"""llama-server: the command line, starting it, readiness and stopping it, for scripts/serve_local.py (a
server kept running by hand) and scripts/select_checkpoint.py (one server per checkpoint, through
LocalServer).

The command reproduces the S1 and S1b server (docs/m0-验证.md) and the server behind the M2 baselines
(runs/eval/serve-base-2.log): -ngl 99, -c 8192, --jinja, bound to 127.0.0.1, llama-server's cache of past
prompts off (--cache-ram 0; scripts/serve_local.py says why), and with an adapter --lora <gguf>
--lora-init-without-apply. llama.cpp b11349 applies a loaded adapter at scale 1.0 to every request without a
"lora" field, even under --lora-init-without-apply (spike S1b), so the base model is reached only by an
explicit scale 0, which LocalClient sends. The binary lives outside the repository: --llama-server or the
LLAMA_SERVER environment variable names it (llama_server).

Readiness is the server's own log line "listening on http", which b11349 prints once the model is loaded,
read from its merged stdout and stderr with a timeout (wait_ready). LocalServer also requires the port to be
free before the start, so that no other server already listening there can pass for the new one, and /health
to answer 200 after the ready line (wait_healthy). It then lists the adapters the server loaded
(/lora-adapters, which names each by the path given on the command line) and its /props, keeps writing the
server's output to a log file, and stops the server when the block ends, an exception included. Every HTTP
request goes through LocalClient, which ignores connection settings from the environment.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import IO, Protocol, Self

from .local import LocalClient

SERVER_VAR = "LLAMA_SERVER"  # names the llama-server binary, which lives outside the repository
MODEL = "models/Qwen3-1.7B-Q8_0.gguf"
READY_PATTERN = "listening on http"
HOST = "127.0.0.1"


class Process(Protocol):
    """What is used of subprocess.Popen; the tests pass a fake."""

    stdout: object
    returncode: int | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class ServerError(RuntimeError):
    """A server LocalServer cannot use."""


class PortBusy(ServerError):
    """Something already answers on the port: a stale server could pass for the new one."""


class NotReady(ServerError):
    """The server exited, never printed its ready line, or never answered /health."""


def llama_server(explicit: str | None, env: Mapping[str, str] = os.environ) -> Path:
    """--llama-server, else $LLAMA_SERVER; LookupError when neither is set (the binary is per machine)."""
    value = explicit or env.get(SERVER_VAR)
    if not value:
        raise LookupError(f"no llama-server given: pass --llama-server or set {SERVER_VAR}")
    return Path(value)


def server_command(binary: str | Path, *, model: str = MODEL, adapter: str | None = None, port: int = 8080,
                   ctx_size: int = 8192, ngl: int = 99, cache_ram: int = 0,
                   extra: Iterable[str] = ()) -> list[str]:
    """The llama-server command line (module docstring); extra arguments go last, as they are."""
    cmd = [str(binary), "-m", model, "-ngl", str(ngl), "-c", str(ctx_size), "--jinja", "--host", HOST,
           "--port", str(port), "--cache-ram", str(cache_ram)]
    if adapter:
        cmd += ["--lora", adapter, "--lora-init-without-apply"]
    return cmd + list(extra)


def start(cmd: list[str], *, popen=subprocess.Popen) -> Process:
    """The server with stdout and stderr merged into one text pipe."""
    return popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                 errors="replace", bufsize=1)


def _read_lines(stream, lines: queue.Queue) -> None:
    for line in stream:
        lines.put(line)
    lines.put(None)  # end of output


def _echo(line: str) -> None:
    print(line, flush=True)


def wait_ready(proc: Process, *, timeout: float, pattern: str = READY_PATTERN,
               echo: Callable[[str], None] = _echo,
               clock: Callable[[], float] = time.monotonic) -> tuple[bool, queue.Queue]:
    """True once a line of the server's output contains pattern.

    False when the server exits or closes its output first, or when timeout seconds pass. Every line read is
    echoed; the queue carries the lines that arrive afterwards (None marks the end of output).
    """
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=_read_lines, args=(proc.stdout, lines), daemon=True).start()
    deadline = clock() + timeout
    while clock() < deadline:
        try:
            line = lines.get(timeout=0.2)
        except queue.Empty:
            if proc.poll() is not None:
                return False, lines
            continue
        if line is None:
            return False, lines
        echo(line.rstrip("\r\n"))
        if pattern in line:
            return True, lines
    return False, lines


def pump(proc: Process, lines: queue.Queue, echo: Callable[[str], None] = _echo) -> int:
    """Echo the server's output until it ends, then return the exit code (KeyboardInterrupt propagates)."""
    while True:
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            continue
        if line is None:
            return proc.wait()
        echo(line.rstrip("\r\n"))


def stop(proc: Process) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class SupportsHealth(Protocol):
    def health(self) -> bool: ...


def wait_healthy(client: SupportsHealth, *, timeout: float, alive: Callable[[], bool] = lambda: True,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 interval: float = 0.5) -> bool:
    """True once client.health() is True (/health answered 200); False as soon as alive() turns False (the
    process exited) or when timeout seconds pass without it."""
    deadline = clock() + timeout
    while True:
        if client.health():
            return True
        if not alive() or clock() >= deadline:
            return False
        sleep(interval)


def same_path(a: str | Path, b: str | Path) -> bool:
    """Whether two spellings name the same file; relative ones are taken from the current folder, where
    llama-server runs, and the comparison ignores case and separators where the file system does (Windows)."""
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class LocalServer:
    """One llama-server for a block of work (module docstring): with LocalServer(...) as server: ... starts it
    on a free port, waits for its ready line and /health, and stops it when the block ends.

    Inside the block: url, cmd, process, client (a LocalClient of its own), adapters (/lora-adapters as listed
    once the server was ready) and props (/props, None when not answered). log, when given, receives the
    server's output line by line. PortBusy or NotReady when the server cannot be used; a server that was
    started is stopped before either is raised.
    """

    def __init__(self, binary: str | Path, *, model: str = MODEL, adapter: str | None = None,
                 port: int = 8080, ctx_size: int = 8192, ngl: int = 99, cache_ram: int = 0,
                 extra: Iterable[str] = (), timeout: float = 120.0, log: str | Path | None = None,
                 popen=subprocess.Popen, client: Callable[[str], LocalClient] = LocalClient,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.cmd = server_command(binary, model=model, adapter=adapter, port=port, ctx_size=ctx_size, ngl=ngl,
                                  cache_ram=cache_ram, extra=extra)
        self.url = f"http://{HOST}:{port}"
        self.timeout, self.log = timeout, None if log is None else Path(log)
        self._popen, self._client, self._clock, self._sleep = popen, client, clock, sleep
        self.process: Process | None = None
        self.client: LocalClient | None = None
        self.adapters: list = []
        self.props: dict | None = None
        self._log: IO[str] | None = None
        self._pump: threading.Thread | None = None

    def __enter__(self) -> Self:
        probe = self._client(self.url)
        try:
            busy = probe.health()
        finally:
            probe.close()
        if busy:
            raise PortBusy(f"a server already answers at {self.url}: stop it first, or choose another port")
        try:
            if self.log is not None:
                self.log.parent.mkdir(parents=True, exist_ok=True)
                self._log = self.log.open("w", encoding="utf-8", newline="\n")
            started = self._clock()
            self.process = start(self.cmd, popen=self._popen)
            ready, lines = wait_ready(self.process, timeout=self.timeout, echo=self._write, clock=self._clock)
            if not ready:
                if self.process.poll() is None:
                    raise NotReady(f"llama-server did not become ready within {self.timeout:g} s")
                raise NotReady(f"llama-server exited with {self.process.returncode} before becoming ready")
            self._pump = threading.Thread(target=pump, args=(self.process, lines, self._write), daemon=True)
            self._pump.start()
            self.client = self._client(self.url)
            left = max(self.timeout - (self._clock() - started), 0.0)
            process = self.process
            if not wait_healthy(self.client, timeout=left, alive=lambda: process.poll() is None,
                                clock=self._clock, sleep=self._sleep):
                raise NotReady(f"llama-server printed its ready line, but {self.url}/health did not answer "
                               f"200 within {self.timeout:g} s")
            self.adapters = self.client.adapters()
            self.props = self.client.props()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop the server and release the client and the log; safe to call more than once."""
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.process is not None:
            stop(self.process)
        if self._pump is not None:
            self._pump.join(timeout=10)  # the output ends with the process
            self._pump = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def _write(self, line: str) -> None:
        log = self._log
        if log is None:
            return
        try:
            log.write(line + "\n")
            log.flush()
        except ValueError:  # closed meanwhile by close(), while the pump thread still had a line
            pass
