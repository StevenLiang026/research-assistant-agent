"""Stand-ins for what the bake-off tooling drives (research_agent.eval.bakeoff, research_agent.eval.fidelity
and scripts/bakeoff.py): the project's scripts as subprocesses, a llama-server behind a port, the machine
probes of the preflight, and the clock. Nothing here starts a process, loads a model or reads the GPU.

FakeRunner stands in for the subprocess runner: a handler per script name (train.py, export_adapter.py,
run_eval.py, score.py, pytest) is called with the script's arguments and may write the files the real script
would; it returns the exit code and the text the script would print. Every command is kept in commands, as
given, with its environment and its log stem. A script without a handler fails the command (exit 1), so a test
sees every script a step runs. FakeExport is export_adapter.py's handler as llama.cpp's converter behaves
under it: it writes the GGUF at --outfile and creates no folder, so a missing folder fails the export as it
failed C3's first S4.

ServedLlama puts tests/support/llama_fakes' FakeLlama behind research_agent.llm.server.LocalServer: the port
answers /health only once the fake server process was started (LocalServer refuses a port that already
answers), the process prints llama-server's ready line and runs until stopped, and every start is kept with
its command line. FakeProbes answers the preflight's three probes from scripted values, each a sequence read
one value per call (the last value repeats), so a test can make a probe clear up after a few polls.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import httpx

from research_agent.llm.local import LocalClient
from support.fake_llama import READY_LINE, ServerProcess
from support.llama_fakes import FakeLlama

Handler = Callable[[list[str]], tuple[int, str]]  # (the script's arguments) -> (exit code, printed text)


class FakeRunner:
    """The subprocess runner of the bake-off, scripted per script (module docstring)."""

    def __init__(self, handlers: Mapping[str, Handler] | None = None) -> None:
        self.handlers = dict(handlers or {})
        self.commands: list[list[str]] = []
        self.logs: list[Path] = []
        self.envs: list[dict] = []

    def __call__(self, cmd: list[str], *, log: Path, env: Mapping[str, str] | None = None
                 ) -> subprocess.CompletedProcess:
        self.commands.append(list(cmd))
        self.logs.append(Path(log))
        self.envs.append(dict(env or {}))
        name = script_name(cmd)
        handler = self.handlers.get(name)
        if handler is None:
            return subprocess.CompletedProcess(cmd, 1, stdout=f"no handler for {name}\n", stderr="")
        code, text = handler(script_args(cmd))
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(f"{log}.out.log").write_text(text, encoding="utf-8")  # the stem as the default runner takes it
        return subprocess.CompletedProcess(cmd, code, stdout=text, stderr="")

    def of(self, name: str) -> list[list[str]]:
        """Every command of one script, in order."""
        return [cmd for cmd in self.commands if script_name(cmd) == name]

    def start(self, cmd: list[str], *, log: Path, env: Mapping[str, str] | None = None) -> FakeProcess:
        """The starter of a long-running script (the pilot's training): the handler runs at once and the
        process is already over when it is polled."""
        return FakeProcess(self(cmd, log=log, env=env).returncode)


class FakeExport:
    """export_adapter.py's handler (module docstring): the GGUF, content, written at --outfile when its folder
    exists; when it does not, exit 1 with the converter's FileNotFoundError, nothing written. written=False
    makes a converter that exits 0 without writing. calls keeps every outfile asked for with whether its
    folder existed at the time."""

    def __init__(self, content: bytes = b"GGUF adapter", *, written: bool = True) -> None:
        self.content, self.written = content, written
        self.calls: list[tuple[Path, bool]] = []

    def __call__(self, argv: list[str]) -> tuple[int, str]:
        outfile = Path(argv[argv.index("--outfile") + 1])
        ready = outfile.parent.is_dir()
        self.calls.append((outfile, ready))
        if not ready:  # convert_lora_to_gguf.py opens the file for writing and creates no folder
            return 1, (f"FileNotFoundError: [Errno 2] No such file or directory: {str(outfile)!r}\n"
                       "converter exited with 1\n")
        if self.written:
            outfile.write_bytes(self.content)
        return 0, f"wrote {outfile}\n"


class FakeProcess:
    """A started script that has already exited."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


def script_name(cmd: list[str]) -> str:
    """The script a command runs: the file name after the interpreter, or "pytest" for python -m pytest."""
    if len(cmd) > 2 and cmd[1] == "-m":
        return cmd[2]
    return Path(cmd[1]).name if len(cmd) > 1 else cmd[0]


def script_args(cmd: list[str]) -> list[str]:
    """The arguments after the script (after the module for python -m)."""
    return list(cmd[3:]) if len(cmd) > 2 and cmd[1] == "-m" else list(cmd[2:])


class ServedLlama:
    """A FakeLlama behind LocalServer (module docstring)."""

    def __init__(self, fake: FakeLlama, *, exit_code: int | None = None) -> None:
        self.fake, self.exit_code = fake, exit_code
        self.started: list[list[str]] = []
        self.processes: list[ServerProcess] = []

    def popen(self, cmd, **kwargs) -> ServerProcess:
        self.started.append(list(cmd))
        process = ServerProcess([READY_LINE], exit_code=self.exit_code)
        self.processes.append(process)
        return process

    def client(self, url: str) -> LocalClient:
        def handle(request: httpx.Request) -> httpx.Response:
            if not self.processes or self.processes[-1].returncode is not None:
                return httpx.Response(503, json={"error": "no server"})
            return self.fake.handle(request)

        return LocalClient(url, transport=httpx.MockTransport(handle))


class FakeProbes:
    """The preflight's probes, scripted (module docstring)."""

    def __init__(self, gpu: Iterable[int | None] = (0,), ram: Iterable[float | None] = (8.0,),
                 ac: Iterable[bool | None] = (True,)) -> None:
        self._gpu, self._ram, self._ac = list(gpu), list(ram), list(ac)
        self.calls: list[str] = []

    def _next(self, values: list, name: str):
        self.calls.append(name)
        return values.pop(0) if len(values) > 1 else values[0]

    def gpu_used_mib(self) -> int | None:
        return self._next(self._gpu, "gpu")

    def available_ram_gb(self) -> float | None:
        return self._next(self._ram, "ram")

    def on_ac_power(self) -> bool | None:
        return self._next(self._ac, "ac")


class FakeClock:
    """A clock that advances by what sleep is asked for."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
