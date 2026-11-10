"""A llama-server stand-in for the tests of research_agent.llm.server and scripts/select_checkpoint.py: a
process that prints its log lines and runs until it is stopped. Nothing is started, no model is loaded."""

from __future__ import annotations

import threading

READY_LINE = "main: server is listening on http://127.0.0.1:8091 - starting the main loop\n"


class ServerProcess:
    """Prints its lines, then runs until terminated (its output ends then), or exits with exit_code."""

    def __init__(self, lines: list[str], *, exit_code: int | None = None) -> None:
        self.lines, self.exit_code = list(lines), exit_code
        self.returncode: int | None = None
        self.terminated = False
        self._stopped = threading.Event()
        self.stdout = self._output()

    def _output(self):
        yield from self.lines
        if self.exit_code is not None:
            self.returncode = self.exit_code
            return
        self._stopped.wait(10)  # a running server: its output ends when it is stopped

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if self.returncode is None:
            self.returncode = 1
        self._stopped.set()

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode
