"""research_agent.llm.server offline: the command line, readiness and stop of llama-server, and LocalServer
over a fake process and a fake HTTP server (nothing is started, no model is loaded)."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from support.fake_llama import READY_LINE, ServerProcess

from research_agent.llm.local import LocalClient
from research_agent.llm.server import (
    LocalServer,
    NotReady,
    PortBusy,
    same_path,
    server_command,
    wait_healthy,
)

ADAPTER = "runs/train/ft/gguf/checkpoint-209-f16.gguf"


class FakeHttp:
    """The HTTP side: nothing listens until the process starts; then /health answers the given codes in turn
    (the last one repeats), /lora-adapters lists the adapter passed with --lora, /props the build."""

    def __init__(self, health: list[int] = (200,), *, listening: bool = False) -> None:
        self.health, self.listening = list(health), listening
        self.adapter: str | None = None
        self.paths: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if not self.listening:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/health":
            code = self.health.pop(0) if len(self.health) > 1 else self.health[0]
            return httpx.Response(code, json={})
        if request.url.path == "/lora-adapters":
            listed = [] if self.adapter is None else [{"id": 0, "path": self.adapter, "scale": 1.0}]
            return httpx.Response(200, json=listed)
        if request.url.path == "/props":
            return httpx.Response(200, json={"build_info": "b11349-fb4b2737a"})
        return httpx.Response(404)

    def client(self, url: str) -> LocalClient:
        return LocalClient(url, transport=httpx.MockTransport(self.handler))


def popen_for(http: FakeHttp, process: ServerProcess, started: list):
    def popen(cmd, **kwargs):
        started.append((cmd, kwargs))
        http.listening = True
        if "--lora" in cmd:
            http.adapter = cmd[cmd.index("--lora") + 1]
        return process

    return popen


def test_the_command_is_the_m2_server_with_the_adapter_and_extra_arguments_after_it():
    assert server_command("srv.exe") == ["srv.exe", "-m", "models/Qwen3-1.7B-Q8_0.gguf", "-ngl", "99", "-c",
                                         "8192", "--jinja", "--host", "127.0.0.1", "--port", "8080",
                                         "--cache-ram", "0"]
    cmd = server_command(Path("srv.exe"), model="m.gguf", adapter=ADAPTER, port=8091, ctx_size=4096, ngl=20,
                         cache_ram=512, extra=["--chat-template-kwargs", "{}"])
    assert cmd == ["srv.exe", "-m", "m.gguf", "-ngl", "20", "-c", "4096", "--jinja", "--host", "127.0.0.1",
                   "--port", "8091", "--cache-ram", "512", "--lora", ADAPTER, "--lora-init-without-apply",
                   "--chat-template-kwargs", "{}"]


def test_a_local_server_is_ready_on_its_own_line_and_health_and_lists_its_adapter_until_it_stops(tmp_path):
    http, started = FakeHttp([503, 200]), []
    process = ServerProcess(["loading model\n", READY_LINE, "srv  update_slots: all slots are idle\n"])
    log = tmp_path / "logs" / "server.log"
    server = LocalServer("srv.exe", adapter=ADAPTER, port=8091, timeout=5, log=log,
                         popen=popen_for(http, process, started), client=http.client,
                         sleep=lambda seconds: None)
    with server as running:
        assert running.url == "http://127.0.0.1:8091" and running.cmd == started[0][0]
        assert running.cmd == server_command("srv.exe", adapter=ADAPTER, port=8091)
        assert running.adapters == [{"id": 0, "path": ADAPTER, "scale": 1.0}]
        assert running.props == {"build_info": "b11349-fb4b2737a"} and not process.terminated
    assert process.terminated
    assert http.paths[0] == "/health" and http.paths.count("/health") == 3  # busy check, 503, 200
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == ["loading model", READY_LINE.rstrip("\n")]
    assert lines[-1] == "srv  update_slots: all slots are idle"  # read after the ready line, then logged


def test_a_busy_port_is_refused_before_anything_starts():
    http, started = FakeHttp(listening=True), []
    server = LocalServer("srv.exe", port=8080, popen=popen_for(http, ServerProcess([READY_LINE]), started),
                         client=http.client)
    with pytest.raises(PortBusy, match="already answers at http://127.0.0.1:8080"):
        server.__enter__()
    assert started == []


def test_a_server_that_exits_before_its_ready_line_is_refused():
    http, started = FakeHttp(), []
    process = ServerProcess(["error: failed to load model\n"], exit_code=3)
    server = LocalServer("srv.exe", popen=popen_for(http, process, started), client=http.client, timeout=5)
    with pytest.raises(NotReady, match="exited with 3"):
        server.__enter__()
    assert len(started) == 1


def test_a_server_that_never_prints_its_ready_line_is_stopped_and_refused():
    http = FakeHttp()
    process = ServerProcess(["still loading\n"])
    server = LocalServer("srv.exe", popen=popen_for(http, process, []), client=http.client, timeout=0.3)
    with pytest.raises(NotReady, match="within 0.3 s"):
        server.__enter__()
    assert process.terminated


def test_a_ready_line_without_health_is_not_ready_and_the_server_is_stopped():
    http = FakeHttp([503])
    process = ServerProcess([READY_LINE])
    ticks = iter(range(1000))
    server = LocalServer("srv.exe", popen=popen_for(http, process, []), client=http.client, timeout=5,
                         clock=lambda: next(ticks), sleep=lambda seconds: None)
    with pytest.raises(NotReady, match="/health"):
        server.__enter__()
    assert process.terminated


def test_the_server_is_stopped_when_the_work_inside_fails(tmp_path):
    http = FakeHttp()
    process = ServerProcess([READY_LINE])
    server = LocalServer("srv.exe", popen=popen_for(http, process, []), client=http.client,
                         log=tmp_path / "server.log", timeout=5)
    with pytest.raises(RuntimeError, match="the run failed"), server:
        raise RuntimeError("the run failed")
    assert process.terminated


class Health:
    def __init__(self, *answers: bool) -> None:
        self.answers, self.asked = list(answers), 0

    def health(self) -> bool:
        self.asked += 1
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def test_wait_healthy_polls_until_health_answers_and_gives_up_on_a_dead_process_or_the_deadline():
    slept = []
    assert wait_healthy(Health(False, False, True), timeout=10, clock=lambda: 0.0, sleep=slept.append)
    assert len(slept) == 2
    dead = Health(False)
    assert not wait_healthy(dead, timeout=10, alive=lambda: False, clock=lambda: 0.0, sleep=slept.append)
    assert dead.asked == 1
    ticks = iter(range(100))
    late = Health(False)
    assert not wait_healthy(late, timeout=3, clock=lambda: next(ticks), sleep=lambda seconds: None)
    assert late.asked <= 4


def test_same_path_compares_spellings_of_one_file_from_the_current_folder():
    assert same_path(ADAPTER, Path.cwd() / "runs" / "train" / "ft" / "gguf" / "checkpoint-209-f16.gguf")
    assert same_path("./" + ADAPTER, ADAPTER)
    assert not same_path(ADAPTER, "runs/train/ft/gguf/checkpoint-418-f16.gguf")
    if os.name == "nt":  # llama-server lists the path as it was passed: case and separators may differ
        assert same_path(ADAPTER.upper().replace("/", "\\"), ADAPTER)
