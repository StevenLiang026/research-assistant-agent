"""scripts/run_bfcl.py offline: a fake repository root (the base model, the tokenizer folder, the BFCL
interpreter, the adapters chosen on the development set), a fake llama-server (process and HTTP) and a fake
BFCL that writes result and score files where the real one would, under BFCL_PROJECT_ROOT. Nothing starts
llama-server or BFCL, and no model or tokenizer is loaded. ROUTE_VIA stands for an inherited variable that
keeps HTTP clients from the local server: while BFCL's environment holds one, the fake BFCL cannot reach the
server."""

from __future__ import annotations

import errno
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from fractions import Fraction
from pathlib import Path

import httpx
import pytest
from support.fake_llama import ServerProcess

from research_agent.eval.lock import sha256_of
from research_agent.llm.local import LocalClient

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bfcl = load_script("run_bfcl")
nothink = load_script("bfcl_nothink")

READY = "main: server is listening on http://127.0.0.1:8080 - starting the main loop\n"
# questions per category (BFCL's own run from 16 to 1,053), and the right answers per system
N = {"simple_python": 4, "multiple": 2, "parallel": 2, "parallel_multiple": 2, "irrelevance": 3,
     "live_simple": 3, "live_multiple": 10, "live_parallel": 1, "live_parallel_multiple": 1,
     "live_irrelevance": 8, "live_relevance": 1}
CORRECT = {
    "b0": {"simple_python": 4, "multiple": 2, "parallel": 1, "parallel_multiple": 1, "irrelevance": 3,
           "live_simple": 2, "live_multiple": 5, "live_parallel": 0, "live_parallel_multiple": 1,
           "live_irrelevance": 6, "live_relevance": 1},
    "ft": {"simple_python": 3, "multiple": 2, "parallel": 2, "parallel_multiple": 1, "irrelevance": 2,
           "live_simple": 3, "live_multiple": 7, "live_parallel": 1, "live_parallel_multiple": 0,
           "live_irrelevance": 5, "live_relevance": 1},
    "b4": {"simple_python": 4, "multiple": 1, "parallel": 2, "parallel_multiple": 2, "irrelevance": 3,
           "live_simple": 1, "live_multiple": 6, "live_parallel": 1, "live_parallel_multiple": 1,
           "live_irrelevance": 7, "live_relevance": 0},
}
ADAPTERS = {"ft": "runs/train/ft/gguf/checkpoint-30-f16.gguf",
            "b4": "runs/train/b4/gguf/checkpoint-20-f16.gguf"}
BASE_NAME, ADAPTER_NAME = "Qwen/Qwen3-1.7B-FC-nothink", "research-agent/Qwen3-1.7B-FT-FC-nothink"
INFERENCE_ERROR = "Error during inference: Error code: 503"  # what BFCL records when a request fails
CONNECTION = "route_via"  # the name the stand-in blocking variables share


def group(category: str) -> str:
    """BFCL's folder for a single-turn category (bfcl_eval.utils.get_general_grouping)."""
    return "live" if "live" in category else "non_live"


def last_value(cmd: list[str], option: str) -> str | None:
    """The value after the last occurrence of option in cmd (llama.cpp's parser keeps the last), or None."""
    at = [i for i, arg in enumerate(cmd) if arg == option]
    return cmd[at[-1] + 1] if at else None


class FakeLlama:
    """llama-server for run_bfcl: popen starts it with the adapter its --lora names, if any, the context of
    its last -c, the slots of its last -np (default_slots without one: anything but 1, so that a command that
    lost -np 1 is caught), and its prompt cache on unless the last of --cache-prompt and --no-cache-prompt in
    it is the latter; over HTTP it answers /health, /lora-adapters (listed(path) of that adapter, then beside
    when given and an adapter is loaded, then phantom when given), /props (those slots and that context) and
    the probes' completions, whose first token's log-probability is -0.5 at lora scale 1 and -0.25 at scale 0
    (the base model); a request without a "lora" field gets default_scale while an adapter is loaded, 1.0 as
    b11349 does, and with ignores_lora every request does, whatever its field says. The one slot keeps the
    latest prompt, and a request whose cache_prompt (the server's setting when the field is absent) is on
    reuses the prefix it shares with that prompt, all but its last character at most (characters standing
    for tokens); the answer reports the reused part as usage.prompt_tokens_details.cached_tokens, as b11349
    does, or has no usage at all when usage is False. With a status other than 200, every completion fails
    with that status. With start_error, popen raises it, as subprocess.Popen raises an OSError for a binary it
    cannot start."""

    def __init__(self, *, default_scale: float = 1.0, ignores_lora: bool = False, listed=lambda path: path,
                 beside: str | None = None, phantom: str | None = None, default_slots: int = 4,
                 exit_code: int | None = None, already_running: bool = False, usage: bool = True,
                 status: int = 200, start_error: OSError | None = None) -> None:
        self.default_scale, self.ignores_lora, self.listed = default_scale, ignores_lora, listed
        self.beside, self.phantom, self.default_slots = beside, phantom, default_slots
        self.exit_code, self.usage, self.status, self.start_error = exit_code, usage, status, start_error
        self.adapter: str | None = None
        self.ctx = self.slots = 0
        self.cache = True  # llama-server's default
        self.slot = ""  # the prompt the slot holds
        self.process: ServerProcess | None = ServerProcess([]) if already_running else None
        self.processes: list[ServerProcess] = []
        self.commands: list[list[str]] = []
        self.payloads: list[dict] = []

    def popen(self, cmd, **kwargs):
        self.commands.append(list(cmd))
        if self.start_error is not None:
            raise self.start_error
        self.adapter = last_value(cmd, "--lora")
        self.ctx = int(last_value(cmd, "-c"))
        self.slots = int(last_value(cmd, "-np") or self.default_slots)
        switches = [arg for arg in cmd if arg in ("--cache-prompt", "--no-cache-prompt")]
        self.cache, self.slot = not switches or switches[-1] == "--cache-prompt", ""
        self.process = ServerProcess([] if self.exit_code is not None else [READY], exit_code=self.exit_code)
        self.processes.append(self.process)
        return self.process

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.process is None or self.process.returncode is not None:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/lora-adapters":
            listing = [] if self.adapter is None else [{"id": 0, "path": self.listed(self.adapter),
                                                         "scale": 1.0}]
            for extra in (self.beside if self.adapter is not None else None, self.phantom):
                if extra is not None:
                    listing.append({"id": len(listing), "path": extra, "scale": 1.0})
            return httpx.Response(200, json=listing)
        if path == "/props":
            return httpx.Response(200, json={"build_info": "b11349-fb4b2737a", "total_slots": self.slots,
                                             "default_generation_settings": {"n_ctx": self.ctx}})
        payload = json.loads(request.content)
        self.payloads.append(payload)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"code": self.status}})
        if "lora" in payload and not self.ignores_lora:
            scale = payload["lora"][0]["scale"]
        else:
            scale = self.default_scale if self.adapter is not None else 0.0
        logprob = -0.5 if scale == 1.0 else -0.25
        token = {"token": "I", "logprob": logprob, "top_logprobs": [{"token": "I", "logprob": logprob}]}
        prompt = payload["prompt"]
        reuse = payload.get("cache_prompt", self.cache)  # the server's setting when the request names none
        reused = len(os.path.commonprefix([self.slot, prompt])) if reuse else 0
        self.slot = prompt
        answer: dict = {"choices": [{"text": "I", "finish_reason": "length",
                                     "logprobs": {"content": [token]}}]}
        if self.usage:
            answer["usage"] = {"completion_tokens": 1, "prompt_tokens": len(prompt),
                               "prompt_tokens_details": {"cached_tokens": min(reused, len(prompt) - 1)}}
        return httpx.Response(200, json=answer)

    def client(self, url: str) -> LocalClient:
        return LocalClient(url, transport=httpx.MockTransport(self.handler))


class FakeBfcl:
    """BFCL in its own environment; run stands in for subprocess.run. The version check prints bfcl_eval's
    version; the reach check fails while the environment holds a blocking variable (a name containing
    CONNECTION) and needs the server running. "generate" writes one result file per category under
    BFCL_PROJECT_ROOT, an answer per question (errors[category] of them BFCL's inference error text);
    "evaluate" one score file per category, BFCL's header and a line per failed question, with
    CORRECT[system] right, the system being the folder's name (none for the categories in unscored; a total
    one too high for those in miscounted; for those in inflated a total one too high with as many failed
    questions as it implies, so that only the result file's count disagrees; an accuracy other than correct /
    total for those in misreported). Every command is recorded with its environment; generation must find the
    server running, evaluation stopped. A step named in raises (version, reach, generate, evaluate) raises
    that OSError instead, as subprocess.run does for an interpreter it cannot start."""

    def __init__(self, llama: FakeLlama, *, errors: dict | None = None, exit_codes: dict | None = None,
                 unscored: tuple[str, ...] = (), miscounted: tuple[str, ...] = (), fewer: dict | None = None,
                 garbled: tuple[str, ...] = (), version: subprocess.CompletedProcess | None = None,
                 raises: dict[str, OSError] | None = None, inflated: tuple[str, ...] = (),
                 misreported: tuple[str, ...] = ()) -> None:
        self.llama, self.errors, self.exit_codes = llama, errors or {}, exit_codes or {}
        self.unscored, self.miscounted, self.version = unscored, miscounted, version
        self.inflated, self.misreported = inflated, misreported
        self.fewer = fewer or {}  # system -> a category in which its run has one question fewer
        self.garbled = garbled  # categories whose result file ends in a line that is not JSON
        self.raises = raises or {}
        self.calls: list[tuple[str, list[str], dict]] = []

    def run(self, cmd, **kwargs) -> subprocess.CompletedProcess:
        cmd, env = list(cmd), dict(kwargs["env"])
        step = cmd[2] if cmd[1] != "-c" else "version" if cmd[2] == bfcl.VERSION_CODE else "reach"
        self.calls.append((step, cmd, env))
        if step in self.raises:
            raise self.raises[step]
        if step == "version":
            return self.version or subprocess.CompletedProcess(cmd, 0, stdout="2026.3.23\n", stderr="")
        if step == "reach":
            assert cmd[2] == bfcl.REACH_CODE, cmd
            assert cmd[3] == env["REMOTE_OPENAI_BASE_URL"]
            blocked = any(CONNECTION in name.lower() for name in env)
            code = 0 if self.llama.process.returncode is None and not blocked else 1
            return subprocess.CompletedProcess(cmd, code, stdout="", stderr="ConnectionError" if code else "")
        root = Path(env["BFCL_PROJECT_ROOT"])
        model = cmd[cmd.index("--model") + 1].replace("/", "_")
        categories = cmd[cmd.index("--test-category") + 1].split(",")
        for folder in ("result", "score", ".file_locks"):  # what BFCL makes when it is imported
            (root / folder).mkdir(parents=True, exist_ok=True)
        for category in categories:
            n = N[category] - (self.fewer.get(root.name) == category)
            if step == "generate":
                assert self.llama.process.returncode is None, "BFCL generates against a running server"
                path = root / "result" / model / group(category) / f"BFCL_v4_{category}_result.json"
                bad = self.errors.get(category, 0)
                lines = [{"id": f"{category}_{i}", "result": INFERENCE_ERROR if i < bad else "<tool_call>",
                          "input_token_count": 100, "output_token_count": 10, "latency": 1.5}
                         for i in range(n)]
            else:
                assert step == "evaluate", cmd
                assert self.llama.process.terminated, "the server is stopped before BFCL evaluates"
                if category in self.unscored:
                    continue
                correct = CORRECT[root.name][category]
                total = n + 1 if category in self.miscounted or category in self.inflated else n
                failed = (total if category in self.inflated else n) - correct
                accuracy = (correct + 1 if category in self.misreported else correct) / total
                path = root / "score" / model / group(category) / f"BFCL_v4_{category}_score.json"
                lines = [{"accuracy": accuracy, "correct_count": correct, "total_count": total},
                         *({"id": f"{category}_{i}", "valid": False} for i in range(failed))]
            path.parent.mkdir(parents=True, exist_ok=True)
            tail = "{\"id\": \"cut off\n" if step == "generate" and category in self.garbled else ""
            path.write_text("".join(json.dumps(line) + "\n" for line in lines) + tail, encoding="utf-8")
        kwargs["stdout"].write(f"BFCL {step} {model} \U0001f98d\n".encode())  # BFCL prints emoji
        return subprocess.CompletedProcess(cmd, self.exit_codes.get(step, 0))


def failing(*args, **kwargs):
    raise AssertionError("nothing may be started")


def select(root: Path, system: str) -> None:
    """The adapter scripts/select_checkpoint.py chose for system: its GGUF and the selection file that names
    it."""
    gguf = root / ADAPTERS[system]
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(f"GGUF {system}".encode())
    selection = {"run": f"runs/train/{system}", "system": system,
                 "chosen": {"step": 30, "gguf": ADAPTERS[system], "sha256": sha256_of(gguf), "n": 150,
                            "successes": 120, "rate": 0.8}}
    out = root / "results" / f"checkpoint-selection-{system}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A repository root, as the current folder, with the base model, the tokenizer folder, a llama-server
    binary, the BFCL interpreter at its default place and the adapters chosen for ft and b4 (all fake)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bfcl, "git_commit", lambda repo: "c0ffee")
    monkeypatch.setattr(bfcl, "git_dirty", lambda repo: False)
    monkeypatch.setattr(bfcl, "NOTHINK", tmp_path / "scripts" / "bfcl_nothink.py")
    for name in [name for name in os.environ if CONNECTION in name.lower()]:
        monkeypatch.delenv(name)
    (tmp_path / "models" / "Qwen3-1.7B").mkdir(parents=True)
    (tmp_path / "models" / "Qwen3-1.7B-Q8_0.gguf").write_bytes(b"GGUF base")
    (tmp_path / "llama-server.exe").write_bytes(b"")
    python = tmp_path / bfcl.BFCL_PYTHON
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    for system in ADAPTERS:
        select(tmp_path, system)
    return tmp_path


ARGV = ["--llama-server", "llama-server.exe"]
OUT = Path("results/bfcl-m3.json")


def run_bfcl(llama: FakeLlama, fake: FakeBfcl, *extra: str) -> int:
    return bfcl.main([*ARGV, *extra], popen=llama.popen, run=fake.run, client=llama.client)


def steps(fake: FakeBfcl) -> list[str]:
    return [step for step, _, _ in fake.calls]


def mean(system: str, categories=tuple(N)) -> Fraction:
    return sum(Fraction(CORRECT[system][c], N[c]) for c in categories) / len(categories)


def test_the_bfcl_model_names_are_the_ones_bfcl_nothink_registers():
    assert nothink.MODELS[bfcl.BFCL_MODELS["b0"]] == "base model"
    assert nothink.MODELS[bfcl.BFCL_MODELS["ft"]] == "fine-tuned adapter"
    assert bfcl.BFCL_MODELS["b4"] == bfcl.BFCL_MODELS["ft"]  # an adapter server either way, folders apart
    assert bfcl.CATEGORIES == tuple(N)  # the 11 single-turn categories of BFCL's "python" group, M1's set


def test_the_child_environment_drops_the_variables_the_patterns_name_and_adds_bfcls_settings():
    inherited = {"PATH": "p", "ROUTE_VIA": "x", "route_via_tls": "x", "Route_Via_Except": "x",
                 "SYSTEMROOT": "s"}
    env = bfcl.child_env(inherited, ["route_via"], {"BFCL_PROJECT_ROOT": "root", "PYTHONUTF8": "1"})
    assert env == {"PATH": "p", "SYSTEMROOT": "s", "BFCL_PROJECT_ROOT": "root", "PYTHONUTF8": "1"}
    assert inherited["ROUTE_VIA"] == "x" and len(inherited) == 5  # the caller's mapping is untouched
    assert bfcl.child_env(inherited, [], {}) == inherited  # nothing named, nothing dropped


def test_the_aggregate_is_the_unweighted_mean_of_the_category_accuracies():
    categories = {"simple_python": {"n": 400, "correct": 369, "accuracy": 369 / 400},
                  "live_parallel": {"n": 16, "correct": 8, "accuracy": 0.5}}
    assert bfcl.aggregate(categories) == {"categories": 2, "n": 416,
                                          "accuracy": float((Fraction(369, 400) + Fraction(1, 2)) / 2)}
    assert bfcl.aggregate(categories)["accuracy"] != 377 / 416  # not the pooled rate


def test_a_run_records_each_systems_server_adapter_categories_and_mean(root, monkeypatch, capsys):
    monkeypatch.setenv("ROUTE_VIA", "x")
    monkeypatch.setenv("route_via_tls", "x")
    before = dict(os.environ)
    llama = FakeLlama()
    fake = FakeBfcl(llama)
    assert run_bfcl(llama, fake, "--drop-env", CONNECTION) == 0
    assert dict(os.environ) == before  # nothing was written to this process's environment

    text = (root / OUT).read_text(encoding="utf-8")
    assert str(root) not in text and "\\" not in text  # repository-relative POSIX paths only
    assert CONNECTION not in text.lower()  # the record says nothing of the dropped variables
    results = json.loads(text)
    assert list(results["systems"]) == ["b0", "ft"]
    assert results["categories"] == list(N) and results["bfcl_eval"] == "2026.3.23"
    assert (results["git_commit"], results["git_dirty"]) == ("c0ffee", False)
    assert "mean" in results["aggregate_rule"] and "Overall" in results["aggregate_rule"]
    base = ["llama-server", "-m", "models/Qwen3-1.7B-Q8_0.gguf", "-ngl", "99", "-c", "12288", "--jinja",
            "--host", "127.0.0.1", "--port", "8080", "--cache-ram", "0"]
    fixed = ["-np", "1", "--no-cache-prompt"]  # one slot, and no prompt reused from the request before
    b0, ft = results["systems"]["b0"], results["systems"]["ft"]
    assert b0["server_command"] == [*base, *fixed] and b0["adapter"] is None
    assert ft["server_command"] == [*base, "--lora", ADAPTERS["ft"], "--lora-init-without-apply", *fixed]
    assert ft["adapter"] == {"gguf": ADAPTERS["ft"], "sha256": sha256_of(ADAPTERS["ft"]),
                             "selection": "results/checkpoint-selection-ft.json"}
    assert (b0["bfcl_model"], ft["bfcl_model"]) == (BASE_NAME, ADAPTER_NAME)
    assert b0["server_build"] == "b11349-fb4b2737a" and b0["raw"] == "runs/bfcl/m3/b0"
    for system, row in (("b0", b0), ("ft", ft)):
        assert row["categories"] == {c: {"n": N[c], "correct": CORRECT[system][c],
                                         "accuracy": CORRECT[system][c] / N[c]} for c in N}
        assert row["aggregate"] == {"categories": 11, "n": sum(N.values()), "accuracy": float(mean(system))}
    pooled = Fraction(sum(CORRECT["b0"].values()), sum(N.values()))
    assert b0["aggregate"]["accuracy"] != float(pooled)  # the premise: a mean of rates, not the pooled rate

    # one server per system, each stopped; each was probed before BFCL ran: the cache probe on both (the same
    # prompt twice, as BFCL sends a request: neither a lora nor a cache_prompt field), after the adapter probe
    # on the adapter server (scale 1, scale 0, no lora field)
    assert [c[c.index("-c") + 1] for c in llama.commands] == ["12288", "12288"]
    # what each record says is the command the server was started with, the binary named llama-server
    started = [["llama-server", *cmd[1:]] for cmd in llama.commands]
    assert started == [b0["server_command"], ft["server_command"]]
    assert "--lora" not in llama.commands[0]
    assert llama.commands[1][llama.commands[1].index("--lora") + 1] == ADAPTERS["ft"]
    assert all(p.terminated for p in llama.processes) and len(llama.processes) == 2
    assert [p.get("lora") for p in llama.payloads] == [None, None, [{"id": 0, "scale": 1.0}],
                                                       [{"id": 0, "scale": 0.0}], None, None, None]
    assert ["cache_prompt" in p for p in llama.payloads] == [False, False, True, True, True, False, False]
    assert len({p["prompt"] for p in llama.payloads}) == 1
    # BFCL, in its own environment: the version, then per system the reach check and generation against the
    # running server, and the evaluation after it stopped
    assert steps(fake) == ["version", "reach", "generate", "evaluate", "reach", "generate", "evaluate"]
    generate = fake.calls[2][1]
    assert Path(generate[0]) == bfcl.BFCL_PYTHON
    assert generate[1:] == ["scripts/bfcl_nothink.py", "generate", "--model", BASE_NAME, "--test-category",
                            ",".join(N), "--skip-server-setup", "--local-model-path", "models/Qwen3-1.7B"]
    assert fake.calls[6][1][1:] == ["scripts/bfcl_nothink.py", "evaluate", "--model", ADAPTER_NAME,
                                    "--test-category", ",".join(N)]
    for step, _, env in fake.calls:
        assert not [name for name in env if CONNECTION in name.lower()], step
        assert (env["PYTHONUTF8"], env["HF_HUB_OFFLINE"], env["TQDM_DISABLE"]) == ("1", "1", "1")
    for (_, _, env), system in zip(fake.calls[1:], ["b0"] * 3 + ["ft"] * 3, strict=True):
        assert env["BFCL_PROJECT_ROOT"] == os.path.abspath(f"runs/bfcl/m3/{system}")
        assert env["REMOTE_OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1"
    for system in ("b0", "ft"):
        for name in ("server.log", "generate.log", "evaluate.log"):
            assert (root / "runs" / "bfcl" / "m3" / system / name).is_file(), name
    assert "\U0001f98d" in (root / "runs" / "bfcl" / "m3" / "b0" / "generate.log").read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert f"{100 * float(mean('ft')):.2f}" in out and OUT.as_posix() in out


def test_the_reach_check_asks_the_server_with_requests_and_through_the_openai_sdk_as_bfcl_does(monkeypatch):
    """BFCL's readiness check uses requests, and its requests go through the OpenAI SDK, over whatever
    transport the SDK chooses: the check in BFCL's environment does both against the server's /v1 and brings
    no HTTP client of its own (run here over fake requests and openai modules, httpx made unimportable)."""
    asked: list = []

    class Response:
        def raise_for_status(self) -> None:
            asked.append("raise_for_status")

    class OpenAI:
        def __init__(self, **kwargs) -> None:
            asked.append(("OpenAI", kwargs))
            self.models = types.SimpleNamespace(list=lambda: asked.append("models.list"))

    requests = types.ModuleType("requests")
    requests.get = lambda url, timeout: asked.append(("get", url, timeout)) or Response()
    openai = types.ModuleType("openai")
    openai.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "httpx", None)
    monkeypatch.setattr(sys, "argv", ["-c", "http://127.0.0.1:8080/v1"])
    exec(compile(bfcl.REACH_CODE, "<reach check>", "exec"), {})  # noqa: S102  the check's own code, as -c runs it
    assert asked == [("get", "http://127.0.0.1:8080/v1/models", 30), "raise_for_status",
                     ("OpenAI", {"base_url": "http://127.0.0.1:8080/v1", "api_key": "EMPTY", "timeout": 30,
                                 "max_retries": 0}), "models.list"]


def test_without_dropping_the_blocking_variables_bfcl_cannot_reach_the_server_and_the_run_exits_2(
        root, monkeypatch, capsys):
    monkeypatch.setenv("ROUTE_VIA", "x")
    llama = FakeLlama()
    fake = FakeBfcl(llama)
    assert run_bfcl(llama, fake) == 2  # BFCL's own readiness check would wait for ever
    assert steps(fake) == ["version", "reach"]
    assert "--drop-env" in capsys.readouterr().out
    assert all(p.terminated for p in llama.processes) and not (root / OUT).exists()


def test_the_results_file_is_byte_identical_on_a_rerun(root):
    llama = FakeLlama()
    assert run_bfcl(llama, FakeBfcl(llama)) == 0
    first = (root / OUT).read_bytes()
    shutil.rmtree(root / "runs" / "bfcl")
    llama = FakeLlama()
    assert run_bfcl(llama, FakeBfcl(llama)) == 0
    assert (root / OUT).read_bytes() == first


def test_b4_runs_on_a_server_with_its_own_adapter_and_folder(root):
    llama = FakeLlama()
    assert run_bfcl(llama, FakeBfcl(llama), "--systems", "b0,ft,b4") == 0
    results = json.loads((root / OUT).read_text(encoding="utf-8"))
    b4 = results["systems"]["b4"]
    assert list(results["systems"]) == ["b0", "ft", "b4"]
    assert b4["adapter"]["gguf"] == ADAPTERS["b4"] and b4["raw"] == "runs/bfcl/m3/b4"
    assert b4["bfcl_model"] == ADAPTER_NAME and b4["aggregate"]["accuracy"] == float(mean("b4"))
    assert llama.commands[2][llama.commands[2].index("--lora") + 1] == ADAPTERS["b4"]


def test_dry_run_prints_the_commands_and_starts_and_writes_nothing(root, monkeypatch, capsys):
    monkeypatch.setenv("ROUTE_VIA", "x")
    monkeypatch.setenv("route_via_tls", "x")
    argv = [*ARGV, "--systems", "b0,ft,b4", "--drop-env", CONNECTION, "--dry-run"]
    assert bfcl.main(argv, popen=failing, run=failing, client=failing) == 0
    out = capsys.readouterr().out
    for system in ("ft", "b4"):
        assert f"--lora {ADAPTERS[system]} --lora-init-without-apply" in out
        assert sha256_of(ADAPTERS[system]) in out
    assert "-c 12288" in out and "-np 1 --no-cache-prompt" in out and "--cache-ram 0" in out
    assert f"generate --model {BASE_NAME} --test-category {','.join(N)} --skip-server-setup" in out
    assert f"evaluate --model {ADAPTER_NAME} --test-category {','.join(N)}" in out
    assert "REMOTE_OPENAI_BASE_URL=http://127.0.0.1:8080/v1" in out
    assert "2 inherited variables dropped" in out
    assert not (root / "runs" / "bfcl").exists() and not (root / OUT).exists()


def test_b0_and_ft_are_required_and_b4_is_optional(root, capsys):
    for systems in ("b0", "ft", "b0,b4", "b0,ft,b9"):
        argv = [*ARGV, "--systems", systems, "--dry-run"]
        assert bfcl.main(argv, popen=failing, run=failing, client=failing) == 2, systems
    argv = [*ARGV, "--b4-adapter", ADAPTERS["b4"], "--dry-run"]  # an adapter for a system not run
    assert bfcl.main(argv, popen=failing, run=failing, client=failing) == 2
    capsys.readouterr()
    argv = [*ARGV, "--systems", "b4,ft,b0", "--dry-run"]
    assert bfcl.main(argv, popen=failing, run=failing, client=failing) == 0
    out = capsys.readouterr().out
    assert out.index("runs/bfcl/m3/b0") < out.index("runs/bfcl/m3/ft") < out.index("runs/bfcl/m3/b4")


def test_a_drop_env_pattern_that_is_not_a_regular_expression_is_a_usage_error(root, capsys):
    argv = [*ARGV, "--drop-env", "route_(via", "--dry-run"]
    assert bfcl.main(argv, popen=failing, run=failing, client=failing) == 2
    assert "--drop-env" in capsys.readouterr().out


def test_the_adapter_is_the_selected_checkpoint_and_must_still_match_its_sha256(root, capsys):
    gguf = root / ADAPTERS["ft"]
    gguf.write_bytes(b"GGUF ft, overwritten after the selection")
    assert bfcl.main([*ARGV, "--dry-run"], popen=failing, run=failing, client=failing) == 2
    assert "checkpoint-selection-ft.json" in capsys.readouterr().out
    argv = [*ARGV, "--ft-adapter", ADAPTERS["ft"], "--dry-run"]  # named explicitly: taken as it is
    assert bfcl.main(argv, popen=failing, run=failing, client=failing) == 0
    assert sha256_of(gguf) in capsys.readouterr().out
    (root / "results" / "checkpoint-selection-ft.json").unlink()
    assert bfcl.main([*ARGV, "--dry-run"], popen=failing, run=failing, client=failing) == 2
    assert "--ft-adapter" in capsys.readouterr().out
    argv = [*ARGV, "--ft-adapter", "runs/train/ft/gguf/missing.gguf", "--dry-run"]
    assert bfcl.main(argv, popen=failing, run=failing, client=failing) == 2


def test_a_raw_folder_holding_an_earlier_run_is_refused_before_anything_starts(root, capsys):
    (root / "runs" / "bfcl" / "m3" / "b0").mkdir(parents=True)  # empty: fine
    assert bfcl.main([*ARGV, "--dry-run"], popen=failing, run=failing, client=failing) == 0
    capsys.readouterr()
    earlier = root / "runs" / "bfcl" / "m3" / "ft" / "result" / ADAPTER_NAME.replace("/", "_") / "x.json"
    earlier.parent.mkdir(parents=True)
    earlier.write_text("{}\n", encoding="utf-8")
    assert bfcl.main(ARGV, popen=failing, run=failing, client=failing) == 2
    assert "runs/bfcl/m3/ft" in capsys.readouterr().out
    assert not (root / OUT).exists()


def test_inference_errors_stop_the_run_before_evaluation_with_exit_4(root, capsys):
    llama = FakeLlama()
    fake = FakeBfcl(llama, errors={"live_multiple": 2})
    assert run_bfcl(llama, fake) == 4
    assert steps(fake) == ["version", "reach", "generate"]
    out = capsys.readouterr().out
    assert "live_multiple" in out and "inference error" in out
    assert all(p.terminated for p in llama.processes) and not (root / OUT).exists()


@pytest.mark.parametrize("step", ["generate", "evaluate"])
def test_a_bfcl_step_that_fails_stops_the_run_with_exit_4(root, capsys, step):
    llama = FakeLlama()
    fake = FakeBfcl(llama, exit_codes={step: 1})
    assert run_bfcl(llama, fake) == 4
    assert steps(fake)[-1] == step and "runs/bfcl/m3/b0" in capsys.readouterr().out
    assert all(p.terminated for p in llama.processes) and not (root / OUT).exists()


def test_a_category_without_a_score_or_with_counts_that_disagree_exits_1(root, capsys):
    llama = FakeLlama()
    assert run_bfcl(llama, FakeBfcl(llama, unscored=("live_relevance",))) == 1
    assert "live_relevance" in capsys.readouterr().out and not (root / OUT).exists()
    shutil.rmtree(root / "runs" / "bfcl")
    llama = FakeLlama()
    assert run_bfcl(llama, FakeBfcl(llama, miscounted=("parallel",))) == 1
    assert "parallel" in capsys.readouterr().out and not (root / OUT).exists()
    shutil.rmtree(root / "runs" / "bfcl")
    llama = FakeLlama()
    assert run_bfcl(llama, FakeBfcl(llama, garbled=("multiple",))) == 1
    assert "BFCL_v4_multiple_result.json" in capsys.readouterr().out and not (root / OUT).exists()


@pytest.mark.parametrize("options, category", [
    ({"inflated": ("live_simple",)}, "live_simple"),  # a header that agrees with itself, not with the answers
    ({"misreported": ("parallel",)}, "parallel"),  # an accuracy that is not correct / total
])
def test_a_score_header_at_odds_with_the_result_file_or_with_its_own_counts_exits_1(root, capsys, options,
                                                                                     category):
    llama = FakeLlama()
    fake = FakeBfcl(llama, **options)
    assert run_bfcl(llama, fake) == 1
    assert steps(fake)[-1] == "evaluate" and not (root / OUT).exists()
    assert f"BFCL_v4_{category}_score.json: the header" in capsys.readouterr().out


def test_bfcl_files_are_split_at_line_feeds_only_so_an_answer_may_hold_unicode_line_separators(root):
    """BFCL writes its files without escaping non-ASCII text, so an answer may hold U+2028, U+2029 or U+0085,
    at which str.splitlines() would cut it; a line ends at a line feed only, and a CR before one is JSON
    whitespace."""
    separators = "".join(map(chr, (0x2028, 0x2029, 0x85)))
    entries = [{"id": "live_simple_0", "result": f"[get_weather(city='Paris'){separators}]"},
               {"id": "live_simple_1", "result": "<tool_call>"}]
    path = root / "BFCL_v4_live_simple_result.json"
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\r\n" for e in entries), encoding="utf-8",
                    newline="")
    assert bfcl.json_lines(path) == entries


def test_systems_with_a_different_number_of_questions_in_a_category_exit_1(root, capsys):
    llama = FakeLlama()
    fake = FakeBfcl(llama, fewer={"ft": "live_multiple"})  # each run consistent in itself
    assert run_bfcl(llama, fake) == 1
    assert steps(fake)[-1] == "evaluate" and "live_multiple" in capsys.readouterr().out
    assert not (root / OUT).exists()


def test_an_adapter_server_must_apply_the_adapter_to_requests_without_a_lora_field(root, capsys):
    llama = FakeLlama(default_scale=0.0)  # the base model for BFCL's requests: FT would measure B0
    fake = FakeBfcl(llama)
    assert run_bfcl(llama, fake) == 3
    assert steps(fake) == ["version", "reach", "generate", "evaluate"]  # b0's only
    assert "lora" in capsys.readouterr().out
    assert all(p.terminated for p in llama.processes) and not (root / OUT).exists()


@pytest.mark.parametrize("server, extra, said", [
    ({"listed": lambda path: ADAPTERS["b4"]}, (), "FAIL: ft: the server lists the adapters"),  # another GGUF
    ({"beside": ADAPTERS["b4"]}, (), "FAIL: ft: the server lists the adapters"),  # a second one beside ft's
    ({"phantom": ADAPTERS["b4"]}, (), "FAIL: b0: the server lists the adapters"),  # the base model's server
    # four slots, the last -np winning: BFCL's answers would depend on what they were batched with
    ({}, ("--", "-np", "4"), "FAIL: b0: the server reports 4 slots of 12288 tokens"),
    # the evaluation server's context (scripts/serve_local.py): too short for BFCL's longest prompt and reply
    ({}, ("--", "-c", "8192"), "FAIL: b0: the server reports 1 slots of 8192 tokens"),
    # a server that ignores the lora field: the probe cannot tell that BFCL's requests get the adapter
    ({"ignores_lora": True}, (), "FAIL: ft: a completion without a lora field"),
])
def test_a_server_with_other_adapters_slots_or_context_or_one_ignoring_lora_stops_the_run_with_exit_3(
        root, capsys, server, extra, said):
    llama = FakeLlama(**server)
    fake = FakeBfcl(llama)
    assert run_bfcl(llama, fake, *extra) == 3
    assert said in capsys.readouterr().out
    assert all(p.terminated for p in llama.processes) and not (root / OUT).exists()
    assert steps(fake)[-1] in ("version", "evaluate")  # nothing ran against the server that failed


@pytest.mark.parametrize("server, extra, said", [
    # the prompt cache turned back on after "--" (the last switch wins): the second request reuses the first
    ({}, ("--", "--cache-prompt"), f"came back with [0, {len(bfcl.PROBE_PROMPT) - 1}] prompt tokens taken"),
    ({"usage": False}, (), "came back with [None, None] prompt tokens taken"),  # it does not say
    ({"status": 503}, (), "probing the server failed: HTTPStatusError"),
])
def test_a_server_that_reuses_a_cached_prompt_or_does_not_say_whether_stops_the_run_with_exit_3(
        root, capsys, server, extra, said):
    llama = FakeLlama(**server)
    fake = FakeBfcl(llama)
    assert run_bfcl(llama, fake, *extra) == 3  # BFCL's answers would depend on the request before them
    assert steps(fake) == ["version"]  # the b0 server failed the probe before BFCL ran
    out = capsys.readouterr().out
    assert "FAIL: b0: " in out and said in out
    assert all(p.terminated for p in llama.processes) and not (root / OUT).exists()


def windows_image_error(code: int, text: str) -> OSError:
    """CreateProcess's failure for a file that is no program of this machine: errno EINVAL, the Windows code
    in winerror (set by hand, so that the case runs on every system)."""
    exc = OSError(errno.EINVAL, text)
    exc.winerror = code
    return exc


@pytest.mark.parametrize("error, code", [
    (OSError(errno.ENOEXEC, "Exec format error"), 2),  # not a program for this system: a usage error
    (windows_image_error(216, "This version of %1 is not compatible with this Windows"), 2),
    (windows_image_error(1392, "The file or directory is corrupted and unreadable"), 2),
    (OSError(errno.EINVAL, "Invalid argument"), 3),  # EINVAL without an image error: the start failed
    (PermissionError(errno.EACCES, "Permission denied"), 2),
    (OSError(errno.EAGAIN, "Resource temporarily unavailable"), 3),  # the binary is fine, the start failed
])
def test_a_llama_server_that_cannot_be_started_stops_the_run_with_a_fail_line(root, capsys, error, code):
    """Popen's OSError is a FAIL line, not a traceback: exit 2 for a binary that cannot be run at all, 3
    otherwise; the empty log of the server that never started is not left behind, so the next run can
    start."""
    llama = FakeLlama(start_error=error)
    fake = FakeBfcl(llama)
    assert run_bfcl(llama, fake) == code
    out = capsys.readouterr().out
    assert "FAIL: b0: llama-server could not be started" in out and error.strerror in out
    assert steps(fake) == ["version"] and len(llama.commands) == 1 and not (root / OUT).exists()
    assert not (root / "runs" / "bfcl" / "m3" / "b0").exists()
    llama = FakeLlama()
    assert run_bfcl(llama, FakeBfcl(llama)) == 0  # the binary fixed, the next run is not refused


def test_a_bfcl_interpreter_that_cannot_be_started_exits_2_before_any_server_starts(root, capsys):
    llama = FakeLlama()
    fake = FakeBfcl(llama, raises={"version": FileNotFoundError(errno.ENOENT, "No such file or directory")})
    assert run_bfcl(llama, fake) == 2 and llama.processes == []
    out = capsys.readouterr().out
    assert "FAIL: " in out and "No such file or directory" in out and "--bfcl-python" in out
    assert not (root / "runs" / "bfcl").exists() and not (root / OUT).exists()


@pytest.mark.parametrize("step, code", [("reach", 2), ("generate", 4), ("evaluate", 4)])
def test_a_bfcl_step_that_cannot_be_started_stops_the_run_with_a_fail_line(root, capsys, step, code):
    """An OSError where a BFCL process starts (the reach check exits 2 like a server it cannot reach, the
    generation and the evaluation 4 like a failed step): a FAIL line, the server stopped, nothing written."""
    llama = FakeLlama()
    fake = FakeBfcl(llama, raises={step: OSError(errno.ENOEXEC, "Exec format error")})
    assert run_bfcl(llama, fake) == code
    out = capsys.readouterr().out
    assert "FAIL: b0: " in out and "Exec format error" in out and steps(fake)[-1] == step
    assert all(p.terminated for p in llama.processes) and not (root / OUT).exists()


def test_a_busy_port_exits_2_and_a_server_that_never_gets_ready_exits_3(root, capsys):
    busy = FakeLlama(already_running=True)
    assert run_bfcl(busy, FakeBfcl(busy)) == 2 and busy.processes == []
    assert "already answers" in capsys.readouterr().out
    dead = FakeLlama(exit_code=1)
    assert run_bfcl(dead, FakeBfcl(dead)) == 3 and len(dead.processes) == 1
    assert "exited with 1" in capsys.readouterr().out
    assert not (root / OUT).exists()


def test_only_the_single_turn_python_categories_can_be_run(root, capsys):
    for categories in ("simple_java", "multi_turn_base", "simple_python,web_search_base"):
        argv = [*ARGV, "--categories", categories, "--dry-run"]
        assert bfcl.main(argv, popen=failing, run=failing, client=failing) == 2, categories
    llama = FakeLlama()
    fake = FakeBfcl(llama)
    assert run_bfcl(llama, fake, "--categories", "live_simple,simple_python") == 0
    results = json.loads((root / OUT).read_text(encoding="utf-8"))
    assert results["categories"] == ["simple_python", "live_simple"]
    generate = fake.calls[2][1]
    assert generate[generate.index("--test-category") + 1] == "simple_python,live_simple"
    ft = results["systems"]["ft"]
    assert list(ft["categories"]) == ["simple_python", "live_simple"]
    assert ft["aggregate"] == {"categories": 2, "n": 7,
                               "accuracy": float(mean("ft", ("simple_python", "live_simple")))}


def test_extra_server_arguments_after_a_double_dash_reach_every_server_and_the_record(root):
    llama = FakeLlama()
    assert run_bfcl(llama, FakeBfcl(llama), "--", "--no-warmup") == 0
    assert [cmd[-2:] for cmd in llama.commands] == [["--no-cache-prompt", "--no-warmup"]] * 2
    results = json.loads((root / OUT).read_text(encoding="utf-8"))
    assert all(row["server_command"][-2:] == ["--no-cache-prompt", "--no-warmup"]
               for row in results["systems"].values())


def test_paths_outside_the_current_folder_are_refused(root, capsys, monkeypatch):
    argv = [*ARGV, "--out", "../bfcl-m3.json", "--dry-run"]
    assert bfcl.main(argv, popen=failing, run=failing, client=failing) == 2
    assert "outside" in capsys.readouterr().out
    # run from another folder than the repository root, the script that BFCL runs lies outside it too
    monkeypatch.setattr(bfcl, "NOTHINK", root.parent / "elsewhere" / "scripts" / "bfcl_nothink.py")
    assert bfcl.main([*ARGV, "--dry-run"], popen=failing, run=failing, client=failing) == 2
    assert "bfcl_nothink.py" in capsys.readouterr().out


def test_absolute_paths_inside_the_current_folder_are_recorded_relative(root):
    llama = FakeLlama()
    fake = FakeBfcl(llama)
    folder, model = root / "models" / "Qwen3-1.7B", root / "models" / "Qwen3-1.7B-Q8_0.gguf"
    assert run_bfcl(llama, fake, "--tokenizer", str(folder), "--model", str(model)) == 0
    text = (root / OUT).read_text(encoding="utf-8")
    assert str(root) not in text and "\\" not in text
    generate = fake.calls[2][1]
    assert generate[generate.index("--local-model-path") + 1] == "models/Qwen3-1.7B"
    b0 = json.loads(text)["systems"]["b0"]
    assert b0["server_command"][b0["server_command"].index("-m") + 1] == "models/Qwen3-1.7B-Q8_0.gguf"


def test_without_llama_server_or_the_bfcl_environment_the_run_stops_with_exit_2(root, capsys, monkeypatch):
    monkeypatch.delenv("LLAMA_SERVER", raising=False)
    assert bfcl.main(["--dry-run"], popen=failing, run=failing, client=failing) == 2
    assert "LLAMA_SERVER" in capsys.readouterr().out
    llama = FakeLlama()
    broken = FakeBfcl(llama, version=subprocess.CompletedProcess([], 1, stdout="", stderr="No module"))
    assert run_bfcl(llama, broken) == 2 and llama.processes == []  # checked before any server starts
    assert "bfcl_eval" in capsys.readouterr().out
    (root / bfcl.BFCL_PYTHON).unlink()
    assert bfcl.main([*ARGV, "--dry-run"], popen=failing, run=failing, client=failing) == 2
    assert "--bfcl-python" in capsys.readouterr().out
