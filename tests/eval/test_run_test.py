"""scripts/run_test.py offline, in a fake repository (a temporary git repository with synthetic locked sets,
their own access logs and a configuration): the preflight's refusals one by one, the dry run that touches
development material only, exactly one open per access log, a session's order and results, the refusal after
an unfinished open, a cap that stops B2, and two sessions over the same answers writing the same bytes; every
real session after a dry run of its commit. M5 in a fake repository after a complete M3 session: ftr alone,
its dry run over the whole development set gated by the frozen point, the references checked against the M3
completion markers and scored again, recorded and tested comparisons, ftr's routing, resources and cap. M6's
ft2 and ft2r against the records of M3 and M5. M7 on the v2 family of test sets, beside the used-up v1 family:
ft2 on the 1.7B server, ft3 and ft3r on a server with its own base model and tokenizer, ft3r routed to B2, the
dry run over the v2 family's own development set (dev and dev-v2 jointly), which must be here. FT3 thinking
on the family path (its adapter's generation): the dry run and the session single step and end to end, the
generation in the records and the results, the configuration errors, the routing file chosen in that
generation; and every session without a generation writing what it wrote before (M3 to M6, M7's legacy path).
M7's final local system (docs/m7-本地提升计划.md WP5): final, voting and ruled, and finalg, greedy and single
step only, on one server, frozen by scripts/freeze_final.py from development runs scripts/run_eval.py made on
the fake server: the preflight's checks of the frozen file, the dry run over every development item held
against it, both scorers in the single-step results and the comparisons, the vote's and the rules' counts,
the budget of B2 alone and the configuration's per-system options. The example configuration of M7. The
machine is faked: llama-server and nvidia-smi as processes, the server's HTTP side (the family path's in
tests/support/llama_fakes.py's form), DeepSeek's SDK. Nothing here opens the real test sets, starts a server
or loads a model."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path, PurePosixPath, PureWindowsPath

import httpx
import pytest
from support.fake_llama import READY_LINE, ServerProcess
from support.family_fakes import ASSISTANT, THINK_CLOSE, THINK_OPEN, FakeFamilyTokenizer
from support.llama_fakes import FakeLlama, Scripted, install_fake_family
from train.tokenizer_fake import FakeTokenizer

from research_agent.agent.prompts import system_prompt
from research_agent.agent.rules import RULES, RULES_VERSION
from research_agent.canonical import canonical_json
from research_agent.data.synth import OwnerFunctions
from research_agent.eval import report
from research_agent.eval.e2e import World
from research_agent.eval.lock import TEST_FAMILIES, read_jsonl, sha256_of, write_jsonl, write_lock
from research_agent.eval.metrics import (
    cluster_noninferiority_p,
    holm,
    noninferiority_p,
    stratified_noninferiority_p,
)
from research_agent.eval.systems import (
    DEMONSTRATIONS,
    DEMONSTRATIONS_HEADING,
    block_hash,
    demonstrations_block,
    load_demonstrations,
)
from research_agent.family.core import GREEDY, Decoding, Mode, load_family
from research_agent.llm.budget import Budget
from research_agent.llm.generation import request_seed
from research_agent.llm.local import LocalClient
from research_agent.net.fetch import Fetched, ReplayMiss
from research_agent.policy import decide
from research_agent.tools.adapters import fulltext_url, paper_url
from research_agent.tools.corpus_search import CorpusIndex
from research_agent.train.samples import prompt_renderer
from research_agent.validation import check_call

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_test = load_script("run_test")


@pytest.fixture(scope="module", autouse=True)
def ledgers_at_the_list_price():
    """The sessions' ledgers book the list price, as every ledger did before M7: off peak they would book half
    of it (research_agent.llm.budget), so what they book would depend on the hour the tests run."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("research_agent.eval.systems.load_off_peak", lambda: None)
        yield


TODAY = "2026-10-10"
NOW = "2026-10-17T03:00:00+00:00"
CONFIG = "results/test-config-m3.json"
FT_GGUF = "runs/train/ft/gguf/checkpoint-418-f16.gguf"
B4_GGUF = "runs/train/b4/gguf/checkpoint-278-f16.gguf"
BINARY = "runs/bin/llama-server.exe"
SYSTEMS = ("ft", "b0", "b1", "b4", "b2")
TEST_LOG, E2E_LOG = "data/eval/test-access.jsonl", "data/eval/e2e-test-access.jsonl"
COMPLETIONS = "data/eval/test-completions.jsonl"
PURPOSE = "M3 phase 4: FT against B0, B1, B4 and B2 on the locked test sets (a test)"

# ---- the sets -------------------------------------------------------------------------------------------

SYSTEM = {"role": "system", "content": system_prompt(TODAY)}
SEARCH = {"name": "search_arxiv", "arguments": {"query": "hallucination", "max_results": 5},
          "arguments_json": '{"query": "hallucination", "max_results": 5}'}
FIND = {"content": "", "tool_calls": [SEARCH]}
BYE = {"content": "See you!", "tool_calls": []}


def item(id_: str, user: str, expected: dict, type_: str, lang: str, split: str, *,
         handwritten: bool = False) -> dict:
    source = {"handwritten": True} if handwritten else {"request": id_.rsplit("-s", 1)[0], "step": 1}
    family = f"handwritten/{type_}/{lang}" if handwritten else f"find/{lang}"
    return {"id": id_, "messages": [SYSTEM, {"role": "user", "content": user}], "expected": expected,
            "type": type_, "lang": lang, "family": family, "split": split,
            "slots": {"topic": "hallucination"}, "source": source}


TEST_ITEMS = [item("req-test-0001-s1", "Find hallucination papers.", FIND, "call", "en", "test"),
              item("req-test-0002-s1", "找几篇幻觉(hallucination)的论文。", FIND, "call", "zh", "test"),
              item("hw-test-001", "Thanks, that's all for today.", BYE, "no_call", "en", "test",
                   handwritten=True),
              item("hw-test-002", "Search for hallucination papers, please.", FIND, "call", "en", "test",
                   handwritten=True)]
DEV_ITEMS = [item("req-dev-0001-s1", "Look for hallucination papers.", FIND, "call", "en", "dev"),
             item("hw-dev-001", "Thanks, that's all.", BYE, "no_call", "en", "dev", handwritten=True)]
ID1, ID2 = "2609.00001", "2609.00002"
EXPECT = {"status": "answered", "must_write_note": False, "note_about": None,
          "must_cite_ids_within": "tool_outputs", "expected_tools_any": ["get_paper"]}


def task(id_: str, text: str, arxiv_id: str, lang: str, split: str) -> dict:
    return {"id": id_, "intent": "lookup_paper", "scenario": "normal", "lang": lang, "split": split,
            "family": "e2e/lookup_paper", "text": text, "slots": {"arxiv_id": arxiv_id}, "expect": EXPECT}


TEST_TASKS = [task("e2e-test-01", f"What is {ID2} about?", ID2, "en", "test"),
              task("e2e-test-02", f"{ID1} 这篇论文讲什么?", ID1, "zh", "test")]
DEV_TASKS = [task("e2e-dev-01", f"What is {ID1} about?", ID1, "en", "dev")]
DEV_TEXTS = {"Look for hallucination papers.", "Thanks, that's all.", f"What is {ID1} about?"}

# ---- the world of the end-to-end tasks --------------------------------------------------------------------

CORPUS = [{"arxiv_id": ID1, "title": "Adaptive Draft Length for Speculative Decoding",
           "authors": ["A. Author"], "published": "2026-09-15",
           "abstract": "Speculative decoding drafts tokens with a small model.", "categories": ["cs.CL"]},
          {"arxiv_id": ID2, "title": "Tree Verification at Scale", "authors": ["A. Author"],
           "published": "2026-09-20", "categories": ["cs.CL"],
           "abstract": "Tree-structured verification checks drafted tokens at once."}]


class Pages:
    """The replay fetcher: each paper's metadata as a JSON feed and a full-text page."""

    def get(self, url: str) -> Fetched:
        for p in CORPUS:
            if url == paper_url(p["arxiv_id"]):
                return Fetched(url=url, final_url=url, status=200, body=json.dumps([p]).encode(),
                               content_type="")
            if url == fulltext_url(p["arxiv_id"]):
                return Fetched(url=url, final_url=url, status=200, body=b"<html><p>Method text.</p></html>",
                               content_type="")
        raise ReplayMiss(f"no cassette for {url}")


def parse_feed(xml: str) -> list[dict] | None:
    try:
        return json.loads(xml)
    except json.JSONDecodeError:
        return None


def world() -> World:
    owner = OwnerFunctions(check_call, decide, parse_feed, lambda html, section, n: "Method text.")
    return World(owner, Pages(), CorpusIndex(CORPUS), None, {})


# ---- the machine ----------------------------------------------------------------------------------------

CALL = ('<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "hallucination", "max_results": 5}}\n'
        '</tool_call>')
OFF = '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "cats"}}\n</tool_call>'
USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
ARXIV = re.compile(r"\d{4}\.\d{5}")
SAMPLER_LINES = (["4100, 80.0\n", "4600, 120.0\n"], ["4000, 70.0\n", "4400, 110.0\n"])
UNSURE = ("please", "幻觉", "Look for", "讲什么")  # requests the local model answers with probability 0.2


def confidence(prompt: str) -> float:
    """The probability of every local answer: low for the requests of UNSURE (ftr routes them), high
    otherwise."""
    user = USER.findall(prompt)[-1]
    return 0.2 if any(word in user for word in UNSURE) else 0.95


class Machine:
    """This machine as run_test meets it, faked. popen starts llama-server (with the base model its -m and
    the adapter its --lora names; one whose path holds failing exits at once) or nvidia-smi's sampling loop;
    run answers nvidia-smi's one-off query with idle_mib; the HTTP side answers /health, /lora-adapters,
    /props (model_path, by default the base model the server was started with) and the completions. Every
    local answer depends on the system the request is for: the adapter loaded and the lora scale it sends, b1
    told from b0 by the demonstrations block. FT and FT3 (the M7 adapter) answer every single-step item
    right, FT2 (the M6 adapter) and B4 all but the no-call one, B1 only that one, B0 none; on an end-to-end
    task all but B0 look the paper up and then answer. Every answer comes as one token with its bytes and the
    log-probability of confidence(prompt), which ftr, ft2r and ft3r route on."""

    def __init__(self, *, idle_mib: float = 512.0, failing: str | None = None,
                 model_path: str | None = None, slots: int = 1) -> None:
        self.idle_mib, self.failing, self.model_path, self.slots = idle_mib, failing, model_path, slots
        self.started: list[list[str]] = []
        self.adapter: str | None = None
        self.model: str | None = None  # the base model of the server running
        self.process: ServerProcess | None = None
        self.samplers = 0
        self.prompts: list[tuple[str, str]] = []  # (system, prompt) of every completion

    def popen(self, cmd, **kwargs):
        self.started.append(list(cmd))
        if cmd[0] == "nvidia-smi":
            self.samplers += 1
            return ServerProcess(SAMPLER_LINES[(self.samplers - 1) % 2])
        self.adapter, self.model = cmd[cmd.index("--lora") + 1], cmd[cmd.index("-m") + 1]
        failing = self.failing is not None and self.failing in self.adapter
        self.process = ServerProcess([] if failing else [READY_LINE], exit_code=1 if failing else None)
        return self.process

    def run(self, cmd, **kwargs):
        assert cmd[0] == "nvidia-smi", cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{self.idle_mib:g}, 12.0\n", stderr="")

    def servers(self) -> list[str]:
        return [cmd[cmd.index("--lora") + 1] for cmd in self.started if cmd[0] != "nvidia-smi"]

    def models(self) -> list[str]:
        """The base model of every server started, in order."""
        return [cmd[cmd.index("-m") + 1] for cmd in self.started if cmd[0] != "nvidia-smi"]

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.process is None or self.process.returncode is not None:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/lora-adapters":
            return httpx.Response(200, json=[{"id": 0, "path": self.adapter, "scale": 1.0}])
        if path == "/props":
            return httpx.Response(200, json={"build_info": "b11349-fb4b2737a",
                                             "model_path": self.model_path or self.model,
                                             "total_slots": self.slots})
        payload = json.loads(request.content)
        text = self.answer(payload["prompt"], payload["lora"][0]["scale"])
        token = {"token": text, "logprob": math.log(confidence(payload["prompt"])),
                 "bytes": list(text.encode("utf-8"))}
        return httpx.Response(200, json={"choices": [{"text": text, "finish_reason": "stop",
                                                      "logprobs": {"content": [token]}}],
                                         "usage": {"prompt_tokens": 900, "completion_tokens": 20},
                                         "timings": {"prompt_ms": 80.0, "predicted_ms": 20.0}})

    def answer(self, prompt: str, scale: float) -> str:
        if scale == 1.0:
            system = next((name for name in ("ft3", "ft2", "ft") if f"/{name}/" in self.adapter), "b4")
        else:
            system = "b1" if DEMONSTRATIONS_HEADING in prompt else "b0"
        self.prompts.append((system, prompt))
        user = USER.findall(prompt)[-1]
        ids = ARXIV.findall(user)
        if ids:  # an end-to-end task
            if system == "b0":
                return "I cannot tell."
            if "<|im_start|>tool" in prompt:
                return f"{ids[0]} is about tree verification."
            lookup = json.dumps({"name": "get_paper", "arguments": {"arxiv_id": ids[0]}})
            return f"<tool_call>\n{lookup}\n</tool_call>"
        no_call = "that's all" in user
        # ft2, the M6 adapter, answers every single-step item with the call: right but for the goodbyes
        answers = {"ft": BYE["content"] if no_call else CALL, "ft2": CALL, "b4": CALL, "b1": BYE["content"],
                   "b0": OFF, "ft3": BYE["content"] if no_call else CALL}
        return answers[system]

    def client(self, url: str) -> LocalClient:
        return LocalClient(url, transport=httpx.MockTransport(self.handler))

    def users(self) -> set[str]:
        """The user messages every local prompt carried."""
        return {USER.findall(prompt)[-1] for _, prompt in self.prompts}


class Response:
    def __init__(self, data: dict) -> None:
        self.data = data

    def model_dump(self, **kwargs) -> dict:
        return self.data


class Cloud:
    """DeepSeek's SDK, faked: B2 (or B3, the response naming the model asked) answers every single-step item
    right but the Chinese one, and looks every end-to-end task's paper up before answering. Every request is
    recorded."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **payload) -> Response:
        self.requests.append(payload)
        messages = payload["messages"]
        user = [m["content"] for m in messages if m["role"] == "user"][-1]
        ids = ARXIV.findall(user)
        if ids and messages[-1]["role"] == "tool":
            message = {"role": "assistant", "content": f"{ids[0]} is about tree verification."}
        elif ids:
            message = {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_7", "type": "function",
                 "function": {"name": "get_paper", "arguments": json.dumps({"arxiv_id": ids[0]})}}]}
        elif "that's all" in user:
            message = {"role": "assistant", "content": "See you!"}
        else:
            arguments = json.dumps({"query": "cats" if "幻觉" in user else "hallucination", "max_results": 5})
            message = {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_3", "type": "function",
                 "function": {"name": "search_arxiv", "arguments": arguments}}]}
        return Response({"id": "chatcmpl-1", "model": payload["model"],
                         "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                         "usage": {"prompt_tokens": 1000, "completion_tokens": 50}})

    def users(self) -> set[str]:
        return {[m["content"] for m in r["messages"] if m["role"] == "user"][-1] for r in self.requests}


def ticking(step: float = 0.25):
    """A clock that moves step seconds forward every time it is read."""
    now = [0.0]

    def clock() -> float:
        now[0] += step
        return now[0]

    return clock


def session(machine: Machine, cloud: Cloud, *extra: str, config: str = CONFIG, binary: str = BINARY,
            load_render=lambda folder: prompt_renderer(FakeTokenizer()),
            load_counter=lambda folder: None, load_family=None) -> int:
    """run_test.main on the fake machine; load_family, when given, loads the family of a family-path server's
    tokenizer folder (run_test's own, research_agent.family.load_family, otherwise)."""
    family = {} if load_family is None else {"load_family": load_family}
    return run_test.main(["--config", config, "--llama-server", binary, *extra], popen=machine.popen,
                         run=machine.run, client=machine.client, openai_client=cloud, load_render=load_render,
                         load_counter=load_counter, load_world=world, now=lambda: NOW, clock=ticking(),
                         **family)


def dry_run(config: str = CONFIG, expect: int = 0) -> None:
    """The dry run a real session needs first (of the same commit and configuration), on a machine of its
    own."""
    assert session(Machine(), Cloud(), "--dry-run", config=config) == expect


# ---- the repository ---------------------------------------------------------------------------------------

# the fake repositories' commits are the same bytes, and so the same hashes, on every machine: no signature
# either (the cloud check's git signs every commit unless told not to)
IDENTITY = ["-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "core.autocrlf=false",
            "-c", "commit.gpgsign=false"]
DATES = {"GIT_AUTHOR_DATE": "2026-10-16T00:00:00+00:00", "GIT_COMMITTER_DATE": "2026-10-16T00:00:00+00:00"}


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    done = subprocess.run(["git", "-C", str(root), *IDENTITY, *args], capture_output=True, text=True,
                          env={**os.environ, **DATES}, check=False)
    assert done.returncode == 0, done.stderr
    return done


def commit(root: Path, message: str = "a change") -> None:
    git(root, "add", "-A")
    if git(root, "status", "--porcelain").stdout.strip():
        git(root, "commit", "-q", "-m", message)


def write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8", newline="\n")
    else:
        path.write_bytes(data)
    return path


def locked(path: Path, rows: list[dict]) -> None:
    write_jsonl(rows, path)
    write_lock(path)


def demonstrations(root: Path, answer: str = "No: query is required and has no default.") -> str:
    """The b1 demonstrations (fake training set and requests file) under root; their block's hash."""
    (call_id, _), (ask_id, _), (answer_id, _) = DEMONSTRATIONS
    system = {"role": "system", "content": "你是科研助理。"}

    def record(rid: str, user: str, assistant: dict) -> dict:
        return {"id": rid, "run": "train", "history": "agent",
                "messages": [system, {"role": "user", "content": user}, assistant]}

    def calling(name: str, arguments: dict) -> dict:
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}

    rows = [record(call_id, "搜 KV cache 的论文", calling("search_arxiv", {"query": "KV cache"})),
            record(ask_id, "Note the paper I just closed.",
                   calling("ask_clarification", {"question": "Which paper?", "missing": ["paper_id"]})),
            record(answer_id, "Does query have a default?", {"role": "assistant", "content": answer})]
    training = write(root / "data/raw/synth/synth.jsonl",
                     "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    requests = write(root / "data/synth/requests.jsonl",
                     "".join(json.dumps({"id": rid, "split": "train"}) + "\n" for rid, _ in DEMONSTRATIONS))
    return block_hash(demonstrations_block(load_demonstrations(training, requests)))


def configuration(root: Path, block: str, **changes) -> dict:
    config = {
        "milestone": "M3", "purpose": PURPOSE, "systems": list(SYSTEMS),
        "adapters": {"ft": {"gguf": FT_GGUF, "sha256": sha256_of(root / FT_GGUF), "selection": None},
                     "b4": {"gguf": B4_GGUF, "sha256": sha256_of(root / B4_GGUF), "selection": None}},
        "server": {"model": "models/Qwen3-1.7B-Q8_0.gguf", "port": 8080, "ctx_size": 8192, "timeout_s": 30,
                   "llama_cpp_build": "b11349-fb4b2737a"},
        "tokenizer": "models/Qwen3-1.7B", "b1_block_sha256_16": block, "b2": {"max_cost_rmb": 5.0},
        "e2e": {"seed": 42, "today": TODAY}, "statistics": {"seed": 42, "resamples": 200},
        "comparisons": [{"name": "ft-vs-b0", "a": "ft", "b": "b0"},
                        {"name": "ft-vs-b1", "a": "ft", "b": "b1"},
                        {"name": "ft-vs-b4", "a": "ft", "b": "b4"},
                        {"name": "ftr-vs-b2", "a": "ft+r", "b": "b2", "deferred": "FT+R is built in M4",
                         "p": 1.0}],
        "gpu": {"idle_max_mib": 1500, "sample_ms": 200},
    }
    config.update(changes)
    return config


def make_repository(root: Path, *, items: list[dict] = TEST_ITEMS, **changes) -> None:
    write(root / ".gitignore", "runs/\nmodels/\ndata/raw/\n")
    write(root / "README.md", "A fake repository for scripts/run_test.py.\n")
    eval_dir = root / "data" / "eval"
    locked(eval_dir / "dev.jsonl", DEV_ITEMS)
    locked(eval_dir / "e2e-dev.jsonl", DEV_TASKS)
    locked(eval_dir / "test.jsonl", items)
    locked(eval_dir / "e2e-test.jsonl", TEST_TASKS)
    block = demonstrations(root)
    write(root / "models/Qwen3-1.7B-Q8_0.gguf", b"GGUF base")
    (root / "models" / "Qwen3-1.7B").mkdir(parents=True)
    write(root / FT_GGUF, b"GGUF ft step 418")
    write(root / B4_GGUF, b"GGUF b4 step 278")
    write(root / BINARY, b"MZ llama-server")
    config = configuration(root, block, **changes)
    write(root / CONFIG, json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    git(root, "init", "-q")
    commit(root, "the fake repository")


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """The fake repository as the current folder and run_test's repository root."""
    make_repository(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_test, "REPO", tmp_path)
    return tmp_path


def result(root: Path, name: str) -> dict:
    return json.loads((root / "results" / name).read_text(encoding="utf-8"))


def logs(root: Path) -> dict[str, bytes | None]:
    """The two access logs and the completion markers as bytes; None for one that does not exist."""
    paths = (TEST_LOG, E2E_LOG, COMPLETIONS)
    return {p: (root / p).read_bytes() if (root / p).exists() else None for p in paths}


# ---- a session --------------------------------------------------------------------------------------------


def test_a_session_opens_each_test_set_once_and_runs_ft_b0_b1_then_b4_then_b2(root, capsys):
    dry_run()
    machine, cloud = Machine(), Cloud()
    assert session(machine, cloud) == 0
    head = git(root, "rev-parse", "HEAD").stdout.strip()
    purpose = f"{PURPOSE} (git commit {head}; config {CONFIG} sha256 {sha256_of(root / CONFIG)})"
    opens = {}
    for log, test_set in ((TEST_LOG, "data/eval/test.jsonl"), (E2E_LOG, "data/eval/e2e-test.jsonl")):
        (entry,) = read_jsonl(root / log)  # exactly one open of each set
        assert entry == {"milestone": "M3", "purpose": purpose, "sha256": sha256_of(root / test_set),
                         "ts": NOW}
        opens[log] = entry
    assert machine.servers() == [FT_GGUF, B4_GGUF]  # one server after the other
    assert all(cmd[cmd.index("-np") + 1] == "1" for cmd in machine.started if cmd[0] != "nvidia-smi")
    assert list(dict.fromkeys(system for system, _ in machine.prompts)) == ["ft", "b0", "b1", "b4"]
    assert machine.users() | cloud.users() == {i["messages"][1]["content"] for i in TEST_ITEMS} | {
        t["text"] for t in TEST_TASKS}  # test material only, every item and task of it

    rates = {s: result(root, f"test-m3-{s}.json")["summary"]["success_rate"] for s in SYSTEMS}
    assert rates == {"ft": 1.0, "b0": 0.0, "b1": 0.25, "b4": 0.75, "b2": 0.75}
    b4 = result(root, "test-m3-b4.json")
    assert b4["open"] == opens[TEST_LOG] and b4["set"] == {"path": "data/eval/test.jsonl",
                                                          "sha256": opens[TEST_LOG]["sha256"], "n": 4}
    subsets = b4["subsets"]
    assert (subsets["drafted"]["success_rate"], subsets["handwritten"]["success_rate"]) == (1.0, 0.5)
    assert b4["info"]["adapters"] == [{"id": 0, "path": B4_GGUF, "sha256": sha256_of(root / B4_GGUF)}]
    assert b4["records"] == "runs/test-m3/open-1/b4.jsonl" and (root / b4["records"]).is_file()
    b1 = result(root, "test-m3-b1.json")
    assert b1["info"]["demonstrations"]["block_sha256_16"] == json.loads(
        (root / CONFIG).read_text(encoding="utf-8"))["b1_block_sha256_16"]
    assert result(root, "test-m3-b2.json")["modes"] == {"thinking": 4}
    e2e = {s: result(root, f"test-m3-e2e-{s}.json") for s in SYSTEMS}
    assert {s: r["summary"]["success_rate"] for s, r in e2e.items()} == {"ft": 1.0, "b0": 0.0, "b1": 1.0,
                                                                         "b4": 1.0, "b2": 1.0}
    assert e2e["ft"]["open"] == opens[E2E_LOG] and e2e["b2"]["modes"] == {"non-thinking": 2, "thinking": 2}

    comparisons = result(root, "test-m3-comparisons.json")
    assert [(c["name"], c["p"], c["p_holm"]) for c in comparisons["comparisons"]] == [
        ("ft-vs-b0", 0.125, 0.5), ("ft-vs-b1", 0.25, 0.75), ("ft-vs-b4", 1.0, 1.0), ("ftr-vs-b2", 1.0, 1.0)]
    first = comparisons["comparisons"][0]
    assert (first["n"], first["diff"], first["mcnemar"]["a_only"]) == (4, 1.0, 4)
    assert first["cluster_ci"]["clusters"] == 4  # two drafted families and two handwritten items
    resources = result(root, "test-m3-resources.json")
    assert resources["servers"]["ft"]["systems"] == ["ft", "b0", "b1"]
    servers = resources["servers"]
    assert (servers["ft"]["vram_mib"], servers["b4"]["vram_mib"]) == (4088.0, 3888.0)  # the peaks over 512
    assert resources["systems"]["ft"]["single_step"]["n"] == 4 and resources["systems"]["ft"]["cost"] is None
    b2 = resources["systems"]["b2"]
    assert b2["single_step"]["basis"] == "paid calls" and b2["single_step"]["n"] == 4
    assert b2["cost"]["paid_rmb"] == pytest.approx(8 * (1000 * 2.0 + 50 * 8.0) / 1e6)
    assert b2["cost"]["ledger_rmb"] == pytest.approx(b2["cost"]["paid_rmb"])

    markers = read_jsonl(root / COMPLETIONS)
    assert [(m["access_log"], m["open"], m["status"]) for m in markers] == [
        (TEST_LOG, opens[TEST_LOG], "complete"), (E2E_LOG, opens[E2E_LOG], "complete")]
    written = sorted(f"results/{p.name}" for p in (root / "results").glob("test-m3-*.json"))
    assert sorted(markers[0]["outputs"]) == written and len(written) == 2 * len(SYSTEMS) + 2
    assert all(sha256_of(root / path) == digest for path, digest in markers[0]["outputs"].items())
    numbers, _ = report.build(root)
    assert numbers["eval.test-m3-ft.success"] == "100.0%" and numbers["e2e.test-m3-b0.success"] == "0.0%"
    assert numbers["compare.test-m3.ft-vs-b0.p_holm"] == "0.500"
    assert "open 1 of 3" in capsys.readouterr().out


def test_the_dry_run_runs_one_development_item_and_task_per_system_and_never_reads_a_test_set(root):
    for name in ("test.jsonl", "e2e-test.jsonl"):
        path = root / "data" / "eval" / name
        path.write_text("not json: never parsed\n", encoding="utf-8")  # a parse would fail
        write_lock(path)
    commit(root, "test sets nobody may parse")
    machine, cloud = Machine(), Cloud()
    assert session(machine, cloud, "--dry-run") == 0
    assert logs(root) == {TEST_LOG: None, E2E_LOG: None, COMPLETIONS: None}  # nothing opened
    assert machine.users() | cloud.users() <= DEV_TEXTS and machine.users() and cloud.users()
    assert machine.servers() == [FT_GGUF, B4_GGUF]
    dry = root / "runs" / "test-m3" / "dry-run"
    for system in SYSTEMS:
        assert [r["id"] for r in read_jsonl(dry / f"{system}.jsonl")] == ["req-dev-0001-s1"]
        assert (dry / f"e2e-{system}" / "transcripts" / "e2e-dev-01.json").is_file()
        written = json.loads((dry / "results" / f"test-m3-{system}.json").read_text(encoding="utf-8"))
        assert written["open"] is None
    assert not list((root / "results").glob("test-m3-*.json"))
    assert git(root, "status", "--porcelain").stdout == ""  # the dry run leaves the tree clean


# ---- the preflight ----------------------------------------------------------------------------------------


def dirty(root: Path, machine: Machine) -> str:
    (root / "README.md").write_text("changed\n", encoding="utf-8")
    return CONFIG


def uncommitted_config(root: Path, machine: Machine) -> str:
    other = "results/test-config-m3-copy.json"
    shutil.copyfile(root / CONFIG, root / other)
    return other


def missing_adapter(root: Path, machine: Machine) -> str:
    (root / FT_GGUF).unlink()
    return CONFIG


def changed_adapter(root: Path, machine: Machine) -> str:
    (root / B4_GGUF).write_bytes(b"GGUF b4, retrained")
    return CONFIG


def other_demonstrations(root: Path, machine: Machine) -> str:
    demonstrations(root, answer="Yes.")
    commit(root, "requests")  # requests.jsonl is tracked; the training set is not
    return CONFIG


def spent_budget(root: Path, machine: Machine) -> str:
    budget = Budget(root / "runs" / "llm" / "budget.sqlite", stage="M3", stage_cap_rmb=30.0)
    budget.charge("deepseek-flash", {"prompt_tokens": 13_000_000, "completion_tokens": 0})  # 26 RMB
    budget.close()
    return CONFIG


def changed_set(name: str):
    def change(root: Path, machine: Machine) -> str:
        path = root / "data" / "eval" / name
        path.write_bytes(path.read_bytes() + b"\n")  # committed, the lock left as it was
        commit(root, f"{name} changed")
        return CONFIG

    return change


def used_opens(log: str):
    def use(root: Path, machine: Machine) -> str:
        entries = [{"milestone": "M2", "purpose": f"open {k}", "sha256": "0" * 64, "ts": NOW}
                   for k in range(3)]
        write_jsonl(entries, root / log)
        commit(root, f"{log} used up")
        return CONFIG

    return use


def busy_gpu(root: Path, machine: Machine) -> str:
    machine.idle_mib = 5200.0
    return CONFIG


def busy_port(root: Path, machine: Machine) -> str:
    machine.process = ServerProcess([])  # a server left running: it answers /health
    return CONFIG


def select(root: Path, choices: dict[str, tuple[str, str]], message: str) -> str:
    """Selection files that chose (gguf, sha256) per system, named in the committed configuration."""
    config = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    for system, (gguf, digest) in choices.items():
        selection = f"results/checkpoint-selection-{system}.json"
        write(root / selection, json.dumps({"chosen": {"step": 209, "gguf": gguf, "sha256": digest}}))
        config["adapters"][system]["selection"] = selection
    write(root / CONFIG, json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    commit(root, message)
    return CONFIG


def other_selection(root: Path, machine: Machine) -> str:
    return select(root, {"ft": ("runs/train/ft/gguf/checkpoint-209-f16.gguf", "1" * 64)},
                  "a selection that chose another checkpoint")


@pytest.mark.parametrize("change, message", [
    (dirty, "the working tree has uncommitted changes"),
    (uncommitted_config, "results/test-config-m3-copy.json is not committed"),
    (missing_adapter, f"the ft adapter {FT_GGUF} is missing"),
    (changed_adapter, f"the b4 adapter {B4_GGUF} has SHA-256"),
    (other_demonstrations, "b1's demonstrations block hashes to"),
    (spent_budget, "the M3 API budget cannot take B2's cap of 5 RMB"),
    (changed_set("test.jsonl"), "test.jsonl does not match its lock"),
    (changed_set("e2e-test.jsonl"), "e2e-test.jsonl does not match its lock"),
    (used_opens(TEST_LOG), "data/eval/test.jsonl has no open left"),
    (used_opens(E2E_LOG), "data/eval/e2e-test.jsonl has no open left"),
    (busy_gpu, "the GPU is not idle: 5200 MiB in use"),
    (busy_port, "a server already answers at http://127.0.0.1:8080"),
    (other_selection, "is not the checkpoint results/checkpoint-selection-ft.json chose"),
])
def test_each_preflight_check_refuses_with_exit_4_before_anything_is_opened_or_started(root, capsys, change,
                                                                                         message):
    dry_run()
    machine, cloud = Machine(), Cloud()
    config = change(root, machine)
    before = logs(root)
    assert session(machine, cloud, config=config) == 4
    assert message in capsys.readouterr().out
    assert logs(root) == before and machine.started == [] and cloud.requests == []
    assert not (root / "runs" / "test-m3" / "open-1").exists()


def test_a_missing_llama_server_or_model_is_a_usage_error_with_exit_2(root, capsys):
    machine, cloud = Machine(), Cloud()
    assert session(machine, cloud, binary="runs/bin/nothing.exe") == 2
    assert "llama-server not found" in capsys.readouterr().out
    (root / "models" / "Qwen3-1.7B-Q8_0.gguf").unlink()
    assert session(machine, cloud) == 2
    assert "models/Qwen3-1.7B-Q8_0.gguf" in capsys.readouterr().out
    assert logs(root) == {TEST_LOG: None, E2E_LOG: None, COMPLETIONS: None} and machine.started == []


def test_a_server_with_more_than_one_slot_stops_the_session_before_any_system_runs_on_it(root, capsys):
    dry_run()
    machine = Machine(slots=4)  # the slots would share one KV cache: an answer could depend on the one before
    assert session(machine, Cloud()) == 6
    out = capsys.readouterr().out
    assert "the ft server reports 4 slots (/props), not 1" in out and "the open is used up" in out
    assert machine.prompts == [] and machine.servers() == [FT_GGUF]


# ---- an unfinished open -----------------------------------------------------------------------------------


def test_an_open_without_a_completion_marker_refuses_the_next_session_until_the_owner_abandons_it(root,
                                                                                                    capsys):
    dry_run()
    machine, cloud = Machine(failing="/b4/"), Cloud()
    assert session(machine, cloud) == 6  # the B4 server dies after FT, B0 and B1 ran on the test sets
    out = capsys.readouterr().out
    assert "exited with 1 before becoming ready" in out and "the open is used up" in out
    assert [len(read_jsonl(root / log)) for log in (TEST_LOG, E2E_LOG)] == [1, 1]
    assert not (root / COMPLETIONS).exists() and not list((root / "results").glob("test-m3-*.json"))

    before, machine = logs(root), Machine()
    assert session(machine, cloud) == 5  # not resumed: the owner decides (the M3 plan, default 5)
    out = capsys.readouterr().out
    assert f"{TEST_LOG}: the M3 open of {NOW} has no completion marker" in out
    assert "--abandon" in out and logs(root) == before and machine.started == []

    assert session(machine, cloud, "--abandon", "the B4 server would not start") == 0
    markers = read_jsonl(root / COMPLETIONS)
    assert [(m["access_log"], m["status"], m["reason"]) for m in markers] == [
        (TEST_LOG, "abandoned", "the B4 server would not start"),
        (E2E_LOG, "abandoned", "the B4 server would not start")]
    assert markers[0]["open"] == read_jsonl(root / TEST_LOG)[0] and machine.started == []
    commit(root, "the crashed session, abandoned")
    dry_run()  # of the new commit
    assert session(Machine(), Cloud()) == 0  # a new session, a new open
    assert [len(read_jsonl(root / log)) for log in (TEST_LOG, E2E_LOG)] == [2, 2]
    assert [m["status"] for m in read_jsonl(root / COMPLETIONS)] == ["abandoned", "abandoned", "complete",
                                                                     "complete"]
    assert (root / "runs" / "test-m3" / "open-2" / "ft.jsonl").is_file()


def test_an_unexpected_error_after_the_opens_exits_6_and_says_the_open_is_used_up(root, capsys, monkeypatch):
    def full_disk(data: dict, path: Path) -> None:
        raise OSError(28, "No space left on device", str(path))

    dry_run()
    monkeypatch.setattr(run_test, "write_json", full_disk)  # first called when the results are written
    assert session(Machine(), Cloud()) == 6
    out = capsys.readouterr().out
    assert "OSError" in out and "the open is used up" in out and "--abandon" in out
    assert [len(read_jsonl(root / log)) for log in (TEST_LOG, E2E_LOG)] == [1, 1]
    assert not (root / COMPLETIONS).exists()


def test_the_advice_after_a_crash_names_both_access_logs_and_the_completion_markers(root, capsys):
    dry_run()
    assert session(Machine(failing="/b4/"), Cloud()) == 6
    capsys.readouterr()
    assert session(Machine(), Cloud()) == 5
    advice = capsys.readouterr().out.split("--abandon", 1)[1]
    assert all(path in advice for path in (TEST_LOG, E2E_LOG, COMPLETIONS))
    assert session(Machine(), Cloud(), "--abandon", "the B4 server would not start") == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert all(path in last for path in (TEST_LOG, E2E_LOG, COMPLETIONS))


def test_a_completion_marker_whose_open_its_access_log_no_longer_holds_is_refused(root, capsys):
    dry_run()
    assert session(Machine(), Cloud()) == 0  # one complete session: an open and a marker per set
    commit(root, "the session's access logs, markers and results")
    (root / E2E_LOG).unlink()  # e.g. a git clean of an access log: its opens would count again
    commit(root, "the e2e access log lost")
    before = logs(root)
    for extra in ((), ("--abandon", "anything")):
        machine = Machine()
        assert session(machine, Cloud(), *extra) == 4
        out = capsys.readouterr().out
        assert E2E_LOG in out and "no longer holds" in out
        assert logs(root) == before and machine.started == []


def test_abandon_run_from_another_folder_is_a_usage_error(root, capsys, monkeypatch):
    (root / "runs").mkdir(exist_ok=True)
    monkeypatch.chdir(root / "runs")
    assert run_test.main(["--config", str(root / CONFIG), "--abandon", "a reason"]) == 2
    assert "run from the repository root" in capsys.readouterr().out


def test_a_dry_run_passes_with_selections_that_chose_the_configured_adapters(root):
    select(root, {"ft": (FT_GGUF, sha256_of(root / FT_GGUF)), "b4": (B4_GGUF, sha256_of(root / B4_GGUF))},
           "the selections")
    assert session(Machine(), Cloud(), "--dry-run") == 0


def test_test_items_the_comparisons_could_not_cluster_stop_the_session_before_any_model_runs(tmp_path,
                                                                                               monkeypatch,
                                                                                               capsys):
    sourceless = [{k: v for k, v in TEST_ITEMS[0].items() if k != "source"}, *TEST_ITEMS[1:]]
    make_repository(tmp_path, items=sourceless)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_test, "REPO", tmp_path)
    dry_run()
    machine, cloud = Machine(), Cloud()
    assert session(machine, cloud) == 6
    out = capsys.readouterr().out
    assert "the test material is malformed" in out and "cannot tell whether it is drafted" in out
    assert "the open is used up" in out and len(read_jsonl(tmp_path / TEST_LOG)) == 1
    assert machine.prompts == [] and cloud.requests == [] and machine.servers() == []


# ---- the cap and the outputs ------------------------------------------------------------------------------


def test_the_results_hold_the_model_path_a_server_reports_relative_to_the_repository(root):
    dry_run()
    machine = Machine(model_path=str(root / "models" / "Qwen3-1.7B-Q8_0.gguf"))  # as a machine path
    assert session(machine, Cloud()) == 0
    for system in ("ft", "b0", "b4"):
        assert result(root, f"test-m3-{system}.json")["server"] == {
            "build_info": "b11349-fb4b2737a", "model_path": "models/Qwen3-1.7B-Q8_0.gguf"}
    assert result(root, "test-m3-b2.json")["server"] is None
    assert run_test.repository_path(str(root.parent / "elsewhere" / "base.gguf")) == "base.gguf"


def test_a_machine_path_in_either_systems_spelling_is_reduced_to_its_file_name():
    for path in ("C:\\models\\ft.gguf", "D:/runs/ft.gguf", "/opt/models/ft.gguf"):
        assert run_test.repository_path(path) == "ft.gguf", path
    assert run_test.repository_path("models/ft.gguf") == "models/ft.gguf"


@pytest.mark.parametrize("flavour", [PurePosixPath, PureWindowsPath])
def test_a_relative_path_in_either_systems_spelling_is_stored_with_forward_slashes(flavour, monkeypatch):
    # Path is the system's own flavour: on Linux a backslash is a character of a name, not a separator
    monkeypatch.setattr(run_test, "Path", flavour)
    for path in ("models\\ft.gguf", "models/ft.gguf", "runs\\train/ft.gguf"):
        assert run_test.repository_path(path) == path.replace("\\", "/"), path


def test_a_b2_run_stopped_by_its_cap_writes_every_result_and_exits_3(tmp_path, monkeypatch):
    make_repository(tmp_path, b2={"max_cost_rmb": 0.003})  # one paid turn costs 0.0024 RMB
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_test, "REPO", tmp_path)
    dry_run(expect=3)  # b2's single development item and task spend the cap too
    assert session(Machine(), Cloud()) == 3
    statuses = [r["status"] for r in read_jsonl(tmp_path / "runs" / "test-m3" / "open-1" / "b2.jsonl")]
    assert statuses == ["ok", "budget_stop", "skipped", "skipped"]
    assert result(tmp_path, "test-m3-e2e-b2.json")["summary"]["statuses"] == {"budget_stop": 1}
    assert [m["status"] for m in read_jsonl(tmp_path / COMPLETIONS)] == ["budget_stop", "budget_stop"]
    assert result(tmp_path, "test-m3-comparisons.json")["comparisons"][0]["p"] == 0.125


def test_two_sessions_over_the_same_answers_write_the_same_bytes(tmp_path, monkeypatch):
    outputs = {}
    for name in ("first", "second"):
        root = tmp_path / name
        make_repository(root)
        monkeypatch.chdir(root)
        monkeypatch.setattr(run_test, "REPO", root)
        dry_run()
        assert session(Machine(), Cloud()) == 0
        files = [*sorted((root / "results").glob("test-m3-*.json")), root / TEST_LOG, root / E2E_LOG,
                 root / COMPLETIONS]
        outputs[name] = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
    assert outputs["first"] == outputs["second"]
    for path, data in outputs["first"].items():
        assert b"\r\n" not in data and data.endswith(b"\n"), path
        assert b"wall_ms" not in data or path.endswith("resources.json"), path  # times only in resources


# ---- the configuration -----------------------------------------------------------------------------------


@pytest.mark.parametrize("changes, message", [
    ({"systemz": ["ft"]}, "unknown key systemz"),
    ({"gpu": {"idle_max_mib": 1500}}, "gpu lacks sample_ms"),
    ({"systems": ["ft", "b0", "b3"]}, "systems: b3 is not one of"),
    ({"systems": []}, "systems must be a non-empty list"),
    ({"comparisons": [{"name": "ft-vs-b4", "a": "ft", "b": "b4"}], "systems": ["ft", "b0"],
      "adapters": {"ft": {"gguf": FT_GGUF, "sha256": "0" * 64, "selection": None}}},
     "comparisons: ft-vs-b4 compares b4, which is not among the systems"),
    ({"tokenizer": "/models/Qwen3-1.7B"}, "tokenizer must be a path relative to the repository root"),
    ({"tokenizer": "../models/Qwen3-1.7B"}, "tokenizer must be a path relative to the repository root"),
    ({"b2": {"max_cost_rmb": 0}}, "b2.max_cost_rmb must be a number above 0"),
    ({"milestone": ["M3"]}, "milestone must be one of"),
])
def test_a_malformed_configuration_is_refused_with_exit_1_naming_the_key(root, capsys, changes, message):
    config = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    config.update(changes)
    write(root / CONFIG, json.dumps(config, indent=2) + "\n")
    commit(root, "a malformed configuration")
    machine = Machine()
    assert session(machine, Cloud()) == 1
    assert message in capsys.readouterr().out and machine.started == []


def key_paths(node: object, prefix: str = "") -> set[str]:
    """Every key of a configuration, nested ones as parent.key, an adapter's under adapters.<system>, a
    system's options under systems.<system>, a comparison's under comparisons[]."""
    paths: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "_doc":
                continue
            name = "<system>" if prefix in ("adapters", "systems") else key
            path = f"{prefix}.{name}" if prefix else name
            paths |= {path} | key_paths(value, path)
    elif isinstance(node, list) and node and all(isinstance(v, dict) for v in node):
        for value in node:
            paths |= key_paths(value, f"{prefix}[]")
    return paths


def test_the_example_configuration_documents_every_key_and_cannot_run_as_it_is():
    example = json.loads((REPO / "results" / "test-config-m3.example.json").read_text(encoding="utf-8"))
    assert key_paths(example) == set(example["_doc"])  # every key documented, nothing else
    with pytest.raises(run_test.ConfigError, match="adapters.ft.sha256 must be 64 lower-case hex digits"):
        run_test.check_config(example)
    filled = json.loads(json.dumps(example))
    for system in ("ft", "b4"):
        filled["adapters"][system].update(gguf=f"runs/train/{system}/gguf/checkpoint-1-f16.gguf",
                                          sha256="0" * 64,
                                          selection=f"results/checkpoint-selection-{system}.json")
    filled["server"]["llama_cpp_build"] = "b11349-fb4b2737a"
    run_test.check_config(filled)  # every other value is one the session can run with


def test_the_m3_configuration_as_committed_is_still_one_the_session_can_run_with():
    config = json.loads((REPO / "results" / "test-config-m3.json").read_text(encoding="utf-8"))
    assert run_test.check_config(config) is config


def test_an_adapter_of_a_system_the_session_does_not_run_may_stay_in_the_configuration(root):
    config = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    config.update(systems=["ft", "b0"], comparisons=[{"name": "ft-vs-b0", "a": "ft", "b": "b0"}])
    run_test.check_config(config)  # b4's adapter and b1's block are kept, and not checked
    del config["b1_block_sha256_16"]
    run_test.check_config(config)  # b1's block may be left out when b1 does not run


# ---- the dry run a real session needs ---------------------------------------------------------------------

DRY = "runs/test-m3/dry-run/session.json"


def test_a_real_session_needs_a_passed_dry_run_of_the_same_commit_and_configuration(root, capsys):
    machine, cloud = Machine(), Cloud()
    assert session(machine, cloud) == 4
    assert f"{DRY} is missing" in capsys.readouterr().out
    dry_run()
    dry = json.loads((root / DRY).read_text(encoding="utf-8"))
    head = git(root, "rev-parse", "HEAD").stdout.strip()
    assert (dry["dry_run"], dry["git_commit"]) == (True, head)
    assert dry["config"] == {"path": CONFIG, "sha256": sha256_of(root / CONFIG)}
    assert dry["dev_check"] == {"passed": True, "routed": None, "successes": None, "expected": None}  # no ftr
    write(root / "README.md", "Another commit.\n")
    commit(root, "another commit")
    assert session(machine, cloud) == 4
    assert f"{DRY} is a dry run of git commit {head}, not of HEAD" in capsys.readouterr().out
    config = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    dry_run()
    write(root / CONFIG, json.dumps({**config, "purpose": "another purpose"}, indent=2) + "\n")
    commit(root, "another configuration")
    dry["git_commit"] = git(root, "rev-parse", "HEAD").stdout.strip()  # the commit matches, the file does not
    write(root / DRY, json.dumps(dry))
    assert session(machine, cloud) == 4
    assert f"{DRY} is a dry run of the configuration with SHA-256 {sha256_of(root / CONFIG)}" not in (
        out := capsys.readouterr().out) and f"{DRY} is a dry run of the configuration with SHA-256" in out
    assert logs(root) == {TEST_LOG: None, E2E_LOG: None, COMPLETIONS: None} and machine.started == []


# ---- M5: ftr against the M3 records of b2 -----------------------------------------------------------------

M5_CONFIG = "results/test-config-m5.json"
ROUTING = "results/routing-fake.json"  # not ftr's default path: the session takes the configuration's
DEV_RECORDS = {"b2": "runs/eval/dev-b2-m3.jsonl", "ft": "runs/eval/select-ft/checkpoint-1668.jsonl"}
M3_COMPARISONS = "results/test-m3-comparisons.json"
PAID_TURN = (1000 * 2.0 + 50 * 8.0) / 1e6  # what one paid deepseek-flash turn of the fake costs, in RMB


def dev_records(root: Path, system: str) -> list[dict]:
    """system's records of the development items, each the record it gave in the M3 session to the test item
    of the same kind (a search for hallucination papers, a goodbye)."""
    m3 = {r["id"]: r for r in read_jsonl(root / "runs" / "test-m3" / "open-1" / f"{system}.jsonl")}
    same = {"req-dev-0001-s1": "req-test-0001-s1", "hw-dev-001": "hw-test-001"}
    return [{**m3[same[i["id"]]], "id": i["id"],
             "item": {k: v for k, v in i.items() if k not in ("id", "messages")}} for i in DEV_ITEMS]


def routing(root: Path, **changes) -> dict:
    """A frozen routing configuration: below 0.5 a local answer is routed. On the development items the fake
    routes "Look for hallucination papers." (confidence 0.2), and B2 answers it right: 1 routed, 2
    successes."""
    data = {"signal": "min_decision_prob", "threshold": 0.5, "rule_met": True,
            "cloud_model": "deepseek-flash", "dev_sha256": sha256_of(root / "data" / "eval" / "dev.jsonl"),
            "adapter_sha256": sha256_of(root / FT_GGUF), "chosen_point": {"routed": 1, "successes": 2}}
    data.update(changes)
    return data


def m5_configuration(root: Path, **changes) -> dict:
    m3 = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    reference = {s: {"results": f"results/test-m3-{s}.json", "records": f"runs/test-m3/open-1/{s}.jsonl",
                     "records_sha256": sha256_of(root / "runs" / "test-m3" / "open-1" / f"{s}.jsonl"),
                     "dev_records": DEV_RECORDS[s], "dev_records_sha256": sha256_of(root / DEV_RECORDS[s])}
                 for s in ("b2", "ft")}
    config = {"milestone": "M5", "purpose": "M5: FT+R against B2 on the locked test sets (a test)",
              "systems": ["ftr"], "adapters": {"ft": m3["adapters"]["ft"]}, "server": m3["server"],
              "tokenizer": m3["tokenizer"], "b2": {"max_cost_rmb": 5.0}, "e2e": m3["e2e"],
              "statistics": m3["statistics"],
              "routing": {"path": ROUTING, "sha256": sha256_of(root / ROUTING)},
              "reference": {"milestone": "M3", "systems": reference},
              "comparisons": [*({"name": f"ft-vs-{b}", "a": "ft", "b": b, "recorded": M3_COMPARISONS}
                                for b in ("b0", "b1", "b4")),
                              {"name": "ftr-vs-b2", "a": "ftr", "b": "b2", "margin": 0.02}],
              "gpu": m3["gpu"]}
    config.update(changes)
    return config


def write_m5(root: Path, config: dict | None = None, message: str = "the M5 configuration") -> None:
    config = config or m5_configuration(root)
    write(root / M5_CONFIG, json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    commit(root, message)


def write_routing(root: Path, **changes) -> None:
    write(root / ROUTING, json.dumps(routing(root, **changes), indent=2) + "\n")


@pytest.fixture(scope="module")
def m3_done(tmp_path_factory) -> Path:
    """A fake repository after a complete M3 session (its dry run, its open, its results and markers
    committed), with the development records of b2 and ft, the routing file and the M5 configuration
    committed. The response cache is removed, so ftr's routed turns are paid."""
    root = tmp_path_factory.mktemp("m5") / "repo"
    make_repository(root)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        patch.setattr(run_test, "REPO", root)
        dry_run()
        assert session(Machine(), Cloud()) == 0
    commit(root, "the M3 session")
    for system, path in DEV_RECORDS.items():
        write_jsonl(dev_records(root, system), root / path)
    for cache in (root / "runs" / "llm").glob("cache.sqlite*"):
        cache.unlink()
    write_routing(root)
    write_m5(root)
    return root


@pytest.fixture
def m5(m3_done, tmp_path, monkeypatch) -> Path:
    """A copy of m3_done as the current folder and run_test's repository root."""
    root = tmp_path / "repo"
    shutil.copytree(m3_done, root)
    monkeypatch.chdir(root)
    monkeypatch.setattr(run_test, "REPO", root)
    return root


def test_an_m5_session_runs_ftr_alone_and_tests_it_against_the_m3_records_of_b2(m5, capsys):
    dry_run(M5_CONFIG)
    dry = m5 / "runs" / "test-m5" / "dry-run"
    assert json.loads((dry / "session.json").read_text(encoding="utf-8"))["dev_check"] == {
        "passed": True, "routed": 1, "successes": 2, "expected": {"routed": 1, "successes": 2}}
    assert [r["id"] for r in read_jsonl(dry / "ftr.jsonl")] == [i["id"] for i in DEV_ITEMS]  # every one
    dry_ftr = json.loads((dry / "results" / "test-m5-ftr.json").read_text(encoding="utf-8"))
    assert dry_ftr["set"]["n"] == 2 and dry_ftr["routing"]["agreement"] == {
        "local": {"reference": "ft", "n": 1, "equal": 1}, "routed": {"reference": "b2", "n": 1, "equal": 1}}

    machine, cloud = Machine(), Cloud()
    assert session(machine, cloud, config=M5_CONFIG) == 0
    assert machine.servers() == [FT_GGUF] and {system for system, _ in machine.prompts} == {"ft"}
    assert cloud.users() == {"找几篇幻觉(hallucination)的论文。", "Search for hallucination papers, please.",
                             f"{ID1} 这篇论文讲什么?"}  # what ftr routed; b2 itself is not run
    assert [len(read_jsonl(m5 / log)) for log in (TEST_LOG, E2E_LOG)] == [2, 2]

    ftr = result(m5, "test-m5-ftr.json")
    assert ftr["records"] == "runs/test-m5/open-2/ftr.jsonl" and ftr["summary"]["success_rate"] == 0.75
    assert ftr["info"]["routing"]["path"] == ROUTING and ftr["info"]["routing"]["threshold"] == 0.5
    assert ftr["routing"] == {"n": 4, "routed": 2, "api_share": 0.5, "hard_triggers": {},
                              "reference_success": {"ft": 1.0, "b2": 0.75}, "random_success": 0.875,
                              "agreement": {"local": {"reference": "ft", "n": 2, "equal": 2},
                                            "routed": {"reference": "b2", "n": 2, "equal": 2}}}
    e2e = result(m5, "test-m5-e2e-ftr.json")
    assert e2e["summary"]["success_rate"] == 1.0
    assert e2e["routing"] == {"tasks": 2, "tasks_routed": 1, "local_calls": 2, "cloud_calls": 2,
                              "local_steps": 2, "cloud_steps": 2}

    comparisons = result(m5, "test-m5-comparisons.json")
    rows = comparisons["comparisons"]
    assert rows[0] == {"name": "ft-vs-b0", "a": "ft", "b": "b0", "recorded": M3_COMPARISONS, "p": 0.125,
                       "p_holm": 0.5}
    assert [(r["name"], r["p"], r["p_holm"]) for r in rows] == [
        ("ft-vs-b0", 0.125, 0.5), ("ft-vs-b1", 0.25, 0.75), ("ft-vs-b4", 1.0, 1.0), ("ftr-vs-b2", 1.0, 1.0)]
    tested = rows[3]
    assert (tested["n"], tested["success_a"], tested["success_b"], tested["diff"]) == (4, 0.75, 0.75, 0.0)
    assert tested["mcnemar"] == {"a_only": 0, "b_only": 0, "discordant": 0, "p": 1.0}
    paired, cluster = tested["paired_ci"]["low"], tested["cluster_ci"]["low"]
    assert tested["noninferiority"] == {"margin": 0.02, "paired_low": paired, "paired_above": paired > -0.02,
                                        "cluster_low": cluster, "cluster_above": cluster > -0.02}
    assert comparisons["rescored"] == {"b2": {"equal": True, "differ": []},
                                       "ft": {"equal": True, "differ": []}}
    assert comparisons["reference"]["systems"]["b2"] == {
        "records": "runs/test-m3/open-1/b2.jsonl", "sha256": sha256_of(m5 / "runs/test-m3/open-1/b2.jsonl")}

    resources = result(m5, "test-m5-resources.json")
    entry = resources["systems"]["ftr"]
    assert entry["server"] == "ft" and resources["servers"]["ft"]["systems"] == ["ftr"]
    assert {k: entry["single_step"][k]["n"] for k in ("local", "routed", "routed_paid")} == {
        "local": 2, "routed": 2, "routed_paid": 2}
    assert entry["cost"]["single_step"]["turns"] == 2
    assert entry["cost"]["single_step"]["paid_rmb"] == pytest.approx(2 * PAID_TURN)
    assert entry["cost"]["e2e"]["paid_rmb"] == pytest.approx(2 * PAID_TURN)
    assert entry["cost"]["ledger_rmb"] == pytest.approx(4 * PAID_TURN)

    markers = read_jsonl(m5 / COMPLETIONS)
    assert [(m["open"]["milestone"], m["status"]) for m in markers] == [("M3", "complete")] * 2 + [
        ("M5", "complete")] * 2
    assert sorted(markers[-1]["outputs"]) == [f"results/test-m5-{name}.json" for name in (
        "comparisons", "e2e-ftr", "ftr", "resources")]


def test_an_ftr_dry_run_off_the_frozen_point_fails_after_writing_its_session_and_bars_the_real_one(m5,
                                                                                                    capsys):
    write_routing(m5, chosen_point={"routed": 2, "successes": 2})
    write_m5(m5, m5_configuration(m5), "another frozen point")
    dry_run(M5_CONFIG, expect=6)
    out = capsys.readouterr().out
    assert "the dev check failed" in out and "nothing was opened" in out
    dry = m5 / "runs" / "test-m5" / "dry-run"
    assert json.loads((dry / "session.json").read_text(encoding="utf-8"))["dev_check"] == {
        "passed": False, "routed": 1, "successes": 2, "expected": {"routed": 2, "successes": 2}}
    assert (dry / "results" / "test-m5-ftr.json").is_file()
    before, machine = logs(m5), Machine()
    assert session(machine, Cloud(), config=M5_CONFIG) == 4
    assert "records a dev check that did not pass" in capsys.readouterr().out
    assert logs(m5) == before and machine.started == []


def changed_file(path: str):
    def change(root: Path) -> None:
        target = root / path
        target.write_bytes(target.read_bytes() + b"\n")
        commit(root, f"{path} changed")

    return change


def m5_changes(**changes):
    def change(root: Path) -> None:
        write_m5(root, m5_configuration(root, **changes), "a changed M5 configuration")

    return change


def other_routing(**changes):
    def change(root: Path) -> None:
        write_routing(root, **changes)
        write_m5(root, m5_configuration(root), "another routing file")

    return change


def unmarked_results(root: Path) -> None:
    shutil.copyfile(root / "results/test-m3-b2.json", root / "results/test-m3-b2-copy.json")
    config = m5_configuration(root)
    config["reference"]["systems"]["b2"]["results"] = "results/test-m3-b2-copy.json"
    write_m5(root, config, "a results file no marker records")


def recorded_rows(*names: str):
    rows = [{"name": name, "a": "ft", "b": "b2", "recorded": M3_COMPARISONS} for name in names]
    return m5_changes(comparisons=[*rows, {"name": "ftr-vs-b2-m5", "a": "ftr", "b": "b2"}])


@pytest.mark.parametrize("change, message", [
    (changed_file(ROUTING), f"the routing file {ROUTING} has SHA-256"),
    (other_routing(dev_sha256="0" * 64), f"the routing file {ROUTING} records dev_sha256 {'0' * 64}"),
    (other_routing(adapter_sha256="1" * 64), f"the routing file {ROUTING} records adapter_sha256 "),
    (changed_file("results/test-m3-b2.json"), "results/test-m3-b2.json has SHA-256"),
    (changed_file(M3_COMPARISONS), f"{M3_COMPARISONS} has SHA-256"),
    (unmarked_results, "results/test-m3-b2-copy.json is not among the outputs the M3 completion markers"),
    (changed_file("runs/test-m3/open-1/b2.jsonl"),
     "the b2 records runs/test-m3/open-1/b2.jsonl have SHA-256"),
    (changed_file(DEV_RECORDS["ft"]), f"the ft development records {DEV_RECORDS['ft']} have SHA-256"),
    (recorded_rows("ft-vs-b9"), f"{M3_COMPARISONS} has no comparison ft-vs-b9"),
    (recorded_rows("ftr-vs-b2"), f"{M3_COMPARISONS} holds ftr-vs-b2 as deferred"),
])
def test_each_m5_preflight_check_refuses_with_exit_4_naming_what_does_not_match(m5, capsys, change, message):
    change(m5)
    before, machine, cloud = logs(m5), Machine(), Cloud()
    assert session(machine, cloud, config=M5_CONFIG) == 4
    assert message in capsys.readouterr().out
    assert logs(m5) == before and machine.started == [] and cloud.requests == []


def test_reference_records_scored_otherwise_now_are_recorded_and_this_sessions_scores_are_compared(m5):
    path = m5 / "runs" / "test-m3" / "open-1" / "b2.jsonl"
    records = read_jsonl(path)
    right = next(r["output"] for r in records if r["id"] == "req-test-0001-s1")
    write_jsonl([{**r, "output": right} if r["id"] == "req-test-0002-s1" else r for r in records], path)
    write_m5(m5, m5_configuration(m5), "b2's records as they are now")
    dry_run(M5_CONFIG)
    assert session(Machine(), Cloud(), config=M5_CONFIG) == 0
    comparisons = result(m5, "test-m5-comparisons.json")
    assert comparisons["rescored"]["b2"] == {"equal": False, "differ": ["req-test-0002-s1"]}
    tested = comparisons["comparisons"][3]
    assert (tested["success_b"], tested["diff"]) == (1.0, -0.25)


def test_ftr_spends_within_one_cap_over_its_single_step_and_end_to_end_runs(m5):
    write_m5(m5, m5_configuration(m5, b2={"max_cost_rmb": 0.003}), "a small cap")
    dry_run(M5_CONFIG)  # one routed development item: 0.0024 RMB
    assert session(Machine(), Cloud(), config=M5_CONFIG) == 3
    statuses = [r["status"] for r in read_jsonl(m5 / "runs" / "test-m5" / "open-2" / "ftr.jsonl")]
    assert statuses == ["ok", "ok", "ok", "budget_stop"]  # the second routed item is over the cap
    assert "budget_stop" in result(m5, "test-m5-e2e-ftr.json")["summary"]["statuses"]
    assert [m["status"] for m in read_jsonl(m5 / COMPLETIONS)][-2:] == ["budget_stop", "budget_stop"]


# ---- the M5 configuration ---------------------------------------------------------------------------------

DROP = object()


def m5_dict(**changes) -> dict:
    """A valid M5 configuration without a repository (made-up hashes); a change to DROP removes the key."""
    entry = {"results": "results/test-m3-b2.json", "records": "runs/test-m3/open-1/b2.jsonl",
             "records_sha256": "a" * 64, "dev_records": "runs/eval/dev-b2-m3.jsonl",
             "dev_records_sha256": "b" * 64}
    config = {"milestone": "M5", "purpose": "M5", "systems": ["ftr"],
              "adapters": {"ft": {"gguf": FT_GGUF, "sha256": "9" * 64, "selection": None}},
              "server": {"model": "models/Qwen3-1.7B-Q8_0.gguf", "port": 8080, "ctx_size": 8192,
                         "timeout_s": 30, "llama_cpp_build": None},
              "tokenizer": "models/Qwen3-1.7B", "b2": {"max_cost_rmb": 5.0},
              "e2e": {"seed": 42, "today": None},
              "statistics": {"seed": 42, "resamples": 100},
              "routing": {"path": "results/routing-m4.json", "sha256": "1" * 64},
              "reference": {"milestone": "M3", "systems": {"b2": entry, "ft": dict(entry)}},
              "comparisons": [{"name": "ft-vs-b0", "a": "ft", "b": "b0", "recorded": M3_COMPARISONS},
                              {"name": "ftr-vs-b2", "a": "ftr", "b": "b2", "margin": 0.02}],
              "gpu": {"idle_max_mib": 1500, "sample_ms": 200}}
    for key, value in changes.items():
        if value is DROP:
            del config[key]
        else:
            config[key] = value
    return config


def test_an_m5_configuration_with_ftr_routing_references_and_three_kinds_of_rows_is_valid():
    config = m5_dict()
    config["comparisons"].insert(1, {"name": "ftr-vs-b9", "a": "ftr", "b": "b9", "deferred": "later", "p": 1})
    assert run_test.check_config(config) is config


TESTED_ONLY = [{"name": "ft-vs-b2", "a": "ft", "b": "b2"}]
REFERENCE_FT = {"milestone": "M3", "systems": {"ft": m5_dict()["reference"]["systems"]["ft"]}}


@pytest.mark.parametrize("changes, message", [
    ({"routing": DROP}, "routing is required when ftr runs"),
    ({"systems": ["ft"], "comparisons": [{"name": "ft-vs-b2", "a": "ft", "b": "b2"}]},
     "routing is only for a routed system (ftr, ft2r, ft3r), none of which this session runs"),
    ({"adapters": {}}, "adapters lacks ft"),
    ({"systems": ["ftr", "b1"]}, "b1_block_sha256_16 must be 16 lower-case hex digits, not None"),
    ({"systems": ["ftr", "ftr"]}, "systems names a system twice"),
    ({"routing": {"path": "results/routing-m4.json", "sha256": "1" * 64, "x": 1}}, "routing: unknown key x"),
    ({"routing": {"path": "results/routing-m4.json", "sha256": "1"}}, "routing.sha256 must be 64 lower-case"),
    ({"routing": {"path": "C:/results/routing-m4.json", "sha256": "1" * 64}},
     "routing.path must be a path relative to the repository root"),
    ({"comparisons": [{"name": "ft-vs-b2", "a": "ft", "b": "b2"}]},
     "comparisons: ft-vs-b2 compares no system this session runs"),
    ({"comparisons": [{"name": "ftr-vs-b9", "a": "ftr", "b": "b9"}]},
     "comparisons: ftr-vs-b9 compares b9, which is not among the systems"),
    ({"comparisons": [{"name": "ftr-vs-b2", "a": "ftr", "b": "b2", "margin": 0}]},
     "comparisons[1].margin must be a number above 0 and below 1, not 0"),
    ({"comparisons": [{"name": "ftr-vs-b2", "a": "ftr", "b": "b2", "margin": 1.5}]},
     "comparisons[1].margin must be a number above 0 and below 1, not 1.5"),
    ({"comparisons": [{"name": "ftr-vs-b2", "a": "ftr", "b": "b2", "recorded": M3_COMPARISONS, "p": 1}]},
     "comparisons[1]: unknown key p"),
    ({"reference": DROP}, "comparisons[1] takes its p from an earlier milestone's comparisons: reference"),
    ({"reference": {**REFERENCE_FT, "milestone": "M5"}}, "reference.milestone must be a milestone before M5"),
    ({"reference": {"milestone": "M3", "systems": {}}}, "reference.systems must be a non-empty object"),
    ({"systems": ["ftr", "b2"]}, "reference.systems.b2 is a system this session runs"),
    ({"reference": {"milestone": "M3", "systems": {"b9": {}}}}, "reference.systems: b9 is not one of"),
    ({"reference": {"milestone": "M3", "systems": {"b2": {"results": "results/test-m3-b2.json"}}}},
     "reference.systems.b2 lacks records"),
    ({"reference": REFERENCE_FT}, "comparisons: ftr-vs-b2 compares b2, which is not among the systems"),
])
def test_a_malformed_m5_configuration_is_refused_naming_the_key(changes, message):
    with pytest.raises(run_test.ConfigError) as refused:
        run_test.check_config(m5_dict(**changes))
    assert message in str(refused.value)


def test_a_reference_path_or_hash_of_the_wrong_kind_is_refused():
    for key, value, message in (
            ("records", "/runs/b2.jsonl", "reference.systems.b2.records must be a path"),
            ("dev_records_sha256", "x", "reference.systems.b2.dev_records_sha256 must be")):
        config = m5_dict()
        config["reference"]["systems"]["b2"] = {**config["reference"]["systems"]["b2"], key: value}
        with pytest.raises(run_test.ConfigError, match=re.escape(message)):
            run_test.check_config(config)



# ---- M6: the committed M5 configuration and its routing file still pass the checks -----------------------


def test_the_m5_configuration_as_committed_is_still_one_the_session_can_run_with(monkeypatch):
    config = json.loads((REPO / "results" / "test-config-m5.json").read_text(encoding="utf-8"))
    assert run_test.check_config(config) is config
    if not (REPO / config["routing"]["path"]).is_file() or not (REPO / run_test.DEV_SET).is_file():
        pytest.skip("the routing file or the development set is not here")
    monkeypatch.chdir(REPO)  # check_routing reads the routing file and the dev set by their relative paths
    assert run_test.check_routing(config) == {"routed": 94, "successes": 132}


def test_the_m6_configuration_as_committed_is_still_one_the_session_can_run_with(monkeypatch):
    config = json.loads((REPO / "results" / "test-config-m6.json").read_text(encoding="utf-8"))
    assert run_test.check_config(config) is config and run_test.family_of(config) == TEST_FAMILIES["v1"]
    if not (REPO / config["routing"]["path"]).is_file() or not (REPO / run_test.DEV_SET).is_file():
        pytest.skip("the routing file or the development set is not here")
    monkeypatch.chdir(REPO)
    assert run_test.check_routing(config) == {"routed": 23, "successes": 132}  # deepseek-flash, its rules


# ---- M6: ft2 and ft2r against the records of M3 (ft, b2) and M5 (ftr) -------------------------------------

M6_CONFIG = "results/test-config-m6.json"
FT2_GGUF = "runs/train/ft2/gguf/checkpoint-1752-f16.gguf"
ROUTING_M6 = "results/routing-m6-fake.json"
ACE_B0, ACE_FT2 = "runs/acebench/m4/b0", "runs/acebench/m6/ft"
FTR_RECORDS, FTR_DEV_RECORDS = "runs/test-m5/open-2/ftr.jsonl", "runs/test-m5/dry-run/ftr.jsonl"
P5_RULES = {"text_exempt": True, "param_triggers": [
    {"tool": "search_notes", "param": "top_k", "allowed": [3]},
    {"tool": "list_notes", "param": "limit", "allowed": [10]}]}
# the fabricated ACEBench runs: answers per (language, category) and the ones each system got right
ACE_ITEMS = {("zh", "normal_atom_bool"): 4, ("zh", "special_incomplete"): 2, ("en", "normal_atom_bool"): 4,
             ("en", "special_incomplete"): 2}
ACE_RIGHT = {"b0": {("zh", "normal_atom_bool"): {0, 1, 2}, ("zh", "special_incomplete"): {0},
                    ("en", "normal_atom_bool"): {0, 1}, ("en", "special_incomplete"): {0, 1}},
             "ft2": {("zh", "normal_atom_bool"): {0, 1, 2, 3}, ("zh", "special_incomplete"): {1},
                     ("en", "normal_atom_bool"): {1, 2}, ("en", "special_incomplete"): {0}}}


def ace_ids(category: str, n: int) -> list[str]:
    return [f"{category}_{i}" for i in range(n)]


def write_acebench(root: Path, folder: str, right: dict, *, ids: dict | None = None) -> None:
    """A raw ACEBench folder as scripts/run_acebench.py copies it: per language, one result file (an answer
    per question) and one score file (a header, then a line per wrong answer) per category."""
    for (language, category), n in ACE_ITEMS.items():
        names = (ids or {}).get((language, category)) or ace_ids(category, n)
        wrong = [i for k, i in enumerate(names) if k not in right[(language, category)]]
        base = root / folder / language
        write(base / "result" / f"data_{category}_result.json",
              "".join(json.dumps({"id": i, "result": "[f(x=1)]"}) + "\n" for i in names))
        header = {"accuracy": (n - len(wrong)) / n, "correct_count": n - len(wrong), "total_count": n}
        rows = [header, *({"id": i, "valid": False, "error": ["wrong"]} for i in wrong)]
        write(base / "score" / f"data_{category}_score.json", "\n".join(json.dumps(r) for r in rows))


def ace_strata(system: str) -> dict:
    """The per-answer successes of a fabricated run per stratum, in the order the comparison pairs them."""
    return {key: [k in ACE_RIGHT[system][key] for k in range(n)] for key, n in sorted(ACE_ITEMS.items())}


def routing_m6(root: Path, **changes) -> dict:
    """A frozen M6 routing configuration (scripts/route_curve.py under the cost objective): the policy P5
    for the ft2 adapter. On the development items the fake routes "Look for hallucination papers." (a call
    at confidence 0.2), which B2 answers right, and answers the goodbye locally with a call: 1 routed, 1
    success."""
    point = {"label": "threshold", "threshold": 0.5, "routed": 1, "api_share": 0.5, "successes": 1,
             "success": 0.5, "cost_per_100_rmb": 0.12, "random_success": 0.5}
    data = {"rule": "the lowest cost per 100 items at success >= B2 - 0.02 (a test)", "objective": "cost",
            "margin": 0.02, "rule_met": True, "signal": "min_decision_prob", "threshold": 0.5,
            "chosen_policy": "P5", "rules": P5_RULES, "cloud_model": "deepseek-flash",
            "dev_sha256": sha256_of(root / "data" / "eval" / "dev.jsonl"),
            "adapter_sha256": sha256_of(root / FT2_GGUF), "chosen_point": {**point, "policy": "P5"},
            "policies": {"P0": {"label": "threshold only", "rules": None, "curve": [], "chosen": None},
                         "P5": {"label": "text exemption + non-default parameter triggers",
                                "rules": P5_RULES, "curve": [point], "chosen": point}}}
    data.update(changes)
    return data


def write_routing_m6(root: Path, *, remove: tuple[str, ...] = (), **changes) -> None:
    data = {k: v for k, v in routing_m6(root, **changes).items() if k not in remove}
    write(root / ROUTING_M6, json.dumps(data, indent=2) + "\n")


def m6_configuration(root: Path, **changes) -> dict:
    m3 = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    reference = m5_configuration(root)["reference"]["systems"]  # ft and b2 of M3
    reference["ftr"] = {"milestone": "M5", "results": "results/test-m5-ftr.json", "records": FTR_RECORDS,
                        "records_sha256": sha256_of(root / FTR_RECORDS), "dev_records": FTR_DEV_RECORDS,
                        "dev_records_sha256": sha256_of(root / FTR_DEV_RECORDS)}
    config = {"milestone": "M6",
              "purpose": "M6: FT2 and FT2+R2 against the records of FT, B2 and FT+R (a test)",
              "systems": ["ft2", "ft2r"],
              "adapters": {"ft2": {"gguf": FT2_GGUF, "sha256": sha256_of(root / FT2_GGUF),
                                   "selection": None}},
              "server": m3["server"], "tokenizer": m3["tokenizer"], "b2": {"max_cost_rmb": 5.0},
              "e2e": m3["e2e"], "e2e_routing": "stepwise", "statistics": m3["statistics"],
              "routing": {"path": ROUTING_M6, "sha256": sha256_of(root / ROUTING_M6)},
              "reference": {"milestone": "M3", "systems": reference},
              "comparisons": [{"name": "ft2-vs-ft", "a": "ft2", "b": "ft", "test": "mcnemar"},
                              {"name": "ft2r-vs-b2", "a": "ft2r", "b": "b2",
                               "noninferiority": {"margin": 0.02}},
                              {"name": "ft2-vs-b0-acebench",
                               "acebench": {"a": ACE_FT2, "b": ACE_B0, "margin": 0.02}},
                              {"name": "ft2r-vs-ftr", "a": "ft2r", "b": "ftr", "descriptive": True}],
              "gpu": m3["gpu"]}
    config.update(changes)
    return config


def write_m6(root: Path, config: dict | None = None, message: str = "the M6 configuration") -> None:
    config = config or m6_configuration(root)
    write(root / M6_CONFIG, json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    commit(root, message)


@pytest.fixture(scope="module")
def m5_done(m3_done, tmp_path_factory) -> Path:
    """m3_done after a complete M5 session (its dry run, its open, its results and markers committed), with
    the ft2 adapter, the M6 routing file, the fabricated ACEBench runs of b0 (M4) and ft2 (M6) and the M6
    configuration committed. The response cache now holds every call M5 routed."""
    root = tmp_path_factory.mktemp("m6") / "repo"
    shutil.copytree(m3_done, root)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        patch.setattr(run_test, "REPO", root)
        dry_run(M5_CONFIG)
        assert session(Machine(), Cloud(), config=M5_CONFIG) == 0
    commit(root, "the M5 session")
    write(root / FT2_GGUF, b"GGUF ft2 step 1752")
    write_routing_m6(root)
    write_acebench(root, ACE_B0, ACE_RIGHT["b0"])
    write_acebench(root, ACE_FT2, ACE_RIGHT["ft2"])
    write_m6(root)
    return root


@pytest.fixture
def m6(m5_done, tmp_path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(m5_done, root)
    monkeypatch.chdir(root)
    monkeypatch.setattr(run_test, "REPO", root)
    return root


def test_an_m6_session_runs_ft2_then_ft2r_and_tests_the_three_hypotheses_against_the_records(m6):
    dry_run(M6_CONFIG)
    dry = m6 / "runs" / "test-m6" / "dry-run"
    assert json.loads((dry / "session.json").read_text(encoding="utf-8"))["dev_check"] == {
        "passed": True, "routed": 1, "successes": 1, "expected": {"routed": 1, "successes": 1}}
    assert [r["id"] for r in read_jsonl(dry / "ft2r.jsonl")] == [i["id"] for i in DEV_ITEMS]
    assert [r["id"] for r in read_jsonl(dry / "ft2.jsonl")] == ["req-dev-0001-s1"]

    machine, cloud = Machine(), Cloud()
    assert session(machine, cloud, config=M6_CONFIG) == 0
    assert machine.servers() == [FT2_GGUF] and {system for system, _ in machine.prompts} == {"ft2"}
    assert cloud.requests == []  # every routed call is one M5 routed too: served from the cache
    assert [len(read_jsonl(m6 / log)) for log in (TEST_LOG, E2E_LOG)] == [3, 3]

    ft2 = result(m6, "test-m6-ft2.json")
    assert ft2["records"] == "runs/test-m6/open-3/ft2.jsonl" and ft2["summary"]["success_rate"] == 0.75
    assert ft2["info"]["adapters"] == [{"id": 0, "path": FT2_GGUF, "sha256": sha256_of(m6 / FT2_GGUF)}]
    assert (ft2["info"]["system"], ft2["info"]["built_as"]) == ("ft2", "ft")
    ft2r = result(m6, "test-m6-ft2r.json")
    assert ft2r["summary"]["success_rate"] == 0.5 and ft2r["info"]["built_as"] == "ftr"
    assert ft2r["info"]["routing"]["rules"] == P5_RULES and ft2r["info"]["routing"]["policy"] == "P5"
    assert ft2r["routing"] == {"n": 4, "routed": 2, "api_share": 0.5, "hard_triggers": {},
                               "reference_success": {"ft2": 0.75, "b2": 0.75}, "random_success": 0.75,
                               "agreement": {"local": {"reference": "ft2", "n": 2, "equal": 2},
                                             "routed": {"reference": "b2", "n": 2, "equal": 2}}}
    assert result(m6, "test-m6-e2e-ft2.json")["summary"]["success_rate"] == 1.0
    e2e = result(m6, "test-m6-e2e-ft2r.json")
    assert e2e["summary"]["success_rate"] == 1.0 and e2e["e2e_routing"] == "stepwise"
    assert e2e["routing"] == {"tasks": 2, "tasks_routed": 1, "local_calls": 3, "cloud_calls": 1,
                              "local_steps": 3, "cloud_steps": 1}
    assert "e2e_routing" not in result(m6, "test-m6-e2e-ft2.json")  # not a routed system

    comparisons = result(m6, "test-m6-comparisons.json")
    h1, h2, h3, described = comparisons["comparisons"]
    assert [r["name"] for r in (h1, h2, h3, described)] == ["ft2-vs-ft", "ft2r-vs-b2",
                                                           "ft2-vs-b0-acebench", "ft2r-vs-ftr"]
    assert h1["test"] == "mcnemar" and h1["p"] == 1.0
    assert h1["mcnemar"] == {"a_only": 0, "b_only": 1, "discordant": 1, "p": 1.0}
    # ft2r: the first item right locally, the Chinese one routed and wrong, the goodbye answered locally
    # with a call, the last routed and right; b2 is right on all but the Chinese one
    ft2r_ok, b2_ok = [True, False, False, True], [True, False, True, True]
    paired = noninferiority_p(ft2r_ok, b2_ok, 0.02, seed=42, n=200)
    families = [("family", "find/en"), ("family", "find/zh"), ("item", "hw-test-001"),
                ("item", "hw-test-002")]
    clustered = cluster_noninferiority_p(ft2r_ok, b2_ok, families, 0.02, seed=42, n=200)
    assert h2["test"] == "noninferiority" and (h2["success_a"], h2["success_b"]) == (0.5, 0.75)
    assert h2["diff"] == -0.25
    assert h2["noninferiority"] == {"margin": 0.02, "p_ni": paired["p_ni"], "below": paired["below"],
                                    "cluster_p_ni": clustered["p_ni"], "cluster_below": clustered["below"]}
    assert 0.55 < h2["p"] == paired["p_ni"] < 0.8  # P(the goodbye is drawn at least once in 4) = 0.68
    assert h2["mcnemar"]["b_only"] == 1 and "paired_ci" in h2 and "cluster_ci" in h2
    stratified = stratified_noninferiority_p(ace_strata("ft2"), ace_strata("b0"), 0.02, seed=42, n=200)
    assert h3["test"] == "acebench" and (h3["a"], h3["b"]) == (ACE_FT2, ACE_B0)
    block = h3["acebench"]
    assert block["a"]["path"] == ACE_FT2 and len(block["a"]["sha256"]) == 64 and block["a"]["files"] == 8
    assert (block["a"]["items"], block["b"]["items"], block["strata"], block["items"]) == (12, 12, 4, 12)
    assert block["margin"] == 0.02 and block["diff"] == pytest.approx(-0.0625)
    assert (block["a"]["mean"], block["b"]["mean"]) == (pytest.approx(0.625), pytest.approx(0.6875))
    assert (block["p_ni"], block["below"]) == (stratified["p_ni"], stratified["below"])
    assert 0 < h3["p"] == block["p_ni"] < 1 and block["paired_ci"]["low"] == stratified["low"]
    assert [(s["language"], s["category"], s["n"], s["correct_a"], s["correct_b"])
            for s in block["per_stratum"]] == [("en", "normal_atom_bool", 4, 2, 2),
                                               ("en", "special_incomplete", 2, 1, 2),
                                               ("zh", "normal_atom_bool", 4, 4, 3),
                                               ("zh", "special_incomplete", 2, 1, 1)]
    assert [r["p_holm"] for r in (h1, h2, h3)] == holm([h1["p"], h2["p"], h3["p"]])
    assert described["descriptive"] is True and "p" not in described and "p_holm" not in described
    assert "p" not in described["mcnemar"]  # nor nested: a comparison outside the family carries no test
    assert (described["diff"], described["mcnemar"]["b_only"]) == (-0.25, 1)  # FT+R got the goodbye right
    m3_records = {s: f"runs/test-m3/open-1/{s}.jsonl" for s in ("b2", "ft")}
    assert comparisons["reference"] == {"milestone": "M3", "systems": {  # a system's own milestone as given
        **{s: {"records": path, "sha256": sha256_of(m6 / path)} for s, path in m3_records.items()},
        "ftr": {"milestone": "M5", "records": FTR_RECORDS, "sha256": sha256_of(m6 / FTR_RECORDS)}}}
    assert comparisons["rescored"] == {s: {"equal": True, "differ": []} for s in ("b2", "ft", "ftr")}

    resources = result(m6, "test-m6-resources.json")
    assert resources["servers"]["ft2"]["systems"] == ["ft2", "ft2r"] and list(resources["servers"]) == ["ft2"]
    assert resources["servers"]["ft2"]["adapter"]["gguf"] == FT2_GGUF
    assert resources["systems"]["ft2"]["server"] == "ft2" and resources["systems"]["ft2"]["cost"] is None
    entry = resources["systems"]["ft2r"]
    assert entry["server"] == "ft2" and entry["single_step"]["routed_paid"]["n"] == 0
    assert entry["single_step"]["local"]["n"] == 2 and entry["single_step"]["routed"]["n"] == 2
    assert entry["cost"]["single_step"] == {"turns": 2, "paid": 0, "cached": 2, "paid_rmb": 0.0,
                                            "list_rmb": pytest.approx(2 * PAID_TURN)}
    assert (entry["cost"]["e2e"]["paid_calls"], entry["cost"]["e2e"]["cache_hits"]) == (0, 1)
    assert entry["cost"]["ledger_rmb"] == 0.0

    markers = read_jsonl(m6 / COMPLETIONS)
    assert [(m["open"]["milestone"], m["status"]) for m in markers[-2:]] == [("M6", "complete")] * 2
    assert sorted(markers[-1]["outputs"]) == [f"results/test-m6-{name}.json" for name in (
        "comparisons", "e2e-ft2", "e2e-ft2r", "ft2", "ft2r", "resources")]
    for fake in (ROUTING, ROUTING_M6):  # the fakes are not whole routing files: the report would refuse them
        (m6 / fake).unlink()
    numbers, _ = report.build(m6)
    assert numbers["compare.test-m6.ft2r-vs-b2.margin"] == "2.0"
    assert numbers["compare.test-m6.ft2r-vs-b2.p_ni"] == f"{h2['p']:.3f}"
    assert numbers["compare.test-m6.ft2-vs-b0-acebench.diff"] == "-6.25"
    assert numbers["compare.test-m6.ft2r-vs-ftr.diff"] == "-25.0"
    assert "compare.test-m6.ft2r-vs-ftr.p" not in numbers
    assert numbers["eval.test-m6-ft2r.routing.reference_success.ft2"] == "75.0%"
    assert numbers["e2e.test-m6-ft2r.routing.mode"] == "stepwise"
    assert numbers["resources.test-m6.server.ft2.vram_mib"] == "4088"


def test_without_e2e_routing_the_end_to_end_run_stays_in_the_cloud_once_routed_as_m5_did(m6):
    config = m6_configuration(m6)
    del config["e2e_routing"]
    write_m6(m6, config, "sticky routing")
    dry_run(M6_CONFIG)
    assert session(Machine(), Cloud(), config=M6_CONFIG) == 0
    e2e = result(m6, "test-m6-e2e-ft2r.json")
    assert e2e["e2e_routing"] == "sticky"
    assert e2e["routing"] == {"tasks": 2, "tasks_routed": 1, "local_calls": 2, "cloud_calls": 2,
                              "local_steps": 2, "cloud_steps": 2}
    assert result(m6, "test-m6-ft2r.json")["routing"]["routed"] == 2  # single step is the same either way


def other_routing_m6(*, remove: tuple[str, ...] = (), **changes):
    def change(root: Path) -> None:
        write_routing_m6(root, remove=remove, **changes)
        write_m6(root, m6_configuration(root), "another M6 routing file")

    return change


def changed_ft2_adapter(root: Path) -> None:
    (root / FT2_GGUF).write_bytes(b"GGUF ft2, retrained")


def acebench_without(path: str):
    def change(root: Path) -> None:
        (root / path).unlink()

    return change


def acebench_other_ids(root: Path) -> None:
    ids = {("zh", "special_incomplete"): ["special_incomplete_7", "special_incomplete_1"]}
    write_acebench(root, ACE_FT2, ACE_RIGHT["ft2"], ids=ids)


@pytest.mark.parametrize("change, message", [
    (changed_file(ROUTING_M6), f"the routing file {ROUTING_M6} has SHA-256"),
    (other_routing_m6(remove=("rules", "chosen_policy", "policies")),
     f"the routing file {ROUTING_M6} has no rules"),
    (other_routing_m6(remove=("policies",)), f"the routing file {ROUTING_M6} has no rules"),
    (other_routing_m6(chosen_policy="P0"), "the rules of the policy P0 are not the file's rules"),
    (other_routing_m6(chosen_policy="P9"), "names the policy 'P9', which its policies do not hold"),
    (other_routing_m6(rules={"text_exempt": "yes"}), f"the routing file {ROUTING_M6} has malformed rules"),
    (other_routing_m6(chosen_point={"routed": 2, "successes": 1, "policy": "P5"}),
     "the chosen point is not the chosen point of the policy P5"),
    (other_routing_m6(adapter_sha256="1" * 64), f"the routing file {ROUTING_M6} records adapter_sha256 "),
    (other_routing_m6(dev_sets=[{"path": "data/eval/dev.jsonl", "sha256": "0" * 64},
                                {"path": "data/eval/v2/dev-v2.jsonl", "sha256": "1" * 64}]),
     (f"the routing file {ROUTING_M6} was chosen on 2 development sets, not on the 1 of the v1 test sets "
      "(data/eval/dev.jsonl)")),
    (changed_ft2_adapter, f"the ft2 adapter {FT2_GGUF} has SHA-256"),
    (changed_file("results/test-m5-ftr.json"), "results/test-m5-ftr.json has SHA-256"),
    (changed_file(FTR_RECORDS), f"the ftr records {FTR_RECORDS} have SHA-256"),
    (changed_file(FTR_DEV_RECORDS), f"the ftr development records {FTR_DEV_RECORDS} have SHA-256"),
    (acebench_without(f"{ACE_FT2}/en/result/data_special_incomplete_result.json"),
     "data_special_incomplete_result.json is missing"),
    (acebench_without(f"{ACE_B0}/en/score/data_special_incomplete_score.json"),
     f"{ACE_FT2} and {ACE_B0} do not hold the same"),
    (acebench_other_ids, "zh/special_incomplete"),
])
def test_each_m6_preflight_check_refuses_with_exit_4_naming_what_does_not_match(m6, capsys, change, message):
    change(m6)
    before, machine, cloud = logs(m6), Machine(), Cloud()
    assert session(machine, cloud, config=M6_CONFIG) == 4
    assert message in capsys.readouterr().out
    assert logs(m6) == before and machine.started == [] and cloud.requests == []


def test_an_m6_dry_run_off_the_frozen_point_fails_and_bars_the_real_session(m6, capsys):
    data = routing_m6(m6)
    for point in (data["chosen_point"], data["policies"]["P5"]["chosen"]):
        point.update(routed=2, successes=2)
    write(m6 / ROUTING_M6, json.dumps(data, indent=2) + "\n")
    write_m6(m6, m6_configuration(m6), "another frozen point")
    dry_run(M6_CONFIG, expect=6)
    assert "the dev check failed: ft2r routed 1" in capsys.readouterr().out
    assert session(Machine(), Cloud(), config=M6_CONFIG) == 4
    assert "records a dev check that did not pass" in capsys.readouterr().out


# ---- the M6 configuration ---------------------------------------------------------------------------------


def m6_dict(**changes) -> dict:
    """A valid M6 configuration without a repository (made-up hashes); a change to DROP removes the key."""
    base = m5_dict()
    entry = base["reference"]["systems"]["b2"]
    reference = {"ft": dict(entry), "b2": dict(entry),
                 "ftr": {**entry, "milestone": "M5", "results": "results/test-m5-ftr.json",
                         "records": FTR_RECORDS, "dev_records": FTR_DEV_RECORDS}}
    config = {**base, "milestone": "M6", "purpose": "M6", "systems": ["ft2", "ft2r"],
              "adapters": {"ft2": {"gguf": FT2_GGUF, "sha256": "8" * 64, "selection": None}},
              "e2e_routing": "stepwise", "routing": {"path": "results/routing-m6.json", "sha256": "1" * 64},
              "reference": {"milestone": "M3", "systems": reference},
              "comparisons": [{"name": "ft2-vs-ft", "a": "ft2", "b": "ft", "test": "mcnemar"},
                              {"name": "ft2r-vs-b2", "a": "ft2r", "b": "b2",
                               "noninferiority": {"margin": 0.02}},
                              {"name": "ft2-vs-b0-acebench",
                               "acebench": {"a": ACE_FT2, "b": ACE_B0, "margin": 0.02}},
                              {"name": "ft2r-vs-ftr", "a": "ft2r", "b": "ftr", "descriptive": True}]}
    for key, value in changes.items():
        if value is DROP:
            del config[key]
        else:
            config[key] = value
    return config


def test_an_m6_configuration_with_the_aliases_three_kinds_of_test_and_a_descriptive_row_is_valid():
    config = m6_dict()
    assert run_test.check_config(config) is config
    plain = m6_dict(comparisons=[{"name": "ft2-vs-ft", "a": "ft2", "b": "ft"}])  # McNemar is the default
    assert run_test.check_config(plain) is plain
    kind = run_test.comparison_kind
    assert kind({"name": "x", "a": "a", "b": "b"}) == "tested"
    assert kind({"name": "x", "a": "a", "b": "b", "noninferiority": {}}) == "noninferiority"
    assert kind({"name": "x", "acebench": {}}) == "acebench"
    assert kind({"name": "x", "a": "a", "b": "b", "descriptive": True}) == "descriptive"
    assert run_test.RUN_ORDER.index("ft2") < run_test.RUN_ORDER.index("ft2r")
    assert run_test.SERVER_OF["ft2"] == run_test.SERVER_OF["ft2r"] == "ft2" and "ft2r" in run_test.SPENDING


def comparisons_of(*rows: dict) -> list[dict]:
    return [{"name": "ft2-vs-ft", "a": "ft2", "b": "ft", "test": "mcnemar"}, *rows]


def ftr_reference(milestone: object) -> dict:
    systems = m6_dict()["reference"]["systems"]
    return {"milestone": "M3", "systems": {**systems, "ftr": {**systems["ftr"], "milestone": milestone}}}


@pytest.mark.parametrize("changes, message", [
    ({"systems": ["ft2", "ft4"]}, "systems: ft4 is not one of"),
    ({"systems": ["ft2", "ft2r", "ftr"], "adapters": {**m6_dict()["adapters"], **m5_dict()["adapters"]}},
     "at most one routed system"),
    ({"adapters": {"ft": {"gguf": FT_GGUF, "sha256": "9" * 64, "selection": None}}}, "adapters lacks ft2"),
    ({"systems": ["ft2"], "routing": DROP, "e2e_routing": DROP},
     "comparisons: ft2r-vs-b2 compares ft2r, which is not among"),
    ({"systems": ["ft2"], "comparisons": comparisons_of()},
     "routing is only for a routed system (ftr, ft2r, ft3r), none of which this session runs"),
    ({"systems": ["ft2"], "comparisons": comparisons_of(), "routing": DROP},
     "e2e_routing is only for a routed system"),
    ({"e2e_routing": "always"}, "e2e_routing must be sticky or stepwise, not 'always'"),
    ({"comparisons": comparisons_of({"name": "x", "a": "ft2", "b": "ft", "test": "wilcoxon"})},
     "comparisons[2].test must be mcnemar"),
    ({"comparisons": comparisons_of({"name": "x", "a": "ft2r", "b": "b2", "noninferiority": {"margin": 0}})},
     "comparisons[2].noninferiority.margin must be a number above 0 and below 1"),
    ({"comparisons": comparisons_of({"name": "x", "a": "ft2r", "b": "b2", "noninferiority": {}})},
     "comparisons[2].noninferiority lacks margin"),
    ({"comparisons": comparisons_of({"name": "x", "a": "ft2r", "b": "b2", "noninferiority": {"margin": 0.02},
                                     "margin": 0.02})}, "comparisons[2]: unknown key margin"),
    ({"comparisons": comparisons_of({"name": "x", "acebench": {"a": ACE_FT2, "b": ACE_FT2, "margin": 0.02}})},
     "comparisons: x compares runs/acebench/m6/ft with itself"),
    ({"comparisons": comparisons_of({"name": "x", "acebench": {"a": "C:/ace/ft", "b": ACE_B0,
                                                                "margin": 0.02}})},
     "comparisons[2].acebench.a must be a path relative to the repository root"),
    ({"comparisons": comparisons_of({"name": "x", "acebench": {"a": ACE_FT2, "b": ACE_B0}})},
     "comparisons[2].acebench lacks margin"),
    ({"comparisons": comparisons_of({"name": "x", "a": "ft2r", "b": "ftr", "descriptive": "yes"})},
     "comparisons[2].descriptive must be true"),
    ({"comparisons": comparisons_of({"name": "x", "a": "ft", "b": "ftr", "descriptive": True})},
     "comparisons: x compares no system this session runs"),
    ({"reference": ftr_reference("M6")}, "reference.systems.ftr.milestone must be a milestone before M6"),
    ({"reference": ftr_reference(5)}, "reference.systems.ftr.milestone must be a milestone before M6"),
])
def test_a_malformed_m6_configuration_is_refused_naming_the_key(changes, message):
    with pytest.raises(run_test.ConfigError) as refused:
        run_test.check_config(m6_dict(**changes))
    assert message in str(refused.value)


# ---- M7: ft3 and ft3r on the v2 test sets, ft3's server on its own base model and tokenizer ---------------

M7_CONFIG = "results/test-config-m7.json"
FT3_GGUF = "runs/train/ft3/gguf/checkpoint-548-f16.gguf"
BASE_17B, BASE_4B = "models/Qwen3-1.7B-Q8_0.gguf", "models/Qwen3-4B-Q8_0.gguf"
TOKENIZER_4B = "models/Qwen3-4B"
ROUTING_M7 = "results/routing-m7-fake.json"
V1, V2 = TEST_FAMILIES["v1"], TEST_FAMILIES["v2"]
V2_SETS = tuple(test_set.as_posix() for test_set, _ in V2.sets)
V2_LOGS = tuple(log.as_posix() for _, log in V2.sets)
V2_COMPLETIONS = V2.completions.as_posix()
DEV_PATH, DEV_V2_PATH = "data/eval/dev.jsonl", "data/eval/v2/dev-v2.jsonl"
# FT3 answers all three right, FT2 all but the goodbye; FT3+R3 routes the Chinese one to B2, which errs
TEST_ITEMS_V2 = [item("req-test-0301-s1", "Find recent hallucination papers.", FIND, "call", "en", "test"),
                 item("hw-test-165", "Thanks, that's all for this week.", BYE, "no_call", "en", "test",
                      handwritten=True),
                 item("hw-test-166", "帮我找幻觉(hallucination)方向的论文。", FIND, "call", "zh", "test",
                      handwritten=True)]
TEST_TASKS_V2 = [task("e2e-test-61", f"Tell me about {ID1}.", ID1, "en", "test"),
                 task("e2e-test-62", f"{ID2} 这篇讲什么?", ID2, "zh", "test")]
# The v2 family's own development set beside data/eval/dev.jsonl (docs/m7-计划.md §5): an unsure call the
# fake routes and B2 answers right, and a goodbye FT3 answers locally
DEV_V2_ITEMS = [item("req-dev-0401-s1", "Find hallucination papers, please.", FIND, "call", "en", "dev"),
                item("req-dev-0402-s1", "Thanks, that's all for now.", BYE, "no_call", "en", "dev")]
DEV_V2_TEXTS = {"Find hallucination papers, please.", "Thanks, that's all for now."}
P5_BUDGET_RULES = {**P5_RULES, "budget_forced": True}


def routing_m7(root: Path, **changes) -> dict:
    """A frozen M7 routing configuration (scripts/route_curve.py under the target objective, chosen on the
    development sets dev and dev-v2 jointly, "dev_sets"): the policy P5 with the budget trigger for the ft3
    adapter, routed to deepseek-flash (B2). Over the four development items the fake routes the two unsure
    calls ("Look for hallucination papers." and "Find hallucination papers, please."), which B2 answers
    right, and FT3 answers both goodbyes: 2 routed, 4 successes."""
    data = routing_m6(root, adapter_sha256=sha256_of(root / FT3_GGUF), objective="target",
                      rules=P5_BUDGET_RULES)
    del data["dev_sha256"]
    data["dev_sets"] = [{"path": path, "sha256": sha256_of(root / path), "n": 2}
                        for path in (DEV_PATH, DEV_V2_PATH)]
    data["policies"]["P5"]["rules"] = P5_BUDGET_RULES
    for point in (data["chosen_point"], data["policies"]["P5"]["chosen"]):
        point.update(routed=2, successes=4, success=1.0)
    data.update(changes)
    return data


def write_routing_m7(root: Path, **changes) -> None:
    write(root / ROUTING_M7, json.dumps(routing_m7(root, **changes), indent=2) + "\n")


def m7_configuration(root: Path, **changes) -> dict:
    m3 = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    config = {"milestone": "M7", "purpose": "M7: FT3 and FT3+R3 on the v2 test sets (a test)",
              "test_sets": "v2", "systems": ["ft3", "ft3r", "ft2"],
              "adapters": {"ft2": {"gguf": FT2_GGUF, "sha256": sha256_of(root / FT2_GGUF), "selection": None},
                           "ft3": {"gguf": FT3_GGUF, "sha256": sha256_of(root / FT3_GGUF), "selection": None,
                                   "model": BASE_4B, "tokenizer": TOKENIZER_4B}},
              "server": m3["server"], "tokenizer": m3["tokenizer"], "b2": {"max_cost_rmb": 5.0},
              "e2e": m3["e2e"], "e2e_routing": "stepwise", "statistics": m3["statistics"],
              "routing": {"path": ROUTING_M7, "sha256": sha256_of(root / ROUTING_M7)},
              "comparisons": [{"name": "ft3-vs-ft2", "a": "ft3", "b": "ft2", "test": "mcnemar"},
                              {"name": "ft3r-vs-ft3", "a": "ft3r", "b": "ft3", "descriptive": True}],
              "gpu": m3["gpu"]}
    config.update(changes)
    return config


def write_m7(root: Path, config: dict | None = None, message: str = "the M7 configuration") -> None:
    config = config or m7_configuration(root)
    write(root / M7_CONFIG, json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    commit(root, message)


@pytest.fixture
def m7(tmp_path, monkeypatch) -> Path:
    """A fake repository whose v1 access logs hold their three opens, with the v2 test sets and the v2
    development set locked beside them, the ft2 and ft3 adapters, the 4B base model and tokenizer folder,
    the M7 routing file and the M7 configuration committed, as the current folder and run_test's repository
    root."""
    root = tmp_path / "repo"
    make_repository(root)
    for log in (TEST_LOG, E2E_LOG):
        write_jsonl([{"milestone": m, "purpose": f"{m} (a test)", "sha256": "0" * 64, "ts": NOW}
                     for m in ("M3", "M5", "M6")], root / log)
    for path, rows in zip(V2_SETS, (TEST_ITEMS_V2, TEST_TASKS_V2), strict=True):
        locked(root / path, rows)
    locked(root / DEV_V2_PATH, DEV_V2_ITEMS)
    write(root / FT2_GGUF, b"GGUF ft2 step 1644")
    write(root / FT3_GGUF, b"GGUF ft3 step 548")
    write(root / BASE_4B, b"GGUF Qwen3-4B base")
    (root / TOKENIZER_4B).mkdir(parents=True)
    write_routing_m7(root)
    write_m7(root)
    monkeypatch.chdir(root)
    monkeypatch.setattr(run_test, "REPO", root)
    return root


def family_files(root: Path) -> dict[str, bytes | None]:
    """Both families' access logs and completion markers as bytes; None for one that does not exist."""
    paths = (TEST_LOG, E2E_LOG, COMPLETIONS, *V2_LOGS, V2_COMPLETIONS)
    return {p: (root / p).read_bytes() if (root / p).exists() else None for p in paths}


def tagged(loaded: list[str]):
    """A chat template loader that records the folders it is given and starts every prompt with the folder's
    name, so a prompt tells which tokenizer rendered it."""

    def load(folder: str):
        loaded.append(folder)
        render = prompt_renderer(FakeTokenizer())
        return lambda messages, tools: f"<!-- {folder} -->\n" + render(messages, tools)

    return load


def test_an_m7_session_opens_the_v2_sets_and_runs_ft2_then_ft3_and_ft3r_on_the_4b_server(m7, capsys):
    before = family_files(m7)
    dry_run(M7_CONFIG)
    dry = m7 / "runs" / "test-m7" / "dry-run"
    written = json.loads((dry / "session.json").read_text(encoding="utf-8"))
    # the v2 family's development set is dev and dev-v2 jointly: ft3r runs all four items, the others the
    # first, and the dev check holds ft3r against the frozen point over both sets
    assert written["dev_check"] == {"passed": True, "routed": 2, "successes": 4,
                                    "expected": {"routed": 2, "successes": 4}}
    assert written["sets"]["single_step"] == {"sets": [
        {"path": DEV_PATH, "sha256": sha256_of(m7 / DEV_PATH)},
        {"path": DEV_V2_PATH, "sha256": sha256_of(m7 / DEV_V2_PATH)}], "n": 1}
    assert [r["id"] for r in read_jsonl(dry / "ft3r.jsonl")] == [i["id"] for i in DEV_ITEMS + DEV_V2_ITEMS]
    assert [r["id"] for r in read_jsonl(dry / "ft3.jsonl")] == ["req-dev-0001-s1"]
    dry_ft3r = json.loads((dry / "results" / "test-m7-ft3r.json").read_text(encoding="utf-8"))
    assert dry_ft3r["set"]["n"] == 4 and dry_ft3r["routing"]["routed"] == 2
    assert family_files(m7) == before  # the dry run opened nothing

    machine, cloud, loaded, counted = Machine(), Cloud(), [], []
    assert session(machine, cloud, config=M7_CONFIG, load_render=tagged(loaded),
                   load_counter=lambda folder: counted.append(folder)) == 0
    assert loaded == counted == ["models/Qwen3-1.7B", TOKENIZER_4B]  # one per tokenizer folder
    assert machine.servers() == [FT2_GGUF, FT3_GGUF] and machine.models() == [BASE_17B, BASE_4B]
    firsts = {system: {prompt.split("\n", 1)[0] for s, prompt in machine.prompts if s == system}
              for system in ("ft2", "ft3")}
    assert firsts == {"ft2": {"<!-- models/Qwen3-1.7B -->"}, "ft3": {f"<!-- {TOKENIZER_4B} -->"}}
    assert [r["model"] for r in cloud.requests] == ["deepseek-flash"] * 2  # what ft3r routed, to B2
    assert machine.users() | cloud.users() == {i["messages"][1]["content"] for i in TEST_ITEMS_V2} | {
        t["text"] for t in TEST_TASKS_V2}  # the v2 material only
    after = family_files(m7)
    assert {p: after[p] for p in (TEST_LOG, E2E_LOG, COMPLETIONS)} == {
        p: before[p] for p in (TEST_LOG, E2E_LOG, COMPLETIONS)}  # the v1 family is left alone
    opens = [read_jsonl(m7 / log) for log in V2_LOGS]
    assert [[e["milestone"] for e in entries] for entries in opens] == [["M7"], ["M7"]]
    (single_open,), (e2e_open,) = opens

    ft3 = result(m7, "test-m7-ft3.json")
    assert ft3["records"] == "runs/test-m7/open-1/ft3.jsonl" and ft3["summary"]["success_rate"] == 1.0
    assert ft3["open"] == single_open and ft3["set"] == {"path": V2_SETS[0], "sha256": single_open["sha256"],
                                                         "n": 3}
    assert (ft3["info"]["system"], ft3["info"]["built_as"]) == ("ft3", "ft")
    assert (ft3["info"]["model"], ft3["info"]["tokenizer"]) == ("qwen3-4b:lora=1", TOKENIZER_4B)
    assert ft3["info"]["adapters"] == [{"id": 0, "path": FT3_GGUF, "sha256": sha256_of(m7 / FT3_GGUF)}]
    assert ft3["server"]["model_path"] == BASE_4B
    assert {r["model"] for r in read_jsonl(m7 / ft3["records"])} == {"qwen3-4b:lora=1"}
    ft2 = result(m7, "test-m7-ft2.json")
    assert (ft2["info"]["model"], ft2["server"]["model_path"]) == ("qwen3-1.7b:lora=1", BASE_17B)
    assert ft2["summary"]["success_rate"] == pytest.approx(2 / 3)
    ft3r = result(m7, "test-m7-ft3r.json")
    assert (ft3r["info"]["built_as"], ft3r["info"]["cloud_model"]) == ("ftr", "deepseek-flash")
    assert ft3r["info"]["cloud"]["system"] == "b2" and ft3r["info"]["routing"]["policy"] == "P5"
    assert ft3r["info"]["routing"]["rules"] == P5_BUDGET_RULES
    assert ft3r["info"]["routing"]["dev_sets"] == [{"path": path, "sha256": sha256_of(m7 / path)}
                                                   for path in (DEV_PATH, DEV_V2_PATH)]
    assert ft3r["info"]["model"] == "qwen3-4b:lora=1>deepseek-flash"
    assert ft3r["summary"]["success_rate"] == pytest.approx(2 / 3)
    assert ft3r["routing"] == {"n": 3, "routed": 1, "api_share": pytest.approx(1 / 3), "hard_triggers": {},
                               "reference_success": {"ft3": 1.0, "b2": None}, "random_success": None,
                               "agreement": {"local": {"reference": "ft3", "n": 2, "equal": 2},
                                             "routed": {"reference": "b2", "n": 1, "equal": None}}}
    e2e = result(m7, "test-m7-e2e-ft3r.json")
    assert e2e["open"] == e2e_open and e2e["set"]["path"] == V2_SETS[1]
    assert e2e["summary"]["success_rate"] == 1.0 and e2e["e2e_routing"] == "stepwise"
    assert e2e["routing"] == {"tasks": 2, "tasks_routed": 1, "local_calls": 3, "cloud_calls": 1,
                              "local_steps": 3, "cloud_steps": 1}

    comparisons = result(m7, "test-m7-comparisons.json")
    tested, described = comparisons["comparisons"]
    assert (tested["name"], tested["p"], tested["p_holm"]) == ("ft3-vs-ft2", 1.0, 1.0)
    assert tested["mcnemar"] == {"a_only": 1, "b_only": 0, "discordant": 1, "p": 1.0}
    assert described["descriptive"] is True and described["diff"] == pytest.approx(-1 / 3)
    assert "reference" not in comparisons

    resources = result(m7, "test-m7-resources.json")
    servers = resources["servers"]
    assert list(servers) == ["ft2", "ft3"] and servers["ft3"]["systems"] == ["ft3", "ft3r"]
    assert (servers["ft2"]["model"], servers["ft3"]["model"]) == (BASE_17B, BASE_4B)
    cost = resources["systems"]["ft3r"]["cost"]
    assert cost["single_step"]["paid_rmb"] == pytest.approx(PAID_TURN)  # priced as deepseek-flash
    assert cost["e2e"]["paid_rmb"] == pytest.approx(PAID_TURN)
    assert cost["ledger_rmb"] == pytest.approx(2 * PAID_TURN)
    written = json.loads((m7 / "runs" / "test-m7" / "open-1" / "session.json").read_text(encoding="utf-8"))
    assert {name: row["model"] for name, row in written["servers"].items()} == {"ft2": BASE_17B,
                                                                                "ft3": BASE_4B}

    markers = read_jsonl(m7 / V2_COMPLETIONS)
    assert [(m["access_log"], m["open"], m["status"]) for m in markers] == [
        (V2_LOGS[0], single_open, "complete"), (V2_LOGS[1], e2e_open, "complete")]
    assert sorted(markers[0]["outputs"]) == [f"results/test-m7-{name}.json" for name in (
        "comparisons", "e2e-ft2", "e2e-ft3", "e2e-ft3r", "ft2", "ft3", "ft3r", "resources")]
    out = capsys.readouterr().out
    assert f"opened {V2_SETS[0]} (open 1 of 3, 3 items)" in out
    assert f"{V2_COMPLETIONS}: both opens complete" in out
    (m7 / ROUTING_M7).unlink()  # the fake is not a whole routing file: the report would refuse it
    numbers, _ = report.build(m7)
    assert numbers["eval.test-m7-ft3.success"] == "100.0%"
    assert numbers["eval.test-m7-ft3r.routing.routed"] == "1"
    assert numbers["resources.test-m7.ft3r.cost.single_step.paid_rmb"] == "0.002"


def test_an_m7_dry_run_never_reads_a_v2_test_set_and_opens_nothing(m7):
    for path in V2_SETS:
        (m7 / path).write_text("not json: never parsed\n", encoding="utf-8")  # a parse would fail
        write_lock(m7 / path)
    commit(m7, "v2 test sets nobody may parse")
    before, machine, cloud = family_files(m7), Machine(), Cloud()
    assert session(machine, cloud, "--dry-run", config=M7_CONFIG) == 0
    assert family_files(m7) == before
    assert machine.users() | cloud.users() <= DEV_TEXTS | DEV_V2_TEXTS and cloud.users()
    assert machine.servers() == [FT2_GGUF, FT3_GGUF] and machine.models() == [BASE_17B, BASE_4B]
    assert not list((m7 / "results").glob("test-m7-*.json"))


def changed_v2_set(index: int):
    def change(root: Path) -> None:
        path = root / V2_SETS[index]
        path.write_bytes(path.read_bytes() + b"\n")  # committed, the lock left as it was
        commit(root, f"{V2_SETS[index]} changed")

    return change


def used_v2_opens(root: Path) -> None:
    entries = [{"milestone": "M7", "purpose": f"open {k}", "sha256": "0" * 64, "ts": NOW} for k in range(3)]
    write_jsonl(entries, root / V2_LOGS[0])
    write_jsonl([{"access_log": V2_LOGS[0], "open": entry, "status": "abandoned", "recorded": NOW}
                 for entry in read_jsonl(root / V2_LOGS[0])], root / V2_COMPLETIONS)
    commit(root, "the v2 single-step set used up")


def other_routing_m7(**changes):
    def change(root: Path) -> None:
        write_routing_m7(root, **changes)
        write_m7(root, m7_configuration(root), "another M7 routing file")

    return change


def changed_ft3_adapter(root: Path) -> None:
    (root / FT3_GGUF).write_bytes(b"GGUF ft3, retrained")


def spent_m7_budget(root: Path) -> None:
    budget = Budget(root / "runs" / "llm" / "budget.sqlite", stage="M7", stage_cap_rmb=50.0)
    budget.charge("deepseek-v4-pro", {"prompt_tokens": 5_000_000, "completion_tokens": 0})  # 45 RMB
    budget.close()


def routing_m7_on_dev_alone(root: Path) -> None:
    """The M7 routing file as one chosen on data/eval/dev.jsonl alone ("dev_sha256", as M4 to M6 wrote)."""
    data = routing_m7(root)
    del data["dev_sets"]
    data["dev_sha256"] = sha256_of(root / DEV_PATH)
    write(root / ROUTING_M7, json.dumps(data, indent=2) + "\n")
    write_m7(root, m7_configuration(root), "an M7 routing file chosen on dev alone")


def routing_m7_before_dev_v2_changed(root: Path) -> None:
    """The M7 routing file as one chosen on another dev-v2 than the one locked now."""
    data = routing_m7(root)
    data["dev_sets"][1]["sha256"] = "0" * 64
    write(root / ROUTING_M7, json.dumps(data, indent=2) + "\n")
    write_m7(root, m7_configuration(root), "an M7 routing file chosen on another dev-v2")


@pytest.mark.parametrize("change, message", [
    (changed_v2_set(0), "test-v2.jsonl does not match its lock"),
    (changed_v2_set(1), "e2e-test-v2.jsonl does not match its lock"),
    (used_v2_opens, f"{V2_SETS[0]} has no open left: {V2_LOGS[0]} holds 3 opens"),
    (other_routing_m7(cloud_model="deepseek-v4-pro"),
     f"the routing file {ROUTING_M7} routes to 'deepseek-v4-pro', not to deepseek-flash"),
    (other_routing_m7(adapter_sha256="1" * 64), f"the routing file {ROUTING_M7} records adapter_sha256 "),
    (routing_m7_on_dev_alone,
     (f"the routing file {ROUTING_M7} was chosen on 1 development set, not on the 2 of the v2 test sets "
      f"({DEV_PATH}, {DEV_V2_PATH})")),
    (routing_m7_before_dev_v2_changed,
     (f"the routing file {ROUTING_M7} records the development set 2 ({DEV_V2_PATH}) with SHA-256 "
      f"{'0' * 64}, not that of {DEV_V2_PATH}")),
    (changed_ft3_adapter, f"the ft3 adapter {FT3_GGUF} has SHA-256"),
    (spent_m7_budget, "the M7 API budget cannot take FT3+R3's cap of 5 RMB"),
])
def test_each_m7_preflight_check_refuses_with_exit_4_and_leaves_both_families_alone(m7, capsys, change,
                                                                                    message):
    dry_run(M7_CONFIG)
    change(m7)
    before, machine, cloud = family_files(m7), Machine(), Cloud()
    assert session(machine, cloud, config=M7_CONFIG) == 4
    assert message in capsys.readouterr().out
    assert family_files(m7) == before and machine.started == [] and cloud.requests == []
    assert not (m7 / "runs" / "test-m7" / "open-1").exists()


def test_without_the_v2_development_set_a_v2_session_or_dry_run_is_refused_with_exit_2(m7, capsys):
    (m7 / DEV_V2_PATH).unlink()  # not built yet (docs/m7-计划.md WP-I 5): nothing may run in its place
    commit(m7, "dev-v2 not built yet")
    before, machine, cloud = family_files(m7), Machine(), Cloud()
    for extra in (("--dry-run",), ()):
        assert session(machine, cloud, *extra, config=M7_CONFIG) == 2
        assert f"the development set {DEV_V2_PATH} of the v2 test sets is not here" in capsys.readouterr().out
    assert family_files(m7) == before and machine.started == [] and cloud.requests == []
    assert not (m7 / "runs" / "test-m7").exists()


def test_a_v2_dry_run_refuses_development_sets_that_share_an_item(m7, capsys):
    locked(m7 / DEV_V2_PATH, [DEV_ITEMS[0], *DEV_V2_ITEMS])  # req-dev-0001-s1 is data/eval/dev.jsonl's too
    write_routing_m7(m7)
    write_m7(m7, m7_configuration(m7), "a dev-v2 sharing an item with dev")
    machine = Machine()
    assert session(machine, Cloud(), "--dry-run", config=M7_CONFIG) == 4
    repeated = "item 3: its id 'req-dev-0001-s1' is missing or repeated"
    assert f"the development material of the dry run: {repeated}" in capsys.readouterr().out
    assert machine.prompts == []


def test_a_missing_base_model_or_tokenizer_folder_of_an_adapter_is_a_usage_error_with_exit_2(m7, capsys):
    before, machine = family_files(m7), Machine()
    (m7 / BASE_4B).unlink()
    assert session(machine, Cloud(), config=M7_CONFIG) == 2
    assert f"the base model {BASE_4B} of the ft3 server is not here (adapters.ft3.model)" in (
        capsys.readouterr().out)
    write(m7 / BASE_4B, b"GGUF Qwen3-4B base")
    (m7 / TOKENIZER_4B).rmdir()
    assert session(machine, Cloud(), config=M7_CONFIG) == 2
    assert f"the tokenizer folder {TOKENIZER_4B} of the ft3 server is not here (adapters.ft3.tokenizer)" in (
        capsys.readouterr().out)
    assert family_files(m7) == before and machine.started == []


def test_an_unfinished_v2_open_refuses_the_next_v2_session_until_abandoned_in_the_v2_markers(m7, capsys):
    dry_run(M7_CONFIG)
    v1 = {p: (m7 / p).read_bytes() for p in (TEST_LOG, E2E_LOG)}
    assert session(Machine(failing="/ft3/"), Cloud(), config=M7_CONFIG) == 6  # the 4B server dies after ft2
    out = capsys.readouterr().out
    assert "exited with 1 before becoming ready" in out and "the open is used up" in out
    assert [len(read_jsonl(m7 / log)) for log in V2_LOGS] == [1, 1] and not (m7 / V2_COMPLETIONS).exists()

    machine = Machine()
    assert session(machine, Cloud(), config=M7_CONFIG) == 5
    out = capsys.readouterr().out
    assert f"{V2_LOGS[0]}: the M7 open of {NOW} has no completion marker" in out
    assert all(path in out.split("--abandon", 1)[1] for path in (*V2_LOGS, V2_COMPLETIONS))
    assert machine.started == []
    assert session(machine, Cloud(), "--abandon", "the 4B server would not start", config=M7_CONFIG) == 0
    assert [(m["access_log"], m["status"]) for m in read_jsonl(m7 / V2_COMPLETIONS)] == [
        (V2_LOGS[0], "abandoned"), (V2_LOGS[1], "abandoned")]
    assert {p: (m7 / p).read_bytes() for p in v1} == v1 and not (m7 / COMPLETIONS).exists()
    commit(m7, "the crashed M7 session, abandoned")
    dry_run(M7_CONFIG)  # of the new commit
    assert session(Machine(), Cloud(), config=M7_CONFIG) == 0  # a new open of the v2 sets
    assert [len(read_jsonl(m7 / log)) for log in V2_LOGS] == [2, 2]
    assert (m7 / "runs" / "test-m7" / "open-2" / "ft3.jsonl").is_file()


def test_unfinished_the_markers_and_the_reference_outputs_are_those_of_the_family_given(tmp_path,
                                                                                       monkeypatch):
    monkeypatch.chdir(tmp_path)
    entry = {"milestone": "M7", "purpose": "an open (a test)", "sha256": "0" * 64, "ts": NOW}
    for _, log in V1.sets:
        write_jsonl([entry], log)
    assert run_test.unfinished("M7") == [(log, entry) for _, log in V1.sets]  # v1 unless told otherwise
    assert run_test.unfinished("M7", V2) == []
    for _, log in V2.sets:
        write_jsonl([entry], log)
    run_test.mark(run_test.unfinished("M7", V2), "complete", lambda: NOW, family=V2,
                  outputs={"results/test-m7-ft3.json": "a" * 64})
    assert run_test.unfinished("M7", V2) == [] and len(run_test.unfinished("M7")) == 2
    assert [m["access_log"] for m in read_jsonl(V2.completions)] == list(V2_LOGS)
    assert not V1.completions.exists()
    assert run_test.reference_outputs("M7", V2) == {"results/test-m7-ft3.json": "a" * 64}
    with pytest.raises(run_test.Refused, match="holds 0 complete M7 opens"):
        run_test.reference_outputs("M7")


# ---- the M7 configuration ---------------------------------------------------------------------------------


def m7_dict(**changes) -> dict:
    """A valid M7 configuration without a repository (made-up hashes); a change to DROP removes the key."""
    base = m6_dict()
    config = {**base, "milestone": "M7", "purpose": "M7", "test_sets": "v2",
              "systems": ["ft2", "ft3", "ft3r"],
              "adapters": {**base["adapters"],
                           "ft3": {"gguf": FT3_GGUF, "sha256": "7" * 64, "selection": None, "model": BASE_4B,
                                   "tokenizer": TOKENIZER_4B}},
              "routing": {"path": "results/routing-m7.json", "sha256": "1" * 64},
              "comparisons": [{"name": "ft3-vs-ft2", "a": "ft3", "b": "ft2", "test": "mcnemar"},
                              {"name": "ft3-vs-b0-acebench",
                               "acebench": {"a": "runs/acebench/m7/ft3", "b": "runs/acebench/m7/b0",
                                            "margin": 0.02}},
                              {"name": "ft3r-vs-ft3", "a": "ft3r", "b": "ft3", "descriptive": True}]}
    del config["reference"]
    for key, value in changes.items():
        if value is DROP:
            del config[key]
        else:
            config[key] = value
    return config


def with_ft3(**changes) -> dict:
    """The M7 adapters with ft3's entry changed; a change to DROP removes the key."""
    adapters = m7_dict()["adapters"]
    adapters["ft3"] = {**adapters["ft3"], **changes}
    adapters["ft3"] = {k: v for k, v in adapters["ft3"].items() if v is not DROP}
    return adapters


def test_m7_adds_ft3_and_ft3r_built_as_ft_and_ftr_on_the_ft3_server_ft3r_routed_to_b2():
    assert run_test.RUN_ORDER == ("ft", "b0", "b1", "ftr", "ft2", "ft2r", "ft3", "ft3r", "final", "finalg",
                                  "b4", "b2")
    assert run_test.ADAPTERS == ("ft", "ft2", "ft3", "final", "b4")
    assert run_test.SERVER_OF["ft3"] == run_test.SERVER_OF["ft3r"] == "ft3"
    assert (run_test.BUILT_AS["ft3"], run_test.BUILT_AS["ft3r"]) == ("ft", "ftr")
    assert run_test.ROUTED["ft3r"] == ("ft3", "b2") and "ft3r" in run_test.RULED  # R3's fallback is flash
    assert {cloud for _, cloud in run_test.ROUTED.values()} == {"b2"}
    assert "ft3r" in run_test.SPENDING and run_test.SPENDER_NAMES["ft3r"] == "FT3+R3"


def test_each_family_of_test_sets_has_its_development_sets_v2_dev_and_dev_v2_jointly():
    assert set(run_test.FAMILY_DEV_SETS) == set(TEST_FAMILIES)
    assert run_test.FAMILY_DEV_SETS["v1"] == (Path(DEV_PATH),)
    assert run_test.FAMILY_DEV_SETS["v2"] == (Path(DEV_PATH), Path(DEV_V2_PATH))
    assert run_test.dev_sets_of(m6_dict()) == (Path(DEV_PATH),) and run_test.dev_sets_of(m7_dict())[1] == (
        Path(DEV_V2_PATH))


def test_an_m7_configuration_of_the_v2_test_sets_with_ft3_on_its_own_base_model_is_valid():
    config = m7_dict()
    assert run_test.check_config(config) is config
    assert run_test.family_of(config) == V2
    plain = m7_dict(adapters=with_ft3(model=DROP, tokenizer=DROP))  # server.model and tokenizer then
    assert run_test.check_config(plain) is plain
    assert run_test.family_of(m6_dict()) == run_test.family_of(m6_dict(test_sets="v1")) == V1
    assert run_test.check_config(m6_dict(test_sets="v1"))


@pytest.mark.parametrize("changes, message", [
    ({"test_sets": "v3"}, "test_sets must be one of ['v1', 'v2'], not 'v3'"),
    ({"test_sets": ["v2"]}, "test_sets must be one of ['v1', 'v2'], not ['v2']"),
    ({"reference": m6_dict()["reference"]}, "reference is only for the v1 test sets"),
    ({"systems": ["ft2", "ft2r", "ft3", "ft3r"]}, "at most one routed system"),
    ({"systems": ["ft2", "ft3"], "comparisons": [{"name": "ft3-vs-ft2", "a": "ft3", "b": "ft2"}]},
     "routing is only for a routed system (ftr, ft2r, ft3r), none of which this session runs"),
    ({"adapters": m6_dict()["adapters"]}, "adapters lacks ft3"),
    ({"adapters": with_ft3(model="C:/models/Qwen3-4B-Q8_0.gguf")},
     "adapters.ft3.model must be a path relative to the repository root"),
    ({"adapters": with_ft3(tokenizer="../models/Qwen3-4B")},
     "adapters.ft3.tokenizer must be a path relative to the repository root"),
    ({"adapters": with_ft3(model=None)}, "adapters.ft3.model must be a non-empty string"),
    ({"adapters": with_ft3(base=BASE_4B)}, "adapters.ft3: unknown key base"),
])
def test_a_malformed_m7_configuration_is_refused_naming_the_key(changes, message):
    with pytest.raises(run_test.ConfigError) as refused:
        run_test.check_config(m7_dict(**changes))
    assert message in str(refused.value)


# ---- M7: FT3's generation, the family path (docs/m7-计划.md §2 推理 and 解码, WP-K, WP-M) ------------------

SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)  # the fake family's think mode samples
GENERATION_THINK = {"mode": "think", "decoding": "sampled", "reasoning_budget": 64, "base_seed": 0}
# The generation as scripts/run_eval.py records it (llm.generation.GenerationSettings.record()) of a run in
# GENERATION_THINK on that family, and of one in think mode decoded greedily (its --greedy): what the routing
# file's ft_generation and the results' info hold
RECORD_THINK = {"mode": "think", "decoding": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                              "presence_penalty": 0.0},
                "reasoning_budget": 64, "answer_tokens": 512, "base_seed": 0, "sample": 0}
RECORD_GREEDY = {**RECORD_THINK, "decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                                              "presence_penalty": 0.0}}
SAMPLED_FIELDS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty")
REASONING = "Thinking it through."
TOOL_RESPONSE = "<tool_response>"


def own_users(prompt: str) -> list[str]:
    """The user messages of a prompt, without the tool outputs the fake family's template writes as user
    turns."""
    return [user for user in USER.findall(prompt) if not user.startswith(TOOL_RESPONSE)]


class FamilyMachine(Machine):
    """Machine, whose FT3 server also answers the family path: a prompt sent as the ids of the fake family's
    tokenizer (tests/support/family_fakes.py) is decoded and answered as Machine answers FT3, the request
    being the last user message that is no tool output; in think mode with a reasoning, the close and the
    answer, in no_think mode with the answer alone, in tests/support/llama_fakes.py's form (every token
    listed with its id and bytes, the answer's at the log-probability of confidence, the generated ids and the
    stop type when the request asks for them). Every request of the family path is kept (family_requests)."""

    def __init__(self, tokenizer: FakeFamilyTokenizer, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.family_requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        running = self.process is not None and self.process.returncode is None
        if running and request.url.path == "/v1/completions":
            payload = json.loads(request.content)
            if isinstance(payload["prompt"], list):
                return self.generate(request, payload)
        return super().handler(request)

    def generate(self, request: httpx.Request, payload: dict) -> httpx.Response:
        # the tests here run FT3 alone on the family path
        assert "/ft3/" in self.adapter and payload["lora"] == [{"id": 0, "scale": 1.0}]
        self.family_requests.append(payload)
        prompt = self.tokenizer.decode(payload["prompt"])
        self.prompts.append(("ft3", prompt))
        user = own_users(prompt)[-1]
        ids = ARXIV.findall(user)
        if ids:  # an end-to-end task: the paper looked up, then the answer
            lookup = json.dumps({"name": "get_paper", "arguments": {"arxiv_id": ids[0]}})
            answer = (f"{ids[0]} is about tree verification." if TOOL_RESPONSE in prompt
                      else f"<tool_call>\n{lookup}\n</tool_call>")
        else:
            answer = BYE["content"] if "that's all" in user else CALL
        sure = math.log(0.2 if any(word in user for word in UNSURE) else 0.95)
        tokens = (REASONING, THINK_CLOSE, answer) if prompt.endswith(THINK_OPEN) else (answer,)
        scripted = Scripted(tokens, logprobs=(*[-0.1] * (len(tokens) - 1), sure))
        return FakeLlama([scripted]).handle(request)

    def users(self) -> set[str]:
        return {own_users(prompt)[-1] for _, prompt in self.prompts}


def think_configuration(root: Path, generation: dict | None = GENERATION_THINK, **changes) -> dict:
    """m7_configuration with ft3's adapter generating as generation (none when None)."""
    config = m7_configuration(root, **changes)
    if generation is not None:
        config["adapters"]["ft3"]["generation"] = generation
    return config


@pytest.fixture
def m7_think(m7, monkeypatch) -> tuple[Path, FakeFamilyTokenizer]:
    """m7 with the fake family in the 4B tokenizer folder (its think mode sampling as SAMPLING), ft3's adapter
    generating as GENERATION_THINK and the routing file chosen on FT3's runs in that generation, committed;
    (the repository, the family's tokenizer)."""
    tokenizer = install_fake_family(monkeypatch, m7 / TOKENIZER_4B,
                                    decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    write_routing_m7(m7, ft_generation=RECORD_THINK)
    write_m7(m7, think_configuration(m7), "FT3 thinks")
    return m7, tokenizer


def thinking_of(seed: int) -> dict:
    """The thinking record of an FT3 step the fake answers in think mode: two reasoning tokens (the reasoning
    and the close), the answer's one, in one request with its seed."""
    return {"mode": "think", "reasoning_tokens": 2, "answer_tokens": 1, "forced": False, "phases": 1,
            "seeds": [seed]}


def test_a_thinking_ft3_dry_run_runs_the_family_path_single_step_and_end_to_end(m7_think):
    root, tokenizer = m7_think
    before, machine, rendered, counted, families = family_files(root), FamilyMachine(tokenizer), [], [], []

    def render(folder: str):
        rendered.append(folder)
        return prompt_renderer(FakeTokenizer())

    def family(folder: str):
        families.append(folder)
        return load_family(folder)

    assert session(machine, Cloud(), "--dry-run", config=M7_CONFIG, load_render=render,
                   load_counter=counted.append, load_family=family) == 0
    dry = root / "runs" / "test-m7" / "dry-run"
    written = json.loads((dry / "session.json").read_text(encoding="utf-8"))
    # FT3+R3 thinking routes the two unsure calls of dev and dev-v2 to B2: the routing file's frozen point
    assert written["dev_check"] == {"passed": True, "routed": 2, "successes": 4,
                                    "expected": {"routed": 2, "successes": 4}}
    assert family_files(root) == before and machine.users() <= DEV_TEXTS | DEV_V2_TEXTS
    # the 1.7B folder's chat template and counter for FT2 and B2; the 4B folder's family, once, for FT3
    assert (rendered, counted, families) == (["models/Qwen3-1.7B"], ["models/Qwen3-1.7B"], [TOKENIZER_4B])
    # every request of FT3 and FT3+R3: the prompt's ids in think mode, the budget, the family's sampling and
    # a seed of its own (llm.generation.request_seed), the generated ids asked for
    requests = machine.family_requests
    prompts = [tokenizer.decode(r["prompt"]) for r in requests]
    assert len(requests) == 1 + 2 + 4 + 2  # FT3: an item, a task of two steps; FT3+R3: four items, two steps
    assert all(prompt.endswith(ASSISTANT + THINK_OPEN) for prompt in prompts)
    assert {(r["max_tokens"], r["return_tokens"], r["cache_prompt"]) for r in requests} == {(64, True, False)}
    assert all({k: r[k] for k in SAMPLED_FIELDS} == RECORD_THINK["decoding"] for r in requests)
    seeds = [request_seed(0, r["prompt"], 1, 0) for r in requests]
    assert [r["seed"] for r in requests] == seeds
    # a seed per prompt: FT3+R3 sends the item and the task FT3 sent with the same seed
    assert len(set(seeds)) == len({tuple(r["prompt"]) for r in requests}) == 6
    # the records: FT3's and FT3+R3's local ones with the reasoning and the thinking, FT2's as before
    (ft3,) = read_jsonl(dry / "ft3.jsonl")
    assert (ft3["reasoning"], ft3["thinking"]) == (REASONING, thinking_of(seeds[0]))
    assert ft3["model"] == "qwen3-4b:lora=1"
    ft3r = read_jsonl(dry / "ft3r.jsonl")
    assert [r["route"]["answered_by"] for r in ft3r] == ["cloud", "local", "cloud", "local"]
    assert [r.get("thinking") for r in ft3r] == [None, thinking_of(seeds[4]), None, thinking_of(seeds[6])]
    assert all("reasoning" not in r for r in ft3r if r["route"]["answered_by"] == "cloud")
    (ft2,) = read_jsonl(dry / "ft2.jsonl")
    assert "reasoning" not in ft2 and "thinking" not in ft2
    transcript = json.loads((dry / "e2e-ft3" / "transcripts" / "e2e-dev-01.json").read_text(encoding="utf-8"))
    attempts = [a for step in transcript["steps"] for a in step["attempts"]]
    assert [(a["reasoning"], a["thinking"]) for a in attempts] == [(REASONING, thinking_of(seeds[1])),
                                                                   (REASONING, thinking_of(seeds[2]))]
    # the results: the generation and the family in every FT3 system's build, FT2's without either
    results = {name: json.loads((dry / "results" / f"test-m7-{name}.json").read_text(encoding="utf-8"))
               for name in ("ft3", "e2e-ft3", "ft3r", "e2e-ft3r", "ft2")}
    for name in ("ft3", "e2e-ft3", "ft3r", "e2e-ft3r"):
        info = results[name]["info"]
        assert (info["generation"], info["family"]["name"], info["temperature"]) == (RECORD_THINK,
                                                                                    "fake-hybrid", 0.6), name
    assert results["ft3r"]["info"]["local"]["generation"] == RECORD_THINK
    assert not {"generation", "family"} & set(results["ft2"]["info"])


def test_a_thinking_ft3_session_opens_the_v2_sets_once_and_its_results_carry_the_generation(m7_think, capsys):
    root, tokenizer = m7_think
    assert session(FamilyMachine(tokenizer), Cloud(), "--dry-run", config=M7_CONFIG) == 0
    machine, cloud = FamilyMachine(tokenizer), Cloud()
    assert session(machine, cloud, config=M7_CONFIG) == 0
    opens = [read_jsonl(root / log) for log in V2_LOGS]
    assert [[e["milestone"] for e in entries] for entries in opens] == [["M7"], ["M7"]]
    assert machine.users() | cloud.users() == {i["messages"][1]["content"] for i in TEST_ITEMS_V2} | {
        t["text"] for t in TEST_TASKS_V2}  # the v2 material only
    open_1, requests = root / "runs" / "test-m7" / "open-1", machine.family_requests
    ft3 = result(root, "test-m7-ft3.json")
    assert ft3["summary"]["success_rate"] == 1.0 and ft3["info"]["generation"] == RECORD_THINK
    # FT3's three items, one request each, in item order: every record with its reasoning and its seed
    assert [(r["id"], r["reasoning"], r["thinking"]) for r in read_jsonl(open_1 / "ft3.jsonl")] == [
        (i["id"], REASONING, thinking_of(request_seed(0, r["prompt"], 1, 0)))
        for i, r in zip(TEST_ITEMS_V2, requests[:3], strict=True)]
    e2e = result(root, "test-m7-e2e-ft3.json")
    assert e2e["summary"]["success_rate"] == 1.0 and e2e["info"]["generation"] == RECORD_THINK
    for task in TEST_TASKS_V2:
        path = open_1 / "e2e-ft3" / "transcripts" / f"{task['id']}.json"
        steps = json.loads(path.read_text(encoding="utf-8"))["steps"]
        assert steps and all(a["reasoning"] == REASONING and a["thinking"]["mode"] == "think"
                             for step in steps for a in step["attempts"])
    # FT3+R3 routes the Chinese item, which B2 gets wrong, and answers the two others thinking, as FT3+R3 on
    # the legacy path does
    ft3r = result(root, "test-m7-ft3r.json")
    assert ft3r["summary"]["success_rate"] == pytest.approx(2 / 3)
    assert ft3r["info"]["generation"] == RECORD_THINK and ft3r["info"]["routing"]["rules"] == P5_BUDGET_RULES
    assert (ft3r["routing"]["routed"], ft3r["routing"]["hard_triggers"]) == (1, {})
    assert ft3r["routing"]["agreement"]["local"] == {"reference": "ft3", "n": 2, "equal": 2}
    assert result(root, "test-m7-e2e-ft3r.json")["summary"]["success_rate"] == 1.0
    ft2 = result(root, "test-m7-ft2.json")
    assert "generation" not in ft2["info"] and ft2["summary"]["success_rate"] == pytest.approx(2 / 3)
    tested, _ = result(root, "test-m7-comparisons.json")["comparisons"]
    assert (tested["name"], tested["mcnemar"]["a_only"], tested["p"]) == ("ft3-vs-ft2", 1, 1.0)
    assert [m["status"] for m in read_jsonl(root / V2_COMPLETIONS)] == ["complete", "complete"]
    assert f"{V2_COMPLETIONS}: both opens complete" in capsys.readouterr().out


def test_a_greedy_thinking_ft3_sends_no_seed_whatever_its_familys_sampling(m7, monkeypatch):
    tokenizer = install_fake_family(monkeypatch, m7 / TOKENIZER_4B,
                                    decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    write_routing_m7(m7, ft_generation=RECORD_GREEDY)
    greedy = {"mode": "think", "decoding": "greedy", "reasoning_budget": 64}
    write_m7(m7, think_configuration(m7, greedy), "FT3 thinks, decoded greedily")
    machine = FamilyMachine(tokenizer)
    assert session(machine, Cloud(), "--dry-run", config=M7_CONFIG) == 0
    requests = machine.family_requests
    assert len(requests) == 9 and {(r["temperature"], r["max_tokens"]) for r in requests} == {(0.0, 64)}
    # temperature 0 alone: no seed, no other sampling field
    assert not any({"seed", "top_p", "top_k", "min_p", "presence_penalty"} & set(r) for r in requests)
    dry = m7 / "runs" / "test-m7" / "dry-run"
    (ft3,) = read_jsonl(dry / "ft3.jsonl")
    assert (ft3["reasoning"], ft3["thinking"]["mode"], ft3["thinking"]["seeds"]) == (REASONING, "think", None)
    for name in ("ft3", "ft3r"):
        info = json.loads((dry / "results" / f"test-m7-{name}.json").read_text(encoding="utf-8"))["info"]
        assert (info["generation"], info["temperature"]) == (RECORD_GREEDY, 0.0), name


def test_a_sampled_ft3_draws_every_seed_from_the_base_seed_its_adapter_names(m7, monkeypatch):
    """Base seed 3, not the 0 a greedy decoding stands on: the settings, the seed of every request of FT3 and
    FT3+R3, single step and end to end, the records' thinking and the results' generation all come from it,
    and the routing file was chosen on FT3's runs from it."""
    tokenizer = install_fake_family(monkeypatch, m7 / TOKENIZER_4B,
                                    decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    seeded = {**RECORD_THINK, "base_seed": 3}
    write_routing_m7(m7, ft_generation=seeded)
    config = think_configuration(m7, {**GENERATION_THINK, "base_seed": 3})
    write_m7(m7, config, "FT3 samples from base seed 3")
    assert run_test.generations(run_test.check_config(config))["ft3"].settings.base_seed == 3
    machine = FamilyMachine(tokenizer)
    assert session(machine, Cloud(), "--dry-run", config=M7_CONFIG) == 0
    requests = machine.family_requests
    seeds = [request_seed(3, r["prompt"], 1, 0) for r in requests]
    assert len(requests) == 9 and [r["seed"] for r in requests] == seeds
    assert seeds != [request_seed(0, r["prompt"], 1, 0) for r in requests]  # base seed 0 draws others
    dry = m7 / "runs" / "test-m7" / "dry-run"
    (ft3,) = read_jsonl(dry / "ft3.jsonl")
    assert (ft3["reasoning"], ft3["thinking"]) == (REASONING, thinking_of(seeds[0]))
    for name in ("ft3", "e2e-ft3", "ft3r", "e2e-ft3r"):
        info = json.loads((dry / "results" / f"test-m7-{name}.json").read_text(encoding="utf-8"))["info"]
        assert info["generation"] == seeded, name


def test_the_family_path_is_resolved_per_served_adapter_with_a_generation_its_folder_loaded_once(tmp_path,
                                                                                                monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_fake_family(monkeypatch, tmp_path / TOKENIZER_4B,
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    loaded = []

    def load(folder: str):
        loaded.append(folder)
        return load_family(folder)

    no_think = {"mode": "no_think", "decoding": "greedy", "reasoning_budget": 64}
    adapters = {**m7_dict()["adapters"],
                "ft": {"gguf": FT_GGUF, "sha256": "9" * 64, "selection": None, "tokenizer": TOKENIZER_4B,
                       "generation": no_think}}
    adapters["ft3"]["generation"] = GENERATION_THINK
    config = run_test.check_config(m7_dict(adapters=adapters, systems=["ft", "ft2", "ft3", "ft3r"],
                                           routing={"path": "results/routing-m7.json", "sha256": "1" * 64}))
    generated = run_test.generations(config, load)
    # ft and ft3 share one folder, loaded once; ft2's server runs on the legacy path
    assert list(generated) == ["ft", "ft3"] and loaded == [TOKENIZER_4B]
    assert generated["ft"].family is generated["ft3"].family
    assert generated["ft3"].settings.record() == RECORD_THINK
    assert generated["ft"].settings.record() == {**RECORD_GREEDY, "mode": "no_think"}
    assert run_test.tokenizers(config) == ["models/Qwen3-1.7B"]
    # each counts its context on the prompt in its own mode: in think mode the prompt ends with the open
    # block, "<think>" and "\n"; in no_think mode with the block closed empty, "<think>", "\n\n", "</think>"
    # and "\n\n" (the fake tokenizer's tokens): two more
    messages, tools = [SYSTEM, {"role": "user", "content": "Find hallucination papers."}], []
    think, plain = (generated[name].count_tokens(messages, tools) for name in ("ft3", "ft"))
    assert plain - think == 2


def test_a_mode_its_family_lacks_or_a_sampling_it_does_not_have_is_a_configuration_error(m7, monkeypatch,
                                                                                         capsys):
    write_routing_m7(m7, ft_generation=RECORD_THINK)
    write_m7(m7, think_configuration(m7), "FT3 thinks")
    before, machine = family_files(m7), Machine()
    install_fake_family(monkeypatch, m7 / TOKENIZER_4B, "none")  # a family that never thinks
    for extra in (("--dry-run",), ()):
        assert session(machine, Cloud(), *extra, config=M7_CONFIG) == 1
        assert ("adapters.ft3.generation.mode: fake-none has no think mode (its modes: no_think)"
                in capsys.readouterr().out)
    install_fake_family(monkeypatch, m7 / TOKENIZER_4B)  # it thinks, but decodes its think mode greedily
    assert session(machine, Cloud(), "--dry-run", config=M7_CONFIG) == 1
    assert ("adapters.ft3.generation.decoding: sampled, but fake-hybrid decodes its think mode greedily"
            in capsys.readouterr().out)
    assert family_files(m7) == before and machine.started == []
    assert not (m7 / "runs" / "test-m7").exists()


def test_a_family_path_tokenizer_folder_that_is_no_registered_family_is_a_missing_prerequisite(m7, capsys):
    write_routing_m7(m7, ft_generation=RECORD_THINK)
    write_m7(m7, think_configuration(m7), "FT3 thinks")  # models/Qwen3-4B holds no model files
    machine = Machine()
    assert session(machine, Cloud(), "--dry-run", config=M7_CONFIG) == 2
    out = capsys.readouterr().out
    assert f"the family of {TOKENIZER_4B} (adapters.ft3.tokenizer) cannot be loaded" in out
    assert machine.started == [] and not (m7 / "runs" / "test-m7").exists()


@pytest.mark.parametrize("routing_changes, generation, message", [
    ({}, GENERATION_THINK,
     (f"the routing file {ROUTING_M7} records no ft_generation, while the systems on the ft3 server run with "
      "the generation")),
    ({"ft_generation": RECORD_GREEDY}, GENERATION_THINK,
     (f"the routing file {ROUTING_M7} records ft_generation {canonical_json(RECORD_GREEDY)}, not "
      f"{canonical_json(RECORD_THINK)}, the generation of adapters.ft3")),
    ({"ft_generation": RECORD_THINK}, {**GENERATION_THINK, "base_seed": 3},
     (f"the routing file {ROUTING_M7} records ft_generation {canonical_json(RECORD_THINK)}, not "
      f"{canonical_json({**RECORD_THINK, 'base_seed': 3})}, the generation of adapters.ft3")),
    ({"ft_generation": RECORD_THINK}, None,
     (f"the routing file {ROUTING_M7} records ft_generation {canonical_json(RECORD_THINK)}, while "
      "adapters.ft3 names no generation")),
], ids=["none-in-the-file", "another-in-the-file", "another-base-seed-in-the-file", "none-in-the-adapter"])
def test_the_routing_file_must_be_chosen_on_ft3s_runs_in_the_generation_ft3r_runs_with(
        m7, monkeypatch, capsys, routing_changes, generation, message):
    install_fake_family(monkeypatch, m7 / TOKENIZER_4B,
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    write_routing_m7(m7, **routing_changes)
    write_m7(m7, think_configuration(m7, generation), "a routing file of another generation")
    before, machine, cloud = family_files(m7), Machine(), Cloud()
    for extra in (("--dry-run",), ()):
        assert session(machine, cloud, *extra, config=M7_CONFIG) == 4
        assert message in capsys.readouterr().out
    assert family_files(m7) == before and machine.started == [] and cloud.requests == []
    assert not (m7 / "runs" / "test-m7").exists()


# ---- the M7 configuration: an adapter's generation ---------------------------------------------------------


def generated(generation: object) -> dict:
    """m7_dict's adapters with ft3's generation given."""
    return with_ft3(generation=generation)


def test_an_adapters_generation_is_its_mode_decoding_reasoning_budget_and_a_sampled_ones_base_seed():
    for generation in (GENERATION_THINK, {"mode": "think", "decoding": "greedy", "reasoning_budget": 1280},
                       {"mode": "no_think", "decoding": "greedy", "reasoning_budget": 1024},
                       {"mode": "no_think", "decoding": "sampled", "reasoning_budget": 1, "base_seed": 7}):
        config = m7_dict(adapters=generated(generation))
        assert run_test.check_config(config) is config


@pytest.mark.parametrize("generation, message", [
    ("think", "adapters.ft3.generation must be an object"),
    ({"mode": "think", "decoding": "greedy"}, "adapters.ft3.generation lacks reasoning_budget"),
    ({**GENERATION_THINK, "seed": 0}, "adapters.ft3.generation: unknown key seed"),
    ({**GENERATION_THINK, "mode": "thinking"},
     "adapters.ft3.generation.mode must be think or no_think, not 'thinking'"),
    ({**GENERATION_THINK, "mode": None}, "adapters.ft3.generation.mode must be think or no_think, not None"),
    ({**GENERATION_THINK, "decoding": "beam"},
     "adapters.ft3.generation.decoding must be greedy or sampled, not 'beam'"),
    ({**GENERATION_THINK, "reasoning_budget": 0},
     "adapters.ft3.generation.reasoning_budget must be an integer of at least 1, not 0"),
    ({**GENERATION_THINK, "reasoning_budget": 64.0},
     "adapters.ft3.generation.reasoning_budget must be an integer of at least 1, not 64.0"),
    ({**GENERATION_THINK, "reasoning_budget": True},
     "adapters.ft3.generation.reasoning_budget must be an integer of at least 1, not True"),
    ({"mode": "think", "decoding": "greedy", "reasoning_budget": 64, "base_seed": 0},
     "adapters.ft3.generation.base_seed: a greedy decoding sends no seed"),
    ({"mode": "think", "decoding": "sampled", "reasoning_budget": 64},
     "adapters.ft3.generation lacks base_seed"),
    ({**GENERATION_THINK, "base_seed": -1},
     "adapters.ft3.generation.base_seed must be an integer of at least 0, not -1"),
    ({**GENERATION_THINK, "base_seed": "0"},
     "adapters.ft3.generation.base_seed must be an integer of at least 0, not '0'"),
])
def test_a_malformed_generation_is_refused_naming_the_key(generation, message):
    with pytest.raises(run_test.ConfigError) as refused:
        run_test.check_config(m7_dict(adapters=generated(generation)))
    assert message in str(refused.value)


def test_a_thinking_servers_context_holds_the_prompt_limit_the_reasoning_budget_and_520_tokens_more():
    # docs/m7-计划.md §2 端到端预算: -c >= 4096 + R + 520, 4680 tokens for R = 64
    server = m7_dict()["server"]
    at = m7_dict(adapters=generated(GENERATION_THINK), server={**server, "ctx_size": 4680})
    assert run_test.check_config(at) is at
    below = m7_dict(adapters=generated(GENERATION_THINK), server={**server, "ctx_size": 4679})
    with pytest.raises(run_test.ConfigError) as refused:
        run_test.check_config(below)
    assert "adapters.ft3.generation: server.ctx_size 4679 is below 4680" in str(refused.value)
    # without thinking the context holds no reasoning: no bound of its own
    plain = m7_dict(adapters=generated({"mode": "no_think", "decoding": "greedy", "reasoning_budget": 64}),
                    server={**server, "ctx_size": 4679})
    assert run_test.check_config(plain) is plain


# ---- without a generation, every session writes what it wrote before ---------------------------------------


def written(root: Path, milestone: str) -> dict[str, str]:
    """The SHA-256 of every file the milestone's dry run and session wrote, by its path relative to root:
    results/test-<m>-*.json, everything under runs/test-<m>/ but the servers' logs, and the access logs and
    completion markers of both families of test sets."""
    m = milestone.lower()
    logs = (TEST_LOG, E2E_LOG, COMPLETIONS, *V2_LOGS, V2_COMPLETIONS)
    paths = [*(root / "results").glob(f"test-{m}-*.json"),
             *(p for p in (root / "runs" / f"test-{m}").rglob("*")
               if p.is_file() and not p.name.startswith("server-")),
             *(root / p for p in logs if (root / p).exists())]
    return {p.relative_to(root).as_posix(): sha256_of(p) for p in sorted(paths)}


def golden(root: Path, milestone: str) -> tuple[int, str]:
    """How many files the milestone's sessions wrote (written) and the SHA-256 of their canonical JSON."""
    files = written(root, milestone)
    return len(files), hashlib.sha256(canonical_json(files).encode("utf-8")).hexdigest()


# Computed with scripts/run_test.py and research_agent of commit 64250c0 (before an adapter could name a
# generation) on the fake repositories of this file (their LF inputs): golden() of the M3 session of m3_done,
# the M5 session of m5_done, and the dry run and session of M6 on m6 and of M7 on m7 (ft2, ft3 and ft3r on the
# legacy path)
SESSIONS_GOLDEN = {"M3": (74, "eca46acfd0ad5ccf01371a07ad53b9a0fa4b705c74dbaad95eabd09442eeab94"),
                   "M5": (22, "09b0153d7a00d6f9d399cfe3b35acaa85a37d6eafb682ba40f4413b7180b6218"),
                   "M6": (35, "274aac7da0dfdcb57fc47b179be8ede1a55e113e47503b31bd0515b6193563ad"),
                   "M7": (50, "fbc08de3b675a77b032e8e2f5e1618cee2c286ae3b83ab126a21fe75512f503e"),
                   # the dry run and session of M7's final system on m7_final (its fake family's answers, its
                   # development runs, its frozen file), computed with the code that added it (WP5, 10-30)
                   "M7-final": (56, "df9d74a19ecbd367c987378ca453888c9bf6581404072cf1e7fc4f9b90b013c5")}


def test_without_a_generation_the_m3_and_m5_sessions_wrote_byte_for_byte_what_they_wrote_before(m3_done,
                                                                                                 m5_done):
    assert golden(m3_done, "M3") == SESSIONS_GOLDEN["M3"]
    assert golden(m5_done, "M5") == SESSIONS_GOLDEN["M5"]


def test_without_a_generation_an_m6_session_writes_byte_for_byte_what_it_wrote_before(m6):
    dry_run(M6_CONFIG)
    assert session(Machine(), Cloud(), config=M6_CONFIG) == 0
    assert golden(m6, "M6") == SESSIONS_GOLDEN["M6"]


def test_without_a_generation_an_m7_session_writes_byte_for_byte_what_it_wrote_before(m7):
    dry_run(M7_CONFIG)
    assert session(Machine(), Cloud(), config=M7_CONFIG) == 0
    assert golden(m7, "M7") == SESSIONS_GOLDEN["M7"]


# ---- M7: the final local system (docs/m7-本地提升计划.md WP5; docs/m7-预登记.md, revision 2) ------------

run_eval = load_script("run_eval")
freeze_final = load_script("freeze_final")

# FT3's checkpoint 1635 (in FT3's folder, which FamilyMachine answers on the family path) over its own base
FINAL_GGUF = "runs/train/ft3/gguf/checkpoint-1635-f16.gguf"
BASE_NB, TOKENIZER_NB = "models/Nanbeige4.1-3B-Q8_0.gguf", "models/Nanbeige4.1-3B"
FROZEN = "results/final-m7.json"
GENERATION_FINAL = {"mode": "think", "decoding": "sampled", "reasoning_budget": 64, "base_seed": 0}
VOTES = 2  # the samples of the fixture's vote
ACE_FT_M7, ACE_B0_M7 = "runs/acebench/m7/ft", "runs/acebench/m7/b0"
FINAL_RUNS = {DEV_PATH: "runs/m7/dev-final", DEV_V2_PATH: "runs/m7/dev-v2-final"}  # the development runs
ALL_DEV = [i["id"] for i in DEV_ITEMS + DEV_V2_ITEMS]


def final_configuration(root: Path, frozen_sha256: str = "0" * 64, **changes) -> dict:
    """The M7 configuration of the final system: FT2, final (FT3 voting over VOTES samples in the final key
    mode with the runtime rules, frozen in FROZEN), finalg (its greedy self, single step only) and B2; the
    normalized scorer beside the default one; H1 on the normalized scorer, H3 on ACEBench and three
    descriptive rows."""
    m3 = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    config = {"milestone": "M7", "purpose": "M7: the final local system on the v2 test sets (a test)",
              "test_sets": "v2", "scorers": ["normalized"],
              "systems": {"ft2": {},
                          "final": {"votes": VOTES, "vote_mode": "final", "rules": True, "glossary": None,
                                    "frozen": {"path": FROZEN, "sha256": frozen_sha256}},
                          "finalg": {"decoding": "greedy", "end_to_end": False}, "b2": {}},
              "adapters": {"ft2": {"gguf": FT2_GGUF, "sha256": sha256_of(root / FT2_GGUF), "selection": None},
                           "final": {"gguf": FINAL_GGUF, "sha256": sha256_of(root / FINAL_GGUF),
                                     "selection": None, "model": BASE_NB, "tokenizer": TOKENIZER_NB,
                                     "generation": GENERATION_FINAL}},
              "server": m3["server"], "tokenizer": m3["tokenizer"], "b2": {"max_cost_rmb": 5.0},
              "e2e": m3["e2e"], "statistics": m3["statistics"],
              "comparisons": [{"name": "final-vs-ft2", "a": "final", "b": "ft2", "test": "mcnemar",
                               "scorer": "normalized"},
                              {"name": "final-vs-b0-acebench",
                               "acebench": {"a": ACE_FT_M7, "b": ACE_B0_M7, "margin": 0.02}},
                              {"name": "final-vs-finalg", "a": "final", "b": "finalg", "scorer": "normalized",
                               "descriptive": True},
                              {"name": "final-vs-b2", "a": "final", "b": "b2", "scorer": "normalized",
                               "descriptive": True},
                              {"name": "final-vs-ft2-default", "a": "final", "b": "ft2", "scorer": "default",
                               "descriptive": True}],
              "gpu": m3["gpu"]}
    config.update(changes)
    return config


def write_final(root: Path, config: dict | None = None, message: str = "the M7 final configuration") -> None:
    """The configuration (by default final_configuration recording the frozen file's SHA-256), committed."""
    config = config or final_configuration(root, sha256_of(root / FROZEN))
    write(root / M7_CONFIG, json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    commit(root, message)


def install_nanbeige(monkeypatch, root: Path) -> FakeFamilyTokenizer:
    """The fake family in the final adapter's tokenizer folder, its think mode sampling as SAMPLING."""
    return install_fake_family(monkeypatch, root / TOKENIZER_NB,
                               decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})


def development_runs(tokenizer: FakeFamilyTokenizer) -> dict[str, list[str]]:
    """The final adapter's development runs as scripts/run_eval.py makes them on the fake server: per set the
    greedy run (--greedy) and one run per sample (--seed k), each with its manifest."""
    machine = FamilyMachine(tokenizer)
    machine.adapter, machine.model, machine.process = FINAL_GGUF, BASE_NB, ServerProcess([READY_LINE])
    client = machine.client("http://127.0.0.1:8080")
    runs: dict[str, list[str]] = {}
    for dev, stem in FINAL_RUNS.items():
        runs[dev] = []
        for name, extra in (("greedy", ["--greedy"]), *((f"s{k}", ["--seed", str(k)]) for k in range(VOTES))):
            out = f"{stem}-{name}.jsonl"
            argv = [dev, "--system", "ft", "--out", out, "--tokenizer", TOKENIZER_NB, "--mode", "think",
                    "--reasoning-budget", "64", *extra]
            assert run_eval.main(argv, local_client=client, clock=ticking()) == 0
            runs[dev].append(out)
    return runs


@pytest.fixture(scope="module")
def m7_final_done(tmp_path_factory) -> Path:
    """A fake repository like m7's (the v1 access logs used up, the v2 sets and dev-v2 locked) with the ft2
    and final adapters, the final adapter's base model and tokenizer folder (the fake family), the fabricated
    ACEBench runs, the final adapter's development runs, the frozen file scripts/freeze_final.py wrote from
    them and the M7 configuration of the final system recording its SHA-256, committed."""
    root = tmp_path_factory.mktemp("m7final") / "repo"
    make_repository(root)
    for log in (TEST_LOG, E2E_LOG):
        write_jsonl([{"milestone": m, "purpose": f"{m} (a test)", "sha256": "0" * 64, "ts": NOW}
                     for m in ("M3", "M5", "M6")], root / log)
    for path, rows in zip(V2_SETS, (TEST_ITEMS_V2, TEST_TASKS_V2), strict=True):
        locked(root / path, rows)
    locked(root / DEV_V2_PATH, DEV_V2_ITEMS)
    write(root / FT2_GGUF, b"GGUF ft2 step 1644")
    write(root / FINAL_GGUF, b"GGUF ft3 step 1635")
    write(root / BASE_NB, b"GGUF Nanbeige4.1-3B base")
    write_acebench(root, ACE_FT_M7, ACE_RIGHT["ft2"])
    write_acebench(root, ACE_B0_M7, ACE_RIGHT["b0"])
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        patch.setattr(run_test, "REPO", root)
        patch.setattr(run_eval, "git_commit", lambda repo: "0" * 40)  # the manifests name no real commit
        patch.setattr(run_eval, "git_dirty", lambda repo: False)
        runs = development_runs(install_nanbeige(patch, root))
        # the configuration before the frozen file's SHA-256 is known, which scripts/freeze_final.py reads
        write(root / M7_CONFIG, json.dumps(final_configuration(root), indent=2) + "\n")
        arguments = [arg for dev, paths in runs.items() for arg in ("--runs", dev, *paths)]
        assert freeze_final.main(["--config", M7_CONFIG, *arguments]) == 0
        write_final(root)
    return root


@pytest.fixture
def m7_final(m7_final_done, tmp_path, monkeypatch) -> tuple[Path, FakeFamilyTokenizer]:
    """A copy of m7_final_done as the current folder and run_test's repository root, its fake family
    installed; (the repository, the family's tokenizer)."""
    root = tmp_path / "repo"
    shutil.copytree(m7_final_done, root)
    monkeypatch.chdir(root)
    monkeypatch.setattr(run_test, "REPO", root)
    return root, install_nanbeige(monkeypatch, root)


def dev_check(root: Path) -> dict:
    return json.loads((root / "runs" / "test-m7" / "dry-run" / "session.json").read_text(encoding="utf-8"))[
        "dev_check"]


def test_an_m7_final_dry_run_holds_final_and_finalg_against_the_frozen_file_on_every_dev_item(m7_final):
    root, tokenizer = m7_final
    before, machine = family_files(root), FamilyMachine(tokenizer)
    assert session(machine, Cloud(), "--dry-run", config=M7_CONFIG) == 0
    assert family_files(root) == before  # nothing opened
    assert dev_check(root) == {"passed": True, "routed": None, "successes": None, "expected": None,
                               "frozen": {"path": FROZEN, "sha256": sha256_of(root / FROZEN), "passed": True,
                                          "final": {"n": 4, "differ": {"default": [], "normalized": []},
                                                    "outputs_differ": 0},
                                          "finalg": {"n": 4, "outputs_differ": []}}}
    dry = root / "runs" / "test-m7" / "dry-run"
    for system in ("final", "finalg"):  # every development item, dev then dev-v2; FT2 and B2 the first
        assert [r["id"] for r in read_jsonl(dry / f"{system}.jsonl")] == ALL_DEV
    assert [r["id"] for r in read_jsonl(dry / "ft2.jsonl")] == [ALL_DEV[0]]
    # final: per step the greedy candidate and the samples, each with the rules, and the vote in its record;
    # finalg: one greedy request per item and no end-to-end run
    assert all(r["votes"]["k"] == VOTES and r["votes"]["mode"] == "final" and r["rules"] == []
               for r in read_jsonl(dry / "final.jsonl"))
    assert not any("votes" in r or "rules" in r for r in read_jsonl(dry / "finalg.jsonl"))
    assert len(machine.family_requests) == (4 + 2) * (1 + VOTES) + 4  # final's items and task steps, finalg's
    assert (dry / "e2e-final" / "transcripts" / "e2e-dev-01.json").is_file()
    assert not (dry / "e2e-finalg").exists() and not (dry / "results" / "test-m7-e2e-finalg.json").exists()


def test_an_m7_final_session_runs_ft2_then_final_and_finalg_on_one_server_then_b2_under_both_scorers(m7_final,
                                                                                                     capsys):
    root, tokenizer = m7_final
    assert session(FamilyMachine(tokenizer), Cloud(), "--dry-run", config=M7_CONFIG) == 0
    machine, cloud = FamilyMachine(tokenizer), Cloud()
    assert session(machine, cloud, config=M7_CONFIG) == 0
    assert machine.servers() == [FT2_GGUF, FINAL_GGUF] and machine.models() == [BASE_17B, BASE_NB]
    assert [len(read_jsonl(root / log)) for log in V2_LOGS] == [1, 1]
    assert machine.users() | cloud.users() == {i["messages"][1]["content"] for i in TEST_ITEMS_V2} | {
        t["text"] for t in TEST_TASKS_V2}  # the v2 material only

    final = result(root, "test-m7-final.json")
    keys = list(final)
    # the default scorer's results where they always were, the normalized one's right after the subsets
    assert keys[keys.index("subsets") + 1] == "scorers" and "scorer" not in final
    block = final["scorers"]["normalized"]
    assert list(block) == ["summary", "subsets", "scores"]
    assert list(block["subsets"]) == ["drafted", "handwritten"]
    assert [s["id"] for s in block["scores"]] == [i["id"] for i in TEST_ITEMS_V2]
    assert final["summary"]["success_rate"] == block["summary"]["success_rate"] == 1.0
    info = final["info"]
    assert (info["system"], info["built_as"], info["vote_mode"]) == ("final", "ft", "final")
    assert info["generation"]["votes"] == VOTES and info["rules"] == {"version": RULES_VERSION,
                                                                      "names": list(RULES)}
    assert final["votes"] == {"steps": 3, "unanimous": 3, "agreement": 1.0, "chosen_not_greedy": 0,
                              "no_sample_voted": 0, "candidate_errors": 0}
    assert final["rules"] == {"steps": 3, "fired": {"transient_retry": 0, "id_repair": 0}}
    finalg = result(root, "test-m7-finalg.json")
    assert finalg["info"]["generation"]["decoding"]["temperature"] == 0.0
    assert "votes" not in finalg["info"]["generation"] and not {"rules", "vote_mode"} & set(finalg["info"])
    assert "votes" not in finalg and finalg["scorers"]["normalized"]["summary"]["success_rate"] == 1.0
    assert not (root / "results" / "test-m7-e2e-finalg.json").exists()
    e2e = result(root, "test-m7-e2e-final.json")
    assert e2e["summary"]["success_rate"] == 1.0
    assert (e2e["votes"]["steps"], e2e["votes"]["unanimous"]) == (4, 4)  # two tasks of two steps
    assert e2e["rules"] == {"steps": 4, "fired": {"transient_retry": 0, "id_repair": 0}}
    assert "scorers" in result(root, "test-m7-ft2.json")
    assert "votes" not in result(root, "test-m7-e2e-ft2.json")

    rows = {row["name"]: row for row in result(root, "test-m7-comparisons.json")["comparisons"]}
    assert list(rows) == ["final-vs-ft2", "final-vs-b0-acebench", "final-vs-finalg", "final-vs-b2",
                          "final-vs-ft2-default"]
    h1, h3 = rows["final-vs-ft2"], rows["final-vs-b0-acebench"]
    assert (h1["scorer"], h1["test"], h1["mcnemar"]["a_only"]) == ("normalized", "mcnemar", 1)
    assert h1["success_b"] == pytest.approx(2 / 3) and "scorer" not in h3
    assert [name for name, row in rows.items() if "p_holm" in row] == ["final-vs-ft2", "final-vs-b0-acebench"]
    assert [h1["p_holm"], h3["p_holm"]] == holm([h1["p"], h3["p"]])
    assert rows["final-vs-ft2-default"]["scorer"] == "default" and "p" not in rows["final-vs-ft2-default"]
    assert (rows["final-vs-finalg"]["diff"], rows["final-vs-b2"]["scorer"]) == (0.0, "normalized")
    resources = result(root, "test-m7-resources.json")
    assert list(resources["servers"]) == ["ft2", "final"]
    assert resources["servers"]["final"]["systems"] == ["final", "finalg"]
    assert resources["systems"]["finalg"]["e2e_step"]["n"] == 0
    markers = read_jsonl(root / V2_COMPLETIONS)
    assert sorted(markers[0]["outputs"]) == [f"results/test-m7-{name}.json" for name in (
        "b2", "comparisons", "e2e-b2", "e2e-final", "e2e-ft2", "final", "finalg", "ft2", "resources")]
    numbers, _ = report.build(root)
    assert numbers["eval.test-m7-final.success"] == "100.0%"
    assert numbers["eval.test-m7-final-normalized.success"] == "100.0%"
    assert numbers["eval.test-m7-final-normalized-handwritten.success"] == "100.0%"
    assert numbers["eval.test-m7-final-normalized.errors.unclassified"] == "0"
    assert numbers["compare.test-m7.final-vs-ft2.diff"] == "+33.3"
    out = capsys.readouterr().out
    assert "finalg: single step 3 items, success 100.0%; end to end not run" in out


def test_an_m7_final_session_writes_byte_for_byte_what_it_wrote_when_frozen(m7_final):
    root, tokenizer = m7_final
    assert session(FamilyMachine(tokenizer), Cloud(), "--dry-run", config=M7_CONFIG) == 0
    assert session(FamilyMachine(tokenizer), Cloud(), config=M7_CONFIG) == 0
    assert golden(root, "M7") == SESSIONS_GOLDEN["M7-final"]


def unfrozen(root: Path) -> None:
    (root / FROZEN).unlink()
    commit(root, "the frozen file removed")


def frozen_changed(root: Path) -> None:
    (root / FROZEN).write_bytes((root / FROZEN).read_bytes() + b"\n")
    commit(root, "the frozen file changed, its SHA-256 in the configuration not")


def refrozen(**changes):
    """The frozen file with fields changed (a callable changes the field it is given), and the configuration
    recording its new SHA-256, committed."""

    def change(root: Path) -> None:
        data = json.loads((root / FROZEN).read_text(encoding="utf-8"))
        for key, value in changes.items():
            data[key] = value(data[key]) if callable(value) else value
        write(root / FROZEN, json.dumps(data, indent=2) + "\n")
        write_final(root, message="another frozen file")

    return change


def changed_entry(position: int, **entry):
    """A system's frozen block with the entry of the item at position changed (a callable changes the value
    it is given)."""

    def change(block: dict) -> dict:
        items = [dict(e) for e in block["items"]]
        items[position] = {**items[position], **{k: v(items[position][k]) if callable(v) else v
                                                 for k, v in entry.items()}}
        return {**block, "items": items}

    return change


def flipped(scorer: str):
    def flip(success: dict) -> dict:
        return {**success, scorer: not success[scorer]}

    return flip


@pytest.mark.parametrize("change, message", [
    (unfrozen, f"the frozen file {FROZEN} is missing (systems.final.frozen.path)"),
    (frozen_changed, f"the frozen file {FROZEN} has SHA-256"),
    (refrozen(adapter={"path": FINAL_GGUF, "sha256": "1" * 64}),
     f"the frozen file {FROZEN} records adapter "),
    (refrozen(base=lambda base: {**base, "sha256": "2" * 64}), f"the frozen file {FROZEN} records base "),
    (refrozen(generation=lambda record: {**record, "votes": 3}),
     f"the frozen file {FROZEN} records generation "),
    (refrozen(vote_mode="reproduction"), f"the frozen file {FROZEN} records vote_mode "),
    (refrozen(rules={"version": "m7-wp3-0", "names": list(RULES)}),
     f"the frozen file {FROZEN} records rules "),
    (refrozen(glossary={"path": "data/glossary/zh-en.json", "sha256": "3" * 64}),
     f"the frozen file {FROZEN} records glossary "),
    (refrozen(dev_sets=lambda sets: [sets[0], {**sets[1], "sha256": "0" * 64}]),
     f"the frozen file {FROZEN} records dev_sets "),
    (refrozen(finalg=lambda block: {**block, "generation": {**block["generation"], "base_seed": 1}}),
     f"the frozen file {FROZEN} records finalg.generation "),
], ids=["missing", "sha256", "adapter", "base", "generation", "vote-mode", "rules-version", "glossary",
        "dev-set", "finalg-generation"])
def test_each_frozen_file_check_refuses_with_exit_4_before_anything_is_opened_or_started(m7_final, capsys,
                                                                                          change, message):
    root, tokenizer = m7_final
    change(root)
    before, machine, cloud = family_files(root), FamilyMachine(tokenizer), Cloud()
    for extra in (("--dry-run",), ()):
        assert session(machine, cloud, *extra, config=M7_CONFIG) == 4
        assert message in capsys.readouterr().out
    assert family_files(root) == before and machine.started == [] and cloud.requests == []
    assert not (root / "runs" / "test-m7").exists()


def test_a_dry_run_whose_final_differs_from_a_frozen_vector_fails_with_exit_6_and_bars_the_session(m7_final,
                                                                                                    capsys):
    root, tokenizer = m7_final
    refrozen(final=changed_entry(1, success=flipped("normalized")))(root)
    assert session(FamilyMachine(tokenizer), Cloud(), "--dry-run", config=M7_CONFIG) == 6
    out = capsys.readouterr().out
    assert f"the dev check failed: final's successes differ from {FROZEN} on 1 of the 4 items" in out
    assert "nothing was opened" in out
    frozen = dev_check(root)["frozen"]
    assert dev_check(root)["passed"] is False and frozen["passed"] is False
    assert frozen["final"]["differ"] == {"default": [], "normalized": [ALL_DEV[1]]}
    assert frozen["finalg"] == {"n": 4, "outputs_differ": []}
    assert session(FamilyMachine(tokenizer), Cloud(), config=M7_CONFIG) == 4
    assert "records a dev check that did not pass" in capsys.readouterr().out


def test_a_dry_run_whose_finalg_answers_otherwise_than_frozen_fails_with_exit_6(m7_final, capsys):
    root, tokenizer = m7_final
    refrozen(finalg=changed_entry(2, output_sha256="0" * 64))(root)
    assert session(FamilyMachine(tokenizer), Cloud(), "--dry-run", config=M7_CONFIG) == 6
    assert f"finalg's outputs differ from {FROZEN} on 1 of the 4 items" in capsys.readouterr().out
    frozen = dev_check(root)["frozen"]
    assert frozen["finalg"] == {"n": 4, "outputs_differ": [ALL_DEV[2]]}
    assert frozen["final"]["differ"] == {"default": [], "normalized": []}


def test_finals_own_output_differences_are_counted_but_its_successes_decide(m7_final):
    root, tokenizer = m7_final
    refrozen(final=changed_entry(0, output_sha256="0" * 64))(root)
    assert session(FamilyMachine(tokenizer), Cloud(), "--dry-run", config=M7_CONFIG) == 0
    frozen = dev_check(root)["frozen"]
    assert frozen["passed"] is True and frozen["final"]["outputs_differ"] == 1


def test_with_b2_the_only_system_that_spends_the_m7_budget_needs_room_for_one_cap(m7_final):
    root, _ = m7_final
    config = run_test.load_config(root / M7_CONFIG)
    budget = Budget(root / "runs" / "llm" / "budget.sqlite", stage="M7", stage_cap_rmb=50.0)
    budget.charge("deepseek-v4-pro", {"prompt_tokens": 5_000_000, "completion_tokens": 0})  # 45 RMB
    budget.close()
    run_test.check_budget(config)  # 45 + 5 <= 50: final and finalg spend nothing
    budget = Budget(root / "runs" / "llm" / "budget.sqlite", stage="M7", stage_cap_rmb=50.0)
    budget.charge("deepseek-v4-pro", {"prompt_tokens": 100_000, "completion_tokens": 0})  # 0.9 RMB more
    budget.close()
    with pytest.raises(run_test.Refused, match="the M7 API budget cannot take B2's cap of 5 RMB"):
        run_test.check_budget(config)


# ---- the M7 configuration: the final system's ------------------------------------------------------------


def final_dict(**changes) -> dict:
    """A valid M7 configuration of the final system without a repository (made-up hashes); a change to DROP
    removes the key."""
    base = m7_dict()
    config = {**base, "scorers": ["normalized"],
              "systems": {"ft2": {},
                          "final": {"votes": 5, "vote_mode": "final", "rules": True, "glossary": None,
                                    "frozen": {"path": FROZEN, "sha256": "6" * 64}},
                          "finalg": {"decoding": "greedy", "end_to_end": False}, "b2": {}},
              "adapters": {"ft2": base["adapters"]["ft2"],
                           "final": {"gguf": FINAL_GGUF, "sha256": "5" * 64, "selection": None,
                                     "model": BASE_NB, "tokenizer": TOKENIZER_NB,
                                     "generation": {**GENERATION_FINAL, "reasoning_budget": 1280}}},
              "comparisons": [{"name": "final-vs-ft2", "a": "final", "b": "ft2", "test": "mcnemar",
                               "scorer": "normalized"},
                              {"name": "final-vs-finalg", "a": "final", "b": "finalg", "descriptive": True}]}
    for key in ("routing", "e2e_routing"):
        del config[key]
    for key, value in changes.items():
        if value is DROP:
            del config[key]
        else:
            config[key] = value
    return config


def with_options(system: str, **options) -> dict:
    """final_dict's systems with a system's options changed; an option set to DROP is removed."""
    systems = final_dict()["systems"]
    systems[system] = {k: v for k, v in {**systems[system], **options}.items() if v is not DROP}
    return systems


def with_final_adapter(**changes) -> dict:
    adapters = final_dict()["adapters"]
    adapters["final"] = {k: v for k, v in {**adapters["final"], **changes}.items() if v is not DROP}
    return adapters


def test_the_final_configuration_is_valid_and_its_frozen_sha256_may_wait_while_the_system_is_frozen():
    config = final_dict()
    assert run_test.check_config(config) is config
    assert run_test.check_config(m7_dict()) is not None  # a list of systems, as before
    waiting = final_dict(systems=with_options("final", frozen={"path": FROZEN, "sha256": "<freeze_final's>"}))
    with pytest.raises(run_test.ConfigError, match=re.escape("systems.final.frozen.sha256 must be 64")):
        run_test.check_config(waiting)
    assert run_test.check_config(waiting, freezing=True) is waiting  # scripts/freeze_final.py reads it so
    assert run_test.SERVER_OF["final"] == run_test.SERVER_OF["finalg"] == "final"
    assert run_test.BUILT_AS["final"] == run_test.BUILT_AS["finalg"] == "ft"
    assert not {"final", "finalg"} & set(run_test.SPENDING)
    order = run_test.RUN_ORDER
    assert order.index("ft2") < order.index("final") < order.index("finalg") < order.index("b2")
    assert set(run_test.SYSTEM_KEYS) == {"votes", "vote_mode", "rules", "glossary", "frozen", "decoding",
                                         "end_to_end"}


@pytest.mark.parametrize("changes, message", [
    ({"systems": with_options("final", votez=5)}, "systems.final: unknown key votez"),
    ({"systems": with_options("final", votes=0)},
     "systems.final.votes must be an integer of at least 1, not 0"),
    ({"systems": with_options("final", votes=True)},
     "systems.final.votes must be an integer of at least 1, not True"),
    ({"systems": with_options("final", votes=DROP)},
     "systems.final.vote_mode chooses the keys of a vote: it needs votes"),
    ({"systems": with_options("final", vote_mode="majority")},
     "systems.final.vote_mode must be reproduction or final, not 'majority'"),
    ({"systems": with_options("final", rules="yes")}, "systems.final.rules must be true or false, not 'yes'"),
    ({"systems": with_options("final", glossary={"path": "data/glossary/zh-en.json"})},
     "systems.final.glossary lacks sha256"),
    ({"systems": with_options("final", glossary={"path": "/zh-en.json", "sha256": "1" * 64})},
     "systems.final.glossary.path must be a path relative to the repository root"),
    ({"systems": with_options("final", frozen=DROP)}, "systems.final lacks frozen"),
    ({"systems": with_options("final", frozen={"path": FROZEN})}, "systems.final.frozen lacks sha256"),
    ({"systems": with_options("final", decoding="greedy")}, "systems.final: votes with decoding greedy"),
    ({"systems": with_options("finalg", decoding="sampled")},
     "systems.finalg.decoding must be greedy, not 'sampled'"),
    ({"systems": with_options("finalg", end_to_end="no")},
     "systems.finalg.end_to_end must be true or false, not 'no'"),
    ({"systems": with_options("finalg", frozen={"path": FROZEN, "sha256": "6" * 64})},
     "systems.finalg.frozen: only final is frozen"),
    # finalg is the adapter's greedy answer alone, held against final's frozen file (the WP5 review)
    ({"systems": with_options("finalg", rules=True)},
     "systems.finalg.rules: finalg answers greedily without the inference wrappers"),
    ({"systems": with_options("finalg", votes=5)},
     "systems.finalg.votes: finalg answers greedily without the inference wrappers"),
    ({"systems": with_options("finalg", glossary={"path": "data/glossary/zh-en.json", "sha256": "1" * 64})},
     "systems.finalg.glossary: finalg answers greedily without the inference wrappers"),
    ({"systems": with_options("finalg", decoding=DROP)},
     "systems.finalg lacks decoding: finalg is the adapter's generation decoded greedily"),
    ({"systems": {"ft2": {}, "finalg": {"decoding": "greedy", "end_to_end": False}, "b2": {}},
      "comparisons": [{"name": "finalg-vs-ft2", "a": "finalg", "b": "ft2", "test": "mcnemar"}]},
     "systems.finalg: finalg is held against final's frozen file, so final runs too"),
    ({"systems": with_options("b2", rules=True)},
     "systems.b2.rules: the inference wrappers and the decoding are a local system's"),
    ({"systems": {**final_dict()["systems"], "finalg": []}}, "systems.finalg must be an object"),
    ({"systems": {}}, "systems must be a non-empty list of system names"),
    ({"adapters": with_final_adapter(generation=DROP)}, "adapters.final lacks generation"),
    ({"adapters": with_final_adapter(generation={"mode": "think", "decoding": "greedy",
                                                 "reasoning_budget": 1280})},
     "systems.final.votes: a vote's samples sample, but adapters.final.generation decodes greedily"),
    ({"scorers": ["default"]}, "scorers must be a non-empty list of distinct scorers other than default"),
    ({"scorers": ["normalized", "normalized"]}, "scorers must be a non-empty list of distinct scorers"),
    ({"scorers": DROP},
     "comparisons[1].scorer normalized: the session scores its systems so only when scorers lists it"),
    ({"comparisons": [{"name": "x", "a": "final", "b": "ft2", "scorer": "strict"}]},
     "comparisons[1].scorer must be one of ['default', 'normalized'], not 'strict'"),
    ({"comparisons": [{"name": "x", "acebench": {"a": ACE_FT_M7, "b": ACE_B0_M7, "margin": 0.02},
                       "scorer": "normalized"}]}, "comparisons[1]: unknown key scorer"),
])
def test_a_malformed_final_configuration_is_refused_naming_the_key(changes, message):
    with pytest.raises(run_test.ConfigError) as refused:
        run_test.check_config(final_dict(**changes))
    assert message in str(refused.value)


# ---- the M7 example configuration --------------------------------------------------------------------------

M7_EXAMPLE = REPO / "results" / "test-config-m7.example.json"
# the keys the M7 example documents and leaves out: a glossary's, whose note the final system of M7 does not
# carry (the owner's decision of 2026-10-30: system A)
GLOSSARY_ONLY = {"systems.<system>.glossary.path", "systems.<system>.glossary.sha256"}
PLACEHOLDER = "systems.final.frozen.sha256"  # the one value not known before scripts/freeze_final.py runs


def test_the_m7_example_configuration_documents_every_key_and_runs_once_its_frozen_file_is_written():
    example = json.loads(M7_EXAMPLE.read_text(encoding="utf-8"))
    assert key_paths(example) == set(example["_doc"]) - GLOSSARY_ONLY  # every key documented, nothing else
    assert b"\r\n" not in M7_EXAMPLE.read_bytes()
    # docs/m7-本地提升计划.md WP5 and the owner's decision of 10-30: FT2, then final (FT3 at step 1635 voting
    # over K = 5 samples in the final key mode, with the runtime rules and no glossary) and finalg (the same
    # adapter greedy, single step only) on one server, then B2; the normalized scorer is the primary one
    assert (example["milestone"], example["test_sets"], example["scorers"]) == ("M7", "v2", ["normalized"])
    assert list(example["systems"]) == ["ft2", "final", "finalg", "b2"]
    assert not {"routing", "e2e_routing"} & set(example)  # R3 is no longer run
    final, finalg = example["systems"]["final"], example["systems"]["finalg"]
    assert (final["votes"], final["vote_mode"], final["rules"], final["glossary"]) == (5, "final", True, None)
    assert final["frozen"]["path"] == "results/final-m7.json"
    assert finalg == {"decoding": "greedy", "end_to_end": False}
    adapter = example["adapters"]["final"]
    assert (adapter["gguf"], adapter["model"], adapter["tokenizer"]) == (
        "runs/train/ft3/gguf/checkpoint-1635-f16.gguf", "models/Nanbeige4.1-3B-Q8_0.gguf",
        "models/Nanbeige4.1-3B")
    assert adapter["generation"] == {"mode": "think", "decoding": "sampled", "reasoning_budget": 1280,
                                     "base_seed": 0}
    selection = REPO / adapter["selection"]
    if selection.is_file():  # the selection that chose FT3's checkpoint 1635
        chosen = json.loads(selection.read_text(encoding="utf-8"))["chosen"]
        assert (chosen["gguf"], chosen["sha256"]) == (adapter["gguf"], adapter["sha256"])
    # H1 on the normalized scorer by McNemar and H3 by ACEBench (FT3, the final adapter, against its base),
    # Holm over both; the descriptive rows after them
    h1, h3, *described = example["comparisons"]
    assert (h1["a"], h1["b"], h1["test"], h1["scorer"]) == ("final", "ft2", "mcnemar", "normalized")
    assert h3["acebench"] == {"a": "runs/acebench/m7/ft", "b": "runs/acebench/m7/b0", "margin": 0.02}
    assert [(row["a"], row["b"], row["scorer"]) for row in described] == [
        ("final", "finalg", "normalized"), ("final", "b2", "normalized"), ("final", "ft2", "default")]
    assert all(row["descriptive"] is True for row in described)
    with pytest.raises(run_test.ConfigError, match=re.escape(PLACEHOLDER)):
        run_test.check_config(example)  # the frozen file's SHA-256, a placeholder until it is written
    assert run_test.check_config(example, freezing=True) is example
    filled = json.loads(M7_EXAMPLE.read_text(encoding="utf-8"))
    filled["systems"]["final"]["frozen"]["sha256"] = "0" * 64
    assert "<" not in json.dumps({k: v for k, v in filled.items() if k != "_doc"})  # no placeholder left
    assert run_test.check_config(filled) is filled


