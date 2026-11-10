"""scripts/run_acebench.py offline: a fake repository root (the base models and their model folders, the
ACEBench interpreter, the adapters chosen on the development set, the recorded patch), a fake ACEBench clone
(category.py, a few data files) with a fake git that knows its upstream tree, a fake llama-server (process and
HTTP) and a fake ACEBench that writes result, score and Excel files where the real one would, under the clone.
The model folders stand for registered families (FAMILY_FOLDERS: research_agent.family.core.folder_family is
replaced by a lookup, since a family is identified by its real chat template). Nothing starts llama-server,
ACEBench or git, and no model is loaded. ROUTE_VIA stands for an inherited variable that keeps HTTP clients
from the local server: while ACEBench's environment holds one, the fake ACEBench cannot reach the server.

Two tests read real files of the repository: the M4 and M6 results (results/acebench-m4*.json,
results/acebench-m6.json), whose recorded commands the no_think path must repeat byte for byte, and the local
ACEBench clone (runs/third_party/ACEBench, not tracked: the test skips without it), from which the upstream
files are only read (git cat-file), to check that the recorded patch applies to the pinned revision."""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import types
from fractions import Fraction
from pathlib import Path

import httpx
import pytest
from support.fake_llama import ServerProcess

import research_agent.family.core as family_core
from research_agent.eval import report
from research_agent.eval.lock import sha256_of
from research_agent.family.core import Decoding, FamilyError, Mode
from research_agent.family.registry import (
    FAMILIES,
    GRANITE_42_3B,
    LFM25_26B,
    QWEN3_HYBRID,
    QWEN3_INSTRUCT_2507,
    QWEN3_THINKING_2507,
    QWEN35_4B,
)
from research_agent.llm.local import LocalClient
from research_agent.llm.server import same_path

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_spike():
    """scripts/spikes/patch_acebench.py, the script that writes the local patches into a clean clone."""
    spec = importlib.util.spec_from_file_location("spike_patch_acebench",
                                                  REPO / "scripts" / "spikes" / "patch_acebench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ace = load_script("run_acebench")
spike = load_spike()

READY = "main: server is listening on http://127.0.0.1:8080 - starting the main loop\n"
HEAD = "56dd66cf6439b0d9655ee1b353e4cd745c6f664e"
CLONE = Path("runs/third_party/ACEBench")
BASE_NAME, ADAPTER_NAME = "local-qwen3-1.7b-base", "local-qwen3-1.7b-lora"
# the fake clone's groups (category.py), questions per category, and right answers per system (Chinese; one
# fewer in English where there is one to lose)
GROUPS = {"normal": ["normal_single_turn_single_function", "normal_multi_turn_user_adjust",
                     "normal_atom_bool"],
          "special": ["special_incomplete", "special_irrelevant"]}
CATEGORIES = [category for group in GROUPS.values() for category in group]
SMOKE = "normal_single_turn_single_function"
N = {"normal_single_turn_single_function": 4, "normal_multi_turn_user_adjust": 3, "normal_atom_bool": 2,
     "special_incomplete": 2, "special_irrelevant": 3}
CORRECT = {
    "b0": {"normal_single_turn_single_function": 3, "normal_multi_turn_user_adjust": 1, "normal_atom_bool": 2,
           "special_incomplete": 1, "special_irrelevant": 3},
    "ft": {"normal_single_turn_single_function": 1, "normal_multi_turn_user_adjust": 2, "normal_atom_bool": 0,
           "special_incomplete": 2, "special_irrelevant": 1},
    "b4": {"normal_single_turn_single_function": 2, "normal_multi_turn_user_adjust": 0, "normal_atom_bool": 1,
           "special_incomplete": 0, "special_irrelevant": 2},
}
# the fine-tuned model's answers in its trained format, per category (the rest in ACEBench's)
TOOL_CALLS = {"normal_single_turn_single_function": 3, "normal_multi_turn_user_adjust": 1,
              "normal_atom_bool": 2, "special_incomplete": 1, "special_irrelevant": 0}
TOOL_ANSWER = '<tool_call>\n{"name": "Api", "arguments": {"ids": ["a"]}}\n</tool_call>'
ACE_ANSWER = "[Api(a='1', b=[1, 2])]"
SPECIAL_ANSWER = "Missing necessary parameters (a) for the api (Api)"
END, PROCESS = 0.5, 0.75  # ACEBench's turn-level accuracies of a multi-turn category
ADAPTERS = {"ft": "runs/train/ft/gguf/checkpoint-30-f16.gguf",
            "b4": "runs/train/b4/gguf/checkpoint-20-f16.gguf"}
PATCH = b"diff --git a/category.py b/category.py\n+    \"smoke_single_function\": [...],\n"
CONNECTION = "route_via"  # the name the stand-in blocking variables share
SECRET = "sentinel-must-not-leak-0001"


# The fake model folders and the registered family each stands for (fake_folder_family); each has its GGUF
# beside it, <folder>-Q8_0.gguf, as models/ holds them.
FAMILY_FOLDERS = {"models/Qwen3-1.7B": QWEN3_HYBRID, "models/Qwen3-4B": QWEN3_HYBRID,
                  "models/Qwen3-4B-Thinking-2507": QWEN3_THINKING_2507,
                  "models/Qwen3-4B-Instruct-2507": QWEN3_INSTRUCT_2507, "models/Qwen3.5-4B": QWEN35_4B,
                  "models/granite-4.2-3b": GRANITE_42_3B, "models/LFM2.5-2.6B": LFM25_26B}
REASONING = "The user asks for the weather in Paris, so I call get_weather with the city."
PROBE_ANSWER = "[get_weather(city='Paris')]"
# llama-server b11349's defaults of the sampling settings /props reports (default_generation_settings.params,
# as results/dev-e2e-*.json recorded them), the command-line flags that set them, and the whole-number ones
SAMPLING_DEFAULTS = {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "presence_penalty": 0.0,
                     "repeat_penalty": 1.0, "repeat_last_n": 64, "seed": 4294967295}
SAMPLING_FLAGS = {"--temp": "temperature", "--top-k": "top_k", "--top-p": "top_p", "--min-p": "min_p",
                  "--presence-penalty": "presence_penalty", "--repeat-penalty": "repeat_penalty",
                  "--repeat-last-n": "repeat_last_n", "--seed": "seed"}
WHOLE = ("top_k", "repeat_last_n", "seed")


def float32(value: float) -> float:
    """value as llama-server keeps and reports a sampling setting: a 32-bit float (0.95 comes back as
    0.949999988079071)."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fake_folder_family(folder, *, name=None, families=None):
    """research_agent.family.core.folder_family for the fake model folders (FAMILY_FOLDERS): the family the
    folder stands for, which a name given must be; a folder that is none of them has no config.json."""
    key = Path(os.path.relpath(folder)).as_posix()
    if key not in FAMILY_FOLDERS:
        raise FamilyError(f"{folder} has no config.json")
    family = FAMILY_FOLDERS[key]
    if name is not None and name != family.name:
        raise FamilyError(f"--family {name!r} does not agree with the folder, whose family is {family.name}")
    return family


def correct(system: str, language: str, category: str) -> int:
    right = CORRECT[system][category]
    return right - 1 if language == "en" and right else right


def question_ids(category: str) -> list[str]:
    """ACEBench's ids: <category>_<i>, multi-turn ones <category>_<turn>_<i>."""
    if "multi_turn" in category:
        return [f"{category}_{i // 2}_{i}" for i in range(N[category])]
    return [f"{category}_{i}" for i in range(N[category])]


def last_value(cmd: list[str], option: str) -> str | None:
    at = [i for i, arg in enumerate(cmd) if arg == option]
    return cmd[at[-1] + 1] if at else None


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # git's object name


class FakeGit:
    """git for the clone checks; run stands in for subprocess.run. It knows the upstream tree of data_all as
    the fixture made it (later edits on disk are what the check must find), the clone's HEAD, its diff against
    the upstream commit and its untracked files."""

    def __init__(self, clone: Path) -> None:
        self.clone, self.head, self.diff, self.untracked = clone, HEAD, PATCH, b""
        self.tree = {path.relative_to(clone).as_posix(): blob(path.read_bytes())
                     for path in sorted((clone / "data_all").rglob("*")) if path.is_file()}
        self.calls: list[list[str]] = []

    def run(self, cmd, **kwargs) -> subprocess.CompletedProcess:
        cmd = list(cmd)
        assert cmd[:2] == ["git", "-C"] and same_path(cmd[2], self.clone), cmd
        args = cmd[3:]
        self.calls.append(args)
        if args[0] == "rev-parse":
            out = (self.head + "\n").encode()
        elif args[0] == "ls-tree":
            assert ace.UPSTREAM in args and args[-1] == "data_all", args
            out = b"".join(f"100644 blob {sha}\t{path}".encode() + b"\0" for path, sha in self.tree.items())
        elif args[0] == "diff":
            assert ace.UPSTREAM in args and "--no-ext-diff" in args, args
            out = self.diff
        else:
            assert args[:3] == ["ls-files", "--others", "--exclude-standard"], args
            out = self.untracked
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr=b"")


class FakeLlama:
    """llama-server for run_acebench: popen starts it with the adapter its --lora names, the context of its
    last -c, the slots of its last -np (4 without one), its prompt cache on unless the last of --cache-prompt
    and --no-cache-prompt is the latter, thinking on unless its last --chat-template-kwargs turns it off, the
    reasoning format of its last --reasoning-format ("auto" without one) and the sampling defaults of its last
    --temp, --top-p, --top-k, --min-p, --presence-penalty, --repeat-penalty, --repeat-last-n and --seed
    (b11349's own without them, SAMPLING_DEFAULTS); over HTTP it answers /health, /lora-adapters
    (listed(path), then beside, then phantom when given), /props (build, those slots and that context, and the
    sampling defaults as 32-bit floats in default_generation_settings.params), the probes' completions (first
    token log-probability -0.5 at lora scale 1, -0.25 at scale 0; without a "lora" field default_scale while
    an adapter is loaded, every request so with ignores_lora; the one slot reuses the prefix a request shares
    with the prompt before it while the cache is on, reported as cached_tokens) and chat completions: a server
    that does not think answers "5"; a thinking one writes reasoning (REASONING unless given) and then answer,
    the reasoning parsed out into reasoning_content under the formats "auto" and "deepseek", left in content
    in <think> tags under "none" (or with raw_think), and both under "deepseek-legacy"."""

    def __init__(self, *, default_scale: float = 1.0, ignores_lora: bool = False, listed=lambda path: path,
                 beside: str | None = None, phantom: str | None = None, builds=("b11349-fb4b2737a",),
                 exit_code: int | None = None, already_running: bool = False, raw_think: bool = False,
                 start_error: OSError | None = None, answer: str = PROBE_ANSWER,
                 reasoning: str = REASONING) -> None:
        self.default_scale, self.ignores_lora, self.listed = default_scale, ignores_lora, listed
        self.beside, self.phantom, self.builds, self.raw_think = beside, phantom, list(builds), raw_think
        self.exit_code, self.start_error, self.answer = exit_code, start_error, answer
        self.reasoning = reasoning
        self.adapter: str | None = None
        self.ctx = self.slots = 0
        self.cache = self.thinking = True
        self.reasoning_format = "auto"
        self.params: dict = {}
        self.slot = ""
        self.process: ServerProcess | None = ServerProcess([]) if already_running else None
        self.processes: list[ServerProcess] = []
        self.commands: list[list[str]] = []
        self.payloads: list[tuple[str, dict]] = []

    def popen(self, cmd, **kwargs):
        self.commands.append(list(cmd))
        if self.start_error is not None:
            raise self.start_error
        self.adapter = last_value(cmd, "--lora")
        self.ctx = int(last_value(cmd, "-c"))
        self.slots = int(last_value(cmd, "-np") or 4)
        switches = [arg for arg in cmd if arg in ("--cache-prompt", "--no-cache-prompt")]
        self.cache, self.slot = not switches or switches[-1] == "--cache-prompt", ""
        kwargs_json = last_value(cmd, "--chat-template-kwargs")
        self.thinking = kwargs_json is None or json.loads(kwargs_json).get("enable_thinking", True)
        self.reasoning_format = last_value(cmd, "--reasoning-format") or "auto"
        self.params = {}
        for flag, key in SAMPLING_FLAGS.items():
            given = last_value(cmd, flag)
            if given is None:
                value = SAMPLING_DEFAULTS[key]
            else:
                value = int(given) if key in WHOLE else float(given)
            self.params[key] = value if key in WHOLE else float32(value)
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
            props = {"total_slots": self.slots,
                     "default_generation_settings": {"n_ctx": self.ctx, "params": dict(self.params)}}
            build = self.builds[min(len(self.processes), len(self.builds)) - 1]
            if build is not None:
                props["build_info"] = build
            return httpx.Response(200, json=props)
        payload = json.loads(request.content)
        self.payloads.append((path, payload))
        if path == "/v1/chat/completions":
            raw = f"<think>\n{self.reasoning}\n</think>\n\n{self.answer}"
            if not self.thinking:
                message = {"role": "assistant", "content": "5"}
            elif self.raw_think or self.reasoning_format == "none":
                message = {"role": "assistant", "content": raw}
            elif self.reasoning_format == "deepseek-legacy":
                message = {"role": "assistant", "content": raw, "reasoning_content": self.reasoning}
            else:  # "auto", the server's default, and "deepseek": the reasoning parsed out of the content
                message = {"role": "assistant", "content": self.answer, "reasoning_content": self.reasoning}
            return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}]})
        if "lora" in payload and not self.ignores_lora:
            scale = payload["lora"][0]["scale"]
        else:
            scale = self.default_scale if self.adapter is not None else 0.0
        logprob = -0.5 if scale == 1.0 else -0.25
        token = {"token": "I", "logprob": logprob, "top_logprobs": [{"token": "I", "logprob": logprob}]}
        prompt = payload["prompt"]
        reuse = payload.get("cache_prompt", self.cache)
        reused = len(os.path.commonprefix([self.slot, prompt])) if reuse else 0
        self.slot = prompt
        return httpx.Response(200, json={
            "choices": [{"text": "I", "finish_reason": "length", "logprobs": {"content": [token]}}],
            "usage": {"completion_tokens": 1, "prompt_tokens": len(prompt),
                      "prompt_tokens_details": {"cached_tokens": min(reused, len(prompt) - 1)}}})

    def client(self, url: str) -> LocalClient:
        return LocalClient(url, transport=httpx.MockTransport(self.handler))

    def system(self) -> str:
        """The system the running server serves, told by its adapter."""
        return next((s for s, gguf in ADAPTERS.items() if self.adapter == gguf), "b0")


class FakeAce:
    """ACEBench in its own environment, run from the clone; run stands in for subprocess.run. The version
    check prints openai's version, the reach check fails while the environment holds a blocking variable or
    the server is down, the summary check reads the fake result.xlsx (JSON here) as the real one reads the
    workbook. generate.py writes one result file per category of its groups, an answer per question (none for
    the last question of the categories in short, null for the first nulls[(language, category)], then empty
    or whitespace text for the next empties[(language, category)]), the system's answers: ft's first
    TOOL_CALLS in <tool_call> form; eval_main.py writes ACEBench's score files
    (header and failed questions; normal_atom_bool one failed line more, as ACEBench writes for a question
    right on a later candidate answer; misreported: a header accuracy that is not ACEBench's; unscored: no
    file), result.xlsx and the Excel files beside the clone. Generation must find the server running,
    evaluation stopped. A step named in raises raises that OSError; exit_codes sets a step's exit code."""

    def __init__(self, llama: FakeLlama, root: Path, *, nulls: dict | None = None,
                 empties: dict | None = None, short: tuple = (), unscored: tuple = (),
                 misreported: tuple = (), exit_codes: dict | None = None, raises: dict | None = None,
                 summary_rows: int = 1) -> None:
        self.llama, self.clone = llama, root / CLONE
        self.nulls, self.short, self.unscored, self.misreported = nulls or {}, short, unscored, misreported
        self.empties = empties or {}
        self.exit_codes, self.raises, self.summary_rows = exit_codes or {}, raises or {}, summary_rows
        self.system = "b0"
        self.calls: list[tuple[str, str | None, list[str], dict]] = []

    def run(self, cmd, **kwargs) -> subprocess.CompletedProcess:
        cmd, env = list(cmd), dict(kwargs["env"])
        assert same_path(kwargs["cwd"], self.clone), kwargs.get("cwd")
        if cmd[1] == "-c":
            step = {ace.VERSION_CODE: "version", ace.REACH_CODE: "reach", ace.SUMMARY_CODE: "summary"}[cmd[2]]
        else:
            step = {"generate.py": "generate", "eval_main.py": "evaluate"}[cmd[1]]
        language = last_value(cmd, "--language") if step in ("generate", "evaluate") else None
        if step == "summary":
            language = Path(cmd[3]).parent.name.removeprefix("score_")
        self.calls.append((step, language, cmd, env))
        if step in self.raises:
            raise self.raises[step]
        if step == "version":
            return subprocess.CompletedProcess(cmd, 0, stdout="1.64.0\n", stderr="")
        if step == "reach":
            assert cmd[3] == env["LOCAL_OPENAI_BASE_URL"]
            blocked = any(CONNECTION in name.lower() for name in env)
            code = 0 if self.llama.process.returncode is None and not blocked else 1
            why = "APIConnectionError" if code else ""
            return subprocess.CompletedProcess(cmd, code, stdout="", stderr=why)
        if step == "summary":
            path, model = Path(cmd[3]), cmd[4]
            table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            rows = [table[model]] * self.summary_rows if model in table else []
            out = json.dumps({"file": path.is_file(), "rows": rows})
            return subprocess.CompletedProcess(cmd, 0, stdout=out + "\n", stderr="")
        model = last_value(cmd, "--model")
        at = cmd.index("--category") + 1
        groups = []
        while at < len(cmd) and not cmd[at].startswith("--"):
            groups.append(cmd[at])
            at += 1
        categories = [c for g in groups for c in ({"smoke_single_function": [SMOKE]}.get(g) or GROUPS[g])]
        if step == "generate":
            assert self.llama.process.returncode is None, "ACEBench generates against a running server"
            self.system = self.llama.system()
            folder = self.clone / "result_all" / f"result_{language}" / model
            folder.mkdir(parents=True, exist_ok=True)
            for category in categories:
                ids = question_ids(category)[:-1] if category in self.short else question_ids(category)
                lines = [{"id": id_, "result": self.answer(language, category, i)}
                         for i, id_ in enumerate(ids)]
                (folder / f"data_{category}_result.json").write_text(
                    "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
        else:
            assert self.llama.process.terminated, "the server is stopped before ACEBench evaluates"
            scores = self.clone / "score_all" / f"score_{language}"
            excel = self.clone.parent / "result_excel" / language / model
            for category in categories:
                excel.mkdir(parents=True, exist_ok=True)
                (excel / f"data_{category}.xlsx").write_bytes(b"PK xlsx")
                if category in self.unscored:
                    continue
                self.score(scores / model, language, category)
            table_path = scores / "result.xlsx"
            table = json.loads(table_path.read_text(encoding="utf-8")) if table_path.is_file() else {}
            table[model] = {"Model": model, "normal_summary": 0.5, "special_summary": 0.25, "Summary": 0.356}
            table_path.write_text(json.dumps(table), encoding="utf-8")
        kwargs["stdout"].write(f"ACEBench {step} {model} \U0001f50d\n".encode())  # ACEBench prints emoji
        return subprocess.CompletedProcess(cmd, self.exit_codes.get(step, 0))

    def answer(self, language: str, category: str, i: int) -> str | None:
        nulls = self.nulls.get((language, category), 0)
        if i < nulls:
            return None
        if i < nulls + self.empties.get((language, category), 0):
            return "" if i % 2 == 0 else " \n\t"  # no text: empty, or whitespace only
        if self.system == "ft" and i < TOOL_CALLS[category]:
            return TOOL_ANSWER
        return SPECIAL_ANSWER if category.startswith("special") else ACE_ANSWER

    def score(self, folder: Path, language: str, category: str) -> None:
        n, right = N[category], correct(self.system, language, category)
        header = {"accuracy": round(right / n, 3), "correct_count": right, "total_count": n}
        failed = n - right + (category == "normal_atom_bool")
        if category.startswith("special"):
            header["accuracy"] = right / n
        if "multi_turn" in category:
            header.update(accuracy=END, process_accuracy=PROCESS)
        if category in self.misreported:
            header["accuracy"] = 0.999
        lines = [header, *({"id": f"{category}_{i}", "valid": False} for i in range(failed))]
        folder.mkdir(parents=True, exist_ok=True)
        # no final newline, as ACEBench writes them
        text = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
        (folder / f"data_{category}_score.json").write_text(text, encoding="utf-8")


def failing(*args, **kwargs):
    raise AssertionError("nothing may be started")


def select(root: Path, system: str) -> None:
    gguf = root / ADAPTERS[system]
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(f"GGUF {system}".encode())
    selection = {"run": f"runs/train/{system}", "system": system,
                 "chosen": {"step": 30, "gguf": ADAPTERS[system], "sha256": sha256_of(gguf)}}
    out = root / "results" / f"checkpoint-selection-{system}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")


def make_clone(clone: Path) -> None:
    """category.py with the fake groups (and the patch's smoke group, and a group not run), and the data of
    every category in both languages, with an answer file each."""
    groups = {"smoke_single_function": [SMOKE], **GROUPS, "agent": ["agent_multi_step"]}
    clone.mkdir(parents=True)
    (clone / "category.py").write_text(f"ACE_DATA_CATEGORY = {json.dumps(groups, indent=4)}\n",
                                       encoding="utf-8")
    for script in ("generate.py", "eval_main.py"):
        (clone / script).write_text("raise SystemExit('never run in the tests')\n", encoding="utf-8")
    for language in ace.LANGUAGES:
        folder = clone / "data_all" / f"data_{language}"
        (folder / "possible_answer").mkdir(parents=True)
        for category in CATEGORIES:
            lines = [{"id": id_, "question": f"问题 {id_}", "function": []} for id_ in question_ids(category)]
            text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
            (folder / f"data_{category}.json").write_text(text, encoding="utf-8")
            (folder / "possible_answer" / f"data_{category}.json").write_text(text, encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A repository root as the current folder: the base models with their model folders (FAMILY_FOLDERS), a
    llama-server binary, the ACEBench interpreter, the adapters chosen for ft and b4, the recorded patch and
    the clone (all fake)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ace, "git_commit", lambda repo: "c0ffee")
    monkeypatch.setattr(ace, "git_dirty", lambda repo: False)
    monkeypatch.setattr(ace, "folder_family", fake_folder_family)
    for name in [name for name in os.environ if CONNECTION in name.lower()]:
        monkeypatch.delenv(name)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "Qwen3-1.7B-Q8_0.gguf").write_bytes(b"GGUF base")
    for folder in FAMILY_FOLDERS:
        (tmp_path / folder).mkdir()
        gguf = tmp_path / f"{folder}-Q8_0.gguf"
        if not gguf.exists():
            gguf.write_bytes(f"GGUF {folder}".encode())
    (tmp_path / "llama-server.exe").write_bytes(b"")
    python = tmp_path / ace.ACE_PYTHON
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    for system in ADAPTERS:
        select(tmp_path, system)
    (tmp_path / "third_party_patches").mkdir()
    (tmp_path / "third_party_patches" / "acebench.diff").write_bytes(PATCH)
    make_clone(tmp_path / CLONE)
    return tmp_path


@pytest.fixture
def git(root):
    return FakeGit(root / CLONE)


BINARY = ["--llama-server", "llama-server.exe"]
ARGV = [*BINARY, "--mode", "no_think"]  # the M4 run: Qwen3-1.7B without thinking (the default folder)
OUT = Path("results/acebench-m4.json")


def run_ace(llama: FakeLlama, fake: FakeAce, git: FakeGit, *extra: str) -> int:
    return ace.main([*ARGV, *extra], popen=llama.popen, run=fake.run, client=llama.client, git=git.run)


def dry(git: FakeGit, *extra: str) -> int:
    return ace.main([*ARGV, *extra, "--dry-run"], popen=failing, run=failing, client=failing, git=git.run)


def steps(fake: FakeAce) -> list[str]:
    return [step if language is None else f"{step} {language}" for step, language, _, _ in fake.calls]


def mean(system: str, languages=ace.LANGUAGES, categories=tuple(CATEGORIES)) -> float:
    rates = [Fraction(correct(system, lang, c), N[c]) for lang in languages for c in categories]
    return float(sum(rates) / len(rates))


def test_the_model_names_are_the_ones_the_patch_registers():
    patch = (REPO / "third_party_patches" / "acebench.diff").read_text(encoding="utf-8")
    assert {ace.MODELS["b0"], ace.MODELS["ft"]} == {BASE_NAME, ADAPTER_NAME}
    assert f'+        "{BASE_NAME}",' in patch and f'+        "{ADAPTER_NAME}",' in patch
    assert ace.MODELS["b4"] == ace.MODELS["ft"]  # an adapter server either way
    assert f'"{ace.SMOKE_GROUP}"' in patch and "LOCAL_OPENAI_BASE_URL" in patch


def test_acebench_format_is_a_bracketed_list_of_calls_as_acebench_parses_it():
    for text in ("[Api(a='1')]", " [A(x=[1, 2]), b.C()] ", "Sure: [Api(q='是')] done", "[]"):
        assert ace.acebench_format(text), text
    for text in (TOOL_ANSWER, "Api(a='1')", "[Api(a='1')", "[1, 2]", "[Api(a=)]", SPECIAL_ANSWER, None, ""):
        assert not ace.acebench_format(text), text
    found = {"c": [{"result": TOOL_ANSWER}, {"result": ACE_ANSWER}, {"result": f"{TOOL_ANSWER} {ACE_ANSWER}"},
                   {"result": "no call"}, {"result": None}, {"result": ""}, {"result": " \n\t"}]}
    # the third answer holds both, and ACEBench's parser, taking the first outermost brackets, finds no call;
    # the last two are text without a character to read (empty), the null before them no text at all
    assert ace.format_counts(found) == {"c": {"n": 7, "tool_call": 2, "acebench_format": 1, "empty": 2}}


def test_the_aggregate_rule_and_the_mean_over_both_languages():
    assert "unweighted mean" in ace.AGGREGATE_RULE and "Summary" in ace.AGGREGATE_RULE
    rows = {"zh": {"a": {"n": 100, "correct": 50}, "b": {"n": 2, "correct": 2}},
            "en": {"a": {"n": 100, "correct": 25}, "b": {"n": 2, "correct": 0}}}
    assert ace.overall(rows) == {"categories": 4, "n": 204, "accuracy": float(Fraction(175, 400))}


def test_a_run_records_each_system_language_and_category_and_copies_the_outputs(root, git, monkeypatch,
                                                                                capsys):
    monkeypatch.setenv("ROUTE_VIA", SECRET)
    monkeypatch.setenv("route_via_tls", SECRET)
    before = dict(os.environ)
    llama = FakeLlama()
    fake = FakeAce(llama, root)
    assert run_ace(llama, fake, git, "--drop-env", CONNECTION) == 0
    assert dict(os.environ) == before  # nothing was written to this process's environment

    raw_bytes = (root / OUT).read_bytes()
    assert b"\r" not in raw_bytes and raw_bytes.endswith(b"}\n") and b'\n  "systems": {' in raw_bytes
    text = raw_bytes.decode("utf-8")
    # repository-relative POSIX paths only (the one backslash JSON needs: the quotes of the template kwargs)
    assert str(root) not in text and "\\" not in text.replace('\\"', "")
    assert CONNECTION not in text.lower() and SECRET not in text  # nothing of the dropped variables
    results = json.loads(text)
    assert list(results["systems"]) == ["b0", "ft"] and results["languages"] == ["zh", "en"]
    assert results["groups"] == {"normal": GROUPS["normal"], "special": GROUPS["special"]}
    assert results["generation"] == {"temperature": 0, "top_p": 1, "max_tokens": 1200, "num_threads": 1}
    assert (results["git_commit"], results["git_dirty"]) == ("c0ffee", False)
    assert results["acebench"]["commit"] == HEAD and results["acebench"]["openai"] == "1.64.0"
    assert results["acebench"]["patch"] == {"file": "third_party_patches/acebench.diff",
                                            "sha256": hashlib.sha256(PATCH).hexdigest()}
    assert results["acebench"]["data"]["files"] == len(git.tree) == 2 * 2 * len(CATEGORIES)
    assert results["acebench"]["data"]["identical_to"] == HEAD
    assert results["environment"] == {"LOCAL_OPENAI_BASE_URL": "http://127.0.0.1:8080/v1",
                                      "LOCAL_OPENAI_API_KEY": "EMPTY", "PYTHONUTF8": "1", "TQDM_DISABLE": "1"}
    assert results["drop_env"] == {"patterns": 1, "variables_dropped": 2}
    assert "unweighted mean" in results["aggregate_rule"] and "<tool_call>" in results["format_rule"]

    base = ["llama-server", "-m", "models/Qwen3-1.7B-Q8_0.gguf", "-ngl", "99", "-c", "8192", "--jinja",
            "--host", "127.0.0.1", "--port", "8080", "--cache-ram", "0"]
    fixed = ["-np", "1", "--no-cache-prompt", "--chat-template-kwargs", '{"enable_thinking":false}']
    b0, ft = results["systems"]["b0"], results["systems"]["ft"]
    assert b0["server_command"] == [*base, *fixed] and b0["adapter"] is None
    assert ft["server_command"] == [*base, "--lora", ADAPTERS["ft"], "--lora-init-without-apply", *fixed]
    assert ft["adapter"] == {"gguf": ADAPTERS["ft"], "sha256": sha256_of(ADAPTERS["ft"]),
                             "selection": "results/checkpoint-selection-ft.json"}
    assert (b0["acebench_model"], ft["acebench_model"]) == (BASE_NAME, ADAPTER_NAME)
    assert b0["server_build"] == ft["server_build"] == "b11349-fb4b2737a"
    assert (b0["raw"], ft["raw"]) == ("runs/acebench/m4/b0", "runs/acebench/m4/ft")
    for system, row in (("b0", b0), ("ft", ft)):
        for language in ("zh", "en"):
            block = row["languages"][language]
            expected = {}
            for category in CATEGORIES:
                right = correct(system, language, category)
                expected[category] = {"n": N[category], "correct": right, "accuracy": right / N[category]}
                if "multi_turn" in category:
                    expected[category].update(acebench_accuracy=END, acebench_process_accuracy=PROCESS)
            assert block["categories"] == expected
            assert block["aggregate"] == {"categories": 5, "n": sum(N.values()),
                                          "accuracy": mean(system, (language,))}
            assert block["groups"]["special"] == {
                "categories": 2, "n": 5, "accuracy": mean(system, (language,), tuple(GROUPS["special"]))}
            assert block["inference_errors"] == dict.fromkeys(CATEGORIES, 0)
            assert block["acebench_summary"] == {"Model": row["acebench_model"], "normal_summary": 0.5,
                                                 "special_summary": 0.25, "Summary": 0.356}
            tool = TOOL_CALLS if system == "ft" else dict.fromkeys(CATEGORIES, 0)
            assert block["format"] == {
                c: {"n": N[c], "tool_call": tool[c],
                    "acebench_format": 0 if c.startswith("special") else N[c] - tool[c], "empty": 0}
                for c in CATEGORIES}
            model = row["acebench_model"]
            assert block["generate"] == [
                ace.ACE_PYTHON.as_posix(), "generate.py", "--model", model, "--category", "normal", "special",
                "--language", language, "--temperature", "0", "--top-p", "1", "--max-tokens", "1200",
                "--num-threads", "1"]
            assert block["evaluate"] == [ace.ACE_PYTHON.as_posix(), "eval_main.py", "--language", language,
                                         "--model", model, "--category", "normal", "special"]
            copy = root / "runs" / "acebench" / "m4" / system / language
            for category in CATEGORIES:
                assert (copy / "result" / f"data_{category}_result.json").read_bytes() == (
                    root / CLONE / "result_all" / f"result_{language}" / model /
                    f"data_{category}_result.json").read_bytes()
                assert (copy / "score" / f"data_{category}_score.json").is_file()
                assert (copy / "excel" / f"data_{category}.xlsx").is_file()
            assert (copy / "acebench-summary.xlsx").is_file()
            assert (copy / "generate.log").is_file() and (copy / "evaluate.log").is_file()
        assert row["aggregate"] == {"categories": 10, "n": 2 * sum(N.values()), "accuracy": mean(system)}
        assert (root / "runs" / "acebench" / "m4" / system / "server.log").is_file()
    # the clone keeps ACEBench's own folders: copied, never moved or deleted
    assert (root / CLONE / "result_all" / "result_zh" / BASE_NAME).is_dir()

    # one server per system, b0 first, each stopped; each probed before ACEBench ran: the adapter probe on the
    # adapter server (scale 1, scale 0, no lora field), the cache probe and the thinking check on both
    assert len(llama.processes) == 2 and all(p.terminated for p in llama.processes)
    started = [["llama-server", *cmd[1:]] for cmd in llama.commands]
    assert started == [b0["server_command"], ft["server_command"]]
    assert [path for path, _ in llama.payloads] == (["/v1/completions"] * 2 + ["/v1/chat/completions"]
                                                    + ["/v1/completions"] * 5 + ["/v1/chat/completions"])
    assert [p.get("lora") for _, p in llama.payloads] == [None, None, None, [{"id": 0, "scale": 1.0}],
                                                          [{"id": 0, "scale": 0.0}], None, None, None, None]
    # the thinking check, as M4 and M6 sent it, field for field: no lora, cache_prompt or chat_template_kwargs
    m4_check = {"messages": [{"role": "system", "content": "You are an AI assistant."},
                             {"role": "user", "content": "What is 2 + 3? Answer with the number only."}],
                "temperature": 0, "max_tokens": 32}
    assert llama.payloads[2][1] == m4_check and llama.payloads[-1][1] == m4_check
    assert results["mode"] == "no_think" and results["budget"] == {"reasoning": None, "reply": 1200,
                                                                  "max_tokens": 1200, "context": 8192}
    assert results["family"] == {"name": "qwen3-hybrid", "thinking": "hybrid", "folder": "models/Qwen3-1.7B"}
    assert results["decoding"]["greedy"] is True and results["decoding"]["seed"] is None
    assert "probe" not in b0 and "probe" not in ft
    assert report.acebench_numbers(results, "m4")["acebench.m4.b0.aggregate"] == f"{100 * mean('b0'):.2f}%"
    # ACEBench in its own environment, from the clone: the version, then per system the reach check and both
    # languages' generation against the running server, then each language's evaluation and summary
    per_system = ["reach", "generate zh", "generate en", "evaluate zh", "summary", "evaluate en", "summary"]
    ran = [s.split(" ")[0] if s.startswith("summary") else s for s in steps(fake)]
    assert ran == ["version", *per_system * 2]
    for step, _, cmd, env in fake.calls:
        assert same_path(cmd[0], root / ace.ACE_PYTHON), step
        assert not [name for name in env if CONNECTION in name.lower()], step
        assert (env["PYTHONUTF8"], env["TQDM_DISABLE"], env["LOCAL_OPENAI_API_KEY"]) == ("1", "1", "EMPTY")
        assert env["LOCAL_OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert "\U0001f50d" in (root / "runs/acebench/m4/b0/zh/generate.log").read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert SECRET not in out and f"{100 * mean('ft'):.2f}" in out and OUT.as_posix() in out
    assert "empty answers" not in out  # none to report


def test_the_results_file_is_not_overwritten_without_force(root, git, capsys):
    (root / OUT).write_text("{}\n", encoding="utf-8")
    assert dry(git) == 2
    assert "--force" in capsys.readouterr().out
    assert (root / OUT).read_text(encoding="utf-8") == "{}\n"
    llama = FakeLlama()
    assert run_ace(llama, FakeAce(llama, root), git, "--force") == 0
    assert json.loads((root / OUT).read_text(encoding="utf-8"))["systems"]


@pytest.mark.parametrize("folder", [
    "result_all/result_zh/local-qwen3-1.7b-base",
    "score_all/score_en/local-qwen3-1.7b-lora",
    "../result_excel/zh/local-qwen3-1.7b-lora",
])
def test_existing_acebench_folders_of_a_model_and_language_are_refused_and_kept(root, git, capsys, folder):
    """ACEBench keeps the answers it finds and generates only the missing ones, so a run would mix with the
    one before it: its folders are refused, by name, and never deleted."""
    earlier = root / CLONE / folder
    earlier.mkdir(parents=True)
    (earlier / "data_x.json").write_text("{}\n", encoding="utf-8")
    assert ace.main(ARGV, popen=failing, run=failing, client=failing, git=git.run) == 2
    out = capsys.readouterr().out
    assert Path(os.path.normpath(CLONE / folder)).as_posix() in out and "move" in out
    assert (earlier / "data_x.json").is_file() and not (root / OUT).exists()


def test_ft_and_b4_share_acebenchs_model_name_and_cannot_run_together(root, git, capsys):
    assert dry(git, "--systems", "b0,ft,b4") == 2
    assert ADAPTER_NAME in capsys.readouterr().out
    llama = FakeLlama()
    fake = FakeAce(llama, root)
    assert run_ace(llama, fake, git, "--systems", "b4,b0") == 0
    results = json.loads((root / OUT).read_text(encoding="utf-8"))
    assert list(results["systems"]) == ["b0", "b4"]  # in the order they run
    b4 = results["systems"]["b4"]
    assert b4["adapter"]["gguf"] == ADAPTERS["b4"] and b4["raw"] == "runs/acebench/m4/b4"
    assert b4["acebench_model"] == ADAPTER_NAME and b4["aggregate"]["accuracy"] == mean("b4")
    assert llama.commands[1][llama.commands[1].index("--lora") + 1] == ADAPTERS["b4"]


def test_dry_run_prints_the_plan_without_the_patterns_and_starts_and_writes_nothing(root, git, monkeypatch,
                                                                                    capsys):
    monkeypatch.setenv("ROUTE_VIA", SECRET)
    monkeypatch.setenv("route_via_tls", SECRET)
    secret_pattern = "route_via|zz_unlikely_name"
    assert dry(git, "--drop-env", secret_pattern) == 0
    out = capsys.readouterr().out
    assert secret_pattern not in out and SECRET not in out and "zz_unlikely_name" not in out
    assert "2 inherited variables dropped" in out
    assert f"--lora {ADAPTERS['ft']} --lora-init-without-apply" in out and sha256_of(ADAPTERS["ft"]) in out
    assert "-c 8192" in out and "-np 1 --no-cache-prompt" in out and "enable_thinking" in out
    assert "generate.py --model local-qwen3-1.7b-base --category normal special --language zh" in out
    assert "eval_main.py --language en --model local-qwen3-1.7b-lora --category normal special" in out
    assert "LOCAL_OPENAI_BASE_URL=http://127.0.0.1:8080/v1" in out and HEAD in out
    assert not (root / "runs" / "acebench").exists() and not (root / OUT).exists()
    assert not (root / CLONE / "result_all").exists()


def test_a_drop_env_pattern_that_is_not_a_regular_expression_is_refused_without_echoing_it(root, git, capsys):
    assert dry(git, "--drop-env", "route_(via") == 2
    out = capsys.readouterr().out
    assert "--drop-env" in out and "route_(via" not in out


@pytest.mark.parametrize("change, said", [
    ("head", "56dd66c"),  # the clone at another commit
    ("byte", "data_all/data_en/data_special_irrelevant.json"),  # a data file not as upstream has it
    ("crlf", "data_all/data_zh/data_normal_atom_bool.json"),  # the same text, other bytes
    ("extra", "data_all/data_zh/data_extra.json"),  # a data file upstream does not have
    ("missing", "data_all/data_zh/possible_answer/data_special_incomplete.json"),
    ("code", "third_party_patches/acebench.diff"),  # code changed beyond the recorded patches
    ("untracked", "planted.py"),  # code beside ACEBench's own
    ("smoke", "smoke_single_function"),  # the patch's smoke group absent
])
def test_a_clone_that_is_not_upstream_plus_the_recorded_patches_is_refused(root, git, capsys, change, said):
    clone = root / CLONE
    if change == "head":
        git.head = "0" * 40
    elif change == "byte":
        path = clone / "data_all/data_en/data_special_irrelevant.json"
        path.write_bytes(path.read_bytes().replace("问题".encode(), "问".encode()))
    elif change == "crlf":
        path = clone / "data_all/data_zh/data_normal_atom_bool.json"
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    elif change == "extra":
        (clone / "data_all/data_zh/data_extra.json").write_text("{}\n", encoding="utf-8")
    elif change == "missing":
        (clone / "data_all/data_zh/possible_answer/data_special_incomplete.json").unlink()
    elif change == "code":
        git.diff = PATCH + b"+print('changed')\n"
    elif change == "untracked":
        git.untracked = b"planted.py\0"
    else:
        (clone / "category.py").write_text(f"ACE_DATA_CATEGORY = {json.dumps(GROUPS)}\n", encoding="utf-8")
    assert dry(git, *(("--smoke",) if change == "smoke" else ())) == 2
    out = capsys.readouterr().out
    assert "FAIL: " in out and said in out


@pytest.mark.parametrize("raw_think, kwargs", [(False, '{"enable_thinking":true}'), (True, "{}")])
def test_a_server_that_thinks_stops_the_run_with_exit_3(root, git, capsys, raw_think, kwargs):
    """ACEBench does not send enable_thinking, so the server's template default must turn thinking off: an
    answer with a thinking block (parsed out into reasoning_content, or raw in the text) is refused before
    ACEBench runs."""
    llama = FakeLlama(raw_think=raw_think)
    fake = FakeAce(llama, root)
    assert run_ace(llama, fake, git, "--", "--chat-template-kwargs", kwargs) == 3
    out = capsys.readouterr().out
    assert "FAIL: b0: " in out and "thinking" in out
    assert steps(fake) == ["version"] and all(p.terminated for p in llama.processes)
    assert not (root / OUT).exists()


@pytest.mark.parametrize("server, extra, said", [
    ({"listed": lambda path: ADAPTERS["b4"]}, (), "FAIL: ft: the server lists the adapters"),
    ({"beside": ADAPTERS["b4"]}, (), "FAIL: ft: the server lists the adapters"),
    ({"phantom": ADAPTERS["b4"]}, (), "FAIL: b0: the server lists the adapters"),
    ({}, ("--", "-np", "4"), "FAIL: b0: the server reports 4 slots of 8192 tokens"),
    ({}, ("--", "-c", "4096"), "FAIL: b0: the server reports 1 slots of 4096 tokens"),
    ({"ignores_lora": True}, (), "FAIL: ft: a completion without a lora field"),
    ({"default_scale": 0.0}, (), "FAIL: ft: a completion without a lora field"),
    ({}, ("--", "--cache-prompt"), "prompt tokens taken from the cache"),
    ({"builds": (None,)}, (), "FAIL: b0: the server reports no build"),
    ({"builds": ("b11349-fb4b2737a", "b12000-0000000")}, (), "FAIL: ft: the server reports build b12000"),
])
def test_a_server_that_fails_a_check_stops_the_run_with_exit_3(root, git, capsys, server, extra, said):
    llama = FakeLlama(**server)
    fake = FakeAce(llama, root)
    assert run_ace(llama, fake, git, *extra) == 3
    assert said in capsys.readouterr().out
    assert all(p.terminated for p in llama.processes) and not (root / OUT).exists()
    assert steps(fake)[-1].split(" ")[0] in ("version", "summary")  # nothing ran against the failed server


def test_the_reach_check_loads_dotenv_and_asks_through_the_openai_sdk_as_acebench_does(monkeypatch):
    """ACEBench's client: load_dotenv() (which may bring variables of a .env file above the clone), then the
    OpenAI SDK against the server's /v1; the check does both, run here over fake dotenv and openai modules."""
    asked: list = []

    class OpenAI:
        def __init__(self, **kwargs) -> None:
            asked.append(("OpenAI", kwargs))
            self.models = types.SimpleNamespace(list=lambda: asked.append("models.list"))

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: asked.append("load_dotenv")
    openai = types.ModuleType("openai")
    openai.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setattr(sys, "argv", ["-c", "http://127.0.0.1:8080/v1"])
    exec(compile(ace.REACH_CODE, "<reach check>", "exec"), {})  # noqa: S102  the check's own code
    assert asked == ["load_dotenv", ("OpenAI", {"base_url": "http://127.0.0.1:8080/v1", "api_key": "EMPTY",
                                                "timeout": 30, "max_retries": 0}), "models.list"]


def test_the_summary_check_reads_the_models_row_of_acebenchs_workbook(tmp_path, monkeypatch, capsys):
    rows = [("Model", "normal_summary", "Summary"), ("other", 0.1, 0.2), (BASE_NAME, 0.5, 0.356)]
    sheet = types.SimpleNamespace(iter_rows=lambda values_only: iter(rows))
    openpyxl = types.ModuleType("openpyxl")
    openpyxl.load_workbook = lambda path, read_only: types.SimpleNamespace(active=sheet)
    monkeypatch.setitem(sys.modules, "openpyxl", openpyxl)
    book = tmp_path / "result.xlsx"
    for exists in (False, True):
        if exists:
            book.write_bytes(b"PK")
        monkeypatch.setattr(sys, "argv", ["-c", str(book), BASE_NAME])
        exec(compile(ace.SUMMARY_CODE, "<summary check>", "exec"), {})  # noqa: S102
        found = json.loads(capsys.readouterr().out)
        row = {"Model": BASE_NAME, "normal_summary": 0.5, "Summary": 0.356}
        assert found == ({"file": True, "rows": [row]} if exists else {"file": False, "rows": []})


def test_without_dropping_the_blocking_variables_acebench_cannot_reach_the_server_and_the_run_exits_2(
        root, git, monkeypatch, capsys):
    monkeypatch.setenv("ROUTE_VIA", SECRET)
    llama = FakeLlama()
    fake = FakeAce(llama, root)
    assert run_ace(llama, fake, git) == 2
    assert steps(fake) == ["version", "reach"] and "--drop-env" in capsys.readouterr().out
    assert all(p.terminated for p in llama.processes) and not (root / OUT).exists()


def test_inference_errors_stop_the_run_before_evaluation_with_exit_4(root, git, capsys):
    llama = FakeLlama()
    fake = FakeAce(llama, root, nulls={("en", "special_incomplete"): 2})
    assert run_ace(llama, fake, git) == 4
    assert steps(fake) == ["version", "reach", "generate zh", "generate en"]
    out = capsys.readouterr().out
    # per language and category, the ones without an error left out (zh has none)
    assert ("FAIL: b0: ACEBench recorded 2 inference errors (en: special_incomplete 2): answers without "
            "content, which would count as wrong answers") in out
    assert all(p.terminated for p in llama.processes) and not (root / OUT).exists()
    # the answers are copied for inspection all the same
    assert (root / "runs/acebench/m4/b0/en/result/data_special_incomplete_result.json").is_file()


def test_missing_answers_or_scores_and_inconsistent_headers_exit_1(root, git, capsys):
    cases = [({"short": ("normal_atom_bool",)}, "data_normal_atom_bool_result.json"),
             ({"unscored": ("special_irrelevant",)}, "data_special_irrelevant_score.json"),
             ({"misreported": ("normal_single_turn_single_function",)},
              "data_normal_single_turn_single_function_score.json"),
             ({"misreported": ("special_incomplete",)}, "data_special_incomplete_score.json"),
             ({"summary_rows": 2}, "result.xlsx")]
    for options, said in cases:
        for folder in ("result_all", "score_all"):
            for path in (root / CLONE / folder).glob("*/*"):
                if path.is_dir():
                    for item in path.iterdir():
                        item.unlink()
                    path.rmdir()
        excel = root / CLONE.parent / "result_excel"
        for path in sorted(excel.rglob("*"), reverse=True) if excel.exists() else []:
            path.unlink() if path.is_file() else path.rmdir()
        for path in sorted((root / "runs" / "acebench").rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        llama = FakeLlama()
        assert run_ace(llama, FakeAce(llama, root, **options), git) == 1, options
        out = capsys.readouterr().out
        assert said in out, (options, out)
        assert not (root / OUT).exists()


@pytest.mark.parametrize("step", ["generate", "evaluate"])
def test_an_acebench_step_that_fails_stops_the_run_with_exit_4(root, git, capsys, step):
    llama = FakeLlama()
    fake = FakeAce(llama, root, exit_codes={step: 1})
    assert run_ace(llama, fake, git) == 4
    assert steps(fake)[-1] == f"{step} zh" and "runs/acebench/m4/b0/zh" in capsys.readouterr().out
    assert all(p.terminated for p in llama.processes) and not (root / OUT).exists()


@pytest.mark.parametrize("step, code", [("version", 2), ("reach", 2), ("generate", 4), ("evaluate", 4)])
def test_an_acebench_process_that_cannot_be_started_stops_the_run_with_a_fail_line(root, git, capsys, step,
                                                                                    code):
    llama = FakeLlama()
    fake = FakeAce(llama, root, raises={step: OSError(errno.ENOEXEC, "Exec format error")})
    assert run_ace(llama, fake, git) == code
    out = capsys.readouterr().out
    assert "FAIL: " in out and "Exec format error" in out and steps(fake)[-1].split(" ")[0] == step
    assert all(p.terminated for p in llama.processes) and not (root / OUT).exists()


def test_a_busy_port_a_dead_server_and_a_binary_that_cannot_run(root, git, capsys):
    busy = FakeLlama(already_running=True)
    assert run_ace(busy, FakeAce(busy, root), git) == 2 and busy.processes == []
    assert "already answers" in capsys.readouterr().out
    dead = FakeLlama(exit_code=1)
    assert run_ace(dead, FakeAce(dead, root), git) == 3 and len(dead.processes) == 1
    assert "exited with 1" in capsys.readouterr().out
    for path in sorted((root / "runs" / "acebench").rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    broken = FakeLlama(start_error=PermissionError(errno.EACCES, "Permission denied"))
    assert run_ace(broken, FakeAce(broken, root), git) == 2
    assert "llama-server could not be started" in capsys.readouterr().out
    assert not (root / "runs" / "acebench" / "m4" / "b0").exists() and not (root / OUT).exists()


def test_smoke_runs_the_smoke_group_only_into_files_of_its_own(root, git):
    llama = FakeLlama()
    fake = FakeAce(llama, root)
    assert run_ace(llama, fake, git, "--smoke") == 0
    assert not (root / OUT).exists()
    results = json.loads((root / ace.SMOKE_OUT).read_text(encoding="utf-8"))
    assert results["smoke"] is True and results["groups"] == {"smoke_single_function": [SMOKE]}
    generate = next(cmd for step, _, cmd, _ in fake.calls if step == "generate")
    groups = generate[generate.index("--category") + 1:generate.index("--language")]
    assert groups == ["smoke_single_function"]
    for system in ("b0", "ft"):
        block = results["systems"][system]["languages"]["zh"]
        assert list(block["categories"]) == [SMOKE]
        assert results["systems"][system]["raw"] == f"{ace.SMOKE_RAW.as_posix()}/{system}"


def test_languages_can_be_chosen(root, git, capsys):
    assert dry(git, "--languages", "fr") == 2
    capsys.readouterr()
    llama = FakeLlama()
    fake = FakeAce(llama, root)
    assert run_ace(llama, fake, git, "--languages", "en") == 0
    results = json.loads((root / OUT).read_text(encoding="utf-8"))
    assert results["languages"] == ["en"] and list(results["systems"]["b0"]["languages"]) == ["en"]
    assert results["systems"]["b0"]["aggregate"]["accuracy"] == mean("b0", ("en",))


def test_inputs_that_are_missing_or_outside_the_current_folder_are_refused(root, git, capsys, monkeypatch):
    assert dry(git, "--out", "../acebench-m4.json") == 2
    assert "outside" in capsys.readouterr().out
    (root / ADAPTERS["ft"]).write_bytes(b"GGUF ft, overwritten after the selection")
    assert dry(git) == 2
    assert "checkpoint-selection-ft.json" in capsys.readouterr().out
    assert dry(git, "--ft-adapter", ADAPTERS["ft"], "--b4-adapter", ADAPTERS["b4"]) == 2  # b4 not run
    capsys.readouterr()
    (root / ace.ACE_PYTHON).unlink()
    assert dry(git, "--ft-adapter", ADAPTERS["ft"]) == 2
    assert "--acebench-python" in capsys.readouterr().out
    monkeypatch.delenv("LLAMA_SERVER", raising=False)
    assert ace.main(["--dry-run"], popen=failing, run=failing, client=failing, git=git.run) == 2
    assert "LLAMA_SERVER" in capsys.readouterr().out


def test_a_raw_folder_holding_an_earlier_run_is_refused(root, git, capsys):
    earlier = root / "runs" / "acebench" / "m4" / "ft" / "zh" / "generate.log"
    earlier.parent.mkdir(parents=True)
    earlier.write_text("x\n", encoding="utf-8")
    assert dry(git) == 2
    assert "runs/acebench/m4/ft" in capsys.readouterr().out


# ---- M7 WP-L: ACEBench in think mode (docs/m7-计划.md WP-L and G4) ----------------------------------------

THINK_FOLDER, THINK_GGUF = "models/Qwen3-4B-Thinking-2507", "models/Qwen3-4B-Thinking-2507-Q8_0.gguf"
THINK_ARGV = [*BINARY, "--tokenizer", THINK_FOLDER, "--model", THINK_GGUF]
THINK_OUT = Path("results/acebench-m7.json")
THINK_NAMES = ("local-qwen3-thinking-2507-base", "local-qwen3-thinking-2507-lora")
ONE_SLOT = ["-np", "1", "--no-cache-prompt"]
THINKING = ["--reasoning-format", "deepseek", "--reasoning-budget"]  # followed by the budget
QWEN3_SAMPLING = ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0", "--presence-penalty",
                  "0.0"]
M4_SWITCH = ["--chat-template-kwargs", '{"enable_thinking":false}']
CLONE_ON_DISK = REPO / "runs" / "third_party" / "ACEBench"


def server_base(model: str, context: int) -> list[str]:
    """How every recorded server command starts: llama-server with the base model and the context."""
    return ["llama-server", "-m", model, "-ngl", "99", "-c", str(context), "--jinja", "--host", "127.0.0.1",
            "--port", "8080", "--cache-ram", "0"]


def planned(git: FakeGit, *argv: str):
    """The plans run_acebench makes of the arguments, by system, with the run's setup and the parsed
    arguments (prepare fills in the default results and raw folders)."""
    args = ace.build_parser().parse_args([*BINARY, *argv])
    plans, _, _, setup = ace.prepare(args, git.run)
    return {p.system: p for p in plans}, setup, args


def served(folder: str) -> tuple[str, ...]:
    """The arguments that serve a model folder's GGUF as its family."""
    return "--model", f"{folder}-Q8_0.gguf", "--tokenizer", folder


def sampling(temperature: str, top_p: str, top_k: str, min_p: str, presence: str) -> list[str]:
    return ["--temp", temperature, "--top-p", top_p, "--top-k", top_k, "--min-p", min_p, "--presence-penalty",
            presence]


SWITCH_ON = ["--chat-template-kwargs", '{"enable_thinking":true}']
# case: (arguments, base model GGUF, context, what follows the base command, generate.py's temperature, top_p
# and max_tokens, ACEBench's names for b0 and ft)
COMMANDS = {
    # the default model folder, Qwen3-1.7B (qwen3-hybrid): a family that thinks runs in think mode by
    # default, with its template's switch turned on
    "qwen3-hybrid": (
        (), "models/Qwen3-1.7B-Q8_0.gguf", 12288,
        [*ONE_SLOT, *THINKING, "1024", *SWITCH_ON, *QWEN3_SAMPLING, "--seed", "0"],
        ("0.6", "0.95", "2224"), ("local-qwen3-hybrid-base", "local-qwen3-hybrid-lora")),
    # without thinking: the M4 command and the M4 names
    "qwen3-hybrid no_think": (
        ("--mode", "no_think"), "models/Qwen3-1.7B-Q8_0.gguf", 8192, [*ONE_SLOT, *M4_SWITCH],
        ("0", "1", "1200"), ("local-qwen3-1.7b-base", "local-qwen3-1.7b-lora")),
    # the same family on another base model: the M4 command, the family's names
    "qwen3-hybrid 4B no_think": (
        ("--mode", "no_think", *served("models/Qwen3-4B")), "models/Qwen3-4B-Q8_0.gguf", 8192,
        [*ONE_SLOT, *M4_SWITCH], ("0", "1", "1200"), ("local-qwen3-hybrid-base", "local-qwen3-hybrid-lora")),
    # a family that always thinks: its template opens the block unasked, no switch is sent
    "qwen3-thinking-2507": (
        served(THINK_FOLDER), THINK_GGUF, 12288,
        [*ONE_SLOT, *THINKING, "1024", *QWEN3_SAMPLING, "--seed", "0"], ("0.6", "0.95", "2224"), THINK_NAMES),
    # another budget and seed, and arguments after "--", which come last
    "qwen3-thinking-2507 R 1280": (
        (*served(THINK_FOLDER), "--reasoning-budget", "1280", "--seed", "7", "--", "--threads", "8"),
        THINK_GGUF, 12288,
        [*ONE_SLOT, *THINKING, "1280", *QWEN3_SAMPLING, "--seed", "7", "--threads", "8"],
        ("0.6", "0.95", "2480"), THINK_NAMES),
    # thinking decoded greedily: ACEBench's temperature 0 decides, no sampling flag, no seed
    "qwen3-thinking-2507 greedy": (
        (*served(THINK_FOLDER), "--greedy"), THINK_GGUF, 12288, [*ONE_SLOT, *THINKING, "1024"],
        ("0", "1", "2224"), THINK_NAMES),
    # Qwen3.5 thinks unless switched off: no switch in think mode; its presence penalty
    "qwen3.5-4b": (
        served("models/Qwen3.5-4B"), "models/Qwen3.5-4B-Q8_0.gguf", 12288,
        [*ONE_SLOT, *THINKING, "1024", *sampling("1.0", "0.95", "20", "0.0", "1.5"), "--seed", "0"],
        ("1.0", "0.95", "2224"), ("local-qwen3.5-4b-base", "local-qwen3.5-4b-lora")),
    # Granite: top_k and min_p off, where the server's own defaults are 40 and 0.05
    "granite-4.2-3b": (
        served("models/granite-4.2-3b"), "models/granite-4.2-3b-Q8_0.gguf", 12288,
        [*ONE_SLOT, *THINKING, "1024", *SWITCH_ON, *sampling("1.0", "0.95", "0", "0.0", "0.0"), "--seed",
         "0"],
        ("1.0", "0.95", "2224"), ("local-granite-4.2-3b-base", "local-granite-4.2-3b-lora")),
    # LFM2.5: its repetition penalty over the whole sequence, a window of the context
    "lfm2.5-2.6b": (
        served("models/LFM2.5-2.6B"), "models/LFM2.5-2.6B-Q8_0.gguf", 12288,
        [*ONE_SLOT, *THINKING, "1024", *sampling("0.1", "1.0", "50", "0.0", "0.0"), "--repeat-penalty", "1.1",
         "--repeat-last-n", "12288", "--seed", "0"],
        ("0.1", "1.0", "2224"), ("local-lfm2.5-2.6b-base", "local-lfm2.5-2.6b-lora")),
    # a family that never thinks runs without thinking by default, and has no switch to send
    "qwen3-instruct-2507": (
        served("models/Qwen3-4B-Instruct-2507"), "models/Qwen3-4B-Instruct-2507-Q8_0.gguf", 8192, [*ONE_SLOT],
        ("0", "1", "1200"), ("local-qwen3-instruct-2507-base", "local-qwen3-instruct-2507-lora")),
}


@pytest.mark.parametrize("case", list(COMMANDS))
def test_the_server_and_acebench_commands_per_family_and_mode(root, git, case):
    argv, model, context, extra, (temperature, top_p, max_tokens), names = COMMANDS[case]
    plans, setup, args = planned(git, *argv)
    b0, ft = plans["b0"], plans["ft"]
    assert b0.record == [*server_base(model, context), *extra]
    assert ft.record == [*server_base(model, context), "--lora", ADAPTERS["ft"], "--lora-init-without-apply",
                         *extra]
    assert [b0.serve[0], *b0.record[1:]] == b0.serve  # the command started is the one recorded
    assert (b0.model, ft.model) == names
    for p, name in ((b0, names[0]), (ft, names[1])):
        for language in ("zh", "en"):
            assert ace.as_recorded(p.generate[language]) == [
                ace.ACE_PYTHON.as_posix(), "generate.py", "--model", name, "--category", "normal", "special",
                "--language", language, "--temperature", temperature, "--top-p", top_p, "--max-tokens",
                max_tokens, "--num-threads", "1"]
            assert ace.as_recorded(p.evaluate[language]) == [ace.ACE_PYTHON.as_posix(), "eval_main.py",
                                                              "--language", language, "--model", name,
                                                              "--category", "normal", "special"]
    assert setup.context == context
    m7 = context == 12288  # a thinking run's results and raw outputs are M7's
    expected = (THINK_OUT, Path("runs/acebench/m7")) if m7 else (OUT, Path("runs/acebench/m4"))
    assert (args.out, args.raw) == expected


def test_smoke_runs_of_either_mode_keep_files_of_their_own(root, git):
    _, _, args = planned(git, "--smoke")
    assert (args.out, args.raw) == (Path("results/acebench-m7-smoke.json"), Path("runs/acebench/m7-smoke"))
    _, _, args = planned(git, "--smoke", "--mode", "no_think")
    assert (args.out, args.raw) == (ace.SMOKE_OUT, ace.SMOKE_RAW) == (Path("results/acebench-m4-smoke.json"),
                                                                       Path("runs/acebench/m4-smoke"))


@pytest.mark.parametrize("argv, said", [
    (("--model", "models/Qwen3-4B-Instruct-2507-Q8_0.gguf", "--tokenizer", "models/Qwen3-4B-Instruct-2507",
      "--mode", "think"), "never thinks"),
    (("--model", THINK_GGUF, "--tokenizer", THINK_FOLDER, "--mode", "no_think"), "always thinks"),
    (("--mode", "no_think", "--greedy"), "--greedy"),
    (("--mode", "no_think", "--seed", "1"), "--seed"),
    (("--mode", "no_think", "--reasoning-budget", "512"), "--reasoning-budget"),
    (("--greedy", "--seed", "1"), "--seed"),
    (("--reasoning-budget", "0"), "--reasoning-budget"),
    (("--reasoning-budget", "5489"), "5488"),  # 12,288 - the prompts' 5,600 - the reply's 1,200
    (("--seed", "-1"), "--seed"),
    (("--seed", "4294967295"), "from 0 to 2147483647"),  # 0xFFFFFFFF, llama.cpp's random seed
    (("--family", "qwen3.5-4b"), "qwen3.5-4b"),  # the default folder is qwen3-hybrid
    (("--model", THINK_GGUF), "--tokenizer"),  # another base model, served with the default folder's family
    (("--model", THINK_GGUF, "--tokenizer", "models/nowhere"), "models/nowhere"),
    (("--model", THINK_GGUF, "--tokenizer", "results"), "config.json"),  # a folder that is no model folder
])
def test_runs_that_cannot_be_served_as_asked_are_refused_before_anything_starts(root, git, capsys, argv,
                                                                                said):
    assert ace.main([*BINARY, *argv, "--dry-run"], popen=failing, run=failing, client=failing,
                    git=git.run) == 2
    out = capsys.readouterr().out
    assert "FAIL: " in out and said in out
    assert not (root / "runs" / "acebench").exists()


def test_no_think_refuses_a_family_that_samples_without_thinking(root, git, capsys, monkeypatch):
    """A no_think run decodes greedily, as M4 did. Every registered hybrid family answers greedily without
    thinking, so the family here is made up: qwen3-hybrid sampling in its no_think mode."""
    sampling = dataclasses.replace(
        QWEN3_HYBRID, name="sampling-hybrid",
        decoding={Mode.THINK: QWEN3_HYBRID.decoding[Mode.THINK], Mode.NO_THINK: Decoding(temperature=0.7)})
    monkeypatch.setitem(FAMILY_FOLDERS, "models/Sampling-Hybrid", sampling)
    (root / "models" / "Sampling-Hybrid").mkdir()
    (root / "models" / "Sampling-Hybrid-Q8_0.gguf").write_bytes(b"GGUF sampling")
    argv = [*BINARY, "--mode", "no_think", *served("models/Sampling-Hybrid"), "--dry-run"]
    assert ace.main(argv, popen=failing, run=failing, client=failing, git=git.run) == 2
    assert "sampling-hybrid samples without thinking" in capsys.readouterr().out
    assert ace.main([*argv[:2], *argv[4:]], popen=failing, run=failing, client=failing, git=git.run) == 0


def test_the_thinking_tags_are_thinks_and_the_familys_own():
    """The tags no probe answer may hold: "<think>", "</think>" and the family's own, stripped, each once.
    Every registered family's own are those two, so a family with others is made up here."""
    assert ace.think_tags(QWEN3_THINKING_2507) == ace.think_tags(LFM25_26B) == ["<think>", "</think>"]
    seed = types.SimpleNamespace(think_open="<seed:think>\n", think_close="\n</seed:think>\n")
    assert ace.think_tags(seed) == ["<think>", "</think>", "<seed:think>", "</seed:think>"]
    blank = types.SimpleNamespace(think_open=None, think_close=" \n")
    assert ace.think_tags(blank) == ["<think>", "</think>"]


def test_a_think_dry_run_prints_the_mode_family_decoding_budget_and_commands(root, git, capsys):
    assert ace.main([*THINK_ARGV, "--dry-run"], popen=failing, run=failing, client=failing, git=git.run) == 0
    out = capsys.readouterr().out
    assert "mode think" in out and "qwen3-thinking-2507" in out and THINK_FOLDER in out
    assert "-c 12288" in out and "--reasoning-format deepseek --reasoning-budget 1024" in out
    assert "--top-k 20 --min-p 0.0 --presence-penalty 0.0 --seed 0" in out
    assert "--temperature 0.6 --top-p 0.95 --max-tokens 2224" in out
    assert "generate.py --model local-qwen3-thinking-2507-base" in out and THINK_OUT.as_posix() in out
    assert not (root / "runs" / "acebench").exists() and not (root / THINK_OUT).exists()
    # where R comes from is said: the family's unless --reasoning-budget gives it, so no default is silent
    assert ("reasoning budget 1024 (qwen3-thinking-2507's reasoning_budget: --reasoning-budget not given)"
            in out)
    assert ace.main([*THINK_ARGV, "--reasoning-budget", "1280", "--dry-run"], popen=failing, run=failing,
                    client=failing, git=git.run) == 0
    assert "reasoning budget 1280 (--reasoning-budget)" in capsys.readouterr().out
    # --family names the folder's family, or the run is refused (above)
    assert ace.main([*BINARY, "--family", "qwen3-hybrid", "--dry-run"], popen=failing, run=failing,
                    client=failing, git=git.run) == 0
    assert "generate.py --model local-qwen3-hybrid-base" in capsys.readouterr().out


def test_a_think_run_serves_the_family_probes_each_server_and_records_mode_family_decoding_and_budget(
        root, git, capsys):
    llama = FakeLlama()
    fake = FakeAce(llama, root)
    assert ace.main(THINK_ARGV, popen=llama.popen, run=fake.run, client=llama.client, git=git.run) == 0
    assert not (root / OUT).exists()
    results = json.loads((root / THINK_OUT).read_text(encoding="utf-8"))
    assert results["benchmark"] == ("ACEBench 56dd66c with the local patches of third_party_patches/"
                                    "acebench.diff, through llama-server in thinking mode")
    assert results["mode"] == "think" and results["base_model"] == THINK_GGUF
    assert results["family"] == {"name": "qwen3-thinking-2507", "thinking": "only", "folder": THINK_FOLDER}
    assert results["generation"] == {"temperature": 0.6, "top_p": 0.95, "max_tokens": 2224, "num_threads": 1}
    assert results["budget"] == {"reasoning": 1024, "reply": 1200, "max_tokens": 2224, "context": 12288}
    decoding = results["decoding"]
    assert {key: decoding[key] for key in ("greedy", "temperature", "top_p", "top_k", "min_p",
                                           "presence_penalty", "seed")} == {
        "greedy": False, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 0.0, "seed": 0}
    assert decoding["acebench_sends"] == ["temperature", "top_p", "max_tokens"]
    assert decoding["server_sets"] == ["top_k", "min_p", "presence_penalty", "seed"]
    assert "--seed" in decoding["rule"] and "one slot" in decoding["rule"]
    extra = [*ONE_SLOT, *THINKING, "1024", *QWEN3_SAMPLING, "--seed", "0"]
    b0, ft = results["systems"]["b0"], results["systems"]["ft"]
    assert b0["server_command"] == [*server_base(THINK_GGUF, 12288), *extra]
    assert ft["server_command"] == [*server_base(THINK_GGUF, 12288), "--lora", ADAPTERS["ft"],
                                    "--lora-init-without-apply", *extra]
    assert (b0["acebench_model"], ft["acebench_model"]) == THINK_NAMES
    assert (b0["raw"], ft["raw"]) == ("runs/acebench/m7/b0", "runs/acebench/m7/ft")
    for row in (b0, ft):
        assert row["probe"] == {"answer": PROBE_ANSWER, "reasoning_chars": len(REASONING),
                                "finish_reason": "stop"}
        for language in ("zh", "en"):
            assert row["languages"][language]["generate"][-8:] == [
                "--temperature", "0.6", "--top-p", "0.95", "--max-tokens", "2224", "--num-threads", "1"]
        assert row["aggregate"]["categories"] == 10
    assert (root / "runs/acebench/m7/ft/server.log").is_file()
    # each server was probed with one chat completion sent as ACEBench sends one: the model's name, a system
    # and a user message, the run's temperature, top_p and max_tokens, and nothing else (no lora,
    # cache_prompt, seed, chat_template_kwargs or other sampling field); the no-think check is not sent
    chats = [payload for path, payload in llama.payloads if path == "/v1/chat/completions"]
    assert chats == [{"model": name, "messages": ace.PROBE_MESSAGES, "temperature": 0.6, "top_p": 0.95,
                      "max_tokens": 2224} for name in THINK_NAMES]
    assert [message["role"] for message in ace.PROBE_MESSAGES] == ["system", "user"]
    assert "[ApiName(" in ace.PROBE_MESSAGES[0]["content"]
    assert len(llama.processes) == 2 and all(p.terminated for p in llama.processes)
    per_system = ["reach", "generate", "generate", "evaluate", "summary", "evaluate", "summary"]
    assert [step.split(" ")[0] for step in steps(fake)] == ["version", *per_system * 2]
    printed = capsys.readouterr().out
    assert THINK_OUT.as_posix() in printed and "empty answers" not in printed
    for row in (b0, ft):
        for block in row["languages"].values():
            assert {cell["empty"] for cell in block["format"].values()} == {0}
    # the report reads the file as it reads M4's and M6's (results/acebench-<name>.json)
    numbers = report.acebench_numbers(results, "m7")
    assert numbers["acebench.m7.ft.aggregate"] == f"{100 * ft['aggregate']['accuracy']:.2f}%"
    assert numbers["acebench.m7.b0.zh.format.normal.n"] == str(sum(N[c] for c in GROUPS["normal"]))


@pytest.mark.parametrize("argv, server, extra, said", [
    # the server leaves the reasoning in the content: nothing is parsed out
    (THINK_ARGV, {}, ("--", "--reasoning-format", "none"), "reasoning_content"),
    # the reasoning is parsed out, and its tags are kept in the content as well
    (THINK_ARGV, {}, ("--", "--reasoning-format", "deepseek-legacy"), "<think>"),
    # an opening tag alone left in the content, before an answer the parser reads
    (THINK_ARGV, {"answer": f"<think>\n{PROBE_ANSWER}"}, (), "content holds <think> ("),
    # a reasoning_content of whitespace alone: no reasoning was separated
    (THINK_ARGV, {"reasoning": " \n\t"}, (), "came back without reasoning_content"),
    # a hybrid model whose switch is turned off again after the run's own: it answers without thinking
    (BINARY, {}, ("--", "--chat-template-kwargs", '{"enable_thinking":false}'), "reasoning_content"),
    # answers ACEBench's parser cannot read: the fine-tuned format, or text
    (THINK_ARGV, {"answer": TOOL_ANSWER}, (), "[ApiName(key='value')]"),
    (THINK_ARGV, {"answer": "It is sunny in Paris today."}, (), "[ApiName(key='value')]"),
])
def test_a_failed_thinking_probe_stops_the_run_with_its_reason(root, git, capsys, argv, server, extra, said):
    llama = FakeLlama(**server)
    fake = FakeAce(llama, root)
    assert ace.main([*argv, *extra], popen=llama.popen, run=fake.run, client=llama.client, git=git.run) == 3
    out = capsys.readouterr().out
    assert "FAIL: b0: the thinking probe" in out and said in out
    assert steps(fake) == ["version"] and len(llama.processes) == 1 and llama.processes[0].terminated
    assert not (root / THINK_OUT).exists()


@pytest.mark.parametrize("extra, said", [
    (("--", "--top-k", "40"), "top_k"),
    (("--", "--min-p", "0.05"), "min_p"),
    (("--", "--seed", "5"), "seed"),
    # off by about 1e-4, far more than a 32-bit float's rounding (0.95 comes back 1.2e-8 off)
    (("--", "--min-p", "0.0001"), "min_p"),
])
def test_a_server_whose_sampling_defaults_are_not_the_familys_stops_the_run(root, git, capsys, extra, said):
    """What ACEBench cannot send is the server's default: /props must report the family's settings."""
    llama = FakeLlama()
    fake = FakeAce(llama, root)
    assert ace.main([*THINK_ARGV, *extra], popen=llama.popen, run=fake.run, client=llama.client,
                    git=git.run) == 3
    out = capsys.readouterr().out
    assert "FAIL: b0: " in out and said in out and "/props" in out
    assert steps(fake) == ["version"] and not [path for path, _ in llama.payloads
                                               if path == "/v1/chat/completions"]
    assert all(p.terminated for p in llama.processes) and not (root / THINK_OUT).exists()


# qwen3-hybrid without thinking on a base model other than the default: the family's names, which its think
# runs use as well
HYBRID_4B_NO_THINK = [*BINARY, "--mode", "no_think", *served("models/Qwen3-4B")]


def plant(root: Path, *names: str) -> list[Path]:
    """ACEBench's result, score and Excel folders of names, as an earlier run leaves them in the clone."""
    folders = [root / CLONE / folder for name in names
               for folder in (f"result_all/result_zh/{name}", f"score_all/score_en/{name}",
                              f"../result_excel/zh/{name}")]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "data_x.json").write_text("{}\n", encoding="utf-8")
    return folders


@pytest.mark.parametrize("argv, folder", [
    # the default model folder thinks by default: qwen3-hybrid's names, in each of ACEBench's three folders
    (BINARY, "result_all/result_zh/local-qwen3-hybrid-base"),
    (BINARY, "score_all/score_en/local-qwen3-hybrid-lora"),
    (BINARY, "../result_excel/zh/local-qwen3-hybrid-lora"),
    # a family that always thinks
    (THINK_ARGV, "result_all/result_en/local-qwen3-thinking-2507-lora"),
    # a think run's folders of qwen3-hybrid on Qwen3-4B block its no_think run, whose names they are too
    (HYBRID_4B_NO_THINK, "result_all/result_zh/local-qwen3-hybrid-base"),
    (HYBRID_4B_NO_THINK, "../result_excel/zh/local-qwen3-hybrid-lora"),
])
def test_acebench_folders_under_the_runs_own_names_are_refused_in_either_mode(
        root, git, capsys, argv, folder):
    """The refusal of the M4 run (above) for the names a run of the family has: generate.py would keep an
    earlier run's answers, a think run's among them, and mix them into this one."""
    earlier = root / CLONE / folder
    earlier.mkdir(parents=True)
    (earlier / "data_x.json").write_text("{}\n", encoding="utf-8")
    assert ace.main(list(argv), popen=failing, run=failing, client=failing, git=git.run) == 2
    out = capsys.readouterr().out
    assert Path(os.path.normpath(CLONE / folder)).as_posix() in out and "move" in out
    assert (earlier / "data_x.json").is_file()
    assert not (root / OUT).exists() and not (root / THINK_OUT).exists()


def test_acebench_folders_under_other_names_leave_a_run_alone(root, git):
    """A run refuses its own names' folders only: M4's (local-qwen3-1.7b-*), which M6 left in the clone,
    block neither a think run nor a run of another base model, and the families' folders block no M4
    rerun."""
    m4 = plant(root, BASE_NAME, ADAPTER_NAME)
    for argv in (BINARY, THINK_ARGV, HYBRID_4B_NO_THINK):
        assert ace.main([*argv, "--dry-run"], popen=failing, run=failing, client=failing,
                        git=git.run) == 0, argv
    for folder in m4:
        shutil.rmtree(folder)
    plant(root, "local-qwen3-hybrid-base", "local-qwen3-hybrid-lora", *THINK_NAMES)
    assert dry(git) == 0


@pytest.mark.parametrize("server, extra, said", [
    ({"default_scale": 0.0}, (), "FAIL: ft: a completion without a lora field"),
    ({"ignores_lora": True}, (), "FAIL: ft: a completion without a lora field"),
    ({}, ("--", "--cache-prompt"), "FAIL: b0: the same prompt sent twice without a cache_prompt field"),
    ({}, ("--", "-np", "4"), "FAIL: b0: the server reports 4 slots of 12288 tokens"),
    ({}, ("--", "-c", "8192"), "FAIL: b0: the server reports 1 slots of 8192 tokens"),
    ({"phantom": ADAPTERS["b4"]}, (), "FAIL: b0: the server lists the adapters"),
    ({"builds": ("b11349-fb4b2737a", "b12000-0000000")}, (), "FAIL: ft: the server reports build b12000"),
])
def test_a_think_server_that_fails_a_check_stops_the_run_before_its_probe(root, git, capsys, server, extra,
                                                                         said):
    """Every server check of the M4 run holds in think mode as well, before the sampling check and the
    thinking probe: a server that fails one gets no probe, and ACEBench never runs against it."""
    llama = FakeLlama(**server)
    fake = FakeAce(llama, root)
    assert ace.main([*THINK_ARGV, *extra], popen=llama.popen, run=fake.run, client=llama.client,
                    git=git.run) == 3
    assert said in capsys.readouterr().out
    assert all(p.terminated for p in llama.processes) and not (root / THINK_OUT).exists()
    assert steps(fake)[-1].split(" ")[0] in ("version", "summary")  # nothing ran against the failed server
    chats = [payload for path, payload in llama.payloads if path == "/v1/chat/completions"]
    assert len(chats) == (1 if said.startswith("FAIL: ft") else 0)  # b0's probe alone, when b0 passed


def test_a_think_run_with_a_repetition_penalty_sets_and_records_its_window_as_the_context(root, git):
    """LFM2.5's repetition penalty covers the whole sequence: the server's window is the think context, 12,288
    tokens, on the command line, in /props and in the results."""
    llama = FakeLlama()
    fake = FakeAce(llama, root)
    assert ace.main([*BINARY, *served("models/LFM2.5-2.6B")], popen=llama.popen, run=fake.run,
                    client=llama.client, git=git.run) == 0
    results = json.loads((root / THINK_OUT).read_text(encoding="utf-8"))
    assert results["family"] == {"name": "lfm2.5-2.6b", "thinking": "only", "folder": "models/LFM2.5-2.6B"}
    decoding = {key: value for key, value in results["decoding"].items() if key != "rule"}
    assert decoding == {"greedy": False, "temperature": 0.1, "top_p": 1.0, "top_k": 50, "min_p": 0.0,
                        "presence_penalty": 0.0, "repetition_penalty": 1.1, "repetition_window": 12288,
                        "seed": 0, "acebench_sends": ["temperature", "top_p", "max_tokens"],
                        "server_sets": ["top_k", "min_p", "presence_penalty", "repeat_penalty",
                                        "repeat_last_n", "seed"]}
    assert results["budget"] == {"reasoning": 1024, "reply": 1200, "max_tokens": 2224, "context": 12288}
    for system, row in results["systems"].items():
        command = row["server_command"]
        at = command.index("--repeat-penalty")
        assert command[at:at + 4] == ["--repeat-penalty", "1.1", "--repeat-last-n", "12288"], system
        assert row["acebench_model"] == f"local-lfm2.5-2.6b-{'base' if system == 'b0' else 'lora'}"
    assert (llama.params["repeat_penalty"], llama.params["repeat_last_n"]) == (float32(1.1), 12288)


@pytest.mark.parametrize("argv, out, why", [
    (THINK_ARGV, THINK_OUT, ("a reasoning that does not close within max_tokens 2224 leaves no answer, "
                             "which the server's reasoning budget (1024) should prevent")),
    (ARGV, OUT, "the model answered nothing"),
], ids=["think", "no_think"])
def test_empty_answers_are_counted_per_category_and_printed(root, git, capsys, argv, out, why):
    """An answer of empty or whitespace text is no inference error (ACEBench got content, and scores it as
    wrong): it is counted per category beside the format counts and printed per system, so that a run whose
    reasoning ran out of max_tokens, the reasoning budget not ending it, shows as such."""
    llama = FakeLlama()
    fake = FakeAce(llama, root, empties={("zh", "normal_atom_bool"): 2, ("en", "special_irrelevant"): 1})
    assert ace.main(list(argv), popen=llama.popen, run=fake.run, client=llama.client, git=git.run) == 0
    printed = capsys.readouterr().out
    results = json.loads((root / out).read_text(encoding="utf-8"))
    assert "empty" in results["format_rule"]
    for system in ("b0", "ft"):
        assert (f"{system}: 3 empty answers (zh: normal_atom_bool 2; en: special_irrelevant 1), which "
                f"ACEBench scores as wrong: {why}") in printed
        languages = results["systems"][system]["languages"]
        empty = {language: {c: cell["empty"] for c, cell in languages[language]["format"].items()}
                 for language in ("zh", "en")}
        assert empty == {"zh": {**dict.fromkeys(CATEGORIES, 0), "normal_atom_bool": 2},
                         "en": {**dict.fromkeys(CATEGORIES, 0), "special_irrelevant": 1}}
        cell = languages["zh"]["format"]["normal_atom_bool"]
        assert (cell["n"], cell["tool_call"], cell["acebench_format"]) == (2, 0, 0)  # neither format
        assert languages["zh"]["inference_errors"] == languages["en"]["inference_errors"] == dict.fromkeys(
            CATEGORIES, 0)


@pytest.mark.parametrize("name, system", [("acebench-m4.json", "b0"), ("acebench-m4.json", "ft"),
                                          ("acebench-m4-b4.json", "b4"), ("acebench-m6.json", "ft")])
def test_no_think_repeats_the_recorded_m4_and_m6_commands_byte_for_byte(root, git, name, system):
    """The no_think path of Qwen3-1.7B against the commands results/acebench-m4*.json and acebench-m6.json
    recorded: the server command, ACEBench's model name, raw folder and both steps per language, the
    generation settings and the benchmark's description."""
    recorded = json.loads((REPO / "results" / name).read_text(encoding="utf-8"))
    row = recorded["systems"][system]
    python = row["languages"]["zh"]["generate"][0]  # as recorded on Windows
    (root / python).parent.mkdir(parents=True, exist_ok=True)
    (root / python).write_bytes(b"")
    argv = ["--mode", "no_think", "--systems", system, "--acebench-python", python, "--raw",
            Path(row["raw"]).parent.as_posix(), "--out", f"results/{name}"]
    if row["adapter"] is not None:
        gguf = root / row["adapter"]["gguf"]
        gguf.parent.mkdir(parents=True, exist_ok=True)
        gguf.write_bytes(b"GGUF as recorded")
        argv += [f"--{system}-adapter", row["adapter"]["gguf"]]
    plans, setup, _ = planned(git, *argv)
    p = plans[system]
    assert p.record == row["server_command"]
    assert p.model == row["acebench_model"] and ace.rel(p.root) == row["raw"]
    assert list(p.generate) == list(row["languages"]) == ["zh", "en"]
    for language, block in row["languages"].items():
        assert ace.as_recorded(p.generate[language]) == block["generate"]
        assert ace.as_recorded(p.evaluate[language]) == block["evaluate"]
    assert setup.generation == recorded["generation"]
    assert ace.benchmark(setup) == recorded["benchmark"]


def added_models(diff: str) -> list[str]:
    """The model names the recorded patch adds to ACEBench's API models (model_inference/inference_map.py)."""
    section = diff.split("diff --git a/model_inference/inference_map.py ")[1].split("\ndiff --git ")[0]
    return re.findall(r'^\+        "([^"]+)",$', section, flags=re.MULTILINE)


def test_the_patch_registers_the_m4_names_and_a_base_and_a_lora_name_per_family():
    names = added_models((REPO / "third_party_patches" / "acebench.diff").read_text(encoding="utf-8"))
    expected = ["local-qwen3-1.7b-base", "local-qwen3-1.7b-lora",
                *(f"local-{family.name}-{kind}" for family in FAMILIES for kind in ("base", "lora"))]
    assert names == expected == ace.registered_models() == spike.LOCAL_MODELS
    assert len(set(names)) == len(names) == 2 + 2 * len(FAMILIES)
    for family in FAMILIES:
        assert ace.acebench_model(family.name, "b0") == f"local-{family.name}-base"
        assert ace.acebench_model(family.name, "ft") == ace.acebench_model(family.name, "b4") == \
            f"local-{family.name}-lora"
    for name in names:
        # ACEBench's client reaches the local server for a name that starts with "local-" unless "gpt" or
        # "deepseek-r1" in it sends it to another provider first (model_inference/apimodel_inference.py), and
        # reads the answers of a name holding "FC" as function-calling JSON (model_inference/utils.py)
        assert name.startswith("local-") and "gpt" not in name and "deepseek-r1" not in name
        assert "FC" not in name


def git_out(*args: str, cwd: Path) -> bytes:
    """git's output, never taking an optional lock (so a read of the clone writes nothing into it)."""
    done = subprocess.run(["git", "--no-optional-locks", *args], cwd=cwd, capture_output=True, check=False)
    assert done.returncode == 0, (args, done.stderr.decode("utf-8", "replace"))
    return done.stdout


@pytest.mark.skipif(not (CLONE_ON_DISK / "generate.py").is_file() or shutil.which("git") is None,
                    reason="needs the local ACEBench clone (runs/ is not tracked) and git")
def test_the_recorded_patch_applies_to_the_pinned_revision_and_is_what_the_spike_writes(tmp_path):
    """Read-only on the clone: the upstream files of the pinned commit are read with git cat-file into two
    scratch repositories; the recorded patch applies to the first and the spike script edits the second, and
    either's git diff under the clone check's options is the recorded patch byte for byte, so a clean clone
    patched either way passes the clone check."""
    patch = REPO / "third_party_patches" / "acebench.diff"
    files = re.findall(r"^diff --git a/(\S+) b/\1$", patch.read_text(encoding="utf-8"), flags=re.MULTILINE)
    assert files == ["category.py", "model_inference/apimodel_inference.py",
                     "model_inference/inference_map.py", "model_inference/model_infer.py"]
    head = git_out("rev-parse", "HEAD", cwd=CLONE_ON_DISK)
    for name in ("applied", "spiked"):
        work = tmp_path / name
        work.mkdir()
        git_out("init", "-q", ".", cwd=work)
        git_out("config", "core.autocrlf", "false", cwd=work)
        for path in files:
            target = work / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(git_out("cat-file", "blob", f"{ace.UPSTREAM}:{path}", cwd=CLONE_ON_DISK))
        git_out("add", "--", *files, cwd=work)
    git_out("apply", "--check", str(patch), cwd=tmp_path / "applied")
    git_out("apply", str(patch), cwd=tmp_path / "applied")
    spike.main(tmp_path / "spiked")
    for name in ("applied", "spiked"):
        assert git_out(*ace.DIFF_ARGS, "--", cwd=tmp_path / name) == patch.read_bytes(), name
    for path in files:
        assert (tmp_path / "applied" / path).read_bytes() == (tmp_path / "spiked" / path).read_bytes()
    assert git_out("rev-parse", "HEAD", cwd=CLONE_ON_DISK) == head


def test_the_family_is_read_from_the_model_folder_by_the_family_layer():
    """Outside the fake root: the script identifies the family as the family layer does, from the folder's
    config.json and chat template (research_agent.family.core.folder_family)."""
    assert ace.folder_family is family_core.folder_family
