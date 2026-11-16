"""scripts/acebench_lengths.py offline (docs/m11-计划.md, 1. 报告补项: ACEBench 生成长度). Synthetic
server logs in the lines llama-server b11349 wrote at verbosity 3 for M7's run (the probes first, three on
b0's server and six on ft's, then one request per answer, zh before en) and a results file as
scripts/run_acebench.py writes one in think mode give each system's generated lengths per language and over
both, the probes left out: the counts, the percentiles, the requests over the reasoning budget and those at
max_tokens. Every anomaly fails the run with exit 1 and writes nothing: a task without one of its three lines
or with one twice, n_tokens other than P + G - 1, a truncated prompt, more tokens than max_tokens, requests
that are not the probes and one per answer, probes that are not what the code sends, a budget the recorded
commands do not carry, a source of another kind, a missing log or source. The same inputs give the same
bytes; an existing file is kept unless --force is given. The probes left out are the requests
scripts/run_acebench.py's own server_problem sends a think-mode server before ACEBench starts (over a fake
server here). Two tests read the real logs of M7's run (runs/acebench/m7, which the public copy leaves
out)."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from research_agent.family.core import GREEDY, Mode
from research_agent.family.registry import NANBEIGE41_3B
from research_agent.llm.local import LocalClient

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


acebench_lengths = load_script("acebench_lengths")
ace = load_script("run_acebench")  # its server_problem: the probes a server gets before ACEBench starts

OUT = Path("results/lengths-acebench-m7.json")
SOURCE = Path("results/acebench-m7.json")
SYSTEMS = ("b0", "ft")
LANGUAGES = ("zh", "en")
R, MAX_TOKENS = 20, 40  # the synthetic run's reasoning budget and max_tokens (M7's: 1280 and 2480)
ADAPTER = "runs/train/ft3/gguf/checkpoint-1635-f16.gguf"
# the answers per language and category: ACEBench sends one request per answer
N = {"zh": {"normal_atom_bool": 3, "special_incomplete": 2},
     "en": {"normal_atom_bool": 2, "special_incomplete": 2}}
# the tokens generated for each answer, in the order ACEBench asked for them
GENERATED = {"b0": {"zh": [5, 21, 40, 12, 30], "en": [8, 20, 40, 40]},
             "ft": {"zh": [16, 2, 39, 24, 7], "en": [33, 11, 6, 19]}}
PROBE_PROMPT, THINK_PROMPT = 28, 200  # the probes' prompt tokens in M7's logs
THINKING = {"b0": 33, "ft": 27}  # the thinking probe's tokens: above R, so a probe counted would show
# the blocks by hand: linear interpolation between closest ranks; over_budget G > R (b0's 20 in en is not),
# at_max_tokens G == MAX_TOKENS
EXPECTED = {
    "b0": {"zh": {"n": 5, "tokens_p50": 21, "tokens_p95": 38, "tokens_max": 40, "over_budget": 3,
                  "at_max_tokens": 1},
           "en": {"n": 4, "tokens_p50": 30, "tokens_p95": 40, "tokens_max": 40, "over_budget": 2,
                  "at_max_tokens": 2},
           "all": {"n": 9, "tokens_p50": 21, "tokens_p95": 40, "tokens_max": 40, "over_budget": 5,
                   "at_max_tokens": 3}},
    "ft": {"zh": {"n": 5, "tokens_p50": 16, "tokens_p95": 36, "tokens_max": 39, "over_budget": 2,
                  "at_max_tokens": 0},
           "en": {"n": 4, "tokens_p50": 15, "tokens_p95": 30.9, "tokens_max": 33, "over_budget": 1,
                  "at_max_tokens": 0},
           "all": {"n": 9, "tokens_p50": 16, "tokens_p95": 36.6, "tokens_max": 39, "over_budget": 3,
                   "at_max_tokens": 0}},
}
BLOCK_KEYS = ["n", "tokens_p50", "tokens_p95", "tokens_max", "over_budget", "at_max_tokens"]

# the lines a log of M7's run opens with, and the tokenizer's warning before each one-token probe
HEADER = [
    "0.00.297.629 I srv  llama_server: initializing ...",
    ("0.00.297.646 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI "
     "arg)"),
    "0.00.310.728 I srv    load_model: loading model 'models/Nanbeige4.1-3B-Q8_0.gguf'",
    "0.04.412.761 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 12288, kv_unified = 'false'",
    "0.04.425.172 I srv  llama_server: model loaded",
    "0.04.425.179 I srv  llama_server: listening on http://127.0.0.1:8080",
]
BOS = ("0.04.436.786 W llama_vocab::impl::tokenize: Added a BOS token to the prompt as specified by the "
       "model but the prompt also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
       "Are you sure this is what you want?")


@dataclasses.dataclass(frozen=True)
class Request:
    """A request of a synthetic log: its prompt and generated tokens, n_tokens when not P + G - 1, its
    truncated flag, the kinds of its lines left out, and whether it is a one-token probe (the tokenizer's
    warning before it, as in M7's logs)."""

    prompt: int
    generated: int
    tokens: int | None = None
    truncated: int = 0
    without: tuple[str, ...] = ()
    one_token: bool = False


def request_lines(task: int, request: Request) -> list[str]:
    """The lines llama-server writes for one request, task, as in M7's logs: the slot chosen (task -1), the
    task started, a progress line, the four timing lines and the release, by kind (Request.without)."""
    prompt, generated = request.prompt, request.generated
    tokens = prompt + generated - 1 if request.tokens is None else request.tokens
    at = f"I slot print_timing: id  0 | task {task} |"
    lines = {
        "slot": "0.04.508.358 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 4201300",
        "launch": f"0.04.509.630 I slot launch_slot_: id  0 | task {task} | processing task, is_child = 0",
        "progress": f"0.07.563.571 {at} n_gen = {generated:6d}, tg =  54.93 t/s, tg_3s =  55.27 t/s",
        "prompt": (f"0.09.123.231 {at} prompt eval time =      50.30 ms / {prompt:5d} tokens (    0.25 ms "
                   "per token,  3975.91 tokens per second)"),
        "eval": (f"0.09.123.240 {at}        eval time =    4563.28 ms / {generated:5d} tokens (   19.75 ms "
                 "per token,    50.62 tokens per second)"),
        "total": f"0.09.123.241 {at}       total time =    4613.58 ms / {prompt + generated:5d} tokens",
        "graphs": f"0.09.123.242 {at}    graphs reused =        231",
        "stop": (f"0.09.123.289 I slot      release: id  0 | task {task} | stop processing: n_tokens = "
                 f"{tokens}, truncated = {request.truncated}"),
    }
    return [line for kind, line in lines.items() if kind not in request.without]


def write_log(path: Path, requests: list[Request], tail: tuple[str, ...] = ()) -> None:
    """A server log of requests, numbered as M7's logs number them (the next task G + 1 after the last)."""
    lines, task = list(HEADER), 1
    for request in requests:
        if request.one_token:
            lines.append(BOS)
        lines += request_lines(task, request)
        task += request.generated + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in [*lines, *tail]), encoding="utf-8", newline="\n")


def probes(system: str) -> list[Request]:
    """The probes scripts/run_acebench.py sends a think-mode server before ACEBench, as M7's logs show them:
    on ft's the adapter probe's three one-token completions, on both the cache probe's two, then the thinking
    probe's chat completion."""
    one_token = [Request(PROBE_PROMPT, 1, one_token=True) for _ in range(5 if system == "ft" else 2)]
    return [*one_token, Request(THINK_PROMPT, THINKING[system])]


def answers(system: str, languages: tuple[str, ...] = LANGUAGES) -> list[Request]:
    """ACEBench's requests, one per answer, the languages one after the other."""
    return [Request(300 + 10 * i, generated) for language in languages
            for i, generated in enumerate(GENERATED[system][language])]


def requests(system: str) -> list[Request]:
    return [*probes(system), *answers(system)]


def results(languages: tuple[str, ...] = LANGUAGES) -> dict:
    """A results file as scripts/run_acebench.py writes one in think mode (greedy, one slot, no prompt
    cache): the fields the script reads, and a few beside them."""
    systems = {}
    for system in SYSTEMS:
        adapter = None if system == "b0" else {"gguf": ADAPTER, "sha256": "a" * 64, "selection": None}
        model = f"local-nanbeige4.1-3b-{'base' if adapter is None else 'lora'}"
        lora = [] if adapter is None else ["--lora", ADAPTER, "--lora-init-without-apply"]
        command = ["llama-server", "-m", "models/Nanbeige4.1-3B-Q8_0.gguf", "-ngl", "99", "-c", "12288",
                   "--jinja", "--host", "127.0.0.1", "--port", "8080", "--cache-ram", "0", *lora, "-np", "1",
                   "--no-cache-prompt", "--reasoning-format", "deepseek", "--reasoning-budget", str(R)]
        blocks = {}
        for language in languages:
            generate = [".venv-acebench/Scripts/python.exe", "generate.py", "--model", model, "--category",
                        "normal", "special", "--language", language, "--temperature", "0", "--top-p", "1",
                        "--max-tokens", str(MAX_TOKENS), "--num-threads", "1"]
            categories = {category: {"n": n, "correct": n - 1, "accuracy": (n - 1) / n}
                          for category, n in N[language].items()}
            blocks[language] = {"generate": generate, "categories": categories}
        systems[system] = {"acebench_model": model, "server_command": command,
                           "server_build": "b11349-fb4b2737a", "adapter": adapter,
                           "raw": f"runs/acebench/m7/{system}", "languages": blocks}
    return {"benchmark": "ACEBench", "smoke": False, "languages": list(languages),
            "generation": {"temperature": 0, "top_p": 1, "max_tokens": MAX_TOKENS, "num_threads": 1},
            "mode": "think",
            "budget": {"reasoning": R, "reply": MAX_TOKENS - R, "max_tokens": MAX_TOKENS, "context": 12288},
            "systems": systems}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def log_path(root: Path, system: str) -> Path:
    return root / "runs" / "acebench" / "m7" / system / "server.log"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def run(tmp_path, monkeypatch):
    """M7's run in miniature as the current folder: its results file and both systems' server logs."""
    monkeypatch.chdir(tmp_path)
    write_json(tmp_path / SOURCE, results())
    for system in SYSTEMS:
        write_log(log_path(tmp_path, system), requests(system))
    return tmp_path


def written(root: Path) -> dict:
    return json.loads((root / OUT).read_text(encoding="utf-8"))


def test_each_systems_lengths_per_language_and_over_both_leave_the_probes_out(run, capsys):
    assert acebench_lengths.main([]) == 0
    out = capsys.readouterr().out
    data = written(run)
    assert list(data) == ["kind", "definition", "source", "source_sha256", "budget", "languages", "systems"]
    assert data["kind"] == "acebench-lengths"
    assert data["source"] == "results/acebench-m7.json" and data["source_sha256"] == sha256(run / SOURCE)
    budget = data["budget"]
    assert list(budget) == ["reasoning", "max_tokens", "from"]
    assert (budget["reasoning"], budget["max_tokens"]) == (R, MAX_TOKENS)  # read from the source
    assert all(part in budget["from"] for part in ("results/acebench-m7.json", "budget.reasoning",
                                                    "--reasoning-budget", "--max-tokens"))
    assert data["languages"] == ["zh", "en"] and list(data["systems"]) == ["b0", "ft"]
    for system, row in data["systems"].items():
        assert list(row) == ["log", "log_sha256", "probes", "requests", "zh", "en", "all"]
        assert row["log"] == f"runs/acebench/m7/{system}/server.log"
        assert row["log_sha256"] == sha256(run / row["log"])
        assert (row["probes"], row["requests"]) == {"b0": (3, 12), "ft": (6, 15)}[system]
        for part in ("zh", "en", "all"):
            assert list(row[part]) == BLOCK_KEYS
            assert row[part] == pytest.approx(EXPECTED[system][part])
    definition = data["definition"]
    assert "\n" not in definition  # one paragraph
    for said in ("reasoning", "end tag", "answer", "probe", "one per answer", "upper bound", "max_tokens",
                 "T = P + G - 1"):
        assert said in definition
    assert "results/acebench-m7.json: reasoning budget 20, max_tokens 40; languages zh, en" in out
    assert "b0: runs/acebench/m7/b0/server.log, 12 requests, 3 probes left out" in out
    assert "ft: runs/acebench/m7/ft/server.log, 15 requests, 6 probes left out" in out
    assert "  zh   n 5, p50 21, p95 38, max 40, over_budget 3 (60.0%), at_max_tokens 1 (20.0%)" in out
    assert "results/lengths-acebench-m7.json: written" in out


def test_the_requests_split_by_language_in_the_order_the_results_list_them(run):
    order = ("en", "zh")
    write_json(run / SOURCE, results(languages=order))
    for system in SYSTEMS:
        write_log(log_path(run, system), [*probes(system), *answers(system, order)])
    assert acebench_lengths.main([]) == 0
    data = written(run)
    assert data["languages"] == ["en", "zh"]
    for system, row in data["systems"].items():
        assert list(row)[-3:] == ["en", "zh", "all"]
        for part in ("zh", "en", "all"):
            assert row[part] == pytest.approx(EXPECTED[system][part])


def test_the_same_inputs_give_the_same_bytes_with_relative_paths(run):
    assert acebench_lengths.main([]) == 0
    first = (run / OUT).read_bytes()
    assert acebench_lengths.main(["--force"]) == 0
    assert (run / OUT).read_bytes() == first
    text = first.decode("utf-8")
    assert text.startswith('{\n  "kind": "acebench-lengths",\n  "definition": ') and text.endswith("}\n")
    assert "\r" not in text
    for spelling in (str(run), run.as_posix(), json.dumps(str(run))[1:-1]):  # repository-relative paths only
        assert spelling not in text


def test_an_existing_file_is_kept_unless_force_is_given(run, capsys):
    write_json(run / OUT, {"kept": True})
    assert acebench_lengths.main([]) == 2
    assert "results/lengths-acebench-m7.json exists; --force overwrites it" in capsys.readouterr().err
    assert written(run) == {"kept": True}
    assert acebench_lengths.main(["--force"]) == 0
    assert written(run)["kind"] == "acebench-lengths"


def test_an_argument_other_than_force_is_a_usage_error(run):
    with pytest.raises(SystemExit) as stopped:
        acebench_lengths.main(["--out", "elsewhere.json"])
    assert stopped.value.code == 2
    assert not (run / OUT).exists()


# ---- fail closed -----------------------------------------------------------------------------------------


def changed(system: str, index: int, **changes) -> Callable[[Path], None]:
    """Rewrite system's log with its request at index (the probes first) changed."""
    def change(root: Path) -> None:
        rewritten = requests(system)
        rewritten[index] = dataclasses.replace(rewritten[index], **changes)
        write_log(log_path(root, system), rewritten)
    return change


def relogged(system: str, rewritten: list[Request], tail: tuple[str, ...] = ()) -> Callable[[Path], None]:
    def change(root: Path) -> None:
        write_log(log_path(root, system), rewritten, tail)
    return change


def doubled(system: str, text: str, occurrence: int) -> Callable[[Path], None]:
    """Repeat the occurrence-th line of system's log that holds text right after it."""
    def change(root: Path) -> None:
        path = log_path(root, system)
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        at = [i for i, line in enumerate(lines) if text in line][occurrence]
        lines.insert(at + 1, lines[at])
        path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return change


def removed(path: str) -> Callable[[Path], None]:
    def change(root: Path) -> None:
        (root / path).unlink()
    return change


def recorded(change_results: Callable[[dict], None]) -> Callable[[Path], None]:
    """Rewrite the results file after change_results changed it."""
    def change(root: Path) -> None:
        data = results()
        change_results(data)
        write_json(root / SOURCE, data)
    return change


def set_at(path: str, value: object) -> Callable[[dict], None]:
    """Set a value of the results by its dotted path (a list item by its index)."""
    def change(data: dict) -> None:
        *parents, last = path.split(".")
        for key in parents:
            data = data[int(key)] if isinstance(data, list) else data[key]
        if isinstance(data, list):
            data[int(last)] = value
        else:
            data[last] = value
    return change


def written_text(text: str) -> Callable[[Path], None]:
    def change(root: Path) -> None:
        (root / SOURCE).write_text(text, encoding="utf-8")
    return change


B0_LOG, FT_LOG = "runs/acebench/m7/b0/server.log", "runs/acebench/m7/ft/server.log"
LAUNCHED = "0.09.200.001 I slot launch_slot_: id  0 | task 900 | processing task, is_child = 0"


@pytest.mark.parametrize(("change", "said"), [
    pytest.param(changed("b0", 3, without=("prompt",)), [B0_LOG, "has no 'prompt eval time' line"],
                 id="no prompt eval line"),
    pytest.param(changed("ft", 9, without=("eval",)), [FT_LOG, "has no 'eval time' line"], id="no eval line"),
    pytest.param(changed("b0", 7, without=("stop",)), [B0_LOG, "has no 'stop processing' line"],
                 id="no stop line"),
    pytest.param(relogged("b0", requests("b0"), (LAUNCHED,)),
                 [B0_LOG, "task 900 has no 'prompt eval time', 'eval time' and 'stop processing' lines"],
                 id="started and never finished"),
    pytest.param(doubled("ft", "|        eval time =", 8), [FT_LOG, "has two 'eval time' lines"],
                 id="a line twice"),
    pytest.param(changed("b0", 5, tokens=320 + 40), [B0_LOG, "n_tokens = 360, not P + G - 1 = 320 + 40 - 1"],
                 id="n_tokens off"),
    pytest.param(changed("ft", 7, truncated=1), [FT_LOG, "truncated = 1"], id="truncated"),
    pytest.param(changed("b0", 4, generated=MAX_TOKENS + 1),
                 [B0_LOG, "generated 41 tokens, more than max_tokens 40"], id="more than max_tokens"),
    pytest.param(relogged("b0", [*requests("b0"), Request(330, 9)]),
                 [B0_LOG, "holds 13 requests, not 3 probes + 5 zh + 4 en answers = 12"],
                 id="a request more"),
    pytest.param(relogged("ft", requests("ft")[:-1]),
                 [FT_LOG, "holds 14 requests, not 6 probes + 5 zh + 4 en answers = 15"],
                 id="a request fewer"),
    pytest.param(changed("ft", 1, generated=2),
                 [FT_LOG, "probe 2 (task 3), a one-token completion, generated 2 tokens"],
                 id="a one-token probe of two tokens"),
    pytest.param(changed("ft", 4, prompt=PROBE_PROMPT + 1),
                 [FT_LOG, "the one-token probes' prompts are of 28, 29 tokens"], id="probes of two prompts"),
    pytest.param(changed("b0", 2, generated=1), [B0_LOG, "the thinking probe (task 5) generated 1 token"],
                 id="a thinking probe of one token"),
    pytest.param(removed(B0_LOG), [B0_LOG, "is missing"], id="no b0 log"),
    pytest.param(removed(FT_LOG), [FT_LOG, "is missing"], id="no ft log"),
    pytest.param(removed("results/acebench-m7.json"), ["results/acebench-m7.json is missing"],
                 id="no source"),
    pytest.param(written_text("{"), ["results/acebench-m7.json is not JSON"], id="source not JSON"),
    pytest.param(recorded(set_at("mode", "no_think")), ["results/acebench-m7.json", "not a think-mode run"],
                 id="no-think source"),
    pytest.param(recorded(set_at("systems.ft.server_command.-1", str(R + 1))),
                 ["system ft", "--reasoning-budget 21, not budget.reasoning 20"],
                 id="another reasoning budget on the server"),
    pytest.param(recorded(set_at("systems.b0.languages.en.generate.14", str(MAX_TOKENS + 1))),
                 ["system b0", "en", "--max-tokens 41, not budget.max_tokens 40"],
                 id="another max_tokens in a generate command"),
    pytest.param(recorded(set_at("generation.max_tokens", MAX_TOKENS + 1)),
                 ["generation.max_tokens 41, not budget.max_tokens 40"], id="another max_tokens generated"),
    pytest.param(recorded(set_at("systems.b0.raw", "/runs/acebench/m7/b0")),
                 ["system b0", "raw folder", "not a repository-relative path"], id="a raw folder elsewhere"),
    pytest.param(recorded(set_at("systems.ft.languages.zh.categories.normal_atom_bool.n", 0)),
                 ["system ft", "zh", "answers"], id="a category without answers"),
])
def test_an_anomaly_fails_the_run_with_exit_1_and_writes_nothing(run, capsys, change, said):
    change(run)
    assert acebench_lengths.main([]) == 1
    err = capsys.readouterr().err
    assert err.startswith("FAIL: ") and "nothing is written" in err
    assert all(part in err for part in said), err
    assert not (run / OUT).exists()


def test_a_failed_run_leaves_an_existing_file_as_it_was_even_with_force(run):
    write_json(run / OUT, {"kept": True})
    changed("ft", 9, truncated=1)(run)
    assert acebench_lengths.main(["--force"]) == 1
    assert written(run) == {"kept": True}


# ---- what the code sends and what the real run wrote -----------------------------------------------------


def probes_sent(system: str) -> list[tuple[str, dict]]:
    """The requests scripts/run_acebench.py's server_problem sends a think-mode server of system before
    ACEBench starts (M7's run: greedy, R 1280, one slot), over a fake server that passes every check: (path,
    payload), in the order sent."""
    sent: list[tuple[str, dict]] = []
    adapter = None if system == "b0" else ace.Adapter(gguf=Path(ADAPTER), sha256="a" * 64, selection=None)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append((request.url.path, payload))
        if request.url.path == "/v1/chat/completions":
            message = {"role": "assistant", "content": "[get_weather(city='Paris')]",
                       "reasoning_content": "The user asks for the weather in Paris."}
            return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}]})
        scale = payload["lora"][0]["scale"] if "lora" in payload else (0.0 if adapter is None else 1.0)
        logprob = -0.5 if scale == 1.0 else -0.25
        token = {"token": "I", "logprob": logprob, "top_logprobs": [{"token": "I", "logprob": logprob}]}
        return httpx.Response(200, json={
            "choices": [{"text": "I", "finish_reason": "length", "logprobs": {"content": [token]}}],
            "usage": {"completion_tokens": 1, "prompt_tokens": PROBE_PROMPT,
                      "prompt_tokens_details": {"cached_tokens": 0}}})

    setup = ace.Setup(family=NANBEIGE41_3B, folder=Path("models/Nanbeige4.1-3B"), mode=Mode.THINK,
                      decoding=GREEDY, seed=None, reasoning_budget=1280, legacy=False, budget_given=True)
    plan = ace.Plan(system=system, model=ace.acebench_model(NANBEIGE41_3B.name, system), adapter=adapter,
                    root=Path("runs/acebench/m7") / system, serve=[], record=[], extra=[], generate={},
                    evaluate={}, settings={})
    server = types.SimpleNamespace(
        props={"total_slots": 1, "build_info": "b11349-fb4b2737a",
               "default_generation_settings": {"n_ctx": 12288, "params": {}}},
        adapters=[] if adapter is None else [{"id": 0, "path": ADAPTER, "scale": 0.0}],
        client=LocalClient("http://127.0.0.1:8080", transport=httpx.MockTransport(handler)))
    problem, _ = ace.server_problem(plan, server, None, setup)
    assert problem is None
    return sent


@pytest.mark.parametrize("system", SYSTEMS)
def test_the_probes_left_out_are_what_run_acebench_sends_a_think_server_before_acebench(system):
    sent = probes_sent(system)
    adapter = system == "ft"
    assert len(sent) == acebench_lengths.probes(adapter) == {"b0": 3, "ft": 6}[system]
    *one_token, think = sent
    # every probe but the last a one-token completion of one prompt; the last the thinking probe, a chat
    # completion with the run's max_tokens (R + 1,200)
    assert [path for path, _ in one_token] == ["/v1/completions"] * len(one_token)
    assert {payload["max_tokens"] for _, payload in one_token} == {1}
    assert len({payload["prompt"] for _, payload in one_token}) == 1
    assert think[0] == "/v1/chat/completions" and think[1]["max_tokens"] == 1280 + 1200
    assert acebench_lengths.THINK_PROBE == 1


REAL = ("results/acebench-m7.json", "runs/acebench/m7/b0/server.log", "runs/acebench/m7/ft/server.log")


@pytest.mark.needs_data(*REAL)  # the logs of M7's run: left out of the public copy
def test_the_real_logs_hold_the_probes_and_one_request_per_answer():
    data = acebench_lengths.summarize(REPO)
    b0, ft = data["systems"]["b0"], data["systems"]["ft"]
    assert (b0["probes"], b0["requests"]) == (3, 1943)
    assert (ft["probes"], ft["requests"]) == (6, 1946)
    for row in (b0, ft):
        assert (row["zh"]["n"], row["en"]["n"], row["all"]["n"]) == (967, 973, 1940)
        assert all(row[part]["tokens_max"] <= data["budget"]["max_tokens"] for part in ("zh", "en", "all"))
    assert (data["budget"]["reasoning"], data["budget"]["max_tokens"]) == (1280, 2480)
    assert data["languages"] == ["zh", "en"]


@pytest.mark.needs_data(*REAL, "results/lengths-acebench-m7.json")
def test_the_lengths_file_in_results_is_what_the_script_writes_from_the_real_logs():
    text = (REPO / "results" / "lengths-acebench-m7.json").read_text(encoding="utf-8")
    assert json.loads(text) == acebench_lengths.summarize(REPO)
