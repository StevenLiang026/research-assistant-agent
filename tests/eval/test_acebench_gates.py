"""scripts/acebench_gates.py (docs/m8-计划.md WP5): gate G4 as M7 computed it (the archived acebench_g4.py:
the normal answers' parse rate, pooled over both languages, at least B0's less 5 points; the special answers'
phrase rate at least half B0's), the settings of the new run checked against B0's M7 run field by field and
its whole server command but the adapter, the gated adapter checked against WP4's decision
(results/decision-m8.json: GO, its chosen checkpoint's SHA-256), and H3's statistic as scripts/run_test.py
computes it (eval.acebench.per_item and pair_strata, then eval.metrics.stratified_noninferiority_p with
margin 0.02, seed 42 and 10,000 resamples). On fabricated ACEBench raw folders, result files and decisions in
a temporary folder, and on the M7 files when they are here: M7's G4 and H3 numbers come back
(docs/m7-进度.md), with --allow-no-decision, which only that check may use."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from research_agent.eval.acebench import pair_strata, per_item
from research_agent.eval.metrics import stratified_noninferiority_p

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gates = load_script("acebench_gates")

COMMIT = "56dd66cf6439b0d9655ee1b353e4cd745c6f664e"
PATCH_SHA256 = "2ded6928e4bdf648210ad9ae1148b34cf18b4859dd310068f3437c4e59a1faa1"
BUILD = "b11349-fb4b2737a"
BASE = "models/Base-Q8_0.gguf"
BASE_BYTES = b"GGUF the base model"
ADAPTER = "runs/train/ft5/gguf/checkpoint-112-f16.gguf"
LANGUAGES = ["zh", "en"]
GROUPS = {"normal": ["normal_atom_bool", "normal_multi_turn_user_adjust"],
          "special": ["special_incomplete", "special_error_param"]}
NEW, M7 = "results/acebench-m8.json", "results/acebench-m7.json"
B0_RAW, NEW_RAW, M7_FT_RAW = "runs/acebench/m7/b0", "runs/acebench/m8/ft", "runs/acebench/m7/ft"
OUT = "out/gates-acebench-m8.json"


def test_the_default_file_is_not_read_as_an_acebench_result():
    # research_agent.eval.report reads every results/acebench-<name>.json as an ACEBench result
    assert gates.OUT.parent.as_posix() == "results" and not gates.OUT.name.startswith("acebench-")
DECISION = "results/decision-m8.json"  # scripts/decide_m8.py's, the default of --decision
ADAPTER_SHA256 = "a" * 64  # the SHA-256 the run records of its adapter (system below)
CALL, BROKEN = "[Api(a='1')]", "Api(a='1')"  # in ACEBench's format, and not
TOOL = '<tool_call>\n{"name": "Api", "arguments": {"a": "1"}}\n</tool_call>'
MISSING = "Missing necessary parameters (a) for the api (Api)"
WRONG_VALUE = "There is incorrect value (2) for the parameters (a) in the conversation history."
OTHER = "I cannot do that."
# per (language, category) the answers, each (text, right): B0's normal answers parse 10 of 12, its special
# answers hold a phrase 6 of 8
B0_ANSWERS = {
    ("zh", "normal_atom_bool"): [(CALL, True), (CALL, True), (CALL, False), (BROKEN, False)],
    ("zh", "normal_multi_turn_user_adjust"): [(CALL, True), (CALL, True)],
    ("en", "normal_atom_bool"): [(CALL, True), (CALL, True), (CALL, True), (CALL, False)],
    ("en", "normal_multi_turn_user_adjust"): [(CALL, True), (TOOL, False)],
    ("zh", "special_incomplete"): [(MISSING, True), (MISSING, False)],
    ("zh", "special_error_param"): [(WRONG_VALUE, True), (OTHER, False)],
    ("en", "special_incomplete"): [(MISSING, True), (OTHER, False)],
    ("en", "special_error_param"): [(WRONG_VALUE, True), (WRONG_VALUE, True)],
}
# the new run's: parse 10 of 12 (>= 5/6 - 1/20), phrase 4 of 8 (>= 0.5 x 3/4)
NEW_ANSWERS = {
    ("zh", "normal_atom_bool"): [(CALL, True), (CALL, False), (CALL, False), (CALL, True)],
    ("zh", "normal_multi_turn_user_adjust"): [(CALL, True), (TOOL, False)],
    ("en", "normal_atom_bool"): [(CALL, True), (CALL, True), (BROKEN, False), (CALL, True)],
    ("en", "normal_multi_turn_user_adjust"): [(CALL, True), (CALL, True)],
    ("zh", "special_incomplete"): [(MISSING, True), (OTHER, False)],
    ("zh", "special_error_param"): [(WRONG_VALUE, True), (OTHER, False)],
    ("en", "special_incomplete"): [(OTHER, False), (OTHER, False)],
    ("en", "special_error_param"): [(WRONG_VALUE, True), (WRONG_VALUE, False)],
}


def write_raw(folder: str, answers: dict) -> Path:
    """A raw folder as scripts/run_acebench.py copies one: per language and category the result file (one
    answer per line, its id and result) and the score file (ACEBench's header, then a line per wrong
    answer)."""
    root = Path(folder)
    for (language, category), rows in answers.items():
        ids = [f"{category}_{i}" for i in range(len(rows))]
        result = root / language / "result" / f"data_{category}_result.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text("".join(json.dumps({"id": i, "result": text}, ensure_ascii=False) + "\n"
                                  for i, (text, _) in zip(ids, rows, strict=True)), encoding="utf-8")
        wrong = [i for i, (_, right) in zip(ids, rows, strict=True) if not right]
        header = {"accuracy": round((len(rows) - len(wrong)) / len(rows), 3),
                  "correct_count": len(rows) - len(wrong), "total_count": len(rows)}
        score = root / language / "score" / f"data_{category}_score.json"
        score.parent.mkdir(parents=True, exist_ok=True)
        score.write_text("\n".join(json.dumps(row) for row in
                                   [header, *({"id": i, "valid": False, "error": ["x"]} for i in wrong)]),
                         encoding="utf-8")
    return root


def system(raw: str, *, adapter: str | None = None) -> dict:
    """A system's record as scripts/run_acebench.py writes it in think mode (greedy, R 1280, one slot)."""
    command = ["llama-server", "-m", BASE, "-ngl", "99", "-c", "12288", "--jinja", "--host", "127.0.0.1",
               "--port", "8080", "--cache-ram", "0"]
    if adapter is not None:
        command += ["--lora", adapter, "--lora-init-without-apply"]
    command += ["-np", "1", "--no-cache-prompt", "--reasoning-format", "deepseek", "--reasoning-budget",
                "1280"]
    return {"acebench_model": "local-nanbeige4.1-3b-" + ("lora" if adapter else "base"),
            "server_command": command, "server_build": BUILD,
            "adapter": None if adapter is None else {"gguf": adapter, "sha256": ADAPTER_SHA256,
                                                     "selection": None},
            "raw": raw, "probe": {"answer": "[get_weather(city='Paris')]"},
            "aggregate": {"categories": 8, "n": 20, "accuracy": 0.7}, "languages": {}}


def write_decision(decision: str = "GO", sha256: str | None = ADAPTER_SHA256) -> Path:
    """WP4's decision as scripts/decide_m8.py writes it (the fields the gates read), at --decision's
    default."""
    chosen = None if sha256 is None else {"step": 112, "gguf": ADAPTER, "sha256": sha256, "sum": 29}
    out = Path(DECISION)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"decision": decision, "chosen": chosen, "reason": "fixture"}) + "\n",
                   encoding="utf-8")
    return out


def write_result(path: str, systems: dict, **changes) -> Path:
    """A result file as scripts/run_acebench.py writes one in think mode; changes set values by dotted path
    ("acebench.commit", "systems.ft.server_build")."""
    data = {"benchmark": "ACEBench 56dd66c with the local patches", "smoke": False, "languages": LANGUAGES,
            "groups": GROUPS,
            "generation": {"temperature": 0, "top_p": 1, "max_tokens": 2480, "num_threads": 1},
            "mode": "think", "family": {"name": "nanbeige4.1-3b", "thinking": "only",
                                        "folder": "models/Nanbeige4.1-3B"},
            "decoding": {"greedy": True, "temperature": 0, "top_p": 1, "seed": None,
                         "acebench_sends": ["temperature", "top_p", "max_tokens"], "server_sets": [],
                         "rule": "Greedy."},
            "budget": {"reasoning": 1280, "reply": 1200, "max_tokens": 2480, "context": 12288},
            "base_model": BASE,
            "acebench": {"clone": "runs/third_party/ACEBench", "commit": COMMIT,
                         "patch": {"file": "third_party_patches/acebench.diff", "sha256": PATCH_SHA256},
                         "data": {"files": 68, "identical_to": COMMIT, "check": "blob names"}},
            "git_commit": "c" * 40, "git_dirty": False, "systems": systems}
    for dotted, value in changes.items():
        *head, last = dotted.split(".")
        holder = data
        for key in head:
            holder = holder[key]
        holder[last] = value
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return out


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """B0's M7 run (results/acebench-m7.json with b0 and ft, their raw folders), the new run of ft
    (results/acebench-m8.json, runs/acebench/m8/ft), with the base GGUF, and WP4's GO for the run's adapter
    (results/decision-m8.json), in a temporary folder."""
    monkeypatch.chdir(tmp_path)
    Path(BASE).parent.mkdir(parents=True)
    Path(BASE).write_bytes(BASE_BYTES)
    write_raw(B0_RAW, B0_ANSWERS)
    write_raw(M7_FT_RAW, NEW_ANSWERS)
    write_raw(NEW_RAW, NEW_ANSWERS)
    write_result(M7, {"b0": system(B0_RAW), "ft": system(M7_FT_RAW, adapter="runs/train/ft3/x.gguf")})
    write_result(NEW, {"ft": system(NEW_RAW, adapter=ADAPTER)})
    write_decision()
    return tmp_path


BASE_SHA256 = hashlib.sha256(BASE_BYTES).hexdigest()


def arguments(*extra: str, result: str = NEW, b0_raw: str = B0_RAW) -> list[str]:
    return ["--result", result, "--b0-raw", b0_raw, "--b0-result", M7, "--b0-base-sha256", BASE_SHA256,
            "--out", OUT, *extra]


def written() -> dict:
    return json.loads(Path(OUT).read_text(encoding="utf-8"))


def test_g4_and_the_settings_pass_and_h3_is_run_tests_statistic(runs, capsys):
    assert gates.main(arguments()) == 0
    printed = capsys.readouterr().out
    out = written()
    g4 = out["g4"]
    assert g4["counts"]["ft"]["pooled"] == {"normal": 12, "parses": 10, "special": 8, "phrase": 4}
    assert g4["counts"]["b0"]["pooled"] == {"normal": 12, "parses": 10, "special": 8, "phrase": 6}
    assert g4["counts"]["b0"]["zh"] == {"normal": 6, "parses": 5, "special": 4, "phrase": 3}
    assert (g4["a"]["ft"], g4["a"]["b0"]) == (10 / 12, 10 / 12)
    assert g4["a"]["threshold"] == pytest.approx(10 / 12 - 0.05, abs=1e-15)  # 47/60, compared exactly
    assert (g4["b"]["ft"], g4["b"]["b0"], g4["b"]["threshold"]) == (0.5, 0.75, 0.375)
    assert g4["a"]["passed"] and g4["b"]["passed"] and g4["passed"]
    # H3: the stratified paired bootstrap of the new run against B0's raw answers, as run_test computes it
    strata_a, strata_b = pair_strata(per_item(NEW_RAW), per_item(B0_RAW))
    stats = stratified_noninferiority_p(strata_a, strata_b, 0.02, seed=42, n=10000)
    h3 = out["h3"]
    assert (h3["margin"], h3["strata"], h3["items"]) == (0.02, 8, 20)
    assert (h3["diff"], h3["p_ni"], h3["below"]) == (stats["diff"], stats["p_ni"], stats["below"])
    assert h3["paired_ci"] == {key: stats[key] for key in ("diff", "mean", "low", "high", "n", "seed")}
    assert h3["a"]["path"] == NEW_RAW and h3["b"]["path"] == B0_RAW
    assert h3["a"]["sha256"] == per_item(NEW_RAW).sha256 and h3["b"]["items"] == 20
    assert [(row["language"], row["n"]) for row in h3["per_stratum"]] == [  # the strata sorted
        ("en", 4), ("en", 2), ("en", 2), ("en", 2), ("zh", 4), ("zh", 2), ("zh", 2), ("zh", 2)]
    assert out["settings"]["equal"] is True
    assert out["settings"]["fields"]["llama.cpp build"] == BUILD
    assert out["settings"]["fields"]["base GGUF SHA-256"] == BASE_SHA256
    # the whole server command is B0's once the adapter's --lora pair and --lora-init-without-apply are out
    command = out["settings"]["fields"]["server command (without the adapter)"]
    assert command == system(B0_RAW)["server_command"]
    assert out["result"] == {"path": NEW, "sha256": hashlib.sha256(Path(NEW).read_bytes()).hexdigest()}
    assert out["system"] == "ft" and out["b0"]["raw"] == B0_RAW
    # the gated adapter is the one WP4's decision chose (--decision's default, results/decision-m8.json)
    assert out["allow_no_decision"] is False and gates.DECISION == Path(DECISION)
    chosen = {"step": 112, "gguf": ADAPTER, "sha256": ADAPTER_SHA256, "sum": 29}
    digest = hashlib.sha256(Path(DECISION).read_bytes()).hexdigest()
    assert out["decision"] == {"path": DECISION, "sha256": digest, "decision": "GO", "chosen": chosen,
                               "adapter": {"gguf": ADAPTER, "sha256": ADAPTER_SHA256}}
    assert "G4 (a): ft 0.8333 >= b0 0.8333 - 0.05 = 0.7833: pass" in printed
    assert "G4 (b): ft 0.5000 >= 0.5 x b0 0.7500 = 0.3750: pass" in printed
    assert "H3:" in printed and "p_NI" in printed
    assert Path(OUT).read_text(encoding="utf-8").endswith("}\n")


@pytest.mark.parametrize(("changes", "code", "a", "b"), [
    ({("zh", "normal_atom_bool"): [(BROKEN, False), (CALL, False), (CALL, False), (CALL, True)]}, 1, False,
     True),  # 9 of 12 parse: below 5/6 - 1/20
    ({("en", "special_error_param"): [(WRONG_VALUE, True), (OTHER, False)]}, 0, True, True),  # 3/8: on 0.375
    ({("en", "special_error_param"): [(OTHER, False), (OTHER, False)]}, 1, True, False),  # 2/8 < 0.375
], ids=["parse-below", "phrase-on-the-threshold", "phrase-below"])
def test_g4_compares_the_pooled_rates_exactly_and_a_failure_exits_1_with_the_file_written(runs, capsys,
                                                                                        changes, code, a, b):
    write_raw(NEW_RAW, {**NEW_ANSWERS, **changes})
    assert gates.main(arguments()) == code
    g4 = written()["g4"]
    assert (g4["a"]["passed"], g4["b"]["passed"], g4["passed"]) == (a, b, a and b)
    printed = capsys.readouterr().out
    assert ("FAIL" in printed) == (not (a and b))


def test_the_phrases_are_found_as_m7_found_them(runs):
    # case-sensitive substrings of the answer as text (a non-text result is read as its str()), and a normal
    # answer parses when its first outermost [...] is a list of calls (run_acebench.acebench_format)
    answers = {**NEW_ANSWERS,
               ("zh", "special_incomplete"): [("missing necessary parameters (a)", False), (None, False)],
               ("zh", "normal_atom_bool"): [("Here: [Api(a='1')] done", True), ("[1, 2]", False),
                                            (None, False), ("[Api(a='1'), Api(b=2)]", True)]}
    counts = gates.g4_counts(write_raw("runs/other", answers), GROUPS, LANGUAGES)
    assert counts["zh"]["phrase"] == 1 and counts["zh"]["special"] == 4  # only zh special_error_param's
    assert counts["zh"]["parses"] == 3 and counts["zh"]["normal"] == 6
    assert gates.PHRASES == ("Missing necessary parameters", "There is incorrect value",
                             "the limitations of the function")


@pytest.mark.parametrize(("changes", "field"), [
    ({"systems.ft.server_build": "b11350-0000000"}, "llama.cpp build"),
    ({"acebench.commit": "0" * 40}, "ACEBench clone commit"),
    ({"acebench.patch.sha256": "0" * 64}, "ACEBench patch SHA-256"),
    ({"acebench.data.identical_to": "0" * 40}, "ACEBench data"),
    ({"budget.context": 16384}, "context size (budget)"),
    ({"decoding.greedy": False}, "greedy"),
    ({"decoding.temperature": 0.6}, "decoding temperature"),
    ({"generation.max_tokens": 2048}, "ACEBench max_tokens"),
    ({"budget.reasoning": 1024}, "reasoning budget (budget)"),
    ({"mode": "no_think"}, "mode"),
    ({"family.name": "qwen3-hybrid"}, "family"),
    ({"groups": {"normal": ["normal_atom_bool"], "special": GROUPS["special"]}}, "groups"),
], ids=["build", "commit", "patch", "data", "context", "greedy", "temperature", "max-tokens", "budget",
        "mode", "family", "groups"])
def test_a_setting_that_differs_from_b0s_m7_run_is_refused_by_its_name(runs, capsys, changes, field):
    write_result(NEW, {"ft": system(NEW_RAW, adapter=ADAPTER)}, **changes)
    assert gates.main(arguments()) == 2
    printed = capsys.readouterr()
    assert f"{field}:" in printed.out + printed.err and not Path(OUT).exists()


@pytest.mark.parametrize(("flag", "value", "field"), [
    ("-c", "16384", "context size (-c)"), ("--reasoning-budget", "1024", "reasoning budget (server)"),
    ("-np", "2", "slots (-np)")])
def test_the_server_commands_settings_are_compared_too(runs, capsys, flag, value, field):
    record = system(NEW_RAW, adapter=ADAPTER)
    command = record["server_command"]
    command[command.index(flag) + 1] = value
    write_result(NEW, {"ft": record})
    assert gates.main(arguments()) == 2
    assert f"{field}:" in capsys.readouterr().out and not Path(OUT).exists()


@pytest.mark.parametrize(("change", "named"), [
    (lambda c: c.remove("--jinja"), "--jinja"),  # the chat template's handling
    (lambda c: c.__setitem__(c.index("-ngl") + 1, "80"), "-ngl"),  # the GPU offload, which can move numerics
    (lambda c: c.__setitem__(c.index("--cache-ram") + 1, "1024"), "--cache-ram"),
    (lambda c: c.extend(["--flash-attn", "on"]), "--flash-attn"),
    (lambda c: c.__setitem__(0, "llama-server-other"), "llama-server-other"),
    (lambda c: c.append(c.pop(c.index("--jinja"))), "another order"),
], ids=["jinja-dropped", "ngl", "cache-ram", "flag-added", "binary", "order"])
def test_the_whole_server_command_but_the_adapter_must_be_b0s(runs, capsys, change, named):
    record = system(NEW_RAW, adapter=ADAPTER)
    change(record["server_command"])
    write_result(NEW, {"ft": record})
    assert gates.main(arguments()) == 2
    printed = capsys.readouterr().out
    assert "server command" in printed and named in printed and not Path(OUT).exists()


@pytest.mark.parametrize(("case", "named"), [
    ("no-go", "NO-GO"),  # WP4 chose no checkpoint: no ACEBench run is gated
    ("other-checkpoint", "adapter SHA-256"),  # the run is not of the checkpoint WP4 chose
    ("no-decision", "decision-m8.json"),
    ("other-lora", "adapter GGUF"),  # the server loaded another file than the one whose SHA-256 is recorded
    ("no-adapter", "no adapter"),  # the gated system ran on the base model alone
], ids=["no-go", "other-checkpoint", "no-decision", "other-lora", "no-adapter"])
def test_the_gated_adapter_must_be_the_checkpoint_wp4_chose(runs, capsys, case, named):
    if case == "no-go":
        write_decision("NO-GO", sha256=None)
    elif case == "other-checkpoint":
        write_decision(sha256="b" * 64)
    elif case == "no-decision":
        Path(DECISION).unlink()
    elif case == "other-lora":
        record = system(NEW_RAW, adapter=ADAPTER)
        command = record["server_command"]
        command[command.index("--lora") + 1] = "runs/train/ft5/gguf/checkpoint-224-f16.gguf"
        write_result(NEW, {"ft": record})
    else:
        write_result(NEW, {"ft": system(NEW_RAW)})  # B0's command and no adapter
    assert gates.main(arguments()) == 2
    printed = capsys.readouterr().out
    assert named in printed and "FAIL" in printed and not Path(OUT).exists()


def test_only_the_m7_reproduction_check_gates_without_a_decision_and_the_file_says_so(runs, capsys):
    Path(DECISION).unlink()
    m7 = ["--result", M7, "--b0-raw", B0_RAW, "--b0-result", M7, "--b0-base-sha256", BASE_SHA256,
          "--out", OUT]
    assert gates.main([*m7, "--allow-no-decision"]) == 0  # ft of B0's own result file
    out = written()
    assert out["allow_no_decision"] is True and out["decision"] is None
    Path(OUT).unlink()
    # any other run needs the decision, and the two options do not go together
    assert gates.main([*arguments(), "--allow-no-decision"]) == 2
    assert "--allow-no-decision" in capsys.readouterr().out and not Path(OUT).exists()
    assert gates.main([*m7, "--allow-no-decision", "--decision", DECISION]) == 2
    assert "--decision" in capsys.readouterr().out and not Path(OUT).exists()


def test_the_base_gguf_is_compared_by_its_bytes(runs, capsys):
    Path(BASE).write_bytes(BASE_BYTES + b" requantized")  # the new run's base: another file at the same path
    assert gates.main(arguments()) == 2
    printed = capsys.readouterr().out
    assert "base GGUF SHA-256:" in printed and not Path(OUT).exists()
    # a result file that records its base's SHA-256 is taken at its word, the file not read
    Path(BASE).write_bytes(BASE_BYTES)
    write_result(NEW, {"ft": system(NEW_RAW, adapter=ADAPTER)}, base_model_sha256=BASE_SHA256)
    Path(BASE).unlink()
    assert gates.main(arguments()) == 0
    assert written()["settings"]["fields"]["base GGUF SHA-256"] == BASE_SHA256
    # B0's result records none (M7's): the pinned SHA-256 stands for it unless --b0-base-sha256 says otherwise
    Path(OUT).unlink()
    pinned = [arg for arg in arguments() if arg not in ("--b0-base-sha256", BASE_SHA256)]
    assert gates.main(pinned) == 2
    assert f"base GGUF SHA-256: {gates.B0_BASE_SHA256!r} (B0)" in capsys.readouterr().out
    # and one that records it is taken at its word
    write_result(M7, {"b0": system(B0_RAW), "ft": system(M7_FT_RAW, adapter="runs/train/ft3/x.gguf")},
                 base_model_sha256=BASE_SHA256)
    assert gates.main(pinned) == 0


def test_the_pinned_b0_base_is_the_sha256_the_bake_off_recorded_when_the_gguf_was_made():
    smoke = REPO / "runs" / "bakeoff" / "m7" / "c5" / "smoke.json"
    if not smoke.is_file():
        pytest.skip("the bake-off's smoke record of Nanbeige4.1-3B is not here")
    s0 = json.loads(smoke.read_text(encoding="utf-8"))["steps"]["S0"]
    assert s0["gguf"] == "models/Nanbeige4.1-3B-Q8_0.gguf" and s0["gguf_sha256"] == gates.B0_BASE_SHA256


def test_an_existing_output_is_never_overwritten_and_nothing_is_read_then(runs, capsys):
    Path(OUT).parent.mkdir(parents=True)
    Path(OUT).write_text("{}\n", encoding="utf-8")
    assert gates.main(arguments(result="results/none.json")) == 2  # refused before the inputs are read
    assert "exists" in capsys.readouterr().out and Path(OUT).read_text(encoding="utf-8") == "{}\n"


@pytest.mark.parametrize("case", ["no-result", "no-system", "other-b0-raw", "unreadable-raw", "no-b0-result",
                                  "no-special-group"])
def test_missing_or_mismatched_inputs_are_refused_with_exit_2(runs, capsys, case):
    argv = arguments()
    if case == "no-special-group":  # both files alike, so the settings agree: G4 has no special group to read
        groups = {"normal": GROUPS["normal"]}
        write_result(NEW, {"ft": system(NEW_RAW, adapter=ADAPTER)}, groups=groups)
        write_result(M7, {"b0": system(B0_RAW), "ft": system(M7_FT_RAW, adapter="runs/train/ft3/x.gguf")},
                     groups=groups)
    elif case == "no-result":
        argv = arguments(result="results/none.json")
    elif case == "no-system":
        argv = arguments("--system", "b4")
    elif case == "other-b0-raw":
        write_raw("runs/elsewhere/b0", B0_ANSWERS)
        argv = arguments(b0_raw="runs/elsewhere/b0")  # not the raw folder of B0's result
    elif case == "unreadable-raw":
        Path(NEW_RAW, "zh", "result", "data_normal_atom_bool_result.json").unlink()
    else:
        argv = [*argv[:4], "--b0-result", "results/none.json", *argv[6:]]
    assert gates.main(argv) == 2
    assert "FAIL" in capsys.readouterr().out and not Path(OUT).exists()


M7_FILES = (REPO / "results" / "acebench-m7.json", REPO / "runs" / "acebench" / "m7" / "b0",
            REPO / "runs" / "acebench" / "m7" / "ft")


@pytest.mark.skipif(not all(p.exists() for p in M7_FILES), reason="M7's ACEBench result and raw folders are "
                                                                   "not here")
def test_m7s_g4_and_h3_come_back_from_its_own_files(tmp_path, monkeypatch, capsys):
    # results/acebench-m7.json's ft against runs/acebench/m7/b0: docs/m7-进度.md's G4 (parse FT3 0.916 against
    # B0 0.894, phrase 0.823 against 0.853) and H3 (-3.96 points, [-5.83, -2.09], p_NI 0.980). The base GGUF
    # (4 GB) is not hashed here: the settings check reads the pinned SHA-256 for it. FT3's run has no WP4
    # decision, so the check runs with --allow-no-decision, which the file records.
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(gates, "file_sha256", lambda path: gates.B0_BASE_SHA256)
    out = tmp_path / "gates.json"
    code = gates.main(["--result", "results/acebench-m7.json", "--system", "ft", "--b0-raw",
                       "runs/acebench/m7/b0", "--allow-no-decision", "--out", str(out)])
    assert code == 0, capsys.readouterr()
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["allow_no_decision"] is True and result["settings"]["equal"] is True
    g4, h3 = result["g4"], result["h3"]
    assert (round(g4["a"]["ft"], 4), round(g4["a"]["b0"], 4)) == (0.9159, 0.8939)
    assert (round(g4["b"]["ft"], 4), round(g4["b"]["b0"], 4)) == (0.8233, 0.8533)
    assert g4["passed"] is True
    assert round(h3["diff"] * 100, 2) == -3.96 and (h3["strata"], h3["items"]) == (30, 1940)
    ci = h3["paired_ci"]
    assert (round(ci["low"] * 100, 2), round(ci["high"] * 100, 2)) == (-5.83, -2.09)
    assert round(h3["p_ni"], 3) == 0.980
