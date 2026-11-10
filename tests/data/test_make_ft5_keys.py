"""scripts/make_ft5_keys.py (docs/m8-计划.md WP3): FT5's sample keys, every sample of special-mix and a
quarter of FT3's samples drawn per step type with a fixed seed, built as scripts/train.py builds the FT5 mix
(FT3's data files and sidecars with special-mix and its sidecar, under M8's overlap report) and read back
with train.py's own --sample-keys reader; the counts per step type; and the FT5 train.py command, FT3's
recipe from its manifest with the preregistered changes, printed and never run. Offline, over the fake family
(every character a token, so the mixes are built with a --max-length no sample reaches) and fabricated data
files."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from support.fakes import SAMPLE_MESSAGES
from support.family_fakes import FakeFamilyTokenizer, fake_family, model_folder

from research_agent.data.reasoning import ReasoningIndex, sidecar_row, write_sidecar
from research_agent.family.core import load_family

REPO = Path(__file__).resolve().parents[2]
BIG = 10**6


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


make = load_script("make_ft5_keys")
train = load_script("train")

INIT = "runs/train/ft3/trainer/checkpoint-1635"
WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Weather of a city.",
                                                 "parameters": {"type": "object", "properties": {
                                                     "city": {"type": "string"}}, "required": ["city"]}}}


def weather_call(city: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": city}}}]}


def weather(city: str) -> dict:
    """A public sample of two steps: a call (step 1, "call"), then the answer after its result
    ("next_step")."""
    return {"messages": [{"role": "user", "content": f"What is the weather in {city}?"}, weather_call(city),
                         {"role": "tool", "content": '{"temperature_c": 18}'},
                         {"role": "assistant", "content": f"It is 18 degrees in {city}."}],
            "tools": [WEATHER_TOOL]}


def refusal(i: int) -> dict:
    """A special-mix record that answers with a sentence instead of a call (step 1 without a call:
    "no_call")."""
    return {"id": f"sm-refuse-{i}", "tools": [WEATHER_TOOL],
            "messages": [{"role": "system", "content": "Answer with a call, or with the sentence asked for."},
                         {"role": "user", "content": f"Weather for the city number {i}?"},
                         {"role": "assistant", "content": f"The value 'number {i}' of city is not valid."}]}


def valid(i: int) -> dict:
    return {"id": f"sm-call-{i}", "tools": [WEATHER_TOOL],
            "messages": [{"role": "user", "content": f"Weather in Oslo-{i}, please."},
                         weather_call(f"Oslo-{i}")]}


def posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


def report_for(path: Path, files: list[Path]) -> Path:
    entries = [{"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                "records_overlapping": []} for f in files]
    path.write_text(json.dumps({"files": entries}), encoding="utf-8")
    return path


def hook(model_dir: str, name: str | None):
    tokenizer = FakeFamilyTokenizer("hybrid")
    return load_family(model_dir, name=name, tokenizer=tokenizer, families=[fake_family(tokenizer)])


def ft3_run(tmp_path: Path) -> dict:
    """FT3 as its manifest records it: two data files (public.jsonl, eight histories of a call then two calls
    at once, 8 call and 8 parallel steps; weather.jsonl, six of a call then the answer, 6 call and 6
    next_step), a sidecar for one turn, its own overlap report, the recipe's flags (QLoRA, the save ratio),
    and the manifest train.py writes: its args (recorded_args), the per-file accounts and the family. Then
    special-mix (four refusals, two calls) with a sidecar for every turn, and M8's overlap report over all
    three files."""
    public = write_jsonl(tmp_path / "public.jsonl", [{"messages": SAMPLE_MESSAGES, "tools": []}] * 8)
    cities = ("Paris", "Rome", "Lima", "Oslo", "Kyiv", "Pune")
    weather_file = write_jsonl(tmp_path / "weather.jsonl", [weather(city) for city in cities])
    teacher = tmp_path / "teacher.jsonl"
    write_sidecar([sidecar_row("public:0", 1, SAMPLE_MESSAGES[4], source="teacher", reasoning="Fetch both.",
                               model="deepseek-flash", cache_key="a" * 64, context_equal=True)], teacher)
    model = model_folder(tmp_path / "model")
    ft3_report = report_for(tmp_path / "overlap-ft3.json", [public, weather_file])
    argv = [str(public), str(weather_file), "--out", "runs/train/ft3", "--model", str(model), "--reasoning",
            str(teacher), "--max-length", str(BIG), "--qlora", "--save-steps", "0.125", "--overlap-report",
            str(ft3_report)]
    args = train.build_parser().parse_args(argv)
    family = hook(str(model), None)
    samples, _, per_file = train.assemble([public, weather_file], family, BIG,
                                          reasoning=ReasoningIndex([teacher]))
    manifest = {"config": {"family": family.record(), "args": train.recorded_args(args),
                           "data": {"files": per_file, "samples": len(samples),
                                    "reasoning": {"files": train.sidecar_files([str(teacher)])},
                                    "sample_keys": None}},
                "data_files": [{"path": str(p), "sha256": train.sha256_of(p)}
                               for p in (public, weather_file)]}
    path = tmp_path / "ft3-manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    special = write_jsonl(tmp_path / "special-mix.jsonl", [*(refusal(i) for i in range(4)),
                                                          *(valid(i) for i in range(2))])
    special_sidecar = tmp_path / "reasoning-special-mix.jsonl"
    records = [*(refusal(i) for i in range(4)), *(valid(i) for i in range(2))]
    write_sidecar([sidecar_row(r["id"], 0, r["messages"][-1], source="teacher",
                               reasoning=f"Reasoning for {r['id']}.", model="deepseek-flash",
                               cache_key="b" * 64, context_equal=True) for r in records], special_sidecar)
    report = report_for(tmp_path / "overlap-m8.json", [public, weather_file, special])
    return {"manifest": path, "args": train.recorded_args(args), "data": [public, weather_file],
            "teacher": teacher, "special": special, "special_sidecar": special_sidecar, "report": report,
            "keys": tmp_path / "ft5" / "ft5-sample-keys.jsonl", "model": model}


def arguments(run: dict, *extra: str) -> list[str]:
    return ["--ft3-manifest", str(run["manifest"]), "--special", str(run["special"]), "--special-reasoning",
            str(run["special_sidecar"]), "--overlap-report", str(run["report"]), "--init-adapter", INIT,
            "--out", str(run["keys"]), *extra]


def main(argv: list[str], capsys) -> tuple[int, str, str, dict | None]:
    code = make.main(argv, load_family=hook)
    out, err = capsys.readouterr()
    last = out.splitlines()[-1] if out.strip() else ""
    summary = json.loads(last.removeprefix("summary: ")) if last.startswith("summary: ") else None
    return code, out, err, summary


def ft5_samples(run: dict) -> list[dict]:
    """The samples train.py builds for FT5: FT3's files and special-mix, FT3's sidecar and special-mix's."""
    files = [*run["data"], run["special"]]
    skips, _ = train.overlap_gate(run["report"], files, check=True)
    samples, _, _ = train.assemble(files, hook(str(run["model"]), None), BIG, skips=skips,
                                   reasoning=ReasoningIndex([run["teacher"], run["special_sidecar"]]))
    return samples


def test_every_special_mix_sample_and_a_quarter_of_ft3s_per_step_type_are_listed(tmp_path, capsys):
    run = ft3_run(tmp_path)
    code, out, err, summary = main(arguments(run), capsys)
    assert code == 0, err
    # train.py's own reader takes the file over the FT5 mix as train.py builds it
    samples = ft5_samples(run)
    selected = train.select_samples(samples, train.load_sample_keys(run["keys"]))
    keys = [tuple(s["key"]) for s in selected]
    special = [tuple(s["key"]) for s in samples if Path(s["file"]) == run["special"]]
    assert len(special) == 6 and set(special) <= set(keys)  # every special-mix sample
    assert all(s["mode"] == "think" for s in selected if Path(s["file"]) == run["special"])
    # 14 call, 8 parallel and 6 next_step samples of FT3: a quarter of each, rounded half up
    assert summary["strata"] == {
        "call": {"ft3_built": 14, "ft3_listed": 4, "special": 2, "listed": 6},
        "next_step": {"ft3_built": 6, "ft3_listed": 2, "special": 0, "listed": 2},
        "no_call": {"ft3_built": 0, "ft3_listed": 0, "special": 4, "listed": 4},
        "parallel": {"ft3_built": 8, "ft3_listed": 2, "special": 0, "listed": 2}}
    assert summary["listed"] == len(keys) == 14 and (summary["share"], summary["seed"]) == (0.25, 42)
    assert keys == sorted(keys, key=[tuple(s["key"]) for s in samples].index)  # in the mix's order
    assert summary["keys"] == {"path": posix(run["keys"]), "sha256": train.sha256_of(run["keys"]),
                               "listed": 14, "written": True}
    assert summary["ft3_check"] == []  # FT3's files build as FT3's manifest recorded them
    for line in ("call", "parallel", "next_step", "no_call", "total"):
        assert any(row.split()[:1] == [line] for row in out.splitlines()), line
    assert run["keys"].read_bytes() == "".join(json.dumps({"id": i, "k": k}, ensure_ascii=False) + "\n"
                                               for i, k in keys).encode("utf-8")


def test_the_draw_is_fixed_by_the_seed_and_a_rerun_writes_nothing_new(tmp_path, capsys):
    run = ft3_run(tmp_path)
    assert main(arguments(run), capsys)[0] == 0
    first = run["keys"].read_bytes()
    code, out, _, summary = main(arguments(run), capsys)
    assert code == 0 and summary["keys"]["written"] is False and run["keys"].read_bytes() == first
    assert "unchanged" in out
    other = tmp_path / "other-keys.jsonl"
    code, _, _, summary = main([*arguments(run)[:-2], "--out", str(other), "--seed", "7"], capsys)
    assert code == 0 and summary["strata"]["call"]["ft3_listed"] == 4  # the same quotas, another draw
    draws = {make.stratified_share([{"key": ("r", i), "type": "call"} for i in range(40)], 0.25, seed)
             .__repr__() for seed in (1, 2, 3)}
    assert len(draws) == 3  # the seed decides the draw
    assert make.stratified_share([{"key": ("r", i), "type": "call"} for i in range(40)], 0.25, 1) == \
        make.stratified_share([{"key": ("r", i), "type": "call"} for i in range(40)], 0.25, 1)


def test_a_stratums_quota_is_its_share_rounded_half_up():
    assert [make.quota(n, 0.25) for n in (0, 1, 2, 3, 4, 6, 14, 20907)] == [0, 0, 1, 1, 1, 2, 4, 5227]
    assert make.quota(10, 1.0) == 10


def test_the_ft5_command_is_ft3s_recipe_with_the_preregistered_changes(tmp_path, capsys):
    run = ft3_run(tmp_path)
    code, out, _, summary = main(arguments(run), capsys)
    assert code == 0
    parsed = train.build_parser().parse_args(summary["command"])
    assert parsed.data == [*map(str, run["data"]), str(run["special"])]
    assert parsed.reasoning == [str(run["teacher"]), str(run["special_sidecar"])]
    assert (parsed.init_adapter, parsed.sample_keys) == (INIT, posix(run["keys"]))
    assert parsed.reasoning_budget == 1280
    assert (parsed.lr, parsed.epochs, parsed.save_steps, parsed.out) == (3e-5, 1, 0.25, "runs/train/ft5")
    assert parsed.overlap_report == str(run["report"])
    changed = {"data", "reasoning", "out", "lr", "epochs", "save_steps", "overlap_report", "sample_keys"}
    for key, value in run["args"].items():  # everything else is FT3's
        if key not in changed:
            assert getattr(parsed, key) == value, key
    assert parsed.qlora is True and parsed.max_length == BIG
    assert ".venv/Scripts/python.exe scripts/train.py " + " ".join(summary["command"]) in out
    assert "not run" in out


def test_the_docstring_holds_the_ft5_command_derived_from_ft3s_own_manifest():
    path = REPO / "runs" / "train" / "ft3" / "manifest.json"
    if not path.is_file():
        pytest.skip("FT3's manifest is not here")
    args = json.loads(path.read_text(encoding="utf-8"))["config"]["args"]
    inputs = {"special": make.SPECIAL, "special_reasoning": make.SPECIAL_REASONING,
              "overlap_report": make.OVERLAP_REPORT, "keys": posix(make.OUT),
              "init_adapter": make.INIT_ADAPTER}
    argv = make.ft5_command(args, **inputs)
    make.check_command(argv, args, **inputs)  # FT3's recipe with the changes, as train.py's parser reads it
    assert ("scripts/train.py " + " ".join(argv)) in " ".join(make.__doc__.split())


def test_ft3s_files_that_build_otherwise_than_its_manifest_says_are_reported(tmp_path, capsys):
    run = ft3_run(tmp_path)
    manifest = json.loads(run["manifest"].read_text(encoding="utf-8"))
    manifest["config"]["data"]["files"][0]["samples"] = 99  # as if FT3 had built more
    run["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    code, out, _, summary = main(arguments(run), capsys)
    assert code == 0 and len(summary["ft3_check"]) == 1 and "samples 99 -> 16" in summary["ft3_check"][0]
    assert "FT3's files" in out


def test_a_keys_file_that_would_change_is_kept_unless_forced(tmp_path, capsys):
    run = ft3_run(tmp_path)
    run["keys"].parent.mkdir(parents=True)
    run["keys"].write_text(json.dumps({"id": "public:0", "k": 0}) + "\n", encoding="utf-8")
    code, _, err, _ = main(arguments(run), capsys)
    assert code == 2 and "--force" in err
    assert run["keys"].read_text(encoding="utf-8") == json.dumps({"id": "public:0", "k": 0}) + "\n"
    code, _, _, summary = main([*arguments(run), "--force"], capsys)
    assert code == 0 and summary["keys"]["written"] is True and summary["keys"]["listed"] == 14


def test_a_dry_run_writes_nothing(tmp_path, capsys):
    run = ft3_run(tmp_path)
    code, out, _, summary = main([*arguments(run), "--dry-run"], capsys)
    assert code == 0 and not run["keys"].exists() and summary["keys"]["written"] is False
    assert summary["listed"] == 14 and "dry run" in out


@pytest.mark.parametrize("case", ["ft3-keys", "no-special", "no-sidecar", "special-is-ft3s", "share"])
def test_inputs_that_cannot_give_ft5s_keys_are_refused_with_exit_2(tmp_path, capsys, case):
    run = ft3_run(tmp_path)
    argv = arguments(run)
    if case == "ft3-keys":  # FT3 trained on all its samples: a manifest with sample keys is another run
        manifest = json.loads(run["manifest"].read_text(encoding="utf-8"))
        manifest["config"]["args"]["sample_keys"] = str(run["keys"])
        run["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    elif case == "no-special":
        run["special"].unlink()
    elif case == "no-sidecar":
        run["special_sidecar"].unlink()
    elif case == "special-is-ft3s":
        argv = [*argv[:2], "--special", str(run["data"][1]), *argv[4:]]
    else:
        argv = [*argv, "--share", "1.5"]
    code, _, err, _ = main(argv, capsys)
    assert code == 2 and "FAIL" in err and not run["keys"].exists()


def test_a_special_mix_the_overlap_report_does_not_cover_stops_as_train_py_stops(tmp_path, capsys):
    run = ft3_run(tmp_path)
    report_for(run["report"], run["data"])  # M8's report without special-mix
    code, _, err, _ = main(arguments(run), capsys)
    assert code == 4 and "check_benchmark_overlap" in err and not run["keys"].exists()
