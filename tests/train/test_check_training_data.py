"""scripts/check_training_data.py (docs/m7-本地提升计划.md, WP6 样本层面的核对): the samples of a data mix
built as scripts/train.py builds them and keyed as its --sample-keys keys them, grouped by the hash of their
NO_THINK prompt within the dedup scope; duplicates, conflicts (an unresolved one of synthetic or changed
samples fails the check), the FT4 changes files and how they match the samples built, the cap on new samples
(copies that move a before-mix turn are not new), the sample-keys file (read back with train.py's own
loader), a manifest's own sample keys and the before/after table. Offline, over the fake family of
tests/train/family_stub.py: every character is a token there, so the mixes are built with a --max-length no
sample reaches."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from family_stub import family_hook, loaded_fake_family
from support.fakes import SAMPLE_MESSAGES, call
from support.family_fakes import FakeFamilyTokenizer, model_folder
from test_train_family_samples import REASONING, sidecar
from test_train_samples import LONG, PUBLIC_SAMPLE

from research_agent.agent.context import step_contexts
from research_agent.agent.prompts import PROMPT_HASH
from research_agent.data.reasoning import ReasoningIndex, answer_sha256
from research_agent.family.core import Mode
from research_agent.glossary import NOTE_PREFIX
from research_agent.tools import schemas

REPO = Path(__file__).resolve().parents[2]
BIG = 10**6  # a --max-length no fake-family sample reaches (the fake reads every character as a token)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the script's dataclasses resolve their postponed annotations through it
    spec.loader.exec_module(module)
    return module


check = load_script("check_training_data")
train = load_script("train")

SYSTEM, REQUEST, SEARCH = SAMPLE_MESSAGES[:3]
OTHER_SEARCH = {"role": "assistant", "content": "",
                "tool_calls": [call("search_arxiv", "c1", query="draft models").to_api()]}
NOTE = "术语对照:推测解码 = speculative decoding"
ALLOW = "--allow-unresolved"  # the base mix's conflict (c) has no changed version: allowed where beside it
# PUBLIC_SAMPLE answered otherwise at its last step: one prompt, another target
PUBLIC_OTHERWISE = {**PUBLIC_SAMPLE, "messages": [*PUBLIC_SAMPLE["messages"][:3],
                                                  {"role": "assistant", "content": "Overcast, 18 degrees."}]}


def agent(record_id: str, messages: list[dict]) -> dict:
    """An agent-history record as the synthetic files hold them."""
    return {"id": record_id, "history": "agent", "messages": messages, "tools": schemas.MODEL_TOOLS,
            "schema_hash": schemas.SCHEMA_HASH, "prompt_hash": PROMPT_HASH}


def derived(record_id: str, messages: list[dict], context: int) -> dict:
    """A record scripts/transform_synth.py --ft4 derives: a cut copy whose first context assistant turns only
    show the context of the turns after them (train.samples: Context turns)."""
    return {**agent(record_id, messages), "context_turns": context}


def asked(text: str) -> list[dict]:
    """LONG with another request: four steps whose prompts no other fixture shares."""
    return [SYSTEM, {**REQUEST, "content": text}, *LONG[2:]]


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


def report_for(path: Path, files: list[Path], skipped: dict[Path, list[int]] | None = None) -> Path:
    """A benchmark-overlap report listing the files at their SHA-256, naming the records to skip."""
    entries = [{"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                "records_overlapping": [{"index": i} for i in (skipped or {}).get(f, [])]} for f in files]
    path.write_text(json.dumps({"files": entries}), encoding="utf-8")
    return path


def change(record_id: str, k: int, kind: str, *, target: bool = True, rule: str = "R8",
           source: list | None = None) -> dict:
    return {"id": record_id, "k": k, "rule": rule, "kind": kind, "target": target, "from": source}


def write_changes(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def write_keys(path: Path, keys: list[tuple[str, int]]) -> Path:
    path.write_text("".join(json.dumps({"id": i, "k": k}) + "\n" for i, k in keys), encoding="utf-8")
    return path


def base_mix(tmp_path: Path) -> list[str]:
    """The arguments of a two-file mix. synth.jsonl: a (LONG, four steps); b, a's first step again; c, a's
    first context with another search (a conflict without a changed version); d, a's first step with a
    glossary note in its system message. public.jsonl: one public sample twice. An overlap report lists both;
    the model folder is a fake one."""
    synth = write_jsonl(tmp_path / "synth.jsonl", [
        agent("a", LONG), agent("b", SAMPLE_MESSAGES[:3]), agent("c", [SYSTEM, REQUEST, OTHER_SEARCH]),
        agent("d", [{**SYSTEM, "content": SYSTEM["content"] + "\n" + NOTE}, REQUEST, SEARCH])])
    public = write_jsonl(tmp_path / "public.jsonl", [PUBLIC_SAMPLE, PUBLIC_SAMPLE])
    report = report_for(tmp_path / "overlap.json", [synth, public])
    model = model_folder(tmp_path / "model")
    return [str(synth), str(public), "--model", str(model), "--overlap-report", str(report), "--max-length",
            str(BIG)]


def changed_mix(tmp_path: Path, name: str = "synth.jsonl") -> tuple[Path, Path]:
    """A synthetic file and its changes as FT4 writes them: c relabels a's first step (another search); e is
    derived from a, its first turn context only (a row with target false) and its second, a's second turn, a
    copy that is a target; f is untouched (four steps of its own) and n is new."""
    synth = write_jsonl(tmp_path / name, [
        agent("a", LONG), agent("c", [SYSTEM, REQUEST, OTHER_SEARCH]), derived("e", SAMPLE_MESSAGES[:5], 1),
        agent("f", asked("另一个请求")), agent("n", [SYSTEM, {**REQUEST, "content": "新的请求"}, SEARCH])])
    changes = write_changes(tmp_path / "changes.jsonl", [
        change("c", 0, "relabel", source=["a", 0]),
        change("e", 0, "copy", target=False, rule="R10", source=["a", 0]),
        change("e", 1, "copy", rule="R10", source=["a", 1]),
        change("n", 0, "new", rule="R14")])
    return synth, changes


def changed_argv(tmp_path: Path, synth: Path, changes: Path, *extra) -> list:
    return [synth, "--model", model_folder(tmp_path / "model"), "--max-length", BIG, "--overlap-report",
            report_for(tmp_path / "overlap.json", [synth]), "--changes", changes, *extra]


def run(argv: list, capsys, hook=None) -> tuple[int, str, str, dict | None]:
    """main's exit code, stdout, stderr and its last stdout line as the summary (None when there is none)."""
    code = check.main([str(a) for a in argv], load_family=hook or family_hook())
    out, err = capsys.readouterr()
    lines = out.splitlines()
    last = lines[-1] if lines else ""
    summary = json.loads(last.removeprefix("summary: ")) if last.startswith("summary: ") else None
    return code, out, err, summary


def train_manifest(path: Path, data: list[Path], *, model: Path, report: Path, reasoning: tuple = (),
                   glossary: Path | None = None, sample_keys: Path | None = None) -> Path:
    """What scripts/train.py's manifest records of a mix that the check reads back: config.args (the
    glossary flags only with a glossary; sample_keys always), the per-file accounts of train.assemble, the
    samples the run trained on (its sample keys applied, as train.prepare applies them) and the keys file,
    the sidecars, the glossary file, the family record and the data files' SHA-256."""
    family = loaded_fake_family(model)
    skips, _ = train.overlap_gate(report, data, check=True)
    index = ReasoningIndex(reasoning) if reasoning else None
    notes = None if glossary is None else train.GlossaryNotes(train.load_glossary(glossary), 0.5)
    samples, _, per_file = train.assemble(data, family, BIG, skips=skips, reasoning=index, refused=[],
                                          notes=notes)
    keys_info = None
    if sample_keys is not None:
        keys = train.load_sample_keys(sample_keys)
        samples = train.select_samples(samples, keys)
        keys_info = {"path": str(sample_keys), "sha256": train.sha256_of(sample_keys), "listed": len(keys),
                     "selected": len(samples)}
    args = {"data": [str(p) for p in data], "model": str(model), "family": None,
            "reasoning": [str(p) for p in reasoning], "no_think": False, "max_length": BIG,
            "overlap_report": str(report), "no_overlap_check": False, "out": "runs/train/x", "epochs": 2,
            "sample_keys": None if sample_keys is None else str(sample_keys)}
    sidecars = {"files": train.sidecar_files([str(p) for p in reasoning])}
    recorded = {"files": per_file, "samples": len(samples), "reasoning": sidecars, "sample_keys": keys_info}
    if glossary is not None:
        args |= {"glossary": str(glossary), "glossary_dropout": 0.5}
        recorded["glossary"] = {"path": str(glossary), "sha256": train.sha256_of(glossary), "dropout": 0.5}
    manifest = {"config": {"family": family.record(), "args": args, "data": recorded},
                "data_files": [{"path": str(p), "sha256": train.sha256_of(p)} for p in data]}
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_the_samples_are_train_pys_own_keyed_by_record_name_and_turn(tmp_path):
    argv = base_mix(tmp_path)
    teacher = tmp_path / "teacher.jsonl"
    sidecar(teacher, [("a", 1, LONG[4], REASONING)])
    spec = check.spec_from_args(check.build_parser().parse_args([*argv, "--reasoning", str(teacher)]))
    mix = check.build_mix(spec, family_hook())
    family = loaded_fake_family(tmp_path / "model")
    files = [tmp_path / "synth.jsonl", tmp_path / "public.jsonl"]
    expected, dropped, per_file = train.assemble(files, family, BIG, reasoning=ReasoningIndex([teacher]))
    assert [(s["key"], s["file"], s["mode"], s["think_fallback"], s["tokens"]) for s in mix.samples] == [
        (tuple(s["key"]), s["file"], s["mode"], s["think_fallback"], s["tokens"]) for s in expected]
    assert mix.per_file == per_file and mix.dropped == dropped
    assert mix.built is mix.samples and mix.selection is None and mix.dropped_keys == []  # no keys, no drop
    assert [s["key"] for s in mix.samples] == [
        ("a", 0), ("a", 1), ("a", 2), ("a", 3), ("b", 0), ("c", 0), ("d", 0),
        ("public:0", 0), ("public:0", 1), ("public:1", 0), ("public:1", 1)]  # no id: the file stem and index
    # the step type is the rule that types the dev items, over the step's context, target and number
    assert [s["type"] for s in mix.samples] == ["call", "parallel", "next_step", "next_step", "call", "call",
                                               "call", "call", "next_step", "call", "next_step"]
    assert [s["synthetic"] for s in mix.samples] == [True] * 7 + [False] * 4
    think = mix.samples[1]  # a THINK sample: its prompt hash is still the NO_THINK rendering's
    context, target = step_contexts(LONG)[1]
    prompt = family.prompt(family.to_template_messages(context), schemas.MODEL_TOOLS, Mode.NO_THINK)
    assert think["mode"] == "think" and think["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert think["answer_sha256"] == answer_sha256(target)
    assert not {"input_ids", "prompt", "completion"} & think.keys()  # only what the check needs is kept


def kept_and_dropped_ids(dropout: float) -> tuple[str, str]:
    """A record id that keeps its glossary notes at the dropout and one that does not (WP4's hash rule)."""
    from research_agent.train.samples import keeps_note

    ids = [f"r{n}" for n in range(64)]
    return next(i for i in ids if keeps_note(i, dropout)), next(i for i in ids if not keeps_note(i, dropout))


def test_glossary_notes_are_built_as_train_py_builds_them_and_the_prompt_hash_ignores_them(tmp_path, capsys):
    noted, plain = kept_and_dropped_ids(0.5)
    request = {**REQUEST, "content": "帮我找关于推测解码的论文"}
    synth = write_jsonl(tmp_path / "synth.jsonl", [agent(noted, [SYSTEM, request, SEARCH]),
                                                   agent(plain, [SYSTEM, request, SEARCH])])
    glossary = tmp_path / "zh-en.json"
    glossary.write_text(json.dumps({"推测解码": "speculative decoding"}, ensure_ascii=False),
                        encoding="utf-8")
    model, report = model_folder(tmp_path / "model"), report_for(tmp_path / "overlap.json", [synth])
    argv = [str(a) for a in (synth, "--model", model, "--max-length", BIG, "--overlap-report", report,
                             "--glossary", glossary)]
    mix = check.build_mix(check.spec_from_args(check.build_parser().parse_args(argv)), family_hook())
    notes = train.GlossaryNotes(train.load_glossary(glossary), 0.5)  # train.py's default dropout
    expected, _, per_file = train.assemble([synth], loaded_fake_family(model), BIG, notes=notes)
    assert [(s["key"], s["tokens"]) for s in mix.samples] == [(tuple(s["key"]), s["tokens"])
                                                              for s in expected]
    assert mix.per_file == per_file and [s["noted"] for s in mix.samples] == [True, False]
    assert expected[0]["tokens"] > expected[1]["tokens"]  # the note is in the first prompt only
    code, out, err, summary = run(argv, capsys)
    assert code == 0, err
    assert summary["duplicates"] == {"groups": 1, "samples": 2, "completions_differ": 0}  # the note aside
    assert summary["before"]["noted"] == 1 and "notes on 1 sample(s)" in out
    code, _, err, summary = run([*argv, "--glossary-dropout", "0"], capsys)  # every record keeps its notes
    assert code == 0 and summary["before"]["noted"] == 2, err
    manifest = train_manifest(tmp_path / "manifest.json", [synth], model=model, report=report,
                              glossary=glossary)
    code, out, err, summary = run(["--manifest", manifest], capsys)
    assert code == 0 and "manifest check: equal" in out and summary["before"]["noted"] == 1, err
    code, _, err, _ = run(["--manifest", manifest, "--glossary", glossary], capsys)
    assert code == 2 and "not both" in err
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"term": "no Han"}), encoding="utf-8")
    code, _, err, _ = run([*argv[:-1], bad], capsys)
    assert code == 2 and "no Chinese character" in err
    code, _, err, _ = run([*argv, "--glossary-dropout", "1.5"], capsys)
    assert code == 2 and "dropout" in err


def test_the_prompt_hash_ignores_a_glossary_note_at_the_end_of_the_system_message(tmp_path):
    family = loaded_fake_family(tmp_path / "model")
    context = [SYSTEM, REQUEST]
    noted = [{**SYSTEM, "content": SYSTEM["content"] + "\n" + NOTE}, REQUEST]
    assert check.without_glossary_note(noted) == context and noted[0]["content"].endswith(NOTE)
    assert check.without_glossary_note(context) is context
    inside = [{**SYSTEM, "content": NOTE + "\nsystem prompt"}, REQUEST]  # only a last line is a note
    assert check.without_glossary_note(inside) is inside
    # a note starts with research_agent.glossary's own NOTE_PREFIX, colon included; another line is no note
    table = [{**SYSTEM, "content": SYSTEM["content"] + "\n术语对照表见附录"}, REQUEST]
    assert check.without_glossary_note(table) is table
    assert check.NOTE_PREFIX is NOTE_PREFIX
    plain = check.prompt_sha256(family, context, schemas.MODEL_TOOLS)
    assert check.prompt_sha256(family, noted, schemas.MODEL_TOOLS) == plain
    assert check.prompt_sha256(family, inside, schemas.MODEL_TOOLS) != plain
    assert check.prompt_sha256(family, table, schemas.MODEL_TOOLS) != plain
    assert check.prompt_sha256(family, [SYSTEM, {**REQUEST, "content": "另一个请求"}],
                               schemas.MODEL_TOOLS) != plain
    assert check.prompt_sha256(family, context, None) != plain  # the tools are part of the rendering


def test_duplicates_keep_their_first_sample_and_a_conflict_without_a_changed_version_keeps_each_target(
        tmp_path, capsys):
    keys = tmp_path / "keys.jsonl"
    code, out, err, summary = run([*base_mix(tmp_path), "--dedup-scope", "all", ALLOW, "--keys-out", keys],
                                  capsys)
    assert code == 0, err
    # a's first prompt holds a, b, d (one target; d's glossary note ignored) and c (another target)
    assert summary["prompts"] == {"distinct": 6, "shared": 3, "mixed": 0}
    assert summary["duplicates"] == {"groups": 2, "samples": 4,  # the public sample's two steps, twice
                                     "completions_differ": 0}
    assert summary["conflicts"] == {"groups": 1, "targets": 1, "modes": 0, "resolved": 0, "unresolved": 1,
                                    "flagged": 1, "samples": 4}
    assert summary["excluded"] == {"conflict": 0, "duplicate": 4}
    after = summary["after"]
    assert (after["built"], after["selected"], after["samples"], after["synthetic"]) == (11, 11, 7, 5)
    listed = [json.loads(line) for line in keys.read_text(encoding="utf-8").splitlines()]
    assert [(r["id"], r["k"]) for r in listed] == [("a", 0), ("a", 1), ("a", 2), ("a", 3), ("c", 0),
                                                   ("public:0", 0), ("public:0", 1)]
    assert "conflicts: 1 prompt" in out and "unresolved 1" in out
    # where the samples of the duplicate and the conflict prompts are
    synth, public = (tmp_path / "synth.jsonl").as_posix(), (tmp_path / "public.jsonl").as_posix()
    assert summary["grouped_files"] == {"duplicates": {public: 4}, "conflicts": {synth: 4}}
    assert f"  samples by file: {public} 4" in out and f"  samples by file: {synth} 4" in out


def test_the_dedup_scope_is_the_synthetic_samples_unless_all_is_asked_for(tmp_path, capsys):
    argv = base_mix(tmp_path)
    twin = write_jsonl(tmp_path / "twin.jsonl",  # a public sample of a's first step
                       [{"messages": SAMPLE_MESSAGES[:3], "tools": schemas.MODEL_TOOLS}])
    argv = [argv[0], argv[1], twin, *argv[2:4], "--overlap-report",
            report_for(tmp_path / "overlap.json", [Path(argv[0]), Path(argv[1]), twin]), *argv[6:], ALLOW]
    code, out, err, summary = run(argv, capsys)
    assert code == 0, err
    # only the synthetic samples are grouped: the public duplicates and a's public twin stay, as FT3 trained
    assert summary["dedup_scope"] == "synthetic"
    assert summary["prompts"] == {"distinct": 4, "shared": 1, "mixed": 1}  # the twin shares a's first prompt
    assert summary["duplicates"]["groups"] == 0 and summary["conflicts"]["groups"] == 1
    assert summary["excluded"] == {"conflict": 0, "duplicate": 2}  # b and d repeat a in its conflict
    assert (summary["after"]["built"], summary["after"]["samples"]) == (12, 10)
    assert summary["scopes"] == {"synthetic": {"conflict": 0, "duplicate": 2},
                                 "all": {"conflict": 0, "duplicate": 5}}
    assert "dedup scope: synthetic" in out and "the scope all would leave out 5" in out
    assert "1 held by synthetic and other samples" in out
    code, out, err, summary = run([*argv, "--dedup-scope", "all"], capsys)
    assert code == 0, err
    assert summary["dedup_scope"] == "all" and summary["excluded"] == {"conflict": 0, "duplicate": 5}
    assert summary["after"]["samples"] == 7 and summary["prompts"]["mixed"] == 1
    assert "dedup scope: all" in out and "the scope synthetic would leave out 2" in out


def test_an_unresolved_conflict_of_synthetic_or_changed_samples_fails_unless_allowed(tmp_path, capsys):
    keys = tmp_path / "keys.jsonl"
    argv = [*base_mix(tmp_path), "--keys-out", keys]
    code, out, err, summary = run(argv, capsys)  # c's other search for a's first prompt, no changed version
    assert code == 2 and not keys.exists() and not list(tmp_path.glob("keys.jsonl*"))
    assert summary["conflicts"]["unresolved"] == 1 and summary["conflicts"]["flagged"] == 1
    assert summary["flagged_conflicts"] == [[["a", 0], ["b", 0], ["c", 0], ["d", 0]]]
    assert "1 unresolved conflict(s) of synthetic or changed samples" in err and ALLOW in err
    assert "('a', 0), ('b', 0), ('c', 0), ('d', 0)" in err
    code, out, err, summary = run([*argv, ALLOW], capsys)
    assert code == 0 and keys.is_file(), err
    assert summary["conflicts"]["flagged"] == 1 and "of them with a synthetic or changed sample 1" in out
    # a conflict among public samples only is FT3's own (glaive): reported, not failed on
    public = write_jsonl(tmp_path / "public.jsonl", [PUBLIC_SAMPLE, PUBLIC_OTHERWISE])
    plain = [public, "--model", model_folder(tmp_path / "model"), "--max-length", BIG, "--overlap-report",
             report_for(tmp_path / "overlap.json", [public]), "--dedup-scope", "all"]
    code, _, err, summary = run(plain, capsys)
    assert code == 0, err
    assert summary["conflicts"] == {"groups": 1, "targets": 1, "modes": 0, "resolved": 0, "unresolved": 1,
                                    "flagged": 0, "samples": 2}
    assert summary["flagged_conflicts"] == []


def test_a_think_and_a_no_think_sample_of_one_prompt_and_answer_conflict(tmp_path, capsys):
    synth = write_jsonl(tmp_path / "synth.jsonl", [agent("a", LONG), agent("b", SAMPLE_MESSAGES[:3])])
    teacher = tmp_path / "teacher.jsonl"
    sidecar(teacher, [("b", 0, SEARCH, REASONING)])
    argv = [synth, "--model", model_folder(tmp_path / "model"), "--max-length", BIG, "--overlap-report",
            report_for(tmp_path / "overlap.json", [synth]), "--reasoning", teacher]
    code, _, err, summary = run(argv, capsys)  # synthetic samples: unresolved, it fails
    assert code == 2 and summary["conflicts"]["flagged"] == 1 and "conflict" in err
    code, _, err, summary = run([*argv, ALLOW], capsys)
    assert code == 0, err
    assert summary["conflicts"] == {"groups": 1, "targets": 0, "modes": 1, "resolved": 0, "unresolved": 1,
                                    "flagged": 1, "samples": 2}
    assert summary["excluded"] == {"conflict": 0, "duplicate": 0}
    assert summary["after"]["modes"] == {"think": 1, "no_think": 4, "think_fallback": 0}
    code, _, err, summary = run([*argv, "--no-think"], capsys)  # both NO_THINK: a plain duplicate
    assert code == 0, err
    assert summary["conflicts"]["groups"] == 0
    assert summary["duplicates"] == {"groups": 1, "samples": 2, "completions_differ": 0}
    assert summary["excluded"]["duplicate"] == 1 and summary["after"]["samples"] == 4


def test_duplicates_whose_completions_differ_are_counted_and_their_first_sample_stays(tmp_path, capsys):
    synth = write_jsonl(tmp_path / "synth.jsonl", [agent(name, SAMPLE_MESSAGES[:3]) for name in "abc"])
    teacher = tmp_path / "teacher.jsonl"  # one target and mode, b with another reasoning
    sidecar(teacher, [("a", 0, SEARCH, REASONING), ("b", 0, SEARCH, "Another reasoning."),
                      ("c", 0, SEARCH, REASONING)])
    keys = tmp_path / "keys.jsonl"
    code, out, err, summary = run([synth, "--model", model_folder(tmp_path / "model"), "--max-length", BIG,
                                   "--overlap-report", report_for(tmp_path / "overlap.json", [synth]),
                                   "--reasoning", teacher, "--keys-out", keys], capsys)
    assert code == 0, err
    assert summary["duplicates"] == {"groups": 1, "samples": 3, "completions_differ": 1}
    assert summary["excluded"]["duplicate"] == 2 and train.load_sample_keys(keys) == [("a", 0)]
    assert "1 of them with completions that differ" in out


def test_the_changes_rows_name_the_samples_built_and_the_changed_version_wins_a_conflict(tmp_path, capsys):
    synth, changes = changed_mix(tmp_path)
    keys = tmp_path / "keys.jsonl"
    code, out, err, summary = run(changed_argv(tmp_path, synth, changes, "--keys-out", keys), capsys)
    assert code == 0, err
    # c's relabel wins a's first prompt; e's copy repeats a's second step, which comes first
    assert summary["excluded"] == {"conflict": 1, "duplicate": 1}
    assert summary["conflicts"] == {"groups": 1, "targets": 1, "modes": 0, "resolved": 1, "unresolved": 0,
                                    "flagged": 0, "samples": 2}
    account = summary["changes"]
    assert account["rows"] == 4 and account["targets"] == {"true": 3, "false": 1}
    assert account["kinds"] == {"relabel": {"rows": 1, "kept": 1}, "new": {"rows": 1, "kept": 1},
                                "copy": {"rows": 2, "kept": 0}}
    assert account["rules"] == {"R10": {"rows": 2, "kept": 0}, "R14": {"rows": 1, "kept": 1},
                                "R8": {"rows": 1, "kept": 1}}
    nothing = {"count": 0, "keys": []}
    assert account["over_length"] == account["without_sample"] == account["context_built"] == nothing
    assert account["uncovered"] is None  # no before mix to look in
    after = summary["after"]
    assert (after["built"], after["samples"], after["synthetic"], after["new"]) == (11, 9, 9, 1)
    assert summary["new"] == {"new": 1, "copies_new": 0, "copies_moved": 0}
    assert summary["share"] == pytest.approx(1 / 9) and summary["within_cap"] is True
    listed = [(r["id"], r["k"]) for r in map(json.loads, keys.read_text(encoding="utf-8").splitlines())]
    assert listed == [("a", 1), ("a", 2), ("a", 3), ("c", 0), ("f", 0), ("f", 1), ("f", 2), ("f", 3),
                      ("n", 0)]
    assert "target true: 3 name a sample built, 0 a turn dropped over the length, 0 none" in out
    assert "target false: 0 name a sample built (a context turn gives none)" in out


@pytest.mark.parametrize("row", [
    change("gone", 0, "new", rule="R14"),  # a misspelled id
    change("c", 1, "relabel"),  # an off-by-one k
    change("e", 2, "copy", rule="R10"),  # a copy that is a target names a sample as well
], ids=["new", "relabel", "copy"])
def test_a_changes_row_that_is_a_target_and_names_no_sample_built_fails_the_check(tmp_path, capsys, row):
    synth, changes = changed_mix(tmp_path)
    with changes.open("a", encoding="utf-8") as rows:
        rows.write(json.dumps(row) + "\n")
    keys = tmp_path / "keys.jsonl"
    code, _, err, summary = run(changed_argv(tmp_path, synth, changes, "--keys-out", keys), capsys)
    assert code == 2 and not keys.exists() and not list(tmp_path.glob("keys.jsonl*"))
    key = (row["id"], row["k"])
    assert summary["changes"]["without_sample"] == {"count": 1, "keys": [list(key)]}
    assert "1 changes row(s) with target true name no sample built" in err and repr(key) in err
    assert summary["new"]["new"] == 1  # what was counted is still reported


def test_a_changes_row_naming_a_turn_dropped_over_the_length_is_recorded_and_passes(tmp_path, capsys):
    long = {**REQUEST, "content": "长" * 20_000}
    longer = {**REQUEST, "content": "长" * 30_000}
    synth = write_jsonl(tmp_path / "synth.jsonl", [agent("n", [SYSTEM, long, SEARCH]), agent("a", LONG),
                                                   agent("m", [SYSTEM, longer, SEARCH])])
    changes = write_changes(tmp_path / "changes.jsonl", [change("n", 0, "new", rule="R14")])
    model, report = model_folder(tmp_path / "model"), report_for(tmp_path / "overlap.json", [synth])
    spec = check.spec_from_args(check.build_parser().parse_args(
        [str(a) for a in (synth, "--model", model, "--overlap-report", report, "--max-length", BIG)]))
    tokens = {s["key"]: s["tokens"] for s in check.build_mix(spec, family_hook()).samples}
    longest = max(tokens[("a", k)] for k in range(4))
    assert min(tokens[("n", 0)], tokens[("m", 0)]) > longest
    argv = [synth, "--model", model, "--overlap-report", report, "--max-length", longest, "--changes",
            changes]
    mix = check.build_mix(check.spec_from_args(check.build_parser().parse_args([str(a) for a in argv])),
                          family_hook())
    # the assembly's drops, by key: one before a's samples, one after the last of them
    assert mix.dropped_keys == [("n", 0), ("m", 0)] and mix.dropped.over_length == 2
    code, out, err, summary = run(argv, capsys)
    assert code == 0, err
    assert summary["changes"]["over_length"] == {"count": 1, "keys": [["n", 0]]}
    assert summary["changes"]["without_sample"] == {"count": 0, "keys": []}
    assert "target true: 0 name a sample built, 1 a turn dropped over the length, 0 none" in out
    # a record the overlap report skips is no such exception: its row names no sample
    report_for(report, [synth], skipped={synth: [0]})
    code, _, err, summary = run([*argv[:-4], "--max-length", BIG, *argv[-2:]], capsys)
    assert code == 2 and "('n', 0)" in err and summary["changes"]["without_sample"]["count"] == 1


def test_a_changes_row_whose_target_is_false_must_name_a_context_turn_no_sample_is_built_for(tmp_path,
                                                                                            capsys):
    synth, changes = changed_mix(tmp_path)  # e's first turn: context only, as its row says
    code, _, err, summary = run(changed_argv(tmp_path, synth, changes), capsys)
    assert code == 0 and summary["changes"]["context_built"] == {"count": 0, "keys": []}, err
    records = [json.loads(line) for line in synth.read_text(encoding="utf-8").splitlines()]
    del records[2]["context_turns"]  # e without its context turns: a run would train the turn left out
    write_jsonl(synth, records)
    keys = tmp_path / "keys.jsonl"
    code, out, err, summary = run(changed_argv(tmp_path, synth, changes, "--keys-out", keys), capsys)
    assert code == 2 and not keys.exists()
    assert summary["changes"]["context_built"] == {"count": 1, "keys": [["e", 0]]}
    assert "1 changes row(s) with target false name a sample built" in err and "('e', 0)" in err
    assert "target false: 1 name a sample built" in out


def test_with_a_before_mix_a_synthetic_sample_it_did_not_build_needs_a_changes_row(tmp_path, capsys):
    before_synth = write_jsonl(tmp_path / "synth-before.jsonl",
                               [agent("a", LONG), agent("f", asked("另一个请求"))])
    unlisted = agent("x", [SYSTEM, {**REQUEST, "content": "没有登记的请求"}, SEARCH])
    after_synth = write_jsonl(tmp_path / "synth-after.jsonl",
                              [agent("a", LONG), agent("f", asked("另一个请求")), unlisted])
    public = write_jsonl(tmp_path / "public.jsonl", [PUBLIC_SAMPLE])  # new, but no synthetic sample
    model = model_folder(tmp_path / "model")
    report = report_for(tmp_path / "overlap.json", [before_synth, after_synth, public])
    before = train_manifest(tmp_path / "manifest.json", [before_synth], model=model, report=report)
    argv = [after_synth, public, "--model", model, "--max-length", BIG, "--overlap-report", report]
    code, _, err, summary = run([*argv, "--before", before], capsys)
    assert code == 2 and "('x', 0)" in err
    assert "1 synthetic sample(s) whose key the before mix did not build have no changes row" in err
    assert summary["changes"]["uncovered"] == {"count": 1, "keys": [["x", 0]]}
    changes = write_changes(tmp_path / "changes.jsonl", [change("x", 0, "new", rule="R14")])
    code, _, err, summary = run([*argv, "--before", before, "--changes", changes], capsys)
    assert code == 0 and summary["changes"]["uncovered"] == {"count": 0, "keys": []}, err
    assert summary["after"]["new"] == 1
    code, _, err, summary = run(argv, capsys)  # without a before mix there is nothing to look in
    assert code == 0 and summary["changes"]["uncovered"] is None, err


def test_a_copy_that_is_a_target_is_moved_when_the_before_mix_holds_its_prompt_and_answer(tmp_path, capsys):
    """FT4's kept teacher asks: the turn leaves its cut record for a derived one, its prompt and answer
    unchanged."""
    before_synth = write_jsonl(tmp_path / "synth-before.jsonl",
                               [agent("a", LONG), agent("f", asked("另一个请求"))])
    # a relabelled at its first step and cut there; x keeps a's second turn; y's second turn is no before turn
    after_synth = write_jsonl(tmp_path / "synth-after.jsonl", [
        agent("a", [SYSTEM, REQUEST, OTHER_SEARCH]), agent("f", asked("另一个请求")),
        derived("x", SAMPLE_MESSAGES[:5], 1), derived("y", asked("第三个请求")[:5], 1)])
    changes = write_changes(tmp_path / "changes.jsonl", [
        change("a", 0, "relabel", source=["a", 0]),
        change("x", 0, "copy", target=False, rule="R10", source=["a", 0]),
        change("x", 1, "copy", rule="R10", source=["a", 1]),
        change("y", 0, "copy", target=False, rule="R10", source=["y", 0]),
        change("y", 1, "copy", rule="R10", source=["y", 1])])
    model = model_folder(tmp_path / "model")
    report = report_for(tmp_path / "overlap.json", [before_synth, after_synth])
    before = train_manifest(tmp_path / "manifest.json", [before_synth], model=model, report=report)
    argv = [after_synth, "--model", model, "--max-length", BIG, "--overlap-report", report, "--changes",
            changes]
    code, out, err, summary = run([*argv, "--before", before], capsys)
    assert code == 0, err
    assert summary["new"] == {"new": 0, "copies_new": 1, "copies_moved": 1}
    after = summary["after"]
    assert (after["samples"], after["synthetic"], after["new"]) == (7, 7, 1)
    assert after["new_types"]["parallel"] == 1 and sum(after["new_types"].values()) == 1
    assert ("new samples: 1: changes kind new 0, copies 1 (moved, their prompt and answer in the before mix: "
            "1)") in out
    # without a before mix no copy can be shown to move: both count as new, 2 of the 7 kept is over 15 %
    code, out, err, summary = run(argv, capsys)
    assert code == 3 and summary["new"] == {"new": 0, "copies_new": 2, "copies_moved": 0}
    assert "no before mix: every copy kept counts as new" in out


def test_with_a_before_mix_the_cap_is_a_share_of_its_synthetic_samples_and_both_readings_print(tmp_path,
                                                                                                capsys):
    before_synth = write_jsonl(tmp_path / "synth-before.jsonl", [agent("a", LONG)])
    after_synth = write_jsonl(tmp_path / "synth-after.jsonl", [
        agent("a", LONG), agent("n", [SYSTEM, {**REQUEST, "content": "新的请求"}, SEARCH])])
    changes = write_changes(tmp_path / "changes.jsonl", [change("n", 0, "new", rule="R14")])
    model = model_folder(tmp_path / "model")
    report = report_for(tmp_path / "overlap.json", [before_synth, after_synth])
    before = train_manifest(tmp_path / "manifest.json", [before_synth], model=model, report=report)
    keys = tmp_path / "keys.jsonl"
    argv = [after_synth, "--model", model, "--max-length", BIG, "--overlap-report", report, "--changes",
            changes, "--cap", "0.2", "--keys-out", keys]
    code, out, err, summary = run([*argv, "--before", before], capsys)
    # 1 new sample is 25 % of the before mix's 4 synthetic samples, 20 % of the 5 kept: the first decides
    assert code == 3 and not keys.exists()
    assert "new samples exceed the cap: 1 of the before mix's 4 synthetic samples, more than 20%" in err
    assert summary["cap_base"] == "before" and summary["share"] == pytest.approx(0.25)
    assert summary["shares"] == {"before": pytest.approx(0.25), "after": pytest.approx(0.2)}
    assert summary["at_most"] == 0 and summary["within_cap"] is False
    assert "cap 20% of the before mix's 4 synthetic samples: 1 (25.0%), at most 0: EXCEEDED" in out
    assert "the other reading, of the 5 synthetic samples kept: 1 (20.0%), at most 1" in out
    code, out, err, summary = run(argv, capsys)  # without a before mix the samples kept are the base
    assert code == 0 and keys.is_file(), err
    assert summary["cap_base"] == "after" and summary["share"] == pytest.approx(0.2)
    assert summary["shares"] == {"before": None, "after": pytest.approx(0.2)} and summary["at_most"] == 1
    assert "cap 20% of the 5 synthetic samples kept: 1 (20.0%), at most 1: ok" in out
    assert "the other reading" not in out


@pytest.mark.parametrize(("line", "message"), [
    ("{oops", "line 2 is not JSON"),
    (json.dumps({"id": "b", "k": 0}), "line 2: a changes row is an object with exactly the keys id, k, rule"),
    (json.dumps({**change("b", 0, "new"), "note": "x"}), "line 2: a changes row is an object with exactly"),
    (json.dumps(change("b", 0, "moved")), "line 2: kind is not one of relabel, new, copy"),
    (json.dumps({**change("b", 0, "new"), "target": "yes"}), "line 2: target is not true or false"),
    (json.dumps(change("b", -1, "new")), "line 2: k is not an integer >= 0"),
    (json.dumps({**change("b", 0, "new"), "k": True}), "line 2: k is not an integer >= 0"),
    (json.dumps(change("", 0, "new")), "line 2: id is not a non-empty string"),
    (json.dumps(change("b", 0, "new", rule="")), "line 2: rule is not a non-empty string"),
    (json.dumps(change("b", 0, "copy", source=["a"])), r"line 2: from is neither null nor \[id, k\]"),
    (json.dumps(change("b", 0, "copy", source=["a", "0"])), r"line 2: from is neither null nor \[id, k\]"),
    (json.dumps(change("a", 0, "copy")), r"line 2: turn \('a', 0\) is also in .*line 1"),
])
def test_a_malformed_changes_row_is_refused_naming_its_line(tmp_path, capsys, line, message):
    path = tmp_path / "changes.jsonl"
    path.write_text(json.dumps(change("a", 0, "new")) + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(check.ChangesError, match=message):
        check.load_changes([path])
    code, _, err, _ = run([*base_mix(tmp_path), "--changes", path], capsys)
    assert code == 2 and "FAIL" in err and "line 2" in err


def test_a_turn_in_two_changes_files_is_refused_and_the_rows_of_several_files_are_read(tmp_path):
    first = write_changes(tmp_path / "m6.jsonl", [change("a", 0, "new"),
                                                  change("a", 1, "copy", target=False)])
    second = write_changes(tmp_path / "m7.jsonl", [change("b", 0, "relabel", source=["b", 0])])
    rows = check.load_changes([first, second])
    assert list(rows) == [("a", 0), ("a", 1), ("b", 0)] and rows[("b", 0)]["from"] == ["b", 0]
    write_changes(second, [change("a", 1, "new")])
    with pytest.raises(check.ChangesError, match=r"m7.jsonl: line 1: turn \('a', 1\) is also in .*m6.jsonl"):
        check.load_changes([first, second])
    assert check.load_changes([]) == {}


def light(key: tuple[str, int], prompt: str, answer: str, mode: str = "no_think", *,
          synthetic: bool = False) -> dict:
    """A sample as resolve reads it: its key, prompt hash, answer hash, mode and whether it is synthetic."""
    return {"key": key, "prompt_sha256": prompt, "answer_sha256": answer, "mode": mode,
            "synthetic": synthetic}


def test_the_changed_versions_of_a_prompt_win_and_several_of_them_stay_unresolved():
    samples = [light(("a", 0), "p", "x"), light(("b", 0), "p", "x"), light(("c", 0), "p", "y"),
               light(("d", 0), "p", "z"), light(("e", 0), "p", "y"), light(("f", 0), "q", "x")]
    changes = {("c", 0): change("c", 0, "relabel"), ("d", 0): change("d", 0, "new")}
    resolution = check.resolve(samples, changes)
    assert [s["key"] for s in resolution.kept] == [("c", 0), ("d", 0), ("f", 0)]
    assert resolution.excluded == {"conflict": 2, "duplicate": 1}  # a, b lose; e repeats c
    assert resolution.conflicts == {"groups": 1, "targets": 1, "modes": 0, "resolved": 0, "unresolved": 1,
                                    "flagged": 1, "samples": 5}  # two changed versions: flagged
    assert resolution.flagged == [[("a", 0), ("b", 0), ("c", 0), ("d", 0), ("e", 0)]]
    assert resolution.prompts == {"distinct": 2, "shared": 1, "mixed": 0}
    assert [s["key"] for s in resolution.grouped["conflicts"]] == [("a", 0), ("b", 0), ("c", 0), ("d", 0),
                                                                    ("e", 0)]
    assert resolution.grouped["duplicates"] == []
    one = check.resolve(samples[:3], {("c", 0): change("c", 0, "relabel")})
    assert [s["key"] for s in one.kept] == [("c", 0)] and one.conflicts["resolved"] == 1
    assert one.conflicts["flagged"] == 0 and one.flagged == []
    # without a changed version, the conflict is flagged when a synthetic sample is in it
    neither = check.resolve([light(("a", 0), "p", "x"), light(("b", 0), "p", "y")], {})
    assert neither.conflicts["unresolved"] == 1 and neither.flagged == []
    synthetic = check.resolve([light(("a", 0), "p", "x"), light(("b", 0), "p", "y", synthetic=True)], {})
    assert synthetic.flagged == [[("a", 0), ("b", 0)]]
    # a changed THINK version of a NO_THINK step: the modes alone differ, the changed one wins
    modes = check.resolve([light(("a", 0), "p", "x"), light(("b", 0), "p", "x", "think")],
                          {("b", 0): change("b", 0, "relabel")})
    assert [s["key"] for s in modes.kept] == [("b", 0)]
    assert (modes.conflicts["modes"], modes.conflicts["targets"], modes.excluded["conflict"]) == (1, 0, 1)
    # a new sample that repeats an existing one is a plain duplicate: the first one stays
    repeated = check.resolve([light(("a", 0), "p", "x"), light(("n", 0), "p", "x")],
                             {("n", 0): change("n", 0, "new")})
    assert [s["key"] for s in repeated.kept] == [("a", 0)] and repeated.excluded["duplicate"] == 1
    assert repeated.duplicates == {"groups": 1, "samples": 2, "completions_differ": 0}
    assert repeated.conflicts["groups"] == 0


def test_the_synthetic_scope_groups_only_the_synthetic_samples():
    samples = [light(("a", 0), "p", "x"), light(("b", 0), "p", "x"),
               light(("c", 0), "p", "x", synthetic=True), light(("d", 0), "p", "x", synthetic=True),
               light(("e", 0), "q", "y", synthetic=True)]
    scoped = check.resolve(samples, {}, scope="synthetic")
    assert [s["key"] for s in scoped.kept] == [("a", 0), ("b", 0), ("c", 0), ("e", 0)]  # d repeats c
    assert scoped.excluded == {"conflict": 0, "duplicate": 1}
    assert scoped.prompts == {"distinct": 2, "shared": 1, "mixed": 1}  # p: synthetic and other samples
    every = check.resolve(samples, {}, scope="all")
    assert [s["key"] for s in every.kept] == [("a", 0), ("e", 0)] and every.excluded["duplicate"] == 3
    assert check.resolve(samples, {}).excluded == every.excluded  # by default, all the samples given


def test_new_samples_beyond_the_cap_fail_the_check_and_no_keys_file_is_written(tmp_path, capsys):
    synth, changes = changed_mix(tmp_path)
    keys = tmp_path / "keys.jsonl"
    argv = changed_argv(tmp_path, synth, changes, "--keys-out", keys)
    code, out, err, summary = run([*argv, "--cap", "0.1"], capsys)  # 1 new of 9 synthetic: 11.1 %
    assert code == 3 and not keys.exists() and not list(tmp_path.glob("keys.jsonl*"))
    assert "11.1%" in out and "cap 10%" in out and "new samples exceed" in err
    assert summary["within_cap"] is False and summary["sample_keys"] is None
    code, _, err, summary = run(argv, capsys)  # the default cap, 15 %
    assert code == 0 and keys.is_file() and summary["cap"] == 0.15, err
    assert check.within_cap(15, 100, 0.15) and not check.within_cap(16, 100, 0.15)  # at the cap passes
    assert check.within_cap(1272, 8483, 0.15) and not check.within_cap(1273, 8483, 0.15)  # FT3's synthetic
    assert check.within_cap(0, 0, 0.15) and not check.within_cap(1, 0, 0.15)


def test_the_sample_keys_file_selects_exactly_the_kept_samples_through_train_pys_own_loader(tmp_path, capsys):
    argv = base_mix(tmp_path)
    keys = tmp_path / "keys.jsonl"
    code, _, err, summary = run([*argv, ALLOW, "--keys-out", keys], capsys)
    assert code == 0, err
    files = [Path(argv[0]), Path(argv[1])]
    samples, _, _ = train.assemble(files, loaded_fake_family(tmp_path / "model"), BIG)
    listed = train.load_sample_keys(keys)
    selected = train.select_samples(samples, listed)
    assert [tuple(s["key"]) for s in selected] == listed and len(listed) == 9  # b and d left out
    positions = [next(i for i, s in enumerate(samples) if s is chosen) for chosen in selected]
    assert positions == sorted(positions)  # the mix's own order
    assert keys.read_bytes() == "".join(json.dumps({"id": i, "k": k}, ensure_ascii=False) + "\n"
                                        for i, k in listed).encode("utf-8")  # UTF-8, LF
    assert summary["sample_keys"] == {"path": str(keys), "sha256": train.sha256_of(keys), "listed": 9}
    # train.py's dry run takes it as --sample-keys
    dry = [*argv, "--out", str(tmp_path / "run"), "--dry-run", "--sample-keys", str(keys)]
    assert train.main(dry, load_family=family_hook()) == 0
    assert "sample keys: 9 listed, 9 samples selected" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.partial"))


def test_samples_sharing_a_key_cannot_be_listed_in_a_keys_file(tmp_path, capsys):
    one = write_jsonl(tmp_path / "one.jsonl", [agent("a", LONG)])
    two = write_jsonl(tmp_path / "two.jsonl", [agent("a", asked("另一个请求"))])  # the same id, another file
    keys = tmp_path / "keys.jsonl"
    argv = [one, two, "--model", model_folder(tmp_path / "model"), "--max-length", BIG, "--overlap-report",
            report_for(tmp_path / "overlap.json", [one, two])]
    code, out, err, summary = run([*argv, "--keys-out", keys], capsys)
    assert code == 2 and "('a', 0)" in err and not keys.exists() and not list(tmp_path.glob("*.partial"))
    assert summary["shared_keys"] == 4 and "shared keys: 4" in out
    code, _, err, summary = run(argv, capsys)  # without a keys file the check itself goes through
    assert code == 0 and summary["after"]["samples"] == 8, err


def test_the_tables_count_by_step_type_and_by_source_file_before_and_after(tmp_path, capsys):
    public = write_jsonl(tmp_path / "public.jsonl", [PUBLIC_SAMPLE, PUBLIC_SAMPLE])
    before_synth = write_jsonl(tmp_path / "synth-before.jsonl", [
        agent("a", LONG), agent("b", SAMPLE_MESSAGES[:3]), agent("f", asked("另一个请求"))])
    after_synth, changes = changed_mix(tmp_path, "synth-after.jsonl")
    model = model_folder(tmp_path / "model")
    report = report_for(tmp_path / "overlap.json", [public, before_synth, after_synth])
    before = train_manifest(tmp_path / "manifest.json", [before_synth, public], model=model, report=report)
    argv = [after_synth, public, "--model", model, "--max-length", BIG, "--overlap-report", report,
            "--changes", changes, "--before", before]
    code, out, err, summary = run(argv, capsys)
    assert code == 0, err
    assert summary["before"]["source"] == str(before) and summary["before"]["samples"] == 13
    assert summary["before"]["types"] == {"call": 5, "next_step": 6, "no_call": 0, "clarify": 0, "recover": 0,
                                          "parallel": 2}
    assert summary["before"]["files"] == {before_synth.as_posix(): 9, public.as_posix(): 4}
    assert summary["before"]["synthetic"] == 9
    after = summary["after"]  # the public duplicates stay: the dedup scope is the synthetic samples
    assert (after["built"], after["samples"], after["synthetic"], after["new"]) == (15, 13, 9, 1)
    assert after["types"] == {"call": 5, "next_step": 6, "no_call": 0, "clarify": 0, "recover": 0,
                              "parallel": 2}
    assert after["synthetic_types"] == {"call": 3, "next_step": 4, "no_call": 0, "clarify": 0, "recover": 0,
                                        "parallel": 2}
    assert after["new_types"]["call"] == 1 and sum(after["new_types"].values()) == 1
    assert after["files"] == {after_synth.as_posix(): 9, public.as_posix(): 4}
    assert summary["changes"]["uncovered"] == {"count": 0, "keys": []}  # c, e and n have rows; a, f before
    lines = out.splitlines()
    head = next(i for i, line in enumerate(lines) if line.startswith("== samples by step type"))
    assert lines[head + 1].split() == ["type", "before", "after", "change", "before", "synth", "after",
                                       "synth", "new"]
    assert lines[head + 2].split() == ["call", "5", "5", "0", "3", "3", "1"]
    assert lines[head + 4].split() == ["no_call", "0", "0", "0", "0", "0", "0"]
    assert lines[head + 7].split() == ["parallel", "2", "2", "0", "2", "2", "0"]
    assert lines[head + 8].split() == ["total", "13", "13", "0", "9", "9", "1"]
    head = next(i for i, line in enumerate(lines) if line.startswith("== samples by source file"))
    assert lines[head + 1].split() == ["file", "before", "after", "change", "new"]
    assert lines[head + 2].split() == [before_synth.as_posix(), "9", "0", "-9", "0"]
    assert lines[head + 3].split() == [public.as_posix(), "4", "4", "0", "0"]
    assert lines[head + 4].split() == [after_synth.as_posix(), "0", "9", "+9", "1"]
    assert lines[head + 5].split() == ["total", "13", "13", "0", "1"]
    assert "cap 15% of the before mix's 9 synthetic samples: 1 (11.1%), at most 1: ok" in out


def test_without_a_before_mix_the_table_compares_the_mix_as_built_with_what_is_kept(tmp_path, capsys):
    code, out, err, summary = run([*base_mix(tmp_path), ALLOW], capsys)
    assert code == 0, err
    assert summary["before"]["source"] == "the mix as built" and summary["before"]["samples"] == 11
    assert summary["before"]["types"]["call"] == 6 and summary["after"]["types"]["call"] == 4  # b, d out
    assert "before: the mix as built" in out and summary["sample_keys"] is None


def test_a_mix_given_by_a_train_manifest_is_rebuilt_and_compared_with_the_manifest(tmp_path, capsys):
    argv = base_mix(tmp_path)
    synth, public, model, report = Path(argv[0]), Path(argv[1]), Path(argv[3]), Path(argv[5])
    teacher = tmp_path / "teacher.jsonl"
    sidecar(teacher, [("a", 1, LONG[4], REASONING)])
    manifest = train_manifest(tmp_path / "manifest.json", [synth, public], model=model, report=report,
                              reasoning=(teacher,))
    code, out, err, summary = run(["--manifest", manifest, ALLOW], capsys)
    assert code == 0, err
    assert "manifest check: equal" in out and summary["manifest_check"] == {"after": [], "before": None}
    assert summary["after"]["source"] == str(manifest) and summary["after"]["built"] == 11
    assert summary["after"]["modes"] == {"think": 1, "no_think": 8, "think_fallback": 0}
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    recorded["config"]["data"]["files"][0]["samples"] = 8
    recorded["config"]["data"]["reasoning"]["files"][0]["sha256"] = "0" * 64
    recorded["config"]["family"]["template_sha256"] = "1" * 64
    manifest.write_text(json.dumps(recorded), encoding="utf-8")
    code, out, err, summary = run(["--manifest", manifest, ALLOW], capsys)
    assert code == 0, err  # reported, not failed on
    assert "manifest check: 3 difference(s)" in out
    assert any("samples 8 -> 7" in line for line in summary["manifest_check"]["after"])
    assert any(line.startswith("family.template_sha256") for line in summary["manifest_check"]["after"])
    for extra in ([str(synth)], ["--max-length", "5"], ["--no-think"], ["--reasoning", str(teacher)]):
        code, _, err, _ = run(["--manifest", manifest, *extra], capsys)
        assert code == 2 and "not both" in err
    code, _, err, _ = run(["--manifest", tmp_path / "missing.json"], capsys)
    assert code == 2 and "missing.json" in err
    code, _, err, _ = run([], capsys)
    assert code == 2 and "no data file" in err


def test_a_manifest_whose_run_trained_on_sample_keys_gives_the_samples_they_select(tmp_path, capsys):
    argv = base_mix(tmp_path)
    synth, public, model, report = Path(argv[0]), Path(argv[1]), Path(argv[3]), Path(argv[5])
    selection = write_keys(tmp_path / "selected.jsonl", [("public:0", 1), ("a", 1), ("b", 0), ("a", 0)])
    manifest = train_manifest(tmp_path / "manifest.json", [synth, public], model=model, report=report,
                              sample_keys=selection)
    keys = tmp_path / "keys.jsonl"
    code, out, err, summary = run(["--manifest", manifest, "--keys-out", keys], capsys)
    assert code == 0, err
    assert "manifest check: equal" in out and summary["manifest_check"]["after"] == []
    after = summary["after"]
    assert (after["built"], after["selected"], after["synthetic"]) == (11, 4, 2)
    # in the keys' order b comes before a's first step, which repeats it
    assert summary["excluded"] == {"conflict": 0, "duplicate": 1} and after["samples"] == 3
    assert train.load_sample_keys(keys) == [("public:0", 1), ("a", 1), ("b", 0)]
    assert f"sample keys: {selection.as_posix()} lists 4, selecting 4 of the 11 samples built" in out
    # the before mix of the table is what its run trained on, too
    code, _, err, summary = run([*argv, ALLOW, "--before", manifest], capsys)
    assert code == 0, err
    assert (summary["before"]["built"], summary["before"]["selected"]) == (11, 4)
    assert summary["before"]["samples"] == 4 and summary["before"]["synthetic"] == 3
    assert summary["changes"]["uncovered"] == {"count": 0, "keys": []}  # the before mix built every key
    # a keys file that changed since the run (its SHA-256 is not data.sample_keys.sha256): the samples the run
    # trained on cannot be told, so the check stops before anything is built
    recorded = json.loads(manifest.read_text(encoding="utf-8"))["config"]["data"]["sample_keys"]["sha256"]
    write_keys(selection, [("public:0", 1), ("a", 1), ("b", 0)])
    for args in (["--manifest", manifest], [*argv, ALLOW, "--before", manifest]):
        code, _, err, summary = run(args, capsys)
        assert code == 2 and summary is None, err
        assert "selected.jsonl" in err and recorded in err and train.sha256_of(selection) in err
    selection.unlink()  # a run's keys file that is gone: its samples cannot be told
    for args in (["--manifest", manifest], [*argv, ALLOW, "--before", manifest]):
        code, _, err, summary = run(args, capsys)
        assert code == 2 and "selected.jsonl" in err and summary is None


def test_an_empty_glossary_or_a_dropout_without_one_stops_the_check_before_anything_is_read(tmp_path, capsys):
    """train.py's glossary_flags_problem: an empty --glossary names no file, and --glossary-dropout without
    --glossary would be ignored; both are exit 2 before any file is read (the missing ones here are not
    reported)."""
    missing = tmp_path / "missing.json"
    for argv, message in (([missing, "--glossary-dropout", "0.3"], "--glossary-dropout needs --glossary"),
                          (["--manifest", missing, "--glossary-dropout", "0.3"],
                           "--glossary-dropout needs --glossary"),
                          ([missing, "--glossary", ""], "--glossary names no file (an empty path)"),
                          ([missing, "--glossary=", "--glossary-dropout", "0.3"],
                           "--glossary names no file")):
        code, out, err, summary = run(argv, capsys)
        assert code == 2 and summary is None and message in err and "missing.json" not in err, argv
        assert out == ""


def test_input_errors_stop_the_check_with_train_pys_exit_codes(tmp_path, capsys):
    argv = base_mix(tmp_path)
    synth, report = Path(argv[0]), Path(argv[5])
    assert run([str(tmp_path / "*.missing"), *argv[1:]], capsys)[0] == 2  # no data file
    code, _, err, _ = run([*argv, "--model", tmp_path / "no-model"], capsys)
    assert code == 2 and "no model folder" in err
    bare = tmp_path / "bare"  # a folder that is no model folder: no family
    bare.mkdir()
    code, _, err, _ = run([*argv, "--model", bare], capsys, hook=train.default_load_family)
    assert code == 2 and "config.json" in err
    teacher = tmp_path / "teacher.jsonl"
    sidecar(teacher, [("a", 1, LONG[2], REASONING)])  # written for another answer: stale
    code, _, err, _ = run([*argv, "--reasoning", teacher], capsys)
    assert code == 2 and "stale" in err
    code, _, err, _ = run([*argv, "--reasoning", tmp_path / "none.jsonl"], capsys)
    assert code == 2 and "none.jsonl" in err
    sidecar(teacher, [("a", 0, SEARCH, "<s> a BOS in the reasoning")])  # a sample the family refuses
    code, _, err, _ = run([*argv, "--reasoning", teacher], capsys,
                          hook=family_hook(tokenizer=FakeFamilyTokenizer("hybrid", bos="token")))
    assert code == 2 and "refuses 1 sample" in err and "bos ('a', 0)" in err
    code, _, err, _ = run([*argv, "--cap", "1.5"], capsys)
    assert code == 2 and "--cap" in err
    # the overlap gate as train.py runs it: an unlisted file stops the check (4) unless unchecked
    report_for(report, [Path(argv[1])])
    code, _, err, _ = run(argv, capsys)
    assert code == 4 and "synth.jsonl" in err
    code, _, err, summary = run([*argv, ALLOW, "--no-overlap-check"], capsys)
    assert code == 0 and summary["after"]["built"] == 11, err
    # the records the report names are skipped, as train.py skips them
    report_for(report, [synth, Path(argv[1])], skipped={synth: [1]})
    code, _, err, summary = run([*argv, ALLOW], capsys)
    assert code == 0 and summary["after"]["built"] == 10, err
    assert summary["excluded"] == {"conflict": 0, "duplicate": 1}  # b skipped: only d repeats a
