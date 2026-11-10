"""The per-answer successes of an ACEBench run, read back from the raw folder scripts/run_acebench.py copies
(docs/m6-计划.md §5, H3: the stratified bootstrap over the (language, category) strata,
eval.metrics.stratified_noninferiority_p, needs every answer's success, paired by id between two runs).

Layout: <folder>/<language>/result/data_<category>_result.json, one JSON object per line, an answer per
question with its id, and <folder>/<language>/score/data_<category>_score.json, ACEBench's header
{accuracy, correct_count, total_count}, then one line per wrong answer with its id and "valid": false. A
language is a subfolder of the run holding a score folder, a category every score file of it, and a stratum
a (language, category) pair; a multi-turn category's answers are its turns, one line each, as the results
file counts them (docs/m4-进度.md). An answer is right unless the score file holds a wrong line for its id,
with one exception: for a question with several candidate answers (MULTI_CANDIDATE), ACEBench writes a wrong
line when the answer failed the first candidate and passed a later one, and still counts it correct, so the
file holds one more wrong line than its header's wrong count; that answer is right, and the header's counts
are reproduced exactly. A run whose files cannot give one success per answer is refused (AceItemsError,
naming the file): no language folder or score file, a missing result file, answers that are not one per
question with distinct ids, a header at odds with itself or with the answers, a wrong line without an id or
with "valid" other than false, a wrong id that names no answer or appears twice, fewer wrong lines than the
header's wrong count, or more of them than the multi-candidate questions among them account for.

per_item(folder) reads a run: its strata (sorted), each as {id: success} in the result file's order, the
headers {n, correct, later_candidates}, the files read as (path relative to the run, SHA-256) and one digest
over them, the SHA-256 of the sorted "relative path\\tsha256\\n" lines, which scripts/run_test.py records
beside the run's path. pair_strata(a, b) pairs two runs by stratum and, within each, by answer id in a's
order (ValueError when the strata or the ids differ): the inputs of stratified_noninferiority_p.

MULTI_CANDIDATE lists, per stratum, the questions of ACEBench's data_all at upstream commit 56dd66c (the one
scripts/run_acebench.py checks the clone against, data files included) whose possible answer lists more than
one candidate: four of the 1,640 normal questions; no special question has more than one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .lock import sha256_of

Stratum = tuple[str, str]  # (language, category)
MULTI_CANDIDATE: Mapping[Stratum, frozenset[str]] = {
    ("zh", "normal_atom_bool"): frozenset({"normal_atom_bool_21"}),
    ("zh", "normal_atom_object_deep"): frozenset({"normal_atom_object_deep_44"}),
    ("en", "normal_atom_bool"): frozenset({"normal_atom_bool_11"}),
    ("en", "normal_atom_object_short"): frozenset({"normal_atom_object_short_42"}),
}
PREFIX, SCORE_SUFFIX, RESULT_SUFFIX = "data_", "_score.json", "_result.json"


class AceItemsError(ValueError):
    """A run whose files cannot give one success per answer (module docstring), naming the file."""


@dataclass(frozen=True)
class PerItem:
    """One run's per-answer successes (module docstring: per_item)."""

    path: str  # the run's folder as given, POSIX
    strata: dict[Stratum, dict[str, bool]]  # sorted by stratum; each in the result file's order
    headers: dict[Stratum, dict]  # {"n", "correct", "later_candidates"} per stratum
    files: list[tuple[str, str]]  # (path relative to the run, SHA-256), sorted
    sha256: str  # the digest over files

    @property
    def items(self) -> int:
        return sum(len(answers) for answers in self.strata.values())


def stratum_label(stratum: Stratum) -> str:
    return f"{stratum[0]}/{stratum[1]}"


def _lines(path: Path) -> list[dict]:
    """The JSON objects of an ACEBench file, one per line (lines end at a line feed only, as
    scripts/run_bfcl.py reads them); AceItemsError naming the file otherwise."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AceItemsError(f"{path.as_posix()} cannot be read: {exc}") from None
    try:
        rows = [json.loads(line) for line in text.split("\n") if line.strip()]
    except ValueError as exc:
        raise AceItemsError(f"{path.as_posix()} is not JSON lines: {exc}") from None
    if not all(isinstance(row, dict) for row in rows):
        raise AceItemsError(f"{path.as_posix()} holds a line that is not a JSON object")
    return rows


def _stratum(result_path: Path, score_path: Path, candidates: frozenset[str]) -> tuple[dict[str, bool], dict]:
    """One stratum's {id: success} and header (module docstring)."""
    ids = [answer.get("id") for answer in _lines(result_path)]
    if not ids or not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        raise AceItemsError(f"{result_path.as_posix()} must hold one answer per question, each with a "
                            "distinct string id")
    rows = _lines(score_path)
    if not rows:
        raise AceItemsError(f"{score_path.as_posix()} is empty: no header")
    header, wrong_lines = rows[0], rows[1:]
    n, right = header.get("total_count"), header.get("correct_count")
    if type(n) is not int or type(right) is not int or not 0 <= right <= n or n != len(ids):
        raise AceItemsError(f"{score_path.as_posix()}: the header {header} does not agree with the "
                            f"{len(ids)} answers of {result_path.name}")
    wrong = []
    for row in wrong_lines:
        if row.get("valid") is not False or not isinstance(row.get("id"), str):
            shown = json.dumps(row, ensure_ascii=False)[:200]
            raise AceItemsError(f"{score_path.as_posix()}: the line {shown} is not a wrong answer (valid "
                                "false, with an id)")
        wrong.append(row["id"])
    known = set(ids)
    stray = [i for i in wrong if i not in known]
    if stray:
        raise AceItemsError(f"{score_path.as_posix()}: the wrong answer {stray[0]!r} is no answer of "
                            f"{result_path.name}")
    if len(set(wrong)) != len(wrong):
        raise AceItemsError(f"{score_path.as_posix()}: a wrong answer is listed twice")
    extra = len(wrong) - (n - right)
    if extra < 0:
        raise AceItemsError(f"{score_path.as_posix()}: {len(wrong)} wrong answers listed, the header says "
                            f"{n - right} of {n} are wrong")
    later = [i for i in wrong if i in candidates]
    if extra and len(later) != extra:
        raise AceItemsError(f"{score_path.as_posix()}: {extra} more wrong answer(s) listed than the header's "
                            f"{n - right}, and {len(later)} of them are questions with several candidate "
                            "answers (a question answered right on a later candidate): the successes cannot "
                            "be told answer by answer")
    right_later = set(later) if extra else set()
    wrong_set = set(wrong)
    successes = {i: i not in wrong_set or i in right_later for i in ids}
    return successes, {"n": n, "correct": right, "later_candidates": extra}


def per_item(folder: str | Path, *,
             candidates: Mapping[Stratum, frozenset[str]] = MULTI_CANDIDATE) -> PerItem:
    """The per-answer successes of the run at folder (module docstring); AceItemsError naming what cannot
    be read as one success per answer."""
    root = Path(folder)
    languages = sorted(p.name for p in root.iterdir() if (p / "score").is_dir()) if root.is_dir() else []
    if not languages:
        raise AceItemsError(f"{root.as_posix()} holds no language folder with ACEBench score files")
    strata: dict[Stratum, dict[str, bool]] = {}
    headers: dict[Stratum, dict] = {}
    files: list[tuple[str, str]] = []
    for language in languages:
        score_dir = root / language / "score"
        names = sorted(p.name for p in score_dir.iterdir()
                       if p.is_file() and p.name.startswith(PREFIX) and p.name.endswith(SCORE_SUFFIX))
        if not names:
            raise AceItemsError(f"{score_dir.as_posix()} holds no score file")
        for name in names:
            category = name[len(PREFIX):-len(SCORE_SUFFIX)]
            score_path = score_dir / name
            result_path = root / language / "result" / f"{PREFIX}{category}{RESULT_SUFFIX}"
            if not result_path.is_file():
                raise AceItemsError(f"{result_path.as_posix()} is missing: no answers of "
                                    f"{language}/{category}")
            stratum = (language, category)
            strata[stratum], headers[stratum] = _stratum(result_path, score_path,
                                                         candidates.get(stratum, frozenset()))
            for path in (result_path, score_path):
                files.append((path.relative_to(root).as_posix(), sha256_of(path)))
    files.sort()
    digest = hashlib.sha256("".join(f"{name}\t{sha}\n" for name, sha in files).encode("utf-8")).hexdigest()
    return PerItem(root.as_posix(), dict(sorted(strata.items())), dict(sorted(headers.items())), files,
                   digest)


def pair_strata(a: PerItem, b: PerItem) -> tuple[dict[Stratum, list[bool]], dict[Stratum, list[bool]]]:
    """The successes of two runs paired by stratum and by answer id, in a's order, as
    eval.metrics.stratified_noninferiority_p takes them; ValueError when the runs do not hold the same
    strata or, in a stratum, the same answers."""
    if set(a.strata) != set(b.strata):
        only_a = sorted(stratum_label(s) for s in set(a.strata) - set(b.strata))
        only_b = sorted(stratum_label(s) for s in set(b.strata) - set(a.strata))
        raise ValueError(f"{a.path} and {b.path} do not hold the same (language, category) strata: only "
                         f"the first has {only_a}, only the second {only_b}")
    out_a: dict[Stratum, list[bool]] = {}
    out_b: dict[Stratum, list[bool]] = {}
    for stratum, answers in a.strata.items():
        other = b.strata[stratum]
        if set(answers) != set(other):
            raise ValueError(f"{a.path} and {b.path} differ in the answers of {stratum_label(stratum)}: "
                             f"{len(answers)} and {len(other)} ids, "
                             f"{len(set(answers) ^ set(other))} not shared")
        out_a[stratum] = list(answers.values())
        out_b[stratum] = [other[i] for i in answers]
    return out_a, out_b
