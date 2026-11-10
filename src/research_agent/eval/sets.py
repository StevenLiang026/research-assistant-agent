"""Assemble the evaluation sets of M2 from the reviewed drafted items and the handwritten ones (docs/plan.md,
评测: development 150, test 300; at least 30% of the items written from scratch).

The rule, per split (dev, test) and stratum (item type, language). The stratum's size is its share of the
split's quotas (QUOTAS_DEV, QUOTAS_TEST) as the drafter splits each type's quota over the languages
(items.select): full at share 1.0, target at share 0.7 (items.scaled). The drafted items an independent review
kept are taken in the order of the review pool (the ranks in data/eval/review/pool-ids.jsonl, which
tools/eval-review/build_pool.py documents) up to target; the rest of full is filled with the handwritten items
of the split and stratum that the same review kept, in id order. So every stratum is at least 30% handwritten,
and more where the review dropped drafted items. A verdict other than keep (drop, review_failed, none at all)
never enters a set.

assemble refuses (SetError) a pool id that the generator runs do not draft, a drafted item recorded under
another split than its pool row, an item id seen twice, a handwritten item of an unknown split, and a stratum
without enough kept items to fill it. The sets come ordered by the plan's type order, then language, then id.

The second family (v2, M7: docs/m7-计划.md WP-I 5; tools/eval-review-v2/README.md) has three single-step
sets and one review pool for them. test-v2 is assembled by v1's rule over the test split at QUOTAS_TEST
(assemble). dev-v2 and dev-v2-ext hold QUOTAS_DEV scaled to 140 and to 70 items (DEV_QUOTAS_V2) and no
handwritten item: per stratum of the dev split, the drafted items the review kept go in review-pool order
first to dev-v2, up to its stratum quota, and the next ones to dev-v2-ext, up to its own (fill_in_order); a
stratum with too few kept items leaves the sets short, dev-v2-ext first, instead of being refused.
assemble_v2 makes all three. Some items of the v2 review were reviewed twice, independently:
combined_verdicts gives every item the verdict it counts with, keep only when every review of it keeps it.
Every stratum of the three sets is recorded with v1's fields and "short", what the set lacks there
(full - drafted - handwritten).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from fractions import Fraction

from .items import LANGS, QUOTAS_DEV, QUOTAS_TEST, SPLITS, TYPES, scaled, select

DRAFTED_SHARE = 0.7
QUOTAS = {"dev": QUOTAS_DEV, "test": QUOTAS_TEST}
KEEP = "keep"
# The second family's sets (module docstring): test-v2 at QUOTAS_TEST, and the two drafted development sets in
# the order the dev split's kept items fill them
TEST_V2 = "test-v2"
DEV_QUOTAS_V2 = {"dev-v2": scaled(QUOTAS_DEV, Fraction(140, 150)),
                 "dev-v2-ext": scaled(QUOTAS_DEV, Fraction(70, 150))}


class SetError(ValueError):
    """The sets cannot be assembled from these inputs."""


def per_language(quotas: Mapping[str, int]) -> dict[tuple[str, str], int]:
    """(type, language) -> the stratum's quota, as items.select splits each type's quota over LANGS."""
    _, report = select([], dict(quotas), seed=0)
    return {(t, lang): report[t][lang]["quota"] for t in quotas for lang in LANGS}


def sort_key(item: dict) -> tuple[int, int, str]:
    """The plan's type order, then language, then id."""
    return TYPES.index(item["type"]), LANGS.index(item["lang"]), item["id"]


def assemble(pool_rows: Iterable[dict], verdicts: Mapping[str, str], drafted: Mapping[str, dict],
             handwritten: Iterable[dict], handwritten_verdicts: Mapping[str, str], *,
             quotas: Mapping[str, Mapping[str, int]] = QUOTAS,
             share: float = DRAFTED_SHARE) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """The sets by split and the table of every stratum (module docstring for the rule).

    pool_rows are the rows of pool-ids.jsonl ({split, type, lang, rank, id}); drafted maps each pool id to the
    item the generator runs draft for it; verdicts and handwritten_verdicts map item ids to the review's
    verdict. The table is {split: {"<type>/<lang>": {"full", "target", "drafted_kept", "drafted",
    "handwritten_kept", "handwritten"}}}.
    """
    rows = sorted(pool_rows, key=lambda r: (r["split"], r["type"], r["lang"], r["rank"]))
    for row in rows:
        item = drafted.get(row["id"])
        if item is None:
            raise SetError(f"pool id {row['id']} is not drafted by the generator runs")
        if (item["split"], item["type"], item["lang"]) != (row["split"], row["type"], row["lang"]):
            raise SetError(f"pool id {row['id']} is a {item['split']}/{item['type']}/{item['lang']} item, "
                           f"but its pool row says {row['split']}/{row['type']}/{row['lang']}")
    handwritten = sorted(handwritten, key=lambda item: item["id"])
    repeated = sorted(i for i, n in Counter([r["id"] for r in rows] + [h["id"] for h in handwritten]).items()
                      if n > 1)
    if repeated:
        raise SetError(f"item ids seen twice: {repeated}")
    unknown = sorted(h["id"] for h in handwritten if h.get("split") not in quotas)
    if unknown:
        raise SetError(f"handwritten items of no known split: {unknown}")
    sets: dict[str, list[dict]] = {}
    table: dict[str, dict] = {}
    for split in (s for s in SPLITS if s in quotas):
        full, target = per_language(quotas[split]), per_language(scaled(dict(quotas[split]), share))
        chosen: list[dict] = []
        table[split] = {}
        for type_ in quotas[split]:
            for lang in LANGS:
                stratum = (split, type_, lang)
                kept = [drafted[r["id"]] for r in rows if (r["split"], r["type"], r["lang"]) == stratum
                        and verdicts.get(r["id"]) == KEEP]
                taken = kept[:target[type_, lang]]
                fill = [h for h in handwritten if (h["split"], h["type"], h["lang"]) == stratum
                        and handwritten_verdicts.get(h["id"]) == KEEP]
                missing = full[type_, lang] - len(taken)
                if len(fill) < missing:
                    raise SetError(f"{split}/{type_}/{lang} needs {missing} handwritten items the review "
                                   f"kept and has {len(fill)}")
                chosen += taken + fill[:missing]
                table[split][f"{type_}/{lang}"] = {
                    "full": full[type_, lang], "target": target[type_, lang], "drafted_kept": len(kept),
                    "drafted": len(taken), "handwritten_kept": len(fill), "handwritten": missing}
        sets[split] = sorted(chosen, key=sort_key)
    return sets, table


def verdict_counts(rows: Iterable[dict]) -> dict:
    """{"verdicts": {verdict: n}, "categories": {verdict: {category: n}}} of review rows."""
    rows = list(rows)
    verdicts = Counter(str(r.get("verdict")) for r in rows)
    categories: dict[str, Counter] = {}
    for r in rows:
        categories.setdefault(str(r.get("verdict")), Counter())[str(r.get("category"))] += 1
    return {"verdicts": dict(sorted(verdicts.items())),
            "categories": {v: dict(sorted(c.items())) for v, c in sorted(categories.items())}}


# ---- the second family (v2) -------------------------------------------------------------------------------


def combined_verdicts(first: Iterable[dict], second: Iterable[dict] = ()) -> dict[str, str]:
    """id -> the verdict an item counts with, when some items were reviewed twice (the module docstring): keep
    only when every review of it keeps it, else the first of its verdicts that is not keep.

    first are the review lines of the first reviews ({"id", "verdict", ...}), one per item; second those of
    the second reviews, of items the first reviews judge. SetError for a line without an id or a verdict, an
    item the first or the second reviews judge twice, or a second review of an item no first review judges."""
    verdicts: dict[str, list[str]] = {}
    for number, row in enumerate(first, 1):
        item, verdict = row.get("id"), row.get("verdict")
        if not isinstance(item, str) or not isinstance(verdict, str):
            raise SetError(f"first review line {number} has no id or no verdict")
        if item in verdicts:
            raise SetError(f"the first reviews judge {item} twice")
        verdicts[item] = [verdict]
    again: set[str] = set()
    for number, row in enumerate(second, 1):
        item, verdict = row.get("id"), row.get("verdict")
        if not isinstance(item, str) or not isinstance(verdict, str):
            raise SetError(f"second review line {number} has no id or no verdict")
        if item not in verdicts:
            raise SetError(f"a second review judges {item}, which no first review judges")
        if item in again:
            raise SetError(f"the second reviews judge {item} twice")
        again.add(item)
        verdicts[item].append(verdict)
    return {item: next((v for v in given if v != KEEP), KEEP) for item, given in verdicts.items()}


def fill_in_order(pool_rows: Iterable[dict], verdicts: Mapping[str, str], drafted: Mapping[str, dict], *,
                  split: str, quotas: Mapping[str, Mapping[str, int]]) -> tuple[dict[str, list[dict]],
                                                                                dict[str, dict]]:
    """Sets of one split without a handwritten part, filled one after the other (the module docstring): the
    sets by name, in quotas' order, and the table of every stratum of each.

    Per (type, language) stratum of split, the drafted items the review kept (verdicts) in review-pool order
    (the rows' ranks) go to the first set up to its stratum quota (items.select's split of each type's quota
    over the languages, as per_language gives it), the next ones to the second set up to its own, and so on;
    a set whose stratum runs out of kept items is short there. Rows of other splits are not read. The table
    is {set: {"<type>/<lang>": {"full", "target", "drafted_kept", "drafted", "handwritten_kept",
    "handwritten", "short"}}}, target being full (the set has no handwritten part), drafted_kept the
    stratum's kept items in the pool (shared by the sets) and short = full - drafted. SetError for a row whose
    id the generator runs do not draft or whose item is of another split, type or language, a pool id seen
    twice, and sets whose quotas name different types. The sets come ordered as assemble orders them
    (sort_key)."""
    rows = sorted((r for r in pool_rows if r["split"] == split),
                  key=lambda r: (r["type"], r["lang"], r["rank"]))
    for row in rows:
        item = drafted.get(row["id"])
        if item is None:
            raise SetError(f"pool id {row['id']} is not drafted by the generator runs")
        if (item["split"], item["type"], item["lang"]) != (row["split"], row["type"], row["lang"]):
            raise SetError(f"pool id {row['id']} is a {item['split']}/{item['type']}/{item['lang']} item, "
                           f"but its pool row says {row['split']}/{row['type']}/{row['lang']}")
    repeated = sorted(i for i, n in Counter(r["id"] for r in rows).items() if n > 1)
    if repeated:
        raise SetError(f"pool ids seen twice: {repeated}")
    types = [list(q) for q in quotas.values()]
    if any(t != types[0] for t in types):
        raise SetError(f"the sets of {split} have different types: {dict(zip(quotas, types, strict=True))}")
    full = {name: per_language(q) for name, q in quotas.items()}
    sets: dict[str, list[dict]] = {name: [] for name in quotas}
    table: dict[str, dict] = {name: {} for name in quotas}
    for type_ in types[0] if types else []:
        for lang in LANGS:
            kept = [drafted[r["id"]] for r in rows
                    if (r["type"], r["lang"]) == (type_, lang) and verdicts.get(r["id"]) == KEEP]
            start = 0
            for name in quotas:
                size = full[name][type_, lang]
                taken = kept[start:start + size]
                start += len(taken)
                sets[name] += taken
                table[name][f"{type_}/{lang}"] = {
                    "full": size, "target": size, "drafted_kept": len(kept), "drafted": len(taken),
                    "handwritten_kept": 0, "handwritten": 0, "short": size - len(taken)}
    return {name: sorted(items, key=sort_key) for name, items in sets.items()}, table


def assemble_v2(pool_rows: Iterable[dict], verdicts: Mapping[str, str], drafted: Mapping[str, dict],
                handwritten: Iterable[dict], handwritten_verdicts: Mapping[str, str], *,
                test_quotas: Mapping[str, int] = QUOTAS_TEST,
                dev_quotas: Mapping[str, Mapping[str, int]] = DEV_QUOTAS_V2,
                share: float = DRAFTED_SHARE) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """The second family's sets (the module docstring): TEST_V2, then the development sets of dev_quotas in
    their order, and the table of every stratum of each (fill_in_order's fields).

    TEST_V2 is assemble's test set over the test split at test_quotas and share, from every pool row (so every
    pool id, of either split, must be drafted) and the handwritten items, which must all be of the test split;
    its strata gain "short", 0 since assemble fills every stratum or refuses. The development sets are
    fill_in_order's over the dev split. SetError as those two raise it."""
    pool_rows = list(pool_rows)
    sets, tables = assemble(pool_rows, verdicts, drafted, handwritten, handwritten_verdicts,
                            quotas={"test": test_quotas}, share=share)
    dev_sets, dev_table = fill_in_order(pool_rows, verdicts, drafted, split="dev", quotas=dev_quotas)
    table = {TEST_V2: {key: {**row, "short": row["full"] - row["drafted"] - row["handwritten"]}
                       for key, row in tables["test"].items()}, **dev_table}
    return {TEST_V2: sets["test"], **dev_sets}, table
