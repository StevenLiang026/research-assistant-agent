"""scripts/figs/route_curve_tex.py: route_curve.m's figure of FT+R's routing curve as a standalone XeLaTeX
document, built from a fabricated routing file, and the build that turns it into the PNG and the PDF. The
document is checked as text and the build with xelatex and Ghostscript faked, so no TeX runs here and the
check passes on a machine without TeX Live (the cloud check on Ubuntu). One test reads the repository's
results/routing-m4.json: the text was placed by looking at that file's figure."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def load_script(path: str):
    spec = importlib.util.spec_from_file_location(f"script_{Path(path).stem}", REPO / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


route_curve_tex = load_script("scripts/figs/route_curve_tex.py")
MINUS = "\N{MINUS SIGN}"
THRESHOLD = 0.7123456


def point(label: str, threshold: float | None, routed: int, successes: int, n: int, ft: float,
          b2: float) -> dict:
    """A point as scripts/route_curve.py writes it, its random-routing success included. Its cost grows
    faster than its API share, as when the first items routed are the dearest, so that a point's cost and
    random routing's expected cost at its share differ."""
    share = routed / n
    return {"label": label, "threshold": threshold, "routed": routed, "api_share": share,
            "successes": successes, "success": successes / n,
            "cost_per_100_rmb": round(0.04 * routed * (routed + 3), 6),
            "random_success": (1 - share) * ft + share * b2}


def routing() -> dict:
    """Seven items, FT 3/7 and B2 6/7, a margin of 0.03; three of them handwritten, FT 1/3 and B2 2/3 there.
    Routing every item costs 2.8 per 100 items, every handwritten one 0.72. Every number differs from the
    repository's results/routing-m4.json, and most need rounding."""
    n, ft, b2 = 7, 3 / 7, 6 / 7
    scan = [point("route_nothing", None, 0, 3, n, ft, b2), point("threshold", 0.2, 1, 4, n, ft, b2),
            point("threshold", 0.45, 2, 5, n, ft, b2), point("threshold", THRESHOLD, 3, 6, n, ft, b2),
            point("threshold", 0.9, 5, 6, n, ft, b2), point("route_everything", None, 7, 6, n, ft, b2)]
    oracle = [{"routed": k, "api_share": k / n, "successes": s, "success": s / n}
              for k, s in enumerate([3, 4, 5, 6, 6, 6, 6, 6])]
    hn, hft, hb2 = 3, 1 / 3, 2 / 3
    hand = [point("route_nothing", None, 0, 1, hn, hft, hb2), point("threshold", 0.45, 1, 2, hn, hft, hb2),
            point("threshold", THRESHOLD, 2, 2, hn, hft, hb2),
            point("route_everything", None, 3, 2, hn, hft, hb2)]
    return {"rule": "the preregistered rule", "margin": 0.03, "rule_met": True, "signal": "min_decision_prob",
            "threshold": THRESHOLD, "n": n, "ft_success": ft, "b2_success": b2, "chosen_point": scan[3],
            "curves": {"routing": scan, "oracle": oracle},
            "handwritten": {"n": hn, "ft_success": hft, "b2_success": hb2, "scan": hand,
                            "at_threshold": hand[2]}}


def cost_oracle() -> dict:
    """scripts/route_cost_oracle.py's file for routing(): three gains, routed for 0.1, 0.25 and 0.6 per 100
    items. Its routing.sha256 is filled in when it is written beside a routing file (write_routing)."""
    costs = [0.0, 0.1, 0.25, 0.6]
    return {"rule": "the most items right at a cost", "routing": {"path": "routing.json", "sha256": None},
            "inputs": {}, "cloud_model": "deepseek-flash", "n": 7, "ft_successes": 3, "gains": 3,
            "b2_cost_per_100_rmb": 2.8,
            "curve": [{"routed": j, "cost_per_100_rmb": c, "successes": 3 + j, "success": (3 + j) / 7}
                      for j, c in enumerate(costs)]}


def not_met() -> dict:
    """The file route_curve.py writes when no threshold meets the rule."""
    data = routing()
    data.update(rule_met=False, threshold=None, chosen_point=None)
    data["handwritten"]["at_threshold"] = None
    return data


def panels(document: str) -> list[str]:
    """The document's four panels, row by row: all items and the handwritten ones against the API share,
    then the same against the cost per 100 items."""
    parts = document.split(r"\nextgroupplot")
    assert len(parts) == 5
    return parts[1:]


def share_row(document: str) -> list[str]:
    """The upper row, route_curve.m's figure: all items (left) and the handwritten ones (right)."""
    return panels(document)[:2]


def cost_row(document: str) -> list[str]:
    """The lower row, against the cost per 100 items: all items (left) and the handwritten ones (right)."""
    return panels(document)[2:]


def options(panel: str) -> str:
    """The options of a panel's \\nextgroupplot."""
    return panel.split("\n", 1)[0]


def plots(panel: str) -> dict[str, list[tuple[float, float]]]:
    """Each plot of a panel by its style: the coordinates it draws, in order."""
    found = {}
    for style, body in re.findall(r"\\addplot\[([^\]]*)\]\s*coordinates\s*\{([^}]*)\};", panel):
        assert style not in found
        found[style] = [(float(x), float(y)) for x, y in re.findall(r"\(([-\d.]+),([-\d.]+)\)", body)]
    return found


def nodes(panel: str) -> dict[str, tuple[float, float]]:
    """Each text node of a panel: its text and where it is placed in the axis' coordinates."""
    return {text: (float(x), float(y)) for x, y, text in
            re.findall(r"\\node\[[^\]]*\] at \(axis cs:([-\d.]+),([-\d.]+)\) \{(.*?)\};", panel)}


def legend(panel: str) -> list[str]:
    return re.findall(r"\\addlegendentry\{(.*?)\}\n", panel)


def leaders(panel: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Each leader line of a panel: its two ends in the axis' coordinates."""
    return [((float(x0), float(y0)), (float(x1), float(y1))) for x0, y0, x1, y1 in re.findall(
        r"\\draw\[leader\] \(axis cs:([-\d.]+),([-\d.]+)\) -- \(axis cs:([-\d.]+),([-\d.]+)\);", panel)]


def styles(document: str) -> dict[str, set[str]]:
    """Each style the preamble defines (name/.style={...}): its keys, split at the commas outside braces."""
    return {name: {key.strip() for key in re.split(r",(?![^{}]*\})", body)} for name, body in
            re.findall(r"(\w[\w ]*)/\.style=\{((?:[^{}]|\{[^{}]*\})*)\}", document)}


def at(x: float, y: float) -> tuple[float, float]:
    """A JSON point as the figure draws it: in percent, rounded to the printed decimals."""
    return round(100 * x, 4), round(100 * y, 4)


def cost_at(cost: float, y: float) -> tuple[float, float]:
    """A JSON point as the lower row draws it: its cost per 100 items as it is, its success in percent."""
    return round(cost, 4), round(100 * y, 4)


def tex(text: str) -> str:
    """A label as the document writes it (escape has its own tests below)."""
    return route_curve_tex.escape(text)


# ---- the curves and the points ----------------------------------------------------------------------------


def test_the_target_line_is_the_routing_files_own_target_under_the_target_objective():
    # M7's R3 (scripts/route_curve.py --objective target) writes a target and no margin; M4's and M6's files
    # keep the preregistered rule's B2 minus the margin
    target, label = route_curve_tex._target({"objective": "target", "target": 0.92, "b2_success": 0.9})
    assert target == pytest.approx(92.0) and label == "目标 92.0%"
    target, label = route_curve_tex._target({"b2_success": 0.9, "margin": 0.02})
    assert target == pytest.approx(88.0) and label.startswith("目标 88.0%(B2 ")


def test_every_scan_random_and_oracle_point_is_drawn_at_100_times_its_value():
    data = routing()
    left, right = (plots(p) for p in share_row(route_curve_tex.document(data)))
    scan, oracle, hand = data["curves"]["routing"], data["curves"]["oracle"], data["handwritten"]["scan"]
    assert left["scan curve"] == [at(p["api_share"], p["success"]) for p in scan]
    assert left["random curve"] == [at(p["api_share"], p["random_success"]) for p in scan]
    assert left["oracle curve"] == [at(p["api_share"], p["success"]) for p in oracle]
    assert right["scan curve"] == [at(p["api_share"], p["success"]) for p in hand]
    assert right["random curve"] == [at(p["api_share"], p["random_success"]) for p in hand]
    assert "oracle curve" not in right and "target line" not in right  # as route_curve.m: left panel only
    assert left["scan curve"][1] == (14.2857, 57.1429)  # 1/7 and 4/7: four decimals, rounded


def test_the_frozen_threshold_is_drawn_where_the_json_puts_it_on_both_panels():
    data = routing()
    left, right = (plots(p) for p in share_row(route_curve_tex.document(data)))
    chosen, hand = data["chosen_point"], data["handwritten"]["at_threshold"]
    assert left["frozen point"] == [at(chosen["api_share"], chosen["success"])] == [(42.8571, 85.7143)]
    assert right["frozen point"] == [at(hand["api_share"], hand["success"])] == [(66.6667, 66.6667)]


def test_the_target_line_spans_the_axis_at_b2_minus_the_margin_with_its_label_on_it():
    document = route_curve_tex.document(routing())
    left, right = share_row(document)
    target = round(100 * (6 / 7 - 0.03), 4)
    assert plots(left)["target line"] == [(0.0, target), (100.0, target)]
    assert target == 82.7143
    label = tex(f"目标 82.7%(B2 {MINUS} 3 个百分点)")  # the margin is the file's, not a fixed 2
    # the label's lower left corner (its anchor) on the line, at TARGET_LABEL_X: it moves along the line only
    assert nodes(left)[label] == (route_curve_tex.TARGET_LABEL_X, target)
    assert "anchor=south west" in styles(document)["target label"]
    assert label not in nodes(right)


def test_the_curves_are_drawn_in_route_curve_ms_line_styles_and_colours():
    document = route_curve_tex.document(routing())
    style = styles(document)
    rgb = dict(re.findall(r"\\definecolor\{(\w+)\}\{rgb\}\{([^}]*)\}", document))

    def colour(name: str) -> str:
        return next(rgb[key.removeprefix("draw=")] for key in style[name] if key.startswith("draw="))

    # MATLAB's first three line colours, in route_curve.m's order of plots, and its '-', '--' and ':'
    solid = not {"dashed", "dotted", "dash dot"} & style["scan curve"]
    assert colour("scan curve") == "0,0.447,0.741" and solid
    assert colour("random curve") == "0.85,0.325,0.098" and "dashed" in style["random curve"]
    assert colour("oracle curve") == "0.929,0.694,0.125" and "dotted" in style["oracle curve"]
    # yline's '-.' with HandleVisibility off: no legend entry, so that each entry stays with the plot it names
    assert {"dash dot", "forget plot"} <= style["target line"]
    assert colour("frozen point") == "0.494,0.184,0.556" and {"only marks", "mark=o"} <= style["frozen point"]


# ---- the text ---------------------------------------------------------------------------------------------


def test_titles_legends_and_annotations_show_the_files_numbers_as_route_curve_m_formats_them():
    left, right = share_row(route_curve_tex.document(routing()))
    assert "title={" + tex("全部 7 道") + "}" in left
    assert "title={" + tex("只看手写题 3 道(描述)") + "}" in right
    curves = ["FT+R:扫描阈值", "随机路由(同调用比例)"]
    assert legend(left) == [tex(t) for t in (*curves, "理想路由", "冻结阈值 0.712")]
    assert legend(right) == [tex(t) for t in (*curves, "冻结阈值 0.712 处")]
    assert tex("调用比例 42.9%") + r"\\" + tex("成功率 85.7%") in nodes(left)
    assert tex("调用比例 66.7%") + r"\\" + tex("成功率 66.7%") in nodes(right)


def test_the_end_labels_sit_at_the_left_and_right_edges_at_fts_and_b2s_success():
    left, right = (nodes(p) for p in share_row(route_curve_tex.document(routing())))
    for labels, ft, b2, ft_text, b2_text in ((left, 3 / 7, 6 / 7, r"FT 42.9\%", r"B2 85.7\%"),
                                             (right, 1 / 3, 2 / 3, r"FT 33.3\%", r"B2 66.7\%")):
        x, y = labels[ft_text]
        assert x <= 5 and y == round(100 * ft, 4)
        x, y = labels[b2_text]
        assert x >= 95 and y == round(100 * b2, 4)


def test_the_panels_share_the_axes_of_route_curve_m_which_the_lower_row_gives_its_cost_axis():
    document = route_curve_tex.document(routing())
    shared = document.split(r"\nextgroupplot")[0]
    for option in ("xmin=0", "xmax=100", "ymin=40", "ymax=100", "grid=major", "legend pos=south east",
                   r"xlabel={API 调用比例(\%)}", r"ylabel={开发集单步成功率(\%)}"):
        assert option in shared
    assert r"\begin{groupplot}" in shared and "group size=2 by 2" in shared
    assert not [panel for panel in share_row(document) if "xmax" in options(panel)]  # route_curve.m's axes


def test_the_document_is_a_standalone_xelatex_document_in_microsoft_yahei():
    document = route_curve_tex.document(routing())
    assert document.startswith("%") and r"\documentclass" in document
    assert re.search(r"\\documentclass(\[[^\]]*\])?\{standalone\}", document)
    for line in (r"\usepackage{xeCJK}", r"\setCJKmainfont{Microsoft YaHei}", r"\usepackage{pgfplots}",
                 r"\usepgfplotslibrary{groupplots}", r"\begin{document}"):
        assert line in document
    assert document.endswith("\\end{document}\n") and "\r" not in document


def test_percent_signs_and_the_other_tex_specials_are_escaped():
    assert route_curve_tex.escape("50% & $5 #1 a_b {c} ~ ^ \\") == (
        r"50\% \& \$5 \#1 a\_b \{c\} \textasciitilde{} \textasciicircum{} \textbackslash{}")
    document = route_curve_tex.document(routing())
    body = [line for line in document.splitlines() if not line.lstrip().startswith("%")]
    assert body and not [line for line in body if re.search(r"(?<!\\)%", line)]


def test_a_space_after_a_chinese_character_is_a_control_space_that_xecjk_keeps():
    # xeCJK drops a plain space after a CJK character, and the preamble's empty CJKecglue adds none back
    # (it also keeps xeCJK from adding a space next to "(" or ":"), so "全部 150 道" would print as
    # "全部150 道"; a space after a Latin character or a digit stays as it is
    assert route_curve_tex.escape("全部 150 道") == "全部\\ 150 道"
    target = route_curve_tex.escape(f"目标 88.0%(B2 {MINUS} 2 个百分点)")
    assert target == f"目标\\ 88.0\\%(B2 {MINUS} 2 个百分点)"
    assert route_curve_tex.escape("API 调用比例(%)") == "API 调用比例(\\%)"
    assert "\\xeCJKsetup{CJKecglue={}}" in route_curve_tex.document(routing())


def test_the_same_input_gives_the_same_bytes_and_is_left_unchanged():
    data = routing()
    before = copy.deepcopy(data)
    first = route_curve_tex.document(data).encode("utf-8")
    assert route_curve_tex.document(json.loads(json.dumps(data))).encode("utf-8") == first
    assert data == before


def test_a_file_whose_rule_was_not_met_is_refused_since_route_curve_m_fails_on_it():
    with pytest.raises(route_curve_tex.NoFrozenThreshold, match="rule"):
        route_curve_tex.document(not_met())


# ---- where the text sits ----------------------------------------------------------------------------------


def test_the_frozen_points_note_is_centred_above_the_point_with_a_leader_down_to_it():
    data = routing()
    document = route_curve_tex.document(data)
    rise = route_curve_tex.NOTE_RISE
    assert rise > 0
    marked = (data["chosen_point"], data["handwritten"]["at_threshold"])
    shown = (("42.9", "85.7"), ("66.7", "66.7"))
    for panel, point, (share, success) in zip(share_row(document), marked, shown, strict=True):
        x, y = at(point["api_share"], point["success"])
        above = round(100 * point["success"] + rise, 4)
        note = tex(f"调用比例 {share}%") + r"\\" + tex(f"成功率 {success}%")
        # the point stays where the file puts it; the note's lower edge is centred on it (the style's
        # anchor), NOTE_RISE higher, and the leader runs straight down from there to the point
        assert plots(panel)["frozen point"] == [(x, y)]
        assert nodes(panel)[note] == (x, above)
        assert leaders(panel) == [((x, y), (x, above))]
    assert "anchor=south" in styles(document)["note"]


def test_the_placements_clear_the_curves_of_the_repositorys_routing_file_as_its_render_showed():
    # TARGET_LABEL_X and NOTE_RISE were chosen by looking at the figure of results/routing-m4.json; this
    # checks what that look relied on, so that a changed placement, or a new routing file, fails here until
    # someone looks again. The label's right end (left of the frozen point) needs TeX's widths: seen only.
    data = json.loads((REPO / "results" / "routing-m4.json").read_text(encoding="utf-8"))
    oracle = [(100 * p["api_share"], 100 * p["success"]) for p in data["curves"]["oracle"]]
    plateau = max(y for _, y in oracle)
    risen = min(x for x, y in oracle if y == plateau)  # where the oracle's rise ends (x 26 there)
    chosen = data["chosen_point"]
    share, target = 100 * chosen["api_share"], 100 * (data["b2_success"] - data["margin"])
    # the target's label starts right of the oracle's rise, which crosses the target line and the label's
    # height, and left of the frozen point; on that stretch the oracle runs at its plateau, above the label,
    # and the scan below the target line, under the label
    assert risen < route_curve_tex.TARGET_LABEL_X < share
    assert all(y == plateau for x, y in oracle if risen <= x <= share)
    assert all(100 * p["success"] < target for p in data["curves"]["routing"] if 100 * p["api_share"] < share)
    # the frozen point's note starts above the oracle's plateau, which it would otherwise cover
    assert 100 * chosen["success"] + route_curve_tex.NOTE_RISE > plateau


# ---- the lower row: success against the cost per 100 items ----------------------------------------------


def test_the_lower_row_draws_the_scan_at_its_cost_and_random_routing_at_its_expected_cost():
    # docs/plan.md asks for success against the API share and against the cost per 100 items. Random routing
    # sends each item to B2 with the probability of its share, so in expectation it costs the share times
    # the cost of routing every item
    data = routing()
    left, right = (plots(p) for p in cost_row(route_curve_tex.document(data)))
    for drawn, points, total in ((left, data["curves"]["routing"], 2.8),
                                 (right, data["handwritten"]["scan"], 0.72)):
        assert drawn["scan curve"] == [cost_at(p["cost_per_100_rmb"], p["success"]) for p in points]
        assert drawn["random curve"] == [cost_at(p["api_share"] * total, p["random_success"]) for p in points]
    assert left["scan curve"][1] == (0.16, 57.1429) and left["random curve"][1] == (0.4, 48.9796)
    # without scripts/route_cost_oracle.py's file there is no oracle at a cost to draw: the routing file gives
    # its oracle's points no cost
    assert not {"oracle curve", "cost oracle curve"} & {*left, *right}


def test_the_lower_row_draws_the_oracle_at_a_cost_in_steps_on_the_left_panel_only():
    # scripts/route_cost_oracle.py's curve, flat past its last point to the cost of routing every item (more
    # money buys no more success). Between two points the success is the earlier point's, so the curve is
    # drawn in steps. As in the upper row, the handwritten panel has no oracle
    left, right = cost_row(route_curve_tex.document(routing(), cost_oracle()))
    assert plots(left)["cost oracle curve"] == [(0.0, 42.8571), (0.1, 57.1429), (0.25, 71.4286),
                                                (0.6, 85.7143), (2.8, 85.7143)]
    assert legend(left) == [tex(t) for t in ("FT+R:扫描阈值", "随机路由(同成本)", "理想路由(同成本)",
                                             "冻结阈值 0.712")]
    assert not {"oracle curve", "cost oracle curve"} & set(plots(right))
    assert {"oracle curve", "const plot"} <= styles(route_curve_tex.document(routing()))["cost oracle curve"]


def test_an_oracle_that_ends_at_the_cost_of_routing_every_item_gets_no_extra_point():
    costs = cost_oracle()
    costs["curve"][-1]["cost_per_100_rmb"] = 2.8
    left, _ = cost_row(route_curve_tex.document(routing(), costs))
    assert plots(left)["cost oracle curve"][-2:] == [(0.25, 71.4286), (2.8, 85.7143)]


def test_the_lower_row_marks_the_frozen_threshold_at_its_cost_with_its_note_above_it():
    data = routing()
    document = route_curve_tex.document(data)
    rise = route_curve_tex.NOTE_RISE
    marked = (data["chosen_point"], data["handwritten"]["at_threshold"])
    for panel, point, (cost, success) in zip(cost_row(document), marked, (("0.72", "85.7"), ("0.40", "66.7")),
                                             strict=True):
        x, y = cost_at(point["cost_per_100_rmb"], point["success"])
        above = round(100 * point["success"] + rise, 4)
        assert plots(panel)["frozen point"] == [(x, y)]
        assert nodes(panel)[tex(f"每百道 {cost} 元") + r"\\" + tex(f"成功率 {success}%")] == (x, above)
        assert leaders(panel) == [((x, y), (x, above))]


def test_the_lower_row_repeats_the_upper_rows_titles_and_legends_without_the_oracle():
    left, right = cost_row(route_curve_tex.document(routing()))
    assert "title={" + tex("全部 7 道") + "}" in options(left)
    assert "title={" + tex("只看手写题 3 道(描述)") + "}" in options(right)
    # at one x of the lower row the dashed curve is random routing at the same (expected) cost, not at the
    # same API share as the scan's point above it
    curves = ["FT+R:扫描阈值", "随机路由(同成本)"]
    assert legend(left) == [tex(t) for t in (*curves, "冻结阈值 0.712")]
    assert legend(right) == [tex(t) for t in (*curves, "冻结阈值 0.712 处")]


def test_the_lower_rows_panels_share_a_cost_axis_from_0_to_a_round_limit_at_or_above_every_cost():
    for panel in cost_row(route_curve_tex.document(routing())):
        assert "xmin=0, xmax=3, xtick={0,0.5,1,1.5,2,2.5,3}" in options(panel)  # 2.8 at most: steps of 0.5
        assert "xlabel={每百道成本(元,列表价)}" in options(panel)


def test_the_shared_cost_axis_covers_the_dearer_of_the_two_panels_whichever_it_is():
    data = routing()
    data["handwritten"]["scan"][-1]["cost_per_100_rmb"] = 5.0  # the handwritten items now cost the most
    for panel in cost_row(route_curve_tex.document(data)):
        assert "xmax=5, xtick={0,1,2,3,4,5}" in options(panel)


@pytest.mark.parametrize(("top", "limit", "step"), [
    (0.463, 0.5, 0.1), (2.8, 3.0, 0.5), (0.5, 0.5, 0.1), (1.0, 1.0, 0.2), (0.05, 0.05, 0.01), (7.2, 8.0, 2.0),
    (6.0, 6.0, 1.0), (1.4, 1.5, 0.5),
    # the repository's cost of routing every item is 0.46299599999999996: float noise must not add a tick
    (0.5000000000000001, 0.5, 0.1), (3.0000000000000004, 3.0, 0.5), (1e-12, 1e-12, 2e-13)])
def test_the_cost_axis_steps_by_the_smallest_1_2_or_5_times_a_power_of_ten_that_fits(top, limit, step):
    xmax, ticks = route_curve_tex.cost_axis(top)
    close = {"rel": 1e-9, "abs": 0}  # pytest.approx's default absolute 1e-12 would pass a limit of 0 here
    assert xmax == pytest.approx(limit, **close) and ticks[0] == 0 and ticks[-1] == xmax
    assert all(b - a == pytest.approx(step, **close) for a, b in itertools.pairwise(ticks))
    assert len(ticks) - 1 <= route_curve_tex.COST_INTERVALS


@pytest.mark.parametrize("top", [0.0, -1.0, float("nan"), float("inf")])
def test_routing_that_costs_nothing_or_no_number_has_no_cost_axis(top):
    with pytest.raises(ValueError, match="cost"):
        route_curve_tex.cost_axis(top)


def test_a_cheap_routing_file_keeps_its_coordinates_and_its_costs_note_apart():
    # a thousandth of the fabricated costs: the coordinates take more decimals and the note shows the cost
    # to one place more than the ticks' step, so that neither rounds to the same value or to 0.00
    data, costs = routing(), cost_oracle()
    for point in data["curves"]["routing"] + data["handwritten"]["scan"] + costs["curve"]:
        point["cost_per_100_rmb"] = round(point["cost_per_100_rmb"] / 1000, 9)
    left, _ = cost_row(route_curve_tex.document(data, costs))
    assert "xmax=0.003, xtick={0,0.0005,0.001,0.0015,0.002,0.0025,0.003}" in options(left)

    def drawn(points: list[dict]) -> list[tuple[float, float]]:
        return [(round(p["cost_per_100_rmb"], 6), round(100 * p["success"], 6)) for p in points]

    assert plots(left)["scan curve"] == drawn(data["curves"]["routing"])
    oracle = drawn(costs["curve"])
    assert plots(left)["cost oracle curve"] == [*oracle, (0.0028, oracle[-1][1])]  # 0.00025: six decimals
    assert tex("每百道 0.00072 元") + r"\\" + tex("成功率 85.7%") in nodes(left)


def test_the_lower_rows_end_labels_put_ft_at_no_cost_and_b2_at_the_cost_of_routing_every_item():
    left, right = (nodes(p) for p in cost_row(route_curve_tex.document(routing())))
    for labels, ft, b2, total, ft_text, b2_text in ((left, 3 / 7, 6 / 7, 2.8, r"FT 42.9\%", r"B2 85.7\%"),
                                                    (right, 1 / 3, 2 / 3, 0.72, r"FT 33.3\%", r"B2 66.7\%")):
        x, y = labels[ft_text]
        assert 0 < x <= 0.05 * 3 and y == round(100 * ft, 4)
        assert labels[b2_text] == (round(total, 4), round(100 * b2, 4))


def test_the_lower_rows_target_line_spans_its_axis_with_its_label_right_of_the_oracles_rise():
    document = route_curve_tex.document(routing(), cost_oracle())
    left, right = cost_row(document)
    target = round(100 * (6 / 7 - 0.03), 4)
    assert plots(left)["target line"] == [(0.0, target), (3.0, target)]
    # as in the upper row, the label starts right of where the oracle rises across the target (on the
    # repository's file: the next test), a fixed share of the axis in
    label = tex(f"目标 82.7%(B2 {MINUS} 3 个百分点)")
    assert nodes(left)[label] == (round(route_curve_tex.TARGET_LABEL_COST_AT * 3, 4), target)
    assert "target line" not in plots(right) and label not in nodes(right)


def test_the_lower_rows_text_clears_its_curves_on_the_repositorys_routing_file():
    # as for the upper row: what the look at the lower row of results/routing-m4.json relied on
    data = json.loads((REPO / "results" / "routing-m4.json").read_text(encoding="utf-8"))
    costs = json.loads((REPO / "results" / "cost-oracle-m4.json").read_text(encoding="utf-8"))
    scan, hand = data["curves"]["routing"], data["handwritten"]
    total = route_curve_tex.total_cost(scan)
    limit, _ = route_curve_tex.cost_axis(max(total, route_curve_tex.total_cost(hand["scan"])))
    chosen, target = data["chosen_point"], 100 * (data["b2_success"] - data["margin"])
    cost, best = chosen["cost_per_100_rmb"], costs["curve"][-1]
    # the target's label starts right of where the oracle at a cost reaches its best, which then runs above
    # the label (three points above the target at least: the label's height), and left of the frozen point;
    # on that stretch the scan and random routing stay under the target line
    assert best["cost_per_100_rmb"] < route_curve_tex.TARGET_LABEL_COST_AT * limit < cost
    assert 100 * best["success"] >= target + 3
    assert all(100 * p["success"] < target for p in scan if p["cost_per_100_rmb"] < cost)
    assert all(100 * p["random_success"] < target for p in scan if p["api_share"] * total < cost)
    # each frozen point's note starts above every curve of its panel, the oracle at a cost included
    for point, points, oracle in ((chosen, scan, 100 * best["success"]),
                                  (hand["at_threshold"], hand["scan"], 0)):
        highest = max([100 * max(p["success"], p["random_success"]) for p in points] + [oracle])
        assert 100 * point["success"] + route_curve_tex.NOTE_RISE > highest


# ---- the log checks of the build --------------------------------------------------------------------------


def test_a_glyph_the_font_lacks_is_found_in_the_log():
    missing = f"Missing character: There is no {MINUS} (U+2212) in font [lmroman10-regular]:mapping=tex-text;"
    assert route_curve_tex.missing_characters(f"(./x.tex\n{missing}\nOutput written on x.pdf\n") == [missing]
    assert route_curve_tex.missing_characters("Output written on x.pdf (1 page).\n") == []


def test_a_log_that_asks_for_another_run_is_recognized():
    assert route_curve_tex.needs_rerun("LaTeX Warning: Label(s) may have changed. Rerun to get "
                                       "cross-references right.\n")
    assert not route_curve_tex.needs_rerun("Output written on route-curve-m4.pdf (1 page).\n")


# ---- the command line, up to the point where it would run TeX ---------------------------------------------


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def beside(source: Path) -> Path:
    """Where write_routing puts the cost oracle made from the routing file at source."""
    return source.with_name(f"cost-oracle-{source.stem}.json")


def write_routing(path: Path, data: dict, costs: dict | None = None) -> Path:
    """The routing file at path, and beside it the cost oracle (by default cost_oracle()) made from it: its
    routing.sha256 is the routing file's."""
    write_json(path, data)
    costs = copy.deepcopy(costs or cost_oracle())
    costs["routing"] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(beside(path), costs)
    return path


def argv(tmp_path: Path, source: Path, folder: str = "build") -> list[str]:
    return ["--routing", str(source), "--cost-oracle", str(beside(source)), "--build", str(tmp_path / folder),
            "--out", str(tmp_path / "figs" / "curve.png")]


@pytest.mark.parametrize("missing", ["xelatex", "rungs"])
def test_a_missing_tool_is_named_and_nothing_is_written(tmp_path, monkeypatch, capsys, missing):
    source = write_routing(tmp_path / "routing.json", routing())
    monkeypatch.setattr(route_curve_tex.shutil, "which", lambda tool: None if tool == missing else tool)
    assert route_curve_tex.main(argv(tmp_path, source)) == 3
    err = capsys.readouterr().err
    assert missing in err and "PATH" in err and "Traceback" not in err
    assert not (tmp_path / "build").exists() and not (tmp_path / "figs").exists()


def test_a_tool_that_cannot_start_is_reported_without_a_traceback(tmp_path, monkeypatch, capsys):
    source = write_routing(tmp_path / "routing.json", routing())
    monkeypatch.setattr(route_curve_tex.shutil, "which", lambda tool: str(tmp_path / f"no-such-{tool}"))
    assert route_curve_tex.main(argv(tmp_path, source)) == 4
    err = capsys.readouterr().err
    assert "no-such-xelatex" in err and "Traceback" not in err
    assert not (tmp_path / "figs").exists()


def test_a_file_that_cannot_be_drawn_exits_1_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(route_curve_tex.shutil, "which", lambda tool: None)  # never reach a TeX run
    assert route_curve_tex.main(argv(tmp_path, write_routing(tmp_path / "r.json", not_met()))) == 1
    assert "rule" in capsys.readouterr().err
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert route_curve_tex.main(argv(tmp_path, tmp_path / "bad.json")) == 1
    assert route_curve_tex.main(argv(tmp_path, tmp_path / "absent.json")) == 1
    shapeless = routing()
    del shapeless["curves"]
    assert route_curve_tex.main(argv(tmp_path, write_routing(tmp_path / "s.json", shapeless))) == 1
    endless = routing()
    endless["curves"]["routing"].pop()  # no route_everything point: no cost of routing every item
    assert route_curve_tex.main(argv(tmp_path, write_routing(tmp_path / "e.json", endless))) == 1
    doubled = routing()
    doubled["curves"]["routing"].append(doubled["curves"]["routing"][-1])  # two of them: which one counts?
    assert route_curve_tex.main(argv(tmp_path, write_routing(tmp_path / "d.json", doubled))) == 1
    boundless = routing()
    boundless["curves"]["routing"][-1]["cost_per_100_rmb"] = float("inf")  # JSON's Infinity
    assert route_curve_tex.main(argv(tmp_path, write_routing(tmp_path / "i.json", boundless))) == 1
    assert "Traceback" not in capsys.readouterr().err
    assert not (tmp_path / "build").exists() and not (tmp_path / "figs").exists()


def test_a_cost_oracle_that_is_not_made_from_the_routing_file_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(route_curve_tex.shutil, "which", lambda tool: None)  # never reach a TeX run
    source = write_routing(tmp_path / "routing.json", routing())
    costs = json.loads(beside(source).read_text(encoding="utf-8"))
    write_json(beside(source), {**costs, "routing": {**costs["routing"], "sha256": "0" * 64}})
    assert route_curve_tex.main(argv(tmp_path, source)) == 1
    assert "another routing file" in capsys.readouterr().err
    write_json(beside(source), {**costs, "n": 8})  # the routing file's own SHA-256, but other items
    assert route_curve_tex.main(argv(tmp_path, source)) == 1
    # route_curve's own SHA-256 and n, but no curve to draw, or one that is not a point per gain routed
    for broken in ({**costs, "curve": []}, {**costs, "gains": 2}, {**costs, "curve": "none"},
                   {key: value for key, value in costs.items() if key != "curve"}):
        write_json(beside(source), broken)
        assert route_curve_tex.main(argv(tmp_path, source)) == 1
    beside(source).write_text("{", encoding="utf-8")  # not JSON
    assert route_curve_tex.main(argv(tmp_path, source)) == 1
    beside(source).unlink()
    assert route_curve_tex.main(argv(tmp_path, source)) == 1  # none at all
    assert "Traceback" not in capsys.readouterr().err
    assert not (tmp_path / "build").exists() and not (tmp_path / "figs").exists()


@pytest.mark.parametrize(("windows", "tools"), [(True, ("xelatex", "rungs")),
                                                 (False, ("xelatex", "rungs", "gs"))])
def test_rungs_runs_tex_lives_own_ghostscript_on_windows_and_gs_from_path_elsewhere(windows, tools):
    assert route_curve_tex.required_tools(windows) == tools


def test_a_missing_gs_off_windows_is_named_before_any_build(tmp_path, monkeypatch, capsys):
    source = write_routing(tmp_path / "routing.json", routing())
    monkeypatch.setattr(route_curve_tex, "WINDOWS", False)
    monkeypatch.setattr(route_curve_tex.shutil, "which", lambda tool: None if tool == "gs" else tool)
    assert route_curve_tex.main(argv(tmp_path, source)) == 3
    err = capsys.readouterr().err
    # gs does not come with TeX Live off Windows: the hint is Ghostscript's, not TeX Live's bin folder
    assert err.startswith("FAIL: gs not found on PATH") and "Ghostscript" in err and "bin folder" not in err
    assert not (tmp_path / "build").exists() and not (tmp_path / "figs").exists()


def test_on_windows_the_build_needs_no_gs_on_path(tmp_path, monkeypatch):
    # rungs runs TeX Live's own Ghostscript there, and the development machine has no gs on PATH
    monkeypatch.setattr(route_curve_tex, "WINDOWS", True)
    tools = Tools()
    source = write_routing(tmp_path / "routing.json", routing())
    monkeypatch.setattr(route_curve_tex.shutil, "which", lambda tool: None if tool == "gs" else tool)
    monkeypatch.setattr(route_curve_tex, "_run", tools)
    assert route_curve_tex.main(argv(tmp_path, source)) == 0
    assert [call[0] for call in tools.calls] == ["xelatex", "rungs"]


# ---- the build, with xelatex and Ghostscript faked --------------------------------------------------------


def png(width: int, height: int) -> bytes:
    """The start of a PNG, its signature and header chunk: all of Ghostscript's file that main() reads."""
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(header)) + b"IHDR" + header
            + struct.pack(">I", zlib.crc32(b"IHDR" + header)))


CLEAN = "Output written on route-curve-m4.pdf (1 page).\n"
RERUN = "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.\n" + CLEAN
HOLE = "Missing character: There is no \N{COMMERCIAL MINUS SIGN} (U+2052) in font Microsoft YaHei!\n" + CLEAN


class Tools:
    """xelatex and rungs as main() runs them through _run, faked so that no TeX runs. Each call is recorded
    with its folder and the environment it adds. xelatex writes pdf and the next of logs (the last one again
    once they run out) into its folder, named as the document is; rungs writes image to the file
    -sOutputFile names, in its folder, which it reads as Ghostscript does: %% is a percent sign, and a
    single % starts a format (%d, the page number), which this fake refuses. None writes nothing, and each
    tool exits with its code of exits."""

    def __init__(self, logs: tuple[str, ...] = (CLEAN,), pdf: bytes | None = b"%PDF-1.5\n% fake\n",
                 image: bytes | None = png(2395, 899), exits: tuple[int, int] = (0, 0)) -> None:
        self.logs, self.pdf, self.image = list(logs), pdf, image
        self.exits = dict(zip(("xelatex", "rungs"), exits, strict=True))
        self.calls: list[tuple[str, list[str], Path | None, dict[str, str] | None]] = []

    def __call__(self, command: list[str], cwd: Path | None = None,
                 env: dict[str, str] | None = None) -> tuple[int, str]:
        tool = command[0]
        self.calls.append((tool, command, cwd, env))
        if tool == "xelatex":
            job = Path(cwd) / Path(command[-1]).stem
            log = self.logs.pop(0) if len(self.logs) > 1 else self.logs[0]
            Path(f"{job}.log").write_text(log, encoding="utf-8")
            if self.pdf is not None:
                Path(f"{job}.pdf").write_bytes(self.pdf)
        elif self.image is not None:
            name = next(arg for arg in command if arg.startswith("-sOutputFile="))
            name = name.removeprefix("-sOutputFile=")
            if "%" in name.replace("%%", ""):
                return 1, "the output file's name holds a format"
            if cwd is None:  # never into the folder the tests run in
                return 1, "no working folder"
            (Path(cwd) / name.replace("%%", "%")).write_bytes(self.image)
        return self.exits[tool], "the tool's last line"

    def runs(self, tool: str) -> list[tuple[str, list[str], Path | None, dict[str, str] | None]]:
        return [call for call in self.calls if call[0] == tool]


def build(tmp_path: Path, monkeypatch, tools: Tools, folder: str = "build") -> int:
    """main() over the fabricated routing file, with tools in place of xelatex and rungs."""
    source = write_routing(tmp_path / "routing.json", routing())
    monkeypatch.setattr(route_curve_tex.shutil, "which", lambda tool: tool)
    monkeypatch.setattr(route_curve_tex, "_run", tools)
    return route_curve_tex.main(argv(tmp_path, source, folder))


def test_a_build_compiles_once_converts_at_200_dpi_and_puts_the_pdf_beside_the_png(tmp_path, monkeypatch,
                                                                                     capsys):
    out, pdf, folder = tmp_path / "figs" / "curve.png", tmp_path / "figs" / "curve.pdf", tmp_path / "build"
    out.parent.mkdir()
    out.write_bytes(b"an earlier figure")
    pdf.write_bytes(b"an earlier figure")
    tools = Tools()
    assert build(tmp_path, monkeypatch, tools) == 0
    assert out.read_bytes() == png(2395, 899) and pdf.read_bytes() == tools.pdf  # both overwritten
    costs = json.loads(beside(tmp_path / "routing.json").read_text(encoding="utf-8"))
    assert (folder / "route-curve-m4.tex").read_bytes() == route_curve_tex.document(routing(), costs).encode(
        "utf-8")  # the cost oracle drawn in the lower row
    assert [call[0] for call in tools.calls] == ["xelatex", "rungs"]
    [(_, xelatex, cwd, _)] = tools.runs("xelatex")
    assert cwd == folder and xelatex[-1] == "route-curve-m4.tex"
    assert {"-interaction=nonstopmode", "-halt-on-error"} <= set(xelatex)  # never waits, stops at an error
    [(_, rungs, cwd, _)] = tools.runs("rungs")
    assert cwd == folder  # Ghostscript too runs in the build folder, on the files' bare names
    assert rungs == ["rungs", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r200",
                     "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", "-sOutputFile=route-curve-m4.png",
                     "route-curve-m4.pdf"]
    assert "(2395 x 899 px at 200 dpi" in capsys.readouterr().out


def test_xelatex_runs_with_a_fixed_source_date_so_that_the_same_file_gives_the_same_pdf(tmp_path,
                                                                                         monkeypatch):
    # xdvipdfmx otherwise writes the build's time into the PDF (its creation date, ID and creator), and the
    # committed PDF would change with every build of the same routing file
    tools = Tools(logs=(RERUN, CLEAN))
    assert build(tmp_path, monkeypatch, tools) == 0
    environments = [env for _, _, _, env in tools.runs("xelatex")]
    assert len(environments) == 2
    for env in environments:
        assert env["FORCE_SOURCE_DATE"] == "1"  # XeTeX's own date too, not only xdvipdfmx's
        date = datetime.fromtimestamp(int(env["SOURCE_DATE_EPOCH"]), UTC)
        assert date == datetime(2026, 10, 17, tzinfo=UTC)


def test_ghostscript_never_sees_the_build_folders_name_so_that_any_name_works(tmp_path, monkeypatch):
    # TeX Live's rungs.exe hands a non-ASCII argument on to Ghostscript as "??" on Windows, and Ghostscript
    # reads a % in -sOutputFile as a format: it runs in the build folder, on the files' bare names
    tools = Tools()
    assert build(tmp_path, monkeypatch, tools, folder="构建 100%") == 0
    [(_, rungs, cwd, _)] = tools.runs("rungs")
    assert cwd == tmp_path / "构建 100%" and not [arg for arg in rungs if "构建" in arg or "100" in arg]
    assert (tmp_path / "figs" / "curve.png").read_bytes() == png(2395, 899)


@pytest.mark.parametrize(("logs", "runs"), [((CLEAN,), 1), ((RERUN, CLEAN), 2), ((RERUN,), 2)])
def test_xelatex_runs_a_second_time_only_when_its_first_log_asks_for_it_and_never_a_third(
        tmp_path, monkeypatch, logs, runs):
    tools = Tools(logs=logs)
    assert build(tmp_path, monkeypatch, tools) == 0
    assert len(tools.runs("xelatex")) == runs


def test_a_glyph_the_font_lacks_fails_the_build_before_ghostscript_runs(tmp_path, monkeypatch, capsys):
    # XeTeX exits 0 with a hole in a label; only its log tells
    tools = Tools(logs=(HOLE,))
    assert build(tmp_path, monkeypatch, tools) == 4
    err = capsys.readouterr().err
    assert "glyph" in err and "U+2052" in err and "Traceback" not in err
    assert not tools.runs("rungs") and not (tmp_path / "figs").exists()
    assert (tmp_path / "build" / "route-curve-m4.log").read_text(encoding="utf-8") == HOLE  # the log stays


def test_xelatex_failing_fails_the_build_even_when_it_leaves_a_pdf(tmp_path, monkeypatch, capsys):
    tools = Tools(exits=(1, 0))
    assert build(tmp_path, monkeypatch, tools) == 4
    err = capsys.readouterr().err
    assert "xelatex failed (exit 1)" in err and "Traceback" not in err
    assert not tools.runs("rungs") and not (tmp_path / "figs").exists()


@pytest.mark.parametrize("silent", ["xelatex", "rungs"])
def test_a_pdf_or_png_of_an_earlier_build_cannot_stand_in(tmp_path, monkeypatch, capsys, silent):
    folder = tmp_path / "build"
    folder.mkdir()
    (folder / "route-curve-m4.pdf").write_bytes(b"%PDF-1.5\n% an earlier build\n")
    (folder / "route-curve-m4.png").write_bytes(png(640, 480))
    # the tool exits 0 but writes nothing
    tools = Tools(pdf=None) if silent == "xelatex" else Tools(image=None)
    assert build(tmp_path, monkeypatch, tools) == 4
    assert "Traceback" not in capsys.readouterr().err
    assert not (tmp_path / "figs").exists()


@pytest.mark.parametrize(("written", "exits"), [(b"GPL Ghostscript: no PNG here", (0, 0)),
                                                (png(2395, 899), (0, 1))])
def test_ghostscript_must_exit_0_with_a_png_or_the_build_fails(tmp_path, monkeypatch, capsys, written, exits):
    assert build(tmp_path, monkeypatch, Tools(image=written, exits=exits)) == 4
    err = capsys.readouterr().err
    assert "Ghostscript" in err and "Traceback" not in err
    assert not (tmp_path / "figs").exists()


def test_a_tool_runs_without_input_with_the_environment_and_its_additions_and_minutes_to_finish(monkeypatch):
    seen = {}

    def run(command, **options):
        seen.update(options)
        return subprocess.CompletedProcess(command, 3, b"first\nthe last line\n\n", b"")

    monkeypatch.setattr(route_curve_tex.subprocess, "run", run)
    folder = Path("build")
    assert route_curve_tex._run(["xelatex", "x.tex"], cwd=folder, env={"FORCE_SOURCE_DATE": "1"}) == (
        3, "the last line")
    assert seen["cwd"] == folder and seen["env"] == {**os.environ, "FORCE_SOURCE_DATE": "1"}
    assert seen["stdin"] == subprocess.DEVNULL  # a run that stops for input cannot wait for it
    assert seen["timeout"] == route_curve_tex.TIMEOUT_S >= 300  # a first XeTeX run may build its font cache


def test_a_tool_that_does_not_finish_in_time_fails_the_build(tmp_path, monkeypatch, capsys):
    def run(command, **options):
        raise subprocess.TimeoutExpired(command, options["timeout"])

    source = write_routing(tmp_path / "routing.json", routing())
    monkeypatch.setattr(route_curve_tex.shutil, "which", lambda tool: tool)
    monkeypatch.setattr(route_curve_tex.subprocess, "run", run)
    assert route_curve_tex.main(argv(tmp_path, source)) == 4
    err = capsys.readouterr().err
    assert f"xelatex did not finish in {route_curve_tex.TIMEOUT_S} s" in err and "Traceback" not in err
    assert not (tmp_path / "figs").exists()


def test_a_tools_error_line_is_reported_rather_than_its_last_line(monkeypatch):
    # Ghostscript's last line is "Unrecoverable error"; the line that says why starts with "Error:"
    printed = (b"GPL Ghostscript 10.08.0 (2026-03-04)\nError: /undefinedfilename in (x.pdf)\n"
               b"Operand stack:\n\nGPL Ghostscript 10.08.0: Unrecoverable error, exit code 1\n")
    monkeypatch.setattr(route_curve_tex.subprocess, "run",
                        lambda command, **options: subprocess.CompletedProcess(command, 1, printed, b""))
    assert route_curve_tex._run(["rungs", "x.pdf"]) == (1, "Error: /undefinedfilename in (x.pdf)")


# ---- the targets ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["curve.pdf", "curve", "curve.png.txt"])
def test_an_out_that_is_not_a_png_is_a_usage_error_and_nothing_is_written(tmp_path, monkeypatch, capsys,
                                                                           name):
    # with curve.pdf, the PNG would be copied there and then overwritten by the PDF
    source = write_routing(tmp_path / "routing.json", routing())
    monkeypatch.setattr(route_curve_tex.shutil, "which", lambda tool: tool)
    monkeypatch.setattr(route_curve_tex, "_run", Tools())
    with pytest.raises(SystemExit) as stop:
        route_curve_tex.main(["--routing", str(source), "--build", str(tmp_path / "build"), "--out",
                              str(tmp_path / "figs" / name)])
    assert stop.value.code == 2 and ".png" in capsys.readouterr().err
    assert not (tmp_path / "build").exists() and not (tmp_path / "figs").exists()


@pytest.mark.parametrize("held", ["curve.pdf", "curve.png"])
def test_a_target_another_program_holds_fails_the_run_before_either_file_changes(tmp_path, monkeypatch,
                                                                                 capsys, held):
    # a PDF viewer on Windows holds its file against writing; the PNG and the PDF change both or neither.
    # The fake refuses to open the held file for writing, as Windows does
    figs = tmp_path / "figs"
    figs.mkdir()
    for name in ("curve.png", "curve.pdf"):
        (figs / name).write_bytes(b"an earlier figure")
    opened = Path.open

    def holding(self, mode="r", *args, **kwargs):
        if self == figs / held and any(letter in mode for letter in "wax+"):
            raise PermissionError(13, "Permission denied", str(self))
        return opened(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", holding)
    assert build(tmp_path, monkeypatch, Tools()) == 4
    err = capsys.readouterr().err
    assert "Permission denied" in err and "Traceback" not in err
    assert sorted(path.name for path in figs.iterdir()) == ["curve.pdf", "curve.png"]  # nothing staged stays
    assert {path.read_bytes() for path in figs.iterdir()} == {b"an earlier figure"}


def earlier_figures(tmp_path: Path) -> Path:
    figs = tmp_path / "figs"
    figs.mkdir()
    for name in ("curve.png", "curve.pdf"):
        (figs / name).write_bytes(b"an earlier figure")
    return figs


@pytest.mark.parametrize("failing", [1, 2])
def test_a_copy_that_fails_halfway_changes_neither_target_and_leaves_no_staged_file(tmp_path, monkeypatch,
                                                                                    capsys, failing):
    # the disk fills while the PDF (1) or the PNG (2) is staged: a part of it is written, then the copy fails
    figs = earlier_figures(tmp_path)
    copies, real_copy = [], shutil.copyfile

    def copyfile(source, target):
        copies.append(target)
        if len(copies) == failing:
            Path(target).write_bytes(b"part of a fig")
            raise OSError(28, "No space left on device")
        return real_copy(source, target)

    monkeypatch.setattr(route_curve_tex.shutil, "copyfile", copyfile)
    assert build(tmp_path, monkeypatch, Tools()) == 4
    assert "No space left" in capsys.readouterr().err
    assert sorted(path.name for path in figs.iterdir()) == ["curve.pdf", "curve.png"]
    assert {path.read_bytes() for path in figs.iterdir()} == {b"an earlier figure"}


def test_a_png_that_cannot_be_replaced_after_its_pdf_says_the_two_now_differ(tmp_path, monkeypatch, capsys):
    # the residue publish() states: a program that lets the PNG be written but not replaced. The PDF goes
    # first, so it is the new one, and the message says that the two come from different builds
    figs = earlier_figures(tmp_path)
    replace = os.replace

    def replacing(source, target):
        if Path(target).name == "curve.png":
            raise PermissionError(5, "Access is denied", str(target))
        return replace(source, target)

    monkeypatch.setattr(route_curve_tex.os, "replace", replacing)
    assert build(tmp_path, monkeypatch, Tools()) == 4
    err = capsys.readouterr().err
    assert "different builds" in err and "Traceback" not in err
    assert (figs / "curve.pdf").read_bytes() == Tools().pdf and (figs / "curve.png").read_bytes() == (
        b"an earlier figure")
    assert sorted(path.name for path in figs.iterdir()) == ["curve.pdf", "curve.png"]
