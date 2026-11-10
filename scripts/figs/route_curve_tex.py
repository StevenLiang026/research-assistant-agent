"""Draw FT+R's routing curve on the development set (docs/plan.md, 路由曲线) with TeX Live: the figure that
scripts/figs/route_curve.m specifies, with the cost row docs/plan.md asks for under it, from
results/routing-m4.json to docs/figs/route-curve-m4.png and .pdf.

    .venv/Scripts/python.exe scripts/figs/route_curve_tex.py [--routing results/routing-m4.json] \\
        [--cost-oracle results/cost-oracle-m4.json] [--build runs/figs/route-curve-m4] \\
        [--out docs/figs/route-curve-m4.png]

Run from the repository root with TeX Live's bin folder on PATH: xelatex, and rungs, TeX Live's front end to
Ghostscript, which runs TeX Live's own Ghostscript on Windows and gs from PATH elsewhere (required_tools).
The script itself needs the standard library only.

Why it exists beside route_curve.m. route_curve.m is the figure's design and stays the specification of its
upper row. On 2026-10-17 MATLAB refused to run on the development machine (a license error), and the owner
chose to draw the same figure with TeX Live instead: pgfplots under XeLaTeX, converted to PNG by
Ghostscript.

The upper row, as route_curve.m specifies it. Two panels side by side. Left, "全部 n 道": FT+R's threshold
scan (curves.routing, solid), random routing at the same API share (each scan point's random_success,
dashed), the oracle (curves.oracle, dotted), the preregistered rule's target at 100 x (b2_success - margin)
as a dash-dot line with its label, the frozen threshold (chosen_point) as a circle with its API share and
success, and FT's and B2's success at the left and the right edge. Right, "只看手写题 n 道(描述)": the
handwritten items' scan and random routing (handwritten.scan), the point of the frozen threshold on them
(handwritten.at_threshold) and their FT and B2 success. Both: x from 0 to 100 (API share, percent), y from
40 to 100 (success, percent), a light grid, the legend in the lower right without a box, MATLAB's default
line colours.

The lower row, which route_curve.m does not draw. docs/plan.md asks for success against the API share and
against the cost per 100 items, and route_curve.m draws the first only. The lower row draws the upper row's
panels again with the cost per 100 items on x (yuan at list price: what B2's answers to the routed items
cost), from 0 to a round limit both panels share (cost_axis): the scan at each point's cost_per_100_rmb;
random routing at the same API share at its expected cost, the share times the cost of routing every item
(total_cost), which is exact since random routing sends each item to B2 with the probability of the share,
so that at one x the dashed curve is random routing at the same cost ("随机路由(同成本)"); on the left panel,
as the upper row has its oracle there, the oracle at a cost ("理想路由(同成本)", dotted) and the target line
with its label; the frozen point with its cost and success; FT at no cost and B2 at the cost of routing
every item. The routing file's oracle cannot be drawn here: it gives its points no cost, and which items it
routes at a given count, and so what they cost, is not fixed. The oracle at a cost is
scripts/route_cost_oracle.py's (--cost-oracle), made from the records the routing file was made from and
checked to belong to it (its routing.sha256 is the routing file's SHA-256, its n the routing file's): it
routes only the items FT fails and B2 passes, the cheapest first. Between two of its points the success is
the earlier point's, so it is drawn in steps (pgfplots' const plot), and past its last point, where every
such item is routed, it runs flat to the cost of routing every item: more money buys no more success.

Where the layout departs from route_curve.m. Its own positions put text on the curves of
results/routing-m4.json (seen on the first render), so the text moves while what it marks stays put. FT's
label sits under the curves' common start and B2's just right of their common end, outside the right edge
in the upper row (route_curve.m: both above the curves, inside the edges); the frozen point's note is
centred above the point with a leader (route_curve.m: below right of it, where the random line runs); the
target's label, at 8 pt, starts right of the oracle's rise in both rows (route_curve.m: at the left edge,
which the rise crosses). Text inside the
axes, the legend included, sits on white without an outline, so no grid line runs through it (MATLAB
leaves a legend without a box transparent). The frozen point is purple on every panel; MATLAB's colour
order would make it yellow on the right, where it is the third plot.

Text. Microsoft YaHei for all of it, as route_curve.m sets it, through xeCJK. xeCJK drops a space after a
Chinese character and by default adds one between Chinese and Latin text, next to "(" and ":" too; the
preamble turns the second off (an empty CJKecglue) and escape() writes the first as a control space, so every
label prints as route_curve.m types it ("全部 150 道", "随机路由(同调用比例)").

Numbers. Every number the figure shows comes from the file, formatted as route_curve.m formats it:
percentages with one decimal (%.1f), the threshold with three (%.3f), item counts as integers (%d); the
lower row's costs, which route_curve.m does not show, with one decimal more than the cost axis' tick step has
(two for this file's step of 0.1 yuan). The one number route_curve.m types by hand, the "2" of the target's
label, is 100 x margin here (%g). The curves' coordinates are the file's values (times 100 for the
percentages) printed with DECIMALS decimals, and the lower row's with one more for each power of ten its
axis falls below 0.1 yuan (places), so that no coordinate is off by more than 5e-4 of its axis, half a pixel
of the 913 the axis spans at 200 dpi.

A file that cannot be drawn. When its rule was not met, scripts/route_curve.py writes "rule_met" false,
"chosen_point" null and "handwritten.at_threshold" null. route_curve.m stops on such a file (MATLAB cannot
index the empty chosen point), so no figure comes out of it; here document() raises NoFrozenThreshold and
the command exits 1 with nothing written. A figure without a frozen threshold would need a design
route_curve.m does not give. A file whose scan has no single route_everything point, or whose routing costs
nothing, cannot be drawn either: the lower row needs the cost of routing every item. Nor can one without a
cost oracle made from it.

Build. The document goes to --build/route-curve-m4.tex (UTF-8, LF) and is compiled there with xelatex
(nonstop, halting on the first error), a second time only when its log asks for one (needs_rerun). xelatex
runs with SOURCE_DATE_EPOCH and FORCE_SOURCE_DATE=1 added to its environment (REPRODUCIBLE), so the PDF
carries SOURCE_DATE (2026-10-17 00:00 UTC) in its creation date, its ID and its creator line instead of the
time and the time zone of the build: the same routing file gives the same PDF bytes, as it gives the same PNG
bytes, on the same TeX Live and fonts. A glyph the font lacks fails the build: XeTeX only logs it ("Missing
character") and exits 0, leaving a hole in a label. Ghostscript converts the PDF to a PNG at 200 dpi (png16m,
anti-aliased text and graphics), also in the build folder and on the files' bare names: TeX Live's rungs.exe
hands a non-ASCII argument on to Ghostscript as "??", and Ghostscript reads a % in a file name as a format,
so the folder's name stays out of its arguments. The PNG then goes to --out and the PDF beside it (same
name, .pdf), both or neither (publish): each is first copied beside its target under a temporary name, a
target another program holds against writing (a PDF viewer on Windows) fails the run before either changes,
and then the PDF and the PNG replace their targets; a temporary file never stays. Only a program that lets
the PNG be written but not replaced can stop its replacement after the PDF's, and the run then says that the
two come from different builds. Both are overwritten, as route_curve.m's exportgraphics overwrote its PNG.

Exit 0 when both are written; 1 when the routing file cannot be drawn (missing, not JSON, not
route_curve.py's output, its rule not met, no finite cost of routing every item, or no cost oracle made from
it: missing, malformed, or made from another routing file); 2 on a usage error, an
--out that does not end in .png among them; 3 when a tool the build runs is not on PATH; 4 when the build
fails: xelatex or Ghostscript fails, cannot start or does not finish in TIMEOUT_S, the font lacks a glyph,
or a file cannot be written (TeX's log stays in the build folder). Nothing reaches --out unless the build
succeeded, and nothing at all is written on exits 1 to 3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

ROUTING = Path("results/routing-m4.json")
COST_ORACLE = Path("results/cost-oracle-m4.json")
BUILD = Path("runs/figs/route-curve-m4")
OUT = Path("docs/figs/route-curve-m4.png")
STEM = "route-curve-m4"  # ASCII without "%": Ghostscript reads the files' names as they are
TOOLS = ("xelatex", "rungs")
WINDOWS = os.name == "nt"
DPI = 200
DECIMALS = 4  # of the plotted coordinates, in percent or yuan
TIMEOUT_S = 600  # a first XeTeX run may build its font cache
# The date the PDF carries, whatever the day of its build: the day this script took over from route_curve.m.
# xdvipdfmx writes SOURCE_DATE_EPOCH as the PDF's creation date and derives the file's ID from it;
# FORCE_SOURCE_DATE=1 makes it XeTeX's own date too, which goes into the PDF's creator line
SOURCE_DATE = datetime(2026, 10, 17, tzinfo=UTC)
REPRODUCIBLE = {"SOURCE_DATE_EPOCH": str(int(SOURCE_DATE.timestamp())), "FORCE_SOURCE_DATE": "1"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MINUS = "\N{MINUS SIGN}"  # route_curve.m's target label writes U+2212, not a hyphen
SPECIALS = {"\\": r"\textbackslash{}", "%": r"\%", "&": r"\&", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}

# route_curve.m's labels (its DisplayName, title and axis label texts), and the lower row's x axis
SCAN, RANDOM, ORACLE = "FT+R:扫描阈值", "随机路由(同调用比例)", "理想路由"
RANDOM_BY_COST = "随机路由(同成本)"  # the lower row: at one x, random routing at the same expected cost
ORACLE_BY_COST = "理想路由(同成本)"  # the lower row: scripts/route_cost_oracle.py's oracle at a cost
COST = "每百道成本(元,列表价)"
COST_INTERVALS = 6  # the most intervals between the lower row's x ticks
# Placements, chosen by looking at the figure of results/routing-m4.json, where route_curve.m's own positions
# put text on curves. The target's label starts right of where the oracle rises across the target and ends
# left of the frozen point: in the upper row from x 27 (the rise ends at 26, the point is at 62.7), in the
# lower row from TARGET_LABEL_COST_AT of the axis (0.14 yuan: the oracle at a cost reaches its best at 0.133,
# the point is at 0.323). The frozen points' notes are centred NOTE_RISE above the points, clear of the
# oracles' best (92) and of every curve; FT's labels sit below the curves' common start, at INSET, B2's just
# right of their common end. Another routing file may need another look; the tests check what this look
# relied on.
INSET = 0.015  # of the axis' width: route_curve.m's ends() puts FT's label at x 1.5 of 100
TARGET_LABEL_X = 27.0
TARGET_LABEL_COST_AT = 0.28  # of the lower row's axis
NOTE_RISE = 5.0  # percentage points

PREAMBLE = r"""% FT+R's routing curve on the development set (docs/plan.md, 路由曲线): the figure of
% scripts/figs/route_curve.m and its cost row, written by scripts/figs/route_curve_tex.py. Edit that script,
% not this file.
\documentclass[10pt,border=4pt]{standalone}
\usepackage{xeCJK}
\setmainfont{Microsoft YaHei}
\setCJKmainfont{Microsoft YaHei}
\xeCJKsetup{CJKecglue={}}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
% MATLAB's default colours: its first four line colours and the colour of a constant line
\definecolor{matlabblue}{rgb}{0,0.447,0.741}
\definecolor{matlaborange}{rgb}{0.85,0.325,0.098}
\definecolor{matlabyellow}{rgb}{0.929,0.694,0.125}
\definecolor{matlabpurple}{rgb}{0.494,0.184,0.556}
\definecolor{matlabink}{rgb}{0.15,0.15,0.15}
\pgfplotsset{
  scan curve/.style={draw=matlabblue, line width=1.4pt},
  random curve/.style={draw=matlaborange, dashed, line width=1.2pt},
  oracle curve/.style={draw=matlabyellow, dotted, line width=1.4pt},
  cost oracle curve/.style={oracle curve, const plot},
  target line/.style={draw=matlabink, dash dot, line width=0.6pt, forget plot},
  frozen point/.style={draw=matlabpurple, only marks, mark=o, mark size=3.5pt,
    mark options={solid, line width=1.5pt}},
}
% Text inside the axes sits on white, without an outline, so that no grid line runs through it; none of it
% covers a curve
\tikzset{
  end label/.style={inner sep=2pt, fill=white},
  note/.style={anchor=south, align=center, inner sep=1pt, fill=white},
  leader/.style={draw=matlabink, line width=0.4pt, shorten <=5pt, shorten >=1.5pt},
  target label/.style={anchor=south west, inner sep=1pt, yshift=0.5pt, fill=white,
    font=\fontsize{8}{10}\selectfont},
}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 2, horizontal sep=3.6cm, vertical sep=2.8cm},
  scale only axis, width=11.6cm, height=9cm,
  xmin=0, xmax=100, ymin=40, ymax=100,
  xtick={0,10,...,100}, ytick={40,50,...,100},
  xticklabel={\pgfmathprintnumber[assume math mode=true]{\tick}},
  yticklabel={\pgfmathprintnumber[assume math mode=true]{\tick}},
  axis lines*=left, tick align=outside,
  grid=major, major grid style={draw=matlabink, draw opacity=0.15, line width=0.4pt},
  xlabel={API 调用比例(\%)}, ylabel={开发集单步成功率(\%)},
  label style={font=\fontsize{11}{13}\selectfont},
  title style={font=\fontsize{11}{13}\selectfont},
  legend pos=south east, legend cell align=left,
  legend style={draw=none, fill=white, font=\fontsize{9}{11}\selectfont},
  clip mode=individual,
]
"""
END = "\\end{groupplot}\n\\end{tikzpicture}\n\\end{document}\n"


class NoFrozenThreshold(ValueError):
    """A routing file whose rule was not met: it has no chosen point, and route_curve.m fails on it."""


class ForeignCostOracle(ValueError):
    """A cost oracle that was not made from the routing file: another SHA-256 of it, or other items."""


class BuildError(RuntimeError):
    """xelatex or Ghostscript failed, could not start or did not finish, or the font lacks a glyph: exit 4."""


# ---- the document -----------------------------------------------------------------------------------------


def escape(text: str) -> str:
    """text with TeX's special characters escaped, so that it prints as it reads. A space after a wide (CJK)
    character becomes a control space: xeCJK drops a plain one there, and with the preamble's empty CJKecglue
    it puts nothing in its place."""
    out = []
    for i, c in enumerate(text):
        after_wide = i > 0 and unicodedata.east_asian_width(text[i - 1]) in ("W", "F")
        out.append("\\ " if c == " " and after_wide else SPECIALS.get(c, c))
    return "".join(out)


def total_cost(points: list[dict]) -> float:
    """The cost per 100 items of routing every item, B2's alone: the scan's route_everything point's."""
    ends = [p["cost_per_100_rmb"] for p in points if p["label"] == "route_everything"]
    if len(ends) != 1:
        raise KeyError(f"the scan has {len(ends)} route_everything points, not one")
    return ends[0]


def cost_axis(top: float) -> tuple[float, list[float]]:
    """The lower row's x axis for costs up to top: its limit and its ticks. The ticks step by the smallest of
    1, 2 or 5 times a power of ten that reaches top in at most COST_INTERVALS steps, and the limit is the
    first tick at or above top."""
    if not (top > 0 and math.isfinite(top)):
        raise ValueError(f"routing every item costs {top!r} per 100 items: there is no cost axis to draw")
    power = 10.0 ** math.floor(math.log10(top / COST_INTERVALS))
    # ten times the power reaches top in fewer than COST_INTERVALS steps, so one of the four fits
    step = next(factor * power for factor in (1, 2, 5, 10)
                if math.ceil(round(top / (factor * power), 9)) <= COST_INTERVALS)
    count = math.ceil(round(top / step, 9))
    return _clean(count * step), [_clean(i * step) for i in range(count + 1)]


def _clean(value: float) -> float:
    """value without a product's float noise (3 x 0.1 is 0.30000000000000004), to 12 significant digits."""
    return float(f"{value:.12g}")


def places(limit: float) -> int:
    """The decimals of the coordinates on an axis from 0 to limit: DECIMALS down to an axis of 0.1, and one
    more for each power of ten below that (module docstring, Numbers)."""
    return DECIMALS + max(0, -math.floor(math.log10(limit)) - 1)


def _cost_text(cost: float, axis: tuple[float, list[float]]) -> str:
    """A cost per 100 items as the lower row's notes show it, one decimal more than the ticks' step has."""
    step = axis[1][1]
    return f"每百道 {cost:.{max(0, -math.floor(math.log10(step))) + 1}f} 元"


def _number(value: float, digits: int = DECIMALS) -> str:
    return f"{value:.{digits}f}"


def _plot(style: str, points: list[tuple[float, float]], digits: int = DECIMALS) -> str:
    """An \\addplot of the points (x, y), six to a line."""
    coordinates = [f"({_number(x, digits)},{_number(y, digits)})" for x, y in points]
    rows = [" ".join(coordinates[i:i + 6]) for i in range(0, len(coordinates), 6)]
    return f"\\addplot[{style}] coordinates {{\n" + "\n".join(rows) + "\n};\n"


def _node(style: str, x: float, y: float, text: str, digits: int = DECIMALS) -> str:
    """A text node at (x, y) in the axis' coordinates; text is TeX, already escaped."""
    return f"\\node[{style}] at (axis cs:{_number(x, digits)},{_number(y, digits)}) {{{text}}};\n"


def _legend(text: str) -> str:
    return f"\\addlegendentry{{{escape(text)}}}\n"


def _panel(title: str, axis: tuple[float, list[float]] | None = None) -> str:
    """A panel's start with its title; in the lower row, with its cost axis (cost_axis) in place of the
    API share's."""
    options = f"title={{{escape(title)}}}"
    if axis is not None:
        limit, ticks = axis
        options += (f", xmin=0, xmax={limit:g}, xtick={{{','.join(f'{tick:g}' for tick in ticks)}}}, "
                    f"xlabel={{{escape(COST)}}}")
    return f"\\nextgroupplot[{options}]\n"


def _curves(points: list[dict]) -> str:
    """The threshold scan and random routing at the same API share, with their legend entries."""
    return (_plot("scan curve", [(100 * p["api_share"], 100 * p["success"]) for p in points])
            + _legend(SCAN)
            + _plot("random curve", [(100 * p["api_share"], 100 * p["random_success"]) for p in points])
            + _legend(RANDOM))


def _cost_curves(points: list[dict], total: float, digits: int) -> str:
    """The threshold scan at each point's cost per 100 items, and random routing at the same API share at its
    expected cost, the share times total (the cost of routing every item), with their legend entries."""
    return (_plot("scan curve", [(p["cost_per_100_rmb"], 100 * p["success"]) for p in points], digits)
            + _legend(SCAN)
            + _plot("random curve", [(p["api_share"] * total, 100 * p["random_success"]) for p in points],
                    digits)
            + _legend(RANDOM_BY_COST))


def _frozen(x: float, y: float, where: str, legend: str, digits: int = DECIMALS) -> str:
    """The frozen threshold's point at (x, y), y its success in percent, its legend entry, and its note:
    where (its place on the x axis) over its success, as route_curve.m's text() has it, centred NOTE_RISE
    above the point with a leader down to it."""
    note = escape(where) + r"\\" + escape(f"成功率 {y:.1f}%")
    leader = (f"\\draw[leader] (axis cs:{_number(x, digits)},{_number(y, digits)}) -- "
              f"(axis cs:{_number(x, digits)},{_number(y + NOTE_RISE, digits)});\n")
    return (_plot("frozen point", [(x, y)], digits) + _legend(legend) + leader
            + _node("note", x, y + NOTE_RISE, note, digits))


def _ends(ft: float, b2: float, left: float, right: float, digits: int = DECIMALS) -> str:
    """FT's success at x left, under the curves' start, and B2's at x right, just right of their end
    (route_curve.m's ends())."""
    return (_node("end label, anchor=north west, yshift=-1pt", left, 100 * ft, escape(f"FT {100 * ft:.1f}%"),
                  digits)
            + _node("end label, anchor=west", right, 100 * b2, escape(f"B2 {100 * b2:.1f}%"), digits))


def _target(routing: dict) -> tuple[float, str]:
    """The target in percent and its label: under scripts/route_curve.py's target objective (M7's R3) the
    routing file's own target; otherwise the preregistered rule's 100 x (b2_success - margin) (M4, M6)."""
    if routing.get("objective") == "target":
        target = 100 * routing["target"]
        return target, f"目标 {target:.1f}%"
    target = 100 * (routing["b2_success"] - routing["margin"])
    return target, f"目标 {target:.1f}%(B2 {MINUS} {100 * routing['margin']:g} 个百分点)"


def _all_items(routing: dict, chosen: dict) -> str:
    target, label = _target(routing)
    oracle = routing["curves"]["oracle"]
    share = 100 * chosen["api_share"]
    return (_panel(f"全部 {routing['n']:d} 道")
            + _curves(routing["curves"]["routing"])
            + _plot("oracle curve", [(100 * p["api_share"], 100 * p["success"]) for p in oracle])
            + _legend(ORACLE)
            + _plot("target line", [(0.0, target), (100.0, target)])
            + _node("target label", TARGET_LABEL_X, target, escape(label))
            + _frozen(share, 100 * chosen["success"], f"调用比例 {share:.1f}%",
                      f"冻结阈值 {chosen['threshold']:.3f}")
            + _ends(routing["ft_success"], routing["b2_success"], 100 * INSET, 100.0))


def _handwritten(hand: dict, threshold: float) -> str:
    at = hand["at_threshold"]
    share = 100 * at["api_share"]
    return (_panel(f"只看手写题 {hand['n']:d} 道(描述)")
            + _curves(hand["scan"])
            + _frozen(share, 100 * at["success"], f"调用比例 {share:.1f}%", f"冻结阈值 {threshold:.3f} 处")
            + _ends(hand["ft_success"], hand["b2_success"], 100 * INSET, 100.0))


def _cost_oracle(costs: dict | None, total: float, digits: int) -> str:
    """scripts/route_cost_oracle.py's oracle at a cost, in steps and flat past its last point to total (the
    cost of routing every item), with its legend entry; nothing without its file."""
    if costs is None:
        return ""
    points = [(p["cost_per_100_rmb"], 100 * p["success"]) for p in costs["curve"]]
    if total > points[-1][0]:
        points.append((total, points[-1][1]))
    return _plot("cost oracle curve", points, digits) + _legend(ORACLE_BY_COST)


def _all_items_by_cost(routing: dict, chosen: dict, total: float, axis: tuple[float, list[float]],
                       costs: dict | None) -> str:
    target, label = _target(routing)
    limit, cost = axis[0], chosen["cost_per_100_rmb"]
    digits = places(limit)
    return (_panel(f"全部 {routing['n']:d} 道", axis)
            + _cost_curves(routing["curves"]["routing"], total, digits)
            + _cost_oracle(costs, total, digits)
            + _plot("target line", [(0.0, target), (limit, target)], digits)
            + _node("target label", TARGET_LABEL_COST_AT * limit, target, escape(label), digits)
            + _frozen(cost, 100 * chosen["success"], _cost_text(cost, axis),
                      f"冻结阈值 {chosen['threshold']:.3f}", digits)
            + _ends(routing["ft_success"], routing["b2_success"], INSET * limit, total, digits))


def _handwritten_by_cost(hand: dict, threshold: float, total: float, axis: tuple[float, list[float]]) -> str:
    at = hand["at_threshold"]
    cost, digits = at["cost_per_100_rmb"], places(axis[0])
    return (_panel(f"只看手写题 {hand['n']:d} 道(描述)", axis)
            + _cost_curves(hand["scan"], total, digits)
            + _frozen(cost, 100 * at["success"], _cost_text(cost, axis), f"冻结阈值 {threshold:.3f} 处",
                      digits)
            + _ends(hand["ft_success"], hand["b2_success"], INSET * axis[0], total, digits))


def document(routing: dict, costs: dict | None = None) -> str:
    """The figure over a routing file of scripts/route_curve.py, route_curve.m's figure with the cost row
    under it, as a complete standalone XeLaTeX document (module docstring); the cost row has its oracle when
    costs, scripts/route_cost_oracle.py's file made from the routing file, is given (main() always gives it).
    Raises NoFrozenThreshold when the file's rule was not met, and KeyError, TypeError or ValueError when it
    is not such a file or gives no cost of routing every item."""
    chosen = routing["chosen_point"]
    if chosen is None:
        raise NoFrozenThreshold("the preregistered rule was not met (rule_met false): the file has no "
                                "chosen point, so there is no frozen threshold to draw, and route_curve.m "
                                "fails on it")
    hand = routing["handwritten"]
    totals = total_cost(routing["curves"]["routing"]), total_cost(hand["scan"])
    axis = cost_axis(max(totals))
    return (PREAMBLE + _all_items(routing, chosen) + _handwritten(hand, chosen["threshold"])
            + _all_items_by_cost(routing, chosen, totals[0], axis, costs)
            + _handwritten_by_cost(hand, chosen["threshold"], totals[1], axis) + END)


def load_cost_oracle(path: Path, routing_bytes: bytes, routing: dict) -> dict:
    """scripts/route_cost_oracle.py's file at path, checked to be made from the routing file of these bytes
    and this content: its routing.sha256 is theirs and its n the routing file's (ForeignCostOracle); and to
    be that script's output, one point of its curve for each number of gains routed, from none to all
    (ValueError)."""
    costs = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(routing_bytes).hexdigest()
    if costs["routing"]["sha256"] != digest or costs["n"] != routing["n"]:
        raise ForeignCostOracle(f"{path.as_posix()} was made from another routing file (routing SHA-256 "
                                f"{costs['routing']['sha256']}, n {costs['n']}; this one's {digest}, n "
                                f"{routing['n']})")
    curve = costs["curve"]
    if not isinstance(curve, list) or not curve or costs["gains"] != len(curve) - 1:
        raise ValueError(f"{path.as_posix()}'s curve is not one point for each number of gains routed, from "
                         f"none to its {costs['gains']!r} gains")
    return costs


# ---- the build ---------------------------------------------------------------------------------------------


def missing_characters(log: str) -> list[str]:
    """XeTeX's lines for glyphs the font lacks: each one is a hole in the figure's text."""
    return [line for line in log.splitlines() if line.startswith("Missing character:")]


def needs_rerun(log: str) -> bool:
    """Whether LaTeX asks for another run (references that changed)."""
    return "Rerun to get" in log or "Rerun LaTeX" in log


def required_tools(windows: bool) -> tuple[str, ...]:
    """The programs the build needs on PATH: xelatex, and rungs, which runs TeX Live's own Ghostscript on
    Windows and gs from PATH elsewhere (TeX Live's rungs.lua), so gs too off Windows."""
    return TOOLS if windows else (*TOOLS, "gs")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Draw route_curve.m's routing figure with TeX Live.")
    ap.add_argument("--routing", type=Path, default=ROUTING,
                    help=f"scripts/route_curve.py's routing file (default {ROUTING.as_posix()})")
    ap.add_argument("--cost-oracle", type=Path, default=COST_ORACLE,
                    help=f"scripts/route_cost_oracle.py's oracle at a cost, made from the routing file "
                         f"(default {COST_ORACLE.as_posix()})")
    ap.add_argument("--build", type=Path, default=BUILD,
                    help=f"the folder of the document and TeX's files (default {BUILD.as_posix()})")
    ap.add_argument("--out", type=Path, default=OUT,
                    help=f"the PNG; the PDF is written beside it (default {OUT.as_posix()})")
    return ap


def _run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run a tool to its end, in this process' environment with env's variables added (for this run only):
    its exit code and the line of its output that says most, the first that starts with "Error:"
    (Ghostscript's reason, which its last line, "Unrecoverable error", does not give), else the last."""
    try:
        done = subprocess.run(command, cwd=cwd, env={**os.environ, **(env or {})}, stdin=subprocess.DEVNULL,
                              capture_output=True, check=False, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise BuildError(f"{command[0]} did not finish in {TIMEOUT_S} s") from None
    except OSError as exc:
        raise BuildError(f"{command[0]} could not start: {exc}") from None
    printed = [line for line in (done.stdout + done.stderr).decode("utf-8", "replace").splitlines()
               if line.strip()]
    return done.returncode, next((line for line in printed if line.startswith("Error:")),
                                 printed[-1] if printed else "")


def _build(tex: str, tools: dict[str, str], folder: Path) -> tuple[Path, Path, bytes]:
    """Compile the document in folder and convert it to PNG (module docstring, Build): the PDF, the PNG and
    the PNG's bytes. A PDF or PNG of an earlier build is removed first, so that neither can stand in for
    this one."""
    folder.mkdir(parents=True, exist_ok=True)
    tex_file, log_file = folder / f"{STEM}.tex", folder / f"{STEM}.log"
    pdf, png = folder / f"{STEM}.pdf", folder / f"{STEM}.png"
    for stale in (pdf, png):
        stale.unlink(missing_ok=True)
    tex_file.write_text(tex, encoding="utf-8", newline="\n")
    xelatex = [tools["xelatex"], "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
               tex_file.name]
    for _ in range(2):
        code, _told = _run(xelatex, cwd=folder, env=REPRODUCIBLE)
        log = log_file.read_text(encoding="utf-8", errors="replace") if log_file.is_file() else ""
        if code != 0 or not pdf.is_file():
            raise BuildError(f"xelatex failed (exit {code}); see {log_file.as_posix()}")
        if not needs_rerun(log):
            break
    holes = missing_characters(log)
    if holes:
        raise BuildError(f"the font lacks {len(holes)} glyph(s) of the figure's text, e.g. {holes[0]!r}; see "
                         f"{log_file.as_posix()}")
    code, told = _run([tools["rungs"], "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", f"-r{DPI}",
                       "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", f"-sOutputFile={png.name}", pdf.name],
                      cwd=folder)
    data = png.read_bytes() if png.is_file() else b""
    if code != 0 or not data.startswith(PNG_SIGNATURE):
        raise BuildError(f"Ghostscript (rungs) did not convert {pdf.as_posix()} to a PNG "
                         f"(exit {code}: {told})")
    return pdf, png, data


class HalfPublished(OSError):
    """The PDF was replaced and the PNG could not be: the two now come from different builds (publish)."""


def publish(png: Path, pdf: Path, out: Path) -> None:
    """Put the build's PNG at out and its PDF beside it, both or neither (module docstring, Build). A target
    that another program holds against writing raises PermissionError before either changes, and a temporary
    file never stays. Only a program that lets the PNG be written but not replaced can stop its replacement
    after the PDF's: HalfPublished then says so."""
    pairs = ((pdf, out.with_suffix(".pdf")), (png, out))
    out.parent.mkdir(parents=True, exist_ok=True)
    for _, target in pairs:
        if target.exists():
            with target.open("r+b"):  # Windows refuses this while a viewer holds the file
                pass
    staged: list[tuple[Path, Path]] = []
    try:
        for source, target in pairs:
            # staged before the copy, which can fail with a part of the file written
            staged.append((target.with_name(f"{target.name}.new"), target))
            shutil.copyfile(source, staged[-1][0])
        for done, (temporary, target) in enumerate(staged):
            try:
                os.replace(temporary, target)
            except OSError as exc:
                if not done:
                    raise
                raise HalfPublished(f"{target.as_posix()} cannot be replaced ({exc}), but "
                                    f"{staged[0][1].as_posix()} already was: the two now come from different "
                                    f"builds; run again once {target.name} is free") from exc
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.out.suffix.lower() != ".png":
        parser.error(f"--out must name a .png file, with the PDF beside it: {args.out.as_posix()}")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    source = args.routing.as_posix()
    try:
        raw = args.routing.read_bytes()
        routing = json.loads(raw.decode("utf-8"))
        tex = document(routing, load_cost_oracle(args.cost_oracle, raw, routing))
    except NoFrozenThreshold as exc:
        print(f"FAIL: {source}: {exc}; nothing was written", file=sys.stderr)
        return 1
    except ForeignCostOracle as exc:
        print(f"FAIL: {exc}: run scripts/route_cost_oracle.py on {source}; nothing was written",
              file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"FAIL: {source} cannot be drawn with {args.cost_oracle.as_posix()}: not a routing file of "
              f"scripts/route_curve.py and a cost oracle of scripts/route_cost_oracle.py made from it "
              f"({type(exc).__name__}: {exc}); nothing was written", file=sys.stderr)
        return 1
    tools = {tool: shutil.which(tool) for tool in required_tools(WINDOWS)}
    missing = [tool for tool, path in tools.items() if path is None]
    if missing:
        hints = []
        if [tool for tool in missing if tool in TOOLS]:
            hints.append("put TeX Live's bin folder on PATH (xelatex compiles the figure; rungs converts it "
                         "with Ghostscript, TeX Live's own on Windows)")
        if "gs" in missing:
            hints.append("off Windows rungs runs Ghostscript's gs from PATH: install Ghostscript and put gs "
                         "on PATH")
        print(f"FAIL: {' and '.join(missing)} not found on PATH: {'; '.join(hints)}; nothing was written",
              file=sys.stderr)
        return 3
    try:
        pdf, png, data = _build(tex, tools, args.build)
        publish(png, pdf, args.out)
    except BuildError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 4
    except OSError as exc:
        print(f"FAIL: a file of the build cannot be written: {exc}", file=sys.stderr)
        return 4
    width, height = struct.unpack(">II", data[16:24])
    print(f"written {args.out.as_posix()} ({width} x {height} px at {DPI} dpi, {len(data)} bytes) and "
          f"{args.out.with_suffix('.pdf').as_posix()}, from {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
