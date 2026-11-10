"""scripts/route_cost_oracle.py: the oracle of the routing curve's cost row, from the records the frozen
routing file was made from. Offline over the fabricated records of tests/eval/test_route_curve.py, whose
scripts/route_curve.py run freezes the routing file these tests start from."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from research_agent.eval.lock import sha256_of

REPO = Path(__file__).resolve().parents[2]


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


route_cost_oracle = load("scripts/route_cost_oracle.py", "script_route_cost_oracle")
# route_curve.py's fabricated items: FT right on a, b, e and B2 on a, c, d, e, so c and d are the gains (FT
# wrong, B2 right), and c is the cheaper (smaller usage); b is a loss, f wrong for both
curve_tests = load("tests/eval/test_route_curve.py", "route_curve_tests_for_cost_oracle")
INPUTS = ("ft_records", "ft_scores", "b2_records", "b2_scores")


# ---- the oracle at a cost ---------------------------------------------------------------------------------


def test_the_oracle_at_a_cost_routes_only_the_gains_cheapest_first():
    c, d = curve_tests.COSTS[2], curve_tests.COSTS[3]
    assert c < d
    points = route_cost_oracle.cost_oracle(curve_tests.rows())
    assert [p["routed"] for p in points] == [0, 1, 2]
    assert [p["successes"] for p in points] == [3, 4, 5]  # FT's 3, then one more per gain: never the loss b
    assert [p["success"] for p in points] == [3 / 6, 4 / 6, 5 / 6]
    assert [p["cost_per_100_rmb"] for p in points] == [0.0, 100 * c / 6, 100 * (c + d) / 6]


def test_the_gains_go_cheapest_first_whatever_their_order_among_the_items():
    rows = curve_tests.rows()
    rows[2]["cost"], rows[3]["cost"] = rows[3]["cost"], rows[2]["cost"]  # now d is the cheaper gain
    points = route_cost_oracle.cost_oracle(rows)
    assert points[1]["cost_per_100_rmb"] == 100 * curve_tests.COSTS[2] / 6


def test_without_gains_the_oracle_is_fts_success_at_no_cost():
    rows = [{**row, "b2": row["ft"]} for row in curve_tests.rows()]
    assert route_cost_oracle.cost_oracle(rows) == [
        {"routed": 0, "cost_per_100_rmb": 0.0, "successes": 3, "success": 0.5}]


# ---- the script -------------------------------------------------------------------------------------------


def routing_file(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    """route_curve.py's fabricated inputs and the routing file it freezes from them, under tmp_path."""
    paths = curve_tests.fixture(tmp_path)
    routing = tmp_path / "results" / "routing-m4.json"
    assert curve_tests.route_curve.main(curve_tests.argv(paths, routing), root=tmp_path) == 0
    return paths, routing


def run(tmp_path: Path, routing: Path, out: Path, *extra: str) -> int:
    return route_cost_oracle.main(["--routing", str(routing), "--out", str(out), *extra], root=tmp_path)


def test_the_script_writes_the_oracle_with_the_routing_file_and_inputs_it_comes_from(tmp_path, capsys):
    _, routing = routing_file(tmp_path)
    out = tmp_path / "results" / "cost-oracle-m4.json"
    assert run(tmp_path, routing, out) == 0
    raw = out.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"}\n") and raw.startswith(b'{\n  "')
    result = json.loads(raw)
    frozen = json.loads(routing.read_text(encoding="utf-8"))
    assert result["routing"] == {"path": "results/routing-m4.json", "sha256": sha256_of(routing)}
    assert result["inputs"] == {name: frozen["inputs"][name] for name in INPUTS}
    assert (result["cloud_model"], result["n"], result["ft_successes"], result["gains"]) == (
        "deepseek-flash", 6, 3, 2)
    assert result["b2_cost_per_100_rmb"] == frozen["curves"]["routing"][-1]["cost_per_100_rmb"]
    assert result["curve"] == route_cost_oracle.cost_oracle(curve_tests.rows())
    assert "2 gains" in capsys.readouterr().out


@pytest.mark.parametrize("name", INPUTS)
def test_inputs_that_changed_since_the_routing_file_was_frozen_are_refused(tmp_path, capsys, name):
    _, routing = routing_file(tmp_path)
    frozen = json.loads(routing.read_text(encoding="utf-8"))
    changed = tmp_path / frozen["inputs"][name]["path"]
    changed.write_text(changed.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="\n")
    out = tmp_path / "o.json"
    assert run(tmp_path, routing, out) == 2
    err = capsys.readouterr().err
    assert "changed" in err and frozen["inputs"][name]["path"] in err and not out.exists()


def rewrite(routing: Path, change) -> None:
    """The routing file with change applied to its content, as if it had been made so."""
    frozen = json.loads(routing.read_text(encoding="utf-8"))
    change(frozen)
    routing.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8", newline="\n")


@pytest.mark.parametrize(("key", "what"), [("ft_success", "FT's success"), ("b2_success", "B2's success"),
                                           ("everything", "the cost of routing every item")])
def test_rows_that_do_not_give_the_routing_files_numbers_are_refused(tmp_path, capsys, key, what):
    # the inputs as the routing file recorded them, but one of its numbers is not theirs (as with another
    # price of the cloud model, or scores that were edited): these are not the rows it was made from. The
    # check is exact, as the routing file's numbers come from the same arithmetic on the same rows
    _, routing = routing_file(tmp_path)

    def change(frozen: dict) -> None:
        if key == "everything":
            frozen["curves"]["routing"][-1]["cost_per_100_rmb"] *= 1 + 1e-15
        else:
            frozen[key] += 1e-15

    rewrite(routing, change)
    out = tmp_path / "o.json"
    assert run(tmp_path, routing, out) == 2
    assert what in capsys.readouterr().err and not out.exists()


@pytest.mark.parametrize("ends", [0, 2])
def test_a_scan_without_a_single_route_everything_point_is_malformed(tmp_path, capsys, ends):
    _, routing = routing_file(tmp_path)

    def change(frozen: dict) -> None:
        scan = frozen["curves"]["routing"]
        scan[-1:] = scan[-1:] * ends

    rewrite(routing, change)
    out = tmp_path / "o.json"
    assert run(tmp_path, routing, out) == 1
    err = capsys.readouterr().err
    assert "route_everything" in err and "Traceback" not in err and not out.exists()


def test_an_existing_result_is_kept_unless_forced(tmp_path):
    _, routing = routing_file(tmp_path)
    out = tmp_path / "o.json"
    out.write_text("{}\n", encoding="utf-8")
    assert run(tmp_path, routing, out) == 3
    assert out.read_text(encoding="utf-8") == "{}\n"
    assert run(tmp_path, routing, out, "--force") == 0
    assert json.loads(out.read_text(encoding="utf-8"))["gains"] == 2


def test_a_routing_file_that_is_not_route_curves_is_malformed_input(tmp_path, capsys):
    out = tmp_path / "o.json"
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    assert run(tmp_path, tmp_path / "broken.json", out) == 1
    (tmp_path / "other.json").write_text('{"n": 6}\n', encoding="utf-8")
    assert run(tmp_path, tmp_path / "other.json", out) == 1
    assert run(tmp_path, tmp_path / "absent.json", out) == 1
    assert "Traceback" not in capsys.readouterr().err and not out.exists()


def test_test_material_is_refused(tmp_path):
    # a routing file made from records that hold a test item: route_curve.py would have refused them, so the
    # routing file here records the changed file's SHA-256 by hand
    paths, routing = routing_file(tmp_path)
    records = [json.loads(line) for line in paths["ft"].read_text(encoding="utf-8").splitlines()]
    records[0]["item"]["split"] = "test"
    curve_tests.write_jsonl(paths["ft"], records)
    frozen = json.loads(routing.read_text(encoding="utf-8"))
    frozen["inputs"]["ft_records"]["sha256"] = sha256_of(paths["ft"])
    routing.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8", newline="\n")
    out = tmp_path / "o.json"
    assert run(tmp_path, routing, out) == 4
    assert not out.exists()
