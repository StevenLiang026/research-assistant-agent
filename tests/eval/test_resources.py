"""research_agent.eval.resources offline: latency percentiles from timing sidecars (local calls, paid cloud
calls only, end-to-end steps), the cost of a cloud system's turns, and the GPU over a fake nvidia-smi (a
one-off query and the sampling loop). Nothing runs nvidia-smi."""

from __future__ import annotations

import subprocess

import pytest
from support.fake_llama import ServerProcess

from research_agent.eval.resources import (
    GpuError,
    GpuSample,
    GpuSampler,
    cloud_cost,
    e2e_step_latency,
    gpu_sample,
    parse_sample,
    percentile,
    server_resources,
    single_step_latency,
)
from research_agent.llm.budget import Price


def test_percentiles_interpolate_between_closest_ranks_as_numpy_does():
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.5  # unsorted input
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)
    assert percentile([7.0], 95) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)


def call(wall_ms: float, *, error: str | None = None, paid: bool | None = None, server: dict | None = None,
         step: int | None = None) -> dict:
    """One call of a timing sidecar (eval.timing.TimedModel)."""
    record = {"wall_ms": wall_ms, "error": error, "paid": paid, "cached": None if paid is None else not paid,
              "server": server}
    return record if step is None else {"step": step, **record}


def test_single_step_latency_of_a_local_system_is_over_the_calls_that_came_back_with_the_servers_own_time():
    items = [{"id": "a", "status": "ok", "wall_ms": 120.0,
              "calls": [call(100.0, paid=False, server={"prompt_ms": 60.0, "predicted_ms": 30.0})]},
             {"id": "b", "status": "ok", "wall_ms": 220.0,
              "calls": [call(200.0, paid=False, server={"prompt_ms": 150.0, "predicted_ms": 30.0})]},
             {"id": "c", "status": "error", "wall_ms": 5.0, "calls": [call(4.0, error="HTTPError")]},
             {"id": "d", "status": "context_overflow", "wall_ms": 1.0, "calls": []}]
    latency = single_step_latency(items, cloud=False)
    assert latency == {"basis": "calls", "n": 2, "p50_ms": 150.0, "p95_ms": 195.0,
                       "server": {"n": 2, "p50_ms": 135.0, "p95_ms": 175.5}}


def test_single_step_latency_of_a_cloud_system_counts_the_paid_calls_only():
    items = [{"id": "a", "status": "ok", "calls": [call(3000.0, paid=True)]},
             {"id": "b", "status": "ok", "calls": [call(2.0, paid=False)]},  # served from the cache
             {"id": "c", "status": "ok", "calls": [call(5000.0, paid=True)]}]
    latency = single_step_latency(items, cloud=True)
    assert latency == {"basis": "paid calls", "n": 2, "p50_ms": 4000.0, "p95_ms": 4900.0}
    nothing_paid = single_step_latency([items[1]], cloud=True)
    assert nothing_paid == {"basis": "paid calls", "n": 0, "p50_ms": None, "p95_ms": None}


def test_e2e_step_latency_is_per_step_and_for_a_cloud_system_over_the_steps_whose_calls_were_all_paid():
    tasks = [{"id": "t1", "steps": [{"index": 1, "calls": 2, "wall_ms": 300.0},
                                    {"index": 2, "calls": 1, "wall_ms": 100.0}],
              "calls": [call(100.0, paid=True, step=1), call(200.0, paid=True, step=1),
                        call(100.0, paid=False, step=2)]},
             {"id": "t2", "steps": [{"index": 1, "calls": 1, "wall_ms": 700.0},
                                    {"index": 2, "calls": 0, "wall_ms": 0.0}],
              "calls": [call(700.0, paid=True, step=1)]},
             {"id": "t3", "steps": [{"index": 1, "calls": 1, "wall_ms": 50.0}],
              "calls": [call(50.0, error="RuntimeError", step=1)]}]  # a step whose call raised
    assert e2e_step_latency(tasks, cloud=False) == {"basis": "steps", "n": 3, "p50_ms": 300.0,
                                                    "p95_ms": 660.0}
    assert e2e_step_latency(tasks, cloud=True) == {"basis": "steps whose calls were all paid", "n": 2,
                                                   "p50_ms": 500.0, "p95_ms": 680.0}


PRICE = Price(input_cache_miss=2.0, output=8.0, input_cache_hit=0.5)


def test_cloud_cost_is_what_the_paid_turns_cost_and_what_every_turn_would_cost_at_the_list_price():
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 600,
             "prompt_cache_miss_tokens": 400}
    records = [{"id": "a", "status": "ok", "usage": usage, "paid": True, "cached": False},
               {"id": "b", "status": "ok", "usage": usage, "paid": False, "cached": True},
               {"id": "c", "status": "budget_stop"}]
    tasks = [{"id": "t1", "cost_rmb": 0.004, "paid_calls": 2, "cache_hits": 1,
              "tokens": {"prompt_tokens": 2000, "completion_tokens": 200}},
             {"id": "t2", "cost_rmb": 0.0, "paid_calls": 0, "cache_hits": 3,
              "tokens": {"prompt_tokens": 1000, "completion_tokens": 100}}]
    cost = cloud_cost(PRICE, records, tasks)
    one_turn = (600 * 0.5 + 400 * 2.0 + 100 * 8.0) / 1e6
    assert cost["single_step"] == {"turns": 2, "paid": 1, "cached": 1, "paid_rmb": pytest.approx(one_turn),
                                   "list_rmb": pytest.approx(2 * one_turn)}
    every_task = (3000 * 2.0 + 300 * 8.0) / 1e6  # token totals, every prompt token at the cache-miss price
    assert cost["e2e"] == {"tasks": 2, "paid_calls": 2, "cache_hits": 4, "paid_rmb": pytest.approx(0.004),
                           "list_rmb": pytest.approx(every_task)}
    assert cost["paid_rmb"] == pytest.approx(one_turn + 0.004)
    assert cost["list_rmb"] == pytest.approx(2 * one_turn + every_task)
    assert cost["per_task_fen"] == pytest.approx(0.2)  # 0.004 RMB over 2 tasks
    assert cloud_cost(PRICE, [], [])["per_task_fen"] is None


# ---- the GPU ----------------------------------------------------------------------------------------------


def test_a_sample_line_is_the_memory_in_mib_and_the_power_in_watts_when_reported():
    assert parse_sample("4100, 85.32\n") == GpuSample(4100.0, 85.32)
    assert parse_sample("4100, [N/A]") == GpuSample(4100.0, None)
    assert parse_sample("4100, [Not Supported]\r\n") == GpuSample(4100.0, None)
    for line in ("", "\n", "memory.used, power.draw", "No devices were found", "[N/A], 30.0"):
        assert parse_sample(line) is None, line


def test_gpu_sample_asks_nvidia_smi_once_for_gpu_0():
    asked = []

    def run(cmd, **kwargs):
        asked.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="512, 12.5\n", stderr="")

    assert gpu_sample(run=run) == GpuSample(512.0, 12.5)
    assert asked == [["nvidia-smi", "--id=0", "--query-gpu=memory.used,power.draw",
                      "--format=csv,noheader,nounits"]]


@pytest.mark.parametrize("outcome, message", [
    (OSError("[WinError 2] The system cannot find the file specified"), "nvidia-smi cannot be run"),
    (subprocess.CompletedProcess([], 9, stdout="", stderr="NVIDIA-SMI has failed"), "exited with 9"),
    (subprocess.CompletedProcess([], 0, stdout="No devices were found\n", stderr=""), "no sample"),
])
def test_gpu_sample_raises_gpu_error_when_nvidia_smi_gives_no_sample(outcome, message):
    def run(cmd, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    with pytest.raises(GpuError, match=message):
        gpu_sample(run=run)


def test_the_sampler_runs_nvidia_smis_loop_and_keeps_every_sample_until_stopped():
    process = ServerProcess(["3900, 40.0\n", "4600, 120.5\n", "garbage\n", "4500, [N/A]\n"])
    started = []

    def popen(cmd, **kwargs):
        started.append(cmd)
        return process

    with GpuSampler(interval_ms=200, popen=popen) as sampler:
        pass
    assert started == [["nvidia-smi", "--id=0", "--query-gpu=memory.used,power.draw",
                        "--format=csv,noheader,nounits", "--loop-ms=200"]]
    assert process.terminated and sampler.samples == [GpuSample(3900.0, 40.0), GpuSample(4600.0, 120.5),
                                                      GpuSample(4500.0, None)]


def test_a_sampler_that_cannot_start_raises_gpu_error():
    def popen(cmd, **kwargs):
        raise FileNotFoundError(2, "No such file", cmd[0])

    with pytest.raises(GpuError, match="nvidia-smi cannot be run"):
        GpuSampler(popen=popen).start()


def test_server_resources_are_the_peak_over_the_idle_baseline_and_the_power_over_the_samples():
    samples = [GpuSample(3900.0, 40.0), GpuSample(4600.0, 120.0), GpuSample(4500.0, None)]
    resources = server_resources(GpuSample(512.0, 10.0), samples, interval_ms=200)
    assert resources == {"baseline_mib": 512.0, "peak_mib": 4600.0, "vram_mib": 4088.0, "samples": 3,
                         "interval_ms": 200, "power_mean_w": 80.0, "power_peak_w": 120.0,
                         "energy_wh": pytest.approx(160 * 0.2 / 3600)}
    empty = server_resources(GpuSample(512.0, None), [], interval_ms=200)
    assert empty["vram_mib"] is None and empty["power_mean_w"] is None and empty["samples"] == 0
