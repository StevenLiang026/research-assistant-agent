"""Resources of the phase-4 test session (docs: the M3 plan, phase 4; docs/plan.md, 数字对账): the latency of
the model calls from the timing sidecars, the GPU's memory and power from nvidia-smi, and the cost of a cloud
system's turns.

Latency (research_agent.eval.timing's sidecars: ItemTimer's items, TaskTimer's tasks). single_step_latency is
over the wall time of every model call that came back (no error); for a cloud system over the paid calls only,
since a turn served from the response cache took no API time and would pull the percentiles down; for a local
system also over llama.cpp's own time of each completion, prompt plus generation (the server's timings,
prompt_ms + predicted_ms), where the server reported it. e2e_step_latency is over the wall time of every step
of a task (the sum of its calls': the model's time, not the replayed tools'), the steps whose calls all came
back, and for a cloud system the steps whose calls were all paid. Each gives n with p50 and p95, by linear
interpolation between closest ranks (percentile: numpy's default, as eval.metrics' bootstrap intervals), None
over no value. Local latency is an upper bound: every prompt is evaluated from scratch (LocalClient sends
cache_prompt false), where a deployment would reuse the prefix a step shares with the step before.

GPU (nvidia-smi, on PATH). gpu_sample asks once for GPU 0's memory used (MiB) and power draw (W; None where
the GPU does not report it); GpuSampler runs nvidia-smi's own loop (--loop-ms) beside a server and keeps one
sample per interval until it is stopped. server_resources gives a server's VRAM, the peak memory over its
samples minus the idle baseline sampled just before it started (Windows reports no memory per process, so
the difference is what can be attributed to the server), and its power: the mean and peak draw over the
samples and the energy, the sum of the draws times the interval (samples without a draw count as none).

Costs (llm.budget's price list, an upper bound). cloud_cost gives what the paid turns cost (the single-step
records with paid true, and the end-to-end tasks' cost_rmb, which counts their paid turns) and what all the
turns would cost at the list price, those served from the cache included (each single-step record's usage,
with the API's own split into cache hits and misses where it was recorded; each task's token totals, every
prompt token at the cache-miss price), and the paid cost per end-to-end task in fen.
"""

from __future__ import annotations

import math
import subprocess
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Self

from ..llm.budget import Price, cost_rmb
from ..llm.server import Process, stop
from .runner import OK

SMI = "nvidia-smi"
QUERY = ("--query-gpu=memory.used,power.draw", "--format=csv,noheader,nounits")
PAID_CALLS, CALLS = "paid calls", "calls"
PAID_STEPS, STEPS = "steps whose calls were all paid", "steps"


class GpuError(RuntimeError):
    """nvidia-smi cannot be run or gives no sample."""


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile of values by linear interpolation between closest ranks (numpy's default);
    ValueError for no value."""
    if not values:
        raise ValueError("no value to take a percentile of")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def latency(values: Iterable[float]) -> dict:
    """{"n", "p50_ms", "p95_ms"} of the values (milliseconds); the percentiles are None for no value."""
    values = list(values)
    if not values:
        return {"n": 0, "p50_ms": None, "p95_ms": None}
    return {"n": len(values), "p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95)}


def server_ms(call: dict) -> float | None:
    """llama.cpp's own time of a call's completion, prompt plus generation; None without its timings."""
    timings = call.get("server")
    if not isinstance(timings, dict):
        return None
    prompt, predicted = timings.get("prompt_ms"), timings.get("predicted_ms")
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in (prompt, predicted)):
        return None
    return prompt + predicted


def single_step_latency(items: list[dict], *, cloud: bool) -> dict:
    """The latency of a single-step run from its sidecar's items (module docstring)."""
    calls = [c for item in items for c in item["calls"]
             if c.get("error") is None and (c.get("paid") or not cloud)]
    out = {"basis": PAID_CALLS if cloud else CALLS, **latency(c["wall_ms"] for c in calls)}
    if not cloud:
        out["server"] = latency(ms for c in calls if (ms := server_ms(c)) is not None)
    return out


def e2e_step_latency(tasks: list[dict], *, cloud: bool) -> dict:
    """The latency per step of an end-to-end run from its sidecar's tasks (module docstring)."""
    walls = []
    for task in tasks:
        for step in task["steps"]:
            calls = [c for c in task["calls"] if c.get("step") == step["index"]]
            if not calls or any(c.get("error") is not None for c in calls):
                continue
            if cloud and not all(c.get("paid") for c in calls):
                continue
            walls.append(step["wall_ms"])
    return {"basis": PAID_STEPS if cloud else STEPS, **latency(walls)}


def cloud_cost(price: Price, records: list[dict], tasks: list[dict]) -> dict:
    """What a cloud system's turns cost and would cost at the list price (module docstring)."""
    ok = [r for r in records if r.get("status") == OK]
    paid = [r for r in ok if r.get("paid")]
    single = {"turns": len(ok), "paid": len(paid), "cached": sum(bool(r.get("cached")) for r in ok),
              "paid_rmb": sum(cost_rmb(price, r.get("usage") or {}) for r in paid),
              "list_rmb": sum(cost_rmb(price, r.get("usage") or {}) for r in ok)}
    e2e = {"tasks": len(tasks), "paid_calls": sum(t.get("paid_calls") or 0 for t in tasks),
           "cache_hits": sum(t.get("cache_hits") or 0 for t in tasks),
           "paid_rmb": sum(t.get("cost_rmb") or 0.0 for t in tasks),
           "list_rmb": sum(cost_rmb(price, t.get("tokens") or {}) for t in tasks)}
    return {"paid_rmb": single["paid_rmb"] + e2e["paid_rmb"],
            "list_rmb": single["list_rmb"] + e2e["list_rmb"],
            "per_task_fen": 100 * e2e["paid_rmb"] / len(tasks) if tasks else None,
            "single_step": single, "e2e": e2e}


# ---- the GPU ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuSample:
    memory_mib: float
    power_w: float | None


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse_sample(line: str) -> GpuSample | None:
    """One line of the query ("4100, 85.32"); None for a line that holds no memory figure."""
    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) != 2:
        return None
    memory = _number(parts[0])
    return None if memory is None else GpuSample(memory, _number(parts[1]))


def command(gpu: int = 0, *, loop_ms: int | None = None) -> list[str]:
    cmd = [SMI, f"--id={gpu}", *QUERY]
    return cmd if loop_ms is None else [*cmd, f"--loop-ms={loop_ms}"]


def gpu_sample(*, run: Callable = subprocess.run, gpu: int = 0) -> GpuSample:
    """GPU gpu's memory used and power draw now (one nvidia-smi query); GpuError when there is none."""
    try:
        done = run(command(gpu), capture_output=True, text=True, encoding="utf-8", errors="replace",
                   timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuError(f"{SMI} cannot be run: {exc}") from None
    if done.returncode != 0:
        why = " ".join((done.stderr or done.stdout or "").split())[:200]
        raise GpuError(f"{SMI} exited with {done.returncode}: {why}")
    for line in (done.stdout or "").splitlines():
        sample = parse_sample(line)
        if sample is not None:
            return sample
    raise GpuError(f"{SMI} gave no sample: {(done.stdout or '').strip()[:200]!r}")


class GpuSampler:
    """nvidia-smi's sampling loop (module docstring): start, then stop, which ends the loop and returns the
    samples; or with GpuSampler(...) as sampler: ..., whose samples are complete once the block has ended."""

    def __init__(self, *, interval_ms: int = 200, gpu: int = 0, popen: Callable = subprocess.Popen) -> None:
        self.interval_ms, self.gpu, self._popen = interval_ms, gpu, popen
        self.samples: list[GpuSample] = []
        self.process: Process | None = None
        self._reader: threading.Thread | None = None

    def start(self) -> Self:
        try:
            self.process = self._popen(command(self.gpu, loop_ms=self.interval_ms), stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                                       errors="replace", bufsize=1)
        except (OSError, subprocess.SubprocessError) as exc:
            raise GpuError(f"{SMI} cannot be run: {exc}") from None
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        return self

    def _read(self) -> None:
        for line in self.process.stdout:
            sample = parse_sample(line)
            if sample is not None:
                self.samples.append(sample)

    def stop(self) -> list[GpuSample]:
        """End the loop and return every sample read; safe to call more than once."""
        if self.process is not None:
            stop(self.process)
        if self._reader is not None:
            self._reader.join(timeout=10)  # its output ends with the process
            self._reader = None
        return list(self.samples)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def server_resources(baseline: GpuSample, samples: list[GpuSample], *, interval_ms: int) -> dict:
    """A server's VRAM over the idle baseline and its power over the samples (module docstring)."""
    peak = max((s.memory_mib for s in samples), default=None)
    powers = [s.power_w for s in samples if s.power_w is not None]
    return {"baseline_mib": baseline.memory_mib, "peak_mib": peak,
            "vram_mib": None if peak is None else peak - baseline.memory_mib, "samples": len(samples),
            "interval_ms": interval_ms, "power_mean_w": sum(powers) / len(powers) if powers else None,
            "power_peak_w": max(powers, default=None),
            "energy_wh": sum(powers) * interval_ms / 1000 / 3600 if powers else None}
