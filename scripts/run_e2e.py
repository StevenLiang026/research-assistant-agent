"""Run one system end to end over a set of tasks on the frozen recordings and score every task
(research_agent.eval.e2e; docs/plan.md, 评测: 端到端).

    .venv/Scripts/python.exe scripts/run_e2e.py data/eval/e2e-dev.jsonl --system b0|b1|b2|b3|ft|b4|ftr \\
        --out runs/eval/e2e-<name>/ [--server http://127.0.0.1:8080] [--stage M3] [--max-cost RMB] \\
        [--limit N] [--tokenizer models/Qwen3-1.7B | --no-tokenizer] [--seed 42] [--today YYYY-MM-DD] \\
        [--mode think|no_think] [--base-seed N] [--reasoning-budget N] [--greedy] [--training-length N] \\
        [--votes K [--vote-mode final|reproduction]] [--rules] [--glossary data/glossary/zh-en.json] \\
        [--routing-config results/routing-m7.json] [--fault-rate 0|0.2|0.5] [--fault-persistent]

The systems are those of scripts/run_eval.py (research_agent.eval.systems; b1's demonstrations are appended to
the system message of every step). There is no replay system here: an end-to-end task has no stored answer.

Routing (ftr; docs/m7-计划.md WP-K). Without --routing-config, ftr routes as it always has: by its default
frozen configuration, results/routing-m4.json, checked against data/eval/dev.jsonl, and sticky (once a step is
routed, the cloud system answers the rest of the task), and the run writes what it wrote before M7.
--routing-config FILE (ftr only; a usage error for any other system) routes by FILE instead, checked against
the development sets FILE says it was chosen on, as they are now (eval.systems.load_routing: dev and dev-v2
for M7's R3), and step by step (agent.route.RoutedModel's stepwise routing): every step is the local model's
first, checked with the frozen policy (agent.route.route_reason with FILE's signal, threshold and rules: the
hard triggers, the budget trigger of a reasoning closed by force, the parameter triggers, the text exemption,
the threshold) on the signals of that step's answer, and a routed step is answered by the cloud system FILE
names (b2, deepseek-flash, for R3) instead, the next step asked of the local model again; a final text answer
is never routed unless a hard or the budget trigger fires. The manifest then says "e2e_routing": "stepwise",
and every task record carries "routing": the transcript steps with a turn the cloud answered
("routed_steps"), the turns the local model and the cloud answered, each routed turn's step, the reason it
was routed, whether it was paid and its cost at the list price ("routed"), and the cloud's cost in all
("cloud_cost_rmb", the record's cost_rmb, since local turns cost nothing). Each model turn is placed in the
step whose attempt it was, as the timing sidecar places calls (research_agent.eval.timing).
A local system runs on the family path as there (the --tokenizer folder's family renders and encodes every
step's prompt, sent as token ids; research_agent.llm.generation), and --mode, --reasoning-budget and
--base-seed set its generation as run_eval's --mode, --reasoning-budget and --seed do (the base seed has its
own name here, since --seed is the tasks' seed below); --greedy is run_eval's --greedy too: think mode decoded
greedily (temperature 0 alone is sent, so no seed goes out), whatever the family's sampling for that mode, as
the FT3 decoding rule may choose (docs/m7-进度.md, FT3 的选点与解码规则); it cannot be combined with --mode
no_think. The manifest records the settings ("generation", --greedy's as run_eval records them: mode think,
the greedy decoding, the base seed given) with the family ("family"), and every attempt of a thinking
generation in a transcript carries its "reasoning" and "thinking", while a greedy no_think transcript is as it
was before M7. --training-length N is the max_length the local model was trained with (default 4096, FT2's;
5120 for the M7 runs), and the manifest counts the local steps whose prompt and generation together exceeded
it ("training_length", "steps_over_training_length"; docs/m7-计划.md §2 端到端预算): a step's length is its
prompt ids as the context check measured them plus what it generated, for a turn with a thinking record its
reasoning tokens, a forced close's tokens when its reasoning was closed by force (the ids generation appends,
family.encode_close()), and its answer tokens, for any other turn its usage's completion tokens (StepLengths).
The steps the cloud answered are not measured, and without the context check (--no-tokenizer) none is, so the
count is null.

The inference wrappers (M7, docs/m7-本地提升计划.md WP2 to WP4), as run_eval's: --votes K answers every
step by a vote of K samples, the greedy answer breaking ties (sample k with the base seed --base-seed + k),
--vote-mode keys the answers (final, the default, or the analysis' reproduction), --rules applies the runtime
rules to every answer over the transcript so far (each candidate's when voting; the loop then checks the
rewritten calls) and --glossary FILE puts the glossary's note on every prompt, never into the transcript.
Every attempt of a voted or ruled step carries its "votes" and "rules" in the transcript, a voted step's
usage and its timings in the sidecar sum its K + 1 requests, and the manifest records each option only when
used, as run_eval's does. A candidate that fails (raises: a HarnessError, which alone ends a task as
harness_error, or any other exception) does not end the step unless the vote chooses its answer: the vote
goes on without it and lists it under the attempt's votes "errors" (agent.vote, Failed candidates). So that
they stay visible, with --votes the summary counts the failed candidates of every transcript's votes by
error type ("candidate_errors", {} when none failed; a task whose transcript is lost to an exception counts
none), and the run prints them; without --votes the summary has no such key. Without the four flags every
output is what it was.

Which sets, as in scripts/run_eval.py: a locked test set (data/eval/e2e-test.jsonl or data/eval/test.jsonl,
however the path is spelled, or a copy with a test set's digest) is refused before anything is read (exit
4), since only the phase-4 test runner opens one, and so are, by location, the files of the test set's
handwritten items with their gold turns (data/eval/handwritten.jsonl, handwritten-specs.jsonl), and, once
read, any tasks file that holds test material (a task of split "test" or with an id of the test split,
eval.lock.refuse_test_material), before any task runs; the end-to-end development tasks
data/eval/e2e-dev.jsonl, and any tasks file with a lock beside it, must match their lock (exit 5 otherwise).

Each task runs through research_agent.data.synth.run_request in replay over data/cassettes: search_arxiv on
the corpus index of the 300 recorded papers, get_paper and fetch_fulltext on the recorded pages, no rewriting
and, by default, no fault injection, the owner's functions from the tree, today the corpus recording date
(2026-10-10).
Each task has its own note store in <out>/notes/<id>/, emptied and seeded with the five scaffold notes that
generation gives a request of its split (the split's topics from data/synth/requests-manifest.json). The
context check uses the tokenizer as the agent loop does (--no-tokenizer skips it).

Fault injection (M4; docs/plan.md, 故障注入). --fault-rate R (default 0; the plan's rates are 0, 0.2 and
0.5, net.faults.RATES) gives every task its own pair of injectors, a FaultInjector over the fetcher and
SearchFaults over the search, keyed by --seed and the task id, with all seven faults (net.faults.FAULTS; the
search can meet four of them) and no fault budget; --fault-persistent makes every request of a task for a URL
or a search meet the same fault. The delay fault sleeps for real (net.faults.DELAY_S). Every fault a task
meets is recorded in its record ("faults") and its transcript (research_agent.eval.e2e, Fault injection), the
manifest records the condition as "fault_injection": {"rate", "persistent", "faults", "seed"}, and the timing
sidecar gives every task its arXiv tool calls ("tools": tool, step, call, wall time, the faults met), from
which scripts/fault_table.py measures the delay. At rate 0 nothing is injected or traced and the run writes
exactly what it wrote before M4: no record and no manifest names a fault (--fault-persistent included). A run
folder belongs to one condition: --out must be absent or empty (exit 6 otherwise), so the records of two
conditions or two systems never mix.

Scoring (research_agent.eval.e2e.score_task): check_trajectory's five checks, the task's expect fields
(final status, a note on the right paper or topic when one is required, cited ids within the tool outputs,
at least one expected tool or no call at all) and the evidence the model quotes in every scaffold write_note
call, each fragment found, letters and digits compared, in text an earlier step retrieved about that paper
(the stored note's tool-picked sentences are reported for information only). Recorded per task: the checks,
steps, tool calls, tool errors by code, tokens, cost and call counts, and no time.

Writes <out>/transcripts/<id>.json, <out>/notes/<id>/ and <out>/report.json: {"manifest": the system, model,
lora scale, prompt and schema hashes, the tasks file, its SHA-256 and whether its lock was checked
("tasks_locked"), the corpus manifest's SHA-256, today, the seed, the date, the server's /props and the
adapters it loaded (each one's path as the server was given it and the GGUF's SHA-256) for a local system,
the stage, the cap and the turns per DeepSeek thinking mode ("modes", research_agent.eval.systems) for a
cloud system, git; "summary": success overall, per check, per intent and per language, statuses, totals
(with --votes, the votes' failed candidates); "tasks": one record per task}, which holds no wall time, so
that two runs with the same answers write the same summary and tasks; and the timing sidecar
<out>/timing.json (research_agent.eval.timing): per task its wall time (from the previous task's record to
its own), its steps' model time and every model call (wall time, error, paid or cached, llama.cpp's own
timings) with its step. Prints one line per task and the summary, its wall time from the sidecar.

Exit 0 when the tasks ran (whatever their scores); 1 on malformed tasks; 2 when the system cannot be built
here (no key, no price, no server, no training set for b1, a loaded adapter whose GGUF is not here to hash, a
routing configuration that ftr cannot run or whose development sets changed, a thinking run of a family that
never thinks, a vote whose decoding does not sample, a glossary that cannot be read) or on a usage error
(--mode, --reasoning-budget, --greedy or --training-length with a system that is not local, --greedy with
--mode no_think, --routing-config with a system that is not routed, --votes, --rules or --glossary with a
system that is not local, --vote-mode without --votes, --votes with --greedy); 3 when
the spending cap stopped the run (the report covers the tasks run until then); 4 when the tasks are a locked
test set, a file of the handwritten test items or any other test material and 5 when they do not match their
lock (nothing is written); 6 when --out exists and is not empty (nothing runs, nothing is written). A
--fault-rate outside [0, 1] is a usage error (exit 2, from argparse).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from research_agent.agent.context import TokenCounter, family_token_counter, qwen_token_counter
from research_agent.agent.loop import Model
from research_agent.agent.prompts import PROMPT_HASH, PROMPT_VERSION
from research_agent.agent.route import CLOUD, LOCAL
from research_agent.agent.types import BUDGET_STOP, AssistantTurn
from research_agent.agent.vote import FINAL
from research_agent.data.synth import FaultSetting, corpus_recorded_on, probe
from research_agent.eval.e2e import REPORT_FILE, World, build_world, load_tasks, run_tasks, summarize
from research_agent.eval.lock import (
    LockError,
    RefusedTestSet,
    check_lock,
    lock_required,
    refuse_test_material,
    refuse_test_sets,
    sha256_of,
)
from research_agent.eval.systems import (
    CLOUD_THINKING,
    DEFAULT_STAGE,
    LOCAL_SCALES,
    ROUTED,
    STAGE_CAPS,
    SYSTEMS,
    TRAINING_SET,
    SystemUnavailable,
    add_wrapper_flags,
    build_system,
    with_votes,
    wrapper_flags_error,
)
from research_agent.eval.timing import TaskTimer, TimedModel, write_timing
from research_agent.family.core import GREEDY, LoadedFamily, Mode, Thinking, load_family
from research_agent.llm.generation import GenerationSettings
from research_agent.tools.corpus_search import MANIFEST
from research_agent.tools.schemas import CONTEXT_TOKENS, SCHEMA_HASH
from research_agent.train.manifest import REPO, git_commit, git_dirty

TOKENIZER = "models/Qwen3-1.7B"
LLM_DIR = "runs/llm"
SERVER = "http://127.0.0.1:8080"
CASSETTES = "data/cassettes"
REQUESTS_MANIFEST = "data/synth/requests-manifest.json"
TIMING_FILE = "timing.json"
E2E_SYSTEMS = tuple(s for s in SYSTEMS if s != "replay")
STEPWISE = "stepwise"  # how a run with --routing-config routes (the manifest's e2e_routing)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def iso_date(text: str) -> str:
    if not _DATE.fullmatch(text):
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}")
    return text


def fault_rate(text: str) -> float:
    try:
        rate = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number in [0, 1], got {text!r}") from None
    if not 0.0 <= rate <= 1.0:
        raise argparse.ArgumentTypeError(f"expected a number in [0, 1], got {text!r}")
    return rate


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run one system end to end over tasks and score them.")
    ap.add_argument("tasks", type=Path, help="the tasks (JSONL), e.g. data/eval/e2e-dev.jsonl")
    ap.add_argument("--system", required=True, choices=E2E_SYSTEMS)
    ap.add_argument("--out", type=Path, required=True, help="the run folder, e.g. runs/eval/e2e-<name>/")
    ap.add_argument("--server", default=SERVER, help=f"llama-server for local systems (default {SERVER})")
    ap.add_argument("--stage", default=DEFAULT_STAGE, choices=sorted(STAGE_CAPS))
    ap.add_argument("--cloud-thinking", default="auto", choices=CLOUD_THINKING,
                    help="a cloud system's thinking (or a routed system's cloud side): auto or off")
    ap.add_argument("--max-cost", type=float, default=None, metavar="RMB",
                    help="what this invocation may add to the stage's spending "
                         "(default: up to the stage cap)")
    ap.add_argument("--limit", type=int, default=None, metavar="N", help="run only the first N tasks")
    ap.add_argument("--tokenizer", default=TOKENIZER,
                    help=f"the model folder (a registered family, research_agent.family.registry) whose "
                         f"template renders every prompt and measures the context (default {TOKENIZER})")
    ap.add_argument("--no-tokenizer", action="store_true", help="skip the context check")
    ap.add_argument("--seed", type=int, default=42, help="the tasks' seed: the scaffold notes and the faults")
    ap.add_argument("--mode", choices=[str(mode) for mode in Mode], default=None,
                    help="the generation mode of a local system (default: the family's inference mode)")
    ap.add_argument("--base-seed", type=int, default=0, metavar="N",
                    help="the base seed of a sampled generation (default 0), run_eval's --seed")
    ap.add_argument("--reasoning-budget", type=int, default=None, metavar="N",
                    help="the tokens the reasoning may take before it is closed by force (default: the "
                         "family's)")
    ap.add_argument("--greedy", action="store_true",
                    help="think mode decoded greedily (temperature 0, no seed), whatever the family's "
                         "sampling for that mode: run_eval's --greedy")
    ap.add_argument("--training-length", type=int, default=None, metavar="N",
                    help=f"the max_length the local model was trained with (default {CONTEXT_TOKENS}, FT2's; "
                         "5120 for the M7 runs): the manifest counts the steps whose prompt and generation "
                         "together exceed it")
    add_wrapper_flags(ap)
    ap.add_argument("--routing-config", type=Path, default=None, metavar="FILE",
                    help="ftr only: route by this frozen configuration (scripts/route_curve.py), every step "
                         "on its own, and record the routed steps (default: ftr's results/routing-m4.json, a "
                         "routed task staying in the cloud)")
    ap.add_argument("--today", type=iso_date, default=None, help="default: the corpus recording date")
    ap.add_argument("--fault-rate", type=fault_rate, default=0.0, metavar="R",
                    help="the share of tool requests that meet an injected fault (default 0: none)")
    ap.add_argument("--fault-persistent", action="store_true",
                    help="every request of a task for a URL or a search meets the same fault")
    ap.add_argument("--cassettes", default=CASSETTES, help=argparse.SUPPRESS)
    ap.add_argument("--corpus-manifest", default=MANIFEST.as_posix(), help=argparse.SUPPRESS)
    ap.add_argument("--requests-manifest", default=REQUESTS_MANIFEST, help=argparse.SUPPRESS)
    ap.add_argument("--llm-dir", default=LLM_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--training-set", default=str(TRAINING_SET), help=argparse.SUPPRESS)
    return ap


def local_date() -> str:
    """Today's date on this machine's clock, for the manifest."""
    return datetime.datetime.now(datetime.UTC).astimezone().date().isoformat()


def percent(rate: float | None) -> str:
    return "-" if rate is None else f"{100 * rate:.1f}%"


class StepLengths:
    """The model, with every local step measured against the training length (module docstring): the prompt
    as the loop measured it (count_tokens: the ids the local model is sent) plus what the step generated, for
    a turn with a thinking record its reasoning tokens, the forced close's tokens when the reasoning was
    closed by force, and its answer tokens, for any other turn the completion tokens of its usage. The steps
    the cloud answered (a cloud system's, a routed step's) are not measured: the training length is the local
    model's. Everything else is the wrapped model's (name, cost, begin_task...)."""

    def __init__(self, model: Model, *, count_tokens: TokenCounter, training_length: int,
                 close_tokens: int) -> None:
        self.model, self.count_tokens = model, count_tokens
        self.training_length, self.close_tokens = training_length, close_tokens
        self.steps = 0  # the local steps measured
        self.over = 0  # of them, those longer than the training length

    @property
    def name(self) -> str:
        return self.model.name

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        turn = self.model.next_turn(messages, tools)
        if not turn.paid and not turn.cached and (turn.route or {}).get("answered_by") != CLOUD:
            self.steps += 1
            length = self.count_tokens(messages, tools) + generated_tokens(turn, self.close_tokens)
            if length > self.training_length:
                self.over += 1
        return turn

    def __getattr__(self, name: str):
        if name == "model":  # not set yet: no recursion through self.model
            raise AttributeError(name)
        return getattr(self.model, name)


def generated_tokens(turn: AssistantTurn, close_tokens: int) -> int:
    """The tokens a local turn's step generated (StepLengths)."""
    thinking = turn.thinking
    if thinking is None:
        return int(turn.usage.get("completion_tokens") or 0)
    forced = close_tokens if thinking.get("forced") else 0
    return int(thinking.get("reasoning_tokens") or 0) + forced + int(thinking.get("answer_tokens") or 0)


class RouteNotes:
    """The routed model, with what each turn it gave was kept until run_tasks makes the task's record
    (module docstring: routing): who answered it, why a routed step was routed, whether the cloud's answer
    was paid and its cost at the list price. note, run_tasks' on_task hook, places the turns in the
    transcript's steps (the loop records one attempt per turn that came back, in order; the last step takes
    any turn left over, as research_agent.eval.timing.TaskTimer places calls) and gives the record its
    "routing". Everything else is the wrapped model's (name, cost, begin_task...)."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.turns: list[dict] = []  # since the last record

    @property
    def name(self) -> str:
        return self.model.name

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        turn = self.model.next_turn(messages, tools)
        route = turn.route or {}
        cost = getattr(self.model, "cost", None)
        self.turns.append({"answered_by": route.get("answered_by"), "reason": route.get("reason"),
                           "paid": turn.paid, "cost_rmb": 0.0 if cost is None else cost(turn)})
        return turn

    def __getattr__(self, name: str):
        if name == "model":  # not set yet: no recursion through self.model
            raise AttributeError(name)
        return getattr(self.model, name)

    def note(self, record: dict, transcript: object | None) -> None:
        """run_tasks' on_task hook: the record's "routing" (module docstring: routing)."""
        turns, self.turns = self.turns, []
        steps = list(getattr(transcript, "steps", None) or [])
        placed: list[tuple[int | None, dict]] = []
        position = 0
        for number, step in enumerate(steps):
            count = len(step.get("attempts") or [])
            if number == len(steps) - 1:  # what the transcript shows no attempt of: the last step's
                count = max(len(turns) - position, 0)
            mine = turns[position:position + count]
            position += len(mine)
            placed += [(step["index"], turn) for turn in mine]
        placed += [(None, turn) for turn in turns[position:]]  # no transcript, no step
        routed = [{"step": step, "reason": turn["reason"], "paid": turn["paid"], "cost_rmb": turn["cost_rmb"]}
                  for step, turn in placed if turn["answered_by"] == CLOUD]
        record["routing"] = {"routed_steps": sorted({r["step"] for r in routed if r["step"] is not None}),
                             "local_turns": sum(turn["answered_by"] == LOCAL for _, turn in placed),
                             "cloud_turns": len(routed), "routed": routed,
                             "cloud_cost_rmb": sum(r["cost_rmb"] for r in routed)}


def failed_candidates(transcript: object | None) -> Counter:
    """The failed candidates a transcript's votes list (every attempt's votes "errors"), by error type
    (module docstring: The inference wrappers); none without a transcript. A step whose chosen answer is
    itself a failed candidate ends as the loop's harness_error and keeps no vote (agent/loop.py records the
    error alone), so its candidates are not counted here; the task's harness_error is."""
    count: Counter = Counter()
    for step in getattr(transcript, "steps", None) or []:
        for attempt in step.get("attempts") or []:
            count.update(error["error_type"] for error in (attempt.get("votes") or {}).get("errors") or [])
    return count


def close_tokens(family: LoadedFamily | None) -> int:
    """The tokens a forced close adds to a step: the ids research_agent.llm.generation appends,
    family.encode_close() (think_close after a word: one id fewer than the close encoded on its own under a
    tokenizer with a dummy prefix, the two Nanbeige families'); 0 without a family, or for one that never
    thinks, whose generations are never closed whatever close it names."""
    if family is None or family.spec.thinking is Thinking.NONE:
        return 0
    return len(family.encode_close())


def main(argv: list[str] | None = None, *, model: object | None = None, world: World | None = None,
         openai_client: object | None = None, local_client: object | None = None, render=None,
         clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep) -> int:
    """model and world stand in for the built system and the recorded world in tests; openai_client,
    local_client and render for the real clients, clock for the timing sidecar's, sleep for the delay
    fault's."""
    parser = build_parser()
    args = parser.parse_args(argv)
    local_side = args.system in LOCAL_SCALES or args.system in ROUTED
    if ((args.mode is not None or args.reasoning_budget is not None or args.greedy
         or args.training_length is not None) and not local_side):
        parser.error(f"--mode, --reasoning-budget, --greedy and --training-length concern the local systems "
                     f"only ({', '.join((*LOCAL_SCALES, *ROUTED))}), not {args.system}")
    if args.greedy and args.mode == str(Mode.NO_THINK):
        parser.error("--greedy is a thinking run decoded greedily: it cannot be combined with "
                     "--mode no_think")
    if args.training_length is not None and args.training_length < 1:
        parser.error(f"--training-length must be a positive number of tokens, got {args.training_length}")
    if args.routing_config is not None and args.system not in ROUTED:
        parser.error(f"--routing-config routes a routed system ({', '.join(ROUTED)}) step by step, not "
                     f"{args.system}")
    wrapper_error = wrapper_flags_error(args, local_side=local_side)
    if wrapper_error is not None:
        parser.error(wrapper_error)
    training_length = CONTEXT_TOKENS if args.training_length is None else args.training_length
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        print(f"FAIL: {args.out} is not empty: a run folder holds one run (another --out)", file=sys.stderr)
        return 6
    faults = None if args.fault_rate == 0 else FaultSetting(rate=args.fault_rate,
                                                            persistent=args.fault_persistent, sleep=sleep)
    roots = (Path.cwd(), REPO)
    try:
        refuse_test_sets(args.tasks, roots)
        locked = lock_required(args.tasks, roots)
        if locked:
            check_lock(args.tasks)
        tasks = load_tasks(args.tasks)
        refuse_test_material(tasks, args.tasks)
    except RefusedTestSet as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 4
    except LockError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 5
    except (OSError, ValueError) as exc:
        print(f"FAIL: malformed tasks: {exc}", file=sys.stderr)
        return 1
    selected = tasks if args.limit is None else tasks[:args.limit]
    today = args.today or corpus_recorded_on(args.corpus_manifest)
    family = generation = None
    if model is None and local_side and render is None:  # the family path (module docstring)
        try:
            family = load_family(args.tokenizer)
            generation = GenerationSettings.for_family(family, mode=Mode.THINK if args.greedy else args.mode,
                                                       reasoning_budget=args.reasoning_budget,
                                                       base_seed=args.base_seed)
        except ValueError as exc:  # a FamilyError is one: the folder is no registered family
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
        if args.greedy:  # the mode's own decoding samples; this run decodes it greedily (module docstring)
            generation = dataclasses.replace(generation, decoding=GREEDY)
    if model is None:  # an injected model is run as it is
        try:
            generation = with_votes(generation, args.votes)
        except ValueError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
    count_tokens = None
    if not args.no_tokenizer:  # loads the folder's tokenizer (transformers): only here
        count_tokens = (family_token_counter(family, generation.mode) if family is not None
                        else qwen_token_counter(args.tokenizer))
    built = None
    # a routing configuration given: checked against the development sets it names (module docstring: routing)
    routed = {} if args.routing_config is None else {"routing_config": args.routing_config, "dev_set": None}
    if model is None:
        try:
            built = build_system(args.system, server=args.server, tokenizer=args.tokenizer,
                                 count_tokens=count_tokens, llm_dir=args.llm_dir, stage=args.stage,
                                 max_cost=args.max_cost, training_set=args.training_set,
                                 openai_client=openai_client, local_client=local_client, render=render,
                                 cloud_thinking=args.cloud_thinking, generation=generation, family=family,
                                 rules=args.rules, glossary=args.glossary, vote_mode=args.vote_mode or FINAL,
                                 **routed)
        except SystemUnavailable as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
        model, count_tokens, info = built.model, built.count_tokens, built.info
    else:
        info = {"system": args.system, "model": model.name, "provider": "injected"}
    notes = None
    if args.routing_config is not None:  # every step on its own, each routed step noted
        if hasattr(model, "stepwise"):
            model.stepwise = True  # agent.route.RoutedModel
        model = notes = RouteNotes(model)
    lengths = None
    if local_side and count_tokens is not None:  # the steps over the training length (module docstring)
        lengths = StepLengths(model, count_tokens=count_tokens, training_length=training_length,
                              close_tokens=close_tokens(family))
    failed = None if args.votes is None else Counter()  # the votes' failed candidates (module docstring)
    try:
        if world is None:
            world = build_world(cassettes=args.cassettes, corpus_manifest=args.corpus_manifest,
                                requests_manifest=args.requests_manifest)
        probe(world.owner, today)
        local = getattr(built, "local_client", None)
        props = getattr(local, "props", None)
        server_props = props() if props is not None else None
        print(f"{len(selected)} of {len(tasks)} tasks, system {args.system} ({info['model']}), today {today}",
              flush=True)
        timings = getattr(built, "timings", None)  # a voted system's, summed over its requests
        timed = TimedModel(model if lengths is None else lengths, clock=clock,
                           server_timings=timings or (lambda: getattr(local, "last_timings", None)))
        timer = TaskTimer(timed, clock=clock)

        def on_tools(record: dict, tools: list[dict]) -> None:  # after the timer's own entry for the record
            timer.tasks[-1]["tools"] = tools

        def on_task(record: dict, transcript: object | None) -> None:
            """The timer's hook, then the routing notes' and the count of the votes' failed candidates."""
            timer(record, transcript)
            if notes is not None:
                notes.note(record, transcript)
            if failed is not None:
                failed.update(failed_candidates(transcript))

        records = run_tasks(selected, model=timed, world=world, today=today, out=args.out,
                            count_tokens=count_tokens, seed=args.seed,
                            log=lambda line: print(line, flush=True),
                            on_task=timer if notes is None and failed is None else on_task, faults=faults,
                            clock=clock, on_tools=on_tools)
    finally:
        if built is not None:
            built.close()
    summary = summarize(records)
    if failed is not None:
        summary["candidate_errors"] = dict(sorted(failed.items()))
    corpus = Path(args.corpus_manifest)
    modes = getattr(model, "modes", None)  # a CloudModel counts its turns per mode; transcripts carry none
    manifest = {**info, "prompt_version": PROMPT_VERSION, "prompt_hash": PROMPT_HASH,
                "schema_hash": SCHEMA_HASH, "tasks": args.tasks.as_posix(),
                "tasks_sha256": sha256_of(args.tasks), "tasks_locked": locked, "tasks_total": len(tasks),
                "tasks_run": len(records), "limit": args.limit, "today": today, "seed": args.seed,
                "date": local_date(),
                "corpus_manifest_sha256": sha256_of(corpus) if corpus.is_file() else None,
                "requests_manifest": args.requests_manifest, "server_props": server_props,
                "context_check": None if args.no_tokenizer else args.tokenizer,
                **({"training_length": training_length,
                    "steps_over_training_length": None if lengths is None else lengths.over}
                   if local_side else {}),
                **({} if notes is None else {"e2e_routing": STEPWISE}),
                **({"modes": dict(sorted(modes.items()))} if modes else {}),
                **({} if faults is None else {"fault_injection": {
                    "rate": faults.rate, "persistent": faults.persistent, "faults": list(faults.faults),
                    "seed": args.seed}}),
                "git_commit": git_commit(REPO), "git_dirty": git_dirty(REPO)}
    report = {"manifest": manifest, "summary": summary, "tasks": records}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / REPORT_FILE).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8", newline="\n")
    write_timing({"report": REPORT_FILE, "tasks": timer.tasks}, args.out / TIMING_FILE)
    print(f"{args.out / REPORT_FILE}: {summary['n']} tasks, success {percent(summary['success_rate'])}")
    if faults is not None:
        met = sum(len(r.get("faults", [])) for r in records)
        print(f"  faults: rate {faults.rate:g}{', persistent' if faults.persistent else ''}, {met} met in "
              f"{sum(bool(r.get('faults')) for r in records)} task(s)")
    print("  by check: " + ", ".join(f"{k} {percent(v)}" for k, v in summary["by_check"].items()))
    for intent, cell in summary["by_intent"].items():
        print(f"  {intent:<20} n {cell['n']:>3}  success {percent(cell['success_rate'])}")
    if lengths is not None:
        print(f"  steps over the training length {training_length}: {lengths.over} of {lengths.steps}")
    if failed is not None:
        print("  failed vote candidates: " + (", ".join(f"{kind} {n}" for kind, n
                                                        in summary["candidate_errors"].items()) or "none"))
    if notes is not None:
        noted = [r["routing"] for r in records]
        print(f"  routed {sum(len(n['routed_steps']) for n in noted)} step(s) in "
              f"{sum(bool(n['routed_steps']) for n in noted)} of {len(records)} task(s) by "
              f"{args.routing_config.as_posix()}: {sum(n['local_turns'] for n in noted)} local and "
              f"{sum(n['cloud_turns'] for n in noted)} cloud turn(s), cloud cost "
              f"{sum(n['cloud_cost_rmb'] for n in noted):.4f} RMB")
    wall = sum(task["wall_ms"] for task in timer.tasks) / 1000  # the sidecar's: the report holds no time
    print(f"  tool errors {summary['tool_errors']}, wall {wall:.1f} s, tokens {summary['tokens']}, "
          f"cost {summary['cost_rmb']:.4f} RMB")
    return 3 if any(r["status"] == BUDGET_STOP for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
