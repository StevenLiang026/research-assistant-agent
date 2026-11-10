"""ACEBench regression of M4 (通用能力回归, docs/plan.md; spike S6 of docs/m0-验证.md) and, in think mode, of
M7 (docs/m7-计划.md WP-L, gate G4): a base model (b0) and adapters on it (ft, optionally b4) on ACEBench's
normal and special categories in Chinese and English, each system served by a llama-server of its own, scored
by ACEBench, summed up by our own aggregate, with the answers that forgot ACEBench's format counted.

    .venv/Scripts/python.exe scripts/run_acebench.py [--systems b0,ft] [--languages zh,en] [--smoke] \\
        [--model models/Qwen3-1.7B-Q8_0.gguf] [--tokenizer models/Qwen3-1.7B] [--family NAME] \\
        [--mode think|no_think] [--reasoning-budget N] [--seed N] [--greedy] \\
        [--ft-adapter GGUF] [--b4-adapter GGUF] [--drop-env PATTERN ...] [--out results/acebench-m7.json] \\
        [--raw runs/acebench/m7] [--force] [--clone runs/third_party/ACEBench] \\
        [--acebench-python .venv-acebench/Scripts/python.exe] [--llama-server PATH] [--port 8080] \\
        [--timeout 120] [--dry-run] [-- extra llama-server arguments]

Family and mode (M7). The base model GGUF (--model) is served as the family of its model folder (--tokenizer:
research_agent.family.core.folder_family, which reads config.json's architectures and the chat template's
SHA-256 from the folder's files and names the registered family that pins them; --family, when given, must be
that family). --tokenizer is models/Qwen3-1.7B for the default base model and must be given with any other, so
that no GGUF is served with another model's settings. --mode think or no_think: think by default for a family
that thinks (its thinking is "only" or "hybrid": qwen3-hybrid, the default folder's family, included),
no_think for one that never does. A family that never thinks has no think mode, and a family that always
thinks has no no_think mode here: its no_think prompt closes the block its template opens
(research_agent.family.core, Prompts), which ACEBench's chat requests cannot ask the server for. Either is a
usage error. --mode no_think on qwen3-hybrid repeats the M4 and M6 runs byte for byte: their server command,
ACEBench's model names and both steps' commands (results/acebench-m4.json, acebench-m4-b4.json and
acebench-m6.json record them).

Systems and ACEBench's model names. b0 is the base model on a server without an adapter; ft and b4 are the
base model on a server that loads an adapter GGUF. ACEBench knows a model by its name, which the local patch
registers (registered_models): local-<family>-base for b0 and local-<family>-lora for ft and b4, one pair per
registered family; and the M4 pair, local-qwen3-1.7b-base and local-qwen3-1.7b-lora, which the default base
model (Qwen3-1.7B) keeps without thinking, so the M4 and M6 runs repeat under their names. ACEBench keeps its
result, score and Excel folders by name, so a hybrid family's think and no_think runs share them and the
second is refused until the first one's folders are moved away (below). ACEBench cannot send llama.cpp's
per-request "lora" field, and llama.cpp b11349 applies a loaded adapter at scale 1.0 to every request without
one (spike S1b), so each system has a server of its own, one after the other, in the order b0, ft, b4. The
adapter is the "chosen" GGUF of results/checkpoint-selection-<system>.json, whose SHA-256 must still be the
one recorded, unless --ft-adapter or --b4-adapter names a GGUF (scripts/run_bfcl.py, whose helpers this script
imports; M6 named FT2's on the command line, M7 names FT3's). --systems defaults to b0,ft. ft and b4 share
ACEBench's model name, so they would share its result and score folders, which this script never deletes: they
cannot run in one invocation; b4 runs in one of its own (--systems b4, another --out) once the earlier folders
are moved away.

Servers. research_agent.llm.server.LocalServer with the command of run_bfcl (-ngl 99, --jinja, 127.0.0.1,
--cache-ram 0), then -np 1 and --no-cache-prompt, as for BFCL: on several slots or with a prompt prefix reused
from the request before, an answer depends on the requests before it (docs/m3-进度.md,
测试运行器干跑与单槽修订). The longest prompt of the categories run is under 5,600 tokens (PROMPT_TOKENS:
2,623 tokens of functions, question and profile, measured with the Qwen3 tokenizer on data_all, plus at most
2,954 tokens of ACEBench's prompt templates, which is the size of a whole template file), and the reply budget
is 1,200 tokens.
- no_think: -c 8192, the prompt and the reply; then the family's no_think template switch,
  --chat-template-kwargs {"enable_thinking":false} for qwen3-hybrid (ACEBench sends no enable_thinking, and
  Qwen3's template thinks by default, S6), none for a template without a thinking block.
- think: -c 12288 (THINK_CONTEXT), which holds the longest prompt, the reasoning budget R and the reply; then
  --reasoning-format deepseek, so the server parses the reasoning out of the answer into reasoning_content and
  ACEBench, which reads message.content, gets the answer alone; --reasoning-budget R, so the server ends a
  reasoning of R tokens with its template's end tag and the answer follows, as the two-phase generation closes
  one at R (research_agent.llm.generation; the server's forced close is its own end tag, not the family's
  think_close); the family's think template switch (--chat-template-kwargs {"enable_thinking":true} for
  qwen3-hybrid, Granite, MiniCPM5 and Spark, none for a family that thinks unasked); and the decoding's server
  settings (below). R is the family's reasoning_budget unless --reasoning-budget N gives it, from 1 to
  MAX_BUDGET, 12,288 - 5,600 - 1,200 = 5,488 tokens; the plan printed says which. ACEBench's --max-tokens is
  R + 1,200: the reasoning and the reply; after a forced close the reply keeps 1,200 tokens less the end
  tag's.
Arguments after "--" follow, and are recorded. The server's output goes to <raw>/<system>/server.log; the
server serves both languages and is stopped once ACEBench has generated them, before the evaluation.

Decoding and determinism. no_think decodes greedily, as M4 did: generate.py --temperature 0 --top-p 1, and
nothing else applies. think decodes with the family's THINK settings (registry: Qwen3's temperature 0.6, top_p
0.95, top_k 20, min_p 0; the vendors' for the others), or greedily with --greedy (run_eval's --greedy).
ACEBench's client (model_inference/apimodel_inference.py) sends messages, model, temperature, top_p and
max_tokens with every request and nothing else, so the settings it cannot send, top_k, min_p,
presence_penalty, a repetition penalty with its window and the seed, are set on the server's command line as
the defaults every request takes (--top-k, --min-p, --presence-penalty, --repeat-penalty with --repeat-last-n
set to the context, which covers the whole sequence as the family's per-request window does, and --seed), with
--temp and --top-p as well, which ACEBench's own fields override with the same values. No request carries a
seed of its own, as the two-phase generation's do (a seed per prompt, phase and sample): llama-server seeds
each request's sampler afresh from the one --seed (N, default 0), so on one slot, without a prompt cache and
with ACEBench's requests sent one at a time (--num-threads 1), an answer depends on its own request alone, and
a rerun with the same seed repeats the run. A greedy run sends no sampling flag and no seed: temperature 0
takes the most probable token. --seed is for a sampled thinking run, --reasoning-budget and --greedy for a
thinking run.

Before ACEBench starts, every server is checked: /props must report one slot of the mode's context and a build
(the same build for every system of the run); the b0 server must list no adapter, an adapter server exactly
its system's GGUF, and a completion without a lora field must match the one at scale 1 and differ from scale
0's (run_bfcl's probe); the same prompt sent twice without a cache_prompt field must take no prompt token from
the cache (run_bfcl's cache probe). Then, without thinking, a chat completion sent as ACEBench sends one
(system and user messages, no lora, cache_prompt or chat_template_kwargs field, temperature 0) must carry no
thinking block: no "<think>" or "</think>" in its content and no reasoning_content. In think mode instead, a
sampled run's /props must report the decoding's server settings as its defaults
(default_generation_settings.params, compared as the 32-bit floats it keeps), and the thinking probe must
pass: one chat completion sent as ACEBench sends one (PROBE_MESSAGES: an ACEBench-style system message with
one API of our own and a request for it; the model's name, the run's temperature, top_p and max_tokens,
nothing else) must come back with reasoning_content filled (the server separates the reasoning), a content
free of the thinking tags ("<think>", "</think>" and the family's own), and a content ACEBench's parser reads
as a call list (acebench_format: its first outermost [...] is a Python list of calls). A failed check stops
the run with the reason. A family whose thinking tags are control tokens, which llama-server writes no text
for, cannot have its reasoning separated by the server: the probe stops it.

ACEBench. The clone (--clone) must be upstream commit 56dd66c with exactly the local patches of
third_party_patches/acebench.diff (scripts/spikes/patch_acebench.py on a clean clone writes them): its HEAD is
that commit, its git diff against it (DIFF_ARGS) is the recorded patch byte for byte, it has no untracked file
outside its .gitignore, and every file under data_all/ is byte-identical to the commit's (each file's git blob
name, computed from its bytes, equals the one of git ls-tree; no file missing, none added). The groups come
from the clone's category.py: normal and special, or with --smoke the patch's one-file group
smoke_single_function only. ACEBench runs from the clone's root in its own environment (--acebench-python, the
.venv-acebench interpreter): generate.py --model <name> --category <groups> --language <zh|en> --temperature
<T> --top-p <P> --max-tokens <M> --num-threads 1 (no_think: 0, 1 and 1,200, greedy, where ACEBench's default
temperature is 0.7; think: the decoding's and R + 1,200), then eval_main.py --language <lang> --model <name>
--category <groups>. generate.py writes result_all/result_<lang>/<name>/data_<category>_result.json, the
answers ACEBench read from message.content (the reasoning, in reasoning_content, is not kept), eval_main.py
score_all/score_<lang>/<name>/data_<category>_score.json (a header {accuracy, correct_count, total_count},
then one line per wrong answer), ../result_excel/<lang>/<name>/data_<category>.xlsx beside the clone, and
score_all/score_<lang>/result.xlsx, its summary table of every model scored in that language. generate.py
keeps the answers it finds and generates only the missing ones, so before anything starts the result, score
and Excel folders of every model and language to run must not exist: they are refused by name and never
deleted. The variables set apply to ACEBench's processes only, never to this one or to any global setting:
LOCAL_OPENAI_BASE_URL (the server's /v1) and LOCAL_OPENAI_API_KEY (EMPTY), which the patch reads, PYTHONUTF8=1
(eval_main.py prints emoji), TQDM_DISABLE=1, and without the inherited variables whose names match a
--drop-env pattern (a regular expression, searched in any case): those that would keep ACEBench's HTTP
clients from reaching the local server (S5). The patterns come from the command line only and are neither
printed nor recorded, only how many there were.
Before generating, a check from the clone's root in that environment does what ACEBench's client does:
load_dotenv() (python-dotenv may add the variables of a .env file above the clone, a dropped one included),
then the OpenAI SDK's models.list() against the server.

Checks of ACEBench's output. ACEBench retries a failed request six times and then stops generation with a
non-zero exit (exit 4); the record of a failed request it can leave is an answer whose result is not text
(null: the server answered without content). After both languages are generated, every category must have its
result file with one answer per question of data_all, ids as the data's (exit 1), and no such inference error
(counted per category, exit 4), before anything is evaluated. An answer that is text without a character to
read (empty or whitespace) is no inference error: ACEBench got content, and scores it as wrong. Such answers
are counted per category (the format counts' empty, below) and, when there are any, printed per system once
both languages are generated: in think mode a reasoning that does not close within max_tokens leaves one,
which the server's reasoning budget should prevent (no check before the run sees whether the budget takes
effect: llama-server passes it to the sampler only for a template whose thinking end tags it knows). Every
category needs its score file, whose header must agree with itself and with the answers: total_count = the
number of questions, 0 <= correct_count <= total_count; accuracy = correct / n for special categories,
round(correct / n, 3) for single-turn normal ones, for normal_multi_turn_* ACEBench's turn-level accuracy (a
turn right when all its answers are) with its process_accuracy, each in [0, 1]; one wrong-answer line per
wrong answer for special and multi-turn categories, at least that many for the others (eval_main.py also
writes a line for a question answered right on a later candidate answer after failing an earlier one).
ACEBench's own summary is read from result.xlsx (the model's row, through openpyxl in ACEBench's environment)
when eval_main.py wrote one; a workbook without exactly one row of the model is an inconsistency (exit 1).

Format forgetting. ACEBench's prompts ask for [ApiName(key='value')]; the fine-tuned model may answer in its
trained <tool_call> JSON instead. Per system, language and category: n, the answers holding "<tool_call>", the
answers in ACEBench's format, as ACEBench's parser takes them: the first outermost [...] of the answer is a
Python list of calls, and the empty answers (above), which are in neither. The counts are reported as they
are; special categories ask for text, so their answers are mostly in neither form.

The aggregate (AGGREGATE_RULE), our own: per system and language, the unweighted mean of the per-category
accuracies correct / n; the same per group (normal, special); over both languages, the same mean over every
(language, category) pair. ACEBench's own Summary weights normal, special and agent and counts the agent
categories, not run, as 0: it is recorded, not used.

Writes <out>: JSON, two-space indent, LF, repository-relative POSIX paths and no time. Its default is
results/acebench-m4.json without thinking and results/acebench-m7.json in think mode; with --smoke
results/acebench-m4-smoke.json or results/acebench-m7-smoke.json, so that a smoke run cannot pass for a real
one. It refuses to overwrite an existing file without --force. It holds the languages, groups and categories,
the generation settings (generate.py's), the mode, the family (name, thinking, model folder), the decoding
(its settings, the seed, which fields ACEBench sends and which the server's command line sets, and the rule of
the module docstring), the budget (reasoning R, reply, ACEBench's max_tokens, the context; reasoning null
without thinking), the rules, the base model, the clone (commit, patch file and SHA-256, data check,
interpreter, openai's version in it), the variables set for ACEBench (names and values) and the drop-env
counts, the git commit and whether the tree was dirty, and per system: ACEBench's model name, the server
command (the binary named llama-server), the build, the adapter {gguf, sha256, selection} (null for b0), the
raw folder, in think mode the thinking probe's {answer, reasoning_chars, finish_reason}, the aggregate over
both languages, and per language the generate and evaluate commands, per category {n, correct, accuracy} (and
ACEBench's acebench_accuracy and acebench_process_accuracy for multi-turn ones), the aggregate, the groups'
aggregates, the format counts, the inference errors per category and ACEBench's summary row (null when it
wrote none). The M4 and M6 results keep reading as they did: every key they hold is written as it was, and the
raw folders keep their layout (research_agent.eval.acebench reads them). The raw outputs are copied to
<raw>/<system>/ (runs/acebench/m4 without thinking, runs/acebench/m7 in think mode, and their -smoke folders):
server.log, and per language generate.log, evaluate.log, result/, score/, excel/ and acebench-summary.xlsx;
the clone keeps its own copies, which the next run of the same model refuses.

--dry-run makes the same checks (systems, languages, binary, base model, interpreter, the model folder and its
family, the mode and its settings, adapters and their SHA-256, the clone, ACEBench's folders, clean raw
folders, the results file, paths inside the current folder) and prints the plan with the environment ACEBench
would get, without the patterns; it starts nothing and writes nothing. Run from the repository root: a path
outside the current folder is refused.

Exit 0 when the results are written (or, with --dry-run, listed); 1 when ACEBench's output is incomplete or
inconsistent (a category without its result or score file, a file that is not JSON lines, answers that are not
one per question, a score header at odds with itself or the answers, a summary workbook without exactly one
row of the model, a summary check that fails); 2 on a usage error (an unknown system or language, ft and b4
together, a --drop-env pattern that is not a regular expression, no llama-server binary, base model or
ACEBench interpreter, a base model other than the default without --tokenizer, a model folder that is not here
or names no registered family or another than --family, a mode the family cannot be served in, --greedy,
--seed or --reasoning-budget where they do not apply or out of range, an interpreter that cannot be started or
has no openai, a llama-server binary that cannot be run at all, an adapter that is missing or changed since
its selection, a clone that is not upstream plus the recorded patches, a group missing from its category.py,
existing ACEBench folders of a model and language to run, a raw folder holding an earlier run, an existing
results file without --force, a path outside the current folder, a busy port, ACEBench's environment unable to
reach the server); 3 when a server could not be started for another reason, did not become ready, did not
report one slot of the mode's context or a build, reported another build than the run's first server, listed
other adapters than its system's, did not apply its adapter to a request without a lora field, took prompt
tokens from its cache, answered with a thinking block without thinking, or, in think mode, reported other
sampling defaults than the decoding's or failed the thinking probe; 4 when an ACEBench step exited non-zero or
could not be started, or its answers hold inference errors. Every failure is a FAIL line.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx

from research_agent.eval.lock import sha256_of
from research_agent.family.core import GREEDY, Decoding, Family, FamilyError, Mode, Thinking, folder_family
from research_agent.family.registry import FAMILIES
from research_agent.llm.local import LocalClient
from research_agent.llm.server import (
    HOST,
    MODEL,
    SERVER_VAR,
    LocalServer,
    NotReady,
    PortBusy,
    llama_server,
    same_path,
    server_command,
)
from research_agent.train.manifest import REPO, git_commit, git_dirty

SCRIPTS = Path(__file__).resolve().parent


def _load_run_bfcl():
    """scripts/run_bfcl.py as a module of its own name (scripts/ is no package), as its tests load it."""
    spec = importlib.util.spec_from_file_location("_run_bfcl_shared", SCRIPTS / "run_bfcl.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bfcl = _load_run_bfcl()
UsageError, OutputError, Adapter, rel = bfcl.UsageError, bfcl.OutputError, bfcl.Adapter, bfcl.rel

CLONE = Path("runs/third_party/ACEBench")
UPSTREAM = "56dd66c"  # ACEBench's commit, "update readme"
PATCH = Path("third_party_patches/acebench.diff")
# the clone check's git diff against UPSTREAM, whose output must be the recorded patch byte for byte
DIFF_ARGS = ("diff", "--no-ext-diff", "--no-color", "--no-textconv", "--no-renames", "--diff-algorithm=myers",
             "--src-prefix=a/", "--dst-prefix=b/")
ACE_PYTHON = (Path(".venv-acebench/Scripts/python.exe") if os.name == "nt"
              else Path(".venv-acebench/bin/python"))
OUT, SMOKE_OUT = Path("results/acebench-m4.json"), Path("results/acebench-m4-smoke.json")
RAW, SMOKE_RAW = Path("runs/acebench/m4"), Path("runs/acebench/m4-smoke")
THINK_OUT, THINK_SMOKE_OUT = Path("results/acebench-m7.json"), Path("results/acebench-m7-smoke.json")
THINK_RAW, THINK_SMOKE_RAW = Path("runs/acebench/m7"), Path("runs/acebench/m7-smoke")
SYSTEMS, ADAPTER_SYSTEMS = bfcl.SYSTEMS, bfcl.ADAPTER_SYSTEMS  # in the order they run
DEFAULT_SYSTEMS = ("b0", "ft")
TOKENIZER = "models/Qwen3-1.7B"  # the default base model's folder
LEGACY_FAMILY = "qwen3-hybrid"  # Qwen3-1.7B's, which keeps the M4 names without thinking
# the M4 names, one per kind of server (without an adapter, with one), which the local patch has registered
# since M4: Qwen3-1.7B without thinking (module docstring)
MODELS = {"b0": "local-qwen3-1.7b-base", "ft": "local-qwen3-1.7b-lora", "b4": "local-qwen3-1.7b-lora"}
GROUPS = ("normal", "special")
SMOKE_GROUP = "smoke_single_function"
LANGUAGES = ("zh", "en")
PROMPT_TOKENS = 5600  # the longest prompt is under it (module docstring)
CONTEXT = 8192  # without thinking: the longest prompt plus 1,200 reply tokens
THINK_CONTEXT = 12288  # thinking: the longest prompt, the reasoning budget and the reply
SLOTS = 1
GENERATION = {"temperature": 0, "top_p": 1, "max_tokens": 1200, "num_threads": 1}  # M4's, greedy
REPLY_TOKENS = GENERATION["max_tokens"]
MAX_BUDGET = THINK_CONTEXT - PROMPT_TOKENS - REPLY_TOKENS  # 5,488 tokens of reasoning at most
REASONING_FORMAT = "deepseek"  # the reasoning in reasoning_content, the answer alone in content
SEED_MAX = 0x7FFFFFFF  # never llama.cpp's random seed, 0xFFFFFFFF
# the server's flag for each setting of the decoding it sets (module docstring: Decoding and determinism),
# under the name /props reports it by
SERVER_FLAGS = {"temperature": "--temp", "top_p": "--top-p", "top_k": "--top-k", "min_p": "--min-p",
                "presence_penalty": "--presence-penalty", "repeat_penalty": "--repeat-penalty",
                "repeat_last_n": "--repeat-last-n", "seed": "--seed"}
ACEBENCH_SENDS = ["temperature", "top_p", "max_tokens"]  # every request's fields besides messages and model
PROPS_TOLERANCE = 1e-6  # /props reports the settings as 32-bit floats (0.95 as 0.949999988...)
SAMPLED_RULE = ("ACEBench's client sends temperature, top_p and max_tokens with every request (generate.py's "
                "--temperature, --top-p and --max-tokens) and no other field, so the decoding's other "
                "settings and the seed cannot be sent: the server's command line sets them as the defaults "
                "every request takes (server_sets: --top-k, --min-p, --presence-penalty, --repeat-penalty "
                "with --repeat-last-n the context, --seed), with --temp and --top-p too, and /props must "
                "report them. No request has a seed of its own: llama-server seeds each request's sampler "
                "afresh from the one --seed, so on one slot, without a prompt cache and with ACEBench's "
                "requests one at a time, an answer depends on its own request alone and a rerun with the "
                "same seed repeats the run.")
GREEDY_RULE = ("Greedy: ACEBench sends temperature 0 and top_p 1 with every request and the sampler takes "
               "the most probable token, so no other setting and no seed applies; one slot, no prompt cache, "
               "ACEBench's requests one at a time.")
ACE_ENV = {"LOCAL_OPENAI_API_KEY": "EMPTY", "PYTHONUTF8": "1", "TQDM_DISABLE": "1"}
VERSION_CODE = "import importlib.metadata as m; print(m.version('openai'))"
# what ACEBench's client does (apimodel_inference.APIModelInference.__init__): load_dotenv(), then the OpenAI
# SDK against the server's /v1 given as the argument; no retries, so an unreachable server fails within 30 s
REACH_CODE = ("import sys; from dotenv import load_dotenv; load_dotenv(); from openai import OpenAI; "
              "OpenAI(base_url=sys.argv[1], api_key='EMPTY', timeout=30, max_retries=0).models.list()")
REACH_TIMEOUT = 120.0
# the rows of ACEBench's summary workbook (argument 1) whose first cell is the model (argument 2), as JSON
SUMMARY_CODE = ("import json, os, sys; path, model = sys.argv[1:3]; exists = os.path.isfile(path); "
                "rows = list(__import__('openpyxl').load_workbook(path, read_only=True).active"
                ".iter_rows(values_only=True)) if exists else []; "
                "print(json.dumps({'file': exists, 'rows': [dict(zip(rows[0], row)) for row in rows[1:] "
                "if row and row[0] == model]}, default=str))")
THINK_MESSAGES = [{"role": "system", "content": "You are an AI assistant."},
                  {"role": "user", "content": "What is 2 + 3? Answer with the number only."}]
THINK_TOKENS = 32
THINK_TAGS = ("<think>", "</think>")
# the thinking probe (module docstring): an ACEBench-style request of our own, answered in ACEBench's format
PROBE_FUNCTION = {"name": "get_weather", "description": "Get the current weather of a city.",
                  "parameters": {"type": "object",
                                 "properties": {"city": {"type": "string",
                                                         "description": "The city's name."}},
                                 "required": ["city"]}}
PROBE_MESSAGES = [
    {"role": "system",
     "content": ("You are an AI assistant with the role name \"assistant\". Based on the API specifications "
                 "below, generate the API request the assistant should make for the user's request. Output "
                 "it in the format [ApiName(key1='value1', key2='value2', ...)], replacing ApiName with the "
                 "API's name, key1, key2 with parameter names and value1, value2 with their values, starting "
                 "with \"[\" and ending with \"]\", and nothing else.\n\nAPI Specifications:\n"
                 + json.dumps([PROBE_FUNCTION], ensure_ascii=False))},
    {"role": "user",
     "content": "Conversation history 1..t:\nuser: What is the weather like in Paris right now?"},
]
TOOL_CALL = "<tool_call>"
AGGREGATE_RULE = ("Per system and language, the unweighted mean of the per-category accuracies (correct / n) "
                  "over the categories run, each category counting once whatever its n, which is recorded "
                  "beside it; per group the same over the group's categories; over both languages the "
                  "same over every (language, category) pair. correct and n are the score file's "
                  "correct_count and total_count: for normal_multi_turn_* they count single answers, while "
                  "ACEBench's own "
                  "accuracy there (acebench_accuracy) counts a turn right only when all its answers are. "
                  "ACEBench's own Summary is not used: it weights normal, special and agent and counts the "
                  "agent categories, not run, as 0.")
FORMAT_RULE = ("Per category: n answers; tool_call, the answers holding <tool_call> (the fine-tuned "
               "model's trained format); acebench_format, the answers whose first outermost [...] parses as "
               "a Python list of calls, as ACEBench's parser takes them ([ApiName(key='value')]); empty, the "
               "answers that are text without a character to read, empty or whitespace (in think mode also "
               "the answer of a reasoning that does not close within max_tokens), which ACEBench scores as "
               "wrong. An answer may be counted in both tool_call and acebench_format, or in neither; "
               "special categories ask for text.")


def acebench_model(family: str, system: str) -> str:
    """ACEBench's name for a system served as a family (module docstring): local-<family>-base for b0,
    local-<family>-lora for an adapter system."""
    return f"local-{family}-{'base' if system == 'b0' else 'lora'}"


def registered_models() -> list[str]:
    """Every name the local patch must register, in its order: the M4 pair, then a base and a lora name per
    registered family (research_agent.family.registry.FAMILIES)."""
    return [*dict.fromkeys(MODELS.values()),
            *(acebench_model(family.name, system) for family in FAMILIES for system in ("b0", "ft"))]


@dataclass(frozen=True)
class Clone:
    """The checked ACEBench clone: its commit, how many data files matched upstream, the groups run with their
    categories, and the question ids of every category per language."""

    root: Path
    commit: str
    data_files: int
    groups: dict[str, list[str]]
    questions: dict[str, dict[str, list[str]]]

    @property
    def categories(self) -> list[str]:
        return [category for categories in self.groups.values() for category in categories]


@dataclass(frozen=True)
class Setup:
    """How a run serves its base model (module docstring: Family and mode; Servers; Decoding and determinism):
    the family and its model folder, the mode, the decoding (GREEDY when it decodes greedily), the server's
    seed (None when greedy), the reasoning budget R (None without thinking), whether the M4 names apply, and
    whether --reasoning-budget gave R (the family's otherwise)."""

    family: Family
    folder: Path
    mode: Mode
    decoding: Decoding
    seed: int | None
    reasoning_budget: int | None
    legacy: bool
    budget_given: bool = False

    @property
    def think(self) -> bool:
        return self.mode is Mode.THINK

    @property
    def greedy(self) -> bool:
        return self.decoding.temperature == 0

    @property
    def context(self) -> int:
        return THINK_CONTEXT if self.think else CONTEXT

    @property
    def max_tokens(self) -> int:
        """ACEBench's --max-tokens: the reply budget, and in think mode the reasoning budget before it."""
        return REPLY_TOKENS + (self.reasoning_budget or 0)

    @property
    def generation(self) -> dict:
        """generate.py's settings: M4's when greedy (temperature 0, top_p 1), the decoding's otherwise."""
        if self.greedy:
            temperature, top_p = GENERATION["temperature"], GENERATION["top_p"]
        else:
            temperature, top_p = self.decoding.temperature, self.decoding.top_p
        return {"temperature": temperature, "top_p": top_p, "max_tokens": self.max_tokens,
                "num_threads": GENERATION["num_threads"]}

    @property
    def server_defaults(self) -> dict:
        """The settings the server's command line sets for a sampled run, by the names /props reports them
        under, in the order of the flags; none for a greedy run."""
        if self.greedy:
            return {}
        decoding = self.decoding
        settings = {"temperature": decoding.temperature, "top_p": decoding.top_p, "top_k": decoding.top_k,
                    "min_p": decoding.min_p, "presence_penalty": decoding.presence_penalty}
        if decoding.repetition_penalty != 1:
            settings.update(repeat_penalty=decoding.repetition_penalty, repeat_last_n=self.context)
        settings["seed"] = self.seed
        return settings

    def sampling_flags(self) -> list[str]:
        return [part for key, value in self.server_defaults.items()
                for part in (SERVER_FLAGS[key], str(value))]

    def template_switch(self) -> str | None:
        """The family's template keyword arguments for the mode as compact JSON, None when it has none."""
        kwargs = dict(self.family.template_kwargs[self.mode])
        return json.dumps(kwargs, separators=(",", ":")) if kwargs else None

    def names(self) -> dict[str, str]:
        """ACEBench's model name per system (module docstring: Systems and ACEBench's model names)."""
        if self.legacy:
            return dict(MODELS)
        return {system: acebench_model(self.family.name, system) for system in SYSTEMS}

    def family_record(self) -> dict:
        return {"name": self.family.name, "thinking": str(self.family.thinking), "folder": rel(self.folder)}

    def decoding_record(self) -> dict:
        """The decoding as the results record it (module docstring: Writes)."""
        if self.greedy:
            return {"greedy": True, "temperature": GENERATION["temperature"], "top_p": GENERATION["top_p"],
                    "seed": None, "acebench_sends": ACEBENCH_SENDS, "server_sets": [], "rule": GREEDY_RULE}
        decoding = self.decoding
        record = {"greedy": False, "temperature": decoding.temperature, "top_p": decoding.top_p,
                  "top_k": decoding.top_k, "min_p": decoding.min_p,
                  "presence_penalty": decoding.presence_penalty}
        if decoding.repetition_penalty != 1:
            record.update(repetition_penalty=decoding.repetition_penalty, repetition_window=self.context)
        record["seed"] = self.seed
        record["acebench_sends"] = ACEBENCH_SENDS
        record["server_sets"] = [key for key in self.server_defaults if key not in ("temperature", "top_p")]
        record["rule"] = SAMPLED_RULE
        return record

    def budget_record(self) -> dict:
        return {"reasoning": self.reasoning_budget, "reply": REPLY_TOKENS, "max_tokens": self.max_tokens,
                "context": self.context}

    def describe(self) -> str:
        family = f"family {self.family.name} (thinking {self.family.thinking}) of {rel(self.folder)}"
        if self.greedy:
            decoding = "greedy (temperature 0; no sampling setting or seed applies)"
        else:
            settings = ", ".join(f"{key} {value}" for key, value in self.server_defaults.items())
            decoding = (f"decoding {settings} (temperature and top_p sent by ACEBench, the rest the server's "
                        "defaults)")
        if self.think:
            source = ("--reasoning-budget" if self.budget_given
                      else f"{self.family.name}'s reasoning_budget: --reasoning-budget not given")
            budget = (f"reasoning budget {self.reasoning_budget} ({source}), reply {REPLY_TOKENS}, "
                      f"max_tokens {self.max_tokens}")
        else:
            budget = f"reply {REPLY_TOKENS} tokens"
        return f"mode {self.mode}: {family}; {decoding}; {budget}, context {self.context}"


@dataclass(frozen=True)
class Plan:
    """What a run does for one system: its server and ACEBench's two steps per language."""

    system: str
    model: str  # ACEBench's model name
    adapter: Adapter | None
    root: Path  # the system's raw folder
    serve: list[str]
    record: list[str]  # the server command with the binary named llama-server
    extra: list[str]  # what follows the server's own arguments (server_extra)
    generate: dict[str, list[str]]  # per language; the interpreter absolute, as ACEBench runs in the clone
    evaluate: dict[str, list[str]]
    settings: dict[str, str]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ACEBench regression: a base model and adapters on it, each on "
                                             "a llama-server of its own; normal and special, zh and en; with "
                                             "or without thinking.")
    ap.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS),
                    help="comma-separated, of b0, ft, b4 (default b0,ft; ft and b4 not together)")
    ap.add_argument("--languages", default=",".join(LANGUAGES),
                    help="comma-separated, of zh, en (default both)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"run the group {SMOKE_GROUP} only, into results and raw folders of its own unless "
                         "--out and --raw say otherwise")
    for system in ADAPTER_SYSTEMS:
        ap.add_argument(f"--{system}-adapter", type=Path, default=None,
                        help=f"the {system} adapter GGUF (default: the chosen one of "
                             f"{bfcl.SELECTION.format(system)})")
    ap.add_argument("--drop-env", action="append", default=None, metavar="PATTERN",
                    help="drop the inherited variables whose names match PATTERN (a regular expression, any "
                         "case) from ACEBench's processes only; repeatable; never printed or recorded")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"the results (default {OUT.as_posix()} without thinking, {THINK_OUT.as_posix()} in "
                         "think mode)")
    ap.add_argument("--raw", type=Path, default=None,
                    help=f"the folder the raw outputs are copied to (default {RAW.as_posix()} without "
                         f"thinking, {THINK_RAW.as_posix()} in think mode)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results file")
    ap.add_argument("--clone", type=Path, default=CLONE,
                    help=f"the ACEBench clone (default {CLONE.as_posix()})")
    ap.add_argument("--acebench-python", type=Path, default=ACE_PYTHON,
                    help=f"ACEBench's interpreter (default {ACE_PYTHON.as_posix()})")
    ap.add_argument("--llama-server", default=None,
                    help=f"llama-server executable (default: ${SERVER_VAR}; one of the two is required)")
    ap.add_argument("--model", default=MODEL, help=f"base model GGUF (default {MODEL})")
    ap.add_argument("--tokenizer", default=None,
                    help=f"the base model's folder, whose family (config.json and chat template) serves it "
                         f"(default {TOKENIZER} for the default base model; required with another)")
    ap.add_argument("--family", default=None,
                    help="the registered family the folder must be (research_agent.family.registry)")
    ap.add_argument("--mode", choices=[str(mode) for mode in Mode], default=None,
                    help="think or no_think (default: think for a family that thinks, no_think for one that "
                         "never does)")
    ap.add_argument("--reasoning-budget", type=int, default=None, metavar="N",
                    help=f"think mode: the reasoning's tokens before the server closes it (default: the "
                         f"family's; 1 to {MAX_BUDGET})")
    ap.add_argument("--seed", type=int, default=None, metavar="N",
                    help="a sampled thinking run: the server's seed, every request's (default 0)")
    ap.add_argument("--greedy", action="store_true",
                    help="think mode decoded greedily (temperature 0, no seed), whatever the family's "
                         "sampling")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for each server")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the environment; start nothing, write nothing")
    ap.add_argument("server_args", nargs="*", help="extra llama-server arguments, after --")
    return ap


def git_output(git, clone: Path, *args: str) -> bytes:
    """stdout of git in the clone, as bytes; UsageError when git cannot run or fails."""
    try:
        done = git(["git", "-C", os.fspath(clone), *args], capture_output=True)
    except OSError as exc:
        raise UsageError(f"git cannot be started for the clone check ({exc})") from None
    if done.returncode != 0:
        why = (done.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-1:]
        raise UsageError(f"git {args[0]} failed in {rel(clone)} ({' '.join(why) or 'no error text'})")
    return done.stdout or b""


def blob_name(data: bytes, length: int) -> str:
    """git's object name of a file's bytes (SHA-1, or SHA-256 for a repository of 64-digit names)."""
    digest = hashlib.sha256 if length == 64 else hashlib.sha1
    return digest(b"blob %d\0" % len(data) + data).hexdigest()


def upstream_data(git, clone: Path) -> dict[str, str]:
    """{path: blob name} of every file under data_all/ in the upstream commit (git ls-tree)."""
    tree = {}
    for entry in git_output(git, clone, "ls-tree", "-r", "-z", UPSTREAM, "--", "data_all").split(b"\0"):
        if entry:
            meta, path = entry.decode("utf-8").split("\t", 1)
            tree[path] = meta.split()[2]
    return tree


def ace_groups(clone: Path) -> dict[str, list[str]]:
    """ACE_DATA_CATEGORY of the clone's category.py, read as a literal (nothing of the clone is run here)."""
    path = clone / "category.py"
    try:
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == \
                    ["ACE_DATA_CATEGORY"]:
                return ast.literal_eval(node.value)
    except (OSError, SyntaxError, ValueError) as exc:
        raise UsageError(f"{rel(path)} cannot be read ({type(exc).__name__}: {exc})") from None
    raise UsageError(f"{rel(path)} defines no ACE_DATA_CATEGORY")


def check_clone(git, clone: Path, languages: list[str], groups: tuple[str, ...]) -> Clone:
    """The clone's checks of the module docstring; UsageError for the first that fails."""
    if not (clone / "generate.py").is_file():
        raise UsageError(f"no ACEBench clone at {rel(clone)} (--clone)")
    commit = git_output(git, clone, "rev-parse", "HEAD").decode("utf-8").strip()
    if not commit.startswith(UPSTREAM):
        raise UsageError(f"{rel(clone)} is at commit {commit}, not ACEBench's {UPSTREAM}")
    tree = upstream_data(git, clone)
    disk = {path.relative_to(clone).as_posix(): path
            for path in (clone / "data_all").rglob("*") if path.is_file()}
    missing, added = sorted(set(tree) - set(disk)), sorted(set(disk) - set(tree))
    changed = sorted(path for path in set(tree) & set(disk)
                     if blob_name(disk[path].read_bytes(), len(tree[path])) != tree[path])
    for what, paths in (("differ from", changed), ("are missing from", missing), ("are not in", added)):
        if paths:
            raise UsageError(f"{len(paths)} files under {rel(clone)}/data_all {what} commit {UPSTREAM} "
                             f"({', '.join(paths[:3])}{' ...' if len(paths) > 3 else ''}): ACEBench's data "
                             "must be byte-identical to upstream")
    diff = git_output(git, clone, *DIFF_ARGS, UPSTREAM, "--")
    if not PATCH.is_file() or diff != PATCH.read_bytes():
        raise UsageError(f"the code of {rel(clone)} is not commit {UPSTREAM} plus the recorded patches: its "
                         f"git diff differs from {PATCH.as_posix()} (scripts/spikes/patch_acebench.py on a "
                         "clean clone)")
    untracked = [p for p in git_output(git, clone, "ls-files", "--others", "--exclude-standard", "-z")
                 .decode("utf-8").split("\0") if p]
    if untracked:
        raise UsageError(f"{rel(clone)} holds files git does not track and does not ignore "
                         f"({', '.join(untracked[:3])}): the clone must be upstream plus the recorded "
                         "patches")
    defined = ace_groups(clone)
    absent = [group for group in groups if group not in defined]
    if absent:
        raise UsageError(f"{rel(clone)}/category.py defines no group {', '.join(absent)} (the patch adds "
                         f"{SMOKE_GROUP}: scripts/spikes/patch_acebench.py)")
    chosen = {group: list(defined[group]) for group in groups}
    questions = {}
    for language in languages:
        questions[language] = {}
        for category in dict.fromkeys(c for categories in chosen.values() for c in categories):
            path = clone / "data_all" / f"data_{language}" / f"data_{category}.json"
            if not path.is_file():
                raise UsageError(f"{rel(path)} is not here: no data for {category} in {language}")
            questions[language][category] = [entry.get("id") for entry in bfcl.json_lines(path)]
    return Clone(clone, commit, len(tree), chosen, questions)


def outputs(clone: Path, language: str, model: str) -> dict[str, Path]:
    """ACEBench's folders of a model and language: result, score, and the Excel folder beside the clone."""
    return {"result": clone / "result_all" / f"result_{language}" / model,
            "score": clone / "score_all" / f"score_{language}" / model,
            "excel": clone.parent / "result_excel" / language / model}


def ace_command(python: Path, step: str, model: str, groups: list[str], language: str,
                generation: Mapping[str, object]) -> list[str]:
    if step == "generate":
        return [os.path.abspath(python), "generate.py", "--model", model, "--category", *groups,
                "--language", language, "--temperature", str(generation["temperature"]),
                "--top-p", str(generation["top_p"]), "--max-tokens", str(generation["max_tokens"]),
                "--num-threads", str(generation["num_threads"])]
    return [os.path.abspath(python), "eval_main.py", "--language", language, "--model", model,
            "--category", *groups]


def server_extra(args: argparse.Namespace, setup: Setup) -> list[str]:
    """What follows llama.server's own arguments (module docstring: Servers): one slot and the prompt cache
    off; in think mode the reasoning format and budget; the family's template switch for the mode; the
    decoding's server settings; then the arguments given after "--"."""
    extra = ["-np", str(SLOTS), "--no-cache-prompt"]
    if setup.think:
        extra += ["--reasoning-format", REASONING_FORMAT, "--reasoning-budget", str(setup.reasoning_budget)]
    switch = setup.template_switch()
    if switch is not None:
        extra += ["--chat-template-kwargs", switch]
    return [*extra, *setup.sampling_flags(), *args.server_args]


def run_family(args: argparse.Namespace) -> tuple[Family, Path]:
    """The family the base model is served as, and its model folder (module docstring: Family and mode);
    UsageError otherwise. folder_family is looked up when called, so the tests can stand in for it."""
    if args.tokenizer is None:
        if not same_path(args.model, MODEL):
            raise UsageError(f"--model {args.model} is not the default base model {MODEL}: name its model "
                             "folder with --tokenizer, whose family (config.json and chat template) serves "
                             "it")
        folder = Path(TOKENIZER)
    else:
        folder = Path(args.tokenizer)
    if not folder.is_dir():
        raise UsageError(f"the model folder {folder.as_posix()} is not here (--tokenizer)")
    try:
        family = folder_family(folder, name=args.family)
    except (FamilyError, OSError, ValueError) as exc:
        raise UsageError(f"{rel(folder)} is not the model folder of a registered family as given: {exc}") \
            from None
    return family, folder


def choose_setup(args: argparse.Namespace, family: Family, folder: Path) -> Setup:
    """The run's mode, decoding, seed and budget (module docstring: Family and mode; Decoding and
    determinism); UsageError for what the family cannot be served with, or flags that do not apply."""
    name = family.name
    if args.mode is not None:
        mode = Mode(args.mode)
    else:
        mode = Mode.NO_THINK if family.thinking is Thinking.NONE else Mode.THINK
    if mode is Mode.THINK and family.thinking is Thinking.NONE:
        raise UsageError(f"--mode think: {name} never thinks: its template has no thinking block (run it "
                         "with --mode no_think)")
    if mode is Mode.NO_THINK and family.thinking is Thinking.ONLY:
        raise UsageError(f"--mode no_think: {name} always thinks: its template opens a thinking block in "
                         "every prompt and its no_think prompt closes it (research_agent.family.core, "
                         "Prompts), which ACEBench's chat requests cannot ask the server for (run it with "
                         "--mode think)")
    if mode is Mode.NO_THINK:
        given = [flag for flag, value in (("--greedy", args.greedy or None), ("--seed", args.seed),
                                          ("--reasoning-budget", args.reasoning_budget)) if value is not None]
        if given:
            raise UsageError(f"{' and '.join(given)} set a thinking run, and --mode no_think answers at "
                             "once, greedily, without a reasoning budget")
        if family.decoding[Mode.NO_THINK].temperature != 0:
            raise UsageError(f"--mode no_think: {name} samples without thinking, and a no_think run decodes "
                             "greedily as M4 did")
        return Setup(family=family, folder=folder, mode=mode, decoding=GREEDY, seed=None,
                     reasoning_budget=None, legacy=name == LEGACY_FAMILY and same_path(args.model, MODEL))
    budget = family.reasoning_budget if args.reasoning_budget is None else args.reasoning_budget
    if not 1 <= budget <= MAX_BUDGET:
        raise UsageError(f"--reasoning-budget {budget}: from 1 to {MAX_BUDGET} tokens, so that the longest "
                         f"prompt (under {PROMPT_TOKENS}), the reasoning and the {REPLY_TOKENS} reply tokens "
                         f"fit the context of {THINK_CONTEXT}")
    decoding = GREEDY if args.greedy else family.decoding[Mode.THINK]
    seed = None
    if decoding.temperature == 0:
        if args.seed is not None:
            raise UsageError("--seed seeds a sampled thinking run, and this one decodes greedily"
                             + (" (--greedy)" if args.greedy else ""))
    else:
        seed = 0 if args.seed is None else args.seed
        if not 0 <= seed <= SEED_MAX:
            raise UsageError(f"--seed {seed}: from 0 to {SEED_MAX} (llama-server reads -1 and 0xFFFFFFFF as "
                             "a random seed)")
    return Setup(family=family, folder=folder, mode=mode, decoding=decoding, seed=seed,
                 reasoning_budget=budget, legacy=False, budget_given=args.reasoning_budget is not None)


def prepare(args: argparse.Namespace, git) -> tuple[list[Plan], Path, Clone, Setup]:
    """The plan of every system, the llama-server binary, the checked clone and the run's setup, after the
    checks of the module docstring that need nothing started; UsageError otherwise."""
    systems = bfcl.chosen(args.systems, SYSTEMS, "--systems")
    languages = bfcl.chosen(args.languages, LANGUAGES, "--languages")
    for i, pattern in enumerate(args.drop_env or [], 1):
        try:
            re.compile(pattern)
        except re.error as exc:  # the pattern itself is not shown: it may name the machine's settings
            raise UsageError(f"--drop-env pattern {i} is not a regular expression ({exc.msg})") from None
    for system in ADAPTER_SYSTEMS:
        if getattr(args, f"{system}_adapter") is not None and system not in systems:
            raise UsageError(f"--{system}-adapter names an adapter for {system}, which --systems does not "
                             "run")
    try:
        binary = llama_server(args.llama_server)
    except LookupError as exc:
        raise UsageError(str(exc)) from None
    if not binary.is_file():
        raise UsageError(f"llama-server not found: {binary} (--llama-server or {SERVER_VAR})")
    if not Path(args.model).is_file():
        raise UsageError(f"the base model {args.model} is not here (--model)")
    if not args.acebench_python.is_file():
        raise UsageError(f"ACEBench's interpreter {rel(args.acebench_python)} is not here "
                         "(--acebench-python; the .venv-acebench environment of docs/m0-验证.md, S6)")
    family, folder = run_family(args)
    setup = choose_setup(args, family, folder)
    names = setup.names()
    if "ft" in systems and "b4" in systems:
        raise UsageError(f"--systems {args.systems!r}: ft and b4 are both {names['ft']} to ACEBench and "
                         "would share its result and score folders; run b4 on its own (--systems b4, another "
                         "--out) once ft's folders are moved away")
    if setup.think:
        args.out = args.out or (THINK_SMOKE_OUT if args.smoke else THINK_OUT)
        args.raw = args.raw or (THINK_SMOKE_RAW if args.smoke else THINK_RAW)
    else:
        args.out = args.out or (SMOKE_OUT if args.smoke else OUT)
        args.raw = args.raw or (SMOKE_RAW if args.smoke else RAW)
    if args.out.exists() and not args.force:
        raise UsageError(f"{rel(args.out)} exists: pass --force to overwrite it, or another --out")
    adapters = {system: bfcl.chosen_adapter(system, getattr(args, f"{system}_adapter"))
                for system in systems if system in ADAPTER_SYSTEMS}
    paths = [args.model, folder, args.acebench_python, args.raw, args.out, args.clone, PATCH,
             *(adapter.gguf for adapter in adapters.values())]
    outside = [rel(path) for path in paths if not bfcl.inside(path)]
    if outside:
        raise UsageError(f"run from the repository root: {outside} lie outside the current folder, and the "
                         "results hold repository-relative paths only")
    groups = (SMOKE_GROUP,) if args.smoke else GROUPS
    clone = check_clone(git, args.clone, languages, groups)
    existing = [rel(made) for system in systems for language in languages
                for made in outputs(args.clone, language, names[system]).values() if made.exists()]
    if existing:
        raise UsageError(f"ACEBench's folders of an earlier run exist: {', '.join(dict.fromkeys(existing))}; "
                         "generate.py would keep their answers and generate only the missing ones: move them "
                         "away first (this script never deletes them)")
    url = f"http://{HOST}:{args.port}/v1"
    extra = server_extra(args, setup)
    plans = []
    for system in systems:
        root = args.raw / system
        earlier = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
        if earlier:
            raise UsageError(f"{rel(root)} holds files of an earlier run ({rel(earlier[0])} and "
                             f"{len(earlier) - 1} more): move the folder away first")
        adapter = adapters.get(system)
        gguf = None if adapter is None else rel(adapter.gguf)
        model = names[system]
        plans.append(Plan(
            system=system, model=model, adapter=adapter, root=root,
            serve=server_command(binary, model=args.model, adapter=gguf, port=args.port,
                                 ctx_size=setup.context, extra=extra),
            record=server_command("llama-server", model=rel(args.model), adapter=gguf, port=args.port,
                                  ctx_size=setup.context, extra=extra),
            extra=extra,
            generate={lang: ace_command(args.acebench_python, "generate", model, list(groups), lang,
                                        setup.generation) for lang in languages},
            evaluate={lang: ace_command(args.acebench_python, "evaluate", model, list(groups), lang,
                                        setup.generation) for lang in languages},
            settings={"LOCAL_OPENAI_BASE_URL": url, **ACE_ENV}))
    return plans, binary, clone, setup


def as_recorded(cmd: list[str]) -> list[str]:
    return [rel(cmd[0]), *cmd[1:]]


def benchmark(setup: Setup) -> str:
    """The results' description of what was run (M4 and M6 recorded the non-thinking one)."""
    mode = "thinking" if setup.think else "non-thinking"
    return (f"ACEBench {UPSTREAM} with the local patches of {PATCH.as_posix()}, through llama-server in "
            f"{mode} mode")


def describe(plans: list[Plan], clone: Clone, dropped: int, setup: Setup, args: argparse.Namespace) -> None:
    settings = ", ".join(f"{name}={value}" for name, value in plans[0].settings.items())
    print(f"ACEBench regression: systems {', '.join(p.system for p in plans)}; languages "
          f"{', '.join(plans[0].generate)}; groups {', '.join(clone.groups)} "
          f"({len(clone.categories)} categories)")
    print(setup.describe())
    print(f"clone {rel(clone.root)} at {clone.commit}, plus {PATCH.as_posix()}; {clone.data_files} data "
          "files byte-identical to upstream")
    print(f"ACEBench's processes only: {settings}; {dropped} inherited variables dropped (--drop-env)")
    print(f"results {rel(args.out)}; raw outputs under {rel(args.raw)}")
    for p in plans:
        if p.adapter is None:
            adapter = "no adapter"
        else:
            source = "named on the command line" if p.adapter.selection is None else rel(p.adapter.selection)
            adapter = f"adapter {rel(p.adapter.gguf)} sha256 {p.adapter.sha256} ({source})"
        print(f"{p.system}: ACEBench model {p.model}, {adapter}, raw {rel(p.root)}")
        print(f"  server:   {subprocess.list2cmdline(p.serve)}")
        for language in p.generate:
            print(f"  generate: {subprocess.list2cmdline(as_recorded(p.generate[language]))}")
        for language in p.evaluate:
            print(f"  evaluate: {subprocess.list2cmdline(as_recorded(p.evaluate[language]))}", flush=True)


def thinking(client: LocalClient) -> str | None:
    """The thinking block of a chat completion sent as ACEBench sends one, or None when it has none."""
    response = client.http.post("/v1/chat/completions", json={"messages": THINK_MESSAGES, "temperature": 0,
                                                              "max_tokens": THINK_TOKENS})
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    content, reasoning = message.get("content") or "", message.get("reasoning_content") or ""
    if reasoning.strip():
        return f"reasoning_content {reasoning[:40]!r}"
    if "<think>" in content or "</think>" in content:
        return f"content {content[:40]!r}"
    return None


def think_tags(family: Family) -> list[str]:
    """The tags no answer may hold: "<think>", "</think>" and the family's own opening and closing tags."""
    tags = dict.fromkeys(THINK_TAGS)
    for tag in (family.think_open, family.think_close):
        if tag and tag.strip():
            tags[tag.strip()] = None
    return list(tags)


def think_probe(client: LocalClient, p: Plan, setup: Setup) -> tuple[str | None, dict]:
    """The thinking probe of the module docstring: why it failed (None when it passed), and its record."""
    generation = setup.generation
    response = client.http.post("/v1/chat/completions", json={
        "model": p.model, "messages": PROBE_MESSAGES, "temperature": generation["temperature"],
        "top_p": generation["top_p"], "max_tokens": generation["max_tokens"]})
    response.raise_for_status()
    choice = response.json()["choices"][0]
    message = choice["message"]
    content, reasoning = message.get("content") or "", message.get("reasoning_content") or ""
    finish = choice.get("finish_reason")
    record = {"answer": content, "reasoning_chars": len(reasoning), "finish_reason": finish}
    if not reasoning.strip():
        return (f"the thinking probe, a chat completion sent as ACEBench sends one, came back without "
                f"reasoning_content (content {content[:60]!r}): the server does not separate the reasoning "
                f"(--reasoning-format {REASONING_FORMAT}), or the model did not think"), record
    leaked = [tag for tag in think_tags(setup.family) if tag in content]
    if leaked:
        return (f"the thinking probe's content holds {', '.join(leaked)} ({content[:60]!r}): the reasoning "
                "would reach the answers ACEBench reads"), record
    if not acebench_format(content):
        return (f"the thinking probe's answer {content[:80]!r} (finish_reason {finish}) is not in ACEBench's "
                "format, a call list its parser reads ([ApiName(key='value')])"), record
    return None, record


def same_setting(reported: object, wanted: float) -> bool:
    return (isinstance(reported, int | float) and not isinstance(reported, bool)
            and abs(reported - wanted) <= PROPS_TOLERANCE)


def sampling_problem(props: Mapping, setup: Setup) -> str | None:
    """Why the server's sampling defaults are not the decoding's server settings (module docstring), or None:
    /props default_generation_settings.params, as 32-bit floats."""
    wanted = setup.server_defaults
    params = (props.get("default_generation_settings") or {}).get("params") or {}
    off = {key: params.get(key) for key, value in wanted.items() if not same_setting(params.get(key), value)}
    if not off:
        return None
    sets = {key: wanted[key] for key in off}
    return (f"the server's default sampling (/props default_generation_settings.params) has {off}, where the "
            f"decoding sets {sets}: ACEBench's requests, which send temperature and top_p alone, would "
            "sample otherwise")


def server_problem(p: Plan, server: LocalServer, build: str | None,
                   setup: Setup) -> tuple[str | None, dict | None]:
    """Why the started server cannot serve p (module docstring), or None, with the thinking probe's record in
    think mode; build is the run's first server's."""
    props = server.props or {}
    slots, context = props.get("total_slots"), (props.get("default_generation_settings") or {}).get("n_ctx")
    if (slots, context) != (SLOTS, setup.context):
        return (f"the server reports {slots} slots of {context} tokens (/props), not {SLOTS} of "
                f"{setup.context}"), None
    reported = props.get("build_info")
    if not isinstance(reported, str) or not reported:
        return "the server reports no build (/props build_info)", None
    if build is not None and reported != build:
        return f"the server reports build {reported}, and the run's first server was {build}", None
    listed = [a.get("path") if isinstance(a, dict) else a for a in server.adapters]
    if p.adapter is None and listed:
        return f"the server lists the adapters {listed}, and the base model needs none", None
    if p.adapter is not None:
        if len(listed) != 1 or not isinstance(listed[0], str) or not same_path(listed[0], p.adapter.gguf):
            return f"the server lists the adapters {listed}, not exactly {rel(p.adapter.gguf)}", None
        if not bfcl.applies_adapter_by_default(server.client):
            return ("a completion without a lora field does not match the one at lora scale 1, or matches "
                    "scale 0 too: ACEBench's requests would not measure the adapter"), None
    cached = bfcl.cached_tokens(server.client)
    if cached != [0, 0]:
        return (f"the same prompt sent twice without a cache_prompt field came back with {cached} prompt "
                "tokens taken from the cache (usage.prompt_tokens_details.cached_tokens), not [0, 0]: "
                "ACEBench's answers would depend on the request before them"), None
    if setup.think:
        off = sampling_problem(props, setup)
        if off:
            return off, None
        return think_probe(server.client, p, setup)
    block = thinking(server.client)
    if block:
        switch = setup.template_switch()
        start = f"--chat-template-kwargs {switch}" if switch else "a template without a thinking block"
        return (f"a chat completion sent as ACEBench sends one answered with a thinking block ({block}): "
                f"the server must start with {start}"), None
    return None, None


def outermost_bracket(text: str) -> str | None:
    """The first outermost [...] of text, as ACEBench's eval_main.extract_outermost_bracket_content takes
    it."""
    start, depth = -1, 0
    for i, char in enumerate(text):
        if char == "[":
            if depth == 0:
                start = i
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start:i + 1]
    return None


def acebench_format(text: object) -> bool:
    """Whether an answer is in ACEBench's format: its first outermost [...] is a Python list of calls."""
    inner = outermost_bracket(text) if isinstance(text, str) else None
    if inner is None:
        return False
    try:
        tree = ast.parse(inner, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    return isinstance(tree.body, ast.List) and all(isinstance(e, ast.Call) for e in tree.body.elts)


def format_counts(found: Mapping[str, list[dict]]) -> dict[str, dict]:
    """Per category: the answers, those holding <tool_call>, those in ACEBench's format, and those that are
    text without a character to read (FORMAT_RULE)."""
    counts = {}
    for category, entries in found.items():
        results = [entry.get("result") for entry in entries]
        counts[category] = {"n": len(results),
                            "tool_call": sum(1 for r in results if isinstance(r, str) and TOOL_CALL in r),
                            "acebench_format": sum(1 for r in results if acebench_format(r)),
                            "empty": sum(1 for r in results if isinstance(r, str) and not r.strip())}
    return counts


def answers(clone: Clone, language: str, model: str) -> dict[str, list[dict]]:
    """Every category's answers, one per question of data_all; OutputError otherwise."""
    found = {}
    for category in clone.categories:
        path = outputs(clone.root, language, model)["result"] / f"data_{category}_result.json"
        if not path.is_file():
            raise OutputError(f"{rel(path)} is missing: ACEBench generated no answer for {category} "
                              f"({language})")
        entries = bfcl.json_lines(path)
        expected = clone.questions[language][category]
        if sorted(map(str, (entry.get("id") for entry in entries))) != sorted(map(str, expected)):
            raise OutputError(f"{rel(path)} holds {len(entries)} answers, and their ids are not the "
                              f"{len(expected)} questions of {category} ({language})")
        found[category] = entries
    return found


def inference_errors(found: Mapping[str, list[dict]]) -> dict[str, int]:
    """Per category, the answers that are no text: ACEBench's record of a request that got no content."""
    return {category: sum(1 for entry in entries if not isinstance(entry.get("result"), str))
            for category, entries in found.items()}


def counted(per_language: Mapping[str, Mapping[str, int]]) -> tuple[int, str]:
    """The total of counts per language and category, and the ones not 0 as "zh: a 2, b 1; en: c 1"."""
    total = sum(sum(rows.values()) for rows in per_language.values())
    text = "; ".join(f"{language}: " + ", ".join(f"{c} {n}" for c, n in rows.items() if n)
                     for language, rows in per_language.items() if any(rows.values()))
    return total, text


def fraction(value: object) -> bool:
    return type(value) in (int, float) and 0 <= value <= 1


def scores(clone: Clone, language: str, model: str, found: Mapping[str, list[dict]]) -> dict[str, dict]:
    """{n, correct, accuracy} per category from ACEBench's score files, checked (module docstring)."""
    scored = {}
    for category, entries in found.items():
        path = outputs(clone.root, language, model)["score"] / f"data_{category}_score.json"
        if not path.is_file():
            raise OutputError(f"{rel(path)} is missing: ACEBench scored no answer of {category} ({language})")
        header, *failed = bfcl.json_lines(path) or [{}]
        n, right, accuracy = header.get("total_count"), header.get("correct_count"), header.get("accuracy")
        valid = type(n) is int and type(right) is int and 0 <= right <= n and n == len(entries) > 0
        multi_turn, special = "normal_multi_turn" in category, category.startswith("special")
        if valid:
            if special:
                valid = accuracy == right / n
            elif multi_turn:
                valid = fraction(accuracy) and fraction(header.get("process_accuracy"))
            else:
                valid = accuracy == round(right / n, 3)
        if valid:
            valid = len(failed) == n - right if special or multi_turn else len(failed) >= n - right
        if not valid:
            raise OutputError(f"{rel(path)}: the header {header} does not agree with its {len(failed)} wrong "
                              f"answers or with the {len(entries)} answers of the result file ({category}, "
                              f"{language})")
        scored[category] = {"n": n, "correct": right, "accuracy": right / n}
        if multi_turn:
            scored[category].update(acebench_accuracy=accuracy,
                                    acebench_process_accuracy=header["process_accuracy"])
    return scored


def overall(languages: Mapping[str, Mapping[str, Mapping]]) -> dict:
    """The aggregate over every (language, category) pair (AGGREGATE_RULE)."""
    return bfcl.aggregate({f"{language}/{category}": row for language, rows in languages.items()
                           for category, row in rows.items()})


def ace_step(run, cmd: list[str], env: dict[str, str], cwd: Path, log: Path) -> int:
    """One ACEBench step, run from the clone, with its output in log; its exit code (OSError if it cannot
    start)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as out:
        return run(cmd, env=env, cwd=os.path.abspath(cwd), stdout=out, stderr=subprocess.STDOUT).returncode


def summary_row(run, python: str, clone: Clone, language: str, model: str,
                env: dict[str, str]) -> dict | None:
    """The model's row of ACEBench's summary workbook, None when eval_main.py wrote none; OutputError when the
    workbook cannot be read or has not exactly one row of the model."""
    book = clone.root / "score_all" / f"score_{language}" / "result.xlsx"
    try:
        done = run([python, "-c", SUMMARY_CODE, os.path.abspath(book), model], env=env,
                   cwd=os.path.abspath(clone.root), capture_output=True, text=True, encoding="utf-8",
                   errors="replace")
        found = json.loads(done.stdout) if done.returncode == 0 else None
    except (OSError, ValueError) as exc:
        raise OutputError(f"ACEBench's summary {rel(book)} cannot be read "
                          f"({type(exc).__name__}: {exc})") from None
    if not isinstance(found, dict):
        why = " ".join((done.stderr or "").strip().splitlines()[-1:]) or "no error text"
        raise OutputError(f"ACEBench's summary {rel(book)} cannot be read ({why})")
    if not found.get("file"):
        return None
    rows = found.get("rows") or []
    if len(rows) != 1:
        raise OutputError(f"ACEBench's summary {rel(book)} has {len(rows)} rows of {model}, not one")
    return rows[0]


def copy_folder(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target)


def evaluate_system(p: Plan, *, args: argparse.Namespace, binary: Path, clone: Clone, build: str | None,
                    setup: Setup, popen, run,
                    client: Callable[[str], LocalClient]) -> tuple[int, dict | None]:
    """Serve, generate both languages, stop, check, evaluate and copy one system (module docstring): (0, its
    row) or (exit code, None)."""
    p.root.mkdir(parents=True, exist_ok=True)
    env = bfcl.child_env(os.environ, args.drop_env or [], p.settings)
    url, server_log = p.settings["LOCAL_OPENAI_BASE_URL"], p.root / "server.log"
    python = next(iter(p.generate.values()))[0]
    served = "the base model" if p.adapter is None else rel(p.adapter.gguf)
    print(f"{p.system}: serving {served} (server log {rel(server_log)})", flush=True)
    probe = None
    try:
        with LocalServer(binary, model=args.model, adapter=None if p.adapter is None else rel(p.adapter.gguf),
                         port=args.port, ctx_size=setup.context, extra=p.extra, timeout=args.timeout,
                         log=server_log, popen=popen, client=client) as server:
            try:
                problem, probe = server_problem(p, server, build, setup)
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                problem = f"probing the server failed: {type(exc).__name__}: {exc}"
            if problem:
                print(f"FAIL: {p.system}: {problem}")
                return 3, None
            build = server.props["build_info"]
            try:
                reach = run([python, "-c", REACH_CODE, url], env=env, cwd=os.path.abspath(clone.root),
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=REACH_TIMEOUT)
                reached, why = reach.returncode == 0, (reach.stderr or "").strip().splitlines()[-1:]
            except subprocess.TimeoutExpired:
                reached, why = False, [f"no answer within {REACH_TIMEOUT:g} s"]
            except OSError as exc:
                print(f"FAIL: {p.system}: ACEBench's interpreter {rel(python)} cannot be started for the "
                      f"reach check ({exc}; --acebench-python)")
                return 2, None
            if not reached:
                print(f"FAIL: {p.system}: ACEBench's environment cannot reach the server at {url} "
                      f"({' '.join(why) or 'no error text'}): drop the inherited variables that keep it from "
                      "the server with --drop-env PATTERN (docs/m0-验证.md, S5)")
                return 2, None
            for language, cmd in p.generate.items():
                log = p.root / language / "generate.log"
                print(f"{p.system}: ACEBench generate {language} (log {rel(log)})", flush=True)
                try:
                    code = ace_step(run, cmd, env, clone.root, log)
                except OSError as exc:
                    print(f"FAIL: {p.system}: ACEBench generate cannot be started ({exc}; {rel(python)})")
                    return 4, None
                copy_folder(outputs(clone.root, language, p.model)["result"], p.root / language / "result")
                if code != 0:
                    print(f"FAIL: {p.system}: ACEBench generate {language} exited with {code} "
                          f"(log {rel(log)})")
                    bfcl.show_tail(log)
                    return 4, None
    except PortBusy as exc:
        print(f"FAIL: {exc}")
        return 2, None
    except NotReady as exc:
        print(f"FAIL: {p.system}: {exc} (server log {rel(server_log)})")
        return 3, None
    except OSError as exc:  # starting llama-server: ACEBench's processes report their own
        bfcl.discard_unstarted(p.root, server_log)
        unusable = bfcl.cannot_run(exc)
        print(f"FAIL: {p.system}: llama-server could not be started ({exc})"
              + (f": check --llama-server or {SERVER_VAR}" if unusable else ""))
        return (2 if unusable else 3), None
    found = {}
    for language in p.generate:
        try:
            found[language] = answers(clone, language, p.model)
        except OutputError as exc:
            print(f"FAIL: {p.system}: {exc} (log {rel(p.root / language / 'generate.log')})")
            return 1, None
    total, counts = counted({language: inference_errors(rows) for language, rows in found.items()})
    if total:
        print(f"FAIL: {p.system}: ACEBench recorded {total} inference errors ({counts}): answers without "
              "content, which would count as wrong answers")
        return 4, None
    formats = {language: format_counts(rows) for language, rows in found.items()}
    total, counts = counted({language: {c: cell["empty"] for c, cell in rows.items()}
                             for language, rows in formats.items()})
    if total:  # no failure: ACEBench got text and scores it (module docstring: Checks of ACEBench's output)
        why = (f"a reasoning that does not close within max_tokens {setup.max_tokens} leaves no answer, "
               f"which the server's reasoning budget ({setup.reasoning_budget}) should prevent" if setup.think
               else "the model answered nothing")
        print(f"{p.system}: {total} empty answers ({counts}), which ACEBench scores as wrong: {why}",
              flush=True)
    languages = {}
    for language, cmd in p.evaluate.items():
        log, folder = p.root / language / "evaluate.log", p.root / language
        print(f"{p.system}: ACEBench evaluate {language} (log {rel(log)})", flush=True)
        try:
            code = ace_step(run, cmd, env, clone.root, log)
        except OSError as exc:
            print(f"FAIL: {p.system}: ACEBench evaluate cannot be started ({exc}; {rel(python)})")
            return 4, None
        made = outputs(clone.root, language, p.model)
        copy_folder(made["score"], folder / "score")
        copy_folder(made["excel"], folder / "excel")
        book = clone.root / "score_all" / f"score_{language}" / "result.xlsx"
        if book.is_file():
            shutil.copyfile(book, folder / "acebench-summary.xlsx")
        if code != 0:
            print(f"FAIL: {p.system}: ACEBench evaluate {language} exited with {code} (log {rel(log)})")
            bfcl.show_tail(log)
            return 4, None
        try:
            scored = scores(clone, language, p.model, found[language])
            acebench_summary = summary_row(run, python, clone, language, p.model, env)
        except OutputError as exc:
            print(f"FAIL: {p.system}: {exc} (log {rel(log)})")
            return 1, None
        languages[language] = {
            "generate": as_recorded(p.generate[language]), "evaluate": as_recorded(cmd),
            "categories": scored, "aggregate": bfcl.aggregate(scored),
            "groups": {group: bfcl.aggregate({c: scored[c] for c in categories})
                       for group, categories in clone.groups.items()},
            "format": formats[language],
            "inference_errors": inference_errors(found[language]),
            "acebench_summary": acebench_summary,
        }
    row = {
        "acebench_model": p.model, "server_command": p.record, "server_build": build,
        "adapter": None if p.adapter is None else {
            "gguf": rel(p.adapter.gguf), "sha256": p.adapter.sha256,
            "selection": None if p.adapter.selection is None else rel(p.adapter.selection)},
        "raw": rel(p.root),
        **({"probe": probe} if setup.think else {}),
        "aggregate": overall({language: block["categories"] for language, block in languages.items()}),
        "languages": languages,
    }
    print(f"{p.system}: {100 * row['aggregate']['accuracy']:.2f}% (mean over "
          f"{row['aggregate']['categories']} language-category pairs)", flush=True)
    return 0, row


def summary(rows: dict[str, dict], clone: Clone) -> None:
    labels = [f"{language} {category}" for language in clone.questions for category in clone.categories]
    width = max(len(label) for label in labels) + 2
    print("category".ljust(width) + "".join(f"{system} (tool_call)".rjust(22) for system in rows))
    for language in clone.questions:
        for category in clone.categories:
            cells = []
            for row in rows.values():
                cell, counts = row["languages"][language]["categories"][category], \
                    row["languages"][language]["format"][category]
                cells.append(f"{100 * cell['accuracy']:.2f}% ({counts['tool_call']}/{cell['n']})")
            print(f"{language} {category}".ljust(width) + "".join(cell.rjust(22) for cell in cells))
    print("mean".ljust(width) + "".join(f"{100 * row['aggregate']['accuracy']:.2f}%".rjust(22)
                                        for row in rows.values()))


def main(argv: list[str] | None = None, *, popen=subprocess.Popen, run=subprocess.run,
         client: Callable[[str], LocalClient] = LocalClient, git=subprocess.run) -> int:
    """popen, run, client and git stand in for subprocess.Popen (llama-server), subprocess.run (ACEBench's
    processes), LocalClient and subprocess.run (git, for the clone checks) in tests."""
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        plans, binary, clone, setup = prepare(args, git)
    except UsageError as exc:
        print(f"FAIL: {exc}")
        return 2
    drop = args.drop_env or []
    dropped = len(os.environ) - len(bfcl.child_env(os.environ, drop, {}))
    describe(plans, clone, dropped, setup, args)
    if args.dry_run:
        return 0
    python = next(iter(plans[0].generate.values()))[0]
    try:
        done = run([python, "-c", VERSION_CODE], env=bfcl.child_env(os.environ, drop, plans[0].settings),
                   cwd=os.path.abspath(clone.root), capture_output=True, text=True, encoding="utf-8",
                   errors="replace")
    except OSError as exc:
        print(f"FAIL: {rel(python)} cannot be started ({exc}): is it ACEBench's interpreter "
              "(--acebench-python)?")
        return 2
    version = (done.stdout or "").strip() if done.returncode == 0 else ""
    if not version:
        why = " ".join((done.stderr or "").strip().splitlines()[-1:]) or "no error text"
        print(f"FAIL: {rel(python)} cannot tell the version of openai ({why}): is it ACEBench's environment "
              "(--acebench-python)?")
        return 2
    rows: dict[str, dict] = {}
    build = None
    for p in plans:
        code, row = evaluate_system(p, args=args, binary=binary, clone=clone, build=build, setup=setup,
                                    popen=popen, run=run, client=client)
        if code:
            return code
        rows[p.system], build = row, row["server_build"]
    results = {
        "benchmark": benchmark(setup),
        "smoke": args.smoke, "languages": list(clone.questions), "groups": clone.groups,
        "generation": setup.generation, "mode": str(setup.mode), "family": setup.family_record(),
        "decoding": setup.decoding_record(), "budget": setup.budget_record(),
        "aggregate_rule": AGGREGATE_RULE, "format_rule": FORMAT_RULE,
        "base_model": rel(args.model),
        "acebench": {"clone": rel(clone.root), "commit": clone.commit,
                     "patch": {"file": PATCH.as_posix(), "sha256": sha256_of(PATCH)},
                     "data": {"files": clone.data_files, "identical_to": clone.commit,
                              "check": "every file under data_all/ has the git blob name of the upstream "
                                       "commit's, computed from its bytes; none missing, none added"},
                     "python": rel(args.acebench_python), "openai": version},
        "environment": plans[0].settings,
        "drop_env": {"patterns": len(drop), "variables_dropped": dropped},
        "git_commit": git_commit(REPO), "git_dirty": git_dirty(REPO),
        "systems": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                        newline="\n")
    summary(rows, clone)
    print(f"results {rel(args.out)}; raw outputs under {rel(args.raw)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
