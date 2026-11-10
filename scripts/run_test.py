"""Open the locked test sets once and evaluate the configured systems on them in one session: the phase-4 test
runner (docs: the M3 plan, phase 4 and defaults 2, 4 and 5; docs/m5-预登记.md; docs/plan.md, 测试集锁).

    .venv/Scripts/python.exe scripts/run_test.py --config results/test-config-m3.json [--dry-run] \\
        [--llama-server PATH]
    .venv/Scripts/python.exe scripts/run_test.py --config results/test-config-m3.json --abandon "<reason>"

The only command that opens a test set (research_agent.eval.lock, who opens what). Run it from the repository
root once the checkpoints are chosen and the configuration is committed with the preregistration, with the GPU
free (no training, no other server) and the DeepSeek key in the environment: first with --dry-run, then, on
the same commit, without it.

The systems (RUN_ORDER): ft, b0, b1, ftr, ft2, ft2r, ft3, ft3r, final, finalg and b4 run on llama-server, b2
on DeepSeek. ftr
(FT+R, eval.systems) is the fine-tuned model on the FT server, routed to b2 under the frozen routing
configuration; its routed calls are b2's calls. M3 ran ft, b0, b1, b4 and b2; M5 runs ftr alone and compares
it with the records b2 left in M3 (docs/m5-预登记.md), without running b2 again. ft2 and ft2r are M6's aliases
(docs/m6-计划.md §4 WP8, §5): ft2 is the ft system with the second adapter, adapters.ft2 (FT2), and ft2r its
routed version (FT2+R2), ft2 on the same server routed to b2 under results/routing-m6.json; each is built as
the system it stands for (BUILT_AS: ft, ftr) and its info names both ("system", "built_as"). M6 runs ft2 then
ft2r and compares them with the records of ft and b2 (M3) and of ftr (M5, descriptive). ft3 and ft3r are M7's
aliases (docs/m7-计划.md WP2): ft3 is the ft system with the third adapter, adapters.ft3 (FT3, on Qwen3-4B:
its server loads the adapter's own base model, and its prompts are rendered and counted with the adapter's own
tokenizer folder, on the family path when the adapter names a generation, below), and ft3r its routed version
(FT3+R3), ft3 on the same server routed to b2
(deepseek-flash, the fallback of R3, docs/m7-计划.md WP-K), built as ft and ftr; M7 no longer runs them (the
owner's decision of 2026-10-29), which older configurations still may. final and finalg are M7's final local
system and its greedy self (docs/m7-本地提升计划.md WP5; docs/m7-预登记.md, revision 2), both built as ft on
one server, the one with the final adapter, adapters.final (FT3's checkpoint 1635, the owner's decision of
2026-10-30), on the family path: final answers by a vote over the samples its options give (votes, vote_mode)
with the runtime rules (rules) and, when configured, the glossary's note, finalg greedily (decoding) without a
wrapper and single step only (end_to_end); final is held against its frozen file (below). A routed system's
cloud system is ROUTED's, b2 for each of them: its routed calls are that system's calls, priced at its model's
prices, and its routing file must name that model (cloud_model, eval.systems.CLOUD_MODELS). M7 opens the
second family of test sets, v2 (test_sets), whose development set is its own (FAMILY_DEV_SETS:
data/eval/dev.jsonl and data/eval/v2/dev-v2.jsonl jointly, docs/m7-计划.md §5 开发集口径 and WP-I 7).
The configuration (results/test-config-m3.json; results/test-config-m3.example.json documents every M3 key
under "_doc", and its placeholders are refused, as are those of results/test-config-m7.example.json, which
documents an M7 session on the v2 sets with FT3's generation). milestone: the stage the opens and the spending
are booked under (M3, M5); purpose: what every open is for; test_sets (optional): the family of test sets the
session opens (eval.lock.TEST_FAMILIES): "v1", the default, data/eval/test.jsonl and e2e-test.jsonl, which M3
to M6 opened, with their access logs data/eval/test-access.jsonl and e2e-test-access.jsonl and their
completion markers data/eval/test-completions.jsonl, or "v2" (M7, docs/m7-计划.md WP4),
data/eval/v2/test-v2.jsonl and e2e-test-v2.jsonl with their own logs, test-v2-access.jsonl and
e2e-test-v2-access.jsonl, and markers, test-completions-v2.jsonl, in the same folder; the access logs, markers
and opens below are the family's, as is the development set (FAMILY_DEV_SETS: v1's data/eval/dev.jsonl; v2's
data/eval/dev.jsonl and data/eval/v2/dev-v2.jsonl jointly, in that order), and a v2 configuration has no
reference (reference records are those of opens of v1); systems: those the session runs, a non-empty list of
distinct names of RUN_ORDER, or an object from each such name to its options (M7, below; {} for none), at
most one of them routed (ftr, ft2r or ft3r, ROUTED); adapters: for ft, ft2,
ft3, final and b4, the GGUF with its SHA-256, optionally the checkpoint selection that chose it and,
optionally, the base model its server loads (model, else server.model), the model folder whose chat template
renders its systems' prompts and counts their context (tokenizer, else the configuration's tokenizer), so that
one session can run FT2 on a Qwen3-1.7B server and FT3 on a Qwen3-4B one, and how its systems generate
(generation, the family path below: {"mode", "decoding", "reasoning_budget", "base_seed"}); one for the server
of every local system the session runs (ft for ft, b0, b1 and ftr; ft2 for ft2 and ft2r; ft3 for ft3 and ft3r;
final for final and finalg, which needs a generation; b4 for b4), the others optional and not checked but for
their form; scorers (optional, M7): the scorers besides the default one (eval.metrics.SCORERS: normalized)
whose results the single-step files carry too and the comparisons may use; server: the base model of a server
whose adapter names none, the port, the context size, the readiness timeout and, optionally, the llama.cpp
build the servers' /props must report; tokenizer: the model folder of the chat template of a local system
whose adapter names none, whose token counter b2 uses too; b1_block_sha256_16: the SHA-256 prefix of b1's
demonstrations block, required when b1 runs and otherwise optional; b2.max_cost_rmb: what each system that
spends (SPENDING: b2, and the routed calls of ftr, ft2r and ft3r) may spend in all, its single-step and
end-to-end runs together (default 5); e2e: the seed and the date (null: the corpus recording date);
e2e_routing (optional, for a routed system): "sticky", the default, keeps a task in the cloud once a step is
routed, as M4 and M5 did; "stepwise" (M6, docs/m6-计划.md §3.3 C7) decides every step on its own and never
routes a final text answer (agent.route.RoutedModel); statistics: the bootstraps' seed and resamples; routing
(required when a routed system runs, refused otherwise): the path of its frozen routing configuration
(results/routing-m4.json for ftr, results/routing-m6.json for ft2r, results/routing-m7.json for ft3r) and its
SHA-256; reference (optional): the milestone, earlier than this one, of the systems that the comparisons use
without running them, and per system (a name of RUN_ORDER the session does not run) its results file, its
single-step records of that milestone's open and their SHA-256, its records of the development set with their
SHA-256 and, optionally, its own earlier milestone when it is not the block's (M6: ft and b2 from M3, ftr from
M5); comparisons: the preregistered primary comparisons, rows of these kinds: tested by McNemar (name, a, b,
each a system of the session or a reference system and at least one of the session, optionally "test":
"mcnemar", with an optional margin in (0, 1) for the descriptive non-inferiority bound), noninferiority (name,
a, b and {"margin"}: the one-sided paired bootstrap of H2, docs/m6-计划.md §5, entering Holm with p_NI),
acebench (name and {"a", "b", "margin"}: the raw ACEBench folders of two runs, read answer by answer
(eval.acebench) for the stratified bootstrap of H3, entering Holm with its p_NI), descriptive (name, a, b,
"descriptive": true: compared like a tested row but outside the Holm family, with no p), deferred to a later
milestone (name, a, b, the reason and the p it enters Holm with: FT+R against B2 at p = 1 in M3) and recorded
(name, a, b and the earlier milestone's comparisons file whose row of that name gives the p; reference is then
required), and a tested, non-inferiority or descriptive row may name the scorer its systems' single-step
successes are compared under ("scorer", M7: one of eval.metrics.SCORERS, the default one when absent; another
one must be among scorers); gpu: the most memory in use that still counts as idle, and the sampling interval.
A missing or unknown key, a value of the wrong kind and a path that is not relative to the repository root are
refused.

A system's options (M7, docs/m7-本地提升计划.md WP5; SYSTEM_KEYS, each optional): votes, a whole number from
1: the system answers by a vote over that many samples, the greedy answer breaking ties (agent.vote,
eval.systems: its generation with that many votes, eval.systems.with_votes, which needs the adapter's
generation to sample); vote_mode, with votes, the vote's answer keys (agent.vote.VOTE_MODES; final when
absent); rules, true or false: the runtime rules (agent.rules) on every answer, every candidate's when voting;
glossary, null or {"path", "sha256"}: the zh-en glossary whose note every prompt carries (agent.noted);
decoding, "greedy": the adapter's generation decoded greedily (finalg's); end_to_end, true or false: false
runs the system single step only (finalg's), and it has no end-to-end results; frozen, {"path", "sha256"}:
final's alone, and required for it, the file scripts/freeze_final.py writes (results/final-m7.json). votes,
vote_mode, rules, glossary and decoding are for a local system that is not routed, votes and decoding on the
family path (its adapter names a generation), and not votes with decoding. finalg takes decoding greedy and
none of votes, vote_mode, rules and glossary, and runs only beside final, whose frozen file holds it
(check_options). While scripts/freeze_final.py (or
scripts/decide_m7.py) reads the configuration, frozen.sha256 may be a placeholder (check_config's freezing):
the frozen file is not written yet. A system without options, and every system of a list, runs as it ran
before M7's final system.

The family path (M7; docs/m7-计划.md §2 推理 and 解码, WP-M: the decoding, the seed and R are configured). An
adapter's generation, {"mode": "think" or "no_think", "decoding": "greedy" or "sampled", "reasoning_budget":
R, "base_seed": N, a sampled decoding's and required there}, runs every system on its server (ft3 and ft3r's
local side for adapters.ft3) on the family path, as scripts/run_eval.py and run_e2e.py run a local system:
eval.systems.build_system without a render, over the family of the adapter's tokenizer folder
(research_agent.family.load_family, loaded once per folder) and these generation settings
(llm.generation.GenerationSettings: the mode; the family's own decoding of that mode when sampled, greedy
decoding otherwise, as run_eval's --greedy decodes think mode; R; the base seed, 0 for a greedy decoding,
which sends none), its context counted on the family's prompt in that mode
(agent.context.family_token_counter), single step and end to end alike. Its records then carry the reasoning
and the thinking record of every step generated thinking or sampling, and its results' info the generation
(GenerationSettings.record(), as run_eval's manifest records it) and the family. The configuration check
refuses (exit 1) a mode other than think or no_think, a decoding other than greedy or sampled, a reasoning
budget below 1, a base seed with a greedy decoding and none with a sampled one, and, in think mode, a
server.ctx_size below 4096 + R + 520 tokens (the prompt's limit, R and the answer with its close,
docs/m7-计划.md §2 端到端预算); once the family is loaded, a mode the family does not have and a sampled
decoding of a mode the family decodes greedily. A server whose adapter names no generation runs its systems on
the legacy path, as M3 to M6 ran them: the folder's chat template renders the text the server completes
(load_render, load_counter). So every session without a generation (the M3 to M6 configurations) writes byte
for byte what it wrote before.

The final system's frozen file (M7, docs/m7-本地提升计划.md WP5): scripts/freeze_final.py, never a hand,
writes results/final-m7.json from the development runs of final's candidates: frozen_identity of the
configuration and, per system (final; finalg with its generation) and per development item, its success under
every scorer and output_hash of its record, the SHA-256 of the canonical JSON of what the model answered (its
status and, when ok, its text, its calls by name with their parsed arguments and its reasoning; no timing,
usage, latency, signals, vote or route bookkeeping, nor the call ids, which differ between runs). The
preflight checks the file and the dry run holds the online runs against it (below).

Before anything (exit 2): the current folder is the repository root, and the llama-server binary
(--llama-server or LLAMA_SERVER), the base models and the tokenizer folders (server.model, tokenizer and those
that the adapters of the servers the session starts name), when b1 runs, b1's training set and requests file,
and the development sets of the family (the dry run runs them and a routing file must have been chosen on
them; v2's data/eval/v2/dev-v2.jsonl is built by docs/m7-计划.md WP-I 5, and no session or dry run of v2 runs
without it) are here; then the family of the tokenizer folder of every server the session starts whose adapter
names a generation is loaded (exit 2 when it cannot be) and the generation checked against it (exit 1: the
family path above). Then the preflight, before any test set is read (exit 4, nothing opened, nothing started),
in this order: no open of the milestone without a completion marker (exit 5, below); the configuration
committed (in HEAD and unchanged); a clean working tree; an open left on both access logs of the family (at
most 3 each) and no session folder yet for the next open; both test sets of the family matching their locks
(their bytes are hashed, not parsed); the adapter of every server the session starts present with its recorded
SHA-256 and, with a selection, the GGUF it chose; when b1 runs, b1's demonstrations block hashing to the
recorded prefix; when a routed system runs, the routing file present with its recorded SHA-256, chosen on the
family's development sets as they are now (as many as the family has, each with the SHA-256 the file records
for it, in order: v1's file records data/eval/dev.jsonl's as dev_sha256, v2's records both of its sets as
dev_sets, eval.systems.routing_dev_sets), its adapter_sha256 that of the adapter of the routed system's server
(adapters.ft for ftr, adapters.ft2 for ft2r, adapters.ft3 for ft3r), its ft_generation
(scripts/route_curve.py: FT's generation over the runs the routing was chosen on) the generation of that
server's systems (GenerationSettings.record() of the adapter's generation; neither one on the legacy path),
its cloud_model the model of the routed system's cloud system (deepseek-flash, b2's, for all three), and a
chosen point with its routed count and successes, and for ft2r and ft3r (RULED) the M6 policy fields of
scripts/route_curve.py (rules, chosen_policy, policies) present and consistent: the rules well formed, the
chosen policy among the policies with the file's rules and the chosen point its own; every glossary a system's
options name present with its recorded SHA-256; when final runs, its frozen file (systems.final.frozen)
present with its recorded SHA-256 and recording, field by field, what the configuration gives now
(frozen_identity: the final adapter's GGUF and SHA-256, its base model with the SHA-256 of the file, its
tokenizer folder and family, final's generation with its votes, the votes, the vote mode, the runtime rules of
this code, RULES_VERSION, the glossary with the SHA-256 of its file or null, and the family's development sets
with the SHA-256 of their files), and, when finalg runs, finalg's generation (finalg.generation); with
references, every
reference results file and every recorded comparisons file having the SHA-256 that the reference milestone's
two completion markers (one per access log, status complete; each reference system's own milestone when it
names one) record for it among their outputs, each results file holding its system's configured records, every
reference records file (the test and the development records) present with its configured SHA-256, and every
recorded comparison found by name in its file as a tested comparison of the same a and b, whose p it takes;
for every acebench comparison, both raw folders readable answer by answer and holding the same (language,
category) strata with the same answers (eval.acebench), which the session then keeps in memory; a real session
only: runs/test-<m>/dry-run/session.json is that of a dry run of the same git commit (HEAD) and the same
configuration SHA-256 whose development check passed (a dry run needs no earlier dry run); when a system
spends, the milestone's API budget able to take the cap of each such system (what the stage has spent plus the
caps within the stage cap, and within the cap over all stages); nothing answering on the servers' port; the
GPU idle (nvidia-smi: memory in use at most gpu.idle_max_mib). Each refusal names what does not match. Then,
still before any open (exit 2 on failure): the chat template and the token counter of every tokenizer folder
the session uses on the legacy path (the configuration's, and those of the servers whose adapter names no
generation), once each, and the recorded world of the end-to-end tasks are loaded, the owner's functions
probed, and, when a system spends, the cloud system it calls built once (b2: its key, its price).

--dry-run: the preflight and the whole session below over the first item of the family's development set and
the first end-to-end development task (DevSet, each development set lock-checked, and the lock of
data/eval/e2e-dev.jsonl; never a test item), except that the routed system, final and finalg run every item of
the development set single step (v2: every item of data/eval/dev.jsonl, then of data/eval/v2/dev-v2.jsonl, no
id in both): every system, every server, the GPU sampling and the result files, all of it under
runs/test-<m>/dry-run/ (the result files in its results/). The session's "sets" name the development set as
the test set's would be named, by its path and SHA-256 (v1), or, for a family of several, by "sets", each
with its path and SHA-256. The references are their development records, scored against the development
items. Its session.json records the git commit, the configuration and "dev_check":
{"passed", "routed", "successes", "expected": {"routed", "successes"}}: the routed system's routed items
(route answered_by "cloud") and successes over the development set against the routing file's chosen point,
which a live run must reproduce (passed, with null counts, when no routed system runs); when final runs, also
"frozen" (Session.frozen_check): {"path", "sha256", "passed", "final": {"n", "differ": per scorer of
eval.metrics.SCORERS the ids whose success is not the frozen file's, "outputs_differ": how many of its outputs
(output_hash) are not, a count that decides nothing}, "finalg": {"n", "outputs_differ": the ids whose output
is not the frozen one}}, the online runs over every development item held against the records frozen from the
development runs; a mismatch (a routed count or success, a success of final, an output of finalg) fails the
dry run (exit 6) once session.json is written, and a real session is refused until a dry run passes. Nothing
is opened, no marker is written and results/ is left alone.

The opening: of the family, LockedSet(single-step set, its access log).open(milestone, purpose) once, and
LockedSet(end-to-end set, its access log).open(milestone, purpose) once (v1: data/eval/test.jsonl with
data/eval/test-access.jsonl, data/eval/e2e-test.jsonl with data/eval/e2e-test-access.jsonl), the purpose being
the configuration's with the git commit and the configuration's SHA-256. The items and tasks stay in memory.
The single-step items reach the runner and the scorer as the OpenedTestSet the open returned; the tasks,
checked as load_tasks checks a file (eval.e2e.check_tasks), as an OpenedTestSet carrying their open's entry
(eval.lock: the test runner's path through the test-split guard).

The session, in this order: the reference systems' records (the test records; in a dry run the development
records) are scored against the items in memory by eval.metrics.score_items; the FT server
(research_agent.llm.server.LocalServer with the ft adapter, which it must list as its only adapter, the
configured build and one slot, SLOTS) runs ft at lora scale 1, then b0 and b1 at scale 0 (b1 with its
demonstrations), then ftr, ft at scale 1 routed to b2 by the configured routing file, each single step and
then end to end; it is stopped; the server with the ft2 adapter runs ft2 and then ft2r the same way; it is
stopped; the server with the ft3 adapter runs ft3 and then ft3r, routed to b2, the same way; it is stopped;
the server with the final adapter runs final and then finalg the same way, a system whose end_to_end is false
single step only; it is stopped; the B4 server runs b4 the same way; it is stopped; b2 runs on DeepSeek (M7:
FT2, then final and finalg, then B2). Each server loads its adapter's base
model (else server.model), and each system is built with its adapter's tokenizer folder (else tokenizer; a
cloud system with tokenizer). A system that spends (b2, and the routed calls of ftr, ft2r and ft3r) goes
through the response cache and the ledger in runs/llm/, booked under the milestone, its single-step and
end-to-end runs within one cap. Each run is built by eval.systems.build_system, a system with options with
them (system_generation: its adapter's generation, decoded greedily under decoding, with its votes; rules,
glossary and vote_mode: eval.systems' inference wrappers), and goes through eval.runner.run_items and
eval.e2e.run_tasks with the timing sidecars (a voted step's llama.cpp timings summed over its requests,
eval.systems.Built.timings), as scripts/run_eval.py and scripts/run_e2e.py run the development sets, and each
single-step run is scored in this process by eval.metrics.score_items and summarize, as scripts/score.py
scores them, under the default scorer and under each of scorers (final and finalg, when final runs, under
every scorer: the frozen file holds both). Beside each server nvidia-smi samples
the GPU (eval.resources), its idle baseline sampled just before the server starts.

Outputs. Under runs/test-<m>/open-<n>/ (n: which open of the family's single-step set this is): <system>.jsonl
with <system>.timing.json, e2e-<system>/ (transcripts, note stores, timing.json), server-<name>.log and
session.json (its servers with their command and base model). In results/ (JSON, two-space indent, LF,
repository-relative paths, no time but the opens'): test-<m>-<system>.json (the open, the set with the number
of items the system ran, the system as it was built, the records' statuses and, for a system that spends, its
cloud turns per thinking mode, the summary with the drafted and the handwritten subsets', the per-item scores;
with scorers configured, right after the subsets "scorers": {scorer: {"summary", "subsets": {"drafted",
"handwritten"}, "scores"}} per scorer of scorers, the default scorer's results staying where they were and the
file naming no scorer of its own (eval.report.scorer_blocks gives the block's numbers under
<system>-<scorer>); for a voted or ruled system "votes" and "rules", counted over its records (wrapper_counts,
descriptive); for a routed system (ftr, ft2r, ft3r) also "routing": n, routed and api_share, the hard triggers
counted by kind (the hard reason up to its colon), reference_success (the mean success of its local system and
of its cloud system over the same items, from the session or the references: ft2's single-step run and b2's M3
records for ft2r; null for a system with neither, as b2 in a v2 session that does not run it, which has no
references), random_success, the exact
expected success of routing as many items at random, (1 - api_share) x the local system's + api_share x the
cloud system's, and agreement: of the locally answered items, how many have exactly the output of the local
system's record of the item, and of the routed ones, how many have that of the cloud system's, each with its
n); test-<m>-e2e-<system>.json (the summary and the task records; for a routed system also "e2e_routing", the
mode its tasks ran under, and "routing": the tasks, those with a cloud call and the local and cloud calls,
from the task records, and the steps whose last attempt the local or the cloud model gave, from the
transcripts; for a voted or ruled system "votes" and "rules" counted over the attempts of its transcripts;
none for a system whose end_to_end is false); test-<m>-comparisons.json (the family in the configuration's
order: a McNemar comparison
("test": "mcnemar") by eval.metrics.compare_systems on the single-step successes of the session or of the
reference, under the row's scorer ("scorer", written when the row names one),
over the items both cover: the paired and the cluster bootstrap and the exact McNemar test, with
"noninferiority" (margin, paired_low, paired_above: paired_low > -margin, cluster_low, cluster_above) when it
has a margin, which is descriptive and leaves p alone; a non-inferiority comparison ("test": "noninferiority")
the same with "noninferiority": {margin, p_ni, below, cluster_p_ni, cluster_below}
(eval.metrics.noninferiority_p and its cluster version over the same draws as the intervals) and p = p_ni; the
ACEBench comparison ("test": "acebench") as a and b (the raw folders) and "acebench": per side the path, the
SHA-256 digest over the files read, their count, the answers and the stratum-mean accuracy, then the margin,
the strata, the answers, the stratum-mean difference with its paired interval, p_ni and the draws below
(eval.metrics.stratified_noninferiority_p) and per stratum its language, category, n, both correct counts and
both later-candidate counts (eval.acebench), with p = p_ni; a descriptive comparison ("descriptive": true)
like a McNemar one but without p; a deferred one with its p; a recorded one as name, a, b, recorded and the p
of its file; Holm over every row with a p; with references, the records each was scored from (and a system's
own milestone when the configuration names one), and in a real session "rescored": per reference system,
whether this session's scores equal, item by item, the "scores" of its results file, and the ids that differ,
a difference being recorded, not fatal, the comparisons using this session's scores); test-<m>-resources.json
(eval.resources: latency p50 and p95 per system with their n, local per call and per end-to-end step, cloud
over the paid calls only; for a routed system single step the local items' calls, the routed items' calls and
the routed and paid ones (a routed item's call covers the local attempt and the cloud's answer together), its
cost over the routed records only and its tasks' paid cost, and the ledger's, priced at its cloud system's
model; per server (named after its adapter: ft, ft2, ft3, final, b4) its base model, the VRAM over the idle
baseline and the power; b2's cost, paid and at the list price for all its turns, and the ledger's).
scripts/build_report.py turns them into numbers.

A crashed session is not resumed (the M3 plan, default 5): the items were in memory only, and an open whose
session did not reach its end is used up. At the end of a session each open gets its completion marker in the
family's markers (v1: data/eval/test-completions.jsonl; the access log, the open's entry, the outcome, the
result files with their SHA-256), and the runner refuses to start while an open of the milestone in the
family's access logs has none, telling the owner (exit 5). The owner decides; --abandon "<reason>" (run from
the repository root too) records the decision as a marker with status "abandoned" for every such open and runs
nothing, after which a session makes a new open (3 per set in all); both access logs and the markers are then
committed together. A marker whose open its access log no longer holds is refused (exit 4) by sessions and by
--abandon alike: an access log that lost entries, for instance to a git clean, would otherwise count its opens
again.

Exit 0 when the session is complete (with --dry-run: done and its development check passed; with --abandon:
recorded); 1 on a malformed configuration (a generation its family cannot run included); 2 on a usage error or
a missing prerequisite (above); 3 when a cap
stopped a run of a system that spends (every result and both markers are written, the statuses show the
stop); 4 when a preflight check refused; 5 when an earlier open of the milestone has no completion marker; 6
when the session failed after it began (a server that does not start, lists another adapter or reports
another build, a system that cannot be built, reference records that cannot be scored, malformed test
material, a dry run whose development check failed, or any unforeseen error, such as a result file that
cannot be written), the open then being used up and without a marker (a dry run opened nothing); the output
says which.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from research_agent.agent.context import TokenCounter, family_token_counter
from research_agent.agent.prompts import PROMPT_HASH, PROMPT_VERSION
from research_agent.agent.route import CLOUD, LOCAL, RouteRules
from research_agent.agent.rules import RULES, RULES_VERSION
from research_agent.agent.types import BUDGET_STOP
from research_agent.agent.vote import FINAL as VOTE_FINAL
from research_agent.agent.vote import VOTE_MODES
from research_agent.canonical import canonical_json
from research_agent.data.synth import corpus_recorded_on, probe
from research_agent.eval.acebench import AceItemsError, PerItem, pair_strata, per_item
from research_agent.eval.e2e import World, build_world, check_tasks, load_tasks, run_tasks
from research_agent.eval.e2e import summarize as summarize_tasks
from research_agent.eval.lock import (
    DEV_SET,
    E2E_DEV_SET,
    MAX_TEST_SET_OPENS,
    TEST_FAMILIES,
    DevSet,
    JsonlError,
    LockedSet,
    LockError,
    OpenedTestSet,
    TestFamily,
    check_lock,
    read_jsonl,
    refuse_test_material,
    sha256_of,
)
from research_agent.eval.metrics import (
    DEFAULT_SCORER,
    SCORERS,
    TYPE_CHECKS,
    cluster_keys,
    compare_systems,
    holm,
    score_items,
    stratified_noninferiority_p,
    summarize,
)
from research_agent.eval.report import SUBSETS, in_subset
from research_agent.eval.resources import (
    GpuError,
    GpuSample,
    GpuSampler,
    cloud_cost,
    e2e_step_latency,
    gpu_sample,
    latency,
    server_ms,
    server_resources,
    single_step_latency,
)
from research_agent.eval.runner import OK, run_items, write_results
from research_agent.eval.systems import (
    CLOUD_MODELS,
    DEV_SET_V2,
    REQUESTS_FILE,
    STAGE_CAPS,
    TRAINING_SET,
    Built,
    SystemUnavailable,
    block_hash,
    build_system,
    demonstrations_block,
    load_demonstrations,
    routing_dev_sets,
    with_votes,
)
from research_agent.eval.timing import ItemTimer, TaskTimer, TimedModel, write_timing
from research_agent.family.core import GREEDY, LoadedFamily, Mode, load_family
from research_agent.llm.budget import Budget, load_prices
from research_agent.llm.generation import GenerationSettings
from research_agent.llm.local import LocalClient
from research_agent.llm.server import (
    HOST,
    SERVER_VAR,
    LocalServer,
    NotReady,
    PortBusy,
    llama_server,
    same_path,
)
from research_agent.net.fetch import utc_now
from research_agent.tools.corpus_search import MANIFEST
from research_agent.tools.schemas import CONTEXT_TOKENS, SCHEMA_HASH
from research_agent.train.manifest import REPO, git_commit, git_dirty

# the systems a session may run, in the order it runs them; ft2 and ft2r are the M6 aliases, ft3 and ft3r,
# final and finalg the M7 ones (module docstring)
RUN_ORDER = ("ft", "b0", "b1", "ftr", "ft2", "ft2r", "ft3", "ft3r", "final", "finalg", "b4", "b2")
# the adapter of the server a local system uses; ftr's local model is ft on the FT server, ft2r's ft2 on the
# server with the ft2 adapter, ft3r's ft3 on the server with the ft3 adapter; final and finalg share the
# server with the final adapter
SERVER_OF = {"ft": "ft", "b0": "ft", "b1": "ft", "ftr": "ft", "ft2": "ft2", "ft2r": "ft2", "ft3": "ft3",
             "ft3r": "ft3", "final": "final", "finalg": "final", "b4": "b4"}
ADAPTERS = ("ft", "ft2", "ft3", "final", "b4")
# an alias and the system of eval.systems it is built as
BUILT_AS = {"ft2": "ft", "ft2r": "ftr", "ft3": "ft", "ft3r": "ftr", "final": "ft", "finalg": "ft"}
# M7's final local system and its greedy self (module docstring): the systems held against the frozen file,
# which run every development item in a dry run
FINAL, FINALG = "final", "finalg"
FROZEN_SYSTEMS = (FINAL, FINALG)
# a system's options in a configuration whose systems are an object (module docstring: A system's options)
VOTES, VOTE_MODE, RULED_KEY, GLOSSARY, FROZEN, DECODING, END_TO_END = (
    "votes", "vote_mode", "rules", "glossary", "frozen", "decoding", "end_to_end")
SYSTEM_KEYS = (VOTES, VOTE_MODE, RULED_KEY, GLOSSARY, FROZEN, DECODING, END_TO_END)
WRAPPING = (VOTES, VOTE_MODE, RULED_KEY, GLOSSARY, DECODING)  # a local system's that is not routed
FILE_KEYS = ("path", "sha256")  # a glossary's and the frozen file's
SYSTEMS_FORM = ("systems must be a non-empty list of system names, or an object from each system's name to "
                "its options")
SCORERS_KEY, SCORER = "scorers", "scorer"  # the configuration's other scorers; a comparison row's
# a routed system and its (local, cloud) systems; the cloud system's model (eval.systems.CLOUD_MODELS) is the
# one its routing file must name: deepseek-flash for each, R3's fallback included (docs/m7-计划.md WP-K)
ROUTED = {"ftr": ("ft", "b2"), "ft2r": ("ft2", "b2"), "ft3r": ("ft3", "b2")}
RULED = ("ft2r", "ft3r")  # the routed systems whose routing file must carry the M6 policy fields
SPENDING = ("ftr", "ft2r", "ft3r", "b2")  # the systems that call DeepSeek, each within b2.max_cost_rmb
SPENDER_NAMES = {"ftr": "FT+R", "ft2r": "FT2+R2", "ft3r": "FT3+R3", "b2": "B2"}
STICKY, STEPWISE = "sticky", "stepwise"  # e2e_routing: how a routed system routes end to end
E2E_ROUTING = (STICKY, STEPWISE)
DRY_RUN = "dry-run"  # the dry run's folder under runs/test-<m>/
# test_sets: the family of test sets a session opens (eval.lock.TEST_FAMILIES), v1 unless the configuration
# names another
DEFAULT_FAMILY = "v1"
DEFAULT_SETS = TEST_FAMILIES[DEFAULT_FAMILY]
# the development set of each family of test sets, in order (module docstring): what its dry run runs and what
# a routing file must have been chosen on; v2's is dev and dev-v2 jointly (docs/m7-计划.md §5 开发集口径)
FAMILY_DEV_SETS = {"v1": (DEV_SET,), "v2": (DEV_SET, DEV_SET_V2)}
LLM_DIR = Path("runs/llm")
RESULTS = Path("results")
CASSETTES = "data/cassettes"
REQUESTS_MANIFEST = "data/synth/requests-manifest.json"
DOC = "_doc"
# One slot (-np 1), checked on /props: a request's output then does not depend on the requests before it. On
# llama-server's default four slots sharing one KV cache, a request sent to another slot than the last one
# runs beside the cells that slot left behind, and its greedy output can change: the dry run of 2026-10-16
# gave b1 another answer on the first development item after ft's and b0's requests (docs/m3-进度.md).
SLOTS = 1
SCHEMA = {  # every key a configuration must have; an object's keys, or None for a value checked on its own
    "milestone": None, "purpose": None, "systems": None, "adapters": None,
    "server": ("model", "port", "ctx_size", "timeout_s", "llama_cpp_build"), "tokenizer": None,
    "b2": ("max_cost_rmb",), "e2e": ("seed", "today"), "statistics": ("seed", "resamples"),
    "comparisons": None, "gpu": ("idle_max_mib", "sample_ms"),
}
OPTIONAL = {  # the keys a configuration may have, as SCHEMA; when each is required, the module docstring
    "b1_block_sha256_16": None, "routing": ("path", "sha256"), "reference": ("milestone", "systems"),
    "e2e_routing": None, "test_sets": None, SCORERS_KEY: None,
}
ADAPTER_KEYS = ("gguf", "sha256", "selection")
# an adapter's optional keys (module docstring): its server's base model and its tokenizer folder, both paths,
# and the generation of the systems on its server (the family path)
ADAPTER_PATHS = ("model", "tokenizer")
GENERATION = "generation"
ADAPTER_OPTIONAL = (*ADAPTER_PATHS, GENERATION)
GENERATION_KEYS = ("mode", "decoding", "reasoning_budget")  # a generation's; a sampled one also has BASE_SEED
BASE_SEED = "base_seed"
GREEDY_DECODING, SAMPLED = "greedy", "sampled"
DECODINGS = (GREEDY_DECODING, SAMPLED)  # a generation's decoding: greedy, or the family's sampling of a mode
# what a thinking server's context holds beyond the prompt's limit (CONTEXT_TOKENS) and the reasoning budget
# R: -c >= 4096 + R + 520 (docs/m7-计划.md §2 端到端预算)
THINK_RESERVE = 520
FT_GENERATION = "ft_generation"  # a routing file's FT generation (scripts/route_curve.py)
REFERENCE_KEYS = ("results", "records", "records_sha256", "dev_records", "dev_records_sha256")
REFERENCE_MILESTONE = "milestone"  # a reference system's own milestone, when not the block's
COMPARISON_KEYS = ("name", "a", "b")  # a tested comparison's, which may add MARGIN and TEST
MARGIN, TEST, MCNEMAR = "margin", "test", "mcnemar"
DEFERRED_KEYS = ("name", "a", "b", "deferred", "p")
RECORDED_KEYS = ("name", "a", "b", "recorded")
NONINFERIORITY_KEYS = ("name", "a", "b", "noninferiority")
ACEBENCH_KEYS = ("name", "acebench")
ACEBENCH_BLOCK = ("a", "b", "margin")
DESCRIPTIVE_KEYS = ("name", "a", "b", "descriptive")
KIND_KEYS = {"deferred": DEFERRED_KEYS, "recorded": RECORDED_KEYS, "noninferiority": NONINFERIORITY_KEYS,
             "acebench": ACEBENCH_KEYS, "descriptive": DESCRIPTIVE_KEYS, "tested": COMPARISON_KEYS}
# the optional keys of a row by its kind: a row comparing two systems' single-step successes may name its
# scorer
KIND_EXTRA = {"tested": (MARGIN, TEST, SCORER), "noninferiority": (SCORER,), "descriptive": (SCORER,)}
HEX64, HEX16 = re.compile(r"[0-9a-f]{64}"), re.compile(r"[0-9a-f]{16}")
NAME = re.compile(r"[A-Za-z0-9_-]+")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
COMPLETE, BUDGET_STOPPED, ABANDONED = "complete", "budget_stop", "abandoned"
DONE, MALFORMED, USAGE, CAPPED, REFUSED, UNFINISHED, FAILED = range(7)
FAMILY = ("the preregistered primary comparisons of the configuration, in its order, adjusted together by "
          "Holm's method: a McNemar comparison with McNemar's p, a non-inferiority comparison with the "
          "one-sided paired bootstrap's p_NI (docs/m6-计划.md §5, H2), the ACEBench comparison with the "
          "stratified bootstrap's p_NI (H3), a deferred comparison with its p and a recorded one with the "
          "p an earlier milestone's comparisons file holds for it (the M3 plan, default 2); a descriptive "
          "comparison is outside the family and has no p; the intervals are not adjusted, and the "
          "non-inferiority bound beside a McNemar comparison with a margin is descriptive")
NOTES = {
    "latency": "p50 and p95 in milliseconds by linear interpolation, with the n they are over: a local "
               "system per model call that came back and per end-to-end step, a cloud system over its paid "
               "calls and the steps whose calls were all paid. Local latency is an upper bound: every prompt "
               "is evaluated from scratch (cache_prompt false). server: llama.cpp's own prompt plus "
               "generation time.",
    "servers": "vram_mib: the peak GPU memory in use while the server ran minus the idle baseline sampled "
               "before it started (nvidia-smi every interval_ms; Windows reports no memory per process); "
               "power over the same samples, energy as the sum of the draws times the interval.",
    "cost": "RMB at the price list (an upper bound): paid_rmb what the paid turns cost, list_rmb what every "
            "turn would cost, those served from the cache included, ledger_rmb what the spending ledger "
            "booked during the runs, per_task_fen the paid cost per end-to-end task.",
    "routed": "A routed system (ftr, ft2r, ft3r) single step: local over the calls of the items its local "
              "model answered "
              "(with llama.cpp's time), routed over the calls of the items it routed and routed_paid over "
              "those that were paid; a routed item's call is the local attempt and the cloud's answer "
              "together. Its cost is over the routed records only; end to end the paid cost of the tasks, "
              "and no list price, since a task's tokens mix local and cloud turns. Its e2e_step is over "
              "every step whose calls came back, local and cloud alike.",
}


class ConfigError(ValueError):
    """A configuration the session cannot run with (exit 1)."""


class Missing(RuntimeError):
    """A prerequisite missing on this machine, or a usage error (exit 2)."""


class Refused(RuntimeError):
    """A preflight check that refused (exit 4)."""


class Unfinished(RuntimeError):
    """An open of the milestone without a completion marker (exit 5)."""


class Failed(RuntimeError):
    """The session failed after it began (exit 6)."""


# ---- the configuration ------------------------------------------------------------------------------------


def _object(node: object, keys: Sequence[str], where: str, *, extra: Sequence[str] = ()) -> dict:
    if not isinstance(node, dict):
        raise ConfigError(f"{where} must be an object")
    missing = [key for key in keys if key not in node]
    if missing:
        raise ConfigError(f"{where} lacks {missing[0]}")
    unknown = [key for key in node if key not in keys and key not in extra]
    if unknown:
        raise ConfigError(f"{where}: unknown key {unknown[0]}")
    return node


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")
    return value


def _path(value: object, where: str) -> str:
    text = _text(value, where)
    windows = PureWindowsPath(text)
    if windows.anchor or PurePosixPath(text).is_absolute() or ".." in windows.parts:
        raise ConfigError(f"{where} must be a path relative to the repository root, not {text!r}")
    return text


def _number(value: object, where: str, *, above: float | None = None, at_least: float | None = None) -> float:
    ok = isinstance(value, int | float) and not isinstance(value, bool)
    if ok and above is not None:
        ok = value > above
    if ok and at_least is not None:
        ok = value >= at_least
    if not ok:
        bound = f" above {above:g}" if above is not None else f" of at least {at_least:g}" if at_least else ""
        raise ConfigError(f"{where} must be a number{bound}, not {value!r}")
    return value


def _integer(value: object, where: str, *, low: int, high: int | None = None) -> int:
    integer = isinstance(value, int) and not isinstance(value, bool)
    if not integer or value < low or (high is not None and value > high):
        span = f"from {low} to {high}" if high is not None else f"of at least {low}"
        raise ConfigError(f"{where} must be an integer {span}, not {value!r}")
    return value


def check_config(config: object, *, freezing: bool = False) -> dict:
    """config when it is a configuration the session can run with (module docstring); ConfigError naming the
    first key that is missing, unknown or of the wrong kind otherwise. The keys of every object are checked
    before any value. freezing: the frozen file is being written (scripts/freeze_final.py) or not yet
    (scripts/decide_m7.py), so systems.final.frozen.sha256 is not checked (module docstring: A system's
    options)."""
    _object(config, list(SCHEMA), "the configuration", extra=(DOC, *OPTIONAL))
    for key, keys in {**SCHEMA, **OPTIONAL}.items():
        if keys is not None and key in config:
            _object(config[key], keys, key)
    systems = config["systems"]
    if isinstance(systems, dict):
        if not systems:
            raise ConfigError(SYSTEMS_FORM)
        for system, options in systems.items():
            _object(options, (), f"systems.{system}", extra=SYSTEM_KEYS)
            for key in (FROZEN, GLOSSARY):
                if key in options and (key == FROZEN or options[key] is not None):
                    _object(options[key], FILE_KEYS, f"systems.{system}.{key}")
    elif not isinstance(systems, list) or not systems or not all(isinstance(s, str) for s in systems):
        raise ConfigError(SYSTEMS_FORM)
    for system in systems:
        if system not in RUN_ORDER:
            raise ConfigError(f"systems: {system} is not one of {list(RUN_ORDER)}")
    if len(set(systems)) != len(systems):
        raise ConfigError("systems names a system twice")
    adapters = _object(config["adapters"], served(systems), "adapters", extra=ADAPTERS)
    for system, entry in adapters.items():
        _object(entry, ADAPTER_KEYS, f"adapters.{system}", extra=ADAPTER_OPTIONAL)
        if GENERATION in entry:
            _object(entry[GENERATION], GENERATION_KEYS, f"adapters.{system}.{GENERATION}", extra=(BASE_SEED,))
    routed = [s for s in systems if s in ROUTED]
    if len(routed) > 1:
        raise ConfigError(f"systems: at most one routed system per session (routing names one routing file), "
                          f"not {routed}")
    if routed and "routing" not in config:
        raise ConfigError(f"routing is required when {routed[0]} runs: the frozen routing configuration and "
                          "its SHA-256")
    if "routing" in config and not routed:
        raise ConfigError(f"routing is only for a routed system ({', '.join(ROUTED)}), none of which this "
                          "session runs")
    if "e2e_routing" in config and not routed:
        raise ConfigError(f"e2e_routing is only for a routed system ({', '.join(ROUTED)}), none of which "
                          "this session runs")
    references: dict = {}
    if "reference" in config:
        references = config["reference"]["systems"]
        if not isinstance(references, dict) or not references:
            raise ConfigError("reference.systems must be a non-empty object")
        for system, entry in references.items():
            if system not in RUN_ORDER:
                raise ConfigError(f"reference.systems: {system} is not one of {list(RUN_ORDER)}")
            if system in systems:
                raise ConfigError(f"reference.systems.{system} is a system this session runs: its scores "
                                  "would come from two places")
            _object(entry, REFERENCE_KEYS, f"reference.systems.{system}", extra=(REFERENCE_MILESTONE,))
    comparisons = config["comparisons"]
    if not isinstance(comparisons, list) or not comparisons:
        raise ConfigError("comparisons must be a non-empty list")
    for number, row in enumerate(comparisons, 1):
        kind = comparison_kind(row)
        _object(row, KIND_KEYS[kind], f"comparisons[{number}]", extra=KIND_EXTRA.get(kind, ()))

    if not isinstance(config["milestone"], str) or config["milestone"] not in STAGE_CAPS:
        raise ConfigError(f"milestone must be one of {sorted(STAGE_CAPS)}, not {config['milestone']!r}")
    family = config.get("test_sets", DEFAULT_FAMILY)
    if not isinstance(family, str) or family not in TEST_FAMILIES:
        raise ConfigError(f"test_sets must be one of {sorted(TEST_FAMILIES)}, not {family!r}")
    if family != DEFAULT_FAMILY and "reference" in config:
        raise ConfigError(f"reference is only for the {DEFAULT_FAMILY} test sets, whose opens the reference "
                          f"records come from, not for test_sets {family}")
    _text(config["purpose"], "purpose")
    for system, entry in adapters.items():
        _path(entry["gguf"], f"adapters.{system}.gguf")
        _hex64(entry["sha256"], f"adapters.{system}.sha256")
        if entry["selection"] is not None:
            _path(entry["selection"], f"adapters.{system}.selection")
        for key in ADAPTER_PATHS:
            if key in entry:
                _path(entry[key], f"adapters.{system}.{key}")
    server = config["server"]
    _path(server["model"], "server.model")
    _integer(server["port"], "server.port", low=1, high=65535)
    _integer(server["ctx_size"], "server.ctx_size", low=1)
    _number(server["timeout_s"], "server.timeout_s", above=0)
    if server["llama_cpp_build"] is not None:
        _text(server["llama_cpp_build"], "server.llama_cpp_build")
    for system, entry in adapters.items():
        if GENERATION in entry:
            check_generation(entry[GENERATION], server["ctx_size"], f"adapters.{system}.{GENERATION}")
    check_options(config, freezing=freezing)
    check_scorers(config)
    _path(config["tokenizer"], "tokenizer")
    block = config.get("b1_block_sha256_16")
    if (block is not None or "b1" in systems) and (not isinstance(block, str) or not HEX16.fullmatch(block)):
        raise ConfigError(f"b1_block_sha256_16 must be 16 lower-case hex digits, not {block!r}")
    if "routing" in config:
        _path(config["routing"]["path"], "routing.path")
        _hex64(config["routing"]["sha256"], "routing.sha256")
    if "e2e_routing" in config and config["e2e_routing"] not in E2E_ROUTING:
        raise ConfigError(f"e2e_routing must be {' or '.join(E2E_ROUTING)}, not {config['e2e_routing']!r}")
    if "reference" in config:
        _earlier_milestone(config["reference"]["milestone"], config["milestone"], "reference.milestone")
        for system, entry in references.items():
            if REFERENCE_MILESTONE in entry:
                _earlier_milestone(entry[REFERENCE_MILESTONE], config["milestone"],
                                   f"reference.systems.{system}.{REFERENCE_MILESTONE}")
            for key in REFERENCE_KEYS:
                check = _hex64 if key.endswith("sha256") else _path
                check(entry[key], f"reference.systems.{system}.{key}")
    _number(config["b2"]["max_cost_rmb"], "b2.max_cost_rmb", above=0)
    _integer(config["e2e"]["seed"], "e2e.seed", low=0)
    today = config["e2e"]["today"]
    if today is not None and not (isinstance(today, str) and DATE.fullmatch(today)):
        raise ConfigError(f"e2e.today must be YYYY-MM-DD or null, not {today!r}")
    _integer(config["statistics"]["seed"], "statistics.seed", low=0)
    _integer(config["statistics"]["resamples"], "statistics.resamples", low=1)
    names = set()
    for number, row in enumerate(comparisons, 1):
        name, where = row["name"], f"comparisons[{number}]"
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in names:
            raise ConfigError(f"{where}.name must be a new name of letters, digits, '-' and '_', not "
                              f"{name!r}")
        names.add(name)
        kind = comparison_kind(row)
        if kind == "acebench":
            block = _object(row["acebench"], ACEBENCH_BLOCK, f"{where}.acebench")
            a, b = _path(block["a"], f"{where}.acebench.a"), _path(block["b"], f"{where}.acebench.b")
            if a == b:
                raise ConfigError(f"comparisons: {name} compares {a} with itself")
            _margin(block["margin"], f"{where}.acebench.margin")
            continue
        a, b = _text(row["a"], f"{where}.a"), _text(row["b"], f"{where}.b")
        if a == b:
            raise ConfigError(f"comparisons: {name} compares {a} with itself")
        if kind == "deferred":
            _text(row["deferred"], f"{where}.deferred")
            _number(row["p"], f"{where}.p", at_least=0)
            if row["p"] > 1:
                raise ConfigError(f"{where}.p must be a p-value, not {row['p']!r}")
            continue
        if kind == "recorded":
            if "reference" not in config:
                raise ConfigError(f"{where} takes its p from an earlier milestone's comparisons: reference "
                                  "(the milestone whose completion markers record that file) is required")
            _path(row["recorded"], f"{where}.recorded")
            continue
        for system in (a, b):
            if system not in systems and system not in references:
                raise ConfigError(f"comparisons: {name} compares {system}, which is not among the systems or "
                                  "the reference systems")
        if a not in systems and b not in systems:
            raise ConfigError(f"comparisons: {name} compares no system this session runs ({a}, {b})")
        if SCORER in row:
            scorer = row[SCORER]
            if not isinstance(scorer, str) or scorer not in SCORERS:
                raise ConfigError(f"{where}.{SCORER} must be one of {list(SCORERS)}, not {scorer!r}")
            if scorer != DEFAULT_SCORER and scorer not in (config.get(SCORERS_KEY) or []):
                raise ConfigError(f"{where}.{SCORER} {scorer}: the session scores its systems so only when "
                                  f"{SCORERS_KEY} lists it")
        if kind == "noninferiority":
            block = _object(row["noninferiority"], (MARGIN,), f"{where}.noninferiority")
            _margin(block[MARGIN], f"{where}.noninferiority.margin")
        elif kind == "descriptive":
            if row["descriptive"] is not True:
                raise ConfigError(f"{where}.descriptive must be true, not {row['descriptive']!r}")
        else:
            if TEST in row and row[TEST] != MCNEMAR:
                raise ConfigError(f"{where}.test must be {MCNEMAR}, not {row[TEST]!r}")
            if MARGIN in row:
                _margin(row[MARGIN], f"{where}.margin")
    _number(config["gpu"]["idle_max_mib"], "gpu.idle_max_mib", at_least=0)
    _integer(config["gpu"]["sample_ms"], "gpu.sample_ms", low=1)
    return config


def check_generation(block: dict, ctx_size: int, where: str) -> dict:
    """block, an adapter's generation (module docstring: the family path), when a server of ctx_size tokens
    can run it whatever the family; ConfigError naming what is wrong otherwise. Its keys are checked
    already."""
    modes = [str(mode) for mode in Mode]
    if block["mode"] not in modes:
        raise ConfigError(f"{where}.mode must be {' or '.join(modes)}, not {block['mode']!r}")
    if block["decoding"] not in DECODINGS:
        raise ConfigError(f"{where}.decoding must be {' or '.join(DECODINGS)}, not {block['decoding']!r}")
    budget = _integer(block["reasoning_budget"], f"{where}.reasoning_budget", low=1)
    if block["decoding"] == SAMPLED:
        if BASE_SEED not in block:
            raise ConfigError(f"{where} lacks {BASE_SEED}: a sampled decoding draws every request's seed "
                              "from it (research_agent.llm.generation)")
        _integer(block[BASE_SEED], f"{where}.{BASE_SEED}", low=0)
    elif BASE_SEED in block:
        raise ConfigError(f"{where}.{BASE_SEED}: a greedy decoding sends no seed; only a sampled one has a "
                          "base seed")
    need = CONTEXT_TOKENS + budget + THINK_RESERVE
    if block["mode"] == str(Mode.THINK) and ctx_size < need:
        raise ConfigError(f"{where}: server.ctx_size {ctx_size} is below {need}, the {CONTEXT_TOKENS} tokens "
                          f"of the prompt's limit, the reasoning budget {budget} and {THINK_RESERVE} more, "
                          "which a thinking server's context must hold (docs/m7-计划.md §2 端到端预算)")
    return block


def check_options(config: dict, *, freezing: bool = False) -> None:
    """The systems' options (module docstring: A system's options), their keys checked already; ConfigError
    naming the first that is wrong. final and finalg run on the family path: adapters.final needs a
    generation. finalg is the adapter's generation decoded greedily without the inference wrappers, held
    against final's frozen file, so it runs only beside final."""
    for system in FROZEN_SYSTEMS:
        if system in config["systems"] and GENERATION not in config["adapters"][SERVER_OF[system]]:
            raise ConfigError(f"adapters.{SERVER_OF[system]} lacks {GENERATION}: {FINAL} and {FINALG} run on "
                              "the family path, whose generation the frozen file records")
    if FINALG in config["systems"] and FINAL not in config["systems"]:
        raise ConfigError(f"systems.{FINALG}: {FINALG} is held against {FINAL}'s frozen file, so {FINAL} "
                          "runs too")
    for system in config["systems"]:
        options, where = options_of(config, system), f"systems.{system}"
        if system == FINALG:
            wrapped = [key for key in (VOTES, VOTE_MODE, RULED_KEY, GLOSSARY) if key in options]
            if wrapped:
                raise ConfigError(f"{where}.{wrapped[0]}: {FINALG} answers greedily without the inference "
                                  "wrappers")
            if DECODING not in options:
                raise ConfigError(f"{where} lacks {DECODING}: {FINALG} is the adapter's generation decoded "
                                  "greedily")
        wrapping = [key for key in WRAPPING if key in options]
        if wrapping and (system not in SERVER_OF or system in ROUTED):
            raise ConfigError(f"{where}.{wrapping[0]}: the inference wrappers and the decoding are a local "
                              f"system's that is not routed, not {system}'s")
        if VOTES in options:
            _integer(options[VOTES], f"{where}.{VOTES}", low=1)
        if VOTE_MODE in options:
            if VOTES not in options:
                raise ConfigError(f"{where}.{VOTE_MODE} chooses the keys of a vote: it needs {VOTES}")
            if options[VOTE_MODE] not in VOTE_MODES:
                raise ConfigError(f"{where}.{VOTE_MODE} must be {' or '.join(VOTE_MODES)}, not "
                                  f"{options[VOTE_MODE]!r}")
        for key in (RULED_KEY, END_TO_END):
            if key in options and not isinstance(options[key], bool):
                raise ConfigError(f"{where}.{key} must be true or false, not {options[key]!r}")
        if options.get(GLOSSARY) is not None:
            _path(options[GLOSSARY]["path"], f"{where}.{GLOSSARY}.path")
            _hex64(options[GLOSSARY]["sha256"], f"{where}.{GLOSSARY}.sha256")
        if DECODING in options and options[DECODING] != GREEDY_DECODING:
            raise ConfigError(f"{where}.{DECODING} must be {GREEDY_DECODING}, not {options[DECODING]!r}: the "
                              "adapter's generation decoded greedily")
        if VOTES in options or DECODING in options:
            adapter = SERVER_OF[system]
            generation = config["adapters"][adapter].get(GENERATION)
            if generation is None:
                raise ConfigError(f"{where}: {VOTES} and {DECODING} change the generation of "
                                  f"adapters.{adapter}, which names none (the family path)")
            if VOTES in options and DECODING in options:
                raise ConfigError(f"{where}: {VOTES} with {DECODING} {GREEDY_DECODING}: a vote's samples "
                                  "sample (its greedy candidate is the vote's own)")
            if VOTES in options and generation["decoding"] != SAMPLED:
                raise ConfigError(f"{where}.{VOTES}: a vote's samples sample, but "
                                  f"adapters.{adapter}.{GENERATION} decodes greedily")
        if FROZEN in options:
            if system != FINAL:
                raise ConfigError(f"{where}.{FROZEN}: only {FINAL} is frozen")
            _path(options[FROZEN]["path"], f"{where}.{FROZEN}.path")
            if not freezing:
                _hex64(options[FROZEN]["sha256"], f"{where}.{FROZEN}.sha256")
        elif system == FINAL:
            raise ConfigError(f"{where} lacks {FROZEN}: {FINAL} is held against the file "
                              "scripts/freeze_final.py writes, its path and SHA-256")


def check_scorers(config: dict) -> None:
    """ConfigError unless the configuration's scorers, when it has them, are distinct scorers of
    eval.metrics.SCORERS other than the default one."""
    if SCORERS_KEY not in config:
        return
    scorers, others = config[SCORERS_KEY], [s for s in SCORERS if s != DEFAULT_SCORER]
    if not (isinstance(scorers, list) and scorers and all(isinstance(s, str) and s in others for s in scorers)
            and len(set(scorers)) == len(scorers)):
        raise ConfigError(f"{SCORERS_KEY} must be a non-empty list of distinct scorers other than "
                          f"{DEFAULT_SCORER} ({', '.join(others)}), not {scorers!r}")


def options_of(config: dict, system: str) -> dict:
    """A system's options: its entry when the configuration's systems are an object, else none."""
    systems = config["systems"]
    return (systems.get(system) or {}) if isinstance(systems, dict) else {}


def system_generation(config: dict, system: str, settings: GenerationSettings) -> GenerationSettings:
    """A system's generation on its server's family path (module docstring: A system's options): settings, the
    adapter's, decoded greedily under its decoding and with its votes (eval.systems.with_votes)."""
    options = options_of(config, system)
    if options.get(DECODING) == GREEDY_DECODING:
        settings = dataclasses.replace(settings, decoding=GREEDY)
    return with_votes(settings, options.get(VOTES))


def wrappers_of(config: dict, system: str) -> dict:
    """build_system's inference wrappers of a system (module docstring: A system's options): the runtime
    rules, the glossary file and the vote mode; build_system's defaults for a system without them."""
    options = options_of(config, system)
    glossary = options.get(GLOSSARY)
    return {"rules": options.get(RULED_KEY, False),
            "glossary": None if glossary is None else Path(glossary["path"]),
            "vote_mode": options.get(VOTE_MODE, VOTE_FINAL)}


def frozen_identity(config: dict, family: LoadedFamily, settings: GenerationSettings) -> dict:
    """What the frozen file records of the final system as the configuration gives it now, field by field
    (module docstring: The final system's frozen file; the preflight compares them): the final adapter's GGUF
    and SHA-256, its base model with the SHA-256 of the file, its tokenizer folder, the family's record,
    final's generation (system_generation of settings, the adapter's, with the votes) and its votes, the vote
    mode (None without a vote), the runtime rules of this code (None without them), the glossary with the
    SHA-256 of its file (None without one) and the family's development sets with the SHA-256 of their
    files."""
    adapter, options = config["adapters"][SERVER_OF[FINAL]], options_of(config, FINAL)
    base, glossary = base_model(config, SERVER_OF[FINAL]), options.get(GLOSSARY)
    final = system_generation(config, FINAL, settings)
    return {"adapter": {"path": adapter["gguf"], "sha256": adapter["sha256"]},
            "base": {"path": base, "sha256": sha256_of(base)},
            "tokenizer": adapter_tokenizer(config, SERVER_OF[FINAL]), "family": family.record(),
            "generation": final.record(), "votes": final.votes,
            "vote_mode": options.get(VOTE_MODE, VOTE_FINAL) if final.votes else None,
            "rules": {"version": RULES_VERSION, "names": list(RULES)} if options.get(RULED_KEY) else None,
            "glossary": None if glossary is None else {"path": glossary["path"],
                                                       "sha256": sha256_of(glossary["path"])},
            "dev_sets": [{"path": dev.as_posix(), "sha256": sha256_of(dev)} for dev in dev_sets_of(config)]}


def output_hash(record: dict) -> str:
    """The SHA-256 (hex) of what a runner record says the model answered (module docstring: The final system's
    frozen file): the canonical JSON of its status and, for an ok record, its text, its calls by name with
    their parsed arguments (the text of arguments that are no JSON) and its reasoning; nothing else, so
    neither the timing, the usage, the signals, the vote's or the route's bookkeeping nor the call ids change
    it."""
    answer: dict = {"status": record.get("status")}
    if record.get("status") == OK:
        output = record.get("output") if isinstance(record.get("output"), dict) else {}
        answer.update(content=output.get("content") or "", reasoning=record.get("reasoning"),
                      calls=[_answered(call) for call in output.get("tool_calls") or []])
    return hashlib.sha256(canonical_json(answer).encode("utf-8")).hexdigest()


def _answered(call: object) -> dict:
    """A call of an answer as output_hash reads it: its name and its parsed arguments."""
    function = (call.get("function") or {}) if isinstance(call, dict) else {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {"name": function.get("name"), "arguments_text": arguments}
    return {"name": function.get("name"), "arguments": arguments}


def wrapper_counts(turns: Iterable[dict]) -> dict:
    """The votes and the rules of a voted or ruled system's turns (runner records, or the attempts of its
    transcripts), counted (module docstring: outputs; descriptive): "votes": the steps voted, those whose
    chosen class every sample held, the mean share of the samples in the chosen class over the steps some
    sample voted on (agreement), the steps whose chosen class is not the greedy answer's, those no sample
    voted on and the candidates that failed (the vote's "errors"); "rules": the steps ruled and per rule how
    many answers it changed. Each only when some turn carries it."""
    turns = list(turns)
    votes = [turn["votes"] for turn in turns if isinstance(turn.get("votes"), dict)]
    rules = [turn["rules"] for turn in turns if isinstance(turn.get("rules"), list)]
    out: dict = {}
    if votes:
        decided = [v for v in votes if v.get("chosen") is not None]
        shares = [v["counts"][v["chosen"]] / v["k"] for v in decided]
        out["votes"] = {"steps": len(votes),
                        "unanimous": sum(v["counts"][v["chosen"]] == v["k"] for v in decided),
                        "agreement": round(sum(shares) / len(shares), 6) if shares else None,
                        "chosen_not_greedy": sum(v["chosen"] != v.get("greedy") for v in decided),
                        "no_sample_voted": len(votes) - len(decided),
                        "candidate_errors": sum(_how_many(v.get("errors")) for v in votes)}
    if rules:
        fired = Counter(change.get("rule") for changes in rules for change in changes
                        if isinstance(change, dict))
        out["rules"] = {"steps": len(rules), "fired": {name: fired[name] for name in RULES}}
    return out


def _how_many(errors: object) -> int:
    """The failed candidates a vote's "errors" names: a list or an object of them, or their number."""
    if isinstance(errors, list | dict):
        return len(errors)
    return errors if isinstance(errors, int) and not isinstance(errors, bool) else 0


def _margin(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 1:
        raise ConfigError(f"{where} must be a number above 0 and below 1, not {value!r}")
    return value


def _earlier_milestone(value: object, milestone: str, where: str) -> str:
    before = list(STAGE_CAPS)
    known = isinstance(value, str) and value in STAGE_CAPS
    if not known or before.index(value) >= before.index(milestone):
        raise ConfigError(f"{where} must be a milestone before {milestone}, not {value!r}")
    return value


def comparison_kind(row: object) -> str:
    """deferred, recorded, acebench, noninferiority, descriptive or tested (McNemar): a comparison row's
    kind, by the key that marks it."""
    if isinstance(row, dict):
        for kind in ("deferred", "recorded", "acebench", "noninferiority", "descriptive"):
            if kind in row:
                return kind
    return "tested"


def routed_system(config: dict) -> str | None:
    """The routed system the session runs, if any (check_config allows one at most)."""
    return next((s for s in config["systems"] if s in ROUTED), None)


def family_of(config: dict) -> TestFamily:
    """The family of test sets the session opens (test_sets, eval.lock.TEST_FAMILIES): v1 unless the
    configuration names another."""
    return TEST_FAMILIES[config.get("test_sets", DEFAULT_FAMILY)]


def dev_sets_of(config: dict) -> tuple[Path, ...]:
    """The development sets of the family the session opens, in order (FAMILY_DEV_SETS)."""
    return FAMILY_DEV_SETS[config.get("test_sets", DEFAULT_FAMILY)]


def served(systems: Sequence[str]) -> list[str]:
    """The adapters of the servers a session of these systems starts, in their order (ADAPTERS)."""
    return [name for name in ADAPTERS if any(SERVER_OF.get(s) == name for s in systems)]


def base_model(config: dict, adapter: str) -> str:
    """The base model the server with this adapter loads: the adapter's model, else server.model."""
    return config["adapters"][adapter].get("model", config["server"]["model"])


def adapter_tokenizer(config: dict, adapter: str) -> str:
    """The tokenizer folder of the prompts of the systems on an adapter's server: the adapter's tokenizer,
    else the configuration's."""
    return config["adapters"][adapter].get("tokenizer", config["tokenizer"])


def tokenizer_of(config: dict, system: str) -> str:
    """The tokenizer folder of a system's prompts and token counts: its server adapter's tokenizer, else the
    configuration's (a cloud system's too)."""
    adapter = SERVER_OF.get(system)
    if adapter is None:
        return config["tokenizer"]
    return adapter_tokenizer(config, adapter)


def tokenizers(config: dict) -> list[str]:
    """Every tokenizer folder whose chat template and token counter the session loads, once: the
    configuration's, then those of the adapters of the servers it starts whose systems run on the legacy path
    (no generation), in their order. A family-path server's folder is loaded as its family (generations)."""
    adapters = config["adapters"]
    folders = [config["tokenizer"], *(adapter_tokenizer(config, name) for name in served(config["systems"])
                                      if GENERATION not in adapters[name])]
    return list(dict.fromkeys(folders))


@dataclass(frozen=True)
class Generated:
    """The family path of the systems on one server (module docstring): the family of its adapter's tokenizer
    folder, their generation settings and the token counter of their context, over the family's prompt in the
    generation's mode."""

    family: LoadedFamily
    settings: GenerationSettings
    count_tokens: TokenCounter


def generation_settings(family: LoadedFamily, block: dict, where: str) -> GenerationSettings:
    """The settings of an adapter's generation (check_generation) on the family (module docstring: the family
    path): its mode, decoded with the family's own decoding of that mode when sampled and greedily otherwise
    (as scripts/run_eval.py's --greedy decodes think mode), its reasoning budget and its base seed (0 for a
    greedy decoding, which sends none). ConfigError for a mode the family does not have and for a sampled
    decoding of a mode the family decodes greedily."""
    try:
        settings = GenerationSettings.for_family(family, mode=block["mode"],
                                                 reasoning_budget=block["reasoning_budget"],
                                                 base_seed=block.get(BASE_SEED, 0))
    except ValueError as exc:  # a mode the family does not have
        raise ConfigError(f"{where}.mode: {exc}") from None
    if block["decoding"] == GREEDY_DECODING:
        return dataclasses.replace(settings, decoding=GREEDY)
    if not settings.samples:
        raise ConfigError(f"{where}.decoding: {SAMPLED}, but {family.spec.name} decodes its {settings.mode} "
                          f"mode greedily (temperature 0): give {GREEDY_DECODING}")
    return settings


def generations(config: dict, load: Callable[[str], LoadedFamily] = load_family) -> dict[str, Generated]:
    """Per adapter of a server the session starts whose entry names a generation, in their order, the family
    path of the systems on that server (Generated): the family of the adapter's tokenizer folder (load, once
    per folder), the settings of generation_settings and the token counter of the family's prompt in their
    mode (agent.context.family_token_counter, what scripts/run_eval.py counts on the family path). Missing
    when a folder's family cannot be loaded; ConfigError from generation_settings."""
    families: dict[str, LoadedFamily] = {}
    out = {}
    for name in served(config["systems"]):
        block = config["adapters"][name].get(GENERATION)
        if block is None:
            continue
        folder = adapter_tokenizer(config, name)
        if folder not in families:
            try:
                families[folder] = load(folder)
            except Exception as exc:  # noqa: BLE001  whatever it is (no registered family...), nothing has run
                where = (f"adapters.{name}.tokenizer" if "tokenizer" in config["adapters"][name]
                         else "tokenizer")
                raise Missing(f"the family of {folder} ({where}) cannot be loaded, and the systems on the "
                              f"{name} server run on the family path ({GENERATION}): {type(exc).__name__}: "
                              f"{exc}") from None
        family = families[folder]
        settings = generation_settings(family, block, f"adapters.{name}.{GENERATION}")
        out[name] = Generated(family, settings, family_token_counter(family, settings.mode))
    return out


def cloud_of(system: str) -> str:
    """The cloud system whose model a spending system calls: a routed system's (ROUTED), else the system."""
    return ROUTED[system][1] if system in ROUTED else system


def reference_milestone(config: dict, system: str) -> str:
    """The milestone a reference system's records come from: its own, else the reference block's."""
    return config["reference"]["systems"][system].get(REFERENCE_MILESTONE) or config["reference"]["milestone"]


def _hex64(value: object, where: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ConfigError(f"{where} must be 64 lower-case hex digits, not {value!r}")
    return value


def load_config(path: Path, *, freezing: bool = False) -> dict:
    """The configuration at path, checked (check_config, with freezing)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{Path(path).as_posix()} cannot be read: {exc}") from None
    except ValueError as exc:
        raise ConfigError(f"{Path(path).as_posix()} is not valid JSON: {exc}") from None
    return check_config(data, freezing=freezing)


# ---- git and the completion markers -----------------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def committed(path: str) -> bool:
    """Whether path (relative to the repository root) is in HEAD and unchanged in the working tree."""
    tree, status = _git("ls-tree", "HEAD", "--", path), _git("status", "--porcelain", "--", path)
    return (tree is not None and tree.returncode == 0 and bool(tree.stdout.strip())
            and status is not None and status.returncode == 0 and not status.stdout.strip())


def markers(family: TestFamily = DEFAULT_SETS) -> list[dict]:
    return read_jsonl(family.completions) if family.completions.exists() else []


def unfinished(milestone: str, family: TestFamily = DEFAULT_SETS) -> list[tuple[Path, dict]]:
    """(access log, entry) of every open of the milestone in the family's access logs that has no completion
    marker in the family's markers, in log order.

    Refused when a marker names an open its access log no longer holds (module docstring)."""
    logged = {log: LockedSet(test_set, log).log_entries() for test_set, log in family.sets}
    held = {canonical_json([log.as_posix(), entry]) for log, entries in logged.items() for entry in entries}
    closed = set()
    for marker in markers(family):
        key = canonical_json([marker.get("access_log"), marker.get("open")])
        if key not in held:
            opened = marker.get("open") if isinstance(marker.get("open"), dict) else {}
            raise Refused(f"{family.completions.as_posix()} marks the open of {opened.get('ts')} in "
                          f"{marker.get('access_log')}, which that access log no longer holds: restore the "
                          "access logs from git before any session or --abandon")
        closed.add(key)
    pending = []
    for log, entries in logged.items():
        for entry in entries:
            if entry.get("milestone") == milestone and canonical_json([log.as_posix(), entry]) not in closed:
                pending.append((log, entry))
    return pending


def to_commit(family: TestFamily = DEFAULT_SETS) -> str:
    """What a session's end or an --abandon leaves to commit, together: the family's two access logs and its
    markers."""
    (_, single), (_, e2e) = family.sets
    return f"{single.as_posix()}, {e2e.as_posix()} and {family.completions.as_posix()}"


def mark(opens: list[tuple[Path, dict]], status: str, now: Callable[[], str],
         family: TestFamily = DEFAULT_SETS, **fields: object) -> None:
    """One completion marker per open, appended to the family's markers (canonical JSON lines; v1:
    data/eval/test-completions.jsonl)."""
    family.completions.parent.mkdir(parents=True, exist_ok=True)
    with family.completions.open("a", encoding="utf-8", newline="\n") as f:
        for log, entry in opens:
            f.write(canonical_json({"access_log": log.as_posix(), "open": entry, "status": status,
                                    "recorded": now(), **fields}) + "\n")


def describe_unfinished(pending: list[tuple[Path, dict]], milestone: str, config: str,
                        family: TestFamily = DEFAULT_SETS) -> str:
    lines = [f"{log.as_posix()}: the {milestone} open of {entry.get('ts')} has no completion marker "
             f"(purpose: {entry.get('purpose')})" for log, entry in pending]
    advice = ("That session did not reach its end. A crashed session is not resumed (the M3 plan, default "
              "5): its items were in memory only and its open is used up. The owner decides what comes next; "
              f"to record the decision and allow a new open, run scripts/run_test.py --config {config} "
              f"--abandon \"<reason>\" and commit {to_commit(family)}.")
    return "\n".join([*lines, advice])


def abandon(config: dict, reason: str, now: Callable[[], str]) -> int:
    milestone, family = config["milestone"], family_of(config)
    try:
        pending = unfinished(milestone, family)
    except (LockError, JsonlError) as exc:
        raise Refused(f"the access logs or the completion markers cannot be read: {exc}") from None
    if not pending:
        print(f"nothing to abandon: every {milestone} open has its completion marker")
        return DONE
    mark(pending, ABANDONED, now, family, reason=reason)
    for log, entry in pending:
        print(f"{log.as_posix()}: the {milestone} open of {entry.get('ts')} is marked abandoned in "
              f"{family.completions.as_posix()}")
    print(f"commit {to_commit(family)}; the next session makes a new open")
    return DONE


# ---- before the opening -----------------------------------------------------------------------------------


@dataclass
class Machine:
    """What the session uses of this machine; main's keyword arguments, which the tests replace."""

    popen: Callable  # llama-server, and nvidia-smi's sampling loop
    run: Callable  # nvidia-smi's one-off query
    client: Callable[[str], LocalClient]
    openai_client: object | None
    load_render: Callable[[str], Callable]
    load_counter: Callable[[str], TokenCounter | None]
    load_world: Callable[[], World]
    now: Callable[[], str]  # the opens' and the markers' timestamps
    clock: Callable[[], float]  # the timing sidecars'
    load_family: Callable[[str], LoadedFamily] = load_family  # a family-path server's (generations)


def training_renderer(tokenizer: str) -> Callable:
    """The local prompt as training rendered it (train.samples.prompt_renderer), over the model folder's
    chat template, loaded once for every local run."""
    from transformers import AutoTokenizer  # heavy import, only for a real session

    from research_agent.train.samples import prompt_renderer

    return prompt_renderer(AutoTokenizer.from_pretrained(tokenizer))


def token_counter(tokenizer: str) -> TokenCounter:
    from research_agent.agent.context import qwen_token_counter  # imports transformers: only here

    return qwen_token_counter(tokenizer)


def recorded_world() -> World:
    return build_world(cassettes=CASSETTES, corpus_manifest=MANIFEST, requests_manifest=REQUESTS_MANIFEST)


def check_root() -> None:
    """Missing unless the current folder is the repository root, where the sets, logs and results are."""
    if Path.cwd().resolve() != Path(REPO).resolve():
        raise Missing(f"run from the repository root ({REPO}): the sets, logs and results are taken from "
                      "there")


def prerequisites(config: dict, llama: str | None) -> Path:
    """The llama-server binary, when everything a session needs on this machine is here; Missing otherwise."""
    check_root()
    try:
        binary = llama_server(llama)
    except LookupError as exc:
        raise Missing(str(exc)) from None
    if not binary.is_file():
        raise Missing(f"llama-server not found: {binary} (--llama-server or {SERVER_VAR})")
    if not Path(config["server"]["model"]).is_file():
        raise Missing(f"the base model {config['server']['model']} is not here (server.model)")
    if not Path(config["tokenizer"]).is_dir():
        raise Missing(f"the tokenizer folder {config['tokenizer']} is not here (tokenizer)")
    for name in served(config["systems"]):
        entry = config["adapters"][name]
        if "model" in entry and not Path(entry["model"]).is_file():
            raise Missing(f"the base model {entry['model']} of the {name} server is not here "
                          f"(adapters.{name}.model)")
        if "tokenizer" in entry and not Path(entry["tokenizer"]).is_dir():
            raise Missing(f"the tokenizer folder {entry['tokenizer']} of the {name} server is not here "
                          f"(adapters.{name}.tokenizer)")
    if "b1" in config["systems"]:
        for path in (TRAINING_SET, REQUESTS_FILE):
            if not path.is_file():
                raise Missing(f"b1: its demonstrations come from {path.as_posix()}, which is not here")
    for path in dev_sets_of(config):
        if not path.is_file():
            family = config.get("test_sets", DEFAULT_FAMILY)
            raise Missing(f"the development set {path.as_posix()} of the {family} test sets is not here: the "
                          "dry run runs it and a routing file must have been chosen on it")
    return binary


@dataclass(frozen=True)
class Frozen:
    """The final system's frozen file as the preflight read it (check_frozen): its path, its SHA-256 and what
    it holds."""

    path: str
    sha256: str
    data: dict


@dataclass
class Ready:
    """What the preflight established."""

    config_path: str  # relative to the repository root
    config_sha256: str
    commit: str
    purpose: str
    folder: Path  # the session's folder under runs/
    recorded: dict[str, float] = field(default_factory=dict)  # a recorded comparison's p, by its name
    # the routed system's: the routed count and the successes of the routing file's chosen point
    frozen_point: dict | None = None
    # an ACEBench comparison's two runs read answer by answer, by the comparison's name
    acebench: dict[str, tuple[PerItem, PerItem]] = field(default_factory=dict)
    frozen: Frozen | None = None  # the final system's frozen file, when final runs


def check_adapter(config: dict, system: str) -> None:
    entry = config["adapters"][system]
    gguf = Path(entry["gguf"])
    if not gguf.is_file():
        raise Refused(f"the {system} adapter {entry['gguf']} is missing")
    digest = sha256_of(gguf)
    if digest != entry["sha256"]:
        raise Refused(f"the {system} adapter {entry['gguf']} has SHA-256 {digest}, the configuration records "
                      f"{entry['sha256']}")
    if entry["selection"] is None:
        return
    try:
        chosen = json.loads(Path(entry["selection"]).read_text(encoding="utf-8"))["chosen"]
        same = same_path(chosen["gguf"], gguf) and chosen["sha256"] == entry["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Refused(f"the {system} adapter's selection {entry['selection']} cannot be read: "
                      f"{type(exc).__name__}: {exc}") from None
    if not same:
        raise Refused(f"the {system} adapter {entry['gguf']} is not the checkpoint {entry['selection']} "
                      f"chose ({chosen['gguf']}, SHA-256 {chosen['sha256']})")


def check_demonstrations(config: dict) -> None:
    try:
        digest = block_hash(demonstrations_block(load_demonstrations(TRAINING_SET, REQUESTS_FILE)))
    except (OSError, ValueError, KeyError) as exc:
        raise Refused(f"b1's demonstrations cannot be built: {type(exc).__name__}: {exc}") from None
    if digest != config["b1_block_sha256_16"]:
        raise Refused(f"b1's demonstrations block hashes to {digest}, the configuration records "
                      f"{config['b1_block_sha256_16']}: {TRAINING_SET.as_posix()} is not the training set "
                      "the development runs used")


def spent(milestone: str) -> tuple[float, float, float]:
    """What the milestone has spent, what all stages have, and the cap over all stages, in RMB."""
    budget = Budget(LLM_DIR / "budget.sqlite", stage=milestone, stage_cap_rmb=STAGE_CAPS[milestone])
    try:
        return budget.spent(milestone), budget.spent(), budget.total_cap
    finally:
        budget.close()


def check_budget(config: dict) -> None:
    """Refused unless the milestone's budget can take the cap of every system that spends (SPENDING),
    whichever cloud model it calls."""
    milestone, cap = config["milestone"], config["b2"]["max_cost_rmb"]
    spenders = [SPENDER_NAMES[s] for s in SPENDING if s in config["systems"]]
    caps = cap * len(spenders)
    stage, total, total_cap = spent(milestone)
    if stage + caps > STAGE_CAPS[milestone] or total + caps > total_cap:
        what = (f"{spenders[0]}'s cap of {cap:g} RMB" if len(spenders) == 1
                else f"the caps of {' and '.join(spenders)} ({cap:g} RMB each)")
        raise Refused(f"the {milestone} API budget cannot take {what}: the stage has spent {stage:.4f} of "
                      f"{STAGE_CAPS[milestone]:g} RMB, all stages {total:.4f} of {total_cap:g}")


def check_routing(config: dict, generation: dict | None = None) -> dict:
    """The routed system's frozen point (the routing file's chosen_point: routed, successes) when the
    routing file is the one configured, chosen on the family's development sets as they are
    (check_dev_sets), with the adapter of the routed system's server and on FT's runs in the generation its
    systems run with (generation: the record of their settings, GenerationSettings.record(); None on the
    legacy path, whose routing file records none: check_generation_of), names the model of the routed system's
    cloud system (ROUTED, CLOUD_MODELS) and, for a system of RULED (ft2r, ft3r), holds the M6 policy fields
    consistently (check_rules); Refused otherwise."""
    system = routed_system(config)
    entry = config["routing"]
    path = Path(entry["path"])
    if not path.is_file():
        raise Refused(f"the routing file {entry['path']} is missing (routing.path)")
    digest = sha256_of(path)
    if digest != entry["sha256"]:
        raise Refused(f"the routing file {entry['path']} has SHA-256 {digest}, the configuration records "
                      f"{entry['sha256']}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        point = {key: data["chosen_point"][key] for key in ("routed", "successes")}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Refused(f"the routing file {entry['path']} has no chosen point to check the dry run against: "
                      f"{type(exc).__name__}: {exc}") from None
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in point.values()):
        raise Refused(f"the routing file {entry['path']} has a chosen point without counts: {point}")
    check_dev_sets(config, data, entry["path"])
    adapter_name = SERVER_OF[system]
    adapter = config["adapters"][adapter_name]["sha256"]
    if data.get("adapter_sha256") != adapter:
        raise Refused(f"the routing file {entry['path']} records adapter_sha256 "
                      f"{data.get('adapter_sha256')}, not that of adapters.{adapter_name} ({adapter})")
    check_generation_of(data, generation, entry["path"], adapter_name)
    cloud_system = ROUTED[system][1]
    cloud = CLOUD_MODELS[cloud_system]
    if data.get("cloud_model") != cloud:
        raise Refused(f"the routing file {entry['path']} routes to {data.get('cloud_model')!r}, not to "
                      f"{cloud}, the model of {cloud_system}, which {system} routes to")
    if system in RULED:
        check_rules(data, entry["path"])
    return point


# what each field of the frozen file stands for in the configuration, for the preflight's refusals
FROZEN_FIELDS = {
    "adapter": "adapters.final.gguf and sha256", "base": "the base model of the final server, as it is now",
    "tokenizer": "the final server's tokenizer folder", "family": "the family of that folder",
    "generation": "adapters.final.generation with systems.final's decoding and votes",
    "votes": "systems.final.votes", "vote_mode": "systems.final.vote_mode",
    "rules": "systems.final.rules, with the rules of this code (RULES_VERSION)",
    "glossary": "systems.final.glossary, as its file is now",
    "dev_sets": "the development sets of the test sets, as they are now",
}


def check_glossaries(config: dict) -> None:
    """Refused unless every glossary a system's options name is here with its configured SHA-256."""
    for system in config["systems"]:
        glossary = options_of(config, system).get(GLOSSARY)
        if glossary is None:
            continue
        path = Path(glossary["path"])
        if not path.is_file():
            raise Refused(f"the glossary {glossary['path']} of {system} is missing "
                          f"(systems.{system}.glossary)")
        digest = sha256_of(path)
        if digest != glossary["sha256"]:
            raise Refused(f"the glossary {glossary['path']} of {system} has SHA-256 {digest}, the "
                          f"configuration records {glossary['sha256']}")


def check_frozen(config: dict, generated: dict[str, Generated]) -> Frozen:
    """The final system's frozen file (module docstring: the preflight) when it is the one configured and
    records, field by field, what the configuration gives now (frozen_identity over the final server's family
    path, generated), and finalg's generation when finalg runs; Refused naming the first field that does
    not."""
    entry = options_of(config, FINAL)[FROZEN]
    where, path = entry["path"], Path(entry["path"])
    if not path.is_file():
        raise Refused(f"the frozen file {where} is missing (systems.{FINAL}.frozen.path): "
                      "scripts/freeze_final.py writes it from the development runs")
    digest = sha256_of(path)
    if digest != entry["sha256"]:
        raise Refused(f"the frozen file {where} has SHA-256 {digest}, the configuration records "
                      f"{entry['sha256']}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        held = [data[system]["items"] for system in FROZEN_SYSTEMS]
        ok = all(isinstance(items, list) and all(isinstance(e, dict) for e in items) for items in held)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Refused(f"the frozen file {where} cannot be read: {type(exc).__name__}: {exc}") from None
    if not ok:
        raise Refused(f"the frozen file {where} holds no list of items for {FINAL} and {FINALG}")
    family_path = generated[SERVER_OF[FINAL]]
    for name, value in frozen_identity(config, family_path.family, family_path.settings).items():
        if data.get(name) != value:
            raise Refused(f"the frozen file {where} records {name} {canonical_json(data.get(name))}, not "
                          f"{canonical_json(value)}, {FROZEN_FIELDS[name]}: freeze the configured system "
                          "(scripts/freeze_final.py) or configure the frozen one")
    if FINALG in config["systems"]:
        greedy = system_generation(config, FINALG, family_path.settings).record()
        if data[FINALG].get(GENERATION) != greedy:
            raise Refused(f"the frozen file {where} records finalg.generation "
                          f"{canonical_json(data[FINALG].get(GENERATION))}, not {canonical_json(greedy)}, "
                          f"{FINALG}'s (adapters.final.generation with systems.{FINALG}'s options)")
    return Frozen(where, digest, data)


def check_generation_of(data: dict, generation: dict | None, path: str, adapter: str) -> None:
    """Refused unless the routing file was chosen on FT's runs in the generation the systems on the adapter's
    server run with: its ft_generation (scripts/route_curve.py) equal to generation (the record of their
    settings), or neither one there (the legacy path, M4 to M6)."""
    chosen = data.get(FT_GENERATION)
    if chosen == generation:
        return
    if chosen is None:
        raise Refused(f"the routing file {path} records no {FT_GENERATION}, while the systems on the "
                      f"{adapter} server run with the generation {canonical_json(generation)} "
                      f"(adapters.{adapter}.{GENERATION}): it was not chosen on FT's runs in it")
    if generation is None:
        raise Refused(f"the routing file {path} records {FT_GENERATION} {canonical_json(chosen)}, while "
                      f"adapters.{adapter} names no {GENERATION}: its systems run on the legacy path, not on "
                      "the one the routing was chosen on")
    raise Refused(f"the routing file {path} records {FT_GENERATION} {canonical_json(chosen)}, not "
                  f"{canonical_json(generation)}, the generation of adapters.{adapter}")


def check_dev_sets(config: dict, data: dict, path: str) -> None:
    """Refused unless the routing file was chosen on the family's development sets (dev_sets_of) as they are
    now: as many as the family has, each with the SHA-256 the file records for it (eval.systems.
    routing_dev_sets: v1's dev_sha256, v2's dev_sets), in order."""
    dev_sets = dev_sets_of(config)
    try:
        recorded = routing_dev_sets(data)
    except ValueError as exc:
        raise Refused(f"the routing file {path}: {exc}") from None
    if len(recorded) != len(dev_sets):
        chosen = f"{len(recorded)} development set{'' if len(recorded) == 1 else 's'}"
        raise Refused(f"the routing file {path} was chosen on {chosen}, not on the {len(dev_sets)} of the "
                      f"{config.get('test_sets', DEFAULT_FAMILY)} test sets "
                      f"({', '.join(dev.as_posix() for dev in dev_sets)})")
    for number, ((name, digest), dev) in enumerate(zip(recorded, dev_sets, strict=True), 1):
        now = sha256_of(dev)
        if digest == now:
            continue
        if "dev_sets" not in data:  # a file of one set, as before M7
            raise Refused(f"the routing file {path} records dev_sha256 {digest}, not that of "
                          f"{dev.as_posix()} ({now})")
        raise Refused(f"the routing file {path} records the development set {number} ({name}) with SHA-256 "
                      f"{digest}, not that of {dev.as_posix()} ({now})")


def check_rules(data: dict, path: str) -> None:
    """Refused unless the routing file holds the M6 policy fields scripts/route_curve.py writes (rules,
    chosen_policy, policies) consistently: the rules well formed (agent.route.RouteRules; null for P0), the
    chosen policy among the policies with the file's rules, and the chosen point that policy's own."""
    if not all(key in data for key in ("rules", "chosen_policy", "policies")):
        raise Refused(f"the routing file {path} has no rules (rules, chosen_policy and policies, which "
                      "scripts/route_curve.py writes from the M6 policies): not a routing file of the M6 "
                      "policies")
    rules, policy, policies = data["rules"], data["chosen_policy"], data["policies"]
    if rules is not None:
        try:
            RouteRules.from_json(rules)
        except ValueError as exc:
            raise Refused(f"the routing file {path} has malformed rules: {exc}") from None
    if (not isinstance(policy, str) or not isinstance(policies, dict) or policy not in policies
            or not isinstance(policies[policy], dict)):
        raise Refused(f"the routing file {path} names the policy {policy!r}, which its policies do not hold")
    if policies[policy].get("rules") != rules:
        raise Refused(f"the routing file {path}: the rules of the policy {policy} are not the file's rules")
    point = {k: v for k, v in data["chosen_point"].items() if k != "policy"}
    if policies[policy].get("chosen") != point:
        raise Refused(f"the routing file {path}: the chosen point is not the chosen point of the policy "
                      f"{policy}")


def reference_outputs(milestone: str, family: TestFamily = DEFAULT_SETS) -> dict[str, str]:
    """The result files of the milestone's opens of the family's sets with their SHA-256, as its two
    completion markers (one per access log, status complete) record them; Refused unless there are exactly
    those two and they agree."""
    complete: dict[str, list[dict]] = {}
    for marker in markers(family):
        opened = marker.get("open") if isinstance(marker.get("open"), dict) else {}
        if opened.get("milestone") == milestone and marker.get("status") == COMPLETE:
            complete.setdefault(marker.get("access_log"), []).append(marker)
    logs = [log.as_posix() for _, log in family.sets]
    for log in logs:
        if len(complete.get(log, [])) != 1:
            raise Refused(f"{family.completions.as_posix()} holds {len(complete.get(log, []))} complete "
                          f"{milestone} opens of {log}, not exactly one: the reference milestone's results "
                          "cannot be told")
    outputs = [complete[log][0].get("outputs") for log in logs]
    if not isinstance(outputs[0], dict) or outputs[0] != outputs[1]:
        raise Refused(f"the {milestone} completion markers of {logs[0]} and {logs[1]} record different "
                      "outputs")
    return outputs[0]


def check_marked(path: str, outputs: dict[str, str], milestone: str,
                 family: TestFamily = DEFAULT_SETS) -> None:
    """Refused unless path is a result file the milestone's completion markers record, with their SHA-256."""
    if path not in outputs:
        raise Refused(f"{path} is not among the outputs the {milestone} completion markers record "
                      f"({family.completions.as_posix()})")
    if not Path(path).is_file():
        raise Refused(f"{path} is missing: the {milestone} completion markers record it")
    digest = sha256_of(path)
    if digest != outputs[path]:
        raise Refused(f"{path} has SHA-256 {digest}, the {milestone} completion markers record "
                      f"{outputs[path]}")


def recorded_p(config: dict, row: dict) -> float:
    """The p of a recorded comparison: its file's row of the same name, a tested comparison with the same
    systems; Refused otherwise."""
    path, name = row["recorded"], row["name"]
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))["comparisons"]
        found = [r for r in rows if r.get("name") == name]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise Refused(f"{path} cannot be read as comparisons: {type(exc).__name__}: {exc}") from None
    if not found:
        raise Refused(f"{path} has no comparison {name}")
    there = found[0]
    for kind in ("deferred", "recorded"):
        if kind in there:
            raise Refused(f"{path} holds {name} as {kind}, not as a tested comparison: it has no p of its "
                          "own")
    if (there.get("a"), there.get("b")) != (row["a"], row["b"]):
        raise Refused(f"{path} compares {there.get('a')} with {there.get('b')} under {name}, not {row['a']} "
                      f"with {row['b']}")
    p = there.get("p")
    if isinstance(p, bool) or not isinstance(p, int | float) or not 0 <= p <= 1:
        raise Refused(f"{path}: {name} has no p-value ({p!r})")
    return float(p)


def check_references(config: dict) -> dict[str, float]:
    """The p of every recorded comparison, by name, once the reference results and the recorded comparisons
    files are those the reference milestones' markers record (each reference system's own milestone, else
    the block's) and every reference records file has its configured SHA-256 (module docstring); Refused
    otherwise."""
    recorded = [row for row in config["comparisons"] if comparison_kind(row) == "recorded"]
    if "reference" not in config:
        return {}
    references, family = config["reference"]["systems"], family_of(config)
    milestones = list(dict.fromkeys([config["reference"]["milestone"],
                                     *(reference_milestone(config, s) for s in references)]))
    try:
        outputs = {milestone: reference_outputs(milestone, family) for milestone in milestones}
    except JsonlError as exc:
        raise Refused(f"the completion markers cannot be read: {exc}") from None
    for system, entry in references.items():
        milestone = reference_milestone(config, system)
        check_marked(entry["results"], outputs[milestone], milestone, family)
        try:
            results = json.loads(Path(entry["results"]).read_text(encoding="utf-8"))
            held = (results.get("system"), results.get("records"))
        except (OSError, ValueError, AttributeError) as exc:
            raise Refused(f"{entry['results']} cannot be read: {type(exc).__name__}: {exc}") from None
        if held != (system, entry["records"]):
            raise Refused(f"{entry['results']} holds the records {held[1]} of {held[0]}, not "
                          f"{entry['records']} of {system} (reference.systems.{system})")
    for path in dict.fromkeys(row["recorded"] for row in recorded):
        holders = [milestone for milestone in milestones if path in outputs[milestone]]
        if not holders:
            raise Refused(f"{path} is not among the outputs the {' or '.join(milestones)} completion markers "
                          f"record ({family.completions.as_posix()})")
        check_marked(path, outputs[holders[0]], holders[0], family)
    for system, entry in references.items():
        for key, label in (("records", f"the {system} records"), ("dev_records",
                                                                   f"the {system} development records")):
            path, expected = entry[key], entry[f"{key}_sha256"]
            if not Path(path).is_file():
                raise Refused(f"{label} {path} are missing (reference.systems.{system}.{key})")
            digest = sha256_of(path)
            if digest != expected:
                raise Refused(f"{label} {path} have SHA-256 {digest}, the configuration records {expected}")
    return {row["name"]: recorded_p(config, row) for row in recorded}


def check_acebench(config: dict) -> dict[str, tuple[PerItem, PerItem]]:
    """Per ACEBench comparison, by name, its two runs read answer by answer (eval.acebench.per_item) and
    paired (pair_strata), before any test set is read; Refused naming what cannot be read or does not
    match."""
    pairs = {}
    for row in config["comparisons"]:
        if comparison_kind(row) != "acebench":
            continue
        block = row["acebench"]
        try:
            a, b = per_item(block["a"]), per_item(block["b"])
            pair_strata(a, b)
        except (AceItemsError, ValueError) as exc:
            raise Refused(f"the ACEBench comparison {row['name']}: {exc}") from None
        pairs[row["name"]] = (a, b)
    return pairs


def check_dry_run(path: Path, commit: str, digest: str) -> None:
    """Refused unless path is the session.json of a dry run of this commit and this configuration whose
    development check passed (module docstring: the dry run)."""
    where = path.as_posix()
    try:
        dry = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise Refused(f"{where} is missing: a real session follows a dry run (--dry-run) of the same commit "
                      "and configuration whose development check passed") from None
    except (OSError, ValueError) as exc:
        raise Refused(f"{where} cannot be read: {exc}") from None
    if not isinstance(dry, dict) or dry.get("dry_run") is not True:
        raise Refused(f"{where} is not a dry run's session")
    if dry.get("git_commit") != commit:
        raise Refused(f"{where} is a dry run of git commit {dry.get('git_commit')}, not of HEAD {commit}: "
                      "run --dry-run again")
    config = dry.get("config") if isinstance(dry.get("config"), dict) else {}
    if config.get("sha256") != digest:
        raise Refused(f"{where} is a dry run of the configuration with SHA-256 {config.get('sha256')}, not "
                      f"{digest}: run --dry-run again")
    check = dry.get("dev_check")
    if not isinstance(check, dict) or check.get("passed") is not True:
        raise Refused(f"{where} records a dev check that did not pass ({canonical_json(check)}): the "
                      "routed system's routing on the development set is not the frozen point's")


def idle_gpu(config: dict, run: Callable) -> GpuSample:
    try:
        sample = gpu_sample(run=run)
    except GpuError as exc:
        raise Refused(f"cannot tell whether the GPU is idle: {exc}") from None
    limit = config["gpu"]["idle_max_mib"]
    if sample.memory_mib > limit:
        raise Refused(f"the GPU is not idle: {sample.memory_mib:.0f} MiB in use, at most {limit:g} count as "
                      "idle (gpu.idle_max_mib); stop what uses it (a training run, a server) first")
    return sample


def free_port(config: dict, client: Callable[[str], LocalClient]) -> None:
    """Refused when a server already answers on the port the session's servers will use: LocalServer would
    refuse it after the opening, using an open up."""
    url = f"http://{HOST}:{config['server']['port']}"
    probe = client(url)
    try:
        busy = probe.health()
    finally:
        probe.close()
    if busy:
        raise Refused(f"a server already answers at {url}: stop it first, since the session starts its own "
                      "(server.port)")


def preflight(config: dict, config_path: Path, *, run: Callable, client: Callable[[str], LocalClient],
              dry_run: bool, generated: dict[str, Generated] | None = None) -> Ready:
    """The checks before any test set is read (module docstring), in its order; generated is the family path
    of the servers whose adapter names a generation (generations), which the routing file must have been
    chosen in and the frozen file frozen in."""
    milestone, family = config["milestone"], family_of(config)
    try:
        relative = config_path.resolve().relative_to(Path(REPO).resolve()).as_posix()
    except ValueError:
        raise Missing(f"{config_path} lies outside the repository") from None
    try:
        pending = unfinished(milestone, family)
    except (LockError, JsonlError) as exc:
        raise Refused(f"the access logs or the completion markers cannot be read: {exc}") from None
    if pending:
        raise Unfinished(describe_unfinished(pending, milestone, relative, family))
    if not committed(relative):
        raise Refused(f"{relative} is not committed: an open is made with a configuration that is in HEAD, "
                      "unchanged (the preregistration)")
    dirty = git_dirty(REPO)
    if dirty is None:
        raise Refused("git cannot tell whether the working tree is clean")
    if dirty:
        raise Refused("the working tree has uncommitted changes: commit them first, so that the commit an "
                      "open states is what ran")
    used = []
    for test_set, log in family.sets:
        left = LockedSet(test_set, log).opens_left()
        if left <= 0:
            raise Refused(f"{test_set.as_posix()} has no open left: {log.as_posix()} holds "
                          f"{MAX_TEST_SET_OPENS} opens, the limit")
        used.append(MAX_TEST_SET_OPENS - left)
    base = Path("runs") / f"test-{milestone.lower()}"
    folder = base / (DRY_RUN if dry_run else f"open-{used[0] + 1}")  # n from the single-step set's log
    if not dry_run and folder.exists():
        raise Refused(f"{folder.as_posix()} already exists: the folder of the next open must be new")
    for test_set, _ in family.sets:
        try:
            check_lock(test_set)  # hashes the bytes; nothing is parsed
        except LockError as exc:
            raise Refused(str(exc)) from None
    for name in served(config["systems"]):  # the adapters of the servers the session starts
        check_adapter(config, name)
    if "b1" in config["systems"]:
        check_demonstrations(config)
    check_glossaries(config)
    frozen = check_frozen(config, generated or {}) if FINAL in config["systems"] else None
    routed, frozen_point = routed_system(config), None
    if routed is not None:  # FT's runs the routing was chosen on: in its server's generation, if any
        family_path = (generated or {}).get(SERVER_OF[routed])
        frozen_point = check_routing(config, None if family_path is None else family_path.settings.record())
    recorded = check_references(config)
    acebench = check_acebench(config)
    commit, digest = git_commit(REPO), sha256_of(config_path)
    if not dry_run:
        check_dry_run(base / DRY_RUN / "session.json", commit, digest)
    if any(s in SPENDING for s in config["systems"]):
        check_budget(config)
    free_port(config, client)
    idle_gpu(config, run)
    purpose = f"{config['purpose']} (git commit {commit}; config {relative} sha256 {digest})"
    return Ready(relative, digest, commit, purpose, folder, recorded, frozen_point, acebench, frozen)


@dataclass
class Setup:
    renders: dict[str, Callable]  # the chat template of each tokenizer folder of the legacy path, by its path
    counters: dict[str, TokenCounter | None]  # and its token counter
    world: World
    today: str
    # the family path of the servers whose adapter names a generation, by the adapter's name (generations)
    generated: dict[str, Generated] = field(default_factory=dict)


def prepare(config: dict, machine: Machine, generated: dict[str, Generated] | None = None) -> Setup:
    """What every run needs and can fail without a server: Missing when it cannot be had here. generated is
    the family path of the servers whose adapter names a generation (generations), loaded already."""
    try:
        renders, counters = {}, {}
        for folder in tokenizers(config):
            renders[folder] = machine.load_render(folder)
            counters[folder] = machine.load_counter(folder)
        world = machine.load_world()
        today = config["e2e"]["today"] or corpus_recorded_on(MANIFEST)
        probe(world.owner, today)
    except Exception as exc:  # noqa: BLE001  whatever it is, nothing has been opened yet
        raise Missing(f"the session cannot be set up: {type(exc).__name__}: {exc}") from None
    # the cloud systems the spending systems call (b2, a routed system's cloud): the key, the price
    for cloud in dict.fromkeys(cloud_of(s) for s in config["systems"] if s in SPENDING):
        try:
            build_system(cloud, llm_dir=LLM_DIR, stage=config["milestone"],
                         max_cost=config["b2"]["max_cost_rmb"], openai_client=machine.openai_client).close()
        except SystemUnavailable as exc:
            raise Missing(str(exc)) from None
    return Setup(renders, counters, world, today, dict(generated or {}))


# ---- the material -----------------------------------------------------------------------------------------


@dataclass
class Material:
    """The items and tasks a session runs, the sets they came from and their opens (None in a dry run)."""

    items: list[dict]
    tasks: list[dict]
    sets: dict[str, dict]
    opens: dict[str, dict | None]
    # every single-step item of the set: what a routed system runs and the references are scored against;
    # in a dry run the whole development set, of which items holds the first only
    full: list[dict]


def check_items(items: list[dict]) -> None:
    """ValueError for single-step items the runner, the scorer or the comparisons would stop at (an item
    neither drafted nor handwritten has no cluster: eval.metrics.cluster_keys), before any model runs."""
    seen = set()
    for number, item in enumerate(items, 1):
        name = item.get("id") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"item {number}: its id {name!r} is missing or repeated")
        seen.add(name)
        if item.get("type") not in TYPE_CHECKS:
            raise ValueError(f"item {name}: type {item.get('type')!r} is not one of {list(TYPE_CHECKS)}")
        if not isinstance(item.get("messages"), list) or not item["messages"]:
            raise ValueError(f"item {name} has no messages")
    cluster_keys(items)


def development_material(dev_sets: Sequence[Path] = (DEV_SET,)) -> Material:
    """The dry run's (module docstring: --dry-run): the first item of the development set (dev_sets, the
    family's, in order: jointly for v2) and the first end-to-end development task, and every item of the
    development set for the routed system and the references."""
    full, opened = [], []
    try:
        for path in dev_sets:
            dev = DevSet(path)
            items = dev.open()
            refuse_test_material(items, path)
            full += items
            opened.append({"path": path.as_posix(), "sha256": dev.last_open["sha256"]})
        digest = check_lock(E2E_DEV_SET)
        tasks = load_tasks(E2E_DEV_SET)[:1]
        refuse_test_material(tasks, E2E_DEV_SET)
        check_items(full)
    except (LockError, OSError, ValueError) as exc:
        raise Refused(f"the development material of the dry run: {exc}") from None
    items = full[:1]
    single_step = {**opened[0], "n": len(items)} if len(opened) == 1 else {"sets": opened, "n": len(items)}
    sets = {"single_step": single_step,
            "e2e": {"path": E2E_DEV_SET.as_posix(), "sha256": digest, "n": len(tasks)}}
    return Material(items, tasks, sets, {"single_step": None, "e2e": None}, full)


def open_test_sets(config: dict, ready: Ready, now: Callable[[], str]) -> Material:
    """The opening (module docstring): one open of each test set of the family, the items checked before
    anything runs."""
    milestone, purpose, family = config["milestone"], ready.purpose, family_of(config)
    (test_set, test_log), (e2e_set, e2e_log) = family.sets
    try:
        opened = LockedSet(test_set, test_log, now=now).open(milestone, purpose)
    except (LockError, JsonlError, OSError) as exc:  # nothing was written: no open is used up
        raise Refused(f"{test_set.as_posix()} could not be opened: {exc}") from None
    try:
        tasks = LockedSet(e2e_set, e2e_log, now=now).open(milestone, purpose)
    except (LockError, JsonlError, OSError) as exc:
        raise Failed(f"{e2e_set.as_posix()} could not be opened: {exc}") from None
    used = [len(LockedSet(path, log).log_entries()) for path, log in family.sets]
    print(f"opened {test_set.as_posix()} (open {used[0]} of {MAX_TEST_SET_OPENS}, {len(opened)} items) and "
          f"{e2e_set.as_posix()} (open {used[1]} of {MAX_TEST_SET_OPENS}, {len(tasks)} tasks)",
          flush=True)
    try:
        check_items(opened)
        checked = OpenedTestSet(check_tasks(tasks, e2e_set.as_posix()), tasks.entry)
    except ValueError as exc:
        raise Failed(f"the test material is malformed: {exc}") from None
    sets = {"single_step": {"path": test_set.as_posix(), "sha256": opened.entry["sha256"], "n": len(opened)},
            "e2e": {"path": e2e_set.as_posix(), "sha256": tasks.entry["sha256"], "n": len(checked)}}
    return Material(opened, checked, sets, {"single_step": opened.entry, "e2e": tasks.entry}, opened)


# ---- the session ------------------------------------------------------------------------------------------


@dataclass
class SystemRun:
    system: str
    info: dict  # the single-step run's build (eval.systems.Built.info)
    e2e_info: dict
    server: dict | None  # a local system's server: its build and model path
    records: list[dict]
    scores: list[dict]
    items_timing: list[dict]
    tasks: list[dict]  # the end-to-end task records
    tasks_timing: list[dict]
    modes: dict | None  # a spending system's single-step cloud turns per thinking mode
    e2e_modes: dict | None  # and its end-to-end ones
    ledger_rmb: float | None  # what the ledger booked during a spending system's runs
    items: list[dict]  # the single-step items it ran
    # its single-step scores per scorer, the default one's (scores) among them
    scored: dict[str, list[dict]] = field(default_factory=dict)
    e2e: bool = True  # whether it ran end to end (its end_to_end option)


@dataclass
class Reference:
    """A system of an earlier milestone that the comparisons use without running it: its records, scored in
    this session against the items in memory, and its own milestone when the configuration names one."""

    system: str
    path: str
    sha256: str
    records: list[dict]
    scores: list[dict]
    milestone: str | None = None
    scored: dict[str, list[dict]] = field(default_factory=dict)  # per scorer, as SystemRun's


def cloud_modes(model: object) -> dict:
    """The turns per thinking mode of a cloud model, or of a routed model's cloud model."""
    modes = getattr(model, "modes", None)
    if modes is None:
        modes = getattr(getattr(model, "cloud", None), "modes", None) or {}
    return dict(sorted(modes.items()))


def answered_by(record: dict) -> str | None:
    """A routed system's record: who answered it (agent.route: local or cloud); None for any other record."""
    route = record.get("route")
    return route.get("answered_by") if isinstance(route, dict) else None


def noninferiority(tested: dict, margin: float) -> dict:
    """The descriptive non-inferiority bound of a comparison: whether each interval's low end is above
    -margin."""
    paired, cluster = tested["paired_ci"]["low"], tested["cluster_ci"]["low"]
    return {"margin": margin, "paired_low": paired, "paired_above": paired > -margin,
            "cluster_low": cluster, "cluster_above": cluster > -margin}


def rel(path: Path) -> str:
    return Path(path).as_posix()


def repository_path(path: object) -> object:
    """A path a server reported, as the results may hold it: relative to the repository root when it lies
    inside, its file name when it lies outside (no machine path), anything else as it is. A path that is
    absolute only in the other system's spelling (C:\\... read on Linux, /... read on Windows) cannot lie in
    this repository and is reduced to its file name without being resolved; a relative one is stored with
    forward slashes whichever system's separators it uses (on Linux a backslash would otherwise stay)."""
    if not isinstance(path, str) or not path:
        return path
    windows = PureWindowsPath(path)
    if not Path(path).is_absolute():
        if windows.anchor or PurePosixPath(path).is_absolute():
            return windows.name
        return windows.as_posix()
    try:
        return Path(path).resolve().relative_to(Path(REPO).resolve()).as_posix()
    except (OSError, ValueError):
        return windows.name


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def percent(rate: float | None) -> str:
    return "-" if rate is None else f"{100 * rate:.1f}%"


def subset_summaries(scores: list[dict], items: list[dict]) -> dict:
    """The summary of the scores over each subset of the items (eval.report.SUBSETS: the drafted and the
    handwritten ones), None for a subset without items."""
    out = {}
    for subset in SUBSETS:
        chosen = [item for item in items if in_subset(item, subset)]
        ids = {item["id"] for item in chosen}
        out[subset] = summarize([s for s in scores if s["id"] in ids], chosen) if chosen else None
    return out


class Session:
    """The session's runs in order (module docstring: the session), then its outputs."""

    def __init__(self, config: dict, ready: Ready, setup: Setup, material: Material, machine: Machine,
                 binary: Path, *, dry_run: bool) -> None:
        self.config, self.ready, self.setup, self.material = config, ready, setup, material
        self.machine, self.binary, self.dry_run = machine, binary, dry_run
        self.folder = ready.folder
        self.milestone = config["milestone"]
        self.runs: dict[str, SystemRun] = {}
        self.references: dict[str, Reference] = {}
        self.servers: dict[str, dict] = {}
        self.commands: dict[str, list[str]] = {}
        self.scorers: list[str] = list(config.get(SCORERS_KEY) or [])  # besides the default scorer

    def scorers_of(self, system: str) -> list[str]:
        """The scorers a system's single-step run is scored under besides the default one: the
        configuration's, and every scorer for final and finalg when the session holds a frozen file."""
        if self.ready.frozen is not None and system in FROZEN_SYSTEMS:
            return [s for s in SCORERS if s != DEFAULT_SCORER]
        return self.scorers

    def run(self) -> None:
        self.score_references()
        systems = [s for s in RUN_ORDER if s in self.config["systems"]]
        for name in ADAPTERS:
            group = [s for s in systems if SERVER_OF.get(s) == name]
            if group:
                self.serve(name, group)
        for system in systems:
            if system in CLOUD_MODELS:
                self.runs[system] = self.run_system(system, None, None)

    def score_references(self) -> None:
        """The reference systems' records scored against the items in memory, before any model runs: the
        development records in a dry run, the test records otherwise."""
        if "reference" not in self.config:
            return
        key = "dev_records" if self.dry_run else "records"
        for system, entry in self.config["reference"]["systems"].items():
            try:
                records = read_jsonl(entry[key])
                scored = {scorer: score_items(records, self.material.full, scorer)
                          for scorer in (DEFAULT_SCORER, *self.scorers)}
            except (OSError, JsonlError, ValueError) as exc:
                raise Failed(f"the {system} reference records {entry[key]} cannot be scored against the "
                             f"items: {exc}") from None
            self.references[system] = Reference(system, entry[key], entry[f"{key}_sha256"], records,
                                                scored[DEFAULT_SCORER], entry.get(REFERENCE_MILESTONE),
                                                scored)

    def scores_of(self, system: str, scorer: str = DEFAULT_SCORER) -> list[dict] | None:
        """A system's single-step scores under scorer: the session's run, else the reference's; None for
        neither."""
        if system in self.runs:
            return self.runs[system].scored[scorer]
        reference = self.references.get(system)
        return None if reference is None else reference.scored[scorer]

    def records_of(self, system: str) -> list[dict] | None:
        if system in self.runs:
            return self.runs[system].records
        reference = self.references.get(system)
        return None if reference is None else reference.records

    def serve(self, name: str, group: list[str]) -> None:
        """One server with name's adapter, over its base model, for the systems of group, the GPU sampled
        beside it."""
        server, adapter = self.config["server"], self.config["adapters"][name]
        model = base_model(self.config, name)
        interval = self.config["gpu"]["sample_ms"]
        try:
            baseline = gpu_sample(run=self.machine.run)
            sampler = GpuSampler(interval_ms=interval, popen=self.machine.popen).start()
        except GpuError as exc:
            raise Failed(f"the GPU before the {name} server: {exc}") from None
        print(f"{name} server: {adapter['gguf']} over {model} for {', '.join(group)}", flush=True)
        try:
            with LocalServer(self.binary, model=model, adapter=adapter["gguf"], port=server["port"],
                             ctx_size=server["ctx_size"], timeout=server["timeout_s"],
                             extra=("-np", str(SLOTS)), log=self.folder / f"server-{name}.log",
                             popen=self.machine.popen,
                             client=self.machine.client) as local:
                self.commands[name] = ["llama-server", *local.cmd[1:]]
                props = self.check_server(name, local, adapter)
                for system in group:
                    self.runs[system] = self.run_system(system, local, props)
        except (PortBusy, NotReady) as exc:
            raise Failed(f"the {name} server: {exc}") from None
        finally:
            samples = sampler.stop()
        self.servers[name] = {"systems": group, "model": model,
                              "adapter": {"gguf": adapter["gguf"], "sha256": adapter["sha256"]},
                              **server_resources(baseline, samples, interval_ms=interval)}

    def check_server(self, name: str, local: LocalServer, adapter: dict) -> dict:
        listed = [a.get("path") if isinstance(a, dict) else a for a in local.adapters]
        if len(listed) != 1 or not isinstance(listed[0], str) or not same_path(listed[0], adapter["gguf"]):
            raise Failed(f"the {name} server lists the adapters {listed}, not exactly {adapter['gguf']}")
        props = local.props or {}
        if props.get("total_slots") != SLOTS:
            raise Failed(f"the {name} server reports {props.get('total_slots')} slots (/props), not {SLOTS}")
        build = self.config["server"]["llama_cpp_build"]
        if build is not None and props.get("build_info") != build:
            raise Failed(f"the {name} server reports the build {props.get('build_info')!r}, the "
                         f"configuration records {build!r}")
        return {"build_info": props.get("build_info"), "model_path": repository_path(props.get("model_path"))}

    def build(self, system: str, local: LocalServer | None, max_cost: float | None) -> Built:
        """The system built (eval.systems.build_system; an alias as the system it stands for, BUILT_AS, its
        info naming both) over its tokenizer folder (tokenizer_of): on the family path with its server's
        family and its own generation settings (system_generation) when its adapter names a generation, else
        with the folder's chat template (module docstring: the family path), with the inference wrappers its
        options give (wrappers_of); checked against the configuration, Failed otherwise."""
        url = local.url if local else f"http://{HOST}:{self.config['server']['port']}"  # b2 does not use it
        routing = self.config.get("routing")
        routed = ({"routing_config": Path(routing["path"]), "dev_set": dev_sets_of(self.config)}
                  if system in ROUTED else {})
        tokenizer = tokenizer_of(self.config, system)
        family_path = self.setup.generated.get(SERVER_OF.get(system))  # None: the legacy path, a cloud system
        prompting = ({"count_tokens": self.setup.counters[tokenizer], "render": self.setup.renders[tokenizer]}
                     if family_path is None else
                     {"count_tokens": family_path.count_tokens, "render": None, "family": family_path.family,
                      "generation": system_generation(self.config, system, family_path.settings)})
        wrapped = wrappers_of(self.config, system) if options_of(self.config, system) else {}
        try:
            built = build_system(BUILT_AS.get(system, system), server=url, tokenizer=tokenizer,
                                 llm_dir=LLM_DIR, stage=self.milestone, max_cost=max_cost,
                                 training_set=TRAINING_SET, requests=REQUESTS_FILE,
                                 openai_client=self.machine.openai_client,
                                 local_client=local.client if local else None, **prompting, **routed,
                                 **wrapped)
        except SystemUnavailable as exc:
            raise Failed(f"{system} cannot be built: {exc}") from None
        glossary = options_of(self.config, system).get(GLOSSARY)
        if glossary is not None and built.info["glossary"]["sha256"] != glossary["sha256"]:
            built.close()
            raise Failed(f"{system} would note the glossary {glossary['path']} with SHA-256 "
                         f"{built.info['glossary']['sha256']}, the configuration records "
                         f"{glossary['sha256']}")
        if system in BUILT_AS:
            built.info = {"system": system, "built_as": BUILT_AS[system],
                          **{k: v for k, v in built.info.items() if k != "system"}}
        if routed and built.info["routing"]["sha256"] != routing["sha256"]:
            built.close()
            raise Failed(f"{system} would route by {routing['path']} with SHA-256 "
                         f"{built.info['routing']['sha256']}, the configuration records {routing['sha256']}")
        if system in SERVER_OF:
            entry = self.config["adapters"][SERVER_OF[system]]
            listed = built.info.get("adapters") or []
            if len(listed) != 1 or listed[0].get("sha256") != entry["sha256"] \
                    or not same_path(listed[0].get("path") or "", entry["gguf"]):
                built.close()
                raise Failed(f"{system} would run over the adapters {listed}, not {entry['gguf']} "
                             f"(SHA-256 {entry['sha256']})")
        recorded = self.config.get("b1_block_sha256_16")
        if system == "b1" and built.info["demonstrations"]["block_sha256_16"] != recorded:
            built.close()
            raise Failed(f"b1's demonstrations block is {built.info['demonstrations']['block_sha256_16']}, "
                         f"the configuration records {recorded}")
        return built

    def run_system(self, system: str, local: LocalServer | None, props: dict | None) -> SystemRun:
        """One system single step and then end to end (module docstring: the session). A system that spends
        (SPENDING: b2, and a routed system's calls to its cloud system, b2) is booked under the
        milestone within b2.max_cost_rmb, one cap for both of its runs."""
        spends = system in SPENDING
        cap = self.config["b2"]["max_cost_rmb"]
        before = spent(self.milestone)[0] if spends else 0.0
        clock = self.machine.clock
        # a dry run's routed system, final and finalg run every development item
        items = self.material.full if system in ROUTED or system in FROZEN_SYSTEMS else self.material.items
        built = self.build(system, local, cap if spends else None)
        try:
            timed = TimedModel(built.model, clock=clock, server_timings=built.timings or (
                lambda: getattr(built.local_client, "last_timings", None)))  # a vote's timings, summed
            timer = ItemTimer(timed, clock=clock)
            records = run_items(items, timed, count_tokens=built.count_tokens, on_item=timer)
            info = built.info
        finally:
            built.close()
        write_results(records, self.folder / f"{system}.jsonl")
        write_timing({"records": f"{system}.jsonl", "items": timer.items},
                     self.folder / f"{system}.timing.json")
        scored = {scorer: score_items(records, items, scorer)
                  for scorer in (DEFAULT_SCORER, *self.scorers_of(system))}
        scores = scored[DEFAULT_SCORER]
        e2e = options_of(self.config, system).get(END_TO_END, True)
        tasks, task_times, e2e_info, e2e_modes = [], [], None, None
        if e2e:
            tasks, task_times, e2e_info, e2e_modes = self.end_to_end_run(system, local, before)
        modes = dict(sorted(Counter(r["mode"] for r in records if "mode" in r).items())) if spends else None
        run = SystemRun(system, info, e2e_info, props, records, scores, timer.items, tasks, task_times,
                        modes, e2e_modes, spent(self.milestone)[0] - before if spends else None, items,
                        scored, e2e)
        success = sum(s["success"] for s in scores) / len(scores) if scores else None
        if not e2e:
            print(f"{system}: single step {len(scores)} items, success {percent(success)}; end to end not "
                  f"run (systems.{system}.{END_TO_END} false)", flush=True)
            return run
        e2e_success = summarize_tasks(tasks)["success_rate"]
        print(f"{system}: single step {len(scores)} items, success {percent(success)}; end to end "
              f"{len(tasks)} tasks, success {percent(e2e_success)}", flush=True)
        return run

    def end_to_end_run(self, system: str, local: LocalServer | None,
                       before: float) -> tuple[list[dict], list[dict], dict, dict | None]:
        """A system's end-to-end run after its single-step one (module docstring: the session): the task
        records, their timing, the build's info and, for a system that spends, its cloud turns per thinking
        mode; within what is left of its cap (before: what the stage had spent when it began)."""
        spends, clock = system in SPENDING, self.machine.clock
        cap = self.config["b2"]["max_cost_rmb"]
        left = max(cap - (spent(self.milestone)[0] - before), 0.0) if spends else None  # one cap for both
        built = self.build(system, local, left)
        if system in ROUTED and self.e2e_mode() == STEPWISE:
            built.model.stepwise = True  # agent.route.RoutedModel: every step decided on its own
        out = self.folder / f"e2e-{system}"
        try:
            timed = TimedModel(built.model, clock=clock, server_timings=built.timings or (
                lambda: getattr(built.local_client, "last_timings", None)))
            task_timer = TaskTimer(timed, clock=clock)
            tasks = run_tasks(self.material.tasks, model=timed, world=self.setup.world,
                              today=self.setup.today, out=out, count_tokens=built.count_tokens,
                              seed=self.config["e2e"]["seed"],
                              log=lambda line: print(f"  {system} {line}", flush=True), on_task=task_timer)
            e2e_info = built.info
            e2e_modes = cloud_modes(built.model) if spends else None
        finally:
            built.close()
        write_timing({"tasks": task_timer.tasks}, out / "timing.json")
        return tasks, task_timer.tasks, e2e_info, e2e_modes

    def e2e_mode(self) -> str:
        """How a routed system routes end to end (e2e_routing): sticky unless the configuration says."""
        return self.config.get("e2e_routing", STICKY)

    # ---- the outputs ----

    def head(self, part: str) -> dict:
        return {"milestone": self.milestone, "set": self.material.sets[part],
                "open": self.material.opens[part],
                "config": {"path": self.ready.config_path, "sha256": self.ready.config_sha256},
                "git_commit": self.ready.commit}

    def hashes(self, info: dict) -> dict:
        return {**info, "prompt_version": PROMPT_VERSION, "prompt_hash": PROMPT_HASH,
                "schema_hash": SCHEMA_HASH}

    def single_step(self, run: SystemRun) -> dict:
        """A system's single-step results (module docstring: outputs): the default scorer's where they always
        were, each of the configuration's scorers' in "scorers" right after the subsets."""
        items = run.items
        statuses = Counter(r["status"] for r in run.records)
        head = self.head("single_step")
        head["set"] = {**head["set"], "n": len(items)}  # a routed system's dry run covers the whole dev set
        others = {scorer: {"summary": summarize(run.scored[scorer], items),
                           "subsets": subset_summaries(run.scored[scorer], items),
                           "scores": run.scored[scorer]}
                  for scorer in self.scorers}
        return {"system": run.system, **head, "info": self.hashes(run.info),
                "server": run.server, "records": rel(self.folder / f"{run.system}.jsonl"),
                "statuses": dict(sorted(statuses.items())),
                **({"modes": run.modes} if run.modes is not None else {}),
                "summary": summarize(run.scores, items), "subsets": subset_summaries(run.scores, items),
                **({SCORERS_KEY: others} if others else {}),
                **({"routing": self.routing(run)} if run.system in ROUTED else {}),
                **wrapper_counts(run.records), "scores": run.scores}

    def routing(self, run: SystemRun) -> dict:
        """A routed system's single-step routing (module docstring: outputs): how many items it routed and
        why, the success of routing as many at random, and how often its answers are those of its local and
        its cloud system's own records."""
        local_system, cloud_system = ROUTED[run.system]
        n = len(run.records)
        local = [r for r in run.records if answered_by(r) == LOCAL]
        routed = [r for r in run.records if answered_by(r) == CLOUD]
        hard = Counter(r["route"]["hard_reason"].split(":", 1)[0] for r in run.records
                       if answered_by(r) is not None and r["route"].get("hard_reason"))
        share = len(routed) / n if n else None
        success = {system: self.mean_success(system, [item["id"] for item in run.items])
                   for system in (local_system, cloud_system)}
        random = None
        if share is not None and None not in success.values():
            random = (1 - share) * success[local_system] + share * success[cloud_system]
        return {"n": n, "routed": len(routed), "api_share": share,
                "hard_triggers": dict(sorted(hard.items())), "reference_success": success,
                "random_success": random,
                "agreement": {"local": self.agreement(local, local_system),
                              "routed": self.agreement(routed, cloud_system)}}

    def mean_success(self, system: str, ids: list[str]) -> float | None:
        """The mean success of system's scores over the items ids; None without scores for all of them."""
        by_id = {s["id"]: s for s in self.scores_of(system) or []}
        if not ids or any(i not in by_id for i in ids):
            return None
        return sum(bool(by_id[i]["success"]) for i in ids) / len(ids)

    def agreement(self, records: list[dict], system: str) -> dict:
        """Of records, how many have exactly the "output" of system's own record of the same item."""
        own = self.records_of(system)
        if own is None:
            return {"reference": system, "n": len(records), "equal": None}
        by_id = {r.get("id"): r for r in own}
        equal = sum("output" in r and r["output"] == by_id.get(r["id"], {}).get("output") for r in records)
        return {"reference": system, "n": len(records), "equal": equal}

    def end_to_end(self, run: SystemRun) -> dict:
        """A system's end-to-end results (module docstring: outputs); a voted or ruled system's with the votes
        and the rules counted over the attempts of its transcripts."""
        wrapped = any(key in run.e2e_info for key in ("vote_mode", "rules"))
        return {"system": run.system, **self.head("e2e"), "info": self.hashes(run.e2e_info),
                "server": run.server, "today": self.setup.today, "seed": self.config["e2e"]["seed"],
                "transcripts": rel(self.folder / f"e2e-{run.system}" / "transcripts"),
                **({"modes": run.e2e_modes} if run.e2e_modes is not None else {}),
                "summary": summarize_tasks(run.tasks),
                **({"e2e_routing": self.e2e_mode(), "routing": self.e2e_routing(run)}
                   if run.system in ROUTED else {}),
                **(wrapper_counts(self.attempts(run)) if wrapped else {}), "tasks": run.tasks}

    def attempts(self, run: SystemRun) -> list[dict]:
        """Every attempt of every step of a system's end-to-end transcripts, in task and step order."""
        out = []
        for task in run.tasks:
            if task.get("transcript"):
                path = self.folder / f"e2e-{run.system}" / task["transcript"]
                for step in json.loads(path.read_text(encoding="utf-8")).get("steps") or []:
                    out += [a for a in step.get("attempts") or [] if isinstance(a, dict)]
        return out

    def e2e_routing(self, run: SystemRun) -> dict:
        """A routed system's end-to-end routing (its mode is the file's e2e_routing): from the task records,
        the tasks with a cloud call (paid or from the cache) and the local and cloud calls (every model call,
        rejected attempts included); from the transcripts, the steps whose last attempt the local or the
        cloud model gave."""
        cloud_model = run.e2e_info.get("cloud_model")
        steps = Counter()
        for task in run.tasks:
            if not task.get("transcript"):
                continue
            path = self.folder / f"e2e-{run.system}" / task["transcript"]
            for step in json.loads(path.read_text(encoding="utf-8")).get("steps") or []:
                attempts = step.get("attempts") or []
                if attempts:
                    steps[CLOUD if attempts[-1].get("model") == cloud_model else LOCAL] += 1
        cloud_calls = [(t.get("paid_calls") or 0) + (t.get("cache_hits") or 0) for t in run.tasks]
        return {"tasks": len(run.tasks), "tasks_routed": sum(calls > 0 for calls in cloud_calls),
                "local_calls": sum(t.get("local_calls") or 0 for t in run.tasks),
                "cloud_calls": sum(cloud_calls),
                "local_steps": steps[LOCAL], "cloud_steps": steps[CLOUD]}

    def comparisons(self) -> dict:
        """The comparisons file (module docstring: outputs): every row of the configuration in its order,
        Holm over the rows that carry a p (FAMILY)."""
        statistics, rows = self.config["statistics"], []
        for row in self.config["comparisons"]:
            kind = comparison_kind(row)
            if kind == "acebench":
                rows.append(self.acebench_comparison(row))
                continue
            base = {"name": row["name"], "a": row["a"], "b": row["b"]}
            if kind == "deferred":
                rows.append({**base, "deferred": row["deferred"], "p": float(row["p"])})
                continue
            if kind == "recorded":
                rows.append({**base, "recorded": row["recorded"], "p": self.ready.recorded[row["name"]]})
                continue
            scorer = row.get(SCORER, DEFAULT_SCORER)
            base.update({SCORER: row[SCORER]} if SCORER in row else {})
            scores_a, scores_b = self.scores_of(row["a"], scorer), self.scores_of(row["b"], scorer)
            # a dry run's routed system ran every development item, the other systems the first only
            common = {s["id"] for s in scores_a} & {s["id"] for s in scores_b}
            items = [item for item in self.material.full if item["id"] in common]
            margin = row["noninferiority"][MARGIN] if kind == "noninferiority" else None
            tested = compare_systems([s for s in scores_a if s["id"] in common],
                                     [s for s in scores_b if s["id"] in common], items,
                                     seed=statistics["seed"], n=statistics["resamples"], margin=margin)
            if kind == "noninferiority":
                rows.append({**base, TEST: "noninferiority", **tested, "p": tested["noninferiority"]["p_ni"]})
            elif kind == "descriptive":
                # outside the Holm family: the paired numbers only, no test, not even nested
                described = {**tested, MCNEMAR: {k: v for k, v in tested[MCNEMAR].items() if k != "p"}}
                rows.append({**base, "descriptive": True, **described})
            else:
                bound = {"noninferiority": noninferiority(tested, row[MARGIN])} if MARGIN in row else {}
                rows.append({**base, TEST: MCNEMAR, **tested, "p": tested["mcnemar"]["p"], **bound})
        family = [row for row in rows if "p" in row]
        for row, adjusted in zip(family, holm([row["p"] for row in family]), strict=True):
            row["p_holm"] = adjusted
        out = {**self.head("single_step"), "family": FAMILY, "seed": statistics["seed"],
               "resamples": statistics["resamples"], "comparisons": rows}
        if self.references:
            out["reference"] = {"milestone": self.config["reference"]["milestone"],
                                "systems": {name: {**({"milestone": ref.milestone} if ref.milestone else {}),
                                                   "records": ref.path, "sha256": ref.sha256}
                                            for name, ref in self.references.items()}}
            if not self.dry_run:
                out["rescored"] = self.rescored()
        return out

    def acebench_comparison(self, row: dict) -> dict:
        """The ACEBench comparison (module docstring: outputs; docs/m6-计划.md §5, H3): the two runs the
        preflight read answer by answer, paired by stratum and id, under the stratified one-sided bootstrap
        of eval.metrics.stratified_noninferiority_p."""
        statistics, block = self.config["statistics"], row["acebench"]
        a, b = self.ready.acebench[row["name"]]
        strata_a, strata_b = pair_strata(a, b)
        stats = stratified_noninferiority_p(strata_a, strata_b, block["margin"], seed=statistics["seed"],
                                            n=statistics["resamples"])

        def side(run: PerItem, strata: dict) -> dict:
            return {"path": run.path, "sha256": run.sha256, "files": len(run.files), "items": run.items,
                    "mean": sum(sum(s) / len(s) for s in strata.values()) / len(strata)}

        per_stratum = [{"language": language, "category": category, "n": len(strata_a[key]),
                        "correct_a": sum(strata_a[key]), "correct_b": sum(strata_b[key]),
                        "later_candidates_a": a.headers[key]["later_candidates"],
                        "later_candidates_b": b.headers[key]["later_candidates"]}
                       for key in strata_a for language, category in [key]]
        interval = {key: stats[key] for key in ("diff", "mean", "low", "high", "n", "seed")}
        return {"name": row["name"], TEST: "acebench", "a": block["a"], "b": block["b"],
                "acebench": {"a": side(a, strata_a), "b": side(b, strata_b), "margin": stats["margin"],
                             "strata": stats["strata"], "items": stats["items"], "diff": stats["diff"],
                             "paired_ci": interval, "p_ni": stats["p_ni"], "below": stats["below"],
                             "per_stratum": per_stratum},
                "p": stats["p_ni"]}

    def rescored(self) -> dict:
        """Per reference system, whether this session's scores of its records equal, item by item, the scores
        of its results file, and the items where they do not."""
        out = {}
        for system, reference in self.references.items():
            path = self.config["reference"]["systems"][system]["results"]
            recorded = json.loads(Path(path).read_text(encoding="utf-8")).get("scores") or []
            by_id = {s.get("id"): s for s in recorded if isinstance(s, dict)}
            ours = {s["id"] for s in reference.scores}
            differ = [s["id"] for s in reference.scores
                      if canonical_json(by_id.get(s["id"])) != canonical_json(s)]
            differ += sorted(str(i) for i in by_id if i not in ours)
            out[system] = {"equal": not differ, "differ": differ}
        return out

    def resources(self) -> dict:
        prices = load_prices()
        systems = {}
        for system, run in self.runs.items():
            if system in ROUTED:
                systems[system] = self.routed_resources(run, prices)
                continue
            cloud = system in CLOUD_MODELS
            cost = None
            if cloud:
                cost = {**cloud_cost(prices[CLOUD_MODELS[system]], run.records, run.tasks),
                        "ledger_rmb": run.ledger_rmb}
            systems[system] = {"server": SERVER_OF.get(system),
                               "single_step": single_step_latency(run.items_timing, cloud=cloud),
                               "e2e_step": e2e_step_latency(run.tasks_timing, cloud=cloud), "cost": cost}
        return {"milestone": self.milestone,
                "config": {"path": self.ready.config_path, "sha256": self.ready.config_sha256},
                "git_commit": self.ready.commit, "notes": NOTES, "systems": systems, "servers": self.servers}

    def routed_resources(self, run: SystemRun, prices: dict) -> dict:
        """A routed system's latency and cost (NOTES["routed"])."""
        by = {r["id"]: answered_by(r) for r in run.records}
        calls = {kind: [c for item in run.items_timing if by.get(item["id"]) == kind for c in item["calls"]
                        if c.get("error") is None] for kind in (LOCAL, CLOUD)}
        local = {**latency(c["wall_ms"] for c in calls[LOCAL]),
                 "server": latency(ms for c in calls[LOCAL] if (ms := server_ms(c)) is not None)}
        single = {"local": local, "routed": latency(c["wall_ms"] for c in calls[CLOUD]),
                  "routed_paid": latency(c["wall_ms"] for c in calls[CLOUD] if c.get("paid"))}
        price = prices[CLOUD_MODELS[ROUTED[run.system][1]]]
        routed = cloud_cost(price, [r for r in run.records if by.get(r["id"]) == CLOUD], [])["single_step"]
        e2e_paid = sum(t.get("cost_rmb") or 0.0 for t in run.tasks)
        e2e = {"tasks": len(run.tasks), "paid_calls": sum(t.get("paid_calls") or 0 for t in run.tasks),
               "cache_hits": sum(t.get("cache_hits") or 0 for t in run.tasks), "paid_rmb": e2e_paid,
               "list_rmb": None}
        cost = {"paid_rmb": routed["paid_rmb"] + e2e_paid, "list_rmb": None,
                "per_task_fen": 100 * e2e_paid / len(run.tasks) if run.tasks else None,
                "single_step": routed, "e2e": e2e, "ledger_rmb": run.ledger_rmb}
        return {"server": SERVER_OF[run.system], "single_step": single,
                "e2e_step": e2e_step_latency(run.tasks_timing, cloud=False), "cost": cost}

    def dev_check(self) -> dict:
        """A dry run's development check (module docstring): the routed system's routed count and successes
        over the whole development set against the routing file's chosen point, passed without one; and,
        when final runs, final and finalg against the frozen file (frozen_check)."""
        routed = routed_system(self.config)
        run = None if routed is None else self.runs.get(routed)
        if run is None:
            check = {"passed": True, "routed": None, "successes": None, "expected": None}
        else:
            routed = sum(answered_by(r) == CLOUD for r in run.records)
            successes = sum(bool(s["success"]) for s in run.scores)
            expected = self.ready.frozen_point
            passed = (routed, successes) == (expected["routed"], expected["successes"])
            check = {"passed": passed, "routed": routed, "successes": successes, "expected": expected}
        if self.ready.frozen is None:
            return check
        frozen = self.frozen_check()
        return {**check, "passed": check["passed"] and frozen["passed"], "frozen": frozen}

    def frozen_check(self) -> dict:
        """The dry run's runs of final and finalg over every development item against the frozen file (module
        docstring: the dry run): per scorer the items whose success of final is not the frozen one, how many
        of its outputs are not (output_hash; a count that decides nothing), and the items whose output of
        finalg is not; passed when no success of final and no output of finalg differs."""
        frozen = self.ready.frozen
        out: dict = {"path": frozen.path, "sha256": frozen.sha256}
        passed = True
        final = self.runs.get(FINAL)
        if final is not None:
            held = {entry.get("id"): entry for entry in frozen.data[FINAL]["items"]}
            differ = {}
            for scorer in SCORERS:
                got = {s["id"]: bool(s["success"]) for s in final.scored[scorer]}
                differ[scorer] = sorted(i for i in got.keys() | held.keys()
                                        if got.get(i) != (held.get(i) or {}).get("success", {}).get(scorer))
            outputs = sum(output_hash(r) != (held.get(r["id"]) or {}).get("output_sha256")
                          for r in final.records)
            passed = passed and not any(differ.values())
            out[FINAL] = {"n": len(final.records), "differ": differ, "outputs_differ": outputs}
        greedy = self.runs.get(FINALG)
        if greedy is not None:
            held = {entry.get("id"): entry.get("output_sha256") for entry in frozen.data[FINALG]["items"]}
            got = {r["id"]: output_hash(r) for r in greedy.records}
            differ = sorted(i for i in got.keys() | held.keys() if got.get(i) != held.get(i))
            passed = passed and not differ
            out[FINALG] = {"n": len(greedy.records), "outputs_differ": differ}
        return {"path": out.pop("path"), "sha256": out.pop("sha256"), "passed": passed, **out}

    def dev_check_failure(self, check: dict) -> str:
        """What a failed development check says: the routed system's counts against the frozen point, final's
        successes and finalg's outputs against the frozen file, whichever did not pass."""
        parts = []
        expected = check["expected"]
        if expected is not None and (check["routed"], check["successes"]) != (expected["routed"],
                                                                             expected["successes"]):
            routed, path = routed_system(self.config), self.config["routing"]["path"]
            parts.append(f"{routed} routed {check['routed']} of the development items and got "
                         f"{check['successes']} right, the frozen point of {path} {expected['routed']} and "
                         f"{expected['successes']} (chosen_point)")
        frozen = check.get("frozen") or {}
        final, greedy = frozen.get(FINAL), frozen.get(FINALG)
        if final is not None and any(final["differ"].values()):
            ids = sorted({i for differ in final["differ"].values() for i in differ})
            per = ", ".join(f"{scorer}: {final['differ'][scorer]}" for scorer in SCORERS)
            parts.append(f"{FINAL}'s successes differ from {frozen['path']} on {len(ids)} of the "
                         f"{final['n']} items ({per})")
        if greedy is not None and greedy["outputs_differ"]:
            differ = greedy["outputs_differ"]
            parts.append(f"{FINALG}'s outputs differ from {frozen['path']} on {len(differ)} of the "
                         f"{greedy['n']} items ({differ})")
        return (f"the dev check failed: {'; '.join(parts)}; {rel(self.folder / 'session.json')} records it "
                "and a real session is refused")

    def finish(self) -> int:
        """Write the outputs and, in a real session, the completion markers; the exit code. A dry run whose
        development check failed raises Failed once its session.json is written."""
        results = self.folder / "results" if self.dry_run else RESULTS
        m = self.milestone.lower()
        outputs: dict[Path, dict] = {}
        for system, run in self.runs.items():
            outputs[results / f"test-{m}-{system}.json"] = self.single_step(run)
            if run.e2e:  # a system whose end_to_end is false has no end-to-end results
                outputs[results / f"test-{m}-e2e-{system}.json"] = self.end_to_end(run)
        outputs[results / f"test-{m}-comparisons.json"] = self.comparisons()
        outputs[results / f"test-{m}-resources.json"] = self.resources()
        for path, data in outputs.items():
            write_json(data, path)
        stopped = any(r["status"] == BUDGET_STOP for run in self.runs.values()
                      for r in [*run.records, *run.tasks])
        check = self.dev_check() if self.dry_run else None
        write_json({"dry_run": self.dry_run, "purpose": self.ready.purpose, "git_commit": self.ready.commit,
                    "config": {"path": self.ready.config_path, "sha256": self.ready.config_sha256},
                    "opens": self.material.opens, "sets": self.material.sets,
                    "servers": {name: {"command": self.commands.get(name), **row}
                                for name, row in self.servers.items()},
                    "outputs": [rel(path) for path in outputs], "budget_stop": stopped,
                    **({"dev_check": check} if check is not None else {})},
                   self.folder / "session.json")
        for path in outputs:
            print(f"wrote {rel(path)}")
        if check is not None and not check["passed"]:
            raise Failed(self.dev_check_failure(check))
        if not self.dry_run:
            family = family_of(self.config)
            (_, single), (_, e2e) = family.sets
            opens = [(single, self.material.opens["single_step"]), (e2e, self.material.opens["e2e"])]
            mark(opens, BUDGET_STOPPED if stopped else COMPLETE, self.machine.now, family,
                 git_commit=self.ready.commit, config_sha256=self.ready.config_sha256,
                 outputs={rel(path): sha256_of(path) for path in outputs})
            print(f"{family.completions.as_posix()}: both opens complete"
                  + ("; a cap stopped a run (its statuses show where)" if stopped else ""))
        return CAPPED if stopped else DONE


# ---- the command line -------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Open the locked test sets once and evaluate every system on "
                                             "them.")
    ap.add_argument("--config", type=Path, required=True,
                    help="the configuration, results/test-config-m3.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="the preflight and one development item and task per system; nothing is opened")
    ap.add_argument("--llama-server", default=None,
                    help=f"llama-server executable (default: ${SERVER_VAR}; one of the two is required)")
    ap.add_argument("--abandon", default=None, metavar="REASON",
                    help="record the owner's decision about the opens without a completion marker; runs "
                         "nothing")
    return ap


def main(argv: list[str] | None = None, *, popen: Callable = subprocess.Popen, run: Callable = subprocess.run,
         client: Callable[[str], LocalClient] = LocalClient, openai_client: object | None = None,
         load_render: Callable[[str], Callable] = training_renderer,
         load_counter: Callable[[str], TokenCounter | None] = token_counter,
         load_world: Callable[[], World] = recorded_world, now: Callable[[], str] = utc_now,
         clock: Callable[[], float] = time.perf_counter,
         load_family: Callable[[str], LoadedFamily] = load_family) -> int:
    """The keyword arguments stand in for this machine in tests (Machine): the processes (llama-server and
    nvidia-smi), the local and the cloud clients, the chat template, the token counter, the recorded world,
    the clocks and the family of a family-path server's tokenizer folder."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.abandon is not None and (args.dry_run or not args.abandon.strip()):
        parser.error("--abandon takes the owner's reason and no --dry-run: it records a decision and runs "
                     "nothing")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"FAIL: {exc}")
        return MALFORMED
    machine = Machine(popen, run, client, openai_client, load_render, load_counter, load_world, now, clock,
                      load_family)
    family, opens_before = family_of(config), None
    try:
        if args.abandon is not None:
            check_root()
            return abandon(config, args.abandon, now)
        binary = prerequisites(config, args.llama_server)
        generated = generations(config, machine.load_family)
        ready = preflight(config, args.config, run=run, client=client, dry_run=args.dry_run,
                          generated=generated)
        setup = prepare(config, machine, generated)
        if args.dry_run:
            material = development_material(dev_sets_of(config))
        else:
            opens_before = len(LockedSet(*family.sets[0]).log_entries())
            material = open_test_sets(config, ready, now)
        session = Session(config, ready, setup, material, machine, binary, dry_run=args.dry_run)
        session.run()
        return session.finish()
    except ConfigError as exc:  # a generation the family cannot run (generations): before anything is opened
        print(f"FAIL: {exc}")
        return MALFORMED
    except Missing as exc:
        print(f"FAIL: {exc}")
        return USAGE
    except Refused as exc:
        print(f"FAIL: {exc}")
        return REFUSED
    except Unfinished as exc:
        print(f"FAIL: {exc}")
        return UNFINISHED
    except Failed as exc:
        print(f"FAIL: {exc}")
        return used_up(opens_before, args.config, family)
    except Exception as exc:  # noqa: BLE001  deliberate: unforeseen (an HTTP error, a full disk) is a failure
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return used_up(opens_before, args.config, family)


def used_up(opens_before: int | None, config: Path, family: TestFamily = DEFAULT_SETS) -> int:
    """After a failure: say whether this session's open of the family's sets is used up and what the owner
    does next; FAILED."""
    try:
        now_held = len(LockedSet(*family.sets[0]).log_entries())
        opened = opens_before is not None and now_held > opens_before
    except (LockError, JsonlError):
        opened = opens_before is not None  # the log cannot be read: assume the worst
    if opened:
        print("the open is used up and has no completion marker: the next session refuses to start until "
              f"the owner decides (scripts/run_test.py --config {config.as_posix()} --abandon "
              "\"<reason>\"); the records so far are under runs/")
    else:
        print("nothing was opened")
    return FAILED


if __name__ == "__main__":
    sys.exit(main())
