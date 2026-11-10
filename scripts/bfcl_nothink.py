"""BFCL with Qwen3 in non-thinking mode: BFCL's stock Qwen FC handler plus an empty thinking block.

The stock handler of "Qwen/Qwen3-1.7B-FC" renders the prompt the way Qwen3's chat template does in thinking
mode: it ends with the generation prompt and the model opens a thinking block of its own (S5 in
docs/m0-验证.md, about 3.6 s per question). This project trains and serves the model in non-thinking mode
(enable_thinking=False), where the template puts an empty thinking block right after the generation prompt.
The subclass built here appends that block and changes nothing else, as docs/plan.md specifies. It is
registered under two names, one per llama-server instance: BFCL cannot send llama.cpp's per-request "lora"
field, so the base model and the fine-tuned adapter are served by separate servers (scripts/serve_local.py
without and with --adapter), and the two names keep their results apart. BFCL turns "_" in a result folder
name back into "/", so the names contain no underscore.

The script runs in the BFCL environment (.venv-bfcl, see S5), not the project one. The first argument is
"self-check" or a BFCL command; the remaining arguments go to BFCL's own command line:

  .venv-bfcl/Scripts/python.exe scripts/bfcl_nothink.py self-check
  .venv-bfcl/Scripts/python.exe scripts/bfcl_nothink.py generate --model Qwen/Qwen3-1.7B-FC-nothink
      --test-category python --skip-server-setup --local-model-path models/Qwen3-1.7B
  .venv-bfcl/Scripts/python.exe scripts/bfcl_nothink.py evaluate --model Qwen/Qwen3-1.7B-FC-nothink
      --test-category python

Set for the command: BFCL_PROJECT_ROOT, an absolute path (BFCL writes result/ and score/ there; with a
relative path its generate and evaluate steps resolve it differently), REMOTE_OPENAI_BASE_URL (the server,
e.g. http://127.0.0.1:8080/v1) and PYTHONUTF8=1 (BFCL prints emoji, which cp1252 cannot encode once the
output is redirected). BFCL's "python" category group is the plan's set: simple_python, multiple, parallel,
parallel_multiple, irrelevance and the six live categories. Start the server with -np 1 for one request at a
time (each answer then does not depend on what else was batched with it). BFCL itself sends up to 100
requests at once, so its latency and cost columns count queueing time and are not used; the project measures
latency in its own harness.

self-check renders the questions of those categories through BFCL's own loader with the stock handler, with
the subclass and with the local tokenizer's chat template (enable_thinking=False). The subclass must equal the
stock prompt plus the block on every question and the template's rendering on the first simple_python
question. It reports, without failing, the stock handler's one known difference from the template, which the
subclass keeps so that base and fine-tuned runs see identical prompts: tool descriptions are written with
json.dumps' ASCII escaping, so non-ASCII text in them reaches the model as backslash-u escapes. It also
reports the longest prompt in tokens, which sets the server's context size (prompt plus BFCL's 4,096-token
reply budget). Exit code 1 on a mismatch.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Callable, MutableMapping

STOCK = "Qwen/Qwen3-1.7B-FC"
GENERATION_PROMPT = "<|im_start|>assistant\n"
NO_THINK = "<think>\n\n</think>\n\n"  # what Qwen3's template writes after the generation prompt, thinking off
MODELS = {  # registered name -> what the server behind it holds
    "Qwen/Qwen3-1.7B-FC-nothink": "base model",
    "research-agent/Qwen3-1.7B-FT-FC-nothink": "fine-tuned adapter",
}
CATEGORY_GROUP = "python"
MODEL_DIR = "models/Qwen3-1.7B"


def no_think(prompt: str) -> str:
    """prompt, which must end with the generation prompt, followed by the empty thinking block."""
    if not prompt.endswith(GENERATION_PROMPT):
        raise ValueError(f"the prompt does not end with the generation prompt: ...{prompt[-40:]!r}")
    return prompt + NO_THINK


def nothink_handler(base: type) -> type:
    """A subclass of BFCL's Qwen FC handler whose prompts end with the empty thinking block."""

    class Handler(base):
        def _format_prompt(self, messages, function):
            return no_think(super()._format_prompt(messages, function))

        # BFCL's handlers have the overrides package's metaclass, which refuses an overriding method without
        # the mark that its @override decorator sets. The decorator itself looks the base class up by name in
        # module globals and cannot find one passed in as an argument, so the mark is set directly.
        _format_prompt.__override__ = True

    Handler.__name__ = Handler.__qualname__ = f"{base.__name__}NoThink"
    return Handler


def register(mapping: MutableMapping, handler: type, *, replace: Callable = dataclasses.replace) -> list[str]:
    """Add MODELS to BFCL's model mapping: copies of the stock Qwen3-1.7B FC config with handler."""
    stock = mapping[STOCK]
    for name, served in MODELS.items():
        display = f"{stock.display_name} non-thinking {served}"
        if "_" in name:
            raise ValueError(f"BFCL reads '_' in result folder names as '/': {name!r}")
        if "," in display:  # its score tables are written with ",".join(row), nothing quoted
            raise ValueError(f"a ',' in a display name shifts every column of BFCL's score tables: "
                             f"{display!r}")
        mapping[name] = replace(stock, display_name=display, model_handler=handler)
    return list(MODELS)


def first_turn_prompt(handler, entry: dict) -> str:
    """The prompt BFCL's generation sends for a single-turn entry, built the way its inference loop does."""
    data = handler._pre_query_processing_prompting(entry)
    data = handler.add_first_turn_message_prompting(data, entry["question"][0])
    return handler._format_prompt(data["message"], data["function"])


def self_check() -> int:
    from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
    from bfcl_eval.utils import load_dataset_entry
    from transformers import AutoTokenizer

    ours_cls = nothink_handler(QwenFCHandler)
    # built without __init__, so no client exists and nothing is sent
    stock, ours = QwenFCHandler.__new__(QwenFCHandler), ours_cls.__new__(ours_cls)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    ok = True
    counts, non_ascii, longest = {}, 0, (0, "")
    for category in TEST_COLLECTION_MAPPING[CATEGORY_GROUP]:
        entries = load_dataset_entry(category)
        counts[category] = len(entries)
        for entry in entries:
            plain = first_turn_prompt(stock, json.loads(json.dumps(entry)))  # the handlers may edit entries
            mine = first_turn_prompt(ours, json.loads(json.dumps(entry)))
            if mine != plain + NO_THINK:
                print(f"MISMATCH {entry['id']}: the subclass changed more than the thinking block")
                ok = False
            if any(not s.isascii() for s in (json.dumps(f, ensure_ascii=False) for f in entry["function"])):
                non_ascii += 1
            tokens = len(tokenizer(mine)["input_ids"])
            longest = max(longest, (tokens, entry["id"]))

    entry = load_dataset_entry("simple_python")[0]
    template = tokenizer.apply_chat_template(entry["question"][0], tools=entry["function"], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    same = first_turn_prompt(ours, json.loads(json.dumps(entry))) == template
    print(f"questions: {sum(counts.values()):,} ({', '.join(f'{c} {n}' for c, n in counts.items())})")
    print(f"subclass = stock prompt + empty thinking block on every question: {'yes' if ok else 'NO'}")
    print(f"subclass = chat template (enable_thinking=False) on {entry['id']}: {'yes' if same else 'NO'}")
    print(f"questions whose tool descriptions contain non-ASCII text (escaped by the stock handler): "
          f"{non_ascii:,}")
    print(f"longest prompt: {longest[0]:,} tokens ({longest[1]}); server context needed with BFCL's "
          f"4,096-token reply budget: {longest[0] + 4096:,}")
    return 0 if ok and same else 1


def main(argv: list[str]) -> int:
    if argv[:1] == ["self-check"]:
        return self_check()
    from bfcl_eval.__main__ import cli
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

    register(MODEL_CONFIG_MAPPING, nothink_handler(QwenFCHandler))
    sys.argv = [sys.argv[0], *argv]
    return cli()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
