"""Apply the local patches ACEBench needs to evaluate a self-hosted OpenAI-compatible model on Windows.

1. model_inference/model_infer.py: make the module-level vllm import optional (vllm is Linux-only).
2. model_inference/apimodel_inference.py: models named "local-*" read LOCAL_OPENAI_BASE_URL / LOCAL_OPENAI_API_KEY.
3. model_inference/inference_map.py: register the local model names as API models (LOCAL_MODELS): the M4 pair,
   which Qwen3-1.7B keeps without thinking, then a base and a lora name per registered model family
   (scripts/run_acebench.py, registered_models; M7, docs/m7-计划.md WP-L).
4. category.py: add a one-file group "smoke_single_function" for quick runs.

Every edit must match exactly once, so an upstream change stops the script instead of half-patching.
Run from the repository root: python scripts/spikes/patch_acebench.py runs/third_party/ACEBench
On a clean clone (upstream 56dd66c), its git diff is third_party_patches/acebench.diff byte for byte
(tests/eval/test_run_acebench.py checks it on a scratch copy of the upstream files).
"""

from __future__ import annotations

import sys
from pathlib import Path

NL = chr(10)
LOCAL_MODELS = [
    "local-qwen3-1.7b-base", "local-qwen3-1.7b-lora",
    "local-qwen3-hybrid-base", "local-qwen3-hybrid-lora",
    "local-qwen3-instruct-2507-base", "local-qwen3-instruct-2507-lora",
    "local-qwen3-thinking-2507-base", "local-qwen3-thinking-2507-lora",
    "local-qwen3.5-4b-base", "local-qwen3.5-4b-lora",
    "local-granite-4.2-3b-base", "local-granite-4.2-3b-lora",
    "local-nanbeige4-thinking-2511-base", "local-nanbeige4-thinking-2511-lora",
    "local-nanbeige4.1-3b-base", "local-nanbeige4.1-3b-lora",
    "local-minicpm5-2b-base", "local-minicpm5-2b-lora",
    "local-spark-x2.5-4b-base", "local-spark-x2.5-4b-lora",
    "local-lfm2.5-2.6b-base", "local-lfm2.5-2.6b-lora",
]


def patch(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {n}: {old[:60]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="")
    print(f"patched {path}")


def main(root: Path) -> None:
    patch(root / "model_inference" / "model_infer.py",
          "from vllm import LLM, SamplingParams" + NL,
          "try:" + NL
          + "    from vllm import LLM, SamplingParams" + NL
          + "except ImportError:  # vllm is Linux-only; API-based evaluation does not need it" + NL
          + "    LLM = SamplingParams = None" + NL)
    patch(root / "model_inference" / "apimodel_inference.py",
          '        elif "o1" in self.model_name:' + NL,
          '        elif self.model_name.startswith("local-"):' + NL
          + '            api_key = os.getenv("LOCAL_OPENAI_API_KEY", "EMPTY")' + NL
          + '            base_url = os.getenv("LOCAL_OPENAI_BASE_URL")' + NL
          + '        elif "o1" in self.model_name:' + NL)
    imap = root / "model_inference" / "inference_map.py"
    patch(imap, "    APIModelInference: [" + NL,
          "    APIModelInference: [" + NL + "".join(f'        "{m}",' + NL for m in LOCAL_MODELS))
    # 4. A one-file group for smoke runs (generate.py only accepts group names).
    patch(root / "category.py", "ACE_DATA_CATEGORY = {" + NL,
          "ACE_DATA_CATEGORY = {" + NL
          + '    "smoke_single_function": ["normal_single_turn_single_function"],' + NL)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
