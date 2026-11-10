"""Qwen3 in full precision with transformers, the counterpart of the llama-server path for the M3 comparison
of the HF model with the quantized one (docs: the M3 plan, phase 3 and default 7: the development set at
lora scale 1 and at scale 0, which separates the base model's quantization error from the adapter's
conversion error).

HFClient answers complete() as LocalClient does, so it serves under LocalModel (research_agent.agent.models):
the prompt is the one LocalModel renders (train.samples.prompt_renderer, the model's own chat template with
thinking off, the template call of llm.local.hf_renderer), and LocalModel's own parsing reads the completion
into a turn. Only the model differs: load_client loads the model folder (models/Qwen3-1.7B) in bfloat16 on the
GPU with sdpa attention, as scripts/train.py loads it, and a PEFT LoRA adapter folder on top when given (a
training run's adapter/ or one of its trainer checkpoint folders). The adapter is never merged into the
weights: lora scale 1.0 applies it whole, and 0.0 runs inside PEFT's disable_adapter(), where every adapted
layer runs its base layer alone. As on llama-server (LocalClient), a model with an adapter needs the scale
given explicitly, and a scale above 0 without an adapter is refused, so that a run labelled with the adapter
cannot measure the base model. A GGUF adapter is llama-server's form and is refused (check_adapter): this
backend loads the folder it was converted from.

Decoding follows LocalClient's requests to llama-server: greedy (temperature 0.0; the model folder's sampling
defaults are unset, never inherited), at most MAX_TOKENS new tokens, the prompt tokenized as it is with no
special token added (Qwen3 has no BOS) and evaluated from scratch. Generation ends at the model's
end-of-generation tokens (its generation config's eos_token_id: <|im_end|> and <|endoftext|> for Qwen3),
where llama-server ends it too. The text is the new tokens decoded without the special ones, as llama-server
renders a completion (<tool_call> and <think> are not special in Qwen3's tokenizer and stay), cut before the
first stop string. llama-server also stops generating at a stop string; here a stop string that is no end
token only cuts the text, which comes out the same, and completion_tokens then counts the tokens generated
after it. finish_reason is "stop" or "length" as llama-server reports it; the usage holds prompt_tokens,
completion_tokens (the end token counted, as llama-server counts it) and total_tokens. There are no
log-probabilities and no server timings (last_timings stays None).

Nothing here imports torch, transformers or peft at module level: load_client imports them, and the tests run
HFClient over fake model and tokenizer objects.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

from ..train.manifest import cuda_device_name, package_version
from .local import MAX_TOKENS, Completion

DTYPE = "bfloat16"
DEVICE = "cuda:0"
ATTENTION = "sdpa"  # scripts/train.py's attn_implementation
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
VERSIONED = ("torch", "transformers", "peft")  # the packages whose versions props() reports


class AdapterError(ValueError):
    """An adapter this backend cannot load: no PEFT folder with safetensors weights (a GGUF included)."""


def check_adapter(path: str | Path) -> Path:
    """path as a PEFT LoRA adapter folder, which holds adapter_config.json and adapter_model.safetensors.

    AdapterError for a GGUF (llama-server's form of an adapter) or any other file, a path that is not there,
    and a folder without those two files (one with the older adapter_model.bin only included).
    """
    folder = Path(path)
    if folder.suffix.lower() == ".gguf":
        raise AdapterError(f"{path} is a GGUF adapter, llama-server's form; the hf backend loads the PEFT "
                           "folder it was converted from (runs/train/<run>/adapter or a trainer checkpoint "
                           "folder)")
    if folder.is_file():
        raise AdapterError(f"{path} is a file, not a PEFT adapter folder")
    if not folder.is_dir():
        raise AdapterError(f"adapter folder {path} not found")
    missing = [name for name in (ADAPTER_CONFIG, ADAPTER_WEIGHTS) if not (folder / name).is_file()]
    if missing:
        raise AdapterError(f"{path} has no {' and no '.join(missing)}: not a PEFT adapter folder with "
                           "safetensors weights")
    return folder


def _ints(row: Any) -> list[int]:
    """A row of token ids as a list: a tensor's tolist(), a list (the tests' fakes) as it is."""
    return row.tolist() if hasattr(row, "tolist") else [int(t) for t in row]


class HFClient:
    """A loaded model and its tokenizer, answering complete() as LocalClient does (module docstring).

    model is a transformers causal LM, or PEFT's wrapper over one when adapter names the folder it was loaded
    from; tokenizer is the model folder's. load_client builds both; the tests pass fakes. model_dir, device,
    dtype and gpu are what props() reports.
    """

    def __init__(self, model: Any, tokenizer: Any, *, adapter: str | Path | None = None,
                 model_dir: str | Path | None = None, device: str = DEVICE, dtype: str = DTYPE,
                 gpu: str | None = None) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.adapter = None if adapter is None else Path(adapter)
        self.model_dir = None if model_dir is None else Path(model_dir)
        self.device, self.dtype, self.gpu = device, dtype, gpu
        eos = model.generation_config.eos_token_id
        self.end_ids = [eos] if isinstance(eos, int) else list(eos or [])
        if not self.end_ids:
            raise ValueError("the model's generation config names no end-of-generation token (eos_token_id)")
        # LocalClient keeps llama.cpp's timings of its latest completion here (eval.timing reads them);
        # transformers reports none.
        self.last_timings: dict | None = None

    def complete(self, prompt: str, *, lora_scale: float | None = None, max_tokens: int = MAX_TOKENS,
                 temperature: float = 0.0, stop: list[str] | None = None) -> Completion:
        """prompt's completion (module docstring). lora_scale as LocalClient takes it: 0.0 the base model, 1.0
        the adapter, None only for a model without one. ValueError, before anything is generated, for another
        scale, a scale this model's adapter (or its lack of one) does not allow, or a temperature other than
        0.0."""
        if temperature != 0.0:
            raise ValueError(f"the hf backend decodes greedily: temperature 0.0, not {temperature:g}")
        if lora_scale is None and self.adapter is not None:
            raise ValueError("this model has a LoRA adapter loaded: pass lora_scale=0.0 (base) or 1.0 "
                             "explicitly")
        if lora_scale not in (None, 0.0, 1.0):
            raise ValueError(f"lora_scale {lora_scale:g}: the hf backend applies the adapter whole (1.0) or "
                             "disables it (0.0)")
        if lora_scale == 1.0 and self.adapter is None:
            raise ValueError("lora_scale 1 needs an adapter, and this model has none loaded (load_client "
                             "with an adapter folder loads one)")
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        prompt_ids = _ints(encoded["input_ids"][0])
        base_only = lora_scale == 0.0 and self.adapter is not None
        t0 = time.perf_counter()
        with self.model.disable_adapter() if base_only else contextlib.nullcontext():
            output = self.model.generate(**encoded, max_new_tokens=max_tokens, do_sample=False, num_beams=1,
                                         temperature=None, top_p=None, top_k=None,
                                         eos_token_id=list(self.end_ids))
        wall_ms = (time.perf_counter() - t0) * 1000
        new_ids = _ints(output[0])[len(prompt_ids):]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        cuts = [at for at in (text.find(s) for s in stop or [] if s) if at >= 0]
        if cuts:
            text = text[:min(cuts)]
        ended = bool(new_ids) and new_ids[-1] in self.end_ids
        usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(new_ids),
                 "total_tokens": len(prompt_ids) + len(new_ids)}
        return Completion(text=text, finish_reason="stop" if ended or cuts else "length",
                          wall_ms=round(wall_ms, 1), raw={"usage": usage})

    def props(self) -> dict:
        """What a run manifest records under server_props in place of llama-server's /props: the backend, the
        model folder, dtype, device, GPU and attention, and the versions of torch, transformers and peft as
        the training manifest records them ("absent" for one that is not installed)."""
        return {"backend": "hf", "model_path": None if self.model_dir is None else self.model_dir.as_posix(),
                "dtype": self.dtype, "device": self.device, "gpu": self.gpu, "attn_implementation": ATTENTION,
                "versions": {name: package_version(name) for name in VERSIONED}}

    def health(self) -> bool:
        """True while the model is loaded (until close)."""
        return self.model is not None

    def close(self) -> None:
        """Lets the model and the tokenizer go."""
        self.model = self.tokenizer = None


def load_client(model_dir: str | Path, adapter: str | Path | None = None) -> HFClient:
    """An HFClient over the model folder's weights in bfloat16 on the GPU, with the PEFT adapter folder on top
    when given, and the folder's tokenizer (module docstring). Local files only: the folders are checked
    before anything heavy is imported (AdapterError from check_adapter; FileNotFoundError for a missing model
    folder, which transformers would otherwise take for a Hub repository id); RuntimeError when torch sees no
    GPU."""
    folder = None if adapter is None else check_adapter(adapter)
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(f"model folder {model_dir} not found")
    import torch  # heavy imports, only when a model is loaded
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("the hf backend runs on CUDA, and torch sees no GPU")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    # device_map puts the weights straight on the GPU instead of staging a copy in system RAM, as
    # scripts/train.py loads them
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, DTYPE),
                                                 attn_implementation=ATTENTION, device_map=DEVICE,
                                                 local_files_only=True)
    if folder is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, folder)  # inference mode; the adapter stays its own modules
    model.eval()
    return HFClient(model, tokenizer, adapter=adapter, model_dir=model_dir, gpu=cuda_device_name())
