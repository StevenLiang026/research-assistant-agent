"""The registered model families (docs/m7-家族层设计.md §2, 登记表) and how a model folder's family is
found.

Ten families: the three Qwen3 families of the bake-off's first wave (docs/m7-计划.md, 候选), qwen3-hybrid
(Qwen3-1.7B, FT2's base, and Qwen3-4B, C0), qwen3-instruct-2507 (C1) and qwen3-thinking-2507 (C2); and the
seven of the later waves, qwen3.5-4b (C3), nanbeige4-thinking-2511 (C4), nanbeige4.1-3b (C5), minicpm5-2b
(C6), spark-x2.5-4b (C7), granite-4.2-3b (C8) and lfm2.5-2.6b (C9).

Template hashes. Each family pins the SHA-256 of its chat template: the lowercase hex SHA-256 of the
UTF-8 bytes of the template text the tokenizer exposes as tokenizer.chat_template after
AutoTokenizer.from_pretrained, or, when that is empty, of chat_template.jinja's text as transformers reads
it (UTF-8, universal newlines); that is core.template_hash. The values below were computed from the local
model folders on 2026-10-19 with transformers 5.18 (the Qwen3 trio) and on 2026-10-20 (the seven; their
HF revisions are in runs/bakeoff/m7/downloads/<folder>.json). models/Qwen3-1.7B and models/Qwen3-4B expose
byte-identical templates, so qwen3-hybrid pins one hash; every other family one each. A folder whose
template hashes to anything else belongs to no registered family (FamilyError): a template that changes
upstream cannot change training or inference text unnoticed.

Resolution (resolve): the family whose pinned hashes hold the folder's template hash and whose
architectures hold all of config.json's. The three Qwen3 families share one architecture,
Qwen3ForCausalLM, and the two Nanbeige and MiniCPM5 share LlamaForCausalLM, so the template tells them
apart. A name given (the --family option) must be the resolved family's.

What the Qwen3 trio share (登记表, 三者): no BOS, "<|im_end|>" as the turn terminator and the one stop
string (agent.models.STOP), Qwen3's JSON tool calls, think_close "\\n</think>\\n\\n", and LoRA on the seven
projections scripts/train.py targets.

The seven, as their templates were probed (tests/family/test_golden.py holds every fact against the real
templates; tests/family/test_core.py pins them):

- qwen3.5-4b (Qwen3_5ForConditionalGeneration; config.json names the multimodal class, and
  AutoModelForCausalLM instantiates the text model Qwen3_5ForCausalLM from it): hybrid, thinks unless
  enable_thinking is False; its render of a target always holds the block, empty or with the reasoning
  trimmed. Qwen3-Coder XML calls. Its template iterates the arguments and so refuses raw text: inline. LoRA
  as one regex over the language model's layers alone: the attention projections of its 8 attention
  layers, in_proj_qkv, in_proj_z and out_proj of its 24 linear-attention layers (not in_proj_a and
  in_proj_b, the decay and gate vectors) and the three MLP projections, and nothing of the vision tower or
  an MTP head, which the multimodal class carries under other names. No generation_config.json in the
  folder: the model card's thinking settings for general tasks, temperature 1.0, top_p 0.95, top_k 20,
  min_p 0, presence_penalty 1.5.
- granite-4.2-3b (GraniteForCausalLM): hybrid; thinking on by default, enable_thinking False gives
  "<|im_start|>assistant\\n<think></think>". Qwen3-Coder XML calls, with "\\n" after each block (so a
  completion with calls ends "...</tool_call>\\n<|im_end|>") and "\\n" after the empty block before a
  call. Its THINK render writes "<think>\\n" + reasoning + "\\n</think>\\n" and the turn's text as one
  string with its end trimmed (`content | trim`), then the newline before the calls: "\\n</think>\\n" is
  the close before a text answer, while a turn of calls alone has the close trimmed to "\\n</think>" and
  the calls' newline after it, "</think>\\n<tool_call>", where reasoning + think_close + the NO_THINK
  completion would hold one newline more. The family declares that join (think_join "trim", core: THINK
  samples), so its THINK samples are the template's own renders and reasoning_field is declared, the
  template's render cross-checked against every THINK sample; a THINK sample's text after the reasoning
  still starts with think_close whenever the answer is not empty, what a forced close appends. Inline raw
  text (the template iterates the arguments). The model card: temperature 1.0, top_p 0.95 (its
  generation_config.json's).
- nanbeige4-thinking-2511 and nanbeige4.1-3b (LlamaForCausalLM, SentencePiece tokenizers; the templates
  differ by 4.1's keep_all_think switch): always thinking; the template's generation prompt ends at
  "<|im_start|>assistant\\n" and the family appends "<think>\\n", while its render of a target after the
  last user query holds the block, empty or with the reasoning trimmed. Hermes JSON calls, raw text printed
  verbatim (string). "<|im_start|>" is the BOS token and opens every turn (turn). The model cards:
  temperature 0.6, top_p 0.95. The Thinking-2511 tokenizer is legacy (tokenizer_config.json "legacy":
  true): it encodes a space token after every added token ("<think>", "<tool_call>", "<|im_start|>"), so
  its decode of a sample's ids is not the sample's text (4.1's is; the two share one tokenizer.json), which
  scripts/train.py's mask check refuses. Both tokenizers write a word-start marker "▁" at the start of a
  text encoded on its own, which no sample holds after the reasoning: a forced close takes the close's ids
  from core.LoadedFamily.encode_close.
- minicpm5-2b (LlamaForCausalLM): hybrid; without enable_thinking the generation prompt holds no block,
  with True "<think>\\n", with False the empty block; its render of a target holds the block as Qwen3's
  does. MiniCPM XML calls with CDATA. Its eos_token is "</s>" while every turn ends with "<|im_end|>": the
  terminator is turn_end. The template writes the BOS "<s>" once at the start (template). "<function",
  "</function>", "<param", "</param>", "<tool_call>" and "<tools>" are special tokens, which llama-server
  leaves out of its text: text_from_ids. Inline raw text (the template calls .items() on the arguments).
  The model card and generation_config.json: temperature 1.0, top_p 0.95, min_p 0.
- spark-x2.5-4b (Spark2_5ForCausalLM, vendor code under trust_remote_code: modeling_spark.py and
  configuration_spark.py reviewed on 2026-10-19 and pinned by SHA-256, with the HF revision): hybrid;
  "<|Bot|><think>" opens the thinking, "<|Bot|></think>" the answer, nothing between the close and the
  answer. The template prepends its default system text and the tools block before the system message.
  GLM-4.5 key-value calls, nothing between blocks. "<｜start▁of▁sentence｜>" is the BOS and opens every
  turn (turn); "<｜end▁of▁sentence｜>" (the eos) ends every turn. The template refuses arguments that are
  not a mapping: inline. LoRA on the fused q_k_v_proj, out_proj and the three MLP projections, not the
  head-wise gate g_proj. generation_config.json: temperature 1.0, top_p 0.95, top_k -1 (off). The template
  writes the reasoning as given, without stripping a newline at either end (the Qwen-style templates
  strip), so a sidecar reasoning with one would fail the training path's cross-check; the sidecars hold
  none. The pinned modeling_spark.py was written for transformers 4.57 (config.json's
  transformers_version), and transformers 5.18 cannot build or run it as it stands; the family mends it at
  load, the files as pinned (core: Model patches, Module adapters), so that it is built, trained under QLoRA
  and run by the HF backend as every other family is (tests/family/test_compat.py, tests/train/
  test_train_cli.py and tests/eval/test_fidelity.py run a two-layer Spark with the real vocabulary through
  build_causal_lm on the CPU, forward and backward). In turn:
  1. It declares its tied weights as a list, _tied_weights_keys = ["lm_head.weight"], which 5.18 reads as
     a mapping from each tied parameter to its source whenever tie_word_embeddings is on, as config.json
     has it: as written, no construction of the model succeeds. The family patches the class at load
     instead of changing the file (core: Model patches): the mapping lm_head.weight ->
     model.embedding.weight, the embedding being Spark2_5Model.embedding (the checkpoint holds
     model.embedding.weight and no lm_head.weight, and the vendor's forward computes the logits with the
     embedding's weight when the two are tied).
  2. Its attention is eager only: Spark2_5Attention.forward calls the module's own eager_attention_forward
     whatever the setting, and the class declares no SDPA support, so 5.18 refuses "sdpa", the attention
     every other family is built with, when it builds the class. The family declares "eager" (core:
     Attention). With both, the model builds, its head tied to the embedding and the files as pinned
     (tests/family/test_lora_targets.py).
  3. Spark2_5Model.forward calls create_causal_mask and create_sliding_window_causal_mask with
     input_embeds= and cache_position=; 5.18's functions take the first as inputs_embeds= and the second
     not at all: as written, the first forward pass raises TypeError. The family adapts the module's two
     names at load (compat: masking_4_57): wrappers of the installed functions that pass input_embeds on as
     inputs_embeds and drop cache_position once it is checked to be the query positions 5.18 derives from
     the cache, so the masks are the ones 4.57 made for the same call (causal, the sliding window, the
     padding), bit for bit.
  4. Spark2_5DecoderLayer.forward casts its input to the dtype of mlp.gate_proj's weight before the
     attention and the MLP, which under bitsandbytes NF4 (QLoRA's base, the bake-off's training and S6) is
     the packed storage, uint8: the forward pass fails at the MLP's activation (no 5.18 change: the cast
     fails under any release). The family replaces the method at load (compat: spark_projection_dtype) with
     the vendor's own source but for that cast, which goes to the layer's compute dtype (bfloat16) when
     its weight is quantized and to the weight's dtype otherwise, as before.
- lfm2.5-2.6b (Lfm2ForCausalLM): a pure reasoning model (its model card); the generation prompt always ends
  "<|im_start|>assistant\\n<think>" and its render of a target with reasoning is "<think>" + reasoning +
  "</think>" + the answer, without reasoning the answer alone: always thinking, think_close "</think>",
  and the empty block constructed. Python-style calls in one list per step. The template writes the BOS
  "<|startoftext|>" once (template); tool messages are turns of their own. "<|tool_call_start|>" and
  "<|tool_call_end|>" are added tokens the tokenizer does not call special, but llama.cpp's converter marks
  every <|...|> token as a control token (the Qwen3.5 and Granite GGUFs in models/ show <|fim_prefix|>,
  special false in their tokenizer.json, as type CONTROL), so llama-server leaves them out of its text:
  text_from_ids. The template refuses string arguments: inline. LoRA on the attention projections of its 8
  attention layers (q, k, v, out_proj), in_proj and out_proj of its 22 short-convolution layers and w1, w2,
  w3 of every feed-forward block. generation_config.json (and the model card): temperature 0.1, top_k 50,
  repetition_penalty 1.1, sent as llama-server's repeat_penalty with repeat_last_n, its window, set to
  every request's prompt and max_tokens, so that it covers the whole sequence, prompt included, at every
  step, as transformers applies it (core.Decoding: The penalty's window; the server's own window is the
  last 64 tokens). Like Spark's, the template writes the reasoning as given, newlines at its ends included.

Decoding (docs/m7-计划.md §2 and docs/m7-底座对比预登记.md, 解码): thinking samples with the vendor's
settings, listed per family above and, for the Qwen3 trio, temperature 0.6, top_p 0.95, top_k 20, min_p 0
(what the generation_config.json of each Qwen3 folder that thinks holds and what Qwen3-4B-Thinking-2507's
model card recommends). A family that always thinks samples in both modes; a hybrid family answers at once
greedily (the bake-off's one no-think run per hybrid family, and qwen3-hybrid's G2 setting, whose requests
are today's bytes); qwen3-instruct-2507 never thinks and is greedy. Every family that thinks has THINK as
its inference mode except qwen3-hybrid, which keeps NO_THINK for the FT2 records. These are each mode's
defaults; a run may still decode a thinking mode greedily, as the bake-off's one greedy thinking run per
family does.
"""

from __future__ import annotations

from collections.abc import Sequence

from .compat import MASKING_4_57, SPARK_PROJECTION_DTYPE
from .core import GREEDY, Decoding, Family, FamilyError, Mode, Thinking
from .toolformats.glm_kv import GLM_KV
from .toolformats.minicpm_xml import MINICPM_XML
from .toolformats.pythonic import PYTHONIC
from .toolformats.qwen_json import QWEN_JSON
from .toolformats.qwen_xml import QWEN_XML

_ASSISTANT = "<|im_start|>assistant\n"
_THINK_OPEN, _THINK_CLOSE = "<think>\n", "\n</think>\n\n"
_TURN_END = "<|im_end|>"
_LLAMA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_QWEN3 = {
    "architectures": ("Qwen3ForCausalLM",),
    "think_close": _THINK_CLOSE,
    "turn_end": _TURN_END,
    "stop": (_TURN_END,),
    "bos": "none",
    "tool_format": QWEN_JSON,
    "raw_arguments": "string",
    "tool_call_ids": False,
    "lora_targets": _LLAMA_TARGETS,
}
_THINKING_SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)  # module docstring: Decoding
_SWITCH = {Mode.THINK: {"enable_thinking": True}, Mode.NO_THINK: {"enable_thinking": False}}
_NO_SWITCH = {Mode.THINK: {}, Mode.NO_THINK: {}}
_QWEN_STYLE_PROMPTS = {Mode.THINK: _ASSISTANT + _THINK_OPEN,
                       Mode.NO_THINK: _ASSISTANT + _THINK_OPEN + _THINK_CLOSE}
_NANBEIGE_SAMPLING = Decoding(temperature=0.6, top_p=0.95)
_LFM25_SAMPLING = Decoding(temperature=0.1, top_k=50, repetition_penalty=1.1)

QWEN3_HYBRID = Family(
    name="qwen3-hybrid",
    template_sha256=frozenset({"a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"}),
    thinking=Thinking.HYBRID,
    template_kwargs=_SWITCH,
    generation_prompt=_QWEN_STYLE_PROMPTS,
    think_open=_THINK_OPEN,
    reasoning_field="reasoning_content",
    decoding={Mode.THINK: _THINKING_SAMPLING, Mode.NO_THINK: GREEDY},
    **_QWEN3,
)
QWEN3_INSTRUCT_2507 = Family(
    name="qwen3-instruct-2507",
    template_sha256=frozenset({"64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326"}),
    thinking=Thinking.NONE,
    template_kwargs={Mode.NO_THINK: {}},  # its template has no thinking switch
    generation_prompt={Mode.NO_THINK: _ASSISTANT},
    think_open=None,
    reasoning_field=None,
    decoding={Mode.NO_THINK: GREEDY},
    **_QWEN3,
)
QWEN3_THINKING_2507 = Family(
    name="qwen3-thinking-2507",
    template_sha256=frozenset({"3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7"}),
    thinking=Thinking.ONLY,
    template_kwargs=_NO_SWITCH,  # its template opens a block whatever it is passed
    generation_prompt=_QWEN_STYLE_PROMPTS,
    think_open=_THINK_OPEN,
    reasoning_field="reasoning_content",
    decoding={Mode.THINK: _THINKING_SAMPLING, Mode.NO_THINK: _THINKING_SAMPLING},
    inference_mode=Mode.THINK,
    **_QWEN3,
)

# The seven families of the later waves (module docstring).
_HYBRID_DECODING = {Mode.THINK: Decoding(temperature=1.0, top_p=0.95), Mode.NO_THINK: GREEDY}
QWEN35_4B = Family(
    name="qwen3.5-4b",
    architectures=("Qwen3_5ForConditionalGeneration",),
    template_sha256=frozenset({"a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"}),
    thinking=Thinking.HYBRID,
    template_kwargs={Mode.THINK: {}, Mode.NO_THINK: {"enable_thinking": False}},  # thinks by default
    generation_prompt=_QWEN_STYLE_PROMPTS,
    think_open=_THINK_OPEN,
    think_close=_THINK_CLOSE,
    turn_end=_TURN_END,
    stop=(_TURN_END,),
    bos="none",
    tool_format=QWEN_XML,
    raw_arguments="inline",
    tool_call_ids=False,
    reasoning_field="reasoning_content",
    lora_targets=(r"model\.(language_model\.)?layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\."
                  r"(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)"),
    decoding={Mode.THINK: Decoding(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5),
              Mode.NO_THINK: GREEDY},
    inference_mode=Mode.THINK,
)
GRANITE_42_3B = Family(
    name="granite-4.2-3b",
    architectures=("GraniteForCausalLM",),
    template_sha256=frozenset({"f0ba43f79b3cabca5e5a7584c77aec92f49feae45cdf7779c87d9fc54cd90258"}),
    thinking=Thinking.HYBRID,
    template_kwargs=_SWITCH,
    generation_prompt={Mode.THINK: _ASSISTANT + _THINK_OPEN, Mode.NO_THINK: _ASSISTANT + "<think></think>"},
    think_open=_THINK_OPEN,
    think_close="\n</think>\n",
    turn_end=_TURN_END,
    stop=(_TURN_END,),
    bos="none",
    tool_format=QWEN_XML,
    raw_arguments="inline",
    tool_call_ids=False,
    reasoning_field="reasoning_content",
    lora_targets=_LLAMA_TARGETS,
    think_join="trim",  # module docstring: the close of a turn of calls alone is trimmed
    decoding=_HYBRID_DECODING,
    inference_mode=Mode.THINK,
)
_NANBEIGE = {
    "architectures": ("LlamaForCausalLM",),
    "thinking": Thinking.ONLY,
    "template_kwargs": _NO_SWITCH,  # no thinking switch: the model always thinks
    "generation_prompt": _QWEN_STYLE_PROMPTS,
    "think_open": _THINK_OPEN,
    "think_close": _THINK_CLOSE,
    "turn_end": _TURN_END,
    "stop": (_TURN_END,),
    "bos": "turn",
    "tool_format": QWEN_JSON,
    "raw_arguments": "string",
    "tool_call_ids": False,
    "reasoning_field": "reasoning_content",
    "lora_targets": _LLAMA_TARGETS,
    "decoding": {Mode.THINK: _NANBEIGE_SAMPLING, Mode.NO_THINK: _NANBEIGE_SAMPLING},
    "inference_mode": Mode.THINK,
}
NANBEIGE4_THINKING_2511 = Family(
    name="nanbeige4-thinking-2511",
    template_sha256=frozenset({"7db1a76279e4bdabda27ca6cf0c523ad0576b3d61b647e2bd49b5f065c9eb117"}),
    **_NANBEIGE,
)
NANBEIGE41_3B = Family(
    name="nanbeige4.1-3b",
    template_sha256=frozenset({"757f108242fb797d23b79f72c088cb5b4bc37fb3744e29dc8771c1dc2b1cb335"}),
    **_NANBEIGE,
)
MINICPM5_2B = Family(
    name="minicpm5-2b",
    architectures=("LlamaForCausalLM",),
    template_sha256=frozenset({"bc1f78ef8a86b6bdf64a056cf89dd0ac506db4f198ab363d85168e2500ab44dd"}),
    thinking=Thinking.HYBRID,
    template_kwargs=_SWITCH,
    generation_prompt=_QWEN_STYLE_PROMPTS,
    think_open=_THINK_OPEN,
    think_close=_THINK_CLOSE,
    turn_end=_TURN_END,  # not the eos ("</s>"): module docstring
    stop=(_TURN_END,),
    bos="template",
    tool_format=MINICPM_XML,
    raw_arguments="inline",
    tool_call_ids=False,
    reasoning_field="reasoning_content",
    lora_targets=_LLAMA_TARGETS,
    text_from_ids=True,
    decoding=_HYBRID_DECODING,
    inference_mode=Mode.THINK,
)
SPARK_X25_4B = Family(
    name="spark-x2.5-4b",
    architectures=("Spark2_5ForCausalLM",),
    template_sha256=frozenset({"705345721970da40e6aa56db548a46ecd1d8d94a5418808fa180ac19ee77a9ba"}),
    thinking=Thinking.HYBRID,
    template_kwargs=_SWITCH,
    generation_prompt={Mode.THINK: "<|Bot|><think>", Mode.NO_THINK: "<|Bot|></think>"},
    think_open="<think>",
    think_close="</think>",
    turn_end="<｜end▁of▁sentence｜>",
    stop=("<｜end▁of▁sentence｜>",),
    bos="turn",
    tool_format=GLM_KV,
    raw_arguments="inline",
    tool_call_ids=False,
    reasoning_field="reasoning_content",
    lora_targets=("q_k_v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"),
    remote_files={
        "modeling_spark.py": "9cf0d1ad2b54b9f7088792779ddd8b5cb4d4fe63bfea054da7f2dd16362bcaf4",
        "configuration_spark.py": "02218597240490f490659b184052db940b08163ff9c3af7b7e59323f6f963722"},
    model_patches={"_tied_weights_keys": {"lm_head.weight": "model.embedding.weight"}},  # docstring 1
    module_adapters={"create_causal_mask": MASKING_4_57,  # module docstring 3
                     "create_sliding_window_causal_mask": MASKING_4_57,
                     "Spark2_5DecoderLayer.forward": SPARK_PROJECTION_DTYPE},  # module docstring 4
    attn_implementation="eager",  # module docstring 2: the vendor attention is eager only
    trust_remote_code=True,
    revision="0bcb35678590218655dff3765b9e61c83b35e9c4",
    decoding=_HYBRID_DECODING,
    inference_mode=Mode.THINK,
)
LFM25_26B = Family(
    name="lfm2.5-2.6b",
    architectures=("Lfm2ForCausalLM",),
    template_sha256=frozenset({"ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef"}),
    thinking=Thinking.ONLY,
    template_kwargs=_NO_SWITCH,
    generation_prompt={Mode.THINK: _ASSISTANT + "<think>", Mode.NO_THINK: _ASSISTANT + "<think></think>"},
    think_open="<think>",
    think_close="</think>",
    turn_end=_TURN_END,
    stop=(_TURN_END,),
    bos="template",
    tool_format=PYTHONIC,
    raw_arguments="inline",
    tool_call_ids=False,
    reasoning_field="reasoning_content",
    lora_targets=("q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"),
    empty_block="constructed",
    text_from_ids=True,
    decoding={Mode.THINK: _LFM25_SAMPLING, Mode.NO_THINK: _LFM25_SAMPLING},
    inference_mode=Mode.THINK,
)
FAMILIES: tuple[Family, ...] = (QWEN3_HYBRID, QWEN3_INSTRUCT_2507, QWEN3_THINKING_2507, QWEN35_4B,
                                GRANITE_42_3B, NANBEIGE4_THINKING_2511, NANBEIGE41_3B, MINICPM5_2B,
                                SPARK_X25_4B, LFM25_26B)


def resolve(architectures: Sequence[str], template_sha256: str, *, name: str | None = None,
            families: Sequence[Family] = FAMILIES) -> Family:
    """The family of a model folder with these config.json architectures and this template hash (module
    docstring: Resolution); FamilyError when none is, when several claim it, or when name is another's."""
    registered = [f.name for f in families]
    if name is not None and name not in registered:
        raise FamilyError(f"unknown family {name!r}; registered: {', '.join(registered)}")
    claims = [f for f in families if template_sha256 in f.template_sha256]
    matches = [f for f in claims if set(architectures) <= set(f.architectures)]
    if len(matches) > 1:
        raise FamilyError(f"families {', '.join(f.name for f in matches)} all claim chat template "
                          f"{template_sha256} with architectures {list(architectures)}")
    if not matches and claims:
        theirs = "; ".join(f"{f.name}: {list(f.architectures)}" for f in claims)
        pinned = ", ".join(f.name for f in claims)
        raise FamilyError(f"chat template {template_sha256} is pinned by {pinned}, but the folder's "
                          f"architectures {list(architectures)} are not among {theirs}")
    if not matches:
        raise FamilyError(f"unknown chat template {template_sha256} (architectures "
                          f"{list(architectures)}): no registered family pins it; registered: "
                          f"{', '.join(registered)}")
    (family,) = matches
    if name is not None and name != family.name:
        raise FamilyError(f"--family {name!r} does not agree with the folder, whose family is {family.name} "
                          f"(architectures {list(architectures)}, chat template {template_sha256})")
    return family
