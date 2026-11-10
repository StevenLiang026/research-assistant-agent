"""Load-time adapters of a family's vendor code (research_agent.family.core: Module adapters): vendor code
written for another stack, run unchanged on the installed one.

A family with vendor code (trust_remote_code, its files pinned by SHA-256) may declare module_adapters: names
in its vendor module, a function the module holds (create_causal_mask) or an attribute of a class it defines
(Spark2_5DecoderLayer.forward), each mapped to a registered adapter (ADAPTERS) that replaces it with what the
installed stack can run. core.patch_model_class applies them (adapt_vendor_module) to the module the auto
class builds the folder's model from (transformers' dynamic module of the folder, named by the folder and
its code's hash) before the model is built, wherever a family's model is built (core.build_causal_lm:
scripts/train.py's load_model, eval.fidelity's HF backend). The vendor files are never changed and keep
their pins: an adapter changes names of the imported module, in this process, and nothing else. Every
adapter holds what it replaces to what it was written for (the installed function the vendor imported; a
vendor function whose source hashes to the reviewed one) and refuses anything else (FamilyError); all the
targets of a module are checked before any is replaced, so a refusal leaves the module as it was. A
replacement carries its adapter's name (the attribute ADAPTED), so adapting a module again replaces nothing.

The two adapters, both Spark-X2.5-4B's (registry: spark-x2.5-4b):

masking_4_57 (create_causal_mask, create_sliding_window_causal_mask). modeling_spark.py was written for
transformers 4.57 (config.json's transformers_version 4.57.1), whose two mask functions took input_embeds=
and cache_position=; 5.18's take the embeddings as inputs_embeds= and no positions at all, so the first
forward pass raises TypeError. The module imports the two functions by name (from transformers.masking_utils
import ...) and Spark2_5Model.forward calls them through the module's globals with keywords alone, so the
module's two names are replaced by thin wrappers of the installed functions (functools.wraps: __wrapped__ is
the installed function) that take the vendor's call: input_embeds goes on as inputs_embeds, and
cache_position is dropped once it is checked to hold exactly the query positions the installed function
places the queries at by itself. 4.57 made the mask with its queries at cache_position (and the key length
and offset from the cache at cache_position); 5.18 makes it with its queries at the cache's query offset
(get_query_offset of the cache's first full-attention layer for create_causal_mask, of its first
sliding-window layer for create_sliding_window_causal_mask, unless layer_idx names one; 0 without a cache),
one per embedding, and the key length and offset from the cache at that query length. The two are the same
mask exactly when cache_position is that range: the vendor computes cache_position as the range after the
tokens its cache has seen (past_key_values.get_seq_length(), every layer of a DynamicCache having seen them
all), and transformers' generate, which still hands remote code a cache_position, the same. A call whose
positions are any other is refused (ValueError) rather than given another mask than the vendor's. Both
functions build the eager mask from the same mask functions in both releases (causal; within the window,
kv_idx > q_idx - sliding_window; the padding of a 2-D attention mask; an additive mask of 0 and the dtype's
least value), which tests/family/test_compat.py holds bit for bit against masks made by hand. The adapter
refuses a vendor name that is not the installed function, and an installed function that takes input_embeds
or cache_position, or lacks config, inputs_embeds, attention_mask or past_key_values: it was written for
5.18's signature alone.

spark_projection_dtype (Spark2_5DecoderLayer.forward). The vendor's decoder layer casts its normalised input
to the dtype of a projection's weight before the attention and before the MLP (hidden_states.to(self.mlp.
gate_proj.weight.dtype)): the model's compute dtype, while weights are stored in it. Under bitsandbytes NF4,
QLoRA's base model (scripts/train.py --qlora, the bake-off's training; eval.fidelity's HF backend, S6), a
quantized layer stores its weight as packed 4-bit codes in a uint8 tensor and computes in its compute_dtype
(bfloat16 in both), so the cast makes the hidden states bytes: the forward pass fails at the MLP's activation
(NotImplementedError: "GeluKernelImpl" not implemented for 'Byte'), as it would under any transformers
release; it is no 5.18 change, but it is in the same path, the forward pass of the model the bake-off trains.
The adapter replaces the method with spark_decoder_layer_forward, the vendor's method with that expression
changed in both places to projection_dtype(self.mlp.gate_proj): the weight's dtype when it is a
floating-point one, the vendor's own reading (a model that is not quantized runs the vendor's arithmetic bit
for bit), and the compute dtype of a quantized layer otherwise (a PEFT layer's read through its base layer).
It refuses a vendor method whose source does not hash to VENDOR_DECODER_FORWARD_SHA256, the method of the
pinned modeling_spark.py, which the replacement copies (tests/family/test_compat.py compares the two
syntax trees: they differ in the cast alone).

Nothing here imports torch or transformers at import time: the wrappers import torch when a call is checked,
the mask adapter imports transformers.masking_utils when it adapts.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .core import FamilyError, _Frozen

ADAPTED = "__family_adapter__"  # the attribute a replacement carries: its adapter's name
MASKING_4_57 = "masking_4_57"
SPARK_PROJECTION_DTYPE = "spark_projection_dtype"
MASKING = "transformers.masking_utils"
# The mask functions masking_4_57 adapts, each with the kind of layer its cache layer is chosen among: whether
# that layer slides (True: the first sliding-window layer; False: the first full-attention one).
_MASKS = {"create_causal_mask": False, "create_sliding_window_causal_mask": True}
_TAKES = ("config", "inputs_embeds", "attention_mask", "past_key_values")  # 5.18's mask functions take these
_TOOK = ("input_embeds", "cache_position")  # 4.57's keywords, which 5.18's take no more
_DECODER_FORWARD = "Spark2_5DecoderLayer.forward"
# The SHA-256 of the source of Spark2_5DecoderLayer.forward in the pinned modeling_spark.py (registry:
# spark-x2.5-4b, SHA-256 9cf0d1ad...), as inspect.getsource reads it: the method spark_decoder_layer_forward
# copies.
VENDOR_DECODER_FORWARD_SHA256 = "e27a5a2b1172531bd7069eec043aed83845c85228a0192a0e5b8ce840424e4e8"


@dataclass(frozen=True)
class ModuleAdapter:
    """A registered adapter: its name, the names of a vendor module it adapts (targets) and how it makes the
    replacement of one of them from what the module holds there (replace(current, target, family) -> the
    replacement; FamilyError when it refuses what it finds)."""

    name: str
    targets: frozenset[str]
    replace: Callable[[object, str, str], object]


def adapt_vendor_module(module: object, adapters: Mapping[str, str], family: str) -> list[str]:
    """The vendor module with its targets replaced by their adapters' replacements (module docstring);
    adapters maps each target (a name the module holds, or Class.attribute) to an adapter's name, family names
    the family in messages. Every target is checked first, then all are replaced, so a FamilyError leaves the
    module as it was. A target that already holds its adapter's replacement is left as it is. Returns the
    targets replaced, in the order given."""
    planned = []
    for target, name in adapters.items():
        adapter = ADAPTERS.get(name)
        if adapter is None or target not in adapter.targets:
            raise FamilyError(f"{family}: {name!r} is no registered adapter of {target} (registered: "
                              f"{', '.join(f'{a.name} of {sorted(a.targets)}' for a in ADAPTERS.values())})")
        owner, attribute = _holder(module, target, family)
        current = getattr(owner, attribute)
        if getattr(current, ADAPTED, None) == name:
            continue  # adapted already
        planned.append((target, owner, attribute, adapter.replace(current, target, family)))
    for _, owner, attribute, replacement in planned:
        setattr(owner, attribute, replacement)
    return [target for target, *_ in planned]


def _holder(module: object, target: str, family: str) -> tuple[object, str]:
    """What a target names in the module: (the module, the name) or (the class, the attribute); FamilyError
    when the module has no such name or class, or the class no such attribute."""
    where = getattr(module, "__name__", repr(module))
    owner_name, _, attribute = target.rpartition(".")
    owner = module
    if owner_name:
        owner = getattr(module, owner_name, None)
        if not isinstance(owner, type):
            raise FamilyError(f"{family}: the vendor module {where} defines no class {owner_name}, so its "
                              f"{target} cannot be adapted")
    if not hasattr(owner, attribute):
        raise FamilyError(f"{family}: the vendor module {where} holds no {target} to adapt")
    return owner, attribute


# ---- masking_4_57 ------------------------------------------------------------------------------------------


def _adapt_mask(current: object, target: str, family: str) -> Callable:
    """The wrapper of the installed mask function the vendor imported as target (module docstring:
    masking_4_57); FamilyError when the module's name is not that function, or the function is not 5.18's."""
    masking = importlib.import_module(MASKING)
    installed = getattr(masking, target, None)
    if installed is None or current is not installed:
        raise FamilyError(f"{family}: {target} in the vendor module is {current!r}, not {MASKING}.{target}, "
                          f"the function {MASKING_4_57} wraps")
    try:
        parameters = list(inspect.signature(installed).parameters)
    except (TypeError, ValueError) as exc:
        raise FamilyError(f"{family}: the signature of {MASKING}.{target} cannot be read ({exc})") from exc
    if not set(_TAKES) <= set(parameters) or set(_TOOK) & set(parameters):
        raise FamilyError(f"{family}: {MASKING}.{target} takes ({', '.join(parameters)}): {MASKING_4_57} "
                          "passes the vendor's input_embeds on as inputs_embeds and drops its "
                          f"cache_position, for transformers 5.18's signature ({', '.join(_TAKES)}, ...), "
                          "and was written for no other")
    return _mask_wrapper(installed, target, sliding=_MASKS[target])


def _mask_wrapper(installed: Callable, name: str, *, sliding: bool) -> Callable:
    """The installed mask function taking the vendor's keywords (module docstring: masking_4_57)."""

    @functools.wraps(installed)
    def adapted(*args, **kwargs):
        if args:
            raise TypeError(f"{name}, adapted by {MASKING_4_57}, takes keyword arguments alone, as the "
                            f"vendor passes them; got {len(args)} positional")
        if "input_embeds" in kwargs:
            if "inputs_embeds" in kwargs:
                raise TypeError(f"{name}: input_embeds (4.57's keyword) and inputs_embeds (5.18's) both "
                                "given")
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        positions = kwargs.pop("cache_position", None)
        if positions is not None:
            _check_positions(name, positions, kwargs, sliding=sliding)
        return installed(**kwargs)

    setattr(adapted, ADAPTED, MASKING_4_57)
    return adapted


def _check_positions(name: str, positions: object, kwargs: dict, *, sliding: bool) -> None:
    """ValueError unless positions (the vendor's cache_position) are the query positions the installed
    function places its queries at (module docstring: masking_4_57): the range from the cache's query offset
    of the layer it sizes the mask by, one position per embedding."""
    import torch  # only a forward pass calls this, with tensors

    positions = torch.as_tensor(positions)
    length = int(kwargs["inputs_embeds"].shape[1])
    offset = 0
    cache = kwargs.get("past_key_values")
    if cache is not None:
        layer = kwargs.get("layer_idx")
        if layer is None:  # as 5.18's function chooses it
            kinds = getattr(cache, "is_sliding", None)
            layer = kinds.index(sliding) if kinds is not None and sliding in kinds else 0
        offset = int(cache.get_query_offset(layer))
    expected = torch.arange(offset, offset + length, device=positions.device)
    if not torch.equal(positions.to(expected.dtype), expected):  # the shape too: (length,)
        shown = positions.flatten().tolist()
        shown = shown if len(shown) <= 8 else [*shown[:4], "...", *shown[-4:]]
        raise ValueError(f"{name}: cache_position {shown} (shape {list(positions.shape)}) is not the range "
                         f"{offset}..{offset + length - 1} of {length} query positions that transformers "
                         "5.18 places the queries at from the cache's query offset: its mask would not be "
                         "the one the vendor's call asks for, 4.57's at cache_position")


# ---- spark_projection_dtype --------------------------------------------------------------------------------


def source_sha256(function: Callable) -> str:
    """The SHA-256 (lowercase hex) of the UTF-8 bytes of a function's source as inspect.getsource reads it."""
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def projection_dtype(linear: object) -> object:
    """The dtype a linear layer computes in: its weight's dtype when that is a floating-point dtype (the
    vendor's reading), else the compute dtype of a quantized layer (bitsandbytes' Linear4bit stores packed
    4-bit codes as uint8 and computes in compute_dtype), read on the layer a PEFT layer wraps (get_base_layer)
    first. TypeError for a layer that stores no floating-point weight and names no compute dtype."""
    dtype = linear.weight.dtype
    if dtype.is_floating_point:
        return dtype
    layers = (linear.get_base_layer(), linear) if hasattr(linear, "get_base_layer") else (linear,)
    for layer in layers:
        compute = getattr(layer, "compute_dtype", None)
        if compute is not None:
            return compute
    raise TypeError(f"{type(linear).__name__} stores its weight as {dtype}, not a floating-point dtype, and "
                    "names no compute dtype to cast its input to")


def spark_decoder_layer_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    position_ids=None,
    **kwargs,
):
    """Spark2_5DecoderLayer.forward of the pinned modeling_spark.py with its two casts to
    self.mlp.gate_proj.weight.dtype made casts to projection_dtype(self.mlp.gate_proj) (module docstring:
    spark_projection_dtype); every other statement is the vendor's, its annotations left out."""

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = hidden_states.to(projection_dtype(self.mlp.gate_proj))

    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        position_ids=position_ids,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = hidden_states.to(projection_dtype(self.mlp.gate_proj))

    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


setattr(spark_decoder_layer_forward, ADAPTED, SPARK_PROJECTION_DTYPE)


def _adapt_decoder_forward(current: object, target: str, family: str) -> Callable:
    """spark_decoder_layer_forward, for a vendor method whose source is the reviewed one (module docstring:
    spark_projection_dtype); FamilyError otherwise."""
    try:
        digest = source_sha256(current)
    except (OSError, TypeError) as exc:
        raise FamilyError(f"{family}: the source of {target} cannot be read ({exc}), so it cannot be held to "
                          "the reviewed source") from exc
    if digest != VENDOR_DECODER_FORWARD_SHA256:
        raise FamilyError(f"{family}: {target} has source sha256 {digest}, not the reviewed "
                          f"{VENDOR_DECODER_FORWARD_SHA256}: {SPARK_PROJECTION_DTYPE} replaces it with a "
                          "copy of the reviewed source and was written for no other")
    return spark_decoder_layer_forward


# The registered adapters, read-only (core._Frozen): name -> ModuleAdapter.
ADAPTERS: Mapping[str, ModuleAdapter] = _Frozen({
    MASKING_4_57: ModuleAdapter(MASKING_4_57, frozenset(_MASKS), _adapt_mask),
    SPARK_PROJECTION_DTYPE: ModuleAdapter(SPARK_PROJECTION_DTYPE, frozenset({_DECODER_FORWARD}),
                                          _adapt_decoder_forward),
})
