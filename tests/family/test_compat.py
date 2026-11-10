"""Load-time adapters of a family's vendor code (research_agent.family.compat; family.core: Module adapters).

Offline, without a model folder: the adapters' registry; the mask functions' wrapper over a fake
transformers.masking_utils, which takes 4.57's keywords and calls the installed function with 5.18's,
dropping cache_position only when it holds the query positions the installed function derives; what the
adapting refuses (a vendor name that is not the installed function, an installed signature the adapter was
not written for, a target the module lacks, a vendor function whose source is not the reviewed one); a second
adapting that leaves the first's replacements as they are; and projection_dtype, the cast target of the
replacement decoder-layer forward.

With models/Spark-X2.5-4B (skipped without it, or without torch, transformers or bitsandbytes): a Spark of
two layers (one sliding-window, one full attention, the window shorter than the sequences) with random
weights and the real vocabulary, its ids a real prompt of the family, on the CPU. As the vendor wrote it,
transformers 5.18 runs no forward pass of it (TypeError: create_causal_mask takes no input_embeds); built by
build_causal_lm, adapted, the pass runs and its logits equal, bit for bit, the vendor layers' under masks
made here as transformers 4.57 made them (causal, the window, a batch's padding), with and without a cache,
and a cached step agrees with the whole pass. Under bitsandbytes NF4 (QLoRA's base model, the bake-off's
training and S6) the vendor layer casts its input to a quantized weight's packed uint8 storage and the pass
fails (NotImplementedError for 'Byte'); adapted, it runs and agrees with bf16, while a model that is not
quantized runs the vendor's arithmetic bit for bit. The replacement forward's syntax tree is the vendor's
but for the cast, and the vendor files keep their pins.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent.family import compat, registry
from research_agent.family.compat import (
    ADAPTED,
    ADAPTERS,
    MASKING_4_57,
    SPARK_PROJECTION_DTYPE,
    VENDOR_DECODER_FORWARD_SHA256,
    adapt_vendor_module,
    projection_dtype,
    source_sha256,
    spark_decoder_layer_forward,
)
from research_agent.family.core import FamilyError, Mode

REPO = Path(__file__).resolve().parents[2]
SPARK = REPO / "models" / "Spark-X2.5-4B"
SPEC = registry.SPARK_X25_4B
MASKS = ("create_causal_mask", "create_sliding_window_causal_mask")
FORWARD = "Spark2_5DecoderLayer.forward"
spark_present = pytest.mark.skipif(not (SPARK / "modeling_spark.py").is_file(),
                                   reason="local model files are not present")


# ---- the registry ----------------------------------------------------------------------------------------


def test_the_registry_holds_the_two_adapters_spark_declares():
    assert set(ADAPTERS) == {MASKING_4_57, SPARK_PROJECTION_DTYPE}
    assert ADAPTERS[MASKING_4_57].targets == frozenset(MASKS)
    assert ADAPTERS[SPARK_PROJECTION_DTYPE].targets == frozenset({FORWARD})
    assert dict(SPEC.module_adapters) == {MASKS[0]: MASKING_4_57, MASKS[1]: MASKING_4_57,
                                          FORWARD: SPARK_PROJECTION_DTYPE}
    assert all(not f.module_adapters for f in registry.FAMILIES if f is not SPEC)


# ---- the mask functions' wrapper, over a fake transformers.masking_utils ---------------------------------


def fake_masking(monkeypatch, *, old: bool = False) -> tuple[types.ModuleType, list]:
    """transformers.masking_utils in sys.modules for this test: the two mask functions with 5.18's keywords
    (old: 4.57's, input_embeds and cache_position), each keeping the keywords of every call."""
    calls: list = []

    def make(name: str):
        if old:
            def function(config, input_embeds, attention_mask, cache_position, past_key_values,
                         position_ids=None, or_mask_function=None, and_mask_function=None):
                calls.append((name, dict(locals())))
                return name
        else:
            def function(config, inputs_embeds, attention_mask, past_key_values, position_ids=None,
                         or_mask_function=None, and_mask_function=None, block_sequence_ids=None,
                         layer_idx=None, allow_is_causal_skip=True):
                calls.append((name, dict(locals())))
                return name
        function.__name__ = name
        return function

    package = types.ModuleType("transformers")
    masking = types.ModuleType("transformers.masking_utils")
    for name in MASKS:
        setattr(masking, name, make(name))
    package.masking_utils = masking
    monkeypatch.setitem(sys.modules, "transformers", package)
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", masking)
    return masking, calls


def fake_vendor(masking: types.ModuleType, **names) -> types.ModuleType:
    """A vendor module that imported the two mask functions as Spark's does, with the given names besides."""
    module = types.ModuleType("transformers_modules.fake.modeling_vendor")
    for name in MASKS:
        setattr(module, name, getattr(masking, name))
    for name, value in names.items():
        setattr(module, name, value)
    return module


MASKING_ONLY = {MASKS[0]: MASKING_4_57, MASKS[1]: MASKING_4_57}


class Cache:
    """A cache as the installed functions read it: the tokens seen per layer, and which layers slide."""

    def __init__(self, seen: int, is_sliding: list[bool]) -> None:
        self.seen, self.is_sliding, self.asked = seen, is_sliding, []

    def get_query_offset(self, layer_idx: int = 0) -> int:
        self.asked.append(layer_idx)
        return self.seen


def test_the_mask_wrapper_calls_the_installed_function_with_its_keywords_and_checks_what_it_drops(
        monkeypatch):
    torch = pytest.importorskip("torch")
    masking, calls = fake_masking(monkeypatch)
    module = fake_vendor(masking)
    assert adapt_vendor_module(module, MASKING_ONLY, "fake") == list(MASKS)
    embeds = torch.zeros(1, 5, 8)
    # the vendor's call, the positions 0..4 of a step without a cache, which 5.18 derives itself
    vendor_call = {"config": "config", "input_embeds": embeds, "attention_mask": None,
                   "cache_position": torch.arange(5), "past_key_values": None,
                   "position_ids": torch.arange(5)[None]}
    assert module.create_causal_mask(**vendor_call) == MASKS[0]
    assert module.create_sliding_window_causal_mask(**vendor_call) == MASKS[1]
    (name, sent), (other, _) = calls
    assert (name, other) == MASKS and sent["inputs_embeds"] is embeds and sent["past_key_values"] is None
    assert sent["position_ids"] is vendor_call["position_ids"] and sent["config"] == "config"
    # with a cache: the positions after the tokens it has seen, as the installed function sizes the mask by
    # the cache's query offset of its first full layer (the causal mask) or first sliding one
    cache = Cache(7, [True, False])
    step = {**vendor_call, "input_embeds": embeds[:, :1], "cache_position": torch.tensor([7]),
            "past_key_values": cache, "position_ids": torch.tensor([[7]])}
    module.create_causal_mask(**step)
    module.create_sliding_window_causal_mask(**step)
    assert cache.asked == [1, 0]
    # positions the installed function would not use are refused: the mask would be another than 4.57's
    for wrong in (torch.arange(1, 6), torch.arange(4), torch.tensor([0, 1, 2, 4, 5]), torch.arange(5)[None]):
        with pytest.raises(ValueError, match="cache_position"):
            module.create_causal_mask(**{**vendor_call, "cache_position": wrong})
    with pytest.raises(ValueError, match="cache_position"):
        module.create_sliding_window_causal_mask(**{**step, "cache_position": torch.tensor([6])})
    assert len(calls) == 4  # none of the refused calls reached the installed function
    # no positions to check: the call goes on without them
    module.create_causal_mask(**{**vendor_call, "cache_position": None})
    assert "cache_position" not in calls[-1][1]


def test_the_mask_wrapper_takes_keywords_alone_and_never_two_spellings_of_the_embeddings(monkeypatch):
    masking, calls = fake_masking(monkeypatch)
    module = fake_vendor(masking)
    adapt_vendor_module(module, MASKING_ONLY, "fake")
    with pytest.raises(TypeError, match="keyword"):
        module.create_causal_mask("config", "embeds", None, None, None)
    with pytest.raises(TypeError, match="input_embeds"):
        module.create_causal_mask(config="c", input_embeds="e", inputs_embeds="e", attention_mask=None,
                                  past_key_values=None)
    assert calls == []
    # a call already in the installed keywords goes through as it is
    module.create_causal_mask(config="c", inputs_embeds="e", attention_mask=None, past_key_values=None)
    assert calls[0][1]["inputs_embeds"] == "e"


def test_adapting_a_module_again_keeps_its_replacements_and_marks_them(monkeypatch):
    masking, _ = fake_masking(monkeypatch)
    module = fake_vendor(masking)
    adapt_vendor_module(module, MASKING_ONLY, "fake")
    first = (module.create_causal_mask, module.create_sliding_window_causal_mask)
    assert adapt_vendor_module(module, MASKING_ONLY, "fake") == []  # adapted already: nothing replaced
    assert (module.create_causal_mask, module.create_sliding_window_causal_mask) == first
    assert all(getattr(f, ADAPTED) == MASKING_4_57 for f in first)
    assert module.create_causal_mask.__wrapped__ is masking.create_causal_mask  # one wrapper, not two


def test_a_vendor_name_that_is_not_the_installed_function_is_refused(monkeypatch):
    masking, _ = fake_masking(monkeypatch)
    module = fake_vendor(masking, create_causal_mask=lambda **kwargs: "the vendor's own")
    with pytest.raises(FamilyError, match="create_causal_mask.*not transformers.masking_utils"):
        adapt_vendor_module(module, MASKING_ONLY, "fake")


def test_an_installed_signature_the_adapter_was_not_written_for_is_refused(monkeypatch):
    # a transformers that still takes input_embeds and cache_position needs no translation: the adapter
    # refuses to change calls it was not written for rather than guess
    masking, _ = fake_masking(monkeypatch, old=True)
    module = fake_vendor(masking)
    with pytest.raises(FamilyError, match="input_embeds"):
        adapt_vendor_module(module, MASKING_ONLY, "fake")
    assert module.create_causal_mask is masking.create_causal_mask  # nothing was replaced
    # nor one that names the embeddings, or any other argument the wrapper passes on, otherwise than 5.18
    masking, _ = fake_masking(monkeypatch)
    for name in MASKS:
        def renamed(config, embeds, attention_mask, past_key_values, position_ids=None):
            return None

        setattr(masking, name, renamed)
    module = fake_vendor(masking)
    with pytest.raises(FamilyError, match="inputs_embeds"):
        adapt_vendor_module(module, MASKING_ONLY, "fake")
    assert module.create_causal_mask is masking.create_causal_mask
    # nor one that takes 5.18's embeddings but still takes positions: dropping them would change its mask
    for name in MASKS:
        def positioned(config, inputs_embeds, attention_mask, past_key_values, cache_position=None):
            return None

        setattr(masking, name, positioned)
    module = fake_vendor(masking)
    with pytest.raises(FamilyError, match="cache_position"):
        adapt_vendor_module(module, MASKING_ONLY, "fake")
    assert module.create_causal_mask is masking.create_causal_mask


def test_an_adapter_that_is_not_registered_for_its_target_is_refused(monkeypatch):
    masking, _ = fake_masking(monkeypatch)
    module = fake_vendor(masking)
    for adapters in ({MASKS[0]: "no_such_adapter"}, {MASKS[0]: SPARK_PROJECTION_DTYPE}):
        with pytest.raises(FamilyError, match="no registered adapter of create_causal_mask"):
            adapt_vendor_module(module, adapters, "fake")
    assert module.create_causal_mask is masking.create_causal_mask


def test_a_refusal_of_one_target_leaves_every_target_of_the_module_as_it_was(monkeypatch):
    # the targets are all checked before any is replaced: the second refused, the first stays the vendor's
    masking, _ = fake_masking(monkeypatch)
    module = fake_vendor(masking, create_sliding_window_causal_mask=lambda **kwargs: "the vendor's own")
    with pytest.raises(FamilyError, match="create_sliding_window_causal_mask"):
        adapt_vendor_module(module, MASKING_ONLY, "fake")
    assert module.create_causal_mask is masking.create_causal_mask
    assert not hasattr(module.create_causal_mask, ADAPTED)


def test_the_mask_wrapper_checks_the_positions_against_the_layer_a_call_names(monkeypatch):
    # a call naming its cache layer (layer_idx, which 5.18 takes and the vendor never passes) is sized by
    # that layer, the positions checked against its query offset, and the layer goes on to the function
    torch = pytest.importorskip("torch")
    masking, calls = fake_masking(monkeypatch)
    module = fake_vendor(masking)
    adapt_vendor_module(module, MASKING_ONLY, "fake")
    cache = Cache(3, [True, False])
    module.create_causal_mask(config="c", input_embeds=torch.zeros(1, 2, 4), attention_mask=None,
                              cache_position=torch.tensor([3, 4]), past_key_values=cache, layer_idx=0)
    assert cache.asked == [0] and calls[-1][1]["layer_idx"] == 0
    # a cache that names no sliding layer: the first layer, as the installed function takes it
    flat = Cache(3, [False, False])
    module.create_sliding_window_causal_mask(config="c", input_embeds=torch.zeros(1, 2, 4),
                                             attention_mask=None, cache_position=torch.tensor([3, 4]),
                                             past_key_values=flat)
    assert flat.asked == [0]


def test_a_target_the_vendor_module_lacks_is_refused(monkeypatch):
    masking, _ = fake_masking(monkeypatch)
    module = fake_vendor(masking)
    del module.create_sliding_window_causal_mask
    with pytest.raises(FamilyError, match="create_sliding_window_causal_mask"):
        adapt_vendor_module(module, MASKING_ONLY, "fake")
    with pytest.raises(FamilyError, match="Spark2_5DecoderLayer"):
        adapt_vendor_module(module, {FORWARD: SPARK_PROJECTION_DTYPE}, "fake")


class DecoderLayer:
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None,
                cache_position=None, position_ids=None, **kwargs):
        return hidden_states  # not the reviewed code


def test_a_vendor_decoder_forward_whose_source_is_not_the_reviewed_one_is_refused(monkeypatch):
    masking, _ = fake_masking(monkeypatch)
    module = fake_vendor(masking, Spark2_5DecoderLayer=DecoderLayer)
    assert source_sha256(DecoderLayer.forward) != VENDOR_DECODER_FORWARD_SHA256
    with pytest.raises(FamilyError, match=f"sha256 .*{VENDOR_DECODER_FORWARD_SHA256}"):
        adapt_vendor_module(module, {FORWARD: SPARK_PROJECTION_DTYPE}, "fake")
    assert module.Spark2_5DecoderLayer.forward is DecoderLayer.forward


def test_source_sha256_hashes_the_functions_source_as_python_reads_it():
    text = inspect.getsource(DecoderLayer.forward)
    assert source_sha256(DecoderLayer.forward) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_a_vendor_decoder_forward_without_a_readable_source_is_refused(monkeypatch):
    # a method made without a file (exec) has no source to hold to the reviewed one
    masking, _ = fake_masking(monkeypatch)
    namespace: dict = {}
    source = ("class DecoderLayer:\n    def forward(self, hidden_states, position_embeddings, **kwargs):\n"
              "        return hidden_states\n")
    exec(source, namespace)  # noqa: S102  a method without a source file
    module = fake_vendor(masking, Spark2_5DecoderLayer=namespace["DecoderLayer"])
    with pytest.raises(FamilyError, match="source of Spark2_5DecoderLayer.forward cannot be read"):
        adapt_vendor_module(module, {FORWARD: SPARK_PROJECTION_DTYPE}, "fake")


class Dtype:
    def __init__(self, name: str, floating: bool) -> None:
        self.name, self.is_floating_point = name, floating


BF16, F32, PACKED = Dtype("bf16", True), Dtype("f32", True), Dtype("uint8", False)


def test_projection_dtype_is_the_weights_dtype_or_a_quantized_layers_compute_dtype():
    plain = SimpleNamespace(weight=SimpleNamespace(dtype=F32))
    assert projection_dtype(plain) is F32  # the vendor's own reading
    quantized = SimpleNamespace(weight=SimpleNamespace(dtype=PACKED), compute_dtype=BF16)
    assert projection_dtype(quantized) is BF16

    class Lora:  # a PEFT layer: its weight is its base layer's, read through get_base_layer
        def __init__(self, base) -> None:
            self.base = base

        @property
        def weight(self):
            return self.base.weight

        def get_base_layer(self):
            return self.base

    assert projection_dtype(Lora(quantized)) is BF16 and projection_dtype(Lora(plain)) is F32
    with pytest.raises(TypeError, match="compute dtype"):
        projection_dtype(SimpleNamespace(weight=SimpleNamespace(dtype=PACKED)))


# ---- the real vendor code: a two-layer Spark on the CPU ---------------------------------------------------

WINDOW = 4  # the two-layer model's sliding window, shorter than every sequence below
CONTEXT = [{"role": "system", "content": "You are a research assistant."},
           {"role": "user", "content": "找 KV cache 的论文"}]


def holder(module: types.ModuleType, target: str) -> tuple[object, str]:
    """What a module adapter's target names: (the module or its class, the attribute)."""
    owner, _, attribute = target.rpartition(".")
    return (getattr(module, owner) if owner else module), attribute


def keep(monkeypatch, model_class: type) -> types.ModuleType:
    """model_class's module, with what build_causal_lm sets on the class and the module restored when the
    test ends; the module as the vendor wrote it is checked first (another test left nothing adapted)."""
    import transformers.masking_utils as masking

    module = sys.modules[model_class.__module__]
    for attribute in SPEC.model_patches:
        monkeypatch.setattr(model_class, attribute, getattr(model_class, attribute))
    for target in SPEC.module_adapters:
        owner, attribute = holder(module, target)
        assert not hasattr(getattr(owner, attribute), ADAPTED), f"{target} was left adapted"
        monkeypatch.setattr(owner, attribute, getattr(owner, attribute))
    assert all(getattr(module, name) is getattr(masking, name) for name in MASKS)
    return module


def vendor_class(transformers, folder: Path) -> type:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = transformers.AutoConfig.from_pretrained(folder, trust_remote_code=True)
    return get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], folder)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    """A Spark folder of two layers with random weights and the real vocabulary, saved by the pinned vendor
    class (its tied weights patched only to build it) with the vendor files copied as they are."""
    if not (SPARK / "modeling_spark.py").is_file():
        pytest.skip("local model files are not present")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    folder = tmp_path_factory.mktemp("spark") / "Spark-X2.5-4B-two-layers"
    config = transformers.AutoConfig.from_pretrained(SPARK, trust_remote_code=True)
    assert config.vocab_size == 131072 and config.tie_word_embeddings  # the real vocabulary, tied
    config.num_hidden_layers, config.layer_types = 2, ["sliding_attention", "full_attention"]
    config.hidden_size, config.intermediate_size, config.sliding_window = 64, 128, WINDOW
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 16
    model_class = vendor_class(transformers, SPARK)
    with pytest.MonkeyPatch.context() as patch:  # saving reads the tied weights again
        patch.setattr(model_class, "_tied_weights_keys", dict(SPEC.model_patches["_tied_weights_keys"]))
        torch.manual_seed(0)
        model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True,
                                                              attn_implementation="eager")
        model.to(torch.bfloat16).save_pretrained(folder)
    for name in SPEC.remote_files:
        (folder / name).write_bytes((SPARK / name).read_bytes())
    return folder


@pytest.fixture
def vendor(tiny, monkeypatch) -> types.ModuleType:
    """The two-layer folder's vendor module, restored when the test ends (keep)."""
    transformers = pytest.importorskip("transformers")
    return keep(monkeypatch, vendor_class(transformers, tiny))


@pytest.fixture(scope="module")
def prompt() -> list[int]:
    """A real prompt of the family (a THINK prompt, its BOS first) followed by an answer: 40-odd ids."""
    from research_agent.family.core import load_family

    family = load_family(SPARK)
    ids = family.encode_prompt(family.prompt(family.to_template_messages(CONTEXT), None, Mode.THINK))
    return ids + family.encode_text("Let me search arXiv for KV cache papers.")


def masks(torch, length: int, dtype, padding=None) -> dict:
    """The masks the vendor's layers ran under in transformers 4.57, made here: additive, 0 where a query
    attends a key and the dtype's least value where not; the full layer's causal (a key at or before the
    query), the sliding layer's also within the window (fewer than WINDOW positions back); a padded key
    (padding 0) attended by no query."""
    i = torch.arange(length)
    causal = i[None, :] <= i[:, None]
    window = causal & (i[:, None] - i[None, :] < WINDOW)
    keep_keys = torch.ones(1, length, dtype=torch.bool) if padding is None else padding.bool()
    least = torch.finfo(dtype).min

    def additive(allowed):
        both = allowed[None, None] & keep_keys[:, None, None, :]
        return torch.where(both, torch.tensor(0.0, dtype=dtype), torch.tensor(least, dtype=dtype))

    return {"full_attention": additive(causal), "sliding_attention": additive(window)}


@spark_present
def test_as_the_vendor_wrote_it_the_installed_transformers_runs_no_forward_pass(tiny, vendor):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model_class = vendor.Spark2_5ForCausalLM
    model_class._tied_weights_keys = dict(SPEC.model_patches["_tied_weights_keys"])  # restored (keep)
    model = transformers.AutoModelForCausalLM.from_pretrained(tiny, trust_remote_code=True,
                                                              attn_implementation="eager",
                                                              dtype=torch.float32)
    with torch.no_grad(), pytest.raises(TypeError, match="create_causal_mask.*'input_embeds'"):
        model(input_ids=torch.tensor([[0, 5, 6, 7]]))
    # the vendor's keywords against the installed signature: renamed, and one gone
    import transformers.masking_utils as masking

    for name in MASKS:
        parameters = set(inspect.signature(getattr(masking, name)).parameters)
        assert {"inputs_embeds", "past_key_values"} <= parameters
        assert not {"input_embeds", "cache_position"} & parameters


@spark_present
def test_adapted_the_forward_pass_equals_the_vendors_layers_under_the_masks_4_57_made(tiny, vendor, prompt):
    torch = pytest.importorskip("torch")
    from research_agent.family.core import build_causal_lm

    model = build_causal_lm(SPEC, tiny, dtype=torch.float32).eval()
    assert type(model).__module__ == vendor.__name__ and all(
        getattr(getattr(*holder(vendor, target)), ADAPTED) == name
        for target, name in SPEC.module_adapters.items())
    ids = torch.tensor([prompt])
    length = ids.shape[1]
    assert length > 4 * WINDOW and ids[0, 0].item() == 0  # the window matters; the BOS opens the prompt
    reference = masks(torch, length, torch.float32)
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=reference).logits
        for run in ({}, {"attention_mask": torch.ones_like(ids)}, {"use_cache": False},
                    {"attention_mask": torch.ones_like(ids), "use_cache": False}):
            assert torch.equal(model(input_ids=ids, **run).logits, expected), run
        # the window matters: the full layer's mask on the sliding layer gives other logits
        unwindowed = {**reference, "sliding_attention": reference["full_attention"]}
        assert not torch.allclose(model(input_ids=ids, attention_mask=unwindowed).logits, expected)
        # causal: what follows a position does not move its logits
        changed = ids.clone()
        changed[0, 20:] = torch.flip(ids[0, 20:], dims=[0])
        assert torch.equal(model(input_ids=changed).logits[0, :20], expected[0, :20])
        # a padded batch: the second sequence padded on the left, its padding attended by no query
        padded = torch.cat([ids, torch.cat([torch.full((1, 3), 2), ids[:, :-3]], dim=1)])
        attention = torch.ones_like(padded)
        attention[1, :3] = 0
        padding = masks(torch, length, torch.float32, attention)
        expected = model(input_ids=padded, attention_mask=padding).logits
        assert torch.equal(model(input_ids=padded, attention_mask=attention).logits, expected)
        # a cached step: the last token after the others, its query at the cache's offset, agrees with the
        # whole pass at that position
        whole = model(input_ids=ids).logits[0, -1]
        prefix = model(input_ids=ids[:, :-1], use_cache=True)
        step = model(input_ids=ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True)
        assert torch.allclose(step.logits[0, -1], whole, atol=1e-5, rtol=1e-4)


def shim_4_57(torch, sliding: bool):
    """A mask function with transformers 4.57's signature, written here: the queries at cache_position, the
    keys from the cache's offset for as many as the cache sizes the mask (get_mask_sizes of its first layer of
    the kind; the queries' own count without a cache), each query attending the keys at or before it and, for
    a sliding layer, fewer than the window back, and no padded key; additive in the embeddings' dtype."""

    def make(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids=None):
        queries = cache_position
        if past_key_values is None:
            length, offset = input_embeds.shape[1], 0
        else:
            kinds = past_key_values.is_sliding
            length, offset = past_key_values.get_mask_sizes(queries.shape[0], kinds.index(sliding))
        keys = torch.arange(offset, offset + length)
        allowed = keys[None, :] <= queries[:, None]
        if sliding:
            allowed = allowed & (keys[None, :] > queries[:, None] - config.sliding_window)
        allowed = allowed[None, None].expand(input_embeds.shape[0], 1, -1, -1)
        if attention_mask is not None:
            allowed = allowed & attention_mask.bool()[:, None, None, offset:offset + length]
        dtype = input_embeds.dtype
        least = torch.tensor(torch.finfo(dtype).min, dtype=dtype)
        return torch.where(allowed, torch.tensor(0.0, dtype=dtype), least)

    return make


@spark_present
def test_adapted_the_logits_are_the_vendor_codes_under_a_shim_of_4_57s_signature_cached_steps_included(
        tiny, vendor, prompt):
    # the reference of the cached path: the vendor code with its two names holding 4.57-signature functions
    # written here (shim_4_57) instead of the adapters; after the prefix, a chunk of three tokens over the
    # cache (queries at an offset, keys the cache's and the chunk's), then one token at a time past the
    # window, so the sliding layer's cache holds the last window-1 keys alone and its mask starts at an offset
    torch = pytest.importorskip("torch")
    from research_agent.family.core import build_causal_lm

    model = build_causal_lm(SPEC, tiny, dtype=torch.float32).eval()
    ids = torch.tensor([prompt])
    steps, chunk = WINDOW + 2, 3
    padded = torch.cat([ids, torch.cat([torch.full((1, 3), 2), ids[:, :-3]], dim=1)])
    attention = torch.ones_like(padded)
    attention[1, :3] = 0

    def outputs() -> list:
        with torch.no_grad():
            got = [model(input_ids=ids).logits, model(input_ids=ids, use_cache=False).logits,
                   model(input_ids=padded, attention_mask=attention).logits]
            start = ids.shape[1] - steps
            cache = model(input_ids=ids[:, :start], use_cache=True).past_key_values
            out = model(input_ids=ids[:, start:start + chunk], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            got.append(out.logits)
            for t in range(start + chunk, ids.shape[1]):
                out = model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                got.append(out.logits)
        return got

    adapted = outputs()
    assert cache_holds_a_window(model, ids, steps, torch)
    vendor.create_causal_mask = shim_4_57(torch, sliding=False)  # restored when the test ends (keep)
    vendor.create_sliding_window_causal_mask = shim_4_57(torch, sliding=True)
    shimmed = outputs()
    assert len(adapted) == 3 + 1 + steps - chunk
    assert all(torch.equal(a, b) for a, b in zip(adapted, shimmed, strict=True))


def cache_holds_a_window(model, ids, steps: int, torch) -> bool:
    """Whether, after the prefix and the steps, the sliding layer's cache holds the last WINDOW - 1 keys alone
    and the full layer's every key (DynamicCache built from the model's configuration)."""
    with torch.no_grad():
        cache = model(input_ids=ids[:, :-steps], use_cache=True).past_key_values
        for t in range(ids.shape[1] - steps, ids.shape[1]):
            cache = model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True).past_key_values
    sliding, full = cache.layers
    return sliding.keys.shape[-2] == WINDOW - 1 and full.keys.shape[-2] == ids.shape[1]


@spark_present
def test_under_nf4_the_vendor_layer_casts_to_packed_storage_and_adapted_it_agrees_with_bf16(tiny, vendor,
                                                                                          prompt):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("bitsandbytes")
    from research_agent.family.core import build_causal_lm

    original = vendor.Spark2_5DecoderLayer.forward  # the vendor's, restored when the test ends (keep)
    quant = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_compute_dtype=torch.bfloat16)
    nf4 = build_causal_lm(SPEC, tiny, dtype=torch.bfloat16, quantization_config=quant,
                          device_map={"": "cpu"}).eval()
    bf16 = build_causal_lm(SPEC, tiny, dtype=torch.bfloat16).eval()
    gate = nf4.model.layers[0].mlp.gate_proj
    assert gate.weight.dtype == torch.uint8 and gate.compute_dtype == torch.bfloat16  # packed 4-bit codes
    ids = torch.tensor([prompt])
    with torch.no_grad():
        adapted, plain = nf4(input_ids=ids).logits.float(), bf16(input_ids=ids).logits
        vendor.Spark2_5DecoderLayer.forward = original
        with pytest.raises(NotImplementedError, match="Byte"):
            nf4(input_ids=ids)
        # not quantized, the replacement is the vendor's arithmetic: bit for bit the same logits
        assert torch.equal(bf16(input_ids=ids).logits, plain)
    cosine = torch.nn.functional.cosine_similarity(adapted.flatten(), plain.float().flatten(), dim=0)
    agree = (adapted.argmax(-1) == plain.float().argmax(-1)).float().mean()
    assert cosine > 0.99 and agree >= 0.9, (float(cosine), float(agree))


def without_annotations(function: ast.FunctionDef) -> tuple[str, str]:
    """A function's arguments and body as syntax trees, its annotations and docstring left out."""
    arguments = function.args
    for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg,
                arguments.kwarg):
        if arg is not None:
            arg.annotation = None
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return ast.dump(arguments), ast.dump(ast.Module(body=body, type_ignores=[]))


class Cast(ast.NodeTransformer):
    """self.mlp.gate_proj.weight.dtype -> projection_dtype(self.mlp.gate_proj), counted."""

    def __init__(self) -> None:
        self.count = 0

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if ast.unparse(node) == "self.mlp.gate_proj.weight.dtype":
            self.count += 1
            gate = ast.parse("self.mlp.gate_proj", mode="eval").body
            return ast.Call(func=ast.Name(id="projection_dtype", ctx=ast.Load()), args=[gate], keywords=[])
        return self.generic_visit(node)


@spark_present
def test_the_replacement_forward_is_the_vendors_with_the_cast_alone_changed(vendor):
    vendor_forward = vendor.Spark2_5DecoderLayer.forward
    assert source_sha256(vendor_forward) == VENDOR_DECODER_FORWARD_SHA256  # the pinned file's
    (theirs,) = ast.parse(textwrap.dedent(inspect.getsource(vendor_forward))).body
    (ours,) = ast.parse(textwrap.dedent(inspect.getsource(spark_decoder_layer_forward))).body
    cast = Cast()
    theirs = ast.fix_missing_locations(cast.visit(theirs))
    assert cast.count == 2  # before the attention and before the MLP
    assert without_annotations(ours) == without_annotations(theirs)
    assert getattr(spark_decoder_layer_forward, ADAPTED) == SPARK_PROJECTION_DTYPE
    assert compat.projection_dtype is projection_dtype


@spark_present
def test_the_vendor_files_keep_their_pins():
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SPARK.glob("*.py")} == dict(
        SPEC.remote_files)
