"""scripts/bfcl_nothink.py: the non-thinking BFCL handler adds exactly Qwen3's empty thinking block.

BFCL lives in its own environment (.venv-bfcl), so these tests use stand-ins for its handler and its model
config; the local tokenizer, when present, shows what Qwen3's chat template writes with thinking off. The
script's self-check covers the real handler on every BFCL question (docs/m1-进度.md records its output).
"""

import dataclasses
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "models" / "Qwen3-1.7B"
spec = importlib.util.spec_from_file_location("bfcl_nothink", REPO / "scripts" / "bfcl_nothink.py")
bfcl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfcl)

PROMPT = "<|im_start|>user\nweather in Paris?<|im_end|>\n<|im_start|>assistant\n"


class StockHandler:
    """Stands in for BFCL's QwenFCHandler: a prompt ending with the generation prompt, and a decoder."""

    def _format_prompt(self, messages, function):
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def decode_ast(self, result, language, has_tool_call_tag):
        return ["stock decoder", result, language]


@dataclasses.dataclass
class Config:
    """Stands in for BFCL's ModelConfig dataclass (the fields register() must keep or change)."""

    model_name: str
    display_name: str
    model_handler: type
    is_fc_model: bool
    underscore_to_dot: bool


def stock_mapping() -> dict:
    return {bfcl.STOCK: Config("Qwen/Qwen3-1.7B", "Qwen3-1.7B (FC)", StockHandler, True, False)}


def test_no_think_appends_exactly_the_empty_thinking_block():
    assert bfcl.no_think(PROMPT) == PROMPT + "<think>\n\n</think>\n\n"


def test_no_think_refuses_a_prompt_that_does_not_end_with_the_generation_prompt():
    with pytest.raises(ValueError, match="generation prompt"):
        bfcl.no_think("<|im_start|>user\nweather in Paris?<|im_end|>\n")
    with pytest.raises(ValueError, match="generation prompt"):
        bfcl.no_think(PROMPT + bfcl.NO_THINK)  # applied twice would double the block


def test_the_subclass_changes_the_prompt_and_nothing_else():
    handler_cls = bfcl.nothink_handler(StockHandler)
    handler, stock = handler_cls(), StockHandler()
    messages = [{"role": "user", "content": "weather in Paris?"}]
    assert handler._format_prompt(messages, []) == stock._format_prompt(messages, []) + bfcl.NO_THINK
    assert handler.decode_ast("x", "python", False) == stock.decode_ast("x", "python", False)
    assert issubclass(handler_cls, StockHandler) and handler_cls.__name__ == "StockHandlerNoThink"


class EnforceMark(type):
    """Mirrors the check of BFCL's metaclass (overrides.EnforceOverridesMeta): a method that overrides one of
    a base class is refused unless it carries the __override__ mark that the @override decorator sets."""

    def __new__(mcls, name, bases, namespace):
        for attr, value in namespace.items():
            overrides = any(callable(getattr(base, attr, None)) for base in bases)
            if not attr.startswith("__") and overrides and not getattr(value, "__override__", False):
                raise TypeError(f"Method {attr} overrides a base method but does not have @override "
                                f"decorator")
        return super().__new__(mcls, name, bases, namespace)


class EnforcedStock(metaclass=EnforceMark):
    def _format_prompt(self, messages, function):
        return PROMPT


def test_the_subclass_passes_bfcls_override_check():
    handler = bfcl.nothink_handler(EnforcedStock)()
    assert handler._format_prompt([], []) == PROMPT + bfcl.NO_THINK
    with pytest.raises(TypeError, match="does not have @override"):  # the stand-in refuses an unmarked one

        class Unmarked(EnforcedStock):
            def _format_prompt(self, messages, function):
                return ""


def test_register_adds_both_names_as_copies_of_the_stock_config_with_the_handler():
    mapping = stock_mapping()
    stock = mapping[bfcl.STOCK]
    handler = bfcl.nothink_handler(StockHandler)
    names = bfcl.register(mapping, handler)
    assert names == list(bfcl.MODELS) and len(names) == 2
    for name in names:
        config = mapping[name]
        assert config.model_handler is handler
        assert (config.model_name, config.is_fc_model, config.underscore_to_dot) == ("Qwen/Qwen3-1.7B", True,
                                                                                     False)
        assert config.display_name.startswith("Qwen3-1.7B (FC) non-thinking ")
        assert "," not in config.display_name  # BFCL writes its score tables with ",".join(row)
        assert "_" not in name  # BFCL reads "_" in result folder names as "/"
    assert mapping[bfcl.STOCK] is stock and stock.model_handler is StockHandler  # stock entry untouched
    assert mapping[names[0]].display_name != mapping[names[1]].display_name


def test_register_refuses_a_name_with_an_underscore(monkeypatch):
    monkeypatch.setitem(bfcl.MODELS, "local/qwen_ft", "fine-tuned adapter")
    with pytest.raises(ValueError, match="'_'"):
        bfcl.register(stock_mapping(), StockHandler)


def test_register_refuses_a_display_name_with_a_comma(monkeypatch):
    # measured 2026-10-10: "Qwen3-1.7B (FC), non-thinking, base model" shifted every column by two
    monkeypatch.setitem(bfcl.MODELS, "local/qwen-ft", "adapter, step 400")
    with pytest.raises(ValueError, match="shifts every column"):
        bfcl.register(stock_mapping(), StockHandler)


def test_first_turn_prompt_follows_bfcls_inference_loop():
    class Handler(StockHandler):
        def _pre_query_processing_prompting(self, entry):
            return {"message": [], "function": entry["function"]}

        def add_first_turn_message_prompting(self, data, first_turn):
            data["message"].extend(first_turn)
            return data

    entry = {"id": "simple_python_0", "question": [[{"role": "user", "content": "weather in Paris?"}]],
             "function": [{"name": "get_weather"}]}
    assert bfcl.first_turn_prompt(Handler(), entry) == PROMPT
    assert bfcl.first_turn_prompt(bfcl.nothink_handler(Handler)(), entry) == PROMPT + bfcl.NO_THINK


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="local model files are not present")
def test_the_block_is_what_qwen3s_chat_template_writes_with_thinking_off():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    messages = [{"role": "user", "content": "weather in Paris?"}]
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather.",
                                               "parameters": {"type": "object", "properties": {}}}}]
    for kwargs in ({}, {"tools": tools}):
        thinking = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=True, **kwargs)
        plain = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False, **kwargs)
        assert thinking.endswith(bfcl.GENERATION_PROMPT)
        assert plain == bfcl.no_think(thinking)
