import json
from pathlib import Path

import httpx
import pytest

from research_agent.llm.local import LocalClient, parse_logprobs, parse_token_bytes, parse_token_ids

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "Qwen3-1.7B"

# Shape recorded from llama-server b11349 in spike S3 (ids and byte arrays trimmed where irrelevant).
SERVER_FORMAT = {"content": [
    {"id": 151657, "token": "<tool_call>", "logprob": -4.0e-05,
     "top_logprobs": [{"id": 151657, "token": "<tool_call>", "logprob": -4.0e-05},
                      {"id": 151665, "token": "<tool_response>", "logprob": -10.8}]},
    {"id": 198, "token": "\n", "logprob": -0.01,
     "top_logprobs": [{"id": 198, "token": "\n", "logprob": -0.01}]},
]}
LEGACY_FORMAT = {"tokens": ["<tool_call>", "\n"], "token_logprobs": [-4.0e-05, -0.01],
                 "top_logprobs": [{"<tool_call>": -4.0e-05, "<tool_response>": -10.8}, {"\n": -0.01}]}
TEXT = '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "speculative decoding"}}\n</tool_call>'


ADAPTER = {"id": 0, "path": "runs/spikes/s4_adapter-f16.gguf", "scale": 1.0}


def server(seen: list, status: int = 200, body: dict | None = None, adapters: list | None = None,
           adapter_queries: list | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/lora-adapters":
            if adapter_queries is not None:
                adapter_queries.append(1)
            return httpx.Response(200, json=adapters or [])
        seen.append((request.url.path, json.loads(request.content) if request.content else None))
        default = {"choices": [{"text": TEXT, "finish_reason": "stop", "logprobs": SERVER_FORMAT}],
                   "timings": {"predicted_per_second": 100.0}}
        return httpx.Response(status, json=default if body is None else body)

    return httpx.MockTransport(handler)


def test_both_logprob_formats_parse_the_same():
    assert parse_logprobs(SERVER_FORMAT) == parse_logprobs(LEGACY_FORMAT)
    tokens, logprobs, top = parse_logprobs(SERVER_FORMAT)
    assert tokens == ["<tool_call>", "\n"] and logprobs == [-4.0e-05, -0.01]
    assert top[0] == {"<tool_call>": -4.0e-05, "<tool_response>": -10.8}


def test_missing_logprobs_give_empty_lists():
    assert parse_logprobs(None) == ([], [], [])


def test_adapter_scale_and_logprobs_are_sent():
    seen = []
    out = LocalClient(transport=server(seen, adapters=[ADAPTER])).complete("PROMPT", lora_scale=1.0)
    path, payload = seen[0]
    assert path == "/v1/completions"
    assert payload == {"prompt": "PROMPT", "max_tokens": 512, "temperature": 0.0, "logprobs": 5,
                       "lora": [{"id": 0, "scale": 1.0}], "cache_prompt": False}
    assert out.text == TEXT and out.finish_reason == "stop" and out.tokens[0] == "<tool_call>"
    assert out.timings == {"predicted_per_second": 100.0} and out.wall_ms >= 0


def test_the_prompt_cache_is_off_unless_asked_for():
    seen = []
    client = LocalClient(transport=server(seen))
    client.complete("PROMPT")
    client.complete("PROMPT", cache_prompt=True)
    assert [payload["cache_prompt"] for _, payload in seen] == [False, True]


def test_scale_zero_is_sent_explicitly_when_an_adapter_is_loaded():
    seen = []
    LocalClient(transport=server(seen, adapters=[ADAPTER])).complete("PROMPT", lora_scale=0.0)
    assert seen[0][1]["lora"] == [{"id": 0, "scale": 0.0}]


def test_without_an_adapter_scale_zero_is_the_base_model_and_a_positive_scale_is_refused():
    seen = []
    client = LocalClient(transport=server(seen))
    client.complete("PROMPT", lora_scale=0.0)
    assert "lora" not in seen[0][1]  # no adapter id 0 to name: the base model is all the server has
    with pytest.raises(ValueError, match="none loaded"):
        client.complete("PROMPT", lora_scale=1.0)
    assert len(seen) == 1  # the refused request was never sent


def test_base_model_request_has_no_adapter_field():
    seen = []
    LocalClient(transport=server(seen)).complete("PROMPT", logprobs=None, stop=["</tool_call>"])
    payload = seen[0][1]
    assert "lora" not in payload and "logprobs" not in payload and payload["stop"] == ["</tool_call>"]


def test_missing_scale_is_refused_when_an_adapter_is_loaded():
    # llama-server b11349 applies a loaded adapter to requests without the field (spike S1b)
    seen = []
    with pytest.raises(ValueError, match="lora_scale"):
        LocalClient(transport=server(seen, adapters=[ADAPTER])).complete("PROMPT")
    assert seen == []  # nothing was generated


def test_explicit_scales_work_with_an_adapter_loaded_and_the_listing_is_fetched_once():
    seen, queries = [], []
    client = LocalClient(transport=server(seen, adapters=[ADAPTER], adapter_queries=queries))
    client.complete("PROMPT", lora_scale=0.0)
    client.complete("PROMPT", lora_scale=1.0)
    assert [p["lora"] for _, p in seen] == [[{"id": 0, "scale": 0.0}], [{"id": 0, "scale": 1.0}]]
    assert client.adapters() == [ADAPTER] and client.adapters() == [ADAPTER] and len(queries) == 1


def test_environment_connection_settings_are_ignored():
    assert LocalClient(transport=server([])).http.trust_env is False


def test_http_errors_surface():
    with pytest.raises(httpx.HTTPStatusError):
        LocalClient(transport=server([], status=500, body={"error": "boom"})).complete("PROMPT")


def test_health():
    assert LocalClient(transport=server([])).health() is True
    assert LocalClient(transport=server([], status=503, body={})).health() is False


def test_the_latest_completions_server_timings_are_kept_until_the_next_request_starts():
    # read by research_agent.eval.timing for the runners' timing sidecars; the payload does not change
    seen = []
    client = LocalClient(transport=server(seen, adapters=[ADAPTER]))
    assert client.last_timings is None
    client.complete("PROMPT", lora_scale=1.0)
    assert client.last_timings == {"predicted_per_second": 100.0}
    with pytest.raises(ValueError):
        client.complete("PROMPT")  # refused before anything is sent: no timings of its own
    assert client.last_timings is None and len(seen) == 1
    failing = LocalClient(transport=server([], status=500, body={"error": "boom"}))
    with pytest.raises(httpx.HTTPStatusError):
        failing.complete("PROMPT")
    assert failing.last_timings is None


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="local model files are not present")
def test_renderer_uses_the_non_thinking_generation_prompt():
    from research_agent.llm.local import hf_renderer

    prompt = hf_renderer(MODEL_DIR)([{"role": "user", "content": "你好"}], None)
    assert prompt.startswith("<|im_start|>user\n你好<|im_end|>\n")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_props_returns_the_server_settings_and_none_when_the_server_has_none():
    props = {"model_path": "models/Qwen3-1.7B-Q8_0.gguf", "build_info": "b11349", "n_ctx": 8192}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=props) if request.url.path == "/props" else httpx.Response(404)

    assert LocalClient(transport=httpx.MockTransport(handler)).props() == props
    missing = httpx.MockTransport(lambda request: httpx.Response(404, text="not found"))
    assert LocalClient(transport=missing).props() is None


# ---- M4: the per-token bytes, for mapping tokens to text positions (research_agent.agent.route) -----------


def test_the_per_token_bytes_are_kept_when_the_server_sends_them_and_none_otherwise():
    with_bytes = {"content": [{**item, "bytes": list(item["token"].encode("utf-8"))}
                              for item in SERVER_FORMAT["content"]]}
    assert parse_token_bytes(with_bytes) == [list(b"<tool_call>"), [10]]
    assert parse_token_bytes(SERVER_FORMAT) is None  # one item without bytes: none are trusted
    assert parse_token_bytes(LEGACY_FORMAT) is None and parse_token_bytes(None) is None
    seen = []
    body = {"choices": [{"text": TEXT, "finish_reason": "stop", "logprobs": with_bytes}]}
    completion = LocalClient(transport=server(seen, body=body)).complete("PROMPT", lora_scale=0.0)
    assert completion.token_bytes == [list(b"<tool_call>"), [10]]
    assert LocalClient(transport=server([])).complete("PROMPT", lora_scale=0.0).token_bytes is None


# ---- M7: the prompt as ids, the sampling fields and the seed, the token ids (research_agent.llm.generation)

# The bytes the client sent before M7 (captured on 2026-10-19 from the client as it was): a greedy request
# without the new fields changes by nothing.
GREEDY_BODY = (b'{"prompt":"PROMPT","max_tokens":512,"temperature":0.0,"cache_prompt":false,"logprobs":5,'
               b'"lora":[{"id":0,"scale":1.0}],"stop":["<|im_end|>"]}')
BARE_BODY = (b'{"prompt":"PROMPT","max_tokens":512,"temperature":0.0,"cache_prompt":false,"logprobs":5,'
             b'"stop":["<|im_end|>"]}')


def raw_server(seen: list[bytes], adapters: list | None = None) -> httpx.MockTransport:
    """server, keeping the bytes of every completion request instead of the decoded payload."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/lora-adapters":
            return httpx.Response(200, json=adapters or [])
        seen.append(request.content)
        return httpx.Response(200, json={"choices": [{"text": TEXT, "finish_reason": "stop",
                                                      "logprobs": SERVER_FORMAT}]})

    return httpx.MockTransport(handler)


def test_a_greedy_request_is_byte_for_byte_the_request_before_m7():
    seen: list[bytes] = []
    LocalClient(transport=raw_server(seen, [ADAPTER])).complete("PROMPT", lora_scale=1.0, stop=["<|im_end|>"])
    LocalClient(transport=raw_server(seen)).complete("PROMPT", lora_scale=0.0, stop=["<|im_end|>"])
    assert seen == [GREEDY_BODY, BARE_BODY]


def test_the_prompt_may_be_the_token_ids():
    seen = []
    LocalClient(transport=server(seen)).complete([151644, 872, 198], lora_scale=0.0)
    assert seen[0][1]["prompt"] == [151644, 872, 198]


def test_the_sampling_fields_and_the_seed_are_sent_only_when_given():
    seen = []
    client = LocalClient(transport=server(seen))
    client.complete("PROMPT", lora_scale=0.0, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                    presence_penalty=0.0, seed=12345, stop=["<|im_end|>"])
    client.complete("PROMPT", lora_scale=0.0, seed=1)
    client.complete("PROMPT", lora_scale=0.0)
    sampled, seeded, plain = (payload for _, payload in seen)
    fields = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed")
    assert {k: sampled[k] for k in fields} == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                               "presence_penalty": 0.0, "seed": 12345}
    assert list(sampled) == ["prompt", "max_tokens", "temperature", "cache_prompt", "logprobs", "stop",
                             "seed", "top_p", "top_k", "min_p", "presence_penalty"]
    assert seeded["seed"] == 1 and not {"top_p", "top_k", "min_p", "presence_penalty"} & set(seeded)
    assert not set(fields[1:]) & set(plain)
    assert not {"repeat_penalty", "repetition_penalty", "repeat_last_n"} & {k for _, payload in seen
                                                                               for k in payload}


def test_a_repeat_penalty_is_sent_under_llama_servers_name_after_the_sampling_fields_when_given():
    # server-schema.cpp reads "repeat_penalty" into the sampler's penalty_repeat (1.0 = disabled) and
    # "repeat_last_n" into its penalty_last_n, the window of last tokens the penalty covers
    seen = []
    client = LocalClient(transport=server(seen))
    client.complete([1, 2], lora_scale=0.0, temperature=0.1, top_p=1.0, top_k=50, min_p=0.0,
                    presence_penalty=0.0, repeat_penalty=1.1, repeat_last_n=514, seed=7, stop=["<|im_end|>"],
                    return_tokens=True)
    client.complete([1, 2], lora_scale=0.0, temperature=0.1, top_k=50, repeat_penalty=None)
    client.complete([1, 2], lora_scale=0.0, temperature=0.1, repeat_penalty=1.1)
    (_, penalised), (_, without), (_, default_window) = seen
    assert (penalised["repeat_penalty"], penalised["repeat_last_n"]) == (1.1, 514)
    assert list(penalised) == ["prompt", "max_tokens", "temperature", "cache_prompt", "logprobs", "stop",
                               "seed", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty",
                               "repeat_last_n", "return_tokens", "verbose", "response_fields"]
    assert not {"repeat_penalty", "repeat_last_n"} & set(without)
    assert "repeat_last_n" not in default_window  # sent only when given (core.Decoding always gives it)


def test_the_token_ids_come_with_the_logprob_items_and_are_empty_when_an_item_lacks_one():
    completion = LocalClient(transport=server([])).complete("PROMPT", lora_scale=0.0)
    assert completion.token_ids == [151657, 198] and completion.finish_reason == "stop"
    without = {"content": [{k: v for k, v in item.items() if k != "id"} for item in SERVER_FORMAT["content"]]}
    body = {"choices": [{"text": TEXT, "finish_reason": "stop", "logprobs": without}]}
    assert LocalClient(transport=server([], body=body)).complete("PROMPT", lora_scale=0.0).token_ids == []
    assert parse_token_ids(SERVER_FORMAT) == [151657, 198]
    assert parse_token_ids(LEGACY_FORMAT) == [] and parse_token_ids(None) == []
    assert parse_token_ids({"content": [{"token": "a", "logprob": -0.1, "id": "7"}]}) == []  # not an int


# ---- M7: the tokens a THINK generation's first request asks for (research_agent.llm.generation) ----------

ASKED = ("return_tokens", "verbose", "response_fields")


def verbose_server(seen: list, verbose: object = None) -> httpx.MockTransport:
    """server, answering a cut completion whose native fields ("__verbose") are the given value, if any."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/lora-adapters":
            return httpx.Response(200, json=[])
        seen.append(json.loads(request.content))
        body = {"choices": [{"text": TEXT, "finish_reason": "length", "logprobs": SERVER_FORMAT}],
                "usage": {"completion_tokens": 3}}
        if verbose is not None:
            body["__verbose"] = verbose
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def test_return_tokens_asks_for_every_generated_id_and_the_stop_type_and_reads_them_back():
    # b11349 lists no token after which the text ends inside a character among the logprobs, so the ids of
    # what it generated come from its native response, which "verbose" adds under "__verbose" and
    # "response_fields" narrows to the two fields (server-task.cpp: to_json_oaicompat, to_json_non_oaicompat)
    seen = []
    client = LocalClient(transport=verbose_server(seen, {"tokens": [151657, 7, 198], "stop_type": "limit"}))
    completion = client.complete([1, 2], lora_scale=0.0, return_tokens=True)
    (payload,) = seen
    assert list(payload)[-3:] == list(ASKED)
    assert {k: payload[k] for k in ASKED} == {"return_tokens": True, "verbose": True,
                                              "response_fields": ["tokens", "stop_type"]}
    assert completion.generated_ids == [151657, 7, 198] and completion.stop_type == "limit"
    assert completion.token_ids == [151657, 198]  # the logprob list, as it came


def test_without_return_tokens_nothing_more_is_asked_and_an_answer_without_the_fields_reads_none():
    seen = []
    completion = LocalClient(transport=verbose_server(seen)).complete([1, 2], lora_scale=0.0)
    assert not set(ASKED) & set(seen[0])
    assert completion.generated_ids is None and completion.stop_type is None
    asked = LocalClient(transport=verbose_server([])).complete([1, 2], lora_scale=0.0, return_tokens=True)
    assert asked.generated_ids is None and asked.stop_type is None  # a server that did not send them


@pytest.mark.parametrize("verbose", [{"tokens": [1, "2"], "stop_type": 7}, {"tokens": [1, True]},
                                     {"tokens": "1 2"}, ["tokens"], {"stop_type": "eos"}])
def test_a_malformed_token_list_or_stop_type_reads_as_none(verbose):
    completion = LocalClient(transport=verbose_server([], verbose)).complete([1], lora_scale=0.0,
                                                                              return_tokens=True)
    assert completion.generated_ids is None
    assert completion.stop_type == (verbose.get("stop_type") if isinstance(verbose, dict)
                                    and isinstance(verbose.get("stop_type"), str) else None)
