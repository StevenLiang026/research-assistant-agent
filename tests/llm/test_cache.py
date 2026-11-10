from research_agent.llm.cache import ResponseCache, request_key


def test_key_ignores_key_order():
    a = {"model": "m", "messages": [{"role": "user", "content": "你好"}], "temperature": 0}
    b = {"temperature": 0, "messages": [{"content": "你好", "role": "user"}], "model": "m"}
    assert request_key(a) == request_key(b)


def test_key_changes_when_any_field_changes():
    base = {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0}
    variants = [base, {**base, "temperature": 0.1}, {**base, "model": "n"},
                {**base, "messages": [{"role": "user", "content": "y"}]}, {**base, "tools": []}]
    assert len({request_key(v) for v in variants}) == len(variants)


def test_key_is_sha256_hex():
    key = request_key({"a": 1})
    assert len(key) == 64 and set(key) <= set("0123456789abcdef")


def test_roundtrip_and_miss(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    assert cache.get("missing") is None
    cache.put("k", "m", {"choices": [{"message": {"content": "推测解码"}}]})
    assert cache.get("k")["choices"][0]["message"]["content"] == "推测解码"
    assert len(cache) == 1
    cache.close()


def test_persists_across_instances(tmp_path):
    path = tmp_path / "sub" / "c.sqlite"
    first = ResponseCache(path)
    first.put("k", "m", {"x": 1})
    first.close()
    second = ResponseCache(path)
    assert second.get("k") == {"x": 1}
    second.close()


def test_in_memory_cache():
    cache = ResponseCache(":memory:")
    cache.put("k", "m", {"x": 1})
    assert cache.get("k") == {"x": 1}
