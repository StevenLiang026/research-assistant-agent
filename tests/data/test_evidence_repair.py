"""The evidence repair of the synthetic training records (research_agent.data.evidence_repair): a scaffold
note whose evidence is not a verbatim quotation gets sentences quoted from the text the model could see."""

from __future__ import annotations

import json
import re

from research_agent.data.evidence_repair import repair_evidence
from research_agent.eval.e2e import evidence_check

ID, OTHER = "2609.00001", "2609.00002"
ABSTRACT = ("Speculative decoding drafts tokens with a small model. The target model verifies every drafted "
            "token in a single forward pass. Acceptance rates decide how long the next draft should be. "
            "The method speeds up generation without changing the output distribution.")
OTHER_ABSTRACT = ("Tree verification checks many drafted branches at once on large batches. It keeps the "
                  "accelerator busy when single drafts are too short to fill it.")
SYSTEM = {"role": "system", "content": "You are a research assistant."}
USER = {"role": "user", "content": f"Read {ID} and write a scaffold note."}
QUOTED = re.compile(r'^"[^"]+"( / "[^"]+")*$')


def call(cid: str, name: str, **args) -> dict:
    return {"id": cid, "type": "function", "function": {"name": name,
                                                        "arguments": json.dumps(args, ensure_ascii=False)}}


def assistant(*calls: dict, content: str = "") -> dict:
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = list(calls)
    return message


def tool(cid: str, body: object) -> dict:
    return {"role": "tool", "tool_call_id": cid, "content": json.dumps(body, ensure_ascii=False)}


def paper(arxiv_id: str = ID, abstract: str = ABSTRACT) -> dict:
    return {"arxiv_id": arxiv_id, "title": "Adaptive Draft Length", "authors": ["A. Author"],
            "published": "2026-09-15", "abstract": abstract}


def note(evidence: object, cid: str = "c2", mode: str = "scaffold", **extra) -> dict:
    return call(cid, "write_note", arxiv_id=ID, mode=mode, evidence=evidence, **extra)


def history(note_call: dict, *, abstract: str = ABSTRACT, final: str | None = f"Noted {ID}.") -> list[dict]:
    messages = [SYSTEM, USER, assistant(call("c1", "get_paper", arxiv_id=ID)),
                tool("c1", paper(abstract=abstract)), assistant(note_call),
                tool(note_call["id"], {"note_id": "n1", "arxiv_id": ID, "path": "n1.md"})]
    return messages + ([assistant(content=final)] if final is not None else [])


def evidence_of(messages: list[dict], cid: str = "c2") -> str:
    for message in messages:
        for c in message.get("tool_calls") or []:
            if c["id"] == cid:
                return json.loads(c["function"]["arguments"])["evidence"]
    raise AssertionError(f"no call {cid}")


def test_a_paraphrased_scaffold_evidence_becomes_verbatim_sentences_of_the_visible_abstract():
    original = history(note("Drafts are checked by the big model, which saves time."))
    repair = repair_evidence(original)
    evidence = evidence_of(repair.messages)
    assert QUOTED.match(evidence), evidence
    for sentence in re.findall(r'"([^"]+)"', evidence):
        assert sentence in ABSTRACT
    assert evidence_check(repair.messages)["ok"] and not evidence_check(original)["ok"]
    assert [c["call_id"] for c in repair.calls] == ["c2"] and not repair.unrepairable
    assert repair.calls[0]["sources"] == {"abstract": len(re.findall(r'"([^"]+)"', evidence))}


def test_a_passing_or_draft_note_and_a_note_without_string_evidence_are_left_byte_identical():
    verbatim = '"The target model verifies every drafted token in a single forward pass."'
    for messages in (history(note(verbatim)),
                     history(note("Drafts are checked by the big model.", mode="draft", problem="p",
                                  method="m", limitations="l")),
                     history(call("c2", "write_note", arxiv_id=ID, mode="scaffold"))):
        repair = repair_evidence(messages)
        assert repair.messages == messages and repair.calls == [] and not repair.final_answer_dropped


def test_the_guards_never_pick_a_cut_tail_a_quote_a_short_span_a_line_break_hyphen_or_a_special_tag():
    abstract = ('Results with "quotes" inside a sentence are long enough to count here. Too short here. '
                "The inter-\nnational standards apply to every single case we measured. "
                "The model emits <tool_call> markers in its plain text answers sometimes. "
                "The verifier accepts most drafted tokens on code and math benchmarks. "
                "The last sentence is cut off before it ends and has no")
    repair = repair_evidence(history(note("Most drafts get accepted by the checker."), abstract=abstract))
    expected = '"The verifier accepts most drafted tokens on code and math benchmarks."'
    assert evidence_of(repair.messages) == expected


def test_sentences_come_only_from_the_previous_step_and_only_from_the_notes_paper():
    fulltext = {"arxiv_id": ID, "section": "method", "source": "html",
                "text": "An older method section sentence that is compressed away by now."}
    messages = [SYSTEM, USER,
                assistant(call("c0", "fetch_fulltext", arxiv_id=ID, section="method")), tool("c0", fulltext),
                assistant(call("c1", "get_paper", arxiv_id=OTHER), call("c3", "get_paper", arxiv_id=ID)),
                tool("c1", paper(OTHER, OTHER_ABSTRACT)), tool("c3", paper()),
                assistant(note("Drafts are checked by the big model.")),
                tool("c2", {"note_id": "n1", "arxiv_id": ID, "path": "n1.md"}), assistant(content="Done.")]
    evidence = evidence_of(repair_evidence(messages).messages)
    sentences = re.findall(r'"([^"]+)"', evidence)
    assert sentences and all(s in ABSTRACT for s in sentences)


def test_ranking_prefers_what_the_teacher_paraphrased_keeps_four_in_text_order_and_is_deterministic():
    abstract = ABSTRACT + (" A second paragraph repeats the setup for code models."
                           " A third paragraph covers the limits on long prompts.")
    paraphrase = "It speeds up generation without changing the output distribution, a lossless method."
    first = repair_evidence(history(note(paraphrase), abstract=abstract))
    sentences = re.findall(r'"([^"]+)"', evidence_of(first.messages))
    assert len(sentences) == 4
    assert "The method speeds up generation without changing the output distribution." in sentences
    assert sentences == sorted(sentences, key=abstract.index)
    assert repair_evidence(history(note(paraphrase), abstract=abstract)).messages == first.messages


def test_only_the_failing_note_of_two_is_repaired():
    good = note('"Acceptance rates decide how long the next draft should be."', cid="c2")
    bad = note("The draft model is tiny.", cid="c4")
    messages = [SYSTEM, USER, assistant(call("c1", "get_paper", arxiv_id=ID)), tool("c1", paper()),
                assistant(good), tool("c2", {"note_id": "n1", "arxiv_id": ID, "path": "n1.md"}),
                assistant(call("c5", "get_paper", arxiv_id=ID)), tool("c5", paper()),
                assistant(bad), tool("c4", {"note_id": "n2", "arxiv_id": ID, "path": "n2.md"}),
                assistant(content="Two notes.")]
    repair = repair_evidence(messages)
    assert evidence_of(repair.messages, "c2") == evidence_of(messages, "c2")
    assert evidence_of(repair.messages, "c4") != evidence_of(messages, "c4")
    assert [c["call_id"] for c in repair.calls] == ["c4"]


def test_a_note_whose_paper_has_no_text_in_the_previous_step_is_unrepairable():
    messages = [SYSTEM, USER, assistant(call("c1", "get_paper", arxiv_id=ID)), tool("c1", paper()),
                assistant(call("c3", "get_paper", arxiv_id=OTHER)), tool("c3", paper(OTHER, OTHER_ABSTRACT)),
                assistant(note("Drafts are checked by the big model.")),
                tool("c2", {"note_id": "n1", "arxiv_id": ID, "path": "n1.md"}), assistant(content="Done.")]
    repair = repair_evidence(messages)
    assert repair.unrepairable == ["c2"] and repair.calls == []


def test_a_final_answer_restating_details_only_the_old_evidence_held_is_dropped():
    detail = "uses a seven billion parameter draft model trained on code"
    messages = history(note(f"The paper {detail}, and the target checks it."),
                       final=f"I noted {ID}: it {detail}.")
    repair = repair_evidence(messages)
    assert repair.final_answer_dropped and repair.messages[-1]["role"] == "tool"
    assert len(repair.messages) == len(messages) - 1


def test_a_final_answer_restating_only_visible_content_is_kept():
    final = f"I noted {ID}: the target model verifies every drafted token in a single forward pass."
    messages = history(note("The checker is the big model, which looks at drafts in one go."), final=final)
    repair = repair_evidence(messages)
    assert not repair.final_answer_dropped and repair.messages[-1]["role"] == "assistant"


def test_the_repaired_arguments_keep_every_other_key_and_its_order():
    repair = repair_evidence(history(note("Drafts are checked by the big model.", new_terms=["drafting"],
                                          open_terms=["acceptance rate"])))
    args = json.loads(repair.messages[4]["tool_calls"][0]["function"]["arguments"])
    assert list(args) == ["arxiv_id", "mode", "evidence", "new_terms", "open_terms"]
    assert (args["arxiv_id"], args["mode"], args["new_terms"], args["open_terms"]) == (
        ID, "scaffold", ["drafting"], ["acceptance rate"])


def test_a_final_answer_restating_a_number_only_the_old_evidence_held_is_dropped():
    messages = history(note("The paper reports a 2.8x speedup on code."),
                       final=f"Noted {ID}; it is 2.8x faster.")
    repair = repair_evidence(messages)
    assert repair.final_answer_dropped and len(repair.messages) == len(messages) - 1
