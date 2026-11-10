"""Two-phase generation on llama-server (docs/m7-家族层设计.md §7; docs/m7-计划.md §2 推理): what a model of a
family generates for a prompt, its thinking closed within a budget, read into the reasoning and the answer.

A family's NO_THINK prompt, or any prompt of a family that never thinks, is answered in one request of at
most answer_tokens tokens that stops at the family's stop strings. Under greedy decoding that request is
byte for byte the one agent.models.LocalModel sent before M7, the prompt as ids in place of its text
(research_agent.llm.local.LocalClient.complete), and the answer Completion is the server's as it came back.

A THINK prompt ends with think_open, and the model writes its reasoning first:

1. The first request asks for at most reasoning_budget (R) tokens, and for the ids of every token it
   generates with the stop type (Token ids, below).
2. When the text holds "</think>", what precedes it is the reasoning and what follows the tag and its
   newlines is the answer (LoadedFamily.split_reasoning). An answer cut short by the budget (finish_reason
   "length") goes on in a second request over the prompt's ids followed by every id the first request
   generated, for the answer tokens left of answer_tokens (the first request's tokens from the answer's
   first on count against them); one that already fills them is left as it is. Where the budget cut the
   answer inside a character, the ids go on exactly, but the text does not join whole: the server writes
   the cut character's first bytes as U+FFFD (it never sends the bytes of the tokens left out at the end of
   a completion) and the second request's leading continuation bytes as U+FFFD too.
3. Without the tag, the reasoning is closed by force: the second request's prompt is the prompt's ids, the
   ids the first request generated and the ids think_close has after a word (LoadedFamily.encode_close),
   and it asks for answer_tokens tokens. A turn the model ended inside its reasoning is closed the same
   way, without the end of that turn: when the server stopped on an end token (stop type "eos": the
   family's terminator or another end token of the model, MiniCPM5's eos "</s>", Qwen3's
   "<|endoftext|>"), that token, the last one generated, is dropped, since an answer cannot follow the end
   of a turn or of a sequence; when the model spelled a stop string as text (stop type "word"), the server
   erased it from the text and dropped from the list as many tokens as the string has encoded on its own
   (server-context.cpp, send_final_response: common_tokenize, special tokens not parsed), and the close
   follows the generated tokens up to the last one listed, none when no token is left listed. A known
   limit, left as it is: the cut is exact when the string starts at a token and the model spelled it with
   as many tokens as the server encodes it to. Otherwise it is off by the difference: spelled with fewer,
   or starting inside a token, the list also lost tokens of the reasoning (that token among them), whose
   text the reasoning keeps and the close does not follow; spelled with more, the close follows tokens of
   the string, whose text the server erased. It takes the model spelling a stop string out as text inside a
   reasoning it never closed: every family's only stop string is its terminator, one token that llama.cpp
   ends generation on (llama-vocab.cpp: "<|im_end|>", "<｜end▁of▁sentence｜>"), so a stop on the terminator
   itself is "eos", whose cut is exact.
4. The answer Completion holds the answer's listed tokens alone: the first request's from the boundary (a
   token straddling it is the answer's first) and the second request's. Its text starts where its first
   token does, so agent.route places its tokens in it; Generation.answer_text is what the model wrote after
   the close, which is parsed and recorded. The usage sums the requests'; raw and timings are the last
   request's.
5. The answer's tokens are counted on what the server generated, not on the listed ones, which leave out the
   parts of a character split across tokens (Token ids, below): Generation.answer_tokens is the first
   request's generated tokens after the reasoning's, and every token a request after the reasoning
   generated as its usage counts them (completion_tokens, the server's count of the tokens it generated, the
   ones left out of the list and the end token included). A NO_THINK generation's is its one request's
   count, None when the server sent no usage (that request asks for no ids, so that it stays today's bytes);
   a THINK generation's request after the reasoning without a count is a GenerationError.

Token ids. llama-server b11349 leaves out of its logprob list every token after which the text generated so
far ends inside a UTF-8 character (tools/server/server-context.cpp, process_token: validate_utf8 finds a lead
byte among the last four bytes without all its continuation bytes after it; a character split across tokens
is listed once, as its last token, with the bytes and the text of the ones before it), so the ids the list
holds can miss tokens the model generated (llm.local, The generated tokens). The first request of a THINK
generation therefore asks for the native list of every generated id (Completion.generated_ids; their number
must be the usage's completion tokens) and the stop type, and a second request goes on from those ids
exactly, the tokens left out of the logprob list included. The reasoning tokens are counted on the same
list. The logprob list is a subsequence of it, and each listed token is placed at the first generated token
with its id after the previous listed token's place; the reasoning tokens of a closed reasoning are the
generated tokens up to and with the last listed token before the answer's first, and the answer tokens the
first request generated, which count against answer_tokens, are the ones after them. The place is exact for
every listed token whose id no token left out of the list carries, whatever the places before it: the
generated tokens with that id are then exactly the listed ones with it, met in order. The last token before
the answer ends with the close or the newlines after it, and a token whose text ends with an ASCII character
is left out only after a lead byte without its continuation bytes, which no valid UTF-8 text holds, a token
without text (a control token) only where the model wrote it inside a character: the count is exact for
every reasoning in valid UTF-8 without a control token inside a character.

Seeds (docs/m7-计划.md §2 解码). A sampled request (temperature above 0) names its seed, request_seed of the
base seed, the prompt's ids, the phase (1 the first request, 2 the one after it) and the sample number, so
two runs with the same base seed repeat each other exactly on a one-slot server, and the samples of one
prompt, and the phases of one sample, draw differently. A greedy request names none: the sampler ignores it,
and the request stays today's bytes. Generation.seeds lists the seeds sent, None under greedy decoding.

Repetition penalty. Under a decoding with a repetition penalty on (LFM2.5's vendor sampling) every request
sends the penalty with its window, repeat_last_n, set to the request's reach, its prompt's ids and its
max_tokens (research_agent.family.core.Decoding: The penalty's window), so that at every step the penalty
covers the whole sequence, the prompt (and, after a forced close, the reasoning and the close) included, as
transformers applies it. A decoding without one sends neither field.

Text from ids (research_agent.family.core: Text from ids). llama-server writes no text and no bytes for a
control token, and a family whose tool-call or thinking tags are such tokens (MiniCPM5, LFM2.5) declares
text_from_ids: every request's completion then has each such token's text decoded from its id by the family
(family.decode, special tokens kept), and that text's UTF-8 bytes as the token's bytes. The completion's text
is then its listed tokens' bytes decoded as UTF-8: the bytes of the listed tokens are the text's bytes, a
control token having none and a listed token carrying the bytes of the ones left out before it (Token ids).
The token spans the signals read so still place every token in the text (agent.route.token_spans), and a
completion without the bytes of every listed token is a GenerationError.
The end token a stopped completion ends on keeps its empty text, as on the text path: the server stops on an
end-of-generation token and lists it last without bytes, and that is the family's terminator or another end
token of the model (MiniCPM5's eos "</s>", which is not its terminator); so does the terminator wherever it
stands. A family without the flag gets the server's completion as it came back, byte for byte, and never
decodes.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass

from ..agent.route import token_spans
from ..canonical import canonical_json
from ..family.core import Decoding, LoadedFamily, Mode
from .local import MAX_TOKENS, Completion

SEED_MASK = 0x7FFFFFFF  # fits a signed 32-bit integer; never llama.cpp's random seed, 0xFFFFFFFF
LENGTH = "length"  # llama-server's finish_reason of a completion the token budget cut short
STOP = "stop"  # its finish_reason of one that ended on an end token or a stop string
EOS, WORD, LIMIT = "eos", "word", "limit"  # its stop types (llm.local, The generated tokens)


class GenerationError(RuntimeError):
    """A generation that cannot go on: the server's answer lacks what the next request needs."""


def _require(ok: bool, message: str) -> None:
    """ValueError naming what is malformed unless ok."""
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class GenerationSettings:
    """How a model generates: the mode (which prompt, and whether a reasoning phase runs), the decoding, the
    reasoning budget R and the answer budget in tokens, and the base seed and sample number of a sampled run
    (module docstring: Seeds). votes (M7, docs/m7-本地提升计划.md WP2) is the number of samples a voted system
    draws besides its greedy answer (agent.vote.voted_settings: sample k with the base seed base_seed + k and
    sample number 0, the greedy one decoded greedily), 0 for a system that does not vote; it needs a decoding
    that samples, and generate itself never reads it."""

    mode: Mode
    decoding: Decoding
    reasoning_budget: int = 1024
    answer_tokens: int = MAX_TOKENS
    base_seed: int = 0
    sample: int = 0
    votes: int = 0

    def __post_init__(self) -> None:
        _require(isinstance(self.mode, Mode), f"generation: mode must be a Mode, got {self.mode!r}")
        _require(isinstance(self.decoding, Decoding),
                 f"generation: decoding must be a Decoding, got {self.decoding!r}")
        for name, least in (("reasoning_budget", 1), ("answer_tokens", 1), ("base_seed", 0), ("sample", 0),
                            ("votes", 0)):
            value = getattr(self, name)
            _require(not isinstance(value, bool) and isinstance(value, int) and value >= least,
                     f"generation: {name} must be a whole number from {least}, got {value!r}")
        _require(not self.votes or self.samples,
                 f"generation: votes need a decoding that samples, got {self.votes} votes with "
                 f"{self.decoding}")

    @classmethod
    def for_family(cls, family: LoadedFamily, *, mode: Mode | str | None = None,
                   reasoning_budget: int | None = None, base_seed: int = 0,
                   sample: int = 0) -> GenerationSettings:
        """The family's own settings: its inference mode (or mode), that mode's decoding, its reasoning
        budget (or reasoning_budget); ValueError for a mode the family does not have."""
        spec = family.spec
        chosen = spec.inference_mode if mode is None else Mode(mode)
        if chosen not in spec.modes:
            raise ValueError(f"{spec.name} has no {chosen} mode (its modes: {', '.join(spec.modes)})")
        return cls(mode=chosen, decoding=spec.decoding[chosen],
                   reasoning_budget=spec.reasoning_budget if reasoning_budget is None else reasoning_budget,
                   base_seed=base_seed, sample=sample)

    @property
    def samples(self) -> bool:
        """Whether the requests sample, and so carry a seed: the temperature is above 0."""
        return self.decoding.temperature > 0

    def record(self) -> dict:
        """What a run manifest keeps: every field, the decoding's five settings as they are, and its
        repetition penalty when it is on, with its window, "sequence": every request sends repeat_last_n as
        its prompt's ids and its max_tokens, the whole sequence it can reach (core.Decoding: The penalty's
        window). A decoding without one records what it recorded before the penalty, and settings that do not
        vote what they recorded before votes: "votes" only when voting."""
        decoding = dataclasses.asdict(self.decoding)
        if decoding["repetition_penalty"] == 1:
            del decoding["repetition_penalty"]
        else:
            decoding["repetition_window"] = "sequence"
        record = {"mode": str(self.mode), "decoding": decoding,
                  "reasoning_budget": self.reasoning_budget, "answer_tokens": self.answer_tokens,
                  "base_seed": self.base_seed, "sample": self.sample}
        if self.votes:
            record["votes"] = self.votes
        return record


def request_seed(base_seed: int, prompt_ids: list[int], phase: int, sample: int) -> int:
    """The seed of one request (module docstring: Seeds): the first four bytes of the SHA-256 of the
    canonical JSON of {"base_seed", "phase", "prompt_ids", "sample"} as a big-endian number, masked to 31
    bits."""
    encoding = canonical_json({"base_seed": base_seed, "phase": phase, "prompt_ids": list(prompt_ids),
                               "sample": sample})
    return int.from_bytes(hashlib.sha256(encoding.encode("utf-8")).digest()[:4], "big") & SEED_MASK


@dataclass(frozen=True)
class Generation:
    """What a prompt generated (module docstring): the reasoning (None in NO_THINK mode), the answer's text,
    the answer Completion (the answer's listed tokens alone), whether the reasoning was closed by force, the
    number of requests, their usage summed, the seeds sent (None under greedy decoding), the number of
    generated ids the reasoning occupies before the answer, the ones the server left out of its list included
    (module docstring: Token ids): the tokens before the answer's first, the close's among them, when the
    model closed its reasoning; the ids the forced close was appended to (without the end of a turn the model
    ended, module docstring 3) when it did not; and the number of tokens generated for the answer, the ones
    left out of the list and the end token included (module docstring 5; None when a NO_THINK generation's
    server counted none). So the prompt's ids, reasoning_tokens, a forced close's ids and answer_tokens are
    the step's whole sequence: the last request's prompt and every token it generated (scripts/run_e2e.py
    counts the steps whose sequence outgrows the training length from them)."""

    reasoning: str | None
    answer_text: str
    answer: Completion
    forced: bool
    phases: int
    usage: dict
    seeds: list[int] | None
    reasoning_tokens: int
    answer_tokens: int | None


def generate(client, family: LoadedFamily, prompt_ids: list[int], settings: GenerationSettings, *,
             lora_scale: float | None) -> Generation:
    """The generation of a prompt (its ids: family.encode_prompt of family.prompt in settings.mode) by the
    procedure of the module docstring. client is llm.local.LocalClient, lora_scale what it takes. ValueError
    for a mode the family does not have; GenerationError when the first request of a THINK generation came
    back without the ids of every token it generated and its stop type, without an id for every listed token,
    or with lists that disagree (module docstring: Token ids)."""
    spec = family.spec
    if settings.mode not in spec.modes:
        raise ValueError(f"{spec.name} has no {settings.mode} mode (its modes: {', '.join(spec.modes)})")
    run = _Run(client, family, prompt_ids, settings, lora_scale)
    if settings.mode is Mode.NO_THINK:
        answer = run.request(prompt_ids, settings.answer_tokens, phase=1)
        return run.done(reasoning=None, answer_text=answer.text, answer=answer, forced=False,
                        reasoning_tokens=0, answer_tokens=_counted(answer, required=False))
    first = run.request(prompt_ids, settings.reasoning_budget, phase=1, generated=True)
    generated = _generated(first)
    positions = _listed_positions(first, generated)
    reasoning, offset = family.split_reasoning(first.text)
    if reasoning is None:
        continued = _continued(first, generated, positions)
        answer = run.request(prompt_ids + continued + family.encode_close(),
                             settings.answer_tokens, phase=2)
        return run.done(reasoning=first.text.strip("\n"), answer_text=answer.text, answer=answer, forced=True,
                        reasoning_tokens=len(continued), answer_tokens=_counted(answer, required=True))
    start, text_start = _boundary(first, offset)
    reasoning_tokens = positions[start - 1] + 1 if start else 0
    answer, answer_text = _sliced(first, start, text_start), first.text[offset:]
    answer_tokens = len(generated) - reasoning_tokens  # the first request's after the reasoning
    left = settings.answer_tokens - answer_tokens
    if first.finish_reason == LENGTH and left > 0:
        rest = run.request(prompt_ids + generated, left, phase=2)
        answer, answer_text = _joined(answer, rest), answer_text + rest.text
        answer_tokens += _counted(rest, required=True)
    return run.done(reasoning=reasoning, answer_text=answer_text, answer=answer, forced=False,
                    reasoning_tokens=reasoning_tokens, answer_tokens=answer_tokens)


class _Run:
    """The requests of one generation: each with its prompt, token budget and phase, under the settings'
    decoding, the family's stop strings and the seed of a sampled request; their usage summed and their
    seeds kept."""

    def __init__(self, client, family: LoadedFamily, prompt_ids: list[int], settings: GenerationSettings,
                 lora_scale: float | None) -> None:
        self.client, self.family, self.prompt_ids, self.settings = client, family, prompt_ids, settings
        self.lora_scale = lora_scale
        self.seeds: list[int] = []
        self.usage: dict = {}
        self.phases = 0

    def request(self, prompt: list[int], max_tokens: int, *, phase: int,
                generated: bool = False) -> Completion:
        """One request; generated asks for the ids of every token it generates and its stop type (module
        docstring: Token ids), which only the first request of a THINK generation does: every other request
        sends what it sent before."""
        settings = self.settings
        seed = None
        if settings.samples:
            seed = request_seed(settings.base_seed, self.prompt_ids, phase, settings.sample)
            self.seeds.append(seed)
        asked = {"return_tokens": True} if generated else {}
        reach = len(prompt) + max_tokens  # a repetition penalty's window: the whole sequence (core.Decoding)
        completion = self.client.complete(prompt, lora_scale=self.lora_scale, max_tokens=max_tokens,
                                          stop=list(self.family.spec.stop), seed=seed,
                                          **settings.decoding.payload(reach=reach), **asked)
        if self.family.spec.text_from_ids:
            completion = _from_ids(completion, self.family)
        self.phases += 1
        usage = completion.raw.get("usage") or {}
        if self.phases == 1:
            self.usage = dict(usage)  # one request: the server's usage as it is, as the records had it
        else:
            for key, value in usage.items():
                total = self.usage.get(key, 0)
                if isinstance(value, int) and isinstance(total, int) and not isinstance(value, bool):
                    self.usage[key] = total + value
        return completion

    def done(self, *, reasoning: str | None, answer_text: str, answer: Completion, forced: bool,
             reasoning_tokens: int, answer_tokens: int | None) -> Generation:
        return Generation(reasoning=reasoning, answer_text=answer_text, answer=answer, forced=forced,
                          phases=self.phases, usage=self.usage,
                          seeds=self.seeds if self.settings.samples else None,
                          reasoning_tokens=reasoning_tokens, answer_tokens=answer_tokens)


def _ids_of(completion: Completion, purpose: str) -> list[int]:
    """The ids of the completion's listed tokens, one per token; GenerationError naming the purpose when the
    server sent a token without its id (the request asks for logprobs, LocalClient.complete's default, and
    b11349 sends every listed token's id). A completion may list no token although it has text: every token it
    generated left the text inside a character (module docstring: Token ids)."""
    ids, tokens = completion.token_ids, completion.tokens
    if len(ids) != len(tokens):
        raise GenerationError(f"the server sent no token id for every token of the completion, so {purpose}: "
                              "the request needs logprobs (LocalClient.complete's default)")
    return list(ids)


def _generated(completion: Completion) -> list[int]:
    """The ids of every token the completion generated (module docstring: Token ids); GenerationError when
    the server sent no list of them or no stop type it knows, or when their number is not its usage's
    completion tokens."""
    ids = completion.generated_ids
    if ids is None or completion.stop_type not in (EOS, WORD, LIMIT):
        raise GenerationError("the server sent no list of the tokens it generated with a stop type, so the "
                              "reasoning's tokens cannot be counted nor the generation go on from them: the "
                              "first request of a THINK generation asks for them (LocalClient.complete's "
                              f"return_tokens; stop type {completion.stop_type!r})")
    count = (completion.raw.get("usage") or {}).get("completion_tokens")
    if isinstance(count, int) and not isinstance(count, bool) and count != len(ids):
        raise GenerationError(f"the server lists {len(ids)} tokens it generated, but its usage counts "
                              f"{count} completion tokens")
    return list(ids)


def _counted(completion: Completion, *, required: bool) -> int | None:
    """The tokens a request generated, as its usage counts them (completion_tokens: every token, the ones
    the logprob list leaves out and the end token included; module docstring 5); None when the usage counts
    none, a GenerationError instead when required (a THINK generation's request after the reasoning, whose
    answer the step's length counts)."""
    count = (completion.raw.get("usage") or {}).get("completion_tokens")
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    if required:
        raise GenerationError("the server's usage counts no completion tokens for the request after the "
                              "reasoning, so the answer's tokens, which the logprob list can undercount "
                              "(module docstring: Token ids), cannot be counted")
    return None


def _listed_positions(completion: Completion, generated: list[int]) -> list[int]:
    """Where each listed token of the completion stands among the generated ones: the first generated token
    with its id after the previous listed token's (module docstring: Token ids); GenerationError when a listed
    token is not among them or has no id."""
    listed = _ids_of(completion, "they cannot be placed among the tokens it generated")
    positions, at = [], 0
    for token_id in listed:
        try:
            at = generated.index(token_id, at)
        except ValueError:
            raise GenerationError(f"the server lists token {token_id}, which is not among the tokens it "
                                  "generated after the ones listed before it") from None
        positions.append(at)
        at += 1
    return positions


def _continued(completion: Completion, generated: list[int], positions: list[int]) -> list[int]:
    """The generated ids a forced close follows (module docstring, 3): without the end token the server
    stopped on (stop type "eos"), up to the last listed token when a stop string ended the completion
    ("word": none when no token is left listed; exact only when the model spelled the string with the
    tokens the server encodes it to, the known limit of the module docstring), every one when the budget
    cut it ("limit")."""
    if completion.stop_type == EOS:
        return generated[:-1]
    if completion.stop_type == WORD:
        return generated[:positions[-1] + 1] if positions else []
    return list(generated)


def _stopped_on_end_token(completion: Completion) -> bool:
    """Whether the completion's last listed token is the end-of-generation token the server stopped on: the
    completion stopped (finish_reason "stop") and the server wrote no bytes for that token, as llama-server
    writes none for an end token, whereas the tokens it lists of a completion a stop string ended are the
    model's text before the string. The caller holds the bytes of every listed token."""
    sent = completion.token_bytes
    return completion.finish_reason == STOP and bool(sent) and not sent[-1]


def _from_ids(completion: Completion, family: LoadedFamily) -> Completion:
    """The completion with the text of its control tokens filled in from their ids (module docstring: Text
    from ids): a token the server sent no bytes for, other than the family's terminator and the end token a
    stopped completion ends on, takes the text its id decodes to as its string and that text's UTF-8 bytes
    as its bytes, and the text becomes the tokens' bytes decoded. The completion itself when no token needs
    it; GenerationError when the server did not send an id and the bytes for every token."""
    ids = _ids_of(completion, "its text cannot be read from them")
    sent = completion.token_bytes
    if sent is None or len(sent) != len(ids):
        raise GenerationError("the server sent no bytes for every token of the completion, so the text of "
                              "its control tokens cannot be placed in it: the request needs logprobs "
                              "(LocalClient.complete's default)")
    ended = len(ids) - 1 if _stopped_on_end_token(completion) else None  # the token the server stopped on
    pieces = {}
    for index, (data, token_id) in enumerate(zip(sent, ids, strict=True)):
        if (not data and token_id != family.turn_end_id and index != ended
                and (piece := family.decode([token_id]))):
            pieces[index] = piece
    if not pieces:
        return completion
    tokens = [pieces.get(index, token) for index, token in enumerate(completion.tokens)]
    token_bytes = [list(pieces[index].encode("utf-8")) if index in pieces else list(data)
                   for index, data in enumerate(sent)]
    text = bytes(b for data in token_bytes for b in data).decode("utf-8", errors="replace")
    return dataclasses.replace(completion, text=text, tokens=tokens, token_bytes=token_bytes)


def _boundary(completion: Completion, offset: int) -> tuple[int, int]:
    """The index of the answer's first token, the first that reaches past the text offset where the answer
    starts (a token straddling it included) or, failing that, the first that starts at it (a token without
    text, such as the end token), and the text offset that token starts at; (the number of tokens, offset)
    when no token is the answer's."""
    spans = token_spans(completion.text, completion.tokens, completion.token_bytes)
    for index, (start, end) in enumerate(spans):
        if end > offset or start >= offset:
            return index, start
    return len(spans), offset


def _sliced(completion: Completion, start: int, text_start: int) -> Completion:
    """The completion's listed tokens from start, its text from text_start; the rest as it is, but for the
    list of every generated token and the stop type, which are the whole completion's (None)."""
    token_bytes = completion.token_bytes
    return dataclasses.replace(
        completion, text=completion.text[text_start:], tokens=completion.tokens[start:],
        logprobs=completion.logprobs[start:], top_logprobs=completion.top_logprobs[start:],
        token_bytes=None if token_bytes is None else token_bytes[start:],
        token_ids=completion.token_ids[start:], generated_ids=None, stop_type=None)


def _joined(head: Completion, rest: Completion) -> Completion:
    """head's tokens followed by rest's; the finish reason, raw response and timings are rest's, the wall
    time the sum."""
    both = head.token_bytes is not None and rest.token_bytes is not None
    return Completion(text=head.text + rest.text, tokens=head.tokens + rest.tokens,
                      logprobs=head.logprobs + rest.logprobs,
                      top_logprobs=head.top_logprobs + rest.top_logprobs,
                      finish_reason=rest.finish_reason, timings=rest.timings,
                      wall_ms=round(head.wall_ms + rest.wall_ms, 1), raw=rest.raw,
                      token_bytes=head.token_bytes + rest.token_bytes if both else None,
                      token_ids=head.token_ids + rest.token_ids)
