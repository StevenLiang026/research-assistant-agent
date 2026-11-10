"""Tool contract shared by the agent framework and the owner's tool functions.

Shapes only: argument types, enums, caps and defaults. Identifier formats, date rules and "this id appeared
in the turn" are semantic checks that belong to the owner's validation module (check_call), so they are
absent here.

The caps come from the token budget (2,048 at first, 3,072 from 2026-10-09, 4,096 since 2026-10-13), measured
with the Qwen3 template: five search hits take about 600 tokens and a 1,500-character section about 255. TOOLS
is the full schema the owner's check_call validates against; MODEL_TOOLS is what the model sees (training and
inference alike), without the validation-only keywords. The system prompt, a one-line request and MODEL_TOOLS
take about 1,048 tokens (measured 2026-10-04), so the rest of the budget holds the history and the reply.
"""

from __future__ import annotations

import hashlib

from ..canonical import canonical_json

# 2,048 -> 3,072 on 2026-10-09 (peak VRAM 5.07 GB, docs/m1-进度.md); 3,072 -> 4,096 on 2026-10-13: the M2 dry
# run overflowed 3,072 in 4 of 10 requests and the peak at 4,096 is 5.27 GB reserved, within the 7.0 GB
# criterion (docs/m2-进度.md, runs/m1/peak_4096.json).
CONTEXT_TOKENS = 4096
REPLY_RESERVE_TOKENS = 256
MAX_CALLS_PER_STEP = 3
MAX_SEARCH_RESULTS = 5
MAX_SNIPPET_CHARS = 200
MAX_ABSTRACT_CHARS = 1000
MAX_FULLTEXT_CHARS = 1500
MAX_NOTE_HITS = 5
MAX_AUTHORS = 5  # get_paper lists the first five authors and then "et al."

# What the owner's parse_atom must return for every paper: a bare canonical id (YYMM.NNNNN, no "arXiv:"
# prefix, no version), the title, a list of author names, the first-version date as YYYY-MM-DD, the abstract,
# and the list of arXiv category terms in feed order (added 2026-10-12 so the replay search over the recorded
# corpus can honour search_arxiv's categories filter; no tool shows it to the model).
PAPER_KEYS = ["arxiv_id", "title", "authors", "published", "abstract", "categories"]
PAPER_LIST_KEYS = ["authors", "categories"]  # the PAPER_KEYS whose value is a list of strings, not a string

CATEGORIES = ["cs.CL", "cs.LG", "cs.AI", "cs.IR", "cs.CV", "stat.ML"]
SECTIONS = ["introduction", "method", "experiments", "conclusion", "all"]
MISSING = ["topic", "paper_id", "date_range", "count", "tags"]

# Error codes a tool result can carry (the model sees them).
EMPTY_RESULTS = "EMPTY_RESULTS"
NOT_FOUND = "NOT_FOUND"
NO_FULLTEXT = "NO_FULLTEXT"  # no HTML full text: the abstract is returned instead and the note is marked
RATE_LIMITED = "RATE_LIMITED"
UPSTREAM_ERROR = "UPSTREAM_ERROR"
INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
INVALID_JSON = "INVALID_JSON"
UNKNOWN_TOOL = "UNKNOWN_TOOL"
SKIPPED = "SKIPPED"  # not executed, e.g. issued together with ask_clarification
ERROR_CODES = [EMPTY_RESULTS, NOT_FOUND, NO_FULLTEXT, RATE_LIMITED, UPSTREAM_ERROR, INVALID_ARGUMENTS,
               INVALID_JSON, UNKNOWN_TOOL, SKIPPED]

# Failure events the loop reports to the owner's decide(event, times) and the actions it may answer with.
EVENTS = ["bad_json", "unknown_tool", "invalid_args", "empty_results", "repeat_call"]
ACTIONS = ["repair", "handoff", "relax", "tell_user", "stop"]
# times counts these three events together within the current step, whichever one each rejected attempt had
# (so "repair once, then hand off" holds for every step) ...
PER_STEP_EVENTS = ["bad_json", "unknown_tool", "invalid_args"]
# ... and empty_results and repeat_call within the whole turn ("relax once, then tell the user").
# An answer outside the event's list is a bug in the policy: the turn ends as a harness error.
ALLOWED_ACTIONS = {
    "bad_json": ["repair", "handoff", "stop"],
    "unknown_tool": ["repair", "handoff", "stop"],
    "invalid_args": ["repair", "handoff", "stop"],
    "empty_results": ["relax", "tell_user"],
    "repeat_call": ["stop", "tell_user"],
}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


TOOLS = [
    _tool("search_arxiv", "Search arXiv. Returns id, title, date and a short snippet per hit.",
          {"query": {"type": "string", "minLength": 1, "maxLength": 300},
           "categories": {"type": "array", "items": {"type": "string", "enum": CATEGORIES},
                          "uniqueItems": True},
           "date_from": {"type": "string", "description": "YYYY-MM-DD"},
           "date_to": {"type": "string", "description": "YYYY-MM-DD"},
           "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_RESULTS,
                           "default": MAX_SEARCH_RESULTS},
           "sort_by": {"type": "string", "enum": ["relevance", "submittedDate"], "default": "relevance"}},
          ["query"]),
    _tool("get_paper", "Metadata and abstract of one arXiv paper.",
          {"arxiv_id": {"type": "string"}}, ["arxiv_id"]),
    _tool("fetch_fulltext", "One section of a paper's HTML full text; falls back to the abstract.",
          {"arxiv_id": {"type": "string"},
           "section": {"type": "string", "enum": SECTIONS, "default": "introduction"},
           "max_chars": {"type": "integer", "minimum": 200, "maximum": MAX_FULLTEXT_CHARS,
                         "default": MAX_FULLTEXT_CHARS}},
          ["arxiv_id"]),
    _tool("search_notes", "Search the owner's past reading notes.",
          {"query": {"type": "string", "minLength": 1},
           "tags": {"type": "array", "items": {"type": "string"}},
           "top_k": {"type": "integer", "minimum": 1, "maximum": MAX_NOTE_HITS, "default": 3}},
          ["query"]),
    _tool("write_note",
          "Write a reading note. scaffold: evidence sentences only; draft: also the four parts.",
          {"arxiv_id": {"type": "string"},
           "mode": {"type": "string", "enum": ["scaffold", "draft"], "default": "scaffold"},
           "problem": {"type": "string"}, "method": {"type": "string"},
           "evidence": {"type": "string"}, "limitations": {"type": "string"},
           "new_terms": {"type": "array", "items": {"type": "string"}},
           "open_terms": {"type": "array", "items": {"type": "string"}}},
          ["arxiv_id"]),
    _tool("list_notes", "List saved notes.",
          {"tag": {"type": "string"}, "since": {"type": "string", "description": "YYYY-MM-DD"},
           "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10}},
          []),
    _tool("ask_clarification", "Ask the user for missing information. Ends the turn.",
          {"question": {"type": "string", "minLength": 1},
           "missing": {"type": "array", "items": {"type": "string", "enum": MISSING}}},
          ["question"]),
]

_VALIDATION_ONLY = ("additionalProperties", "minLength", "maxLength", "uniqueItems", "default")


def _model_view(node: object) -> object:
    if isinstance(node, dict):
        return {k: _model_view(v) for k, v in node.items() if k not in _VALIDATION_ONLY}
    if isinstance(node, list):
        return [_model_view(v) for v in node]
    return node


MODEL_TOOLS = [_model_view(t) for t in TOOLS]
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
DEFAULTS = {t["function"]["name"]: {k: v["default"]
                                    for k, v in t["function"]["parameters"]["properties"].items()
                                    if "default" in v} for t in TOOLS}
SCHEMA_HASH = hashlib.sha256(canonical_json(TOOLS).encode("utf-8")).hexdigest()[:16]
