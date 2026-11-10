"""Tool schemas and topics shared by the spike scripts (seven tools, as in the plan)."""

def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


TOOLS = [
    fn("search_arxiv", "Search arXiv for papers matching a query. Returns id, title, authors, date and a short "
       "abstract snippet for each hit. Use date_from/date_to to restrict the submission date.",
       {"query": {"type": "string", "description": "Search terms, 1-300 characters"},
        "categories": {"type": "array", "items": {"type": "string", "enum": [
            "cs.CL", "cs.LG", "cs.AI", "cs.IR", "cs.CV", "stat.ML"]}},
        "date_from": {"type": "string", "description": "YYYY-MM-DD"},
        "date_to": {"type": "string", "description": "YYYY-MM-DD"},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
        "sort_by": {"type": "string", "enum": ["relevance", "submittedDate", "lastUpdatedDate"]}},
       ["query"]),
    fn("get_paper", "Get metadata and the full abstract of one arXiv paper.",
       {"arxiv_id": {"type": "string", "description": "e.g. 2609.01234"}}, ["arxiv_id"]),
    fn("fetch_fulltext", "Fetch one section of the paper's HTML full text.",
       {"arxiv_id": {"type": "string"},
        "section": {"type": "string", "enum": ["introduction", "method", "experiments", "conclusion", "all"]},
        "max_chars": {"type": "integer", "minimum": 500, "maximum": 8000}}, ["arxiv_id"]),
    fn("search_notes", "Search the owner's past reading notes.",
       {"query": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]),
    fn("write_note", "Write a reading note in the owner's template.",
       {"arxiv_id": {"type": "string"}, "mode": {"type": "string", "enum": ["scaffold", "draft"]},
        "problem": {"type": "string"}, "method": {"type": "string"}, "evidence": {"type": "string"},
        "limitations": {"type": "string"}, "new_terms": {"type": "array", "items": {"type": "string"}}},
       ["arxiv_id", "mode"]),
    fn("list_notes", "List saved notes.",
       {"tag": {"type": "string"}, "since": {"type": "string"}, "limit": {"type": "integer"}}, []),
    fn("ask_clarification", "Ask the user for missing information; ends the turn.",
       {"question": {"type": "string"}, "missing": {"type": "array", "items": {"type": "string", "enum": [
           "topic", "paper_id", "date_range", "count", "tags"]}}}, ["question"]),
]

TOPICS = ["speculative decoding", "推测解码", "mixture of experts", "test-time compute", "retrieval augmented "
          "generation", "LoRA fine-tuning", "长上下文", "tool use agents", "reward hacking", "KV cache compression"]
