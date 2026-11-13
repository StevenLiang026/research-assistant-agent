"""Live use of M7's final system (docs/m9-计划.md WP1): the frozen system unchanged, on the real arXiv,
with a notes library that lasts from one run to the next.

    frozen   the frozen record checked at every start (load_frozen) and a run's identity held against it
             (identity_problems)
    world    the live tools: the arXiv tools over a live fetcher whose rate limits hold across runs, the
             papers a run has seen kept for write_note, and the notes library
    session  one turn as the evaluation runs it, a follow-up after a clarification, the progress lines and
             events, the single-instance lock and the files every run writes
    render   an answer or a note, Markdown as the model writes it, as the page's HTML: every piece escaped,
             arXiv ids as links to their abstract pages
    web      the chat page of ask.py --web: a server on 127.0.0.1 for the page in page/, its access checks,
             the state the page shows and the event stream that keeps it current

scripts/ask.py is the command line over them; scripts/check_final_identity.py and scripts/live_check.py check
what a run wrote.
"""
