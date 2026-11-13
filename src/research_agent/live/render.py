"""Markdown to HTML for the page (docs/m10-计划.md 4, 安全的文本渲染): a turn's answer and a note are rendered
here, on the server, so the page inserts no HTML that did not come from this module.

render_html(text) reads a small subset of Markdown in one pass over the lines and escapes every piece of the
input as it emits it (html.escape, quotes included): no pattern ever runs over HTML already produced, and no
link, image or raw HTML of the input survives as markup. The output holds only the tags and attributes of
ALLOWED, every element closed in order, with no line break between blocks.

Blocks. Lines end at "\\n", "\\r\\n" or "\\r"; a line of spaces and tabs alone is blank; indentation is
counted in columns, a tab advancing to the next multiple of TAB_STOP.
- A fenced code block: three or more backticks after any indentation, then an info string without a backtick
  (dropped), up to a line of at least as many backticks and nothing else, or to the end of the text. Its
  lines, each less up to as many leading spaces as the fence's indentation has columns, as <pre><code>, read
  no further.
- An ATX heading: one to six "#" after at most three spaces, then a space, a tab or the end of the line;
  demoted below the page's own headings: # is h3, ## h4, ### h5, and #### to ###### are h6. A closing run of
  "#" (after a space or a tab, or the whole text) is dropped. A heading closes every open list.
- A list item: "-" or "*" (ul), or a number of at most nine ASCII digits and "." (ol; the number is not kept),
  then a space, a tab or the end of the line. An item indented deeper than the open item above it nests under
  it (notes.render_entry writes two spaces per level); one indented less closes the deeper lists first; one at
  the same indentation is the next item of that list, or starts a new list when it switches between ul and ol.
  A blank line does not end a list.
- Any other line, and a fence, first closes the open lists whose markers stand at its indentation or right of
  it, then continues the innermost item still open: the line's text after a <br> (two after a blank line, none
  right after a nested list), the fence's block as it is. With no item open, the fence's block stands alone
  and the line starts or continues a paragraph: lines each stripped, joined by <br>, up to a blank line, a
  heading, a fence or an item. So a line that is not indented ends a list (there is no lazy continuation).

Inline, in the text of headings, items and every other line: one regular expression of alternatives
(_INLINE), the leftmost match first; every piece of text between matches is escaped as it stands.
- A code span: a run of backticks up to the next run of as many (when both ends of its text are spaces and
  the text is not all spaces, one is dropped at each end), as <code>; its text is never linked.
- An arxiv.org abs or pdf URL (https://, http:// or neither, www. or not, a version and ".pdf" allowed, any
  letter case) as one link that shows the URL as written.
- **bold**: the opening "**" followed by neither white space nor "*", the closing one preceded by no white
  space, no "**" within; as <strong>, its text read inline again.
- A bare new-style id, matched by eval.metrics._ARXIV_ID's pattern, the one the evaluation reads cited ids
  with: never inside a longer number (12345.678901 and 2026.11 hold none), a version suffix allowed.
A link's href is ABS_URL and the id _ARXIV_ID matches in the token with its version dropped, when
normalize_arxiv_id keeps it (else the token stays plain text); it carries target="_blank" and
rel="noopener noreferrer", and its text is the token as written. So, outside code spans, fenced blocks and
fence lines, each occurrence of an id eval.metrics.arxiv_ids reads in the text is exactly one link.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from ..eval.metrics import _ARXIV_ID
from ..normalize import normalize_arxiv_id
from ..notes import ABS_URL

# The tags render_html writes, each with the attributes it may carry; nothing else ever appears in its output.
ALLOWED: dict[str, frozenset[str]] = {
    **dict.fromkeys(("p", "br", "strong", "code", "pre", "ul", "ol", "li", "h3", "h4", "h5", "h6"),
                    frozenset()),
    "a": frozenset({"href", "target", "rel"}),
}
LINK_TARGET, LINK_REL = "_blank", "noopener noreferrer"
HEADING_SHIFT = 2  # "#" is h3: the page's own headings stay above the rendered ones
LOWEST_HEADING = 6
TAB_STOP = 4
_SPACE = " \t"  # what strips a line and makes one blank

_FENCE = re.compile(r"(?P<indent>[ \t]*)(?P<fence>`{3,})(?P<info>[^`]*)")  # fullmatch: an opening fence
_FENCE_END = re.compile(r"[ \t]*(?P<fence>`{3,})[ \t]*")  # fullmatch: closes one at most as long
_HEADING = re.compile(r" {0,3}(?P<hashes>#{1,6})(?:[ \t]+(?P<text>.*))?")  # fullmatch
_ITEM = re.compile(r"(?P<indent>[ \t]*)(?P<marker>[-*]|[0-9]{1,9}\.)(?:[ \t]+(?P<text>.*))?")  # fullmatch
_INLINE = re.compile("|".join((
    r"(?<!`)(?P<ticks>`+)(?!`)(?P<code>.+?)(?<!`)(?P=ticks)(?!`)",
    r"(?P<url>(?i:(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/)" + _ARXIV_ID.pattern + r"(?i:\.pdf)?)",
    r"\*\*(?![\s*])(?P<bold>(?:[^*\n]|\*(?!\*))+?)(?<!\s)\*\*",
    r"(?P<id>" + _ARXIV_ID.pattern + ")",
)))


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _columns(indent: str) -> int:
    """The width of indentation made of spaces and tabs (a tab advances to the next multiple of TAB_STOP)."""
    return len(indent.expandtabs(TAB_STOP))


def _without_closing_hashes(text: str) -> str:
    """A heading's text, stripped, less its closing run of "#" (module docstring: Blocks), with the white
    space before it. Read in time linear in the text's length: a pattern that tries every start inside a
    long run of white space takes time growing with the square of its length."""
    kept = text.rstrip("#")
    if len(kept) < len(text) and (not kept or kept[-1] in _SPACE):
        return kept.rstrip(_SPACE)
    return text


# ---- inline -----------------------------------------------------------------------------------------------


def _code_text(text: str) -> str:
    """A code span's text: one space dropped at each end when both ends are spaces and not all of it is."""
    return text[1:-1] if text[0] == text[-1] == " " and text.strip(" ") else text


def _link(token: str) -> str:
    """token, an arxiv.org URL or a bare id, as a link to the abstract page of the id _ARXIV_ID matches in it,
    its version dropped; plain text when normalize_arxiv_id does not keep that id."""
    arxiv_id = normalize_arxiv_id(_ARXIV_ID.search(token).group(1))
    if arxiv_id is None:
        return _escape(token)
    return (f'<a href="{_escape(ABS_URL + arxiv_id)}" target="{LINK_TARGET}" rel="{LINK_REL}">'
            f"{_escape(token)}</a>")


def _inline(text: str) -> str:
    """One line's text as HTML (module docstring: Inline)."""
    parts, start = [], 0
    for match in _INLINE.finditer(text):
        parts.append(_escape(text[start:match.start()]))
        start = match.end()
        if match.group("code") is not None:
            parts.append(f"<code>{_escape(_code_text(match.group('code')))}</code>")
        elif match.group("bold") is not None:
            parts.append(f"<strong>{_inline(match.group('bold'))}</strong>")
        else:  # an arxiv.org URL or a bare id
            parts.append(_link(match.group()))
    parts.append(_escape(text[start:]))
    return "".join(parts)


# ---- blocks -----------------------------------------------------------------------------------------------


@dataclass
class _List:
    """An open list: its tag, the column of its items' markers, and whether its open item ends with inline
    text (a line that continues the item then starts with <br>)."""

    tag: str
    column: int
    text: bool = False


class _Writer:
    """The HTML of the blocks so far, the open paragraph's lines (as HTML) and the open lists, innermost
    last (module docstring: Blocks)."""

    def __init__(self) -> None:
        self.out: list[str] = []
        self.paragraph: list[str] = []
        self.lists: list[_List] = []

    def end_paragraph(self) -> None:
        if self.paragraph:
            self.out.append(f"<p>{'<br>'.join(self.paragraph)}</p>")
            self.paragraph = []

    def close_lists(self, column: int) -> None:
        """Close every open list whose markers stand at column or deeper."""
        closed = False
        while self.lists and self.lists[-1].column >= column:
            self.out.append(f"</li></{self.lists.pop().tag}>")
            closed = True
        if closed and self.lists:
            self.lists[-1].text = False  # that item now ends with a nested list

    def heading(self, level: int, content: str) -> None:
        self.end_paragraph()
        self.close_lists(0)
        self.out.append(f"<h{level}>{content}</h{level}>")

    def fence(self, column: int, lines: list[str]) -> None:
        """A fenced block, inside the innermost open item whose marker stands left of column."""
        self.end_paragraph()
        self.close_lists(column)
        code = _escape("\n".join(lines))
        self.out.append(f"<pre><code>{code}</code></pre>")
        if self.lists:
            self.lists[-1].text = False

    def item(self, column: int, tag: str, content: str) -> None:
        self.end_paragraph()
        self.close_lists(column + 1)
        top = self.lists[-1] if self.lists else None
        if top is not None and top.column == column and top.tag == tag:
            self.out.append("</li><li>")
        else:
            if top is not None and top.column == column:  # the other kind of list: a new list
                self.close_lists(column)
            self.out.append(f"<{tag}><li>")
            self.lists.append(_List(tag, column))
        self.out.append(content)
        self.lists[-1].text = bool(content)

    def line(self, column: int, content: str, blank: bool) -> None:
        """A line that is no block of its own: it continues the innermost open item whose marker stands left
        of column, else the paragraph; blank says a blank line came just before it."""
        self.close_lists(column)
        if not self.lists:
            self.paragraph.append(content)
            return
        top = self.lists[-1]
        if top.text:
            self.out.append("<br><br>" if blank else "<br>")
        self.out.append(content)
        top.text = True

    def html(self) -> str:
        self.end_paragraph()
        self.close_lists(0)
        return "".join(self.out)


def render_html(text: str) -> str:
    """text, Markdown as a model answers or a note is written, as HTML for the page (module docstring)."""
    writer = _Writer()
    lines = iter(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    blank = False
    for line in lines:
        if not line.strip(_SPACE):
            writer.end_paragraph()
            blank = True
            continue
        if fence := _FENCE.fullmatch(line):
            column, size, body = _columns(fence.group("indent")), len(fence.group("fence")), []
            for inner in lines:  # the same iterator: the text goes on after the closing fence
                end = _FENCE_END.fullmatch(inner)
                if end and len(end.group("fence")) >= size:
                    break
                body.append(inner[min(column, len(inner) - len(inner.lstrip(" "))):])
            writer.fence(column, body)
        elif heading := _HEADING.fullmatch(line):
            level = min(len(heading.group("hashes")) + HEADING_SHIFT, LOWEST_HEADING)
            content = _without_closing_hashes((heading.group("text") or "").strip(_SPACE))
            writer.heading(level, _inline(content))
        elif item := _ITEM.fullmatch(line):
            tag = "ul" if item.group("marker") in ("-", "*") else "ol"
            content = (item.group("text") or "").strip(_SPACE)
            writer.item(_columns(item.group("indent")), tag, _inline(content))
        else:
            stripped = line.strip(_SPACE)
            writer.line(_columns(line[:len(line) - len(line.lstrip(_SPACE))]), _inline(stripped), blank)
        blank = False
    return writer.html()
