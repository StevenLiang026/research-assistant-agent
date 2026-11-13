"""research_agent.live.render offline: the HTML the page shows for an answer or a note. Every injection vector
comes out as text the reader sees, with only the tags and attributes of ALLOWED, every element closed in order
and every link exactly https://arxiv.org/abs/<canonical id> (read back with html.parser, as a browser reads
it, never by searching the output); the Markdown subset, a heading's closing run of "#" dropped as its
pattern drops it but in time linear in the heading's length; arXiv ids and arxiv.org URLs as exactly one link
each, never inside code or a longer number; and a note as notes.render_entry writes it."""

from __future__ import annotations

import random
import re
import time
from html.parser import HTMLParser

import pytest

from research_agent.eval.metrics import _ARXIV_ID, arxiv_ids
from research_agent.live.render import ALLOWED, render_html
from research_agent.normalize import normalize_arxiv_id
from research_agent.notes import render_entry

ABS = "https://arxiv.org/abs/"
LINK = {"target": "_blank", "rel": "noopener noreferrer"}  # besides href, on every link


def a(arxiv_id: str, shown: str | None = None) -> str:
    """The link render_html writes for arxiv_id, showing shown (escaped already; the id by default)."""
    return (f'<a href="{ABS}{arxiv_id}" target="_blank" rel="noopener noreferrer">'
            f"{arxiv_id if shown is None else shown}</a>")


class Parsed(HTMLParser):
    """html as html.parser reads it: the text as shown (character references resolved), the links as (href,
    text), the number of tags, and every problem: a tag or an attribute outside ALLOWED, a repeated attribute,
    a link that is not an arXiv abstract page with LINK, a link inside a link, an end tag out of order, an
    element left open, a self-closing tag, a comment, a declaration or a processing instruction."""

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.problems: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.tags = 0
        self._text: list[str] = []
        self._open: list[str] = []
        self._link: list[str] | None = None
        self.feed(html)
        self.close()
        if self._open:
            self.problems.append(f"left open: {self._open}")
        self.text = "".join(self._text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags += 1
        allowed = ALLOWED.get(tag)
        if allowed is None:
            self.problems.append(f"tag <{tag}>")
            allowed = frozenset()
        names = [name for name, _ in attrs]
        self.problems += [f"attribute {name} on <{tag}>" for name in names if name not in allowed]
        if len(set(names)) != len(names):
            self.problems.append(f"a repeated attribute on <{tag}>: {names}")
        if tag == "a":
            values = dict(attrs)
            href = values.pop("href", None) or ""
            arxiv_id = href.removeprefix(ABS)
            if not href.startswith(ABS) or normalize_arxiv_id(arxiv_id) != arxiv_id or values != LINK:
                self.problems.append(f"link {attrs}")
            if self._link is not None:
                self.problems.append("a link inside a link")
            self._link = [href, ""]
        if tag != "br":
            self._open.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.problems.append(f"self-closing <{tag}/>")

    def handle_endtag(self, tag: str) -> None:
        self.tags += 1
        if not self._open or self._open[-1] != tag:
            self.problems.append(f"</{tag}> closes {self._open[-1:] or 'nothing'}")
            return
        self._open.pop()
        if tag == "a" and self._link is not None:
            self.links.append((self._link[0], self._link[1]))
            self._link = None

    def handle_data(self, data: str) -> None:
        self._text.append(data)
        if self._link is not None:
            self._link[1] += data

    def handle_comment(self, data: str) -> None:
        self.problems.append(f"comment {data!r}")

    def handle_decl(self, decl: str) -> None:
        self.problems.append(f"declaration {decl!r}")

    def unknown_decl(self, data: str) -> None:
        self.problems.append(f"declaration {data!r}")

    def handle_pi(self, data: str) -> None:
        self.problems.append(f"processing instruction {data!r}")


def checked(html: str) -> Parsed:
    """html parsed without a problem, every "<" and ">" in it one of its tags' (its text holds none)."""
    parsed = Parsed(html)
    assert parsed.problems == [], html
    assert html.count("<") == html.count(">") == parsed.tags, html
    return parsed


def test_allowed_is_the_page_allowlist():
    plain = ("p", "br", "strong", "code", "pre", "ul", "ol", "li", "h3", "h4", "h5", "h6")
    assert ALLOWED == {**{tag: set() for tag in plain}, "a": {"href", "target", "rel"}}


# ---- escaping ---------------------------------------------------------------------------------------------

# (vector, what the reader must see of it as text; None when the renderer drops it, as a fence's info string)
VECTORS = [
    pytest.param("<script>alert(1)</script>", "<script>alert(1)</script>", id="script"),
    pytest.param("<SCRIPT SRC=//x.example/x.js></SCRIPT>", "<SCRIPT SRC=//x.example/x.js></SCRIPT>",
                 id="script-src"),
    pytest.param('<img src=x onerror="alert(1)">', '<img src=x onerror="alert(1)">', id="img-onerror"),
    pytest.param("<svg/onload=alert(1)>", "<svg/onload=alert(1)>", id="svg-onload"),
    pytest.param("<iframe src=//x.example></iframe>", "<iframe src=//x.example></iframe>", id="iframe"),
    pytest.param("<style>*{display:none}</style>", "<style>*{display:none}</style>", id="style"),
    pytest.param("[2501.12948](javascript:alert(1))", "](javascript:alert(1))", id="javascript-md-link"),
    pytest.param('<a href="javascript:alert(1)">x</a>', '<a href="javascript:alert(1)">x</a>',
                 id="javascript-anchor"),
    pytest.param("javascript:alert(document.cookie)", "javascript:alert(document.cookie)",
                 id="javascript-text"),
    pytest.param("![x](https://x.example/x.png)", "![x](https://x.example/x.png)", id="markdown-image"),
    pytest.param("<https://x.example/>", "<https://x.example/>", id="autolink"),
    pytest.param('2501.12948" onmouseover="alert(1)', '" onmouseover="alert(1)', id="quote-after-an-id"),
    pytest.param('https://arxiv.org/abs/2501.12948"><script>alert(1)</script>', '"><script>alert(1)</script>',
                 id="quote-after-a-url"),
    pytest.param("https://arxiv.org/pdf/2501.12948'onfocus='alert(1)", "'onfocus='alert(1)",
                 id="single-quote-after-a-url"),
    pytest.param('`" onclick="alert(1)`', '" onclick="alert(1)', id="quote-in-code"),
    pytest.param('**"><img src=x onerror=alert(1)>**', '"><img src=x onerror=alert(1)>', id="quote-in-bold"),
    pytest.param('<a href="https://arxiv.org/abs/2501.12948" target="_blank">2501.12948</a>', '<a href="',
                 id="the-renderers-own-markup"),
    pytest.param("```\n</code></pre><script>alert(1)</script>\n```", "</code></pre><script>alert(1)</script>",
                 id="fence-break-out"),
    pytest.param('```"><script>alert(1)</script>\nx\n```', None, id="fence-info-string"),
    pytest.param("```\n<img src=x onerror=alert(1)>", "<img src=x onerror=alert(1)>", id="unclosed-fence"),
    pytest.param("`</code><script>alert(1)</script>`", "</code><script>alert(1)</script>",
                 id="code-span-break-out"),
    pytest.param("&lt;script&gt;alert(1)&lt;/script&gt;", "&lt;script&gt;alert(1)&lt;/script&gt;",
                 id="named-entities"),
    pytest.param("&#60;img src=x onerror=alert(1)&#62; &#x3C;b&#x3E;",
                 "&#60;img src=x onerror=alert(1)&#62; &#x3C;b&#x3E;", id="numeric-references"),
    pytest.param("&amp;lt;b&amp;gt;", "&amp;lt;b&amp;gt;", id="escaped-entity"),
    pytest.param("<!-- x --><!DOCTYPE html><![CDATA[<b>]]><?xml x?>",
                 "<!-- x --><!DOCTYPE html><![CDATA[<b>]]><?xml x?>", id="comment-and-declarations"),
    pytest.param("a\x00b\x0bc<\x00script>", "<\x00script>", id="control-characters"),
]
# Where a vector may stand: alone, in a heading, an item, a nested item, bold, code, a fence, beside links...
CONTEXTS = ("{}", "# {}", "###### {}", "- {}", "- a\n  - {}", "1. {}", "**{}**", "x `{}` y", "```\n{}\n```",
            "- a\n  ```\n  {}\n  ```", "2501.12948 {} https://arxiv.org/abs/2501.12948", "a\n{}\nb",
            "- a\n\n  {}")


@pytest.mark.parametrize("vector, shown", VECTORS)
def test_every_vector_comes_out_as_text_with_only_the_allowed_tags(vector, shown):
    for context in CONTEXTS:
        checked(render_html(context.format(vector)))
    parsed = checked(render_html(vector))
    if shown is not None:
        assert shown in parsed.text


def test_text_is_escaped_quotes_included_and_entities_are_shown_as_typed():
    assert render_html("""a < b > c & "d" 'e'""") == ("<p>a &lt; b &gt; c &amp; &quot;d&quot; "
                                                      "&#x27;e&#x27;</p>")
    assert render_html("&lt;b&gt; &#60;") == "<p>&amp;lt;b&amp;gt; &amp;#60;</p>"


# ---- the Markdown subset ----------------------------------------------------------------------------------


@pytest.mark.parametrize("markdown, html", [
    pytest.param("# One", "<h3>One</h3>", id="h1-is-h3"),
    pytest.param("## Two", "<h4>Two</h4>", id="h2-is-h4"),
    pytest.param("### Three", "<h5>Three</h5>", id="h3-is-h5"),
    pytest.param("#### Four", "<h6>Four</h6>", id="h4-is-h6"),
    pytest.param("##### Five", "<h6>Five</h6>", id="h5-is-h6"),
    pytest.param("###### Six", "<h6>Six</h6>", id="h6-is-h6"),
    pytest.param("####### Seven", "<p>####### Seven</p>", id="seven-hashes"),
    pytest.param("#Tag", "<p>#Tag</p>", id="no-space-after-the-hashes"),
    pytest.param("   # Indented", "<h3>Indented</h3>", id="three-spaces-before-a-heading"),
    pytest.param("## Closed ##", "<h4>Closed</h4>", id="closing-hashes"),
    pytest.param("## C#", "<h4>C#</h4>", id="a-hash-in-a-word"),
    pytest.param("#", "<h3></h3>", id="empty-heading"),
    pytest.param("# T\ntext", "<h3>T</h3><p>text</p>", id="heading-then-paragraph"),
    pytest.param("one\ntwo\n\nthree", "<p>one<br>two</p><p>three</p>", id="paragraphs"),
    pytest.param("  text  \n\tmore", "<p>text<br>more</p>", id="lines-stripped"),
    pytest.param("a\r\nb\rc", "<p>a<br>b<br>c</p>", id="line-ends"),
    pytest.param("", "", id="empty"),
    pytest.param("\n \n\t\n", "", id="blank-lines"),
    pytest.param("**bold** and `co<de>`", "<p><strong>bold</strong> and <code>co&lt;de&gt;</code></p>",
                 id="bold-and-code"),
    pytest.param("**a**b**c**", "<p><strong>a</strong>b<strong>c</strong></p>", id="two-bolds"),
    pytest.param("***x***", "<p>*<strong>x</strong>*</p>", id="three-stars"),
    pytest.param("*one* and ** two ** and **", "<p>*one* and ** two ** and **</p>", id="not-bold"),
    pytest.param("``a`b``", "<p><code>a`b</code></p>", id="double-backtick-code"),
    pytest.param("`` `x` ``", "<p><code>`x`</code></p>", id="code-with-backticks"),
    pytest.param("`**x**` and `a", "<p><code>**x**</code> and `a</p>", id="code-is-not-bold"),
    pytest.param("**`x` 2**", "<p><strong><code>x</code> 2</strong></p>", id="code-in-bold"),
    pytest.param("- a\n- b", "<ul><li>a</li><li>b</li></ul>", id="dash-list"),
    pytest.param("* a\n* b", "<ul><li>a</li><li>b</li></ul>", id="star-list"),
    pytest.param("1. a\n2. b\n10. c", "<ol><li>a</li><li>b</li><li>c</li></ol>", id="ordered-list"),
    pytest.param("- a\n  - b\n    - c\n- d",
                 "<ul><li>a<ul><li>b<ul><li>c</li></ul></li></ul></li><li>d</li></ul>", id="nested-list"),
    pytest.param("- a\n    - b\n  - c", "<ul><li>a<ul><li>b</li></ul><ul><li>c</li></ul></li></ul>",
                 id="deeper-then-between"),
    pytest.param("- a\n - b\n- c", "<ul><li>a<ul><li>b</li></ul></li><li>c</li></ul>", id="one-column"),
    pytest.param("- a\n\t- b", "<ul><li>a<ul><li>b</li></ul></li></ul>", id="tab-nests"),
    pytest.param("1. a\n   - b\n2. c", "<ol><li>a<ul><li>b</li></ul></li><li>c</li></ol>",
                 id="list-in-ordered-list"),
    pytest.param("- a\n1. b", "<ul><li>a</li></ul><ol><li>b</li></ol>", id="kind-switch"),
    pytest.param("-\n- ", "<ul><li></li><li></li></ul>", id="empty-items"),
    pytest.param("-a and 1.5 and 2026. no", "<p>-a and 1.5 and 2026. no</p>", id="no-markers"),
    pytest.param("Papers:\n- a\n\nAfter.", "<p>Papers:</p><ul><li>a</li></ul><p>After.</p>",
                 id="list-between-paragraphs"),
    pytest.param("1. a\n\n2. b", "<ol><li>a</li><li>b</li></ol>", id="blank-line-between-items"),
    pytest.param("1. **A**\n   about A\n2. B", "<ol><li><strong>A</strong><br>about A</li><li>B</li></ol>",
                 id="continuation"),
    pytest.param("1. A\n\n   about A", "<ol><li>A<br><br>about A</li></ol>", id="continuation-after-blank"),
    pytest.param("- a\n  - b\n  more a", "<ul><li>a<ul><li>b</li></ul>more a</li></ul>",
                 id="continuation-after-nested-list"),
    pytest.param("- a\nnot a", "<ul><li>a</li></ul><p>not a</p>", id="no-lazy-continuation"),
    pytest.param("- a\n## H", "<ul><li>a</li></ul><h4>H</h4>", id="heading-ends-list"),
    pytest.param("```py\nx = '<y>'\n\n  z\n```", "<pre><code>x = &#x27;&lt;y&gt;&#x27;\n\n  z</code></pre>",
                 id="fence"),
    pytest.param("text\n```\ncode\n```\nafter", "<p>text</p><pre><code>code</code></pre><p>after</p>",
                 id="fence-between-paragraphs"),
    pytest.param("- item\n  ```\n  code\n  ```\n- next",
                 "<ul><li>item<pre><code>code</code></pre></li><li>next</li></ul>", id="fence-in-item"),
    pytest.param("- item\n```\ncode\n```", "<ul><li>item</li></ul><pre><code>code</code></pre>",
                 id="fence-ends-list"),
    pytest.param("```\nunclosed\n# no heading\n- no item",
                 "<pre><code>unclosed\n# no heading\n- no item</code></pre>", id="unclosed-fence"),
    pytest.param("````\n```\n````", "<pre><code>```</code></pre>", id="longer-fence"),
    pytest.param("```a`b```", "<p><code>a`b</code></p>", id="backtick-in-info-is-no-fence"),
])
def test_the_markdown_subset(markdown, html):
    assert render_html(markdown) == html
    checked(html)


# A heading's closing run of "#" as a pattern: the reference for the renderer's reading, which is linear
CLOSING_HASHES = re.compile(r"(?:^|[ \t]+)#+\Z")


def test_a_closing_run_of_hashes_goes_after_a_space_or_a_tab_or_when_it_is_the_whole_heading():
    draw = random.Random(70)
    for _ in range(3000):
        text = "".join(draw.choice("# \ta") for _ in range(draw.randint(0, 12)))
        expected = CLOSING_HASHES.sub("", text.strip(" \t"))
        assert render_html(f"# {text}") == f"<h3>{expected}</h3>", repr(text)


@pytest.mark.parametrize("gap", [" ", "\t"], ids=["spaces", "tabs"])
def test_a_heading_with_a_long_run_of_white_space_renders_in_time_linear_in_its_length(gap):
    run = gap * 40_000  # with backtracking over every start in the run, this took over five seconds
    for line, html in ((f"# a{run}#x", f"<h3>a{run}#x</h3>"), (f"# a{run}##", "<h3>a</h3>")):
        started = time.perf_counter()
        assert render_html(line) == html
        assert time.perf_counter() - started < 1.0


# ---- links ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text, links", [
    pytest.param("2501.12948", [("2501.12948", "2501.12948")], id="bare-id"),
    pytest.param("见 2501.12948v2。", [("2501.12948", "2501.12948v2")], id="version-dropped-from-href"),
    pytest.param("arXiv:2501.12948", [("2501.12948", "2501.12948")], id="arxiv-prefix"),
    pytest.param("(2501.12948, 2502.00001)", [("2501.12948", "2501.12948"), ("2502.00001", "2502.00001")],
                 id="two-ids"),
    pytest.param("2501.12948 2501.12948", [("2501.12948", "2501.12948")] * 2, id="each-occurrence"),
    pytest.param("0704.0001 1412.3555", [("0704.0001", "0704.0001"), ("1412.3555", "1412.3555")],
                 id="four-digit-ids"),
    pytest.param("https://arxiv.org/abs/2501.12948v3",
                 [("2501.12948", "https://arxiv.org/abs/2501.12948v3")], id="abs-url"),
    pytest.param("<https://arxiv.org/pdf/2501.12948v2.pdf>",
                 [("2501.12948", "https://arxiv.org/pdf/2501.12948v2.pdf")], id="pdf-url"),
    pytest.param("http://www.arxiv.org/abs/2501.12948.", [("2501.12948", "http://www.arxiv.org/abs/2501.12948")],
                 id="http-www-url"),
    pytest.param("arxiv.org/pdf/2501.12948", [("2501.12948", "arxiv.org/pdf/2501.12948")], id="no-scheme"),
    pytest.param("HTTPS://ArXiv.org/ABS/2501.12948", [("2501.12948", "HTTPS://ArXiv.org/ABS/2501.12948")],
                 id="upper-case-url"),
    pytest.param("[the paper](https://arxiv.org/abs/2501.12948)",
                 [("2501.12948", "https://arxiv.org/abs/2501.12948")], id="markdown-link-is-text"),
    pytest.param("https://x.example/abs/2501.12948", [("2501.12948", "2501.12948")], id="foreign-url"),
    pytest.param("https://arxiv.org/abs/**2501.12948**", [("2501.12948", "2501.12948")], id="split-url"),
    pytest.param("## **B** 2501.12948", [("2501.12948", "2501.12948")], id="in-a-heading"),
    pytest.param("**see 2501.12948** and `x` - 2502.00001",
                 [("2501.12948", "2501.12948"), ("2502.00001", "2502.00001")], id="in-bold-and-beside-code"),
])
def test_an_arxiv_id_or_url_is_exactly_one_link_to_its_abstract_page(text, links):
    assert checked(render_html(text)).links == [(ABS + arxiv_id, shown) for arxiv_id, shown in links]


def test_a_link_is_written_as_the_page_expects():
    assert render_html("2501.12948v2 https://arxiv.org/pdf/2502.00001.pdf") == (
        f"<p>{a('2501.12948', '2501.12948v2')} {a('2502.00001', 'https://arxiv.org/pdf/2502.00001.pdf')}</p>")


@pytest.mark.parametrize("text", ["`2501.12948`", "``https://arxiv.org/abs/2501.12948``",
                                  "```\n2501.12948\nhttps://arxiv.org/abs/2502.00001\n```",
                                  "- a\n  ```\n  2501.12948\n  ```"])
def test_code_is_never_linked(text):
    parsed = checked(render_html(text))
    assert parsed.links == [] and "2501.12948" in parsed.text


@pytest.mark.parametrize("text", [
    pytest.param("12345.678901", id="longer-number"),
    pytest.param("2026.11", id="year-and-month"),
    pytest.param("1.2501.12948", id="after-a-dot"),
    pytest.param("22501.12948", id="digit-before"),
    pytest.param("2501.129481", id="digit-after"),
    pytest.param("2501.12948123", id="digits-after"),
    pytest.param("2513.12345", id="month-13"),
    pytest.param("2500.12345", id="month-00"),
    pytest.param("1412.12345", id="five-digits-before-1501"),
    pytest.param("2501.1234", id="four-digits-from-1501"),
    pytest.param("0612.1234", id="before-0704"),
    pytest.param("https://arxiv.org/abs/2513.12345", id="url-of-no-id"),
    pytest.param("https://arxiv.org/abs/12345.678901", id="url-of-a-longer-number"),
])
def test_no_link_inside_a_longer_number_or_for_an_id_arxiv_does_not_issue(text):
    parsed = checked(render_html(text))
    assert parsed.links == [] and parsed.text == text


# What the random texts are made of: ids valid and not, URL prefixes and suffixes, bold, list and heading
# markers, line breaks, digits and dots (no backtick: code is never linked, so it is left out here)
PIECES = ("2501.12948", "2501.12948v2", "0704.0001", "2513.12345", "12345.678901", "2026.11", "1412.12345",
          "https://arxiv.org/abs/", "http://www.arxiv.org/pdf/", "arxiv.org/abs/", ".pdf", "v3", "**", "*",
          " ", "  ", "\n", "\n\n", "- ", "  - ", "1. ", "# ", "## ", "#", ".", "1", "x", "(", ")", ";", "\t",
          "论文")


def test_outside_code_every_id_the_evaluation_reads_is_one_link_in_text_order():
    draw = random.Random(10)
    for _ in range(600):
        text = "".join(draw.choice(PIECES) for _ in range(draw.randint(1, 40)))
        expected = [ABS + i for i in (normalize_arxiv_id(m.group(1)) for m in _ARXIV_ID.finditer(text)) if i]
        assert [href for href, _ in checked(render_html(text)).links] == expected, repr(text)


def test_the_links_of_an_answer_are_the_ids_the_evaluation_counts_as_cited():
    answer = ("找到 3 篇:\n"
              "1. **Adaptive Draft Length** (2501.12948v2) https://arxiv.org/pdf/2502.00001v1.pdf\n"
              "2. arXiv:0704.0001; not 2513.12345, 12345.678901 or 2026.11\n\n"
              "See [the paper](https://arxiv.org/abs/2603.01234) and 2501.12948.")
    links = checked(render_html(answer)).links
    assert len(links) == 5
    assert ({href.removeprefix(ABS) for href, _ in links} == arxiv_ids(answer)
            == {"2501.12948", "2502.00001", "0704.0001", "2603.01234"})


# ---- a note -----------------------------------------------------------------------------------------------


def test_a_draft_note_keeps_its_heading_source_link_and_nested_summary():
    entry = {"today": "2026-11-10", "title": "Adaptive <Draft> Length", "link": f"{ABS}2611.00001",
             "published": "2026-11-03", "type": "论文", "problem": "Drafts are **too short**.",
             "method": "Adapt the length to the acceptance rate (as in 2611.00002).",
             "new_terms": ["acceptance rate", "draft"]}
    html = render_html(render_entry(entry))
    assert html == (
        "<h5>2026-11-10 Adaptive &lt;Draft&gt; Length</h5>"
        f"<ul><li>来源:{a('2611.00001', f'{ABS}2611.00001')};2026-11-03;论文</li>"
        "<li>与本周所学的关系:</li>"
        "<li>Summary (about 300 words)<ul><li>Problem: Drafts are <strong>too short</strong>.</li>"
        f"<li>Method: Adapt the length to the acceptance rate (as in {a('2611.00002')}).</li>"
        "<li>Evidence:</li><li>Limitations:</li></ul></li>"
        "<li>新术语:acceptance rate, draft</li><li>待解术语:</li></ul>")
    checked(html)


def test_a_scaffold_note_keeps_its_evidence_sentences_nested():
    entry = {"today": "2026-11-10", "title": "T", "link": f"{ABS}2611.00001", "published": "2026-11-03",
             "type": "论文", "evidence_sentences": ["We adapt the <draft> length.", "It is 2x faster."]}
    html = render_html(render_entry(entry))
    assert html == (
        "<h5>2026-11-10 T</h5>"
        f"<ul><li>来源:{a('2611.00001', f'{ABS}2611.00001')};2026-11-03;论文</li>"
        "<li>与本周所学的关系:</li>"
        "<li>Summary (about 300 words)<ul><li>Problem:</li><li>Method:</li><li>Evidence:</li>"
        "<li>Limitations:</li></ul></li>"
        "<li>证据句(原文)<ul><li>We adapt the &lt;draft&gt; length.</li><li>It is 2x faster.</li></ul></li>"
        "<li>新术语:</li><li>待解术语:</li></ul>")
    checked(html)
