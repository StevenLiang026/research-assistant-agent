"""Spike S7: how many recent arXiv papers have an HTML full text, and can we find the main sections?

Politeness: one arXiv API request; HTML pages fetched one at a time with a 15 s gap
(arxiv.org/robots.txt: Crawl-delay 15, Allow /html). Personal research use only.
Output: runs/spikes/s7_arxiv_html.json (gitignored) and a printed summary.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = (
    "https://export.arxiv.org/api/query?search_query=cat:cs.CL+OR+cat:cs.LG"
    "&sortBy=submittedDate&sortOrder=descending&max_results={n}"
)
ATOM = "{http://www.w3.org/2005/Atom}"
UA = "research-assistant-agent-spike/0.1 (personal research use)"
SECTIONS = {
    "introduction": r"introduction",
    "method": r"method|approach|methodology|model|framework",
    "experiments": r"experiment|evaluation|results",
    "conclusion": r"conclusion|discussion|summary",
}
H2 = re.compile(r"<h2[^>]*class=\"[^\"]*ltx_title_section[^\"]*\"[^>]*>(.*?)</h2>", re.S | re.I)
TAG = re.compile(r"<[^>]+>")


def get(url: str, timeout: float = 60.0) -> tuple[int, str, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.geturl(), r.read()
    except urllib.error.HTTPError as e:
        return e.code, url, b""
    except Exception as e:  # network errors are recorded, not retried
        return -1, f"{type(e).__name__}: {e}", b""


def main(n: int = 50) -> int:
    status, _, body = get(API.format(n=n))
    if status != 200:
        print(f"arXiv API request failed: status {status}")
        return 1
    ids = []
    for entry in ET.fromstring(body).iter(f"{ATOM}entry"):
        raw = entry.findtext(f"{ATOM}id", "")
        ids.append(re.sub(r"v\d+$", "", raw.rsplit("/abs/", 1)[-1]))
    results = []
    for i, aid in enumerate(ids):
        time.sleep(15)
        code, final_url, html = get(f"https://arxiv.org/html/{aid}")
        text = html.decode("utf-8", "replace")
        heads = [TAG.sub("", h).strip().lower() for h in H2.findall(text)]
        found = {k: any(re.search(p, h) for h in heads) for k, p in SECTIONS.items()}
        results.append({"id": aid, "status": code, "final_url": final_url, "bytes": len(html),
                        "n_sections": len(heads), "found": found})
        print(f"{i + 1:02d}/{len(ids)} {aid} status={code} bytes={len(html)} sections={len(heads)}", flush=True)
    ok = [r for r in results if r["status"] == 200 and r["n_sections"] > 0]
    summary = {
        "queried": len(ids),
        "html_200": sum(r["status"] == 200 for r in results),
        "html_with_sections": len(ok),
        "section_found_rate_among_ok": {k: round(sum(r["found"][k] for r in ok) / max(len(ok), 1), 3)
                                        for k in SECTIONS},
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = Path("runs/spikes/s7_arxiv_html.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 50))
