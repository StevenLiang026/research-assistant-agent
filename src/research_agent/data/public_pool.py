"""The public records that the M2 draw left unused, read from the local Hugging Face cache only.

scripts/fetch_public_data.py downloaded each source's data files at one revision into the default Hugging
Face cache, converted every record with research_agent.data.public and drew the plan's count into
data/raw/public/<name>.jsonl; data/public/manifest.json records, per source, the Hub id ("repo"), the
"revision", the files with their configs ("per_file") and the drawn file's "sha256". This module reads the
same files back from the cache at that revision (huggingface_hub.try_to_load_from_cache, which never opens a
connection: a file that is not cached is a FileNotFoundError, never a download), converts them with the same
converters and the same source ids (the row index, "config/row" for a file with a config) and leaves out the
source ids of the drawn file. The drawn file must still have the manifest's SHA-256, or the exclusion would
be measured against another draw (ValueError).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

from .public import CONVERTERS, convert_many

RAW_DIR = Path("data/raw/public")
MANIFEST = Path("data/public/manifest.json")


def load_manifest(path: str | Path = MANIFEST) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cached_file(repo: str, filename: str, revision: str) -> Path:
    """The path of a dataset file in the local Hugging Face cache at revision; FileNotFoundError when the
    cache does not hold it (nothing is downloaded)."""
    found = try_to_load_from_cache(repo, filename, revision=revision, repo_type="dataset")
    if not isinstance(found, str) or not Path(found).is_file():
        raise FileNotFoundError(
            f"{repo}/{filename}@{revision[:8]} is not in the Hugging Face cache; "
            "scripts/fetch_public_data.py downloads it"
        )
    return Path(found)


def read_records(path: Path) -> list[dict]:
    """The records of a data file: a JSON list, or the rows of a CSV with a header (read as the fetch did)."""
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"{path.name}: expected a JSON list, got {type(records).__name__}")
    return records


def source_ids(entry: dict, file_index: int, rows: int) -> list[str]:
    """The source ids of a file's rows as the fetch gave them: "config/row" with a config, else the row."""
    config = entry["per_file"][file_index].get("config")
    return [f"{config}/{i}" if config else str(i) for i in range(rows)]


def drawn_source_ids(path: Path, sha256: str | None) -> set[str]:
    """The source ids of a drawn file; ValueError when its SHA-256 is not the manifest's (sha256 None: not
    checked)."""
    data = path.read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{path}: SHA-256 differs from data/public/manifest.json; the drawn file changed")
    return {json.loads(line)["source_id"] for line in data.decode("utf-8").splitlines() if line.strip()}


def load_unused(name: str, entry: dict, raw_dir: str | Path = RAW_DIR) -> tuple[list[dict], dict]:
    """The converted samples of source name that data/raw/public/<name>.jsonl does not hold, in file order,
    and an account: rows, converted, dropped by reason, drawn, unused."""
    drawn = drawn_source_ids(Path(raw_dir) / f"{name}.jsonl", entry.get("sha256"))
    converter = CONVERTERS[name]
    samples: list[dict] = []
    rows = converted = 0
    dropped: dict[str, int] = {}
    for index, item in enumerate(entry["per_file"]):
        records = read_records(cached_file(entry["repo"], item["file"], entry["revision"]))
        ids = source_ids(entry, index, len(records))
        got, reasons = convert_many(records, converter, ids=ids)
        rows, converted = rows + len(records), converted + len(got)
        for reason, n in reasons.items():
            dropped[reason] = dropped.get(reason, 0) + n
        samples += got
    unused = [s for s in samples if s["source_id"] not in drawn]
    return unused, {
        "rows": rows,
        "converted": converted,
        "dropped": dict(sorted(dropped.items())),
        "drawn": len(drawn),
        "unused": len(unused),
    }
