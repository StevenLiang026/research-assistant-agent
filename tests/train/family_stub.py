"""Family stand-ins for the training tests: a LoadedFamily over tests/support/family_fakes' tokenizer, loaded
from a fake model folder, and the load_family hook scripts/train.py's main takes in place of the real loader.

The fake tokenizer reads every character as one token (family_fakes), so the samples of the shared histories
are a few thousand tokens long: a test that wants a sample dropped sets --max-length from a sample's own
"tokens", not from a guess.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from support.family_fakes import FakeFamilyTokenizer, fake_family, model_folder

from research_agent.family.core import LoadedFamily, load_family


def loaded_fake_family(folder: str | Path, style: str = "hybrid", *, name: str | None = None,
                       tokenizer: FakeFamilyTokenizer | None = None, **changes) -> LoadedFamily:
    """load_family over folder (made as a fake model folder when it has no config.json) with a
    FakeFamilyTokenizer of the style, as the family that describes it; changes alter that family."""
    folder = Path(folder)
    if not (folder / "config.json").is_file():
        model_folder(folder)
    tokenizer = tokenizer or FakeFamilyTokenizer(style)
    return load_family(folder, name=name, tokenizer=tokenizer, families=[fake_family(tokenizer, **changes)])


def family_hook(style: str = "hybrid", *, seen: list | None = None,
                **changes) -> Callable[[str, str | None], LoadedFamily]:
    """main(load_family=...)'s hook: (model_dir, name) -> the fake family of that folder; the calls' arguments
    are appended to seen when given."""

    def load(model_dir: str, name: str | None) -> LoadedFamily:
        if seen is not None:
            seen.append((model_dir, name))
        return loaded_fake_family(model_dir, style, name=name, **changes)

    return load
