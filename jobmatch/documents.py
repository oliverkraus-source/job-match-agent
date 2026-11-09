"""Load job ads and CVs, split a CV into searchable lines, and normalise text for comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


def load_text(path: str | Path) -> str:
    """Read a .md / .txt file, or extract the text of a .pdf."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize(text: str) -> str:
    """Lower-case, unify quotes/dashes, drop markdown emphasis and bullets, squash whitespace.

    Used to check whether a quote really appears in the CV, without failing on
    harmless differences like "**Python**" vs "Python" or a curly apostrophe.
    """
    text = text.translate(_QUOTES).lower()
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"(^|\n)\s*(?:[-+•]|\d+\.)\s+", r"\1", text)   # list bullets
    return re.sub(r"\s+", " ", text).strip(" .;,")


@dataclass(frozen=True)
class CVLine:
    section: str
    text: str


def cv_lines(cv_text: str) -> list[CVLine]:
    """Split a CV into non-empty lines, each tagged with the heading it sits under."""
    section = "Profile"
    lines: list[CVLine] = []
    for raw in cv_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = re.match(r"^#{1,6}\s+(.*)", line)
        if heading:
            section = heading.group(1).strip()
            continue
        lines.append(CVLine(section, re.sub(r"^(?:[-*+•]|\d+\.)\s+", "", line)))
    return lines
