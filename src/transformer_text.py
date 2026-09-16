"""Text normalization shared by Transformer training and inference."""

from __future__ import annotations

import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
WHITESPACE_RE = re.compile(r"\s+")


def light_clean(text: str) -> str:
    """Remove HTML/URLs and normalize whitespace while preserving text signal."""
    value = str(text)
    value = HTML_RE.sub(" ", value)
    value = URL_RE.sub(" ", value)
    return WHITESPACE_RE.sub(" ", value).strip()
