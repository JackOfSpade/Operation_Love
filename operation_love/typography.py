"""Canonical dash-folding table -- the single source of truth for the owner's no-dash
rule (openers must contain no em dash or hyphen of any kind; it reads as the single
biggest AI-written tell).

Two independent call sites enforce this rule: ``opener.opener._sanitize`` (the primary
enforcement point -- cleans the LLM's own output) and ``drivers.adb._clean_text_for_input``
(the device-input layer's defence-in-depth safety net -- cleans whatever ``Adb.text()`` is
asked to type, in case something upstream ever bypasses the sanitizer). Both fold through
THIS table so they cannot silently drift apart again (they previously disagreed on which
codepoints counted as a dash, and on how to tidy up the punctuation spacing left behind).

Lives at the operation_love top level, not inside either ``opener/`` or ``drivers/``, so
neither subpackage has to import the other -- the same pattern as ``human.py``.
"""
from __future__ import annotations

import re

# Every dash-like codepoint the two call sites have needed to fold, split into two buckets
# by what a real ASCII dash in that position means: a long, sentence-level dash reads as a
# pause and becomes a comma; a short, word-joining hyphen becomes a plain space. Newly
# encountered dash lookalikes are a one-line addition to one of these two strings.
EM_DASH_CHARS = "—–―﹘"      # — em, – en, ― horizontal bar, ﹘ small em dash
HYPHEN_CHARS = "-‐‑‒−﹣－"  # - hyphen-minus, ‐ hyphen, ‑ non-breaking hyphen, ‒ figure
                            # dash, − minus sign, ﹣ small hyphen-minus, － fullwidth hyphen

# str.translate()-ready mapping (codepoint -> replacement string), for callers that fold
# with one .translate() call (opener.py).
DASH_TRANSLATION: dict[int, str] = {ord(c): ", " for c in EM_DASH_CHARS}
DASH_TRANSLATION.update({ord(c): " " for c in HYPHEN_CHARS})

# Plain char -> replacement mapping, for callers that fold character-by-character while
# also handling other typography in the same pass (adb.py).
DASH_FOLD: dict[str, str] = {c: ", " for c in EM_DASH_CHARS}
DASH_FOLD.update({c: " " for c in HYPHEN_CHARS})


def tidy_punctuation_spacing(text: str) -> str:
    """Collapse whitespace and clean up the artifacts a dash -> comma/space fold leaves
    behind: a stray space before punctuation (a dash that had a space on its left, e.g.
    "it - really" -> "it , really" pre-cleanup), runs of connective punctuation collapsed
    from adjacent dashes, a comma stranded right before terminal punctuation, and
    leading/trailing connective punctuation left by a boundary dash. Shared by
    opener._sanitize (LLM text) and adb._clean_text_for_input (device-input safety net) so
    both dash call sites clean up identically instead of drifting.
    """
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)        # no space before punctuation
    t = re.sub(r"([,;:])(\s*[,;:])+", r"\1", t)   # collapse runs created by dash->comma
    t = re.sub(r",\s*([.!?;:])", r"\1", t)        # drop a comma stranded before terminal punctuation
    t = re.sub(r"^[\s,;:]+|[\s,;:]+$", "", t)     # strip leading/trailing connective punct from a boundary dash
    return t
