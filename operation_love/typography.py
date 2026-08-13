"""Canonical typography-folding tables -- the single source of truth for "what a real
device keyboard can actually type" (and, tied to that, the owner's no-dash rule: openers
must contain no em dash or hyphen of any kind; it reads as the single biggest AI-written
tell).

Two independent call sites fold through this module so they cannot silently drift apart
again: ``opener.opener._sanitize`` (the primary enforcement point -- cleans the LLM's own
output before it is recorded as the sent opener) and ``drivers.adb.Adb.text`` (the device-
input boundary -- folds whatever it is asked to type into its ASCII equivalent, and raises
loudly on anything that still can't be typed after folding, rather than silently dropping
it; see ``fold_to_ascii``/``undeliverable_chars`` below and adb.py's docstring for why a
silent drop is unacceptable: the owner's hard rule is best humanized interaction or FAIL
LOUDLY, never a silent degrade). Before this module existed the two call sites had already
disagreed once on which codepoints counted as a dash and how to tidy up the punctuation
spacing left behind; folding through one shared table/function is what keeps that from
happening again.

Lives at the operation_love top level, not inside either ``opener/`` or ``drivers/``, so
neither subpackage has to import the other -- the same pattern as ``human.py``.
"""
from __future__ import annotations

import re
import unicodedata

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


# Curly quote -> straight quote. adb shell `input text` and every downstream consumer
# (BigQuery, the hub) only need to compare/display straight ASCII quotes; a curly one is
# just as much a "reads like it came from a word processor, not a person" tell as a dash.
QUOTE_FOLD: dict[str, str] = {
    "’": "'", "‘": "'", "‚": "'", "‛": "'",   # curly single: ' ' , '
    "“": '"', "”": '"', "„": '"', "‟": '"',   # curly double: " " „ ‟
}
# nbsp, narrow no-break space, thin space -> a normal ASCII space. `adb shell input text`
# only recognizes an ordinary space (encoded as %s -- see adb.py's _escape_input_text); any
# of these exotic spaces would otherwise pass the printable-ASCII check below (they are NOT
# in 32..126) and get reported as undeliverable for no reason a human would understand.
SPACE_FOLD: dict[str, str] = {" ": " ", " ": " ", " ": " "}

# Latin letters that Unicode's own NFKD decomposition CANNOT reduce to base-letter +
# combining-mark, because they are atomic letters in their own right, not an accented form
# of a plainer one -- NFKD leaves every one of these completely unchanged. fold_to_ascii's
# combining-mark strip (step 2 below) therefore never touches them, so without this table
# they would sail through folding untouched and only get caught, as an opaque "undeliverable
# character", at the Adb.text()/OpenerParseError boundary -- forcing a retry (or a hard
# failure on the device-input safety net) for a perfectly common, easily-substituted
# European surname letter. Spelling out the substitution here instead lets ordinary names
# (Nordic o/oe, German ss, Icelandic thorn/eth, Polish l) come through as readable ASCII on
# the first pass, the same way an accented e already does via NFKD.
LIGATURE_FOLD: dict[str, str] = {
    "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L",
    "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D",
}


def fold_to_ascii(text: str) -> str:
    """The canonical "what the device can actually type" fold -- the single source of truth
    behind both opener.opener._sanitize (the LLM-output sanitizer) and drivers.adb.Adb.text
    (the device-input boundary; see its module docstring). Order matters and mirrors the
    passes a human proofreading typography by hand would make, most-specific first:

    1. Character-by-character table folds -- curly quotes, every dash variant (via the
       canonical DASH_FOLD table), the ellipsis glyph, exotic spaces, and the atomic
       ligature/letter substitutions in LIGATURE_FOLD -- so anything with a fixed, known-good
       ASCII substitute is rewritten before the generic accent-stripping pass below ever
       sees it (NFKD does not touch any of these; see LIGATURE_FOLD's own comment for why).
    2. NFKD decomposition + combining-mark strip: NFKD splits an accented letter into its
       base letter plus one or more combining marks (e.g. e -> e + U+0301 COMBINING ACUTE
       ACCENT), and dropping every codepoint unicodedata.combining() reports as a combining
       mark leaves just the plain ASCII base behind. This is the general-purpose case that
       covers the huge remaining space of accented Latin letters (Sao, cafe, jalapeno, u,
       n, ...) without needing one manual table entry per accent/letter combination.
    3. Newline/CR/tab -> a single space: an opener is one line of comment-box text: a
       control character has no ASCII meaning worth preserving here.
    4. Literal % is left ALONE. `adb shell input text` reserves `%s` as its own space
       escape (drivers.adb._escape_input_text encodes every space as literal `%s`), and an
       earlier version of this function assumed ANY literal `%` would collide with that and
       rewrote it as the word " percent" -- unconditionally, even for the overwhelming
       majority of `%` occurrences that never touch the escape at all. The owner rejected
       that rewrite as unnatural ("50%" must type as "50%", not "50 percent").

       It turns out the blanket rewrite was an overreaction: `%` only collides with the
       space escape when it is immediately followed by a lowercase `s` (Android's
       `Input.java`/`InputShellCommand` `sendText()` unescaper sets an escape flag on `%`
       and only consumes the NEXT character as a space if that character is exactly a
       lowercase `s`; anything else, including uppercase `S`, just clears the flag and both
       characters pass through unharmed). Running fold_to_ascii's own output (nothing else
       in this function touches `%`) through a faithful port of that unescaper, keyed on the
       real drivers.adb._escape_input_text encoding, measured:

           '50%'          -> '50%'          OK
           '50% off'      -> '50% off'      OK
           'up 30% today' -> 'up 30% today' OK
           '50%.'         -> '50%.'         OK
           'a%%b'         -> 'a%%b'         OK
           '100%sure'     -> '100 ure'      COLLISION
           '20%stake'     -> '20 take'      COLLISION
           '%s'           -> ' '            COLLISION

       So the ONLY real collision is `%` directly against a following lowercase `s`, and
       silently rewording the whole opener to dodge that one narrow case is exactly the
       thing this module refuses to do to every other undeliverable character (see this
       docstring's closing paragraph and adb.py's module docstring): fold what has a known-
       good ASCII substitute, and FAIL LOUDLY -- never silently reword -- on what doesn't.
       That narrow case is therefore handled the same way an emoji is: left untouched here,
       and caught downstream as a named, loud rejection by undeliverable_sequences() (below)
       rather than papered over by this function.

       LIVE-VERIFY: the table above is a faithful port of Android's published sendText()
       algorithm exercised against this repo's own _escape_input_text, not a measurement
       taken on the physical Pixel 7a (disconnected as of this writing). Confirm "50%"
       really types as "50%" on-device before leaning on this further.
    5. tidy_punctuation_spacing: cleans up the whitespace/punctuation artifacts steps 1-4
       can leave behind (double spaces, a stray space before punctuation, a comma stranded
       before terminal punctuation, leading/trailing connective punctuation from a boundary
       dash).

    Anything not recognised by any of the above passes through UNTOUCHED -- this function
    never deletes a character it doesn't understand, and (per point 4 above) never rewords a
    character it COULD deliver just because it collides in one narrow, adjacent-character
    case. That is deliberate: silently dropping or rewording is exactly the bug this module
    exists to fix (see adb.py's module docstring and the owner's "best humanized interaction
    or fail loudly" rule). Whatever survives this fold and is still outside printable ASCII
    is for undeliverable_chars (below) to catch and name; the narrow `%`+lowercase-`s`
    collision that survives fully-ASCII is for undeliverable_sequences (below) to catch and
    name instead. Neither function quietly erases or rewrites anything.
    """
    folded_chars: list[str] = []
    for ch in text:
        if ch in QUOTE_FOLD:
            folded_chars.append(QUOTE_FOLD[ch])
        elif ch in DASH_FOLD:
            folded_chars.append(DASH_FOLD[ch])
        elif ch == "…":
            folded_chars.append("...")
        elif ch in SPACE_FOLD:
            folded_chars.append(" ")
        elif ch in LIGATURE_FOLD:
            folded_chars.append(LIGATURE_FOLD[ch])
        else:
            folded_chars.append(ch)
    step1 = "".join(folded_chars)

    decomposed = unicodedata.normalize("NFKD", step1)
    no_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))

    no_control_ws = no_marks.translate({ord("\n"): " ", ord("\r"): " ", ord("\t"): " "})

    # No step 4 here: see docstring point 4 -- a literal '%' is left alone. The narrow real
    # collision (a literal '%' directly against a following lowercase 's') is handled by
    # undeliverable_sequences() as a loud, named rejection, not by rewording here.
    return tidy_punctuation_spacing(no_control_ws)


def undeliverable_chars(text: str) -> list[str]:
    """The sorted, de-duplicated set of characters still outside printable ASCII
    (U+0020..U+007E) after folding `text` through fold_to_ascii -- i.e. exactly what a real
    device keyboard still could not type after every known substitution has been applied.
    An empty list means `text` is fully typeable as-is.

    This is the boundary check both call sites use to turn "still not ASCII after folding"
    into a loud, named failure instead of what adb.py used to do silently: drop the
    character and keep going. See drivers.adb.Adb.text (raises AdbError) and
    opener.opener.GeminiOpener._parse (raises OpenerParseError, which opener/service.py's
    retry loop turns into a corrective re-ask of the model).
    """
    folded = fold_to_ascii(text)
    return sorted({ch for ch in folded if not (32 <= ord(ch) <= 126)})


def undeliverable_sequences(text: str) -> list[str]:
    """The de-duplicated, in-order list of SUBSTRINGS of `fold_to_ascii(text)` that collide
    with `adb shell input text`'s own space escape once drivers.adb._escape_input_text has
    run -- currently only a literal `%` immediately followed by a lowercase `s`.

    This exists as a SEPARATE function from undeliverable_chars, not a case added to it,
    because undeliverable_chars is inherently character-by-character (it asks "is this one
    codepoint typeable"), and this collision cannot be expressed that way: `%` alone is
    perfectly typeable, `s` alone is perfectly typeable, but the two-character SEQUENCE `%s`
    is not, because drivers.adb._escape_input_text encodes every space as the literal two
    characters `%s`, and Android's own `sendText()` unescaper (a faithful port of
    `Input.java`/`InputShellCommand`, exercised in tests/test_adb.py) decodes any `%`
    immediately followed by lowercase `s` back into a space, silently eating both characters.
    Uppercase `%S` is NOT a collision -- the Android comparison is against lowercase 's'
    only -- so it is deliberately not flagged here. Measured round-trip through the real
    escaper (see fold_to_ascii's docstring point 4 for the full table):

        '50%'          -> '50%'          OK (not flagged)
        '100%sure'     -> '100 ure'      COLLISION (flagged: '%s')
        '20%stake'     -> '20 take'      COLLISION (flagged: '%s')
        '%s'           -> ' '            COLLISION (flagged: '%s')

    LIVE-VERIFY: this is a faithful port of Android's published sendText() algorithm run
    against this repo's own escaper, not a measurement taken on the physical Pixel 7a
    (disconnected as of this writing).

    A pure function like undeliverable_chars: never raises, never mutates `text`. Callers
    (drivers.adb.Adb.text, opener.opener.GeminiOpener._parse) turn a non-empty result into a
    loud, named failure -- AdbError / OpenerParseError -- exactly like undeliverable_chars,
    rather than silently rewording the text (see fold_to_ascii's docstring point 4 for why
    that rewording is no longer done here).
    """
    folded = fold_to_ascii(text)
    found: list[str] = []
    for i in range(len(folded) - 1):
        if folded[i] == "%" and folded[i + 1] == "s" and "%s" not in found:
            found.append("%s")
    return found


def describe_char(ch: str) -> str:
    """Render one character the way an operator (or the model, reading a retry hint) can
    actually identify it: the character itself, its codepoint escape (\\uXXXX for the BMP,
    \\UXXXXXXXX above it -- the same width split Python's own string literals use), and its
    Unicode name (falling back to "UNKNOWN" for a codepoint with no assigned name -- e.g. a
    stray control character) rather than an unreadable raw glyph or an opaque ordinal.

    Shared by drivers.adb.Adb.text (names an undeliverable character in the AdbError it
    raises) and opener.opener.GeminiOpener._parse (names one in the OpenerParseError that
    becomes a model-facing retry hint -- see service.py's retry loop), so the same character
    is described identically at both the device-input boundary and the LLM-output boundary.
    """
    code = ord(ch)
    escape = f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}"
    name = unicodedata.name(ch, "UNKNOWN")
    return f"{ch!r} ({escape} {name})"


def format_duration(seconds: float | None) -> str:
    """"1m37s" / "45s" / "1h02m03s" style compact wall-clock duration, or "unknown duration"
    for the None a malformed/missing timestamp pair produces — never raises, never fabricates
    a number for data that isn't there.

    Lives HERE, in the leaf formatting module, rather than in the module that first needed it.
    It has two callers that sit on opposite sides of the dependency graph: bugreport.py's stall
    summary (offline, after the fact) and hinge.py's stuck-screen watchdog message (live, mid
    run). Homing it in bugreport.py made a DRIVER import the BUG REPORTER, which is backwards —
    the reporter is a diagnostics consumer of the drivers, and it already reaches the other way
    (`from .drivers.touchwatch import ...`). That only avoided a circular import because the
    reporter's own drivers import happens to be function-local today; making it module-level,
    a perfectly ordinary refactor, would have broken the entire drivers package at import time.
    typography.py imports nothing from this project (only `re`/`unicodedata`), so depending on
    it can never close a cycle from any direction."""
    if seconds is None:
        return "unknown duration"
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
