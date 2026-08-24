"""Pure, fail-closed detection for Hinge's inline ``Send Like`` composer.

Hinge 9.134.0 renders the selected profile item above an inline comment field
instead of opening the older modal sheet.  Seeing the words ``Send Like`` is
not sufficient permission to touch the screen: that glyph can be copied into
other UI, and the legacy fixed comment/send coordinates land on the selected
item in this layout.  This module therefore requires all three independent
pieces of evidence before returning tap geometry:

* a high-confidence match for the caller-supplied Send Like template;
* a wide, filled CTA surrounding that glyph; and
* a wide, outlined comment input immediately above that CTA.

There is deliberately no driver, ADB, configuration, template loading, or
device I/O here.  Callers supply frame bytes and the already-loaded template,
then must still perform their own profile/item verification before typing.
Failure raises ``ComposerDetectionError`` rather than returning guessed
coordinates.  In particular, callers must not fall back to the old modal
coordinates when this refuses.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    """A half-open pixel rectangle in the decoded frame's coordinate space."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x0 < 0 or self.y0 < 0 or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"invalid rectangle: {self!r}")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x0 + self.x1) // 2, (self.y0 + self.y1) // 2)

    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1


@dataclass(frozen=True)
class ComposerSurface:
    """Vision-located actionable geometry for the measured inline composer only."""

    layout_id: str
    comment_rect: Rect
    send_rect: Rect
    confirm_point: tuple[int, int]


class ComposerDetectionError(RuntimeError):
    """The frame did not affirmatively prove Hinge's inline composer is actionable."""


_LAYOUT_ID = "hinge_inline_v1"
_MIN_FRAME_WIDTH = 720
_MIN_FRAME_HEIGHT = 1_600


def _require_vision():
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001 -- surface as the leaf's explicit failure mode
        raise ComposerDetectionError(
            "inline-composer detection requires opencv-python and numpy "
            f"(extra: operation-love[hinge]); import failed: {exc}") from exc
    return cv2, np


def _background_level(gray, np, confirm_y: int) -> float:
    """Estimate the white composer-page background beside the candidate CTA.

    The keyboard moves the complete composer up by about 20% of the display.
    Sampling side margins around the matched glyph, rather than a fixed input
    rectangle, works in both states and excludes the CTA itself.
    """
    height, width = gray.shape
    pad = max(4, round(width * 0.015))
    y0 = max(0, confirm_y - round(height * 0.12))
    y1 = min(height, confirm_y + round(height * 0.12))
    left = gray[y0:y1, pad:max(pad + 1, round(width * 0.08))]
    right = gray[y0:y1, round(width * 0.92):width - pad]
    samples = [part.reshape(-1) for part in (left, right) if part.size]
    if not samples:
        raise ComposerDetectionError("frame has no side margins from which to validate composer geometry")
    return float(np.median(np.concatenate(samples)))


def _confirm_matches(gray, template, cv2, np, *, threshold: float) -> list[tuple[float, tuple[int, int]]]:
    """Return distinct template peaks in Hinge's possible inline-composer band.

    The band intentionally covers both measured vertical states: untouched
    composer (about .77 display height) and keyboard-focused composer (about
    .58).  The CTA and input validations below, not this coarse envelope,
    establish that a match is actionable.
    """
    if template is None or getattr(template, "ndim", None) != 2:
        raise ComposerDetectionError("no usable grayscale Send Like confirmation template was supplied")
    template = np.asarray(template)
    th, tw = template.shape
    height, width = gray.shape
    if th <= 0 or tw <= 0 or th > height or tw > width:
        raise ComposerDetectionError("confirmation template cannot fit in this frame")
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(result >= threshold)
    if not len(xs):
        raise ComposerDetectionError(
            f"no Send Like confirmation glyph matched at the required threshold {threshold:.2f}")
    candidates: list[tuple[float, tuple[int, int]]] = []
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        point = (x + tw // 2, y + th // 2)
        # The template's literal phrase can occur in selected-item text.  This
        # coarse envelope is deliberately broad enough for either keyboard
        # state; detected surface topology does the affirmative validation.
        if round(width * 0.45) <= point[0] <= round(width * 0.82) and \
                round(height * 0.54) <= point[1] <= round(height * 0.84):
            candidates.append((float(result[y, x]), point))
    if not candidates:
        raise ComposerDetectionError(
            "Send Like glyph matched only outside the inline composer's measured CTA region")

    # A single glyph creates several adjacent correlation hits.  Collapse
    # those first, while retaining independently placed copied labels.
    distinct: list[tuple[float, tuple[int, int]]] = []
    for score, point in sorted(candidates, reverse=True):
        if all(abs(point[0] - seen[1][0]) > tw // 2 or
               abs(point[1] - seen[1][1]) > th // 2 for seen in distinct):
            distinct.append((score, point))
    return distinct


def _dark_mask(gray, np, background: float):
    """Keep composer fill and borders while discarding its white page background."""
    darkness = max(5.0, min(20.0, background * 0.025))
    return (gray < background - darkness).astype(np.uint8)


def _without_thin_bridges(mask, cv2, np):
    """Drop marks too narrow to be either composer control.

    Whenever the owner taps into the comment field to edit the opener, Android
    draws a text-selection handle in the gap between the input and the CTA.
    Measured on the Pixel 7a, that teardrop is about 56px wide and spans the
    whole 32px gap, so under 8-connectivity it welds the two controls into one
    component whose bounds are neither control's -- the CTA validation then
    refuses a composer that is plainly on screen, and the observe loop reports
    the sheet as closed while the human is still typing.  Both real controls
    span most of the display width, so opening with a horizontal element erases
    the handle and leaves their geometry byte-identical.

    The span is forced odd because OpenCV anchors an even kernel half a pixel
    off centre, which translates the result instead of only eroding it.
    """
    span = max(3, round(mask.shape[1] * 0.30)) | 1
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, span), np.uint8))


def _cta_for_confirm(mask, cv2, *, confirm: tuple[int, int]) -> Rect:
    """Find the broad filled CTA component that physically encloses ``confirm``."""
    height, width = mask.shape
    count, _labels, stats, _centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    x, y = confirm
    candidates: list[Rect] = []
    for x0, y0, component_width, component_height, area in stats[1:count]:
        rect = Rect(int(x0), int(y0), int(x0 + component_width), int(y0 + component_height))
        if not rect.contains(x, y):
            continue
        if not (round(width * 0.40) <= rect.width <= round(width * 0.68) and
                round(height * 0.025) <= rect.height <= round(height * 0.075) and
                rect.x0 <= round(width * 0.45) and rect.x1 >= round(width * 0.82)):
            continue
        if area / (rect.width * rect.height) < 0.72:
            continue
        candidates.append(rect)
    if not candidates:
        raise ComposerDetectionError(
            "Send Like text matched but its enclosing lower-screen CTA is not visibly filled")
    if len(candidates) != 1:
        raise ComposerDetectionError("ambiguous filled CTA components surround the Send Like glyph")
    return candidates[0]


def _long_dark_run(row, np, *, minimum: int) -> tuple[int, int] | None:
    """Return the widest contiguous dark run in a row, if it is input-width."""
    changes = np.diff(np.concatenate(([False], row.astype(bool), [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    runs = [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if end - start >= minimum
    ]
    return max(runs, key=lambda run: run[1] - run[0]) if runs else None


def _comment_for_cta(mask, cv2, np, *, send: Rect) -> Rect:
    """Locate the wide outlined input directly above an already-validated CTA."""
    height, width = mask.shape
    minimum_width = round(width * 0.72)
    search_top = max(0, send.y0 - round(height * 0.12))
    search_bottom = max(search_top, send.y0 - round(height * 0.004))
    rows: list[tuple[int, int, int]] = []
    for y in range(search_top, search_bottom):
        run = _long_dark_run(mask[y], np, minimum=minimum_width)
        if run is not None:
            rows.append((y, *run))
    if not rows:
        raise ComposerDetectionError(
            "Send Like text matched but the expected wide outlined comment input was not found")

    # Group adjacent antialiased border rows, then require a top/bottom pair
    # whose height and CTA gap agree with the measured control topology.
    groups: list[list[tuple[int, int, int]]] = []
    for row in rows:
        if not groups or row[0] > groups[-1][-1][0] + 1:
            groups.append([row])
        else:
            groups[-1].append(row)
    candidate_pairs: list[tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]] = []
    for top, bottom in zip(groups, groups[1:], strict=False):
        observed_height = bottom[0][0] - top[0][0]
        cta_gap = send.y0 - bottom[-1][0] - 1
        if (round(height * 0.050) <= observed_height <= round(height * 0.100) and
                round(height * 0.003) <= cta_gap <= round(height * 0.030)):
            candidate_pairs.append((top, bottom))
    if len(candidate_pairs) != 1:
        raise ComposerDetectionError(
            "Send Like text matched but the expected wide outlined comment input was not found")
    top, bottom = candidate_pairs[0]

    # Real rounded rectangles connect their sides, allowing the component's
    # exact outer bounds.  Synthetic/hermetic frames commonly draw only the
    # two border rows, for which their union is the actual rectangle.
    #
    # Label within the input's own row band rather than the whole mask: a
    # text-selection handle drawn below the input welds it to the CTA, and the
    # merged component's bounds are then neither control's.  Nothing outside
    # these rows can belong to the input, so cropping costs no evidence.
    top_y, bottom_y = top[0][0], bottom[-1][0]
    enclosing: list[Rect] = []
    band = mask[top_y:bottom_y + 1]
    count, _labels, stats, _centres = cv2.connectedComponentsWithStats(band, connectivity=8)
    for x0, y0, component_width, component_height, _area in stats[1:count]:
        rect = Rect(int(x0), int(y0 + top_y),
                    int(x0 + component_width), int(y0 + top_y + component_height))
        if (rect.y0 <= top_y and rect.y1 > bottom_y and
                round(width * 0.72) <= rect.width <= round(width * 0.90) and
                round(height * 0.050) <= rect.height <= round(height * 0.105)):
            enclosing.append(rect)
    if len(enclosing) == 1:
        return enclosing[0]
    return Rect(min(row[1] for row in top + bottom), top_y,
                max(row[2] for row in top + bottom), bottom_y + 1)


def locate_inline_composer(frame: bytes, confirm_template, threshold: float = 0.8,
                           *, image=None) -> ComposerSurface:
    """Affirmatively locate Hinge 9.134.0's inline composer or fail closed.

    ``confirm_template`` is the preloaded grayscale image for the literal ``Send Like`` glyph.
    This function does no input.  A returned surface only establishes where the composer controls
    are; the caller must separately establish profile identity and that the selected item is the
    model-selected item before typing.  A detection failure must be treated as *no safe tap*,
    never as permission to use legacy modal coordinates.

    ``image`` lets a caller that already decoded this exact ``frame`` to grayscale (Hinge's
    passive observation path calls this function TWICE on the same frame -- once at the strict
    0.80 threshold, then again at 0.68 only if that failed -- see
    ``HingeDriver._locate_observed_inline_composer``) hand that array straight through instead of
    paying for a second ``cv2.imdecode`` of the SAME bytes (2026-08-23 perf pass). ``None``, the
    default, decodes ``frame`` here exactly as before, so every existing caller is unaffected.
    This does NOT change what either call sees: `_confirm_matches` below still runs
    ``cv2.matchTemplate`` against the full, uncropped array either way, at whatever size that
    array is -- only WHERE the decode happens moves, never the size or content of what gets
    correlated. That distinction matters here specifically: cropping this search image before
    matching is a separate, measured trap (a 0.045 correlation drift against a 0.04 ambiguity
    margin, because cv2 switches between spatial and DFT correlation by input size) that this
    change does not go anywhere near.
    """
    if not isinstance(threshold, (int, float)) or not 0.0 < float(threshold) <= 1.0:
        raise ValueError("threshold must be a finite correlation value in (0, 1]")
    cv2, np = _require_vision()
    gray = image if image is not None else cv2.imdecode(
        np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ComposerDetectionError(f"inline-composer frame did not decode as an image ({len(frame)} bytes)")
    height, width = gray.shape
    if width < _MIN_FRAME_WIDTH or height < _MIN_FRAME_HEIGHT:
        raise ComposerDetectionError(
            f"frame {width}x{height} is too small for the measured Hinge inline-composer geometry")

    matches = _confirm_matches(gray, confirm_template, cv2, np, threshold=float(threshold))
    validated: list[tuple[float, ComposerSurface]] = []
    first_failure: ComposerDetectionError | None = None
    for score, confirm_point in matches:
        try:
            background = _background_level(gray, np, confirm_point[1])
            mask = _dark_mask(gray, np, background)
            # Only the CTA lookup needs the bridge-free view: it is the one step that
            # reads a whole component's bounds, so a selection handle reaching the CTA
            # from the input above corrupts it.  The comment lookup keeps the untouched
            # mask so its measured outer bounds stay exact.
            send = _cta_for_confirm(_without_thin_bridges(mask, cv2, np), cv2, confirm=confirm_point)
            comment = _comment_for_cta(mask, cv2, np, send=send)
        except ComposerDetectionError as exc:
            if first_failure is None:
                first_failure = exc
            continue
        validated.append((score, ComposerSurface(
            layout_id=_LAYOUT_ID,
            comment_rect=comment,
            send_rect=send,
            confirm_point=confirm_point,
        )))
    if not validated:
        if first_failure is None:
            # `_confirm_matches` currently either raises or returns at least one candidate, and
            # every candidate above either validates or records a typed refusal. Keep that
            # invariant fail-closed even under `python -O`, where an `assert` would disappear,
            # and preserve this function's public exception contract if either helper changes.
            raise ComposerDetectionError(
                "inline composer produced no actionable confirmation candidate")
        raise first_failure
    if len(validated) > 1:
        best_score = max(score for score, _surface in validated)
        # Two independently placed literal labels with comparable correlation
        # and complete surrounding geometry are not safely distinguishable.
        if sum(score >= best_score - 0.04 for score, _surface in validated) > 1:
            raise ComposerDetectionError(
                "ambiguous equally plausible Send Like confirmation glyphs in inline composer")
    _score, surface = max(validated, key=lambda candidate: candidate[0])
    return surface
