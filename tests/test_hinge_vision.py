"""Vision locator for Hinge's like/pass buttons — template-matching the glyphs.

The like-heart and pass-X are located at runtime by matching their glyph (not a fixed
coord), because the "Start sending likes" banner + per-profile photo heights shift the
heart and the X's white disc merges into a prompt's white background. These tests synthesize
frames by pasting the SHIPPED glyph templates onto a textured canvas at known spots, so we
verify match location, topmost selection, side filtering, and graceful degradation without
committing real screenshots.

hinge_heart.png vs hinge_like_button.png: the former is Hinge's OUTLINE heart (the "Which do
we have in common" list rows' control), the latter is the real per-card like button (a white
heart in a filled black circle). Only hinge_like_button.png is wired to the "like" role (see
HINGE_SPEC.templates) — hinge_heart.png stays on disk, unused, purely as a reference for what
it actually is. The tests above that reference "hinge_heart.png" directly are exercising
_match_glyph's generic mechanics against a real shipped glyph, not the "like" role itself, so
they are unaffected by that role's re-target and still use it as their fixture on purpose.
"""
import cv2
import numpy as np

from operation_love.drivers import hinge


def _canvas():
    rng = np.random.default_rng(0)
    return rng.integers(60, 200, size=(2400, 1080), dtype=np.uint8)   # textured (avoid flat-NCC artifacts)


def _paste(canvas, templ, cx, cy):
    th, tw = templ.shape
    canvas[cy - th // 2: cy - th // 2 + th, cx - tw // 2: cx - tw // 2 + tw] = templ
    return canvas


def _png(gray):
    ok, buf = cv2.imencode(".png", gray)
    assert ok
    return buf.tobytes()


def test_glyph_templates_load():
    assert hinge._load_template("hinge_heart.png") is not None
    assert hinge._load_template("hinge_pass_x.png") is not None


def test_locate_like_finds_topmost_right_heart():
    heart = hinge._load_template("hinge_heart.png")
    c = _canvas()
    _paste(c, heart, 937, 1600)
    _paste(c, heart, 937, 800)
    pts = hinge._match_glyph(_png(c), heart, side="right")
    assert pts, "heart not found"
    assert pts[0] == (937, 800)                       # topmost first (the first photo after scroll-to-top)
    assert all(x > 1080 * 0.55 for x, _ in pts)       # all on the right


def test_locate_pass_finds_left_x():
    x = hinge._load_template("hinge_pass_x.png")
    c = _canvas()
    _paste(c, x, 125, 2035)
    pts = hinge._match_glyph(_png(c), x, side="left")
    assert pts and pts[0] == (125, 2035)


def test_side_filter_excludes_wrong_side():
    heart = hinge._load_template("hinge_heart.png")
    c = _canvas()
    _paste(c, heart, 140, 1600)                       # heart on the LEFT
    assert hinge._match_glyph(_png(c), heart, side="right") == []


def test_locate_send_like_anyway_modal_text():
    t = hinge._load_template("hinge_send_like_anyway.png")
    assert t is not None
    c = _canvas()
    _paste(c, t, 420, 2197)
    pts = hinge._match_glyph(_png(c), t, side="any")   # centered modal text, no side filter
    assert pts and pts[0] == (420, 2197)


def test_match_glyph_graceful_without_template():
    assert hinge._match_glyph(b"x", None, side="right") == []


def test_match_glyph_graceful_on_undecodable():
    heart = hinge._load_template("hinge_heart.png")
    assert hinge._match_glyph(b"not-a-png", heart, side="right") == []


# --- the "like" role's template: hinge_like_button.png, not hinge_heart.png ------------------
#
# ops/calibration/scroll_20260811T211209Z/ (115 real Hinge frames, 1080x2400, gitignored, real
# profiles -- never committed) established that hinge_heart.png (the "Which do we have in
# common" widget's OUTLINE heart) was the wrong glyph for "like": it matched those outline-heart
# rows at correlation 1.000 while the real per-card button (a white heart in a filled black
# circle) never matched at all (peaked ~0.544, off any actual heart). hinge_like_button.png is
# the fix -- 88x88 grayscale, a pixel-exact crop of the filled-circle button's own bounding
# square (inscribed so the crop is 100% button chrome: no photo, no face, no text), cropped from
# ops/calibration/scroll_20260811T211209Z/00050.png. See the templates dict comment on
# HINGE_SPEC and _LIKE_MATCH_THRESHOLD (both in hinge.py) for the full measured distribution.

def test_like_role_resolves_to_the_button_template_not_the_outline_heart():
    assert hinge.HINGE_SPEC.templates["like"] == "hinge_like_button.png"
    assert hinge.HINGE_SPEC.templates["like"] != "hinge_heart.png"
    assert hinge._load_template(hinge.HINGE_SPEC.templates["like"]) is not None


def test_like_button_template_found_at_its_calibrated_threshold():
    """POSITIVE: hinge_like_button.png, pasted at a plausible right-side card position, is
    found by _match_glyph at _LIKE_MATCH_THRESHOLD (0.75) -- the threshold actually wired into
    _locate_button/_locate_target_heart for role "like", not _match_glyph's generic 0.6
    default (see _LIKE_MATCH_THRESHOLD's comment: 0.6 lets a fixed, unrelated bottom-nav icon
    through as a false "like" hit on every single real frame)."""
    button = hinge._load_template(hinge.HINGE_SPEC.templates["like"])
    c = _canvas()
    _paste(c, button, 938, 660)
    pts = hinge._match_glyph(_png(c), button, side="right", threshold=hinge._LIKE_MATCH_THRESHOLD)
    assert pts and pts[0] == (938, 660)


def test_like_button_template_does_not_fire_on_the_old_outline_heart():
    """NEGATIVE — the regression that actually matters, more than the positive case above.
    Paste the OLD "like" template (hinge_heart.png, Hinge's OUTLINE heart from the "Which do
    we have in common" widget -- a real control, just never the like button) where a real like
    button would be, and confirm the NEW template does not call it a match, at the calibrated
    threshold OR anywhere close to it. Measured live, the new template scores ~ -0.09 against
    real outline-heart rows -- not a near miss. If this test ever starts passing at a
    meaningfully positive correlation, the template has drifted back towards the exact
    confusion (common-row heart vs. real like button) this whole asset swap exists to fix."""
    old_outline_heart = hinge._load_template("hinge_heart.png")
    new_button = hinge._load_template(hinge.HINGE_SPEC.templates["like"])
    c = _canvas()
    _paste(c, old_outline_heart, 938, 660)
    # The calibrated production threshold, not 0.0: an all-noise canvas has SOME location
    # with positive correlation by chance (that's what the noise texture is for -- see
    # _canvas's own comment), so threshold=0.0 would fail this test on noise alone and prove
    # nothing about the outline heart specifically. hinge._LIKE_MATCH_THRESHOLD is exactly
    # what production uses for role "like" (_locate_button/_locate_target_heart), so this is
    # the real question: does a real deployment ever call this a "like" hit here.
    assert hinge._match_glyph(_png(c), new_button, side="right",
                              threshold=hinge._LIKE_MATCH_THRESHOLD) == []


# --- y_band: structurally excluding Hinge's bottom-nav false positives ------------------------
#
# MEASURED 2026-08-11 against ops/calibration/scroll_20260811T211209Z/ (115 real frames,
# gitignored): Hinge's bottom nav bar carries TWO persistent false positives for the "like"
# template, both at y=2258 -- outside HINGE_SPEC.content_band ((0.125, 0.875) -> y 300..2100 on
# a 2400-row frame), while every real card heart (y 570..1890) sits inside it. Previously each
# false positive was excluded only by a thin margin: the "Matches" tab icon by ~0.10 of
# threshold headroom (0.6527 vs 0.75), the "Likes" tab heart by 54px/5% of screen width against
# the side="right" cutoff (a near-perfect ~1.0 correlation match otherwise). y_band excludes
# both by geometry instead, regardless of what threshold or side is passed.

def test_like_button_found_inside_content_band():
    """POSITIVE: a like-button glyph pasted at a plausible card position (inside
    HINGE_SPEC.content_band) is still found once y_band is passed."""
    button = hinge._load_template(hinge.HINGE_SPEC.templates["like"])
    c = _canvas()
    _paste(c, button, 938, 1398)                      # well inside content_band (300..2100)
    pts = hinge._match_glyph(_png(c), button, side="right",
                             threshold=hinge._LIKE_MATCH_THRESHOLD,
                             y_band=hinge.HINGE_SPEC.content_band)
    assert pts and pts[0] == (938, 1398)


def test_like_button_in_nav_bar_band_excluded_by_y_band():
    """NEGATIVE -- the regression guard for the nav-bar false positive. Paste the SAME
    like-button glyph at y=2258 (the real, measured constant y of both of Hinge's bottom-nav
    false positives -- the "Matches" tab icon and the "Likes" tab heart, see _LIKE_MATCH_THRESHOLD's
    comment), which is below content_band's lower edge (0.875 * 2400 = 2100). At the SAME
    calibrated production threshold role "like" actually uses (_LIKE_MATCH_THRESHOLD, not a
    lowered one -- this is a self-paste so a real match scores ~1.0, well clear of it either
    way), a y_band-restricted match must not find it: geometry, not score, is doing the
    exclusion here. Without y_band this same glyph/position IS found (asserted below too),
    which is exactly the fragility (threshold/side-margin only) this parameter exists to
    remove."""
    button = hinge._load_template(hinge.HINGE_SPEC.templates["like"])
    c = _canvas()
    _paste(c, button, 938, 2258)                      # real measured nav-bar false-positive y
    assert hinge._match_glyph(_png(c), button, side="right",
                              threshold=hinge._LIKE_MATCH_THRESHOLD,
                              y_band=hinge.HINGE_SPEC.content_band) == []
    # Without y_band, the same glyph at the same spot IS a hit -- confirms the exclusion above
    # is y_band's doing, not some other unrelated reason (e.g. a bad paste).
    assert hinge._match_glyph(_png(c), button, side="right",
                              threshold=hinge._LIKE_MATCH_THRESHOLD) != []
