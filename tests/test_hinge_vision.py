"""Vision locator for Hinge's like/pass buttons — template-matching the glyphs.

The like-heart and pass-X are located at runtime by matching their glyph (not a fixed
coord), because the "Start sending likes" banner + per-profile photo heights shift the
heart and the X's white disc merges into a prompt's white background. These tests synthesize
frames by pasting the SHIPPED glyph templates onto a textured canvas at known spots, so we
verify match location, topmost selection, side filtering, and graceful degradation without
committing real screenshots.
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
