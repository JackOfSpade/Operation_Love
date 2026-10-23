"""Rendering of navigation_refusal.passive_resample telemetry. Offline."""

import json

from operation_love import bugreport


def _row(passive=None, include=True):
    telemetry = {
        "code": "shift_unmeasured",
        "planned": {"step_px": 600},
        "achieved": {"status": "no_quorum"},
        "return_outcome": "verified",
    }
    if include:
        telemetry["passive_resample"] = passive
    return json.dumps(
        {
            "action": "still_photo_dwell_walk_candidate",
            "outcome": "navigation_refused",
            "heart_ordinal": 2,
            "navigation_refusal": telemetry,
        }
    )


_MAIN = (
    "- page heart 2: dwell-navigation refusal `shift_unmeasured`; step 600px; "
    "estimator `no_quorum`; anchor return `verified`"
)


def _render(row):
    return bugreport._dwell_navigation_refusal_summary_md([row])


def test_legacy_row_without_metadata_unchanged():
    assert _render(_row(include=False)) == _MAIN


def test_attempted_failure():
    out = _render(
        _row(
            {
                "schema_version": 1,
                "attempts": 1,
                "frames_captured": 2,
                "outcome": "still_unmeasured",
                "recovered": False,
            }
        )
    )
    assert out == (
        _MAIN + "\n  - passive screenshot recovery: outcome `still_unmeasured`; "
        "1 attempt; 2 extra screenshots; estimator not recovered"
    )


def test_ineligible_zero_attempts():
    out = _render(
        _row(
            {
                "schema_version": 1,
                "attempts": 0,
                "frames_captured": 0,
                "outcome": "ineligible",
                "recovered": False,
            }
        )
    )
    assert out.splitlines()[1] == (
        "  - passive screenshot recovery: outcome `ineligible`; "
        "not attempted (0 attempts); 0 extra screenshots; estimator not recovered"
    )


def test_malformed_metadata_does_not_crash():
    line = "  - passive screenshot recovery: metadata malformed; not interpreted"
    assert _render(_row("garbage")).splitlines() == [_MAIN, line]
    assert _render(_row(None)).splitlines() == [_MAIN, line]
    assert _render(_row({"schema_version": 2})).splitlines()[1] == (
        "  - passive screenshot recovery: unrecognised metadata schema; not interpreted"
    )
    assert _render(_row({"schema_version": True})).splitlines()[1] == (
        "  - passive screenshot recovery: unrecognised metadata schema; not interpreted"
    )
    out = _render(
        _row(
            {
                "schema_version": 1,
                "attempts": True,
                "frames_captured": 7,
                "outcome": 3,
                "recovered": "yes",
                "stationarity": ["not", "a", "dict"],
            }
        )
    )
    assert out.splitlines() == [
        _MAIN,
        "  - passive screenshot recovery: outcome unrecognised; attempts not recorded; "
        "extra screenshots not recorded; recovery not recorded",
    ]


def test_sanitization_rejects_free_text_outcome():
    secret = "token=sk-SECRET `inject`\n# heading data:image/png;base64,AAAA"
    out = _render(
        _row(
            {
                "schema_version": 1,
                "attempts": 1,
                "frames_captured": 1,
                "outcome": secret,
                "recovered": True,
                "image": "data:image/png;base64,AAAA",
            }
        )
    )
    assert "SECRET" not in out
    assert "base64" not in out
    assert out.splitlines()[1] == (
        "  - passive screenshot recovery: outcome unrecognised; 1 attempt; "
        "1 extra screenshot; estimator recovered"
    )
    long_token = "a" * 49
    out = _render(_row({"schema_version": 1, "outcome": long_token}))
    assert long_token not in out
    assert "outcome unrecognised" in out.splitlines()[1]


def test_stationarity_rendered_via_trace_renderer():
    out = _render(
        _row(
            {
                "schema_version": 1,
                "attempts": 1,
                "frames_captured": 2,
                "outcome": "recovered",
                "recovered": True,
                "stationarity": {"forward": {"status": "measured"}},
            }
        )
    )
    lines = out.splitlines()
    assert lines[2] == "    - stationarity check between extra screenshots:"
    assert lines[3].startswith("      - forward estimator `measured`")
