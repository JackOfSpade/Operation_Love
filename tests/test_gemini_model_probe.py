"""tools/gemini_model_probe.py -- repeatable live audit of usable Gemini models.

No network, ever: every test injects a fake transport (matching
operation_love.opener.opener.GeminiTransport's ``(url, payload, headers, timeout, *,
method="POST") -> (code, body)`` shape). See tests/test_gemini_opener.py for the same pattern
against GeminiOpener itself.
"""
from __future__ import annotations

import json

import pytest
import yaml

from tools import gemini_model_probe as m

_SECRET = "sk-super-secret-abc123"


# ---------------------------------------------------------------------------------------
# local _helpers (no conftest.py in this repo)
# ---------------------------------------------------------------------------------------

def _page(ids_with_methods, next_token=None):
    body = {"models": [{"name": f"models/{model_id}", "supportedGenerationMethods": methods}
                       for model_id, methods in ids_with_methods]}
    if next_token:
        body["nextPageToken"] = next_token
    return (200, body)


def _success(opener_text="Nice view, where is this from?"):
    return (200, {
        "candidates": [{"content": {"parts": [{"text": json.dumps({
            "item_index": 1, "referenced": "thing", "angle": "guessing",
            "item_description": "a photo", "opener": opener_text,
        })}]}}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
    })


def _http_error(code, status, message, *, details=None):
    error = {"code": code, "status": status, "message": message}
    if details is not None:
        error["details"] = details
    return (code, {"error": error})


def _thinking_400(field="thinking_level"):
    return _http_error(
        400, "INVALID_ARGUMENT",
        f'Invalid JSON payload received. Unknown name "{field}" at '
        "'generation_config.thinking_config': Cannot find field.")


def _quota_429(*, quota_id=None, quota_metric=None, quota_value=None):
    violation = {}
    if quota_id is not None:
        violation["quotaId"] = quota_id
    if quota_metric is not None:
        violation["quotaMetric"] = quota_metric
    if quota_value is not None:
        violation["quotaValue"] = quota_value
    details = ([{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [violation]}] if violation else None)
    return _http_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded for this model.",
                       details=details)


def _not_found():
    return _http_error(404, "NOT_FOUND",
                       "models/gemini-x is not found for API version v1beta")


def _busy():
    return _http_error(503, "UNAVAILABLE", "The model is overloaded. Please try again later.")


class _ScriptedTransport:
    """Replays a fixed list of (code, body) responses in call order; records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, payload, headers, timeout, *, method="POST"):
        self.calls.append(
            {"method": method, "url": url, "payload": payload, "headers": headers,
             "timeout": timeout})
        if not self.responses:
            raise AssertionError("no more scripted transport responses")
        return self.responses.pop(0)


def _probe_kwargs(transport, **overrides):
    profile, item_request = m.build_synthetic_profile_and_items()
    kwargs = dict(api_key="test-key", transport=transport, timeout=5, max_tokens=200,
                 style="be nice", profile=profile, item_request=item_request)
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------------------
# ENUMERATE -- pagination, generateContent-only filtering
# ---------------------------------------------------------------------------------------

def test_enumerate_models_follows_nextpagetoken_pagination():
    page1 = _page([("gemini-a", ["generateContent"])], next_token="tok-2")
    page2 = _page([("gemini-b", ["generateContent", "countTokens"])])
    transport = _ScriptedTransport([page1, page2])

    catalog = m.enumerate_models(api_key="k", transport=transport, timeout=5)

    assert catalog == {"gemini-a": ["generateContent"],
                       "gemini-b": ["generateContent", "countTokens"]}
    assert len(transport.calls) == 2
    assert transport.calls[0]["method"] == "GET"
    assert transport.calls[0]["payload"] is None
    assert "pageToken=tok-2" in transport.calls[1]["url"]


def test_enumerate_models_percent_encodes_an_opaque_page_token():
    """The fixture that could not reach the branch it named: "tok-2" encodes to itself.

    Google's page tokens are opaque base64-ish strings. A raw '+' is decoded back as a space and
    a raw '&' or '#' truncates the query, so an unencoded token turns page 2 into a 400 that
    aborts the audit and leaves candidate models invisible with no error.
    """
    token = "a+b/c=&d#e"
    transport = _ScriptedTransport([_page([("gemini-a", ["generateContent"])], next_token=token),
                                    _page([("gemini-b", ["generateContent"])])])

    catalog = m.enumerate_models(api_key="k", transport=transport, timeout=5)

    assert set(catalog) == {"gemini-a", "gemini-b"}
    assert transport.calls[1]["url"].endswith("?pageToken=a%2Bb%2Fc%3D%26d%23e")


def test_enumerate_models_refuses_a_server_that_repeats_a_page_token():
    """Bounded like preflight: a repeated token is an infinite loop, not a catalog."""
    transport = _ScriptedTransport(
        [_page([("gemini-a", ["generateContent"])], next_token="tok") for _ in range(4)])
    with pytest.raises(RuntimeError, match="repeated a pagination token"):
        m.enumerate_models(api_key="k", transport=transport, timeout=5)


def test_enumerate_models_refuses_a_catalog_longer_than_the_page_ceiling(monkeypatch):
    monkeypatch.setattr(m, "_MAX_PREFLIGHT_PAGES", 3)
    transport = _ScriptedTransport(
        [_page([(f"gemini-{page}", ["generateContent"])], next_token=f"tok-{page}")
         for page in range(8)])
    with pytest.raises(RuntimeError, match="exceeded 3 pages"):
        m.enumerate_models(api_key="k", transport=transport, timeout=5)


def test_enumerate_models_raises_cleanly_on_http_error():
    transport = _ScriptedTransport([_http_error(500, "INTERNAL", "server error")])
    with pytest.raises(RuntimeError, match="ListModels failed"):
        m.enumerate_models(api_key="k", transport=transport, timeout=5)


def test_generate_content_ids_filters_to_generatecontent_support():
    catalog = {
        "gemini-a": ["generateContent"],
        "embed-only": ["embedContent"],
        "gemini-b": ["generateContent", "countTokens"],
    }
    assert m.generate_content_ids(catalog) == ["gemini-a", "gemini-b"]


# ---------------------------------------------------------------------------------------
# CLASSIFY -- exclusion substring rules and the three-way bucket split
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", [
    "imagen-3.0-generate-002",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
    "gemini-live-2.5-flash-preview",
    "gemini-2.5-flash-native-audio-preview-09-2025",
    "gemini-robotics-er-1.5-preview",
    "lyria-realtime-exp",
    "text-embedding-004",
    "gemini-embedding-001",
    "gemini-2.5-computer-use-preview",
    "deep-research-preview",
    # Both of these are listed by the REAL ListModels WITH "generateContent" support, and both
    # slip past every substring rule that keys off the technical id, so each needs its own rule.
    # Measured live 2026-08-13: without them the tool proposed spending up to 6 billed requests
    # apiece discovering what their published identity already says.
    "antigravity-preview-05-2026",   # managed agent; no structured-output support at all
    "nano-banana-pro-preview",       # marketing alias for the gemini-3-pro-image family
])
def test_excluded_kinds_are_excluded_with_a_reason(model_id):
    reason = m.excluded_reason(model_id)
    assert reason is not None and reason


@pytest.mark.parametrize("model_id", [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
])
def test_chat_ids_are_not_excluded(model_id):
    assert m.excluded_reason(model_id) is None


def test_classify_models_buckets_configured_candidates_and_excluded():
    ids = ["gemini-3.6-flash", "gemini-new-candidate", "imagen-3.0-generate-002",
          "text-embedding-004"]
    classification = m.classify_models(ids, configured=["gemini-3.6-flash"])

    assert classification.configured == ("gemini-3.6-flash",)
    assert classification.candidates == ("gemini-new-candidate",)
    assert set(classification.excluded) == {"imagen-3.0-generate-002", "text-embedding-004"}


def test_classify_models_configured_wins_over_an_exclusion_match():
    # An owner-configured model is reported as configured even if it happens to match a
    # substring rule -- classify_models never second-guesses an existing config.yaml.
    classification = m.classify_models(["weird-tts-model"], configured=["weird-tts-model"])
    assert classification.configured == ("weird-tts-model",)
    assert classification.excluded == {}


# ---------------------------------------------------------------------------------------
# Thinking-variant escalation -- advances ONLY on a thinking-related 400
# ---------------------------------------------------------------------------------------

def test_probe_model_advances_through_thinking_400s_then_succeeds():
    transport = _ScriptedTransport([_thinking_400(), _thinking_400(), _success()])
    result = m.probe_model("gemini-x", **_probe_kwargs(transport))

    assert result.verdict == m.VERDICT_USABLE
    assert result.thinking_label == "thinkingLevel=medium"   # 3rd variant in the order
    assert result.thinking_config == {"thinkingLevel": "medium"}
    assert len(transport.calls) == 3


def test_probe_model_stops_immediately_on_200_without_trying_further_variants():
    transport = _ScriptedTransport([_success("Great trail, is that Patagonia?")])
    result = m.probe_model("gemini-x", **_probe_kwargs(transport))

    assert result.verdict == m.VERDICT_USABLE
    assert result.thinking_label == "thinkingLevel=minimal"   # 1st variant, never escalated
    assert result.opener_text == "Great trail, is that Patagonia?"
    assert len(transport.calls) == 1


def test_probe_model_stops_immediately_on_404_without_trying_other_variants():
    transport = _ScriptedTransport([_not_found()])
    result = m.probe_model("gemini-x", **_probe_kwargs(transport))

    assert result.verdict == m.VERDICT_RETIRED_404
    assert len(transport.calls) == 1


def test_probe_model_stops_immediately_on_429_without_trying_other_variants():
    transport = _ScriptedTransport([_quota_429(quota_id="Q", quota_metric="M", quota_value="20")])
    result = m.probe_model("gemini-x", **_probe_kwargs(transport))

    assert result.verdict == m.VERDICT_QUOTA_429
    assert len(transport.calls) == 1


def test_probe_model_stops_immediately_on_5xx_without_trying_other_variants():
    transport = _ScriptedTransport([_busy()])
    result = m.probe_model("gemini-x", **_probe_kwargs(transport))

    assert result.verdict == m.VERDICT_BUSY_5XX
    assert len(transport.calls) == 1


def test_probe_model_stops_immediately_on_a_non_thinking_400():
    transport = _ScriptedTransport(
        [_http_error(400, "INVALID_ARGUMENT", "Request payload size exceeds the 20MB limit")])
    result = m.probe_model("gemini-x", **_probe_kwargs(transport))

    assert result.verdict == m.VERDICT_BAD_REQUEST_400
    assert len(transport.calls) == 1


def test_probe_model_exhausting_every_thinking_variant_stays_bad_request():
    transport = _ScriptedTransport([_thinking_400()] * len(m._THINKING_VARIANTS))
    result = m.probe_model("gemini-x", **_probe_kwargs(transport))

    assert result.verdict == m.VERDICT_BAD_REQUEST_400
    assert len(transport.calls) == len(m._THINKING_VARIANTS)


def test_probe_model_transport_failure_stops_immediately():
    class _RaisingTransport:
        def __init__(self):
            self.call_count = 0

        def __call__(self, *args, **kwargs):
            self.call_count += 1
            raise OSError("connection reset by peer")

    transport = _RaisingTransport()
    result = m.probe_model("gemini-x", **_probe_kwargs(transport))

    assert result.verdict == m.VERDICT_TRANSPORT
    assert transport.call_count == 1


def test_probe_model_unusable_response_on_unparseable_success_body():
    transport = _ScriptedTransport([
        (200, {"candidates": [{"content": {"parts": [{"text": "not json at all"}]}}],
              "usageMetadata": {}}),
    ])
    result = m.probe_model("gemini-x", **_probe_kwargs(transport))

    assert result.verdict == m.VERDICT_UNUSABLE_RESPONSE
    assert len(transport.calls) == 1


# ---------------------------------------------------------------------------------------
# is_thinking_related_400 / quota_detail
# ---------------------------------------------------------------------------------------

def test_is_thinking_related_400_true_for_a_thinking_field_message():
    assert m.is_thinking_related_400(
        'Unknown name "thinking_level" at \'generation_config\': Cannot find field.')


def test_is_thinking_related_400_false_for_an_unrelated_message():
    assert not m.is_thinking_related_400("Request payload size exceeds the 20MB limit")


def test_quota_detail_parses_a_quotafailure_violation():
    code, body = _quota_429(quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                            quota_metric="generativelanguage.googleapis.com/foo",
                            quota_value="20")
    assert m.quota_detail(body) == {
        "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        "quotaMetric": "generativelanguage.googleapis.com/foo",
        "quotaValue": "20",
    }


def test_quota_detail_returns_none_when_no_structured_detail_present():
    assert m.quota_detail({"error": {"code": 429, "message": "quota exceeded"}}) is None
    assert m.quota_detail("not even a mapping") is None


# ---------------------------------------------------------------------------------------
# YAML snippet
# ---------------------------------------------------------------------------------------

def test_build_yaml_snippet_contains_all_three_entries_and_is_valid_yaml():
    snippet = m.build_yaml_snippet([
        ("gemini-x", {"thinkingLevel": "minimal"}),
        ("gemini-y", None),   # the "omitted" variant worked -> {} (server default), per config.py
    ])
    doc = yaml.safe_load(snippet)

    assert set(doc) == {"models", "thinking", "pricing"}
    assert doc["models"] == ["gemini-x", "gemini-y"]
    assert doc["thinking"] == {"gemini-x": {"thinkingLevel": "minimal"}, "gemini-y": {}}
    for model_id in ("gemini-x", "gemini-y"):
        assert set(doc["pricing"][model_id]) == {"input", "output", "cache_read", "cache_write"}


def test_build_yaml_snippet_empty_when_no_usable_models():
    assert m.build_yaml_snippet([]) == ""


# ---------------------------------------------------------------------------------------
# main() -- configured-model skip default / --include-configured, cost gating, --json, redaction
# ---------------------------------------------------------------------------------------

def _config_path(tmp_path, models):
    path = tmp_path / "config.yaml"
    models_yaml = ", ".join(models)
    path.write_text(f"opener:\n  models: [{models_yaml}]\n")
    return path


def test_main_skips_already_configured_models_by_default(tmp_path, capsys):
    config_path = _config_path(tmp_path, ["gemini-configured"])
    transport = _ScriptedTransport([
        _page([("gemini-configured", ["generateContent"]),
              ("gemini-candidate", ["generateContent"])]),
        _success(),   # only the candidate should be probed
    ])

    rc = m.main(["--yes", "--config", str(config_path)], transport=transport,
               env={"GEMINI_API_KEY": "k"})

    assert rc == 0
    assert len(transport.calls) == 2   # 1 GET (ListModels) + 1 POST (candidate only)
    assert transport.calls[1]["url"].endswith("gemini-candidate:generateContent")
    out = capsys.readouterr().out
    # gemini-configured is reported as already configured but never probed (no result line
    # for it, and it never appears as the target of a POST above).
    assert "Already configured (opener.models): ['gemini-configured']" in out
    assert "gemini-configured: USABLE" not in out
    assert "gemini-configured: BAD_REQUEST_400" not in out


def test_main_include_configured_flag_reprobes_configured_models(tmp_path):
    config_path = _config_path(tmp_path, ["gemini-configured"])
    transport = _ScriptedTransport([
        _page([("gemini-configured", ["generateContent"])]),
        _success(),
    ])

    rc = m.main(["--yes", "--include-configured", "--config", str(config_path)],
               transport=transport, env={"GEMINI_API_KEY": "k"})

    assert rc == 0
    assert len(transport.calls) == 2
    assert transport.calls[1]["url"].endswith("gemini-configured:generateContent")


def test_main_explicit_models_flag_also_skips_configured_by_default(tmp_path, capsys):
    config_path = _config_path(tmp_path, ["gemini-configured"])
    transport = _ScriptedTransport([
        _page([("gemini-configured", ["generateContent"])]),
        # no POST should be issued: the only --models entry is already configured
    ])

    rc = m.main(["--yes", "--models", "gemini-configured", "--config", str(config_path)],
               transport=transport, env={"GEMINI_API_KEY": "k"})

    assert rc == 0
    assert len(transport.calls) == 1   # ListModels only -- nothing was probed
    out = capsys.readouterr().out
    assert "Skipping already-configured" in out
    assert "gemini-configured" in out


def test_main_requires_confirmation_and_does_not_probe_when_declined(tmp_path):
    missing_config = tmp_path / "missing.yaml"
    transport = _ScriptedTransport([_page([("gemini-candidate", ["generateContent"])])])

    rc = m.main(["--config", str(missing_config)], transport=transport,
               env={"GEMINI_API_KEY": "k"}, confirm=lambda prompt: False)

    assert rc != 0
    assert len(transport.calls) == 1   # only the free ListModels call happened


def test_main_proceeds_when_confirm_accepts(tmp_path):
    missing_config = tmp_path / "missing.yaml"
    transport = _ScriptedTransport([
        _page([("gemini-candidate", ["generateContent"])]),
        _success(),
    ])

    rc = m.main(["--config", str(missing_config)], transport=transport,
               env={"GEMINI_API_KEY": "k"}, confirm=lambda prompt: True)

    assert rc == 0
    assert len(transport.calls) == 2


def test_main_yes_flag_skips_the_confirmation_prompt_entirely(tmp_path):
    missing_config = tmp_path / "missing.yaml"
    transport = _ScriptedTransport([
        _page([("gemini-candidate", ["generateContent"])]),
        _success(),
    ])

    def _boom(prompt):
        raise AssertionError("confirm() must not be called when --yes is given")

    rc = m.main(["--yes", "--config", str(missing_config)], transport=transport,
               env={"GEMINI_API_KEY": "k"}, confirm=_boom)

    assert rc == 0


def test_main_requires_api_key():
    def _boom(*a, **k):
        raise AssertionError("transport must not be called without an API key")

    rc = m.main(["--yes"], transport=_boom, env={})
    assert rc != 0


def test_main_json_mode_emits_only_valid_json_on_stdout(tmp_path, capsys):
    missing_config = tmp_path / "missing.yaml"
    transport = _ScriptedTransport([
        _page([("gemini-candidate", ["generateContent"])]),
        _success("Great trail, is that Patagonia?"),
    ])

    rc = m.main(["--yes", "--json", "--config", str(missing_config)], transport=transport,
               env={"GEMINI_API_KEY": "k"})

    assert rc == 0
    out = capsys.readouterr().out
    doc = json.loads(out)   # must parse cleanly -- no narrative text mixed into stdout
    assert doc["results"][0]["model"] == "gemini-candidate"
    assert doc["results"][0]["verdict"] == "USABLE"
    snippet_doc = yaml.safe_load(doc["yaml_snippet"])
    assert set(snippet_doc) == {"models", "thinking", "pricing"}


def test_main_json_mode_nothing_to_probe_still_emits_valid_json(tmp_path, capsys):
    config_path = _config_path(tmp_path, ["gemini-configured"])
    transport = _ScriptedTransport([_page([("gemini-configured", ["generateContent"])])])

    rc = m.main(["--yes", "--json", "--config", str(config_path)], transport=transport,
               env={"GEMINI_API_KEY": "k"})

    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["results"] == []


# ---------------------------------------------------------------------------------------
# The API key must never reach stdout or stderr, on any path.
# ---------------------------------------------------------------------------------------

def test_api_key_never_appears_in_output_on_listmodels_error(tmp_path, capsys):
    transport = _ScriptedTransport([
        _http_error(400, "INVALID_ARGUMENT", f"API key {_SECRET} is not valid"),
    ])

    rc = m.main(["--yes", "--config", str(tmp_path / "missing.yaml")], transport=transport,
               env={"GEMINI_API_KEY": _SECRET})

    assert rc != 0
    captured = capsys.readouterr()
    assert _SECRET not in captured.out
    assert _SECRET not in captured.err


def test_api_key_never_appears_in_output_on_probe_bad_request(tmp_path, capsys):
    transport = _ScriptedTransport([
        _page([("gemini-candidate", ["generateContent"])]),
        _http_error(400, "INVALID_ARGUMENT", f"malformed request, key={_SECRET}"),
    ])

    rc = m.main(["--yes", "--config", str(tmp_path / "missing.yaml")], transport=transport,
               env={"GEMINI_API_KEY": _SECRET})

    assert rc == 0
    captured = capsys.readouterr()
    assert _SECRET not in captured.out
    assert _SECRET not in captured.err


def test_api_key_never_appears_in_output_on_successful_run(tmp_path, capsys):
    transport = _ScriptedTransport([
        _page([("gemini-candidate", ["generateContent"])]),
        _success(),
    ])

    rc = m.main(["--yes", "--json", "--config", str(tmp_path / "missing.yaml")],
               transport=transport, env={"GEMINI_API_KEY": _SECRET})

    assert rc == 0
    captured = capsys.readouterr()
    assert _SECRET not in captured.out
    assert _SECRET not in captured.err
    # The header sent on the wire legitimately carries the key -- only stdout/stderr are
    # regulated here -- so confirm it really was used, otherwise this test would pass by
    # accident if the key were silently dropped instead of merely not printed.
    assert any(call["headers"].get("X-goog-api-key") == _SECRET for call in transport.calls)


# ---------------------------------------------------------------------------------------
# Synthetic profile -- never real captured data
# ---------------------------------------------------------------------------------------

def test_synthetic_profile_carries_no_real_photos_and_a_placeholder_bio():
    profile, item_request = m.build_synthetic_profile_and_items()

    assert profile.photos == []
    assert "synthetic" in profile.bio.lower()
    assert len(item_request.items) == 1
    assert item_request.items[0].startswith(b"\x89PNG\r\n\x1a\n")   # a real, valid PNG


# ---------------------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------------------

def test_help_does_not_require_network_or_an_api_key(capsys):
    with pytest.raises(SystemExit) as exc_info:
        m.main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--models" in out
    assert "--include-configured" in out
    assert "--json" in out
