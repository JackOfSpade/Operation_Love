"""GeminiOpener's direct REST adapter and in-run capacity fallback.

Every request uses an injected transport: these tests never need an API key, SDK, or
network connection.
"""
import base64
import io
import json
import random
import socket
import threading
import time
import urllib.error

import pytest

from operation_love.opener import opener as opener_module
from operation_love.opener.opener import (
    GeminiAPIError,
    GeminiCapacityExhausted,
    GeminiOpener,
    OpenerAborted,
    OpenerError,
    OpenerParseError,
    _ANCHOR_LABEL,
    _ANCHOR_SYSTEM,
    _SCHEMA,
    _SYSTEM,
)
from operation_love.perception.capture import Profile

# Real free-tier quota id strings (per Google's docs), used to build 429 bodies that carry
# enough structured detail for _classify_quota_exhaustion to tell RPD from RPM.
_PER_DAY_QUOTA_ID = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
_PER_MINUTE_QUOTA_ID = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"


def _quota_exhausted(quota_id=None, quota_metric=None):
    """Build a 429 RESOURCE_EXHAUSTED body, optionally with a QuotaFailure detail entry.

    With no quota_id/quota_metric this mimics a 429 body that carries no parseable quota
    detail at all -- real responses aren't guaranteed to include one.
    """
    error = {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota"}
    if quota_id is not None or quota_metric is not None:
        violation = {}
        if quota_id is not None:
            violation["quotaId"] = quota_id
        if quota_metric is not None:
            violation["quotaMetric"] = quota_metric
        error["details"] = [{
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [violation],
        }]
    return (429, {"error": error})


def _success(opener="That pottery mug has a story. What happened?", *, index=1,
             usage=None):
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps({
            "opener": opener, "referenced": "pottery mug", "referenced_index": index,
        })}]}}],
        "usageMetadata": usage or {
            "promptTokenCount": 11,
            "candidatesTokenCount": 7,
            "thoughtsTokenCount": 2,
            "cachedContentTokenCount": 3,
        },
    }


class _Transport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, url, payload, headers, timeout, *, method="POST"):
        self.calls.append((url, payload, headers, timeout, method))
        return next(self.responses)


def _opener(transport, models=("gemini-primary",), **kwargs):
    return GeminiOpener(models, api_key="test-key", transport=transport, **kwargs)


def test_generate_posts_structured_multimodal_request_and_maps_usage():
    transport = _Transport([(200, _success(opener="Matcha and yoga — noted"))])
    png = b"\x89PNG\r\n\x1a\nfirst"
    jpeg = b"\xff\xd8\xffsecond"
    result = _opener(transport, request_timeout_s=12).generate(
        Profile(photos=[png, jpeg], bio="Weekend potter"), style="be curious")

    assert result.model == "gemini-primary"
    assert result.opener == "Matcha and yoga, noted"
    assert result.referenced == "pottery mug" and result.referenced_index == 1
    assert result.usage.input_tokens == 8       # prompt tokens exclude cache-read subset
    assert result.usage.output_tokens == 9       # candidates + Gemini thinking tokens
    assert result.usage.cache_read_input_tokens == 3

    url, payload, headers, timeout, method = transport.calls[0]
    assert url.endswith("/models/gemini-primary:generateContent")
    assert headers["X-goog-api-key"] == "test-key" and timeout == 12
    assert method == "POST"
    assert payload["systemInstruction"]["parts"][0]["text"] == _SYSTEM
    assert payload["generationConfig"] == {
        "maxOutputTokens": 400,
        "responseMimeType": "application/json",
        "responseJsonSchema": _SCHEMA,
    }
    parts = payload["contents"][0]["parts"]
    assert payload["contents"][0]["role"] == "user"
    assert parts[0]["inlineData"] == {
        "mimeType": "image/png", "data": base64.b64encode(png).decode("ascii"),
    }
    assert parts[1]["inlineData"] == {
        "mimeType": "image/jpeg", "data": base64.b64encode(jpeg).decode("ascii"),
    }
    assert "STYLE GUIDE:\nbe curious" in parts[2]["text"]
    assert "HER PROFILE TEXT:\nWeekend potter" in parts[2]["text"]
    assert "profile in scroll order" in parts[2]["text"]


def test_api_key_comes_from_injected_environment():
    transport = _Transport([(200, _success())])
    GeminiOpener(["gemini-primary"], env={"GEMINI_API_KEY": "env-key"}, transport=transport).generate(
        Profile(), style="s")
    assert transport.calls[0][2]["X-goog-api-key"] == "env-key"


# ---------------------------------------------------------------------------------------
# __init__ guard clauses. Both are only ever exercised indirectly today, via
# supervisor.run()'s pre-flight checks (missing GEMINI_API_KEY, empty opener.models) --
# never as a direct unit test of GeminiOpener itself. Pinned here so a regression in either
# guard fails immediately, in this file, instead of only showing up as a confusing
# supervisor-level symptom two layers away.
# ---------------------------------------------------------------------------------------

def test_empty_model_list_raises_value_error():
    """An empty models list can never produce a usable request -- __init__ must fail fast
    with a clear ValueError instead of constructing a GeminiOpener that would only discover
    it has nothing to call once generate() is actually invoked."""
    with pytest.raises(ValueError, match="at least one configured model"):
        GeminiOpener([], api_key="test-key")


def test_all_blank_model_list_raises_value_error():
    """__init__ filters each model id through `str(model).strip()` before checking for
    emptiness (see GeminiOpener.models), so a config with only blank/whitespace entries
    (a plausible YAML typo, e.g. `models: ["", " "]`) must be rejected exactly like a
    genuinely empty list -- not silently become a cascade of unusable "" model ids that
    404 on every call."""
    with pytest.raises(ValueError, match="at least one configured model"):
        GeminiOpener(["", "   ", ""], api_key="test-key")


def test_missing_api_key_raises_runtime_error_naming_the_env_var():
    """No api_key kwarg and no GEMINI_API_KEY in the (injected, deliberately empty)
    environment must fail fast with a RuntimeError naming the exact env var an operator
    needs to set. Pinned to the literal message: it is a fixed, hardcoded string with no
    interpolation of any secret, so asserting it exactly also proves the failure path can
    never leak key material -- there is nothing but this constant string to leak."""
    with pytest.raises(RuntimeError) as exc_info:
        GeminiOpener(["gemini-primary"], env={})
    assert str(exc_info.value) == "GEMINI_API_KEY is not set"


def _model_calls(transport):
    return [call[0].split("/models/")[1].split(":")[0] for call in transport.calls]


def test_429_per_day_quota_blacklists_model_for_the_run(capsys):
    """RPD resets only at midnight Pacific, so a per-day 429 must permanently retire that
    model for the rest of the run: the second profile should skip straight past it."""
    day_exhausted = _quota_exhausted(quota_id=_PER_DAY_QUOTA_ID)
    transport = _Transport([day_exhausted, (200, _success()), (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    first = opener.generate(Profile(bio="first"), style="s")
    second = opener.generate(Profile(bio="second"), style="s")

    assert first.model == second.model == "gemini-second"
    assert _model_calls(transport) == ["gemini-first", "gemini-second", "gemini-second"]
    output = capsys.readouterr().out
    assert "per-day" in output and "blacklisting" in output
    assert "test-key" not in output


def test_429_per_minute_quota_does_not_blacklist_and_is_retried_next_profile(capsys):
    """A per-minute 429 is transient (free-tier RPM can be as low as 5), so it must NOT
    permanently retire the model: the second profile should try the preferred model first."""
    minute_exhausted = _quota_exhausted(quota_metric="generativelanguage.googleapis.com/"
                                                     "generate_content_requests_per_minute")
    transport = _Transport([minute_exhausted, (200, _success()), (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    first = opener.generate(Profile(bio="first"), style="s")
    second = opener.generate(Profile(bio="second"), style="s")

    assert first.model == "gemini-second"
    assert second.model == "gemini-first"   # preferred model retried first on next profile
    assert _model_calls(transport) == ["gemini-first", "gemini-second", "gemini-first"]
    output = capsys.readouterr().out
    assert "per-minute" in output and "NOT blacklisting" in output


def test_429_with_no_parseable_details_does_not_blacklist(capsys):
    """A 429 body with no QuotaFailure detail at all (real responses aren't guaranteed to
    include one) must default to the safer non-blacklisting treatment, same as per-minute."""
    unclassified = _quota_exhausted()
    transport = _Transport([unclassified, (200, _success()), (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    first = opener.generate(Profile(bio="first"), style="s")
    second = opener.generate(Profile(bio="second"), style="s")

    assert first.model == "gemini-second"
    assert second.model == "gemini-first"
    output = capsys.readouterr().out
    assert "unclassified" in output and "NOT blacklisting" in output


@pytest.mark.parametrize("exhausted", [
    _quota_exhausted(quota_id=_PER_DAY_QUOTA_ID),
    _quota_exhausted(quota_metric="generate_content_requests_per_minute"),
    _quota_exhausted(),
], ids=["day", "minute", "unknown"])
def test_all_429_resource_exhausted_models_raise_distinct_exception(exhausted):
    """Whatever the classification, every configured model 429ing within ONE generate()
    call must still raise GeminiCapacityExhausted -- this is what stops the automation."""
    transport = _Transport([exhausted, exhausted])
    with pytest.raises(GeminiCapacityExhausted):
        _opener(transport, models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    assert len(transport.calls) == 2


def test_all_per_day_exhaustion_says_to_wait_for_the_midnight_pacific_reset():
    """The GeminiCapacityExhausted message becomes the run's stop reason in the hub, so when
    every model really is out of DAILY quota it must say so and name the reset time -- that
    is the one case where the operator genuinely cannot just restart."""
    day = _quota_exhausted(quota_id=_PER_DAY_QUOTA_ID)
    with pytest.raises(GeminiCapacityExhausted) as exc_info:
        _opener(_Transport([day, day]),
                models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    reason = str(exc_info.value)
    assert "per-day" in reason and "midnight Pacific" in reason
    assert "gemini-first" in reason and "gemini-second" in reason


def test_transient_throttle_exhaustion_does_not_blame_the_daily_quota():
    """A cascade that fell through with a per-MINUTE cap in it is very likely transient (RPM
    can be as low as 5 on the free tier). The run still stops -- a bare like with no opener is
    worse than halting -- but the stop reason must NOT tell the operator to wait until midnight
    Pacific when restarting a minute later would work."""
    with pytest.raises(GeminiCapacityExhausted) as exc_info:
        _opener(_Transport([_quota_exhausted(quota_id=_PER_DAY_QUOTA_ID),
                            _quota_exhausted(quota_id=_PER_MINUTE_QUOTA_ID)]),
                models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    reason = str(exc_info.value)
    assert "gemini-first (per-day quota)" in reason
    assert "gemini-second (per-minute throttle)" in reason
    assert "restarting in a minute" in reason


def test_models_retired_earlier_in_the_run_still_count_as_per_day_exhausted():
    """A model blacklisted by an earlier profile's per-day 429 is skipped without a request on
    later profiles, so it contributes no fresh error to classify. It must still be reported as
    per-day exhausted, or a later all-exhausted stop would be misreported as transient."""
    day = _quota_exhausted(quota_id=_PER_DAY_QUOTA_ID)
    opener = _opener(_Transport([day, day, day]), models=("gemini-first", "gemini-second"))
    with pytest.raises(GeminiCapacityExhausted):
        opener.generate(Profile(bio="first"), style="s")      # both retired here
    with pytest.raises(GeminiCapacityExhausted) as exc_info:
        opener.generate(Profile(bio="second"), style="s")     # neither is even called now
    reason = str(exc_info.value)
    assert "per-day" in reason and "midnight Pacific" in reason
    assert "restarting in a minute" not in reason


# ---------------------------------------------------------------------------------------
# 404 NOT_FOUND -- a retired/unavailable model must not take the rest of the cascade with it
#
# EMPIRICAL FINDING: gemini-2.5-flash and gemini-2.5-flash-lite were probed with this
# project's real request shape and both returned HTTP 404 NOT_FOUND, "This model
# models/<id> is no longer available to new users" -- despite BOTH still being returned by
# ListModels with "generateContent" in supportedGenerationMethods (see preflight()'s
# docstring). So this is not a hypothetical: a config naming a model id that preflight
# happily approved can still 404 the first time a live profile is actually processed, and a
# 404 says nothing about whether the OTHER configured models still work.
# ---------------------------------------------------------------------------------------

def _not_found(message="This model models/gemini-first is no longer available to new users"):
    return (404, {"error": {"code": 404, "status": "NOT_FOUND", "message": message}})


def test_404_not_found_retires_model_and_second_model_serves_same_profile(capsys):
    """A 404 must not be raised straight to the caller (that would kill every other
    configured model too) -- it retires only the model that 404d, and the cascade proceeds
    to the next configured model for this same profile."""
    transport = _Transport([_not_found(), (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    result = opener.generate(Profile(bio="first"), style="s")

    assert result.model == "gemini-second"
    assert _model_calls(transport) == ["gemini-first", "gemini-second"]
    output = capsys.readouterr().out
    assert "gemini-first" in output and "404" in output
    assert "dropping it from the cascade" in output
    assert "test-key" not in output


def test_404_retired_model_is_skipped_entirely_on_the_next_profile():
    """Like a per-day 429, a 404 permanently retires the model for the rest of THIS run: the
    next profile must skip straight past it without even making a request."""
    transport = _Transport([_not_found(), (200, _success()), (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    opener.generate(Profile(bio="first"), style="s")
    second = opener.generate(Profile(bio="second"), style="s")

    assert second.model == "gemini-second"
    assert _model_calls(transport) == ["gemini-first", "gemini-second", "gemini-second"]


def test_all_404_cascade_raises_capacity_exhausted_pointing_at_fixing_opener_models():
    """When EVERY configured model is gone (not merely out of quota), the message must NOT
    tell the operator to wait for the midnight Pacific reset -- waiting never fixes a
    retired model id. It must point at fixing opener.models instead."""
    transport = _Transport([_not_found(), _not_found()])
    with pytest.raises(GeminiCapacityExhausted) as exc_info:
        _opener(transport, models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    reason = str(exc_info.value)
    assert "opener.models" in reason
    assert "gemini-first" in reason and "gemini-second" in reason
    assert "midnight Pacific" not in reason


def test_mixed_404_and_per_day_quota_cascade_reports_both_scopes_distinctly():
    """A cascade that falls through with one model gone (404) and another merely out of
    per-day quota must report each under its OWN scope, not collapse them into one -- they
    call for different operator responses (fix the config vs. wait for the reset)."""
    transport = _Transport([_quota_exhausted(quota_id=_PER_DAY_QUOTA_ID), _not_found()])
    with pytest.raises(GeminiCapacityExhausted) as exc_info:
        _opener(transport, models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    reason = str(exc_info.value)
    assert "gemini-first (per-day quota)" in reason
    assert "gemini-second (model unavailable)" in reason


@pytest.mark.parametrize("code,status", [(400, "INVALID_ARGUMENT"), (401, "UNAUTHENTICATED"),
                                           (403, "PERMISSION_DENIED")])
def test_non_capacity_errors_do_not_cascade_to_next_model(code, status):
    """A bad key, a permission problem, or a malformed request will fail identically on every
    other model, so burning the rest of the cascade on it is pure waste -- these are raised
    straight to the caller, which classifies them (see opener/service.py). NOTE: 429 is
    deliberately NOT parametrized here anymore -- every 429 is now treated as a capacity
    signal and cascades (see the tests below), even one whose status doesn't say
    RESOURCE_EXHAUSTED; item C of an adversarial audit found that gating the capacity branch
    on an exact status match let a 429 from an infra-level rate limiter/proxy (which doesn't
    carry Gemini's own status string) fall through to `raise error` here and abandon the
    whole cascade."""
    transport = _Transport([(code, {"error": {"code": code, "status": status, "message": "nope"}})])
    with pytest.raises(GeminiAPIError) as exc_info:
        _opener(transport, models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    assert exc_info.value.http_code == code and exc_info.value.status == status
    assert len(transport.calls) == 1


def test_429_with_unrecognized_status_still_cascades_as_capacity(capsys):
    """C: an infra-level rate limiter or proxy in front of the real API can return a bare
    429 that doesn't carry Gemini's own machine-readable RESOURCE_EXHAUSTED status at all --
    it might carry some other status string, or none. Pre-fix, generate()'s capacity branch
    required BOTH http_code == 429 AND status == "RESOURCE_EXHAUSTED", so a 429 like this
    fell through every branch and hit `raise error`, abandoning the whole cascade even
    though gemini-second was healthy and configured. The HTTP 429 status code alone must be
    the trigger; classification of WHICH scope (day/minute/unknown) is a separate concern
    that only affects blacklisting, not whether the 429 cascades at all. With no parseable
    quota detail, this classifies as "unknown" -- same non-blacklisting treatment as a real
    per-minute cap -- so gemini-first stays first in line for the next profile."""
    weird_429 = (429, {"error": {"code": 429, "status": "RATE_LIMIT_EXCEEDED",
                                   "message": "too many requests"}})
    transport = _Transport([weird_429, (200, _success()), (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    first = opener.generate(Profile(bio="first"), style="s")
    second = opener.generate(Profile(bio="second"), style="s")

    assert first.model == "gemini-second"
    assert second.model == "gemini-first"        # not blacklisted -- retried first next profile
    output = capsys.readouterr().out
    assert "unclassified" in output and "NOT blacklisting" in output


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_provider_5xx_cascades_to_the_next_model_without_retiring_it(code, capsys):
    """OBSERVED LIVE: gemini-3.6-flash returned 503 "this model is currently experiencing high
    demand" while every other configured model was serving fine, which proves a 5xx is a
    per-MODEL condition. Raising it would leave the worker to send a bare like with no opener
    while healthy models sat unused, so it must cascade. And because high demand clears on
    its own, the model must NOT be retired -- it stays first in line on the next profile."""
    busy = (code, {"error": {"code": code, "status": "UNAVAILABLE", "message": "high demand"}})
    transport = _Transport([busy, (200, _success()), (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    first = opener.generate(Profile(bio="first"), style="s")
    second = opener.generate(Profile(bio="second"), style="s")

    assert first.model == "gemini-second"      # cascaded rather than failing the profile
    assert second.model == "gemini-first"      # not retired: preferred model retried next time
    assert _model_calls(transport) == ["gemini-first", "gemini-second", "gemini-first"]
    assert "NOT blacklisting" in capsys.readouterr().out


def test_whole_cascade_5xx_reports_a_transient_stop_reason_not_a_quota_one():
    """When every model is merely busy, the run still stops (a bare like is worse than
    halting) -- but the stop reason must say "restart" rather than sending the operator off
    to wait for a daily quota reset that has nothing to do with the failure."""
    busy = (503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "high demand"}})
    with pytest.raises(GeminiCapacityExhausted) as exc_info:
        _opener(_Transport([busy, busy]),
                models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    reason = str(exc_info.value)
    assert "provider 5xx" in reason and "restarting should succeed" in reason
    assert "midnight Pacific" not in reason and "per-day" not in reason


# ---------------------------------------------------------------------------------------
# Transport-level failures (socket.timeout, urllib.error.URLError, or any other OSError the
# transport raises instead of returning) -- these never reach _stdlib_gemini_transport's own
# HTTPError handling (HTTPError is only raised once urlopen() has already succeeded at the
# socket layer and gotten a non-2xx status back; a timeout or a dropped connection fails
# before that point, and _stdlib_gemini_transport does not catch it). Pre-fix, an exception
# here propagated straight out of generate() and abandoned the whole cascade over one
# dropped connection. EMPIRICAL FINDING: a live run against the real API timed out
# mid-request on the first configured model and produced exactly that outcome -- six other
# healthy, configured models were never tried and the profile got no opener.
# ---------------------------------------------------------------------------------------

class _MixedTransport:
    """Like _Transport, but each queued item may be an Exception INSTANCE (raised, simulating
    a transport-level failure below the HTTP layer) or a (code, body) TUPLE (returned,
    simulating an ordinary HTTP response) -- lets a test mix a raised transport error with
    normal HTTP responses in one cascade, the same way _Transport already lets a test queue a
    sequence of different HTTP status codes."""

    def __init__(self, items):
        self.items = iter(items)
        self.calls = []

    def __call__(self, url, payload, headers, timeout, *, method="POST"):
        self.calls.append((url, payload, headers, timeout, method))
        item = next(self.items)
        if isinstance(item, BaseException):
            raise item
        return item


def test_transport_timeout_cascades_to_the_next_model(capsys):
    """A socket.timeout on the first configured model is an OSError, not an HTTPError, so it
    never reaches _stdlib_gemini_transport's status-code handling at all -- it must still
    cascade to the next configured model for this same profile rather than abandoning the
    whole call."""
    transport = _MixedTransport([socket.timeout("timed out"), (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    result = opener.generate(Profile(bio="first"), style="s")

    assert result.model == "gemini-second"
    assert _model_calls(transport) == ["gemini-first", "gemini-second"]
    output = capsys.readouterr().out
    assert "gemini-first" in output and "transport level" in output
    assert "timeout" in output.lower()


def test_transport_url_error_cascades_to_the_next_model(capsys):
    """Same as the socket.timeout case above, for urllib.error.URLError -- the other common
    real-world transport failure (connection reset, refused connection, DNS failure all
    surface through this type)."""
    transport = _MixedTransport([urllib.error.URLError("Name or service not known"),
                                  (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    result = opener.generate(Profile(bio="first"), style="s")

    assert result.model == "gemini-second"
    assert _model_calls(transport) == ["gemini-first", "gemini-second"]
    output = capsys.readouterr().out
    assert "gemini-first" in output and "transport level" in output
    assert "URLError" in output


def test_transport_failure_does_not_blacklist_model_retried_first_next_profile():
    """A transport failure is transient (the same connection could well succeed a second
    later), so unlike a per-day 429 or a 404 it must NOT permanently retire the model: the
    next profile must try the preferred model first, not skip straight past it."""
    transport = _MixedTransport([socket.timeout("timed out"), (200, _success()), (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    first = opener.generate(Profile(bio="first"), style="s")
    second = opener.generate(Profile(bio="second"), style="s")

    assert first.model == "gemini-second"
    assert second.model == "gemini-first"   # preferred model retried first on next profile
    assert _model_calls(transport) == ["gemini-first", "gemini-second", "gemini-first"]


def test_all_models_transport_failure_raises_transient_capacity_exhausted():
    """When every configured model fails at the transport level within one generate() call,
    the loop must still fall through to GeminiCapacityExhausted -- and because every scope
    ends up "busy" (a member of _TRANSIENT_SCOPES), the message must use the transient
    "restarting should succeed" wording, NOT the daily-quota wording: a total network outage
    should stop the run with an accurate reason, not send the operator off to wait for a
    midnight Pacific reset that has nothing to do with the failure."""
    transport = _MixedTransport([socket.timeout("timed out"), urllib.error.URLError("refused")])
    with pytest.raises(GeminiCapacityExhausted) as exc_info:
        _opener(transport, models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    reason = str(exc_info.value)
    assert "restarting should succeed" in reason
    assert "midnight Pacific" not in reason
    assert "per-day" not in reason


def test_transport_type_error_propagates_unchanged():
    """A TypeError from a broken transport implementation is a programming bug, not a flaky
    network -- generate() catches OSError specifically, not bare Exception, so a TypeError
    must propagate straight out of generate() unchanged rather than being silently retried
    across every configured model, which would hide the bug instead of surfacing it."""
    def broken_transport(url, payload, headers, timeout, *, method="POST"):
        raise TypeError("transport is broken")

    with pytest.raises(TypeError, match="transport is broken"):
        _opener(broken_transport, models=("gemini-first", "gemini-second")).generate(
            Profile(), style="s")


def test_transport_failure_printed_line_never_contains_the_api_key(capsys):
    transport = _MixedTransport([socket.timeout("timed out"), (200, _success())])
    _opener(transport, models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    output = capsys.readouterr().out
    assert "test-key" not in output


def test_unusable_billed_response_raises_parse_error_with_normalized_usage():
    response = {"candidates": [], "usageMetadata": {"promptTokenCount": 5,
                                                        "candidatesTokenCount": 2}}
    with pytest.raises(OpenerParseError) as exc_info:
        _opener(_Transport([(200, response)])).generate(Profile(), style="s")
    assert exc_info.value.model == "gemini-primary"
    assert exc_info.value.usage.input_tokens == 5
    assert exc_info.value.usage.output_tokens == 2


# ---------------------------------------------------------------------------------------
# A: a null/empty/non-string `opener` field must never parse successfully -- it is
# downstream typed straight into the Hinge comment box and sent to a real person
# (Hinge.like's `if opener: self.adb.text(opener)`; see worker.py).
# ---------------------------------------------------------------------------------------

def _response_with_raw_opener(opener_value):
    """Like _success(), but lets a test put an arbitrary (non-string) JSON value in the
    'opener' field -- _success() always json.dumps's a real string, which can't represent
    Gemini returning null or a bare number for 'opener' despite the schema marking it
    required (see _SCHEMA's "required" list -- a generation hint, not a runtime guarantee)."""
    body = {"opener": opener_value, "referenced": "pottery mug", "referenced_index": 0}
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7},
    }


@pytest.mark.parametrize("bad_value", [None, 42, 3.14, True, [], {}], ids=[
    "null", "int", "float", "bool", "list", "dict"])
def test_non_string_opener_field_raises_parse_error_instead_of_stringifying(bad_value):
    """Pre-fix, _sanitize()'s str(text) turned {"opener": null} into the literal string
    "None" (truthy -- Hinge.like sends it) and {"opener": 42} into "42", both of which then
    passed the sentence-count guard and parsed SUCCESSFULLY. Every non-string JSON value
    Gemini could plausibly emit here must instead raise OpenerParseError -- still carrying
    usage so the already-billed call is recorded, per this file's established contract for
    a billed-but-unusable response."""
    with pytest.raises(OpenerParseError) as exc_info:
        _opener(_Transport([(200, _response_with_raw_opener(bad_value))])).generate(
            Profile(), style="s")
    message = str(exc_info.value)
    assert type(bad_value).__name__ in message
    assert exc_info.value.model == "gemini-primary"
    assert exc_info.value.usage.input_tokens == 11    # the call was billed -- usage survives


def test_non_string_opener_field_message_never_becomes_the_literal_string_sent():
    """Guards the actual bug: the message must not itself equal the string Hinge would have
    typed (e.g. plain "None"), which would just relocate the bug into the error text."""
    with pytest.raises(OpenerParseError) as exc_info:
        _opener(_Transport([(200, _response_with_raw_opener(None))])).generate(
            Profile(), style="s")
    assert str(exc_info.value) != "None"


@pytest.mark.parametrize("blank", ["", "   ", "\t\n  "], ids=["empty", "spaces", "whitespace"])
def test_empty_or_whitespace_opener_field_raises_parse_error(blank):
    """An empty opener string was falsy and degraded safely downstream by luck alone (see
    Hinge.like's `if opener:` guard) -- this makes it an explicit, named failure instead of
    an accident of truthiness, and a purely-whitespace opener (falsy check would NOT have
    caught this one) is exactly as unusable."""
    with pytest.raises(OpenerParseError) as exc_info:
        _opener(_Transport([(200, _response_with_raw_opener(blank))])).generate(
            Profile(), style="s")
    assert "empty" in str(exc_info.value).lower()


def test_opener_field_that_sanitizes_to_only_dashes_raises_parse_error():
    """A string that is non-empty and non-whitespace BEFORE sanitizing but folds down to
    nothing afterward (pure dash/hyphen content, which _sanitize turns into stripped
    connective punctuation) must be caught by the post-sanitize check, not just the
    pre-sanitize one."""
    with pytest.raises(OpenerParseError) as exc_info:
        _opener(_Transport([(200, _response_with_raw_opener("--- - ---"))])).generate(
            Profile(), style="s")
    assert "empty" in str(exc_info.value).lower()


def test_valid_string_opener_still_parses_normally():
    """Sanity companion to the rejection tests above: an ordinary valid opener must be
    completely unaffected by the new type/emptiness checks."""
    result = _opener(_Transport([(200, _success(opener="That mug has a story"))])).generate(
        Profile(), style="s")
    assert result.opener == "That mug has a story"


# ---------------------------------------------------------------------------------------
# B: a corrupt/truncated photo must raise OpenerError (skip just this profile), never
# escape _fit_images_to_budget as a raw PIL exception that OpenerService can't classify.
# ---------------------------------------------------------------------------------------

def test_corrupt_photo_during_budget_fit_raises_opener_error_not_a_bare_pil_exception(monkeypatch):
    """Pre-fix, Image.open()/img.save() inside _fit_images_to_budget had no try/except: a
    truncated/corrupt `adb screencap` PNG in an oversized profile raised
    PIL.UnidentifiedImageError straight out of generate(). OpenerService can't classify a
    bare PIL exception as GeminiAPIError, so it fell into the generic transient branch and
    printed "swiping without" forever, identically, with no escalation. It must instead
    surface as OpenerError -- the established "skip just this profile" signal -- naming the
    bad photo's index and byte length."""
    good = _noise_png(seed=0)
    corrupt = b"not a real png, just garbage bytes that PIL cannot decode" * 200
    monkeypatch.setattr(opener_module, "_MAX_INLINE_REQUEST_BYTES", 1)  # force the fit path
    opener = _opener(_Transport([]))
    with pytest.raises(OpenerError) as exc_info:
        opener.generate(Profile(photos=[good, corrupt]), style="s")
    message = str(exc_info.value)
    assert "index 1" in message                 # names WHICH photo (0-based, corrupt is 2nd)
    assert str(len(corrupt)) in message          # names its byte length


# ---------------------------------------------------------------------------------------
# D: GeminiOpener must hold its own lock rather than rely on an external caller
# (OpenerService) to serialize concurrent generate() calls on its behalf.
# ---------------------------------------------------------------------------------------

def test_generate_holds_an_internal_lock_so_two_threads_never_double_spend_on_one_model():
    """A future direct call site (a script, a second service) that skips OpenerService's own
    external RLock must not be able to reintroduce a double-spend race where two threads
    both burn a real billed API call on a model this run has already retired -- an
    adversarial audit demonstrated exactly this pre-fix. Two threads call generate()
    concurrently for two different profiles; gemini-first hits its per-day 429 on whichever
    thread's call is served first. With generate() holding its own lock for its FULL
    duration, the second thread's call cannot begin running (not even the transport call for
    gemini-first) until the first is completely done -- so it must see gemini-first already
    retired and go straight to gemini-second, never calling gemini-first a second time."""
    call_log = []
    log_lock = threading.Lock()

    def transport(url, payload, headers, timeout, *, method="POST"):
        if "gemini-first" in url:
            time.sleep(0.05)   # widen the window a missing lock would race inside
        with log_lock:
            call_log.append(url)
        if "gemini-first" in url:
            return _quota_exhausted(quota_id=_PER_DAY_QUOTA_ID)
        return (200, _success())

    opener = GeminiOpener(("gemini-first", "gemini-second"), api_key="test-key",
                          transport=transport)
    results, errors = [], []

    def worker(bio):
        try:
            results.append(opener.generate(Profile(bio=bio), style="s"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"profile-{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors
    assert len(results) == 2 and all(r.model == "gemini-second" for r in results)
    first_calls = [u for u in call_log if "gemini-first" in u]
    assert len(first_calls) == 1, (
        f"gemini-first was called {len(first_calls)} times across two concurrent "
        "generate() calls -- it should be called exactly once (retired for the run "
        "immediately after), proving generate() serializes concurrent callers on its own "
        "rather than relying on an external lock"
    )


def test_geminiopener_constructs_its_own_lock_instance():
    """Cheap direct pin that GeminiOpener no longer depends entirely on an external lock:
    every instance owns a real lock object it can acquire/release on its own."""
    opener = GeminiOpener(("gemini-primary",), api_key="test-key", transport=_Transport([]))
    assert opener._lock.acquire(blocking=False)
    opener._lock.release()


# ---------------------------------------------------------------------------------------
# retry_hint -- lets a caller (OpenerService's retry loop) tell the model what was wrong
# with its previous attempt, so a retry is a corrected re-ask rather than an identical dice
# roll. The owner's rule is that a commentless like must NEVER go out: when a response comes
# back unusable, OpenerService re-asks with a reason rather than degrading, and this is the
# piece of that loop that actually gets the reason in front of the model.
# ---------------------------------------------------------------------------------------

def test_absent_retry_hint_produces_a_byte_identical_request_to_today():
    """The default ("") must not add so much as a stray delimiter or a blank section --
    an ordinary first attempt (the overwhelmingly common case) must build the exact same
    request it did before retries existed."""
    transport = _Transport([(200, _success())])
    opener = _opener(transport)
    opener.generate(Profile(bio="Weekend potter"), style="be curious")

    baseline_transport = _Transport([(200, _success())])
    baseline_opener = _opener(baseline_transport)
    baseline_opener.generate(Profile(bio="Weekend potter"), style="be curious")

    text_with_default_arg = transport.calls[0][1]["contents"][0]["parts"][-1]["text"]
    text_with_no_retry_hint_call = baseline_transport.calls[0][1]["contents"][0]["parts"][-1]["text"]
    assert text_with_default_arg == text_with_no_retry_hint_call
    assert "RETRY" not in text_with_default_arg


def test_empty_string_retry_hint_matches_omitted_retry_hint():
    """Passing retry_hint="" explicitly (as OpenerService's retry loop will on attempt 1)
    must behave identically to not passing it at all -- both are "no correction to make"."""
    transport = _Transport([(200, _success())])
    _opener(transport).generate(Profile(bio="Weekend potter"), style="be curious", retry_hint="")
    text = transport.calls[0][1]["contents"][0]["parts"][-1]["text"]
    assert "RETRY" not in text
    assert text.endswith("Write the opener now.")


def test_nonempty_retry_hint_appears_in_user_text_after_profile_content():
    """The corrective instruction must be appended AFTER the profile text (so it's the most
    recent thing the model reads) and must contain the caller's specific reason verbatim."""
    transport = _Transport([(200, _success())])
    _opener(transport).generate(
        Profile(bio="Weekend potter"), style="be curious",
        retry_hint="the opener field was empty after sanitizing")

    text = transport.calls[0][1]["contents"][0]["parts"][-1]["text"]
    profile_index = text.index("HER PROFILE TEXT:")
    retry_index = text.index("RETRY")
    assert retry_index > profile_index          # appended after, not before or interleaved
    assert "the opener field was empty after sanitizing" in text
    # The HARD REJECTION rules _parse() actually enforces must be restated, not just the bare
    # reason -- a corrected retry needs the model to re-see what "correct" means structurally,
    # not just what it did wrong last time. Asserted on substance (case-insensitively, via
    # loose substrings) rather than the exact literal paragraph: pinning the precise wording
    # verbatim just re-creates the brittleness that broke the moment the prompt was tuned to
    # separate hard-rejection rules from style guidance (see _text_part's own comment).
    lower = text.lower()
    assert "non-empty" in lower and "string" in lower
    assert "two sentences" in lower
    assert "dash" in lower and "hyphen" in lower
    assert "concrete detail" in lower


def test_retry_hint_reaches_the_second_model_after_a_429_cascade(capsys):
    """A 429 on the first model must not drop the hint -- what the previous attempt got
    wrong is still true no matter which configured model ends up serving this retry."""
    transport = _Transport([_quota_exhausted(quota_metric="generate_content_requests_per_minute"),
                            (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    result = opener.generate(Profile(bio="Weekend potter"), style="be curious",
                             retry_hint="previous opener referenced no photo detail")

    assert result.model == "gemini-second"
    assert len(transport.calls) == 2
    for _, payload, *_ in transport.calls:
        text = payload["contents"][0]["parts"][-1]["text"]
        assert "previous opener referenced no photo detail" in text


def test_images_are_encoded_once_not_per_model_when_a_retry_hint_is_supplied():
    """The existing 'encode images once, reuse across the cascade' optimization must still
    hold with a retry_hint in play -- only the text part is allowed to vary per model."""
    png = b"\x89PNG\r\n\x1a\nfirst"
    call_count = {"n": 0}
    real_image_parts = GeminiOpener._image_parts

    def counting_image_parts(self, photos):
        call_count["n"] += 1
        return real_image_parts(self, photos)

    transport = _Transport([_quota_exhausted(quota_metric="generate_content_requests_per_minute"),
                            (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    import unittest.mock
    with unittest.mock.patch.object(GeminiOpener, "_image_parts", counting_image_parts):
        opener.generate(Profile(photos=[png]), style="s", retry_hint="fix it")

    assert call_count["n"] == 1     # encoded once despite two models being tried
    assert len(transport.calls) == 2
    for _, payload, *_ in transport.calls:
        image_part = payload["contents"][0]["parts"][0]
        assert image_part["inlineData"]["data"] == base64.standard_b64encode(png).decode("ascii")


# ---------------------------------------------------------------------------------------
# Per-model thinkingConfig
# ---------------------------------------------------------------------------------------

def test_thinking_config_is_included_verbatim_for_a_configured_model():
    transport = _Transport([(200, _success())])
    opener = _opener(transport, models=("gemini-with-thinking",),
                     thinking={"gemini-with-thinking": {"thinkingLevel": "minimal"}})
    opener.generate(Profile(), style="s")
    payload = transport.calls[0][1]
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}


def test_thinking_config_is_omitted_entirely_for_a_model_with_no_entry():
    transport = _Transport([(200, _success())])
    # thinking is configured for a DIFFERENT model id, so gemini-plain must get no
    # thinkingConfig key at all -- not an empty dict, not a guessed default.
    opener = _opener(transport, models=("gemini-plain",),
                     thinking={"some-other-model": {"thinkingBudget": 0}})
    opener.generate(Profile(), style="s")
    payload = transport.calls[0][1]
    assert "thinkingConfig" not in payload["generationConfig"]


def test_payload_is_rebuilt_per_model_during_a_cascade():
    """Model A has no thinking entry (2.5-family default); model B does (3.x-family
    minimal). A 429s so the call cascades to B -- B's payload must carry B's config,
    not A's, proving the payload is rebuilt per model rather than reused."""
    transport = _Transport([_quota_exhausted(quota_id=_PER_DAY_QUOTA_ID), (200, _success())])
    opener = _opener(transport, models=("gemini-a", "gemini-b"),
                     thinking={"gemini-b": {"thinkingLevel": "high"}})
    result = opener.generate(Profile(), style="s")

    assert result.model == "gemini-b"
    payload_a = transport.calls[0][1]
    payload_b = transport.calls[1][1]
    assert "thinkingConfig" not in payload_a["generationConfig"]
    assert payload_b["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}


# ---------------------------------------------------------------------------------------
# MAX_TOKENS truncation diagnostics
# ---------------------------------------------------------------------------------------

def test_no_text_with_max_tokens_finish_reason_names_the_thinking_truncation():
    response = {
        "candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}],
        "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 0,
                           "thoughtsTokenCount": 250, "cachedContentTokenCount": 0},
    }
    with pytest.raises(OpenerParseError) as exc_info:
        _opener(_Transport([(200, response)]), max_tokens=250).generate(Profile(), style="s")
    message = str(exc_info.value)
    assert "max_tokens" in message.lower()
    assert "thinking" in message.lower()
    assert "250" in message   # both the thought-token count and configured max_tokens
    # The call was billed (thinking tokens cost money) even though it produced no usable
    # opener, so usage must still be carried for the caller to record spend.
    assert exc_info.value.usage.output_tokens == 250
    assert exc_info.value.usage.input_tokens == 20


def test_no_text_with_other_finish_reason_surfaces_it_instead_of_a_generic_message():
    response = {
        "candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 0},
    }
    with pytest.raises(OpenerParseError) as exc_info:
        _opener(_Transport([(200, response)])).generate(Profile(), style="s")
    assert "SAFETY" in str(exc_info.value)


def test_invalid_json_response_surfaces_finish_reason():
    response = {
        "candidates": [{"content": {"parts": [{"text": "not json"}]}, "finishReason": "RECITATION"}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
    }
    with pytest.raises(OpenerParseError) as exc_info:
        _opener(_Transport([(200, response)])).generate(Profile(), style="s")
    assert "RECITATION" in str(exc_info.value)


# ---------------------------------------------------------------------------------------
# 20MB inline-image request budget
# ---------------------------------------------------------------------------------------

def _noise_png(seed: int, size: int = 48) -> bytes:
    """A small but incompressible-ish PNG (random per-pixel color) so a tiny monkeypatched
    byte budget can be exceeded and then satisfied by JPEG recompression without needing
    anywhere near real 20MB/full-resolution-screenshot test fixtures."""
    from PIL import Image  # lazy, same as operation_love/vision/quality.py
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(size * size)])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_oversized_photos_are_compressed_to_fit_the_budget(monkeypatch, capsys):
    photos = [_noise_png(seed) for seed in range(3)]
    # Real cap is 18MB; shrink it to something these tiny fixture photos can exceed so the
    # test stays fast (no need to synthesize anywhere near 20MB of real pixels).
    monkeypatch.setattr(opener_module, "_MAX_INLINE_REQUEST_BYTES", 30_000)
    transport = _Transport([(200, _success())])
    opener = _opener(transport)
    opener.generate(Profile(photos=photos), style="s")

    payload = transport.calls[0][1]
    parts = payload["contents"][0]["parts"]
    image_parts = parts[:-1]   # last part is the text block
    assert len(image_parts) == 3
    sent_size = sum(len(p["inlineData"]["data"]) for p in image_parts)
    assert sent_size <= 30_000
    for p in image_parts:
        assert p["inlineData"]["mimeType"] == "image/jpeg"   # recompressed from PNG

    output = capsys.readouterr().out
    assert "compressed 3 image" in output


def test_photos_already_under_budget_are_sent_untouched():
    png = b"\x89PNG\r\n\x1a\nsmall photo bytes"
    transport = _Transport([(200, _success())])
    opener = _opener(transport)
    opener.generate(Profile(photos=[png]), style="s")

    payload = transport.calls[0][1]
    image_part = payload["contents"][0]["parts"][0]
    assert image_part["inlineData"] == {
        "mimeType": "image/png", "data": base64.standard_b64encode(png).decode("ascii"),
    }


def test_photos_still_over_budget_after_full_compression_raise_opener_error(monkeypatch):
    photos = [_noise_png(seed) for seed in range(2)]
    # A budget this small can never be hit no matter how much the fixture photos compress.
    monkeypatch.setattr(opener_module, "_MAX_INLINE_REQUEST_BYTES", 10)
    opener = _opener(_Transport([]))
    with pytest.raises(OpenerError) as exc_info:
        opener.generate(Profile(photos=photos), style="s")
    message = str(exc_info.value)
    assert "2 image" in message


def test_photos_still_over_budget_error_names_the_actual_byte_composition(monkeypatch):
    """BUG 4 (adversarial audit): the message used to unconditionally blame 'photo count or
    resolution', even though the TEXT part (style guide + profile text + a retry_hint, which
    can itself be a sizable corrective block) counts against the same budget and never
    shrinks here -- only images do. A large retry_hint can push an otherwise-fine profile
    over budget with the photos barely contributing, so telling the operator to trim photos
    in that case is actively misleading. It must instead report the real composition (photo
    bytes vs. text bytes) rather than assuming."""
    photos = [_noise_png(seed) for seed in range(2)]
    monkeypatch.setattr(opener_module, "_MAX_INLINE_REQUEST_BYTES", 10)
    with pytest.raises(OpenerError) as exc_info:
        _opener(_Transport([])).generate(Profile(photos=photos), style="s")
    message = str(exc_info.value)
    assert "2 image" in message
    assert "bytes of images" in message and "bytes of text" in message
    # Conditional guidance, not a blanket assumption -- the operator is told to look at
    # whichever side actually dominates, not always at the images.
    assert "if images dominate" in message and "if text does" in message


# ---------------------------------------------------------------------------------------
# TEST GAP: no existing budget test combined retry_hint with the image-budget path -- the
# budget tests above all pass no hint, and the retry_hint tests earlier in this file all use
# small, comfortably-under-budget profiles. generate() builds text_part WITH retry_hint
# already applied (see _text_part) BEFORE handing it to _fit_images_to_budget, so the size
# check should already measure the hinted text -- but nothing pinned that against
# regression. An under-count here (sizing the request as though retry_hint were still empty)
# would let a retry sail past Gemini's real 20MB inline-data cap and 400 on the live API.
# ---------------------------------------------------------------------------------------

def test_fit_to_budget_measures_the_hinted_text_part_not_a_hintless_baseline(monkeypatch, capsys):
    """Sets a budget that comfortably fits the photo plus an ORDINARY (hint-less) text part
    untouched, but not once the actual (much longer) RETRY-block text this call sends is
    appended -- isolating exactly what an under-count would miss, since retry_hint is what
    generate() is actually asked to send here."""
    photo = _noise_png(seed=0)
    hint = ("reference a different specific detail from her profile, not the one you picked. "
            * 60)
    monkeypatch.setattr(opener_module, "_MAX_INLINE_REQUEST_BYTES", 12_000)

    # Companion baseline: the SAME budget, WITHOUT the hint, doesn't even need to fit -- this
    # proves any compression below is caused by the hint's own size, not some other setting
    # (e.g. the monkeypatched budget alone would already have been too small).
    baseline_transport = _Transport([(200, _success())])
    _opener(baseline_transport).generate(Profile(photos=[photo]), style="s")
    baseline_image = baseline_transport.calls[0][1]["contents"][0]["parts"][0]
    assert baseline_image["inlineData"]["mimeType"] == "image/png"   # sent untouched, no fit needed

    transport = _Transport([(200, _success())])
    opener = _opener(transport)
    opener.generate(Profile(photos=[photo]), style="s", retry_hint=hint)

    image_part = transport.calls[0][1]["contents"][0]["parts"][0]
    # Recompresses ONLY because the HINTED request (photo + the long retry text) is what got
    # measured against the budget -- the hint-less version of this same request (comfortably
    # under 12,000 bytes, per the baseline above) would never have needed to fit at all.
    assert image_part["inlineData"]["mimeType"] == "image/jpeg"
    assert "compressed 1 image" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------
# should_stop -- BUG 1 (adversarial audit): nothing on this path ever consulted a stop
# signal, so a Stop click mid-cascade was silently ignored for as long as
# max_attempts * len(models) * request_timeout_s (5 * 7 * 90s = ~52 minutes against the
# shipped config) while OpenerService's shared lock stayed held the whole time. Checked at
# the TOP of every model iteration, BEFORE that model's request is issued, so the worst case
# is now bounded by ONE already-in-flight HTTP request (request_timeout_s), not the rest of
# the cascade.
# ---------------------------------------------------------------------------------------

def test_should_stop_aborts_the_cascade_before_the_next_models_request():
    """A stop signaled between models must prevent the NEXT model's request entirely: the
    transport must be called exactly once, for the first (already in flight) model, never
    for the second."""
    transport = _Transport([_quota_exhausted(quota_metric="generate_content_requests_per_minute"),
                            (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))
    seen = {"n": 0}

    def should_stop():
        # False on the check before gemini-first's request (it must still be issued -- the
        # "one in-flight request" worst case), True from then on (simulating a Stop click
        # that landed while gemini-first's request was already on the wire).
        seen["n"] += 1
        return seen["n"] > 1

    with pytest.raises(OpenerAborted) as exc_info:
        opener.generate(Profile(), style="s", should_stop=should_stop)

    assert len(transport.calls) == 1                  # gemini-second's request was never issued
    assert _model_calls(transport) == ["gemini-first"]
    assert "gemini-second" in str(exc_info.value)
    assert "stopping" in str(exc_info.value).lower()


def test_should_stop_true_from_the_start_issues_no_requests_at_all():
    """If the run is already stopping before the very first model is even tried, no request
    should be issued at all -- not even to the preferred model."""
    transport = _Transport([(200, _success())])
    opener = _opener(transport, models=("gemini-first",))

    with pytest.raises(OpenerAborted):
        opener.generate(Profile(), style="s", should_stop=lambda: True)

    assert len(transport.calls) == 0


def test_should_stop_is_checked_for_every_model_including_an_already_retired_one():
    """HOLE 2 (mutation audit): both the class docstring (THREAD SAFETY paragraph) and
    generate()'s own should_stop docstring promise the should_stop() check runs at the TOP
    of every model iteration, BEFORE any other per-model handling -- including the "already
    retired this run" skip for a model already sitting in self._unavailable_models. Reading
    the loop confirms the code matches that contract today (the should_stop check is
    genuinely the very first thing done with each `model`), so no production fix is needed
    here -- this test only PINS that ordering.

    A should_stop() that unconditionally returns True cannot actually distinguish the
    documented order from the swapped one: whichever check runs first, the cascade still
    aborts before any request is ever issued, so transport.calls would read 0 either way.
    What DOES distinguish them is whether should_stop() gets INVOKED AT ALL on an iteration
    for a model that is about to be skipped for being already-retired: the documented order
    calls it on every iteration (dead model included), while the swapped order would only
    reach the should_stop() line for a model that survives the retirement check, i.e. never
    for gemini-dead. So should_stop here is False on its first call (gemini-dead's own
    check) and True from its second call onward (gemini-live's check) -- under the
    documented (and actual) order this raises OpenerAborted with ZERO transport calls, but
    under the swapped order gemini-dead's iteration would never call should_stop() at all,
    so gemini-live's iteration would be should_stop's FIRST call (still False), and the
    cascade would go on to issue a real, billed request to gemini-live instead of aborting.
    """
    transport = _Transport([(200, _success())])
    opener = _opener(transport, models=("gemini-dead", "gemini-live"))
    opener._unavailable_models["gemini-dead"] = "day"   # retired earlier this run

    seen = {"n": 0}

    def should_stop():
        seen["n"] += 1
        return seen["n"] >= 2   # False for gemini-dead's own check, True from then on

    with pytest.raises(OpenerAborted) as exc_info:
        opener.generate(Profile(), style="s", should_stop=should_stop)

    assert len(transport.calls) == 0
    assert "gemini-live" in str(exc_info.value)


# ---------------------------------------------------------------------------------------
# preflight()
# ---------------------------------------------------------------------------------------

def _models_page(ids_with_methods, next_token=None):
    body = {"models": [{"name": f"models/{model_id}", "supportedGenerationMethods": methods}
                       for model_id, methods in ids_with_methods]}
    if next_token:
        body["nextPageToken"] = next_token
    return (200, body)


def test_preflight_passes_when_all_configured_models_are_present_and_usable():
    transport = _Transport([_models_page([
        ("gemini-first", ["generateContent"]),
        ("gemini-second", ["generateContent", "countTokens"]),
    ])])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    opener.preflight()   # must not raise

    url, payload, headers, timeout, method = transport.calls[0]
    assert method == "GET"
    assert payload is None   # GET must never carry a JSON body
    assert headers["X-goog-api-key"] == "test-key"
    assert url == "https://generativelanguage.googleapis.com/v1beta/models"


def test_preflight_raises_naming_the_missing_model_id():
    transport = _Transport([_models_page([("gemini-first", ["generateContent"])])])
    opener = _opener(transport, models=("gemini-first", "gemini-typo"))

    with pytest.raises(RuntimeError) as exc_info:
        opener.preflight()
    message = str(exc_info.value)
    assert "gemini-typo" in message
    assert "gemini-first" in message   # available ids listed so the typo is obvious


def test_preflight_raises_when_a_configured_model_lacks_generatecontent_support():
    transport = _Transport([_models_page([("gemini-embed-only", ["embedContent"])])])
    opener = _opener(transport, models=("gemini-embed-only",))

    with pytest.raises(RuntimeError) as exc_info:
        opener.preflight()
    assert "gemini-embed-only" in str(exc_info.value)


def test_preflight_follows_nextpagetoken_pagination():
    page1 = _models_page([("gemini-first", ["generateContent"])], next_token="tok-2")
    page2 = _models_page([("gemini-second", ["generateContent"])])
    transport = _Transport([page1, page2])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    opener.preflight()   # must not raise -- gemini-second is only on page 2

    assert len(transport.calls) == 2
    assert "pageToken=tok-2" in transport.calls[1][0]


def test_preflight_translates_invalid_api_key_400_without_leaking_the_key():
    transport = _Transport([(400, {"error": {
        "code": 400, "status": "INVALID_ARGUMENT",
        "message": "API key not valid. Please pass a valid API key.",
    }})])
    opener = GeminiOpener(["gemini-first"], api_key="super-secret-key", transport=transport)

    with pytest.raises(RuntimeError) as exc_info:
        opener.preflight()
    message = str(exc_info.value)
    assert "not valid" in message.lower()
    assert "super-secret-key" not in message


# ---------------------------------------------------------------------------------------
# D: skip_models -- a parse-failure retry must not re-hit the model that just failed. Passed
# by OpenerService (see service.py's maybe_opener) with the models that already produced an
# unusable response FOR THIS PROFILE; generate()'s cascade steps over them without marking
# them permanently unavailable (see self._unavailable_models), so they stay fully eligible
# again on a fresh call (a new profile, or this same profile with no skip_models supplied).
# ---------------------------------------------------------------------------------------

def test_skip_models_steps_over_the_named_model_in_the_cascade(capsys):
    """gemini-first is skipped even though nothing marks it unavailable/retired -- the
    cascade goes straight to gemini-second."""
    transport = _Transport([(200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    result = opener.generate(Profile(bio="p"), style="s",
                             skip_models=frozenset({"gemini-first"}))

    assert result.model == "gemini-second"
    assert _model_calls(transport) == ["gemini-second"]   # gemini-first: no request at all
    output = capsys.readouterr().out
    assert "skipping gemini-first" in output
    assert "test-key" not in output


def test_skipped_model_is_not_marked_unavailable_and_is_eligible_again_next_call():
    """Unlike a per-day 429 or a 404, a skipped model is NOT retired: it produced a billed,
    well-formed response, just not a usable opener, so it must be tried FIRST again on a
    fresh call with no skip_models."""
    transport = _Transport([(200, _success()), (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    skipped = opener.generate(Profile(bio="p1"), style="s",
                              skip_models=frozenset({"gemini-first"}))
    fresh = opener.generate(Profile(bio="p2"), style="s")   # no skip_models this time

    assert skipped.model == "gemini-second"
    assert fresh.model == "gemini-first"          # NOT retired -- back at the front of the line
    assert _model_calls(transport) == ["gemini-second", "gemini-first"]
    assert opener._unavailable_models == {}        # never touched by a skip


def test_skip_models_safety_valve_ignores_the_set_when_every_model_is_skipped():
    """If EVERY configured model is in skip_models, a stochastic re-ask of an already-failed
    model still beats returning with no opener at all -- the set is ignored entirely rather
    than raising GeminiCapacityExhausted without ever trying a single model."""
    transport = _Transport([(200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    result = opener.generate(Profile(bio="p"), style="s",
                             skip_models=frozenset({"gemini-first", "gemini-second"}))

    assert result.model == "gemini-first"          # cascade ran normally, safety valve engaged
    assert _model_calls(transport) == ["gemini-first"]


def test_skip_models_default_is_empty_and_does_not_change_existing_behavior():
    """Sanity pin: omitting skip_models entirely (every pre-existing call in this file) must
    behave exactly as before -- the cascade tries every configured model in order."""
    transport = _Transport([(200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))
    result = opener.generate(Profile(bio="p"), style="s")
    assert result.model == "gemini-first"
    assert _model_calls(transport) == ["gemini-first"]


def test_skip_models_combines_with_a_capacity_cascade():
    """skip_models and the ordinary per-day/404/transient cascade logic must compose: a
    skipped model contributes nothing to `scopes` (it was never tried this call), while a
    genuinely exhausted model still does."""
    day_exhausted = _quota_exhausted(quota_id=_PER_DAY_QUOTA_ID)
    transport = _Transport([day_exhausted, day_exhausted])   # gemini-first, gemini-second
    opener = _opener(transport, models=("gemini-first", "gemini-second", "gemini-third"))

    with pytest.raises(GeminiCapacityExhausted) as exc_info:
        opener.generate(Profile(bio="p"), style="s",
                        skip_models=frozenset({"gemini-third"}))

    # gemini-first and gemini-second: both actually tried, both hit their per-day quota.
    # gemini-third: skipped entirely -- no request, and (see the assertion below) nothing
    # reported for it in the exhaustion reason either.
    assert _model_calls(transport) == ["gemini-first", "gemini-second"]
    reason = str(exc_info.value)
    assert "gemini-first" in reason and "gemini-second" in reason
    assert "per-day" in reason and "midnight Pacific" in reason
    assert "gemini-third" not in reason             # skipped, never tried -- has nothing to report


# ---------------------------------------------------------------------------------------
# Anchor image -- a live screenshot of Hinge's own like/comment screen, appended after her
# profile photos, that visually shows the ONE photo or prompt the opener will actually be
# attached to and displayed underneath once she sees it. Without it, generate() has no way to
# know which item that is: the model picks whichever photo or prompt it personally finds most
# interesting to write about, and on a real like/comment screen that is frequently NOT the item
# the comment lands under -- the message reads as if it were written for a different photo
# entirely. See generate()'s own docstring for the full rationale; these tests only pin the
# request SHAPE anchoring produces (parts order, system instruction, main text, budget fit,
# cascade behavior, and the corrupt-image error message).
# ---------------------------------------------------------------------------------------

def test_no_anchor_request_is_byte_identical_to_before_anchoring_existed():
    """Compatibility guarantee: a call with no anchor argument at all must build EXACTLY the
    request anchoring never existed for -- the flat systemInstruction text (byte-identical to
    _SYSTEM) and the flat [img, img, text] parts list, no standalone label part anywhere."""
    png_a = b"\x89PNG\r\n\x1a\nfirst"
    png_b = b"\x89PNG\r\n\x1a\nsecond"
    transport = _Transport([(200, _success())])
    _opener(transport).generate(Profile(photos=[png_a, png_b]), style="s")

    payload = transport.calls[0][1]
    assert payload["systemInstruction"]["parts"][0]["text"] == _SYSTEM
    parts = payload["contents"][0]["parts"]
    assert len(parts) == 3
    assert parts[0]["inlineData"]["data"] == base64.standard_b64encode(png_a).decode("ascii")
    assert parts[1]["inlineData"]["data"] == base64.standard_b64encode(png_b).decode("ascii")
    assert "text" in parts[2]
    assert _ANCHOR_LABEL not in parts[2]["text"]


def test_anchored_request_parts_are_profile_photos_then_label_then_anchor_then_text():
    """The defining shape of an anchored request: her profile photos in scroll order, then the
    standalone _ANCHOR_LABEL text part, then the anchor image itself (LAST among the images),
    then the trailing instructions. The label sits immediately before the anchor image rather
    than only being described in the trailing text several parts away -- adjacency is what
    makes "the next image" unambiguous to a model reading one flat sequence of parts (see
    _assemble_parts's own docstring)."""
    png_a = b"\x89PNG\r\n\x1a\nfirst"
    png_b = b"\x89PNG\r\n\x1a\nsecond"
    jpeg_c = b"\xff\xd8\xffanchor"
    transport = _Transport([(200, _success())])
    _opener(transport).generate(Profile(photos=[png_a, png_b]), style="s", anchor=jpeg_c)

    parts = transport.calls[0][1]["contents"][0]["parts"]
    assert len(parts) == 5
    assert parts[0]["inlineData"] == {
        "mimeType": "image/png", "data": base64.standard_b64encode(png_a).decode("ascii"),
    }
    assert parts[1]["inlineData"] == {
        "mimeType": "image/png", "data": base64.standard_b64encode(png_b).decode("ascii"),
    }
    assert parts[2] == {"text": _ANCHOR_LABEL}
    assert parts[3]["inlineData"] == {
        "mimeType": "image/jpeg", "data": base64.standard_b64encode(jpeg_c).decode("ascii"),
    }
    assert "text" in parts[4] and parts[4]["text"] != _ANCHOR_LABEL


def test_anchored_system_instruction_is_system_plus_anchor_system():
    """Once an anchor is present, the systemInstruction must be _SYSTEM with _ANCHOR_SYSTEM
    appended (never a different combined string), and it must tell the model the extra image
    is not one more thing from her profile."""
    transport = _Transport([(200, _success())])
    _opener(transport).generate(Profile(photos=[b"\x89PNG\r\n\x1a\nfirst"]), style="s",
                                anchor=b"\xff\xd8\xffanchor")
    system_text = transport.calls[0][1]["systemInstruction"]["parts"][0]["text"]
    assert system_text == _SYSTEM + _ANCHOR_SYSTEM
    assert system_text.startswith(_SYSTEM)
    assert "is NOT part of her profile" in system_text


def test_anchored_main_text_names_the_like_screen_and_forbids_app_chrome():
    """The trailing text part of an anchored request must tell the model plainly that the
    LAST image is the like screen, that the message attaches to whatever item is shown there,
    and that the app's own interface elements (comment box, Send Like button, keyboard) are
    not hers and must never be described. Asserted on distinctive substrings rather than the
    whole block, since the exact wording is free to be tuned."""
    transport = _Transport([(200, _success())])
    _opener(transport).generate(Profile(photos=[b"\x89PNG\r\n\x1a\nfirst"]), style="s",
                                anchor=b"\xff\xd8\xffanchor")
    text = transport.calls[0][1]["contents"][0]["parts"][-1]["text"]
    lower = text.lower()
    assert "the last image is the like screen" in lower
    assert "your message attaches to the photo or prompt in that last image" in lower
    assert "comment box" in lower and "send like button" in lower and "keyboard" in lower


def test_anchor_with_no_profile_photos_avoids_the_nonsensical_first_0_images_phrasing():
    """A profile whose scroll capture produced zero photos still has an anchor to write
    about: the wording must not fall through to the generic "The first {photo_count}
    image(s)..." branch, which would render as the nonsensical "The first 0 image(s) are her
    profile in scroll order" and send the model hunting through images that were never sent.
    It must instead say the anchor is the only image in the request, and the parts list must
    only be [label, anchor, text] -- no empty profile-photo parts anywhere."""
    transport = _Transport([(200, _success())])
    anchor = b"\xff\xd8\xffanchor"
    _opener(transport).generate(Profile(photos=[]), style="s", anchor=anchor)

    parts = transport.calls[0][1]["contents"][0]["parts"]
    assert len(parts) == 3
    assert parts[0] == {"text": _ANCHOR_LABEL}
    assert parts[1]["inlineData"]["data"] == base64.standard_b64encode(anchor).decode("ascii")
    text = parts[2]["text"]
    lower = text.lower()
    assert "first 0 image" not in lower
    assert "the like screen image below is the only image in this request" in lower
    assert "your message attaches to the photo or prompt shown in that image" in lower


def test_retry_hint_on_an_anchored_request_keeps_the_reanchoring_sentence():
    """_text_part's anchored branch appends an extra corrective sentence re-anchoring a retry
    ("...still attached to the photo or prompt in the final image...") on top of the ordinary
    HARD REJECTION block -- and the anchor image itself must still be present in the retried
    payload, not dropped by the retry path."""
    jpeg_c = b"\xff\xd8\xffanchor"
    transport = _Transport([(200, _success())])
    _opener(transport).generate(Profile(photos=[b"\x89PNG\r\n\x1a\nfirst"]), style="s",
                                anchor=jpeg_c, retry_hint="opener drifted to a different photo")

    parts = transport.calls[0][1]["contents"][0]["parts"]
    text = parts[-1]["text"]
    assert "opener drifted to a different photo" in text
    assert "still attached to the photo or prompt in the final image" in text
    anchor_part = parts[-2]      # [photo, label, anchor, text] -- anchor is second-to-last
    assert anchor_part["inlineData"]["data"] == base64.standard_b64encode(jpeg_c).decode("ascii")


def test_anchor_image_part_is_identical_across_a_model_cascade():
    """The anchor, exactly like her profile photos, is encoded once and reused across every
    model tried in a capacity cascade (see generate()'s "encode once, reuse across the
    cascade" note) -- a 429 on the first model must not cause the anchor to be dropped or
    re-encoded differently for the second model's payload."""
    jpeg_c = b"\xff\xd8\xffanchor"
    transport = _Transport([_quota_exhausted(quota_metric="generate_content_requests_per_minute"),
                            (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    result = opener.generate(Profile(photos=[b"\x89PNG\r\n\x1a\nfirst"]), style="s", anchor=jpeg_c)

    assert result.model == "gemini-second"
    assert len(transport.calls) == 2
    for _, payload, *_ in transport.calls:
        anchor_part = payload["contents"][0]["parts"][-2]
        assert anchor_part["inlineData"]["mimeType"] == "image/jpeg"
        assert anchor_part["inlineData"]["data"] == base64.standard_b64encode(jpeg_c).decode("ascii")


def test_request_size_bytes_counts_the_anchor_label_and_the_anchor_image():
    """_request_size_bytes must total the standalone _ANCHOR_LABEL text part and the anchor
    image's own encoded bytes, not just her profile photos and the trailing text -- otherwise
    an anchored request could sail past Gemini's real 20MB inline cap while the size estimate
    stayed blind to two of the parts actually sent on the wire (see _assemble_parts)."""
    png = b"\x89PNG\r\n\x1a\nfirst"
    jpeg = b"\xff\xd8\xffanchor"
    opener = _opener(_Transport([]))
    image_parts = opener._image_parts([png, jpeg])
    text_part = {"text": "trailing instructions"}

    unanchored_parts = opener._assemble_parts(image_parts, text_part, anchored=False)
    anchored_parts = opener._assemble_parts(image_parts, text_part, anchored=True)
    unanchored_size = opener._request_size_bytes(unanchored_parts, _SYSTEM)
    anchored_size = opener._request_size_bytes(anchored_parts, _SYSTEM)

    # Both share the exact same two images and trailing text; the only structural difference
    # in the anchored layout is the standalone {"text": _ANCHOR_LABEL} part -- so the size
    # delta must equal exactly the label's own encoded length.
    assert anchored_size - unanchored_size == len(_ANCHOR_LABEL.encode("utf-8"))
    # And the anchor image's own bytes (the second image) are counted in both totals -- proving
    # the method never quietly drops the last image once it's playing the anchor role.
    expected_total = (len(_SYSTEM.encode("utf-8"))
                       + len(image_parts[0]["inlineData"]["data"])
                       + len(image_parts[1]["inlineData"]["data"])
                       + len(text_part["text"].encode("utf-8"))
                       + len(_ANCHOR_LABEL.encode("utf-8")))
    assert anchored_size == expected_total


def test_oversized_anchored_request_recompresses_the_anchor_too_and_keeps_part_order(
        monkeypatch, capsys):
    """_fit_images_to_budget recompresses from the FULL images list generate() builds -- her
    profile photos plus the anchor appended at the end -- so the anchor (a full-resolution
    phone screenshot exactly like a profile photo) is not silently exempt from the same size
    pressure that would otherwise shrink only her photos. After recompression the final parts
    list must still end in the correct anchored ORDER: photos, label, (now-compressed) anchor,
    text."""
    photos = [_noise_png(seed) for seed in range(2)]
    anchor = _noise_png(seed=99)
    monkeypatch.setattr(opener_module, "_MAX_INLINE_REQUEST_BYTES", 30_000)
    transport = _Transport([(200, _success())])
    opener = _opener(transport)
    opener.generate(Profile(photos=photos), style="s", anchor=anchor)

    parts = transport.calls[0][1]["contents"][0]["parts"]
    assert len(parts) == 5                      # photo, photo, label, anchor, text
    assert parts[2] == {"text": _ANCHOR_LABEL}
    for image_part in (parts[0], parts[1], parts[3]):
        assert image_part["inlineData"]["mimeType"] == "image/jpeg"   # recompressed, anchor too
    assert "text" in parts[4]
    sent_image_bytes = sum(len(p["inlineData"]["data"]) for p in (parts[0], parts[1], parts[3]))
    assert sent_image_bytes <= 30_000
    assert "compressed 3 image" in capsys.readouterr().out


def test_anchor_decode_failure_names_it_as_the_anchor_not_a_photo_index(monkeypatch):
    """A corrupt anchor screenshot must be named as the like screen anchor image in the raised
    OpenerError, not as "photo index N" -- reporting it as a profile photo index would send
    the operator hunting through her profile photos for a capture bug that is actually in the
    anchor capture path (see _fit_images_to_budget's own comment on this)."""
    good = _noise_png(seed=0)
    corrupt_anchor = b"not a real image, just garbage bytes PIL cannot decode" * 200
    monkeypatch.setattr(opener_module, "_MAX_INLINE_REQUEST_BYTES", 1)   # force the fit path
    opener = _opener(_Transport([]))

    with pytest.raises(OpenerError) as exc_info:
        opener.generate(Profile(photos=[good]), style="s", anchor=corrupt_anchor)

    message = str(exc_info.value)
    assert "the like screen anchor image" in message
    assert "photo index" not in message
    assert str(len(corrupt_anchor)) in message
