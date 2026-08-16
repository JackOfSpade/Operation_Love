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
    ItemRequest,
    OpenerAborted,
    OpenerError,
    OpenerParseError,
    _CONTEXT_LABEL,
    _ITEM_PREAMBLE,
    _ITEM_PREAMBLE_CONTEXT,
    _SCHEMA,
    _SYSTEM,
    _is_thinking_config_rejection,
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
             usage=None, angle="guessing she threw the mug herself",
             item_description="a photo of a pottery mug"):
    """A well-formed model response, with the fields in the SAME order _SCHEMA declares them
    (item_index, referenced, angle, item_description, opener -- see ops/OPENER-REDESIGN.md 3.4
    and 5.7). The order is cosmetic to _parse, which reads by key, but a fixture that emits the
    old opener-first shape would quietly stop representing what a schema-honouring model
    actually returns.

    ``index`` is the MODEL ITEM INDEX and is 1-BASED (doc 5.7), which is why the default is 1
    and not 0: under this contract 0 is the out-of-band "no item" value (ITEM_INDEX_ABSENT), so
    a fixture defaulting to 0 would make every test that never thinks about the index assert
    against a value the model is not allowed to mean.

    ``angle`` is free text and pure telemetry (doc 3.5): nothing branches on it, so it is only
    ever asserted on, never matched against a fixed vocabulary. ``item_description`` is the
    model's own account of WHAT it picked, returned in both modes (doc 5.7).
    """
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps({
            "item_index": index, "referenced": "pottery mug", "angle": angle,
            "item_description": item_description, "opener": opener,
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
    assert result.referenced == "pottery mug"
    # The MODEL ITEM INDEX, 1-based over the numbered items in the request (doc 5.1/5.7). It
    # is not the old referenced_index: that was a 0-based index into raw scroll frames, and
    # the field no longer exists in the schema or on OpenerResult.
    assert result.item_index == 1
    assert not hasattr(result, "referenced_index")
    # `angle` (ops/OPENER-REDESIGN.md 3.4/3.5) must survive the round trip onto OpenerResult:
    # it is what the stores persist as the telemetry column, so a mapping regression here
    # would blank that column silently rather than fail anything. `item_description` is the
    # same kind of round trip, and doc 5.7 requires it in BOTH modes.
    assert result.angle == "guessing she threw the mug herself"
    assert result.item_description == "a photo of a pottery mug"
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
    # The five-field schema as it goes out on the wire, in the order that reaches Gemini.
    # Asserted on payload rather than only on the imported _SCHEMA constant so that "what is
    # actually sent" is pinned, not merely "the payload references the constant" -- the
    # equality above is true no matter what _SCHEMA contains. Field ORDER is load-bearing;
    # see test_response_schema_orders_referenced_and_angle_before_the_opener for why.
    schema = payload["generationConfig"]["responseJsonSchema"]
    assert list(schema["properties"]) == ["item_index", "referenced", "angle",
                                          "item_description", "opener"]
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
    # 2 images sent -> the model is told the items are numbered 1 to 2, and told which field
    # to answer in. Both numbers are derived from the image count in _text_part, so this also
    # pins that the range can never disagree with what was actually sent.
    assert "numbered 1 to 2 in the order shown" in parts[2]["text"]
    assert "Set item_index to the number of the one your opener is about." in parts[2]["text"]
    assert "scroll order" not in parts[2]["text"]


# ---------------------------------------------------------------------------------------
# responseJsonSchema -- the five fields Gemini is asked to return, and the ORDER it is asked
# to return them in. Both are part of the wire format this file exists to keep from drifting.
# ---------------------------------------------------------------------------------------

def test_response_schema_orders_referenced_and_angle_before_the_opener():
    """FIELD ORDER IS LOAD-BEARING, not cosmetic (ops/OPENER-REDESIGN.md 3.4, and the comment
    above _SCHEMA in opener.py). Every model in the cascade runs with thinkingLevel minimal
    (config.yaml opener.thinking), so the model has NO scratchpad of any kind: an earlier
    OUTPUT field is the only place it can do its grounding work before it writes the message.
    The old schema emitted `opener` first, which forced the message itself to carry the
    description of the photo -- root cause #3 of the over-description bug this redesign fixes.
    `referenced` first discharges the description into a field that is never sent to her, and
    `angle` makes the model commit to what its message is doing before it writes it.

    `item_index` LEADS as of Part B (doc 5.1/5.7): it is now a CHOICE the opener must follow
    rather than a label stuck on an opener that was already written, and emitting it after the
    message would invert that -- the model would write first and then name whichever item its
    message happened to suit, which is the blind-then-repair order Part B deletes.

    Nothing else in the codebase would notice a reorder: a dict literal is order-insensitive
    to every consumer, `required` is only a generation hint, and _parse reads by key. So this
    assertion is the ONLY thing standing between a tidy-looking alphabetical reshuffle of the
    _SCHEMA literal and the silent return of the exact bug Part A was written to remove.
    """
    transport = _Transport([(200, _success())])
    _opener(transport).generate(Profile(bio="Weekend potter"), style="s")
    schema = transport.calls[0][1]["generationConfig"]["responseJsonSchema"]

    assert list(schema["properties"]) == ["item_index", "referenced", "angle",
                                          "item_description", "opener"]
    assert schema["required"] == ["item_index", "referenced", "angle", "item_description",
                                  "opener"]
    assert schema["additionalProperties"] is False
    # The replaced field is GONE, not merely reordered. Its old meaning (a 0-based index into
    # raw scroll frames) no longer names anything, so leaving it declared would invite a model
    # to answer in a space nothing consumes -- see opener.py's OpenerResult.item_index comment.
    assert "referenced_index" not in schema["properties"]

    # The order must survive JSON serialization, since that -- not the Python dict -- is what
    # Gemini actually reads. json.dumps preserves insertion order, so this is really a guard
    # against a future sort_keys=True (or any other normalizing step) creeping into the
    # request path and alphabetizing the properties into angle/item_description/..., which
    # would put `opener` second and undo the reorder without touching opener.py at all.
    serialized = json.dumps(schema)
    assert (serialized.index('"item_index"') < serialized.index('"referenced"')
            < serialized.index('"angle"') < serialized.index('"item_description"')
            < serialized.index('"opener"'))

    # The field descriptions are deliberately ADVERSARIAL to each other (doc 3.4), so the
    # routing still works on a model that ignores declared property order entirely -- which is
    # a hypothesis, not a documented Gemini guarantee. Losing either half turns the schema
    # back into five neutral labels that all invite the same description.
    referenced_description = schema["properties"]["referenced"]["description"].lower()
    angle_description = schema["properties"]["angle"]["description"].lower()
    opener_description = schema["properties"]["opener"]["description"].lower()
    assert "never sent to" in referenced_description
    assert "does not leak into the opener" in referenced_description
    assert "must not describe the item" in opener_description
    # With minimal thinking, angle is the only place to plan the relationship between beats
    # before emitting the opener. Keep the premise-consistency check in that scratch field.
    assert "accepts and advances the first beat's claim" in angle_description
    assert "verifying it, contradicting it" in angle_description
    assert "abandoning it for a nearby generic topic" in angle_description
    assert "must not reuse the words from referenced" in opener_description
    # `angle` is free text and telemetry only (doc 3.5) -- an enum would force a pick from a
    # closed set, which is exactly the shoehorning the move list is designed to avoid.
    assert "enum" not in schema["properties"]["angle"]
    assert schema["properties"]["angle"]["type"] == "string"
    # item_index's description carries the three facts doc 5.7 requires of it, because the
    # schema is the only place the model is told what the number MEANS: which list it indexes,
    # that only numbered items are choosable, and that unnumbered context blocks may be
    # referred to but never picked (doc 5.3's two tiers). Substring-matched on substance
    # rather than the full literal so wording can be tuned without a test edit.
    item_index_description = schema["properties"]["item_index"]["description"].lower()
    assert schema["properties"]["item_index"]["type"] == "integer"
    assert "numbered items in this request" in item_index_description
    assert "numbered from 1" in item_index_description
    assert "you may only choose a number that was actually given" in item_index_description
    assert "without a number" in item_index_description
    assert "never pick one" in item_index_description
    # One answer to two questions (doc 5.1): the item the opener is about IS the item liked.
    assert "also the item that will be liked" in item_index_description
    # item_description is free text describing the ITEM, not the detail -- doc 5.7 keeps it
    # strictly separate from `referenced`, which the redundancy monitor (3.7) compares the
    # opener against.
    assert schema["properties"]["item_description"]["type"] == "string"
    assert "enum" not in schema["properties"]["item_description"]


def test_angle_is_mapped_from_the_response_and_degrades_to_empty_when_omitted():
    """`angle` is pure telemetry: nothing branches on it, and no profile may lose its opener
    because a model skipped it. _SCHEMA's "required" list is a generation HINT the API does
    not enforce on the response (the same lesson the non-string `opener` guard was written
    for), so a response with no angle at all must still parse -- carrying "", which is exactly
    what an OpenerResult built without an angle already holds."""
    with_angle = _Transport([(200, _success(angle="teasing her about the kiln"))])
    assert _opener(with_angle).generate(Profile(), style="s").angle == "teasing her about the kiln"

    body = {"referenced": "pottery mug", "item_index": 1, "opener": "That mug has a story"}
    without_angle = _Transport([(200, {
        "candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7},
    })])
    result = _opener(without_angle).generate(Profile(), style="s")
    assert result.angle == ""
    assert result.opener == "That mug has a story"     # the opener itself is unaffected


def test_item_description_is_mapped_from_the_response_and_degrades_to_empty_when_omitted():
    """`item_description` (ops/OPENER-REDESIGN.md 5.7) round-trips onto OpenerResult, and a
    response that omits it must still parse. Same reasoning as `angle` directly above: the
    "required" list is a generation HINT the API does not enforce on the response, and nothing
    branches on this field today -- the pre-flight cross-check that will read it is a later
    workflow -- so a missing description may cost that check its input but must never cost a
    profile its opener.

    Contrast with `item_index`, which degrades to an out-of-band value instead of a plausible
    one: a description that is missing is merely uninformative, an index that is missing would
    otherwise be indistinguishable from a real pick."""
    with_description = _Transport([(200, _success(item_description="a written prompt card"))])
    assert (_opener(with_description).generate(Profile(), style="s").item_description
            == "a written prompt card")

    body = {"item_index": 2, "referenced": "pottery mug", "angle": "guessing",
            "opener": "That mug has a story"}
    without_description = _Transport([(200, {
        "candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7},
    })])
    # Two photos, so item_index=2 names one that was really sent -- an index past the end of
    # the request is refused as ABSENT, which is a different guard and has its own test.
    result = _opener(without_description).generate(Profile(photos=[b"a", b"b"]), style="s")
    assert result.item_description == ""
    assert result.opener == "That mug has a story"     # the opener itself is unaffected
    assert result.item_index == 2                       # and the pick still survives


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
    body = {"item_index": 1, "referenced": "pottery mug",
            "angle": "guessing she threw the mug herself",
            "item_description": "a photo of a pottery mug", "opener": opener_value}
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


def test_redundancy_monitor_logs_but_never_rejects_an_otherwise_valid_opener(capsys):
    """The redundancy monitor (ops/OPENER-REDESIGN.md 3.7) compares the opener against the
    model's own `referenced` note and ships LOG ONLY, deliberately not as a gate: it is a
    lower bound on redundancy that a terse `referenced` defeats, no threshold has been
    calibrated against real data yet, and five consecutive rejections stop a run -- so an
    uncalibrated gate here is a run-killer, not a safety feature. Pinned end to end through
    generate() rather than only as a unit of _redundant_description_markers, because the
    tempting future edit is to raise on it, which would silently convert a monitor into
    exactly the stop condition it was designed not to be.

    This also explains the monitor line every other test in this file prints: the shared
    _success() fixture's opener restates two content words from its own `referenced` note
    ("pottery mug"), which is precisely the over-description shape Part A targets.
    """
    result = _opener(_Transport([(200, _success())])).generate(Profile(), style="s")

    assert result.opener == "That pottery mug has a story. What happened?"   # sent, not rejected
    assert result.redundancy_markers == ['opener restates the referenced word "pottery"',
                                          'opener restates the referenced word "mug"']
    output = capsys.readouterr().out
    assert "redundancy monitor" in output          # loud, never silent
    assert "the opener is being sent" in output


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
    """The default ("") must not add so much as a stray delimiter or a blank section -- an
    ordinary first attempt (the overwhelmingly common case) must build the exact same request
    it did before retries existed.

    Pinned against a LITERAL expected string rather than against a second identical call.
    Comparing two invocations of the same call could only ever prove _text_part is
    deterministic, which was never in doubt; the thing worth guarding is the exact bytes of
    the unanchored text part, because the whole point of this file is that the request shape
    cannot drift silently. The Part A redesign left this branch of _text_part untouched (only
    the ANCHORED closing gained the "you never need to name it" conclusion -- see
    test_anchored_main_text_names_the_like_screen_and_forbids_app_chrome).

    Part B (ops/OPENER-REDESIGN.md 5.1/5.7) is the first change that DOES move this literal,
    and this profile carries no photos, so it exercises the zero-numbered-items branch: with
    nothing numbered to choose from, the model is told so explicitly and told to answer with
    the out-of-band value, rather than being asked for "the index of the one your opener is
    about" out of an empty list -- which invited a confident 0 that was indistinguishable from
    a real pick of the first item under the old 0-based contract. See
    test_unanchored_text_numbers_the_items_from_one for the ordinary case.
    """
    transport = _Transport([(200, _success())])
    opener = _opener(transport)
    opener.generate(Profile(bio="Weekend potter"), style="be curious")

    text_with_default_arg = transport.calls[0][1]["contents"][0]["parts"][-1]["text"]
    assert text_with_default_arg == (
        "STYLE GUIDE:\nbe curious\n\n"
        "HER PROFILE TEXT:\nWeekend potter\n\n"
        "No images of her profile were captured, so there are no numbered items in this "
        "request and the profile text above is everything you have. Set item_index to 0, "
        "which means you could not pick a numbered item. Write the opener now."
    )
    assert "RETRY" not in text_with_default_arg
    # The anchor-only copy must not leak into an unanchored request from either constant.
    assert transport.calls[0][1]["systemInstruction"]["parts"][0]["text"] == _SYSTEM


def test_unanchored_text_numbers_the_items_from_one():
    """The ordinary unanchored request, pinned as a literal for the same reason as the
    zero-image case above: this paragraph is the model's ONLY statement of what the number it
    is about to return indexes.

    1-BASED (ops/OPENER-REDESIGN.md 5.7), and every number in the sentence is derived from the
    image count rather than written out, so the range can never disagree with what was sent.
    The old copy said "her profile in scroll order (index 0 first)", which named a space --
    raw scroll frames -- that is not an item space at all: one card appears in several frames
    and one frame can hold two cards, so the model had to invent its own enumeration and hope
    it matched the driver's."""
    transport = _Transport([(200, _success())])
    _opener(transport).generate(
        Profile(photos=[b"\x89PNG\r\n\x1a\na", b"\x89PNG\r\n\x1a\nb", b"\x89PNG\r\n\x1a\nc"],
                bio="Weekend potter"),
        style="be curious")

    text = transport.calls[0][1]["contents"][0]["parts"][-1]["text"]
    assert text == (
        "STYLE GUIDE:\nbe curious\n\n"
        "HER PROFILE TEXT:\nWeekend potter\n\n"
        "The 3 image(s) above are her profile items, numbered 1 to 3 in the order shown. "
        "Set item_index to the number of the one your opener is about. "
        "Write the opener now."
    )
    # The replaced field's name and its index space must both be gone from the copy: doc 5.7
    # calls out "every line of prompt copy saying scroll order" as part of the breaking change.
    assert "referenced_index" not in text
    assert "scroll order" not in text


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
    # Root cause #1 (ops/OPENER-REDESIGN.md 1.1) named the retry hint as one of three places
    # that demanded the opener prove it looked, with nothing to stop the model from naming the
    # detail instead of merely being grounded in it. The counterweight must survive here too,
    # not just in _SYSTEM and the anchored closing: the detail is a premise, never named or
    # described back to her, and the corrected opener must still carry a claim that could be
    # wrong.
    assert "premise" in lower and "point" in lower
    assert "never name it or describe it back to her" in lower
    assert "must still carry a claim that could be wrong" in lower
    assert "only one direct inference from that visible or stated premise" in lower
    assert "never stack guesses" in lower
    assert "hidden action, route, effort, goal, cause, or sequence" in lower
    assert "natural under the ordinary competing explanations of the scene" in lower
    assert "a hedge does not rescue a far-fetched premise" in lower
    assert "invented ownership, job, routine, responsibility, or relationship" in lower
    assert "recognize the visible or stated clue that led to the guess immediately" in lower
    assert "without reverse-engineering your logic" in lower


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
# THE ITEM-CROP REQUEST SHAPE (ops/OPENER-REDESIGN.md 5.2 and 5.7)
#
# What the model is shown stops being her raw scroll frames and becomes one cropped image per
# profile item, numbered, plus the unnumbered context crops, her name as text, and a truncation
# flag. These tests pin the NEW shape the same way the tests above pin the old one -- as
# literals -- because doc 5.2's whole argument is that "image k IS item k" has to be true by
# construction rather than by the model counting, and the only thing standing between that and
# a silent drift is this file.
#
# The fixtures below are SYNTHETIC bytes, never a real capture: ops/calibration/ holds real
# people's profiles and nothing from it may become a test fixture.
# ---------------------------------------------------------------------------------------

_ITEM_ONE = b"\x89PNG\r\n\x1a\nitem-one-crop"
_ITEM_TWO = b"\x89PNG\r\n\x1a\nitem-two-crop"
_ITEM_THREE = b"\x89PNG\r\n\x1a\nitem-three-crop"
_CONTEXT_ONE = b"\x89PNG\r\n\x1a\nvitals-context-crop"
_SCROLL_FRAME = b"\x89PNG\r\n\x1a\nraw-scroll-frame"


def _part_shape(parts):
    """One request's parts as ("text", str) / ("image", raw bytes) pairs, in wire order.

    Asserting against the whole list at once is deliberate: it pins ADJACENCY (which label sits
    against which image), ORDER (numbered items before context, text last) and CONTENT in a
    single comparison. A test that only checked "the label is somewhere in the request" would
    pass on the exact failure this shape exists to prevent -- a label that has drifted onto the
    wrong crop."""
    shape = []
    for part in parts:
        if "text" in part:
            shape.append(("text", part["text"]))
        else:
            shape.append(("image", base64.b64decode(part["inlineData"]["data"])))
    return shape


def test_item_crop_request_labels_every_image_and_sends_no_scroll_frames():
    """The full wire shape of an item-crop request, pinned part by part.

    Doc 5.2: "With crops, image 3 in the request IS item 3. Agreement by construction." The
    construction is exactly this: a preamble stating the label convention once, then for every
    image a standalone label part IMMEDIATELY before it, numbered items first and context after
    them, then the trailing instruction block. There is no counting step left for the model.

    The profile also carries a scroll frame, which must NOT appear anywhere in the request --
    doc 5.7's "Not sent: full screenshots, scroll frames, the anchor, endorsement blocks". A
    frame sent alongside the crops would re-introduce the duplication bias doc 5.2 removes,
    since a card straddling a scroll seam appears in several frames and repetition reads as
    salience to a model that is now CHOOSING among items."""
    transport = _Transport([(200, _success())])
    items = ItemRequest(name="Sarah", items=[_ITEM_ONE, _ITEM_TWO], context=[_CONTEXT_ONE])

    _opener(transport).generate(
        Profile(photos=[_SCROLL_FRAME], bio="Weekend potter"), style="be curious", items=items)

    payload = transport.calls[0][1]
    assert _part_shape(payload["contents"][0]["parts"]) == [
        ("text", _ITEM_PREAMBLE + _ITEM_PREAMBLE_CONTEXT),
        ("text", "=== ITEM 1 ==="),
        ("image", _ITEM_ONE),
        ("text", "=== ITEM 2 ==="),
        ("image", _ITEM_TWO),
        ("text", _CONTEXT_LABEL),
        ("image", _CONTEXT_ONE),
        ("text",
         "STYLE GUIDE:\nbe curious\n\n"
         "HER NAME:\nSarah\n\n"
         "HER PROFILE TEXT:\nWeekend potter\n\n"
         "The 2 numbered image(s) above are her profile photos, numbered 1 to 2, each shown "
         "immediately after its own ITEM label, so the image after ITEM 1 is item 1. "
         "The 1 image(s) labelled CONTEXT carry no number: use what they show if it helps, "
         "but never pick one. "
         "Set item_index to the number of the one your opener is about. Write the opener now."),
    ]
    # The frame is not merely absent from the image list -- it is nowhere on the wire at all.
    assert _SCROLL_FRAME not in [img for kind, img in _part_shape(payload["contents"][0]["parts"])
                                 if kind == "image"]


def test_item_crop_request_uses_the_shared_system_prompt():
    """An item-crop request uses plain _SYSTEM and contains no like-screen instructions."""
    transport = _Transport([(200, _success())])
    _opener(transport).generate(Profile(), style="s",
                                items=ItemRequest(name="Sarah", items=[_ITEM_ONE]))

    payload = transport.calls[0][1]
    assert payload["systemInstruction"]["parts"][0]["text"] == _SYSTEM
    for kind, value in _part_shape(payload["contents"][0]["parts"]):
        if kind == "text":
            assert "like screen" not in value


def test_item_crop_request_without_context_omits_every_mention_of_context():
    """No context crops sent means no copy about them, in either the preamble or the closing.

    Explaining a CONTEXT label that never appears would be describing something that is not in
    the request -- the same small lie as anchored copy on an unanchored request, and the schema
    description's promise that context blocks are "shown WITHOUT a number" only stays honest
    while the two agree."""
    transport = _Transport([(200, _success())])
    _opener(transport).generate(
        Profile(bio="Weekend potter"), style="be curious",
        items=ItemRequest(name="Sarah", items=[_ITEM_ONE, _ITEM_TWO, _ITEM_THREE]))

    parts = transport.calls[0][1]["contents"][0]["parts"]
    assert _part_shape(parts) == [
        ("text", _ITEM_PREAMBLE),
        ("text", "=== ITEM 1 ==="),
        ("image", _ITEM_ONE),
        ("text", "=== ITEM 2 ==="),
        ("image", _ITEM_TWO),
        ("text", "=== ITEM 3 ==="),
        ("image", _ITEM_THREE),
        ("text",
         "STYLE GUIDE:\nbe curious\n\n"
         "HER NAME:\nSarah\n\n"
         "HER PROFILE TEXT:\nWeekend potter\n\n"
         "The 3 numbered image(s) above are her profile photos, numbered 1 to 3, each shown "
         "immediately after its own ITEM label, so the image after ITEM 1 is item 1. "
         "Set item_index to the number of the one your opener is about. Write the opener now."),
    ]
    assert "CONTEXT" not in parts[0]["text"]
    assert "CONTEXT" not in parts[-1]["text"]


def test_truncated_capture_tells_the_model_it_is_seeing_only_part_of_her_profile():
    """Doc 5.7's truncation flag. Present ONLY when the capture hit its ceiling, and phrased as
    a fact about our capture rather than a deficiency in her profile -- immediately followed by
    "choose from them anyway", so it can never read as licence to decline or to write about the
    part we did not see."""
    transport = _Transport([(200, _success()), (200, _success())])
    opener = _opener(transport, models=("gemini-primary",))
    truncated = ItemRequest(name="Sarah", items=[_ITEM_ONE], truncated=True)
    opener.generate(Profile(), style="s", items=truncated)
    whole = ItemRequest(name="Sarah", items=[_ITEM_ONE], truncated=False)
    opener.generate(Profile(), style="s", items=whole)

    truncated_text = transport.calls[0][1]["contents"][0]["parts"][-1]["text"]
    whole_text = transport.calls[1][1]["contents"][0]["parts"][-1]["text"]
    assert ("Her profile was longer than we could read, so these are only the items we saw. "
            "Choose from them anyway.") in truncated_text
    assert "longer than we could read" not in whole_text
    # The truncation note never displaces the instruction the model acts on.
    assert truncated_text.endswith(
        "Set item_index to the number of the one your opener is about. Write the opener now.")


def test_item_crop_request_passes_her_name_back_as_text():
    """Doc 5.2: "Her name is lost by cropping and must be passed back as text." It is stated as
    a bare labelled fact with no instruction attached -- the point is to restore what the sticky
    header in every scroll frame already carried, not to introduce a new move. An OCR that read
    nothing renders as an explicit "(not read)" rather than a blank line, which would read as a
    name that is blank."""
    transport = _Transport([(200, _success()), (200, _success())])
    opener = _opener(transport)
    opener.generate(Profile(), style="s", items=ItemRequest(name="  Sarah  ", items=[_ITEM_ONE]))
    opener.generate(Profile(), style="s", items=ItemRequest(name="", items=[_ITEM_ONE]))

    assert "HER NAME:\nSarah\n\n" in transport.calls[0][1]["contents"][0]["parts"][-1]["text"]
    assert "HER NAME:\n(not read)\n\n" in transport.calls[1][1]["contents"][0]["parts"][-1]["text"]


def test_item_crop_request_forwards_the_part_a_style_guide_byte_for_byte():
    """Part B is a PAYLOAD change, not a voice change (doc sections 2 and 3 are shipped and
    working). The owner-tunable style guide must arrive unchanged and in the same leading
    position as on every other shape -- combined with the plain-_SYSTEM assertion above, that
    is every Part A property reaching the model exactly as it did: the one rule, the five
    non-binding moves, the three guardrails, the hedge variety, the two-sentence cap and the
    economy rule, and the ASCII/no-dash hard rules."""
    style = (
        "THE ONE RULE: your opener must contain a claim that could be wrong.\n"
        "HEDGE THE CLAIM, NEVER YOURSELF: I'm going to guess, I bet, I heard, I'm assuming.\n"
        "NEVER INVENT THE SENDER.\n"
        "TWO sentences is the absolute maximum."
    )
    transport = _Transport([(200, _success())])
    _opener(transport).generate(Profile(), style=style,
                                items=ItemRequest(name="Sarah", items=[_ITEM_ONE]))

    text = transport.calls[0][1]["contents"][0]["parts"][-1]["text"]
    assert text.startswith(f"STYLE GUIDE:\n{style}\n\n")


def test_legacy_frame_request_carries_no_item_label_copy_at_all():
    """The migration must not leak backwards. A request built the old way (profile.photos, no
    ItemRequest) has no labels adjacent to its images, so it must not carry copy promising any
    -- the model would go looking for labels that are not there."""
    transport = _Transport([(200, _success())])
    _opener(transport).generate(Profile(photos=[_SCROLL_FRAME], bio="Weekend potter"), style="s")

    for kind, value in _part_shape(transport.calls[0][1]["contents"][0]["parts"]):
        if kind == "text":
            assert "=== ITEM" not in value
            assert _CONTEXT_LABEL not in value
            assert _ITEM_PREAMBLE not in value
            assert "HER NAME:" not in value


def test_retired_anchor_argument_is_refused_before_anything_is_billed():
    transport = _Transport([(200, _success())])
    with pytest.raises(TypeError):
        _opener(transport).generate(Profile(), style="s", anchor=b"\x89PNG\r\n\x1a\nlike-screen",
                                    items=ItemRequest(name="Sarah", items=[_ITEM_ONE]))
    assert transport.calls == []     # nothing billed


def test_item_crop_request_survives_a_retry_hint_without_anchor_copy():
    """The retry block is the same corrective block as every other shape (it describes what the
    previous ATTEMPT got wrong, which does not depend on the payload), appended after the
    profile content -- but the anchored re-anchoring sentence must not follow it, because there
    is no final like-screen image to point at."""
    transport = _Transport([(200, _success())])
    _opener(transport).generate(
        Profile(), style="s", retry_hint="the opener field was empty after sanitizing",
        items=ItemRequest(name="Sarah", items=[_ITEM_ONE, _ITEM_TWO]))

    text = transport.calls[0][1]["contents"][0]["parts"][-1]["text"]
    assert text.index("RETRY") > text.index("HER PROFILE TEXT:")
    assert "the opener field was empty after sanitizing" in text
    assert "the final image" not in text and "like screen" not in text


# --- ItemRequest itself: the value type the numbering rests on -------------------------

def test_item_request_orders_images_numbered_first_then_context():
    """`images` IS the numbering: position k - 1 is item k, and every context crop sits after
    every numbered one. _assemble_parts labels by position against this list and never
    reorders, so this ordering is the whole contract."""
    items = ItemRequest(name="Sarah", items=[_ITEM_ONE, _ITEM_TWO], context=[_CONTEXT_ONE])
    assert items.images == (_ITEM_ONE, _ITEM_TWO, _CONTEXT_ONE)
    assert items.item_count == 2 and items.context_count == 1 and items.image_count == 3
    assert [items.label_for(i) for i in range(3)] == [
        "=== ITEM 1 ===", "=== ITEM 2 ===", _CONTEXT_LABEL]


def test_item_request_labels_are_one_based():
    """1-based per doc 5.7 and FIRST_ITEM_INDEX: the first item is ITEM 1, never ITEM 0. Under
    this contract 0 is ITEM_INDEX_ABSENT, so a label reading ITEM 0 would offer the model a
    number that means "I could not pick one"."""
    items = ItemRequest(name="Sarah", items=[_ITEM_ONE])
    assert items.label_for(0) == f"=== ITEM {opener_module.FIRST_ITEM_INDEX} ==="
    assert "ITEM 0" not in items.label_for(0)


def test_item_request_from_profile_transcribes_the_capture_payload():
    """THE ADAPTER (doc 5.7). The driver enumerates, the Profile carries plain bytes, and this
    turns them into the request -- transcription, not translation: the four fields map one for
    one, in order, and nothing here may renumber, reorder or drop a crop. `items` stays first
    and `context` stays after it, because that ordering IS the numbering."""
    profile = Profile(photos=[b"a frame nobody sends"], name="Sarah",
                      items=(_ITEM_ONE, _ITEM_TWO), item_context=(_CONTEXT_ONE,),
                      items_truncated=True)

    request = ItemRequest.from_profile(profile)

    assert request.items == (_ITEM_ONE, _ITEM_TWO)
    assert request.context == (_CONTEXT_ONE,)
    assert request.name == "Sarah"
    assert request.truncated is True
    assert request.images == (_ITEM_ONE, _ITEM_TWO, _CONTEXT_ONE)
    assert b"a frame nobody sends" not in request.images


def test_item_request_from_profile_refuses_a_capture_that_enumerated_nothing():
    """A capture that could not enumerate says so in `Profile.items_unavailable`, and the
    caller's move is to STOP with that sentence -- never to fall back to `profile.photos`, which
    cannot carry an item number at all (one card appears in several frames, one frame can hold
    two cards). Softening this into an empty request would put the substitution back."""
    with pytest.raises(ValueError) as exc_info:
        ItemRequest.from_profile(Profile(photos=[b"f0", b"f1"],
                                         items_unavailable="the top was never confirmed"))
    assert "at least one numbered item" in str(exc_info.value)


def test_item_request_with_no_numbered_items_is_refused():
    """Doc 5.1's contract is that the model CHOOSES an item, so a request offering none cannot
    be answered honestly -- the only available reply is ITEM_INDEX_ABSENT, and paying for a
    billed call to be told what we already knew is worse than raising. The caller hard stops
    (doc 5.3) instead."""
    with pytest.raises(ValueError) as exc_info:
        ItemRequest(name="Sarah", items=[], context=[_CONTEXT_ONE])
    assert "at least one numbered item" in str(exc_info.value)


@pytest.mark.parametrize("bad", [b"", None, "not bytes"])
def test_item_request_refuses_an_unusable_image_because_numbering_is_positional(bad):
    """A missing or empty crop would not merely lose one item, it would RENUMBER every item
    after it -- item 3's label landing on item 4's crop, with the model's answer confidently
    wrong and nothing downstream able to tell. Refused at construction."""
    with pytest.raises(ValueError) as exc_info:
        ItemRequest(name="Sarah", items=[_ITEM_ONE, bad])
    assert "item 2's crop" in str(exc_info.value)


def test_item_request_refuses_an_unusable_context_crop_by_its_own_name():
    """Same guard, and the message must name the crop the way an operator can find it: a
    context crop is not a numbered item and not "photo index N" either."""
    with pytest.raises(ValueError) as exc_info:
        ItemRequest(name="Sarah", items=[_ITEM_ONE], context=[b""])
    assert "context crop 1 (unnumbered)" in str(exc_info.value)


def test_item_request_is_frozen_against_post_construction_mutation():
    """Normalized to tuples at construction, so a caller holding the list it passed in cannot
    renumber a request that has already been built and sized."""
    original = [_ITEM_ONE, _ITEM_TWO]
    items = ItemRequest(name="Sarah", items=original)
    original.append(_ITEM_THREE)
    assert items.items == (_ITEM_ONE, _ITEM_TWO)
    assert items.item_count == 2


# --- the 20MB budget on the crop shape (doc 5.2: never DROP an image) ------------------

def test_oversized_item_crops_are_all_compressed_and_none_is_dropped(monkeypatch, capsys):
    """_fit_images_to_budget must compress every crop or raise; it must never send fewer.

    On the legacy frame shape a dropped image lost some of what the model could look at. On
    this shape it would RENUMBER everything after the gap, so the guarantee matters more, not
    less -- and the labels must still be positionally correct after recompression, which is why
    this asserts the whole shape rather than just the count."""
    crops = [_noise_png(seed) for seed in range(3)]
    context = [_noise_png(seed=9)]
    monkeypatch.setattr(opener_module, "_MAX_INLINE_REQUEST_BYTES", 30_000)
    transport = _Transport([(200, _success())])
    _opener(transport).generate(
        Profile(), style="s",
        items=ItemRequest(name="Sarah", items=crops, context=context))

    parts = transport.calls[0][1]["contents"][0]["parts"]
    kinds = [kind for kind, _ in _part_shape(parts)]
    # preamble, then (label, image) x 4, then the trailing text: nothing dropped, nothing
    # reordered, every image still preceded by its own label.
    assert kinds == ["text", "text", "image", "text", "image", "text", "image",
                     "text", "image", "text"]
    labels = [value for kind, value in _part_shape(parts) if kind == "text"]
    assert labels[1:5] == ["=== ITEM 1 ===", "=== ITEM 2 ===", "=== ITEM 3 ===", _CONTEXT_LABEL]
    image_parts = [p for p in parts if "inlineData" in p]
    assert sum(len(p["inlineData"]["data"]) for p in image_parts) <= 30_000
    for part in image_parts:
        assert part["inlineData"]["mimeType"] == "image/jpeg"    # recompressed from PNG
    assert "compressed 4 image" in capsys.readouterr().out


def test_over_budget_item_request_error_names_crops_not_photo_indexes(monkeypatch):
    """The operator-facing message must describe what was actually sent. "Reduce photo count"
    on a crop request points at her profile photos, which are not in the request at all --
    the same wrong-place mistake the anchor image's own label exists to avoid."""
    monkeypatch.setattr(opener_module, "_MAX_INLINE_REQUEST_BYTES", 10)
    with pytest.raises(OpenerError) as exc_info:
        _opener(_Transport([])).generate(
            Profile(), style="s",
            items=ItemRequest(name="Sarah", items=[_noise_png(0), _noise_png(1)],
                              context=[_noise_png(2)]))
    message = str(exc_info.value)
    assert "2 numbered item crop(s) and 1 context crop(s)" in message
    assert "not scroll frames" in message


def test_corrupt_item_crop_error_names_the_item_number_not_a_photo_index(monkeypatch):
    """A crop PIL cannot decode must still surface as OpenerError (skip just this profile), and
    must name the ITEM, not "photo index 1" -- on this shape there is no photo 1 to go and
    look at."""
    corrupt = b"not a real png, just garbage bytes that PIL cannot decode" * 200
    monkeypatch.setattr(opener_module, "_MAX_INLINE_REQUEST_BYTES", 1)   # force the fit path
    with pytest.raises(OpenerError) as exc_info:
        _opener(_Transport([])).generate(
            Profile(), style="s",
            items=ItemRequest(name="Sarah", items=[_noise_png(0), corrupt]))
    message = str(exc_info.value)
    assert "item 2's crop" in message
    assert "photo index" not in message
    assert str(len(corrupt)) in message


def test_mislabelled_part_count_is_refused_rather_than_shifting_the_numbering(monkeypatch):
    """_assemble_parts labels positionally, so being handed a different number of encoded image
    parts than the request describes would slide every label past the gap onto the wrong crop.
    There is no safe repair for that, so it raises."""
    items = ItemRequest(name="Sarah", items=[_ITEM_ONE, _ITEM_TWO])
    with pytest.raises(ValueError) as exc_info:
        GeminiOpener._assemble_parts([{"inlineData": {"mimeType": "image/png", "data": "x"}}],
                                     {"text": "trailing"}, items=items)
    assert "labels are positional" in str(exc_info.value)


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
    assert exc_info.value.reason_code == "response_blocked"


def test_prompt_feedback_block_is_classified_without_candidates():
    response = {
        "promptFeedback": {"blockReason": "IMAGE_SAFETY"},
        "usageMetadata": {"promptTokenCount": 5},
    }
    with pytest.raises(OpenerParseError) as exc_info:
        _opener(_Transport([(200, response)])).generate(Profile(), style="s")
    assert "promptFeedback.blockReason=IMAGE_SAFETY" in str(exc_info.value)
    assert exc_info.value.reason_code == "prompt_blocked"
    assert exc_info.value.usage.input_tokens == 5


def test_prompt_block_reason_takes_precedence_over_candidate_finish_reason():
    response = {
        "promptFeedback": {"blockReason": "SAFETY"},
        "candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}],
        "usageMetadata": {"thoughtsTokenCount": 250},
    }
    with pytest.raises(OpenerParseError) as exc_info:
        _opener(_Transport([(200, response)]), max_tokens=250).generate(Profile(), style="s")
    assert exc_info.value.reason_code == "prompt_blocked"
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
    generate() is actually asked to send here.

    The budget is DERIVED from the two request sizes this build actually produces, not
    hardcoded. It only means anything while it sits strictly between them, and both numbers
    move with any prompt tuning -- the Part A rewrite roughly doubled _SYSTEM and _SCHEMA,
    which pushed the hint-LESS baseline over the old literal 12,000 cap and failed this test
    for a reason that had nothing to do with what it measures. Deriving the cap keeps the test
    pinned to its actual subject (hinted vs. hint-less sizing) instead of to prompt length,
    and the explicit ordering assertion below is what would fail, loudly and specifically, if
    the hint ever stopped changing the measured size at all.
    """
    photo = _noise_png(seed=0)
    profile = Profile(photos=[photo])
    hint = ("reference a different specific detail from her profile, not the one you picked. "
            * 60)

    # Measured through GeminiOpener's own helpers, so these are the very numbers
    # _fit_images_to_budget compares against the cap -- not an independent re-derivation that
    # could agree with the test while disagreeing with production.
    sizer = _opener(_Transport([]))
    image_parts = sizer._image_parts([photo])

    def _request_size(retry_hint):
        text_part = sizer._text_part(profile, "s", retry_hint)
        return sizer._request_size_bytes(
            sizer._assemble_parts(image_parts, text_part), _SYSTEM)

    hintless_size = _request_size("")
    hinted_size = _request_size(hint)
    cap = (hintless_size + hinted_size) // 2
    assert hintless_size < cap < hinted_size
    monkeypatch.setattr(opener_module, "_MAX_INLINE_REQUEST_BYTES", cap)

    # Companion baseline: the SAME budget, WITHOUT the hint, doesn't even need to fit -- this
    # proves any compression below is caused by the hint's own size, not some other setting
    # (e.g. the monkeypatched budget alone would already have been too small).
    baseline_transport = _Transport([(200, _success())])
    _opener(baseline_transport).generate(profile, style="s")
    baseline_image = baseline_transport.calls[0][1]["contents"][0]["parts"][0]
    assert baseline_image["inlineData"]["mimeType"] == "image/png"   # sent untouched, no fit needed

    transport = _Transport([(200, _success())])
    opener = _opener(transport)
    opener.generate(profile, style="s", retry_hint=hint)

    image_part = transport.calls[0][1]["contents"][0]["parts"][0]
    # Recompresses ONLY because the HINTED request (photo + the long retry text) is what got
    # measured against the budget -- the hint-less version of this same request (comfortably
    # under `cap`, per the baseline above) would never have needed to fit at all.
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


def test_legacy_profile_photo_request_is_flat_and_uses_the_shared_system_prompt():
    png_a = b"\x89PNG\r\n\x1a\nfirst"
    png_b = b"\x89PNG\r\n\x1a\nsecond"
    transport = _Transport([(200, _success())])
    _opener(transport).generate(Profile(photos=[png_a, png_b]), style="s")

    payload = transport.calls[0][1]
    system_text = payload["systemInstruction"]["parts"][0]["text"]
    assert system_text == _SYSTEM
    parts = payload["contents"][0]["parts"]
    assert len(parts) == 3
    assert parts[0]["inlineData"]["data"] == base64.standard_b64encode(png_a).decode("ascii")
    assert parts[1]["inlineData"]["data"] == base64.standard_b64encode(png_b).decode("ascii")
    assert "text" in parts[2]
    lower = (system_text + parts[2]["text"]).lower()
    assert "like screen" not in lower


# ---------------------------------------------------------------------------------------
# 400 THINKING CONFIG REJECTION -- a per-model capability rejection must not take the rest of
# the cascade with it. This is the narrow exception to "every other 4xx raises straight to the
# caller" (see GeminiOpener's class docstring and generate()'s 400 handling).
#
# EMPIRICAL FINDING, live, 2026-08-13, against the real API: gemini-3.7-flash with
# generationConfig.thinkingConfig = {"thinkingLevel": "minimal"} returned
#
#   HTTP 400 INVALID_ARGUMENT
#   "Thinking level MINIMAL is not supported for this model. Please retry with other
#   thinking level."
#
# while every OTHER configured model accepted the identical thinkingLevel on the same run --
# so this 400, despite its status code, is a property of ONE model id's declared capability,
# exactly like a 404, not of the request or the credentials. See _is_thinking_config_rejection.
# ---------------------------------------------------------------------------------------

def _thinking_rejected(message="Thinking level MINIMAL is not supported for this model. "
                                "Please retry with other thinking level."):
    return (400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": message}})


@pytest.mark.parametrize("message", [
    "Thinking level MINIMAL is not supported for this model. Please retry with other "
    "thinking level.",
    "Invalid value at 'generation_config.thinking_config.thinking_level'",
    "thinkingBudget is not supported for this model",
    "Thinking budget exceeds the maximum allowed for this model",
], ids=["measured_gemini_3_7_flash_message", "thinking_config_field_path_underscored",
        "thinkingBudget_camel_case", "thinking_budget_words"])
def test_is_thinking_config_rejection_matches_thinking_related_400_messages(message):
    """The measured message verbatim, plus the thinkingBudget spelling variant the 2.5 model
    family actually uses (see _payload's own comment on thinkingLevel vs. thinkingBudget), must
    all be recognized as a thinking-config rejection."""
    assert _is_thinking_config_rejection(message) is True


@pytest.mark.parametrize("message", [
    "Invalid JSON payload received. Unknown name \"foo\": Cannot find field.",
    "API key not valid. Please pass a valid API key.",
], ids=["generic_malformed_json_payload", "invalid_api_key"])
def test_is_thinking_config_rejection_does_not_match_generic_400_messages(message):
    """THE GUARD AGAINST OVER-MATCHING: a generic malformed-request 400 and an invalid-API-key
    400 must never be mistaken for a thinking-config rejection -- both are properties of the
    request/credentials, not of one model's capability, and must keep raising straight to the
    caller (see the cascade tests below)."""
    assert _is_thinking_config_rejection(message) is False


def test_thinking_config_400_retires_model_and_second_model_serves_same_profile(capsys):
    """A thinking-config 400 must not be raised straight to the caller (that would kill every
    other configured model too) -- it retires only the model whose thinking config was
    rejected, and the cascade proceeds to the next configured model for this same profile."""
    transport = _Transport([_thinking_rejected(), (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    result = opener.generate(Profile(bio="first"), style="s")

    assert result.model == "gemini-second"
    assert _model_calls(transport) == ["gemini-first", "gemini-second"]
    output = capsys.readouterr().out
    assert "gemini-first" in output and "400" in output
    assert "thinking level or budget" in output
    assert "dropping it from the cascade" in output
    assert "test-key" not in output


def test_thinking_config_400_retired_model_is_skipped_entirely_on_the_next_profile():
    """Like a per-day 429 or a 404, a thinking-config 400 permanently retires the model for the
    rest of THIS run: the next profile must skip straight past it without even making a
    request."""
    transport = _Transport([_thinking_rejected(), (200, _success()), (200, _success())])
    opener = _opener(transport, models=("gemini-first", "gemini-second"))

    opener.generate(Profile(bio="first"), style="s")
    second = opener.generate(Profile(bio="second"), style="s")

    assert second.model == "gemini-second"
    assert _model_calls(transport) == ["gemini-first", "gemini-second", "gemini-second"]


def test_generic_400_still_raises_and_does_not_cascade():
    """The end-to-end guard against over-matching: a 400 whose message carries no
    thinking-related token must still raise GeminiAPIError straight to the caller and must NOT
    cascade to the next model -- exactly the behavior this branch must leave untouched for
    every 400 it does not recognize."""
    transport = _Transport([(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                              "message": "Invalid JSON payload received. "
                                              "Unknown name \"foo\": Cannot find field."}})])
    with pytest.raises(GeminiAPIError) as exc_info:
        _opener(transport, models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    assert exc_info.value.http_code == 400
    assert len(transport.calls) == 1        # never cascaded to gemini-second


def test_invalid_api_key_400_still_raises_so_the_service_latch_is_unaffected():
    """OpenerService's immediate-latch path (_is_invalid_gemini_api_key) depends on this exact
    400 reaching the caller unchanged -- it must not be swept up by the new thinking-config
    branch just because it shares the same HTTP status code."""
    transport = _Transport([(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                              "message": "API key not valid. Please pass a "
                                              "valid API key."}})])
    with pytest.raises(GeminiAPIError) as exc_info:
        _opener(transport, models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    assert "api key not valid" in exc_info.value.message.lower()
    assert len(transport.calls) == 1        # never cascaded


def test_all_thinking_config_400_raises_capacity_exhausted_naming_opener_thinking():
    """When EVERY configured model rejects its configured thinking config, the message must
    point at fixing opener.thinking -- and, like the all-404 case, must NOT suggest waiting for
    a quota reset or a plain restart, because neither ever fixes a capability rejection: the
    model will keep 400ing on the same thinkingConfig until opener.thinking is edited."""
    transport = _Transport([_thinking_rejected(), _thinking_rejected()])
    with pytest.raises(GeminiCapacityExhausted) as exc_info:
        _opener(transport, models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    reason = str(exc_info.value)
    assert "opener.thinking" in reason
    assert "gemini-first" in reason and "gemini-second" in reason
    assert "midnight Pacific" not in reason
    assert "restart" not in reason.lower()
    assert "wait" not in reason.lower()


def test_mixed_thinking_and_404_cascade_reports_both_scopes_distinctly():
    """A cascade that falls through with one model's thinking config rejected and another
    actually gone (404) must report each under its OWN scope, not collapse them -- they call
    for different operator fixes (opener.thinking vs. opener.models)."""
    transport = _Transport([_thinking_rejected(), _not_found()])
    with pytest.raises(GeminiCapacityExhausted) as exc_info:
        _opener(transport, models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    reason = str(exc_info.value)
    assert "gemini-first (thinking config rejected by model)" in reason
    assert "gemini-second (model unavailable)" in reason


def test_thinking_config_400_printed_line_never_contains_the_api_key(capsys):
    """House rule: no operator-facing print line may ever contain the API key, even though the
    error message itself (echoed verbatim from Gemini) never carries it either."""
    transport = _Transport([_thinking_rejected(), (200, _success())])
    _opener(transport, models=("gemini-first", "gemini-second")).generate(Profile(), style="s")
    output = capsys.readouterr().out
    assert "gemini-first" in output           # names the model that was retired
    assert "test-key" not in output
