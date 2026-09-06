"""Storage interface + factory — picks the backend from config."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..costing import Usage

# `opener_outcomes.outcome` vocabulary -- documented, never enforced at write time. Same
# "suggestions, not restrictions" rule this repo already applies to `record_opener`'s `angle`:
# a closed enum would force a real observation into whichever of these categories fits worst,
# and this column exists precisely so an operator (and later, an automated reader -- see
# `record_opener_outcome`) can say what actually happened. These five are what this project
# can currently tell apart by looking at Hinge:
OPENER_OUTCOME_MATCH = "match"                # she matched (mutual like) some time after send
OPENER_OUTCOME_REPLY = "reply"                # she replied in the conversation this opener sent
OPENER_OUTCOME_NO_RESPONSE = "no_response"    # observed after a stated wait; still nothing back
OPENER_OUTCOME_UNMATCH = "unmatch"            # a prior match/conversation later disappeared
OPENER_OUTCOME_UNKNOWN = "unknown"            # observed, but not classifiable into the above
KNOWN_OPENER_OUTCOMES = frozenset({
    OPENER_OUTCOME_MATCH, OPENER_OUTCOME_REPLY, OPENER_OUTCOME_NO_RESPONSE,
    OPENER_OUTCOME_UNMATCH, OPENER_OUTCOME_UNKNOWN,
})

# `opener_outcomes.source` vocabulary -- WHO/WHAT produced the observation, also documented
# rather than enforced. Only `owner` is populated by anything that exists today: collection is
# deliberately manual first (a separate recording CLI, not this data layer, writes these rows).
# `automated` names the shape a future automated reader would use -- WITHOUT any schema change,
# because this column is already plain TEXT/STRING -- so the same table can carry both without
# ever being migrated again for that reason.
OPENER_OUTCOME_SOURCE_OWNER = "owner"
OPENER_OUTCOME_SOURCE_AUTOMATED = "automated"
KNOWN_OPENER_OUTCOME_SOURCES = frozenset(
    {OPENER_OUTCOME_SOURCE_OWNER, OPENER_OUTCOME_SOURCE_AUTOMATED})


@runtime_checkable
class Store(Protocol):
    """Shared storage surface both SQLiteStore and BigQueryStore implement."""

    def load_labels(self) -> list[tuple[bool, list[float]]]: ...
    # capture_truncated: the bot hit its per-profile screencap ceiling without ever reaching
    # the profile's true bottom, i.e. this label was made from an INCOMPLETE read. Carried
    # all the way to the store (rather than left in the local, rotating debug log) so the
    # system of record can answer "which labels came from a partial read?" later — the exact
    # query you want if truncated-read labels turn out to be noisier than complete ones.
    def record_profile(self, run_id: str, app: str, profile_id: str, liked: bool,
                       source: str = "manual", photos: list[bytes] | None = None,
                       photo_count: int = 0, capture_truncated: bool = False,
                       progress=None) -> bool: ...
    def add_label(self, run_id: str, app: str, liked: bool, embedding: list[float],
                  source: str = "manual", photo_count: int = 0,
                  profile_id: str = "", profile_name: str = "") -> None: ...
    def remove_latest_training_label(self) -> dict | None: ...
    def record_decision(self, run_id: str, app: str, decision: str, score: float,
                        source: str = "auto", profile_id: str = "",
                        created_at: object | None = None) -> None: ...
    def retraction_run_rows(self, run_id: str, app: str, source: str) -> dict: ...
    def append_label_retraction(self, row: dict) -> bool: ...
    def advisory_opener_run_rows(self, run_id: str, app: str) -> dict: ...
    def append_opener_retraction(self, row: dict) -> bool: ...
    # `angle` is the model's own free-text words for what the opener is DOING (guess / know /
    # imagine / tease / connect and anything else it invents) — telemetry only, it never
    # constrains generation. Deliberately not an enum: the opener design's move list is
    # explicitly non-binding, and a closed set would force the shoehorning it's meant to avoid.
    # It exists so we can eventually ask which opener shapes correlate with matches, a question
    # that is unaskable today. Trailing with a "" default so every existing positional caller
    # (and every test double implementing this Protocol) keeps working unchanged.
    #
    # `item_description` is the model's own short description of the ITEM it picked to write
    # about and to like (ops/OPENER-REDESIGN.md 5.7), recorded in auto and observe alike. Not
    # a second copy of `referenced`: that is the DETAIL the opener reacts to, this is what the
    # item IS (a photo, a written prompt), which is the coarse class doc 5.8's pre-flight
    # cross-check compares against our own crop. Telemetry here too -- nothing reads it back at
    # runtime -- and trailing with a "" default for the same compatibility reason as `angle`.
    #
    # `prompt_sha256` is the SHA-256 of the PROMPT ERA that produced this row (see
    # prompt_stamp in opener/opener.py for the digest's exact coverage and its deliberate
    # exclusions). It exists so an offline calibration pass can group rows by era instead of
    # reconstructing prompt-rewrite boundaries from created_at against config.yaml's git
    # commit dates. Keyword-only with a None default: every positional caller is untouched,
    # and a caller that does not stamp writes NULL, which means "predates the stamp" rather
    # than "no era". Parameter ORDER here is pinned against BigQueryStore.record_opener by
    # tests/test_bigquery_store.py, so this name must stay last in both.
    # `profile_key` is the STABLE, cross-time attribution key for the profile this opener was
    # written about -- see ranker/profile_key.py's `profile_key_from_identity` for exactly
    # what it is (a hash of the sticky per-profile header fingerprint `drivers.item_identity`
    # already captures) and, just as importantly, what it is NOT (never the raw fingerprint;
    # never a similarity match; not a replacement for `profile_id`). `profile_id` above stays
    # exactly what it always was: an opaque per-CARD lineage id regenerated every time,
    # binding one opener to the ONE decision that sent it. `profile_key` is what lets an
    # outcome learned days later -- a match, a reply, silence -- join back to this row at all,
    # by WHO the card was rather than by which swipe it was. Threaded exactly like
    # `prompt_sha256` immediately above it: keyword-only, trailing, defaulted, so no existing
    # positional caller changes shape. "" (not None) is the "no key could be derived for this
    # card" value, matching `profile_id`'s own empty-string convention; NULL is reserved for
    # rows that predate this column entirely (see the paired store migration).
    def record_opener(self, run_id: str, app: str, model: str, opener: str, referenced: str,
                      angle: str = "", item_description: str = "", *, profile_id: str = "",
                      decision: str = "", decision_source: str = "",
                      decision_created_at: object | None = None,
                      model_item_index: int | None = None,
                      prompt_sha256: str | None = None,
                      profile_key: str = "") -> None: ...
    # Durable record of a REJECTED opener attempt (OpenerParseError -- see opener/service.py's
    # OpenerParseError handling), captured for every attempt including the final one that
    # exhausts a profile's retries. Without this, only SUCCEEDED openers were ever recorded
    # (record_opener above), so there was no way to ask "how often does each guard fire" or
    # "is the scaffolding/sentence-cap detector too strict" -- the rejected text and reason
    # were printed and then lost forever. reason_code/raw_opener may be None (see
    # OpenerParseError's own docstring for exactly when each is None vs populated).
    # `prompt_sha256` is the same prompt-era digest record_opener carries, so a guard's firing
    # rate is attributable to the prompt that provoked it; keyword-only with a None default,
    # leaving the seven existing parameters positional for every caller and test double.
    def record_opener_rejection(self, run_id: str, app: str, model: str, attempt: int,
                                reason_code: str | None, reason: str,
                                raw_opener: str | None, *,
                                prompt_sha256: str | None = None) -> None: ...
    # The OUTCOME half of the measurement gap `profile_key` exists to close: what PERFORMED,
    # as distinct from everything else this store already records about what the prompt
    # PRODUCED. Deliberately has no `run_id`: an outcome is observed on a real conversation
    # some time after any particular automation run ended (a match noticed the next morning,
    # a reply read a week later), often by the owner looking at the phone rather than by
    # anything this codebase ran, so tying it to a run would either force a fabricated run_id
    # or silently drop every outcome collection can't attribute to one. `app` plus
    # `profile_key` is the whole join key back to `openers` (see `joined_opener_outcomes`).
    #
    # `profile_key` is passed through UNVALIDATED, including `""`: an owner who observed a
    # real outcome but could not pin down which captured profile it belongs to must still have
    # that observation LAND -- silently dropping it would be worse than storing it
    # unattributed, because a dropped row leaves no trace that an observation was ever made at
    # all. `""` here reads back exactly like `""` on `openers.profile_key`: "no key", not "the
    # key is the empty string" -- the two never collide because the read side
    # (`joined_opener_outcomes`) excludes empty and NULL keys from its join on BOTH sides.
    #
    # `outcome` and `source` are free text against the vocabularies documented above this
    # class (`KNOWN_OPENER_OUTCOMES` / `KNOWN_OPENER_OUTCOME_SOURCES`) -- suggestions, not a
    # closed set enforced here, for the same reason `angle` is not one on `record_opener`.
    #
    # `observed_at` is WHEN THE OUTCOME HAPPENED (or was noticed), exactly analogous to
    # `record_opener`'s own `decision_created_at`: optional, defaults to "now" when omitted,
    # and accepted as either an epoch number or a timezone-aware ISO-8601 string (see
    # store.py's shared timestamp helpers). It is deliberately distinct from the row's own
    # `created_at` (when this record was WRITTEN), which every backend stamps itself and which
    # is not a parameter here at all -- an owner backfilling a match from three days ago must
    # be able to say so without lying about when the database learned it.
    #
    # Best-effort at the CALL SITE, not inside this method: like every other record_* method
    # here, a real storage failure raises, and it is whoever wires this into a collection path
    # (worker.py, the recording CLI, a future automated reader -- none of which this change
    # touches) who wraps the call in the same non-fatal try/except every other opener-adjacent
    # write already uses (see opener/service.py's record_opener call site for the pattern).
    def record_opener_outcome(self, app: str, profile_key: str, outcome: str, *,
                              observed_at: object | None = None,
                              source: str = OPENER_OUTCOME_SOURCE_OWNER,
                              note: str = "") -> None: ...
    # Read-only. Every `opener_outcomes` row for `app` whose `profile_key` matches an
    # `openers` row's `profile_key` for the same `app` -- the join `profile_key` exists to
    # make possible, kept HERE (in the store) rather than in whatever tool calls this, so a
    # backend-specific join predicate (SQL vs. an in-memory join) never has to leak into a
    # caller that just wants "what happened to the openers written under prompt era X."
    # Pass `prompt_sha256` to scope to exactly that one prompt era; omit it (the default) to
    # see every era's outcomes at once. There is deliberately no way to ask for only the
    # UNSTAMPED ("predates the stamp") rows through this filter -- `prompt_sha256=None` means
    # "no filter," not "match NULL," mirroring how every other `prompt_sha256` consumer in this
    # store treats NULL as "no era" rather than a filterable era of its own; a caller that
    # needs exactly that split reads unfiltered and checks `row["prompt_sha256"] is None`.
    #
    # The join predicate excludes empty-string AND NULL `profile_key` on BOTH sides -- an
    # `openers` row with no derivable key and an `opener_outcomes` row with no derivable key
    # must never be joined to EACH OTHER just because both happen to be `""`; that would
    # attribute a real observation to an unrelated opener that also lacked identity, which is
    # a worse failure than the outcome staying unjoined. See ranker/profile_key.py's own
    # docstring for why an unattributable row is still worth storing in the first place.
    def joined_opener_outcomes(self, app: str, *,
                              prompt_sha256: str | None = None) -> list[dict]: ...
    def record_spend(self, run_id: str, model: str, usage: Usage, cost: float | None) -> None: ...
    def count_today(self, app: str, *, source: str = "auto") -> int: ...
    def spend_today(self) -> float: ...
    def observe_release_persistence_summary(self, run_id: str, app: str) -> dict[str, int]:
        """Return only aggregate, run-scoped manual pass/like and successful-opener counts.

        The Hinge AUTO release verifier uses these exact fields to prove a whole supervised
        manual cycle persisted, without ever exporting profile, embedding, or opener content.
        """
        ...
    def ai_observe_release_persistence_summary(self, run_id: str, app: str,
                                               source: str) -> dict[str, int]:
        """Return run-scoped PASS/LIKE persistence for an explicitly non-manual source.

        This deliberately has a different method and field names from the manual release
        summary so callers cannot relabel manual rows as autonomous evidence by accident.
        """
        ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


def make_store(cfg, ensure: bool = True) -> Store:
    """Build the configured store. ``ensure=False`` skips BigQuery table/bucket setup.

    Evaluation uses that for pure reads, while correction tools may still append non-photo
    tombstones. Profile-photo upload always refuses on an unverified ``ensure=False`` instance.
    """
    s = cfg.storage
    if s.backend == "bigquery":
        from .bigquery_store import DEFAULT_FLUSH_EVERY, BigQueryStore
        bq = s.bigquery
        return BigQueryStore(
            project_id=bq.get("project_id", ""),
            dataset=bq.get("dataset", "operation_love"),
            location=bq.get("location", "US"),
            photo_bucket=bq.get("photo_bucket", ""),
            flush_every=bq.get("flush_every", DEFAULT_FLUSH_EVERY),
            ensure=ensure,
        )
    from .store import SQLiteStore
    return SQLiteStore(cfg.db_file)
