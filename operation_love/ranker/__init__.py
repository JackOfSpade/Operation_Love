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
OPENER_OUTCOME_MATCH = "match"  # she matched (mutual like) some time after send
OPENER_OUTCOME_REPLY = "reply"  # she replied in the conversation this opener sent
OPENER_OUTCOME_NO_RESPONSE = (
    "no_response"  # observed after a stated wait; still nothing back
)
OPENER_OUTCOME_UNMATCH = "unmatch"  # a prior match/conversation later disappeared
OPENER_OUTCOME_UNKNOWN = "unknown"  # observed, but not classifiable into the above
KNOWN_OPENER_OUTCOMES = frozenset(
    {
        OPENER_OUTCOME_MATCH,
        OPENER_OUTCOME_REPLY,
        OPENER_OUTCOME_NO_RESPONSE,
        OPENER_OUTCOME_UNMATCH,
        OPENER_OUTCOME_UNKNOWN,
    }
)

# `opener_outcomes.source` vocabulary -- WHO/WHAT produced the observation, also documented
# rather than enforced. Only `owner` is populated by anything that exists today: collection is
# deliberately manual first (a separate recording CLI, not this data layer, writes these rows).
# `automated` names the shape a future automated reader would use -- WITHOUT any schema change,
# because this column is already plain TEXT/STRING -- so the same table can carry both without
# ever being migrated again for that reason.
OPENER_OUTCOME_SOURCE_OWNER = "owner"
OPENER_OUTCOME_SOURCE_AUTOMATED = "automated"
KNOWN_OPENER_OUTCOME_SOURCES = frozenset(
    {OPENER_OUTCOME_SOURCE_OWNER, OPENER_OUTCOME_SOURCE_AUTOMATED}
)


@runtime_checkable
class Store(Protocol):
    """Shared storage surface both SQLiteStore and BigQueryStore implement."""

    def load_labels(self) -> list[tuple[bool, list[float]]]: ...
    # Exactly the rows load_labels() returns -- same retraction filter, same rows, same order --
    # plus each label's `source` ("" for a NULL one). ranker/threshold.py needs the source to
    # tell MANUAL calibration labels from auto-review answers (a source starting with
    # "auto_review"): a review answer is training data, but it must never set the recall
    # threshold it is also being used to audit. Both backends serve it from the same cache as
    # load_labels (training retrains after every label), and add_label keeps that cache coherent.
    def load_label_rows(self) -> list[tuple[bool, list[float], str]]: ...
    # --- whole-photo features (ranker.feature_set face_crop_whole_photo_v1) ---------------
    # `photo_features` is append-only, keyed by (app, profile_id, feature_version); the row
    # with the latest created_at wins. Nothing below is read in face_crop_v1 mode, and
    # labels.embedding / auto_passes.embedding keep the face-crop vector in every mode.
    # `feature_version` must be vision/photo_cards.py's WHOLE_PHOTO_FEATURE_VERSION, the
    # embedding WHOLE_PHOTO_DIMS finite numbers and card_count >= 1 (zero photo cards is an
    # extraction failure, never a row). `source` is "live" or "backfill". ValueError for a
    # malformed row; True once it is committed (SQLite) or buffered and applied to the
    # in-memory feature map (BigQuery, flushed like a label). `best_effort=True` (face_crop_v1
    # mode, where nothing reads the row) writes it at once and NEVER buffers it, so a rejected
    # row cannot fail a later flush/close: a storage failure is a loud WARNING and False. All
    # keyword-only.
    def add_photo_features(
        self,
        *,
        run_id: str,
        app: str,
        profile_id: str,
        feature_version: str,
        embedding: list[float],
        card_count: int,
        frames_sha256: str,
        source: str,
        best_effort: bool = False,
    ) -> bool: ...
    # Exactly load_label_rows()' rows, in its order, each extended with (photo_vector_or_None,
    # has_photo_row) for `feature_version`, matched by (app, profile_id). has_photo_row False:
    # no row exists (a label with no profile_id never matches). True with None: a zero-card row
    # an earlier build stored, which the label loader counts as incomplete. Served from the
    # same cache as load_label_rows on BigQuery.
    def load_label_feature_rows(
        self, feature_version: str
    ) -> list[tuple[bool, list[float], str, list[float] | None, bool]]: ...
    # {"labels", "with_features", "missing", "zero_cards"} (exact int keys) over those rows:
    # labels == with_features + missing; zero_cards is part of with_features.
    def photo_feature_coverage(self, feature_version: str) -> dict: ...
    # The backfill's work list: one {"run_id","app","profile_id","source","frame_uris",
    # "frame_sha256s"} per non-retracted label lacking a usable row (none, or a stored
    # zero-card row; labels sharing an (app,
    # profile_id) are listed once), frames in photo_index order from profile_photos (manual
    # labels) or auto_pass_photos (auto_review_* labels). Empty frame lists mean this store
    # cannot supply that label's frames (SQLite archives none; a pruned AUTO pass; no
    # profile_id; an incomplete or conflicting archive) -- never a partial set.
    def labels_missing_photo_features(self, feature_version: str) -> list[dict]: ...
    # capture_truncated: the bot hit its per-profile screencap ceiling without ever reaching
    # the profile's true bottom, i.e. this label was made from an INCOMPLETE read. Carried
    # all the way to the store (rather than left in the local, rotating debug log) so the
    # system of record can answer "which labels came from a partial read?" later — the exact
    # query you want if truncated-read labels turn out to be noisier than complete ones.
    def record_profile(
        self,
        run_id: str,
        app: str,
        profile_id: str,
        liked: bool,
        source: str = "manual",
        photos: list[bytes] | None = None,
        photo_count: int = 0,
        capture_truncated: bool = False,
        progress=None,
    ) -> bool: ...
    def add_label(
        self,
        run_id: str,
        app: str,
        liked: bool,
        embedding: list[float],
        source: str = "manual",
        photo_count: int = 0,
        profile_id: str = "",
        profile_name: str = "",
    ) -> None: ...
    def remove_latest_training_label(self) -> dict | None: ...
    def record_decision(
        self,
        run_id: str,
        app: str,
        decision: str,
        score: float,
        source: str = "auto",
        profile_id: str = "",
        created_at: object | None = None,
    ) -> None: ...
    # --- AUTO pass archive + owner review -------------------------------------------------
    # AUTO archives every profile the model PASSES (the screenshots it saw, the score and
    # threshold in force, the embedding) so the owner can later answer "would have liked" or
    # "correct pass"; EVERY answer becomes a training label. `random_sample` marks the unbiased
    # pool the live false-pass rate is measured on. Archiving writes NO label, decision, or
    # profiles/profile_photos row (the training archive is what retraction planning reasons
    # about). Returns False (loudly) when the photos cannot be archived; malformed arguments
    # raise ValueError. All keyword-only.
    def record_auto_pass(
        self,
        *,
        run_id: str,
        app: str,
        profile_id: str,
        score: float,
        threshold: float,
        embedding: list[float],
        photos: list[bytes],
        profile_name: str = "",
        random_sample: bool,
        capture_truncated: bool = False,
    ) -> bool: ...
    # One unreviewed pass of `kind` ("random": uniform among random_sample passes; "near_cut":
    # among the rest, the largest score - threshold), or None when that pool is empty. Returns
    # {"profile_id","app","run_id","created_at" (epoch float),"score","threshold",
    # "photo_count","profile_name","kind"}.
    def next_auto_pass_for_review(
        self, *, kind: str, app: str | None = None
    ) -> dict | None: ...
    def auto_pass_photo(self, *, profile_id: str, index: int) -> bytes | None: ...
    # ValueError for an unknown or already-reviewed pass. Writes the review row AND a training
    # label (source "auto_review_random" / "auto_review_near_cut") as one unit: atomic on
    # SQLite, label-first and retry-idempotent on BigQuery (a pass whose label already exists
    # never gets a second one). Returns {"kind", "label_source"}.
    def record_auto_pass_review(self, *, profile_id: str, would_like: bool) -> dict: ...
    # {"pending_random","pending_near_cut","random_reviewed","random_would_like" (over the most
    # recent `window` random reviews),"near_cut_reviewed","near_cut_would_like","auto_likes",
    # "auto_passes","auto_model_passes"} -- auto_likes/auto_passes count source='auto'
    # decisions at or after the earliest archived pass's own DECISION row (the worker records
    # the decision before it archives, so the window must open there to count that pass; 0/0
    # when none archived). auto_model_passes is the part of auto_passes that entered the archive
    # (the rest are likes the session policy or like-ratio ceiling demoted to a pass, which the
    # random pool never samples). app=None means all apps pooled: the model and its calibration
    # labels are shared.
    def auto_review_stats(
        self, *, app: str | None = None, window: int = 200
    ) -> dict: ...
    # Undo the NEWEST answer to an archived pass: delete its review row and its training label,
    # so the pass is unreviewed again. Returns {"profile_id","profile_name","kind","would_like"}
    # of the undone answer, or None when there is none. ("Remove last saved label" refuses with
    # ValueError instead of touching an answer, and never reaches past one to an older swipe.)
    def undo_latest_auto_review(self) -> dict | None: ...
    # Retention: delete archived passes (manifest, embedding, screenshots -- and on BigQuery the
    # GCS objects) older than `max_age_days` or beyond the `max_passes` newest, answered or not;
    # None/0 leaves a bound off. The owner's ANSWERS (labels, review rows, decisions) are never
    # touched. Returns how many passes were removed.
    def prune_auto_passes(
        self,
        *,
        max_passes: int | None = None,
        max_age_days: float | None = None,
        now: float | None = None,
    ) -> int: ...
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
    def record_opener(
        self,
        run_id: str,
        app: str,
        model: str,
        opener: str,
        referenced: str,
        angle: str = "",
        item_description: str = "",
        *,
        profile_id: str = "",
        decision: str = "",
        decision_source: str = "",
        decision_created_at: object | None = None,
        model_item_index: int | None = None,
        prompt_sha256: str | None = None,
        profile_key: str = "",
    ) -> None: ...
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
    def record_opener_rejection(
        self,
        run_id: str,
        app: str,
        model: str,
        attempt: int,
        reason_code: str | None,
        reason: str,
        raw_opener: str | None,
        *,
        prompt_sha256: str | None = None,
    ) -> None: ...
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
    def record_opener_outcome(
        self,
        app: str,
        profile_key: str,
        outcome: str,
        *,
        observed_at: object | None = None,
        source: str = OPENER_OUTCOME_SOURCE_OWNER,
        note: str = "",
    ) -> None: ...
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
    #
    # The join ALSO keeps only openers that were actually SENT -- `openers.decision = 'like'`,
    # the one value that has ever meant "this opener reached the device" (the same literal
    # tools/opener_outcome_recorder.py calls DECISION_SENT). A drafted-then-DISLIKED opener,
    # a 'synthetic_replay' row, and a NULL/'' decision from before decision tracking are all
    # "never sent", and an outcome is something a SENT opener earned: without this filter a
    # profile drafted-then-disliked in one run and liked in a later one returns the owner's
    # single recorded match TWICE, the second copy crediting a prompt era for a match earned
    # by a draft nobody ever saw.
    #
    # Even WITH the filter this join can still multiply one outcome: a person written to
    # twice, both sent, across two runs is two `openers` rows sharing one `profile_key`, and
    # one `opener_outcomes` row for her joins to BOTH. That is a real ambiguity in the data
    # (nothing records WHICH send a later match answered), not a defect this predicate can
    # remove, so a caller aggregating per-era rates must decide what to do with duplicates
    # rather than assume rows are one-per-outcome.
    def joined_opener_outcomes(
        self, app: str, *, prompt_sha256: str | None = None
    ) -> list[dict]: ...
    def record_spend(
        self, run_id: str, model: str, usage: Usage, cost: float | None
    ) -> None: ...
    def count_today(self, app: str, *, source: str = "auto") -> int: ...
    def spend_today(self) -> float: ...
    def observe_release_persistence_summary(
        self, run_id: str, app: str
    ) -> dict[str, int]:
        """Return only aggregate, run-scoped manual pass/like and successful-opener counts.

        The Hinge AUTO release verifier uses these exact fields to prove a whole supervised
        manual cycle persisted, without ever exporting profile, embedding, or opener content.
        """
        ...

    def ai_observe_release_persistence_summary(
        self, run_id: str, app: str, source: str
    ) -> dict[str, int]:
        """Return run-scoped PASS/LIKE persistence for an explicitly non-manual source.

        This deliberately has a different method and field names from the manual release
        summary so callers cannot relabel manual rows as autonomous evidence by accident.
        """
        ...

    def training_release_persistence_summary(
        self,
        run_id: str,
        app: str,
        *,
        like_profile_id: str,
        pass_profile_id: str,
        like_send_evidence_id: str,
    ) -> dict[str, int]:
        """Return run-scoped manual counts bound to the two profiles a Training release cites.

        Exactly ``like_labels``, ``like_decisions``, ``like_openers``, ``like_send_evidence``,
        ``pass_labels`` and ``pass_decisions``. Each count is filtered to the profile id the
        verified debug chain persisted (and, for the send evidence, the landed evidence id), so
        a Disliked or never-sent draft, another profile's rows, or a retracted label cannot
        stand in for the Like or Pass the release names.
        """
        ...

    def flush(self) -> None: ...
    # Per-table count of rows this store PERMANENTLY GAVE UP ON during this run -- rows that
    # were accepted by a record_* call, never reached the system of record, and never will.
    # Only nonzero tables appear, so `{}` means "nothing was lost" and a caller's whole test is
    # `if store.dropped_rows():`.
    #
    # It exists because "flush() returned without raising" is NOT the same claim as "every row
    # landed", and the shutdown path used to conflate them. BigQueryStore drops a row that has
    # failed _MAX_INSERT_ATTEMPTS consecutive inserts (bad UTF-8 in raw_opener, an over-length
    # field) so one permanently invalid row cannot block every valid row queued behind it; the
    # RuntimeError that carries that news is caught and warned about by opener/service.py so
    # the run continues; by shutdown the buffer is empty and flush() succeeds. Every terminal
    # signal then reads clean, and the operator was told "✅ all data saved" over real data
    # loss. This is the DATA behind BigQueryStore.saved_summary's prose, so supervisor.py can
    # branch on the loss instead of grepping a sentence for it.
    #
    # SQLiteStore returns {} unconditionally and that is a FACT, not a stub: it writes inside
    # the caller's own call and a rejected INSERT raises out of it, so it has no buffer a row
    # can be silently dropped from. Both backends answer, so no caller has to know which one it
    # holds.
    def dropped_rows(self) -> dict[str, int]: ...
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
