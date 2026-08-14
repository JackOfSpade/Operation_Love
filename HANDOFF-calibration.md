# Handoff — Hinge 9.134 inline-composer calibration

This is the current operational handoff. It replaces the old paused-PTY,
manual-only, and pending-release instructions. The AI-reviewed production
OBSERVE release artifact is installed and **AUTO is enabled**.

## Completed calibration

- Hinge 9.134 is calibrated for its inline composer, with photo-only targets.
  Prompt/context cards are readable context but are never selected or tapped.
- The reusable hybrid workflow is implemented and is the default after a Hinge
  build/layout/crop change. It is AI-reviewed automation, not manual evidence:
  every protected action is checkpointed and hash-bound before a separate
  reviewer approves, refuses, retries, restarts, or uses a bounded guarded
  scroll. The capture provenance explicitly records
  `human_ground_truth: false`.
- The clean completed calibration split is
  `ops/calibration/targeting_20260814T_hybrid_alt6_calibration`:
  4 profiles, 84 frames, target sequence `[1, 3, 1, 3]`, one pre-heart skip,
  no abort recoveries. Its manifest SHA-256 is
  `f17a80fd79c990046f582f6525ab3c4ab90fe909fab4305883b5b6c2a7043bd3`.
- The separately captured held-out split is
  `ops/calibration/targeting_20260814T_hybrid_alt7_heldout`:
  3 profiles, 81 frames, target sequence `[1, 3, 1]`, no skips and no abort
  recoveries. Its manifest SHA-256 is
  `79932517345ee19ec1d906cf299346cbbf2ba78a1c6ff5ae1feca441eb940e36`.
- The deterministic second-process review is
  `ops/calibration/targeting_20260814T_hybrid_alt6_alt7-terra-independent-review.json`.
  File SHA-256: `9983cbcde6e0da16dfbf6171eb272211ad6b3479df80a536403b0ae316283a8b`;
  its evidence SHA-256 is
  `ed902a75cec0e8cdffcef2dd60911df2e510936c7d556777d02969482b4cb2e7`.
  It is independent review of recorded evidence, not independent ground truth.
- The measurement ledger is
  `ops/calibration/targeting_20260814T_hybrid_alt7_heldout/measurement_ledger.json`
  (SHA-256 `98ac8b6ae9a2e53ad8fae4a4717c95925201ae60bbe344fb1151fb536890ef95`).
  Frozen bounds are `identity_match_max_dist: 1.9721` and
  `inline_item_max_dist: 14.9099`; held-out results are zero foreign accepts
  and zero false refusals for both checks.
- `config.yaml` has the measured schema-v3 `targeting_calibration` installed
  for Pixel 7a serial `33111JEHN04475`, Hinge `9.134.0`, `1080x2400`, inline
  layout, and photo-only policy.
- Production OBSERVE run `312b80bf4b50` persisted two reviewed Passes, one
  reviewed Like, and one successful Gemini opener. The release sequence binds
  Pass -> hub publication -> item-5 anchor -> post-tap verification -> reviewed
  open -> reviewed send attempt -> frame-bound Like -> verified landing.
- Its action provenance is
  `ops/release/312b80bf4b50/provenance/hinge_ai_observe_action_provenance.json`
  (SHA-256 `2845e7642dcb90f135d6359907b2594e19a7031034bfdb8812c8608740216316`).
  The separate Terra review is
  `ops/release/312b80bf4b50/independent-review/hinge_ai_observe_independent_review.json`
  (SHA-256 `e7244ae68da4f42664dfd02cdb91a796598ed6ab53b9e23b36f558035c9667e5`).
- The verified release artifact is
  `ops/release/312b80bf4b50/release/hinge_ai_observe_release.json`
  (SHA-256 `60952f85f77f5f6ce89a825e08f2b41c7d4818d7ffc1ea7e39620d35b307c303`).
  `config.yaml` binds it under `ai_reviewed_observe_release_evidence`, removes
  the OBSERVE-only controller fields, and sets global mode to `auto`.
- The artifact truthfully records `human_ground_truth: false`; this is an
  explicitly accepted AI-reviewed release, not manual evidence.

## Current live state

No hub or Worker is running. The phone is on Melissa with no composer open and
was deliberately left six capture scrolls below profile top when the completed
run stopped. A subsequent Worker-owned capture must restore/confirm top before
enumeration; do not reuse a stale profile token or anchor.

The owner approved sending Hinge profile data to Gemini and storing the run in
the configured BigQuery/GCS. The release run completed under that approval.

Earlier run `a01fbcd1e9a0` contains a false Malaika Pass at debug row 225 caused
by a now-fixed second-driver race. Its exact label/paired-decision retraction was
appended to BigQuery and an identical retry returned an idempotent no-op. The
bound plan is `ops/corrections/a01fbcd1e9a0-malaika-plan.json` (plan SHA-256
`810137e5ebc27ccb7595c127cb14889e5620863dd0f44379494b1444d75510fa`).
The forensic profile/photos remain; training and release aggregate queries
exclude the retracted pair.

The no-decision opener audit on 2026-08-14 found seven legacy eager-generation
rows across runs `0da87bfda01c`, `712bead758d5`, `a01fbcd1e9a0`,
`ac983d1a3f6e`, and `f588e2fcc34c`. The owner approved cleanup. Five
self-hashed plans under `ops/corrections/opener-cleanup-*.json` appended seven
exact-row `opener_retractions`; a live idempotency replay was a no-op. The
postcheck proved zero visible openers in those runs, seven preserved spend
rows, zero opener/Like mismatches across all Hinge runs, and no orphan
label/photo/retraction records. The consolidated audit is
`ops/corrections/phantom_openers_20260814.json`.

Generation is now staged in both OBSERVE and AUTO. Billing spend is recorded
when the provider call happens, but profile-attributable opener telemetry is
committed only after a verified Like and after its durable decision. New
opener rows carry nullable action lineage (`profile_id`, decision/source/time,
and model item index). Pass, Stop, resync, targeting refusal, paywall, and
failed send paths leave no committed opener row.

## Reusable production OBSERVE release sequence

After a future Hinge build/layout recalibration, keep Hinge in OBSERVE and
repeat one Worker-owned reviewed Pass/Like cycle. Produce provenance, have a
different reviewer identity/process review it, then verify against the active
store:

   ```bash
   python -m tools.hinge_observe_ai_release provenance \
     --config config.yaml --debug-run data/hinge_debug/<worker-run-id> \
     --run-id <worker-run-id> --out ops/release/<worker-run-id>/provenance \
     --source external_ai_review \
     --acceptance I_ACCEPT_AI_REVIEWED_OBSERVE_RELEASE_RISK \
     --executor-model gpt-5.6-sol --executor-id codex-root \
     --executor-version 2026-08-14 \
     --executor-process codex-live-observe-controller-v1

   python -m tools.hinge_observe_ai_release review \
     --debug-run data/hinge_debug/<worker-run-id> --run-id <worker-run-id> \
     --provenance ops/release/<worker-run-id>/provenance/hinge_ai_observe_action_provenance.json \
     --out ops/release/<worker-run-id>/independent-review \
     --reviewer-model gpt-5.6-terra --reviewer-id <different-id> \
     --reviewer-version 2026-08-14 --reviewer-process <different-process>

   python -m tools.hinge_observe_ai_release verify \
     --debug-run data/hinge_debug/<worker-run-id> --run-id <worker-run-id> \
     --provenance ops/release/<worker-run-id>/provenance/hinge_ai_observe_action_provenance.json \
     --review ops/release/<worker-run-id>/independent-review/hinge_ai_observe_independent_review.json \
     --out ops/release/<worker-run-id>/release \
     --acceptance I_ACCEPT_AI_REVIEWED_OBSERVE_RELEASE_RISK
   ```

Install only the emitted `ai_reviewed_observe_release_evidence` mapping,
validate configuration, rerun the full suite, then and only then change `mode`
to `auto`. Never reuse the current release artifact after its calibration,
device, Hinge build, framebuffer, or action log binding changes.

## Reusable recalibration command set

Use fresh campaign directories under an unchanged `config.yaml`; do not append
to or relabel a prior session. The target strategy stays one photo heart then
Pass per profile (`[1,3,1,3]` calibration and `[1,3,1]` held-out), with no
navigation after the inline composer opens.

```bash
python -m tools.hinge_calibrate capture --split calibration --profiles 4 --hybrid-review \
  --confirmation I_ACCEPT_EXTERNAL_REVIEWED_AUTOMATION_RISK \
  --reviewer-model <reviewer-model> --reviewer-process <reviewer-process> \
  --out ops/calibration/targeting_<campaign>-calibration
python -m tools.hinge_calibrate capture --split heldout --profiles 3 --hybrid-review \
  --confirmation I_ACCEPT_EXTERNAL_REVIEWED_AUTOMATION_RISK \
  --reviewer-model <reviewer-model> --reviewer-process <reviewer-process> \
  --out ops/calibration/targeting_<campaign>-heldout
python -m tools.hinge_calibration_review \
  ops/calibration/targeting_<campaign>-calibration \
  ops/calibration/targeting_<campaign>-heldout --config config.yaml \
  --out ops/calibration/targeting_<campaign>-independent-review.json
python -m tools.hinge_calibrate measure \
  ops/calibration/targeting_<campaign>-calibration \
  ops/calibration/targeting_<campaign>-heldout --config config.yaml \
  --accept-hybrid-reviewed-evidence \
  --confirmation I_ACCEPT_EXTERNAL_REVIEWED_AUTOMATION_RISK \
  --unattended-review ops/calibration/targeting_<campaign>-independent-review.json
```

The reviewer checks exact capture/config bytes, PNG hashes, scope-appropriate
target proof, device/build/framebuffer, and action order; it does not import
ADB, capture, or vision code. A refusal, stale hash, malformed reviewer input,
interruption, or unresolved screen state ends that session. Preserve it for
diagnosis and start a fresh session from a confirmed profile top.

## Verification status

- Full test suite after AUTO release installation and no-decision advisory fix:
  **2175 passed, 2 skipped**.
- Ruff is clean for `operation_love`, `tools`, and `tests`; `git diff --check`
  is clean.
- Full-repository Ruff still reports pre-existing legacy-only errors. Do not
  hide them by weakening the scoped quality gate.
- Real profile/debug artifacts under `ops/calibration/` and
  `data/hinge_debug/` are private and gitignored; do not upload or commit them.
