# Training review viewport recovery — 2026-10-23

Run `d577d2074aad` halted while resuming the owner's **Dislike** choice.
The phone and retained error frame agreed: the selected inline composer was
still open, but review scrolling moved its CTA centre to `(699, 1038)` on a
1080×2400 frame. Its comment field was `(95,778)–(985,956)` and CTA was
`(390,988)–(985,1097)`. The strict detector searches CTA centres only at
54–84% display height; this one was at 43.25%.

Simply widening the send search would not complete the checkpoint. Offline
verification of the same frame rejects the selected photo because 683 of its
1109 source rows are hidden, beyond the existing 40% suggestion-shelf reframe
limit. Both safeguards correctly withheld a decision; the missing operation
was restoring the viewport after review.

The resume settle loop now recognises complete composer topology across the
content band for navigation only, proves the stored profile identity, and
drags downward above the input and CTA. It shares the existing three-attempt
bound and final Stop/foreground/input guards. Missing controls, foreign or
unknown identity, or an unchanged viewport cannot produce a decision. Every
restored frame must pass the original strict composer, profile, selected-item,
pass-control and applicable draft checks. The initial pre-review checkpoint
does not acquire navigation recovery.

Live validation used the configured persistent UHID transport under the device
lock. Two swipes restored the same draft without sending Like or Dislike.
The resulting comment field was `(95,1197)–(985,1375)`, CTA was
`(390,1407)–(985,1516)`, and the pass X was at `(116,2062)`.
Private captures and navigation records are in
`data/hinge_debug/checkpoint_20261023/`.

The original model item 1 crop was recovered by exact SHA-256 prefix
`654e4873a00ae61f` from the local replay corpus. An offline reconstruction of
the six source crops restored the original screenshot's sRGB conversion before
deriving grey verification references. The unchanged item verifier rejected
the original viewport and accepted the restored viewport as item 1 at distance
1.1118, below both its relative bound 22.9030 and absolute ceiling 14.9099.
This reconstruction is diagnostic evidence, not a new calibration or label.
No historical decision or store record was changed.

New `training_checkpoint_viewport_restore` records retain the pre-scroll frame,
target item and gesture coordinates. Existing bug reports already include the
action trail and error captures; no report-format change was required.

Validation: 617 focused tests passed, including composer topology, viewport
recovery, identity/item refusal, bounded retries, Stop, Training checkpoints,
foreground/input guards, upsell handling and the earlier Send anyway fix.
Ruff and whitespace checks passed. Test canvases contain generic controls and
geometry only; full profile screenshots remain private under ignored `data/`.
Restart the hub to load the updated driver. The phone was left on the restored,
unsent draft.
