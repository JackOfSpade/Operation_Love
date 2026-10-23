# Hinge 10.9.0 message warning — 2026-10-23

Run `c9b8c72f8615` (reported build `772b72b1`) stopped after a Training
Send Like attempt. `00076_unexpected_error.png` shows a message warning,
not a ready successor card. A read-only ADB screencap during this investigation
confirmed the same warning remained on the connected phone. No live send was
performed during the investigation.

The warning reads **“Are you sure you want to send this?”**, explains that the
recipient might find the language uncomfortable, and presents **Keep editing**
above **Send anyway**. This differs from the already-supported Rose upsell's
**Send Like anyway**. The existing Rose template does not match this warning.
The old post-send verifier treated the changed modal as possible progress;
Training's stronger semantic deck-advance check then refused to record a label.
The failed action must not be retrospectively labelled as delivered.

The private live capture is at
`data/hinge_debug/send_anyway_20261023/live.png`; the original failure capture
remains in the run's private debug folder. Neither full screenshot belongs in
Git. `tests/assets/hinge_message_warning.png` retains only generic warning
chrome (rows 1450–2300 of the 1080×2400 frame), with every other pixel replaced
by white. It contains no profile, account photo, name, or message content.

The shipped templates contain only the heading and two button labels.
Both original captures locate Send anyway at `(540, 2202)`. Runtime recognition
requires all three unique template matches at correlation ≥0.9, centred and
vertically ordered with the measured button separation. The tap follows the
matched label, with no coordinate fallback. This evidence covers the observed
English 10.9.0 rendering at 1080×2400; it does not establish other scales,
languages, themes, or versions.

An offline replay across all 76 screenshots retained from the failed run
matched exactly frames 74, 75, and 76, which show the warning after Send Like.
The preceding 73 frames did not match. The warning-specific tests also cover
shifted controls, the different Rose modal, missing labels, delayed appearance,
Stop, stuck warnings, and a paywall appearing after confirmation.

During a send the owner reviewed (a Training Like, or a reviewed comment sheet),
the post-send verifier can tap this confirmation once, respecting Stop at the
touch boundary. It then waits a bounded interval for the warning to clear and
runs the existing delivery checks, including Training's stable semantic
successor proof. Paywall rejection still takes precedence. Missing controls or a
warning that remains open halt the action without repeated confirmation taps.
Passive Observe performs no new input.

**AUTO never confirms it** (owner decision, 2026-10-23). In AUTO nobody has read
the opener Hinge flagged, so `_verify_like_landed` is called with
`confirm_message_warning=False`. It records the warning, taps nothing, leaves the
warning on screen and raises, which stops the run. The staged opener is filed as
`send_unverified`, because Send Like was tapped. Only `_like_comment_sheet` with a
Training decision hook, and the reviewed-sheet path, pass `True`.

`message_warning` debug records retain the warning screenshot and identify
`send_anyway_attempt`, `unreviewed_refused`, `unrecognised_controls`, or `still_open`; the existing
`like_attempt` and final `like` records continue to distinguish an attempted
send from a confirmed result. Existing bug reports already include this action
trail and screenshots, so no reporting-format change was needed.

Validation: 512 focused Hinge/Training/upsell checks (including 17 warning
cases) and 608 calibration/capture checks passed. Ruff and Git whitespace
checks passed. Runtime phone interaction was limited to the read-only capture;
the pending message was left unsent. Restart the hub to load the updated driver.
