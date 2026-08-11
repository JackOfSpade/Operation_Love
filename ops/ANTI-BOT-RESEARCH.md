# Anti-Bot & Detection Research — Bumble & Hinge / Match Group

**Purpose.** Durable record of what we learned (deep research, mid-2026) about how Bumble
and Hinge/Match Group detect automation — so design decisions aren't re-litigated and
future work inherits the reasoning. **This file is the source of truth.** The assistant's
memory is mutable and only points here.

**Reality check.** None of this makes automating these apps *compliant* — both prohibit
automation in their Terms, and every option below is "lower risk," never "safe." The goal
of the research was to pick the least-bad viable path and to understand the failure modes,
especially the silent ones.

**Confidence taxonomy** (used throughout):
- **HIGH** — official docs / ToS / vendor specs / reproducible behavior.
- **MEDIUM** — credible security research or analyst data, not app-specific confirmation.
- **LOW** — community anecdote, reverse-engineering claims, vendor-stack guesses.

**Provenance.** Two independent deep-research passes per app. Where they disagreed, both
positions are recorded and the disagreement is flagged — do **not** treat the more
confident report as settled.

---

## 1. Bumble (web app, automated via Playwright/Chromium)

We drive the real web client at `bumble.com/app` with a real browser over CDP. We do **not**
forge the signed API requests (`badoo.bma.BadooMessage` over `/mwebapi.phtml`) — the genuine
client constructs them. So the surface is **browser-automation + behavior + account**, not
API-signature forgery.

### Detection vectors

| Vector | What | Confidence | Our mitigation |
|---|---|---|---|
| CDP `Runtime.enable` leak | Vanilla Playwright triggers `Runtime.consoleAPICalled` side-effects, detectable by client scripts. The specific serialization trick was patched in Chromium ~May 2025 → weakened but real. | HIGH (mechanism); Bumble-uses-it = LOW | Switched to **patchright** (routes eval through isolated worlds) |
| Injected page artifacts | `window.__pwInitScripts`, custom globals, and — our own mistake — **injected DOM** (status HUD / busy modal) are page-visible to any script | HIGH | We don't use `add_init_script`; **disabled the in-page HUD/busy by default** (`inpage_overlays:false`) |
| `navigator.webdriver` | `true` by default | HIGH | `--disable-blink-features=AutomationControlled` + `ignore_default_args=["--enable-automation"]` |
| Bundled vs real browser | Bundled Chromium → SwiftShader render + missing `window.chrome` tells | HIGH | `channel="chrome"` (real Google Chrome) |
| Web fingerprint | canvas / WebGL / AudioContext / fonts / JA3-JA4 TLS — coherent on a real headful Chrome; spoofing creates mismatches | HIGH (general); which Bumble uses = LOW | Real Chrome, no spoofing; keep locale/timezone/geo aligned with IP |
| Behavioral | cursor path (Bézier vs zero-time teleport), swipe-cadence regularity, **right-swipe ratio (>70% bot-like; ~20–40% human)**, volume (~25 right-swipes/day on free), session timing | HIGH that behavior is used; exact thresholds = LOW (report 2: no public "safe" number) | human-cursor (Bézier + coord jitter), log-normal cadence, lowered caps + per-run like budget |
| Network / IP | datacenter/VPN penalized; residential trusted | HIGH | run on home residential IP |
| Account / identity | phone (VoIP rejected), photo + selfie verification (FaceVector), device/cookie reuse, new-account-immediate-activity heuristics | HIGH (documented identifiers) | account hygiene |
| Cross-app | **Bumble Inc. (Badoo) ban federation** — a block can propagate across Bumble Group via shared email/phone/IP/device. **Separate from Match Group.** | HIGH (Bumble privacy policy) | aware; Bumble ≠ Match |

### Vendor stack — DISPUTED
- **Report 1 (confident):** Arkose Labs (MatchKey CAPTCHA) + proprietary "Deception Detector" AI + "Gelato" JS error-tracking, stated as fact.
- **Report 2 (skeptical):** found **no public evidence** of any named vendor in the *authenticated* app; only Bumble's own documented "Deception Detector" / anti-spam team is confirmed.
- **Verdict:** treat "Bumble uses Arkose/DataDome/etc." as **UNVERIFIED**. Bumble's internal fraud/behavioral system is documented; its third-party vendor stack is not.

### Consequences (Bumble)
- **Temporary restriction** (documented): error codes `0030 0430 00XX` / `0030-0403-00XX`, guidance to wait 24–48h.
- **Verification interlock** (documented): CAPTCHA or live-selfie gate.
- **Hard ban** (documented): logout + ToS-violation notice; phone/email/payment/device blacklisted; retention 6 years (serious cases up to 15).
- **Shadowban / reduced reach:** widely *reported* but **not a publicly confirmed formal tier** (LOW).
- Appeals: low success.

### Risk by mode
- **Observe** (you swipe, we read): **Low–moderate.** Residual = CDP attachment + any injected page code. With patchright + overlays-off, near-zero added footprint.
- **Auto** (we click): **Moderate.** Behavioral correlation is the unresolved risk; report 2 advises against auto-matching at all.
- **Burner for Bumble?** Report 2 argues a burner is **not** protective (one-account/phone linkage, multi-account enforcement) — the lever is "don't auto-match," not "use a burner." (Contrast Hinge, where we *do* use a burner.)

---

## 2. Hinge / Match Group (Android app, planned automation)

Here the **architecture itself** was the problem, not tunable hardening.

### The two hard gates
1. **Device integrity — Google Play Integrity API.** Match apps use it server-side. An
   **Android emulator (AVD) fails `MEETS_DEVICE_INTEGRITY`** — no hardware root-of-trust /
   locked-bootloader proof. A **physical, stock, unrooted, Play-certified phone passes
   natively** (Pixel 6a+ via Titan M2; also passes STRONG with patches <12 months).
   *(HIGH — Google docs + both reports.)*
   - `MEETS_DEVICE_INTEGRITY` = genuine certified device + locked bootloader.
   - `MEETS_STRONG_INTEGRITY` = DEVICE + a security patch within the last 12 months
     (enforced since ~May 2025). STRONG decays once a phone leaves its update window;
     DEVICE can persist a while longer.
2. **Automation footprint — Play Integrity `appAccessRiskVerdict`.** uiautomator2 installs an
   on-device helper (`io.appium.uiautomator2.server`) with input-control/instrumentation →
   flagged **`UNKNOWN_CONTROLLING`**, **even in OBSERVE mode** (reading the UI hierarchy runs
   the server). Apps can also enumerate enabled accessibility services
   (`getEnabledAccessibilityServiceList`). *(Documented Google API HIGH; whether Hinge
   consumes the verdict = MEDIUM.)*

### Viability matrix (report 1)

| Mode | Hardware | Control | Verdict |
|---|---|---|---|
| Observe / Auto | **Emulator (AVD)** | uiautomator2 | **CRITICAL / UNFEASIBLE** (fails integrity + `UNKNOWN_CONTROLLING`) |
| Observe / Auto | Physical (stock) | on-device helper (uiautomator2/Appium) | POOR (helper APK flagged) |
| **Observe** | **Physical (stock)** | **Isolated USB control — scrcpy/ADB, no helper app** | **LOW / VIABLE** |
| **Auto** | **Physical (stock)** | **scrcpy/ADB** | **MEDIUM / VIABLE** |

→ The only viable path = **physical Android + scrcpy/ADB external control** (no installed
helper, no accessibility service): capture via `adb exec-out screencap`, taps via
`adb input`/scrcpy, opener text via scrcpy clipboard-paste, and UI state read by **vision**
(template/OCR on the screenshot) rather than the accessibility tree.

### Other vectors
- **Behavioral** (HIGH used / thresholds LOW): timing variance, non-linear swipe paths,
  like:pass ~20–45% (>85% or exactly 50% = bot), scroll depth, match:message ratio, session
  windows ~5–20 min.
- **Network / IP** (HIGH): datacenter/VPN penalized; residential/mobile good; JA3/JA4;
  **Incognia** BSSID + GPS vs IP-geolocation consistency.
- **Account / identity**: phone (VoIP rejected; carrier DB checked), GAID, **Widevine/DRM
  device-id survives factory reset**, Firebase/Play install tokens. See §2.1 for what a
  factory reset does and does not rotate.
- **Biometric**: **face-hash federation across Match Group** (Tinder/Hinge/OkCupid/Match) — a
  banned face is recognized on a new account even with new number/device/IP.
- **Cross-brand ban federation** (from privacy policy / litigation): Match's centralized
  safety repository shares device/photo/face/phone/payment across brands; bans cascade and
  persist.

### 2.1 Device-identity reset — what a factory reset buys (added 2026-08-07)

Question that prompted this: *can the Pixel be wiped so Hinge sees a new phone?* Short
answer: **no.** A factory reset rotates the software identifiers and leaves the hardware
identity — the part Match can actually pin you with — completely intact.

| Identifier | Factory reset | Source / confidence |
|---|---|---|
| `ANDROID_ID` / SSAID | **Rotates** | Android O blog: *"only changes if the device is factory reset or if the signing key rotates"* — HIGH |
| Advertising ID (GAID) | **Rotates** (also resettable anytime, no wipe needed) | HIGH |
| Firebase install ID / GSF ID / Play install tokens | **New** | HIGH |
| App data, cookies, accounts | **Gone** | HIGH |
| Widevine / MediaDrm per-app ID | **Persists** | Per-APK scoped since O, but derived from the factory-provisioned keybox — HIGH |
| Hardware attestation key (StrongBox / Titan M2) | **Permanent** | HIGH |
| IMEI / serial | **Permanent**, but unreadable by third-party apps since Android 10 | HIGH — effectively moot for Hinge |
| **Play Integrity Device Recall** | **Survives by design** | See below — HIGH mechanism / MEDIUM that Hinge uses it |

**Play Integrity "Device Recall" — the feature that closes this door.** Google ships an API
whose stated purpose is defeating exactly this manoeuvre. Three custom bits stored **on
Google's servers**, keyed to the device: *"allowing your app to reliably recall your custom
data even after your app is reinstalled or the device is reset."* Google's own example use
is *"has or hasn't been known for high severity abuse."* Bits persist 3 years after last
access. Beta, gated behind an interest form + Play Console opt-in, so adoption is not
publicly enumerable — but Match Group is precisely the applicant profile that gets approved.
**Assume a reset may be worth nothing against Hinge, not merely imperfect.**
<https://developer.android.com/google/play/integrity/device-recall>

**Do not escalate to bootloader unlock / root / ID spoofing.** The physical-Pixel path exists
*only* because the device passes §2 gate 1. Unlocking trades a rotatable soft identifier for
a permanent hard integrity failure — strictly worse. It also means Pixel factory-image
flashing is unavailable (it requires an unlocked bootloader); the only integrity-preserving
wipe is the plain Settings/recovery factory reset, which keeps the bootloader locked.

**Consequence for strategy.** Device IDs were never the binding constraint. Per the vectors
above, the durable linkers are **face-hash federation**, **phone number**, **payment
instrument**, and **IP** — a wipe touches none of them. Rotating hardware identity in any
real sense requires *different physical hardware*, and even that does not clear Recall bits
or a face hash. Budget effort accordingly: account/face/number hygiene ≫ device-ID hygiene.

### Vendor / biometric claims — DISPUTED
- **Report 1 (confident):** Incognia + ThreatMetrix SDKs embedded in Hinge; `MotionEvent`
  touch-biometrics (pressure/size distinguish synthetic taps).
- **Report 2 (skeptical):** **no public teardown names a specific SDK**; **synthetic input
  events are NOT distinguishable** — Android does not tag injected taps, so detection relies
  on environment + behavior, not event metadata.
- **Verdict:** treat Incognia / ThreatMetrix / touch-biometrics as **UNPROVEN**. The robust
  shared core is: *the emulator is the weakest link*, and *an on-device automation helper is
  a real signal*.

### Consequences (Hinge) — the strategic crux
- **Hard ban** (documented ToS): logout + permanent blacklist of phone/email/payment/device;
  cross-brand; low appeal success.
- **Shadowban — the killer:** silent. You log in, swipe, send likes "normally," but your
  profile is hidden from others' feeds and outgoing likes are quietly dropped. **You cannot
  confirm it until weeks/months of no matches.** Partial faster probe: on Hinge a **billing
  error when buying premium/roses/boosts** ≈ shadowban indicator (MEDIUM).
- **Implication (the decisive one):** there is **no way to make autonomous swiping on an
  account you care about both effective and *confirmably* safe.** "Just test it and see if it
  gets banned" is near-worthless for shadowban — it gives a *false green* (deck loads, looks
  fine, account silently demoted). Emulator-vs-physical changes the *probability* of getting
  flagged, never the *confirmability*. This is why Hinge runs on a **disposable burner**, and
  why seeding (observe) is unaffected by shadowban (the deck still loads) while autonomous
  *matching* is the gamble.

---

## 3. Cross-cutting principles
- **Lowest-risk ≠ safe.** Everything here violates ToS and is bannable by policy regardless
  of evasion quality.
- **Observe ≪ Auto.** Observe (human acts, we only read) removes the behavioral evidence that
  Auto generates.
- **Shadowban is silent + lagging + unconfirmable** — the central reason autonomous use of a
  *valued* account is a gamble; only sane on a disposable account.
- **Two reports per app; they disagree on specifics.** Vendor names (Arkose/Incognia/
  ThreatMetrix) and touch-biometrics are the least-supported claims. Don't over-trust the
  confident report.
- **Detection tricks are fragile/short-lived** (e.g., the CDP `Runtime.enable` serialization
  trick was patched out ~May 2025). Re-verify before relying on any single signal.

## 4. Decisions this research drove
- **Bumble:** 6 hardening changes — patchright (no CDP leak), no DOM injection
  (`inpage_overlays:false`), `navigator.webdriver` flags, real Chrome (`channel="chrome"`),
  human-cursor clicks, lowered caps + per-run like budget. See
  `operation_love/drivers/bumble.py`, `operation_love/limits.py`, `config.yaml`. Run
  observe-first; no burner.
- **Hinge:** abandon emulator + uiautomator2; commit to **physical Pixel 7a + scrcpy/ADB +
  vision driver**; **burner** account; **real photos** (risk accepted); home Wi-Fi
  (residential IP) + Tello prepaid eSIM (real US number). Driver rewrite pending.
- **Platform standard:** Android for all phone automation; iOS is structurally unsuitable
  (no ADB; the only routes — Appium/WebDriverAgent helper or jailbreak — reintroduce the
  footprint/flags we engineered away).

### Addendum 2026-08-09 — fixed caps removed (supersedes the cap decisions above)

The lines above are kept as the historical record; this addendum states what actually ships
now. The **default volume caps were removed** — `config.yaml` is `limits: {}`, so
`max_per_run`, `max_per_day`, `max_likes_per_run` and `target_like_ratio` are all unset.
The `RateLimiter` mechanism is retained (every field optional) for a deliberate temporary
ceiling, e.g. a supervised run; nothing sets one by default.

**Rationale (owner decision).** A fixed numeric ceiling is *itself* a bot signature: hitting
the identical wall run after run is a hard step function under distribution analysis, which
no human session boundary produces. Same reasoning retired the fixed-probability session
micro-break (a flat 8%/swipe roll yields an exactly geometric gap distribution) in favor of a
fatigue hazard that ramps with actions-since-break and re-rolls its own rate each stretch.
Volume is now governed by the ranker's decisions plus the human-timing model; auto mode runs
until the deck is exhausted, it hits an unrecognized screen state or error, or it is stopped.

**Accepted risk, stated explicitly.** §1's behavioral row rates right-swipe ratio a
HIGH-confidence signal (">70% bot-like; ~20–40% human"), and `target_like_ratio: 0.35` was
its mitigation. It was **not** reinstated. The argument for accepting this: the ranker trains
on the owner's own observe-mode labels, so its natural like rate should converge on the
owner's real manual like rate — human by construction — whereas an artificial target would
pull it *away* from that and force passes on profiles the model wants. The residual risk is
that this holds only while the model is well-calibrated; the like rate is now an emergent
property, not a governed one.

**Open item:** the ranker's realized like rate is unmeasured. It is the quantity to watch if
detection behavior ever looks off — see the §5 trigger below.

### Addendum 2026-08-10 — Bumble SuperSwipe measured live: no confirmation sheet is a safety net

A prior pass (see `operation_love/drivers/hinge.py`'s `_decide_by_card_swipe` and
`operation_love/drivers/android/bumble.py`'s old `forbidden_zones` comment) claimed there is
"no confirmation modal" after a Bumble SuperSwipe. That claim was wrong, but the corrected
version is more interesting than "actually there is one" — the ground truth, measured on the
real Pixel 7a (1080x2400) 2026-08-10 via a read-only screencap + uiautomator dump (no `adb`
write command issued), is a two-state model:

- **Balance == 0** — attempting a SuperSwipe opens
  `com.badoo.mobile.payments.flow.bumble.BumblePaymentFlowActivity`, a purchase/confirmation
  sheet. Its "Get 30 SuperSwipes for $39.99" CTA occupies x 0.049-0.950, y 0.899-0.951 —
  nearly full width, and (measured against `BUMBLE_SPEC.forbidden_zones`,
  `(0.34, 0.80, 0.66, 1.00)`) both `like_heart` (0.850, 0.900) and `pass_x` (0.150, 0.900)
  land ON that CTA while sitting OUTSIDE the forbidden zone. A "Close sheet" dimmed overlay
  covers x 0.000-1.000, y 0.000-0.394.
- **Balance > 0** (5, on the owner's account at measurement time) — the SuperSwipe is spent
  SILENTLY. No sheet, no prompt, nothing on screen to catch it.

So "there's a confirmation sheet" is true only on the zero-balance path, and even there the
sheet is a **hazard** (its CTA overlaps the deck's own like/pass coordinates), not a
safeguard. On the non-zero-balance path there is no safety net of any kind — the only thing
standing between a mistaken tap and an actual, irreversible, paid action is code that never
aims at the SuperSwipe control in the first place, and code that refuses to act at all unless
the ordinary swipe deck is positively confirmed on screen.

**What shipped in response** (`operation_love/drivers/hinge.py`,
`operation_love/drivers/android_spec.py`, `operation_love/drivers/android/bumble.py`):

1. `AndroidDriver._require_deck_confirmed()` — a new PRE-CONDITION on every autonomous decide
   gesture (tap or card_swipe), not just Bumble's: before issuing a like/pass, the driver must
   positively confirm the deck's own like+pass glyphs are visible, or it refuses
   (`UnconfirmedScreenError`) and the run halts with the screen untouched. This is what
   actually generalises where a static `forbidden_zones` rect cannot — the rect is
   screen-agnostic (it forbids a coordinate no matter what's on screen), while the CTA danger
   above is screen-dependent (the identical point is a harmless like on the deck and a $39.99
   purchase on the sheet). Widening `forbidden_zones` to also cover the CTA was rejected: it
   would forbid `like_heart`/`pass_x` on the ordinary deck too, since on the purchase sheet
   those are literally the same pixels.
2. `AndroidAppSpec.upsell_dismiss_zone` + `AndroidDriver._dismiss_via_zone()` — for a
   paid-upgrade sheet whose real dismiss control is a large "tap outside the sheet" overlay
   rather than a single button (Bumble's case), the driver detects the sheet via the existing
   `upsell_dismiss` template first (no detection, no tap, ever), then taps a FRESH RANDOM
   point inside a declared safe rect on each attempt (never a fixed coordinate — a repeated
   exact point is itself a bot signature), verifies the sheet actually cleared, and HALTS
   (`PaidUpsellStuckError`) rather than tapping again indefinitely if it's still up after 3
   attempts. `BUMBLE_SPEC` declares `upsell_dismiss_zone=(0.15, 0.10, 0.85, 0.34)`, derived
   from the measured overlay with margin for tap jitter, the status bar, and Android's edge
   back-gesture strips — see that field's comment for the arithmetic. It stays inert
   (`templates` has no `upsell_dismiss` entry yet) until a real template is captured live —
   see `ops/RUNBOOK.md`'s Bumble calibration checklist, which was rewritten to test both
   balance states explicitly instead of the old single "confirm nothing is purchased" step.

**Risk accepted going forward:** `_require_deck_confirmed` only fires for a spec that declares
BOTH a `like` and a `pass` glyph template — `BUMBLE_SPEC` currently declares neither (real,
placeholder state), so today this guard is a no-op for Bumble specifically, exactly like every
other vision-gated action in this driver with no template to check. It becomes load-bearing
the moment Bumble's `like`/`pass` templates are captured, which the calibration checklist
already requires before `calibrated` can be set at all. Until then, Bumble's only live
protections remain `decide_gesture="card_swipe"` + `forbidden_zones`, which is why
`calibrated=False` still gates the platform closed regardless.

### Addendum 2026-08-10 — Hinge observe mode recorded a PASS from manual scrolling alone

Owner bug report: in observe mode (the bot watches the phone and learns from the owner's own
like/pass taps), scrolling a profile up and down to read it — no pass/like tap at all — got
silently recorded as a PASS training label. This is a labeling-correctness bug, not a
detection-vector finding, but it shipped alongside a live-behavior change (`scroll_captures`)
covered by this file's LIVE-VERIFY discipline, so it's logged here per that convention.

**Root cause.** `HingeDriver.wait_for_decision` (`operation_love/drivers/hinge.py`)
distinguishes "still the same profile, just scrolled" from "advanced to a new profile" by
downsampling the screen to 24x24 grayscale and comparing it, at the SAME row alignment, against
the handful of frames `_capture_current()` captured automatically while reading the profile
top-to-bottom (`scroll_captures`, up to ~8-10 stops at `read_scroll_frac`, default 0.55,
policy-sampled up to 0.75). A human's own scroll distance essentially never lands on exactly
one of the bot's own stops, so it read as "whole card changed" -> PASS. The owner's own run
showed a second, compounding cause: 2 of the last 3 profiles captured exactly 8 photos (the old
`scroll_captures` ceiling) — the automated read hadn't reached the true bottom before the owner
started reading manually, so nothing captured could match what they scrolled into regardless of
alignment tolerance.

**What shipped:**
1. `_vertical_shift_match()` (`operation_love/drivers/hinge.py`) — before concluding PASS on an
   unmatched "top changed" frame, search vertical offsets (bounded by the same
   `_READ_SCROLL_FRAC_MIN/MAX` the read-scroll sampler is already validated against) for an
   aligned match against any captured frame, not just the exact row position. Recognizes a
   human's scroll landing BETWEEN two of the bot's own capture stops.
2. `capture_truncated` flag (`_capture_current`'s for/else) — set when the automated read
   exhausts its full capture ceiling without ever seeing the repeated frame that means "reached
   the true bottom." Surfaced in `Profile.meta` and the debug log. Monitoring only — it does
   not change what gets decided, because there is no way to distinguish "scrolled past what was
   captured" from "a real pass" once truncation has actually happened; both look identical
   (unmatched) to the perception layer. See risk accepted below.
3. `observe_decision` debug-log entry — `wait_for_decision` now logs `top`/`bot` deltas,
   `min_sig_dist`, `shift_matched`, and `capture_truncated` immediately before concluding PASS
   from pixels alone. The original bug report's `actions.jsonl` had no record of why the PASS
   was recorded; this closes that gap for the next report.
4. `config.yaml`'s `scroll_captures: 8 -> 12` — a live-behavior change, made only after
   confirming with the owner (this is a device-timing change, not a pure code-correctness fix).
   Reduces how often a profile gets truncated before the owner starts reading, directly
   addressing the compounding cause above. Same randomized dwell/scroll-fraction sampling as
   before; no new fixed timing signature introduced.

**Risk accepted going forward.**
- Truncation is narrowed, not eliminated: a profile with more scrollable content than 12
  captures still reach can still produce the original bug. `capture_truncated` makes this
  visible in the debug log (per-profile, in `Profile.meta` and `actions.jsonl`) rather than
  fixing it outright — deliberately, since actively re-scrolling mid-`wait_for_decision` to
  "catch up" would break observe mode's passivity contract (it must never touch the device on
  the owner's behalf) and refusing to conclude PASS at all when truncated would equally break
  detecting a REAL pass on a truncated profile (both cases look identical: unmatched).
- The shift-search widens the match window ~37x (up to 18 vertical offsets in each direction on
  a 24-row downsample) against the same `change_threshold=9.0` used for the original single
  full-frame comparison. At the extreme offsets the compared band shrinks to a quarter of the
  frame, which is mechanically more permissive per-comparison, and there are many more
  comparisons tried. The failure mode this could introduce is the mirror image of the bug it
  fixes: a genuinely different profile's photo coincidentally matching at some offset, silently
  swallowing a real pass and stalling the observe loop on a profile the owner already left. No
  live evidence of this yet; `change_threshold` was tuned for the original zero-shift
  comparison, not this one. See the §5 trigger below.

### Addendum 2026-08-10 (b) — the vertical-shift fix above cannot fire in production; identity-band anchor + read-only touch corroboration shipped instead

The `_vertical_shift_match` addendum immediately above was itself measured live on the Pixel 7a
and does not work as shipped. It shift-searches the **whole** downsampled frame, but Hinge's
status bar, sticky header, and bottom nav bar do **not** translate when content scrolls — only
the middle of the frame does. On real same-profile pairs (different scroll offsets of one
profile) the full-frame best-shift distance is **30.38-41.38** against `change_threshold=9.0` —
3-4x over threshold, so the helper never returns a match in production and the original
scrolling-reads-as-PASS bug is unfixed by it. Restricting the search to the content rows only
(excluding the fixed chrome) drops the same pairs to **3.34-5.41** — under threshold, as it
should be. (A fourth pair, frames 13 of 24 downsampled rows apart, still misses even
content-rows-only, which is exactly why an identity anchor rather than a wider shift search is
the real fix — see below.)

**The real anchor.** Hinge renders a sticky top bar containing the person's name the moment a
profile is scrolled at all; at scroll-top the same band instead shows the app's own
filter-chips row, which is identical on every profile. Measured live: mean-abs-diff (0..255)
across three real frames of the SAME profile at three different scroll offsets is **0.00** —
pixel-identical — and **17.95** between a scrolled frame and that same profile's scroll-top
frame, against the same `change_threshold=9.0`. Separation is 0.00 vs 17.95 across a threshold
of 9.0: unambiguous, and — unlike the shift-searched content — true regardless of how far the
profile was scrolled. This identity band, not any form of frame-shift matching, is what now
gates the PASS decision: a frame whose identity band matches the captured profile's identity
band is the same profile no matter what else changed on screen, full stop, never a decision.

**Read-only touch-stream corroboration.** Alongside the identity anchor, observe mode now reads
the device's own touch event stream, read-only, via a persistent host-side `adb shell getevent`
subprocess held open for the length of an observe session — no input injection, no on-device
agent or helper APK, and no accessibility/uiautomator connection (that stays forbidden per
`ops/HINGE-PIXEL-RUNBOOK.md` §5). Verified working unrooted on the Pixel 7a: the real
touchscreen is `goodix_ts0` on `/dev/input/event3`; `/dev/input/event*` is `root:input
crw-rw----` and the adb `shell` user can read it without root. Risk accepted: a persistent
`adb shell` subprocess for the whole observe session is a larger, longer-lived footprint than
the one-shot `adb` calls used everywhere else in this driver. To bound that risk, the touch
stream is corroboration only — it can veto a label (routing a card advance to a resync instead
of a PASS) but it is never by itself a standalone decision source, and it self-disables (falls
back to identity-only proof, with a one-time warning) if it sees zero events for the whole run,
so a device or transport where `getevent` silently fails does not silently mislabel everything
as a resync.

**What is now refused rather than guessed.** Previously any card advance with no positive
decide-gesture evidence was recorded as a PASS. Now a card advance with no decide-gesture
evidence returns `None` — a resync, recapture with no label written — instead of being guessed
as a PASS.

### Addendum 2026-08-10 (c) — the read-only touch stream does not work on this phone; shipped OFF

Correction to addendum (b) above, which described gesture corroboration as live. It is not, on
this device. Measured on the Pixel 7a running **Android 17 (SDK 37, 2026-06-05 security
patch)**, with the owner deliberately tapping and scrolling the screen for 30s:

- the adb `shell` user **is** in group `1004(input)`;
- `/dev/input/event3` is `crw-rw---- root input u:object_r:input_device:s0`, SELinux Enforcing;
- `getevent -p /dev/input/event3` succeeds and returns the touchscreen's full capability set,
  so the node genuinely opens and ioctls work;
- and `getevent -lt /dev/input/event3` delivered **zero lines** — not unparsed lines, zero
  lines (the watcher now counts raw lines separately from parsed events precisely so those two
  can never be confused again; see `TouchWatcher.raw_line_count` and `tools/touch_selftest.py`).

So Android withholds the input event *stream* from unprivileged readers on this build even
though it grants capability queries. Recording input remains a rooted-device capability. No
code change recovers this, and the earlier reasoning in (b) — which assumed the successful
`getevent -p` probe implied a usable stream — was wrong to treat attach success as proof of
delivery.

**What shipped:** `HINGE_SPEC.observe_touch_watch` and `config.yaml`'s `apps.hinge.
observe_touch_watch` are both **False**. The module, its 20 tests, and `tools/touch_selftest.py`
are kept rather than deleted: the code is correct and costs nothing while off, and the
capability returns the moment this runs rooted or on a platform that permits the read.

**Risk accepted.** Observe mode now rests on the identity anchor plus deck-ready/settle proof
alone. That is what actually fixes the reported bug (a human scroll recorded as a PASS), and it
is verified against real device frames. What is given up is the narrower set of cases only a
real tap could disambiguate, and these are now unmitigated:
- a **rewind-arrow / bottom-nav tap** that changes the card without being a like or pass — the
  `observe_ignore_zones` rects exist but nothing reads them without a touch stream, so such an
  advance is recorded as a PASS rather than routed to a resync;
- the **same-first-name collision**: the identity band holds only a first name, so two adjacent
  profiles sharing one render an identical header. The "two pass-taps in one wait ⇒ resync"
  guard added for exactly this depends on the touch stream and is therefore inert.
Both were previously covered by layer 3 and are now paper risks with no live evidence either
way, which is the honest state to log rather than to claim the layer is protecting anything.

### Addendum 2026-08-10 (d) — Stop now abandons a profile read mid-way; a stopped session can end mid-scroll

Reported by the owner: "when I hit stop, it doesn't stop while it's reading a profile, it
completes the read (by scrolling a bunch) then stops." Measured against the real code with the
shipped config (`scroll_captures: 12`, `dwell_s: 1.1`, screencap ~0.6s, one humanized UHID
gesture ~2.1s): a Stop pressed mid-read was not observed for **~74s**, because observe mode had
exactly one stop check per profile — in `worker.py`, on the line *after* `current_profile()`
returned — and everything before it (`_capture_current`'s 12 screencaps + 11 read-scrolls, then
`_scroll_to_top`'s 11 undo-swipes) was stop-deaf. The same harness after the fix: **worst 2.0s,
mean 0.8s**, bounded by the one ADB call already in flight.

**What shipped.** `should_stop` is threaded into `next_profile`/`current_profile` and polled at
the top of each read iteration, inside the read dwell, and before each undo-swipe. It is passed
only to drivers declaring `DatingAppDriver.supports_interruptible_capture` (True on the Android
family, deliberately False on Bumble web, whose capture waits are not stop-aware). A gesture
already handed to `adb shell hid` is never interrupted — the sleeps live in the device-side
script, and a half-played gesture would leave the virtual finger down.

**Behavior change worth logging here.** A stopped session can now end **mid-profile-scroll**
instead of always completing a full read and unwinding to the top:
- *Toward the app's view of us, this is neutral-to-better.* Always reading every profile to the
  bottom and then scrolling all the way back up before every stop is the more mechanical
  pattern; a person putting the phone down part-way through a profile is not.
- *`read_dwell_s_total` is credited only with time actually slept*, not the sampled dwell, so an
  abandoned read never claims Signals behavior #1 reading time it did not spend. No decision,
  label, or Signals credit is recorded for an abandoned profile at all.
- *The card is left where it is*, per the existing "a stop leaves the screen untouched for
  debugging" rule, and the scroll ledger is deliberately **not** reset (resetting it would make
  the driver claim it is back at the top when it is several screens down, which is how a later
  `like()` could tap a heart on the wrong item). The operator is told this in the stop line.

**The trap this opened, found by review before it shipped, and what closed it.** The driver's
*first* capture of a session assumes frame 0 is at scroll-top — that is what makes
`_identity_top_sig` the app's own filter-chips chrome rather than a real person's sticky
header. Nothing enforced it (`open_session` only foregrounds the app), so leaving the card
scrolled meant the next run seeded the anchor from **that person's header**, leaving
`_identity_sig`/`_identity_name` unset for the whole card. The first assessment written here
said that merely "falls back to the narrower layer-2 proof". **That was wrong, and the correct
answer is much worse:** with no `_identity_sig`, `_identity_of` compared against the top
signature alone and answered **`new`** for any frame that differed from it — including the
card's own true top. `new` is the one verdict that *skips layer 2 entirely*
(wait_for_decision's `identity_state != "new"` guard), so it went straight to the
deck-ready/settle check, and with `observe_touch_watch: false` (the shipped config, see
addendum (c)) layer 3 cannot veto. Reproduced against the real code: a human scrolling that
first card, tapping nothing, was recorded as a **PASS** — a wrong training label, the exact bug
class the whole observe redesign exists to eliminate.

Two fixes shipped, and the second is not specific to Stop at all:
- **`_ensure_session_top`** — once per session, before the first capture, the card is unwound to
  a *confirmed* scroll-top, restoring the invariant across sessions instead of hoping the
  previous one tidied up. Costs one downward swipe and two screencaps when already at the top
  (the settle check ends the loop on the first iteration); it refuses to run while a like sheet
  is open, so it cannot drag a sheet out from under the operator.
- **`_identity_of` no longer manufactures a `new`** from the top signature alone. With no
  per-profile header locked, the chrome can positively prove `top` and nothing else; anything
  else is now `unknown`, which falls through to layer 2 exactly as this method's own docstring
  always said it must ("there is nothing to compare against either way, so this must not be
  read as either 'same' or 'new'"). This was a **pre-existing latent false-PASS path**,
  independent of the Stop change: any profile short enough that its sticky header never appeared
  during capture had the same hole. It is the same asymmetry the OCR layer already enforces —
  a veto on `new`, never the power to create one.

**Risk accepted.** A card that never settles (an animated/video profile) can still leave the
session-start unwind unconfirmed. That is now bounded to the honest outcome: the anchor may be
unavailable for that card and decisions on it rest on layer 2, which is stated on the console
and recorded as a `session_top_unconfirmed` debug action rather than being inferred from a
missing field.

## 5. Re-check triggers
- **Realized auto-mode like rate drifting high** (added 2026-08-09) — with `target_like_ratio`
  unset, nothing holds the right-swipe ratio down. Measure it from the decision store; if it
  approaches the ">70% bot-like" band in §1, treat that as a model-calibration problem first,
  not a reason to reinstate an artificial cap.
- Chromium/Playwright detection changes (CDP tricks come and go) → re-validate the patchright
  approach.
- Play Integrity policy changes; and confirm whether Hinge enforces DEVICE vs STRONG.
- **Device Recall leaving beta / becoming default-on** (§2.1) — would harden the "wipe and
  start over" dead end into a permanent one. Re-check before any burner-rotation plan.
- Bumble's actual vendor stack (only confirmable via authenticated-app inspection).
- **Bumble SuperSwipe geometry drifting** (added 2026-08-10) — the measured CTA/overlay/zone
  coordinates in `BUMBLE_SPEC` (§4's 2026-08-10 addendum) are pinned to one app build on one
  device. Re-measure before ever flipping `calibrated=True`, and again after any Bumble app
  update once it's running live — a layout change could silently move the purchase CTA into
  `upsell_dismiss_zone`, or move `upsell_dismiss_zone` into range of a control it isn't meant
  to touch.
- Hinge's true enforcement aggressiveness (report 2's open question) — resolvable only
  empirically, and only via shadowban-aware testing on a disposable account.
- **Observe-mode decision quality after the 2026-08-10 scroll fix** — watch `capture_truncated`
  frequency in the debug log (still-frequent truncation at `scroll_captures=12` means the
  ceiling needs another look) and watch for the observe loop silently stalling on a profile
  already left (the shift-search false-negative risk in §4's 2026-08-10 addendum). Neither has
  live evidence yet; both are plausible on paper only.
- **Hinge identity-band + touch-watcher geometry and reliability (added 2026-08-10)** — the
  `identity_band`/`content_band`/`observe_ignore_zones` coordinates in `HINGE_SPEC` and the
  `goodix_ts0`/`/dev/input/event3` touch-device selection are pinned to one app build on one
  physical Pixel 7a, exactly like the Bumble SuperSwipe geometry above. Re-measure after any
  Hinge app update. Also watch the `observe_resync` rate in the debug log: a resync fires
  whenever a card advances without decide-gesture evidence (drag-only, tap in an ignore zone, or
  a dead touch watcher past its `event_count==0` health check) — a resync rate that climbs over
  time rather than settling near zero points at geometry drift or the touch watcher losing the
  device, not a real behavior change, and should be treated as a calibration problem first.
- **Gesture corroboration availability (added 2026-08-10)** — `observe_touch_watch` ships False
  because Android 17 withholds the input stream from the adb shell user (see §4's 2026-08-10 (c)
  addendum). Re-test with `python -m tools.touch_selftest` after any Android major-version
  update, or if the bot ever moves to a rooted device: if it prints TAP/DRAG lines while you
  touch the screen, flip `apps.hinge.observe_touch_watch: true` and the rewind-tap and
  same-first-name risks recorded in that addendum close again.
- **First capture of a session starting mid-scroll (added 2026-08-10)** — see §4's 2026-08-10 (d)
  addendum. `_ensure_session_top` is supposed to make this impossible; watch for
  `session_top_unconfirmed` records and for `identity_seen: false` on the FIRST `capture` record
  of a run. Either appearing regularly means the unwind is not reaching a settled top (an
  animated card, or a ceiling too low for how far a previous run scrolled) and the ceiling or
  the settle check needs a look — not a wider identity threshold, which would trade a missing
  anchor for a wrong one.
- **Same-first-name mislabels while layer 3 is unavailable (added 2026-08-10)** — with the touch
  stream off, two consecutive profiles sharing a first name render an identical sticky header
  and the second can read as "still the same card". Watch the observe debug log for a profile
  whose decision never lands (the loop waiting through a real pass), and for `capture_split`
  records. If this shows up live, the fix is a second identity dimension (e.g. including the
  verified badge / photo-count row in the band), not a wider threshold.
