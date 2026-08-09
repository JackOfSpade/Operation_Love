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
- Hinge's true enforcement aggressiveness (report 2's open question) — resolvable only
  empirically, and only via shadowban-aware testing on a disposable account.
