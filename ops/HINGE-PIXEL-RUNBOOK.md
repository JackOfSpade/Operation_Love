# Hinge Pixel 7a Setup Runbook

Turnkey setup for bringing a brand-new physical Pixel 7a online for Hinge
automation with the lowest-risk architecture selected in
[ANTI-BOT-RESEARCH.md](./ANTI-BOT-RESEARCH.md). This is not "safe" or compliant:
the research record says Match/Hinge automation remains bannable by policy and
shadowbans can be silent and lagging.

Fixed setup decisions:
- Hardware: physical Pixel 7a, 100% stock. No root. No bootloader unlock.
- Control path: host-side `scrcpy` + ADB only. No on-device automation helper.
- SIM: Tello prepaid eSIM on T-Mobile MVNO, real US number, about $5/mo no-data.
- Network: home Wi-Fi residential IP for all account creation and usage. No VPN.
- Account: dedicated burner Google account plus burner Hinge account.
- Host: macOS on Apple Silicon.

---

## 1. Device first-boot

Research basis: Hinge/Match's first hard gate is Google Play Integrity. A
physical, stock, unrooted, Play-certified phone can pass
`MEETS_DEVICE_INTEGRITY`; an emulator cannot. See
[ANTI-BOT-RESEARCH.md, "The two hard gates"](./ANTI-BOT-RESEARCH.md#the-two-hard-gates).

- [ ] Unbox and boot the Pixel 7a.
- [ ] Complete factory-fresh Android setup as a new device.
- [ ] When Android asks for network during setup, join home Wi-Fi. Do not use a
      VPN. Section 3 makes this network rule explicit for Hinge usage.
- [ ] Decline restore-from-backup / device-copy prompts. Use a clean device
      fingerprint for this burner setup.
- [ ] Sign in only with the dedicated burner Google account.
- [ ] Finish setup to the stock Pixel home screen.
- [ ] Leave the phone stock:
      - no root
      - no bootloader unlock
      - no custom ROM
- [ ] Keep the Pixel on current security patches (install OS/security updates as
      they arrive) and note the model's end-of-support date. This matters only if
      Hinge enforces `MEETS_STRONG_INTEGRITY` — STRONG needs a security patch
      within the last 12 months and decays once the phone leaves its update
      window; whether Hinge requires STRONG vs DEVICE is currently unconfirmed
      (verify per the research's re-check triggers).
- [ ] Verify on device: Play Store opens under the burner Google account.

Why this matters: `MEETS_DEVICE_INTEGRITY` depends on a genuine certified device
with a locked bootloader. Root or bootloader unlock crosses the integrity line
this setup is built around.

## 2. Tello eSIM activation

Research basis: Hinge/Match account identity includes phone checks; VoIP numbers
are called out as rejected/carrier-checked, and the project decision is Tello
prepaid eSIM as the real US number. See
[ANTI-BOT-RESEARCH.md, "Other vectors"](./ANTI-BOT-RESEARCH.md#other-vectors)
and
[ANTI-BOT-RESEARCH.md, "Decisions this research drove"](./ANTI-BOT-RESEARCH.md#4-decisions-this-research-drove).

- [ ] Activate the Tello prepaid eSIM on the Pixel 7a using Tello's current
      activation flow. Pixel/Tello labels can change; verify the exact path on
      device.
- [ ] If using Android Settings directly, start from:
      `Settings -> Network & internet -> SIMs -> Add SIM`
      and verify the remaining prompts on device.
- [ ] Confirm the activated Tello line shows a US phone number.
- [ ] Open the stock Messages app.
- [ ] From another phone, send a test SMS to the Tello number.
- [ ] Confirm the Pixel receives the SMS.
- [ ] Reply from the Pixel and confirm the other phone receives it.
- [ ] Record the Tello number in the project secrets/notes location you already
      use. Do not put it in git.

This SMS receive check is required before Hinge signup/login because the burner
Hinge account will verify through this Tello number.

## 3. Network

Research basis: the Hinge/Match network vector flags datacenter/VPN IPs as
penalized and residential/mobile as better. Vendor-specific claims are disputed,
so the durable rule here is residential home Wi-Fi and no VPN. See
[ANTI-BOT-RESEARCH.md, "Other vectors"](./ANTI-BOT-RESEARCH.md#other-vectors)
and
[ANTI-BOT-RESEARCH.md, "Vendor / biometric claims - DISPUTED"](./ANTI-BOT-RESEARCH.md#vendor--biometric-claims--disputed).

- [ ] Connect the Pixel to home Wi-Fi.
- [ ] Confirm the Pixel is not using a VPN app or Android VPN profile.
      Verify on device.
- [ ] Use home Wi-Fi for every Hinge-related step:
      - Play Store install
      - burner Hinge account creation/login
      - SMS verification
      - automation sessions
- [ ] Do not use VPN, proxy, cloud/datacenter network, or remote browser/device
      infrastructure for Hinge account creation or usage.

## 4. Enable ADB

Research basis: the viable path is isolated USB control with `scrcpy`/ADB and no
helper app. USB debugging is not root and does not unlock the bootloader; it is
the control line this setup allows. See
[ANTI-BOT-RESEARCH.md, "Viability matrix"](./ANTI-BOT-RESEARCH.md#viability-matrix-report-1).

- [ ] On the Pixel, enable Developer options:
      `Settings -> About phone -> Build number`
      then tap `Build number` seven times. Verify labels on device.
- [ ] Enable USB debugging:
      `Settings -> System -> Developer options -> USB debugging`
      and confirm the Android warning prompt.
- [ ] Connect the Pixel to the Mac with a USB-C data cable.
- [ ] Keep the Pixel unlocked.
- [ ] If `adb` is already available on the Mac, run:

```bash
adb devices
```

- [ ] When Android shows the "Allow USB debugging?" RSA prompt, allow this Mac.
- [ ] Confirm `adb devices` shows the Pixel as `device`, not `unauthorized`.

If `adb` is not available yet on a fresh Mac, complete section 6, then return to
the `adb devices` authorization check before installing or logging into Hinge.

Do not go beyond USB debugging. No root, no bootloader unlock, no accessibility
automation service, and no helper APK.

## 5. THE GUARDRAIL: no on-device automation server

Research basis: `uiautomator2` installs an on-device helper with
input-control/instrumentation and can trip Play Integrity
`appAccessRiskVerdict` as `UNKNOWN`/risk (`UNKNOWN_CONTROLLING` in the research
record), even in observe mode. Apps can also enumerate enabled accessibility
services. See
[ANTI-BOT-RESEARCH.md, "The two hard gates"](./ANTI-BOT-RESEARCH.md#the-two-hard-gates)
and
[ANTI-BOT-RESEARCH.md, "Viability matrix"](./ANTI-BOT-RESEARCH.md#viability-matrix-report-1).

- [ ] Do not install `uiautomator2`.
- [ ] Do not install `atx-agent`.
- [ ] Do not install Appium's Android helper APKs, including
      `io.appium.uiautomator2.server`.
- [ ] Do not install any on-device automation server.
- [ ] Do not enable any accessibility service for automation.
- [ ] Do not use an observe tool that reads the Android UI hierarchy through an
      on-device server.
- [ ] If any tool asks to install an APK/helper on the Pixel, stop.
- [ ] Use only host-side ADB + `scrcpy`:
      - screenshots via `adb exec-out screencap`
      - taps/text via `adb shell input` or `scrcpy`
      - UI state through screenshot vision/OCR, not the accessibility tree

This is the main account-protection guardrail. Stock physical hardware is not
enough if the automation footprint is moved onto the phone.

## 6. Install scrcpy on the Mac

Research basis: the selected Hinge path is physical Android plus external
`scrcpy`/ADB control with no installed helper app. See
[ANTI-BOT-RESEARCH.md, "Viability matrix"](./ANTI-BOT-RESEARCH.md#viability-matrix-report-1).

- [ ] On the Mac, install `scrcpy`:

```bash
brew install scrcpy
```

- [ ] Verify the tools are present:

```bash
scrcpy --version
adb version
```

- [ ] Connect and unlock the Pixel.
- [ ] Start/restart ADB:

```bash
adb kill-server
adb start-server
adb devices
```

- [ ] If the Pixel shows the RSA prompt, allow this Mac.
- [ ] Confirm `adb devices` shows one Pixel as `device`.
- [ ] Start mirroring:

```bash
scrcpy
```

- [ ] Verify the Pixel screen mirrors on the Mac.
- [ ] Close `scrcpy` with `Ctrl-C` in the terminal when done testing.

If Homebrew or the formula behavior differs on this Mac, verify on Mac and keep
the final state the same: `scrcpy` and `adb` installed locally, Pixel authorized,
no helper app installed on the Pixel.

## 7. Install Hinge from Play Store

Research basis: Hinge runs on the disposable burner because shadowban status can
look normal while likes are silently dropped, and Match can federate safety
signals across account/device/phone/payment/photo/face identifiers. See
[ANTI-BOT-RESEARCH.md, "Consequences (Hinge) - the strategic crux"](./ANTI-BOT-RESEARCH.md#consequences-hinge--the-strategic-crux)
and
[ANTI-BOT-RESEARCH.md, "Other vectors"](./ANTI-BOT-RESEARCH.md#other-vectors).

- [ ] Confirm the Pixel is on home Wi-Fi and no VPN is active.
- [ ] Open Play Store under the burner Google account.
- [ ] Install Hinge from the Play Store.
- [ ] Open Hinge on the Pixel.
- [ ] Create or log into only the burner Hinge account.
- [ ] Use the Tello number for Hinge SMS verification.
- [ ] Complete the SMS verification on the Pixel.
- [ ] Verify on device any Hinge prompts that may vary by app version.
- [ ] Do not log into the user's real Apple-keyed Hinge account.
- [ ] Do not link the burner account to the user's Apple ID, real Google account,
      real phone number, or personal payment method.

## 8. Verification checklist

Research basis: the viable control surface is ADB input plus screenshot capture,
with `scrcpy` for mirrored human supervision. See
[ANTI-BOT-RESEARCH.md, "Viability matrix"](./ANTI-BOT-RESEARCH.md#viability-matrix-report-1).

- [ ] `adb devices` shows the Pixel:

```bash
adb devices
```

Expected shape:

```text
List of devices attached
<pixel_serial>    device
```

If it says `unauthorized`, unlock the Pixel and accept the RSA prompt. If it is
missing, verify the USB cable is a data cable and USB debugging is still enabled.

- [ ] `scrcpy` mirrors the Pixel:

```bash
scrcpy
```

- [ ] ADB tap works. With the Pixel unlocked and a harmless target visible in
      the mirrored screen, run:

```bash
adb shell input tap 100 100
```

Verify on device/scrcpy that the tap lands. Coordinates are screen-dependent; if
`100 100` is not a visible target, choose a harmless coordinate from the mirrored
screen and rerun.

- [ ] ADB screenshot capture works:

```bash
adb exec-out screencap -p > /tmp/hinge-pixel-screencap.png
file /tmp/hinge-pixel-screencap.png
ls -lh /tmp/hinge-pixel-screencap.png
```

Expected: `file` reports PNG image data and `ls` shows a non-empty file.

## 9. What could burn the account

Research basis: the durable risks are Play Integrity failure, on-device
automation footprint, bad network/IP, identity linkage, behavior, and silent
shadowban. See
[ANTI-BOT-RESEARCH.md, "Hinge / Match Group"](./ANTI-BOT-RESEARCH.md#2-hinge--match-group-android-app-planned-automation)
and
[ANTI-BOT-RESEARCH.md, "Cross-cutting principles"](./ANTI-BOT-RESEARCH.md#3-cross-cutting-principles).

Do:
- [ ] Keep the Pixel 7a stock, locked, unrooted, and Play-certified.
- [ ] Keep all Hinge activity on home Wi-Fi with no VPN.
- [ ] Keep the burner Google account, Tello number, and burner Hinge account
      separate from the user's real account stack.
- [ ] Use host-side ADB + `scrcpy` only.
- [ ] Prefer observe/human-in-the-loop behavior where possible; autonomous
      behavior is the unresolved risk.
- [ ] Treat a normal-looking Hinge deck as inconclusive. Shadowban can be silent
      and lagging.

Do not:
- [ ] Do not use an emulator or AVD for Hinge.
- [ ] Do not root, unlock the bootloader, or flash a custom OS.
- [ ] Do not install `uiautomator2`, `atx-agent`, Appium helper APKs,
      accessibility automation, or any on-device automation server.
- [ ] Do not use VPN, proxy, datacenter IP, or cloud-hosted device/browser paths.
- [ ] Do not log into the user's real Apple-keyed Hinge account.
- [ ] Do not reuse a banned Match Group phone/email/payment/device identity.
- [ ] Do not use photos of a face that has ever appeared on a banned (or prior)
      Match Group account. Match federates a face-hash across its brands
      (Tinder/Hinge/OkCupid/Match), so a previously-banned face is recognized on
      this burner even with a new number, device, and IP; the "real photos, risk
      accepted" decision assumes the face has no Match-brand ban history.
- [ ] Do not assume Incognia, ThreatMetrix, or touch-biometric claims are proven;
      the research marks those vendor/biometric specifics as disputed. The robust
      guardrails are stock physical hardware, residential network, burner
      account, and no on-device helper.

## 10. Known blocking screen — Hinge's "out of free likes" paywall

Research basis: a 2026-08-11 observe run hung 2.5 minutes because Hinge
refused a Send Like (daily free likes exhausted) and put its Hinge+ upgrade
screen up in place of the deck, and nothing in the codebase recognized it. See
[ANTI-BOT-RESEARCH.md, "Addendum 2026-08-11 — Hinge's out-of-free-likes paywall, and the observe hang it exposed"](./ANTI-BOT-RESEARCH.md#4-decisions-this-research-drove).
This is vision-only detection (screencap + `cv2` template match + best-effort
`tesseract` OCR) — no accessibility tree, per §5's guardrail above; the
uiautomator dump used to measure the geometry below was a one-off read-only
calibration read, the same kind already used for Bumble, never a standing
capability.

Measured live on the Pixel 7a (1080x2400) 2026-08-11. Reference artifacts:
`ops/calibration/hinge_out_of_likes_20260811.png` (screenshot) and
`ops/calibration/hinge_out_of_likes_20260811_uiautomator.xml` (the one-off
read-only dump).

- [ ] **Template crop box** (`operation_love/drivers/assets/hinge_upgrade_tab.png`,
      210x76 grayscale): x 705..915, y 254..330 on a 1080x2400 frame — the
      fixed "HingeX" tab-bar wordmark, chosen because the tab bar is fixed
      chrome while the hero image and benefit list rotate/scroll underneath
      it. `cv2.TM_CCOEFF_NORMED` measured 1.000 on the live paywall, 0.965 to
      1.000 under gain/bias perturbation, and a maximum of 0.4903 over all 88
      real non-paywall frames of the hung run — `_PAYWALL_MATCH_THRESHOLD =
      0.75` sits with wide margin on both sides.
- [ ] **Position gate:** only accept a template match whose centre y is
      `<= 0.30 * height` (`_PAYWALL_MAX_Y_FRAC`) — the tab bar sits at y
      266..318 of 2400 (y_frac 0.111..0.133). Same defensive idiom as the
      like-sheet detector's existing y-gate.
- [ ] **Headline OCR band** (best-effort only, never load-bearing —
      `AndroidAppSpec.paywall_headline_band`): normalized `(0.0556, 0.1958,
      0.9537, 0.3000)`, i.e. px (60,470)-(1030,720) on 1080x2400, covering
      the wrapped headline "You're out of free likes for today."
- [ ] **OCR recipe for that band** — the driver's ordinary dark-text-on-chrome
      recipe fails here (measured: garbage output) because the headline is
      white text over a photograph. What works, measured identically at
      thresholds 180/200/215: grayscale crop, binarize (`pixel > threshold`),
      **invert** so it becomes black text on white, upscale roughly 2-3x, then
      `tesseract --psm 6`.
- [ ] These coordinates are pinned to one Hinge app build on one physical
      Pixel 7a. Re-measure (fresh reference screenshot + uiautomator dump,
      same one-off read-only method) after any Hinge app update, and
      especially after any redesign of the Hinge+ upgrade screen — see
      ANTI-BOT-RESEARCH.md §5's matching re-check trigger.
