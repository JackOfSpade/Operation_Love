# Hinge Pixel 7a Host Setup Runbook

Host-side setup for authorized diagnostic use of a physical Pixel 7a. It covers
ADB, screen mirroring, and private capture handling; it does **not** authorize
dating-app automation, account provisioning, or evasion of platform policy or
enforcement. Stop if the account owner and the service's current terms do not
expressly permit the intended use.

Fixed setup decisions:
- Hardware: physical Pixel 7a, 100% stock. No root. No bootloader unlock.
- Control path: host-side `scrcpy` + ADB only. No on-device automation helper.
- Account and SIM provisioning: deliberately out of scope for this runbook.
- Network: a secure, account-owner-controlled connection that complies with the
  service and network policies.
- Host: macOS on Apple Silicon.

---

## 1. Device first-boot

Research basis: Hinge/Match's first hard gate is Google Play Integrity. A
physical, stock, unrooted, Play-certified phone can pass
`MEETS_DEVICE_INTEGRITY`; an emulator cannot. See
[ANTI-BOT-RESEARCH.md, "The two hard gates"](./ANTI-BOT-RESEARCH.md#the-two-hard-gates).

- [ ] Unbox and boot the Pixel 7a.
- [ ] Complete Android setup using the account owner's approved device and
      account policy.
- [ ] Connect through a secure network authorized by its owner.
- [ ] Sign in only when the Android account holder has authorized it.
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
- [ ] Verify on device: Play Store opens under the authorized Android account.

Why this matters: `MEETS_DEVICE_INTEGRITY` depends on a genuine certified device
with a locked bootloader. Root or bootloader unlock crosses the integrity line
this setup is built around.

## 2. Account and SIM provisioning

This runbook intentionally provides no account-creation, phone-number, or
identity-separation procedure. Do not use a new account, number, device, or
network to bypass a restriction, suspension, or other enforcement action. Where
the intended work is authorized, follow the service's current account and
verification process directly and keep credentials out of the repository.

## 3. Network

Use a secure network that the account owner is authorized to use. This runbook
does not prescribe network characteristics as a way to influence a service's
fraud, safety, or enforcement systems.

- [ ] Connect the Pixel through a stable, authorized network.
- [ ] Follow the service's and network owner's security requirements.
- [ ] Do not use a network, proxy, remote device, or identity change to bypass
      a restriction or enforcement action.

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

Install Hinge only when the service's current terms and the account owner
expressly permit the intended use. This repository is not a guide to creating,
recovering, or separating dating-app accounts.

- [ ] Confirm the intended use is authorized before proceeding.
- [ ] Open Play Store under the authorized Android account.
- [ ] Install Hinge from the Play Store.
- [ ] Open Hinge on the Pixel.
- [ ] Log in only with an account the owner is authorized to use for this work.
- [ ] Follow the service's normal verification process; never use account,
      payment, phone, device, network, or identity changes to evade enforcement.

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

## 9. Authorization and account safety

Do:
- [ ] Confirm current service terms and account-owner approval before each use.
- [ ] Stop when the service signals a restriction, suspension, paywall, or other
      state that the approved workflow does not cover.
- [ ] Keep the Pixel 7a stock, locked, unrooted, and Play-certified.
- [ ] Use host-side ADB + `scrcpy` only.

Do not:
- [ ] Do not automate a service or account without explicit authorization.
- [ ] Do not create, modify, or combine accounts, identities, phone numbers,
      payment methods, networks, devices, or photos to evade enforcement.
- [ ] Do not use an emulator or AVD for Hinge.
- [ ] Do not root, unlock the bootloader, or flash a custom OS.
- [ ] Do not install `uiautomator2`, `atx-agent`, Appium helper APKs,
      accessibility automation, or any on-device automation server.
- [ ] Do not assume Incognia, ThreatMetrix, or touch-biometric claims are proven;
      the research marks those vendor/biometric specifics as disputed.

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
