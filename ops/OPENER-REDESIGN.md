# Opener redesign: substance, item selection, and targeting

> ## Current status — final handoff (2026-08-12)
>
> **The opener redesign is implemented.** Production now uses numbered, model-selected
> items and fail-closed targeting; the former anchored-opener production path is gone.
> The remaining operator task is a **real-device, held-out calibration** of the two
> targeting acceptance bounds. Until that evidence is supplied, this is intentionally
> conservative: AUTO stops before any targeted-like gesture and OBSERVE keeps targeted
> opener text off the hub while manual labels continue. No calibration values are shipped,
> inferred, or guessed in this repository; see `ops/RUNBOOK.md` for the required evidence
> and measurement procedure. The historical record below is preserved as context and may
> describe earlier incomplete states; the final addendum at the end is authoritative.

> **Current driver contract (2026-08-12).** `DatingAppDriver.like` is
> `like(opener=None, item_index=None, *, model_item_index=None, should_stop=None)`. `None` is
> deliberately distinct from capture-order index `0`; `anchored_opener` was removed
> and must fail loudly. Historical passages below that describe a repair callback or
> a default index of `0` are superseded by the final 2026-08-12 addenda.

> **Photo-only targeting amendment (2026-08-13).** The owner contract now permits the system
> to choose and tap photos only, never written prompt cards. Hinge's payload builder numbers
> only crops affirmatively classified as photos; WRITTEN and UNKNOWN crops remain readable,
> unnumbered context. Aspect ratio and a presumed count are not item-type evidence. Dense model item
> numbers retain a translation to the original page-heart
> ordinals, so skipping prompt hearts cannot shift a later photo's tap. The current 9.134.0
> corpus measures six 974x974 photos and three written prompt cards at 974x756 / 974x685; the
> resulting translation is `(1, 3, 4, 6, 8, 9)`. Calibration schema 3 binds this behavior as
> `hinge_photos_only_v1`. The policy additionally excludes a card when its source
> sightings show Hinge's pixel-stable upper-left mute control; the card and heart remain in the
> scroll index, so later photo ordinals do not shift. Historical passages below that say every photo or prompt is selectable
> describe the superseded policy, not the current product contract.

Designed 2026-08-11. This is the agreed design, written down so the reasoning survives.

Sequencing note up front: Part A (wording) works against today's pipeline and ships alone.
Part B (capture and targeting) is the real project. A does not depend on B.

## Status as of 2026-08-11

**Part A: IMPLEMENTED.** Suite 1245 → 1313 green. Shipped: the new `opener.style` block with the
one rule, the five moves marked non-binding and the three guardrails; `_SYSTEM` and both anchored
copies rewritten, including the retry hint (all three sites named in 1.1 root cause #1);
`_SCHEMA` reordered to `referenced, angle, referenced_index, opener` with `angle` as free text;
`_redundant_description_markers` and `_leading_ngram`; the entropy guard, inert under
`advisory=True` and budget-exempt in auto; `angle` persisted in both stores with migrations.

**Correction, 2026-08-11 (later the same day): the line above is wrong, and left unedited only so
this file keeps recording what actually shipped and when it changed.** "Inert under
`advisory=True`" was never a decided design property -- it was an unexamined carry-over from 3.6's
open question about a HARD-REJECTING guard (see the addendum at the end of 3.6 for the full
account), applied to the soft, `retry_hint`-based guard that actually shipped, without re-checking
whether the reasoning still held. It did not: the guard's regeneration draw sits on
`maybe_opener`'s success path, outside the `for attempt in range(...)` loop, so it was already
budget-exempt in EVERY mode, including advisory -- there was never an attempt for `advisory=True`
to lose by running the guard, so skipping it there bought nothing. What it cost was real: Hinge
observe mode is the canary for auto (the opener a human sees in observe must be byte-identical to
what auto would send), and a guard that fires in AUTO but not in observe makes the two diverge
exactly where it matters. A live dry run confirmed this concretely -- 5 openers requested with
`advisory=True` against one real profile came back with 4 sharing an identical leading phrase,
because the guard never looked. Fixed the same day: the guard now runs identically regardless of
`advisory`, with no branch on that flag anywhere in `_apply_entropy_guard`. See `service.py`'s
`_apply_entropy_guard` docstring (ADVISORY ASYMMETRY section) for the corrected reasoning in full.

Two deliberate deviations from the drafts above, both kept:

- `config.yaml` retains "Ground this opener in exactly ONE concrete detail", which 1.1 names as
  root cause #1, but defuses it with a rider ("then do not say that detail back to her. The
  detail is where your claim comes from, not what your message is made of"). Deleting it outright
  would have cost the specificity pressure that section 2 explicitly wants, since a generic opener
  is ranked worse than a descriptive one. `_SYSTEM` carries the equivalent via "profile-specific
  bid" plus the premise/point rule, so the two copies agree in direction.
- The optional `first_draft` A/B arm (3.4) and the schema-order A/B harness (7.3) are not built.
  Both are explicitly optional or deferred to the Measure step.

**Part B: NOT IMPLEMENTED.** One of its two blockers is now cleared and one is confirmed real.
The viewport question (5.4) is closed: no card comes near the viewport height, so no stitching is
needed. The heart-dedup question (5.5) is confirmed as a genuine problem, not a hypothetical one:
naive counting fabricated a phantom item on real data, and this doc's own proposed alternative
missed 2 of 9 items at every threshold tried. It must be re-validated under bot-driven closed-loop
scrolling before anything is built on it. Full measurements in 5.10.

**Addendum 2026-08-12: Part B's PERCEPTION layer is now built; its DEVICE layer is not. The line
above stays unedited, per this file's convention, but read it with this.** The heart-dedup blocker
it names is closed — by construction rather than by a better matcher — and four leaf modules now
exist, each validated offline against the gitignored `ops/calibration/` captures and each with its
own derivation and per-constant measurements in its module docstring:

| Module | What it answers | Recorded in |
|---|---|---|
| `operation_love/drivers/segment.py` | one frame -> ordered blocks, classed by heart | 5.4 addendum, 5.10 addendum |
| `operation_love/drivers/frameshift.py` | two frames -> how far the content moved, or a refusal | 5.10.1 addendum |
| `operation_love/drivers/item_index.py` | a run of frames -> the profile's items, heart ordinals, and 5.3's translation table | 5.3, two addenda |
| `operation_love/drivers/item_crops.py` | an index -> the numbered images actually sent, plus 5.6's reference signatures | 5.2, two addenda |

They are LEAF modules and nothing in `operation_love/` imports them yet: no driver, no worker, no
prompt path. Suite 1410 -> 1493 green, all 83 new tests on synthetic fixtures.

Still unbuilt, and each is a real gate rather than a formality:

- **The closed loop (5.5, 5.10.1).** `config.yaml`'s `read_scroll_frac` is still 0.55, and that is
  the cadence `item_index` REFUSES at the first pair ("moved +1299px, beyond the 900px window").
  Wired in today it would refuse every profile. The step has to be sized against locally measured
  card spacing, which is what `segment.py`'s block extents are for.
- **The scroll-top gate (5.5).** `at_scroll_top` is an argument, and `_scroll_top_evidence` only
  CONTRADICTS a false one — 21 of 21 caught on one capture, 9 of 10 on the other. The affirmative
  filter-chips confirmation stays a hard gate before capture and does not exist in the driver yet.
- **Counting navigation (5.5), post-tap verification (5.6), the pre-flight cross-check (5.8), the
  schema change and the observe inversion (5.7/5.9), anchor removal.** None started. `hinge.py`,
  `worker.py` and the opener path are untouched by all of the above.

**Addendum, later on 2026-08-12: the three bullets above are stale and the list they belong to is
kept unedited per this file's convention. Read them with this.** All three of the gates that
bulleted list names as unbuilt now exist, each validated offline against the same gitignored
captures and each with its derivation in its own module docstring:

| Module | What it answers | Recorded in |
|---|---|---|
| `operation_love/drivers/scroll_top.py` | is this frame at a profile's scroll top — confirmed / refuted / cannot tell | 5.5, addendum |
| `operation_love/drivers/scroll_step.py` | how far the enumeration pass may scroll next, sized against locally measured spacing | 5.5, addendum |
| `operation_love/drivers/item_nav.py` | where item N's heart is on screen right now, and the evidence for it | 5.5, addendum |

`item_nav` is the first of these that touches the device, and the only one: it captures frames and
scrolls through the driver's own humanized methods, while every decision it makes is still made by
a leaf module. 5.3's third blocker is closed with it — `ItemIndex.heart_ordinal_for` now raises on
an `at_scroll_top=False` index rather than answering with a confident int, so that assertion has
exactly one legitimate source. `config.yaml`'s `read_scroll_frac` is deliberately still 0.55, which
stays correct for the swipe-deck read path; the enumeration and navigation passes compute their own
step and never consult it. Still untouched by all of this: `hinge.py` (beyond `_band`'s additive
`size` argument), `worker.py`, the opener path, the tap itself and everything downstream of it.

**Addendum, later still on 2026-08-12: the RESPONSE half of 5.7 is now built, so "the opener path
is untouched" above no longer holds.** `_SCHEMA` is `item_index, referenced, angle,
item_description, opener`; `referenced_index` is deleted rather than renamed; `OpenerResult`,
`OpenerPick` and both stores carry the two new fields, with a column migration each. Suite 1601 ->
1620 green. `OpenerPick.index` now holds the model item index while `driver.like(item_index=...)`
still expects capture order — documented at every site it crosses, closed by 5.3's translation
table in the navigation workflow. Full account, including the three decisions that were not pure
transcription, in 5.7's addendum. Still unbuilt in the model-facing contract: the request payload
(numbered crops in place of scroll frames) and the observe inversion.

**A production defect found by that calibration, and fixed 2026-08-11.** The shipped `like` glyph
template was the wrong control. `hinge_heart.png` is Hinge's OUTLINE heart, which is the glyph in
the "Which do we have in common" rows, not the per-card like button (a white heart in a filled
black circle). Measured against real frames it matched those "common" rows at correlation 1.000
while the real like button was invisible to it, peaking at 0.544 on frame noise rather than on
either visible button. Consequences: `_locate_button("like")` and `_locate_target_heart` could not
see the control, so `_await_button` would raise `UnlocatedControlError`. That is fail-loud rather
than a blind tap, so the humanized-input rule held and nothing was ever mistapped. But it means
auto mode's like path could not have worked against Hinge's current UI, and the template it *did*
match at 1.000 is a different interactive control we must never tap.

Fixed by cropping the real button from calibration frames into
`operation_love/drivers/assets/hinge_like_button.png` (88x88, pure UI chrome, no profile content),
pointing the `like` role at it, and raising that role's threshold to
`_LIKE_MATCH_THRESHOLD = 0.75`. Real buttons score 0.815-1.000; the nav bar's "Matches" icon, a
persistent false positive at 0.6527, sits below the new threshold. Matching for the `like` role is
additionally restricted to `content_band` so both nav-bar false positives are excluded
structurally rather than by margin, the more important of the two being the nav "Likes" heart which
scores ~1.0 and was held out only by a 54px side cutoff.

That band restriction drops ~4% of genuine detections, those at y=2109-2176. **This is deliberate,
not an oversight.** The dark nav bar begins at y~2190, so a heart centred at 2176 has a tap point
only 14px clear of a system control. Those should be scrolled into view rather than tapped, and
the drop is safe by construction because the result is a fail-loud retry, never a mistap.
`hinge_heart.png` is left on disk, unused, with a comment recording what it actually is.

**A third production defect, found the same day by an actual live dry run, and fixed 2026-08-11.**
5 openers generated in a row against one real profile all opened with "I bet" -- exactly the
entropy-collapse risk 3.6 names. Root cause: the HEDGE THE CLAIM guardrail named only two forms
in `_SYSTEM` ("a hedge like I bet or I heard"), first-mentioned and closest to the point of
generation on every request, on every model in the cascade (thinkingLevel minimal, so there is no
scratchpad for the model to reconsider its lead-in). `config.yaml`'s copy was wider (four named
forms) but still put "I bet" second, and neither copy said anything about not repeating the same
opening construction across openers.

Fixed in both copies: the named hedge forms widened from two/four to seven ("I'm going to guess",
"I heard", "I'm assuming", "something tells me", "odds are", "my money is on", "I bet"), framed
explicitly as illustrative rather than a menu (the same escape-clause pattern 2.3 already uses for
the five moves), plus a new standalone VARY THE OPENING instruction telling the model not to open
every message the same way. The underlying guardrail -- hedge the claim, never the sender -- is
unchanged. The contrastive edit pairs (3.3) were checked for the same failure mode and found
already reasonably distributed across constructions ("I bet" appears in exactly one of the six
`Yes` lines), so they were left as shipped rather than churned for its own sake.

This is a prompt-layer mitigation, not the deterministic entropy guard 3.6 describes as the real
fix (a leading-n-gram repetition check on `recent_openers`, soft-signalled via `retry_hint` rather
than hard-rejecting). That guard is still not built. Re-verify with a fresh live dry run once it
ships, since prompt wording alone is not guaranteed to hold under continued volume.

**A fourth production defect, found by a live suggestion and fixed 2026-08-15.** Expanding the
hedge list changed the size of the phrase menu without removing the menu. The model opened with
"my money is on the French Alps", which reads as a bare answer to an imaginary location question:
the message never says what is supposedly in the French Alps, so the reader reasonably asks "for
what?" Its second sentence also introduced an unsupported evaluation rather than following from
the evidence-bound claim.

Fixed in both prompt copies by removing the named hedge forms. HEDGE THE CLAIM now specifies a
semantic function -- express uncertainty naturally when the claim needs it -- and explicitly lets
the specific item and angle determine the construction. A new SELF-CONTAINED CLAIM rule requires
the entire proposition rather than shorthand from an unstated guessing game; shared visual context
may carry the photo description, but not a missing subject or relation. Optional second-beat
questions now inherit the same one-hop evidence boundary and must be omitted when no natural,
evidence-bound question follows. The underlying permission to make playful, correctable guesses is
unchanged.

---

## 1. The problem

Openers over-describe the photo they are attached to.

> Photo: her in an outdoor sauna at sunset.
> Shipped: "That view by the sauna during sunset looks relaxing, where is this from?"
> Wanted:  "That view looks relaxing, where is this from?"

She is looking at the photo while she reads the message. Describing it back to her reads as
if we think she cannot see it.

### 1.1 Root causes, all in the prompt layer

1. **Every layer demands the opener prove it looked.** `_SYSTEM` (`opener.py:49-75`) says
   "Ground this opener in exactly ONE concrete detail"; the anchored closing
   (`opener.py:733-747`) says "use one concrete thing you can genuinely see in it"; the retry
   hint (`opener.py:772-799`) repeats it. Nothing distinguishes *being grounded in* a detail
   from *naming* it, so the model recites the detail inside the message.

2. **The schema already has the right place for the description and never routes it there.**
   `_SCHEMA` (`opener.py:34-47`) has a `referenced` field. No instruction says "put your
   evidence of grounding there and keep it out of `opener`", so the model writes it twice.

3. **Field order fights us, and `thinkingLevel: minimal` amplifies it.** The schema emits
   `opener` first, then `referenced`. All seven models run with minimal thinking
   (`config.yaml:242-249`), so there is no scratchpad, and the field that should absorb the
   grounding work comes after the message that is carrying it.

4. **The anchor copy states the premise and draws the wrong conclusion.** It tells the model
   "she reads your words directly beneath that item" and concludes only that the opener must
   *target* the right item, which pressures the model to disambiguate in the text. The
   conclusion it never draws is the opposite one: because she sees it there, you never need
   to name it.

5. **No few-shot examples exist anywhere in the pipeline.** Style is specified only as
   abstract rules, the weakest possible instrument for word economy. The two-sentence cap is
   a *sentence* cap, so nothing pushes against a long sentence.

### 1.2 What makes the fix safe

On Hinge a like-with-comment is always attached to a specific photo or prompt card. There is
no delivery path where she reads the text without the item in front of her. The
shared-visual-context assumption holds 100% of the time, so the rule can be unconditional.

---

## Part A: wording

### 2026-08-16 superseding correction: conversational value without forced invention

Sections 2 through 3.3 below are retained as decision history, but their mandatory
falsifiability rule, cover-the-photo test, question demotion, move menu and few-shot edit pairs
are no longer normative and no longer ship to the model. They overcorrected the original
description problem: when a photo offered no natural inference, the model manufactured a motive
or circumstance merely to satisfy the required claim shape.

The current contract is:

- A visible detail may be named and may be the subject, premise or setup. The opener fails only
  when its final conversational payoff merely identifies or describes that detail.
- Every visible detail named in the sent opener must be necessary to, and used by, its
  conversational move. If removing a descriptive clause leaves the later point or question
  unchanged, cut it rather than spending a sentence on an orphaned scene inventory.
- A guess or correctable claim is optional. When no natural inference exists, a grounded
  observation or specific question is better than a forced guess.
- Any claim uses the least speculative interpretation supported by what is visible or stated.
  Never invent a purpose, motive, cause, plan, sequence or unseen circumstance to create one.
- Clearly nonliteral playful hyperbole is allowed when its visible anchor is immediate. That is
  playful framing, not permission to present invented biography as literal fact.
- A question may be the whole message when it is the strongest natural angle. If a claim leads a
  two-beat opener, the second beat must accept and advance that premise rather than rechecking,
  contradicting or abandoning it.
- A question asks one coherent thing. An "or" is valid only for parallel, genuinely contrasting
  answers to the same underlying question, never for unrelated dimensions such as what an object
  is and why she was there.
- Production prompt text contains semantic properties and failure categories only. Concrete
  opener examples remain off-wire in tests and historical documentation so they cannot become
  templates for the model.

The authoritative shipped wording is pinned in `tests/test_opener.py`,
`tests/test_config_yaml_real.py`, and `tests/test_gemini_opener.py`.

## 2. The rule

Shortening alone produces a worse failure. "That looks fun, where is it?" is sendable to any
woman alive, and a generic opener reads as *not having looked* just as clearly as a
descriptive one does.

The property we actually want:

> **The opener must carry a claim that could be wrong.**

Describing a photo is unfalsifiable by construction. Nobody can dispute that the husky is
cute. A message that cannot be wrong could have been written without looking, which is why
description feels like proof of attention but is not. She is not checking whether we have
eyes. **Anyone can see. The opener has to show we thought.**

Call the property **unbluffable**: impossible to have written without having looked at this
exact profile.

### 2.1 The visible detail is allowed, as premise

An earlier draft of this rule was "delete every word that is true only because it is
visible." That is too blunt and would ban good openers. "Based on the mountain behind you"
is pure description and is fine.

> **The visible detail may be your premise. It may never be your point.**

Mechanical test: **cover the photo and read the message. Is there still a claim?**

| Opener | Covered |
|---|---|
| "That view by the sauna during sunset looks relaxing, where is this from?" | nothing survives |
| "Based on the mountain behind you, I'm going to guess this is in Norway" | a claim that can be right or wrong |
| "I bet you were freezing your butt off" | a claim about her experience |

### 2.2 Questions are demoted, not banned

"That husky is too cute, is it yours?" is short and still bad. "Too cute" is a null verdict,
and "is it yours?" has a near-certain answer, which makes it an interview question that
dead-ends at one word.

A guess is better than a question even when the guess is wrong, because a correction is the
easiest and most enjoyable reply a person can give. We do the work, she gets the payoff.

> **A claim she can correct beats a question she has to answer.**

Questions stay legal. They stop being the default. This also sits better with the 3% Man
frame than the current "one open, easy to answer question" wording, which is interviewing.

### 2.3 The five moves, explicitly non-binding

These are illustrations of the property, not a menu. **If none fits the item in front of the
model, it writes whatever does fit and keeps the property.** Forcing a move where it does not
belong produces awkwardness, which is worse than plain.

| Move | What it does | Example |
|---|---|---|
| **Guess** | Commit to an inference from visible evidence: where, what, who, what just happened | mountain in frame, guess the region |
| **Know** | Contribute something from your own head that the item triggered | how warm huskies run |
| **Imagine** | Claim something outside the frame: what she felt, what it cost, what happened before or after | "I bet you were freezing" |
| **Tease** | A risky good-natured claim about her, licensed by the evidence | adjacent to Imagine |
| **Connect** | Combine two things she said in different places on the profile | her food prompt against a backpacking photo |

**Connect is the strongest.** The other four are unbluffable by someone who looked at one
photo. Connect is unbluffable by anyone who looked at less than the whole profile. It is not
proof of attention, it is proof of synthesis. Its test is the sharpest one we have: on a
different profile, the same question would come out of nowhere. It also has no creepiness
ceiling, since she volunteered both halves herself.

### 2.4 Guardrails

**Hedge the claim, never hedge yourself.** "I'm going to guess", "I bet", "I heard", "I'm
assuming" are one device. They make a real claim safe to make and trivially easy to answer.

This is what rescues **Know**, which is otherwise the riskiest move: the person in the photo
is usually the expert on the thing in the photo, so "Did you know huskies run warm?" competes
with her expertise, while "I heard huskies run about as warm as a space heater" hands her the
floor to confirm or debunk. Same content, opposite posture. The hedge converts the worst case
into the best case, and it removes the need for a separate ban on invented numbers.

The hedge attaches to the *claim*. It must never attach to the *sender*. "Sorry if this is a
dumb question" is neediness and stays banned, both by the existing style rules and by the
Corey Wayne frame.

**Guess the world, not her identity.** The cap is on *specificity*, not accuracy. The model
may well be able to name the exact resort. It must not. Guess at country, region or park
scale, the way a well-travelled friend would. Never a street, neighbourhood, hotel, specific
venue, or anything that reads as where she lives. The same rule covers reading her employer
off a lanyard, her school off a hoodie, guessing her age, and identifying other people in the
frame.

This guardrail is free: the guess exists to invite a correction, so being slightly wrong is
the feature. Capping precision makes the opener better, not merely safer.

**Never invent the sender.** "I did that trail last fall and it wrecked me" is a tempting
shape. It is unbluffable, it builds rapport, it needs no hedge. It is also the model
inventing the owner's history, which the owner then has to sustain five messages later. The
model must never claim experiences, preferences, or history on his behalf. This is a hard
rule, not a stylistic preference. Note that "I heard" is safe precisely because it claims
nothing about him except that he heard a thing.

**Endorsement blocks are excluded.** Hinge shows a "From people close to [name]" section.
Callbacks land on *authorship*: referencing what she wrote proves we paid attention to her,
referencing what her friends wrote proves we read the page. A tease built on a friend's line
is the worst possible ammunition, since she never signed off on it and has to either validate
or disown someone else's characterization. Dropped entirely, see 5.3.

---

## 3. Prompt and schema changes (Part A)

### 3.1 Where the text lives

`_SYSTEM` (`opener.py:49-75`) and `config.yaml opener.style` (`config.yaml:280-300`) are
near-duplicate copies of the same style text, pinned by two *separate* tests. Editing one and
not the other gives the model contradictory instructions and fails a test in a file nobody
would think to look at.

Recommendation: the long-form rule and the examples live in `config.yaml opener.style`, which
is owner-tunable and is where voice belongs. A compressed one-liner of the property goes in
`_SYSTEM` alongside the mechanics. Flag the duplication itself as debt.

### 3.2 New style block (draft)

```
SHARED CONTEXT RULE: your message is displayed directly under the exact photo or prompt it
attaches to. She is looking at that item while she reads your words. Write like two people
looking at the same thing.

THE ONE RULE: your opener must contain a claim that could be wrong. Describing what is in
the photo can never be wrong, which is exactly why it proves nothing. She is not checking
whether you have eyes. A message that could have been written without looking at her profile
has failed, whether it is too descriptive or too generic.

The thing you can see may be your premise. It may never be your point. Test it by covering
the photo: if nothing is left, start over.

SELF CONTAINED CLAIM: write the whole proposition, not shorthand from an imagined question
or a bare option in an unstated guessing game. Shared visual context may carry the photo
description, but the message must still say what you think is true.

WAYS THIS TENDS TO LOOK. These are examples, not a checklist. If none of them fits the item
in front of you, write whatever does fit and keep the rule.
  Guess something from the evidence and commit to it.
  Say something you know that the item brought to mind.
  Claim something outside the frame: what she felt, what it cost, what happened next.
  Tease her, good naturedly, about something the evidence licenses.
  Connect two things she said in different places on her profile.

A claim she can correct beats a question she has to answer. Questions are allowed as the
second beat after a real one, never as the whole message. That second beat inherits the same
evidence boundary and may not add an unseen expectation, outcome, or earlier conversation.
If no natural evidence bound question follows, stop after the claim.

HEDGE THE CLAIM, NEVER YOURSELF: when a claim is uncertain, express that uncertainty naturally
in wording that fits the specific item. This is a semantic requirement, not a phrase menu.
Choose the construction from context and state the complete proposition. Being wrong is then
part of the fun and she gets to be the expert. Never apologise for writing, never ask
permission, never call your own question dumb.

GUESS THE WORLD, NOT HER IDENTITY: name a country, a region, a park, the way a well travelled
friend would. Never a street, a neighbourhood, a hotel, a specific venue, or anywhere that
could be where she lives. Never guess her employer, her school, or her age, and never
identify anyone else in the photo. You may know more than this. Do not show it.

NEVER INVENT THE SENDER: you may not claim he has been somewhere, done something, or likes
something. You do not know his history and he has to live with whatever you write.
```

The draft deliberately says "the sender" rather than "me". The rest of the prompt addresses the
model in the second person about a third-person man ("the opening message a man sends"), so a
first-person "me" is ambiguous about whose history is meant.

### 3.2.1 The length rule has to soften

`_SYSTEM` and the style block both say "ONE short sentence is preferred and TWO sentences is
the absolute maximum." The two-sentence ceiling stays. The one-sentence *preference* now works
against the design, because a claim that could be wrong needs room to exist. The strongest
opener in the whole design discussion, "I know you were smiling, but I bet you were freezing
out there", is eighteen words and would be pushed against by a rule that prefers brevity for
its own sake.

Replace the preference with an economy rule tied to the actual goal: as short as the claim
allows, with no word spent on anything she can already see. That keeps the pressure on padding
while removing it from substance. We are reallocating the budget, not cutting it.

### 3.3 Contrastive edit pairs

Ship them as before/after **edit pairs**, never as standalone good openers. An edit pair
teaches the transformation; a standalone exemplar teaches a string the model will copy, which
matters given the entropy concern in 3.6.

```
EXAMPLES OF THE EDIT. Same item each time, first is wrong, second is right.
Photo of her in an outdoor sauna at sunset.
  No:  That view by the sauna during sunset looks relaxing, where is this from?
  Yes: That view looks relaxing, where is this from?
Photo of her with a husky in the arctic.
  No:  That husky is too cute, is it yours?
  Yes: I know you were smiling, but I bet you were freezing out there.
  Yes: I heard huskies run about as warm as a space heater, so at least you had that.
Photo of her on a ridge with mountains behind her.
  Yes: Based on that ridgeline I'm going to guess Norway.
Prompt card, two truths and a lie about 30 countries, cilantro, and meeting a pop star.
  No:  Your two truths and a lie about the countries, cilantro and the pop star is fun,
       which one is the lie?
  Yes: The cilantro one is the lie, I can feel it.
Her prompts mention she puts hot sauce on everything; separate photo of her backpacking.
  Yes: How many bottles of hot sauce did that trip cost you?
```

All example openers must be free of em dashes and hyphens (existing owner rule), since the
model copies the shape of what it is shown.

### 3.4 Schema changes

Reorder to `referenced` then `angle` then `opener`, so the grounding work is discharged into
a field before the message is written. This matters specifically because `thinkingLevel:
minimal` leaves the model no other scratchpad. Whether Gemini honours declared property order
in `responseJsonSchema` is a hypothesis, so A/B it rather than assuming.

Rewrite the field descriptions to be adversarial to each other:

- `referenced`: "What you are reacting to, described in full. **This field is never sent to
  her.** Put the whole description here so it does not leak into the opener."
- `angle`: free text, the model's own words for what its opener is doing. Telemetry only,
  never constrains output. Deliberately **not** an enum, see 3.5.
- `opener`: "The bare message. She reads it while looking at the item, so it must not
  describe the item. Do not reuse the words from `referenced`."

Optional A/B arm: a `first_draft` field before `opener`, with `opener` defined as
`first_draft` with every word she can already see removed. In-call self-trim, not a second
model call, so it does not touch the no-LLM-judge decision. `max_tokens` is 2048 against a
one-sentence output, so there is headroom.

### 3.5 Why `angle` is free text and not an enum

An enum forces a pick from a closed set, which is the shoehorning the move list is explicitly
designed to avoid. Free text preserves the commit-to-a-strategy scaffolding and the telemetry
value without constraining the output space. Over time it answers a question we currently
cannot ask at all: which shapes correlate with matches.

### 3.6 Entropy guard

Shortening compresses the output space, and few-shot examples make direct copying a live
risk. Across a burner account sending uncapped volume, near-identical openers are both a
fingerprint and embarrassing if two matches compare screenshots.

The fix is *not* forced move rotation. It is a deterministic repetition check on the leading
n-grams of the last N openers, using the `recent_openers` ring buffer that already exists
(`service.py:773-782`). Pure string comparison, no semantics, no constraint on what the model
may write. Catches "Based on the X, I'm going to guess" recurring without a taxonomy.

Open decision: a hard rejection counts toward the exhaustion budget and can stop a run over a
stylistic near-miss, which is disproportionate. Preference is a soft signal fed into
`retry_hint` without consuming the budget, which needs a small change to the service's attempt
accounting.

The asymmetry that makes this urgent rather than academic: observe mode passes `advisory=True`,
which forces `effective_max_attempts = 1` (`service.py:461-465`), not five. So in the very mode
we plan to canary this redesign in, a single leading-n-gram collision exhausts the only attempt
and disables suggestions for the rest of the session, with no obvious explanation to the
operator. A hard-rejecting entropy guard is therefore strictly worse in observe than in auto,
which is the opposite of what the sequencing in §7 assumes.

#### Addendum 2026-08-11: that asymmetry argument was misapplied to the shipped design

The paragraph above is correct about a HARD-REJECTING guard, and it is exactly why the "Open
decision" two paragraphs up was resolved in favour of the soft, `retry_hint`-based design instead.
But when the soft guard shipped, its advisory handling was built by carrying this section's
conclusion over unexamined: `_apply_entropy_guard` was written to skip itself outright whenever
`advisory=True`, as though the hard-rejection cost described above still applied. It does not. The
soft guard's regeneration draw sits on `maybe_opener`'s SUCCESS path, lexically inside the `for
attempt in range(...)` loop but returning unconditionally -- so it can never produce another
`attempt` iteration and can never consume `effective_max_attempts`, in advisory or in auto alike.
There was never an attempt for `advisory=True` to lose by running the guard. The asymmetry this
section identified is real for the design it was written about; it was never true of the design
that actually shipped, and nobody re-checked that before wiring the skip in.

The consequence was worse than a missed optimisation: Hinge observe mode is meant to be the
canary for auto (the opener a human sees in observe must be byte-identical to what auto would
send for the same profile -- that is the whole reason observe exists as a preview). A guard that
fires in AUTO but is inert in observe makes the two diverge on exactly the profiles the guard is
supposed to change, defeating the canary property silently -- nothing on the hub or in the logs
distinguished "the guard did not fire" from "the guard fired and found nothing." A live dry run
made this concrete: `advisory=True` against one real profile, 5 openers requested, 4 came back
sharing an identical leading phrase, because the guard never even looked.

Fixed the same day: the guard now runs identically whether or not the call is advisory, with no
branch on `advisory` anywhere in `_apply_entropy_guard` (the parameter was removed from its
signature entirely, not just left unused). What makes this safe is not a re-derived asymmetry
argument, it is the guard's own invariants from earlier in this section -- never raises, never
rejects, never touches `stop_requested` or a latch counter, at most one extra draw, ever -- which
hold identically regardless of `advisory`. See `service.py`'s `_apply_entropy_guard` docstring
(ADVISORY ASYMMETRY paragraph) for the full corrected reasoning, and
`tests/test_opener_service.py`'s advisory entropy-guard tests for the pinned contract.

### 3.7 The redundancy monitor, measurement first

Over-description looks semantic but has a deterministic proxy nobody has used: the model's
own `referenced` field. If `referenced` is "outdoor sauna at sunset" and the opener contains
"sauna" and "sunset", the opener is restating its own grounding note.

Feasibility is good: `referenced` sits in the same parsed dict as `opener` inside `_parse`
(`opener.py:1081-1225`), so `_redundant_description_markers(opener, referenced) -> list[str]`
is a drop-in sibling of `_scaffolding_markers` with no signature change and no photo context
needed. On the sauna pair the signal separates cleanly: bad opener overlaps on two content
words, good opener on zero.

**It ships log-only, not as a gate**, for four reasons:

1. It is a lower bound on redundancy, defeatable by a terse `referenced`. It can never be the
   primary defence, only the monitor. The prompt is the fix.
2. Five consecutive rejections stop the run (`service.py:595-612`). An uncalibrated threshold
   is a run-killer.
3. Real opener data exists but is far too thin to set a threshold on. The local
   `data/operation_love.db` has 0 rows, but sqlite is not the system of record:
   `config.yaml:168` sets `storage.backend: bigquery`, and
   `operation-love-2026.operation_love.openers` holds a handful of real rows from live runs.
   A handful is enough to sanity-check the metric, nowhere near enough to pick a cutoff.

   Those rows also confirm the bug is real in production rather than hypothetical: at least one
   shipped opener names the specific location visible in the photo it responds to, alongside a
   genuine claim in the same sentence, which is exactly the redundant naming this redesign
   targets. (The verbatim text is not reproduced here: it is a private message to an
   identifiable person, and this doc is tracked in git.)
4. The threshold can be derived offline at no cost, because `record_opener` already persists
   both `opener` and `referenced` (`store.py:116-122`, `bigquery_store.py:367-373`). No schema
   change required.

Promote to a gate only once real data shows a threshold with a zero false-positive rate on
owner-approved openers. This respects the existing scaffolding-defense decision: deterministic
only, no LLM judge, no classifier.

---

## Part B: capture, item selection, and targeting

## 4. Why this part exists

Two separate findings force it.

**The correctness hole.** `referenced_index` today indexes a *scroll frame*, not an item.
`_locate_target_heart` (`hinge.py:2416-2486`) re-scrolls until a frame's downsample signature
matches, then takes `hearts[0]`, the topmost heart on that frame. The driver's own docstring
(`hinge.py:2559-2562`) admits the consequence: a frame showing a photo and a prompt at once
can land the tap on the neighbour **even when `on_target` reports True**. And `on_target` is a
pre-tap screen check, not a post-tap read of the opened sheet, so a wrong success skips the
repair path. Today "which item did we actually like" is only reliably answered when targeting
already knows it failed.

**Connect needs the whole profile.** There is no text channel at all: `Profile.prompts` is
hardcoded `[]` and `bio` is never set (`hinge.py:2220-2230`), so every word of her profile
reaches the model as pixels. The capture caps at 12-14 frames (`config.yaml:92`, code-level
default 8 at `hinge.py:358`), and in real logged runs `capture_truncated` was true on **2 of the
20 captures that carry the field, about 10%**. There are 29 capture records in
`data/hinge_debug/*/actions.jsonl` in total, but 9 predate the field being logged and carry no
value either way, so they are excluded rather than assumed untruncated. On truncated profiles
the model is shown only the top and Connect silently loses its material.

Either denominator is a tiny sample and neither should be treated as a reliable rate. The point
that survives is only that truncation is real and not rare.

## 5. The design

### 5.1 One call, model picks the item

Auto mode makes a single call, as it does today. The model receives the profile and returns
which item to like plus the opener. The driver then navigates to that item, taps its heart,
verifies, and types.

Selection criterion changes with it: pick the item that yields the **best angle**, not the
most striking photo. A mediocre backpacking shot that connects to her food prompt beats a
great portrait with nothing to say about it.

The anchor image is dropped from the prompt entirely in both modes. It only ever existed
because auto wrote blind, hearted, then repaired. Generation now happens before anything is
tapped, so the anchor becomes a post-tap verification artifact instead of an input.

#### Addendum 2026-08-12: the selection criterion is now stated in the prompt, in both copies

5.7's response contract and request payload gave the model an `item_index` to fill and a numbered
list to fill it from; neither told it how to CHOOSE. Until now the only guidance was one clause
("choose the item you have the best thing to say about, not the most striking picture") that
shipped alongside the schema change. That states the preference but not the tradeoff it exists to
settle, and a model shown a page of photographs has a strong prior toward the best photograph.

Three paragraphs now ship in `config.yaml opener.style` (long form, per 3.1's division of labour)
and in a compressed form in `opener.py`'s `_SYSTEM`, each pinned by its own test
(`tests/test_config_yaml_real.py::test_shipped_opener_style_ships_the_item_selection_rule`,
`tests/test_opener.py::test_system_prompt_keeps_faithful_corey_opener_policy_and_two_sentence_cap`).
Nothing already in Part A was edited: the one rule, the five moves and their escape clause, the
three guardrails, the hedge list, VARY THE OPENING, the two-sentence ceiling, the economy rule,
the hard rules and every edit pair are byte-identical.

**The criterion is a tradeoff, so it ships as one.** This section's own illustration goes in
verbatim in substance ("a mediocre backpacking shot that connects to her food prompt beats a
great portrait you have nothing to say about"), because the tradeoff is the part that gets
decided wrong, not the preference. It is 2.1's premise/point distinction applied one step
earlier: the item is only ever the premise, so selecting by how the item LOOKS optimises the half
that never becomes the message.

**The failure mode is named as a failure, and it is not a restatement of THE ONE RULE.** Picking
a striking item with nothing to say about it is the upstream CAUSE of over-description rather
than a sibling of it: every wording rule in Part A can be obeyed right up to the moment the item
is chosen, and after that description is the only material left. So the remedy is stated at the
stage it goes wrong ("the way out is always to pick a different item, never to describe the one
you picked"), which is the only instruction that can actually be followed once the model notices.

**The context tier is described rather than merely forbidden.** 5.3 confirms from a live capture
that the vitals block (age, job, school, city) carries no heart, so it is sent unnumbered and can
never be chosen. But 2.3 names Connect the strongest move precisely because it combines two
things she said in different places, and the vitals block is prime material for it, so the prompt
now says why to USE it and not only that it cannot be picked. The `_SYSTEM` copy deliberately
makes that point WITHOUT naming the move list: the moves are stated as non-binding examples in
`config.yaml` only, and a compressed copy in `_SYSTEM` would arrive without its escape clause,
which is the regression `test_system_prompt_never_turns_the_five_moves_into_a_binding_menu`
exists to catch.

No live call was made for this change: it is prompt copy, offline-testable, and the one dry run
this workflow was allowed was spent on 5.7's request shape.

### 5.2 Crops, not frames

Cropping is not about legibility or size. Full-resolution pixels are already preserved in the
common case, because `_fit_images_to_budget` tries JPEG-85 recompression with no resize first
and that is nearly always enough. Cropping is about **who owns the numbering**.

1. **Index correspondence.** The driver enumerates by counting hearts. If we send overlapping
   full frames, the model must derive its own independent enumeration and the two must agree
   by luck. With crops, image 3 in the request *is* item 3. Agreement by construction.
2. **Duplication bias.** A card straddling a scroll seam appears in two or three frames.
   Harmless when the model wrote about whatever caught its eye; not harmless now that it is
   *choosing* among items, since repetition reads as salience. We would bias selection by our
   own scroll cadence.
3. **We need the crops anyway.** Post-tap verification is a signature match against the stored
   crop of item N. Without per-item crops there is no reference to match against.

Crops are also smaller and fewer than the frames they came from, so this is the cheaper
option. The cost is the detection, which is already committed for targeting. The crop is a
byproduct.

OCR stays out of the opener's input path. Tesseract is real in production but only ever reads
three narrow fixed bands: the sticky header name (`hinge.py:2138`, psm 7), the scroll-top card
header name (`hinge.py:2892`, psm 6), and the paywall headline (`hinge.py:2345`, psm 6, which
its own docstring notes wraps to two lines). Every one is a known rectangle containing a known
kind of short string. Reading arbitrary multi-line prompt answers off a card is a different
problem, and ANTI-BOT-RESEARCH calls production OCR "best-effort". Gemini reads the pixels
better. Keep OCR as an optional telemetry channel if we later want to ask which prompt topics
correlate with matches; never as the model's view of her words.

Her name is lost by cropping and must be passed back as text. It is already extracted
(`hinge.py:2138`).

#### Addendum 2026-08-12: the crops exist, and the viewport caveat now has a detector

Two-tier crop assembly is built in `operation_love/drivers/item_crops.py`
(`build_item_payload`), on top of `item_index.py`. It takes an index plus the frames it was built
from and returns an `ItemPayload`: one cropped image per selectable item numbered 1..N, the
context blocks cropped and unnumbered, an entry for every block that was NOT sent with the reason
it was withheld, and a `CropSignature` on every crop for 5.6's post-tap check. `ItemPayload.images`
is the whole of what goes to the model, and the module holds no frame bytes at all, so there is
nothing for a caller to append alongside the crops. Full derivation and per-constant measurements
are in that module's docstring.

Measured on the same gitignored corpus (geometry and counts only): the 0.16-cadence capture yields
**9 numbered items and 1 unnumbered context crop** from its 11 blocks, the leading chrome dropped
and nothing uncroppable, each item cropped from a different one of 9 frames. The 10 crops total
8.53MB of PNG against 37.15MB for the 24 frames they came from — 4.4x smaller, which is this
section's "crops are also smaller and fewer than the frames they came from" with a number on it.
The two unusable captures raise before a single crop is produced.

Four things worth recording because they were not in the original design.

**The viewport caveat is now a check, not a hope.** 5.10 closed the stitching question on "the
tallest card fully observed is 1114px against an 1800px content band" and added "a very long
prompt answer could still exceed it, so the crop path should detect the case rather than assume it
away". A block taller than the analysed band is now a hard failure naming stitching as
deliberately unimplemented, and the limit is READ OFF the segmentation rather than declared, so it
follows `content_band`. On the corpus the tallest crop is 1144px, 64% of the band.

**A crop is never a fragment.** An item no frame ever bounded end to end produces no image at all
— it is reported with its page position and the reason. Cropping the visible part would send the
model a half card and then store that half as the verification reference.

**Exclusion is a mechanism here and deliberately not a detector.** 5.10 measured that the
endorsement block did not appear in either capture, so `exclude` is a caller-supplied
`block -> reason` predicate, `EXCLUSION_ENDORSEMENT` is the named reason string, and
`exclude_page_rows` builds a predicate from explicit page rows. Excluding a block never touches
the heart ordinals — 5.3's index trap — but it DOES renumber the model's dense list, so
`ItemPayload.translation` rather than `ItemIndex.translation` is the table to navigate by whenever
anything selectable was excluded. With no exclusions they are equal.

**The signature has one entry point, because the obvious two disagree.** `signature_of` is what a
verification pass must call on its screenshot. `cv2.IMREAD_GRAYSCALE` and "decode colour, then
`cvtColor(BGR2GRAY)`" measure 1.46 grey levels apart on average and up to 9.8 apart on a 32x32
crop signature — where 4.66 is the entire distance between the two most alike items on that
profile, so the wrong decode would land nearer the wrong item. Measured separation at that grid:
the same card re-observed in another frame sits 0.02 from its stored signature, the nearest
DISTINCT item 4.66, the median pair 79.2, and a rect located 1/2/4px off costs 1.47/2.93/5.78. So
re-observation is essentially exact and the binding constraint is how precisely 5.6 locates its
rect, which argues for locating it by segmentation rather than by assumption. No threshold constant
is shipped: every number above is crop-against-crop, and 5.6's real comparison is
crop-against-like-sheet, which no capture in the corpus contains.

Still out of scope and unbuilt at this point: the closed-loop scroll, counting navigation, the
schema change (`item_index`/`item_description`), the observe inversion, the post-tap check itself,
and anchor removal.

#### Addendum 2026-08-12: the separability numbers above are WRONG, and the fix is to ship the measurement

The addendum above reports "the same card re-observed in another frame sits 0.02 from its stored
signature, the nearest DISTINCT item 4.66" and reads that as a ~230x margin. An independent
validation pass found it does not hold on the second profile. Re-measuring found it does not hold
on the FIRST one either, and the cause is the METHOD, not the profile: that measurement compared
only the sightings where segmentation happened to return a byte-identical rect in both frames,
which systematically excludes every card that is MOVING, because a moving card's segmented bottom
edge wobbles by a pixel. It measured the static cards and reported the answer as a property of
the signature.

Re-measured by resampling each crop's own page rows in every frame whose analysed band contains
them (grey levels and counts only):

| | profile B (24 frames, 9 items) | profile A (57 frames, 9 items) |
|---|---|---|
| worst re-observation drift | **2.66** (the last card) | **25.0** (the animated card) |
| that item's distance to its NEAREST other item | 44.7 | 47.9 |
| smallest distance between ANY two items | 4.66 (items 5 and 7) | 6.27 (items 5 and 7) |
| items reported unseparable | 0 of 9 | 0 of 9 |
| items nothing re-observed, so unmeasurable | 1 of 9 | 1 of 9 |

Two readings, both true, and the second is the one that stops this blocking 5.6. The shipped
number was wrong by two orders of magnitude — re-observation is not "essentially exact". But the
comparison a verification pass actually makes is PER ITEM ("is this item 9" only has to beat item
9's own neighbours), and on that basis both captures still come out separable on every item they
could measure, at 1.9x on the worst instead of 230x. Comparing the worst drift against the
smallest distance between ANY two items (25.0 against 6.27) inverts the verdict and is the wrong
comparison, because item 9 is nowhere near items 5 and 7.

A fixed threshold is still ruled out: 0.02 rejects a correct re-observation on both profiles and
30 accepts a wrong item on either. So doc 5.4's alternative is what ships — explicit detection,
per item, no verdict baked in. `ItemCrop.signature_drift`, `ItemCrop.nearest_item_distance` and
the three-valued `ItemCrop.separable`, plus `ItemPayload.unseparable_items` and
`undetermined_items`. **`separable` is None, never False, when nothing re-observed the crop**: a
capture that steps and never revisits measures nothing, and silence must not read as stability.
Neither state makes the payload unusable — an animated card compromises its own reference and
nothing else, and refusing the payload would throw away eight sound references to punish the
ninth. 5.6 decides what an unverifiable item may be used for.

Also closed here, from the same validation: `build_item_payload` now refuses a frame list that is
not byte-identical to the one the index was built from, via a sha256 per frame recorded at
segmentation time (`FrameSegmentation.frame_digest`). The previous guard was the frame count plus
each decoded frame's size, which can never fire on this device — every Hinge screencap is
1080x2400 — and validation drove the same capture in REVERSE order straight through it, getting
ten confidently wrong crops with zero failures and signature distances of 18.8 to 156.5 from the
right ones.

### 5.3 Two tiers, and a driver-owned index space

Confirmed from live-device observations: Hinge profiles contain a **vitals
block** (age, job, school,
city, languages) with **no heart**. Pure heart-delimited cropping would drop it silently, and
it is prime Connect fuel.

- **Selectable items** are heart-bearing: photos and prompt cards. The model may pick one.
- **Context blocks** have no heart: vitals. Sent, read, freely referenced, never selectable.
- **Excluded blocks**: the endorsement section, per 2.4. Not sent at all, because models
  reference what they are shown.

The index trap: if an excluded block turns out to *have* a heart, the driver still counts it,
because navigation counts hearts. Dropping it from the model's list while the driver keeps
counting reintroduces exactly the divergence this redesign eliminates. So:

> **Index space belongs to the driver. Selectability is policy.**

The model receives a dense list of items 1..N that it may choose from. The driver keeps a
private translation table from model index to heart ordinal. Item 3 in the model's list may be
the fourth heart on the page. The model never sees heart ordinals. Any future block type slots
into the same scheme without touching navigation.

Concretely, the table is per-capture driver state built at segmentation time, alongside
`_current_sigs`: an ordered list of records carrying the heart ordinal, the block class
(selectable / context / excluded), the crop, and the crop's signature for later verification.
The model index is the position within the selectable subset only.

Its lifetime is exactly one profile, and it must be invalidated wherever `_current_sigs` is
today, including the deck-advance path (`_current_capture_split`, `hinge.py:2196-2210`).

A stale table is a reliability bug rather than a safety one, and it is worth being precise
about why. Navigating by a previous profile's ordinals would tap a heart on the current
person's card, but the stored crops are stale too, so the post-tap signature check in 5.6
compares the opened sheet against the wrong reference, fails, and stops the run. The failure
mode is therefore a guaranteed false-positive stop on a profile that was fine, not a silent
wrong like. Still fix it, but the mitigation is correct invalidation, not an extra guard.

Treat a missing table as a hard stop, never as a reason to fall back to a fixed coordinate.

#### Addendum 2026-08-12: the index and the translation table now exist, and 5.5's weakest joint is closed

Both index spaces this section specifies are built in `operation_love/drivers/item_index.py`
(`build_item_index`), on top of `segment.py` (per-frame blocks) and `frameshift.py` (per-pair
translation). It returns an `ItemIndex` carrying every block found across the whole scroll in page
order — absolute page extent, class, heart page position, and the frames it was observed in —
plus the heart ordinal for every heart-bearing block, the dense model index over the selectable
subset only, and `translation` between them. Full derivation and per-constant measurements are in
that module's docstring.

**5.5's "weakest joint" is closed by construction rather than by a better matcher.** That section
said heart dedup across frames "needs resolving before implementation, not during", and 5.10
measured naive chaining recovering 9 real items and fabricating a spurious 10th. Deduplication is
now by ABSOLUTE PAGE POSITION under the measured shift, so the same card seen in eight frames is
one block; and a pair the shift estimator refuses yields NO items at all rather than a best guess,
because without that pair there is no way to tell a heart that moved from a heart that arrived.
Three guards make a phantom unreachable: a broken chain returns an empty block list, folding is by
overlap in one coordinate space, and two resolved blocks closer than the gutter window (47px) are
reported as one card that fragmented rather than accepted as two items.

Two consequences worth recording because they were not in the original design.

**Disagreement is a failure, not an average.** Two frames that both bounded the same card are two
measurements of one fixed quantity; if they differ by more than 8px the index says so. The
resolved extent is always an extent some single frame actually observed, never a blend — a blended
extent would be cropped, sent to the model, and then stored as 5.6's post-tap verification
reference, where it would fail against whichever card is really there.

**The scroll-top confirmation 5.5 already requires is now an argument.** `at_scroll_top` is a
required keyword: it is the caller stating it made the affirmative filter-chips check. It buys
absolute heart ordinals, and it is what licenses treating Hinge's header chrome as chrome rather
than as a block that might be hiding heart 1 — segment.py's docstring assigns exactly that job to
its caller, and this is that caller. A heartless block nobody ever bounded is otherwise a hard
failure whenever anything heart-bearing sits below it, since a hidden heart would shift every
ordinal under it.

Measured on the same gitignored corpus (geometry and counts only):

| Capture | Result |
|---|---|
| botscroll 231516Z (frac 0.16), 24 frames, `at_scroll_top=True` | USABLE: **9 selectable items, no phantom** — the same 9 two independent captures agree on. 11 blocks: 1 leading chrome, 9 selectable, 1 context (1144px). Translation 1..9. **Zero** partial blocks. `reached_end`, not truncated |
| the same capture, `at_scroll_top=False` | UNUSABLE on one failure: the 42px leading chrome block "was never observed complete and shows no like heart" |
| botscroll 225314Z (the aliasing cadence that fabricated the phantom) | UNUSABLE at the first pair, "moved +1299px, beyond the 900px window". **Zero blocks, zero phantoms** |
| scroll 211209Z (hand-scrolled, 115 frames) | UNUSABLE, refused at the pair frameshift also refuses (56/57, the animated-card tail). The 56 pairs before it chain cleanly |

Still out of scope and unbuilt at this point: the closed-loop scroll itself, counting navigation,
crop assembly, the schema change, the observe inversion, post-tap verification, anchor removal.

#### Addendum 2026-08-12: `at_scroll_top` is no longer taken entirely on trust

The addendum above records `at_scroll_top` as "the caller stating it made the affirmative
filter-chips check", verified by nothing. An independent validation pass then measured what an
unchallenged FALSE assertion costs, and it is worse than "the ordinals are relative": told
`at_scroll_top=True` about a capture that began three frames into a real profile, the index came
back USABLE with 8 items, translation 1..8, `truncated` False, `complete` True and **zero
failures**, and its model item 1 was really the profile's item 2. Every completeness property
corroborated the lie rather than staying silent, and the leading-chrome rule made it worse: it
relabelled the sliced top fragment as "chrome", which is precisely the evidence against the
claim.

The assertion still cannot be VERIFIED from geometry — segment.py's docstring is right that one
frame cannot distinguish "the header is above card 1" from "we scrolled past it" — but it can be
CONTRADICTED. At a genuine scroll top Hinge's header is the topmost content and is drawn BELOW
the analysed band's first row, with page background between the two: measured **34px** on profile
A and **68px** on profile B, both with the shipped `content_band`. A capture that started
mid-profile has the band's top edge slicing a card instead, and its topmost block begins ON that
row. So `item_index._scroll_top_evidence` requires that some frame saw background above the
topmost block, and the leading-chrome relabelling now requires the same evidence instead of
erasing it.

Measured on the same gitignored corpus (geometry and counts only):

| Capture | Before | After |
|---|---|---|
| botscroll 231516Z, genuine top | USABLE, 9 items, translation 1..9 | unchanged |
| scroll 211209Z frames 0..56, genuine top | USABLE, 9 items, translation 1..9 | unchanged |
| botscroll 231516Z frames 3.., asserted top | **USABLE, 8 items, translation 1..8, zero failures** | UNUSABLE, two failures, leading fragment stays PARTIAL |
| every suffix of botscroll 231516Z, asserted top (21 false tops) | — | **21 of 21 refused, 0 still accepted** |
| every 5th suffix of scroll 211209Z frames 0..56 (10 false tops) | — | **9 refused, 1 still accepted** |

It is a PARTIAL guard and says so in its own docstring. A false top whose band edge lands in a
gutter, or in any other page background, has background above its topmost block just as a genuine
top does. The one miss above is exactly that: its topmost block began 4 rows below the band's
first row. The gutter alone is 53px of a ~1027px card pitch, so roughly 1 scroll position in 20
looks innocent by construction, and the 21 of 21 on the bot capture is a 363px cadence that
happens never to land there rather than a guarantee. **Doc 5.5's filter-chips confirmation
therefore stays a hard gate before capture**, and this turns "undetectable" into "usually
detected", nothing more.

Two smaller corrections from the same validation, recorded because the module docstrings asserted
them and they were false:

- Order is NOT load-bearing for correctness. A fully reversed run, and every adjacent
  transposition of a real capture, still chains and still yields the same 9 items with the same
  translation, because page position is measured pairwise rather than assumed from ordering. What
  order buys is coverage (`reached_end` is read off the last frame) and correspondence between
  neighbours, not a guard against reordering.
- With `at_scroll_top=False` the ordinals are relative and nothing else on the result says so: a
  mid-profile window returns translation (2, 3, 4, 5, 6) for what are really items 5..9, and
  `heart_ordinal_for` still answers with a confident int. Navigation must not count against a
  False index.

### 5.4 Segmentation: hearts and gutters

There is no segmentation capability in the repo: no contours, no Canny, no thresholding, only
`cv2.matchTemplate` against five fixed chrome glyphs. But general layout parsing is not needed.

**Hearts say which blocks are selectable.** Every likeable item has exactly one heart, always
bottom-right (`hinge.py:750`, enforced by `_match_glyph`'s `side="right"` filter at
`hinge.py:795-798`). `_match_glyph` already returns *all* matches sorted top to bottom through
a 12-iteration NMS loop; every caller then throws away everything but the first
(`_locate_target_heart` at `hinge.py:2474`, and `_locate_button` at `hinge.py:1347-1357` does
the same for the swipe-deck heart). The information the new design needs is already computed
and discarded at every call site. A frame can show two hearts at once, and the list-returning API
already handles that.

**Gutters say where each block starts and stops.** Hearts mark where a card *ends* (bottom
right), not where the next begins, so hearts alone cannot give crop edges. Hinge cards are
rounded rectangles on a contrasting page background with visible gutters. A row-wise test for
"is this scanline uniformly page background" finds those boundaries directly: a horizontal
projection over a numpy array, not contour detection.

**Failure classes this must handle explicitly.** None of these are hypothetical; the first two
are already documented as live risks in this codebase.

- **Animated or video cards.** `hinge.py:1879-1880` and `ANTI-BOT-RESEARCH.md:501` record that
  a card which never settles is an observed condition. It defeats the frame-repeat bottom
  detector today, and under Part B it also defeats the deterministic crop-signature match in
  5.6, since two screencaps of the same position are not pixel-identical. Under the
  never-substitute rule that turns every animated card into a guaranteed hard stop. Needs a
  tolerance band on the signature comparison, or explicit detection and a different path.
- **Cards taller than the viewport.** The gutter test needs the top and bottom edge of a block
  visible in one frame to crop it. A long prompt answer, or a photo card on a smaller content
  band, may exceed it. Either stitch across two frames or accept a partial crop, but the doc
  must say which; silently cropping a fragment would poison both the model's view and the
  verification signature.
- **No gutter at the list boundaries.** Above the first card is Hinge's filter-chips header, and
  below the last is the dark bottom nav (`observe_ignore_zones`, `hinge.py:397-400`). Neither is
  page background, so the first and last blocks need special-casing rather than the interior
  gutter rule.
- **Bright uniform regions inside photos.** Snow, sky, a white shirt: a naive "uniform row" test
  can fire mid-photo and split one selectable item into two, which shifts every index below it.
  The discriminator has to be the specific page-background colour plus the expected gutter
  height, not uniformity alone.

#### Addendum 2026-08-11: the list-boundary failure class is SOLVED, not open

The bullet above was written as an unsolved problem, with a caller-declared `list_top_y` /
`list_bottom_y` pair (special-casing the first and last blocks) as the direction. That direction
turned out to be wrong, and calibration proved it before it shipped: the window that recovers a
COMPLETE first card is 554..696 on one profile and 410..478 on the other — DISJOINT, so no single
declared constant, and nothing derived from `identity_top_name_band`, can serve both. The
recommended constant (600, `identity_top_name_band`'s own bottom edge) is correct on the first
profile and silently truncates the second profile's card by 121px (853 of its 974px) while `ok`
stays True. Worse, the self-check meant to catch a wrong declaration caught only 22 of 139
deliberately-mis-declared frames: it can prevent a confidently wrong extent sometimes, but it
cannot detect a wrong row reliably.

The shipped fix (`operation_love/drivers/segment.py`) takes no declared row at all. Hinge cards
are rounded rectangles, and the row projection this section already needs (the colour/span test
above) measures a card's own corner arc for free: at a top edge the non-background span starts at
`card_width - 2r` and grows monotonically to the full card width exactly `r` rows later, where `r`
is the corner radius (a bottom edge is the mirror). Two independent estimates of `r` — one from
the edge row's own span, one from how many rows the ramp takes to reach full width — have to
agree, which is what makes the test self-evidencing: it needs no background above the row and no
declaration from the caller at all (see `_corner_radius` and "THE THIRD PIECE OF EVIDENCE" in
segment.py's module docstring for the full derivation and corpus numbers).

Item 1 and item N are consequently both resolvable now, which is the point this failure class said
could not be done without special-casing. With the corner test and no declaration, the same two
profiles' scroll-top frames cut at their true chrome gaps and return item 1 complete at 697..1671
and 479..1453. Across the whole corpus item 1 goes from complete on 0 frames to complete on 4 of
the 6 frames it appears in (profile A) and 1 of 4 (profile B), and item N — which needed the mirror
declaration under the old design — goes from 0 to 58 of 63 (profile A) and 1 of 4 (profile B). The
remaining misses are frames where the card's far edge is genuinely off the analysed band; those
report an honest PARTIAL block rather than a silently wrong extent.

### 5.5 Closed-loop scrolling

Segment-by-hearts and snap-the-scroll are not alternatives. To snap to a card boundary you must
know where the boundary is, which requires the detection anyway. Card-snapping is heart
detection plus a closed loop on the scroll.

Scroll roughly one item per step, with randomized jitter around the boundary. Against both
criteria it wins: truncation drops to near zero because the cap becomes 12-14 *items* rather
than arbitrary screen fractions, and targeting accuracy improves because one dominant item per
frame removes the ambiguity behind today's bug.

The anti-bot objection does not survive scrutiny. A content-following scroll distance varies
with card height, which varies per item and per profile, and that is a *wider* distribution
than a fixed screen fraction. With jitter on top it has strictly more entropy than what runs
today, so the randomization rule is satisfied rather than strained.

Navigating back: scroll to top, then step forward counting distinct hearts, tap the k-th.

Two dependencies here that an earlier draft of this doc got wrong, both load-bearing.

**Counting does not escape the scroll ledger, it relocates the dependency.** The zero point is
"scroll to top", and `_scroll_to_top` (`hinge.py:1850-1948`) is itself ledger-derived: its swipe
ceiling is `len(ledger) + 3`, and whether it actually arrived is a pixel-diff settle heuristic
whose return value **every pre-existing caller ignores**. Counting forward from a false top
gives a systematic off-by-N. So the design needs an affirmative top confirmation before
counting starts, and one exists: at scroll-top `identity_band` shows Hinge's filter-chips row
rather than a name (`hinge.py:369-382`), which is a positive signal rather than an absence.
Confirm top by that signal, and treat failure to confirm as a hard stop.

**Heart dedup across frames cannot lean on `_vertical_shift_match` as specified.** That
function (defined `hinge.py:568`, called `hinge.py:3546`) was already measured live and
demoted. `ANTI-BOT-RESEARCH.md:356-368` records the finding: full-frame search never matches,
and even the content-band-restricted version still misses on real pairs. The measurement table
in the function's own docstring (`hinge.py:596`) has the numbers, including a real pair at
11.94 against a threshold of 9.0, and the docstring's own conclusion (`hinge.py:604-605`) is
that it is "corroboration (layer 2), never the authoritative check."

Using it for a fine-grained pixel offset is a *more* precision-demanding job than the
same-or-different classification it already failed at.

This is the weakest joint in Part B and it needs resolving before implementation, not during.
The likely answer is to avoid cross-frame dedup entirely: with closed-loop scrolling at roughly
one item per step, count a heart when it appears in the lower band and has left the upper band,
so identity comes from scroll cadence rather than from pixel correspondence. That needs
validating against real captures before it is trusted.

#### Addendum 2026-08-12: the affirmative top confirmation exists, and it is a leaf function

This section's "the design needs an affirmative top confirmation before counting starts" is now
built: `operation_love/drivers/scroll_top.py`, `confirm_scroll_top(frame, identity_band=...)`.
Pure functions over frame bytes plus calibration parameters, on the same terms as `segment.py` /
`frameshift.py` / `item_index.py` / `item_crops.py` — no device, no driver state. Full derivation
and per-constant measurements are in that module's docstring.

It is the signal this section named. At a genuine scroll top `identity_band` shows Hinge's own
filter-chips row; the moment the card scrolls at all the sticky per-profile header covers that
strip with the person's name. Re-measured through the shipped `hinge._band` decode over all three
gitignored calibration captures: the five frames at a genuine top produce a BYTE-IDENTICAL band —
mean-abs distance 0.000 pairwise, across two different profiles and three sessions — while all
143 other frames land at 14.391 or further. That 0.000 across two different people is both why one
constant serves every profile and the proof the strip carries nothing about either.

**Three outcomes, not two.** `confirmed at top` / `confirmed NOT at top` / `cannot tell`, with a
dead zone between the two bounds (3.0 and 9.0 mean-abs grey levels, the second being
`change_threshold` itself). The dead zone is EMPTY on all 148 corpus frames; it exists to be
honest about a band the module has never seen, not to classify one it has. "Cannot tell" is never
"at top": `ScrollTopVerdict.__bool__` raises rather than defaulting to truthy, and
`require_scroll_top` turns anything short of a confirmation into `ScrollTopUnconfirmed`.

| Capture | Frames | Confirmed top | Which | Cannot tell |
|---|---|---|---|---|
| botscroll 231516Z (frac 0.16) | 24 | **1** | frame 1, distance 0.000 | 0 |
| botscroll 225314Z (frac 0.55) | 9 | **1** | frame 1, distance 0.000 | 0 |
| scroll 211209Z (hand-scrolled) | 115 | **3** | frames 1, 2 and 115, all 0.000 | 0 |

**Zero false positives**, and the two non-frame-1 confirmations are genuine tops rather than
misses. Frame 2 of the hand-scrolled capture is a MEASURED 0px from frame 1 (`estimate_shift`
returns delta 0 at confidence 1.0 — its manifest has frame 1 at 0.76s and frame 2 at 17.9s, so
the human had not started scrolling). Frame 115 is the last frame of that capture, where the deck
had been advanced: its `identity_top_name_band` is 94.15 from frame 1's, i.e. a different profile,
and it segments with the scroll-top shape (a heartless leading chrome block with page background
above it, then a `card_corner`-topped complete card). An independent cross-check against
`item_index._scroll_top_evidence`'s geometry agrees on every one of the 148 frames: nothing the
gate confirmed is contradicted by the geometry, and the gate refuses 3 frames the geometry alone
would have permitted — which is exactly the residual that section documents.

This does not replace `_scroll_top_evidence`; the two are complements. The geometry can only
contradict, and misses roughly 1 scroll position in 20. This confirms.

Residual, stated: the confirmation is "the sticky header has not slid in yet", which bounds the
scroll offset rather than pinning it at row 0. Measured on the hand-scrolled capture, the swap
happens somewhere in (0, 202]px — far below one card pitch (738..1027px measured), so it cannot
change any heart ordinal, which is the only thing counting reads.

One operability note. The strip shows the account's own filter chips, so an owner who edits their
filters invalidates the constant. The failure is a hard stop at the gate, never a false top, and
the fix is a re-measurement with the owner present rather than a softened threshold.

#### Addendum 2026-08-12: the closed loop itself now exists, and the step is sized per frame

This section's closed loop — and 5.10.1's "step at most about a third of the LOCALLY MEASURED
card spacing" — is built: `operation_love/drivers/scroll_step.py`, `plan_scroll_step(segmentation)`
returning a `ScrollStep` whose `.frac` and `.x_frac` go to the driver's existing
`_scroll_down_one`. A leaf module on the same terms as `segment.py` / `frameshift.py` /
`item_index.py` / `item_crops.py` / `scroll_top.py`: it has no touch access at all, so the only
thing it can produce is a number and the humanized gesture path is the only way to spend it. Full
derivation and per-constant measurements are in that module's docstring. **This is the change
that makes Part B runnable at all**, because at production's 0.55 the index refuses every profile.

**The measurand is heart-bearing spacing, and one exclusion is load-bearing.** Three
measurements are taken off the current frame and the SMALLEST wins: two like glyphs on one frame
(exact, needs no block edge), a heart-bearing block's top to the top of the block directly below
it (exact, and it reaches a card whose own heart is below the band), and a complete heart-bearing
block's height plus one gutter (a lower bound, and the workhorse). A heartless block never
BEGINS a period. Without that rule profile A's 215px vitals block contributes a 268px top-to-top
pitch, which would size the step at 89px — ~112 captures for one profile, and then a hard stop
under the minimum legal gesture. Two things say 268 is not a real period: aliasing needs a
repeated structure, and the 363px cadence 5.10.1 validated ran straight over that block at a
ratio of 1.35 and still recovered all 9 of profile A's items.

**A previously unrecorded transport fact, and it is exact.** `_scroll_down_one` takes a fraction
of screen height and the content does not move by that fraction. The 0.16 capture's 384px drag
moves the content 363px; the 0.55 capture's 1320px drag moves it 1299px. **384 - 363 = 1320 -
1299 = 21px**, on two cadences 3.4x apart, to the pixel — a fixed offset, not a gain, and equal to
Android's 8dp touch slop at this device's 2.625 density. Anything converting a desired pixel step
into a `frac` must subtract it or it under-steps by 21px every gesture.

**Jitter lives in the ratio, not in the pixels**, so this section's entropy argument survives:
the step is `uniform(0.26, 0.36) x local spacing`, so the delivered distance is the product of the
card in front of us and the draw, and a tall card jitters over a proportionally wider pixel range
than a short one.

Measured on the same gitignored corpus — every frame segmented and planned 200 times, 29,600
plans over 148 frames and two profiles, then again as the ordered loop:

| | Result |
|---|---|
| Chosen steps | **219..363px, 139 distinct values**, mean 301 (memoryless) / 235..262 (with the loop's memory), against the single 363 that `frac 0.16` emits and the 1299 that 0.55 emits |
| Step/spacing ratio vs the spacing each plan measured | **max 0.360, median 0.31, not one at or above 0.4** — against 1.26 for the cadence that fabricated the phantom |
| Refusals | **0** — the smallest spacing anywhere in the corpus is 738px against a 608px refusal threshold |
| Frames offering no measurement | 16.2%, all planned at 219..265px, strictly under any measured plan's ceiling |
| Cost | 36 scrolls for profile B's 8349px, 42 for profile A's 10027px, against the probe's 23 — ~1.6x, inside 5.10.1's "~3x" budget |

Three residuals, stated rather than smoothed over.

- **The below-the-fold card.** The local measurement cannot see the card under the fold, so the
  step is also capped at `_MAX_STEP_PX` = 363 — not a round number but exactly what `frac 0.16`
  delivers, i.e. the only cadence with end-to-end evidence, whose ratio against that same
  capture's smallest 738px spacing is 0.49 and which measured 23 of 23 pairs clean. The loop
  additionally threads the smallest spacing seen ANYWHERE on this profile back into the next
  plan, so a short card costs at most the one step during which it was still below the fold.
  Measured: ~3 steps per profile exceed 0.36 of the profile's eventual minimum, on all three
  captures, and none after the running minimum has met its first short card.
- **A spacing below ~608px is refused, loudly.** The floor is not a constant but a fact about the
  humanized path: the smallest gesture inside the driver's own sanctioned read-scroll window
  (`_READ_SCROLL_FRAC_MIN` = 0.10) moves 219px, so a spacing under 219/0.36 cannot be enumerated
  without exceeding the ratio rule. 5.10.1's HYPOTHETICAL ~600px short prompt card is therefore
  outside what this driver can enumerate safely, and `plan_scroll_step` raises rather than
  stepping further. Nothing in the corpus comes near it.
- **`config.yaml`'s `read_scroll_frac` is untouched**, deliberately: 0.55 stays correct for the
  swipe-deck read path, which indexes nothing. The enumeration pass computes its own step and
  never consults it — but it does need a capture ceiling well above `scroll_captures`'s shipped 8
  (the validated probe ran at 48), which is the caller's configuration problem.

Still out of scope and unbuilt at this point: counting navigation and the `at_scroll_top=False`
refusal in `heart_ordinal_for` (5.3's third blocker), the model schema change, the prompt rewrite,
post-tap verification, anchor removal, the observe inversion.

#### Addendum 2026-08-12: counting navigation exists, and 5.3's third blocker is closed

This section's own sentence — "scroll to top, then step forward counting distinct hearts, tap the
k-th" — is now built up to the comma before "tap":
`operation_love/drivers/item_nav.py`, `navigate_to_item(driver, index, model_index) -> ItemTarget`.
It returns the tap POINT and the evidence for it and never taps; the tap, 5.6's post-tap check and
the hard stop on a miss are the next layer's. Full derivation and per-constant measurements are in
that module's docstring.

**It is the first module of this family that is not a leaf**, and deliberately the only one: it
captures frames and issues gestures because "navigate" is a verb about a device. Every decision it
makes is still made by a leaf — `scroll_top` for the top, `segment` for the hearts, `frameshift`
for the page space, `item_index`'s own folding for the count, `scroll_step` for the distance — and
the only two device verbs it uses are `_screencap` and the driver's humanized scroll methods, so
the ledger, the jitter and the forbidden-zone guard all still apply. It is a module rather than a
`HingeDriver` method for the same reason `scroll_step` added none: hinge.py churn is being kept
minimal while other Part B work edits it.

**5.3's third blocker is closed, in two places rather than one.** `ItemIndex.heart_ordinal_for`
now RAISES on an `at_scroll_top=False` index instead of answering with a confident int — that is
the accessor a navigator calls, so that is where the enforcement belongs — and
`navigate_to_item` refuses such an index before it moves the phone. `translation` still answers,
deliberately: it describes the capture, `item_crops` compares its own renumbering against it, and
a validation pass reads it without navigating anywhere. With this and the top gate in place,
`at_scroll_top=True` has exactly one legitimate source: a `ScrollTopVerdict.confirmed`.

**The count is re-derived and then cross-checked, not read off the index.** Scrolling to a
distance computed from the index would be weaker in two specific ways: the top confirmation bounds
the offset rather than pinning row 0 (residual measured in (0, 202]), so a page row from an
earlier capture is not a row in this one; and a computed distance has nothing to check itself
against. So this pass counts its own hearts and compares them with the index on every frame,
ANCHORED ON HEART 1 rather than on absolute page rows — both passes started at a confirmed top, so
both saw the same physical heart first, and heart-1-to-heart-k is a property of the page rather
than of either capture's origin. Every ordinal up to the target is checked, not just the target,
and the ANCHOR is checked too — a comparison built on the wrong first heart agrees with itself, and
for a target of ordinal 1 nothing else would catch it, so this pass's first heart must sit within
the gate's own (0, 202] residual of the index's heart 1. At the landing frame the card itself is
checked as well: its height and the heart's inset within it must match what the index recorded, or
the count reached the right ordinal on the wrong card, which is what a stale translation table
surviving a deck advance looks like (5.3's named reliability failure). The folding is
`item_index`'s own `_heart_clusters`, imported rather than reimplemented, so this pass counts
hearts by exactly the rule the index it is checked against counted them by.

Measured on the gitignored corpus, replaying `botscroll_20260811T231516Z` (profile B, 24 frames)
through a frame-serving double as though it were a live device. Geometry and counts only.

| | Result |
|---|---|
| Items navigated to, one at a time | **9 of 9 landed on their own heart** — page row exact to 0px on all nine, x exact, card height exact |
| Worst count-vs-index disagreement on any ordinal, any run | **0px** against a 16px tolerance (the replay reproduces the index's own frames, so this measures that the comparison is correctly anchored, not how much slack a live second pass needs) |
| Cost | 1 frame for item 1 up to 24 for item 9, against derived budgets of 9..49 frames |
| The 0.55 aliasing capture and the 115-frame hand-scrolled one | refused at `NAV_INDEX_UNUSABLE`, no gesture issued — inherited whole from `ItemIndex.usable` |
| A usable mid-profile index (translation (2, 3), `at_scroll_top=False`) | refused at `NAV_INDEX_RELATIVE` before the first scroll — blocker 3 on real frames |
| The same replay at the SHIPPED `ratio_window` | refused at `NAV_SCROLL_OVERSHOT` on the FIRST gesture |

That last row is the honest caveat and is worth stating rather than burying: the double advances
one frame per gesture, i.e. that capture's fixed 363px cadence, which is ratio 0.49 against its own
smallest 738px spacing and outside anything `plan_scroll_step` will plan. The validation widens the
ratio window to admit it. So the 9-of-9 is evidence about COUNTING; the cadence itself is validated
in `scroll_step.py` on the same corpus, and nothing here says 363px is a safe navigation step.

Twelve named refusal codes, every one a stop rather than a fallback (5.6's never-substitute rule):
index unusable, index relative, ordinal out of range, top unconfirmed, frame contradicts itself,
chain broken, scroll stalled, scroll overshot, heart missed, count disagrees with the index, item
never fully visible, budget exhausted. Two are worth calling out because they are new failure
classes rather than restatements. **Heart missed** catches the count's silent failure mode: a
forward scroll admits new content at one edge, so a heart first seen in frame `i` must have been
below frame `i-1`'s band, and one that appears inside a band that already covered it means the
matcher missed it and every ordinal below it is one too small. **Item never fully visible** is a
refusal rather than a tap, because the card's extent is what 5.6's check compares against and a
card whose edges were never seen has none.

The frame budget is derived from the index's own geometry and the smallest gesture the driver may
make, NOT from `scroll_captures` — that constant sizes the profile read (shipped 8) and the two are
different jobs. The enumeration pass's own ceiling problem, recorded in the addendum above, is
unchanged and still the caller's.

Still out of scope and unbuilt at this point: the tap itself and post-tap verification (5.6), the
hard stop on a navigation miss, the model schema change, the prompt rewrite, anchor removal, the
observe inversion. Recorded forward for whoever owns 5.6: crop signatures DRIFT on animated cards
(`item_crops.py` measured 0.018 grey levels on a static item against 25.0 on an animated one), so
that check cannot be a fixed threshold.

#### Addendum 2026-08-12: two corrections from an independent validation of the three modules above

An independent pass re-measured all three against the same gitignored corpus and could not break
the top gate (0 false positives over 148 frames plus the paywall control, 5 confirmations all at
distance 0.000, nearest non-top 4.8x the confirm bound away), the ratio rule (24,800 plans, span
0.259..0.3596, not one at 0.4) or counting (9 of 9 items reached, every one exact to 0px). It
found two things. Both are fixed; both are recorded here because each was a claim this doc or a
module docstring made and could not support.

**One: the step's jitter collapsed onto its own clamps, which is the fixed constant the owner rule
forbids.** `plan_scroll_step` drew the step/spacing RATIO uniformly and then clamped the resulting
distance into the legal gesture range. A clamp maps every draw outside the range to the SAME
delivered distance, so the realised distribution grew a point mass at each end: the gesture floor
below (219px, which a short card's `_STEP_RATIO_MIN` falls under) and `_MAX_STEP_PX` above (363px,
which a tall card's `_STEP_RATIO_MAX` exceeds). Measured, 5000 draws per spacing:

| local spacing | window | ratio-then-clamp | drawn in pixels |
|---|---|---|---|
| 609px | 219..219 | 1 value, 100% | 1 value, 100% — forced by the device |
| 620px | 219..223 | 5 values, **95% on one** | 5 values, 21% |
| 738px (corpus minimum) | 219..265 | 46 values, **40% on one** | 46 values, 4% |
| 1027px | 219..363 | 92 values, 7% | 92 values, 2% |
| 1166px | 219..363 | 57 values, **34% on one** | 57 values, 4% |
| 1400px | 219..363 | 2 values, **87% on one** | 139 values, 2% |

The fix keeps the ratio rule where it belongs — as the two ENDS of the draw window, so the
distance still follows the card in front of us, which is 5.5's entropy argument — and draws
uniformly over the integer pixels inside it. Over the three captures in order, threading the
profile's smallest measured spacing, the single most common gesture distance goes from 24-37% of
all gestures to 2-4%, and the cost is unchanged or slightly better (35 scrolls to cover profile
B's 8349px against 34; 43 for profile A's 10027px against 42, because the floor's point mass was
dragging the mean step down). No ratio moved: corpus maximum 0.3596 before and after, none at 0.4.
Two window edges are now explicit rather than emergent — the gesture floor wins over
`_STEP_RATIO_MIN` (below ~842px of spacing), and `_MAX_STEP_PX` wins over it too (past ~1396px,
where the whole ratio window sits above the ceiling and the window's low end drops back to the
floor instead of collapsing onto 363). At ~609px of spacing the floor MEETS the ceiling and one
distance is all the device can legally deliver; that is not fixable at any layer, so the plan
reports it (`ScrollStep.window_px`, plus a sentence in `reason`) rather than presenting a forced
constant as a draw.

**Two: `item_nav._confirm_landing` does not catch an index built against a different profile, and
its docstring claimed it did.** Driving profile A's index over profile B's real frames, model
index 1 returned a tap target: the anchor gap was 218px against a 218px bound on a strict `>`, and
all three of the landing comparisons matched IDENTICALLY (974px card height, 885px heart inset,
x=938) because Hinge's cards are stereotyped — "the wrong card under the right ordinal" is not
what an ordinary foreign profile looks like. Model indices 2..9 all refused, as did the reverse
direction; the hole is ordinal 1 specifically, where the count's own k=1 comparison is zero by
construction.

Fixed at this layer as far as it can be, and the residual is 5.6's (see its addendum):

- the anchor test is now `>=`, which is what the measurement says it always should have been:
  `_TOP_ORIGIN_RESIDUAL_PX` = 202 is the first offset the top gate was measured to REFUTE, so two
  CONFIRMED tops differ by strictly less than it and a gap of exactly the bound is already
  impossible. That catches the measured pair, by 0px rather than missing it by 0px;
- the count/index cross-check now compares EVERY ordinal the count has reached rather than
  stopping at the target, so a target of ordinal 1 gets a real comparison whenever a second heart
  is in view instead of only the trivially-zero one;
- `_confirm_landing`'s docstring now says what it actually catches (the right ordinal on a
  differently-SIZED card) and names what it does not.

Nothing else changed. The top gate, `_MAX_STEP_PX`, `_STEP_RATIO_MAX`, the twelve refusal codes and
`config.yaml`'s `read_scroll_frac: 0.55` are all untouched. Suite 1593 -> 1601 green, the 8 new
tests all on synthetic fixtures (including one that PINS the residual above, so that closing it
has to come back through this document).

### 5.6 Verification and the hard stop

Dropping the anchor from the prompt does not weaken verification, it strengthens it. Once
items are cropped and indexed, confirming the opened sheet is a **deterministic signature
match** between the sheet's displayed item and the stored crop of item N. No model call, no
cost, no judgment. Today's `on_target` is a pre-tap screen check that can report success while
the tap lands on the neighbour; this is a post-tap content check, which is the thing that was
actually missing.

**Owner rule: never substitute a different item.** If we cannot land on the item the model
chose, even after multiple tries, the run stops. No falling back to `hearts[0]`, no "closest
reachable item", no rewriting the opener to match whatever we hit. The fallback at
`hinge.py:2481-2486` comes out.

On a genuine miss: do not type, do not send, never send a commentless like, stop the run with
intended and actual item both recorded, and leave the screen where it is for debugging. This
matches the existing stop-condition rule, and the hub already has the surface to show it.

Consequence to accept: navigation reliability becomes a hard gate on throughput and we have no
measurement of it today. Observe mode cannot supply one, since the human drives the phone. The
failure rate will be discovered by the bot stopping, which is the intent. If that becomes
tiresome, a dry-run mode that navigates and verifies without tapping, then passes, would
measure it over a batch at the cost of burning those profiles.

#### Addendum 2026-08-12: what counting navigation does NOT pre-screen for this section

Recorded here rather than in a handover note, because the temptation this addendum exists to head
off is precisely the one a later reader of 5.5's "counting navigation exists" addendum would have.

**The stale translation table is still yours, and at ordinal 1 the crop-signature check is the
ONLY guard.** 5.3 says a stale table is "a reliability bug rather than a safety one" because "the
stored crops are stale too, so the post-tap signature check in 5.6 compares the opened sheet
against the wrong reference, fails, and stops the run". That argument is intact and it is now
load-bearing rather than belt-and-braces: measured on the corpus, navigation returns a tap target
for model index 1 when driven with another profile's index (see 5.5's addendum for the numbers and
for the anchor fix that catches the one measured pair). Do not weaken the signature comparison and
do not assume navigation pre-screened identity — it cannot, because no geometry it has access to
distinguishes two Hinge profiles' stereotyped first cards.

Four more constraints this section inherits, all of them measured rather than anticipated:

- **A fixed signature threshold is ruled out.** `item_crops` measured re-observation drift at 0.018
  grey levels on a static item against 25.0 on an ANIMATED one, with per-item nearest-neighbour
  distances of 44.7/47.9. Compare against the per-item `separable` / drift numbers each `ItemCrop`
  already carries, never against one constant (5.2's second 2026-08-12 addendum has the table).
- **The stop surface needs more than one exception type.** Routing on `ItemNavigationError.code`
  alone misses three that propagate uncoded BY DESIGN and were confirmed to propagate:
  `ScrollStepError` (a card spacing no permitted gesture can enumerate — any spacing at or below
  ~608px), `SegmentationError` and `ShiftEstimationError`, plus `ScrollTopUnconfirmed` for anything
  calling `require_scroll_top` directly. All four are correct hard stops; §8's stop surface has to
  route them.
- **`navigate_to_item`'s `**plan_kwargs` are an offline-validation door.** They forward straight to
  `plan_scroll_step`, so a caller can widen `ratio_window`, raise `max_step_px` or lower
  `fallback_spacing_px` and quietly leave the validated envelope. The offline replay needs exactly
  one of them (`ratio_window`, because that capture's fixed 363px cadence is otherwise refused at
  `NAV_SCROLL_OVERSHOT` on the first gesture, which is the correct shipped behaviour). A production
  caller passes NONE of them. Every plan's own window and bound are on `ItemTarget.steps`, so a
  stop record shows it if one ever does.
- **The enumeration capture ceiling is still unraised.** `config.yaml` has `scroll_captures: 12`
  and `hinge._capture_limit_for_profile` adds 0..2, so the real ceiling is 12..14 against the
  ~34-42 captures the closed loop needs for one profile. Until the caller raises it the enumeration
  read truncates every profile. `read_scroll_frac` stays 0.55 regardless — that is the swipe-deck
  read path, which indexes nothing.

#### Addendum 2026-08-12: the identity gate exists, and geometry was never going to be enough

The first of this section's two carried-forward requirements is built:
`operation_love/drivers/item_identity.py`, wired as the first thing `item_nav.navigate_to_item`
does. An `ItemIndex` now carries a fingerprint of the profile it was built from, and navigation
compares the screen in front of it against that fingerprint BEFORE it issues a scroll or a tap.
Full derivation and per-constant measurements are in that module's docstring. Post-tap
verification, the hard stop, the `hearts[0]` removal and the anchor removal are NOT in this
change and are all still ahead.

**The requirement was structural, so the fix is structural.** The fingerprint is a FIELD of the
`ItemIndex` (`identity`), `build_item_index` takes `identity_band` as a REQUIRED keyword, and an
index whose identity is UNKNOWN is refused by navigation exactly as a foreign one is. There is no
keyword on `navigate_to_item` that disables the gate. So "navigate with an index nobody checked"
is not a state reachable by forgetting a call; the only way to reach it is with an index that says
outright it does not know whose profile it describes, and that refuses too. Two new refusal codes
join the twelve: `NAV_IDENTITY_MISMATCH` and `NAV_IDENTITY_UNCONFIRMED`.

**THE MATRIX, on the real captures.** Profile A's index (scroll 211209Z frames 0..56, 9 items,
translation 1..9) driven over profile B's frames, and profile B's index (botscroll 231516Z, 9
items, translation 1..9) driven over profile A's, for every model index in both directions:

| | Result |
|---|---|
| Cross-profile navigations attempted | **18** (9 items x 2 directions) |
| Refused | **18 of 18, all at `NAV_IDENTITY_MISMATCH`** |
| Gestures issued across the whole matrix | **0** — and 0 calls to `_scroll_to_top`, so the phone is never touched at all |
| Same-profile control, profile B | 9 of 9 still land on their own heart, page row exact to 0px, identity matched at **0.000** |
| Same-profile control, profile A | item 1 lands, identity 0.000; items 2..9 refuse at `NAV_SCROLL_STALLED` — a property of replaying a HAND-scrolled capture one frame per gesture (that capture has near-static pairs), reached well past the gate, and unchanged by this work |

Before this change the same matrix refused 9 of 18 and let the other 9 through to a count.

**Two things the corpus changed about the design, and both were bugs caught by measuring rather
than by review.**

*One: `change_threshold` does not separate two people.* The gate initially used 9.0, on the
reasoning that `hinge._identity_of` decides same-person / new-person on it. Measured through the
shipped `hinge._band` decode at the shipped `_IDENTITY_DS` grid, the two calibration profiles'
settled sticky headers are **7.287** grey levels apart — inside 9.0, so the gate would have called
two different people the same person and passed for the exact case it exists to refuse. The
shipped bound is **3.0**, which is `scroll_top._CONFIRM_MAX_DIST` (one tolerance on this strip
rather than two). The table it sits in:

| Measured on the gitignored corpus, mean-abs grey levels at 64x16 | |
|---|---|
| the same profile's header against itself, every scroll offset | **0.000** on all 143 refuting frames, four captures, two profiles |
| the out-of-likes paywall's band against profile B's header | 4.289 (the nearest non-match of any kind) |
| profile A's header against profile B's | **7.287**, every frame, both directions |
| that paywall against profile A's header | 9.496 |
| profile A's own header MID-SLIDE-IN | 14.482 |
| the filter-chips row against either header | 15.7..19.5 |

There is no measured nonzero same-profile distance at all, so the bound is set entirely by what
must be kept out, and 3.0 clears the nearest by 1.29. The consequence is stated rather than left
to be discovered: at this grid a header redrawn a few pixels over cannot be tolerated (`scroll_top`
measured a 4px shift on this band at 6.67, which is larger than the 4.289 separating a paywall
from a person), so the gate is deliberately intolerant of drift and a redraw is a false STOP. That
is the safe direction and it is what the owner rule asks for.

This also says something about `_identity_of` that was not previously recorded: at 9.0 it would
call these two profiles the same person. It is safe THERE and is deliberately left alone — that
method's errors are asymmetric by design ("a false 'same' costs at most a missed pass", and the
deck-ready, settle and content checks downstream still have to agree) — but it is worth knowing
that the observe-mode anchor is weaker than its 0.00-vs-17.95 device measurement suggests.

*Two: the fingerprint cannot be the first frame that shows a header.* The obvious rule — take the
first frame whose band refutes the scroll top — picked, on profile A, a TRANSITIONAL frame caught
mid-slide-in: the only frame of 112 that disagrees with the rest, 14.482 from the settled header
the other 111 share to **0.000**. An index fingerprinted from it refuses its OWN profile at every
later frame, which is a guaranteed false stop on a perfectly good read. The shipped rule requires
CORROBORATION — the reading must be agreed with by another refuting frame, and the agreeing group
must be a strict majority of them — which is `item_index`'s own "two frames that both observed one
fixed quantity are two measurements of it, and disagreement is a failure rather than an average",
applied to the header. With it, profile A fingerprints from frame 3 with 54 agreeing frames and 1
recorded outlier, and profile B from frame 1 with 23 agreeing and 0 outliers.

**What makes the fingerprint trustworthy is that the capture had to show BOTH states of the
strip.** Some frame must CONFIRM the scroll top (which proves `identity_band` really is the rect
Hinge draws its own profile-independent filter chips over) and a later one must REFUTE it (which
proves that same rect carries something else once scrolled, i.e. the person). A rect that never
confirms yields no identity at all, and that is not fussiness: a fingerprint taken off a rect that
happens to hold static chrome would be identical on every profile, so the gate would pass for
everybody. A false sense of safety is worse here than no check, and this is the only structure
that rules it out without a second calibration.

**A PRECONDITION THIS IMPOSES ON WHOEVER WIRES THE TAP, and it is not optional.** Identity is not
readable at a scroll top: there the strip is Hinge's own chrome, byte-identical across two
different people (0.000), so there is nothing to tell anyone apart. Navigation must therefore be
entered from where the enumeration read leaves the card — SCROLLED, sticky header showing.
`hinge.like()` calls `_scroll_to_top()` before it locates a heart today; the navigation call has
to go BEFORE that, or every run stops at `NAV_IDENTITY_UNCONFIRMED`. `navigate_to_item` performs
its own affirmative return-to-top anyway, so there is nothing for a caller to do first.

**Refused at capture time as well as at navigation time.** `hinge._index_captured_items` now
refuses a capture whose identity is unknown, on the same terms as every other enumeration failure
(a sentence on `Profile.items_unavailable`, the ranker's frames untouched, worker.py's stop before
the model is asked anything). The reason is a billed call: an index that navigation will refuse at
its entry gate must not buy an opener first.

**This does NOT replace the post-tap crop-signature check, and the second carried-forward
requirement is untouched.** The gate rules out the wrong PERSON before any gesture; 5.6's
signature comparison rules out the wrong ITEM after the tap, against a reference this module does
not hold. Crop signatures still drift on animated cards (0.018 static against 25.0 animated), a
fixed threshold is still ruled out, and nothing here pre-screens any of it.

Offline only — no device and no live API call. `ops/calibration/` was READ for the matrix above,
which is a read of already-captured frames on this machine: nothing was copied out of that
directory, no frame or crop or name or photo description appears in any tracked file, and nothing
reached a remote. The new tests are synthetic end to end, on the same terms as every other Part B
test file: a painted 1080x2400 frame, two flat "sticky headers" 40 grey levels apart, and the
shipped `_SCROLL_TOP_BAND_FINGERPRINT` painted back in so the real scroll-top half runs with its
real reference.

Suite **1697 → 1723 green**: 20 new tests for the gate itself, 4 in `test_item_nav.py` (including the synthetic form of the matrix above, both directions, every model index) and 2 in `test_hinge_item_capture.py`. The one that used to pin the HOLE — `test_the_residual_a_nearer_foreign_profile_is_not_caught_here_and_5_6_owns_it`, whose own docstring said "if a later layer closes this at navigation time, delete this test and say so in 5.6" — was rewritten rather than deleted, so the closure is pinned by the same case that used to prove the gap, and its geometric half is kept as the control that shows what the gate had to exist for.


#### Addendum 2026-08-12: post-tap verification exists, and the sheet turned out to be measurable

The second of this section's two carried-forward requirements is built:
`operation_love/drivers/item_verify.py`, wired into `hinge._like_comment_sheet` as the gate on
typing. Full derivation and per-constant measurements are in that module's docstring. The hard
stop on a NAVIGATION miss, the `hearts[0]` removal and the anchor removal are NOT in this change
and are all still ahead; so is wiring `item_nav.navigate_to_item` itself, which is what will
finally hand the driver a model item number in production.

**The corpus this section said it did not have was on the machine all along.** 5.2's addendum ends
"5.6's real comparison is crop-against-like-sheet, which no capture in the corpus contains", and
every threshold argument since has been made without one. Six real screencaps of an OPEN Hinge
comment sheet were sitting in the gitignored local debug directory (`observe_like_anchor`, which
observe mode has been capturing since 2026-08-10), together with the profile frames captured
either side of each. Everything below is measured off them. Geometry and grey levels only.

**The sheet renders the card, and it renders it BOTTOM-ANCHORED.** That was the finding that
mattered, and the obvious comparison — the whole preview against the whole stored crop — is wrong
by an order of magnitude:

| distance from the preview to the card it renders, at the stored 32x32 grid | four real sheets |
|---|---|
| against the WHOLE card | 9.296 / 18.836 / 1.849 / — |
| against the card's BOTTOM `round(preview_h * card_w / preview_w)` rows | 1.011 / 0.533 / 1.849 / 4.600 |
| against the same number of rows from the TOP | 20.108 / 31.661 / 1.849 / — |

The third sheet's card fitted whole, so its three numbers coincide; that is the control. The rule
needs no constant — the scale is the stored crop's own width over the located preview's — so a
different card width or a different screen re-derives it rather than inheriting it.

**The remaining 4.600 was a 1% scale error, not a content difference.** Sweeping the derived
window height by ±2% and taking the minimum turns those four into **1.011 / 0.533 / 1.846 /
0.946**; the outlier was an 8-row miss on the tallest card. The same sweep moves the WRONG cards
in those frames by at most 0.9 grey levels (52.024→51.738, 58.858→58.058, 61.128→60.928,
59.354→58.772), because a card that is not on the sheet is not nearly aligned with it at any
window height. So the sheet's own reproduction penalty is **≤1.85** at the stored 32x32 grid (and
≤1.91 at the 64x64 grid verification runs at) against a nearest wrong card of **≥51.7** — a 27x
margin, and the number this section never had.

**Sheet geometry, for whoever has to re-measure it after a Hinge update.** The preview spans
columns 95..985 (890px) on five of the six sheets and 91..989 on the sixth, top row 236 on five
and 338 on the sixth; the comment box below it is 846px wide and 178 tall, the Send Like row 799
wide and 109 tall. The preview is the only element both ≥870px wide and ≥300px tall, and the
locator's third test — that it is indented to 95±15 — is what tells a sheet from a PROFILE screen,
whose cards sit at 53. Measured: 6 of 6 sheets located, **139 of 139 profile frames refused**.

**A FIXED THRESHOLD IS RULED OUT, WITH THE MATRIX.** Every re-observation of every item on both
calibration profiles, compared against all nine stored signatures of its own profile:

| | profile B (11 re-observations of 8 items) | profile A (31 re-observations of 8 items) |
|---|---|---|
| diagonal (own crop) | 0.002 .. 2.656 | 0.000 .. 25.009 |
| off-diagonal (the other eight) | 4.670 .. 154.577 | 6.266 .. 150.288 |
| nearest stored signature was the right item | 11 of 11 | 31 of 31 |

42 of 42 correct, and yet the diagonal MAXIMUM (25.009) is five times the off-diagonal MINIMUM
(4.670) — the two distributions overlap, so no single number separates them. **The shipped bound
is per item and derived rather than tuned: half that item's distance to its nearest other item.**
`CropSignature.distance` is a scaled L1 norm, so by the triangle inequality a sheet within
`nearest_k / 2` of item k is the unique nearest stored item; the argmin is corroboration, not a
second threshold.

**End to end, through the shipped code, on the real crops.** Each item's real re-observation
re-rendered as a sheet and verified as each of the nine numbers in turn, at the 64x64 grid the
module verifies at:

| | profile B | profile A |
|---|---|---|
| own number verified | **11 of 11** | **30 of 31** (the 31st is the animated card, below) |
| any WRONG number accepted | **0 of 88** | **0 of 248** |
| diagonal / off-diagonal | 1.085..4.998 / **9.166**..155.963 | 2.178..31.424 / **9.962**..137.226 |

And on the six real sheets, against a payload of every distinct card captured around each: four
recovered their own card at **3.0..4.0** with the runner-up at **52.6..63.9** and exactly one
MATCH each; the other two never captured their source card in a non-sheet frame at all, and both
were correctly refused rather than matched to the nearest thing available — one of them on a
60.126-versus-60.319 near tie, which is precisely the case an argmin alone would have got wrong.

**The animated card gets explicit detection and a different path, not a wider band.** 5.4 offered
"a tolerance band, or explicit detection and a different path". The band is rejected on the
numbers: one wide enough to accept profile A's animated card is 25 grey levels wide, which is five
times the 4.661 separating the two most alike items on profile B, i.e. wide enough to accept a
wrong item. `verification_blocker` runs BEFORE the tap and refuses when
`sheet_render_drift + the item's own measured drift >= half its nearest-neighbour distance`. Over
both profiles' 18 items it refuses exactly **one** — 25.009 + 1.95 against 23.931 — and the
tightest pass clears by 0.36 grey levels. The refusal costs a stop with the screen untouched,
which is what the owner rule asks for.

**The grid is 64x64 here while the stored crops stay at 32x32**, because the module re-cuts its
references into windows anyway and the choice is therefore free. The swept sheet penalty barely
moves with the grid (1.816 / 1.849 / 1.878 / 1.904 / 2.015 at 24/32/48/64/96) while the closest
distinct pair grows (4.372 / 4.661 / 5.990 / 8.024 / 10.021 on profile B), so the ratio the check
lives on goes 2.41 / 2.52 / 3.19 / **4.21** / 4.97. 64 doubles the worst-case separation for a 3%
rise in the penalty and is the finest grid `item_crops`' own published table measured.

**What changed in the driver, and what deliberately did not.** `like()`/`_like_comment_sheet` take
a new `model_item_index`; given one, the sheet is verified before a character is typed and `on_target`
decides nothing — the anchored re-ask is not offered, because repairing the TEXT does not undo
spending the LIKE on an item the model never chose. Without one, the legacy capture-order path is
byte-for-byte what it was, including the repair hatch, because nothing hands the driver a model
item number until counting navigation is wired. `_locate_target_heart`, `hearts[0]`,
`_ANCHOR_SYSTEM`, `anchored_opener`, `worker.py` and `config.yaml` are all untouched.

Offline only — no device and no live API call. `ops/calibration/` and `data/hinge_debug/` were READ
for the numbers above, which is a read of already-captured frames on this machine: nothing was
copied out, no frame or crop or name or photo description appears in any tracked file, and nothing
reached a remote. The new tests are synthetic end to end — painted cards with a vertical ramp,
re-rendered through a painted sheet at the geometry above.

Suite **1723 → 1754 green**: 19 new tests for the comparison itself (`test_item_verify.py`,
including the NxN matrix over real crop machinery) and 12 for the driver's ordering
(`test_hinge_sheet_verification.py`, whose headline is that there is no path from a tapped heart
to `adb.text` that does not pass a match first).

**Still owed after this**, unchanged: the hard stop on a NAVIGATION miss and the `hearts[0]` /
`_await_button("like")` removal in `_locate_target_heart`; anchor removal; wiring
`navigate_to_item` (and with it `should_stop`, blocker 3); 5.8's pre-flight cross-check; the
observe inversion; the hub changes; and §8's stop surface routing `ScrollStepError`,
`SegmentationError`, `ShiftEstimationError` and `ScrollTopUnconfirmed` by type.

**Two residuals to carry.** (a) The bottom-anchored rule is fitted to two real cards that did not
fit the sheet whole (the third fitted, the other three never captured their source card). A future
Hinge that renders a top-anchored or centred window would produce a FALSE STOP, never a false
accept — measured cost 20.1 and 31.7 grey levels against bounds of 4 and up — so it fails in the
safe direction, but it is one measurement and re-measuring belongs with the owner present. (b) Two
items can be too alike for this comparison to have power at all: the closest pair measures 4.661
and 6.265 grey levels on the two profiles, which at the verify grid becomes 8.024 and 8.727
against a ≤1.9 penalty — a 4.2x margin that holds today and would not if a profile ever showed two
cards nearer than about 4 grey levels apart. That case refuses at `verification_blocker` rather
than guessing, and would show up as a stop rate rather than as a wrong like.

#### Addendum 2026-08-12: the substitution paths are gone, the hard stop is wired, and the anchor survives ONLY for observe

This section's owner rule — "never substitute a different item ... the fallback at `hinge.py:2481-2486`
comes out" — is now enforced by the code rather than by the sentence. Three removals and one
addition, all offline, no device, no API call, nothing read from `ops/calibration/`.

**`_locate_target_heart` returns the chosen item's heart or it raises. There is no second return
value any more.** Four routes used to end at `hearts[0]` / `_await_button("like")` — an
out-of-range index, an undecodable target signature, a search that never matched, and a matched
frame carrying no heart — each reported as `on_target=False` for the caller to repair the TEXT
against. All four now raise `HingeTargetingError`. Deleting the flag rather than always returning
True is the point: while it existed, "we are about to tap the wrong item" was a state the type
system permitted, and every caller had to remember to check it.

**Retrying the same item is allowed, and is what happens first.** `_TARGET_HEART_ATTEMPTS` = 2
whole searches, each from its own `_scroll_to_top()`, before the stop. The owner rule draws the
line at substitution, not at persistence: a shaky hand is not a wrong decision. 2 rather than 3+
because a repeat only changes one thing — it re-reads from a re-established zero point, which is
the over/undershoot the search's own `+3` frame slack (HINGE-05) was added for. A third attempt
repeats the same deterministic comparison over frames captured from the same top by the same
gestures, buys another profile's worth of dwell, and the honest reading of "the second read did
not find it either" is that the capture and the screen disagree. Every gesture is still the
driver's own `_scroll_to_top` / `_scroll_down_one`, so the ledger, the jitter and the
forbidden-zone guard all still apply.

**An opener with no item index at all is refused before any gesture.** `_like_comment_sheet`
raises on `opener and item_index is None and model_item_index is None`, above the `_scroll_to_top`,
so the screen is untouched. Not reachable from `worker.py` (which stops on an untranslatable pick
before calling the driver) and guarded anyway, because `like()` is a public driver method and that
is the one input combination that would otherwise attach real text to whichever heart happened to
be topmost. The same `None` WITHOUT an opener stays legal and unchanged — Hinge with
`opener.enabled: false` sends a plain like, and nothing is substituted because nothing was chosen.

**The hard stop is at the worker, and it renders as a stop rather than a crash.**
`base.ItemTargetingError` is a new driver-agnostic exception carrying `stage`
("navigate"/"verify"), `intended`, `actual` and `index_space`; `hinge.HingeTargetingError` is both
that and a `HingeActionError`, so nothing that has always caught this driver's action failures by
type changes behaviour, and `worker.py` catches the one failure class that has a specific stop to
render without importing a Hinge symbol. `_auto_loop` wraps its single `driver.like(...)` call,
takes the failure screenshot (which the generic handler would otherwise have taken), publishes
`state="stopped"`, `stop_kind="opener"` and a reason leading with INTENDED and ACTUAL, sets the
stop event and breaks — so the like is never recorded as a decision, nothing swipes after it, and
the screen stays exactly as the driver left it. `actual` renders as "never got far enough" rather
than as a number when nothing was reached, because "we could not reach item 4" and "we reached
item 6 while aiming at item 4" are different diagnoses. Both numbers are read off the exception's
fields rather than parsed back out of its prose.

Two honest notes on that stop. `stop_kind="opener"` follows the convention the two adjacent opener
stops already set, and the hub's branch for it is titled "opener capacity exhausted" — wrong for a
targeting stop, with the true reason carried as the subtitle. A dedicated kind plus its own banner
belongs with the hub workflow, and `hub.html` was deliberately not touched here. And the catch is
narrow: a stuck deck, a missed Send tap or any other unexpected action failure keeps its red error
banner and its traceback, because those are bugs rather than a rule being obeyed.

**THE ANCHOR IS NOT REMOVED, AND THIS IS THE PART A LATER READER MUST NOT MISREAD.** Doc 5.1 says
the anchor is dropped from the prompt in both modes. Half of that shipped here and half did not,
because the two halves were never the same mechanism:

- REMOVED, entirely: the `anchored_opener` repair callback. The worker's closure, the keyword on
  `base.Driver.like` / `hinge.like` / `_like_comment_sheet` / `_like_direct` / `bumble_web.like`,
  the re-ask branch and its "the replacement produced nothing" halt. In AUTO the anchor was a
  repair — a second billed call that rewrote the message to agree with an item the model never
  chose. Repairing the text does not undo spending the like. It was the last substitution path in
  the auto flow.
- KEPT, byte-for-byte: everything that BUILDS an anchored request. `_ANCHOR_SYSTEM`,
  `_ANCHOR_LABEL`, `_system_text`'s anchored branch, both anchored closing paragraphs, the
  anchored retry sentence, `_assemble_parts`' anchor placement, `generate(anchor=...)`,
  `OpenerService.maybe_opener(anchor=...)` and observe's call site. Observe mode still generates
  its opener AFTER the human taps, and there the anchor is not a repair of a wrong choice: the
  human MADE the choice by tapping that item's heart, so the anchor carries it. Removing any of
  the above would break the one path on which a real message reaches a real person today. 5.9's
  INVERSION is what retires it, and that is the next workflow. Until then observe is the only
  caller of the anchored shape anywhere, and every anchored comment in `opener.py`,
  `opener/service.py`, `hinge.py` and `worker.py` now says so instead of naming a repair hatch
  that no longer exists.

Every test that pinned a removed behaviour was rewritten rather than deleted, and several were
inverted in place so the file still shows what changed: `test_hinge_observe.py`'s five
fallback-and-flag assertions became stop assertions, its four `anchored_opener` like() tests became
"stops and sends nothing" plus a `TypeError` test pinning the keyword's absence,
`test_hinge_sheet_verification.py`'s legacy-path pair became "types its opener when targeting
lands" and "stops instead of repairing the text", and `test_worker.py`'s five ANCHORED_OPENER tests
became seven doc-5.6 stop tests (including one proving an ordinary driver failure is still an
error, and one proving observe cannot reach the new handler at all). Suite **1754 → 1761 green**.

**Still owed after this**, and unchanged: wiring `item_nav.navigate_to_item` (and with it
`should_stop`, blocker 3) so a model item number reaches the driver in production at all; §8's stop
surface routing `ScrollStepError`, `SegmentationError`, `ShiftEstimationError` and
`ScrollTopUnconfirmed` by type; 5.8's pre-flight cross-check; the observe inversion; the hub
changes (including `item_description` display and a targeting-specific stop banner). The residual
`_locate_target_heart` still carries is also unchanged and is stated in its own docstring: the
heart taken from a matched frame is the TOPMOST one on it, so a scroll position showing a photo and
a prompt at once can still land on the neighbour. That is a property of the capture-order space,
which numbers frames rather than items; counting navigation plus the post-tap crop check is what
closes it, and nothing in this change pretends otherwise.

#### Addendum 2026-08-12: post-tap verification accepted a FOREIGN card, and three statements in this doc are measurably false

An independent validation of the workflow above drove a stale payload for one profile against a
sheet rendering another profile's card, through the shipped `hinge._like_comment_sheet`. It
returned `VERIFY_MATCH`, typed the opener and SENT the like. **10 of 540 foreign comparisons were
accepted**, reproducibly across five resampling kernels; the reverse direction refused, but by only
1.118x. Nothing had to be un-shipped — the worker never passes `model_item_index` and invalidation
is thorough, so the path is unreachable in production — but 5.9 plans to make this same comparison
the SOLE guard on the one flow where a real message reaches a real person, so it had to close
first.

**THREE STATEMENTS ABOVE ARE WRONG. They are left in place per the standing rule and corrected
here.**

1. **5.3: "A stale table is a reliability bug rather than a safety one ... the post-tap signature
   check in 5.6 compares the opened sheet against the wrong reference, fails, and stops the run."**
   False, measured 10 times in 540. The correct reading is that a stale table is a SAFETY bug whose
   only real mitigation is the invalidation 5.3 also asks for, plus refusing to act on an item
   nothing can navigate to.
2. **`item_verify.py`'s docstring: "`verification_blocker` is deliberately the STRICTER of the two
   tests ... it errs towards refusing an item that would in fact have verified."** False whenever
   the rendered window pruned a candidate, which is the ordinary case for a short card — verify was
   up to 4x LOOSER than its own pre-tap prediction. True again after the fix below.
3. **`item_nav.py`'s `_confirm_landing` docstring: "doc 5.6's post-tap crop-signature check remain
   in force behind all three."** In force as an ITEM discriminator within one payload; never as a
   whose-profile-is-this backstop. It does not degrade gracefully into one.

**THE DEFECT HAD TWO INDEPENDENT MECHANISMS AND ONLY ONE OF THEM IS FIXABLE WITHOUT A DEVICE.**

*Mechanism one, a plain bug, fixed.* `nearest_other` — the quantity the whole accept bound is
derived from — was computed only over items tall enough to BE the window the sheet renders. An item
shorter than that window correctly gets no `distance` (it cannot be what is on screen) but it was
also losing its REFERENCE, so it fell out of every other item's neighbour set. In the measured case
a 756px item's two nearest neighbours were 685px, a 764px window pruned both, and its bound came out
**27.6 instead of the ~6.9** its own stored `nearest_item_distance` implies. `_compare_item` now
produces a reference for every item and withholds only the distance. The neighbour set is a property
of the payload; it must not depend on how tall the sheet in front of us happens to be.

Re-measured on the same real captures through the same harness, before and after that one change:

| foreign card on the sheet, `ops/calibration/` | before | after |
|---|---|---|
| profile A's cards against profile B's payload | 360 comparisons, **10 ACCEPTED**, tightest distance/bound 0.541 | 360 comparisons, **0 accepted**, tightest **1.463** |
| profile B's cards against profile A's payload | 180 comparisons, 0 accepted, tightest 1.118 | unchanged (no item on that payload was being pruned) |
| the unnumbered context block against its own profile | 99 comparisons, 0 accepted, tightest 2.94x | unchanged |
| within-profile, own number over every re-observation | 0 wrong numbers accepted; own number matched 20 of 20 (B) and 38 of 40 (A), the 2 being the animated card `verification_blocker` already refuses before the tap | **unchanged, exactly** — 0 wrong, 20 of 20 and 38 of 40 |

So the change costs nothing on the correct comparisons and closes every measured false accept —
which is what a bound that got looser for no reason should do when the reason is removed.

*Mechanism two, structural, NOT fixed and deliberately so.* `_SEPARATION_FRACTION`'s 0.5 proves
uniqueness among the STORED items — "if the sheet is within `nearest_k/2` of item k then k is the
unique nearest stored item". It says nothing about whether the sheet is any of them. There is no
absolute ceiling anywhere, so out-of-payload content only has to beat the payload's own internal
spacing. Closing that needs a second, ABSOLUTE term, and on this corpus the correct-item distances
(0.73..10.86) and the foreign accepts (14.91..15.24) leave a gap about 3.5 grey levels wide measured
on two profiles. Per the standing rule that is a calibration task with the owner present, not a
number to pick offline, so it is not shipped. It is recorded loudly in `item_verify`'s docstring
instead ("THIS IS A CLOSED-SET TEST"), and note the class is wider than another person's card: a
block the index resolved PARTIAL or UNCROPPABLE carries a heart, is tappable, has no crop, and is
therefore out-of-payload content from the SAME profile, where an identity gate says nothing either.

**THE DRIVER NOW ASKS WHOSE PROFILE THIS IS, BEFORE IT ASKS WHICH ITEM.** Since mechanism two
cannot be closed inside `item_verify`, the like path closes the case that mattered — a payload
describing somebody else — where it can be closed: `hinge._confirm_payload_profile` runs
`item_identity.compare_profile_identity` against the screen before the scroll and before the tap,
whenever a `model_item_index` is given. That is the same primitive at the same point in the
sequence that `item_nav.navigate_to_item` already uses as the first thing it does, so wiring
counting navigation inherits it rather than replacing it. Identity is unreadable at a scroll top,
which is exactly why it sits above `_scroll_to_top` and why "cannot tell" is a stop rather than a
pass. It is defence in depth and it is stated as such in its own docstring: neither guard is
sufficient alone, they fail in the SAME direction on the same input (see the identity numbers
below), and correct invalidation remains the guard that actually holds.

The validation's decisive case, re-run through the shipped driver against the real captures, with
the driver holding profile B's INDEX as well as its crops (they are set and cleared together, so
that is what a stale table really is) and the phone showing a real profile-A frame until the heart
is tapped:

| | before | after |
|---|---|---|
| stale B payload, phone on profile A, sheet shows A's card | **COMPLETED — the like was SENT**, opener typed, 3 taps | **STOP at the identity gate: 0 taps, 0 gestures, nothing typed** |
| control: same driver, phone on profile B, sheet shows B's card | sends | **still sends** — so it is a gate and not a blanket refusal |

**THE WIRING HAZARD BESIDE IT, ALSO CLOSED.** `_like_comment_sheet`'s pre-flight guard fired only
when `item_index` and `model_item_index` were BOTH None. With a model item number and no
capture-order index — exactly what a caller produces if it hands the driver the model's number and
leaves the other argument at its default, i.e. what wiring counting navigation naively looks like —
`_locate_target_heart(None)` read the None as "nobody named an item" and TAPPED THE TOPMOST HEART.
Measured: `scroll_to_top, locate, tap`, sheet open on a card the model never chose, then a stop on
the verification mismatch. Nothing typed and no like sent, so not a wrong like — but the phone was
touched, and given mechanism two it is the one shape where a false accept would also be a real send.
That combination now raises before the scroll, naming counting navigation as the missing piece.
`item_index=0` beside a model item number stays legal: 0 is a real capture-order target, so the tap
is aimed rather than defaulted. **Whoever wires `item_nav.navigate_to_item` replaces that refusal
with the navigation call — that is the handover.**

**AND A NEW FINDING THAT LANDS ON 5.5's IDENTITY GATE, WHICH IS WHAT THE CROSS-PROFILE MATRIX
RESTS ON.** `item_identity._IDENTITY_MATCH_MAX_DIST` is 3.0, justified above by a single measured
pair ("profile A's settled header against profile B's, **7.287**, ... 3.0 clears the nearest by
1.29"). Re-measured over six real profiles — the six `observe_like_anchor` sheets in the local
gitignored debug directory, each definitely a different person because a like advances the deck —
through the shipped `hinge._band` decode at the gate's own 64x16 grid:

| six real profiles, fifteen pairs | grey levels |
|---|---|
| the closest two DIFFERENT people | **2.565** — inside the 3.0 bound, so the gate MATCHES them |
| the next closest | 3.752 (clears by 25%) |
| the furthest apart | 6.891 |
| every sheet against its own profile's scrolled frames | **0.000**, all six |

So one pair in fifteen is a measured false accept, and the "1.29 grey levels of margin" was an
artifact of having two profiles. A finer grid does not rescue it (2.683 at 128x32, 2.681 at 256x64).
Same-profile distance is still identically 0.000 everywhere, so any bound in (0, 2.565) is correct
on all data anyone has — which is precisely why the number is NOT changed here: choosing among them
is the calibration act this doc keeps deferring to the owner, and tightening it trades a wrong-person
risk for a stop on every redrawn header (measured at 6.67 for a 4px shift). The constant's comment
now carries the table and the consequence.

The consequence is the part that matters for sequencing: **an `IDENTITY_MATCH` is a strong REFUSAL
mechanism and a weak CONFIRMATION one**, and `verify_sheet_item` cannot cover its gap because it has
itself been measured accepting a foreign card. The two guards 5.9 was going to stack are correlated
in the wrong direction. Neither hole is reachable today for the same reason: nothing in production
hands this driver a model item number, and the driver now refuses one it cannot navigate to.

**ONE MEASUREMENT THAT MAKES 5.9's JOB EASIER, though, and it was not previously recorded: THE
COMMENT SHEET DOES NOT OCCLUDE THE STICKY HEADER.** `identity_band` cuts rows 115..226 and the
sheet's preview starts at row 236. On all six real `observe_like_anchor` sheets the band reads
`confirmed_not_top` (it carries a person, not the filter chips) and sits **0.000** from that
profile's own neighbouring scrolled frames. So observe's inverted flow can run
`compare_profile_identity` on the very frame it is already holding — the deck-advance race 5.9
calls "the most serious defect found reviewing this design" is answerable on identity rather than
on the pixels of a card, at the cost of one comparison and no extra screencap. It is a refusal
mechanism, not a proof, per the table above; the point is that it is available where the design
assumed it was not.

**Also worth recording positively, from the same validation:** the cross-profile navigation matrix
is clean at 108/108 refusals with zero gestures and zero `_scroll_to_top` calls; the happy path is
9 of 9 exact to 0px; within-profile post-tap verification accepted no wrong item in 371 measurable
comparisons; no substitution path survives anywhere; humanization is intact (zero references to the
touch transport in all eight leaf modules); invalidation has four sites and `_current_sigs` is
written in exactly one of them; and observe cannot reach any of the new machinery.

Offline only — no device, no live API call. `ops/calibration/` and `data/hinge_debug/` were READ for
the numbers above: nothing was copied out, no frame, crop, name or photo description appears in any
tracked file, and nothing reached a remote. The new tests are synthetic end to end — a second
painted page whose look-alike cards differ in height either side of the window floor, plus a card of
the same family that is on no page the payload describes, which is accepted as a stored item before
the fix and refused after.

**Still owed after this**, unchanged except where noted: wiring `item_nav.navigate_to_item` (and
with it `should_stop`, blocker 3, and the replacement of the refusal above); §8's stop surface
routing `ScrollStepError`, `SegmentationError`, `ShiftEstimationError` and `ScrollTopUnconfirmed` by
type; 5.8's pre-flight cross-check; the observe inversion; the hub changes. **NEW and blocking for
5.9:** the absolute accept term for `verify_sheet_item` and a decision on
`_IDENTITY_MATCH_MAX_DIST`, both of which are measurements to take with the owner and a device
present rather than constants to pick.

### 5.7 Request and response shape

Sent:

- her name, as a string
- items 1..N, each a cropped image, photos and prompt cards alike
- context blocks, cropped, unnumbered
- a truncation flag when the capture hit the ceiling
- the style guide per Part A

Not sent: full screenshots, scroll frames, the anchor, endorsement blocks.

Returned: `item_index`, `referenced`, `angle` (free text, telemetry), `opener`,
`item_description`.

**`referenced` survives Part B and is not the same field as `item_description`.** `referenced`
is the full-text detail the opener is reacting to; it is what the redundancy monitor in 3.7
compares the opener against, what `store.record_opener(run_id, app, model, opener, referenced)`
writes to the `openers` column (`store.py:116-122`, a positional signature), and what the hub
renders as the "about:" caption. `item_description` describes the *item* and is scoped in 5.8
to a coarse type check. Dropping `referenced` would silently blank the telemetry column and
take the monitor dark.

`item_description` is always returned, in **both** modes. Auto ignores and logs it, observe
displays it. A mode-dependent schema would mean auto and observe issue different requests,
which quietly breaks the canary property that makes observe worth having.

It also buys a free deterministic cross-check on the segmentation itself, specified in 5.8.

Note `referenced_index` changes meaning, from an index into scroll frames to an index into
items. Breaking change to the field's semantics and to every line of prompt copy saying
"scroll order", several of which are pinned by tests.

#### Addendum 2026-08-12: the response contract shipped, and `referenced_index` is gone rather than renamed

The RESPONSE half of this section is built. `_SCHEMA` now declares, in this order,
**`item_index`, `referenced`, `angle`, `item_description`, `opener`** — `opener` still last, for
3.4's reason (thinkingLevel is minimal on every model in the cascade, so an earlier output field
is the only scratchpad that exists), and `item_index` FIRST because it is now a choice the opener
has to follow rather than a label attached to an opener that was already written. Emitting it
after the message would let the model write first and then name whichever item its message
happened to suit, which is the blind-then-repair order Part B deletes. `OpenerResult` and
`OpenerPick` carry both new fields, and both stores persist `item_description` beside `angle`
(a `_MIGRATIONS` ALTER in bigquery_store.py, a duplicate-column-guarded ALTER in store.py — the
live table holds real rows, so CREATE TABLE IF NOT EXISTS would never have reached it).

Three things worth recording because they were decisions, not transcription.

**`referenced_index` was DELETED, not renamed, and nothing falls back to it.** The paragraph above
says the field "changes meaning"; in code that framing is the hazard rather than the plan. Old and
new are both small ints, neither carries its base in its type, and they count different things
(0-based over raw scroll frames, where one card appears in several frames and one frame can hold
two cards, versus 1-based over numbered items). So the service reads `item_index` with no getattr
fallback to the old name: a stale producer degrades loudly to ABSENT instead of having a frame
index silently reinterpreted as an item number. That is 5.3's rule applied to the field itself.

**Zero is out of band, and it is a named constant.** `ITEM_INDEX_ABSENT = 0` is what a missing,
null, negative or unparseable index collapses to. Under the old 0-based contract the identical
coercion produced a confident, perfectly legal "the first item"; under 1-based numbering there is
no legal index it can be confused with. What a consumer must DO with it is deliberately not
decided there — 5.3's "treat a missing table as a hard stop, never a fixed coordinate" belongs to
the navigation workflow. The zero-numbered-items request says so in as many words rather than
asking for an index into an empty list.

**`OpenerPick.index` kept its name and changed its meaning, which the driver does not know yet.**
It now carries the model item index and is still handed to `driver.like(item_index=...)`, whose
parameter is the old capture-order space. That gap is documented at every site it crosses
(worker.py's call site, `OpenerPick`, `base.like`, `hinge._locate_target_heart`, `_capture_current`)
and is closed by 5.3's driver-owned translation table in the next workflow. It is bounded rather
than silent: `_locate_target_heart` reports `on_target=False` for anything it cannot honour, and
`_like_comment_sheet` then re-asks against the live anchor instead of shipping the text.

Verified live once against the real API (one dry run, synthetic colour panels, no device and no
real profile): the five-property `responseJsonSchema` is accepted, and gemini-3.6-flash answered
with an in-range `item_index` and an `item_description` matching the panel it picked. The first
attempt of that dry run returned `finishReason=MAX_TOKENS` purely because the throwaway client
omitted `thinkingConfig` — the schema itself never 400'd.

Still out of scope and unbuilt at this point: the request payload (numbered crops in place of
scroll frames), 5.8's pre-flight cross-check, 5.6's post-tap verification and hard stop, the
`hearts[0]` fallback removal, anchor removal from the driver, the observe inversion, and wiring
`item_index` into `_capture_current`.

#### Addendum 2026-08-12: the REQUEST half shipped too, and the numbering is carried by per-image labels

The other half of this section is now built, in `opener.py`: `ItemRequest` (her name as text, the
numbered item crops, the unnumbered context crops, the truncation flag) plus a
`generate(..., items=...)` request shape that sends those crops INSTEAD of `profile.photos`.
`profile` is still passed and still contributes its text; only its photos go unused, because the
caller that has crops also still has the frames and needs them for ranking and embedding.

Three things worth recording because they were decisions, not transcription.

**The numbering is stated by ADJACENT LABELS, not by a paragraph at the end.** 5.2's argument is
that crops make "image k is item k" true by construction — but that is a property of how WE build
the request, and the model still has to know it. Told only in the trailing text ("the images above
are numbered 1 to N in the order shown"), the model has to COUNT images to use it, which is an
inference step on exactly the enumeration this redesign exists to stop leaving to luck. So the
wire shape is: one preamble stating the convention, then for every image a standalone
`=== ITEM k ===` text part IMMEDIATELY before it, numbered items first and the
`=== CONTEXT, NOT NUMBERED, CANNOT BE PICKED ===` crops after them, then the trailing block
(style guide, her name, her profile text, the counts, `Set item_index to ...`). This is the same
placement rule, and the same reason, as `_ANCHOR_LABEL`. The imperative is deliberately NOT
repeated per label: N copies of "set item_index to k" would put a fresh instruction immediately
before generation with the last one read the freshest, which is the same recency mechanism that
produced 5/5 "I bet" openers in 3.6's addendum.

**The anchor path is untouched and still works, and the two shapes are mutually exclusive rather
than ordered by precedence.** 5.1 drops the anchor in both modes; removing it belongs with the
driver-side removal, and until then observe and the driver's repair hatch still take that path.
Passing both an anchor and an item list raises before anything is encoded or billed: the anchor
IS the choice and the numbered list asks for one, so any precedence rule would have to silently
demote a live instruction, and `_ANCHOR_SYSTEM` would ship a paragraph about an image nobody sent.

**`_fit_images_to_budget` kept its never-drop guarantee, and it matters more here, not less.** On
the frame shape a dropped image lost some of what the model could look at; on this shape it would
RENUMBER every item past the gap, so item 5's label would land on item 6's crop and the answer
would be confidently wrong with nothing downstream able to detect it. Same reasoning behind the
two `ValueError`s in `_assemble_parts` (an image-part count that disagrees with the request, and
an anchored item layout) and the one in `ItemRequest.__post_init__` (an empty or non-bytes crop).
An `ItemRequest` with zero numbered items is refused outright: 5.1's contract is that the model
CHOOSES, so a request offering none can only be answered with `ITEM_INDEX_ABSENT`, and paying for
a billed call to be told what we already knew is worse than raising.

Verified live once against the real API (one dry run: four synthetic colour panels, no device, no
real profile, nothing from `ops/calibration/`). The interleaved label/image part list was accepted,
and gemini-3.6-flash answered `item_index=2` with an `item_description` naming the shape that was
in fact panel 2 — i.e. it read the number off the label rather than deriving one — while leaving
the unnumbered context panel unpicked. The opener itself kept every Part A property (a falsifiable
claim, a hedge that was not "I bet", two sentences, plain ASCII, no dashes).

Still out of scope and unbuilt after this: NOTHING BUILDS AN `ItemRequest` YET. `OpenerService`
does not thread the parameter (unlike `anchor`), so the crop shape is unreachable from production
until the driver-side adapter from `ItemPayload` lands — that adapter, 5.3's driver-owned
translation table, 5.8's pre-flight cross-check, 5.6's post-tap verification and hard stop, the
`hearts[0]` fallback removal, anchor removal, the observe inversion, and wiring `item_index` into
`_capture_current` are all still ahead.

#### Addendum 2026-08-12: "bounded rather than silent" was FALSE, and the seam is now closed by a stated index space

The addendum above closes with "`OpenerPick.index` kept its name and changed its meaning, which
the driver does not know yet ... It is bounded rather than silent: `_locate_target_heart` reports
`on_target=False` for anything it cannot honour, and `_like_comment_sheet` then re-asks against
the live anchor instead of shipping the text." That sentence is wrong, it was repeated at five
code sites, and the next workflow's plan was resting on it. An independent validation pass
replayed `hinge._locate_target_heart`'s own branch condition against a 9-frame capture:

| `item_index` | what happens | `on_target` |
|---|---|---|
| 0 | first-heart fast path | **True** |
| 1, 4, 8 | capture-order frame 1/4/8, topmost heart | **True** |
| 9, 12 | first-heart fallback | False |

The guard is `item_index >= len(_current_sigs)`, so only the LAST item and genuinely
out-of-range values were ever caught. Every other 1-based item number was in range for the
frame list, resolved to a real, ADJACENT, WRONG frame, and came back on target — so the repair
hatch never fired and the comment landed one card down with full confidence and no `_dbg_action`
record. Two live defects fell out of the one false claim:

1. **An off-by-one on the shape that actually runs.** Nothing builds an `ItemRequest`, so auto
   still sends `profile.photos` and `_text_part`'s unanchored branch now tells the model the
   images are "numbered 1 to N". The model answers 1-based; `_current_sigs` is 0-based over the
   same frames. `HEAD`'s copy asked for "the 0-based index", which MATCHED. Gated only by
   `mode:`, one config line from live.
2. **`ITEM_INDEX_ABSENT` was a confident like of item 1.** `ITEM_INDEX_ABSENT = 0` reached
   `_locate_target_heart(0)`, took the index-0 fast path, and returned `on_target=True`. A model
   that could not pick got its opener attached to item 1 and sent — the opposite of 5.3's "treat
   a missing table as a hard stop, never as a reason to fall back to a fixed coordinate".

Neither is fixed by a ±1, and neither required building this section's remaining scope. The root
cause is one thing: **a small integer crossed a boundary into a different index space, and
nothing in its type said which space it came from**, so the receiving side could not tell a
converted value from an unconverted one. The fix states the space and refuses when it cannot
convert.

- `opener.py` gains `INDEX_SPACE_PROFILE_PHOTOS` / `INDEX_SPACE_MODEL_ITEMS` and
  `OpenerResult.index_space`, set by `generate()` from the payload it actually built (so there is
  one producer of the fact, not two that can drift). The anchor is deliberately not counted as a
  numbered item.
- `OpenerPick.capture_order_index` is the ONE sanctioned crossing. Profile-photo space converts
  (`index - FIRST_ITEM_INDEX`, exact: the request was built from `profile.photos`, and
  `_current_sigs` is index-aligned with it). Model-item space returns `None` until 5.3's table
  exists. `ITEM_INDEX_ABSENT` returns `None` in every space. Both defaults — on `OpenerResult`
  and on `OpenerPick` — are the UNTRANSLATABLE space, so a fake or a future call site that says
  nothing produces a pick nothing will tap from.
- `driver.like(item_index=...)` accepts `int | None`, defaulting to `None`. `None` means "nobody
  said which item" and is deliberately NOT 0, which is a legal first frame in the driver's own
  space; collapsing the two is exactly how defect 2 happened. `_locate_target_heart(None)` still
  taps the fallback heart (something must open the sheet) but reports **off target**, which
  routes the message through the `anchored_opener` repair, and logs `reason="no_target_index"`
  so a bug report can tell "never told" from "aimed and missed".
- `generate()` now RANGE-CHECKS `item_index` against the number of numbered items it actually
  sent, which is the cheapest fail-loud win here and had no home before: `_parse` never saw the
  request, and the driver counts a different thing entirely (its own frames — 24 of them for 9
  items, so every value 1..23 looked in range). Out of range collapses to `ITEM_INDEX_ABSENT`
  with a printed line rather than raising: the opener TEXT is not what went wrong, and five
  consecutive rejections stop a run, so reusing the one out-of-band value that is already
  handled everywhere beats inventing a second failure mode.

What this deliberately does NOT do: build 5.3's driver-owned translation table, remove the
`hearts[0]` fallback, add post-tap verification or 5.6's hard stop, remove the anchor, invert
observe, or wire `item_index` into `_capture_current`. All still ahead. The consequence is that
the crop shape's index is currently untranslatable BY DESIGN and every crop-shape like would
repair against the live anchor rather than target; that is the honest state until the table
lands, and it is now visible in the types rather than asserted in a comment.

One more thing this makes visible rather than fixes: a bug report now prints the index space
next to the index (`index: 3 (profile_photos)`), because a bare small integer in a human-read
artefact has exactly the ambiguity that caused the bug. Entries written before today carry no
space and render with no suffix rather than a guessed one.

Also closed while the request and the response were finally in the same scope: `_parse` now
accepts only an `item_index` that UNAMBIGUOUSLY names an item. `int()` was happy to turn `3.7`
into item 3 (the truncation invents the answer; 3.7 means neither 3 nor 4) and `True` into item
1 (`bool` is an `int` subclass in Python). Neither is reachable from a schema-conforming model,
which is exactly why it was harmless under the 0-based contract and worth closing under this
one: under 1-based numbering an invented number names a REAL card. An integral float (`3.0`)
and a decimal string (`"3"`) are still accepted, because each names exactly one item and
nothing is invented.

Suite 1643 → 1670 green.

#### Addendum 2026-08-12, final audit of the model-facing contract workflow

The schema + payload + prompt scope is closed and audited. Nothing below is new code; this
records the audited end state, the two things the audit found, and the requirements the
driver-integration workflow must carry forward. Suite **1670 passed** (baseline at the start of
this workflow was 1601).

**Scope held.** Six things were explicitly out of scope and none of them leaked: 5.6's post-tap
verification and hard stop are unwritten (`_verify_like_landed` is the pre-existing "did the like
send" check and compares no signatures); the `hearts[0]` / `_await_button("like")` first-photo
fallback in `hinge._locate_target_heart` is intact; the anchor path is intact end to end
(`_ANCHOR_SYSTEM`, `_ANCHOR_LABEL`, `anchored_opener`, observe's `anchor=` call); observe is not
inverted (`worker._wait_for_observed_decision` still passes the human's like-screen anchor with
`advisory=True`); `item_index` is not wired into `_capture_current` (it gained a comment naming
the two spaces and nothing else); and `config.yaml`'s `read_scroll_frac` is byte-identical at
0.55. The seven leaf modules from the perception workflows are still imported by nothing in
`operation_love/` outside their own tests.

**Part A is intact and still pinned.** Every wording property from sections 2, 2.1, 2.2, 2.3, 2.4
and 3.2.1 is asserted in both copies by name, in
`tests/test_opener.py::test_system_prompt_keeps_faithful_corey_opener_policy_and_two_sentence_cap`
and `test_system_prompt_never_turns_the_five_moves_into_a_binding_menu` for `_SYSTEM`, and in
`tests/test_config_yaml_real.py`'s five `test_shipped_opener_style_*` tests plus
`test_shipped_example_openers_contain_no_hyphen_em_dash_or_non_ascii` for `config.yaml`. Several
assertions are NEGATIVE (the deleted "exactly one concrete detail" framing without its rider, "one
short sentence is preferred", "a hedge like I bet or I heard", any `pick the best photo` phrasing,
`referenced_index`, `scroll order`), which is what makes them regression guards rather than
inventories.

**`item_description` is migrated in both stores.** A schema entry AND an `ALTER` in each:
`store.py`'s `_SCHEMA` line plus a duplicate-column-guarded `ALTER TABLE openers ADD COLUMN`, and
`bigquery_store.py`'s `_TABLES` entry plus `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` in
`_MIGRATIONS`. The live `openers` table already holds rows, so `CREATE TABLE IF NOT EXISTS` alone
would never have reached it — the same trap `angle` documented one addendum earlier.

**Finding 1, and it is the reason this addendum exists: a THIRD live dry run was made, against
REAL profile crops, and this file did not record it.** The two runs recorded above (the response
schema, then the request shape) were synthetic colour panels. A third was then run offline-of-doc
from a session scratchpad: it built a real `ItemPayload` from
`ops/calibration/botscroll_20260811T231516Z` and sent those crops to the Gemini API, five openers,
`advisory=True`. So real people's profile pixels reached a remote, which is exactly what
`.gitignore:26`'s own comment says must never happen ("frames of REAL people's dating profiles ...
they must never reach a remote"), and the accepted-risk note that should have accompanied it was
never written. Nothing leaked into git: no frame, no crop, no name, no prompt text and no generated
opener appears in any tracked file, and `git status --short ops/calibration` is empty. The run's
raw output, which did contain her prompt text verbatim alongside the openers written about her, was
deleted from the scratchpad during this audit. **What ships is unaffected and no rule about the
phone was touched** (no device input, ever). What is accepted, and stated here rather than left
implicit: one profile from the calibration corpus has been through a third-party model, and the
next workflow does not get to treat "offline only, one dry run" as satisfied by silence — if a run
needs real pixels, it goes in this file with the reason before it happens.

**Finding 2, smaller: 5.7's "observe displays it" is not built.** `item_description` is returned in
both modes, carried on `OpenerPick`, and persisted by both stores, but no hub surface renders it —
`hub/state.py` and `assets/hub.html` never mention it. Auto's half ("ignores and logs it") is done.
The display half belongs with the observe inversion and is called out below so it is not assumed
finished.

**What remains before driver integration can run**, over and above 5.6's own addendum list
(a stop surface routing four exception types, the unraised `scroll_captures` ceiling,
`navigate_to_item`'s `**plan_kwargs` staying unused in production):

1. The adapter from `drivers.item_crops.ItemPayload` to `opener.ItemRequest`, and threading
   `items=` through `OpenerService.maybe_opener` — today nothing builds an `ItemRequest`, so the
   crop shape is unreachable from production and every crop-shape like would repair against the
   live anchor rather than target.
2. 5.3's driver-owned translation table (model item index -> heart ordinal), invalidated wherever
   `_current_sigs` is, including the deck-advance path. Until it exists
   `OpenerPick.capture_order_index` returns `None` for `INDEX_SPACE_MODEL_ITEMS` by design.
3. 5.8's pre-flight cross-check, 5.6's post-tap verification and hard stop, removal of the
   `hearts[0]` fallback, anchor removal from prompt and driver, the observe inversion (including
   the `item_description` display above), and wiring `item_index` into `_capture_current`.

**Two carried-forward requirements, repeated verbatim in substance so they cannot be lost:**

- **Crop signatures DRIFT on animated cards, and that constrains 5.6.** Re-observation drift
  measured 2.66 on one profile and 25.0 on the other (the animated card), against per-item
  nearest-neighbour distances of 44.7 and 47.9 — separable per item, at 1.9x on the worst, not the
  230x an earlier measurement claimed. No fixed threshold is admissible: 0.02 rejects a correct
  re-observation and 30 accepts a wrong item. Compare against the per-item `signature_drift` /
  `nearest_item_distance` / `separable` each `ItemCrop` carries, and remember `separable is None`
  means nothing re-observed the crop, never that it is stable.
- **A stale or foreign index is NOT caught at model index 1 today.** `item_nav`'s anchor test is
  the only thing standing between a stale translation table and a confident tap on the wrong
  person's card at ordinal 1, and the one measured cross-profile pair tripped it by exactly 0px
  (218px apart against a 218px bound, which is why that comparison is `>=` and not `>`). A closer
  pair is not caught at all, and no geometry available to navigation can catch it, because Hinge's
  first cards are stereotyped (same 974px height, same 885px heart inset, same x=938 across
  profiles). The next phase must add an IDENTITY check at navigation ENTRY, before any gesture, IN
  ADDITION to 5.6's post-tap verification — not instead of it, and not on the assumption that
  navigation pre-screened it.

#### Addendum 2026-08-12: the capture now enumerates, and an `ItemRequest` is what AUTO sends

The first item on the previous addendum's "what remains" list is done, along with 5.3's table and
the `item_index`-into-`_capture_current` wiring. The chain that did not exist at all now runs end
to end in one direction: `hinge._capture_current` confirms the scroll top affirmatively, sizes
every gesture from `plan_scroll_step` against the card in front of it, folds the frames it kept
into a `build_item_index` / `build_item_payload` pair, and carries the numbered crops, the
unnumbered context crops, her name and the truncation flag out on the `Profile`;
`worker._auto_loop` turns those into `opener.ItemRequest.from_profile(profile)` and
`OpenerService.maybe_opener` threads them to `generate(items=...)`, unconditionally, the way it
threads `anchor`. The seven leaf modules are no longer imported by nothing.

Five things worth recording because they were decisions, not transcription.

**The refusal is a RESULT on the Profile, and the stop is at the worker.** Two consumers want
these frames and they fail independently: the ranker needs faces and the opener needs a numbered
list, so an enumeration that refuses must not discard a capture the ranker was going to use. Every
failure inside the read — the top not confirmed, a spacing no permitted gesture can respect, a
frame that will not segment, an index or a payload that contradicts itself — is caught by name and
written to `Profile.items_unavailable` as one operator-readable sentence, and exactly one of that
field and `Profile.items` is ever set. `worker._auto_loop` reads it before it calls the opener at
all and stops the run (`stop_kind="opener"`, no billed call, no like). **That placement is the
whole of doc 5.2's "never fall back to raw frames"**: the substitution could only ever happen at
the request-building site, so that is where the refusal lives, rather than in a driver crash that
would also throw away the ranker's frames. A driver that enumerates NOTHING (every non-Hinge
driver) leaves both fields empty and keeps the frame shape untouched — "no item space" and "an
item space that failed" are different facts and only the second stops a run.

**Enumeration is AUTO-only for now, and that is a policy choice rather than a capability one.**
The closed loop reads a profile in ~37-43 frames against `scroll_captures`' 12, and observe mode
then scrolls every one of them back up before the operator can act — roughly 3x the ~85s
per-profile wall clock on the one mode a human sits and waits through, in exchange for a payload
observe currently has no use for (5.9's inversion is what gives it one). So
`_item_enumeration_blocker` refuses an observe session first, by name, and the reason is on the
Profile rather than implied by an absence. Observe's cadence, frame count and anchored suggestion
are byte-for-byte what they were. **The observe-inversion workflow owns turning this on**, and it
is one condition in one method.

**The enumeration read has its own ceiling, and `config.yaml` is untouched.** 5.6's "the
enumeration capture ceiling is still unraised" blocker is closed by `_ENUMERATION_CAPTURE_LIMIT`
= 48 — the ceiling the validated bot-driven probe itself ran at, against the 37 and 43 frames the
loop was measured to need for the two calibration profiles. It is a driver constant rather than a
config key because it is a property of the measured card geometry and the gesture window, not an
operator preference, and because `scroll_captures` must stay exactly what it is for the observe
read and for `_ensure_session_top`'s swipe ceiling. `read_scroll_frac` is still 0.55 and is still
what every non-enumeration read uses. The capture-limit jitter is applied on top of the new base,
not replaced by it: a fixed terminal depth does not stop being a bot signature because the
constant got larger.

**The table is invalidated at three places, not one.** 5.3 asks for "wherever `_current_sigs` is,
including the deck-advance path". It is dropped at the top of every capture, on the
`_current_capture_split` path, and — going beyond what 5.3 asks — in a `finally` around `like()`
and `dislike()`, because a failed action is exactly when a stale table survives longest (nobody
then knows where the deck is). `_invalidate_item_index` sets the REASON as it clears, so the state
is never "no payload and no explanation".

**Nothing spends the table yet, and AUTO therefore hard-stops on the crop shape.** This is the
honest state and it is deliberate: `generate()` answers in `INDEX_SPACE_MODEL_ITEMS`,
`OpenerPick.capture_order_index` returns `None` for that space (capture-order frames are the wrong
target — the crop shape resolves to a HEART ORDINAL), and the worker's existing guard stops the
run before touching the driver. So an AUTO like on an enumerated profile currently stops rather
than targets. The guard's stop_reason was corrected rather than left to lie: it used to say this
build "has no driver-owned translation table", which is now false — it has one and nothing spends
it. Wiring the tap is 5.6's workflow (navigate, verify against `ItemPayload.signature_for`, hard
stop, remove the `hearts[0]` fallback and the anchor path), and the two carried-forward
requirements above are unchanged and still owed by it: the identity gate at navigation ENTRY, and
a per-item drift comparison rather than a fixed signature threshold.

Offline only — no device, no live API call, and nothing from `ops/calibration/` was read, sent or
described. The 26 new tests build a synthetic scrollable world from first principles and serve
1080x2400 windows of it through a fake ADB that moves by exactly what the transport model says a
`frac` delivers, so the real segmenter, the real shift estimator and the real shipped scroll-top
fingerprint all run against it. Suite 1671 → 1697 green.

### 5.8 The pre-flight cross-check

`item_description` is a free-text description of the item the model believes it chose. Compare
it against what our own crop at that index actually is. Disagreement means our list
construction is wrong, which means the model's `item_index` refers to something other than what
we think, which means we are about to like the wrong item.

This is the same guarantee as the post-tap verification in 5.6, obtained **before we touch the
phone**, from a field we are already collecting. It is the cheaper half of the same rule.

**Compare item type only, never content.** Semantic comparison would need a second model call,
which is out. What is deterministic is the coarse class: does the description name a photo, or
a written prompt or answer? And is our crop a text card or a photo? The latter is separable
from pixels alone by colour variance and edge density, since text cards are near-uniform
background with sparse high-contrast glyphs and photos are not. Keyword-match one against the
other and stop there.

**Precision over recall, as with the scaffolding detector.** Fail only when the description
unambiguously names one class and our crop is confidently the other. Anything ambiguous passes.
A noisy check here stops runs for nothing, and the cost of a miss is bounded because 5.6 still
catches it after the tap.

**On failure: hard stop**, same as 5.6, for the same reason. A segmentation bug is not a
condition to handle, and there is no safe action once we know our own item list is wrong. This
check should essentially never fire. If it fires with any regularity during bring-up, that is
the signal that hearts-and-gutters is not carving profiles correctly, which is exactly what we
would want to learn before it has liked a hundred wrong items.

**Do not oversell it.** Comparing coarse type cannot catch the wrong photo among several
photos, which is the most likely index bug and exactly what a scroll-to-top undershoot or a
spurious mid-photo gutter would produce. It catches a whole class of gross misalignment early
and cheaply. The post-tap check in 5.6 remains the one doing the real work.

### 5.9 Observe mode

Inverted: the system decides which item to like and tells the human "like item 3" plus the
opener, before the human taps. Same call, same schema, same generated string as auto. Observe
serves as manual debug mode.

Cheap to build, though note the argument is about today's code: `anchor=None` is already a
first-class supported path, unanchored generation is
what the system did before anchoring existed, and the full profile is in hand at
`worker.py:279` well before the human acts. Nothing in `OpenerService` or `Opener` changes;
only the call site moves.

The canary rule survives. The opener sits alone in its own div with `escHtml(opener)` and
nothing else (`openerBox` at `hub.html:227-230`, the opener-only div is line 229); everything
else goes in a separate dimmer `chrome` row on line 230.
"Like item 3" and the description go in `chrome`, never inside the opener block.

Labels are not contaminated. They are whole-profile like/pass with a face embedding;
`referenced_index` never reaches the `labels` table.

Decided against: logging when the human hearts a different item than we chose, as *training
data*. It would make observe an evaluation set for the item picker, but it adds friction to a
mode whose primary job is evaluating like/pass accuracy.

**But the mismatch must still be detected and surfaced, which is a different thing.** This is
the most serious defect found reviewing this design, and it is a regression I introduced.

Today the observe opener is generated *after* the tap, against the sheet the human actually
opened, so it is right by construction: `hinge.py:2563` calls the observe anchor "exact by
construction (a human tapped it)." Inverting the order removes that guarantee on the one path
where a real message reaches a real person on a real account. If the suggestion is about item 3
and the human hearts item 5, nothing in the design as first written notices, and the human
pastes a wrong-item opener under item 5.

Latency makes this a live race rather than an edge case. Generation can take up to 90s
(`request_timeout_s`, `config.yaml:279`), and nothing bounds how quickly a human who has
already read the profile may tap. Whenever the tap wins, the suggestion lands *after* the human
has acted, where it is actively misleading rather than merely late. Publishing READY
immediately and letting the suggestion fill in behind it helps the wait but does nothing about
this race. How often the tap actually wins is unmeasured; the debug logs bound total profile
read time but not post-read tap latency.

The fix costs the human nothing, which is why it does not violate the no-friction decision
above. We already see the opened sheet and we already hold per-item crops, so the same
deterministic signature match from 5.6 answers "is this the item the opener was written for."
On mismatch the hub replaces the opener block with a warning and no text to type. No input is
requested from the human, nothing is recorded as training data, and the failure mode goes from
silent to loud.

The auto-mode rule is "never ship a comment attached to the wrong item." Observe must honour
the same rule; it just enforces it by refusing to show text rather than by stopping the run.

### 5.10 Calibration results, measured 2026-08-11

115 real frames of one profile, hand-scrolled top to bottom at ~0.83s intervals, captured with
`tools/hinge_scroll_capture.py` (read-only: no input is sent to the phone, and the tool has no
input path at all). Frames live in `ops/calibration/`, which is gitignored because they are real
people's profiles. Only geometry is recorded here.

**The capture immediately found a production defect unrelated to Part B. See the Status section.**

**Card geometry (5.4) is measurable and the gutter approach works, with two amendments.**

| Measurement | Value |
|---|---|
| Page background | NOT flat: a vertical gradient, RGB ~(255,254,253) at y=300 to ~(243,243,243) at y=2100 |
| Left / right card margin | 53px each (screen 1080px) |
| Gutter height between cards | 52-53px canonical, 212 of 224 measured gutters within 1px |
| Card corner radius | ~20-23px |
| Tallest card observed end to end | 1114px |

Amendment one: **colour alone is not sufficient.** 11.5% of rows sampled inside confirmed cards
pass a naive "matches background" test, and one contiguous 192-row span inside a single card
would have split it in two. Combining the colour test with the expected 52-53px gutter *length*
resolves it, since the false span is nearly 4x too tall. Because the background is a gradient,
the colour reference must be local to the row, not a single global constant.

Amendment two: a real gutter was **missed** by a naive "any pixel differs from background" test,
because a narrow element spanning only x=30-216 kept the row looking occupied. The test must
require non-background pixels across close to the card's full 53-1026 width, not any single pixel.

Amendment three (2026-08-14): **gutter length is necessary but not sufficient.** A captioned
photo placed a 135px header and its media inside one rounded card with a 47px blank card-white
seam. The old length-only gate accepted that seam as a real gutter, split one photo into a
heartless context block plus a heart-bearing media block, and clipped frames spanning the two
pieces later made the page index refuse contradictory extents. A gutter must therefore satisfy
both structural tests: canonical length *and* affirmative agreement with the row-local page
background read from the side margins. In the incident frame the internal seam's median grey
level differed from the page by 2 levels (up to 3); the genuine 53px gutter immediately below
matched by 0 on every row. The shipped gate allows one level of capture noise. Rejecting an
uncertain gutter merges regions—the conservative, loud direction—and never fabricates or
renumbers an item. This is content-derived segmentation; it has no fixed photo-count rule.

**The viewport question (5.4, and open item in 9) is CLOSED.** The tallest card fully observed is
1114px against an 1800px content band, 62% of the budget. Nothing in this capture approaches the
viewport height, so cropping can work from single frames and **no stitching is required**. Caveat:
one profile. A very long prompt answer could still exceed it, so the crop path should detect the
case rather than assume it away.

**The vitals block (5.3) is confirmed.** 215px tall, tracked across 7 consecutive frames, sits
immediately after the first card, and carries **no heart** in any frame. The two-tier
selectable/context split is correct as designed.

**The endorsement block did not appear** in this capture, by OCR across all 115 frames in two
passes. That is not proof of absence. It remains the open item it was, and the two-tier design
handles either answer.

**Heart counting (5.5) is confirmed as the weak joint, with numbers.** This is the finding that
matters, and it went against the hopeful reading.

- Frame-to-frame heart matching by an independently estimated scroll offset works well *per step*:
  189 of 198 heart instances matched, residual std 2.4px.
- But chaining those matches across the whole scroll recovered 9 real items **and fabricated a
  spurious 10th**, created entirely by one large-jump tracking failure. Naive counting overcounts,
  which is exactly the failure 5.3's index-trap reasoning says we cannot tolerate.
- The alternative this doc proposed in 5.5 (count a heart once it has entered the lower band and
  left the upper band) was simulated at four threshold choices and **missed 2 of 9 real items at
  every one**. Not a tuning problem: at large scroll steps an item crosses both bands inside a
  single frame gap.
- Human scroll steps ranged 0 to 787px, median 116px, against an 1800px band. 11% of steps
  exceeded 500px.

The honest reading: heart counting is not safe as a sole source of truth at *uncontrolled* scroll
cadence. The mitigating argument is that Part B's closed-loop scrolling bounds its own step size,
which is a materially easier problem than this capture posed. That hypothesis was then tested
directly, in 5.10.1.

#### Addendum 2026-08-11: card-corner radius, closing the list-boundary open item

Measured in the same corpus, extending the card-geometry table above and closing the list-boundary
problem 5.4 called out as unsolved (see 5.4's own addendum for the full narrative and the
disjoint-window numbers that ruled out a declared `list_top_y`/`list_bottom_y` constant).

| Measurement | Value |
|---|---|
| Corner-radius agreement at the 219 gutter-proven card edges (edge-row span estimate vs. ramp-length estimate, within 2px) | 209/219 tops (95%), 213/219 bottoms (97%) |
| Nearest false-positive "radius" implied by a non-card row (Hinge's header strip / its name row) | 68.5, 72.5, 75.0, 487px — all >=43px outside the accepted 18..25px window |

The corner test needs no declared row and no background above it, so it bounds item 1's top edge
and item N's bottom edge directly from the card's own geometry instead. Full derivation in
`operation_love/drivers/segment.py`'s module docstring ("THE THIRD PIECE OF EVIDENCE") and
`_corner_radius`.

### 5.10.1 Bot-driven scroll probe: the real cause is step/spacing aliasing, and it is fixable

Two bot-driven runs with `tools/hinge_bot_scroll_probe.py`, which calls the production
`current_profile()` exactly once through the shipped humanized path (same `_scroll_down_one`, same
jitter, same forbidden-zone guard) with every decision and raw-input method monkeypatched to raise.

**Run 1, production default `read_scroll_frac: 0.55`.** 9 frames covered the whole profile, so the
step was ~1150px against ~1027px heart-to-heart spacing. Result: 25% match rate, 6 of 8 pairs
flagged as tracking failures, 9 hearts unclassifiable, a phantom item fabricated.

**That failure is ALIASING, not estimator noise.** When the scroll step equals the item spacing, a
heart translated by one step lands almost exactly where the *next* item's heart already was. Same
heart moved and next heart arrived become geometrically indistinguishable. Confirmed by trying two
independent estimators: 2D phase correlation returned near-zero confidence (0.02-0.24) with
implausibly uniform deltas, and a 1D row-profile cross-correlation saturated at its search bound.
**No better estimator fixes this**, which is why the earlier framing of 5.5 as "needs a better
dedup mechanism" was looking in the wrong place.

**Run 2, `read_scroll_frac: 0.16`, everything else identical.**

| | Run 1 (frac 0.55) | Run 2 (frac 0.16) |
|---|---|---|
| Measured step | ~1150px | 363px |
| Heart spacing | ~1027px | 1027px |
| Step / spacing ratio | ~1.1 | **0.35** |
| Frame-to-frame match rate | 25% | **100%** |
| Tracking failures | 6 of 8 pairs | **0 of 23** |
| Ambiguous items | 9 | **0** |
| Distinct items recovered | 2 confirmed + phantom | **9, no phantom** |
| Phase-correlation response | 0.02-0.24 | 0.54-0.94 |

The 9 items recovered match the 9 real items the independent human-scrolled capture found, so two
unrelated captures agree on the ground truth.

**Consequence for the design.** Heart tracking IS a viable index, but only when the scroll step is
materially smaller than item spacing. Do not read "frac 0.16" as the answer: 363px is safe against
this profile's 1027px spacing, but a short prompt card could space hearts ~600px apart, where the
same 363px step gives a ratio of 0.6 and lands back in the aliasing band.

So the rule is relative, not absolute: **the enumeration pass must step at most about a third of
the locally measured card spacing.** That is implementable precisely because 5.4's gutter detection
already tells us where the next card boundary is, so the closed loop can size each step against
measured local geometry rather than a fixed screen fraction. This is a refinement of 5.5's
closed-loop design, not a replacement for it.

Caveat stated plainly: one profile, one clean run. Validate across more profiles before building
on it. The cost of the finer step is ~3x the frames per profile, which is more dwell time, which
the Signals requirements reward rather than penalise.

#### Addendum 2026-08-11: the cross-frame shift estimator, and two corrections to the numbers above

The estimator this section says is needed now exists as `operation_love/drivers/frameshift.py`
(`estimate_shift`), built to the shape this section's own findings dictate: a bank of 13
horizontal strips cut from frame A across the full analysed band, each located independently in
frame B by `cv2.matchTemplate` / `TM_CCOEFF_NORMED`, with the answer coming from how many strips
AGREE rather than from any one strip's score. Phase correlation is not used, per the measurement
above. Validated offline on all three captures in `ops/calibration/`; full derivation, per-constant
measurements and the corpus numbers are in that module's docstring.

**Correction one: the aliasing capture's step is 1299px, not ~1150px.** The ~1150 above was
inferred from phase correlation on the very capture where phase correlation was measured
unreliable (its own stored deltas there are -501, -501, -501, -501, -555, +272). Five of that
capture's six over-large pairs measure 1299 by strip NCC, consistently and at full confidence, on
the strips that still have content in common. That makes the step/spacing ratio ~1.26 rather than
~1.1 — the conclusion is unchanged and if anything stronger.

**Correction two: heart-to-heart spacing is not a single number.** The 231516Z capture's own
16 measured spacings are 738, 809 and 1027px on ONE profile, not 1027 throughout. So the "step at
most about a third of local spacing" rule has to be evaluated against the *local* pair, exactly as
this section says, and a step sized against 1027 would already be at ratio 0.49 where the spacing
is 738.

Measured behaviour of the estimator on the three captures, which is the evidence the closed loop
can now be built on:

| Capture | Pairs | Measured | Refused | Wrong |
|---|---|---|---|---|
| botscroll 231516Z (frac 0.16) | 23 | 23, median 363px (ledger: 362.99) | 0 | **0** |
| botscroll 225314Z (frac 0.55) | 8 | 2 — the genuine short final step and a settled frame | 6, of which 5 explicitly report "moved 1299px, out of window" | **0** |
| scroll 211209Z (hand-scrolled) | 114 | 89, spanning -18..+787px | 25, all in the animated-card tail | **0** |

"Wrong" is measured independently of the estimator: for every pair it measured, the like glyphs
`_match_glyph` finds in frame A were checked to land where the delta predicts in frame B. 162 of
162 did, on every capture, and that stays 100% across score floors of 0.50, 0.75 and 0.90.

The saturation requirement this section implies is met structurally rather than by tuning: each
strip is searched over its full geometric range and the trust window is applied to the RESULT, so
an over-large shift returns `SHIFT_BEYOND_WINDOW` with the measured magnitude and no `delta_px`,
instead of the search bound dressed up as a measurement.

Still one profile per cadence. The caveat above stands unchanged.

#### Addendum 2026-08-12: what violating the ratio rule actually costs, now that the stack refuses

This section's conclusion — "the enumeration pass must step at most about a third of the locally
measured card spacing" — is unchanged and is what `scroll_step.py` implements. But its stated
CONSEQUENCE ("a step past the ratio produces a confidently wrong index") was a property of the
naive heart-chaining tracker 5.10 measured, and is not a property of the stack that shipped.
Measured by sub-sampling the clean 363px capture against the same real content and re-indexing
with `build_item_index`:

| stride | step per pair | ratio vs that profile's 738px minimum | result |
|---|---|---|---|
| 1 | 363px | 0.49 | usable, 9 selectable, translation 1..9, heart rows exact |
| 2 | 726px | **0.99**, dead centre of the aliasing band | **usable, all 9 heart page rows still EXACT**, but 9 selectable items become 8 — one card was never bounded end to end in a single frame |
| 3 | 1089px | 1.48 | all 7 pairs refused `beyond_window`; unusable, zero blocks, **zero phantoms** |

Because correspondence is measured pairwise and blocks are folded by absolute page position, with
a broken chain yielding no items at all, an over-large step costs COVERAGE and then costs the whole
index — it did not misnumber anything and could not fabricate an item. Recorded in both directions
deliberately: the rule stays exactly where it is (the 0.99 row is one capture, not a licence, and
"most of a profile" is still a worse read), and equally nobody should treat the ratio rule as the
last line of defence against a wrong like. The index's own refusals are that line.

### 5.11 Incidental upsides

Scrolling back up to a specific item to comment on it is *more* human, not less. It is what a
person does: read the whole profile, then return to the thing worth mentioning. Per the Signals
requirements that is the dwell-and-read behaviour the badge rewards.

Heart enumeration also gives two things we do not have today: an exact item count, and a better
end-of-profile signal than the current pixel-repeat heuristic.

Raising `scroll_captures` costs dwell time, which is a cost we want to pay anyway.

---

## 6. Test blast radius

**The duplication trap:** the style text exists in two near-duplicate copies pinned by two
separate tests. Editing one without the other gives the model contradictory instructions and
fails a test in an unrelated file.

Part A:

- `tests/test_opener.py:204` pins ~12 substrings of `_SYSTEM`
- `tests/test_config_yaml_real.py:41-54` pins the same against the real `config.yaml`
- `tests/test_gemini_opener.py`: `test_absent_retry_hint_produces_a_byte_identical_request_to_today`,
  `test_no_anchor_request_is_byte_identical_to_before_anchoring_existed`,
  `test_anchored_main_text_names_the_like_screen_and_forbids_app_chrome`
- `test_generate_posts_structured_multimodal_request_and_maps_usage` if the schema reorders

Part B additionally: every anchor-related test in `tests/test_gemini_opener.py` (the anchor
path is removed), `tests/test_hinge_observe.py`'s `anchored_opener` repair tests,
`tests/test_worker.py`'s opener call-site tests, and the observe-state tests in
`tests/test_status.py`.

## 7. Sequencing

1. **Part A, prompt only.** New style block, edit pairs, schema field descriptions, anchor
   copy fix. Low risk, reversible, no new gate. Targets all five root causes and works against
   today's scroll frames.
2. **Canary it in observe mode.** The observe suggestion is byte-identical to what auto would
   type, so this is a free preview before auto runs on it.
3. **Measure.** A/B the schema reorder and the optional `first_draft` arm. Compute the
   redundancy metric offline from existing `openers` rows.
4. **Part B.** Heart and gutter segmentation, two-tier crops, closed-loop scroll, counting
   navigation, deterministic sheet verification, hard stop.
5. **Gate the monitor**, only if the data justifies a threshold.

## 8. Operability risk: the stop surface

Part B adds three unconditional hard stops with no fallback permitted: a missing translation
table (5.3), a navigation miss after retries (5.6), and a pre-flight type mismatch (5.8). It
simultaneously *removes* the one recovery path that exists today, the `anchored_opener` repair.
These stack on top of an already-substantial set: per-profile attempt exhaustion, the 400
streak latch, the transient-failure latch, capacity exhaustion, paywall block, the undecodable
frame guard, and the stuck-screen watchdog.

The design ships all three with **zero measured reliability**, on brand new cv2 heuristics
(row uniformity, heart counting) stacked on an already-fragile scroll-to-top. If the real
failure rate resembles the truncation rate in §4, the bot halts often enough to need an operator
babysitting it, and every halt is a manual restart.

This is the intended tradeoff, since a wrong-item like is worse than a stop. But it should be
entered knowingly, and it argues for the dry-run measurement in 5.6 being done *before* Part B
runs unattended rather than after it proves annoying.

Related, and worth stating plainly because §5.9 currently reads more reassuring than it should:
observe mode canaries the opener *wording* byte-for-byte, but it cannot canary navigation or
targeting at all, because the human drives the phone. The highest-risk machinery in this
redesign is exactly the part the sequencing never validates before it runs autonomously.

## 9. Open items

RESOLVED by the 2026-08-11 calibration (see 5.10):

- ~~Whether cards can exceed the viewport~~ — CLOSED. Tallest card 1114px against an 1800px band.
  No stitching needed. Detect the over-tall case rather than assume it away.
- ~~Vitals block position and heart status~~ — CLOSED enough to build on. Heart-less, 215px,
  immediately after the first card. Confirms the two-tier split.
- ~~Eval corpus gitignore decision~~ — DONE. `ops/calibration/` is now gitignored, alongside
  `.env`, with the reasoning recorded in `.gitignore` itself.

STILL OPEN:

- ~~Heart dedup under closed-loop scrolling~~ — ANSWERED, see 5.10.1. The cause was step/spacing
  aliasing, and a step of ~1/3 the local card spacing gives a 100% frame-to-frame match rate with
  zero phantoms. What remains is validation breadth, not the mechanism: this is one profile, and
  the step must be sized against *measured local* spacing rather than a fixed screen fraction,
  because a short prompt card can space hearts tightly enough to realias a fixed step.
- Whether the endorsement block carries a heart. It did not appear in this capture at all. Still
  non-blocking: the two-tier scheme handles either answer.
- Whether the vitals block's position in scroll order is fixed across profiles, or only happened
  to follow the first card in this one.
- Free-tier quota: any offline A/B harness must not consume the day's Gemini quota needed for
  live runs.
- Entropy guard: hard rejection versus soft `retry_hint` signal, per 3.6. The observe-mode
  `advisory=True` single-attempt asymmetry makes this close to decided already.

#### Addendum 2026-08-12: audit of the scroll-top / closed-loop / counting-navigation workflow

Recorded here rather than in a fourth 5.5 addendum because it spans sections and because what a
later reader needs from it is the BLOCKER LIST, which is this section's job. Nothing above was
edited; the three 5.5 addenda and the 5.6 one are the change record, this is the state after them.

**What exists now, and where.** Three modules, all new and none wired into `hinge.py` — nothing
imports them yet, which is the honest summary of runnability:

| | Lines | Tests | What it is |
|---|---|---|---|
| `operation_love/drivers/scroll_top.py` | 416 | 26 | the affirmative gate (5.5's blocker 2) |
| `operation_love/drivers/scroll_step.py` | 795 | 50 | the closed loop (5.10.1's blocker 1) |
| `operation_love/drivers/item_nav.py` | 867 | 31 | counting navigation |

Plus two edits outside them: `ItemIndex.heart_ordinal_for` now RAISES on an `at_scroll_top=False`
index (5.3's blocker 3), and `hinge._band` grew an optional `size` argument, defaulting to
`_IDENTITY_DS`, so `scroll_top` reads the identity band through the driver's ONE decode instead of
re-implementing crop-and-resize. Suite **1601 passed**, 0 failed.

**Scope held.** Verified absent from the working tree: any `item_index` / `item_description`
schema field (`opener.py`'s `_SCHEMA` still requires exactly `referenced`, `angle`,
`referenced_index`, `opener`), any prompt rewrite, any post-tap verification, any hard stop wired
to the run surface, anchor removal (`hinge.py`'s `hearts[0]` fallback and the `anchored_opener`
repair path are untouched), the observe inversion (`worker.py`, `hub.html` and the observe path in
`hinge.py` are untouched by this workflow), and any change to `config.yaml` — `read_scroll_frac`
is still 0.55 and `scroll_captures` still 12.

**BLOCKERS REMAINING before selection + verification can run.** In the order they bite:

1. **Nothing is wired.** `hinge.py` imports none of these modules, so the profile read still runs
   at `read_scroll_frac` 0.55 and still produces an index that refuses. The three modules are
   validated leaves plus one sequencer, not a pipeline.
2. **The enumeration capture ceiling is still unraised**, unchanged from 5.6's addendum:
   `scroll_captures: 12` plus `_capture_limit_for_profile`'s 0..2 against the ~34-43 captures the
   closed loop needs for one profile. Until the caller raises it, every enumeration read truncates.
3. **`navigate_to_item` has no `should_stop`.** It calls `_scroll_to_top()` without one and then
   issues up to a budget's worth of gestures (9..49 frames measured) polling nothing. The observe
   read path plumbs `should_stop` through both; this does not. Whoever wires it owns closing that,
   or navigation becomes a new stop-deaf window of exactly the kind the operator has complained
   about before.
4. **The stop surface must route four uncoded exception types**, not just `ItemNavigationError.code`
   — `ScrollStepError`, `SegmentationError`, `ShiftEstimationError`, `ScrollTopUnconfirmed`. Stated
   in 5.6's addendum, still unbuilt.
5. **Ordinal 1 against a stale or foreign index is caught by nothing before the tap.** Measured,
   not feared (5.5's fourth addendum). The mitigations are correct invalidation of the translation
   table wherever `_current_sigs` is cleared, including the deck-advance path, and 5.6's post-tap
   crop-signature check.
6. **That signature check cannot be a fixed threshold**: crop signatures drift 0.018 grey levels on
   a static item against 25.0 on an animated one. This is the blocker handed forward from the
   indexing workflow and it is still open.
7. **Nothing has touched a device.** Every number in the three addenda above is offline replay over
   `ops/calibration/`. `_TOUCH_SLOP_PX`, `_MAX_STEP_PX` and `_SCROLL_TOP_BAND_FINGERPRINT` are all
   properties of one Pixel 7a and one Hinge build; each fails as a hard stop rather than a wrong
   answer, and each needs re-measuring with the owner present if the device, the transport or the
   account's own filter chips change.
8. **`navigate_to_item`'s `**plan_kwargs` remain an open door** into `plan_scroll_step`. A
   production caller passes none; the offline replay needs exactly one. There is no assertion of
   that anywhere, only `ItemTarget.steps` recording it after the fact.

#### Addendum 2026-08-12: the AUTO path now hard-stops on an unresolvable pick instead of repairing one

The "'bounded rather than silent' was FALSE" addendum above (search for that heading) fixed the
type-level confusion — `OpenerPick.capture_order_index` now states the index space and returns
`None` rather than guessing — but the behaviour it wired in for the `None` case was still: hand
the driver `item_index=None` and `anchored_opener`, let it tap the first item to open the sheet,
and let `_like_comment_sheet` decide only THEN whether a repair re-ask can produce text that
matches whatever it landed on. That addendum said so explicitly and accepted it as the honest
state: "every crop-shape like would repair against the live anchor rather than target." An
independent read of that acceptance, against this doc's own 5.3 rule, found it insufficient:
**"treat a missing table as a hard stop, never as a reason to fall back to a fixed coordinate"**
does not describe "fall back to a fixed coordinate, then ask a second model to write honest text
about it." Repairing the TEXT does not undo spending the LIKE on an item the model never chose,
and the driver still touches the phone — taps the heart, opens the sheet, and on a failed re-ask
leaves the sheet open — before anything downstream has decided whether targeting was even
possible. That is a bounded, visible substitution. It is still a substitution, and doc 5.3 asks
for a hard stop instead.

**The fix moves the check from the driver to the worker, and from "after tapping" to "before
calling the driver at all."** `worker.py`'s `_auto_loop`, right after the existing "no opener at
all" stop and before it ever calls `self.driver.like(...)`, now reads
`pick.capture_order_index`: if a pick exists (an opener WAS generated) and its
`capture_order_index` is `None`, the run stops there — `stop_kind="opener"`, a `stop_reason`
naming which of the two causes it was, `self.driver` never touched. The two causes get different
wording because the operator's next move differs: `pick.index == ITEM_INDEX_ABSENT` reads as
"the model gave no item number at all" (an opener-layer/prompt problem); anything else reads as
"opener targets model item N in index space S, and this build has no driver-owned translation
table for that space yet" (a missing-infrastructure problem — 5.3's table, still the next
workflow's job, unchanged by this addendum). Both are the SAME underlying fact
(`capture_order_index is None`); only the message differs, matching the same two-cause split
`_like_comment_sheet`'s own `HingeActionError` already used for the DIFFERENT failure it covers
(a targeting miss discovered AT SWIPE TIME, after a valid, translated index no longer matches
because the deck moved between the decision and the tap).

**The driver's `anchored_opener` repair hatch is NOT removed, and is not wrong to keep.** It
still exists for exactly that swipe-time-drift case — a `pick.capture_order_index` that WAS a
sound, validated conversion when the worker checked it, but no longer matches what
`_locate_target_heart` finds once the driver actually gets there. That is a live, in-the-field
mismatch a second anchor-grounded call can legitimately repair, because the item really was
chosen and really did move; it is a different failure from never having had a target at all, and
it keeps its existing behaviour untouched, including `test_anchored_opener_callback_reasks_...`
and `..._returns_none_when_the_reask_fails` in `test_worker.py` (both updated only to hand the
worker a TRANSLATABLE initial pick, so they keep exercising the repair callback itself rather
than tripping the new earlier stop).

**Observe mode is untouched, structurally, not just by test count.** `worker.py` calls
`self.driver.like(...)` from exactly one place, inside `_auto_loop`; `_observe_loop` and
`_wait_for_observed_decision` never call it at all (the human taps Send Like with the app's own
controls). The new check lives inside the one call site that only AUTO reaches, so there is no
code path through which it could touch observe's behaviour.

**A side effect worth naming: the test fixtures that fake a provider response had to stop being
carelessly untranslatable.** `OpenerResult.index_space` defaults to `INDEX_SPACE_MODEL_ITEMS`
precisely so a careless construction — "the fake clients in tests," by that field's own comment
— fails in the safe direction. Before this addendum, "safe direction" meant "the driver repairs
or refuses"; after it, the same default means "the worker refuses to call the driver at all,"
which is stronger but also meant every `test_worker.py` test built on `FakeOpenerClient` (most of
the file — tests about budgets, rate limits, ratio ceilings, opener fidelity, none of them about
item targeting) started hard-stopping on a pick nobody meant to make untranslatable. Fixed at the
fixture, not the assertions: `FakeOpenerClient.generate()` and `_RecordingResultOpenerClient.generate()`
now set `item_index=FIRST_ITEM_INDEX, index_space=INDEX_SPACE_PROFILE_PHOTOS` on the
`OpenerResult` they return — a realistic, translatable pick matching what the LIVE pipeline
actually sends today (nothing builds an `ItemRequest` yet, so every real call is still in the
profile-photos space) — so a test that isn't about item targeting keeps exercising what it was
written to exercise. Tests that DO want an absent or untranslatable pick construct their own
`OpenerPick` directly (`_SequencedOpenerService`), bypassing this fixture entirely.

Suite 1670 → 1671 green: one net new test
(`test_auto_mode_stop_reason_names_absent_item_index_distinctly_from_untranslatable_space`);
the rest of the delta is existing `test_worker.py` tests renamed and/or re-pointed at the new
behaviour rather than added. No other file's fakes construct an `OpenerResult`/`OpenerPick`
through a shared, unset-by-default fixture, so nothing else needed the same fixture repair —
confirmed by running the full suite, not merely inferred.

#### Addendum 2026-08-12: two NEW blockers, and both are measurements rather than code

Recorded in this section because §9 is where the blocker list lives; the mechanisms, the numbers
and the fixes are in 5.6's "post-tap verification accepted a FOREIGN card" addendum above.

9. **`item_verify.verify_sheet_item` has no absolute accept ceiling.** The 0.5 separation fraction
   proves the chosen item is the UNIQUE NEAREST of the stored items; nothing anywhere bounds how
   far the sheet may be from ALL of them, so out-of-payload content only has to beat the payload's
   own internal spacing. One of the two mechanisms behind the measured 10-of-540 foreign accept is
   fixed (the neighbour set was being pruned by a height filter); this one is not. Closing it needs
   a second, absolute term, and the correct-item and foreign-accept distributions leave about 3.5
   grey levels between them on two profiles — a calibration task with the owner and a device, not a
   constant to pick offline. **Blocking for 5.9**, which plans to make this comparison the sole
   guard on the one flow where a real message reaches a real person.
10. **`item_identity._IDENTITY_MATCH_MAX_DIST` = 3.0 is too wide, measured.** Its justification is
    one pair of profiles at 7.287 grey levels. Over six real profiles the closest two DIFFERENT
    people measure **2.565** — inside the bound — so one pair in fifteen would pass the gate that
    exists to refuse exactly that. Same-profile distance is still identically 0.000 everywhere, so
    the separation is total and any bound under 2.565 is correct on all available data; choosing
    one is the calibration act, and tightening it trades a wrong-person risk for a stop on every
    redrawn header (6.67 for a 4px shift). **Blocking for anything that leans on an
    `IDENTITY_MATCH` as confirmation** rather than using it as a refusal.

Neither is reachable in production: nothing hands the driver a model item number, the driver
refuses one it cannot navigate to, and it now also asks whose profile is on screen before it moves.
Blockers 1 (nothing is wired) through 8 above are unchanged, except that 5 and 6 are answered as
far as they can be offline — 6 by `verify_sheet_item`'s per-item derived bound and 5 by the
identity gate plus this driver-side check, both with the residuals now stated as 9 and 10.

#### Addendum 2026-08-12: final audit of the capture-indexing / targeting workflow, and the headline is that AUTO STILL CANNOT LIKE

Recorded in §9 with the other audits, because what a later reader needs from it is the blocker
list. Nothing above was edited. Offline only: no device, no live API call, and nothing from
`ops/calibration/` or `data/hinge_debug/` was read for this audit.

**Suite 1671 → 1770 green**, 0 failed, over the whole workflow (capture-time indexing and the
`ItemRequest` wiring, the identity gate, post-tap verification, the hard stop, the `hearts[0]`
removal and the `anchored_opener` removal). The five change addenda above are the record of what
each step did; this is the state after all of them.

**THE STATED GAP IS NOT CLOSED. Auto mode stops on every Hinge profile, without exception.**
The chain now runs one way and dead-ends: `_capture_current` enumerates, `_index_captured_items`
builds the index and the crops, `worker._auto_loop` builds an `ItemRequest`, `generate()` answers
in `INDEX_SPACE_MODEL_ITEMS`, `OpenerPick.capture_order_index` returns `None` for that space, and
the worker hard-stops before touching the driver. There is no branch out of that. Traced through
the code, an auto Hinge like has exactly two outcomes and both are stops:

- enumeration SUCCEEDS -> `Profile.items` set -> `ItemRequest` -> model-items space -> the
  `capture_order_index is None` stop;
- enumeration REFUSES -> `Profile.items_unavailable` set -> the `items_unavailable` stop, which
  the worker takes before it calls the opener at all.

`item_nav.navigate_to_item` is imported by nothing in `operation_love/`, and `model_item_index` is
passed by no production caller. So the two guards this workflow built — `_confirm_payload_profile`
(identity) and `_verify_sheet_shows` (post-tap) — are both gated on `model_item_index is not None`
and are therefore UNREACHABLE in production. They are correct, they are measured, and they are
dead code until the navigation call is wired. Blocker 1 ("nothing is wired") is unchanged in
substance: what changed is that the leaf modules are now reachable from the CAPTURE side only.

**TWO DEFECTS THIS WORKFLOW INTRODUCED, both stops rather than wrong likes.**

11. **An enumeration refusal stops an auto run that has no openers at all.** `worker._auto_loop`
    gates the `items_unavailable` stop on `accepts_opener and self.opener_service is not None`,
    and does NOT check `self.opener_service.disabled`. With `opener.enabled: false` the service is
    a live-but-disabled instance rather than `None` (see the `pick is None` guard immediately
    below it, which DOES check `disabled` for exactly this reason), while
    `_item_enumeration_blocker` reads the SPEC's `accepts_opener`, which config does not gate. So
    Hinge auto with openers off still attempts an enumeration read, and any refusal — an
    unconfirmed scroll top, a spacing no gesture can respect, a frame that will not segment —
    stops the run over a numbered list that run was never going to send. Reproduced through the
    shipped `Worker` with a disabled `OpenerService`: `state=stopped`, `stop_kind="opener"`,
    `driver.likes == []`. The fix is one `disabled` check, but it is a behaviour change and is
    left for whoever owns the next workflow to make deliberately.

12. **The auto ranker now scores a ~48-frame read while its training labels come from a 12-frame
    read.** `_ENUMERATION_CAPTURE_LIMIT` = 48 replaces `scroll_captures` = 12 for the enumeration
    pass, and `Profile.photos` carries those frames unchanged into `decider.decide` (worker.py's
    auto loop) exactly as before. Observe — the mode that produces every label the taste model is
    trained on — is excluded from enumeration by `_item_enumeration_blocker`, so it still reads 12.
    Per-profile pooling is not scale-free: ArcFace is a raw mean over detected faces, so a card
    that now spans three times as many frames carries three times the weight, and CLIP's
    `dedup_by_cosine(0.85)` thins near-duplicates without equalising them. The consequence is
    train/serve skew on the one model whose whole job is to predict the owner's taste, plus ~4x
    the embedding compute and ~4x the `profile_photos` archive rows per profile. This is a
    consequence of the capture change, not of the targeting change, and it is not discussed
    anywhere above. It needs either a decision (accept it, and re-measure the ranker) or a
    separation (enumerate for items, hand the ranker a subsampled frame set).

**One property the owner asked for that this workflow has quietly weakened, ahead of 5.9.** The
observe-is-the-auto-canary rule ("the opener shown to type in observe must be byte-identical to
what auto would send") no longer holds by construction: auto sends an `ItemRequest` of numbered
crops with no anchor, observe sends raw scroll frames with an anchor. The SCHEMA is shared, which
is what 5.7 asked for, but the request payload is now mode-dependent, which is the same defect one
layer down. It is currently moot only because auto sends nothing at all; the moment navigation is
wired, the two modes are generating from different inputs. 5.9's inversion is what re-converges
them and it should be treated as a repair of this, not only as a feature.

**What was verified clean.** Scope held: `hub.html` contains no `item_index`, `item_description`,
`model_item` or targeting reference, and `worker._observe_loop` / `_wait_for_observed_decision`
are structurally untouched by the new machinery (the only `driver.like` call in the file is inside
`_auto_loop`). Humanization intact: every enumeration gesture is `self._scroll_down_one(frac,
x_frac)` and every navigation gesture is `_scroll_to_top` / `_scroll_down_one`, so `_scroll`'s
forbidden-zone assertion and the `_capture_scroll_ledger` append still stand behind all of them;
none of the eight leaf modules references `touch`, `scroll_up` or `adb` except in prose. The owner
rule holds on inspection: `_locate_target_heart` returns the asked-for heart or raises,
`operation_love` has exactly one `adb.text(` call site and it sits strictly below
`_verify_sheet_shows`, and `ItemTargetingError` is caught in exactly one place (`worker.py`'s auto
loop) which stops rather than continues. Observe end to end: unchanged cadence, frame count, dwell
and anchored suggestion; the only deltas are `like()`'s signature (a method observe never calls)
and `Profile.items_unavailable` now carrying the observe-exclusion sentence.
`_item_enumeration_blocker` returns on its FIRST condition for an observe session, so observe does
not even pay the template load the other three conditions would cost.

**The residual on the legacy path is unchanged and is worth restating because it is the only
mis-aim left in the codebase.** `_locate_target_heart` takes the TOPMOST heart on the matched
frame, so a frame showing a photo and a prompt at once can still tap the neighbour, and on that
path there is no post-tap check to catch it. It is unreachable from Hinge auto today (every auto
like stops earlier) and it is the path a non-enumerating opener-capable driver would use.

**BLOCKERS, restated in the order they bite.** 1 (nothing spends the table), 3 (`navigate_to_item`
has no `should_stop`), 4 (the stop surface must route `ScrollStepError`, `SegmentationError`,
`ShiftEstimationError`, `ScrollTopUnconfirmed` by type), 8 (`**plan_kwargs`), 9 (no absolute accept
ceiling in `verify_sheet_item`) and 10 (`_IDENTITY_MATCH_MAX_DIST` = 3.0 is measured too wide) are
all unchanged. 2 (the capture ceiling) is closed by `_ENUMERATION_CAPTURE_LIMIT`. 5 and 6 remain
answered only as far as they can be offline. 11 and 12 above are new. 7 is unchanged and is the
one that governs everything else: **nothing in Part B has touched a device.**

**BEFORE AUTO IS EVER RUN, with the owner present.** Confirm on the Pixel 7a that a real
enumeration read completes inside 48 frames and that `_scroll_to_top` genuinely returns to the top
after ~48 forward scrolls (the undo budget is ledger-derived, so this is a physical-travel
question, not a counting one); measure what ~99 gestures and ~3x dwell per profile do to Hinge's
own behaviour before any volume is run; take the two calibration numbers this doc keeps deferring
(the absolute accept term for `verify_sheet_item`, and a bound under 2.565 for
`_IDENTITY_MATCH_MAX_DIST`); re-measure the bottom-anchored sheet rule against the live sheet; and
re-check `_SCROLL_TOP_BAND_FINGERPRINT` against the account's own filter chips. Every one of those
fails as a stop rather than as a wrong like, which is why they are safe to discover on a device —
but they are discoveries, not confirmations.

#### Addendum 2026-08-12: three more defects, found by two independent audits, all three fixed

Recorded here rather than as a new numbered finding because all three span sections already
written above and this section is where the blocker/defect ledger lives. Nothing above was
edited. Offline only -- no device, no live API call, nothing read from `ops/calibration/`. Suite
**1770 -> 1776 green** (6 new regression tests, one per fix below plus two extra edge-case tests;
see each fix for the count).

**BUG 1 (MAJOR, live today), FIXED: the observe prompt contradicted itself.** The 5.1 addendum
above ("the selection criterion is now stated in the prompt, in both copies") added an
unconditional PICK THE ITEM YOURSELF paragraph to `config.yaml`'s `opener.style` -- the block
`_text_part` sends, byte-for-byte, as the user turn's STYLE GUIDE on EVERY request regardless of
shape. Observe still generates via the anchored shape (5.1/5.6's "the anchor survives ONLY for
observe" addendum), whose own closing paragraph tells the model the item is ALREADY decided ("YOUR
MESSAGE ATTACHES TO THE PHOTO OR PROMPT IN THAT LAST IMAGE ... even if another photo or prompt
seems more interesting"). So an anchored request carried two live, opposing instructions about who
chooses the item in the same call, with nothing resolving the clash -- reproduced by building the
real request via `GeminiOpener._text_part(profile, cfg.opener.style, anchored=True)` against the
real shipped config.

`opener.py`'s `_SYSTEM` copy of the same paragraph did NOT have this bug: `_ANCHOR_SYSTEM` already
explicitly overrides it ("THIS OVERRIDES PICK THE ITEM YOURSELF ABOVE ... you do not choose one")
before the model ever writes, so that copy is genuinely shape-aware. **Fix: the three
paragraphs (PICK THE ITEM YOURSELF, THE FAILURE TO AVOID, THE UNNUMBERED IMAGES ARE CONTEXT) were
DELETED from `config.yaml`'s `opener.style`, not given a matching override.** Two designs were
available -- build a second, independent anchored-override mechanism for this copy, or delete the
duplicate and let the one already-correct, shape-aware copy in `_SYSTEM` be the sole source. The
second was chosen: doc 3.1 already names "the duplication trap" (two near-identical copies pinned
by two separate tests, editable independently) as the standing risk, and this bug is exactly that
risk materialising -- an edit landed in one copy (adding the paragraph to both, per the 5.1
addendum) without re-deriving whether the OTHER copy's existing safety mechanism (an anchored
override) needed to travel with it. Building a second override would leave two independently
maintained shape-aware mechanisms doing the same job, which is more surface for the next edit to
drift on, not less. `_SYSTEM`'s copy is compressed, not abridged -- the tradeoff, the failure mode
and the context tier (and why to use it) are all still there in full -- so nothing the model needs
to choose well was lost, only the unconditional, unguarded duplicate of it.

Part A wording (the one rule, the premise/point distinction, the five moves and their escape
clause, "a claim she can correct beats a question she has to answer", the three guardrails, the
widened hedge list, VARY THE OPENING, the two-sentence cap, the economy rule, both HARD RULEs) is
untouched in both copies -- this fix removed exactly the three item-selection paragraphs the 5.1
addendum added and nothing else. `tests/test_config_yaml_real.py`'s
`test_shipped_opener_style_ships_the_item_selection_rule` is now
`test_shipped_opener_style_does_not_ship_the_item_selection_rule`, pinning the paragraphs'
ABSENCE from `config.yaml` rather than their presence, plus a positive check that the surrounding
Part A text (the five moves, "a claim she can correct...") survived untouched.
`tests/test_opener.py`'s `test_system_prompt_keeps_faithful_corey_opener_policy_and_two_sentence_cap`
keeps every existing assertion (no wording changed) with its docstring and inline comments
corrected to state that `_SYSTEM` is now the item-selection paragraph's ONE remaining home rather
than one of two synced copies. A new end-to-end regression test,
`test_anchored_style_guide_never_carries_the_item_selection_instruction`, reproduces the audit's
own repro verbatim (`_text_part(profile, cfg.opener.style, anchored=True)` against the real
config) and asserts the self-selection phrasing is absent while the anchored closing paragraph's
"write about that item only" survives.

**BUG 2 (BLOCKER), FIXED: `opener.enabled: false` hard-stopped a Hinge AUTO run.**
`_item_enumeration_blocker` gated enumeration on `_auto_session` and `accepts_opener` (a static
per-app capability, Hinge is always `True`) but never on whether an opener would actually be
requested. `opener.enabled: false` constructs `OpenerService(client=None, ...)`, which is
`disabled` from construction but is NOT `None` -- `worker.py`'s own comment already named this
exact trap two paragraphs above the bug ("There, `self.opener_service.disabled` is True ... only a
live, still-enabled service that just failed on THIS call reaches here"), but the EARLIER guard in
the same method (the one reading `profile.items_unavailable`) never got the matching check. So a
disabled-opener Hinge AUTO run still paid for a ~40-frame enumeration read on every profile, and
any refusal in it (an unconfirmed scroll top, a spacing no gesture could respect, ...) reached that
guard and stopped a run that was never going to send an opener at all. Reproduced through the
shipped `Worker` with a disabled `OpenerService`: `state=stopped`, `stop_kind="opener"`,
`driver.likes == []`.

**Decision: auto keeps running and sends plain (commentless) likes when openers are administratively
disabled -- it does not refuse to run at all.** This was not a fresh call: `opener.enabled: false`
sending bare likes in AUTO is pre-existing, deliberately tested behaviour
(`tests/test_worker.py::test_worker_with_opener_disabled_by_config_still_likes_normally_in_auto_mode`,
untouched by Part B and still green), and the owner's "never a commentless like" rule is scoped to
a BAD AI RESPONSE ("if the response from the AI is bad, redo the prompt ... if after 5 attempts
it's still a bad response, stop" -- config.py's `OpenerCfg.max_attempts` comment) rather than to an
operator's deliberate choice to run without openers at all. Rewriting that precedent to "auto
cannot run without openers" would be the OTHER defensible answer the task allowed, but it would
mean deleting or inverting an existing, passing, intentionally-written test with no new evidence
that the precedent was wrong -- which is not what this fix is for. So enumeration, whose only
consumer is opener generation, was taught to recognise when there is no consumer.

Two changes, at the two layers that each independently mattered. (1) **The driver now knows
whether openers exist for this session.** `AndroidDriver` gained `self._openers_enabled` (default
`True`, preserving every existing caller including calibration tools that never call the new hook)
and `set_opener_enabled(enabled)`, called once by `Worker._auto_loop` right alongside
`set_auto_session_policy`, from `opener_service is not None and not opener_service.disabled` --
the exact condition already used elsewhere in the same method. `_item_enumeration_blocker` gained
a second real POLICY check (alongside the pre-existing observe exclusion) that refuses before a
single frame is read when openers are off, so the ~40-frame closed-loop scroll never runs at all
for a disabled-opener session -- not merely "does not stop the run", the doc text's stronger claim
("enumeration should not run") is honoured literally. (2) **The worker's stop guard now also
checks `disabled`**, mirroring the identical check the "pick is None" guard a few lines below it
already used: `if accepts_opener and self.opener_service is not None and not
self.opener_service.disabled:` gates the whole `items_unavailable` block. This is the guard that
actually prevents the hard stop and is deliberately NOT redundant with (1) -- `set_opener_enabled`
is a best-effort optional hook (absent on any fake or future driver that does not define it), while
`disabled` is the service's own authoritative state, and a driver that enumerates anyway (a fake,
an older driver) must still not trip the stop.

Four new tests. `tests/test_hinge_item_capture.py`:
`test_openers_disabled_does_not_enumerate_and_says_so_rather_than_going_quiet` (mirrors the
existing observe-exclusion test; calls `set_opener_enabled(False)` on an otherwise-AUTO read and
asserts the ordinary, non-enumerating cadence and ceiling) and
`test_openers_enabled_by_default_preserves_every_other_enumeration_test` (pins the safe default).
`tests/test_worker.py`: `test_auto_does_not_stop_on_items_unavailable_when_openers_are_disabled`
mirrors `test_auto_stops_when_the_capture_could_not_produce_numbered_items` with the one variable
that matters changed (a disabled `OpenerService`) and asserts the run does NOT stop and the like is
sent bare, exercising the worker-side guard independently of the driver-side one.

**BUG 3 (BLOCKER, unintended consequence), FIXED: Part B silently changed the ranker's input from
~12 frames to up to 48.** `_ENUMERATION_CAPTURE_LIMIT` (48) correctly replaced `scroll_captures`
(12) as the CEILING for a read that also builds an item index -- the index needs the finer,
closed-loop cadence doc 5.10.1 derived to avoid step/spacing aliasing, and that part was not
touched. But `_capture_current` handed every frame it read straight to `Profile.photos`
regardless of which ceiling was in force, and `Profile.photos` is also the RANKER's entire view of
the profile (`decider.decide`, `worker.py`'s auto loop). Per-profile pooling is not scale-free
(`aggregation-design.md`: ArcFace is a raw MEAN over every detected face; CLIP's
`dedup_by_cosine(0.85)` thins near-duplicates without equalising their weight), so a card sampled
~4x as often at the enumeration cadence would carry ~4x the weight it carried before Part B -- and
only on AUTO, since observe (the only mode that produces the labels the ranker is trained against)
never enumerates and always reads at the 12-cadence. Left alone this is train/serve skew dressed up
as a ranker regression, plus ~4x the embedding compute and ~4x the `profile_photos` archive rows
per profile, exactly as the finding described.

**Fix: separate the two consumers rather than shrink the enumeration ceiling** (shrinking it was
explicitly ruled out -- it would realias the scroll step against card spacing per 5.10.1 and break
the item index the ceiling exists to build). A new pure function,
`hinge._ranker_frames_from_enumeration(frames, target)`, resamples the FULL enumeration-cadence
capture back down to `target` frames -- `self.scroll_captures`, the configured base with no jitter
applied (the jitter `_capture_limit_for_profile` adds exists to keep the DEVICE-facing scroll
ceiling from being a fixed, externally observable bot signature; this function runs entirely after
every gesture for the profile has already happened and produces nothing Hinge's servers or a human
ever see, so that reasoning does not transfer). It is only applied to the copy that becomes
`Profile.photos`; the copy `_index_captured_items` folds into the item index and crops is the
full, untouched capture, computed first and unaffected by anything below it. Evenly spaced across
the whole captured range (`round(i * (n - 1) / (target - 1))` for `i` in `0..target-1`,
deduplicated), always including the first and last frame, so the ranker keeps seeing top-to-bottom
coverage rather than the head start a naive "keep the first `target` frames" would give it.
Whether the ceiling was raised at all for a given capture (`enumeration_ceiling_raised`) is
captured once, before the loop, separately from the possibly-mid-loop-mutated `enumerating` flag --
a read that raises the ceiling and then falls back to the ordinary cadence part way through (a
`ScrollStepError`) can still walk all the way to 48 frames at that ordinary cadence, and the
ranker must not see all 48 of those either.

**What the ranker now receives, and why that is equivalent, stated plainly rather than assumed.**
It receives `self.scroll_captures` frames (8 by the code-level `HINGE_SPEC` default, 12 as shipped
in `config.yaml`) spanning the same top-to-bottom range the enumeration read covered, evenly
resampled from the finer capture -- the same COUNT as before Part B, covering the same PROFILE, at
a coarser but even spacing. **Not exact equivalence, and the difference is quantified rather than
papered over**: the pre-Part-B read walked the profile in big (~1299px at the shipped 0.55
`read_scroll_frac`) steps and stopped the moment a repeated frame signalled the bottom, so a SHORT
profile could see fewer than `target` frames at that coarse spacing; this function instead
resamples whatever the finer capture already read, which on a short profile can return closer to
the full `target` frame count spanning the same short page -- i.e. it can hand the ranker a
slightly denser sampling of a SHORT profile than the old cadence ever would have, though never more
frames than the device actually produced and never a synthesised one. The ranker's own pooling
(ArcFace's mean, CLIP's dedup+GeM) already collapses near-duplicate detections, so an extra
same-content frame that would not have existed under the old cadence contributes at most a
near-duplicate of a frame the old cadence WOULD have kept, which is why the residual is judged
acceptable rather than closed further. No live re-measurement of the ranker was performed or is
claimed; this is an input-equivalence argument, not a recalibration.

Three new tests in `tests/test_hinge_item_capture.py`.
`test_ranker_frames_from_enumeration_is_pure_and_evenly_spaced` unit-tests the function directly
(untouched-below-target, thinned-above-target with first/last frame preserved and no reordering or
duplication, and the degenerate `target<=0`/`target==1`/empty-input cases named in its docstring).
`test_the_ranker_still_sees_scroll_captures_frames_not_the_enumeration_ceiling` drives a real
synthetic capture through `_capture_current`, confirms the fixture actually needs thinning to
exercise the fix (`len(served) > drv.scroll_captures`), and asserts `Profile.photos` is exactly
`scroll_captures` frames, a genuine subset of what was served, covering from the first frame to
within one frame of the last (the very last served frame is the repeat that told the loop it had
reached the bottom, deliberately never kept in `photos` at all) -- while the item index itself
(`profile.items`, `profile.item_context`) is unaffected, because it was built from the full capture
before any thinning happened.

**None of the three was a misdiagnosis.** All three reproduced exactly as described, against the
shipped code, before any fix; the two BLOCKERs (2 and 3) match findings 11 and 12 already recorded
in this section's earlier addendum almost verbatim, and BUG 1 reproduces with the exact repro
command the assignment specified. Suite 1770 -> 1776 green, 6 net new tests: 1 for BUG 1
(`test_anchored_style_guide_never_carries_the_item_selection_instruction`; its sibling,
`test_shipped_opener_style_ships_the_item_selection_rule`, was rewritten in place into
`test_shipped_opener_style_does_not_ship_the_item_selection_rule` rather than added, so it is not
counted twice), 3 for BUG 2
(`test_openers_disabled_does_not_enumerate_and_says_so_rather_than_going_quiet` and
`test_openers_enabled_by_default_preserves_every_other_enumeration_test` in
`test_hinge_item_capture.py`, `test_auto_does_not_stop_on_items_unavailable_when_openers_are_disabled`
in `test_worker.py`), 2 for BUG 3
(`test_ranker_frames_from_enumeration_is_pure_and_evenly_spaced` and
`test_the_ranker_still_sees_scroll_captures_frames_not_the_enumeration_ceiling`, both in
`test_hinge_item_capture.py`).

#### Addendum 2026-08-12: the REWIND IS GONE — navigation walks UP, and AUTO can finally like

Owner-approved. This closes §9's blocker 1 ("nothing spends the table") and rewrites the second
half of 5.5's own sentence: "scroll to top, then step forward counting distinct hearts" is no
longer what happens. The enumeration read still starts from an affirmatively confirmed scroll top
— that is what makes heart ordinals ABSOLUTE and nothing about it changed — but the top has
stopped being the only anchor navigation will move from.

**What the rewind cost, and why the cost was the smaller half of the argument.** Auto used to make
three passes over one profile: enumerate down (up to 48 gestures), `_scroll_to_top()` undo every
one of them (~51 swipes, `len(ledger) + 3`), then `_locate_target_heart` scroll to the top AGAIN
and walk forward to item N (up to 48 more, doubled by `_TARGET_HEART_ATTEMPTS`). A mid-profile item
cost ~120 gestures and a deep one with a retry approached 200, against ~27 before Part B. The two
reasons the owner approved replacing it are both about correctness rather than about the count, and
both are now stated in `item_nav.py`'s module docstring so they survive the next edit:

- **The anchor becomes MEASURED rather than REPLAYED.** `_scroll_to_top` sizes its swipe budget
  from a counted ledger and decides it arrived by a pixel-diff settle heuristic whose result
  "every pre-existing caller ignores" (5.5's own words). Bottom-up joins this pass's page space to
  the index's with ONE `frameshift.estimate_shift` against the very frame `ItemIndex.offsets[-1]`
  was measured on, and that shift is **exactly 0px** whenever nothing moved the card between the
  read and the like — which is what `hinge._capture_current` produces by construction, because
  both of its terminating paths leave the screen showing its own last KEPT frame (the
  repeated-frame break happens BEFORE the repeat is appended; the ceiling path's final iteration
  issues no scroll). There is no second origin to reconcile: every page row this pass measures is
  already in the index's coordinates.
- **It removes the mechanical down-up-down pattern.** 5.11 already argues that returning to an
  item is more human, not less; what it did not say is that the rewind made the bot do it in the
  one shape a person never would — a full-speed sweep to the very top, then a second full-speed
  sweep back down. That pattern is a stronger signature than the raw gesture count.

**GESTURES PER PROFILE, against the corpus's own geometry** (profile A 10027px at the loop's
measured 233..262px cadence, profile B 8349px):

| | rewind (until today) | bottom-up |
|---|---|---|
| best case, the LAST item on the page | ~51 rewind + ~1 forward = **~52** | **0** |
| typical, a middle item | ~51 + ~22 = **~73**, and ~120 once `_locate_target_heart`'s own scroll-to-top and search are counted | **~22** |
| worst case, item 1 | ~51 + ~43 = **~94**, doubled by `_TARGET_HEART_ATTEMPTS` to **~188** | **~43** |

Bottom-up's WORST case is the rewind's BEST case, and its best case is free. The saving is exactly
the rewind: walking up to item k costs what walking down to it cost, and nothing else. Measured on
the synthetic closed loop, the last item takes 0 gestures, cost rises monotonically up the page,
and the worst item costs no more than the READ's own walk over the same page.

**THE FOUR SIGN FLIPS, and the two that deliberately needed no code.** These are where an
off-by-one hides, so they are written as a table in `item_nav.py` and each is implemented as the
`ascending` branch of a function whose other branch is the descending original — BOTH branches
tested directly, because an inverted comparison in either would simply never fire and would pass
any end-to-end test.

| quantity | descending (the enumeration read) | ascending (navigation) |
|---|---|---|
| `estimate_shift` | delta POSITIVE | delta NEGATIVE |
| progress / stall | `delta <= 0` is a stall | `-delta <= 0` is a stall, which now also covers "it moved the WRONG WAY" |
| new content arrives | at the band's BOTTOM edge | at the band's TOP edge |
| a missed heart looks like | first seen below the previous band's top | ...above its bottom |
| the count's anchor | heart 1, at a confirmed top | the bottom-most heart seen, ordinal matched by nearest page row |
| ordinals run | upward from the anchor | downward from it |
| "we went past it" | heart above the band's top row | heart below its bottom row (`NAV_ITEM_BELOW_ENTRY`) |

The two that needed no flip: the page-space recurrence (`offsets[i+1] = offsets[i] + delta`) is
UNCHANGED, because `estimate_shift` measures the sign and an upward gesture makes the offset
decrease on its own; and the ordinal arithmetic in `_count_disagrees` is direction-FREE — every
cluster's ordinal is `anchor_ordinal + (its position - the anchor's position)`, which counts up
from a first-heart anchor and down from a last-heart one with no second expression to keep in step.
The only direction-dependent line left in that function is the choice of anchor, which is four
lines and is tested both ways.

**THE REVERSE GESTURE IS A DIRECTION OF `_scroll`, NOT A SECOND PATH TO THE TRANSPORT.**
`hinge._scroll_up_one(frac, x_frac)` goes through `_scroll(..., reverse=True)`, so it inherits the
forbidden-zone guard, `scroll_x`'s shared column jitter and the humanized kinematics; the only
difference is which end the finger goes down on. Three things about it are deliberate:

- **both end rows are zone-checked**, which is stricter than `_swipe`'s start-only rule. The
  reverse stroke swaps touch-down and release, so "the start" is a row no other read-scroll in
  this driver presses, and at the enumeration fracs (0.10..0.16 measured) both rows sit inside
  0.42..0.58 of the screen so nothing legal is refused by asking for both;
- **both arguments are required, with no defaults.** `_scroll_down_one(frac=None, x_frac=None)`
  re-samples BOTH from the behaviour policy when either is None, which silently issues
  production's 0.55 cadence — the exact aliasing cadence this whole pass exists to avoid. Having
  nowhere to put a `None` is the cheapest way to make that unreachable;
- **the ledger is APPENDED to, not popped.** Its only consumer is `_scroll_to_top`, where
  `len(ledger)` is a CEILING on undo-swipes and the settle check is what actually ends the loop.
  Overstating outstanding downward travel is free (at the top, one downward swipe changes no
  pixels and the loop exits on its first iteration); UNDERSTATING it leaves the card scrolled and
  the next capture's identity anchor seeded from a real person's sticky header. Popping one
  forward entry per reverse gesture would understate whenever a reverse step is smaller than the
  forward step it undoes, which the per-frame ratio rule makes likely.

Aliasing does not care which way the finger went, so `plan_scroll_step` sizes every reverse
gesture against the card in front of it exactly as the enumeration read sizes a forward one, and
`step_overshoot` is handed the climb magnitude so its two tests keep their meanings unchanged.

**Addendum 2026-08-26: one pre-count lower-edge positioning recovery, not a reversal.** A live
Lara navigation exposed the one geometry the ascending rule alone cannot repair. Page heart 9 was
still the correct translated target, and the index projected its whole card inside the analysed
band (frame rows 901..2010 against 300..2100), but the fresh segmentation could not establish the
card's LOWER edge: it ended in a 90px `background_run`, whose blank pixels can be page background
or an unobserved blank tail of the card. The heart was visible, but no complete target block could
be returned. An upward gesture moves that unresolved lower edge DOWN; the old loop issued one and
then correctly refused only after the target had fallen below the band. The refusal was safe, but
the direction was needlessly terminal.

`navigate_to_item` now has exactly one narrow exception **before a frame contributes an
observation or a heart cluster, and before any ascending gesture**. It may make one planned,
humanized `_scroll_down_one(frac, x_frac)` only when all of these are true:

- this frame's own matched target heart is inside the analysed band;
- the target block has a trustworthy TOP edge and an unobserved LOWER edge (either a band edge or
  an untrusted `background_run`); this deliberately excludes a top-edge partial, for which a
  forward movement would hide the missing edge farther above the band;
- the indexed card's measured full height fits in the band, and the **planned** legal step leaves
  both indexed card edges in it. A too-tall card, or a step that would move either edge out, is a
  refusal without a forward gesture.

The lane still comes from the behaviour sampler, the distance from `plan_scroll_step`, and both
arguments are passed to the normal guarded driver primitive. The resulting forward displacement
must be a positive `estimate_shift` and must respect that plan's `step_overshoot` bound. The next
frame is segmented independently; the same indexed heart must now belong to a complete block. Any
unmeasurable shift, stalled/wrong-direction/oversize step, or still-incomplete target is
`NAV_ENTRY_POSITION_UNRESOLVED`; no second positioning stroke, reverse stroke, nearest-heart
fallback, or tap is permitted.

On success the new page offset is measured and the loop restarts from that frame with empty
observations, clusters, and reverse-step history. Thus the normal ascending count is one clean
count from a measured origin; the forward movement did not reverse or reuse a count that had
already been spent. The existing scroll-top identity recovery also consumes the single sanctioned
forward entry gesture. If it has already run, a later lower-edge uncertainty raises
`NAV_ENTRY_POSITION_UNRESOLVED` immediately rather than attempting a second forward scroll.
`tests/test_item_nav.py` pins the live-style 90px-background-run case, a top-edge control, an
unmeasurable positioning step, a too-tall-card refusal, and the combined scroll-top/low-edge path.

**THE HANDOVER IS TAKEN.** `hinge._like_comment_sheet`'s `model_item_index is not None and
item_index is None` branch — which the code itself named as the handover point — is now
`_navigate_to_model_item`, and there is no `_scroll_to_top` on that path at all.
`_confirm_payload_profile` is KEPT above it even though `navigate_to_item` opens with the same
comparison: it is one band decode, it is the only identity gate on the path if `like()` is ever
called with a model item number by something that is not this branch, and the two guards fail in
the SAME direction on the same input (this doc's own measurement: the closest two DIFFERENT people
sit 2.565 grey levels apart against a 3.0 bound), so neither may be dropped because the other
exists. The legacy capture-order path is byte-for-byte what it was, including `item_index=0` beside
a model item number.

**AND THE WORKER NOW PASSES THE NUMBER.** `INDEX_SPACE_MODEL_ITEMS` is no longer
untranslatable-in-practice. `OpenerPick.capture_order_index` still returns `None` for it, correctly
and unchanged — the model's number resolves to a HEART ORDINAL and not to a captured frame, and
inventing a frame index for it is the bug the two spaces exist to prevent — but `worker._auto_loop`
now branches on `index_space` rather than on that `None`: a model-item pick goes to
`driver.like(model_item_index=...)` with `item_index` left at `None`, which is the one combination
the driver routes to counting navigation. The hard stop narrowed from "any pick without a
capture-order index" to `ITEM_INDEX_ABSENT` and any space this build does not recognise. So the
2026-08-12 audit's headline — "AUTO STILL CANNOT LIKE" — no longer holds, and the two guards that
audit called "correct, measured, and dead code until the navigation call is wired"
(`_confirm_payload_profile` and `_verify_sheet_shows`) are now both reachable in production.

**§9's blocker 4 is closed at the driver boundary.** `ScrollStepError`, `SegmentationError`,
`ShiftEstimationError` and `IdentityError` propagate UNCODED out of the navigation stack by design;
all four (plus the coded `ItemNavigationError`) become `HingeTargetingError` in
`_navigate_to_model_item`, which `worker.py` already routes to a stop with `intended`/`actual`
rendered. Translating at the driver rather than at the worker is what keeps `worker.py` free of
Hinge symbols, which is why `base.ItemTargetingError` exists. **Blocker 8 stays closed by the same
mechanism `_plan_enumeration_step` uses**: `_navigate_to_model_item` forwards no `**plan_kwargs`
at all, so a production caller has nowhere to widen the validated envelope from.

**Two new refusal codes**, joining the fourteen: `NAV_ANCHOR_UNMEASURED` (the screen could not be
put in the index's page space, or it has DRIFTED from where the read left it by at or beyond the
smallest gesture this driver can make — a physical bound, not noise slack, since the expected
drift is exactly 0) and `NAV_ITEM_BELOW_ENTRY` (the target is below the analysed band on the entry
frame, so walking up only takes it further away). `NAV_TOP_UNCONFIRMED` and `_confirm_top` are
GONE, along with `_MAX_TOP_ATTEMPTS`; `navigate_to_item` no longer calls `_scroll_to_top`, reads
`scroll_captures` or writes `_capture_scrolls`.

**ONE THING THIS TRADES AWAY, stated plainly because it is a real loss.** Under the rewind, both
passes anchored on their own confirmed scroll top, so a foreign profile whose first card sat
`_TOP_ORIGIN_RESIDUAL_PX + tolerance` (218px) or more from the index's showed up as a geometric
ANCHOR gap. Bottom-up has no second origin: the entry shift is MEASURED, so a page that is a
uniform translation of the indexed one is absorbed by that measurement and every relative heart
pitch then agrees exactly rather than merely closely. That layer is gone for translations under
one gesture floor (219px). What replaces it is arguably stronger and is stated in the module
docstring: on REAL frames the entry anchor itself refuses a foreign profile, because
`estimate_shift` requires the screen to CORRELATE with the read's own last frame and two different
people's cards do not — the synthetic fixture's "foreign" page is byte-identical content at a
different offset, which is a worst case rather than a likely one. Above the gesture floor the new
drift bound catches it. And the identity gate, which is the layer that actually answers the
question, is unchanged and now runs on the ENTRY frame — a frame that is scrolled by construction,
where the sticky header is readable, instead of on a post-rewind frame where the same strip is
Hinge's own chrome and carries no identity at all. The cross-profile matrix is still 100% refused,
synthetically, in both directions, at every model index, with zero gestures issued.

**`_ENUMERATION_CAPTURE_LIMIT`: RE-DERIVED, AND THE MEASURED GEOMETRY DOES NOT LICENSE TIGHTENING
IT.** This workflow was asked to tighten 48 on the reading that "~28 steps covers a ~10,000px
profile at ~363px". That reading takes `scroll_step._MAX_STEP_PX` as the step, and the loop never
draws it: the window's ends are the ratio rule applied to the SMALLEST spacing seen so far on the
profile, and BOTH calibration profiles contain a 685px card, so both loops end up drawing from
(219, 265). The realised cadence is 233..262px, which is why this doc's own closed-loop addendum
measured 35 gestures (36 frames) for profile B and 43 (44 frames) for profile A. The derivation
that ships is therefore:

| | |
|---|---|
| worst measured page | 10027px, and NINE selectable items — Hinge's own maximum (6 photos + 3 prompts), so close to the structural worst case rather than a sample |
| worst measured frames | **44** |
| safety factor for a longer page | x1.1 -> 48.4, floor **48** |
| absolute worst case, every draw on the 219px gesture floor | `ceil(10027/219) + 1` = **47**, also covered |

So the constant is unchanged at 48 and its justification is replaced: it was "the ceiling the
validated probe ran at", which is headroom, and it is now the arithmetic above. Shaving it to 46
would buy ~2 frames of dwell — a cost 5.11 wants to pay anyway — against a real truncation risk on
the longer of only two measured profiles, and a tighter value could only come from a faster
cadence, which 5.10.1 forbids.

**The other half of that requirement — fail loud rather than truncate silently — IS shipped.**
Truncation was already recorded (`ItemIndex.truncated` -> `Profile.items_truncated` -> the model,
per 5.7), but only the MODEL was ever told, so "silently" was a fair description from the
operator's chair. `_note_enumeration_truncated` now prints the frame count against the derived
ceiling and writes a `capture_enumeration_truncated` debug action, on enumeration reads only — the
ordinary `scroll_captures` read keeps its pre-existing quiet `capture_truncated` flag, because a
notice that fires on every profile is one nobody reads. It is deliberately NOT a hard stop: a
profile longer than the ceiling is not an error, the owner's stop-condition rule scopes stops to an
unrecognized screen or an error rather than to a quota, the item the model picks is inside the
enumerated region by construction so navigation is unaffected, and what is actually lost is Connect
material from the tail — a quality cost, not a correctness one.

**§9 blocker 3 (`navigate_to_item` has no `should_stop`) is MITIGATED, not closed, and that is
deliberate.** The auto like path never receives a stop callable at all — `_scroll_to_top`'s own
STOP paragraph records why ("the auto path's like()/`_locate_target_heart` unwinds never receive
them and so can never be interrupted mid-commit") — so adding one here would be a new contract on
`base.Driver.like`. What changed is the SIZE of the window rather than its existence: it was a full
rewind plus a full forward walk (~51 + up to 48 gestures, doubled on a retry), and it is now 0
gestures for an item near the bottom and at most one walk up from where the read ended. The frame
budget still bounds the loop and is now derived from the ASCENT rather than from the descent.

Offline only — no device, no live API call, and nothing from `ops/calibration/` or
`data/hinge_debug/` was read, sent or described for this workflow. Every fixture is synthetic: the
tall painted world `tests/test_item_nav.py` already builds, served through a fake driver whose
`_scroll_up_one` moves by exactly what the measured transport model says a `frac` delivers, with
the real segmenter, the real shift estimator and the real shipped scroll-top fingerprint running
against it. `tests/test_item_nav.py`'s double now RAISES from `_scroll_to_top` and
`_scroll_down_one` rather than implementing them, so a reintroduced rewind fails every test in the
file by name instead of quietly working at ~100 gestures a profile.

**Still owed after this**, and unchanged: §9's blockers 7 (nothing in Part B has touched a device),
9 (no absolute accept ceiling in `verify_sheet_item`) and 10 (`_IDENTITY_MATCH_MAX_DIST` = 3.0 is
measured too wide) — both of the last two being calibration acts with the owner and a device
present. Also unchanged: 5.8's pre-flight cross-check, the observe inversion, the hub changes
(including a targeting-specific stop banner and `item_description` display), and the residual
`_locate_target_heart` still carries on the legacy capture-order path. **NEW and worth carrying
forward:** the before-auto-is-ever-run list in the previous audit said to "confirm that
`_scroll_to_top` genuinely returns to the top after ~48 forward scrolls" — that question is now
much less load-bearing, because no `_scroll_to_top` runs between the read and the tap at all, but
it is not gone: `_ensure_session_top` still depends on it once per session, and the ledger a
navigation leaves behind is now longer than the read's rather than shorter.

#### Addendum 2026-08-12: OBSERVE IS INVERTED — it runs auto's pipeline, and the canary is back

This section's own design is now built. Observe no longer generates AFTER the human taps: it
enumerates the profile, sends the SAME `ItemRequest` auto sends, the model picks the item and
writes the opener, and the hub tells the human WHICH ITEM to like plus the text to type — before
they touch anything. They still tap and still type; nothing is automated for them.

**What this repairs is named in the previous audit and is bigger than a feature.** That audit's
own closing note ("one property the owner asked for that this workflow has quietly weakened")
recorded that auto sent numbered crops with no anchor while observe sent raw scroll frames with
an anchor, so the two modes issued genuinely different requests — different image count,
different image content, different closing text, and a different cognitive task (report vs
choose). Testing observe therefore told the owner nothing about auto. Both modes now build
`opener.ItemRequest.from_profile(profile)` and pass no anchor at all.

**THE ONE SURVIVING DIVERGENCE, stated because this section is exactly where a silent one would
hide.** Observe still calls with `advisory=True`, i.e. ONE attempt instead of `max_attempts`, and
its exhaustion still routes through `_exhaust(request_stop=False)`. That is a RETRY POLICY
difference and never a request one: the crops, the schema and the prompt are byte-identical
(pinned by `test_observe_suggestion_and_auto_like_type_the_identical_opener_the_client_returned`,
which now asserts the request as well as the text), so any opener observe DOES show is what auto
would send. What differs is coverage — a first-attempt parse failure yields no suggestion here
where auto would retry — and it is kept for the two reasons the advisory flag was introduced for:
an advisory failure must never set `stop_requested` and end a labelling session, and observe now
calls on EVERY card rather than only on hearted ones, which is the second divergence and the one
with a bill attached. Auto asks only when its ranker decides to like; observe cannot know what
the human will do, so it asks for every profile. Expect roughly 3-5x the calls per session
against the free-tier quota, and note that gating observe on the ranker's own verdict would be
worse than the cost: the ranker is cold in the mode that trains it, and a human who chooses to
like a profile the ranker would have passed is exactly the case a suggestion is for.

**WALL CLOCK, measured against the corpus geometry rather than estimated.** The read is what
costs: 12 frames becomes 36 (profile B, 8349px) and 44 (profile A, 10027px) at the loop's own
measured 233..262px cadence, so the ~85s per-profile figure the operator already reports becomes
roughly 3x that. **The scroll back to the top does NOT scale with it**, which is worth recording
because `_item_enumeration_blocker`'s old text implied it did: observe has no auto behaviour
policy, so `_scroll_to_top` throws `read_scroll_frac`-sized (0.55h) undo strokes and ends on its
own settle check — ~9-10 swipes over a ~10,000px page whatever the enumeration cadence was. And
5.5's bottom-up navigation buys observe **nothing**: what it removed was the BOT's
rewind-then-walk on the auto LIKE path, and in observe the human does the navigating. The saving
is real and it is auto's.

**THE MISMATCH GUARD IS ON IDENTITY FIRST, THEN ON THE ITEM, AND EITHER REFUSING MEANS REFUSE.**
`hinge.observe_item_mismatch(sheet, model_item_index)` returns `""` only when
`item_identity.compare_profile_identity` confirms the sheet is on the profile the crops came from
AND `item_verify.verify_sheet_item` confirms it is showing that item. Both run on the frame the
driver already holds — this section's own "we already see the opened sheet" — and the identity
half is possible only because of the measurement in 5.6's last addendum: the comment sheet does
not occlude the sticky header (`identity_band` cuts rows 115..226, the preview starts at 236), so
the deck-advance race this section calls "the most serious defect found reviewing this design" is
answered on identity rather than on card pixels, at the cost of one comparison and no extra
screencap.

The two guards are stacked in the only direction the measurements permit. An `IDENTITY_MATCH` is
a strong REFUSAL and a weak confirmation (the closest two different people of six measure 2.565
grey levels against a 3.0 bound), and `verify_sheet_item` is a CLOSED-SET test with no absolute
accept ceiling that was measured accepting a foreign card 10 times in 540. So the guard promises
only the contrapositive: **either one refusing is enough to withhold the text.** "Both passed" is
never treated as proof anywhere in this workflow, and the two open calibration blockers (§9's 9
and 10) are unchanged and still the owner's.

Four smaller decisions worth recording because they were decisions.

**"Could not look" renders exactly like "looked and it is wrong."** A driver that could not hand
over the sheet frame, an unreadable band, no preview on the frame, a missing payload, an
out-of-range item number: every one is a warning with no text. Under the OLD flow a missing
anchor meant "generate blind and badge it unanchored"; under the inversion it means the one check
between a suggestion and a wrong-item comment could not be made, and to an operator about to type
that calls for the same action.

**Observe refuses to show text; it does not stop.** This section's own rule ("it just enforces it
by refusing to show text rather than by stopping the run"), applied to every refusal including
the ones AUTO hard-stops on. An enumeration failure stops an auto run — there is no honest
request to make and doc 5.2 forbids falling back to raw frames — and in observe it produces no
suggestion and the driver's own sentence on the hub. Nothing about the human's decision, the
label, or the retrain changes.

**Generation is on its own thread, and that is a correctness requirement rather than a courtesy.**
This section asks for READY immediately with the suggestion filling in behind it. Blocking the
loop for up to `request_timeout_s` between publishing READY and entering `wait_for_decision`
would be worse than slow: a human who acts in that window is never observed at all, because the
driver's first frame after the wait starts is already the NEXT card — the decision lost and the
one after it attributed to the wrong profile, which is a corrupted training label. The thread
touches `OpenerService` (lock-guarded), `RunStatus.set_app` (lock-guarded), `print`, and — under
the suggestion object's own lock — the driver's `observe_item_mismatch`, which is pure vision over
a frame in hand and deliberately takes no screencap and writes no debug record (`DebugLog` appends
to one file from one thread by design, and the worker thread is writing to it). `cancel()` takes
the same lock on every exit from the wait, so once it returns no driver access can be in flight or
can start, which is the precondition for the next capture invalidating the item table.

**The hub tells the human the item BEFORE the text, and the canary rule is unchanged.** "like item
3 — the ridgeline photo" is the heading of the opener box; the opener sits alone in its own
delimited row with `escHtml(opener)` and nothing else; `referenced` and the send instruction sit
in the dimmer chrome row below. On a mismatch the box is replaced outright by a WAIT-coloured
warning with nothing to copy. `item_description` now has the display half doc 5.7 asked for and
the previous audit's "Finding 2" recorded as unbuilt.

**THE ANCHOR PATH IS NOW DEAD IN PRODUCTION, AND IS DELIBERATELY STILL HERE.** Nothing passes
`anchor=` to `OpenerService.maybe_opener` or `GeminiOpener.generate` any more — the driver's
repair re-ask went with 5.6 and observe's post-heart suggestion went with this. The
request-building machinery (`_ANCHOR_SYSTEM`, `_ANCHOR_LABEL`, `_system_text`'s anchored branch,
both anchored closing paragraphs, the anchored retry sentence, `_assemble_parts`' anchor
placement, and the `anchor=` parameters themselves) is left intact for one phase so that
inverting observe and retiring the anchor stay separately bisectable. The workflow that deletes it
owns that list, and should confirm the call graph rather than trusting this paragraph.
`AppStatus.opener_anchored` is vestigial for the same reason: still cleared with the rest of the
set, published True by nothing, rendered by nothing.

**One unplanned improvement, recorded because it moves a number the ranker depends on.** The
previous audit's finding 12 was train/serve skew: auto scored a ~48-frame read while observe —
the only mode that produces labels — read 12. Both modes now enumerate and both hand the ranker
`_ranker_frames_from_enumeration(photos, scroll_captures)`, so the two see the same count at the
same spacing over the same page. That closes the skew rather than papering over it, but it does
change observe's own frame set slightly (an even resample of a fine read, rather than the raw
frames of a coarse one), which is an input change to the model whose whole job is the owner's
taste. No re-measurement was performed and none is claimed.

Offline only — no device, no live API call, and nothing from `ops/calibration/` or
`data/hinge_debug/` was read, sent or described for this workflow. Every fixture is synthetic:
the painted cards and painted comment sheet `tests/test_hinge_sheet_verification.py` already
builds, and the synthetic scrollable world `tests/test_hinge_item_capture.py` already serves.

**Still owed after this**, and unchanged: §9's blockers 7 (nothing in Part B has touched a
device), 9 (no absolute accept ceiling in `verify_sheet_item`) and 10
(`_IDENTITY_MATCH_MAX_DIST` = 3.0 is measured too wide), the last two being calibration acts with
the owner and a device present and BOTH of them now load-bearing on the one flow where a real
message reaches a real person. Also unchanged: 5.8's pre-flight cross-check, blocker 3
(`navigate_to_item` has no `should_stop`), the residual `_locate_target_heart` carries on the
legacy capture-order path, and anchor removal. **NEW:** observe's per-profile wall clock and its
per-profile billed call are both real costs the owner is now paying and should re-check after a
session; and a residual on the identity half of the mismatch guard — all six real sheets read
`confirmed_not_top`, but a human who hearts item 1 without scrolling at all is not in that
sample, and if such a sheet is drawn over a scroll-top screen the guard answers "cannot tell" and
withholds the text. That is the safe direction and it is a measurement to take on the device, not
a bound to widen.

#### Addendum 2026-08-12: the hub reads the inverted flow, and a targeting stop stops reading as a quota problem

The hub half of 5.9 is closed, and with it the last item on the previous addendum's "the hub
changes (including a targeting-specific stop banner and `item_description` display)" line. Most of
the observe surface shipped WITH the inversion (the item heading, the isolated opener block, the
mismatch box, `opener_pending` beside a live GO cue), so this records what that left, what it got
wrong, and the two defects found by looking at the rendered output rather than at the code.

**The targeting stop has its own kind and its own banner.** 5.6's own addendum flagged this and
handed it here in as many words: "the hub's branch for it is titled 'opener capacity exhausted',
which is wrong for a targeting stop ... a dedicated kind plus its own banner belongs with the hub
workflow". `worker._auto_loop`'s `ItemTargetingError` handler now publishes
`stop_kind="targeting"` instead of `"opener"`, and `hub.html` renders it as
"stopped — could not like the item the opener was written about" with the reason (which leads with
INTENDED and ACTUAL) on the sub-line. Two reasons it is a KIND rather than better wording on the
old branch: the cause is not OpenerService at all — nothing was exhausted, the opener exists and
is fine, the driver could not reach the item it is about — and the operator's next move is
different, which is to walk over and read a phone the driver deliberately left exactly as it
stopped. Rendered `idle`, not the red error box, for the same reason the worker catches the
exception in the first place: a standing rule being obeyed must not look like a crash.

**Stated rather than smoothed over: `stop_kind="opener"` is still a family.** Besides genuine
capacity exhaustion it carries a per-profile opener failure, a capture that could not be
enumerated into numbered items, and an item number nothing can act on. The title is exact only for
the first and the truth is on the sub-line for the rest. Only targeting was split out, because it
is the only one of them whose operator ACTION differs; splitting the others means a kind each at
their own worker sites and buys nothing until one of them needs a different action.

**DEFECT ONE, FOUND BY RENDERING THE PRE-TAP WINDOW RATHER THAN BY READING THE CODE.** Until
5.9 a suggestion could only be published AFTER the heart tap (`waiting_for_send`), so it never
competed with the 🟢 GO cue that tells the operator it is their turn and that PASSING is one of
the two things they may do. It is published BEFORE the tap now, and the opener branch runs ahead
of the `waiting` branch, so the entire decision window rendered as "like item 3 — the ridgeline
photo, then type exactly this / <opener> / about: … · then tap Send Like in Hinge": no circle
(against the owner's GO/WAIT convention), no mention of the X, and an instruction about a sheet
that is not open yet — in the one mode whose entire output is the owner's own like/pass labels. A
hub that reads as an order to like biases the labels the ranker is trained on. Fixed in the chrome
row only, which keeps the canary rule intact: while the sheet is closed it now reads
"🟢 your call: click pass X or heart", and it becomes "then tap Send Like in Hinge" the moment the
sheet is open. The opener's own row is untouched and still contains `escHtml(opener)` and nothing
else.

**DEFECT TWO, THE SAME CLASS ONE BRANCH OVER: the mismatch box said what went wrong and nothing
about what to do.** This section's own wording is that the warning replaces the OPENER BLOCK, and
it is the only thing on screen while it is up, so a card whose suggestion failed rendered a red box
and stopped there. It compounds: `_ObserveSuggestion` publishes a warning per card whenever
suggestions are applicable but keep failing (a dead free-tier quota, a driver that cannot
enumerate), so a whole session could run without the operator being shown a 🟢 turn cue once. The
box gained a third row carrying the same two cues the suggestion box uses — "🟢 your call: click
pass X or heart" before the tap, "type your own opener, then tap Send Like" once the sheet is open
— and nothing else changed about it: still WAIT-coloured, still nothing to copy.

**The canary is pinned where the two rules meet.** `worker._ObserveSuggestion._display` returns at
the mismatch, so a warning and a suggestion can never be published together — which is exactly why
the hub's own precedence is now pinned rather than assumed (a stale poll or a future producer that
ANNOTATES instead of REPLACING would otherwise put a wrong-item opener back on screen beside a
caveat, and text beside a caveat is text that gets typed). With both fields set the banner offers
nothing copyable: no opener, no "type exactly this", not even the `about:` caption.

**3 net new tests**, all in `tests/test_hub.py`: the targeting banner against the capacity banner
it was split from, the warning-plus-suggestion precedence, and the pre-tap pass cue. One rewritten
in place (`test_a_targeting_miss_stops_the_run_instead_of_crashing_it` now asserts the new kind and
that it is NOT "opener"), and the existing mismatch test extended with defect two's next-action row
rather than a fourth test added beside it.

**On the suite count, stated rather than tidied:** the full suite is **1809 passed, 0 failed** with
this work in, against the 1776 baseline this workflow was handed. The difference is not 3 — another
workflow's uncommitted tests landed in the same working tree while this one ran, so a
before/after number is not attributable to either of us. Recorded as an absolute rather than as an
arrow, because a `1776 → 1779` here would have been a confidently wrong claim of exactly the kind
this file keeps having to correct. Offline only — no device, no live API call, nothing read from
`ops/calibration/` or `data/hinge_debug/`.

**Still owed after this**, and unchanged: §9's blockers 7 (nothing in Part B has touched a
device), 9 (no absolute accept ceiling in `verify_sheet_item`) and 10
(`_IDENTITY_MATCH_MAX_DIST` = 3.0 is measured too wide), 5.8's pre-flight cross-check, blocker 3
(`navigate_to_item` has no `should_stop`), anchor removal (the request-building machinery is still
intact and `AppStatus.opener_anchored` is still vestigial — it is cleared with the rest of the set,
published True by nothing and rendered by nothing), and the residual `_locate_target_heart` carries
on the legacy capture-order path.

#### Final addendum — 2026-08-12: completed redesign, calibration-gated release

This addendum supersedes the preceding historical “still owed” claims without editing them.
The redesign is complete in the production path. Navigation is **bottom-up**, using the
enumeration ledger from the entry anchor rather than rewinding and walking a second time; it
is cancel-aware throughout model-item navigation, so a requested stop cannot leave a new
scroll/tap action in flight. The capture-side model/item translation is preserved end to end.

The flow is deliberately asymmetric by mode. AUTO performs the pure, coarse 5.8 item-type
preflight before device work, confirms profile identity before navigation/tap, and verifies the
opened sheet's profile identity and selected item before it types. Sheet verification now has a
calibrated **absolute** acceptance ceiling as well as its closed-set separation check. A failure
at any of these targeting gates stops the targeted-like path: it never substitutes another item,
rewrites an opener for a different card, types, or sends. Rejected sends, including a paid-upsell
or paywall rejection, are logged as rejected rather than counted as sent.

OBSERVE is inverted: it enumerates first, shows the hub which numbered item a suggestion is
about, and lets the human choose pass or heart. Once the sheet is open, the same profile-identity
then item check decides whether text remains visible; a mismatch or an unreadable check withholds
the targeted opener while ordinary manual decision labels continue. The hub distinguishes a
targeting stop from opener capacity/quota exhaustion.

The obsolete `anchored_opener` repair callback and its anchored request-building machinery have
been removed. Remaining mentions of that design are historical prose, not a fallback path. This
release does **not** embed a numeric
calibration or manufacture one from offline material: the two acceptance ceilings must be
measured and recorded for the actual device before any targeted like or targeted OBSERVE text is
licensed. `ops/RUNBOOK.md` is the operational source for that final handoff.

#### Addendum — 2026-08-21: the run-level setup notice must never restyle the decision window

**DEFECT THREE, of the same class as defects one and two and found the same way — by an operator
actually sitting in the window.** With `targeting_calibration` intentionally absent
(config.yaml, 2026-08-21: withheld until positive still-photo proof exists), the driver's
run-level "no numbered item list" reason reached the hub as `opener_warning` on every card, and
renderSwipe routed every warning through the WAIT-styled `openerWarnBox` — amber, led by
"🔴 targeted opener setup required". In the owner's circle convention (and in every other box of
that function) amber+🔴 means "hands off", so the one mode whose entire output is the owner's own
pass/like labels spent the whole run telling the owner not to act in the exact window that was
waiting on them. The owner obeyed it for 2m24s, stopped the run, and filed it as a bug — twice in
one day; the first fix reworded the string but kept the box's WAIT identity, which is why it was
re-reported.

The fix draws the line the two earlier defects already gestured at: WAIT styling on a warning is
licensed by "there is something here you must not copy/do", and a run-level, intentional,
currently-unfixable condition licenses no such thing. In the decision window the banner is now
byte-identical to the plain waiting state's 🟢 GO box, with the setup situation demoted to the
box's fine-print sub row (which states the still-photo-proof precondition rather than promising
that a fresh calibration would enable suggestions today). With the sheet open it is a GO box
saying to type your own opener. Any other state falls through to that state's own cue instead of
being intercepted. The driver's internal consequence clauses ("…no numbered item list would have
a usable consumer" / "; no opener text is offered") stay in logs and bug reports but are stripped
from the operator surface, degrading harmlessly to the full sentence if the driver wording ever
drifts. Per-card mismatch warnings keep their deliberate WAIT treatment unchanged — there, the
thing not to do is real.

Pinned in `tests/test_hub.py::test_observe_banner_keeps_the_go_cue_when_hinge_targeting_calibration_is_absent`
(rewritten from the test that had pinned the amber shape): GO box style with a single 🟢 in both
decision states, jargon stripped, no-promise phrasing, capturing-state fall-through, the stopping
intercept still outranking the GO cue, and escaping of the displayed reason.

#### Addendum — 2026-08-22: the setup notice must name the step that is actually left

**DEFECT FOUR, in the very sentence defect three added, and reported the same way — by an
operator who read the box and concluded there was nothing they could do.** Defect three's fix
demoted the missing-calibration condition to fine print and, correctly for that day, made the
sub row state the still-photo-proof precondition rather than promising that a fresh calibration
would enable suggestions. Both the hub (`hub.html`) and the worker's startup notice
(`worker.py::_announce_observe_targeting_setup`) hardcoded that sentence.

Hours later the owner accepted the centered-autoplay assumption
(ops/STILL-PHOTO-DISCRIMINATOR.md section 5b) and `config.validate()` began installing a
still-photo licence from `apps.hinge.still_photo_assumption_acceptance`. From that moment
`hinge_targeting_unavailable_reason()` returned `None`, numbering was licensed, and the only
remaining gate on targeted suggestions was the absent `apps.hinge.targeting_calibration` — a
step the owner could have taken that same evening. But both surfaces went on stating a
prerequisite the owner had already satisfied, and neither ever named the calibration campaign.
The run log said `targeted suggestions enabled under an UNMEASURED assumption` two lines above
a banner that said still-photo proof must come first. The owner read the fine print as an
upstream block, filed "still has this error. not giving openers", and the calibration stayed
uncaptured.

The lesson is narrower than defect three's and worth stating separately: **operator guidance
that encodes a precondition must be derived from that precondition, not written down beside
it.** A hardcoded next step is correct exactly until the state it describes changes, and then
it is worse than silence — it actively routes the operator away from the fix. Wording review
cannot catch this, because the sentence was true when it was written and no diff touched it
when it became false.

The fix: `targeting_policy.targeting_setup_next_step()` branches on the installed licence and
returns the one action that would restore suggestions —
`TARGETING_SETUP_NEXT_STEP_BLOCKED` (still-photo proof first) while nothing licenses numbering,
`TARGETING_SETUP_NEXT_STEP_CALIBRATE` (capture, measure, paste the block) once something does.
The worker prints it and publishes it as `RunStatus.targeting_setup_next_step`, run-level and
outside `_OPENER_FIELDS`' clearing net exactly like `targeting_licence_notice`. The hub renders
what it is given: it cannot work the branch out for itself, because the licence lives in the
process that ran config validation. With nothing published the note ends after "manual pass/like
labels still work" rather than asserting a precondition the page cannot verify — silence cannot
go stale.

Everything defect three fixed is unchanged and still pinned: a single 🟢 GO box in both decision
states, jargon stripped, capturing-state fall-through, the stopping intercept outranking the GO
cue, escaping of the displayed reason. Which next step is named never restyles the window.

Pinned in `tests/test_hub.py::test_observe_banner_keeps_the_go_cue_when_hinge_targeting_calibration_is_absent`
(both branches plus the unpublished case) and in `tests/test_worker.py`'s
`test_the_targeting_setup_notice_names_the_calibration_once_numbering_is_licensed`,
`…_names_the_still_photo_proof_while_nothing_licenses_numbering`, and
`test_a_calibrated_run_publishes_no_targeting_setup_step`.

#### Addendum — 2026-08-26: stale live calibration is a typed pre-opener stop

The installed 10.0.1 schema-v3 calibration became historical evidence when the live Hinge build
advanced to 10.1.0. Exact app-version/frame binding remains fail-closed: equal frame dimensions
do not prove that composer, scroll-top, or item geometry is unchanged. The driver therefore marks
the capture `Profile.items_unavailable_kind` as `targeting_calibration` when calibration is absent
or rejected at runtime.

The worker publishes `stop_kind=targeting_calibration` before any opener request, gesture, or
label. The Hub renders **"targeting calibration must be renewed"** and retains the precise
runtime diagnostic. Its next step is to capture and validate schema-v3 calibration for the live
build/frame; that 10.1.0 renewal is now installed and enables Training. AUTO still requires fresh
release evidence for the renewed calibration. This is not the `targeting` state: `targeting` is
reserved for the later case where an opener already exists but cannot be attached to its selected
item. The generic `opener` state now means **"could not prepare a safe opener"** rather than
incorrectly implying capacity exhaustion.

#### Addendum — 2026-08-27: "the signature has one entry point" was not enough, and a correct like sheet was refused

The claim in 5.6 above — that a verification pass is safe once it calls `signature_of` — is
**wrong, and it cost a live training run**. One entry point guarantees one *decoder*. It does not
guarantee one *grey space*, because the two sides of the comparison were never decoding the same
bytes.

`item_crops._crop_image` stores the model's image as a colour re-encode of the frame. Verification
then had to window that image at many heights for the inline reframe sweep, which the single fixed
stored signature cannot answer, so `item_verify._decode_crop` grey-decoded `crop.image` — and
argued in its own docstring that this was safe because it was byte for byte the call
`signature_of` makes. It was the same call on a different PNG.

**The operative difference is one ancillary chunk.** An Android screencap carries an `sRGB` chunk;
`cv2.imencode` writes none. OpenCV's `IMREAD_GRAYSCALE` is sRGB-aware, so identical RGB samples
grey differently depending on whether the chunk is there. Re-inserting it into the re-encode makes
the two decodes **bit-identical** again, which is what identifies the chunk rather than the alpha
channel or the choice of conversion formula as the cause.

Measured on the live Pixel 7a (Hinge 10.1.0, profile "Tina", heart 9): the two paths differ by
4.35 grey levels on average and 64 at the worst pixel. On that card — a dark, heavily saturated
neon photo, which is where the two greys diverge most — the gap was 10.158 at the 64x64
verification grid. The correct sheet therefore measured **10.283 against the 7.00 one-item
ceiling** and the run halted with the sheet open and the opener untyped. Against the same live
frame after the fix, the correct card measures **0.167** and the reciprocal wrong card still
measures **73.965**, so nothing about wrong-card rejection was traded away.

Note what this does NOT say: no calibrated bound moved. `_INLINE_COMPOSER_ONE_ITEM_MAX_DIST`,
`_SHEET_RENDER_DRIFT` and the configured `inline_item_max_dist` are all unchanged. The distances
they were fitted against were inflated by this bias, so removing it makes every existing ceiling
more conservative, never looser. They are now loose rather than tight, and re-fitting them is a
separate measurement on a corpus of real device frames — not something to do while fixing this.

What ships:

* `ItemCrop.verify_image` — the same rows as `image`, as a lossless 8-bit greyscale PNG cut from
  the frame's own `IMREAD_GRAYSCALE` decode. Colour type 0, so it decodes back bit for bit with no
  colour conversion left in the path to disagree about. `image` is untouched and still what the
  model is sent.
* `item_verify._check_reference_provenance` — the reference must reduce back to the signature
  `item_crops` stored for the same rows at the same grid. Expected value 0.000. **Fails closed**,
  because a wrongly-scaled distance is indistinguishable from an honest one by inspection, which
  is exactly why it has to be asserted rather than trusted.
* A verification refusal now records `reason` (which bound regime fired), the per-item comparison
  table, the preview's derivation and the composer rects, and the bug report renders them. The
  original report printed "10.283 against 7.000" and nothing else, which is indistinguishable from
  a genuine wrong-item tap.

**The lesson for future fixtures, and it is the uncomfortable one: no synthetic frame in this repo
can reproduce this fault.** Every painted PNG here is written by `cv2.imencode`, so neither side
carries an `sRGB` chunk and both paths agree. The full suite was green throughout — 4147 tests —
while a live run refused a correct card. The regression test injects the chunk by hand. When a
live run disagrees with a green corpus, suspect the device's own encoding before the geometry.

#### Addendum — 2026-08-28: three defects in the bidirectional preview walk, found by auditing the fix above

The 2026-08-27 grey-space fix shipped alongside an in-progress change that made
`_extend_inline_preview_to_block_edges` walk the preview's edges in BOTH directions rather than
only downward. Auditing that change found three defects. All three are fixed; two are
mutation-tested, and the third is a docstring that was actively misleading.

**1. A foreign card with one pale edge could be welded onto the preview.** `row_support` required
a row to be at least `minimum_width` wide and aligned on ONE edge, and never bounded the opposite
edge. An ordinary profile card at x=53..1027 is correctly rejected — both alignment tests miss by
42px against 27px of slack — but a card whose leading columns are pale enough to read as page
background reports a span that STARTS on the preview's own x0 and still runs 42px past its x1, and
that row passed every test. Measured on the real 00097 composer frame with a pale-left card
painted at rows 110..232: a 236..1092 preview welded to **110..1092**, turning `verify_match` into
`verify_mismatch`. Both polarities reproduce. Fixed by requiring column containment
(`span[0] >= preview.x0 - slack and span[1] <= preview.x1 + slack`) in both support branches;
`slack` already existed, no new constant. A genuine row of THIS photo cannot extend past the
photo's own columns, so containment costs nothing legitimate.

This one is worth dwelling on because the obvious test does not catch it: the repo's existing
"reviewer scrolled a card above" fixture paints its upper card OPAQUE across the full card width,
which the alignment tests reject — so it never reached the upward walk at all and proved nothing
about it. The regression test paints the pale-edge variant, in both polarities.

**2. The bridged-seam budget was summed across both directions and tested against the
single-direction bound.** One legal 3-row seam below plus a single bridged row above pushed
`bridged_runs` to 2, discarded the WHOLE extension, and produced exactly the false-refusal class
this walk exists to prevent. Fixed by judging each edge against the budget independently and
discarding only the offending edge, which restores HEAD's downward behaviour byte for byte and
holds the new upward direction to the identical rule. **This is not a widening of the seam**, and
the distinction matters: what stops the walk reaching a different block is
`_INLINE_PREVIEW_MAX_INTERNAL_GAP_PX` as a cap on CONSECUTIVE unsupported rows (unchanged, so a
four-row gap is still a hard boundary in both directions) plus the containment from (1). Two
three-row seams do not add up to permission to cross a four-row one.

**3. A false sentence in the docstring, deleted rather than reworded.** "A no-op after
`_locate_inline_composer_preview`, which already scans at this same width floor and therefore
already ends at a background row." Both halves are false: that locator scans at 0.78 while the
walk continues at the lower one-edge 0.74 floor, so on this module's own Lauren fixture the
locator returns 713..1092 and the walk carries the top edge 477 rows up to 236, terminating on a
row that still holds a 661px span. At HEAD the sentence named `_locate_inline_compact_preview` — a
function that does not exist — which at least made it obviously stale; the rename swapped in the
real name and left the false substance, laundering a dead sentence into a believable one. These
docstrings are the calibration record, so a claim that cannot be reproduced is removed.

**Also fixed, and it is the reason this took as long as it did:** the driver's own stop message
printed "intended model item 1, actual 1", which reads as nonsense. Two different faults wear that
exception — `nearest != intended` is a real targeting miss, `nearest == intended` is the right
card refused by its bound, i.e. a measurement/calibration problem and a candidate FALSE REFUSAL.
They now print as different sentences.

#### Addendum — 2026-08-28: the one-item ceiling is re-fitted 7.00 → 3.00, and this one IS a threshold change

Stated plainly, because the previous two addenda were careful to say that no calibrated bound
moved: **this one moves a bound.** It is not disguised as a bug fix.

The reason it has to move is that `_INLINE_COMPOSER_ONE_ITEM_MAX_DIST = 7.00` was fitted in units
that no longer exist. Its justification was two held-out inline renders at 3.729 (Shai) and 6.703
(Malaika) — and **both of those figures were decode bias, not render penalty.** They were measured
while the crop reference was recovered from a colour re-encode that had lost the device PNG's
`sRGB` chunk, i.e. through the exact defect the 2026-08-27 addendum above describes. The frames
behind them are not preserved, so that is inference rather than re-measurement; the replacement is
not.

**Measured 2026-08-28**, over every recoverable like-sheet verification in `data/hinge_debug` —
148 sheets across 17 runs, each re-verified through the real
`verify_sheet_item` / `_compare_item` / `locate_inline_composer` path, and validated by
reproducing production's own logged 10.283:

| population | n | min | median | p90 | max |
|---|---|---|---|---|---|
| correct card, corrected space | 148 | 0.068 | 0.249 | 0.963 | **1.380** |
| correct card, defective space | 148 | 0.733 | 1.969 | — | 10.283 |
| foreign card, cross-profile | 218 | **40.823** | 70.535 | — | 111.698 |

Two things fall out of that table beyond the ceiling itself. First, **9 of 148 correct sheets
(6.1%) measured at or above 7.00 in the defective space** — the 2026-08-27 halt was one of nine,
not a one-off, and the grey-space fix retires all nine. Second, a 219th cross pair scored 0.174;
it was hand-checked against the images and is the **same profile captured in two different runs**,
so it is a correct match and is excluded from the foreign set rather than kept as a flattering
minimum. It is also independent evidence that the check works across sessions.

**3.00** sits 2.17× above the worst of 148 real correct readings and 13.6× below the nearest real
foreign card, and refuses none of the 148. The point of moving it is NOT reject power against a
different card — 7.00 already had 5.8× of that against a population whose minimum is 40.8. It is
that in the one-item regime this ceiling is also the **only test of whether the reading is
trustworthy at all**, there being no separation proof to fall back on. At roughly 0.7–0.9 grey
levels per pixel of preview-rect error, 7.00 accepts a preview mislocated by about 8px; 3.00
accepts about 2px, against the ±1px actually observed.

**Why this cannot create a false accept:** the constant feeds exactly one expression,
`if mine.distance >= mine.bound`, so a smaller value can only convert accepts into refusals.

**The risk the owner is accepting, and it is a real one:** availability. A sheet that would have
squeaked through at 7.00 now halts with the composer open and nothing typed. That is the fail-loud
direction the owner rules prefer, and every measured reading clears 3.00 with room — but it is a
live-behaviour change and one constant reverts it.

**Deliberately NOT changed.** `_SHEET_FALSE_MATCH_DISTANCE = 14.91` stays: it is a *measured
wrong-card acceptance* used only as an upper validity clamp on a configured ceiling, and the
corrected foreign population (min 40.8) makes it conservative rather than wrong. The config's own
`apps.hinge.targeting_calibration.inline_item_max_dist` (14.9099 = `nextafter(14.91, 0)`) is that
same clamp and stays too. The module docstring's older corpus separations (4.670 / 8.024 / 8.727)
were also measured through the defect; the direction is safe — correct distances collapsed and
foreign distances grew — so nothing there is now looser than advertised, but the figures are stale
and should be re-derived the next time that section is touched.

#### Addendum — 2026-08-28: a portrait photo is PILLARBOXED in its card, and the locator was measuring the photograph where it needed the card

Two production halts, 2026-08-27 and 2026-08-28, both `verify_sheet_item` → `unreadable`, both the
same sentence: `no candidate at least 694px wide and 300px tall is aligned with it`. Both taps were
CORRECT. The frames show an ordinary open composer with the right photo in it.

`apps.hinge.debug_protect_runs` now protects `f6a15d162e68` and `d605b0837ac2`; they are the only
frames in the corpus that carry this shape and they cannot be regenerated.

**What is actually on the screen.** The card is a PROMPT-PHOTO card whose photograph is portrait, so
Hinge fits it to the card's height and centres it: a 556x932 photo inside a 974px card, 209px of
card background on each side. The sheet renders that whole card into the 890px content column at
the corpus's own 890/974 = 0.9137 scale, so the photograph's INK spans 508px at columns 286..794 —
191px of white card either side, symmetric to the pixel. Against a 694px width floor and 27px of
edge slack, all three of the locator's tests fail, and every one of them is right about the ink and
none of them is about the card.

**The invariant that was never true.** The six-sheet corpus behind this module is six FULL-BLEED
photo cards, where the photograph happens to run the card edge to edge, so preview-ink and content
column coincide. "The preview spans the content column" was read off that and is a property of the
CARD'S CONTENT, not of the composer. The card is always rendered into the whole column; only its
ink may be inset. Swept 2026-08-28 over 98 runs / 297 verify sheets: 287 full-bleed with ink width
EXACTLY 890 and pads of exactly 0, 4 pillarboxed (one photograph, met twice), nothing in between —
a 0.43-wide empty gap in the width fraction. The underlying card shape is 1 of ~1033 distinct cards
seen, 0.10%.

**Widening the width test alone would NOT have fixed it, and this is the part worth remembering.**
`_compare_item` derives `scale = crop_width / preview.width` and the stored crop is the WHOLE card,
so handing it the 508px ink box asks a 1109px card for a window of round(837 * 974/508) = 1604 rows
and it refuses with "this item cannot be what is on screen" — the same halt one stage later, with a
message that blames the card instead of the locator. Measured: ink box → distance None, window_px
1604; content column → **0.579** with the reframe search landing on source rows 185..1088 of 1109,
against production's 14.9099 ceiling, while the same card mirrored scores 21.563.

So `SheetPreview` now carries the RECT and the INK BOX as separate things. The rect is the card's
columns (the comment field's, which is what the crop is scaled against); the ink box is where the
photograph painted, and only the block walk uses it.

**What replaces the two edge-alignment tests, and what is honestly carrying the load.** A pillarbox
candidate must be contained inside the content column and centred in it. Containment refuses an
ordinary full-bleed profile card exactly as decisively as alignment did (53..1027 spills 42px past
both ends). Centring does NOT refuse it — a card and the column share the centre 540 — and neither
test refuses the one impostor that matters: a PILLARBOXED card scrolled above the composer is
contained and centred to the pixel. **The gap bound is what excludes it**, and structurally: a
candidate needs `comment.y0 - y1 <= 0.25 * (y1 - y0)`, i.e. `y1 >= 921`, which an impostor can only
reach if the selected preview renders under ~188px tall — at which point the preview fails the
300-row floor and nothing is located at all. Applying these tests to all 298 composer frames in the
corpus yields exactly one accepted candidate per frame, always the selected preview. No frame has
both a pillarboxed preview and a pillarboxed block above it, so the case is constructed in
`test_a_pillarboxed_card_scrolled_above_the_composer_does_not_displace_the_preview` rather than
left to a live run.

**A safety consequence that had to be corrected, not merely noted.** A pillarboxed comparison rect
is ~30% blank on BOTH sides — card-white in the sheet and card-white in every candidate crop — so
those cells contribute nothing and every distance through it contracts. The relative bound does not
care (`0.5 x nearest_other` is a ratio of two quantities that contract together, which is exactly
why a multi-item run looks safe and hides this). The two ABSOLUTE ceilings do care: they are frozen
numbers fitted on full-bleed renders, and `inline_item_max_dist` is clamped to 0.0001 below the
known 14.91 foreign-card collision ON PURPOSE. Making this regime reachable without correction
would have widened both by the contraction factor — quietly loosening the one guard that catches
content the payload does not contain at all. Both are now scaled by the fraction of the compared
rect that carries ink, which can only ever LOWER a ceiling and therefore cannot create a false
accept.

The contraction is content-dependent and is not one number: comparing over the photograph's columns
alone rather than the content column gives 0.749 for the correct card, 0.867 for a mirrored foreign
one, and 0.681 for a separately composited foreign card. The geometric 0.696 sits at or below all
three, which is the direction that matters. This is a correction to a margin that was already
adequate rather than a rescue: 40 foreign photographs composited into this exact card template
score min 31.32 / median 58.98 through the pillarbox path, none under the uncorrected 14.9099, so
the correction moves 2.1x of headroom to 3.0x. The exact alternative — comparing over the photo's
columns on both sides, which needs no estimate — is the better fix and is deliberately NOT taken:
it re-cuts every item's reference and so changes what `nearest_other` means, and there is exactly
ONE pillarboxed render in the corpus to validate that against.

**Three residuals, recorded rather than glossed.**

1. `_INLINE_PILLARBOX_MIN_WIDTH_FRACTION = 0.40` IS NOT A MEASURED FLOOR. n = 1 render at 0.5708,
   and it sits near the SHALLOW end of the regime: full-bleed previews run 0.83..1.15 in displayed
   aspect and this one is 1.65, so Hinge starts pillarboxing around 1.15 and then caps the rendered
   height, under which the fraction falls without limit (9:16 lands near 0.54, 1:2 near 0.48). A
   taller upload than 0.40 admits will still REFUSE, which is fail-loud and one constant to revisit
   with a frame in hand.
2. THE SHEET IS NOT A RIGID RESCALE FOR A CHROMED CARD. Measured at full resolution: the
   width-derived `window_px` is 916 rows and the true best-fitting window is 905 at start 183 —
   1.23% of vertical anisotropy, 11 rows of error. The full-bleed control fits in 804 rows with
   0.059% and ZERO rows of error. `_SCALE_TOLERANCE`'s ±2% absorbs it, but that constant was fitted
   as "1.0% observed, 2% is that with a factor of two on it", and the factor of two is now 1.65 on
   n = 1. If the anisotropy grows past 2% the symptom is a false VERIFY_MISMATCH on a correct tap:
   fail-loud, but it reads as a targeting fault rather than as a scale one.
3. The centring test MUST use the median span and never the hull. Hinge draws a dark alt-text chip
   on the card's white margin at the top left, so 54 of the block's 837 rows report ink starting at
   x=133..156 while 783 report exactly 286; the hull gives a 153px asymmetry and would refuse the
   very frame this regime was written for.

**Reporting.** The refusal now surveys every non-background block above the field at NO width floor
and prints each one's geometry with the tests it passed AND failed — because on this halt no run
reached the 694px floor at all, so a message written in terms of rejected CANDIDATES would have
described nothing, and the diagnosis was entirely in ink that never became one. The passes matter
as much as the failures: "fails width and both edges, PASSES height and field gap, inset by the
same 191px on both sides" IS the diagnosis. `bugreport.py` also no longer renders an `unreadable`
as a comparison that returned nothing — it used to print `nearest stored item None; distance —
against bound —`, three dashes standing where three numbers stand on every other verdict, when in
fact no card had been compared at all.

#### Addendum 2026-08-28: Hinge 10.1.0 PINS the profile header inside the analysed band, which is amendment four to 5.4 and quietly killed three separate guards

A training capture refused its whole item index with seven fold failures, the first of which read
"frame 16 sees page rows 8005..9234 ... where the block was bounded at 7387..8072 by frame 14 — a
fragment cannot reach past the card that contains it". The scroll chain was never in doubt: frame
15's heart at frame row 2035 maps to frame 16 row 1508 at exactly the measured pair delta of 527.
The fault was in what one frame reported, not in how the frames were joined.

**The measurement.** On the six saved evidence frames of run `8fb11094ef4d`, frame rows 300..516
are byte-identical — maximum absolute difference **0**, across all 15 pairs — while rows 517 and
below differ by a mean of 59..143 grey levels. Those frames sit at page offsets 6180..8485, i.e.
thousands of pixels apart. Hinge 10.1.0 draws the profile header (filter chips, name, verified
badge, back arrow, overflow menu, and a `she | Active today` sub-row) **pinned to the screen**,
inside `content_band`'s rows 300..2100, and clips the scrolling content to begin at row 517.

**Amendment four to 5.4: a page-background run can be neither a gutter nor a card edge, and the
existing tests are structurally blind to it.** The run between the pinned header and the content
below it measures 106px — twice the canonical gutter — so the length gate correctly declines it,
and no card corner sits below it mid-scroll, so amendment one's corner escape declines too. The
run is therefore absorbed, and the 43px header strip merges into the clipped top of the next card.
Note that the run passes every *affirmative* test amendment three added: its median grey level
matches the page reference by exactly 0 on all 106 rows. Level agreement was never the missing
signal here. The missing signal is that the strip above it does not move.

**Why the merge is not merely untidy.** The merged block's top is reported at frame row 368 on
every frame, so its PAGE row is `368 + offset` — a different row on every frame, and every one of
them fabricated, because a screen-pinned element has no page position at all. Whenever the card
above happens to have scrolled into the header's dead zone, that fabricated top lands inside it,
`_overlap_groups` chains two cards into one fold group, and `_resolve_group` refuses. With a 43px
strip against a 53px gutter on a ~1027px card pitch, a swept simulation over this profile's own
layout puts the per-frame hazard at ~9% — about an 84% chance of at least one bad frame per
19-frame profile. This capture was not unlucky; it was typical of the expanded-header state.

**The fix is to name the ambiguity, not to resolve it in the wrong place.** `segment.py` now
reports such a strip as `BLOCK_UNANCHORED` with both edges UNOBSERVED and declines to say what it
is, because chrome-ness is a CROSS-FRAME property: a strip with page background on both sides is
either app chrome pinned to the screen or a slice of page content the band happened to cut that
way, and no single frame distinguishes them. `item_index` then decides from the capture's own
evidence — a strip at the same frame rows, with the same pixel digest, at two or more page offsets
spanning more than its own height, is fixed to the SCREEN and is held out of page space entirely.
Both unproven branches are bit-exact the pre-fix behaviour, verified field-for-field against the
recorded geometry of this very incident, so an unavailable proof degrades to the old loud refusal
and never to a new outcome. Held out, the incident capture folds with **zero** failures.

Rejected alternatives, both on measurement rather than taste. Narrowing `content_band` below the
header fails three ways: `plan_coverage_step`'s blind fallback becomes a hard refusal at any band
top above row **414**; a band top of exactly 517 makes `_scroll_top_evidence` return False on every
genuine scroll top, refusing every capture; and the prefix is **not a constant** — the same run,
same build, same phone shows a *collapsed* header state clipping at 236, so a static 517 would
discard 281 rows of real content on those frames. Splitting on any full-width page-background run
was measured too: it avoids this refusal but fabricates 18 page-space fragments of a fixed screen
element, and lands one in a real gutter about 1% of the time, where it trips the fabricated-item
guard.

**THE THREE GUARDS THIS SILENTLY KILLED, all of which had been reporting success.**

1. `item_index._scroll_top_evidence` — the check credited in its own docstring with refusing 21 of
   21 and 9 of 10 falsely-asserted scroll tops in the corpus. It asks whether page background sat
   above the topmost block. The pinned header puts background there on EVERY frame, so it returned
   True unconditionally, top or not. It now additionally requires a POSITIVE discriminator: the
   first real item must have been seen with its own `EDGE_CARD_CORNER` top. Frames 0 and 6 of the
   incident carry that corner; the six deep mid-scroll frames provably do not.
2. `scroll_top.confirm_scroll_top` — doc 5.5's affirmative filter-chips gate. Its returned reason
   asserts "at any scroll offset past the top the app's sticky per-profile header covers this strip
   with the person's name instead". In the expanded state the chips row is itself pinned at rows
   115..340, so all six deep mid-scroll frames return `confirmed_top` at distance **0.000**. In the
   collapsed state the same band still reads correctly (13.219 and 12.359 on two other profiles in
   the same run), so this is per-screen-state, not blanket.
3. `frameshift`'s own stated hazard, re-introduced. `estimate_shift`'s docstring warns that a strip
   cut across the status bar or a sticky header "would correlate best at a shift of zero no matter
   what the content underneath did, and would vote against every real scroll" — which is why the
   band excludes chrome. The pinned header put exactly such a strip back inside it: strip [300,396]
   reports `pinned`, delta 0, score 1.0 on all 18 pairs, and [442,538] frequently votes a
   structurally meaningless 0 into the median.

Guard 1 was fixed here. Guard 2 is per-screen-state and needs `identity_band` recalibrated against
BOTH header states on the device; until then it fails closed through `capture_profile_identity`,
which already returns no fingerprint and blocks navigation. Guard 3 is currently benign — the
deltas are unaffected and dissent actually falls — but it is the same class of fault and should be
re-checked if the header ever grows.

**Open, and not answerable from this run.** What SELECTS the expanded (clip 517) versus collapsed
(clip 236) header. Both appear in run `8fb11094ef4d`, on the same build and phone. The enumeration
frames of the two profiles that succeeded that day are not saved, so it cannot be settled here.
The gutter-phase arithmetic above says an expanded-header profile should hit a merge overrun on
roughly 10% of frames, which argues the two that succeeded enumerated collapsed rather than got
lucky. Worth finding, because it decides how often this was ever going to fire.

#### Addendum 2026-08-28 (b): the pinned header measured LIVE — one collapsing toolbar, and the fix verified against the capture that broke

Follow-up to the addendum above, run on the connected Pixel 7a on the same Hinge 10.1.0 build,
read-only plus humanized read-scrolls, 199 logged actions and zero like/pass/send/advance/tap.

**The two "header states" are one collapsing toolbar.** Expanded only at a confirmed scroll top:
chips + large name + pronoun row, content clipped at row **517**, the 43px strip at **368..410**.
It collapses on the first downward scroll to a compact bar sitting entirely above the band's first
row, after which content starts at 300. The 517-vs-236 clip lines the previous addendum recorded
are the same toolbar at two scroll positions, not two builds or two surfaces.

**The previous addendum's guess about the two successful captures is WRONG and is corrected here.**
It reasoned that Charlotte and Alana must have "enumerated collapsed" because an expanded read
should hit a merge overrun on ~10% of frames. Their manifests in fact record chrome at page rows
`368..410` — the EXPANDED signature. The whole 2026-08-28 run was in the stuck-expanded state; the
two that succeeded got lucky on card phase, exactly as the ~9%/frame hazard predicts for some
profiles and not others.

**Verified end to end.** `_capture_current()` on the incident's own profile (debug run
`run_20260828_210258`) reproduced its page rows exactly — chrome `[368,410]`, cards at
`7387..8072` and `8125..9234`, the pair that bridged — and returned `usable=True, failures=0,
blocks=11, selectable=9, translation=(1..9)`, fingerprint present, 3 numbered items. Both
dispositions are now confirmed on real pixels: unproven-at-one-offset (collapsed) places the strip
and relabels it `ITEM_LEADING_CHROME` exactly as before; proven-at-six-offsets (the incident's
saved frames) holds it out and turns 7 failures into 0.

**`identity_band` needs no recalibration** — measured 0.000 at the top and 12.547..12.797 on every
scrolled frame, both well outside the 3..9 dead zone, with a present fingerprint. No calibrated
value was changed. See ops/ANTI-BOT-RESEARCH.md's 2026-08-28 (b) addendum for the numbers.

**Still open:** what makes the toolbar STICK. Not reproduced today in ~60 scrolls across five
patterns on the same build, phone and profile. Treat a capture whose leading strip is *proven*
screen-fixed as the signal that it has recurred — that verdict is now printed in the bug report.

#### Addendum 2026-08-28 (c): the coverage margin was defending the blind branch against the one card that cannot be in front of it

Closing the residual the 2026-08-28 (b) addendum left open. Two numbers were supposed to be
re-measured together on the phone; measuring them changed the question.

**`_MAX_CARD_HEIGHT_PX = 1467` was wrong, and in the unsafe direction.** Its own comment admitted
it was "not independently reproduced". Re-measured over every `item_manifest` in the on-disk
archive — 522 blocks that were CROPPED FROM ONE FRAME, so both edges were observed inside a single
band, across 50 captures in 19 runs: min 215, median 974, p95 1109, **max 1609**. So the constant
understated the tallest observed block by 142px. The 1609 is genuine: `kind=context` (heartless),
run `948352f2d4d0`, frame 4 rows 391..2000, 91px of top clearance; its manifest neighbours are
separated by exact 53px gutters throughout, no pair of archived block heights plus a gutter sums
to it, and the run's independently logged gesture ledger puts frame 4 at page offset 1784 against
the manifest's implied 1785. Its raw frame is NOT on disk — enumeration frames survive only when
an index is refused — so it can never be re-segmented.

**Read 1609 as a FLOOR, never a bound.** A block taller than the band can never be observed
complete at all, so it falls outside this population by construction and the sample is
survivorship-biased. That is exactly why the adaptive throttle's proof rests on the BAND HEIGHT
instead, and why it needed no card-height figure to be sound.

**The real defect was the formula, not the number.** `coverage_margin_px = band_height -
max_card_height_px` is reached only when a frame yields NO blocks. For a card to occupy the
analysed band and contribute zero card rows, BOTH its edges must lie outside the band — its height
must exceed the band — for which `band_height - H` is negative and the derivation's own premise
has already failed. A card genuinely 1467px tall in an 1800px band produces rows, gets a block,
and routes to the adaptive branch. The margin was guarding this branch against the one card that
cannot be in front of it: the guarantee was anti-correlated with its own trigger. [Measured over
all 3696 archived frames re-segmented with the production segmenter: exactly 3 yield zero blocks,
0.08%, all three hub `observe_*` screens that never reach this function, with band mean grey 254.0
and per-row standard deviation 0.0 — a blank undrawn screen, not a tall card. Zero enumeration
frames yield zero blocks; zero frames anywhere report a segmentation failure.]

**Shipped:** `_MAX_CARD_HEIGHT_PX` and the card-height margin are deleted. The blind branches are
now capped by the DIRECT-BRIDGE BUDGET — what frameshift's trust window has left once the
enumeration ceiling is spent, `(0.5 - 0.30) * band_height` = 360px on the calibrated device
against the old 333 — because the branch's only real job is to keep `build_item_index`'s
frame-omission recovery able to drop the unusable frame and bridge i-1 to i+1 across it. The
frameshift fraction is imported, not restated. The cap must not become the bare trust ceiling:
540 + 540 = 1080 exceeds the 900px window and would make that recovery unreachable, which is now
pinned by a test.

**REJECTED, on measurement:** computing the margin against the honest scrolling viewport, which
the (b) addendum floated. `_scrolling_viewport_top` returns the band top on exactly the zero-block
state the blind branch triggers on, so it is a no-op there; any threaded-in value is a guess
across at least 15 distinct header states observed in 275 archived frames; and the
`coverage_margin_px < 1` raise runs BEFORE branch selection, so on read-path pinned-header frames
it would have raised on 19 of 43 at 1467 and 43 of 43 at 1609 — killing ordinary throttled steps,
not just blind ones.

**AND A CORRECTION TO WHAT A REFUSAL COSTS.** The (b) work assumed a `ScrollStepError` here merely
ends enumeration and lets the read finish. It does not: it sets `enumeration_reason`, which
reaches `_invalidate_item_index` and then `Profile.items_unavailable`, and in Training
(`worker.py:486-493`) and in AUTO with openers enabled (`worker.py:970-987`) the worker does
`stop_event.set(); break` unconditionally. **The run stops.** Every proposal that adds a refusal
to this path is adding a run stop, and must be priced that way.

**Still open, now with numbers:** on read-path pinned-header frames the honest scrolling viewport
measured 1420 / 1518 / 1583, and 1609 > 1420, so the throttle's completeness proof does not cover
the tallest observed block in the pinned state. Its failure mode there is fail-closed and loud (an
unbounded heartless block above a heart appends `_UNCERTAIN_HEART_NOTE`, `usable` goes False, the
capture refuses), so it costs one profile and can never produce a wrong like. Separately,
`item_crops._over_tall_failures` compares block height to the BAND, not to the viewport, so a
1609px block under a 1420px viewport passes it silently and simply never completes. Both belong in
the same re-check, not in a step gate.

#### Addendum — 2026-09-05: an opener that broke no rule still read as a bot, because every rule governed content and none governed register

The trigger was owner feedback on the newest opener on record (2026-09-04T19:16:01, run
122495fa2f81, `auto_opener_pre_send` — never sent):

> You have a really classic sense of style here. What is your favorite kind of spot when
> you feel like dressing up for a night out?

It passes every Part A rule — two sentences, one compliment, an easy positive question,
profile-specific, nothing invented — and the owner correctly read it as glazing and wordy.
Their rewrite: "That's quite a classic style you got there. What's your favorite spot for a
night out?" Three register properties separate the two: an earnest second-person appraisal
vs. an offhand remark about the thing; written-formal grammar vs. spoken texting register;
restated context and deictic filler ("here") vs. trusted implicature.

MEASURED (offline, read-only: BigQuery `openers` 125 rows + `opener_send_evidence` 64 rows +
data/hinge_debug actions.jsonl 96 unique, deduped, 40 most recent analyzed, 2026-09-02 →
2026-09-04):
- ZERO apostrophes in all 188 unique openers ever generated. Not one contraction. The
  register failure is not occasional; it is the entire distribution.
- 21/40 open with a demonstrative verdict (That/This/Those/These + appraisal).
- 22/40 end in a binary "X or Y" question; 12/40 voice the observation as "looks like";
  7/40 stack two or more intensifiers; 30/40 fit one compound mold: [demonstrative] +
  [appraisal or looks-like hedge] + [caption noun] . [binary or-question].
- Median 23 words. The five openers that read naturally are the short ones that drop the
  appraisal beat entirely.
- A side finding: the no-hyphen HARD RULE was being satisfied by deleting hyphens from
  compounds the model still wrote ("top secret survival gear", "a pose off"), i.e. the rule
  changed spelling where it was meant to change phrasing.

WHAT SHIPS (both prompt copies, doc 3.1 lockstep; plus the two copies no prior entry
recorded — see below):
- SPOKEN REGISTER — say it out loud, then type that: the contractions a relaxed speaker
  would use, plain everyday words over formal noun phrases, natural spoken elision licensed
  (owner decision this date: elision yes, internet slang and chat abbreviations no, borrowed
  caption vocabulary no), one intensifier, and wording-only scope (spelling, capitalization,
  punctuation rules unchanged).
- SAY IT ONCE — the second sentence inherits the first's topic; no restated connections, no
  setup repeated inside the question, questions stripped to the one clause a person would
  text.
- COMPLIMENT AS REMARK — the one allowed compliment (quota unchanged, its line untouched) is
  an offhand remark about the visible thing, the thing as grammatical subject, understated
  over emphatic; it shapes a compliment that occurs and never requires one (deliberately
  scoped to avoid re-imposing a mandatory shape — the same trap the 2026-08-16 correction
  removed from claims).
- VARY THE SHAPE — the verdict-then-question mold, the looks-like frame, and the binary
  or-question are each one available shape, never the default; a repeating shape is a
  template even when words change. This generalizes VARY THE OPENING from first words to
  whole-message structure, against the measured 30/40 mold.
- SHARED CONTEXT RULE gains: never point at the medium (a word gesturing at the photo,
  screen, or profile as an object); a bare demonstrative suffices. The rule's own "she is
  looking at that item" framing was arguably inviting the deictic "here".
- HARD RULE gains: when a compound would need a hyphen, rephrase; never delete the hyphen
  and glue the words.

WHY PROPERTIES AND NOT EXAMPLES OR MENUS: the 2026-08-11 "I bet" collapse (5/5) and the
2026-08-15 "my money is on" incident are both on record above; the 2026-08-16 de-templating
decision stands. No example copy ships in any model-facing string. The motivating opener and
the owner's rewrite are pinned off-wire in the new tests.

FOUR COPIES, NOT TWO — a correction to this doc's own §3.1 framing: the on-wire prompt
carries the rules in FOUR places, not two: config.yaml `opener.style` (user turn), `_SYSTEM`
(systemInstruction), the `_SCHEMA` field descriptions (responseJsonSchema), and the retry
hint block. All four were touched this date, but NOT with the same content, and a later
reader should not expect otherwise: the five register properties ship in `opener.style` and
`_SYSTEM` in lockstep, the `_SCHEMA` opener description carries a compressed form of THREE of
them (SPOKEN REGISTER, SAY IT ONCE, COMPLIMENT AS REMARK) plus the new never-point-at-the-
medium clause — it deliberately does not restate VARY THE SHAPE or the hyphen rephrase clause,
since a field description is read as a constraint on the value and those two are advisory —
and the retry hint block was corrected in an unrelated way — its HARD REJECTION list. That
list now names the scaffolding, disclaimer, shared-future, sensitive-inference and
undeliverable-character (emoji / non-ASCII) causes the code actually enforces, and no longer
spends its "especially" emphasis on the dash rule, which `_sanitize` launders and which
therefore can never be the reason an attempt was rejected. The undeliverable-character clause
is doing double duty: dropping the dash "especially" also dropped the retry turn's only ASCII
reminder, at exactly the moment SPOKEN REGISTER started pushing the model toward informality,
which is the direction emoji come from. One cause remains deliberately unnamed:
`REASON_UNDELIVERABLE_SEQUENCE` (a literal `%` directly against a following lowercase `s`)
stays out of every prompt copy, because a standing `%` warning costs more than the collision
does — that guard's own comment makes the argument, and its own `retry_hint` explains the fix
on the rare occasion it fires. So the list is complete except for one documented, intentional
omission; the invariant for the next editor is recorded at the block itself: a new raising
guard in `_parse()` needs a new clause here, or the list quietly becomes a lie again.

DELIBERATELY NOT CHANGED: the compliment quota line; CONVERSATIONAL VALUE RULE and its
claims-optional stance; POSITIVE SOCIAL FRAMING; REPLY COMFORT; CONFIRMATION BOUNDARY;
APPLICATION RULE's two-sentence ceiling and hard rules; PROFILE TEXT FACT CHECK (still
_SYSTEM-only); the observe/auto byte-identity contract; no new deterministic detector
(owner decision: prompt-only first, entropy guard remains the backstop; revisit only if a
watched batch still fails register — the 2026-08-11 scaffolding-defense decision is not
reopened).

RECORD-KEEPING CORRECTIONS FOLDED IN HERE, because this file stopped tracking Part A wording
on 2026-08-16 while four rule sets shipped after it: REFERENT CLARITY and REPLY COMFORT
(2026-08-17), CONFIRMATION BOUNDARY and INFORMATION GAIN (2026-09-02), POSITIVE SOCIAL
FRAMING (2026-09-04) exist only as dated comment addenda above `style:` in config.yaml and
in opener.py's log. This entry restores the doc as the index: the config comment log is the
de facto Part A changelog; read it alongside §2-3. Also stale as of this date: §3.7's cited
store line numbers; §3.1's statement that "the examples live in config.yaml opener.style"
(superseded 2026-08-16; no examples live anywhere on-wire) and its cited line numbers
`opener.py:49-75` / `config.yaml:280-300` (`_SYSTEM` now begins at opener.py:397, `style:` at
config.yaml:869); and §3.6's two-clause entropy-guard rationale. §3.6's second clause,
"few-shot examples make direct copying a live risk", died with the same 2026-08-16
de-templating pass; its first clause (shortening compresses the output space) is still true
and is still the structural reason collisions are likely at all — SAY IT ONCE shortens the
message further, so that clause gets MORE true this date, not less. The live reason the guard
exists is the measured 5/5 self-collapse of 2026-08-11 plus the fingerprint risk of an
uncapped-volume burner account. §3.6 is the most load-bearing of these three because the code
points AT it by name: the entropy guard comments in `opener.py` (`_leading_ngram`) and
`service.py` both cite "ops/OPENER-REDESIGN.md 3.6", so a reader following the code's own
pointer landed on the stale claim. Both comments now carry the corrected two-part rationale
and warn that §3.6's own text is half stale. Per the append-only rule for dated records, §3.6
itself is NOT edited in place.

RESIDUALS, stated rather than smoothed over:
- No row-level style version stamp exists anywhere (verified: zero hits for any
  prompt/style hash or version constant). The comparability boundary between register eras
  is created_at vs. config.yaml's git commit dates, now recorded in the calibration warning
  beside _redundant_description_markers, which names BOTH boundaries (2026-08-11,
  2026-09-05). A real stamp would need the two-step schema change bigquery_store.py:156-177
  documents; recommended as a follow-up, not smuggled into this change.
- Register compliance is UNMEASURED until a live batch runs. The prediction that is
  cheapest to check: apostrophe frequency moves off zero. The or-question and looks-like
  saturations should fall but VARY THE SHAPE is advisory; the entropy guard still only
  polices leading n-grams within a run.
- opener_rejections remains empty (0 rows ever) and is advisory-blind (rejections recorded
  only when not advisory, at two guarded sites in service.py), so "the guard never fires"
  cannot be read off it; this predates and survives this change.
- The style text is read once at supervisor startup; a running process keeps the old
  register rules until restarted.
- Editing config.yaml changes its whole-file SHA-256, which strands any in-flight
  calibration capture/review/video-bound session bound to the old bytes (five refusal sites
  check exact bytes). Recovery if needed: check out the pre-change config, measure, return.
- The weakest cascade models (two Gemma ids, thinkingLevel minimal, 16K TPM) now carry five
  more property rules; if their rejection rate rises, max_attempts=5 is the run-killing
  budget to watch.

BLAST RADIUS FOUND IN REVIEW AND FIXED THIS DATE — a detector whose safety depended on the
register we just changed. `_strip_wrapping_quotes` silently repairs an opener the model
wrapped in its own quote pair, but it refused to strip whenever the wrapping character also
occurred in the interior. Zero apostrophes in 188 openers meant a single-quote wrapper always
had a clean interior and always repaired; SPOKEN REGISTER makes contractions the norm, so
from this date it usually would NOT have. Verified before the fix: `'That's a nice mug.'`
came back unchanged and `_scaffolding_markers` returned `[]` for it, i.e. the literal leading
and trailing apostrophes would have been typed into her comment box with nothing downstream
catching it. The refusal now applies only to interior occurrences that are not INTRA-WORD (a
letter on both sides): a contraction repairs, a trailing possessive (`'Grams' pie is
unreal.'`) and a nested quote still refuse and still ship untouched, which is the residual.
The residual that MATTERS is a third shape, and it is the one this change itself
manufactures: `_is_intra_word` requires a letter on both sides, so the word-final elision
apostrophe SPOKEN REGISTER explicitly licenses is not intra-word,
`_strip_wrapping_quotes("'Nothin' fancy about that mug.'")` still refuses (verified live,
byte-identical before and after the fix) and those wrapping quotes ship — the possessive and
the nested quote are shapes nothing in this change encourages, this one it invites. The
pre-named fix, a scaffolding rule for a message still wholly wrapped in a matched pair after
the helper declines, is deliberately deferred to the first watched batch, per the prompt-only
owner decision recorded above. The general lesson is worth more than the fix: a measured-zero
property of model output ("it never emits apostrophes") had silently become a precondition of
a repair, and the change that moved that property lives in a different file from the code
that depended on it.

RESIDUALS ADDED IN REVIEW — in-prompt tensions this change created or sharpened. None is a
contradiction; each is a place where two rules pull against each other on `thinkingLevel:
minimal`, and each names the metric that would show it:
- NEGATION PRIMING: VARY THE SHAPE puts the literal phrase "looks like" on the wire in both
  copies in order to suppress a frame measured at 12/40. That is the same mechanism as the
  2026-08-11 "I bet" 5/5 collapse and the 2026-08-15 "my money is on" incident, and the same
  mechanism POSITIVE SOCIAL FRAMING itself states three paragraphs earlier. It does not
  violate the 2026-08-16 de-templating rule, which bans example opener COPY rather than
  naming a shape, and the CI negatives pass. But if the first live batch shows the looks-like
  rate flat or UP rather than down, the three shape names are the first thing to pull, not
  the rule.
- DEMONSTRATIVE PULL: the SHARED CONTEXT addition endorses the bare demonstrative ("a bare
  demonstrative carries all the shared looking the message needs") while VARY THE SHAPE, four
  paragraphs later, deprecates the verdict-then-question mold that the measured 21/40 opened
  with a demonstrative verdict. A bare demonstrative is not a demonstrative verdict and the
  owner's own rewrite is demonstrative-initial, so the copy is defensible as written; watch
  the demonstrative-opening rate as its own metric, separately from the or-question and
  looks-like rates.
- "HERE'S" COLLISION: `_SCAFFOLD_LEADING_PHRASES` still rejects any opener starting with
  "here is" / "here's" as REASON_SCAFFOLDING, and SPOKEN REGISTER raises the odds a model
  writes one — verified live, `_scaffolding_markers("Here's hoping that trail's as steep as
  it looks. ...")` returns a leading-interjection marker, i.e. a perfectly human spoken
  opening is a hard rejection against the same max_attempts=5 budget. The collision predates
  this change; the change raises its probability, and the new never-point-at-the-medium
  clause partly pulls the other way. No code change now: if REASON_SCAFFOLDING rows with a
  leading-interjection marker on "Here's" appear in opener_rejections, the fix is to drop
  "here's"/"here is" from that tuple (rule 1's leading-label clause already catches "Here's
  the response:"), NOT to weaken SPOKEN REGISTER.
- SAY IT ONCE vs CONFIRMATION BOUNDARY / REFERENT CLARITY: a confirm-or-correct second beat
  often needs exactly the qualifier SAY IT ONCE strips ("Am I right that you bake?" ->
  "Am I right?"), which is also where the referent goes ambiguous. Compatible for a careful
  reader, opposed on minimal thinking. If REFERENT CLARITY-shaped failures show up in the
  first watched batch, this is the named cause.
- HYPHEN CLAUSE WORDING: the style text says the model must "never write the compound with
  its hyphen silently removed, because the glued words read as a typo". The mechanism is
  slightly different from what that sentence describes — typography.py folds every hyphen to
  a SPACE, not to nothing, so the artifact is two space-separated words ("top secret survival
  gear", "a pose off"), never a concatenation. Do not go looking for a gluing bug. The rule's
  actionable half ("rephrase the sentence so no hyphen is needed") is correct and is what
  matters; the wording stands as the owner approved it.

Suite 4862 -> 4865 green. Offline only — no device, no live API call, nothing read from
ops/calibration/. New pins: tests/test_config_yaml_real.py::
test_shipped_opener_style_ships_the_spoken_register_rules, tests/test_opener.py::
test_system_prompt_ships_the_spoken_register_rules, tests/test_opener.py::
test_strip_wrapping_quotes_strips_around_a_contraction; retry-block and `_SCHEMA`
opener-description assertions extended in tests/test_gemini_opener.py (the schema copy was
pinned by nothing before this date, so it could have been deleted or drifted out of lockstep
with a fully green suite).

#### Addendum -- 2026-09-05 (b): the era boundary becomes a column

The trigger is the first RESIDUAL of the (a) addendum above, promoted by owner decision the
same day. That entry recorded, correctly, that no row-level style version stamp exists
anywhere, that the comparability boundary between register eras is therefore created_at
against config.yaml's git commit dates, and that a real stamp would need the two-step schema
change bigquery_store.py documents -- "recommended as a follow-up, not smuggled into this
change". This is that follow-up, shipped as its own change with its own tests.

WHAT SHIPS. A nullable `prompt_sha256` on BOTH `openers` and `opener_rejections`, in BOTH
backends (SQLite TEXT, BigQuery STRING). The value is the service's one-time
`prompt_stamp(style)`, computed once in `OpenerService.__init__` because all of its inputs are
fixed for the life of the process, and written on every row that service produces: the
immediate/legacy record site, the staged-commit path, and the rejection site. Both backends
follow their own documented two-step rule -- the `_SCHEMA` / `_TABLES` declaration reaches a
FRESH database only, and a matching ALTER (a new tuple entry plus a new stanza in sqlite's
`_initialize_schema`, two new `_MIGRATIONS` lines in BigQuery) is what carries the column into
a database that already exists. `opener_rejections` had never gained a column before, in
either backend, so this is that table's first migration in both.

STAMPED AT GENERATION, NOT AT COMMIT. The digest is captured onto `_StagedOpenerRecord` when
the draft is produced and replayed verbatim when the draft is committed. Staging is exactly
the gap in which the two could differ, and a draft attributed to a prompt it was not generated
under is worse than no stamp at all. `_record_staged_opener` probes the sink for the new
parameter through a probe that is deliberately ORTHOGONAL to the existing five-name lineage
probe: the two schema extensions shipped on different dates, so a duck-typed store may have
either, both, or neither, and folding them together would drop the stamp for one shape and
raise TypeError on the other -- turning a real Like into a failed worker run, which is the
exact thing that probe exists to prevent.

DIGEST COVERAGE, AND THE DELIBERATE EXCLUSIONS. The payload is SEVEN components joined in a
fixed order by a NUL byte: `opener.style` (the user turn), `_SYSTEM` (systemInstruction), the
canonicalized `_SCHEMA` field descriptions (responseJsonSchema) -- three of the FOUR on-wire
prompt copies the (a) addendum's four-copies correction names -- and then the item-crop shape's
four instruction constants, `_ITEM_PREAMBLE`, `_ITEM_PREAMBLE_CONTEXT`, `_ITEM_LABEL` and
`_CONTEXT_LABEL`. Those last four were folded in during review, before the column reached any
table, because they are model-facing instruction prose rather than bookkeeping -- the preamble
is the FIRST text part of every item-crop request, and between them they carry the
compound-item rule (a title, caption, or prompt above a photo is ONE item with it) and the
cannot-be-picked rule that `_CONTEXT_LABEL` restates on every context image -- and because they
satisfy the same coverage criterion as the first three: stable for a whole run and living in
module-level constants. `_SCHEMA` is canonicalized (sort_keys, compact separators,
ensure_ascii) so a cosmetic dict reorder cannot manufacture a false boundary, and the NUL sits
between EVERY adjacent pair so that prose moved from the end of one constant to the start of
the next cannot reassemble to the same bytes. It EXCLUDES: the retry-hint prose, because it is
request-conditional -- built per attempt from whichever guard rejected the previous draft -- so
including it would give one era as many digests as it has failure modes and would file an
attempt and its own retry under two eras; generationConfig / thinking, which change how hard
the model works rather than what it was told; and the model id, which is already its own
column in both tables and would only make the stamp less joinable.

DELIBERATELY NOT TOUCHED. The opener retraction fingerprint stays the closed five-field dict
(run_id, app, created_at, model, opener) and `advisory_opener_run_rows` stays a three-key
projection (created_at, model, opener_fingerprint). Five on-disk cleanup plans in
ops/corrections/ carry fingerprints computed from exactly those fields; widening either side
would re-key every stored plan against rows it already names, and `append_opener_retraction`
would then refuse them as "fingerprint no longer matches its cleanup plan" -- a plan that can
never be applied again. One pin per backend now asserts both halves, and the BigQuery one also
asserts the SELECT never asks for the column. `opener_send_evidence` is NOT stamped: the owner
scoped this change to the two named tables, and that table joins to `openers` anyway.

NULL SEMANTICS. The column is nullable and nothing backfills it. NULL means "written before
2026-09-05 (b)", not "no era" -- those rows still need the created_at vs. git-dates split, so
an offline pass over a range spanning this date uses both methods at once. That is now stated
in the CALIBRATION WARNING beside `_redundant_description_markers`, appended to rather than
replacing the two-boundary text already there.

MIGRATION EXECUTION TIMING. Nothing runs at edit time. SQLite applies its ALTERs the next time
a store is constructed, BigQuery at the next `ensure=True` init -- i.e. the next live run, in
both cases. BigQuery's grouped-ALTER behavior is unchanged: `_ensure_tables` still collects
every needed addition per table into ONE `ALTER TABLE` statement and submits the whole script
as a single job with the existing rate-limit retry, so adding two columns across two tables
adds two clauses, not two round trips per column. This matters because `flush_every=1` is a
live configuration: a row carrying a field the live table does not declare is rejected on the
FIRST insert of the run and takes its whole batch with it (skip_invalid_rows is unset), which
is the entire reason a `_TABLES` entry without a `_MIGRATIONS` line is a production bug that
passes every local test. The `_MIGRATIONS` lines MOVE that failure mode rather than eliminate
it: BigQuery's streaming insert path does not always see a freshly ALTERed table's new column
immediately, so the very first opener or rejection insert of the run that applies the ALTER can
still be rejected for schema propagation lag and take its batch with it. Every prior `openers`
column shipped exactly this way without incident, and the loss is warning-level rather than a
run failure -- `record_opener` runs inside the service's non-fatal try/except -- so this is
recorded as a known first-run cost, not designed around.

RESIDUALS.
- The stamp cannot distinguish two eras that differ ONLY in retry-hint prose. That is the
  priced cost of the exclusion above, not an oversight; the retry block was in fact edited on
  2026-09-05 (a), so a pair of eras with this exact shape is not hypothetical.
- The retry hint is NOT the only such gap, and "equal digests mean the same era" must not be
  read as unconditional. The one remaining un-hashed body of model-facing instruction prose is
  `_text_part`'s TRAILING text part, and only that: the closing block, its truncation and
  CONTEXT sentences, and the STYLE GUIDE / HER NAME / HER PROFILE TEXT section labels. The
  item-crop preamble and the per-image labels are NOT in this gap -- the first review pass left
  them out and this addendum's DIGEST COVERAGE now hashes all four of those constants, so an
  edit to the compound-item rule or the cannot-be-picked rule DOES move the digest. What
  remains excluded is excluded on the same principle as the retry hint: the trailing block is
  request-conditional as a whole (item counts, name section, and truncation sentence vary per
  profile), so covering it would mean hashing a per-profile string, i.e. a stamp that is no
  longer per-run -- the pricing argument is valid for this body and for no other. Unlike
  bookkeeping it still carries FIXED sentences that are real prompt rules, notably its own
  CONTEXT-images sentence ("Never pick one or make one the opener's main premise") and the
  item_index instruction, and that copy has been rewritten before. Editing one of those
  sentences is a genuine era boundary that produces an identical digest: the precise false
  negative this column exists to prevent, narrowed twice now but not eliminated. Named here
  and in `prompt_stamp`'s own docstring so a later reader does not conclude that any
  prompt-rule edit outside the retry hint moves the digest.
- `opener_rejections` remains advisory-blind -- rejections are recorded only when not advisory,
  at the two guarded sites in service.py -- so a stamped rejection table still cannot answer
  "how often does this guard fire" for observe sessions. Unchanged by this entry, and it
  survives it.
- The stamp answers WHICH era, never WHAT changed. Reading a digest back requires checking out
  the suspected commit and calling `prompt_stamp(cfg.opener.style)`; equal digests mean the
  same era. That recipe is in the function's own docstring so a later reader is not left with
  an opaque hex string.
- Legacy rows stay NULL forever. A backfill would have to guess an era from created_at, which
  is precisely the reconstruction this column exists to replace, so it is deliberately not
  attempted.

TEST PINS BY NAME. New: tests/test_opener.py::test_prompt_stamp_is_a_64_character_hex_sha256,
::test_prompt_stamp_is_deterministic_for_the_same_style,
::test_prompt_stamp_changes_when_the_owner_style_text_changes,
::test_prompt_stamp_changes_when_the_system_prompt_changes,
::test_prompt_stamp_changes_when_a_schema_field_description_changes,
::test_prompt_stamp_changes_when_an_item_crop_instruction_constant_changes (parametrized over
all four of `_ITEM_PREAMBLE`, `_ITEM_PREAMBLE_CONTEXT`, `_ITEM_LABEL`, `_CONTEXT_LABEL`, one
constant at a time, so dropping any single component from the payload fails on its own),
::test_prompt_stamp_separates_adjacent_constants_rather_than_concatenating_them,
::test_prompt_stamp_ignores_schema_dict_key_ordering;
tests/test_sqlite_store.py::test_sqlite_record_opener_persists_the_prompt_era_stamp,
::test_sqlite_record_opener_leaves_the_prompt_stamp_null_when_the_caller_omits_it,
::test_sqlite_record_opener_rejection_persists_the_prompt_era_stamp,
::test_sqlite_migrates_legacy_openers_prompt_sha256_column,
::test_sqlite_migrates_legacy_opener_rejections_prompt_sha256_column,
::test_sqlite_opener_rejections_prompt_sha256_migration_reraises_unexpected_operational_errors,
::test_sqlite_openers_alter_loop_reraises_unexpected_operational_errors,
::test_sqlite_advisory_opener_run_rows_keep_the_prompt_stamp_out_of_the_fingerprint;
tests/test_bigquery_store.py::test_ensure_tables_declares_prompt_sha256_on_the_openers_create_table,
::test_ensure_tables_runs_openers_prompt_sha256_migration,
::test_ensure_tables_declares_prompt_sha256_on_the_opener_rejections_create_table,
::test_ensure_tables_runs_opener_rejections_prompt_sha256_migration,
::test_record_opener_buffers_the_prompt_era_stamp,
::test_record_opener_buffers_a_null_prompt_stamp_when_the_caller_omits_it,
::test_record_opener_rejection_buffers_the_prompt_era_stamp,
::test_record_opener_rejection_signature_stays_aligned_with_the_store_protocol,
::test_bigquery_advisory_opener_run_rows_keep_the_prompt_stamp_out_of_the_fingerprint;
tests/test_opener_service.py::test_rejection_rows_carry_the_prompt_era_stamp_of_the_style_the_service_runs,
::test_immediate_opener_row_carries_the_prompt_era_stamp,
::test_staged_commit_stamps_the_era_the_draft_was_generated_under. Extended:
tests/test_sqlite_store.py::test_sqlite_openers_table_has_the_expected_columns_in_order,
tests/test_bigquery_store.py::test_record_opener_signature_stays_positional_compatible_with_the_store_protocol
(which also pins Protocol/backend signature equality),
tests/test_opener_service.py::test_committed_opener_carries_exact_landed_action_lineage,
tests/test_worker.py::test_auto_persists_landed_decision_before_committing_its_staged_opener.
The pre-existing
tests/test_bigquery_store.py::test_record_opener_writes_exactly_the_columns_the_openers_table_declares
enforces _TABLES-to-row-dict parity on its own and was left alone; it passes.

MUTATION-CHECKED, since a green test is not evidence it reaches the branch it names: dropping
`prompt_sha256` from the sqlite openers INSERT fails two tests; dropping it from the BigQuery
buffered row dict fails three, including the exact-columns parity test; adding a dummy key to
either backend's `advisory_opener_run_rows` emitted dict fails that backend's fingerprint
isolation pin (and three ops/corrections tests besides). Four more were added in review, all on
guards a pass shipped unpinned. (1) Rewriting the new `opener_rejections` ALTER's re-raise
branch as a bare `except sqlite3.OperationalError: pass` left tests/test_sqlite_store.py
entirely green before the fail-loud pin above existed and fails exactly that one test after.
(2) The openers half was FIRST claimed to need no separate pin -- "it rides the existing ALTER
loop and inherits its guard" -- and that claim was mutation-tested and found FALSE: the angle
and item_description pins each intercept a standalone ALTER with its own try/except, while the
loop carries a third handler that nothing reached, so rewriting the LOOP's handler as a bare
`except sqlite3.OperationalError: pass` also left the file green. The fix was to add the
missing pin rather than soften the sentence; with
::test_sqlite_openers_alter_loop_reraises_unexpected_operational_errors in place that same
mutation now fails exactly one test out of the whole suite (4895 passed, 1 failed), and the
docstring clause points at the pin that actually covers it. (3) Replacing the digest payload's
NUL join with a plain concatenation fails only
::test_prompt_stamp_separates_adjacent_constants_rather_than_concatenating_them, which is the
whole reason the separators are there. (4) Making `prompt_sha256` positional on
`BigQueryStore.record_opener_rejection`, or dropping it from the Protocol, each fails only the
new rejection signature pin. Every mutation was restored byte-identically (verified by hash and
by `git diff`) and the suite re-verified green.

Suite 4865 -> 4896 green (31 new tests, 5 skipped, unchanged). Offline only -- no device, no
live API call, no BigQuery read or write from this change; the migrations execute on the next
live store init.

#### Addendum -- 2026-09-05 (c): the nouns were clear, but the modifier attached two ways

The trigger was the first watched Training draft after the spoken-register rewrite. The Hub
showed this opener on Maja's selected snow photo (run `d986f1b30059`, model item 1):

> You look completely at home bundled up in all that snow. Are you genuinely a winter person
> or strictly in it for the cozy lodge after?

The owner's reading -- that she was not literally bundled up *in snow* -- is a reasonable
reading of the word order. The intended composition is also recoverable: she is "bundled up"
in the ordinary sense of wearing warm clothes, while "in all that snow" locates the whole
scene. The defect is that the location phrase immediately follows the clothing phrase, so it
can instead be parsed as what completes "bundled up in ...". An opener that makes its reader
stop and select between those two relationships has failed even when the intended one is
eventually understandable.

THIS WAS NOT IMAGE RECOGNITION OR TARGET BINDING. The model's own `about` text correctly named
Maja's puffer coat, beanie, snowy mountains, and frozen water. The retained phone frame shows
those same things. The action ledger maps model item 1 to page heart 8, then verifies the open
composer against that stored crop three times at 0.236 grey levels against a 36.229 bound
before recording the exact draft. Stop later cancelled the Training checkpoint, so the text
was never sent or committed. The evidence therefore isolates generation wording: the model
saw the right item, understood its relevant objects, and assembled them into an ambiguous
sentence.

THE MISSING PROPERTY. `REFERENT CLARITY` governs what a pronoun, shorthand noun, or question
subject denotes and whether a later beat silently changes that denotation. Every noun in this
incident passes that rule. The ambiguity instead concerns grammatical attachment: which word
or idea a modifying phrase describes, and therefore which semantic relationship the sentence
asserts. Spoken register, setup/payoff continuity, and role consistency do not settle that
either. This is a separate property rather than another exception packed into one of those
rules.

WHAT SHIPS. `MODIFIER CLARITY` is mirrored across the three stable prompt surfaces that carry
semantic wording constraints: the owner-tunable `config.yaml` `opener.style`, `_SYSTEM`, and
the response schema's `opener` field description. All three state the same core property:
every modifying phrase must have only one natural attachment on first reading, and wording
whose placement permits a plausible unintended meaning is reordered or rephrased. This avoids
turning the correction into a blanket adjacency rule that would overconstrain harmless
clause-level modifiers. The config version remains the long style and the other two are
compressed copies. Because all three are inputs to `prompt_stamp`, this edit creates a new
prompt-era digest automatically; no storage migration or stamp plumbing changes.

WHY PROMPT-ONLY. There is no lexical blacklist for "bundled up", "in snow", or their
combination. Each is harmless and natural in other sentences, while the actual class spans
arbitrary verbs, modifiers, and relationships. A regex would catch one report rather than the
defect, create false positives, and turn them into billed retries against the run-stopping
`max_attempts=5` budget. No deterministic parser in this project has enough semantic context
to judge the general class at send confidence. Consequently `_parse`, reason codes,
`OpenerService`, and the retry hard-rejection list are unchanged. A retry already receives the
full style guide; there is no new hard rejection for the list to name.

DE-TEMPLATING STILL HOLDS. The live opener appears only in comments, this dated record, and
test docstrings. It does not occur in `opener.style`, `_SYSTEM`, or `_SCHEMA`; the regression
tests assert that explicitly. The model receives the abstract property, never the salient
wording that produced it.

TEST PINS BY NAME. New:
`tests/test_config_yaml_real.py::test_shipped_opener_style_ships_modifier_clarity_without_incident_copy`
and
`tests/test_opener.py::test_system_prompt_ships_modifier_clarity_without_incident_copy`.
Extended:
`tests/test_gemini_opener.py::test_response_schema_orders_referenced_and_angle_before_the_opener`
pins the third copy and the same off-wire boundary. Focused verification also reruns the style
typography/ASCII contract, the main system-prompt contract, prompt-stamp determinism, and the
no-retry request baseline. Offline only -- no device input, live provider request, or storage
write. Focused prompt/config verification: 345 passed.

#### Addendum -- 2026-09-05 (d): an inferred location gets no second premise

The trigger was the next watched Training draft, again on Maja's snow photo (run
`397c1f944d22`, model item 1):

> That looks a lot like Lake Louise in deep winter. Were you out there ice skating or just
> braving the freeze for the view?

The first sentence correctly marks Lake Louise as an inference. The second sentence does not
wait for Maja to confirm it: both offered answers put her "out there", so the question silently
promotes the guess to fact. This is the same defect as the country-guess incident that caused
CONFIRMATION BOUNDARY on 2026-09-02, not a new tone preference. The action ledger ends at
`auto_opener_pre_send`; there is no approval or resumed-send event, so this draft was typed into
the Training composer but was not sent or committed.

PROMPT-ONLY FAILED. CONFIRMATION BOUNDARY was already present in `config.yaml`, `_SYSTEM`, both
response-schema planning fields, and the retry guidance. A later 2026-09-05 draft had likewise
guessed an Iceland tour and immediately asked whether the vehicle went onto a glacier or into
the highlands. The existing prose therefore described the right property but was not a send
boundary. Three nearby instructions also weakened it under minimal thinking: SAY IT ONCE said a
second sentence inherits the first; REPLY COMFORT discouraged asking to verify an ambiguous
visible detail; and APPLICATION RULE generically allowed a second positive question. VARY THE
SHAPE additionally named the literal looks-like and two-choice structures it was trying to
de-emphasize, the negation-priming risk already called out in addendum (b)'s review.

WHAT NOW SHIPS. The prompt calls an image-derived place guess the entire conversational move
until she replies. The opener may end after the guess or ask only whether the location itself is
right. It may not append an activity, reason, preference, feeling, experience, or consequence at
that place. SAY IT ONCE and APPLICATION RULE explicitly defer to this boundary, while REPLY
COMFORT distinguishes direct location confirmation from interrogating her about an ambiguous
detail. VARY THE SHAPE is now abstract: the item determines sentence count, clause pattern, and
question form, with no concrete high-frequency structures supplied for imitation. The long
`config.yaml` form, compressed `_SYSTEM` mirror, both `_SCHEMA` planning descriptions, and retry
guidance all carry the same boundary. Because these are prompt inputs, `prompt_stamp` creates a
new era digest automatically.

THE SEND BACKSTOP. `_unconfirmed_location_followup_markers` runs after sanitization inside
`GeminiOpener._parse`. It requires both (1) explicit inference language whose conclusion begins
with a capitalized proper name and (2) a later sentence or question-like clause. A standalone
guess passes. A later beat also passes when its whole job is to confirm or correct the place.
Every other later beat is rejected as `REASON_UNCONFIRMED_LOCATION_FOLLOWUP`, preserving the
sanitized candidate in `raw_opener`. `OpenerService` already catches `OpenerParseError`, records
the rejection reason, and retries, so no service control-flow change was needed. The retry turn
now names this hard rejection and tells the model to end after the location guess or ask only
whether it is right.

PRECISION CHECK. The detector was run read-only across all 106 unique locally retained opener
drafts. It selected six, and all six are genuine location-guess-then-assumption failures: Italy,
Spain/Mallorca, Switzerland, Namsan Tower, the Iceland tour, and Lake Louise. It selected none of
the other 100. In particular, the retained Portugal opener that ends by asking whether the
location guess is close and the British Columbia opener that asks for location correction both
remain valid. Ordinary figurative uses such as a pet looking like a guard also remain outside
the gate. This is intentionally not a general named-entity recognizer: prompt semantics still
cover wordings that an outgoing-text check cannot identify at send confidence, while the narrow
guard guarantees the recurring live form cannot pass unchanged again.

TEST PINS BY NAME. New and extended cases in `tests/test_opener.py` cover all six retained live
failures, sentence and comma continuations, a direct location question followed by an assumed
experience, standalone guesses, confirmation tags, correction questions, stated locations,
lowercase interpretations, and figurative pet language. The parser regression proves the exact
Lake Louise draft raises the new reason before it can be returned, and the service regression
proves that rejection is retried and a confirmation-only replacement succeeds. Prompt mirrors
are pinned in `tests/test_opener.py`, `tests/test_config_yaml_real.py`, and
`tests/test_gemini_opener.py`, including the absence of VARY THE SHAPE's two concrete structural
phrases. Focused opener/config verification: 368 passed. Offline only -- no device input,
provider request, or storage write.

#### Addendum -- 2026-09-06: a compliment moved off her is still a score

TRIGGER. Owner feedback on a watched Training draft, the New Year's sushi photo, model item 1:

> Starting the new year with a massive spread of sushi is an elite move. Was that the main
> event for the night or just the appetizer?

The owner's objection, quoted in substance: calling something an elite move sounds so
superficial with nothing interesting underneath, it is the same as saying you are so hot.

ROOT CAUSE, and this is the point of the entry. COMPLIMENT AS REMARK, which shipped the
previous day, produced this opener. It located the defect in a compliment being a verdict on
her and being emphatic, and instructed the model to prefer the thing as the sentence's subject
and to keep it understated rather than emphatic. The draft satisfies every clause: the thing is
the subject, the register is casual, nothing is said about her. The rule worked and the output
still failed, because the defect is not the verdict's target or its volume. It is the verdict.
She chose to post the item, so a judgment of how good it is transfers no information she did
not already have, which is exactly why it lands like a compliment on her appearance.

MEASURED. 95 unique openers recovered from the 26 populated `data/hinge_debug/<run_id>/actions.jsonl`
logs (61 run directories present locally; the store at `data/operation_love.db` is empty and on a
stale pre-redesign schema, no `opener_events` or `actions` table exists there). "Elite move"
appears verbatim twice: the sushi draft above and "Matcha tiramisu for a birthday cake is such an
elite move. Did you make that yourself?" The same grade-shaped predicate appears in roughly 20 of
the 95: is a bold move, is iconic weekend energy, is top tier apres ski energy, is unmatched, is a
masterpiece, is unreal, is impressively professional, a great look (3), the perfect (4), an
incredible or amazing experience (2), is pretty fantastic (2). State plainly that this is a
PARTIAL LOCAL SAMPLE: BigQuery `operation_love.openers` and `opener_rejections` are the system of
record and were not queried, so the true incidence is unmeasured and should be read off BigQuery
before any escalation decision.

WHY NOT A PHRASE BLOCKLIST. It would have caught 2 of about 20. The defect is a predicate shape,
and a vocabulary ban is routed around by the next synonym. This is consistent with the
2026-09-05 modifier-clarity reasoning that a lexical pattern for one observed sentence overfits a
general problem.

STRUCTURAL CAUSE WORTH RECORDING. CONVERSATIONAL VALUE, INFORMATION GAIN, MINIMUM INVENTION, and
NEVER INVENT THE SENDER jointly leave the taste verdict as the only predicate that requires zero
facts about either party, while PRIMARY ITEM RULE still demands the opener justify the Like and
the second beat still needs a referent anchor. The grade is therefore the path of least
resistance the rule set itself created, not a stylistic tic. Any future rule that closes an
escape hatch without opening one should expect the same class of result.

WHAT SHIPS. NO GRADING, prompt-only, in the same three stable semantic surfaces MODIFIER CLARITY
used: `config.yaml` `opener.style` (long form), `_SYSTEM` (compressed), and the response schema
(the `opener` description carries the compressed clause, the `angle` description carries the
self-check because it is the model's only pre-opener scratchpad under minimal thinking). The
rule's operational test, quoted here as it appears on wire: read the sentence again with the item
swapped for a different woman's different photo, and if the predicate still fits unchanged it was
a grade. NO GRADING explicitly narrows COMPLIMENT AS REMARK rather than competing with it, since
moving praise off her and onto the thing does not by itself make it a remark.

THE DESIGN CONSTRAINT THAT SHAPED IT. The 2026-08-16 minimum-invention addendum records that
making falsifiability unconditional pressured the model to invent motives merely to produce a
correctable claim. NO GRADING therefore routes its own failure case to a shorter message, cut the
beat and let one specific question be the whole message, and never to a substitute claim.
Removing that escape hatch in a future edit re-creates the 2026-08-16 bug.

DELIBERATELY NOT CHANGED. No deterministic detector and no new retry reason, so the retry hint
block (copy 4 of 4) is untouched; the 2026-08-11 scaffolding-defense decision is not reopened and
the 2026-09-05 owner decision that prompt-only comes first is followed. No example verdict
vocabulary ships in any model-facing string, per the 2026-08-16 de-templating addendum; the
concrete corpus regressions above are pinned off wire in tests only.

RESIDUALS, stated rather than smoothed over.

- NO GRADING materially narrows what COMPLIMENT AS REMARK still permits. An OWNER DECISION is
  open on whether the one permitted compliment should be narrowed (what shipped) or retired
  outright. As shipped it survives only when the praise is inseparable from a specific
  observation about that item.
- Compliance is UNMEASURED until a live batch runs. Cheapest prediction to check: the "X is
  <positive noun phrase>" first-beat frame falls from roughly 1 in 5 toward zero, and
  single-sentence openers rise, since the escape hatch routes to a shorter message.
- The SUBSTITUTION TEST is advisory prose under minimal thinking, like VARY THE SHAPE, and the
  weakest cascade models now carry one more property rule; watch their rejection rate against
  the `max_attempts` budget.
- `opener_rejections` stays empty and advisory-blind, so "the guard never fires" still cannot be
  read off it. Unchanged by this entry, but it means the escalation trigger has to come from a
  watched batch rather than from a table.
- This edit changes `config.yaml`'s whole-file SHA-256 and `prompt_stamp()`'s digest, so it opens
  a new prompt era: rows before and after are not grade-comparable, and any in-flight calibration
  session bound to the old config bytes is stranded (recovery: check out the pre-change config,
  measure, return).

TEST PINS BY NAME. New:
`tests/test_config_yaml_real.py::test_shipped_opener_style_ships_the_no_grading_rule`,
`tests/test_opener.py::test_system_prompt_ships_the_no_grading_rule`,
`tests/test_opener.py::test_no_grading_rule_ships_no_example_verdict_vocabulary`,
`tests/test_opener.py::test_angle_field_carries_the_no_grading_self_check`,
`tests/test_opener.py::test_no_grading_motivating_openers_stay_off_every_prompt_surface`, and
`tests/test_gemini_opener.py::test_response_schema_opener_description_ships_the_no_grading_rule`.
All six are new, none replace an existing case. Offline only -- no device input, live provider
request, or storage write.

Full suite (`pytest -q -n auto`): 4896 -> 4953 green (5 skipped, unchanged). The six NO GRADING
pins above account for part of that growth; the remainder belongs to other in-flight edits on
this branch (sqlite and BigQuery store, ranker, worker, hub changes) landed concurrently and not
itemized here. 297s wall clock. Offline only -- no device, no live API call, no BigQuery read or
write from this change.

#### Addendum -- 2026-09-06 (b): a carve-out, a subordination clause, and the "HERE'S" collision closed

Three fixes, closing gaps an adversarial review and a separate audit found in the 2026-09-06 NO
GRADING addendum above.

**Fix 1, PLAYFUL HYPERBOLE carve-out (MEDIUM).** NO GRADING's SUBSTITUTION TEST asks whether a
predicate would still fit unchanged under a different woman's different photo; PLAYFUL HYPERBOLE
licenses an unmistakably nonliteral exaggeration whenever its visible anchor is immediate. An
anchored nonliteral line can be exactly as portable as a grade in the swap sense without being one,
and narrowing it would have cost a real, measured move: hyperbole is 13.7% of first beats on the
local 95-opener corpus the 2026-09-06 addendum above names, one of the few non-degenerate moves the
model still reaches for. A carve-out ships in all three on-wire surfaces (`config.yaml`
opener.style, `_SYSTEM`, and the response schema's `opener` description): the swap test targets a
literal verdict on quality, and an unmistakably nonliteral PLAYFUL HYPERBOLE survives it because
playful framing is not an assessment of quality, even where its wording could transfer.

**Fix 2, VARY THE SHAPE subordination (MEDIUM).** NO GRADING routes its failure case to "cut that
beat and let one specific question be the whole message"; VARY THE SHAPE says no structure is the
default. Nothing stated precedence between them, so the one-question fallback could itself become
a standing template. A short subordinating clause now states the fallback remains governed by VARY
THE SHAPE: the construction of that one question must still come from the item. MEASURED CONTEXT,
deliberately kept off wire so as not to overweight a guard against a problem that has not happened:
the local corpus is 94.7% exactly two sentences and 98.9% question final, so today routing MORE
messages to the one-sentence fallback INCREASES shape diversity rather than collapsing it. This
clause guards a future regression; it does not correct a present one.

**Fix 3, the "HERE'S" collision (real live bug, found by audit).** The residual named in the
2026-09-05 register-rewrite addendum above was live: `_SCAFFOLD_LEADING_PHRASES` matched a bare
leading "here is"/"here's", and SPOKEN REGISTER raising the odds a model writes a natural spoken
"Here's ..." opening meant a correct human line burned a retry against `max_attempts=5` -- verified,
`_scaffolding_markers("Here's hoping that trail's as steep as it looks. ...")` returned a
leading-interjection marker. Fixed in the GUARD, not the prompt. `"here is"`/`"here's"` are removed
from the bare phrase tuple; a new `_SCAFFOLD_HERE_IS_OBJECT_RE` narrows the rule to the shape that
actually distinguishes scaffolding from speech: a determiner immediately followed by a meta noun
naming the output itself ("the response", "an option", "a version", "my take", ...), almost always
before a colon. "Here's hoping ...", "Here's to ...", and "Here's the thing, ..." do not match --
"hoping", "to", and "thing" are never the artifact -- so a genuine spoken opening now passes, while
"Here's the response: ..." and "Here's an option: ..." still raise `REASON_SCAFFOLDING`. This
touches no prompt text on any of the four on-wire copies; the fix is entirely inside the guard.

TEST PINS BY NAME. New:
`tests/test_opener.py::test_generate_still_raises_parse_error_on_narrowed_here_is_scaffolding` and
`tests/test_opener.py::test_generate_accepts_a_natural_spoken_here_is_opening`; three new entries
in `_SCAFFOLDING_FALSE_POSITIVES` and two in `_SCAFFOLDING_TRUE_POSITIVES` (same file); and
extended assertions in
`tests/test_config_yaml_real.py::test_shipped_opener_style_ships_the_no_grading_rule`,
`tests/test_opener.py::test_system_prompt_ships_the_no_grading_rule`, and
`tests/test_gemini_opener.py::test_response_schema_opener_description_ships_the_no_grading_rule`.
Each new assertion and corpus entry was mutation tested: temporarily reverting the code or prompt
text it pins made it fail, and the file was restored after.

Fixes 1 and 2 change `config.yaml`'s whole-file bytes and therefore `prompt_stamp()`'s digest again,
opening a new prompt era on top of the 2026-09-06 one; fix 3 changes no on-wire text at all.

Full suite (`pytest -q -n auto`): 4953 -> 4960 green (5 skipped, unchanged), 267s wall clock.
Offline only -- no device, no live API call, no BigQuery read or write from this change.

#### Addendum -- 2026-09-06 (c): the record only kept the openers that worked

TRIGGER. Owner instruction during the NO GRADING change above: make sure we have a sufficient
logging system in place for the opener generation revisions so we don't end up going in
circles.

THE MEASURED JUSTIFICATION, which is the point of this entry. An audit of every prompt change
recorded in this document found that of 8 prompt-only fixes, ZERO are documented as verified
working by a post-ship re-measurement, 3 are documented as having failed, regressed, or
required escalation to a deterministic guard, and 5 shipped with no outcome measurement
recorded at all. Every register-era addendum above closes with some spelling of "UNMEASURED
until a live batch runs." The one intervention in this entire document carrying a real
quantified result is the 2026-09-05 (d) deterministic guard (6/6 true positives, 0/100 false
positives on 106 historical drafts, from that addendum's own PRECISION CHECK). Going in circles
was therefore the predicted outcome of the existing instrumentation, not bad luck.

THE FOUR GAPS FOUND.
1. SURVIVORSHIP BIAS BY CONSTRUCTION, the largest one. Training gated `commit_opener` strictly
   inside `if outcome == "like":` with no else branch, and AUTO inside `if d.decision ==
   "like":` likewise, so every draft a person disliked, stopped, or that hit a driver refusal
   was discarded and never reached the durable record. The stored corpus could therefore never
   answer "how often does the model write a bad opener", because the bad ones were exactly the
   rows deleted.
2. Rejections that were not `OpenerParseError` were never written at all in any mode: the
   `OpenerError` branch and the HTTP 400 and transient latch points had no store call.
3. The in-memory diagnostic ring buffers `bugreport.py` reads (`recent_openers`,
   `recent_rejections`) carried no prompt era stamp, so the one always-on view could not be
   grouped by era even in principle.
4. The only real corpus was 26 populated files out of 61 run directories under
   `data/hinge_debug/`, which is gitignored, and `data/operation_love.db` was empty and on a
   stale pre-redesign schema.

CORRECT THE RECORD, explicitly and without editing the old lines. This document's standing
explanation that `opener_rejections` is empty because it is "advisory-blind" is STALE and was
already stale when last restated. The two `if not advisory:` guards could not have been the
cause: Training and AUTO are the only production callers of `maybe_opener`, and neither ever
passes `advisory=True`, because Observe's call site was removed by commit ea6756e8 on
2026-08-26 and `tests/test_worker.py` pins that AUTO must never pass `advisory=True`. The real
cause was gap 2 above. This corrects, rather than repeats, the earlier addenda's reasoning.

WHAT SHIPS.
- A new `advisory` column on `opener_rejections` in both backends, with BOTH halves of the
  two-step migration the `prompt_sha256` column already proved: the CREATE-table declaration
  for a fresh database AND the paired ALTER for the live already-populated table. Shipping
  only the first half is the one documented way to break production, since local tests all
  pass while streaming inserts fail with "no such field".
- Both `if not advisory:` guards removed so rejections always record, with the advisory value
  preserved as data rather than by omission.
- Rejection recording added to the three previously silent failure branches (`OpenerError`,
  the HTTP 400 path below the latch threshold, and the generic transient-exception path), with
  new `REASON_OPENER_ERROR` / `REASON_BAD_REQUEST` / `REASON_TRANSIENT_ERROR` reason codes.
- `prompt_sha256` added to the `recent_openers` and `recent_rejections` ring buffers.
- `OpenerService.discard_opener` plus `worker.py` calls in the previously missing else
  branches, so drafts that were not sent are durably recorded with the generation-time prompt
  era and a decision recording what actually happened. This is the single biggest lever for
  diffing era N+1 against era N.
- The driver now carries the run's prompt era into the `actions.jsonl` opener events via a
  one-time `set_opener_prompt_sha256` setter following the existing `bind_debug_run` pattern,
  chosen over threading the value through the `Driver.like` ABC, which dozens of test fakes
  implement; justified because `OpenerService.prompt_sha256` is computed once in `__init__`
  and fixed for the life of the process, so a run-level bind is exactly as fresh as per-draft
  threading.
- A new offline report, `tools/opener_corpus_report.py`, computing grade rate, sentence-count
  and question-final distributions, opening trigram diversity, apostrophe rate, "looks like"
  rate and length, grouped by prompt era and by decision, with a compare mode for diffing two
  eras.

THE DESIGN POINT WORTH RECORDING. The deterministic grade detector that was correctly REJECTED
as an online guard is correct as an OFFLINE measurement. Online it would carry a
false-rejection cost against the `max_attempts` budget and would put its own vocabulary on the
wire; offline it has neither cost, so the lexicon lives in the tool only and is never imported
by the prompt or by any guard. This resolves rather than reopens the 2026-08-11
scaffolding-defense decision: measure lexically offline, do not enforce lexically online.

VALIDATION. The tool independently reproduced the hand-measured figures from the 2026-09-06
addendum above on the same 95-opener local corpus: 94.7% exactly two sentences and 98.9%
question final. Its deterministic grade detector reports a materially LOWER rate than that
addendum's structural classification did (12.6%, 12/95, versus roughly 45.3%), because the two
use different definitions: the tool's is a deliberately conservative lower bound over an
explicit lexicon, while the structural read counted any first beat whose predicate was an
evaluative verdict. The tool's value is a fixed definition that tracks DIRECTION era over era,
not an absolute level, and the two numbers must never be compared to each other.

RESIDUALS, stated rather than smoothed over.
- Historical rows are NOT retroactively fixed. Every `openers` row written before this change
  exists only because its draft was sent, so all pre-change store data stays survivorship
  biased. Only rows written after this date carry both populations.
- The local `data/hinge_debug` corpus still carries no era stamp for runs recorded before this
  change, so its numbers aggregate multiple prompt eras and cannot be attributed to any one.
- There is still NO reply, match, or outcome signal anywhere in this system. Everything this
  instrument measures is what the prompt PRODUCES, never what PERFORMS. Adding an outcome
  signal remains the single largest missing input to any future quality claim.
- `data/operation_love.db` on this machine remains a stale pre-redesign artifact (schema:
  `labels`, `decisions`, `openers`, `spend` -- no `opener_rejections`, no `prompt_sha256`, no
  `advisory`) that the current SQLiteStore has evidently never opened; the migrations will
  apply on first real open.
- Whether the now-orphaned `advisory` parameter should be reconnected to a real preview caller
  or retired outright is an OPEN OWNER DECISION, deliberately not made here.
- The first question this instrument should be pointed at, because it is currently
  unanswerable: VARY THE SHAPE deliberately puts the literal phrase "looks like" on the wire to
  suppress a frame the 2026-09-05 addendum above measured at 12/40. The tool measures that same
  frame at 27.4% (26/95) on the current local corpus -- only marginally below the pre-fix 30%
  baseline, not the clean drop the fix was meant to produce -- and with no era stamp on those
  rows it is impossible to tell whether this is a real but diluted suppression, no effect at
  all, or the aggregate simply mixing pre- and post-fix eras together.

TEST PINS BY NAME. New in `tests/test_opener_service.py`:
`test_opener_error_now_records_a_durable_rejection`,
`test_http_400_records_a_durable_rejection_below_the_latch`,
`test_http_400_records_a_rejection_on_every_call_including_the_one_that_latches`,
`test_transient_error_records_a_durable_rejection_below_the_latch`,
`test_transient_error_records_a_rejection_on_every_call_including_the_one_that_latches`,
`test_advisory_opener_error_and_400_and_transient_all_thread_advisory_true`,
`test_rejection_rows_carry_the_prompt_era_stamp_of_the_style_the_service_runs`,
`test_immediate_opener_row_carries_the_prompt_era_stamp`,
`test_staged_commit_stamps_the_era_the_draft_was_generated_under`,
`test_rejection_store_failure_does_not_break_the_opener_error_branch`,
`test_rejection_store_failure_does_not_break_the_http_400_branch`,
`test_rejection_store_failure_does_not_break_the_transient_branch`,
`test_advisory_draft_keeps_billing_but_writes_no_opener_row_unless_committed`,
`test_discard_opener_persists_a_never_sent_row_with_the_generation_time_prompt_stamp`,
`test_discard_opener_returns_false_and_touches_nothing_when_pick_was_never_staged`,
`test_discard_opener_is_exactly_once_like_commit_opener`,
`test_discard_opener_and_commit_opener_are_mutually_exclusive`,
`test_discard_opener_store_failure_is_swallowed_and_returns_false`, and
`test_recent_openers_snapshot_carries_the_prompt_era_stamp`.

New in `tests/test_sqlite_store.py`: `test_sqlite_record_opener_persists_the_prompt_era_stamp`,
`test_sqlite_record_opener_leaves_the_prompt_stamp_null_when_the_caller_omits_it`,
`test_sqlite_record_opener_rejection_persists_the_prompt_era_stamp`,
`test_sqlite_migrates_legacy_openers_prompt_sha256_column`,
`test_sqlite_migrates_legacy_opener_rejections_prompt_sha256_column`,
`test_sqlite_opener_rejections_prompt_sha256_migration_reraises_unexpected_operational_errors`,
`test_sqlite_openers_alter_loop_reraises_unexpected_operational_errors`,
`test_sqlite_record_opener_rejection_persists_the_advisory_flag`,
`test_sqlite_record_opener_rejection_defaults_advisory_to_null_when_omitted`,
`test_sqlite_migrates_legacy_opener_rejections_advisory_column`, and
`test_sqlite_advisory_opener_run_rows_keep_the_prompt_stamp_out_of_the_fingerprint`.

New in `tests/test_bigquery_store.py`: `test_record_opener_buffers_the_prompt_era_stamp`,
`test_record_opener_buffers_a_null_prompt_stamp_when_the_caller_omits_it`,
`test_record_opener_rejection_buffers_the_prompt_era_stamp`,
`test_record_opener_rejection_signature_stays_aligned_with_the_store_protocol`,
`test_ensure_tables_declares_prompt_sha256_on_the_openers_create_table`,
`test_ensure_tables_runs_openers_prompt_sha256_migration`,
`test_ensure_tables_declares_prompt_sha256_on_the_opener_rejections_create_table`,
`test_ensure_tables_runs_opener_rejections_prompt_sha256_migration`,
`test_ensure_tables_declares_advisory_on_the_opener_rejections_create_table`,
`test_ensure_tables_runs_opener_rejections_advisory_migration`,
`test_record_opener_rejection_buffers_the_advisory_flag`,
`test_record_opener_rejection_buffers_a_null_advisory_when_the_caller_omits_it`, and
`test_bigquery_advisory_opener_run_rows_keep_the_prompt_stamp_out_of_the_fingerprint`.

New in `tests/test_worker.py`:
`test_worker_binds_opt_in_driver_to_the_service_prompt_era_before_opening`,
`test_worker_binds_none_prompt_era_when_there_is_no_opener_service`,
`test_worker_run_binds_the_opener_prompt_era_before_any_profile_is_read`,
`test_worker_never_calls_the_prompt_era_hook_on_a_driver_that_lacks_it`,
`test_auto_targeting_failure_discards_the_staged_opener_as_never_sent`,
`test_auto_post_send_paywall_discards_the_staged_opener_as_never_sent`,
`test_auto_generic_like_failure_discards_the_staged_opener_as_never_sent`,
`test_auto_never_sent_discard_when_the_model_names_no_item`,
`test_auto_never_sent_discard_on_run_budget_exhaustion`,
`test_auto_dislike_never_generates_or_discards_an_opener_draft`,
`test_auto_committed_like_decision_and_source_are_unchanged`,
`test_auto_never_sent_discard_store_failure_does_not_change_the_targeting_stop_outcome`, and
`test_auto_stop_during_opener_generation_never_starts_like_but_discards_the_draft`.

New in `tests/test_hinge_auto_presend_snapshot.py`:
`test_opener_evidence_rows_carry_the_bound_prompt_era` and
`test_opener_evidence_rows_carry_no_prompt_era_when_never_bound`.

Extended, not new, in `tests/test_training_worker.py`: the existing
`test_training_persists_verified_human_choice_only_after_hub_choice` (parametrized over
`command` in `["like", "dislike"]`) now also asserts that the Dislike branch calls
`discard_opener` exactly once with `decision="dislike"`, `decision_source="manual"`, the
profile's own `profile_id`, and a float `decision_created_at`, and that the Like branch never
calls it at all.

New file `tests/test_opener_corpus_report.py` (42 cases, all new): the grade detector's lexicon
boundary including verbatim reproductions of the sushi and matcha openers named in the
2026-09-06 addendum above and negative cases for hyperbole, negation, and a same-shape
question (`test_grade_detector_flags_the_sushi_opener_verbatim_from_the_addendum`,
`test_grade_detector_requires_the_complement_to_be_in_the_lexicon`,
`test_grade_detector_only_looks_at_the_first_beat`); the sentence/trigram/apostrophe/looks-like
helper functions (`test_split_sentences_keeps_abbreviation_periods_intact`,
`test_has_looks_like_is_word_bounded`); era-metrics aggregation and its unknown-bucket and
empty-bucket handling (`test_era_metrics_grade_rate_only_counts_the_first_beat`,
`test_build_era_metrics_groups_by_era_and_buckets_none_as_unknown`); cross-source dedupe and
era resolution/compare (`test_dedupe_openers_prefers_a_row_carrying_a_known_era`,
`test_resolve_era_raises_on_ambiguous_prefix`, `test_compare_eras_reports_signed_deltas`); the
recursive opener-string walk (`test_find_opener_strings_nested_inside_list_and_dict`); reading
each of the three sources including their missing-table and pre-redesign-schema paths
(`test_read_jsonl_openers_missing_debug_dir_reports_zero_not_an_error`,
`test_read_sqlite_openers_pre_redesign_schema_reports_missing_tables`,
`test_read_bigquery_openers_missing_table_reports_a_note_not_a_crash`); and the CLI's text,
json, and compare entry points (`test_main_text_mode_reports_sources_and_both_eras`,
`test_main_compare_unknown_token_fails_loudly`).

Full suite (`pytest -q -n auto`): 4960 -> 5035 green (5 skipped, unchanged), 236s wall clock.
The test pins above account for part of that growth; the remainder belongs to other in-flight
edits on this branch (drivers/hinge.py, ranker, worker, hub) landed concurrently and not
itemized here. Offline only -- no device, no live API call, no BigQuery read or write from this
change.

#### Addendum -- 2026-09-06 (d): three owner decisions and a pre registered prediction

Three owner decisions closing questions this document left open across the 2026-09-06 (a)-(c)
addenda above, followed by a falsifiable prediction registered now, before the batch that would
confirm or refute it exists, so neither outcome can be rationalized after the data arrives.
Pre-registration is itself the anti-circling mechanism the 2026-09-06 (c) addendum's audit called
for: a prediction written before the numbers exist cannot be moved once they do.

DECISION 1, the one permitted compliment: NARROWED, not retired. The 2026-09-06 addendum above
left this open as a residual. Rationale on record (`operation_love/opener/opener.py`'s own
comment ahead of `_SYSTEM`): the rule set measured at roughly 69% prohibition and mechanical
instruction against only 21% positive specification, with the corpus at 94.7% exactly two
sentences, 98.9% question final, and 82% of first beats deletable. The space of legal positive
moves is already collapsed, so retiring the one remaining permitted compliment would remove one
more legal positive move from a rule set already starved of them; narrowing it in place is the
correct direction instead. THE FIX ALREADY SHIPPED: the qualification that used to live only in
NO GRADING's later "narrows COMPLIMENT AS REMARK" sentence is now folded into COMPLIMENT AS
REMARK's own sentence, ahead of the technique it gates, in all three on-wire surfaces --
`config.yaml` `opener.style`, the compressed `_SYSTEM` mirror, and the response schema's
`opener` description -- by editing words inside the existing sentence rather than only appending
a new one, so every property already on wire (thing as sentence's subject, praise landing
sideways, understated over emphatic, shaping a compliment that occurs rather than requiring one)
survives the fold unchanged. This resolves the LOW severity finding from the 2026-09-06
adversarial review that COMPLIMENT AS REMARK and NO GRADING read as a rule and a later,
separate correction of it -- a minimal-thinking read of COMPLIMENT AS REMARK alone can no
longer satisfy it while still producing a grade. No example verdict vocabulary was added
anywhere, per the 2026-08-16 de-templating rule. New test pins:
`tests/test_config_yaml_real.py::test_shipped_opener_style_folds_qualification_into_compliment_as_remark`,
`tests/test_opener.py::test_system_prompt_folds_qualification_into_compliment_as_remark`, and
`tests/test_gemini_opener.py::test_response_schema_opener_description_folds_qualification_into_compliment_as_remark`.

DECISION 2, the orphaned advisory parameter: NEUTRALIZED, neither reconnected nor deleted. Read
against the live state of `operation_love/opener/service.py` and `operation_love/config.py`: no
production caller has passed `advisory=True` to `OpenerService.maybe_opener` since commit
ea6756e8 (2026-08-26) removed the last one, Hinge's Observe preview suggestion. `service.py`'s
own `maybe_opener` docstring now states this plainly ("DEAD IN PRODUCTION as of ... Training and
AUTO are the only production callers today and neither passes advisory=True"), and
`config.py` marks `OpenerCfg.advisory_max_attempts` and `advisory_deadline_s` the same way in
their field comments. What shipped is neither of the two options this document's residual framed
the choice as: the parameter, its validation in `config.py`'s `validate()`, and the shortened
retry budget it drives in `OpenerService` are all KEPT, not deleted, and not RECONNECTED to any
real preview caller either -- "kept, not deleted, so a preview caller that returns has a working,
tested budget waiting for it rather than one rebuilt from scratch" (`config.py`'s own comment).
The `opener_rejections.advisory` column the 2026-09-06 (c) addendum above shipped is KEPT either
way, in both the SQLite and BigQuery backends, regardless of which of the three outcomes this
decision landed on. Separately, a test now pins that no production caller passes `advisory=True`
for Training as well as AUTO: `tests/test_worker.py::test_training_loop_opener_call_does_not_pass_advisory`
is NEW alongside the pre-existing `test_auto_loop_like_call_does_not_pass_advisory`, because the
false "advisory-blind" explanation for the empty `opener_rejections` table -- already corrected
once, in the 2026-09-06 (c) addendum above, where the real cause was identified as the three
silent failure branches rather than either `if not advisory:` guard -- had stood on an
AUTO-only pin and let that explanation misdirect diagnosis for months before Training's own call
site was checked at all.

DECISION 3, the structural redesign: DEFERRED pending measurement. The 2026-09-06 (c) addendum's
own audit above found that of 8 prompt-only fixes recorded in this document, 0 are documented as
verified working by a post-ship re-measurement, 3 are documented as having failed, regressed, or
required escalation to a deterministic guard, and 5 shipped with no outcome measurement recorded
at all. A ninth blind structural change has no better prior than those eight. What ships instead
is the pre-registered prediction below, plus the era registry, so the next live batch produces the
measurement that has been missing every other time. Naming the restructure now, so the decision
is already made rather than re-litigated if the falsification criterion below is met: the prompt
gains an explicit positive specification of what an opener IS, stated as a property rather than a
template (the property to name is that the message must be evidence the sender actually looked,
which survives the 2026-08-16 de-templating rule because a property is not a template); an
explicit priority order between rules, so conflicts stop being patched one subordination clause at
a time the way NO GRADING's "narrows COMPLIMENT AS REMARK" sentence and, before it today, Decision
1's fold were each required to state by hand; and a merge of the four rules this document's
2026-09-06 addendum above already identified as jointly policing the same inference property --
CONVERSATIONAL VALUE RULE, INFORMATION GAIN TEST, MINIMUM INVENTION, and NEVER INVENT THE SENDER,
which that addendum's STRUCTURAL CAUSE paragraph found jointly leave the taste verdict as the only
predicate requiring zero facts about either party.

THE PRE REGISTERED PREDICTION. Baseline measured 2026-09-06 by `tools/opener_corpus_report.py`
on the 95 local drafts recovered from `data/hinge_debug/*/actions.jsonl` (verified by running the
tool: `sources.jsonl.unique` reports 95). Every one of these 95 drafts predates the driver-side
prompt era stamp the 2026-09-06 (c) addendum above shipped, so all 95 land in the tool's `unknown`
era bucket and mix however many real prompt eras are represented locally -- say so, rather than
presenting this as one era's number.

| metric | baseline (unknown era, n=95) |
| --- | --- |
| grade_rate | 12.6% (12/95) |
| two_sentence_rate | 94.7% (90/95) |
| question_final_rate | 98.9% (94/95) |
| single_sentence_rate | 5.3% (5/95) |
| trigram_diversity | 87.4% (83 distinct / 95) |
| apostrophe_rate | 2.1% (2/95) |
| looks_like_rate | 27.4% (26/95) |

Predictions for the first NO GRADING era batch of at least 40 drafts. A batch qualifies once
every draft in it was generated under a live `prompt_sha256` computed from on-wire text that
includes NO GRADING -- the 2026-09-06 addendum above and every prompt-changing fix stacked on top
of it since, including today's Decision 1 fold, each open a new era on top of the last by
changing on-wire bytes, and `prompt_stamp()` recomputes the live digest at runtime regardless of
whether `ops/prompt-eras.json` has been regenerated to name it; the registry is backfilled from
committed git history, and today's changes are not yet committed, so none of these newer eras
carry a human label in the registry yet:

- grade_rate FALLS below 5%. This is the primary endpoint.
- single_sentence_rate RISES above 15%, because NO GRADING's escape hatch routes explicitly to
  one question as the whole message. If it does not move, the escape hatch is not being taken and
  the rule is operating as a pure prohibition.
- two_sentence_rate falls below 85%, as the mirror of the above.
- trigram_diversity does NOT fall below 80%. This is the guard against the fix narrowing the
  space rather than redirecting it; a fall here means NO GRADING made the collapse worse.
- No prediction is made for looks_like_rate or apostrophe_rate, because the baseline for both is
  era mixed and therefore not attributable.

THE FALSIFICATION CRITERION, stated up front. If after at least 40 drafts in the new era
grade_rate is still at or above 10% AND single_sentence_rate is still below 8%, the
prohibition-only approach is FALSIFIED and the positive specification restructure named under
DECISION 3 above ships without further argument.

THE CAVEAT. `tools/opener_corpus_report.py`'s `grade_rate` is a deliberately conservative lower
bound over a fixed, explicit lexicon, and it measured 12.6% on this same 95-opener corpus where
an independent structural classification (the 2026-09-06 addendum above, which counted any first
beat whose predicate was an evaluative verdict rather than matching a lexicon) found roughly
45.3%. The prediction above is stated ENTIRELY in the tool's own definition, so it stays checkable
by one repeatable command (`python tools/opener_corpus_report.py --json`, or the equivalent text
report once a live batch exists); the two numbers must never be compared to each other.

THE ERA REGISTRY that shipped alongside this decision: `ops/prompt-eras.json`, generated by
`tools/backfill_prompt_eras.py`, which walks every commit that ever touched `config.yaml` or
`operation_love/opener/opener.py`, recomputes `prompt_stamp()`'s digest for each revision from
the historical bytes alone (AST-parsing the seven digest components rather than importing the
package, since historical `opener.py` revisions carry real package-relative imports that would
fail to exec standalone), and groups same-digest commits into one era. As shipped it records 11
known eras spanning 22 evaluated revisions plus 24 revisions in an explicit UNEVALUABLE bucket
(each with the concrete named reason it could not be evaluated, mostly a missing per-item crop
constant that postdates the 2026-08-12 opener redesign) out of 46 total revisions touching those
two files. `tools/opener_corpus_report.py` now reads that registry (`--eras-file`, default
`ops/prompt-eras.json`) to resolve every printed `prompt_sha256` to a human label, and adds two
flags read directly from the tool's own `_build_arg_parser()`: `--eras` lists every known era,
oldest first, with its label, date range, digest, and the rules that changed versus the previous
era, reading ONLY the eras file and never the corpus sources; `--rule NAME` is the direct answer
to "have we tried this before" -- it prints every known era NAME was on the wire in, when it
first and last appeared, and any measured metrics for those eras, reading both the eras file and
the corpus sources.

Full suite (`pytest -q -n auto`): 5035 -> 5124 passed, 5 skipped (unchanged), 214s wall clock.
Offline only -- no device, no live API call, no BigQuery read or write from this change. This
entry is documentation only: it makes three owner decisions the code and tests already carried
(Decision 1's fold, Decision 2's neutralization, and the new Training advisory pin all exist on
disk as of this addendum) and registers a prediction against no code change of its own; the test
delta above reflects the state of the tree at time of writing, not new work done by this entry.

#### Addendum -- 2026-09-06 (e): compliment as remark defers to the substitution test, and the record corrected

Two further adversarial-review findings against the 2026-09-06 (d) addendum above, fixed
together. Note the lettering: this document's own (b), (c), (d) addenda cover the PLAYFUL
HYPERBOLE/VARY THE SHAPE carve-outs, the durable-logging instrumentation, and three unrelated
owner decisions respectively -- none of them is the entry for the COMPLIMENT AS REMARK fold
itself, which shipped as code-comment-only history ahead of `_SYSTEM` in `opener.py` and inside
`config.yaml`'s own comment block, and is recorded in this document for the first time here.

MEDIUM. DECISION 1 in the 2026-09-06 (d) addendum above ("THE FIX ALREADY SHIPPED") folded a
qualification into COMPLIMENT AS REMARK's own sentence so it would stop reading as a rule
followed by a separate later correction. That fold closed one contradiction and opened another.
It gave COMPLIMENT AS REMARK its OWN portability test ("unable to survive being detached from
what was actually noticed" in `config.yaml` `opener.style` and the `_SYSTEM` mirror; an inline,
unlabeled "inseparable from a specific observation about that item" clause in the `_SCHEMA`
opener description), stated with NO exception. NO GRADING's SUBSTITUTION TEST, a few sentences
away in every one of those same three surfaces, explicitly carves out an unmistakably nonliteral
PLAYFUL HYPERBOLE (2026-09-06 (b) addendum above). An anchored hyperbolic compliment whose
predicate could transfer unchanged to a different woman's photo therefore passed NO GRADING (the
hyperbole exception) but failed COMPLIMENT AS REMARK's own restated test (no exception) --
opposite verdicts on the same sentence under minimal thinking, which is precisely the
rule-versus-later-correction failure shape the fold was meant to close.

THE FIX: COMPLIMENT AS REMARK no longer restates a second, subtly different portability test. It
now points at NO GRADING's SUBSTITUTION TEST directly, so the one test, and its one exception,
lives in exactly one place per surface and cannot drift out of sync with a restated copy again.
This removes words rather than adding a second hyperbole exception, which matters given this rule
set already runs roughly 10,000 tokens per request and every added word costs attention on the
weakest cascade models. Shipped in all three on-wire surfaces: `config.yaml` `opener.style` now
gates the technique on the praise "also passes NO GRADING's SUBSTITUTION TEST below"; `_SYSTEM`
reads "it must also pass NO GRADING's SUBSTITUTION TEST below"; the `_SCHEMA` opener description
reads "it must also pass the SUBSTITUTION TEST below," with a matching "SUBSTITUTION TEST:" label
now added at the schema's own grade-check sentence so the forward reference lands somewhere
named. No example verdict vocabulary was added anywhere, per the 2026-08-16 de-templating rule.
The three test pins DECISION 1 above named were extended in place (same three function names,
not renamed) to assert the new deferring wording and to add a MUTATION GUARD each, asserting the
removed exception-free portability phrasing is now absent so it cannot silently return:
`tests/test_config_yaml_real.py::test_shipped_opener_style_folds_qualification_into_compliment_as_remark`,
`tests/test_opener.py::test_system_prompt_folds_qualification_into_compliment_as_remark`, and
`tests/test_gemini_opener.py::test_response_schema_opener_description_folds_qualification_into_compliment_as_remark`.

LOW, CORRECTING THE RECORD. DECISION 1's paragraph above, and the code comment ahead of `_SYSTEM`
in `operation_love/opener/opener.py` that it quotes, both claimed that "every property already
on wire (thing as sentence's subject, praise landing sideways, understated over emphatic,
shaping a compliment that occurs rather than requiring one) survives the fold unchanged," which
reads as a claim of parity across all three on-wire surfaces. Checked against the actual text,
that parity claim is true only of `config.yaml`'s long `opener.style` form, where all four
properties were genuinely on wire both before and after the fold. It is not true of `_SYSTEM`,
which never carried "praise landing sideways ... in passing" or a separate "shapes a compliment
that occurs" sentence at any point in this saga -- only "thing as the sentence's subject" and
"understated over emphatic" were ever on that wire -- and it is not true of the `_SCHEMA` opener
description, which never carried any of the four. Nothing was LOST by the fold in either
compressed surface: these properties were never present there to lose, which is consistent with
the deliberate compression gap the 2026-09-05 register rewrite already documented (the schema
copy receives a compressed subset of properties, not the full set, and the two compressed
copies are not required to carry the same subset as each other or as the long form). The claim
was overstated, not the code. `opener.py`'s own comment ahead of `_SYSTEM` has been corrected in
place (permitted there, since it is not an append-only record); this paragraph is the
corresponding append-only correction for this document, which cannot edit DECISION 1's sentence
above in place.

Full suite (`pytest -q -n auto`): 5101 passed, 5 skipped, 208s wall clock, unchanged by this
entry. This addendum documents on-wire prompt text and code-comment corrections that were
already on disk (the SUBSTITUTION TEST deferral and the three extended test pins) and adds no
new test beyond the mutation guards folded into those three existing functions; it registers no
prediction and reads no live data. Offline only -- no device, no live API call, no BigQuery read
or write from this change.

#### Addendum -- 2026-09-06 (f): the advisory preview path is gone, not merely marked dead

A note on the letter before anything else: this is (f), not the (e) the brief that produced it
asked for verbatim, because the addendum immediately above already spent (e) on the COMPLIMENT
AS REMARK / SUBSTITUTION TEST fix. That entry's own material landed in the working tree before
this one was written up, even though the removal recorded here happened earlier in the same
day's work; one letter per entry, never reused, is this document's own rule, so this entry takes
the next open slot rather than collide with it. Section 5 below is the cross-reference this
entry owes that fix, not a second telling of it.

1. THE OVERRIDE. The 2026-09-06 (d) addendum's DECISION 2 above recorded the orphaned `advisory`
parameter as NEUTRALIZED -- kept, validated, and load-bearing in config, but reconnected to no
real caller. The owner overrode that decision after reading it: clean it up completely. (d)'s
account of decision 2 is superseded by this entry; (d) itself is not edited, per this document's
append-only rule, and its reasoning stays on the record as the decision that was actually
overridden rather than quietly abandoned.

2. WHAT WAS ACTUALLY REMOVED, read against the current state of the tree rather than assumed.
`OpenerService.maybe_opener` (`operation_love/opener/service.py`) no longer takes an `advisory`
parameter at all -- its signature is now exactly `(self, run_id, app, profile, *, items=None,
should_stop=None, stage=False)`, pinned by a new mutation-guard test named in full below. Every
branch that keyed off it went with it: the shortened `advisory_max_attempts`/`advisory_deadline_s`
retry budget, the `_register_transient_failure(..., request_stop=...)` parameter that let an
advisory streak latch without stopping the run, the `_apply_entropy_guard(..., deadline=...,
client_accepts_deadline=...)` plumbing that threaded a wall-clock cutoff through the entropy
guard's extra draw, and the `_accepts_generate_deadline()` duck-typing probe that decided
whether a client accepted that cutoff at all. `OpenerCfg.advisory_max_attempts` and
`advisory_deadline_s` (`operation_love/config.py`) are gone as fields, along with their three
bool-before-int/range/cross-field `validate()` checks each and the `_MAX_ADVISORY_DEADLINE_S`
ceiling constant and its derivation comment. `supervisor.py`'s `OpenerService(...)` construction
no longer forwards either config value. `bugreport.py`'s `_config_md` no longer prints
`opener.advisory_max_attempts`/`opener.advisory_deadline_s`, and `_recent_openers_md` no longer
reads an `advisory` bool off a hub entry to fall back into an `"advisory"` session-mode label.
Measured directly off `git diff` against HEAD: 301 removed lines mention the word "advisory"
across production and test code (107 in `service.py`, 104 in `tests/test_opener_service.py`, 44
in `config.py`, 18 in `tests/test_config.py`, 8 each in `tests/test_worker.py` and
`tests/test_supervisor.py`, 5 in `bugreport.py`, 4 in `tests/test_bugreport.py`, 2 in
`supervisor.py`, 1 in `tests/test_hub.py`) against only 13 added lines, and every one of those
13 is either the new mutation-guard test below or an untouched reference to the separate
`advisory_opener_run_rows` feature section 3 covers next -- none of it is a live use of the
parameter this entry removes. Wholesale, 33 test functions were deleted outright rather than
edited (27 in `tests/test_opener_service.py`, 5 in `tests/test_config.py`, 1 in
`tests/test_worker.py`, that last one `test_auto_loop_like_call_does_not_pass_advisory` --
its Training-side twin from (d), `test_training_loop_opener_call_does_not_pass_advisory`, is
not in the current tree under that name either, since there is no `advisory` keyword left for
either one to assert the absence of).
`tests/test_supervisor.py`, `tests/test_hub.py`, and `tests/test_bugreport.py` lost no whole
test function; their fixtures and assertions were trimmed in place instead. On the
`opener_rejections.advisory` column the 2026-09-06 (c) addendum shipped: it was added and
removed the same day, still uncommitted throughout, with the table at zero rows in every
backend for its entire existence (this machine's local `data/operation_love.db` predates even
the redesign's `opener_rejections` table per (c)'s own RESIDUALS, and nothing ever shipped to
BigQuery from an uncommitted branch) -- so removing it needed no drop migration and leaves no
schema history behind. `git diff` against `operation_love/ranker/store.py` and
`operation_love/ranker/bigquery_store.py` today shows only the unrelated `prompt_sha256`
column; the `advisory` column appears in neither the added nor the removed lines of either
file, which is exactly what "shipped and retracted inside one uncommitted working tree" looks
like from the outside.

3. THE SCOPE TRAP. "advisory" is not one feature in this codebase, it is two, and only one of
them died today. `operation_love/ranker/opener_retractions.py`, `tools/advisory_opener_cleanup.py`,
and `tests/test_advisory_opener_cleanup.py` implement advisory opener RETRACTION and cleanup
plans -- a fail-closed, append-only tool for correcting incorrectly persisted rows after the
fact, built around `Store.advisory_opener_run_rows()` -- and none of the three carry any diff
today (`git diff --stat` against all three reports nothing). A reader grepping this file for
"advisory" after this entry and finding `advisory_opener_run_rows` still very much alive should
not conclude the word was purged or that this entry is wrong; it should conclude there were
always two different things sharing one adjective, and this entry retired the dead one.

4. WHY IT WAS WORTH DOING. The 2026-09-06 (c) addendum above already corrected the specific
factual error: this document's own standing explanation for why `opener_rejections` sat empty
was that the store was "advisory-blind," and that explanation was FALSE, because Training and
AUTO are the only production callers of `maybe_opener` and neither has ever passed
`advisory=True` -- Observe's call site, the only one that ever did, was removed by commit
ea6756e8 on 2026-08-26. The false explanation stood, restated rather than re-checked, for close
to two weeks after the code it blamed had no live caller left, and the actual cause (gap 2 in
(c)'s own audit: three failure branches with no store call at all) was sitting unexamined the
whole time. A parameter that no longer does anything in production but still reads, in its own
docstrings and its own config comments, as though it does, is not neutral scenery -- it is a
standing invitation to blame the wrong thing, and it took this long to cost only a diagnosis
delay rather than something worse. That is the same failure this entire day's logging and
era-registry work exists to close off from the other direction: dead code that still looks live
corrupts the decision record itself, not merely the runtime.

5. THE PROMPT CONTRADICTION FIXED IN THE SAME PASS. The (e) addendum immediately above this one
already carries the full account: DECISION 1's fold in (d) gave COMPLIMENT AS REMARK its own
portability test with no exception, while NO GRADING's SUBSTITUTION TEST a few sentences away
carves out an unmistakably nonliteral PLAYFUL HYPERBOLE, so the two rules disagreed on an
anchored hyperbolic compliment under minimal thinking; the fix makes COMPLIMENT AS REMARK point
at the SUBSTITUTION TEST directly instead of restating a second, subtly different version of it,
in all three on-wire surfaces (verified again here directly against the current text: `config.yaml`
line 1111 reads "praise also passes NO GRADING's SUBSTITUTION TEST below", `_SYSTEM` in
`operation_love/opener/opener.py` reads "it must also pass NO GRADING's SUBSTITUTION TEST
below", and the `_SCHEMA` opener description at line 244 carries a matching "SUBSTITUTION TEST:"
label). The LOW finding on which folded properties survived is also settled by (e): parity
across all three surfaces held only for `config.yaml`'s long form; `_SYSTEM` never carried
"praise landing sideways ... in passing" or a separate "shapes a compliment that occurs"
sentence, and `_SCHEMA` never carried any of the four properties, at any point in this saga, so
nothing was lost from either compressed copy by today's fold -- it was never there to lose. This
entry does not re-litigate either finding; it records that both are already closed, by the entry
right above it, not by this one.

6. THE ERA REGISTRY NOW COVERS THE WORKING TREE. `tools/backfill_prompt_eras.py` gained a second
mode alongside the committed-history walk (d) described: `evaluate_working_tree()` reads the
CURRENT `config.yaml` and `operation_love/opener/opener.py` straight off disk and recomputes
`prompt_stamp()`'s digest from them, then attaches the result to the registry as a separate
top-level `working_tree` key -- never appended to the `eras` list, and unmistakably marked
`"provisional": true` with `"commit": null` and a label that leads with `UNCOMMITTED WORKING
TREE -- PROVISIONAL, not a shipped era`. The reason to keep it separate rather than folding it
into `eras` is stated in the module's own docstring and holds up against a live run: a rule
added only in the working tree looked, before this, indistinguishable from one that had never
been tried, which is precisely backwards for a registry whose whole purpose is answering "have
we tried this before." Run live just now, `python tools/opener_corpus_report.py --rule
"SUBSTITUTION TEST"` correctly reports the rule absent from all 11 known COMMITTED eras and
present in the provisional working tree, with its metrics line reading "no local openers
recorded under this era digest" rather than anything that could be mistaken for a measurement --
the exact behaviour (d) promised: `--rule` answers correctly for a rule that has not shipped
yet, without ever reporting it as measured. Committed-history coverage is unchanged by any of
today's work, because it depends only on git history and none of today's changes are committed:
22 of the 46 total revisions across both files are evaluated (just under half, spanning 11
known eras), and the other 24 are UNEVALUABLE, each for the same concrete, stated reason --
`missing constant(s) at this revision: ['_ITEM_PREAMBLE', '_ITEM_PREAMBLE_CONTEXT',
'_ITEM_LABEL', '_CONTEXT_LABEL']` -- because every one of those 24 revisions predates the
2026-08-12 item-crop redesign that introduced those four constants into `prompt_stamp()`'s
digest. That is a fact about what the prompt looked like at each of those revisions, not a
limitation of the tool: a digest that hashes seven components cannot be computed at a revision
where four of them did not exist yet, so those 24 revisions have no comparable digest by
construction, and the registry says so by name rather than silently omitting them or guessing.

7. RESIDUALS. The three from (d) still stand exactly as stated there: the pre-registered
prediction is UNTESTED pending a live batch of at least 40 NO GRADING-era drafts; there is still
no reply, match, or outcome signal anywhere in this system, so every metric measures what the
prompt produces and never what performs; and every `openers` row written before today remains
survivorship biased, because only sent (for AUTO, liked) drafts were ever committed to it before
`discard_opener` existed. One more, found while verifying this entry rather than while doing the
removal: `GeminiOpener.generate()` in `operation_love/opener/opener.py` still takes an optional
`deadline` keyword, and `OpenerDeadlineExceeded` is still a live exception class, but neither has
a production caller left -- `service.py` no longer passes `deadline=` to `.generate()` anywhere,
confirmed by grepping the file (zero hits) and by the fact that every remaining `deadline=` call
site in the whole tree is inside `tests/test_gemini_opener.py`, exercising `GeminiOpener` in
isolation. Both the parameter's own docstring ("used by Observe's optional advisory path") and
`OpenerDeadlineExceeded`'s ("`OpenerService.maybe_opener` therefore handles it only on its
advisory path") still assert a caller relationship that `maybe_opener` no longer has, now that
its `advisory` parameter and every branch keyed off it are gone. This is the identical shape of
defect item 4 above just finished explaining the cost of: code that keeps describing itself as
called by something that no longer calls it. It was out of scope for this pass -- the owner's
instruction named the config knobs, the supervisor forwarding, the bugreport diagnostic, and the
`advisory` column, not `GeminiOpener`'s own deadline plumbing -- so it is left here as a found,
not fixed, residual rather than pulled in under this entry's own scope.

TEST PINS BY NAME. New: `tests/test_opener_service.py::test_maybe_opener_has_no_advisory_parameter`,
a mutation guard asserting `"advisory" not in inspect.signature(OpenerService.maybe_opener).parameters`
and pinning the full remaining parameter set (`{"self", "run_id", "app", "profile", "items",
"should_stop", "stage"}`), so a future re-add of the parameter or any of its dead plumbing fails
this test rather than silently landing with no caller to exercise it. No other new test was
added by this removal: `tests/test_config.py`, `tests/test_supervisor.py`, `tests/test_hub.py`,
and `tests/test_bugreport.py` all had their existing tests' fixtures and assertions trimmed in
place instead of gaining a dedicated absence pin, which this entry records rather than papers
over -- their coverage of the removal is exactly "the existing test still passes with the field
gone," nothing stronger.

Full suite (`pytest -q -n auto`), run just now: 5101 passed, 5 skipped, 210.67s wall clock. This
is the same figure the (e) addendum above already reported as "unchanged by this entry" --
correctly, since (e) added no tests of its own -- but that figure already reflects this entry's
removal, which is why (e)'s number and this one match despite this entry deleting 33 test
functions. The (d) addendum above reported 5124 passed before this removal landed; the arithmetic
this entry owes the record is that 33 deletions against 1 addition (the mutation guard above) is
a net of -32, while the observed drop from 5124 to 5101 is only -23 -- the remaining +9 belongs
to other concurrent edits on this branch not itemized in this entry, in keeping with this
document's standing disclaimer (see the 2026-09-05 (b) addendum above) that same-day parallel
work is not always itemized per entry. Offline only -- no device, no live API call, no BigQuery
read or write from this change.

#### Addendum -- 2026-09-06 (g): compliment as remark's magnitude clause was still exception-free

A fourth adversarial review of the same day's COMPLIMENT AS REMARK work, this time walking a
concrete opener through every rule by hand, found that the (d) addendum above closed only HALF
of the contradiction it named, and the (e) correction above it verified the wrong half stayed
closed rather than re-checking the other one.

THE FINDING. COMPLIMENT AS REMARK's one permitted compliment has carried two independent
requirements since the 2026-09-05 register rewrite introduced it, not one: a PORTABILITY
requirement (does the predicate survive being swapped onto a different woman's different
photo) and a separate MAGNITUDE requirement (does it stay understated rather than turning
earnest and emphatic). (d) fixed only the portability half, by making COMPLIMENT AS REMARK
point at NO GRADING's SUBSTITUTION TEST directly instead of restating its own copy of that
test -- and that test already carves out an unmistakably nonliteral PLAYFUL HYPERBOLE, so
portability is genuinely settled. The magnitude requirement was never touched by (d) and never
carried an exception of its own: the affirmative clause read "stays understated rather than
emphatic" and the disqualifier read "or that turns earnest and emphatic," both unconditional. An
unmistakably nonliteral PLAYFUL HYPERBOLE is, by its own definition, emphatic -- exaggeration
that is not emphatic is not hyperbole. So for a strong hyperbolic compliment, PLAYFUL HYPERBOLE
and NO GRADING's SUBSTITUTION TEST said allowed (via (d)'s fix) while COMPLIMENT AS REMARK's own
magnitude clause said disallowed, on the same sentence, under the same minimal-thinking read (d)
diagnosed the first time. (e)'s correction above re-verified only which of the four narrated
properties survived the (c) fold across the three surfaces; it never re-checked whether the
magnitude clause itself was internally consistent with the exception (d) had just added
elsewhere, which is how this half survived three review rounds (the original (c)/(d) pass, and
(e)'s own correction pass) before a fourth review walked a concrete strong-hyperbole opener
through the rule text by hand and hit the contradiction directly.

THE FIX. Do not delete the magnitude requirement: earnest, emphatic praise reading as grading
her or auditioning for her approval is a real, distinct property COMPLIMENT AS REMARK has
carried since 2026-09-05, and (e)'s own correction above already flagged silent property loss as
its own defect for this exact rule. Instead subordinate magnitude to the same exception
portability already has, stated once rather than once per clause. `config.yaml` opener.style now
reads (in relevant part): "...lands sideways the way a friend would mention it in passing, and,
save for that same test's PLAYFUL HYPERBOLE exception, stays understated rather than emphatic.
Praise that fails either requirement reads as grading her or auditioning for her approval..." --
dropping the old disqualifier's restated "or that turns earnest and emphatic" (an unexcepted
echo of the same requirement the affirmative clause already states, and exactly the kind of
second copy (d) warned drifts out of sync) rather than adding a second hyperbole carve-out next
to it. `_SYSTEM` in `operation_love/opener/opener.py` carries the same fix compressed to a single
inserted clause: "...it must also pass NO GRADING's SUBSTITUTION TEST below, whose PLAYFUL
HYPERBOLE exception covers tone too, phrased with the thing as the sentence's subject,
understated over emphatic..." Both edits net FEWER words than before, not more: the fix is a
subordinating cross-reference to an exception already defined at NO GRADING's SUBSTITUTION TEST,
not a restated definition of PLAYFUL HYPERBOLE itself. The `_SCHEMA` opener description is
unchanged, because (per the 2026-09-06 (e) correction above, re-verified again here directly
against the current text) it never carried the magnitude clause at all -- there is nothing to
except there.

THE PROOF, walked end to end against the current wire text. (1) MILD hyperbolic compliment --
"that trail view alone could make anyone fall for mountains" said as an aside under a hiking
photo. PLAYFUL HYPERBOLE (config.yaml): unmistakably nonliteral, visible anchor immediate --
allowed. NO GRADING's SUBSTITUTION TEST: the predicate would transfer to a different woman's
different mountain photo unchanged, so on its own it is a grade, EXCEPT that the same test
explicitly carves out an unmistakably nonliteral PLAYFUL HYPERBOLE -- passes. COMPLIMENT AS
REMARK: passes NO GRADING's SUBSTITUTION TEST (above), lands sideways as an aside, and for the
magnitude clause -- "save for that same test's PLAYFUL HYPERBOLE exception, stays understated
rather than emphatic" -- the same exception applies, so a mildly emphatic hyperbole is not
disqualified on magnitude either. Consistent allow across all three. (2) STRONG, clearly emphatic
hyperbolic compliment -- "that view alone could make anyone fall in love with mountains forever,"
the case that broke under the pre-fix text. PLAYFUL HYPERBOLE: still unmistakably nonliteral with
an immediate visible anchor -- allowed, exactly as before. SUBSTITUTION TEST: the predicate is
even more portable than case (1) (it would fit any dramatic vista unchanged), so without the
exception it reads as a stronger grade -- but the exception does not scale down as the hyperbole
gets stronger; it still applies in full, so this still passes. COMPLIMENT AS REMARK: the same
chain as (1) -- passes the test, lands sideways, and now explicitly "save for that same test's
PLAYFUL HYPERBOLE exception, stays understated rather than emphatic" waives the magnitude
requirement for exactly this case, because it is emphatic only in the excepted, nonliteral-
hyperbole sense the exception names. Consistent allow, matching (1) -- this is the case that used
to fail before this fix, and now does not. (3) Plain literal grade, no hyperbole -- "your smile is
a 10 out of 10." PLAYFUL HYPERBOLE does not apply at all: this reads as a literal claim about how
good something is, not an unmistakably nonliteral exaggeration, so no exception is even in play.
SUBSTITUTION TEST: the predicate would fit unchanged under any other woman's smile -- it is a
grade rather than an observation, and it fails, with no PLAYFUL HYPERBOLE exception available to
rescue it. COMPLIMENT AS REMARK: fails the SUBSTITUTION TEST outright, so it fails "either
requirement" and reads as grading her -- correctly and still rejected. No hole opened by
widening the exception's reach to magnitude: a plain grade never reaches PLAYFUL HYPERBOLE's
gate in the first place.

TEST PINS BY NAME, mirroring the three on-wire surfaces this touches: `tests/test_config_yaml_real.py::test_shipped_opener_style_folds_qualification_into_compliment_as_remark`
(updated in place, since it already pinned the (c)/(d) fold this entry extends rather than
replaces) now additionally asserts the new subordinating clause is present, the old unexcepted
disqualifier restatement is gone, and the affirmative magnitude clause survives byte for byte.
`tests/test_opener.py::test_system_prompt_folds_qualification_into_compliment_as_remark` pins
the compressed `_SYSTEM` mirror the same way. `tests/test_gemini_opener.py` gains one new
mutation-checkable assertion on the existing schema test recording that `_SCHEMA`'s opener
description still carries neither "understated" nor "emphatic" -- a guard against this fix's
cross-reference wording drifting onto the one surface that was never supposed to carry it,
rather than a pin of new wording that does not exist there. Every new assertion was mutation
checked: removing what it pins and re-running the targeted test made it fail, then the text was
restored.

RESIDUALS. This entry does not reopen (d)'s or (e)'s own scope: the portability half of the
contradiction stays fixed exactly as (d) left it, and the property-survival audit (e) ran stays
correct as re-verified here. What this entry adds is the magnitude half neither prior pass
checked. No deterministic guard, no new retry reason, and no change to the retry hint block
(copy 4 of 4) -- prompt text only, per the same prompt-first ordering the 2026-09-05 and
2026-09-06 (b) addenda above already established for this rule family. No example verdict
vocabulary was added anywhere, per the 2026-08-16 de-templating rule. Offline only -- no device,
no live API call, no BigQuery read or write from this change.

#### Addendum -- 2026-09-06 (h): the two smaller fixes owed to this record, and a live hazard found while writing it

A note on the letter, matching this document's own precedent for the same situation (see the
(f) addendum above): this entry was commissioned to record itself as (f), "the contradiction
moved one clause over" -- COMPLIMENT AS REMARK's untouched MAGNITUDE requirement ("stays
understated rather than emphatic" / "turns earnest and emphatic") disagreeing with PLAYFUL
HYPERBOLE and NO GRADING's SUBSTITUTION TEST for a strong hyperbolic compliment, the exact same
relocate-the-defect shape as the day's original "elite move" incident. By the time this entry
was written, letter (f) was already spent on the advisory-preview-removal entry above, and a
separate, concurrent adversarial-review pass had already spent (g) on exactly the finding this
entry was asked to record -- verified directly against `config.yaml` and `operation_love/opener/
opener.py` at the time this entry checked, that (g) account was accurate: (d) above had closed
only the portability half of COMPLIMENT AS REMARK's two independent requirements, and (g) closed
the magnitude half the same way, by subordinating it to the same PLAYFUL HYPERBOLE exception
rather than restating a second carve-out. This entry does not repeat that finding a third time.
It takes the next open letter and covers what (g) did not: a live hazard this entry's own
verification pass turned up in the same file (g) describes, and the two smaller, in-scope fixes
this entry was also asked to record.

1. A HAZARD OBSERVED WHILE VERIFYING (g), SELF-RESOLVED BEFORE THIS ENTRY SHIPPED. `config.yaml`
is outside the three files this entry owns (`operation_love/bugreport.py`, its test file, and
this document), so this entry never touched it -- but it is named here because it briefly made
(g)'s account above false, and any reader who checks this document against the live tree deserves
the full timeline rather than a silently-stale warning. Mid-way through verifying (g) against the
current wire text, `config.yaml`'s working-tree copy was found byte-for-byte identical to `git
show HEAD:config.yaml` (matching MD5 `4d6f021bba53fcc9913aba697b9856ee`, confirmed twice a few
minutes apart) and containing no occurrence of "COMPLIMENT AS REMARK" at all -- meaning every
uncommitted `config.yaml` edit from today, the entire COMPLIMENT AS REMARK / NO GRADING /
SUBSTITUTION TEST rule set this whole day's addenda chain describes and not merely (g)'s own
magnitude-clause fix, had been discarded from the working tree by some action this entry did not
take and could not identify from here (no `git stash`, `checkout`, or `reset` was run by this
entry; the ground rule this document and its sibling tasks operate under forbids exactly those).
`operation_love/opener/opener.py` was unaffected throughout: its `_SYSTEM` constant carried (g)'s
fix verbatim the whole time. By the time this entry was finished, `config.yaml` had been restored
by whatever process owns it in this session -- re-checked just now, its MD5 no longer matches
HEAD, `grep -c "COMPLIMENT AS REMARK" config.yaml` reports 7, and it again carries (g)'s exact
magnitude-clause text ("...save for that same test's PLAYFUL HYPERBOLE exception, stays
understated rather than emphatic..."). So (g)'s account is accurate again as of this entry, and
was never wrong -- there was a window, entirely within today's session and closed before this
entry shipped, where the file on disk did not match it. This is recorded rather than left
unmentioned because a silent gap in a document that other entries and code comments cite as the
authority on "what actually shipped" is itself worth one line of history.

2. THE ERA REGISTRY'S JSON SIDE TABLE OMITTED THE WORKING TREE ERA IT MOST NEEDED TO MARK.
`tools/opener_corpus_report.py`'s `build_report()` emits one side table, `era_labels`
(digest -> human label), so every text-mode formatter and the `--json` output resolve a
prompt_sha256 digest the same way and can never silently disagree. Before today's fix, that
table was built by walking the registry's own `eras`/`order` (a committed-history structure)
rather than by walking `by_era`'s actual keys, so a digest that was never committed to `eras` at
all but happened to match the registry's provisional `working_tree` entry -- e.g. real corpus
rows recorded locally under the current uncommitted prompt edit, before that edit is ever
committed -- could show up in `by_era` with a genuine measured `n`, and carry NO entry in
`era_labels` whatsoever. That is precisely the ambiguity the provisional marker exists to
prevent: a JSON consumer would see real numbers attached to a digest it cannot label as shipped
or provisional. Every text-mode formatter already called `resolve_era_label()`, which already
knew to check `registry.working_tree` for exactly this case, so this was a JSON-only gap, not a
measurement gap. THE FIX: `era_labels` is now built from `by_era`'s own keys (excluding the
"unknown"/"ALL" pseudo-buckets, which are not real digests), resolved through
`resolve_era_label()` -- the same function the text formatters use -- plus the registry's
`working_tree` digest added explicitly even on a run whose corpus happens to contain zero rows
under that exact digest yet, so the provisional era is always labeled once the registry names it,
never only when a run happens to have measured it. Test pins:
`tests/test_opener_corpus_report.py::test_build_report_era_labels_covers_a_working_tree_digest_with_real_rows`
(the bug this closes: a working-tree digest with real `by_era` rows now also appears in
`era_labels`, correctly labeled `UNCOMMITTED WORKING TREE`, without narrowing the table away from
the committed eras a run also measured) and
`tests/test_opener_corpus_report.py::test_build_report_era_labels_includes_working_tree_digest_even_with_zero_rows`
(the provisional entry is present even at zero measured rows). Both pass as of this entry
(`pytest -q tests/test_opener_corpus_report.py -k era_labels`: 5 passed).

3. THE LAST "ADVISORY" REMNANT IN `bugreport.py`. `_recent_openers_md`'s per-app mode line
validated an entry's `session_mode` against the set literal `{"advisory", "training", "auto"}`
before falling back to the last-run status snapshot for anything unrecognized. Nothing in the
current codebase can produce `"advisory"` as a `session_mode` value: the only two writers of
this dict's `session_mode` key, `OpenerService.maybe_opener` and `.commit_opener`
(`operation_love/opener/service.py`), write only the literals `"auto"` and `"training"`
(confirmed by grepping every `session_mode` assignment in the repository). The comment this
entry replaced was itself a fossil of an era when `advisory` was a real boolean-derived value
here (`git log -S` on this file surfaces the exact prior line, `mode_note = "advisory" if
advisory else "auto"`, from before that flag's fallback branch was dropped in favour of today's
`legacy_app_modes` lookup) -- the set literal was never updated when that branch went, so
`"advisory"` sat in the recognized set with nothing left that could ever produce it. The second
half of the brief this entry checked, whether a HISTORICAL persisted run log could still carry
that string into this function: no. `_recent_openers_md`'s only data route is
`HubState.recent_openers()`, which returns either the live `OpenerService`'s in-memory ring
buffer or `HubState._completed_openers`, a detached in-process snapshot taken at shutdown --
both are memory-only for the life of one hub process and are never serialized to or reloaded
from a persisted run log (`_completed_openers` starts as `[]` on every `HubState`
construction). The separate function that DOES read persisted `actions.jsonl` rows for opener
evidence, `_latest_auto_opener_evidence_md`, already normalizes any non-`"training"` value
(missing key or otherwise) to `"auto"` by a plain ternary rather than a recognized-set check, so
it was never exposed to this same defect regardless. DECISION: drop `"advisory"` from the set
literal rather than keep it with a compatibility comment, since no path, live or historical, can
ever hand this function that value. THE FIX: the check is now
`if explicit_mode not in {"training", "auto"}:`, with a comment stating this reasoning inline so
the choice reads as deliberate rather than an oversight to whoever next greps this file for
"advisory". New test:
`tests/test_bugreport.py::test_recent_openers_section_treats_stray_advisory_session_mode_as_unrecognized`,
asserting a stray/corrupted `"advisory"` entry renders as `auto` (the `legacy_app_modes`
fallback), never literally. Mutation checked: reverting the set literal back to include
`"advisory"` made this new test fail (it asserted `" · advisory · "` would render, and it did);
restoring the fix made it pass again. `tests/test_bugreport.py` in full: 227 passed.

A suite run taken mid-way through this entry, while finding 1's `config.yaml` gap was still open,
showed exactly the 13 failures that gap predicts: `operation_love/config.py`'s `OpenerCfg` no
longer defines `advisory_max_attempts`/`advisory_deadline_s` as fields (per the (f) addendum's
own removal above), while `config.yaml`'s reverted-to-HEAD copy still declared both keys under
`opener:`, so `Config._section` raised `ValueError: ... unexpected keyword argument
'advisory_max_attempts'` the instant any test constructed a config from the file on disk --
exactly `tests/test_hub.py` and `tests/test_supervisor.py`'s shared fixture path (13 tests, 5092
passed, 5 skipped otherwise, 307.86s). That run is not the number this entry closes with,
because it measured a cross-file desync finding 1 already explains and that has since cleared,
not a defect in `config.py` or `bugreport.py`. Full suite (`pytest -q -n auto`), run again after
`config.yaml`'s restoration confirmed in finding 1: 5105 passed, 5 skipped, 0 failed, 419.28s wall
clock (slower than the mid-entry run only from other concurrent sessions sharing this machine's
CPU at the same time, per `ps`, not from anything this entry changed). The two fixes this entry
itself records are independently verified by the targeted runs cited in 2 and 3 above, both
green, and are included in this final full-suite number as well. Offline only -- no device, no
live API call, no BigQuery read or write from this entry.

#### Addendum -- 2026-09-06 (h): the contradiction relocated a third time, so the exception was hoisted

Addendum (g) subordinated COMPLIMENT AS REMARK's MAGNITUDE clause to NO GRADING's SUBSTITUTION
TEST and its PLAYFUL HYPERBOLE exception. An adversarial clause-by-clause review then found the
same contradiction alive in a THIRD clause of the same rule, `lands sideways the way a friend
would mention it in passing`, which still carried no exception of its own. The disqualifier
sentence had also gone stale: it said praise failing `either requirement` while the sentence
above it listed THREE coordinate conditions.

THIS IS THE LESSON OF THE ENTIRE DAY, and it is the same mechanism three times over. The original
defect was COMPLIMENT AS REMARK producing "elite move" by RELOCATING a verdict from her onto the
thing and being obeyed perfectly. Fix (e) relocated the contradiction from the portability clause
into the magnitude clause. Fix (g) relocated it from the magnitude clause into the placement
clause. A fix that moves a property rather than governing the whole rule gets complied with and
preserves the failure. Three separate same-day adversarial rounds were each explicitly briefed to
hunt this exact failure shape, and each caught one clause and missed the next.

WHAT SHIPS. The exception is now HOISTED ONCE and governs every condition of COMPLIMENT AS REMARK
together, rather than being bolted onto whichever clause was last reported. The three conditions
are enumerated explicitly and the disqualifier counts all three. The `lands sideways` case is
resolved deliberately in the NON-PERMISSIVE direction and says so on the wire: that condition is
about PLACEMENT rather than volume, so it survives the exception and an exaggeration still has to
be an aside rather than the point of the message. Shipped in config.yaml opener.style and in
_SYSTEM; _SCHEMA never carried the magnitude or placement clauses, and its guard assertions pin
that they stay absent there.

THE STANDING RULE, added to the test docstrings so it is not lost: after subordinating one clause
of a rule to an exception, re-walk EVERY other clause of that same rule against that same
exception before calling it closed. A per-clause carve-out is a defect waiting to relocate; one
exception stated once for the whole rule is the only shape that terminates.

ALSO IN THIS PASS. The last reader of the removed Observe advisory-suggestion flow was deleted
from `bugreport.py`'s `_hub_guidance_md`: its `opener_warning` / `opener_pending` /
`opener_suggestion` branches rendered text like "advisory suggestion is currently shown", but
nothing has written any of those keys into a hub app dict since commit ea6756e8 (2026-08-26), so
all three were unreachable in production. A hand-built test fixture was the only thing keeping
them alive; that test now pins their ABSENCE instead. This is the same class of harm as the
"advisory-blind" explanation corrected in (c): a dead diagnostic that reads as live.

INCIDENT, recorded because it affected this repository's working tree. During this pass a
subagent ran `git checkout -- config.yaml`, which is forbidden by the working agreement precisely
because it discards uncommitted work, and it discarded the uncommitted register-rewrite content
in that file before reconstructing it. The reconstruction was subsequently VERIFIED rather than
trusted: 34 distinct content fragments spanning the whole `opener.style` block were confirmed
present (10 of them exist only in the uncommitted work and not at HEAD), every non-opener section
of config.yaml was confirmed byte-identical to HEAD, the file parses, and the config-pinning test
file passes. RESIDUAL RISK, stated rather than smoothed over: an uncommitted change that was
never observed earlier in the session and is not covered by a test or by those 34 fragments could
not be detected by any of these checks, because no copy of the pre-checkout bytes exists. The
mitigation for next time is to commit before delegating edits to parallel agents.

Suite 5105 passed, 5 skipped. New pins: the (h) assertions inside
tests/test_config_yaml_real.py::test_shipped_opener_style_folds_qualification_into_compliment_as_remark
(exception hoisted, both dimensions covered, placement survives, disqualifier counts three) and
tests/test_opener.py::test_system_prompt_folds_qualification_into_compliment_as_remark (same
invariants in the compressed mirror, plus a guard that (g)'s superseded wording is gone), and
tests/test_bugreport.py::test_status_section_surfaces_current_training_guidance_safely now pinning
the absence of the dead advisory-suggestion guidance. Both prompt assertions were mutation tested
by reverting each surface in turn and confirming the corresponding test fails, then restoring
byte-identically. Offline only: no device, no live API call, no BigQuery read or write.

#### Addendum -- 2026-09-06 (i): what the prompt produces, and for the first time what it did

TRIGGER. Two gaps named at the close of the 2026-09-06 work above. First, the 2026-09-06 (d)
addendum registered a falsifiable prediction for the first NO GRADING era batch of at least 40
drafts, but a pre-registered prediction is only as good as the batch that checks it, and no live
batch existed yet -- the prediction sat untested. Second, the 2026-09-06 (c) addendum's own
RESIDUALS said it plainly: "There is still NO reply, match, or outcome signal anywhere in this
system... Adding an outcome signal remains the single largest missing input to any future
quality claim." Both gaps are closed by this entry, in that order.

THE TWO BLOCKERS FOUND, which changed the shape of both fixes.

First, the numbered item crops actually sent to Gemini (her name as text, the numbered item
crops in order, the unnumbered context crops, the truncation flag -- `opener.opener.ItemRequest`)
were never persisted anywhere. `data/hinge_debug/<run_id>/` holds navigation and verification
screenshots, not the request INPUTS. So none of the 95 historical drafts recovered from that
directory could ever be replayed through a revised prompt: every prompt revision required a
fresh live batch to measure, which is the mechanical reason nothing in this document was ever
measured before/after until tools/opener_corpus_report.py existed, and even that tool could only
ever measure what a live batch happened to produce, never re-run an old batch through a new
prompt.

Second, `openers.profile_id` is `uuid.uuid4().hex` (`worker.py`'s `lineage_profile_id` /
`profile_id`), minted fresh every time a profile is shown, purely to bind one committed opener
row to the ONE decision that sent it. It carries no information about WHO the card was. An
outcome learned days later -- she matched, she replied, she never responded -- had no way to
find its way back to the opener that earned it, because the id that "wrote" that profile is
already forgotten the moment the card is gone. Any outcome table keyed on `profile_id` would
have been useless by construction: it could never join to anything.

WHAT SHIPS FOR REPLAY. `operation_love/opener/replay_corpus.py`: an on-disk format, one
directory per captured request named after a content-derived `replay_id` (her name, the
truncation flag, and every item/context crop's own digest, in order -- so writing the identical
capture twice is a no-op, never a duplicate copy of someone's photos), holding the crops as
plain image files plus a `manifest.json`. It is local-only by default: `DEFAULT_CORPUS_DIR` sits
under `data/`, which this project's `.gitignore` already excludes wholesale, and the module has
no dependency on any store or network client at all. `write_replay_capture` never raises (every
failure comes back as a `ReplayWriteResult`, matching this project's telemetry-must-be-best-
effort rule), so a future call site on the live opener path can call it unconditionally. Gated by
a new config flag, `opener.replay_corpus_enabled` (`config.py`'s `OpenerCfg`, `config.yaml`),
DEFAULT FALSE: this writes REAL PEOPLE'S PHOTOS to local disk on every profile an opener is
generated for, so turning it on is the owner's explicit, opt-in choice, never something that
rides along with the opener pipeline's own on/off switch. The flag is wired end to end already,
not merely defined: `supervisor.py` passes `replay_corpus_dir=DEFAULT_CORPUS_DIR` into
`OpenerService` exactly when `cfg.opener.replay_corpus_enabled` is true (`None` otherwise), and
`OpenerService.maybe_opener` calls the new `_capture_replay_corpus` once per profile, before any
generation attempt, whenever a `replay_corpus_dir` is set -- so flipping the config flag alone is
enough to start capturing on the very next live profile, no driver change required. Wrapped so a
capture failure can never affect generation, decision, send, or refusal. `tools/opener_replay.py`
is the replay harness itself: it
re-generates an opener for each captured request through WHATEVER `config.yaml` currently
configures (never the prompt that was live at capture time) and writes every result to the local
SQLite store under the CURRENT `prompt_sha256`, tagged `decision="synthetic_replay"` so it can
never be miscounted as a real human "sent" decision by any consumer that groups on that column.
DRY RUN IS THE DEFAULT and needs no `GEMINI_API_KEY` at all -- it builds the exact request
payload locally and reports its size; `--live` plus `--yes` or an interactive confirmation is
required to spend real free-tier quota, mirroring `tools/gemini_model_probe.py`'s own COST
TRANSPARENCY gate. THE CONSEQUENCE, stated plainly: from the first run with capture enabled, one
live batch buys unlimited offline prompt iterations, and the 2026-09-06 (d) pre-registered
prediction becomes checkable without further device time -- the exact gap this entry's TRIGGER
names first.

WHAT SHIPS FOR OUTCOMES. `operation_love/ranker/profile_key.py`: the stable, cross-time
attribution key that lets an outcome observed days later find its opener. It hashes (SHA-256,
never stores) `drivers.item_identity.ProfileIdentity.fingerprint` -- a fingerprint that already
exists for an unrelated reason (the navigation safety gate that checks whether the screen still
shows the profile an index was built from) and is already, in that module's own words, "loggable
and storable anywhere." A HASH, deliberately, never the raw fingerprint: the fingerprint is a
1024-integer low-resolution rendering of a real person's first name on flat app chrome, and an
equality join needs only that two rows' KEYS compare equal, never the pixels themselves -- a
cryptographic hash preserves exactly that property (equal input -> equal digest, nothing about
the digest recoverable back to the input) while a bare fingerprint sitting in a table whose whole
purpose is aggregate calibration would be a real rendering of a real name for no attribution
benefit at all. The grid rides inside the hashed payload too, so a future recalibration of
`item_identity._IDENTITY_GRID` opens a new key namespace rather than silently colliding with or
diverging from keys hashed under the old one. No name is ever mixed in, for the same reason the
identity gate itself never reads one: it would make the SAME real profile hash to two different
keys depending on which host captured it (OCR availability differs by machine), which destroys
the one property this key exists to provide. `profile_key_from_identity` returns `None` (stored
as `""`, matching `profile_id`'s own empty-string convention) rather than a placeholder hash when
there is nothing to key -- an unattributable observation is still worth storing, never dropped.
`openers.profile_key` is the new column carrying this key (both halves of the two-step migration
present in both backends: `_SCHEMA`/`_TABLES` for a fresh database, the paired `ALTER` for the
live one), threaded alongside -- never instead of -- `profile_id`, and now flows all the way
through `OpenerService.commit_opener`/`discard_opener` and both of `worker.py`'s Training and AUTO
paths, captured once per card before any action that could invalidate it. `opener_outcomes` is
the new table (a brand-new table, so only the `CREATE TABLE IF NOT EXISTS` half is needed, no
`ALTER` -- it cannot pre-date itself): `app`, `profile_key`, `outcome`, `observed_at`, `source`,
`note`, `created_at`, with no `run_id` and no foreign key into `openers` at all, because an
outcome is observed on a real conversation often well after any automation run ended. `(app,
profile_key)` is the whole join key back to `openers` (`Store.joined_opener_outcomes`), and that
join excludes empty-string AND NULL `profile_key` on BOTH sides -- two unattributable rows must
never join to EACH OTHER just because both happen to carry the same "no key" spelling. `outcome`
(`ranker.KNOWN_OPENER_OUTCOMES`: match / reply / no_response / unmatch / unknown) and `source`
(`KNOWN_OPENER_OUTCOME_SOURCES`: owner / automated) are documented, unenforced vocabularies,
matching the "suggestions, not restrictions" rule this project already applies to `angle`.
`tools/opener_outcome_recorder.py` is the recording CLI itself: `list` reads the last `--limit`
opener rows for an app (ANY decision, most-recent first) but only PRINTS the ones actually SENT
(`decision == "like"`, the one value `OpenerService.commit_opener` has ever written), each
carrying its own name (when a Training label recorded one, joined by the same `profile_id` a
Training decision and its opener share), the opener text, when it was sent, and its prompt era
(resolved through the same `ops/prompt-eras.json` registry `tools/opener_corpus_report.py`
already reads). `record --index N` re-reads the identical window and resolves N through
`resolve_selection()`, which is the one place a write can happen at all: an index outside the
window, or one that resolves to a row that was never sent, is refused with a clear, named reason
BEFORE the store is ever touched -- so it is structurally impossible to record an outcome against
a draft that was never sent, not merely discouraged. The owner never sees or types a raw
`profile_key` hash: `record_outcome()` reads it off the already-resolved row and passes it to
`Store.record_opener_outcome` directly. `source` is never a caller-supplied flag -- every row
this tool writes is stamped `owner`, because "owner-entered" is the one guarantee this manual
entry point exists to make. Finally, `tools/opener_corpus_report.py` gained a third axis,
METRICS BY OUTCOME, built from `opener_outcomes` rows already joined to their `openers` row by
`(app, profile_key)` and grouped by prompt era, the same grouping the PRODUCED-side METRICS BY
PROMPT ERA axis already uses -- with synthetic
replay rows (`decision == "synthetic_replay"`) explicitly excluded, since they were never sent to
a real person and can never have earned a real outcome. `response_rate` counts `match`/`reply` as
"a response" (excluding `unmatch`/`no_response`/`unknown`, whose raw counts are still shown, never
folded into that one summary number either way) -- but ONLY when the bucket holds at least
`MIN_SAMPLE_FOR_OUTCOME_RATE` (10) joined rows; below that, one new observation swings the rate
by more than ten points, so the tool prints an explicit "n too small for a rate" refusal and the
raw counts instead of a number precise-looking enough to be mistaken for a real measurement. This
is a documented judgment call, not a statistically derived bound, and the same guard blocks
`--compare`'s delta whenever EITHER side is below threshold, rather than silently substituting a
withheld rate with 0.0 or any other value.

DELIBERATELY NOT BUILT: no automated Hinge match/chat screen reader. Reading her matches or
messages automatically would add a brand-new screen-reading surface with its own anti-bot
exposure -- exactly the kind of decision `ops/ANTI-BOT-RESEARCH.md` exists to be the canonical
record for, and this entry does not reopen that debate or make that call on its own. It would
also introduce attribution errors of its own kind (matching the right conversation to the right
opener from an automated read is a harder, less honest problem than an owner who was there
looking at her own phone). Collection is therefore manual first, by design, and the table is
shaped so an automated source can populate it later without any schema change at all: `source`
is already plain TEXT against a documented vocabulary that includes `automated`, not a closed
enum that would need a migration to grow.

RESIDUALS, stated rather than smoothed over.
- The 95 historical drafts this document's 2026-09-06 (d) prediction was measured against remain
  PERMANENTLY UNREPLAYABLE: their exact request inputs were never captured (see THE TWO BLOCKERS
  FOUND above), and no amount of retroactive tooling can reconstruct a crop that was never saved.
  The baseline that prediction is checked against therefore stays era-mixed forever, not just
  until the next backfill.
- `profile_key` is an EQUALITY key, never a similarity matcher, inherited directly from the
  identity gate's own fingerprint. A header Hinge renders even slightly differently later -- a
  new app version's font hinting, a redrawn layout -- fingerprints to a different key, so the
  outcome simply has no opener row to join to: UNATTRIBUTED, never MIS-attributed. The opposite
  failure is also real and already measured, not merely theoretical: the identity gate's own
  calibration records two different real profiles as close as 2.565 grey levels apart at this
  grid, so a `profile_key` COLLISION across two different people sharing a similarly-rendered
  name is possible and would silently merge their outcome history under one key.
- Outcome data is owner-observed, not automatically detected, and is therefore incomplete by
  construction and possibly biased toward whichever outcomes were easiest, most memorable, or
  most rewarding to go note down. Silence in this axis means "nobody recorded an outcome for this
  opener," never "nothing happened."
- `n` will be far too small for a rate for a long time -- collection just started -- and
  `tools/opener_corpus_report.py` refuses to print one below `MIN_SAMPLE_FOR_OUTCOME_RATE` (10)
  rather than let a reader mistake an unstable count for a real measurement.
- The replay corpus stores real people's photos locally, gitignored but otherwise unmanaged: how
  long a captured directory should be kept, and whether it is ever pruned or deleted, is an OPEN
  OWNER DECISION this entry deliberately does not make.

TEST PINS BY NAME, reading them out of the test files rather than restating this brief.

`tests/test_profile_key.py` (new file, 10 tests):
`test_compute_profile_key_matches_the_documented_canonical_encoding`,
`test_compute_profile_key_returns_a_64_character_lowercase_hex_digest`,
`test_compute_profile_key_is_deterministic`,
`test_compute_profile_key_changes_when_the_fingerprint_changes`,
`test_compute_profile_key_changes_when_the_grid_changes`,
`test_compute_profile_key_accepts_any_integer_sequence_not_only_tuples`,
`test_profile_key_from_identity_returns_none_for_no_identity`,
`test_profile_key_from_identity_returns_none_when_fingerprint_is_unknown`,
`test_profile_key_from_identity_hashes_the_known_fingerprint_and_grid`,
`test_profile_key_from_identity_never_returns_the_raw_fingerprint`.

`tests/test_sqlite_store.py` and `tests/test_bigquery_store.py`, the `profile_key`/
`opener_outcomes` sections (21 and 16 new tests respectively): schema/migration pins
(`test_sqlite_migrates_legacy_openers_profile_key_column`,
`test_sqlite_opener_outcomes_table_has_the_expected_columns_in_order`,
`test_sqlite_creates_the_opener_outcomes_table_in_a_preexisting_database_that_lacks_it`,
`test_ensure_tables_declares_profile_key_on_the_openers_create_table`,
`test_ensure_tables_runs_openers_profile_key_migration`,
`test_ensure_tables_creates_opener_outcomes_table`), the hash-not-fingerprint pin
(`test_sqlite_record_opener_persists_profile_key_as_the_hash_never_the_raw_fingerprint`,
`test_record_opener_persists_profile_key_as_the_hash_never_the_raw_fingerprint`), and the join's
exclusion rules (`test_sqlite_joined_opener_outcomes_excludes_empty_string_profile_key_on_both_sides`,
`test_sqlite_joined_opener_outcomes_excludes_null_profile_key_openers`,
`test_sqlite_joined_opener_outcomes_does_not_cross_app_boundaries`,
`test_joined_opener_outcomes_query_excludes_empty_and_null_profile_keys_on_both_sides`).

`tests/test_opener_service.py` (5 `profile_key`-threading tests plus 6 replay-corpus-capture
tests): `test_commit_opener_threads_profile_key_to_the_store`,
`test_discard_opener_threads_profile_key_to_the_store`,
`test_commit_and_discard_opener_default_profile_key_to_empty_string`,
`test_commit_opener_degrades_gracefully_when_the_store_predates_profile_key`,
`test_a_committed_and_a_discarded_draft_can_carry_the_same_profile_key`,
`test_replay_corpus_capture_is_not_called_when_disabled`,
`test_replay_corpus_capture_persists_the_exact_request_when_enabled`,
`test_replay_corpus_capture_exception_never_propagates_or_changes_the_outcome`.

`tests/test_worker.py` (5 tests): `test_auto_committed_opener_carries_a_profile_key_from_the_driver`,
`test_auto_discarded_draft_carries_the_same_profile_key_a_commit_would_have`,
`test_auto_committed_opener_profile_key_is_empty_when_the_driver_has_no_identity_hook`,
`test_auto_committed_opener_profile_key_is_empty_when_identity_is_unknown`,
`test_auto_profile_key_is_captured_before_the_like_action_invalidates_it`.

`tests/test_config.py` (2 dedicated tests plus parametrized invalid-value coverage):
`test_opener_replay_corpus_enabled_defaults_to_false`,
`test_opener_replay_corpus_enabled_true_loads_and_validates_cleanly`.

`tests/test_opener_replay.py` (new file, 21 tests, covering both `replay_corpus.py`'s format and
`tools/opener_replay.py`'s CLI): `test_write_then_load_roundtrips_every_field`,
`test_manifest_json_has_exactly_the_documented_shape`,
`test_write_replay_capture_is_idempotent_by_content`,
`test_write_replay_capture_never_raises_on_a_disk_failure`,
`test_main_dry_run_makes_no_network_call_and_no_store_write`,
`test_main_live_requires_confirmation_and_does_not_call_transport_when_declined`,
`test_main_live_writes_replay_marker_under_the_current_prompt_era`,
`test_main_live_replay_marker_is_never_the_sent_decision_value`.

`tests/test_opener_corpus_report.py`, the OUTCOMES-axis section (35 new tests): join/read pins
(`test_read_sqlite_opener_outcomes_joins_by_app_and_profile_key`,
`test_read_sqlite_opener_outcomes_never_joins_across_an_app_boundary`,
`test_read_sqlite_opener_outcomes_excludes_unattributed_rows_on_both_sides`,
`test_read_sqlite_opener_outcomes_excludes_synthetic_replay_rows`,
`test_read_bigquery_opener_outcomes_excludes_synthetic_replay_rows`), the small-sample guard
(`test_outcome_metrics_counts_by_kind_and_response_rate_at_the_threshold`,
`test_outcome_metrics_below_threshold_blocks_the_rate_but_keeps_the_counts`,
`test_outcome_metrics_empty_bucket_blocks_the_rate_without_dividing_by_zero`,
`test_outcome_metrics_unmatch_is_reported_but_never_counted_as_a_response`,
`test_compare_outcome_eras_blocks_the_delta_when_either_side_is_below_threshold`), and the
wiring into `main()`/`build_report()`
(`test_build_report_carries_the_by_outcome_axis`, `test_main_json_mode_carries_the_outcome_axis`,
`test_main_text_mode_prints_the_outcome_section`,
`test_caveats_state_outcomes_are_owner_observed_and_incomplete`).

`tests/test_opener_outcome_recorder.py` (new file, 40 tests, this entry's own CLI): the selection
design (`test_resolve_selection_refuses_an_index_outside_the_window`,
`test_resolve_selection_refuses_a_draft_that_was_never_sent`,
`test_resolve_selection_returns_the_row_for_a_valid_sent_index`), the write itself
(`test_record_outcome_writes_the_resolved_profile_key_and_owner_source`,
`test_record_outcome_passes_through_an_empty_profile_key_rather_than_refusing`), the listing
filter (`test_list_sent_openers_filters_to_sent_and_keeps_true_window_position`,
`test_format_list_text_shows_only_sent_openers_with_name_and_era`), the name-resolution read
(`test_read_recent_openers_sqlite_resolves_the_training_label_name`,
`test_read_recent_openers_sqlite_leaves_name_empty_for_auto_mode_openers`), and the end-to-end
refusal proofs that no orphan row is ever written
(`test_main_record_refuses_an_unsent_draft_and_writes_no_orphan_row`,
`test_main_record_refuses_an_unknown_index_and_writes_no_orphan_row`,
`test_main_record_without_yes_asks_for_confirmation_and_aborts_on_no`). Every assertion pinning
a refusal was mutation checked by hand: the `is_sent` guard, the window-bounds guard, the
listing filter, and the profile_key/source pass-through were each disabled or swapped in turn,
confirmed to make its corresponding test fail, then restored byte-for-byte.

Full suite (`pytest -q -n auto`): 5105 -> 5271 passed, 5 skipped (unchanged), 392.44s wall clock.
Offline only -- no device, no live API call, no BigQuery read or write from this entry's own
additions (`tools/opener_outcome_recorder.py` and its test file); every other component this
entry documents was already on disk when this entry began and is described, not re-implemented,
here.


#### Addendum -- 2026-09-06 (j): capture is on, bounded, and the verdict says where it came from

Three owner decisions, taken together because enabling capture without retention would have been
the wrong order: photos must be bounded from the FIRST capture, never retroactively.

RETENTION, decided. The replay corpus stores real people's photos on disk, so it is now bounded
by BOTH a maximum capture count and a maximum age, pruned oldest first after every successful
capture, with 0 documented as unlimited for either bound. The defaults are 400 captures and 180
days. The count had to comfortably exceed the pre-registered minimum of 40 drafts so retention
can never silently delete the very evidence the prediction needs, while still leaving several
eras comparable; 180 days spans several prompt revisions without anything lingering indefinitely.
An automatic bound was chosen over a documented manual habit on purpose: a retention policy that
depends on someone remembering is not a policy.

THE DELETE PATH was treated as the highest-risk part of the change, because it removes
directories of real people's photos under config control. It resolves the root, validates every
candidate against the id format the module itself produces, refuses separators and traversal,
refuses to follow a symlink out of the root, and confirms the resolved candidate really is inside
the resolved root before removing anything. A capture whose manifest is missing or corrupt is
KEPT, never deleted, and never counted. Like every write on this path it can never raise into the
opener, decision, send or refusal path. Each escape was attempted in a test and refused.

CAPTURE ENABLED. config.yaml now sets opener.replay_corpus_enabled: true, while the CODE default
in config.py stays False: the shipped config opts in, the library does not. Capture happens in
OpenerService.maybe_opener, which is the path Training and AUTO share, so a Training run captures.
It takes effect the next time the supervisor loads config.yaml, not in any process already running.

PROGRESS AND VERDICT REPORTING. tools/opener_corpus_report.py now prints a REPLAY CORPUS section
and a PRE-REGISTERED CHECK: how many drafts exist under the current era, how many more are needed
to reach the minimum, and, once the threshold is met, each pre-registered metric beside its
predicted value with the agreed falsification verdict. Below threshold it prints the shortfall and
refuses a verdict. The threshold lives in one named constant pointing at the (d) addendum that set
it. tools/opener_replay.py gained a purge path that defaults to a dry run.

THE REVIEW FINDING WORTH RECORDING. The first version of that verdict rendered THRESHOLD MET,
PASS/FAIL and FALSIFIED without ever disclosing whether the drafts behind it were live sends or
offline replays. That is the same class of defect as the era axis pooling replay rows silently,
found earlier the same day, and it matters more here because this verdict is the agreed trigger
for shipping the positive-specification restructure. The fix DISCLOSES rather than suppresses:
replayed drafts are genuine model output under the current prompt, so they are valid PRODUCED-side
evidence and are still counted, since making the prediction checkable without further device time
is exactly why replay exists. What is now impossible is mistaking one for the other. The check
carries the synthetic count, a basis of live, mixed or replay, an explicit PROVENANCE paragraph
whenever any draft is synthetic, and the basis is stamped on the VERDICT line itself so a reader
who skims to the verdict alone still sees it.

RESIDUALS. The prediction remains UNTESTED until a live batch actually runs; nothing here changes
that, it only makes the moment it becomes checkable visible. Retention bounds the corpus but the
photos are still real and still local, so deletion remains the owner's call at any time via the
purge path. And the outcome axis stays empty until outcomes are recorded by hand: everything the
report shows about this era is still what the prompt PRODUCED, never what performed.

Suite 5397 -> 5403 passed, 5 skipped, lint clean. New pins include
tests/test_replay_corpus_prune.py (the prune helper and every attempted path escape) and the
provenance pins in tests/test_opener_corpus_report.py (verdict_basis live/mixed/replay, the
synthetic count and PROVENANCE paragraph, the basis stamped on the VERDICT line, a live verdict
left unlabelled, and to_dict carrying both). All were mutation tested.
