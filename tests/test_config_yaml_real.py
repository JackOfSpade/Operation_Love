"""Smoke tests against the shipped config.yaml — catches typos and dead config keys
that unit tests with inline YAML strings would miss.

All tests here are offline (no GCP / no phone / no SDK).

These used to skip validate() entirely, on the stated grounds that it "requires a
BigQuery project_id + photo_bucket ... would fail in a clean CI env". That premise was
wrong: validate() only checks those keys are present, non-empty strings — it opens no
connection and needs no credentials. The consequence was a real hole: every rule
validate() enforces (the registry's check_selection, halt_on_error-vs-auto coherence,
the pacing floor, opener.max_attempts' bounds, ranker.retrain_every) was exercised only
against inline YAML in other test files, never against the file we actually ship. A
config.yaml that could not start was therefore fully CI-green.
"""
import pytest

from operation_love.config import load, validate


@pytest.fixture(scope="module")
def cfg():
    return load("config.yaml")


def test_config_yaml_loads_without_error(cfg):
    """The shipped file parses and passes every offline startup validation."""
    assert cfg.enabled_apps                        # at least one app
    assert cfg.mode in {"observe", "auto"}
    assert cfg.budget.run_budget_usd is not None   # a run budget is set
    # In particular, Hinge's intentionally absent targeting calibration remains a valid
    # observe configuration: it prevents targeted suggestions at runtime, but must not turn
    # the whole shipped application configuration into an unloadable file.
    validate(cfg)


def test_opener_model_has_pricing_entry(cfg):
    """The model named in opener.model must have a pricing entry so spend tracking works."""
    model = cfg.opener.model
    assert model in cfg.budget.pricing, (
        f"opener.model={model!r} has no entry in budget.pricing. "
        f"Available: {sorted(cfg.budget.pricing.keys())}"
    )


def _example_opener_lines(style):
    """The example opener lines the model is actually shown, extracted from the style block.

    Returned verbatim (stripped of leading indentation only), so a caller can assert on the
    exact characters the model copies. Few-shot examples are the strongest instrument in this
    prompt, so the owner rules about what may appear in them are enforced against THESE lines
    specifically rather than against the block as a whole, which contains prose the rules do
    not govern (the HARD RULE has to spell "PA-C" to forbid it).

    Structure of the EXAMPLES OF THE EDIT block (config.yaml, per ops/OPENER-REDESIGN.md 3.3):
    an item description at column 0, then its examples indented and labelled "No:"/"Yes:", and
    one example that WRAPS onto a continuation line indented deeper still. The continuation is
    part of the example the model reads, so it is collected too.
    """
    lines = style.splitlines()
    starts = [i for i, line in enumerate(lines)
              if line.strip().startswith("EXAMPLES OF THE EDIT")]
    assert len(starts) == 1, (
        "expected exactly one 'EXAMPLES OF THE EDIT' heading in opener.style, "
        f"found {len(starts)}"
    )
    collected = []
    in_example = False
    for line in lines[starts[0] + 1:]:
        stripped = line.strip()
        if stripped.startswith(("No:", "Yes:")):
            in_example = True
            collected.append(stripped)
        elif in_example and stripped and line.startswith(" "):
            collected.append(stripped)      # wrapped continuation of the example above
        else:
            in_example = False              # blank line, or an item description at column 0
    return collected


def test_shipped_opener_style_keeps_faithful_corey_framework_and_two_sentence_cap(cfg):
    """The Corey Wayne voice and the length ceiling survive the 2026-08-11 redesign.

    Everything here predates ops/OPENER-REDESIGN.md and must NOT have been lost while the
    substance rules around it were rewritten. The one deliberate casualty is the ONE-sentence
    *preference*, removed per doc 3.2.1: a claim that could be wrong needs room to exist, and
    the strongest opener in the whole design ("I know you were smiling, but I bet you were
    freezing out there") is eighteen words. The two-sentence CEILING stays; the preference is
    replaced by an economy rule aimed at padding rather than at substance, pinned below.
    """
    style = " ".join(cfg.opener.style.lower().split())
    assert "90/10 framework" in style
    assert "genuinely curious" in style
    assert "do not force teasing into every opener" in style
    assert "one open, easy to answer question" in style
    assert "positive, fun conversation" in style
    assert "brief greeting is optional" in style
    assert "two sentences is the absolute maximum" in style
    assert "exactly one concrete detail" in style
    assert "never use an em dash or any hyphen" in style
    assert "low investment so she chases" not in style
    assert "tease her like a bratty little sister" not in style
    assert "no interview questions" not in style
    # The ceiling now lives under its own heading, alongside the economy rule that replaced
    # the brevity preference. Pinning the heading keeps the two from drifting apart.
    assert "application rule: two sentences is the absolute maximum" in style
    assert "as short as the claim allows" in style
    assert "spend no word on anything she can already see" in style
    # Doc 3.2.1: this phrasing must stay GONE. Its return would silently reinstate the
    # brevity-for-its-own-sake pressure the redesign removes, and it would do so without
    # contradicting any other assertion in this file.
    assert "one short sentence is preferred" not in style
    assert "one short sentence" not in style


def test_shipped_opener_style_ships_the_unbluffable_claim_rule(cfg):
    """The substance rule from ops/OPENER-REDESIGN.md 2, 2.1 and 2.2.

    This is the whole point of the redesign. "exactly one concrete detail" (asserted above)
    survived the rewrite but no longer means what it used to: on its own it never distinguished
    being GROUNDED IN a detail from NAMING it, which is how we shipped "That view by the sauna
    during sunset looks relaxing, where is this from?" under a photo of a sauna at sunset. Each
    half is now load-bearing, so each half is pinned:

      - the premise that makes the rule unconditional (she is looking at the item),
      - the falsifiability property that replaced brevity as the fix,
      - the premise-not-point carve-out that keeps "based on that ridgeline" legal,
      - the "do not say the detail back to her" clause bolted onto the grounding sentence,
      - the demotion of questions to a second beat.
    """
    style = " ".join(cfg.opener.style.lower().split())
    assert "shared context rule" in style
    assert "displayed directly under the exact photo or prompt it attaches to" in style
    assert "write like two people looking at the same thing" in style
    assert "the one rule: your opener must contain a claim that could be wrong" in style
    assert "she is not checking whether you have eyes" in style
    assert "may be your premise. it may never be your point" in style
    assert "test it by covering the photo: if nothing is left, start over" in style
    # Bug report 2026-08-15: a deck photo licensed a made-up staircase, exertion and target
    # time. The model may make one uncertain inference; it may not invent facts to serve as
    # premises for further guesses.
    assert "evidence boundary, one hop only" in style
    assert "the premise must be plainly visible in the item or explicitly stated" in style
    assert "never stack guesses" in style
    assert "unseen action, route, effort, goal, cause, or before and after sequence" in style
    assert "an observation deck does not mean she climbed stairs or had a target time" in style
    assert "a summit does not mean she hiked there" in style
    assert "your legs were jelly after climbing all those stairs" in style
    assert "choose a different claim or a different item" in style
    # One-hop is necessary but not sufficient: a single-photo inference can still assume an
    # implausibly specific backstory. Pin evidence-proportional calibration and the reporter's
    # goat/barn counterexample.
    assert "calibrate the guess: a hedge does not rescue a far fetched premise" in style
    assert "natural under most ordinary explanations of the scene" in style
    assert "if it works only under one special backstory, do not use it" in style
    assert "ownership, employment, a routine, a responsibility, or a relationship" in style
    assert "feeding one goat does not mean she owns it, works on a farm" in style
    assert "spent the day cleaning a barn" in style
    assert "it could be a wild encounter on a trail" in style
    assert "guess about the interaction the evidence shows, not an unshown life story" in style
    assert "confidently wrong is playful only when the guess was reasonable" in style
    assert "you became its favorite person the second the snacks came out" in style
    assert "traceability test" in style
    assert "she should instantly see which visible or stated clue led you to that angle" in style
    assert "the path from clue to guess must be obvious without an explanation" in style
    assert "how did you possibly see it that way?" in style
    assert "fantasized a backstory from minimal evidence" in style
    assert "reasoning she can recognize at a glance" in style
    # The grounding sentence and its new second half must stay adjacent: the sentence alone is
    # root cause #1 from doc 1.1, and it is only safe with this clause attached.
    assert "exactly one concrete detail" in style
    assert "then do not say that detail back to her" in style
    # Doc 2.2: questions stay legal, they stop being the default.
    assert "a claim she can correct beats a question she has to answer" in style
    assert "never as the whole message" in style


def test_shipped_opener_style_frames_the_five_moves_as_non_binding_examples(cfg):
    """ops/OPENER-REDESIGN.md 2.3: the five moves are illustrations, never a menu.

    Deliberate and easy to "tidy" into a bug: if the escape clause is ever dropped, the moves
    read as a checklist and the model shoehorns one onto an item it does not suit, which is
    worse than plain. The five moves themselves are pinned so a silent deletion is caught, and
    Connect is pinned as the strongest because that ranking is the reason Part B exists at all.
    """
    style = " ".join(cfg.opener.style.lower().split())
    assert "ways this tends to look. these are examples, not a checklist" in style
    assert "write whatever does fit and keep the rule" in style
    assert "never shoehorn" in style
    assert "guess something from the evidence and commit to it" in style
    assert "say something you know that the item brought to mind" in style
    assert "make one direct inference outside the frame, licensed by what is visible or stated" in style
    assert "claim something outside the frame: what she felt, what it cost, what happened next" not in style
    assert "tease her, good naturedly, about something the evidence licenses" in style
    assert "connect two things she said in different places on her profile" in style
    assert "connecting two separate places on her profile is the strongest" in style


def test_shipped_opener_style_does_not_ship_the_item_selection_rule(cfg):
    """Bug 1 fix, 2026-08-12 audit: this block used to carry the item-selection instruction
    (PICK THE ITEM YOURSELF / THE FAILURE TO AVOID / THE UNNUMBERED IMAGES ARE CONTEXT) and no
    longer does, on purpose.

    opener.py's _SYSTEM keeps the selection criterion -- the tradeoff, the failure mode named as
    a failure, and why to use (never pick) the unnumbered context tier -- in full, compressed but
    not abridged, pinned by tests/test_opener.py. The instruction lives in exactly one place, and
    this test pins its absence here rather than its presence.
    """
    style = " ".join(cfg.opener.style.lower().split())

    for _phrase in (
        "pick the item yourself",
        "the failure to avoid is picking an item you have nothing to say about",
        "the unnumbered images are context, not choices",
        "which one you write about is your choice to make",
        "set item_index to its number",
    ):
        assert _phrase not in style, (
            f"opener.style ships the item-selection instruction again ({_phrase!r}); this "
            "block must not duplicate the selection rule. The selection criterion belongs in "
            "opener.py's _SYSTEM -- see test_opener.py's item-selection assertions."
        )

    # Everything else the redesign shipped around it must survive untouched -- this test exists
    # to catch a regression in ONE paragraph, not to license churn on its neighbours.
    assert "connecting two separate places on her profile is the strongest" in style
    assert "a claim she can correct beats a question she has to answer" in style


def test_shipped_opener_style_ships_the_redesign_guardrails(cfg):
    """ops/OPENER-REDESIGN.md 2.4. Two of these three are safety rules, not style.

    HEDGE THE CLAIM is what makes a wrong guess a feature rather than a risk, and it is the
    only thing that rescues the "say something you know" move from competing with her own
    expertise. GUESS THE WORLD caps the SPECIFICITY of a guess, not its accuracy, and is the
    rule keeping us off her street/employer/school/age. NEVER INVENT THE SENDER stops the model
    fabricating the owner's history, which he then has to sustain five messages later.

    British spellings are the doc's, kept verbatim on the "implement what it says" rule, so
    they are pinned as shipped rather than quietly Americanized here.
    """
    style = " ".join(cfg.opener.style.lower().split())
    assert "hedge the claim, never yourself" in style
    assert "never apologise for writing, never ask permission" in style
    assert "guess the world, not her identity" in style
    assert "never a street, a neighbourhood, a hotel, a specific venue" in style
    assert "the way a well travelled friend would" in style
    assert "never guess her employer, her school, or her age" in style
    assert "never invent the sender" in style
    assert "you may not claim he has been somewhere" in style


def test_shipped_opener_style_hedge_forms_are_wide_and_openings_are_varied(cfg):
    """ops/OPENER-REDESIGN.md 3.6 (entropy guard) plus its 2026-08-11 addendum: a live dry run
    against a real profile produced 5/5 openers that all opened with "I bet". Root cause was the
    HEDGE THE CLAIM guardrail naming too few forms with "I bet" landing first/most salient, both
    here and in opener.py's _SYSTEM (pinned separately by tests/test_opener.py). This test pins
    the two-part fix so the regression cannot come back silently: (1) more than two named hedge
    forms, offered as illustrative rather than a fixed menu, and (2) an explicit instruction not
    to open every message the same way.
    """
    style = " ".join(cfg.opener.style.lower().split())
    _hedge_forms = ["i'm going to guess", "i heard", "i'm assuming", "something tells me",
                    "odds are", "my money is on", "i bet"]
    for form in _hedge_forms:
        assert form in style, f"hedge form {form!r} missing from opener.style"
    _forms_present = sum(1 for form in _hedge_forms if form in style)
    assert _forms_present > 2, (
        f"opener.style's HEDGE THE CLAIM line only names {_forms_present} hedge forms "
        f"(need >2) -- this is the exact 'I bet' monoculture regression"
    )
    assert "these are examples, not a menu to pick the same one from every time" in style
    assert "vary the opening" in style
    assert "do not start every message the same way" in style


def test_shipped_example_openers_contain_no_hyphen_em_dash_or_non_ascii(cfg):
    """Owner rule, and the strictest place it applies: the few-shot examples.

    The model copies the shape of what it is shown, so a single hyphen inside an example opener
    would teach it to write hyphens regardless of what the HARD RULE says in prose. The HARD
    RULE itself has to contain "PA-C" in order to forbid it, which is why this is asserted
    against the example lines rather than the whole block, and why the block-wide check below
    allows exactly that one hyphen and names it.
    """
    raw = cfg.opener.style
    examples = _example_opener_lines(raw)
    # Guard the extractor itself: if a reflow ever breaks the parse, this test must fail loudly
    # rather than silently pass over zero lines. 10 = 2 sauna + 3 husky + 1 ridge + 2 prompt
    # card (one of which wraps, giving 3 lines) + 1 hot sauce.
    assert len(examples) == 10, f"extractor found {len(examples)} example lines: {examples}"
    assert "Yes: That view looks relaxing, where is this from?" in examples
    assert "No:  That view by the sauna during sunset looks relaxing, where is this from?" in examples
    assert "Yes: I know you were smiling, but I bet you were freezing out there." in examples
    assert "which one is the lie?" in examples          # the wrapped continuation line
    for line in examples:
        assert "-" not in line, f"example opener contains a hyphen: {line!r}"
        assert "—" not in line, f"example opener contains an em dash: {line!r}"
        assert "–" not in line, f"example opener contains an en dash: {line!r}"
        assert all(ord(c) < 128 for c in line), f"example opener is not plain ASCII: {line!r}"
    # Block-wide, the same rules hold with exactly one documented exception. Asserting the
    # count (rather than "no hyphen") is what lets this stay strict: a new hyphen anywhere in
    # the style text fails here even if it is added outside the examples.
    assert all(ord(c) < 128 for c in raw), "opener.style must be plain ASCII"
    assert "—" not in raw
    assert raw.count("-") == 1, (
        "opener.style should contain exactly one hyphen, the 'PA-C' the HARD RULE forbids; "
        f"found {raw.count('-')}"
    )
    assert 'not "PA-C"' in raw


def test_hinge_identity_top_name_band_matches_the_measured_ocr_band(cfg):
    """apps.hinge.identity_top_name_band must carry the exact band MEASURED on real Pixel 7a
    frames on 2026-08-10 (tesseract --psm 6 read the card-header name correctly on every
    scroll-top frame tested, both banner-present and banner-gone layouts). This is what fixes
    observe mode recording a pass that advanced Profile A -> Profile B as a scroll of Profile A -- see
    android_spec.py's identity_top_name_band docstring for the full mechanism. A drifted or
    dropped value here would silently defeat the scroll-top name check on the one platform
    that actually runs (Hinge; see live-bringup-status)."""
    band = cfg.apps["hinge"].get("identity_top_name_band")
    assert band is not None, "apps.hinge.identity_top_name_band is missing from config.yaml"
    assert tuple(band) == (0.03, 0.130, 0.75, 0.250)


def test_limits_are_uncapped_by_default(cfg):
    """Auto-mode volume is deliberately uncapped by default (see config.yaml's `limits:`
    block): a fixed numeric ceiling is itself a bot signature (an identical hard step
    every run), so the shipped config must not set any of these by default. Timing remains
    paced; the profile queue, real stop conditions, or a manual stop end the run.
    """
    lim = cfg.limits
    assert lim.get("max_per_run") is None
    assert lim.get("max_per_day") is None
    assert lim.get("max_likes_per_run") is None
    assert lim.get("target_like_ratio") is None


def test_shipped_config_actually_passes_validate():
    """The file we ship must satisfy every rule validate() enforces — not just parse.

    This is the test whose absence let the shipped config drift: validate() is what the
    real entry points call before a run, so a config.yaml that fails it cannot start the
    app at all, yet nothing here checked. It needs no network and no credentials.
    """
    from operation_love.config import validate
    validate(load("config.yaml"))          # must not raise


def test_shipped_config_selects_something_the_registry_can_actually_run():
    """enabled_apps must name a platform that is available right now.

    validate() deliberately allows an UNAVAILABLE platform (you must be able to configure
    Bumble's coordinates before Bumble is calibrated), so it alone cannot catch a shipped
    config that parses, validates, and then refuses to start. That gap is what this covers.
    """
    from operation_love import platforms
    cfg = load("config.yaml")
    assert platforms.check_runnable(cfg.enabled_apps) is None, (
        f"config.yaml ships enabled_apps={cfg.enabled_apps}, which cannot start: "
        f"{platforms.check_runnable(cfg.enabled_apps)}"
    )
