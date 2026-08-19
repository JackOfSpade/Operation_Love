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


def test_shipped_config_defaults_to_hinge_observe_with_bumble_auto_configured(cfg):
    assert cfg.enabled_apps == ["hinge"] and cfg.mode == "observe"
    coords = cfg.apps["bumble"].get("coords", {})
    assert set(coords) == {"swipe_start", "swipe_like_end", "swipe_pass_end"}
    assert "like_heart" not in coords and "pass_x" not in coords


def test_opener_model_has_pricing_entry(cfg):
    """The model named in opener.model must have a pricing entry so spend tracking works."""
    model = cfg.opener.model
    assert model in cfg.budget.pricing, (
        f"opener.model={model!r} has no entry in budget.pricing. "
        f"Available: {sorted(cfg.budget.pricing.keys())}"
    )


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
    assert "one specific, easy, positive question may be the whole message" in style
    assert "positive, fun conversation" in style
    assert "brief greeting is optional" in style
    assert "two sentences is the absolute maximum" in style
    assert "a visible detail may be named" in style
    assert "never use an em dash or any hyphen" in style
    assert "low investment so she chases" not in style
    assert "tease her like a bratty little sister" not in style
    assert "no interview questions" not in style
    # The ceiling now lives under its own heading, alongside the economy rule that replaced
    # the brevity preference. Pinning the heading keeps the two from drifting apart.
    assert "application rule: two sentences is the absolute maximum" in style
    assert "as short as the angle allows" in style
    assert "spend no word merely repeating what she can already see" in style
    assert "never cut necessary setup or the conversational payoff" in style
    # Doc 3.2.1: this phrasing must stay GONE. Its return would silently reinstate the
    # brevity-for-its-own-sake pressure the redesign removes, and it would do so without
    # contradicting any other assertion in this file.
    assert "one short sentence is preferred" not in style
    assert "one short sentence" not in style


def test_shipped_opener_style_requires_value_without_forcing_a_claim(cfg):
    """Visible setup is legal; only description as final payoff fails, and claims are optional."""
    style = " ".join(cfg.opener.style.lower().split())
    assert "shared context rule" in style
    assert "displayed directly under the exact photo or prompt it attaches to" in style
    assert "write like two people looking at the same thing" in style
    assert "photo header rule" in style
    assert "title, caption, or prompt printed with a photo is part of that same item" in style
    assert "defines how the photo is meant to be read" in style
    assert "interpret the visible scene through that text before choosing an angle" in style
    assert "never contradict, reverse, or ignore the header's framing" in style
    assert "conversational value rule" in style
    assert "a visible detail may be named and may be the subject, premise, or setup" in style
    assert "final conversational point is merely that description" in style
    assert "perspective, grounded interpretation, playful framing, connection, or natural question" in style
    assert "does not have to contain a guess or a claim that could be wrong" in style
    assert "claims only when natural" in style
    assert "a correctable inference is one available move, not a requirement" in style
    assert "prefer a grounded observation or specific question over a forced guess" in style
    assert "write the whole proposition rather than a shorthand answer" in style
    assert "choose the least speculative interpretation" in style
    assert "minimum invention" in style
    assert "never invent a purpose, motive, cause, plan, sequence, action, route" in style
    assert "do not assign why she chose or did something unless her profile states it" in style
    assert "any literal premise must be plainly visible in the item or explicitly stated" in style
    assert "never use one invented fact as the premise for another" in style
    assert "a setting or destination does not establish how she arrived" in style
    assert "what effort it took, or whether she pursued a goal" in style
    assert "a hedge does not rescue a far fetched premise" in style
    assert "ownership, employment, a routine, a responsibility, or a relationship" in style
    assert "traceability test" in style
    assert "for any inference, she should instantly see which visible or stated clue" in style
    assert "if the path from clue to inference needs an explanation" in style
    assert "playful hyperbole" in style
    assert "unmistakably nonliteral exaggeration is allowed" in style
    assert "does not license presenting an invented motive" in style
    assert "a natural claim she can correct can be effective, but it is not mandatory" in style
    assert "question may be the whole message when that is the strongest natural angle" in style
    assert "never as the whole message" not in style
    assert "setup payoff continuity" in style
    assert "every visible detail you name must be necessary to, and used by" in style
    assert "if removing a descriptive clause leaves the later point or question unchanged, cut it" in style
    assert "question coherence" in style
    assert "ask one coherent thing at a time" in style
    assert "parallel, genuinely contrasting answers to that same underlying question" in style
    assert "never to join unrelated dimensions" in style
    assert "referent clarity" in style
    assert "every pronoun, shorthand noun, and question subject" in style
    assert "must have one immediately obvious referent" in style
    assert "keep the same referent unless the transition to a new one is explicit" in style
    assert "different ordinary meanings of the same word" in style
    assert "role consistency" in style
    assert "preserve that role across every beat" in style
    assert "same subject an incompatible role later" in style
    assert "different subject, name it explicitly" in style
    assert "reply comfort" in style
    assert "most natural honest reply feel good to give" in style
    assert "preference, perspective, inspiration, or experience, not self justification" in style
    assert "intelligence, sincerity, knowledge, effort" in style
    assert "forced choice whose honest answers make her defend, diminish, or embarrass herself" in style
    assert "rather than asking her to verify its status" in style
    assert "premise consistency" in style
    assert "if the first beat asserts or guesses x" in style
    assert "must accept x as its working premise and move the conversation forward" in style
    assert "never ask whether x itself was true" in style
    assert "restate x as a question" in style
    assert "ask about the opposite of x" in style
    assert "abandon x for a generic question about the surrounding scene" in style
    assert "may extend the angle with clearly nonliteral hyperbole" in style
    assert "may not add a literal invented fact, motive, or backstory" in style
    assert "if no coherent continuation exists, stop after the first beat" in style
    assert "your opener must contain a claim that could be wrong" not in style
    assert "test it by covering the photo" not in style
    assert "then do not say that detail back to her" not in style


def test_shipped_opener_style_derives_the_move_without_an_example_menu(cfg):
    """The profile should determine the move without a model-facing taxonomy to imitate."""
    style = " ".join(cfg.opener.style.lower().split())
    assert "derive the conversational move from the specific profile" in style
    assert "rather than choosing from a fixed taxonomy" in style
    assert "two separate parts of her profile create one natural angle" in style
    assert "ways this tends to look" not in style
    assert "examples, not a checklist" not in style


def test_shipped_opener_style_contains_no_concrete_opener_examples(cfg):
    raw = cfg.opener.style
    lowered = raw.lower()
    assert "examples of the edit" not in lowered
    assert not any(line.strip().startswith(("No:", "Yes:")) for line in raw.splitlines())
    for copied_phrase in (
        "froze your butt off",
        "hot chocolate afterwards",
        "favorite person the second the snacks came out",
        "jelly after climbing all those stairs",
        "based on that ridgeline",
    ):
        assert copied_phrase not in lowered


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

    # The whole-profile connection principle remains, without the former move menu.
    assert "two separate parts of her profile create one natural angle" in style
    assert "a natural claim she can correct can be effective, but it is not mandatory" in style


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
    """Pin the 2026-08-15 correction to the earlier prompt-level entropy mitigation.

    Expanding the named hedge list replaced one recurring template with seven salient templates;
    a live run then copied "my money is on" into a semantically incomplete opener. The prompt now
    specifies the job of uncertainty language and lets the item determine its construction.
    """
    style = " ".join(cfg.opener.style.lower().split())
    assert "when a claim is uncertain, express that uncertainty naturally" in style
    assert "wording that fits the specific item" in style
    assert "the goal is calibrated uncertainty, not a particular lead in" in style
    assert "this is not a phrase menu" in style
    assert "choose the construction from context" in style
    assert "never use a hedge as a substitute for the complete self contained claim" in style
    assert "my money is on" not in style
    assert "vary the opening" in style
    assert "let the specific item and angle determine the wording" in style
    assert "do not rotate or recycle a fixed stock hedge" in style


def test_shipped_opener_style_models_its_typography_rules(cfg):
    """The model-facing style itself should obey the typography it requests."""
    raw = cfg.opener.style
    normalized = " ".join(raw.split())
    assert all(ord(c) < 128 for c in raw), "opener.style must be plain ASCII"
    assert "—" not in raw
    assert "–" not in raw
    assert "-" not in raw
    assert "spell out hyphenated abbreviations" in normalized


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
