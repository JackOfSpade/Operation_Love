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
    assert cfg.mode in {"training", "auto"}
    assert cfg.budget.run_budget_usd is not None   # a run budget is set
    validate(cfg)


def test_the_shipped_targeting_calibration_is_a_complete_bound_calibration(cfg):
    """The shipped 10.4.0 config carries a complete, exact-build calibration.

    This assertion used to be its inverse — the file deliberately shipped WITHOUT one, and
    pinning that absence was how we proved an uncalibrated config was still a loadable,
    valid observe configuration. That premise expired the moment a real calibration was
    measured and installed (commit 0c41bb1a), so the pin flips rather than disappears: the
    shipped block must be COMPLETE and BOUND to this exact device and build, because a
    half-written or foreign calibration is the failure this file exists to catch, and
    `validate()` above already refuses one.

    The bounds are asserted against the hard prior-corpus ceilings rather than the values
    measured on any one campaign — those ceilings (2.565 different-profile, 14.91 foreign
    false-match) are the limits ops/RUNBOOK.md states can never be exceeded, so a future
    recalibration may legitimately move the numbers underneath them without touching this.
    """
    cal = cfg.apps["hinge"]["targeting_calibration"]
    assert cal["schema_version"] == 3
    assert cal["device"] == "33111JEHN04475"
    assert cal["device"] == cfg.apps["hinge"]["serial"], "calibration must bind the exact serial"
    assert cal["hinge_version_name"] == "10.4.0"
    assert list(cal["frame_size_px"]) == [1080, 2400]
    assert cal["item_selection_policy_id"] == "hinge_photos_only_v2"
    assert cal["composer_layout_id"] == "hinge_inline_v1"
    assert 0 < cal["identity_match_max_dist"] < 2.565, "hard different-profile ceiling"
    assert 0 < cal["inline_item_max_dist"] < 14.91, "hard foreign false-match ceiling"
    # The effective bands the calibration was measured under must still be the ones in force.
    assert list(cal["identity_band"]) == list(cfg.apps["hinge"]["identity_band"])
    assert list(cal["content_band"]) == list(cfg.apps["hinge"]["content_band"])
    still_photo = cfg.apps["hinge"]["still_photo_assumption_acceptance"]
    assert still_photo["device"] == cal["device"]
    assert still_photo["hinge_version_name"] == cal["hinge_version_name"]


def test_shipped_config_uses_training_until_live_auto_release_is_renewed(cfg):
    assert cfg.enabled_apps == ["hinge"] and cfg.mode == "training"
    assert cfg.limits == {}
    # 2026-08-28: shipped on the persistent transport after a live A/B through the real
    # driver (284s -> 200s on one profile, 143 gestures, zero non-delivery). Both genuine
    # UHID transports are acceptable here; the degraded constant-pressure `adb` one is not,
    # and that is the property this line exists to defend.
    assert cfg.apps["hinge"]["touch_backend"] == "uhid_persistent"
    assert cfg.apps["hinge"]["touch_backend"] in {"uhid", "uhid_persistent"}
    assert "auto_trial" not in cfg.apps["hinge"]
    evidence = cfg.apps["hinge"]["observe_release_evidence"]
    assert evidence["production_run_id"] == "d8547ff144b4"
    assert evidence["hinge_version_name"] == "10.0.1"
    assert evidence["hinge_version_name"] != cfg.apps["hinge"]["targeting_calibration"][
        "hinge_version_name"]
    assert evidence["verification_file"] == (
        "ops/release/d8547ff144b4/manual-release/hinge_observe_release.json")
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
    assert "terminal plain text smiley, :) is allowed only" in style
    assert "never add it by default" in style
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
    assert "primary item rule" in style
    assert "clear main subject of the referenced note, angle, and opener" in style
    assert "other numbered images are selection only" in style
    assert "do not borrow one for the selected opener's premise, predicate, joke, question" in style
    assert "explicit profile text may support a complete natural connection back to the selected item" in style
    assert "justify why the selected item was liked" in style
    assert "reason, subject, or payoff" in style
    assert "unnumbered context image as supporting context" not in style
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
    assert "safety and dignity" in style
    assert "never infer, tease, or pose a forced choice about self harm, suicide" in style
    assert "a bridge, height, water, travel scene, or recognized location is never evidence" in style
    assert "that she jumped or considered jumping" in style
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
    assert "necessary means its exact identity changes how that move is understood" in style
    assert "not merely that it anchors the reaction" in style
    assert "removing a descriptive clause or replacing it" in style
    assert "use the shorter implicit version" in style
    assert "question coherence" in style
    assert "ask one coherent thing at a time" in style
    assert "parallel, genuinely contrasting answers to that same underlying question" in style
    assert "never to join unrelated dimensions" in style
    assert "casual or punctuation" in style
    assert "never put a comma immediately before 'or'" in style
    assert "write it the way a person would text casually" in style
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
    assert "positive social framing" in style
    assert "state the intended positive observation, question, or invitation directly" in style
    assert ("naming an insulting, judgmental, awkward, pressuring, creepy, or offensive "
            "interpretation") in style
    assert "that denial introduces the negative interpretation" in style
    assert "remove the disclaimer" in style
    assert "rewrite the substantive thought so it sounds confident and stands on its own" in style
    assert "reciprocity before future" in style
    assert "not an audition for a role described in her profile" in style
    assert "never answer one of her preferences by advertising the sender" in style
    assert "promising what he will do for her" in style
    assert "do not assume that a match, date, relationship, or shared future already exists" in style
    assert "possessive language about a first date, place, trip, or other future together" in style
    assert "a proposal is not an established shared plan" in style
    assert "information gain test" in style
    assert "conclusion of a guess must not itself be directly visible or explicitly stated" in style
    assert "its header, a sign, or elsewhere in her profile" in style
    assert "they are clues, not guessed conclusions" in style
    assert "ordinary viewer can read or see the conclusion directly without inference" in style
    assert "confirmation boundary" in style
    assert "a guess remains unconfirmed until she replies" in style
    assert "statement, question, compliment, or invitation that assumes it is correct" in style
    assert "natural next move is to confirm or correct it" in style
    assert "only invite that confirmation or correction without presupposing the answer" in style
    assert "experience, preference, or consequence that only makes sense if the guess is true" in style
    assert "confirmation boundary overrides this inheritance rule" in style
    assert "never let a later beat inherit an unconfirmed claim as fact" in style
    assert "asking only whether an inferred location itself is right" in style
    assert "visual location turn boundary" in style
    assert "that location guess is the only conversational move before she replies" in style
    assert "end after the guess, or ask only whether the location itself is right" in style
    assert "her settling it is the whole payoff either way" in style
    assert ("activity, reason, preference, feeling, experience, or consequence at that place"
            in style)
    assert "subject to confirmation boundary, a second sentence may be" in style
    assert "must accept x as its working premise" not in style
    assert "your opener must contain a claim that could be wrong" not in style
    assert "test it by covering the photo" not in style
    assert "then do not say that detail back to her" not in style


def test_shipped_opener_style_ships_the_she_is_the_one_who_knows_rule(cfg):
    """2026-09-14: a live opener asked "Looks like Rome, right?" under a photo of a woman
    standing on an Italian cobblestone street. The owner's objection: she was standing there,
    so asking her to agree that it "looks like" Rome addresses her as a fellow onlooker
    inferring from the same picture, when she is the one who actually knows. VISUAL LOCATION
    TURN BOUNDARY caused it -- "ask only whether the location itself is right" recommends
    exactly that interrogative shape -- and REPLY COMFORT's carve-out exempted it from the
    reply-quality test. Measured: within the 21 proper-name location guesses in the 234-opener
    corpus, the hedge-plus-agreement-tag shape went from 1 of 11 before 2026-09-05 to 5 of 10
    after.

    SHE IS THE ONE WHO KNOWS is the new rule that fixes this at its root: the model is not
    forbidden from guessing a place, only from writing that guess as an appearance the two of
    them are jointly assessing, because she was actually there and the model was not. This is
    the config-side long form; the compressed _SYSTEM twin lives in
    tests/test_opener.py::test_system_prompt_ships_the_she_is_the_one_who_knows_rule and the
    _SCHEMA opener-field twin lives in
    tests/test_gemini_opener.py::test_response_schema_opener_description_ships_the_she_is_the_one_who_knows_rule.
    """
    style = " ".join(cfg.opener.style.lower().split())
    assert "she is the one who knows" in style
    # 2026-09-06 (c) found a rule whose first, affirmative sentence could be satisfied while
    # still producing the defect it forbade. This rule therefore states all THREE operative
    # clauses in its own first sentence, before any rationale; pin that, not just their
    # presence somewhere in the paragraph.
    first_sentence = style.split("she is the one who knows:", 1)[1].split(". ", 1)[0]
    assert "write any inference about it from your own not knowing" in first_sentence
    assert "never ask her to agree about how something appears" in first_sentence
    assert ("never close a claim about her own life with a tag whose only job is to collect "
            "her agreement") in first_sentence
    assert "the shared looking covers the item on the screen, never the world behind it" in style
    assert "she is not working that world out from a picture, because she was in it" in style
    assert "is news to you and old news to her" in style
    assert "write any inference about it from your own not knowing" in style
    assert "never ask her to agree about how something appears" in style
    assert ("never close a claim about her own life with a tag whose only job is to collect "
            "her agreement") in style
    assert "this narrows the shared context rule rather than competing with it" in style
    assert "put the uncertainty in how sure you are, never in how clear the item is" in style
    assert "asking her outright stays welcome" in style
    assert "never invent the sender still forbids saying where he has or has not been" in style


def test_shipped_opener_style_amends_visual_location_turn_boundary_and_reply_comfort(cfg):
    """SHE IS THE ONE WHO KNOWS reaches into its two neighbour rules rather than merely sitting
    beside them (see test_shipped_opener_style_ships_the_she_is_the_one_who_knows_rule for the
    motivating incident). VISUAL LOCATION TURN BOUNDARY's old payoff line ("the confirmation is
    the conversational payoff") is superseded, not merely supplemented -- its return would
    silently reinstate the wording that produced the "right?" agreement-tag shape, so its
    absence is pinned as a MUTATION GUARD alongside the replacement's presence. REPLY COMFORT's
    carve-out for the location confirmation now says explicitly what it does NOT exempt: the
    wording is still governed by SHE IS THE ONE WHO KNOWS.
    """
    style = " ".join(cfg.opener.style.lower().split())
    # VISUAL LOCATION TURN BOUNDARY amendment.
    assert "her settling it is the whole payoff either way" in style
    assert "the confirmation is the conversational payoff" not in style
    assert "neither ending is the default and vary the shape still chooses between the two" in style
    assert "she settles the place because she was standing in it" in style
    assert "never an appearance she is asked to agree with" in style
    # REPLY COMFORT amendment.
    assert "it exempts that confirmation from nothing else" in style

    # De-templating (2026-08-16): the motivating opener and its distinctive wording must never
    # become prompt copy -- naming a literal construction primes a minimal-thinking model to
    # imitate it, the same mechanism the 2026-08-11 "I bet" incident and the 2026-09-06 "elite
    # move" incident both demonstrated. Word-boundary regex on "rome" because a bare substring
    # match could collide with an ordinary word.
    import re
    assert "looks like rome" not in style
    assert not re.search(r"\brome\b", style)
    assert "right?" not in style
    assert '"right"' not in style


def test_shipped_opener_style_defaults_to_minimum_sufficient_visual_reference(cfg):
    """2026-09-08: the attached item can resolve an implicit reference by itself.

    The watched draft and its rewrite stay in tests and the design record, never in the style
    text. This pins the positive transformation rather than a blacklist of scene vocabulary.
    """
    style = " ".join(cfg.opener.style.lower().split())
    assert "minimum sufficient reference" in style
    assert "default to omission or the least explicit natural reference" in style
    assert "replace each literal description of visible content with an implicit reference" in style
    assert "if the meaning, rhythm, and conversational move survive, use the implicit version" in style
    assert "exact identity changes the point or is needed to distinguish possible referents" in style
    assert "never merely to prove grounding, identify the selected item" in style
    assert "attachment itself can make a referent immediately obvious" in style
    assert "counts as the selected item anchor" in style
    assert "full literal inventory in the private referenced and item_description fields" in style
    assert "do not force implicit wording when it would create real ambiguity" in style
    assert "specificity may come from how the message fits the attached item" in style
    assert "subject to minimum sufficient reference" in style
    assert "tiled bench" not in style
    assert "glass of wine" not in style


def test_shipped_opener_style_derives_the_move_without_an_example_menu(cfg):
    """The selected item and profile text determine the move without a stock taxonomy."""
    style = " ".join(cfg.opener.style.lower().split())
    assert "derive the conversational move from the selected item and explicit profile text" in style
    assert "rather than choosing from a fixed taxonomy" in style
    assert "profile text connection is specific only when the selected item supplies an essential part of the thought" in style
    assert "different image supplies its concept, predicate, joke, or payoff" in style
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

    opener.py's _SYSTEM keeps the selection criterion -- the tradeoff and the failure mode named
    as a failure -- in full, compressed but not abridged, pinned by tests/test_opener.py.
    Unnumbered context is now retained only for replay/forensics and withheld from generation.
    The selection instruction lives in exactly one place, and this test pins its absence here
    rather than its presence.
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

    # The item rule remains one instruction in the payload, but context candidates cannot
    # provide a selected opener's thought after the choice has been made.
    assert "other numbered images are selection only" in style
    assert "do not borrow one for the selected opener's premise, predicate, joke, question" in style
    assert "explicit profile text may support a complete natural connection back to the selected item" in style
    assert "unnumbered context image as supporting context" not in style
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
    assert "when a place comes from recognizing the image rather than from her profile text" in style
    assert "present it only as an inference and obey visual location turn boundary" in style
    assert "do not build on an inferred location as though it were correct" in style
    assert "do not state an inferred location as shared experience" in style
    assert "do not turn it into a generic compliment" in style
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


def test_shipped_opener_style_ships_the_spoken_register_rules(cfg):
    """2026-09-05 register rewrite: a live opener passed every rule above and still read
    as AI. The motivating opener (kept off wire per the 2026-08-16 de-templating rule):

        You have a really classic sense of style here. What is your favorite kind of
        spot when you feel like dressing up for a night out?

    and the owner's rewrite of it:

        That's quite a classic style you got there. What's your favorite spot for a
        night out?

    Measured before the rewrite (40 most recent live openers, three sources merged):
    zero apostrophes across all 188 openers ever generated, 21/40 opening with a
    demonstrative verdict, 22/40 ending in a binary or question, 12/40 on a looks like
    frame, 30/40 in one compound mold. These pins hold the register rules in the shipped
    style text; the compressed mirrors are pinned in tests/test_opener.py.
    """
    style = " ".join(cfg.opener.style.lower().split())
    assert "spoken register" in style
    assert "the contractions a relaxed speaker would use" in style
    assert "natural spoken elision" in style
    assert "no chat abbreviations" in style
    assert "idiom fit" in style
    assert "idiom only when it is contemporary" in style
    assert "everyday, immediately understandable on first reading in ordinary conversation" in style
    assert "semantically apt to the item and point" in style
    assert "natural when spoken" in style
    assert "idioms that are dated, literary, formal, obscure, forced, or tied to a passing trend" in style
    assert "does not license internet slang, memes, or borrowed caption wording" in style
    assert "makes her stop to decode the point" in style
    assert "say it once" in style
    assert "never restate a connection or context the message itself already established" in style
    assert "compliment as remark" in style
    assert "rather than a verdict pronounced on her" in style
    assert "it never requires one" in style
    assert "vary the shape" in style
    assert "never because it is the easy mold" in style
    assert "specific item determine the whole message's sentence count, clause pattern" in style
    assert "no structure is the default" in style
    assert "never point at the medium" in style
    assert "rephrase the sentence so no hyphen is needed" in style
    assert "observation announced as what something looks like" not in style
    assert "question offering exactly two choices" not in style
    # De-templating (2026-08-16): the motivating copy above must stay off wire.
    assert "classic sense of style" not in style
    assert "you got there" not in style
    assert "hold court" not in style


def test_shipped_opener_style_ships_the_no_grading_rule(cfg):
    """2026-09-06: COMPLIMENT AS REMARK told the model to move praise off her and onto the
    visible thing, and the model did exactly that -- openers kept landing on a taste verdict
    on the thing itself ("... is an elite move"). Moving the target of the grade does not
    remove the grade; NO GRADING is the rule that forbids grading at all, with the one
    permitted compliment required to be inseparable from a specific observation rather than a
    score. This is the config-side long form; the compressed mirrors belong in
    tests/test_opener.py (the _SYSTEM twin) and tests/test_gemini_opener.py (the schema
    description twin), per the same mirroring contract as the 2026-09-05 register rewrite
    above.
    """
    style = " ".join(cfg.opener.style.lower().split())
    assert "no grading" in style
    assert "substitution test" in style
    # The escape hatch: when nothing but a grade is available, cut the beat rather than invent one.
    assert "cut that beat and let one specific question be the whole message" in style
    # Subordination to COMPLIMENT AS REMARK, not competition with it -- moving praise onto the
    # thing was already correct and must stay; NO GRADING narrows what counts as landing there.
    assert "narrows compliment as remark rather than competing with it" in style
    # 2026-09-06 (b): PLAYFUL HYPERBOLE carve-out -- the SUBSTITUTION TEST targets a literal
    # verdict on quality, so an anchored nonliteral exaggeration is not disqualified merely
    # because its wording could transfer to a different photo.
    assert "the test targets a literal verdict on how good something is" in style
    assert "playful hyperbole can survive the swap without failing" in style
    assert "stays playful framing rather than an assessment of quality" in style
    # 2026-09-06 (b): the one-question fallback is still subject to VARY THE SHAPE, not a
    # standing template of its own.
    assert "that fallback remains subject to vary the shape" in style
    assert "is itself a template, so its construction must still come from the item" in style
    # De-templating (2026-08-16): naming the banned verdict vocabulary on the wire would hand a
    # minimal-thinking model salient wording to imitate -- the same mechanism that produced the
    # 2026-08-11 "I bet" incident. These words belong only in code comments, the design doc, and
    # tests, never in a model-facing string, so the rule text itself must not name any of them.
    assert "elite" not in style
    assert "iconic" not in style
    assert "top tier" not in style
    assert "masterpiece" not in style
    assert "unmatched" not in style
    assert "power move" not in style


def test_shipped_opener_style_folds_qualification_into_compliment_as_remark(cfg):
    """OWNER DECISION 2026-09-06: NARROW COMPLIMENT AS REMARK, do not retire it. Rationale on
    record: an audit measured this rule set at roughly 69% prohibition and mechanical guidance
    against only 21% positive specification, with the corpus at 94.7% exactly two sentences and
    98.9% question final -- the space of legal positive moves is already collapsed, so retiring
    the one remaining permitted compliment would be the wrong direction.

    THE DEFECT: COMPLIMENT AS REMARK and NO GRADING read as a rule and a later correction of it.
    COMPLIMENT AS REMARK affirmatively taught the exact technique NO GRADING goes on to call
    insufficient ("prefer the thing as the sentence's subject", "keep it understated"), and the
    qualification that rescues it lived only in NO GRADING's later "narrows COMPLIMENT AS
    REMARK" sentence (still pinned above in
    test_shipped_opener_style_ships_the_no_grading_rule). Under minimal thinking, a model can
    read and satisfy the first sentence while still producing a grade -- the exact "... is an
    elite move" regression that motivated NO GRADING in the first place.

    THE FIX (2026-09-06 (c)): fold the qualification into COMPLIMENT AS REMARK's own sentence,
    ahead of the technique it gates, so the rule is self contained rather than repaired
    downstream. NO GRADING's own subordination sentence is UNCHANGED (it still does useful work
    stating the precedence explicitly), but the two are no longer readable as an affirmative
    rule and a separate later fix.

    THE (c) FIX'S OWN DEFECT, closed here (2026-09-06 (d)): the folded qualification
    ("cannot survive being detached from what was actually noticed") was itself a second
    portability test with NO exception, while NO GRADING's SUBSTITUTION TEST (still pinned in
    test_shipped_opener_style_ships_the_no_grading_rule) explicitly lets an unmistakably
    nonliteral PLAYFUL HYPERBOLE survive the same swap. An anchored hyperbolic compliment whose
    predicate could transfer to a different photo therefore passed NO GRADING but failed
    COMPLIMENT AS REMARK's own restated test -- opposite verdicts on one sentence under minimal
    thinking. THE FIX: COMPLIMENT AS REMARK no longer restates its own portability test; it
    defers to NO GRADING's SUBSTITUTION TEST directly, so the one exception lives in one place
    per surface and cannot drift out of sync with a second copy again.

    (d)'S OWN DEFECT, closed here (2026-09-06 (g)): (d) fixed only HALF the contradiction.
    COMPLIMENT AS REMARK carries a SECOND, independent requirement beyond portability: MAGNITUDE
    ("stays understated rather than emphatic" / "turns earnest and emphatic"), and that clause
    carried no exception of its own. An unmistakably nonliteral PLAYFUL HYPERBOLE is by nature
    emphatic, so a strong hyperbolic compliment passed the (d)-fixed portability test while
    still failing the untouched magnitude clause -- opposite verdicts again, under the same
    minimal-thinking read.

    (g)'S OWN DEFECT, closed here (2026-09-06 (h)): (g) subordinated only the MAGNITUDE clause,
    so the contradiction relocated a THIRD time, into the untouched "lands sideways ... in
    passing" condition. That is the same mechanism as the original defect: a fix that moves a
    property instead of governing the whole rule gets obeyed and preserves the failure. THE FIX
    hoists the exception ONCE so it governs ALL THREE conditions together rather than being
    bolted onto whichever clause was last reported, and resolves the "lands sideways" case in
    the NON-permissive direction on purpose: that condition is about PLACEMENT rather than
    volume, so it survives the exception and an exaggeration still has to be an aside rather
    than the point. The disqualifier now counts all three conditions instead of saying "either
    requirement", which had gone stale against a three-item list.

    THE STANDING RULE this encodes: after subordinating one clause of a rule, re-walk EVERY
    other clause of that same rule against the same exception. Three same-day adversarial
    rounds each caught exactly one clause and missed the next.
    """
    style = " ".join(cfg.opener.style.lower().split())
    assert "compliment as remark" in style
    assert "moving the praise onto the thing is not enough by itself" in style
    assert "praise also passes no grading's substitution test below" in style
    # MUTATION GUARD: the (c) fold's own restated portability test must be GONE, not merely
    # supplemented -- proving there is no longer a second, exception-free test that could
    # contradict NO GRADING's SUBSTITUTION TEST (whose PLAYFUL HYPERBOLE exception is pinned in
    # test_shipped_opener_style_ships_the_no_grading_rule).
    assert "remains inseparable from a specific observation about that item" not in style
    assert "cannot survive being detached from what was actually noticed" not in style
    # Distinct properties already on wire must survive the fold (narrow, not retire).
    assert "lands sideways the way a friend would mention it in passing" in style
    assert "stays understated rather than emphatic" in style
    assert "this shapes a compliment that occurs" in style
    assert "it never requires one" in style
    # The forward reference ("below") must be literally true.
    assert style.index("compliment as remark") < style.index("substitution test")
    # De-templating (2026-08-16): no banned verdict vocabulary reaches this rule's own text.
    assert "elite" not in style
    # 2026-09-06 (g): the magnitude clause ("stays understated rather than emphatic") now
    # points at the SAME PLAYFUL HYPERBOLE exception portability already has, stated once so it
    # governs both requirements, immediately before the clause it modifies.
    # (h): ONE exception, hoisted to govern every condition at once. Pinning "all three
    # conditions together" is what stops a future edit from re-attaching it to a single clause
    # and leaving the next one exception-free for a fourth time.
    assert "governs all three conditions together" in style
    assert "covering both the portability and the volume" in style
    # "lands sideways" must be resolved in the NON-permissive direction: it survives the
    # exception, so a hyperbolic compliment is still an aside rather than the point.
    assert "landing sideways survives that exception" in style
    assert "never the point of the message" in style
    # MUTATION GUARD: the old, unexcepted disqualifier restatement must be GONE -- its presence
    # would mean a strong hyperbolic compliment could still be rejected on magnitude alone even
    # though it passes the exception above, which is the exact (g) contradiction.
    assert "or that turns earnest and emphatic" not in style
    # The disqualifier must count all three conditions. "either requirement" had gone stale
    # against a three-item list, which is how a reader loses track of which clause is excepted.
    assert "praise that fails any of the three reads as grading her" in style
    assert "praise that fails either requirement" not in style


def test_shipped_opener_style_ships_modifier_clarity_without_incident_copy(cfg):
    """A live Training draft correctly recognized Maja's coat and snowy setting, but this
    word order gave its final location phrase a second plausible attachment:

        You look completely at home bundled up in all that snow.

    That is not a noun-referent failure: the reader knows what every word names, but can read
    either the setting or the nearby clothing phrase as governing the final modifier. The prompt
    needs the general semantic property, never a one-sentence blacklist or model-facing example.
    opener.py's compressed twin is pinned separately so the two inputs sent in one request cannot
    drift apart.
    """
    style = " ".join(cfg.opener.style.lower().split())
    assert "modifier clarity" in style
    assert "every modifying phrase must have only one natural attachment" in style
    assert "on first reading" in style
    assert "if the phrase's placement permits a plausible unintended meaning" in style
    assert "reorder or rephrase the line" in style
    # De-templating: the motivating output above is evidence for the rule, not prompt copy.
    assert "bundled up in all that snow" not in style
    assert "completely at home bundled up" not in style


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
    fallback = cfg.apps["hinge"].get("identity_top_name_fallback_band")
    assert fallback is not None, "apps.hinge.identity_top_name_fallback_band is missing"
    assert tuple(fallback) == (0.03, 0.130, 0.75, 0.235)
    take_another_look = cfg.apps["hinge"].get(
        "identity_top_name_take_another_look_band")
    assert take_another_look is not None, (
        "apps.hinge.identity_top_name_take_another_look_band is missing")
    assert tuple(take_another_look) == (0.03, 0.215, 0.75, 0.285)


def test_normal_auto_is_uncapped_by_default(cfg):
    """Normal AUTO runs until Stop, deck exhaustion, or a safety halt."""
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
