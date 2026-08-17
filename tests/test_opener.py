"""Provider-independent opener logic: the no-dash sanitizer (Corey-Wayne rules),
sentence-count enforcement, image-media-type sniffing, the shared system prompt, and the
two deterministic text helpers the 2026-08-11 opener redesign added (the redundancy
MONITOR, _redundant_description_markers, and the entropy guard's normalizer,
_leading_ngram).

Gemini is the only opener client this project ships (the legacy Anthropic/Claude path
has been removed from operation_love/opener/opener.py entirely -- see
operation_love/opener/service.py and config.py for the corresponding provider-level
enforcement). Most tests below call the module-level helpers directly since they're
pure functions with no provider dependency; a few OpenerParseError edge cases
(missing "opener" key, a non-int item_index, an opener over the two-sentence
cap) are exercised through GeminiOpener with an injected fake transport, the same
technique tests/test_gemini_opener.py uses for its own (much larger) REST/quota/
thinking-config coverage. No SDK/network either way.
"""
import json

import pytest

from operation_love.opener.opener import (
    FIRST_ITEM_INDEX,
    INDEX_SPACE_MODEL_ITEMS,
    INDEX_SPACE_PROFILE_PHOTOS,
    ITEM_INDEX_ABSENT,
    GeminiOpener,
    ItemRequest,
    OpenerParseError,
    OpenerResult,
    REASON_SCAFFOLDING,
    REASON_TOO_MANY_SENTENCES,
    REASON_UNDELIVERABLE_CHARS,
    REASON_UNDELIVERABLE_SEQUENCE,
    _ITEM_PREAMBLE,
    _SYSTEM,
    _image_media_type,
    _leading_ngram,
    _redundant_description_markers,
    _sanitize,
    _scaffolding_markers,
    _sentence_count,
    _strip_wrapping_quotes,
)
from operation_love.costing import Usage
from operation_love.opener.service import OpenerService
from operation_love.perception.capture import Profile
from operation_love.typography import fold_to_ascii


def test_image_media_type_detects_png():
    assert _image_media_type(b"\x89PNG\r\n\x1a\n" + b"rest") == "image/png"


def test_image_media_type_detects_jpeg():
    assert _image_media_type(b"\xff\xd8\xff" + b"rest") == "image/jpeg"


def test_image_media_type_detects_gif():
    assert _image_media_type(b"GIF89a" + b"rest") == "image/gif"
    assert _image_media_type(b"GIF87a" + b"rest") == "image/gif"


def test_image_media_type_detects_webp():
    assert _image_media_type(b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"rest") == "image/webp"


def test_image_media_type_defaults_to_png_for_unknown_bytes():
    assert _image_media_type(b"not an image") == "image/png"
    assert _image_media_type(b"") == "image/png"
    assert _image_media_type(b"\x00\x01") == "image/png"


def test_sanitize_removes_em_dash_and_hyphen():
    out = _sanitize("Matcha and yoga — but cry-in-the-car energy")
    assert "—" not in out and "-" not in out and "–" not in out


def test_sanitize_credentials_lose_the_hyphen():
    assert _sanitize("PA-C energy") == "PA C energy"


def test_sanitize_no_dangling_comma_from_boundary_dash():
    assert _sanitize("your dog—") == "your dog"          # trailing dash -> no trailing comma
    assert _sanitize("—start") == "start"                # leading dash -> no leading comma
    assert _sanitize("Bold move—!") == "Bold move!"      # dash before terminal -> clean
    assert "," not in _sanitize("nice try—")             # no stray comma anywhere


# Every dash-like codepoint a model has been observed to emit (owner hard rule: NO em dashes,
# NO hyphens of any kind in a generated opener -- it's the single biggest AI-written tell).
# Property-style: loop over the codepoints so a newly-encountered dash is a one-line addition.
_ALL_DASH_CODEPOINTS = [
    "—",  # — em dash
    "–",  # – en dash
    "-",  # -  hyphen-minus
    "‐",  # ‐ hyphen
    "‑",  # ‑ non-breaking hyphen
    "‒",  # ‒ figure dash
    "―",  # ― horizontal bar
    "−",  # − minus sign
    "﹘",  # ﹘ small em dash
    "﹣",  # ﹣ small hyphen-minus
    "－",  # － fullwidth hyphen-minus
]


def test_sanitize_strips_every_dash_codepoint():
    for ch in _ALL_DASH_CODEPOINTS:
        out = _sanitize(f"left{ch}right")
        assert ch not in out, f"dash codepoint U+{ord(ch):04X} survived sanitize: {out!r}"


# --- _sanitize is now WYSIWYG: what it returns is byte-for-byte what Adb.text() will type
# (both delegate to typography.fold_to_ascii). An accented name must survive as its readable
# ASCII spelling -- NOT be mutilated or dropped, the bug this whole feature exists to fix.
def test_sanitize_folds_accented_names_to_readable_ascii():
    assert _sanitize("Your trip to São Paulo") == "Your trip to Sao Paulo"
    assert _sanitize("That café looks great") == "That cafe looks great"
    assert _sanitize("Chloé, those antlers") == "Chloe, those antlers"
    assert _sanitize("that jalapeño had a kick") == "that jalapeno had a kick"


# --- _strip_wrapping_quotes: a pure formatting repair (the model quoting its own message
# back), applied silently in _parse rather than routed through _scaffolding_markers -- see
# that function's and _parse's docstrings for why this doesn't burn a retry.
def test_strip_wrapping_quotes_strips_matched_double_quotes():
    assert _strip_wrapping_quotes('"Nice antlers."') == "Nice antlers."


def test_strip_wrapping_quotes_strips_matched_single_quotes():
    assert _strip_wrapping_quotes("'Nice antlers.'") == "Nice antlers."


def test_strip_wrapping_quotes_leaves_unbalanced_inner_apostrophe_alone():
    # The interior apostrophe in "isn't" means the leading/trailing ' are NOT a clean
    # matched wrapping pair -- stripping them naively would leave a broken, unbalanced
    # string, so this must be returned completely unchanged.
    text = "'Nice antlers, isn't it?'"
    assert _strip_wrapping_quotes(text) == text


def test_strip_wrapping_quotes_leaves_unquoted_text_alone():
    assert _strip_wrapping_quotes("Nice antlers.") == "Nice antlers."


def test_strip_wrapping_quotes_leaves_mismatched_quote_pair_alone():
    assert _strip_wrapping_quotes("\"Nice antlers.'") == "\"Nice antlers.'"


# ---------------------------------------------------------------------------------------
# _scaffolding_markers: deterministic (no second LLM call) detector for scaffolding/preamble
# text leaking INSIDE the opener string, e.g. {"opener": "Here's the response: ..."}. The
# JSON schema stops free text OUTSIDE the field but not a meta-clause inside it -- see
# opener.py's module docstring and the function's own docstring for the exact rule list.
# Precision matters far more than recall (a false positive burns a retry, and 5 consecutive
# rejections stop the whole run), so the false-positive corpus below is the more important
# half of this coverage.
# ---------------------------------------------------------------------------------------

_SCAFFOLDING_FALSE_POSITIVES = [
    "Two options: skiing or the beach?",                               # "option" excluded on purpose
    "That golden retriever has better hair than me.",
    "Your Rome photo just moved it up my list.",
    "Fair warning, I message back too fast.",                          # naturally contains "message"
    "Your photo from the trailhead is doing a lot of work for you.",   # naturally contains "photo"
    "The answer is obviously pineapple.",                              # naturally contains "answer"
    "That's a bold sweater, tell me it's not a dare.",                 # internal apostrophe
    'Your caption said "adventure awaits", so where to first?',        # internal quotes
    "Two truths and a lie: you go first or should I?",                 # leading colon, no label word
    "Sunrise hikes and bad coffee, we might be the same person.",
    "That ski photo has main character energy, what's the story?",
    "Best travel story wins, mine involves a scooter and no plan.",
    # "I cannot"/"I'm unable" are natural English far more often than they are a refusal.
    # Matching the bare phrase (as an earlier draft of the rule did) flagged every one of
    # these perfectly in-style openers, and five rejections in a row stop the run -- hence
    # the refusal-verb requirement in _SCAFFOLD_REFUSAL_RE.
    "I cannot believe you skied that line.",
    "I cannot get over that dog's face.",
    "I cannot help but notice the antlers.",                           # "help but" is the idiom
    "Okay, I'm unable to look away from that sunset shot.",
]


@pytest.mark.parametrize("text", _SCAFFOLDING_FALSE_POSITIVES)
def test_scaffolding_markers_false_positive_corpus_is_clean(text):
    assert _scaffolding_markers(text) == []


_SCAFFOLDING_TRUE_POSITIVES = [
    "Here's the response: Great ocean, where was this taken?",         # owner's own example
    "Opener: Nice antlers.",
    'Sure! Here\'s a great opener: "Nice antlers."',
    "As an AI, I would say...",
    "`Nice antlers, great catch.`",                                    # backtick-wrapped
    '{"opener": "Nice antlers, where was this taken?"}',               # raw JSON leak
    "Certainly, here's a good one: love the ski photo, where was that taken?",
    "I cannot assist with that request.",                              # refusal framing
    "# Nice antlers, where was this taken?",
    "This is my **opener** suggestion for her hiking photo.",
]


@pytest.mark.parametrize("text", _SCAFFOLDING_TRUE_POSITIVES)
def test_scaffolding_markers_true_positive_corpus_is_flagged(text):
    assert _scaffolding_markers(text) != []


# ---------------------------------------------------------------------------------------
# _redundant_description_markers: the deterministic (no second LLM call) MONITOR for the
# over-description bug -- content words the opener restated from the model's own `referenced`
# note. Same shape as _scaffolding_markers above: pure function, list of human-readable
# markers, empty list means clean. See ops/OPENER-REDESIGN.md 3.7.
#
# The stakes are DIFFERENT from the scaffolding detector's, and it's worth being precise
# about how. This monitor is log-only -- it never rejects, so a false positive costs no
# retry today and cannot stop a run (test_redundancy_monitor_is_log_only_... below pins
# that). But its entire purpose is to be the number a future gate is calibrated from, and a
# metric that fires on clean openers would put that cutoff in the wrong place before anyone
# noticed. Precision therefore still matters more than recall, exactly as it does above, and
# the false-positive corpus is again the more important (and the longer) half.
#
# Every entry below is a (opener, referenced) pair, in the argument order of the function.
# ---------------------------------------------------------------------------------------

_REDUNDANCY_FALSE_POSITIVES = [
    # The redesign's own worked example (doc 1): the "Yes" half of the sauna edit pair. The
    # claim ("looks relaxing") survives with the photo covered and names nothing visible.
    ("That view looks relaxing, where is this from?",
     "Photo of her in an outdoor sauna at sunset"),
    # Guess: the visible detail is the PREMISE ("that ridgeline"), the point is the country.
    ("Based on that ridgeline I'm going to guess Norway.",
     "Photo of her on a ridge with mountains behind her"),
    # Imagine: a claim about her experience, entirely outside the frame.
    ("I know you were smiling, but I bet you were freezing out there.",
     "Photo of her with a husky in the arctic"),
    # Know, hedged. Note it says "they", not "that husky" -- the pronoun is what keeps it off
    # the monitor, and it is also what makes it read like two people looking at one thing.
    ("I heard they run about as warm as a space heater, so at least you had that.",
     "Photo of her with a husky in the arctic"),
    # STRUCTURAL NOUNS ONLY. Nearly every `referenced` the model writes opens with
    # "photo of..." or "prompt card...", so counting those words would fire on almost every
    # profile and swamp the real signal -- hence the second stopword group.
    ("I bet that took three tries to get right.",
     "The second photo, a picture of a latte with a leaf poured into it"),
    ("You look like you won that argument.",
     "One of her photos, a picture of her mid debate at a podium"),
    ("That is a lot of confidence for a Tuesday.",
     "Her answer to the two truths prompt card"),
    # Possessive in `referenced` ("dog's") must not leave a fragment that matches anything.
    ("Tell me you did not carry that the whole way up.",
     "Photo of her dog's pack on a summit trail"),
    # Sub-3-character tokens are dropped whether or not they are stopwords ("DJ" here), which
    # is the correct trade for a monitor: a two-character overlap is noise far more often
    # than it is evidence.
    ("I bet that is louder than it looks.",
     "Photo of her DJ setup at home"),
    # A bare digit run in `referenced` ("30") is likewise below the length floor.
    ("I am going to guess that was colder than it looks.",
     "Photo taken at 30 below in Tromso"),
    # Shared FUNCTION words ("where", "she", "the") must never count -- they say nothing
    # about which profile this is.
    ("Where does someone even learn that?",
     "Prompt card where she says she taught herself to weld"),
    ("Fair warning, I message back too fast.",
     "Prompt card about her slow reply times"),
    # Degenerate input: an empty `referenced` has no content words to restate.
    ("I bet the dog picked that hat.", ""),
]


@pytest.mark.parametrize("opener, referenced", _REDUNDANCY_FALSE_POSITIVES)
def test_redundancy_markers_false_positive_corpus_is_clean(opener, referenced):
    assert _redundant_description_markers(opener, referenced) == []


_REDUNDANCY_TRUE_POSITIVES = [
    # The shipped opener this entire redesign exists to stop (doc 1), and the "No" half of
    # the sauna edit pair: cover the photo and nothing is left.
    ("That view by the sauna during sunset looks relaxing, where is this from?",
     "Photo of her in an outdoor sauna at sunset"),
    # Synthetic illustration of a failure shape production data confirmed is real, not just
    # hypothetical (doc 3.7). The second clause is a genuine claim; redundantly naming the
    # location is exactly the pattern being targeted here.
    ("That water at the alpine lake looks freezing, were you brave enough to jump all the way in?",
     "Photo of her swimming in the freezing water at the alpine lake"),
    # The "No" half of the husky edit pair: a null verdict plus a near-certain question.
    ("That husky is too cute, is it yours?",
     "Photo of her with a husky in the arctic"),
    # The "No" half of the prompt-card edit pair: reads the card back to the woman who wrote it.
    ("Your two truths and a lie about the countries, cilantro and the pop star is fun, "
     "which one is the lie?",
     "Prompt card, two truths and a lie about 30 countries, cilantro, and meeting a pop star"),
    # Minimal case: exactly one restated content word is still a restatement.
    ("The antlers are doing a lot of work in that shot.",
     "Photo of her wearing antlers at a party"),
    # Normalization is applied to BOTH sides, so case never hides a restatement.
    ("NORWAY looks unreal in that light.",
     "Photo of a fjord in norway"),
    ("Your rooftop sunset photo is a whole mood.",
     "Photo of a sunset from a rooftop bar"),
]


@pytest.mark.parametrize("opener, referenced", _REDUNDANCY_TRUE_POSITIVES)
def test_redundancy_markers_true_positive_corpus_is_flagged(opener, referenced):
    assert _redundant_description_markers(opener, referenced) != []


def test_redundancy_markers_name_each_distinct_word_once_in_referenced_order():
    """The marker STRING is what gets printed and (via OpenerResult.redundancy_markers) put
    in front of whoever calibrates the future threshold, so its shape is pinned exactly: one
    marker per DISTINCT restated word, ordered by first appearance in `referenced` rather
    than in the opener, with the word quoted so a multi-word marker list stays readable."""
    markers = _redundant_description_markers(
        "That view by the sauna during sunset looks relaxing, where is this from?",
        "Photo of her in an outdoor sauna at sunset")
    assert markers == ['opener restates the referenced word "sauna"',
                       'opener restates the referenced word "sunset"']
    # A word restated twice in the opener is still exactly one marker.
    assert _redundant_description_markers("Sauna today, sauna tomorrow.", "an outdoor sauna") == [
        'opener restates the referenced word "sauna"']


def test_redundancy_markers_is_a_lower_bound_and_fails_by_missing_not_by_alarming():
    """Documented limitations, pinned as tests rather than left in prose, because they are
    the reason the monitor is described as a LOWER BOUND on redundancy (doc 3.7 reason 1) and
    therefore cannot ever be the primary defence -- the prompt is the fix, this only measures
    whether the fix worked. Each case below is a MISS (redundancy present, nothing reported),
    never a false alarm on a clean opener, which is the direction a monitor should fail in.
    If any of these ever starts returning markers that is an improvement, but it invalidates
    any threshold calibrated before it, so it must not happen silently."""
    # 1. A terse `referenced` defeats it completely: the opener describes the scene in full
    #    and there is nothing to match against.
    assert _redundant_description_markers(
        "That outdoor sauna at sunset looks unreal, where is it?", "the third photo") == []
    # 2. It counts words, not meaning, so a paraphrase scores zero.
    assert _redundant_description_markers(
        "That steam room looks like the best part of the trip.",
        "Photo of her in an outdoor sauna at sunset") == []
    # 3. Exact-token match only, so morphology defeats it ("huskies" against "husky").
    assert _redundant_description_markers(
        "I heard huskies run warm.", "Photo of her with a husky in the arctic") == []


def test_redundancy_markers_fires_on_a_good_connect_opener_which_is_why_it_is_not_a_gate():
    """The other half of the honesty, and the sharper reason this must stay log-only: the
    monitor also fires on openers the design explicitly holds up as CORRECT. Both cases below
    are verbatim "Yes" examples from ops/OPENER-REDESIGN.md 3.3.

    Connect is the strongest move in the design (doc 2.3) and it is structurally guaranteed
    to trip a word-overlap metric, because combining two things she said in two different
    places means naming both halves. A word-overlap count cannot tell "you restated the photo
    she is staring at" apart from "you linked her food prompt to her backpacking photo".

    So: promoting this to a gate on a threshold naively fitted to the true-positive corpus
    above would reject the best openers the system can write, and five consecutive rejections
    stop the run (service.py). Whoever calibrates the offline threshold must hold these two
    out as the false-positive set."""
    assert _redundant_description_markers(
        "The cilantro one is the lie, I can feel it.",
        "Prompt card, two truths and a lie about 30 countries, cilantro, and meeting a pop "
        "star") != []
    assert _redundant_description_markers(
        "How many bottles of hot sauce did that trip cost you?",
        "Her prompt says she puts hot sauce on everything, plus a separate photo of her "
        "backpacking") != []


# ---------------------------------------------------------------------------------------
# _leading_ngram: the normalizer behind the entropy guard (ops/OPENER-REDESIGN.md 3.6).
# Shortening the openers compresses the output space and few-shot examples make direct
# copying a live risk, so across a burner account sending uncapped volume the same opening
# words recurring are both a fingerprint and embarrassing if two matches compare
# screenshots. The guard is a plain string comparison of this value against the recent
# openers ring buffer -- no semantics, no taxonomy of moves, no second model call.
#
# This function is deliberately free of POLICY: what counts as a collision, what happens on
# one, and how a collision interacts with the attempt budget all live in
# OpenerService._apply_entropy_guard and are tested in tests/test_opener_service.py. What is
# pinned here is only the definition of the thing being compared, because that definition is
# what makes two differently-punctuated copies of the same opening register as the same.
# ---------------------------------------------------------------------------------------

def test_leading_ngram_takes_the_first_four_words_lowercased_without_punctuation():
    assert _leading_ngram("Based on that ridgeline I'm going to guess Norway.") == \
        "based on that ridgeline"
    assert _leading_ngram("I bet you were freezing out there.") == "i bet you were"


def test_leading_ngram_n_is_configurable_and_a_short_text_returns_all_its_words():
    assert _leading_ngram("Based on that ridgeline I'm going to guess Norway.", 2) == "based on"
    assert _leading_ngram("one two three four five six", 6) == "one two three four five six"
    # Fewer words than n: return what there is rather than padding or failing. This is the
    # case that makes two openers of "Hi" collide with each other, which is correct.
    assert _leading_ngram("Hi!", 6) == "hi"
    assert _leading_ngram("Hi there!", 6) == "hi there"


def test_leading_ngram_collapses_the_noise_two_near_identical_openers_differ_by():
    """The whole point: openers that differ only in case, punctuation or whitespace must
    normalize to the SAME n-gram, or the guard misses the repetition it exists to catch."""
    assert _leading_ngram("I BET you were freezing") == _leading_ngram("i bet, you were freezing!")
    assert _leading_ngram("  I bet,   you were   freezing ") == "i bet you were"
    assert _leading_ngram("I bet -- you were freezing") == "i bet you were"
    # ...while a genuinely different opening must NOT collide, or the guard would regenerate
    # openers for no reason and burn a billed draw each time.
    assert _leading_ngram("I bet you were freezing") != _leading_ngram("That view looks relaxing")


def test_leading_ngram_keeps_apostrophes_inside_words_but_not_at_the_edges():
    """Contraction handling is load-bearing, not cosmetic: the recurring phrase this guard
    was written to catch is "I'm going to guess". Splitting on the apostrophe would leave an
    "i"/"m" fragment pair polluting the n-gram, and would make "I'm going" and "I am going"
    collide with each other. Quote characters wrapping a word are still stripped."""
    assert _leading_ngram("I'm going to guess Norway") == "i'm going to guess"
    assert _leading_ngram("'I'm going to guess'", 3) == "i'm going to"
    assert _leading_ngram("I'm going") != _leading_ngram("I am going")


def test_leading_ngram_degenerate_inputs_return_empty_string_and_never_raise():
    """It runs inside the guard on every successful opener, so a degenerate value must
    degrade to "" (which the service treats as "no comparison possible") rather than raise
    and cost a profile an opener that was already good enough to send."""
    assert _leading_ngram("") == ""
    assert _leading_ngram("   ") == ""
    assert _leading_ngram("...!?") == ""
    assert _leading_ngram("anything at all", 0) == ""
    assert _leading_ngram("anything at all", -3) == ""


def test_leading_ngram_is_pure_and_deterministic():
    text = "Based on that ridgeline I'm going to guess Norway."
    assert _leading_ngram(text) == _leading_ngram(text)


def test_sentence_count_basic_cases():
    assert _sentence_count("One profile-specific thought") == 1
    assert _sentence_count("One thought. One easy question?") == 2
    assert _sentence_count("Dr. Dolittle energy. What's the story?") == 2   # abbreviation guard
    assert _sentence_count("One. Two? Three!") == 3


def test_system_prompt_keeps_faithful_corey_opener_policy_and_two_sentence_cap():
    """_SYSTEM is sent verbatim by GeminiOpener (see test_gemini_opener.py's request-shape
    assertions); its actual wording is checked here directly, once, independent of any
    provider's request format.

    Rewritten 2026-08-11 for ops/OPENER-REDESIGN.md Part A. The Corey Wayne framing, the
    90/10 spirit, both HARD RULEs and the two-sentence ceiling are unchanged and still
    pinned below. Three assertions were REPLACED rather than dropped, each with the
    replacement that covers the new behaviour sitting directly beneath it -- see the inline
    comments. config.yaml's opener.style carries a near-duplicate of the Part A wording below
    (the one rule, the five moves, the guardrails, ...) and is pinned separately by
    tests/test_config_yaml_real.py; the two are sent in the SAME request, so an edit that
    lands here and not there hands the model two contradictory style guides.

    ITEM SELECTION is the one exception: _SYSTEM is the sole copy of the instruction. The
    duplicate was removed from config.yaml; see
    test_shipped_opener_style_does_not_ship_the_item_selection_rule for the other half.
    """
    lowered = _SYSTEM.lower()
    assert "90/10 framework" in _SYSTEM
    assert "genuinely curious" in lowered
    assert "do not force teasing into every opener" in lowered
    assert "positive, fun conversation" in lowered
    assert "brief greeting is optional" in lowered
    assert "two sentences is the absolute maximum" in lowered
    assert "never use an em dash or any hyphen" in lowered
    assert "plain ascii letters and punctuation" in lowered   # WYSIWYG / no-emoji rule
    assert "no emoji" in lowered
    assert "low investment so she chases" not in lowered
    assert "tease her like a bratty little sister" not in lowered

    # Bug report 2026-08-16: unconditional falsifiability forced invented motives on items that
    # offered no natural guess. A visible detail is legal setup; description only fails when it
    # remains the final payoff. Claims are optional and minimum invention wins.
    assert "exactly one concrete detail" not in lowered
    assert "conversational value rule" in lowered
    assert "a visible detail may be named and may be the subject, premise, or setup" in lowered
    assert "final conversational point is merely that description" in lowered
    assert "perspective, grounded interpretation, playful framing, connection, or natural question" in lowered
    assert "does not have to contain a guess or a claim that could be wrong" in lowered
    assert "claims only when natural" in lowered
    assert "a correctable inference is one available move, not a requirement" in lowered
    assert "prefer a grounded observation or specific question over a forced guess" in lowered
    assert "write the whole proposition rather than a shorthand answer" in lowered
    assert "choose the least speculative interpretation" in lowered
    assert "minimum invention" in lowered
    assert "never invent a purpose, motive, cause, plan, sequence, effort, goal" in lowered
    assert "do not assign why she chose or did something unless her profile states it" in lowered
    assert "any literal premise must be plainly visible in the item or explicitly stated" in lowered
    assert "never use one invented fact as the premise for another" in lowered
    assert "a setting or destination does not establish how she arrived" in lowered
    assert "what effort it took, or whether she pursued a goal" in lowered
    assert "a hedge does not rescue a far-fetched premise" in lowered
    assert "ownership, employment, a routine, a responsibility, or a relationship" in lowered
    assert "traceability test" in lowered
    assert "for any inference, she should instantly see which visible or stated clue led there" in lowered
    assert "playful hyperbole" in lowered
    assert "unmistakably nonliteral exaggeration is allowed" in lowered
    assert "does not license presenting an invented motive, circumstance, or event as literal fact" in lowered
    assert ("shared context rule: your message is displayed directly under the exact photo "
            "or prompt it attaches to") in lowered
    assert "she is looking at that item while she reads your words" in lowered
    assert "photo header rule" in lowered
    assert "title, caption, or prompt printed with a photo is part of that same item" in lowered
    assert "defines how the photo is meant to be read" in lowered
    assert "interpret the visible scene through that text before choosing an angle" in lowered
    assert "never contradict, reverse, or ignore the header's framing" in lowered

    # A question may stand alone when it is more natural than a claim. Premise consistency still
    # applies whenever a two-beat opener does begin with a claim.
    assert "one open, easy-to-answer question" not in lowered
    assert "a natural claim she can correct can be effective, but it is not mandatory" in lowered
    assert "question may be the whole message when that is the strongest natural angle" in lowered
    assert "never as the whole message" not in lowered
    assert "setup payoff continuity" in lowered
    assert "every visible detail you name must be necessary to, and used by" in lowered
    assert "if removing a descriptive clause leaves the later point or question unchanged, cut it" in lowered
    assert "question coherence" in lowered
    assert "ask one coherent thing at a time" in lowered
    assert "parallel, genuinely contrasting answers to that same underlying question" in lowered
    assert "never to join unrelated dimensions" in lowered
    assert "premise consistency" in lowered
    assert "if the first beat asserts or guesses x" in lowered
    assert "must accept x as its working premise and move forward from it" in lowered
    assert "never ask whether x itself was true" in lowered
    assert "ask about the opposite of x" in lowered
    assert "abandon x for a generic question about the surrounding scene" in lowered
    assert "may extend the angle with clearly nonliteral hyperbole" in lowered
    assert "may not add a literal invented fact, motive, or backstory" in lowered

    # --- LENGTH: the ceiling stays, the one-sentence PREFERENCE is gone (doc 3.2.1).
    # Preferring brevity for its own sake fought a claim that needs room to exist ("I know
    # you were smiling, but I bet you were freezing out there" is eighteen words and is the
    # strongest opener in the whole design). A regression here silently reinstates it.
    assert "one short sentence is preferred" not in lowered
    assert "application rule: two sentences is the absolute maximum" in lowered
    assert "as short as the angle allows" in lowered
    assert "spend no word merely repeating what she can already see" in lowered
    assert "never cut necessary setup or the conversational payoff" in lowered

    # --- THE THREE GUARDRAILS (doc 2.4). Safety rules rather than style preferences, which
    # is why they are stated in BOTH copies of the prompt rather than only in config.yaml.
    #
    # Addendum 2026-08-15: widening a two-form list to seven did not create contextual freedom;
    # it created seven salient templates and shipped "my money is on" as awkward copy. Pin the
    # semantic requirement and the absence of that stock menu instead. The item and angle, not
    # rotation through an inventory, choose the construction.
    assert "hedge the claim, never yourself" in lowered
    assert "when a claim is uncertain, express that uncertainty naturally" in lowered
    assert "wording that fits the specific item" in lowered
    assert "the goal is calibrated uncertainty, not a particular lead in" in lowered
    assert "this is not a phrase menu" in lowered
    assert "choose the construction from context" in lowered
    assert "never use a hedge as a substitute for the complete self contained claim" in lowered
    assert "my money is on" not in lowered
    assert "vary the opening" in lowered
    assert "let the specific item and angle determine the wording" in lowered
    assert "do not rotate or recycle a fixed stock hedge" in lowered
    assert "never open every message the same way" in lowered
    assert "never apologise for writing" in lowered
    assert "never call your own question dumb" in lowered
    assert "guess the world, not her identity" in lowered
    assert "name a country, a region, or a park" in lowered
    assert "never guess her employer, her school, or her age" in lowered
    assert "never invent the sender" in lowered
    assert "you may not claim he has been somewhere" in lowered

    # --- FIELD ROUTING (doc 1.1 root cause #2 and #3). The schema always had the right place
    # to put the description and nothing ever told the model to route it there, so it wrote
    # the description twice. thinkingLevel is minimal on every model in the cascade, so these
    # earlier output fields are the only scratchpad that exists.
    assert ("fill item_index, referenced, angle and item_description before you write the "
            "opener") in lowered
    assert "is never sent to her" in lowered
    assert "put the literal inventory there" in lowered
    assert "message may use only the setup it needs" in lowered
    assert "must add a conversational payoff" in lowered
    assert "angle is your own short wording for what your opener is doing" in lowered
    assert "item_description says in a few words what the item you picked is" in lowered

    # --- ITEM SELECTION (doc 5.1/5.7). The model CHOOSES the item now, so _SYSTEM has to state
    # the three facts the new contract rests on: the numbering is 1-based over the items in
    # this request, the chosen item is also the one that gets liked (one call, doc 5.1), and
    # unnumbered context blocks may be read but never picked (the two tiers, doc 5.3).
    # Selection is by best ANGLE, not best photo -- 5.1's criterion, and the reason a good
    # opener about a mediocre photo beats a great photo with nothing to say about it.
    assert "pick the item yourself" in lowered
    assert "numbered from 1 in the order they are given" in lowered
    assert "you choose which one to write about" in lowered
    assert "not the most striking picture" in lowered
    assert "it is also the item that gets liked" in lowered
    assert "given without a number is context" in lowered
    assert "never pick it" in lowered

    # --- THE SELECTION CRITERION, spelled out rather than implied (doc 5.1). "Best angle, not
    # most striking photo" is a TRADEOFF, and a model looking at a page of photographs has a
    # strong prior toward the best photograph, so the preference alone does not settle it. It is
    # THE ONE RULE applied one step earlier: the item is only the premise, so choosing by how the
    # item looks optimises the half that never becomes the message.
    assert "the item you have the best conversational angle on, not the most striking picture" in lowered
    assert "natural observation, question, connection, or playful framing" in lowered
    assert "beats a beautiful one you have nothing to say about" in lowered
    assert "the item supplies the material and the conversational payoff is the message" in lowered
    assert "pick that one even when another item is the better picture" in lowered
    # The inverted criterion must never come back: instructing the model to pick by how the item
    # LOOKS is the exact regression this block exists to catch, and it would not contradict any
    # other assertion in this file.
    for _inverted in ("the best photo", "the most striking", "the most interesting photo",
                      "the most eye catching"):
        assert f"pick {_inverted}" not in lowered, f"_SYSTEM selects by appearance: {_inverted!r}"
        assert f"choose {_inverted}" not in lowered, f"_SYSTEM selects by appearance: {_inverted!r}"

    # --- THE FAILURE MODE, named as a failure (doc 5.1). Not a restatement of THE ONE RULE:
    # picking a striking item with nothing to say about it is the upstream CAUSE of
    # over-description, reachable with every Part A wording rule obeyed, because once the item
    # is chosen and no claim is available, description is the only material left. The remedy has
    # to be stated at the stage it goes wrong, i.e. pick a different item.
    assert "the failure to avoid is picking a photo you have nothing to say about" in lowered
    assert "all that is left to write is what it looks like" in lowered
    assert "never enough as the final point" in lowered

    # --- THE CONTEXT TIER, described rather than merely permitted (doc 5.3). The unnumbered
    # block is her vitals, confirmed from a live capture to carry no like heart, so it is
    # referenceable and unpickable. Saying only "you may not pick it" would leave the strongest
    # use of the strongest move (doc 2.3: combining two things she said in different places,
    # which nobody who read one card could have written) unstated.
    assert "usually her vitals: her age, her job, her school, her city" in lowered
    assert "set against a numbered item, is often the best angle on the page" in lowered
    assert "item_index can never refer to it" in lowered
    # The replaced field is gone from the copy entirely, along with the index space it named:
    # doc 5.7 calls out that both the field and "every line of prompt copy saying scroll order"
    # change meaning, and a leftover instruction to fill referenced_index would ask the model
    # for a value the schema no longer declares.
    assert "referenced_index" not in lowered
    assert "scroll order" not in lowered

    # --- Owner invariant, not a wording choice: this text is itself sent to the model.
    assert all(ord(ch) < 128 for ch in _SYSTEM), "_SYSTEM must be pure ASCII"
    assert "—" not in _SYSTEM


_MOVE_LIST_MARKERS = (
    "ways this tends to look",
    "say something you know that the item brought to mind",
    "claim something outside the frame",
    "connect two things she said",
)

_BINDING_MENU_PHRASINGS = (
    "choose one of",
    "pick one of",
    "use one of the following",
    "must use one of",
    "select a move",
    "one of these moves",
)


def test_system_prompt_contains_no_move_menu_or_concrete_opener_copy():
    lowered = _SYSTEM.lower()
    for phrase in _BINDING_MENU_PHRASINGS:
        assert phrase not in lowered, f"_SYSTEM presents the moves as a closed menu: {phrase!r}"
    for marker in _MOVE_LIST_MARKERS:
        assert marker not in lowered
    for copied_phrase in (
        "froze your butt off",
        "favorite person the second the snacks came out",
        "jelly after climbing all those stairs",
        "based on that ridgeline",
    ):
        assert copied_phrase not in lowered


def test_item_preamble_binds_a_photo_header_to_its_numbered_photo():
    """The crop contains both regions; the wire prompt must make them one meaning."""
    lowered = _ITEM_PREAMBLE.lower()
    assert "title, caption, or prompt above its photo" in lowered
    assert "one compound item" in lowered
    assert "must be read together" in lowered


# ---------------------------------------------------------------------------------------
# OpenerParseError edge cases, exercised through GeminiOpener + an injected fake transport
# (GeminiOpener is the only opener client shipped; these parsing rules live in
# GeminiOpener._parse but are provider-independent in spirit -- item_index coercion,
# a required "opener" key, and the two-sentence cap all trace back to the shared _SYSTEM
# contract and _sentence_count/_sanitize above).
# ---------------------------------------------------------------------------------------

class _Transport:
    def __init__(self, responses):
        self.responses = iter(responses)
        # Call count, so a test can assert that a given response did NOT cost a retry.
        # (A rejection is only visible as a second request: OpenerService catches
        # OpenerParseError and asks again, so "was this rejected?" and "how many requests
        # were made?" are the same question from outside.)
        self.calls = 0

    def __call__(self, url, payload, headers, timeout, *, method="POST"):
        self.calls += 1
        return next(self.responses)


def _gemini_response(structured: dict) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps(structured)}]}}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
    }


def _opener(transport) -> GeminiOpener:
    return GeminiOpener(["gemini-test"], api_key="test-key", transport=transport)


@pytest.mark.parametrize("bad_value", ["notanint", None, [], {}, -3, 0], ids=[
    "string", "null", "list", "dict", "negative", "zero"])
def test_generate_defaults_an_unusable_item_index_to_the_absent_value(bad_value):
    """Every way the model can fail to give a usable item number collapses to the SAME
    out-of-band value (ITEM_INDEX_ABSENT), and none of them costs the profile its opener --
    _SCHEMA's "required" list is a generation hint, not something the API enforces on the
    response.

    Zero and negatives are in this list on purpose: item numbering starts at 1
    (ops/OPENER-REDESIGN.md 5.7), so they are not small valid indices, they are "no answer".
    Under the old 0-based referenced_index the identical coercion produced a confident,
    perfectly legal "the first item", which is exactly what made a silently reinterpreted
    integer dangerous (doc 5.3)."""
    payload = _gemini_response({"opener": "hi", "referenced": "x", "item_index": bad_value})
    result = _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert result.item_index == ITEM_INDEX_ABSENT
    assert result.opener == "hi"                # the opener itself is never at risk


def test_generate_defaults_a_missing_item_index_to_the_absent_value():
    """A response with no item_index key at all -- the field absent rather than unusable."""
    payload = _gemini_response({"opener": "hi", "referenced": "x"})
    result = _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert result.item_index == ITEM_INDEX_ABSENT


def test_item_index_is_one_based_and_never_shifted_on_the_way_through():
    """The number the model returns is the number this pipeline carries: no +1, no -1, no
    conversion to a 0-based space anywhere in _parse. A "helpful" adjustment here is the
    silent reinterpretation doc 5.3 exists to prevent, and it would be invisible -- both
    values are small ints and neither carries its base in its type.

    Both profiles below carry enough photos for the returned number to name a real one: an
    index past the end of what was actually sent is a different case entirely and is refused
    (see test_item_index_beyond_what_was_sent_is_refused_as_absent)."""
    payload = _gemini_response({"opener": "hi", "referenced": "x", "item_index": 1})
    assert _opener(_Transport([(200, payload)])).generate(
        Profile(photos=[b"a"]), style="s").item_index == 1

    payload = _gemini_response({"opener": "hi", "referenced": "x", "item_index": 9})
    assert _opener(_Transport([(200, payload)])).generate(
        Profile(photos=[b"a"] * 9), style="s").item_index == 9
    assert FIRST_ITEM_INDEX == 1 and ITEM_INDEX_ABSENT == 0


# --- the range check: an item number that names nothing we sent ----------------------------
# Nothing downstream can catch this. The driver's own bounds check counts its captured FRAMES,
# of which there are always more than there are items, so an out-of-range item number sails
# through it and lands on a real, wrong heart. generate() is the only place the request and
# the response are both in scope, which is why the check lives there.

@pytest.mark.parametrize("returned", [2, 12, 999])
def test_item_index_beyond_what_was_sent_is_refused_as_absent(returned):
    """One photo was sent, so item 1 is the only legal answer. Anything above it names an item
    that does not exist and must collapse to ITEM_INDEX_ABSENT -- never be passed on as a
    plausible small integer -- while leaving the opener text alone."""
    payload = _gemini_response({"opener": "hi", "referenced": "x", "item_index": returned})
    result = _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert result.item_index == ITEM_INDEX_ABSENT
    assert result.opener == "hi"                # a numbering slip never costs the opener


@pytest.mark.parametrize("returned, expected", [
    (3.7, ITEM_INDEX_ABSENT),   # truncation would INVENT item 3; 3.7 names neither 3 nor 4
    (3.0, 3),                   # an integral float names exactly one item, so it is kept
    (True, ITEM_INDEX_ABSENT),  # bool is an int subclass, so int(True) would give "item 1"
    (False, ITEM_INDEX_ABSENT),
    ("3", 3),                   # a decimal string names exactly one item; nothing is invented
    ("3.7", ITEM_INDEX_ABSENT),  # int() raises rather than truncating, which is the right answer
])
def test_only_values_that_unambiguously_name_an_item_survive_as_a_pick(returned, expected):
    """`int()` will happily turn things that never meant an item number into a plausible one,
    and under 1-based numbering a plausible number is worse than a refused one: it names a REAL
    card. None of these is reachable from a schema-conforming model, which is exactly why the
    old truncation was harmless under the 0-based contract and is worth closing under this
    one."""
    payload = _gemini_response({"opener": "hi", "referenced": "x", "item_index": returned})
    result = _opener(_Transport([(200, payload)])).generate(
        Profile(photos=[b"a", b"b", b"c"]), style="s")
    assert result.item_index == expected
    assert result.opener == "hi"                # never at the cost of the opener itself


def test_item_index_at_the_exact_end_of_the_sent_range_is_kept():
    """The boundary in the other direction: N items sent, item N returned. Off-by-one in the
    check itself would silently disable the last item on every profile."""
    payload = _gemini_response({"opener": "hi", "referenced": "x", "item_index": 3})
    result = _opener(_Transport([(200, payload)])).generate(
        Profile(photos=[b"a", b"b", b"c"]), style="s")
    assert result.item_index == 3


def test_item_index_is_refused_when_no_numbered_items_were_sent_at_all():
    """A profile with no photos and no anchor carries no numbered items, and the prompt says
    so and asks for ITEM_INDEX_ABSENT. A model that answers 1 anyway is naming an item that
    was never in the request, so it is refused rather than believed."""
    payload = _gemini_response({"opener": "hi", "referenced": "x", "item_index": 1})
    result = _opener(_Transport([(200, payload)])).generate(Profile(), style="s")
    assert result.item_index == ITEM_INDEX_ABSENT


# --- index_space: the fact a bare small int cannot carry -------------------------------------

def test_generate_records_which_list_the_item_index_counts():
    """`item_index` is meaningless without the list it indexes, and the 2026-08-12 seam bug was
    a consumer assuming that list. generate() states it, derived from the payload it actually
    built, so no consumer has to guess."""
    payload = _gemini_response({"opener": "hi", "referenced": "x", "item_index": 1})
    frames = _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert frames.index_space == INDEX_SPACE_PROFILE_PHOTOS

    payload = _gemini_response({"opener": "hi", "referenced": "x", "item_index": 1})
    items = _opener(_Transport([(200, payload)])).generate(
        Profile(), style="s",
        items=ItemRequest(name="A", items=[b"crop-1"], context=[], truncated=False))
    assert items.index_space == INDEX_SPACE_MODEL_ITEMS


def test_an_openerresult_built_by_hand_defaults_to_the_untranslatable_space():
    """Fakes and future call sites do not have to know about index spaces, but their picks must
    not be convertible into a tap. The default is the space nothing can translate, so being
    wrong here means "no target", never "target 1"."""
    assert OpenerResult("hi", "x", Usage(0, 0), "m").index_space == INDEX_SPACE_MODEL_ITEMS


def test_item_description_is_parsed_defensively_like_angle():
    """Trimmed, coerced to str, and degraded to "" when missing or unusable -- never a reason
    to fail a request. Doc 5.7 requires it in BOTH modes, so it is parsed unconditionally;
    nothing here branches on advisory/auto."""
    payload = _gemini_response({"opener": "hi", "referenced": "x", "item_index": 2,
                                "item_description": "  a photo of a ridge  "})
    result = _opener(_Transport([(200, payload)])).generate(
        Profile(photos=[b"a", b"b"]), style="s")
    assert result.item_description == "a photo of a ridge"

    for bad_value in (None, 0, [], {}):
        payload = _gemini_response({"opener": "hi", "referenced": "x", "item_index": 2,
                                    "item_description": bad_value})
        result = _opener(_Transport([(200, payload)])).generate(
            Profile(photos=[b"a", b"b"]), style="s")
        assert result.item_description == ""
        assert result.item_index == 2           # a bad description never disturbs the pick


def test_generate_raises_parse_error_on_missing_opener_key():
    payload = _gemini_response({"referenced": "x", "item_index": 1})   # no "opener" key
    with pytest.raises(OpenerParseError):
        _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")


def test_generate_raises_parse_error_when_opener_exceeds_two_sentence_maximum():
    payload = _gemini_response({"opener": "One. Two? Three!", "referenced": "x", "item_index": 1})
    with pytest.raises(OpenerParseError, match="two-sentence maximum") as exc:
        _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    # Machine-readable reason_code lets a persisted rejection row be grouped/queried without
    # regex-parsing the human message -- see OpenerParseError's own docstring for the full
    # reason_code/raw_opener contract. raw_opener here is the SANITIZED text (this guard runs
    # on `sanitized`, and sanitize is a no-op on plain ASCII like this fixture).
    assert exc.value.reason_code == REASON_TOO_MANY_SENTENCES
    assert exc.value.raw_opener == "One. Two? Three!"


# --- WYSIWYG: an opener the model wrote that still can't be typed after _sanitize's
# fold_to_ascii pass (almost always an emoji -- an accented name folds cleanly, see
# test_sanitize_folds_accented_names_to_readable_ascii above) must be a loud, named
# OpenerParseError, not a silently-sent opener that later fails (or worse, silently
# truncates) at the device-input boundary (drivers.adb.Adb.text). See opener.py's _parse.
def test_generate_raises_parse_error_naming_an_emoji_in_the_opener():
    payload = _gemini_response({"opener": "Love the \U0001F384 vibes", "referenced": "x",
                                "item_index": 1})
    with pytest.raises(OpenerParseError, match="CHRISTMAS TREE") as exc:
        _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert "\\U0001f384" in str(exc.value)
    assert "🎄" in str(exc.value)
    assert exc.value.reason_code == REASON_UNDELIVERABLE_CHARS
    # raw_opener is the SANITIZED text (undeliverable_chars() runs on `sanitized`, and
    # _sanitize's fold_to_ascii pass doesn't touch an emoji -- there's no ASCII substitute).
    assert exc.value.raw_opener == "Love the \U0001F384 vibes"


# --- % is left ALONE by _sanitize/fold_to_ascii (owner decision: "50%" must type as "50%",
# never "50 percent"); only the narrow %+lowercase-s collision is a loud, named rejection --
# see typography.fold_to_ascii's docstring point 4 and typography.undeliverable_sequences.
def test_generate_accepts_opener_with_percent_sign_unchanged():
    payload = _gemini_response({
        "opener": "That trail run photo is great, looks like you hit 50% of the course already",
        "referenced": "x", "item_index": 1,
    })
    result = _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert "50%" in result.opener
    assert "percent" not in result.opener


def test_generate_raises_parse_error_for_percent_lowercase_s_collision():
    payload = _gemini_response({
        "opener": "That trail run photo is great, 100%sure you crushed it",
        "referenced": "x", "item_index": 1,
    })
    with pytest.raises(OpenerParseError) as exc:
        _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert exc.value.reason_code == REASON_UNDELIVERABLE_SEQUENCE
    msg = str(exc.value)
    assert "'%s'" in msg                      # names the offending sequence
    # Retry hint is actionable and natural: it must not read as "percent signs are banned"
    # (that risks scaring the model off '%' entirely) -- it says '%' is fine on its own and
    # suggests a concrete reword.
    assert "'50%' is fine on its own" in msg
    assert "50 percent" in msg
    assert exc.value.raw_opener == "That trail run photo is great, 100%sure you crushed it"


def test_generate_returns_opener_result_whose_text_is_already_wysiwyg():
    """OpenerResult.opener must already equal fold_to_ascii(OpenerResult.opener) -- i.e. the
    text recorded as sent is already exactly the text a device would type, with nothing left
    for a later fold to change. This is the whole point of the fix: before it, this identity
    could fail silently (the recorded opener carried typography Adb.text() would go on to
    mangle or drop)."""
    payload = _gemini_response({"opener": "that’s a bold choice — love it",
                                "referenced": "x", "item_index": 1})
    result = _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert result.opener == fold_to_ascii(result.opener)
    assert result.opener != ""     # sanity: the fixture text really did need folding


class _NeverBudgetTracker:
    """Minimal CostTracker-shaped fake for the retry test below: never over budget, and
    `record` just needs to return a number -- the actual cost doesn't matter to this test."""
    def budget_reached(self):
        return False

    def record(self, model, usage):
        return 0.0


class _DiscardingStore:
    def record_spend(self, *a):
        pass

    def record_opener(self, *a):
        pass

    def record_opener_rejection(self, *a):
        pass


def test_emoji_opener_is_retried_by_the_service_and_a_clean_second_attempt_succeeds():
    """End-to-end regression for the WYSIWYG fix, through the REAL retry path a live run
    would take: the model's first response has an emoji in its opener field, which
    GeminiOpener._parse rejects (OpenerParseError, naming the emoji -- see
    test_generate_raises_parse_error_naming_an_emoji_in_the_opener above); OpenerService.
    maybe_opener's existing retry loop (service.py) feeds that message back to the model as
    a retry_hint and asks again; the model's second response is clean and is returned. A
    profile whose first draft happened to include an emoji still gets a real opener, rather
    than being skipped or halting the run."""
    bad = _gemini_response({"opener": "Love the \U0001F384 vibes", "referenced": "x",
                            "item_index": 1})
    clean = _gemini_response({"opener": "Love the holiday vibes", "referenced": "x",
                              "item_index": 1})
    client = _opener(_Transport([(200, bad), (200, clean)]))
    service = OpenerService(client, _NeverBudgetTracker(), _DiscardingStore(), "casual")

    pick = service.maybe_opener("r", "hinge", Profile(photos=[b"a"]))

    assert pick is not None
    assert pick.text == "Love the holiday vibes"


# --- Same regression shape, for the scaffolding-text guard (owner's overriding fear:
# preamble/label text leaking INSIDE the opener string, e.g. a "Here's the response:"
# clause that the JSON schema does nothing to prevent since it's inside the string value).
def test_generate_raises_parse_error_on_scaffolded_opener():
    payload = _gemini_response({
        "opener": "Here's the response: Great ocean, where was this taken?",
        "referenced": "x", "item_index": 1,
    })
    with pytest.raises(OpenerParseError, match="scaffolding") as exc:
        _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert exc.value.reason_code == REASON_SCAFFOLDING
    assert exc.value.raw_opener == "Here's the response: Great ocean, where was this taken?"


def test_scaffolded_opener_is_retried_by_the_service_and_a_clean_second_attempt_succeeds():
    """Same end-to-end shape as test_emoji_opener_is_retried_by_the_service... above, for the
    scaffolding-text guard: a first attempt whose opener field contains a leaked preamble
    clause is rejected and retried via service.py's existing retry loop (the rejection
    message is fed back to the model as a retry_hint), and a clean second attempt is
    returned rather than the run being skipped or halted."""
    bad = _gemini_response({
        "opener": "Here's the response: Great ocean, where was this taken?",
        "referenced": "x", "item_index": 1,
    })
    clean = _gemini_response({"opener": "Great ocean, where was this taken?",
                              "referenced": "x", "item_index": 1})
    client = _opener(_Transport([(200, bad), (200, clean)]))
    service = OpenerService(client, _NeverBudgetTracker(), _DiscardingStore(), "casual")

    pick = service.maybe_opener("r", "hinge", Profile(photos=[b"a"]))

    assert pick is not None
    assert pick.text == "Great ocean, where was this taken?"


# ---------------------------------------------------------------------------------------
# The redundancy monitor is LOG ONLY -- the counterpart to the two retry regressions above.
# Both guards directly above turn a bad opener into a rejection plus a retry; this one must
# do NEITHER, and the distinction is the whole reason it can ship before any threshold has
# been calibrated (ops/OPENER-REDESIGN.md 3.7). The fixture is the redesign's own worked
# example: the shipped sauna opener, which restates two content words from its own
# `referenced` note and is exactly what the monitor is built to count.
#
# Why this needs a test rather than a comment: the monitor sits in _parse next to five
# guards that DO raise, it produces a "markers" list identical in shape to
# _scaffolding_markers' (which raises), and there is no REASON_* constant standing in the
# way. Making it reject is a two-line edit that would look like a natural tidy-up, and the
# cost is uncalibrated rejections against a run that halts after five consecutive ones.
# ---------------------------------------------------------------------------------------

_REDUNDANT_OPENER = "That view by the sauna during sunset looks relaxing, where is this from?"
_REDUNDANT_REFERENCED = "Photo of her in an outdoor sauna at sunset"


def test_generate_returns_a_redundant_opener_unchanged_and_records_the_markers(capsys):
    payload = _gemini_response({"opener": _REDUNDANT_OPENER,
                                "referenced": _REDUNDANT_REFERENCED, "item_index": 1})
    transport = _Transport([(200, payload)])

    result = _opener(transport).generate(Profile(photos=[b"a"]), style="s")

    # Not raised, not rewritten, not blanked: the opener comes back exactly as written.
    assert result.opener == _REDUNDANT_OPENER
    assert result.referenced == _REDUNDANT_REFERENCED
    assert transport.calls == 1
    # ...and the measurement is still taken and carried on the result, which is the entire
    # point of shipping it now: the offline calibration pass in doc 3.7 needs these counts.
    assert result.redundancy_markers == _redundant_description_markers(
        _REDUNDANT_OPENER, _REDUNDANT_REFERENCED)
    assert result.redundancy_markers == ['opener restates the referenced word "sauna"',
                                         'opener restates the referenced word "sunset"']
    # "Log only" means it is actually logged -- a monitor nobody can see is not a monitor.
    out = capsys.readouterr().out
    assert "redundancy monitor" in out
    assert 'opener restates the referenced word "sauna"' in out
    assert "the opener is being sent" in out


def test_a_clean_opener_carries_no_redundancy_markers(capsys):
    """Control for the test above: the "Yes" half of the same edit pair, same `referenced`.
    Without this, a monitor that flagged every opener unconditionally would still pass."""
    payload = _gemini_response({"opener": "That view looks relaxing, where is this from?",
                                "referenced": _REDUNDANT_REFERENCED, "item_index": 1})

    result = _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")

    assert result.redundancy_markers == []
    assert "redundancy monitor" not in capsys.readouterr().out


def test_redundant_opener_is_sent_by_the_service_without_burning_a_retry():
    """The same guarantee through the REAL path a live run takes. If the monitor were ever
    promoted to a rejection, the service's retry loop would ask the model again, so a second
    request is the observable signature of a gate; there is only one response queued here, so
    a retry would also exhaust the transport. One request in, one opener out."""
    payload = _gemini_response({"opener": _REDUNDANT_OPENER,
                                "referenced": _REDUNDANT_REFERENCED, "item_index": 1})
    transport = _Transport([(200, payload)])
    service = OpenerService(_opener(transport), _NeverBudgetTracker(), _DiscardingStore(),
                            "casual")

    pick = service.maybe_opener("r", "hinge", Profile(photos=[b"a"]))

    assert pick is not None
    assert pick.text == _REDUNDANT_OPENER
    assert transport.calls == 1
