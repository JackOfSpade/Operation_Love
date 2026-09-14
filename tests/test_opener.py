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
import copy
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
    REASON_PREEMPTIVE_DISCLAIMER,
    REASON_PREMATURE_SHARED_FUTURE,
    REASON_SCAFFOLDING,
    REASON_SENSITIVE_INFERENCE,
    REASON_TOO_MANY_SENTENCES,
    REASON_UNCONFIRMED_LOCATION_FOLLOWUP,
    REASON_UNDELIVERABLE_CHARS,
    REASON_UNDELIVERABLE_SEQUENCE,
    _ITEM_PREAMBLE,
    _SYSTEM,
    _image_media_type,
    _leading_ngram,
    _preemptive_disclaimer_markers,
    _premature_shared_future_markers,
    _redundant_description_markers,
    _sanitize,
    _scaffolding_markers,
    _sentence_count,
    _sensitive_inference_markers,
    _strip_wrapping_quotes,
    _unconfirmed_location_followup_markers,
    prompt_stamp,
)
import operation_love.opener.opener as opener_mod
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


def test_sanitize_removes_comma_immediately_before_or_for_casual_texting():
    assert _sanitize(
        "Were you reading a handwritten note, or just scanning the dessert menu?"
    ) == "Were you reading a handwritten note or just scanning the dessert menu?"
    assert _sanitize("Dessert, OR the handwritten note?") == "Dessert OR the handwritten note?"


def test_sanitize_keeps_other_commas_and_does_not_touch_words_starting_with_or():
    assert _sanitize("Honestly, that looks fun or slightly chaotic") == (
        "Honestly, that looks fun or slightly chaotic"
    )
    assert _sanitize("That orange, obviously") == "That orange, obviously"


# --- _strip_wrapping_quotes: a pure formatting repair (the model quoting its own message
# back), applied silently in _parse rather than routed through _scaffolding_markers -- see
# that function's and _parse's docstrings for why this doesn't burn a retry.
def test_strip_wrapping_quotes_strips_matched_double_quotes():
    assert _strip_wrapping_quotes('"Nice antlers."') == "Nice antlers."


def test_strip_wrapping_quotes_strips_matched_single_quotes():
    assert _strip_wrapping_quotes("'Nice antlers.'") == "Nice antlers."


def test_strip_wrapping_quotes_strips_around_a_contraction():
    """2026-09-05 register rewrite: SPOKEN REGISTER makes contractions the NORM, and a
    contraction's apostrophe is intra-word punctuation, not the other half of the leading
    quote. Refusing to strip on any interior ' would therefore have disabled this repair for
    the common case, and nothing downstream catches it -- _scaffolding_markers does not match
    a wholly quoted message (verified), so the literal wrapping apostrophes would be typed
    into her comment box. Before the rewrite the model produced zero apostrophes in 188
    openers, which is why the old rule never showed the hole."""
    assert _strip_wrapping_quotes("'That's a nice mug.'") == "That's a nice mug."
    assert _strip_wrapping_quotes("'Nice antlers, isn't it?'") == "Nice antlers, isn't it?"
    # A message quoted back with BOTH kinds of interior contraction still repairs.
    assert (_strip_wrapping_quotes("'You'd never guess what that isn't.'")
            == "You'd never guess what that isn't.")


def test_strip_wrapping_quotes_leaves_unbalanced_inner_apostrophe_alone():
    # A trailing possessive apostrophe is NOT intra-word (a space follows it), so the
    # leading/trailing ' are not a clean matched wrapping pair -- stripping them would leave a
    # broken, unbalanced string, so this must be returned completely unchanged. This is the
    # half of the rule the contraction allowance above deliberately does NOT relax.
    text = "'Grams' pie is unreal.'"
    assert _strip_wrapping_quotes(text) == text
    # Same for an inner quoted phrase inside a double-quoted wrapper.
    nested = '"He said "hi" and left."'
    assert _strip_wrapping_quotes(nested) == nested


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
    # 2026-09-06 (b) "HERE'S" collision (ops/OPENER-REDESIGN.md residuals section): a bare
    # leading "Here's"/"Here is" used to be matched unconditionally, so this perfectly natural
    # spoken opening was rejected as REASON_SCAFFOLDING against the same max_attempts=5 budget
    # that stops a whole run. "hoping" is a gerund, never the meta noun naming the output, so
    # _SCAFFOLD_HERE_IS_OBJECT_RE does not match it.
    "Here's hoping that trail's as steep as it looks. Have you done the full loop?",
    "Here's to a good hike, was that trail as brutal as it looks?",     # "to" is not a determiner
    "Here's a photo I love, where was it taken?",                      # "a photo" names a thing, not the output
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
    # 2026-09-06 (b): the narrowed "here's"/"here is" check must still catch genuine preamble
    # that introduces the message as an object -- a determiner plus a meta noun naming the
    # output -- not just the bare phrase removed above.
    "Here's an option: Skiing or the beach, whichever you prefer?",
    "Here is my take: that lake looks unreal, where is it?",
]


@pytest.mark.parametrize("text", _SCAFFOLDING_TRUE_POSITIVES)
def test_scaffolding_markers_true_positive_corpus_is_flagged(text):
    assert _scaffolding_markers(text) != []


@pytest.mark.parametrize("text", [
    "Zero judgment here, you wear those Mickey ears with total confidence.",
    "No judgement from me. Which ride is nonnegotiable?",
    "No offense, but that hat has serious main character energy.",
    "Hey, no pressure! What is the story behind that costume?",
    "Don't take this the wrong way, that sweater is spectacular.",
    "Not to be weird, but your dog looks exactly like my old neighbor's.",
    "I'm not trying to sound nosy, but where was that photo taken?",
    "This might sound dramatic, but that cake deserves its own fan club.",
])
def test_preemptive_disclaimer_markers_flag_negative_framing_at_the_start(text):
    assert _preemptive_disclaimer_markers(text) != []


@pytest.mark.parametrize("text", [
    "You wear those Mickey ears with total confidence. Which ride is nonnegotiable?",
    "That sweater is not subtle, and I respect the commitment.",
    "The no pressure approach clearly does not apply to that competition.",
    "Tell me that dramatic cake tasted as good as it looked.",
    "I cannot get over that dog's face.",
    "Is judging the costume contest as difficult as entering it?",
])
def test_preemptive_disclaimer_markers_leave_ordinary_negation_and_words_alone(text):
    assert _preemptive_disclaimer_markers(text) == []


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
    # 2026-09-08 watched Training incident, implicit control: the attached item resolves the
    # place reference without repeating its visible surface, furniture, drink, or pose.
    ("You look very at home there. Is this your regular spot for a quiet evening out?",
     "Photo of Taylor sitting on a tiled outdoor bench under string lights with her legs "
     "folded up, holding a glass of white wine"),
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
    # 2026-09-08 watched Training incident: its question supplies conversational value, but the
    # setup still copies four obvious content words that an implicit place reference can replace.
    ("Perching right up on the tiled bench with a glass of wine is definitely the way to do it. "
     "Is this your regular spot for a quiet evening out?",
     "Photo of Taylor sitting on a tiled outdoor bench under string lights with her legs "
     "folded up, holding a glass of white wine"),
]


@pytest.mark.parametrize("opener, referenced", _REDUNDANCY_TRUE_POSITIVES)
def test_redundancy_markers_true_positive_corpus_is_flagged(opener, referenced):
    assert _redundant_description_markers(opener, referenced) != []


def test_redundancy_monitor_documents_the_2026_09_08_literal_setup_incident():
    """The monitor is evidence for this prompt fix, not a send gate (see the corpus above)."""
    opener = (
        "Perching right up on the tiled bench with a glass of wine is definitely the way to do "
        "it. Is this your regular spot for a quiet evening out?"
    )
    referenced = (
        "Photo of Taylor sitting on a tiled outdoor bench under string lights with her legs "
        "folded up, holding a glass of white wine"
    )
    assert _redundant_description_markers(opener, referenced) == [
        'opener restates the referenced word "tiled"',
        'opener restates the referenced word "bench"',
        'opener restates the referenced word "glass"',
        'opener restates the referenced word "wine"',
    ]


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


# ---------------------------------------------------------------------------------------
# prompt_stamp: the prompt-era digest every `openers` / `opener_rejections` row carries from
# 2026-09-05 (b). Its whole value is that two rows with the same digest were generated under
# the same prompt, so these tests pin BOTH directions: identical inputs must agree, and each
# input the digest claims to cover must be able to change it.
# ---------------------------------------------------------------------------------------

def test_prompt_stamp_is_a_64_character_hex_sha256():
    """Stored as a nullable STRING/TEXT column in both backends, so its shape is a contract
    with every offline query that GROUPs BY it, not just an implementation detail."""
    stamp = prompt_stamp("casual and warm")
    assert len(stamp) == 64
    assert set(stamp) <= set("0123456789abcdef")


def test_prompt_stamp_is_deterministic_for_the_same_style():
    """Pure and I/O-free: the service computes it ONCE at startup and replays it on every row
    of the run, so a digest that varied per call would file one run's rows under many eras."""
    assert prompt_stamp("casual and warm") == prompt_stamp("casual and warm")


def test_prompt_stamp_changes_when_the_owner_style_text_changes():
    """config.yaml's opener.style is one of the four on-wire prompt copies and the one the
    owner actually edits, so an era boundary usually IS a style edit."""
    assert prompt_stamp("casual and warm") != prompt_stamp("casual and warm.")


def test_prompt_stamp_changes_when_the_system_prompt_changes(monkeypatch):
    """_SYSTEM participates, not just the style.

    The 2026-09-05 register rewrite touched `opener.style` AND `_SYSTEM` in lockstep, but
    nothing forces that: a `_SYSTEM`-only edit is a real era boundary that a style-only digest
    would report as no change at all, which is the exact failure this stamp exists to prevent.
    """
    style = "casual and warm"
    before = prompt_stamp(style)
    monkeypatch.setattr(opener_mod, "_SYSTEM", _SYSTEM + " One more rule.")
    assert prompt_stamp(style) != before


def test_prompt_stamp_changes_when_a_schema_field_description_changes(monkeypatch):
    """The `_SCHEMA` field descriptions are the third on-wire copy (responseJsonSchema); the
    2026-09-05 rewrite shipped a compressed form of three register rules there and nowhere
    else in the schema, so they are prompt text and must move the digest."""
    style = "casual and warm"
    before = prompt_stamp(style)
    edited = copy.deepcopy(opener_mod._SCHEMA)
    edited["properties"]["opener"]["description"] += " Say it out loud first."
    monkeypatch.setattr(opener_mod, "_SCHEMA", edited)
    assert prompt_stamp(style) != before


@pytest.mark.parametrize("constant", ["_ITEM_PREAMBLE", "_ITEM_PREAMBLE_CONTEXT",
                                      "_ITEM_LABEL", "_CONTEXT_LABEL"])
def test_prompt_stamp_changes_when_an_item_crop_instruction_constant_changes(
        monkeypatch, constant):
    """Every fixed prompt-era input, including legacy digest-only values, must move the digest.

    `_ITEM_PREAMBLE` and `_ITEM_LABEL` are current wire rules. `_ITEM_PREAMBLE_CONTEXT` and
    `_CONTEXT_LABEL` are intentionally retained only as historical prompt-stamp inputs so
    stored rows and backfill tooling keep their seven-component contract. Pinned one constant
    at a time because they enter the digest as separate components: a widening that dropped any
    single one would leave the others green.
    """
    style = "casual and warm"
    before = prompt_stamp(style)
    monkeypatch.setattr(opener_mod, constant, getattr(opener_mod, constant) + " One more rule.")
    assert prompt_stamp(style) != before


def test_prompt_stamp_separates_adjacent_constants_rather_than_concatenating_them(monkeypatch):
    """The NUL between every pair of components is what makes the boundaries unambiguous.

    Without it, moving a sentence from the end of one constant to the start of the next -- a
    real edit, and exactly the kind of copy shuffle these constants have had before -- would
    produce a byte-identical payload and therefore an identical digest, silently merging two
    eras. The two monkeypatched arrangements below concatenate to the same string and must
    still stamp differently.
    """
    style = "casual and warm"
    moved_sentence = " Read the label directly above each image."
    monkeypatch.setattr(opener_mod, "_ITEM_PREAMBLE", _ITEM_PREAMBLE + moved_sentence)
    trailing = prompt_stamp(style)
    monkeypatch.setattr(opener_mod, "_ITEM_PREAMBLE", _ITEM_PREAMBLE)
    monkeypatch.setattr(opener_mod, "_ITEM_PREAMBLE_CONTEXT",
                        moved_sentence + opener_mod._ITEM_PREAMBLE_CONTEXT)
    assert prompt_stamp(style) != trailing


def test_prompt_stamp_ignores_schema_dict_key_ordering(monkeypatch):
    """Canonicalized with sort_keys, so reordering the schema's dict literal -- which changes
    no byte the model ever sees -- must not manufacture a false era boundary. Without this,
    every cosmetic reshuffle of _SCHEMA would split one era into two in the stored rows."""
    style = "casual and warm"
    before = prompt_stamp(style)
    reordered = {key: copy.deepcopy(value)
                 for key, value in reversed(list(opener_mod._SCHEMA.items()))}
    reordered["properties"] = {
        key: copy.deepcopy(value)
        for key, value in reversed(list(opener_mod._SCHEMA["properties"].items()))}
    assert list(reordered) != list(opener_mod._SCHEMA)          # the reorder really happened
    monkeypatch.setattr(opener_mod, "_SCHEMA", reordered)
    assert prompt_stamp(style) == before


def test_sentence_count_basic_cases():
    assert _sentence_count("One profile-specific thought") == 1
    assert _sentence_count("One thought. One easy question?") == 2
    assert _sentence_count("Dr. Dolittle energy. What's the story?") == 2   # abbreviation guard
    assert _sentence_count("One. Two? Three!") == 3


def test_sensitive_inference_guard_catches_the_reported_bridge_jump_angle_only():
    """A bridge photo is not evidence of a jump or a willingness to take one.

    The production prompt makes that evidence rule explicit. This tiny deterministic guard is
    the final pre send backstop for the reported wording, and intentionally stays narrower than
    a ban on all sport vocabulary so explicit, benign profile activities remain possible.
    """
    reported = ("Did you work up the courage to jump or were you happy just taking in "
                "the scenery?")
    assert _sensitive_inference_markers(reported) == ["courage to jump"]
    assert _sensitive_inference_markers("Did you jump from the bridge?") == [
        "asking whether she jumped", "jumping from a height"]
    assert _sensitive_inference_markers("Your paragliding story sounds unforgettable.") == []
    assert _sensitive_inference_markers("That bridge is beautiful.") == []


_REPORTED_UNCONFIRMED_LOCATION_FOLLOWUP = (
    "That looks a lot like Lake Louise in deep winter. Were you out there ice skating "
    "or just braving the freeze for the view?"
)

_UNCONFIRMED_LOCATION_MARKER = "unconfirmed location used as a later premise"


@pytest.mark.parametrize("text", [
    _REPORTED_UNCONFIRMED_LOCATION_FOLLOWUP,
    ("That looks like Namsan Tower in Seoul right behind you. Was visiting Korea your "
     "favorite trip so far or is another destination at the top of your list?"),
    ("That looks like an Iceland super jeep tour. Did you guys take that thing out onto a "
     "glacier or into the highlands?"),
    ("That sunny stone street looks like Spain, maybe Mallorca. Did you sneak away there for "
     "a quick escape while you were living in London?"),
    ("Judging by the architecture, my official guess for that backdrop is Italy. What was "
     "your favorite spot you visited while you were there?"),
    "Is that Banff? How long were you there?",
    "I'm guessing Norway, did you hike while you were there?",
    ("My guess is Switzerland, but did that fluffy cat hire himself out as your local tour "
     "guide or just demand a petting break?"),
    "It has to be Lake Louise. How cold was it there?",
    "My guess is Switzerland, you must have loved hiking there.",
    "That looks like Banff and you must have gone skiing.",
    "Was that Lake Louise? Did you skate there?",
    "That looks like Banff. How long were you there?",
    "That looks like Lake Louise. Was that taken during your ski trip?",
    "That looks like Lake Louise. Is that taken before you went skating?",
    "That looks like Lake Louise. Was that taken in Alberta during the ski trip?",
    "That looks like Lake Louise. Was that taken out west after the glacier hike?",
    "That looks like Lake Louise. Is that Banff and did the group skate there?",
    "That looks like Lake Louise. Is that Banff because the group went skiing?",
    "That looks like Banff, must have been freezing there.",
    # Addendum -- 2026-09-14 (b): the 2026-09-14 noun-set/pattern widening (see
    # _LOCATION_CONFIRMATION_NOUN's module comment) only had to make MORE confirmation shapes
    # legal, never fewer rejections. These three put the same newly-legal place nouns into a
    # frame that is NOT a confirmation of the sender's own guess, to prove the widening did not
    # open a hole alongside it.
    "My guess is Rome. Did you get the city right?",
    "My guess is Rome. How close is that gelato?",
    "My guess is Rome. What was the best part of the city?",
], ids=[
    "reported-lake-louise",
    "logged-namsan-tower",
    "logged-iceland-tour",
    "logged-spain-mallorca",
    "logged-official-italy-guess",
    "confirmation-question-then-assumption",
    "single-sentence-comma-then-assumption",
    "logged-switzerland-comma-but-followup",
    "has-to-be-then-assumption",
    "comma-subject-led-statement",
    "conjunction-subject-led-statement",
    "past-tense-location-question-then-assumption",
    "looks-like-with-there-followup",
    "taken-during-activity-is-not-confirmation",
    "taken-before-activity-is-not-confirmation",
    "place-then-temporal-activity-is-not-confirmation",
    "direction-then-activity-is-not-confirmation",
    "named-place-then-coordinated-activity-is-not-confirmation",
    "named-place-then-causal-activity-is-not-confirmation",
    "comma-bare-modal-statement",
    "widened-noun-set-still-rejects-recipient-directed-phrasing",
    "widened-noun-set-still-rejects-a-noun-outside-the-shared-set",
    "widened-noun-set-still-rejects-an-experience-question-after-the-noun",
])
def test_unconfirmed_location_followup_guard_flags_dependent_later_beats(text):
    assert _unconfirmed_location_followup_markers(text) == [
        _UNCONFIRMED_LOCATION_MARKER
    ]


def test_unconfirmed_location_followup_guard_addendum_20260914b_keeps_the_historical_six_rejected():
    """The 2026-09-14 (b) noun-set/pattern widening (ops/OPENER-REDESIGN.md, Addendum --
    2026-09-14 (b), "THE FIX") must not have loosened the rejection side of the guard. These
    are the exact six live drafts the 2026-09-05 (d) PRECISION CHECK named as genuine
    location-guess-then-assumption failures -- Italy, Spain/Mallorca, Switzerland, Namsan
    Tower, the Iceland super jeep tour, and Lake Louise -- already pinned individually above by
    test_unconfirmed_location_followup_guard_flags_dependent_later_beats, and re-asserted here
    word for word as one named addendum regression, so a future noun-set or pattern edit that
    reopens any one of them fails a test that names the addendum rather than an unrelated id
    several hundred lines away.
    """
    historical_failures = [
        _REPORTED_UNCONFIRMED_LOCATION_FOLLOWUP,  # Lake Louise
        ("That looks like Namsan Tower in Seoul right behind you. Was visiting Korea your "
         "favorite trip so far or is another destination at the top of your list?"),
        ("That looks like an Iceland super jeep tour. Did you guys take that thing out onto a "
         "glacier or into the highlands?"),
        ("That sunny stone street looks like Spain, maybe Mallorca. Did you sneak away there "
         "for a quick escape while you were living in London?"),
        ("Judging by the architecture, my official guess for that backdrop is Italy. What was "
         "your favorite spot you visited while you were there?"),
        ("My guess is Switzerland, but did that fluffy cat hire himself out as your local tour "
         "guide or just demand a petting break?"),
    ]
    for text in historical_failures:
        assert _unconfirmed_location_followup_markers(text) == [_UNCONFIRMED_LOCATION_MARKER]


@pytest.mark.parametrize("text", [
    # Existing accepted corpus entries: these must not be lost to a broad guess/there rule.
    "Based on that ridgeline I'm going to guess Norway.",
    "I know you were smiling, but I bet you were freezing out there.",
    "I am going to guess that was colder than it looks.",
    # A later beat may ask only for confirmation or correction of the inferred place.
    "That looks a lot like Lake Louise in deep winter. Am I close?",
    "That looks like Lake Louise. Is that right?",
    "That scenery looks like British Columbia. Was this taken out west or somewhere else?",
    "My guess is Switzerland. Could it be Austria instead?",
    "That looks like Lake Louise. Was that taken in Alberta?",
    "That looks like Lake Louise. Was that taken on the Icefields Parkway?",
    "That looks like Banff and Lake Louise.",
    "That looks like Lake Louise. Was that Banff instead?",
    "That looks like Croatia. Could that be Bosnia and Herzegovina?",
    "That looks like Lake Louise. Could that be right?",
    "That looks like Lake Louise. Is my guess right?",
    "That looks like Lake Louise. Am I even close?",
    "That looks like Lake Louise. Could that be St. Moritz?",
    "That looks like Lake Louise. Am I close? :)",
    "That looks like Mt. Fuji.",
    "That looks like Washington D.C. in spring.",
    "That sweeping metal bridge over the water looks like Portugal. Am I close on that location?",
    "Is that Lake Louise in deep winter?",
    "That has to be Lake Louise, right?",
    # A stated place may come from profile text; outgoing text alone cannot relitigate it.
    "Banff in January sounds intense. How long were you there?",
    # Looks like is often a grounded interpretation rather than a place identification.
    "That looks like serious dedication. Were you out there before sunrise?",
    "That husky looks like he takes guard duty seriously. Which one calls the shots?",
    "That looks like Audrey Hepburn. Where did you find the coat?",
    # Addendum -- 2026-09-14 (b): until this date _DIRECT_LOCATION_CONFIRMATION_PATTERNS'
    # two confirmation shapes carried DIFFERENT place-noun sets ("Am I right about the
    # country?" accepted, "Did I get the country right?" rejected; "city" and "town" in
    # neither), so which verb the model reached for silently decided the verdict, and the
    # "how close" shape only ever matched the literal "How close am I?". One shared
    # _LOCATION_CONFIRMATION_NOUN alternation now backs every shape. The line below is the
    # measured regression: replaying the reported "Looks like Rome, right?" capture through
    # the 2026-09-14 (a) prompt produced this exact draft, and the pre-fix detector rejected
    # it even though it violates no rule on wire (see run draw 1 in the addendum table).
    "My guess is Rome. Did I get the city right?",
    "That looks like Norway. Did I get the country right?",
    "That looks like Tuscany. Did I get the region right?",
    "My guess is Vienna. Did I guess the city right?",
    "Is that Sedona? Did I call the town right?",
    "That looks like Vienna. Am I right about the city?",
    "That looks like Vienna. Was I close on the city?",
    # The exact string from ops/OPENER-REDESIGN.md's 2026-09-05 (d) PRECISION CHECK, which
    # claimed this retained Portugal draft "remain[ed] valid". Re-running the shipped
    # detector on it before this fix showed that claim was never true (the "how close" shape
    # only matched "How close am I?"); the 2026-09-14 (b) addendum corrects the record and
    # this line pins the correction.
    "That rock arch looks a lot like the Algarve coast in Portugal. How close is that guess?",
    "My guess is Norway. How far off was that guess?",
    "My guess is Norway. How close am I?",
], ids=[
    "existing-norway-guess",
    "existing-freezing-guess",
    "existing-coldness-guess",
    "direct-confirmation-followup",
    "direct-pronoun-confirmation",
    "direct-location-correction",
    "direct-named-location-correction",
    "taken-in-proper-place-confirmation",
    "taken-on-proper-place-confirmation",
    "compound-place-guess-with-no-followup",
    "past-tense-named-place-correction",
    "place-name-with-and",
    "could-that-be-right",
    "is-my-guess-right",
    "am-i-even-close",
    "direct-place-confirmation-with-abbreviation",
    "confirmation-with-terminal-smiley",
    "standalone-mount-abbreviation",
    "standalone-dotted-place-abbreviation",
    "exact-portugal-confirmation-control",
    "single-beat-location-question",
    "confirmation-tag",
    "known-location",
    "non-location-looks-like",
    "figurative-pet-looks-like",
    "no-location-context-with-wh",
    "measured-regression-my-guess-is-rome-did-i-get-the-city-right",
    "did-i-get-the-country-right",
    "did-i-get-the-region-right",
    "did-i-guess-the-city-right",
    "did-i-call-the-town-right",
    "am-i-right-about-the-city",
    "was-i-close-on-the-city",
    "portugal-how-close-is-that-guess-precision-check-correction",
    "how-far-off-was-that-guess",
    "how-close-am-i-still-passes",
])
def test_unconfirmed_location_followup_guard_preserves_clean_corpus(text):
    assert _unconfirmed_location_followup_markers(text) == []


def test_location_confirmation_shapes_share_one_noun_set():
    """Regression guard for the 2026-09-14 (b) bug itself, not just its symptoms: two
    confirmation shapes once drew from DIFFERENT place-noun lists ("Am I right about the
    country?" accepted, "Did I get the country right?" rejected; "city" and "town" in
    neither), so which verb the model reached for silently decided the verdict.
    _DIRECT_LOCATION_CONFIRMATION_PATTERNS now interpolates ONE shared alternation,
    _LOCATION_CONFIRMATION_NOUN, into both the "am/was I right/close ... about/on/with the
    <noun>?" shape and the "did I get/guess/call the <noun> right?" shape.

    This test reads that live alternation (never a hardcoded copy of it) and asserts both
    shapes accept a location-confirming later beat for every noun currently in it. A future
    editor who extends one pattern's noun list without the other reproduces exactly the
    2026-09-14 (b) bug and this test fails immediately, without anyone having to write a new
    case naming the specific noun that regressed.
    """
    noun_alternation = opener_mod._LOCATION_CONFIRMATION_NOUN
    assert noun_alternation.startswith("(?:") and noun_alternation.endswith(")"), (
        "this test assumes the simple (?:a|b|c) alternation shape used today; update the "
        "extraction below if _LOCATION_CONFIRMATION_NOUN's construction ever changes"
    )
    nouns = noun_alternation[len("(?:"):-len(")")].split("|")
    assert len(nouns) > 1  # sanity: the alternation really does list more than one noun

    for noun in nouns:
        did_i_get = f"My guess is Rome. Did I get the {noun} right?"
        am_i_right = f"My guess is Rome. Am I right about the {noun}?"
        assert _unconfirmed_location_followup_markers(did_i_get) == [], (
            f"'Did I get the {noun} right?' should be accepted as a location confirmation, "
            f"the same as every other noun in the shared set")
        assert _unconfirmed_location_followup_markers(am_i_right) == [], (
            f"'Am I right about the {noun}?' should be accepted as a location confirmation, "
            f"the same as every other noun in the shared set")


def test_confirmation_shapes_exclude_below_world_scale_nouns():
    """GUESS THE WORLD, NOT HER IDENTITY forbids guessing a street, a neighbourhood, a hotel,
    or a venue at all, so _LOCATION_CONFIRMATION_NOUN deliberately excludes all four even
    though each reads as an ordinary confirmation noun in isolation ("did I get the venue
    right?" parses exactly like "did I get the city right?"). This pins that exclusion
    directly against the live noun set so a future editor who unions in a below-world-scale
    noun, to fix some unrelated complaint, is caught here rather than discovered live.

    Checked by execution, not assumed: these four are rejected for the same generic reason as
    any other non-confirming later beat (REASON_UNCONFIRMED_LOCATION_FOLLOWUP) because the
    confirmation pattern simply fails to fullmatch when the noun is outside the shared
    alternation -- there is no dedicated street/neighbourhood/hotel/venue-specific rejection
    path in the code, only this table's silence on them.
    """
    noun_alternation = opener_mod._LOCATION_CONFIRMATION_NOUN
    nouns = noun_alternation[len("(?:"):-len(")")].split("|")
    excluded_nouns = ("street", "neighbourhood", "hotel", "venue")
    for excluded in excluded_nouns:
        assert excluded not in nouns, (
            f"{excluded!r} must stay out of the shared confirmation noun set -- GUESS THE "
            f"WORLD, NOT HER IDENTITY forbids guessing it at all")
        did_i_get = f"My guess is Rome. Did I get the {excluded} right?"
        am_i_right = f"My guess is Rome. Am I right about the {excluded}?"
        assert _unconfirmed_location_followup_markers(did_i_get) == [
            _UNCONFIRMED_LOCATION_MARKER]
        assert _unconfirmed_location_followup_markers(am_i_right) == [
            _UNCONFIRMED_LOCATION_MARKER]


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
    assert "profile text fact check" in lowered
    assert "never ask for a fact, preference, activity, place, or opinion" in lowered
    assert "if she says she likes apples, do not ask whether she likes apples" in lowered
    assert "positive, fun conversation" in lowered
    assert "brief greeting is optional" in lowered
    assert "two sentences is the absolute maximum" in lowered
    assert "never use an em dash or any hyphen" in lowered
    assert "plain ascii letters and punctuation" in lowered   # WYSIWYG / no-emoji rule
    assert "no emoji" in lowered
    assert "terminal plain text smiley, :) is allowed only" in lowered
    assert "never add it by default" in lowered
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
    assert "safety and dignity" in lowered
    assert "never infer, tease, or pose a forced choice about self harm, suicide" in lowered
    assert "a bridge, height, water, travel scene, or recognized location is never evidence" in lowered
    assert "that she jumped or considered jumping" in lowered
    assert ("shared context rule: your message is displayed directly under the exact photo "
            "or prompt it attaches to") in lowered
    assert "she is looking at that item while she reads your words" in lowered
    assert "primary item rule" in lowered
    assert "clear main subject of referenced, angle, and opener" in lowered
    assert "profile text only when it sharpens a connection back to the selected item" in lowered
    assert "connection back to the selected item" in lowered
    assert "justify why the selected item was liked" in lowered
    assert "it supplies the reason, subject, or payoff" in lowered
    assert "do not let profile text replace the selected item" in lowered
    assert "photo header rule" in lowered
    assert "title, caption, or prompt printed with a photo is part of that same item" in lowered
    assert "defines how the photo is meant to be read" in lowered
    assert "interpret the visible scene through that text before choosing an angle" in lowered
    assert "never contradict, reverse, or ignore the header's framing" in lowered

    # A question may stand alone when it is more natural than a claim. A guess stays unconfirmed
    # across the whole opener so she, rather than the sender, gets to resolve it.
    assert "one open, easy-to-answer question" not in lowered
    assert "a natural claim she can correct can be effective, but it is not mandatory" in lowered
    assert "question may be the whole message when that is the strongest natural angle" in lowered
    assert "never as the whole message" not in lowered
    assert "setup payoff continuity" in lowered
    assert "every visible detail you name must be necessary to, and used by" in lowered
    assert "necessary means its exact identity changes how the move is understood" in lowered
    assert "not merely that it anchors the reaction" in lowered
    assert "removing a descriptive clause or replacing it" in lowered
    assert "use the shorter implicit version" in lowered
    assert "question coherence" in lowered
    assert "ask one coherent thing at a time" in lowered
    assert "parallel, genuinely contrasting answers to that same underlying question" in lowered
    assert "never to join unrelated dimensions" in lowered
    assert "casual or punctuation" in lowered
    assert "never put a comma immediately before 'or'" in lowered
    assert "write it the way a person would text casually" in lowered
    assert "referent clarity" in lowered
    assert "every pronoun, shorthand noun, and question subject" in lowered
    assert "must have one immediately obvious referent" in lowered
    assert "keep the same referent unless the transition to a new one is explicit" in lowered
    assert "different ordinary meanings of the same word" in lowered
    assert "role consistency" in lowered
    assert "preserve that role across every beat" in lowered
    assert "same subject an incompatible role later" in lowered
    assert "different subject, name it explicitly" in lowered
    assert "reply comfort" in lowered
    assert "most natural honest reply feel good to give" in lowered
    assert "preference, perspective, inspiration, or experience, not self justification" in lowered
    assert "intelligence, sincerity, knowledge, effort" in lowered
    assert "forced choice whose honest answers make her defend, diminish, or embarrass herself" in lowered
    assert "rather than asking her to verify its status" in lowered
    assert "positive social framing" in lowered
    assert "state the intended positive observation, question, or invitation directly" in lowered
    assert ("naming an insulting, judgmental, awkward, pressuring, creepy, or offensive "
            "interpretation") in lowered
    assert "that denial introduces the negative interpretation" in lowered
    assert "remove the disclaimer and make the substantive thought stand on its own" in lowered
    assert "reciprocity before future" in lowered
    assert "not an audition for a role described in her profile" in lowered
    assert "never answer one of her preferences by advertising the sender" in lowered
    assert "promising what he will do for her" in lowered
    assert "do not assume that a match, date, relationship, or shared future already exists" in lowered
    assert "possessive language about a first date, place, trip, or other future together" in lowered
    assert "a proposal is not an established shared plan" in lowered
    assert "information gain test" in lowered
    assert "conclusion of a guess must not itself be directly visible or explicitly stated" in lowered
    assert "its header, a sign, or elsewhere in her profile" in lowered
    assert "they are clues, not guessed conclusions" in lowered
    assert "ordinary viewer can read or see the conclusion directly without inference" in lowered
    assert "confirmation boundary" in lowered
    assert "a guess remains unconfirmed until she replies" in lowered
    assert "statement, question, compliment, or invitation that assumes it is correct" in lowered
    assert "natural next move is to confirm or correct it" in lowered
    assert "only invite that confirmation or correction without presupposing the answer" in lowered
    assert "experience, preference, or consequence that only makes sense if the guess is true" in lowered
    assert "confirmation boundary overrides this inheritance rule" in lowered
    assert "never let a later beat inherit an unconfirmed claim as fact" in lowered
    assert "asking only whether an inferred location itself is right" in lowered
    assert "visual location turn boundary" in lowered
    assert "that location guess is the only conversational move before she replies" in lowered
    assert "end after it or ask only whether the location itself is right" in lowered
    assert "her settling it is the whole payoff either way" in lowered
    assert ("activity, reason, preference, feeling, experience, or consequence at that place"
            in lowered)
    assert "must accept x as its working premise" not in lowered

    # --- LENGTH: the ceiling stays, the one-sentence PREFERENCE is gone (doc 3.2.1).
    # Preferring brevity for its own sake fought a claim that needs room to exist ("I know
    # you were smiling, but I bet you were freezing out there" is eighteen words and is the
    # strongest opener in the whole design). A regression here silently reinstates it.
    assert "one short sentence is preferred" not in lowered
    assert "application rule: two sentences is the absolute maximum" in lowered
    assert "as short as the angle allows" in lowered
    assert "spend no word merely repeating what she can already see" in lowered
    assert "never cut necessary setup or the conversational payoff" in lowered
    assert "subject to confirmation boundary, a second sentence may be" in lowered

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
    assert "when a place comes from recognizing the image rather than from her profile text" in lowered
    assert "present it only as an inference and obey visual location turn boundary" in lowered
    assert "do not build on an inferred location as though it were correct" in lowered
    assert "state it as shared experience, or turn it into a generic compliment" in lowered
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
    assert "message must use the least explicit immediately clear reference" in lowered
    assert "may name only the setup whose exact identity changes the conversational move" in lowered
    assert "angle is your own short wording for what your opener is doing" in lowered
    assert "item_description says in a few words what the item you picked is" in lowered

    # --- ITEM SELECTION (doc 5.1/5.7). The model CHOOSES the item now, so _SYSTEM has to state
    # the three facts the new contract rests on: the numbering is 1-based over the items in
    # this request, the chosen item is also the one that gets liked (one call, doc 5.1), and
    # every other numbered candidate is selection-only after the choice.
    # Selection is by best ANGLE, not best photo -- 5.1's criterion, and the reason a good
    # opener about a mediocre photo beats a great photo with nothing to say about it.
    assert "pick the item yourself" in lowered
    assert "numbered from 1 in the order they are given" in lowered
    assert "you choose which one to write about" in lowered
    assert "not the most striking picture" in lowered
    assert "it is also the item that gets liked" in lowered
    assert "other numbered image was only an alternative for selection" in lowered
    assert "never take its facts, concepts, wordplay, or payoff" in lowered

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

    # --- SELECTED-ITEM-ONLY. Context crops are retained for forensics but never shown to Gemini;
    # once it picks, every other numbered candidate stops being usable source material.
    assert "once you choose, every other numbered image was only an alternative for selection" in lowered
    assert "never take its facts, concepts, wordplay, or payoff" in lowered
    assert "given without a number is context" not in lowered
    assert "unnumbered profile image" not in lowered
    # The replaced field is gone from the copy entirely, along with the index space it named:
    # doc 5.7 calls out that both the field and "every line of prompt copy saying scroll order"
    # change meaning, and a leftover instruction to fill referenced_index would ask the model
    # for a value the schema no longer declares.
    assert "referenced_index" not in lowered
    assert "scroll order" not in lowered

    # --- Owner invariant, not a wording choice: this text is itself sent to the model.
    assert all(ord(ch) < 128 for ch in _SYSTEM), "_SYSTEM must be pure ASCII"
    assert "—" not in _SYSTEM


def test_system_prompt_ships_the_she_is_the_one_who_knows_rule():
    """2026-09-14: a live opener asked "Looks like Rome, right?" under a photo of a woman
    standing on an Italian cobblestone street. VISUAL LOCATION TURN BOUNDARY caused it --
    "ask only whether the location itself is right" recommended exactly that interrogative
    shape -- and REPLY COMFORT's carve-out exempted it from the reply-quality test. Measured:
    within the 21 proper-name location guesses in the 234-opener corpus, the
    hedge-plus-agreement-tag shape went from 1 of 11 before 2026-09-05 to 5 of 10 after.

    This is the compressed _SYSTEM twin of the long form pinned in
    tests/test_config_yaml_real.py::test_shipped_opener_style_ships_the_she_is_the_one_who_knows_rule.
    The two copies diverge in exact wording (this one says "co-signing" hyphenated per _SYSTEM's
    plain-ASCII-but-hyphen-tolerant convention for compressed prose, and drops config.yaml's
    "rather than competing with it" and "never" phrasing in a couple of places), so each is
    pinned against its own real wording rather than shared substrings.
    """
    lowered = _SYSTEM.lower()
    assert "she is the one who knows" in lowered
    # 2026-09-06 (c): a rule whose first sentence can be satisfied while the defect survives is
    # the shape that let "... is an elite move" past NO GRADING. All three operative clauses of
    # this rule therefore live in its own first sentence, ahead of any rationale.
    first_sentence = lowered.split("she is the one who knows:", 1)[1].split(". ", 1)[0]
    assert "write any inference about it from your own not knowing" in first_sentence
    assert "never ask her to agree about how something appears" in first_sentence
    assert ("never close a claim about her own life with a tag whose only job is to collect "
            "her agreement") in first_sentence
    assert "the shared looking covers the item on the screen, never the world behind it" in lowered
    assert "she was in the world the item shows and you were not" in lowered
    assert "is news to you and old news to her" in lowered
    assert "write any inference about it from your own not knowing" in lowered
    assert ("an impression the two of you are forming together, or a tag collecting her "
            "agreement, hands the one person who was actually there") in lowered
    assert "never ask her to agree about how something appears" in lowered
    assert ("never close a claim about her own life with a tag whose only job is to collect "
            "her agreement") in lowered
    assert "co-signing" in lowered
    assert "and how it appears is what she can already see" in lowered
    assert "this narrows the shared context rule and sharpens hedge the claim, never yourself" in lowered
    assert "the uncertainty belongs in how sure you are, not in how clear the item is" in lowered
    assert "asking her outright stays welcome" in lowered
    assert "never invent the sender still forbids saying where he has or has not been" in lowered

    # VISUAL LOCATION TURN BOUNDARY amendment: the old payoff line is SUPERSEDED, not merely
    # supplemented -- its return would silently reinstate the wording that produced the "right?"
    # agreement-tag shape.
    assert "her settling it is the whole payoff either way" in lowered
    assert "neither ending is the default" in lowered
    assert "vary the shape still chooses between them" in lowered
    assert "she settles the place because she was standing in it" in lowered
    assert "the confirmation is the conversational payoff" not in lowered

    # REPLY COMFORT amendment: the confirmation carve-out now says explicitly what it does NOT
    # exempt -- SHE IS THE ONE WHO KNOWS still governs how that confirmation is worded.
    assert "it exempts that confirmation from nothing else" in lowered

    # Owner invariant: this text is itself sent to the model.
    assert all(ord(ch) < 128 for ch in _SYSTEM), "_SYSTEM must be pure ASCII"
    assert "—" not in _SYSTEM


def test_system_prompt_defaults_to_minimum_sufficient_visual_reference():
    """2026-09-08: shared visual context should compress setup, not license a private caption.

    The watched failure and its rewrite remain off wire. These assertions pin the semantic edit
    on the compressed system surface without handing a minimal-thinking model example copy.
    """
    lowered = " ".join(_SYSTEM.lower().split())
    assert "minimum sufficient reference" in lowered
    assert "default to omission or the least explicit natural reference" in lowered
    assert "replace each literal visual description with an implicit reference" in lowered
    assert "if meaning and the conversational move survive, use the implicit version" in lowered
    assert "exact identity changes the point or distinguishes possible referents" in lowered
    assert "never merely to prove grounding, identify the selected item, or add textual specificity" in lowered
    assert "attachment itself can supply an immediately obvious referent" in lowered
    assert "full literal inventory in private referenced and item_description, not in the message" in lowered
    assert "do not force implicit wording where it creates real ambiguity" in lowered
    assert "specificity may come from how it fits the attached item" in lowered
    assert "subject to minimum sufficient reference" in lowered
    assert "tiled bench" not in lowered
    assert "glass of wine" not in lowered


def test_system_prompt_ships_the_spoken_register_rules():
    """2026-09-05 register rewrite. Every rule above governs CONTENT or STRUCTURE; none of them
    governed REGISTER, so a live opener could pass all of them and still read as written by a
    machine (an earnest second-person appraisal in flawless written grammar, with deictic
    filler and a question that restated its own setup).

    THE MIRRORING CONTRACT (ops/OPENER-REDESIGN.md 3.1): config.yaml's opener.style carries the
    LONG form of each rule with its reasoning, and this constant carries the COMPRESSED
    statement of the same property. Both are sent in the SAME request, so the two must not
    disagree. The config-side twin of this test is
    tests/test_config_yaml_real.py::test_shipped_opener_style_ships_the_spoken_register_rules;
    an edit landing in one file and not the other fails a test in a file nobody would think to
    look at.

    The motivating opener and the owner's rewrite of it stay OFF the wire (2026-08-16
    de-templating decision) and are quoted only in the config-side twin's docstring; the
    negatives at the bottom of this test are what keep them off it.
    """
    lowered = _SYSTEM.lower()
    assert "spoken register" in lowered
    assert "the contractions a relaxed speaker would use" in lowered
    assert "natural spoken elision" in lowered
    assert "never internet slang, chat abbreviations, meme phrasing" in lowered
    assert "idiom fit" in lowered
    assert "idiom only when it is contemporary" in lowered
    assert "everyday, immediately understandable on first reading in ordinary conversation" in lowered
    assert "semantically apt to the item and point" in lowered
    assert "natural when spoken" in lowered
    assert "idioms that are dated, literary, formal, obscure, forced, or tied to a passing trend" in lowered
    assert "does not license internet slang, memes, or borrowed caption wording" in lowered
    assert "makes her stop to decode the point" in lowered
    # The scope limiter. Its config.yaml twin carries it ("this licenses wording only:
    # spelling, capitalization, and every punctuation rule here stand unchanged"), and this
    # compressed copy sits next to CASUAL OR PUNCTUATION and the no-dash / ASCII HARD RULEs it
    # could otherwise be read as licensing an exception to. Compression is the contract between
    # the two copies; dropping a scope limiter is not compression, it is a wider rule.
    assert "licenses wording only" in lowered
    assert "say it once" in lowered
    assert "never restate a connection or context the message already established" in lowered
    assert "compliment as remark" in lowered
    assert "not an earnest verdict on her" in lowered
    assert "vary the shape" in lowered
    assert "specific item determine the whole message's sentence count, clause pattern" in lowered
    assert "no structure is the default" in lowered
    assert "never point at the medium" in lowered
    assert "rephrase so no hyphen is needed" in lowered
    assert "observation announced as what something looks like" not in lowered
    assert "question offering exactly two choices" not in lowered

    # De-templating: no concrete opener copy, neither the failing draft nor its rewrite, may
    # reach a model-facing string -- a minimal-thinking model imitates whatever wording is
    # salient in front of it, which is how the 2026-08-15 "my money is on" incident happened.
    assert "classic sense of style" not in lowered
    assert "you got there" not in lowered
    assert "hold court" not in lowered


def test_system_prompt_ships_modifier_clarity_without_incident_copy():
    """The system-instruction mirror must cover modifier attachment, not merely whether nouns
    and pronouns have referents. The concrete Maja failure stays here, off wire: “You look
    completely at home bundled up in all that snow” can make the setting sound like what she is
    bundled in even though the model identified the coat, beanie, and snow correctly.
    """
    lowered = _SYSTEM.lower()
    assert "modifier clarity" in lowered
    assert "every modifying phrase must have only one natural attachment" in lowered
    assert "on first reading" in lowered
    assert "if the phrase's placement permits a plausible unintended meaning" in lowered
    assert "reorder or rephrase the line" in lowered
    assert "bundled up in all that snow" not in lowered
    assert "completely at home bundled up" not in lowered


def test_system_prompt_ships_the_no_grading_rule():
    """2026-09-06: COMPLIMENT AS REMARK told the model to move praise off her and onto the
    visible thing, and the model complied perfectly -- openers kept landing on a taste verdict
    on the thing itself ("... is an elite move"). Moving WHO gets graded does not stop the
    grading; NO GRADING is the rule that forbids the verdict itself, whatever it lands on.

    THE MIRRORING CONTRACT (ops/OPENER-REDESIGN.md 3.1): config.yaml's opener.style carries the
    long form with its reasoning (tests/test_config_yaml_real.py::
    test_shipped_opener_style_ships_the_no_grading_rule), and this constant carries the
    compressed statement of the same property. The third copy, the response schema's opener
    description, is pinned in tests/test_gemini_opener.py; the angle field's self-check twin is
    pinned separately in this file (test_angle_field_carries_the_no_grading_self_check).
    """
    lowered = _SYSTEM.lower()
    assert "no grading" in lowered
    assert "substitution test" in lowered
    # The escape hatch: when nothing but a grade is available, cut the beat rather than invent
    # one. This is the same 2026-08-16 minimum-invention lesson applied to a new failure mode --
    # see the code addendum above _SYSTEM for why this must stay a shorter message, never a
    # substitute claim.
    assert "cut that beat and let one specific question be the whole message" in lowered
    # Subordination to COMPLIMENT AS REMARK, not competition with it: moving praise onto the
    # thing was already correct and must stay; NO GRADING narrows what counts as landing there.
    assert "narrows compliment as remark rather than competing with it" in lowered
    # 2026-09-06 (b): PLAYFUL HYPERBOLE carve-out against the SUBSTITUTION TEST.
    assert "except an unmistakably nonliteral playful hyperbole" in lowered
    assert "stays playful framing rather than an assessment of quality" in lowered
    # 2026-09-06 (b): the fallback question is still subject to VARY THE SHAPE.
    assert "still subject to vary the shape so the same single sentence" in lowered
    assert "never becomes its own template" in lowered


def test_system_prompt_folds_qualification_into_compliment_as_remark():
    """OWNER DECISION 2026-09-06: NARROW COMPLIMENT AS REMARK, do not retire it. Long-form
    rationale (69% prohibition/mechanical vs 21% positive specification, 94.7% two sentences,
    98.9% question final) is pinned in tests/test_config_yaml_real.py::
    test_shipped_opener_style_folds_qualification_into_compliment_as_remark; this test pins the
    same fix in the compressed _SYSTEM mirror.

    THE ORIGINAL DEFECT: this compressed copy read the same way as the long form -- COMPLIMENT
    AS REMARK taught "thing as the sentence's subject, understated over emphatic" with no
    qualification of its own, and only NO GRADING's later "narrows COMPLIMENT AS REMARK"
    sentence (still pinned above in test_system_prompt_ships_the_no_grading_rule) supplied the
    missing limit. THE 2026-09-06 (c) FIX folded a qualification into COMPLIMENT AS REMARK's
    own sentence ("unable to survive being detached from what was actually noticed") ahead of
    the technique it gates.

    THE (c) FIX'S OWN DEFECT, closed here (2026-09-06 (d)): that restated qualification was a
    second, subtly different portability test with NO exception, while NO GRADING's
    SUBSTITUTION TEST (still pinned above) explicitly carves out an unmistakably nonliteral
    PLAYFUL HYPERBOLE. An anchored hyperbolic compliment whose predicate could transfer to a
    different photo therefore passed NO GRADING but failed COMPLIMENT AS REMARK's own test --
    opposite verdicts on the same sentence. THE FIX: COMPLIMENT AS REMARK no longer restates its
    own portability test; it defers to NO GRADING's SUBSTITUTION TEST directly, so the one
    exception lives in one place. NO GRADING's own subordination sentence stays UNCHANGED.

    (d)'S OWN DEFECT, closed here (2026-09-06 (g)): (d) fixed only the portability half of
    COMPLIMENT AS REMARK's two independent requirements. The MAGNITUDE half ("understated over
    emphatic") carried no exception of its own, so a strong, unmistakably-nonliteral (and
    therefore emphatic) hyperbolic compliment passed the (d)-fixed portability test while still
    failing magnitude -- opposite verdicts again. THE FIX: this compressed copy now inserts one
    clause right after naming the SUBSTITUTION TEST -- "whose PLAYFUL HYPERBOLE exception covers
    tone too" -- so the same exception governs both the portability test and the magnitude
    phrase that follows it in the same sentence, without restating PLAYFUL HYPERBOLE's own
    definition a second time.
    """
    lowered = _SYSTEM.lower()
    assert "compliment as remark" in lowered
    assert "not an earnest verdict on her" in lowered
    assert "moving the praise onto the thing is not enough alone" in lowered
    assert "it must also pass no grading's substitution test below" in lowered
    # MUTATION GUARD: the (c) fold's own restated portability test must be GONE, not merely
    # supplemented -- its absence is what proves COMPLIMENT AS REMARK no longer carries a
    # second, exception-free test that could contradict NO GRADING's SUBSTITUTION TEST.
    assert "unable to survive being detached from what was actually noticed" not in lowered
    assert "remain inseparable from a specific observation about that item" not in lowered
    # The forward reference ("below") must be literally true: COMPLIMENT AS REMARK's pointer
    # has to precede the SUBSTITUTION TEST sentence it names, not follow it.
    assert lowered.index("compliment as remark") < lowered.index("substitution test")
    # 2026-09-06 (g): the same SUBSTITUTION TEST reference now also covers the magnitude phrase
    # ("understated over emphatic") that follows it in the same sentence, one exception stated
    # once rather than a second carve-out bolted onto the magnitude clause.
    # 2026-09-06 (h): the exception is hoisted ONCE to govern every condition that follows it,
    # rather than being attached to whichever clause was last reported. (g) subordinated only
    # the magnitude clause and the contradiction relocated a third time into "lands sideways".
    assert "exception carries here and governs every condition that follows" in lowered
    assert "covering both portability and volume" in lowered
    # Resolved in the NON-permissive direction: placement survives the exception.
    assert "landing sideways survives that exception" in lowered
    assert "never becomes the point" in lowered
    assert "playful hyperbole exception covers tone too" not in lowered
    assert "understated over emphatic" in lowered


def test_no_grading_rule_ships_no_example_verdict_vocabulary():
    """De-templating (2026-08-16): naming the banned verdict vocabulary on the wire would hand a
    minimal-thinking model salient wording to imitate, exactly the mechanism that produced 5/5
    "I bet" openers from a two-hedge list on 2026-08-11. NO GRADING's rule text must therefore
    state the general SHAPE (a predicate that would fit unchanged under a different woman's
    different photo) without naming any of the concrete corpus phrases that motivated it; those
    stay in code comments, the design doc, and tests only (per the ground rule enforced across
    this file, config.yaml, and _SCHEMA).
    """
    lowered = _SYSTEM.lower()
    schema_text = json.dumps(opener_mod._SCHEMA).lower()
    for word in ("elite", "iconic", "top tier", "masterpiece", "unmatched", "power move"):
        assert word not in lowered, f"_SYSTEM must not name banned verdict vocabulary: {word!r}"
        assert word not in schema_text, f"_SCHEMA must not name banned verdict vocabulary: {word!r}"


def test_angle_field_carries_the_no_grading_self_check():
    """`angle` is the model's only pre-opener scratchpad: every model in the cascade runs with
    thinkingLevel minimal (config.yaml opener.thinking), so there is no hidden reasoning trace
    anywhere else for the model to catch its own grading before it commits to `opener`. The
    schema's own comment near line 302 records that the angle field carries this instruction for
    exactly that reason -- putting the self-check only in `opener`'s description would let the
    model discover the grade only after it had already written the message.
    """
    angle_description = opener_mod._SCHEMA["properties"]["angle"]["description"].lower()
    assert "not a grade, rank, or" in angle_description
    assert "verdict on how good" in angle_description
    assert "would not fit unchanged under a different woman's different photo" in angle_description


def test_angle_field_carries_the_she_is_the_one_who_knows_self_check():
    """Same rationale as the NO GRADING self-check directly above: `angle` is the model's only
    pre-opener scratchpad under thinkingLevel minimal, so the location-inference self-check has
    to land there too, not only in `opener`'s description, or the model discovers the problem
    only after it has already written the message. See
    tests/test_config_yaml_real.py::test_shipped_opener_style_ships_the_she_is_the_one_who_knows_rule
    for the motivating "Looks like Rome, right?" incident.
    """
    angle_description = opener_mod._SCHEMA["properties"]["angle"]["description"].lower()
    assert "she was in the world the item shows and you were not" in angle_description
    assert ("confirm that the opener states any inference about it as your own uncertainty"
            ) in angle_description
    assert "never closes it with a tag whose only job is to collect her agreement" in angle_description
    assert "never as an appearance she is asked to agree with" in angle_description


# 2026-09-06 off-wire regression pin (de-templating rule: this vocabulary and this exact copy
# belong only in comments, the design doc, and tests -- never in a model-facing string). These
# are two of the roughly 20-of-95 local-corpus openers that motivated NO GRADING; see the code
# addendum above _SYSTEM in opener.py for the fuller measurement and its caveats.
_NO_GRADING_MOTIVATING_OPENERS = (
    "Starting the new year with a massive spread of sushi is an elite move. Was that the main "
    "event for the night or just the appetizer?",
    "Matcha tiramisu for a birthday cake is such an elite move. Did you make that yourself?",
)


def test_no_grading_motivating_openers_stay_off_every_prompt_surface():
    """The concrete drafts that motivated NO GRADING must never themselves become prompt copy --
    that would hand a minimal-thinking model salient wording to imitate (2026-08-16
    de-templating decision). Checked against all three on-wire copies this file can see
    directly; the retry hint block (copy 4 of 4) is deliberately untouched by this rule, per the
    code addendum above _SYSTEM.
    """
    schema_text = json.dumps(opener_mod._SCHEMA)
    for opener_text in _NO_GRADING_MOTIVATING_OPENERS:
        assert opener_text not in _SYSTEM
        assert opener_text not in schema_text
        assert "elite move" not in _SYSTEM.lower()
        assert "elite move" not in schema_text.lower()


# 2026-09-14 off-wire regression pin (de-templating rule): the exact live draft that motivated
# SHE IS THE ONE WHO KNOWS -- see
# tests/test_config_yaml_real.py::test_shipped_opener_style_ships_the_she_is_the_one_who_knows_rule
# for the full incident. Naming this literal construction in a model-facing string would prime a
# minimal-thinking model to imitate it, the same mechanism that produced the 2026-08-11 "I bet"
# incident and the 2026-09-06 "elite move" incident.
_SHE_IS_THE_ONE_WHO_KNOWS_MOTIVATING_OPENER = "Looks like Rome, right?"


def test_looks_like_rome_right_motivating_opener_stays_off_every_prompt_surface():
    """The concrete draft that motivated SHE IS THE ONE WHO KNOWS must never itself become
    prompt copy. Checked against all three on-wire copies this file can see directly (config.yaml
    opener.style, _SYSTEM, and json.dumps(_SCHEMA)). The retry hint block (copy 4 of 4) DID
    receive this rule's wording clause -- only the HARD REJECTION list was deliberately left
    unchanged -- but it is built per attempt inside _text_part rather than being a module
    constant, so its own absence pin lives in tests/test_gemini_opener.py::
    test_nonempty_retry_hint_carries_the_she_is_the_one_who_knows_amendment. No surface is exempt
    from the de-templating rule. "rome" is checked
    with a word-boundary regex because a bare substring match could collide with an ordinary word.
    """
    import re

    from operation_love.config import load as load_config

    style = load_config("config.yaml").opener.style
    schema_text = json.dumps(opener_mod._SCHEMA)
    for surface_name, surface in (("config.yaml opener.style", style),
                                  ("_SYSTEM", _SYSTEM),
                                  ("_SCHEMA", schema_text)):
        lowered = surface.lower()
        assert _SHE_IS_THE_ONE_WHO_KNOWS_MOTIVATING_OPENER.lower() not in lowered, surface_name
        assert "looks like rome" not in lowered, surface_name
        assert not re.search(r"\brome\b", lowered), surface_name
        assert "right?" not in lowered, surface_name


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


def test_generate_accepts_a_terminal_plain_text_smiley_unchanged():
    """A sparing, terminal :) is ASCII and reaches the phone exactly as the model wrote it."""
    payload = _gemini_response({
        "opener": "That pirate flag makes this look like the most serious dune expedition :) ",
        "referenced": "x", "item_index": 1,
    })
    result = _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert result.opener == "That pirate flag makes this look like the most serious dune expedition :)"


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
    # Both opener sinks take **kw: from 2026-09-05 (b) the service passes prompt_sha256 (the
    # prompt-era stamp) as a keyword, and a `*a`-only fake would raise TypeError inside the
    # service's non-fatal try/except rather than discarding the row as this double intends.
    def record_spend(self, *a):
        pass

    def record_opener(self, *a, **kw):
        pass

    def record_opener_rejection(self, *a, **kw):
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


def test_generate_still_raises_parse_error_on_narrowed_here_is_scaffolding():
    """2026-09-06 (b) "HERE'S" collision fix: narrowing rule 2 to the determiner-plus-meta-noun
    object shape must not stop catching real preamble of that exact shape. "Here's an option:"
    names the output as an object just as "Here's the response:" does, so it must still burn a
    retry rather than reach her phone."""
    payload = _gemini_response({
        "opener": "Here's an option: Skiing or the beach, whichever you prefer?",
        "referenced": "x", "item_index": 1,
    })
    with pytest.raises(OpenerParseError, match="scaffolding") as exc:
        _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert exc.value.reason_code == REASON_SCAFFOLDING
    assert exc.value.raw_opener == "Here's an option: Skiing or the beach, whichever you prefer?"


def test_generate_accepts_a_natural_spoken_here_is_opening():
    """2026-09-06 (b) "HERE'S" collision fix: the same generate() path that used to reject this
    exact opener (ops/OPENER-REDESIGN.md's live-verified example) must now accept it, since
    "hoping" is never the meta noun the narrowed rule requires."""
    text = "Here's hoping that trail's as steep as it looks. Have you done the full loop?"
    payload = _gemini_response({"opener": text, "referenced": "x", "item_index": 1})
    result = _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert result.opener == text


def test_generate_rejects_a_preemptive_negative_framing_disclaimer():
    text = "Zero judgment here, you wear the Mickey ears with total confidence."
    payload = _gemini_response({"opener": text, "referenced": "Mickey ears", "item_index": 1})
    with pytest.raises(OpenerParseError, match="negative social interpretation") as exc:
        _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert exc.value.reason_code == REASON_PREEMPTIVE_DISCLAIMER
    assert exc.value.raw_opener == text


def test_preemptive_disclaimer_is_retried_and_direct_second_attempt_succeeds():
    bad = _gemini_response({
        "opener": "Zero judgment here, you wear the Mickey ears with total confidence.",
        "referenced": "Mickey ears", "item_index": 1,
    })
    clean_text = "You wear the Mickey ears with total confidence. Which ride is nonnegotiable?"
    clean = _gemini_response({
        "opener": clean_text, "referenced": "Mickey ears", "item_index": 1,
    })
    client = _opener(_Transport([(200, bad), (200, clean)]))
    service = OpenerService(client, _NeverBudgetTracker(), _DiscardingStore(), "casual")

    pick = service.maybe_opener("r", "hinge", Profile(photos=[b"a"]))

    assert pick is not None
    assert pick.text == clean_text


def test_generate_rejects_the_reported_courage_to_jump_inference_before_send():
    payload = _gemini_response({
        "opener": ("Did you work up the courage to jump or were you happy just taking in "
                   "the scenery?"),
        "referenced": "a bridge in a travel photo", "item_index": 1,
    })
    with pytest.raises(OpenerParseError, match="sensitive dangerous activity") as exc:
        _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert exc.value.reason_code == REASON_SENSITIVE_INFERENCE
    assert exc.value.raw_opener == (
        "Did you work up the courage to jump or were you happy just taking in the scenery?")


def test_generate_rejects_the_reported_unconfirmed_location_followup_before_send():
    payload = _gemini_response({
        "opener": _REPORTED_UNCONFIRMED_LOCATION_FOLLOWUP,
        "referenced": "snowy mountains and a frozen lake",
        "angle": "guessing the location, then asking what she did there",
        "item_description": "photo of her beside a frozen lake",
        "item_index": 1,
    })
    transport = _Transport([(200, payload)])

    with pytest.raises(OpenerParseError, match="unconfirmed location guess") as exc:
        _opener(transport).generate(Profile(photos=[b"a"]), style="s")

    assert transport.calls == 1
    assert exc.value.reason_code == REASON_UNCONFIRMED_LOCATION_FOLLOWUP
    assert exc.value.raw_opener == _REPORTED_UNCONFIRMED_LOCATION_FOLLOWUP


def test_unconfirmed_location_followup_is_retried_and_confirmation_only_succeeds():
    bad = _gemini_response({
        "opener": _REPORTED_UNCONFIRMED_LOCATION_FOLLOWUP,
        "referenced": "snowy mountains and a frozen lake",
        "angle": "guessing the location, then asking what she did there",
        "item_description": "photo of her beside a frozen lake",
        "item_index": 1,
    })
    clean_text = "That looks a lot like Lake Louise in deep winter. Am I close?"
    clean = _gemini_response({
        "opener": clean_text,
        "referenced": "snowy mountains and a frozen lake",
        "angle": "guessing the location and leaving it for her to confirm or correct",
        "item_description": "photo of her beside a frozen lake",
        "item_index": 1,
    })
    transport = _Transport([(200, bad), (200, clean)])
    service = OpenerService(
        _opener(transport), _NeverBudgetTracker(), _DiscardingStore(), "casual")

    pick = service.maybe_opener("r", "hinge", Profile(photos=[b"a"]))

    assert transport.calls == 2
    assert pick is not None
    assert pick.text == clean_text


_REPORTED_PREMATURE_FUTURE_OPENER = (
    "That city skyline makes for a great backdrop. Since you appreciate a man who handles "
    "the planning, I will make sure our first spot has a view just as good."
)


def test_premature_shared_future_guard_catches_reported_opener_but_allows_invitations():
    assert _premature_shared_future_markers(_REPORTED_PREMATURE_FUTURE_OPENER) == [
        "assumed shared first date or outing",
        "promise of future performance",
    ]
    assert _premature_shared_future_markers(
        "You seem fun. Let's grab a drink this week if you're free."
    ) == []
    assert _premature_shared_future_markers(
        "That skyline feels like a subtle hint for whoever gets to plan the date."
    ) == []
    assert _premature_shared_future_markers(
        "Good news, I'm exactly the man who handles the planning."
    ) == ["self-advertised dating role"]
    assert _premature_shared_future_markers(
        "I can handle the date planning, you just show up."
    ) == ["promise to handle date planning"]
    assert _premature_shared_future_markers(
        "You seem fun. I'm in if you want to grab a drink."
    ) == []


def test_generate_rejects_the_reported_premature_shared_future_before_send():
    payload = _gemini_response({
        "opener": _REPORTED_PREMATURE_FUTURE_OPENER,
        "referenced": "a red dress against a city skyline", "item_index": 1,
    })
    with pytest.raises(OpenerParseError, match="unaccepted shared plan") as exc:
        _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert exc.value.reason_code == REASON_PREMATURE_SHARED_FUTURE
    assert exc.value.raw_opener == _REPORTED_PREMATURE_FUTURE_OPENER


def test_premature_shared_future_is_retried_and_a_relaxed_second_attempt_succeeds():
    bad = _gemini_response({
        "opener": _REPORTED_PREMATURE_FUTURE_OPENER,
        "referenced": "a red dress against a city skyline", "item_index": 1,
    })
    clean_text = "That skyline feels like a subtle hint for whoever gets to plan the date."
    clean = _gemini_response({
        "opener": clean_text,
        "referenced": "a red dress against a city skyline", "item_index": 1,
    })
    client = _opener(_Transport([(200, bad), (200, clean)]))
    service = OpenerService(client, _NeverBudgetTracker(), _DiscardingStore(), "casual")

    pick = service.maybe_opener("r", "hinge", Profile(photos=[b"a"]))

    assert pick is not None
    assert pick.text == clean_text


def test_sensitive_inference_is_retried_and_a_grounded_second_attempt_succeeds():
    bad = _gemini_response({
        "opener": "Did you work up the courage to jump from the bridge?",
        "referenced": "a bridge in a travel photo", "item_index": 1,
    })
    clean = _gemini_response({
        "opener": "That view looks like it was worth the trip.",
        "referenced": "a bridge in a travel photo", "item_index": 1,
    })
    client = _opener(_Transport([(200, bad), (200, clean)]))
    service = OpenerService(client, _NeverBudgetTracker(), _DiscardingStore(), "casual")

    pick = service.maybe_opener("r", "hinge", Profile(photos=[b"a"]))

    assert pick is not None
    assert pick.text == "That view looks like it was worth the trip."


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
    assert "delivery has not been decided at this parsing stage" in out


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
