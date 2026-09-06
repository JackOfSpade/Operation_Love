"""Provider-backed opener generation, with enforced JSON output.

Structured outputs guarantee the model returns exactly the fields in _SCHEMA below —
no "Sure! Here's a great opener:" preamble (the problem that killed the original
ChatGPT attempt). The provider is behind a small interface so it stays swappable.

Returns the parsed opener AND the token usage, so the caller can record spend
and enforce the per-run budget (see operation_love.costing).
"""
from __future__ import annotations

import base64
import hashlib
from http.client import HTTPException
import io
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ..costing import Usage
from ..perception.capture import Profile
from ..typography import (
    describe_char,
    fold_to_ascii,
    undeliverable_chars,
    undeliverable_sequences,
)

# The value `item_index` carries when the model gave us no usable item number at all: a
# missing field, a null, a non-integer, or anything below the first item. NOT "item 0" and not
# "the first item" -- the item numbering the model is given is 1-BASED (ops/OPENER-REDESIGN.md
# 5.7), so zero is out of band by construction and there is no legal index it can be confused
# with. It exists as a named constant precisely so a reader cannot mistake a bare `0` here for
# the old `referenced_index` default, which really did mean "the first image".
#
# What a consumer must do with it is NOT decided here -- doc 5.3's rule is "treat a missing
# table as a hard stop, never as a reason to fall back to a fixed coordinate", and the
# navigation half of that (the driver-owned translation table, the removal of the hearts[0]
# fallback) is a LATER workflow. What IS decided here, as of the 2026-08-12 correction, is that
# it must never arrive at a consumer wearing a legal value's clothes: see
# OpenerPick.capture_order_index, which maps it to None rather than to any tappable index,
# because the driver's own capture-order space has a perfectly legal 0 in it.
ITEM_INDEX_ABSENT = 0

# The number of the first item in the list the model is shown. 1-based per doc 5.7, and named
# rather than written as a bare 1 in the clamp below because "is this index 0-based or 1-based"
# is the exact question that made the old field unsafe to reinterpret.
FIRST_ITEM_INDEX = 1

# WHICH LIST `item_index` COUNTS. The number alone cannot say, which is the entire class of bug
# doc 5.3 is written against ("index space belongs to the driver"), and leaving it to a comment
# is what let a 1-based item number be handed to a 0-based capture-order parameter and be
# accepted as a confident, in-range, on-target answer. So generate() records the space it
# actually built the request in, on the result, and every consumer branches on it rather than
# on an assumption about which shape production happens to send today.
#
# These are the model-facing spaces only. The DRIVER's capture-order space is deliberately NOT
# one of them: nothing in this module can produce a value in it, and naming it here would
# invite exactly the silent conversion this constant exists to stop.
#
# PROFILE_PHOTOS: the numbered images were `profile.photos`, i.e. the raw scroll frames, sent in
# capture order and numbered 1..len(photos) (see _text_part's frame branches). This is the
# legacy shape and it is what production still sends. The frames are not really items -- one
# card can appear in three of them and one frame can hold two cards -- which is the whole reason
# doc 5.2 replaces them with crops; but the CORRESPONDENCE between the number the model returns
# and the driver's capture order is exact and derivable, so a consumer holding the driver may
# translate it (see OpenerPick.capture_order_index).
#
# MODEL_ITEMS: the numbered images were an ItemRequest's per-item crops (doc 5.1/5.2), numbered
# 1..item_count over the SELECTABLE items only. There is no way to get from one of these to
# anything the driver can tap without doc 5.3's driver-owned translation table, which is the
# next workflow -- so this space is untranslatable on purpose, and a consumer that cannot
# translate it must refuse to target rather than guess.
INDEX_SPACE_PROFILE_PHOTOS = "profile_photos"
INDEX_SPACE_MODEL_ITEMS = "model_items"

# FIELD ORDER IS LOAD-BEARING, not cosmetic (ops/OPENER-REDESIGN.md 3.4 and 5.7): item_index,
# then referenced, then angle, then item_description, then opener LAST. Every model in the
# cascade runs with minimal thinking (config.yaml opener.thinking), so the model has no
# scratchpad of any kind and the only place it can do its grounding work is an earlier OUTPUT
# field. The old order emitted `opener` first, which meant the message itself had to carry the
# description because nothing had absorbed it yet -- root cause #3 of the over-description bug
# this redesign fixes. Putting `referenced` first discharges the description into a field that
# is never sent to her, and `angle` makes the model commit to what its message is doing before
# it writes the message.
#
# `item_index` leads because it is now a CHOICE the opener has to follow, not a label attached
# to an opener that was already written (doc 5.1: "the model receives the profile and returns
# which item to like plus the opener", and selection is by best ANGLE rather than best photo).
# Emitting it after the message would invert that: the model would write first and then pick
# whichever item the message happened to suit, which is the blind-then-repair order Part B
# exists to delete. `item_description` sits immediately before `opener` for the same
# discharge-it-first reason as `referenced`, and because doc 5.8's pre-flight cross-check reads
# it against our own crop at that index.
#
# Whether Gemini actually honours declared property order in responseJsonSchema is a
# HYPOTHESIS, not a documented guarantee (doc 3.4 says to A/B it rather than assume it), so
# this ordering is cheap insurance that costs nothing if the hypothesis is wrong. The field
# descriptions below separate private grounding from public payoff: `referenced` holds the full
# literal description, while `angle` and `opener` require conversational value beyond it. That
# routing survives even on a model that ignores order entirely.
_SCHEMA = {
    "type": "object",
    "properties": {
        "item_index": {
            "type": "integer",
            "description": "The number of the item your opener is about. This is also the item "
                           "that will be liked, so the message and the like always land on the "
                           "same item. It must be the opener's clear primary subject: the "
                           "message should feel natural directly under it, not as an excuse to "
                           "talk about a different image. Your opener must justify why this "
                           "selected item was liked; supporting context may enrich it but cannot "
                           "replace it as the reason, subject, or payoff. "
                           "The list it indexes is the numbered items in THIS "
                           "request and nothing else: they are numbered from 1, in the order "
                           "they are given, and you may only choose a number that was actually "
                           "given to you. Some blocks are shown WITHOUT a number, for context "
                           "only; you may use them to support a connection to the selected item, "
                           "but you can never pick one or make it the opener's main premise.",
        },
        "referenced": {
            "type": "string",
            "description": "What you are reacting to, described in full: the selected item's "
                           "exact photo or prompt detail that is the opener's primary "
                           "anchor. THIS FIELD IS NEVER SENT TO HER. If unnumbered context "
                           "supports the connection, distinguish it as supporting context rather "
                           "than replacing this selected-item anchor. Record only what is visibly "
                           "shown or explicitly stated, not "
                           "an action or backstory you inferred. Put the whole description "
                           "here, including any header, caption, or prompt printed with a photo "
                           "and how it frames the photo. Describe any other visible people "
                           "neutrally unless her profile explicitly states their relationship; "
                           "never infer friend, partner, or family from proximity, so the "
                           "opener does not have to carry it. The opener may name "
                           "only the visible detail needed as setup, and its final conversational "
                           "point must add something beyond that description.",
        },
        "angle": {
            "type": "string",
            "description": "In your own words, what your opener is doing: what you are "
                           "observing, asking, guessing, teasing about, or connecting. State "
                           "the conversational payoff beyond merely identifying what is visible. "
                           "The selected item must remain the angle's primary anchor and reason "
                           "for the Like; unnumbered context may support a connection but cannot "
                           "replace the selected item as the subject or payoff. "
                           "The angle must respect any header, caption, or prompt attached to "
                           "the photo; that text defines the photo's intended context. "
                           "A guess is optional. If you use one, choose the least speculative "
                           "interpretation that fits ordinary explanations of the scene and do "
                           "not invent a motive, purpose, cause, or unseen circumstance. The "
                           "conclusion of a guess must not itself be directly visible or stated "
                           "in the item, its text, a sign, or elsewhere in her profile. Visible "
                           "and stated facts may be clues, not guessed conclusions. If the opener "
                           "uses a guess, state how it leaves the conclusion unconfirmed and gives "
                           "her room to confirm or correct it. Any second beat must not assume the "
                           "guess is true or ask about an experience, preference, or consequence "
                           "that only makes sense if it is true. If a location is inferred from "
                           "an image rather than stated in her profile text, the location guess "
                           "must be the whole conversational move: end after it or ask only "
                           "whether that location itself is right. "
                           "For every visible detail named in the opener, state how it is used "
                           "by the conversational move. If it asks a question, name the one "
                           "underlying question; alternatives must be parallel, contrasting "
                           "answers to it, never unrelated dimensions joined by 'or'. Identify "
                           "the question's subject and confirm that every pronoun or shorthand "
                           "noun has one immediately obvious referent. Across beats, any change "
                           "of referent must be explicit and immediately clear. "
                           "Confirm the opener's conversational point is not a grade, rank, or "
                           "verdict on how good the item is, and that its predicate would not fit "
                           "unchanged under a different woman's different photo. "
                           "Confirm that its "
                           "most natural honest reply lets her share a preference, perspective, "
                           "inspiration, or experience rather than defend or diminish herself. "
                           "Confirm that the opener does not audition for a role described in "
                           "her profile, promise how the sender will perform for her, or assume "
                           "a match, date, relationship, or shared future before she replies. A "
                           "direct invitation may propose a get-together, but it must leave her "
                           "acceptance open rather than speak as though the plan already exists. "
                           "Free text, not a fixed list of choices. Recorded for analysis only, "
                           "never sent to her.",
        },
        "item_description": {
            "type": "string",
            "description": "A short description of the item you picked: say whether it is a "
                           "photo or a written prompt, and in a few words what it shows or "
                           "says. For a photo, include any header, caption, or prompt printed "
                           "with it. Describe any other visible people neutrally unless her "
                           "profile explicitly states their relationship; never infer friend, "
                           "partner, or family from proximity. This is how we check that the "
                           "item you numbered is the item "
                           "we think it is. Never sent to her.",
        },
        "opener": {
            "type": "string",
            "description": "The message to send, bare text only. She reads it while looking at "
                           "the selected item, so it must feel natural directly under it and "
                           "justify why it was liked. Supporting context may enrich the message "
                           "but cannot replace the selected item as its reason, subject, or "
                           "payoff. It may name a visible detail as setup, but its final point "
                           "must do something conversational beyond describing that detail. "
                           "Every named setup detail must be necessary to that move; cut it if "
                           "the later point still works without it. A question asks one coherent "
                           "thing; use 'or' only for parallel, contrasting answers to it. Every "
                           "pronoun, shorthand noun, and question subject must have one "
                           "immediately obvious referent. Across two beats, any change of "
                           "referent must be explicit and immediately clear. "
                           "MODIFIER CLARITY: every modifying phrase must have only one natural "
                           "attachment on first reading. If the phrase's placement permits a "
                           "plausible unintended meaning, reorder or rephrase the line. "
                           "Its most natural honest reply should let her share a preference, "
                           "perspective, inspiration, or experience, never require self "
                           "justification. "
                           "A guess must remain unconfirmed until she replies; no later statement "
                           "or question may assume it is correct. An image-derived location guess "
                           "is the only conversational move before she replies: end after it or "
                           "ask only whether the location itself is right. Its confirmation is "
                           "the payoff; do not ask about an activity, reason, preference, feeling, "
                           "experience, or consequence there before she confirms it. "
                           "Do not advertise the sender as "
                           "the answer to a preference in her profile, promise what he will do "
                           "for her, or assume a match, date, relationship, or shared future. A "
                           "direct invitation may propose a get-together only while leaving her "
                           "acceptance open. "
                           "State the positive substance directly. Do not preface it by naming "
                           "an insulting, judgmental, awkward, pressuring, creepy, or offensive "
                           "interpretation and then denying that interpretation. "
                           "Write it in the spoken register a person texts in, with natural "
                           "contractions; deliver any compliment as an offhand remark about the "
                           "thing rather than an earnest verdict on her; moving the praise onto "
                           "the thing alone is not enough, it must also pass the SUBSTITUTION "
                           "TEST below; never restate context "
                           "the message itself already established, and never point at the photo "
                           "or profile as an object. "
                           "never assign the thing a grade, rank, or verdict on how good it is; "
                           "SUBSTITUTION TEST: if the predicate would fit unchanged under a "
                           "different woman's photo it is a grade rather than an observation, "
                           "except that an unmistakably nonliteral PLAYFUL HYPERBOLE stays "
                           "playful framing rather than an assessment of quality even where its "
                           "wording could transfer; "
                           "when the best thing to say about an item is how good it is, cut that "
                           "beat and let one specific question be the whole message rather than "
                           "inventing a claim to fill it, itself still subject to VARY THE SHAPE so "
                           "it does not become its own repeated template. "
                           "Maximum two sentences. No em dash, no hyphen.",
        },
    },
    "required": ["item_index", "referenced", "angle", "item_description", "opener"],
    "additionalProperties": False,
}

# THIS TEXT IS DUPLICATED, AND THE COPIES ARE SENT IN THE SAME REQUEST. config.yaml's
# opener.style carries a near-duplicate of the substance rules below; it arrives as the user
# turn's STYLE GUIDE while this constant arrives as systemInstruction, so the two disagreeing
# is not a cosmetic inconsistency but a real bug -- the model receives both at once. Each copy
# is pinned by a SEPARATE test (tests/test_opener.py against this constant,
# tests/test_config_yaml_real.py against the real config file), so editing one alone also
# fails a test in a file nobody would think to look at. The duplication itself is known debt,
# recorded in ops/OPENER-REDESIGN.md 3.1.
#
# Division of labour between the two copies, per doc 3.1: the LONG form of the rule and its
# reasoning live in config.yaml, which is owner tunable and is where voice belongs. This
# constant carries a COMPRESSED statement of the same
# property plus the mechanics that have nowhere else to live (field semantics, output format,
# the two HARD RULEs).
#
# Rewritten 2026-08-11 per ops/OPENER-REDESIGN.md Part A. What changed, and why each change:
#   - "Ground this opener in exactly ONE concrete detail from a specific photo or prompt" is
#     GONE (doc 1.1 root cause #1). It never distinguished being GROUNDED IN a detail from
#     NAMING it, so the model recited the detail back inside the message -- under a photo she
#     is already looking at, which reads as if we think she cannot see it. Its replacement is
#     the falsifiability rule: the opener must carry a claim that could be wrong. Description
#     is unfalsifiable by construction, which is exactly why it proves nothing.
#   - "Favor a sincere observation, a direct low-pressure invitation, or one open,
#     easy-to-answer question about that detail" goes with it: it made an unfalsifiable
#     observation and an interview question the two default shapes. Questions are demoted
#     rather than banned (doc 2.2): a claim she can correct is the easiest reply to give.
#   - "ONE short sentence is preferred" is GONE (doc 3.2.1); the TWO sentence ceiling stays.
#     Preferring brevity for its own sake worked against a claim that needs room to exist
#     ("I know you were smiling, but I bet you were freezing out there" is eighteen words and
#     is the strongest opener in the whole design). Word economy is now tied to the actual
#     goal instead: spend no word on anything she can already see. Budget reallocated, not cut.
#   - The three guardrails (hedge the claim not yourself, guess the world not her identity,
#     never invent the sender) are compressed to one line each here; config.yaml carries them
#     in full. They are safety rules rather than style preferences, so both copies state them.
#
# Addendum 2026-08-11 (same day, later): a live dry run against a real profile produced 5/5
# openers that all opened with "I bet" -- exactly the entropy-collapse risk doc 3.6 predicted.
# This constant's HEDGE THE CLAIM line was the sharper half of the root cause: it named only
# two forms ("a hedge like I bet or I heard"), with "I bet" first in the very sentence the
# model reads immediately before it writes, on every request, on every model in the cascade
# (thinkingLevel minimal, so nothing intervenes between reading this and generating). Fixed by
# widening the named forms to seven, marking the list illustrative rather than exhaustive, and
# adding a standalone VARY THE OPENING instruction. config.yaml's copy got the same two changes
# so the pair keeps agreeing (doc 3.1). Neither change touches the underlying guardrail: hedge
# the claim, never the sender, is unchanged word for word at the start of the sentence.
#
# Addendum 2026-08-15: widening that list treated the symptom as a vocabulary problem and made
# the examples a seven-item stock menu. A live suggestion then opened with "my money is on the
# French Alps", a bare answer to an imaginary location question: the reader reasonably asks
# "for what?" The named list is removed. The guardrail now specifies its semantic job -- express
# uncertainty naturally when the claim needs it -- while SELF CONTAINED CLAIM requires the model
# to state the whole proposition. VARY THE OPENING now tells the item and angle to determine the
# wording rather than telling the model to rotate through an inventory. A second-beat question
# explicitly inherits the evidence boundary so it cannot smuggle in another invented premise.
#
# Addendum 2026-08-16: evidence-bounded sentences could still disagree with each other. Live
# suggestions asserted X and then asked whether X, asked about not-X, or abandoned X for a
# generic question about the scene. PREMISE CONSISTENCY makes the optional second beat accept
# the first claim as its working premise and advance it with a consequence or specific detail.
# The angle field carries the same instruction because it is the model's only pre-opener
# scratchpad under minimal thinking.
#
# Addendum 2026-08-16 (prompt de-templating): all concrete opener copy, worked scenarios and
# move menus were removed from the model-facing instructions. Even a labelled counterexample
# supplies salient wording for a minimal-thinking model to imitate. The prompt now specifies
# properties and failure categories only; tests keep the regressions concrete off-wire.
#
# Addendum 2026-08-16 (minimum invention): making falsifiability unconditional overcorrected
# the description problem. On an item with no natural inference it pressured the model to invent
# motives and circumstances merely to produce a correctable claim. A claim is now optional. A
# visible detail may be named as setup and fails only when description is the final payoff. The
# model must prefer the least speculative natural move, while unmistakably nonliteral hyperbole
# remains available because it is playful framing rather than asserted biography.
#
# Addendum 2026-08-16 (photo headers): a captioned photo is one compound profile item. The
# caption is not decoration: it can reverse the ordinary reading of the pixels, so the model
# must inventory it before drafting and treat it as authoritative context for the photo.
#
# Addendum 2026-08-17 (referent clarity): a live draft established one subject in its setup and
# silently switched to an ambiguous shorthand noun in its question. QUESTION COHERENCE already
# constrains the number and shape of questions, while PREMISE CONSISTENCY constrains their logic
# after a claim; neither says that a reader must be able to resolve what each noun refers to.
# REFERENT CLARITY adds that semantic property without blacklisting a word or supplying concrete
# copy that a minimal-thinking model could imitate.
#
# Addendum 2026-08-17 (reply comfort): coherence alone does not rule out a question whose honest
# answer makes her defend her intelligence, sincerity, or effort, or acknowledge an unflattering
# alternative. REPLY COMFORT tells the model to select an angle with an affirming answer space
# while deliberately stating no worked opener or wording pattern for it to imitate.
#
# Addendum 2026-09-02 (guess confirmation and information gain): a live opener guessed a country
# and immediately asked about her experience there, treating the guess as confirmed before she
# could answer. The old PREMISE CONSISTENCY instruction explicitly caused that failure by telling
# the second beat to accept X as its working premise. It is replaced by CONFIRMATION BOUNDARY:
# an uncertain conclusion stays uncertain for the entire opener, and any second beat must leave
# her room to confirm or correct it. INFORMATION GAIN separately prevents the model from dressing
# a plainly visible or stated fact, such as readable location text, up as a guess. Visible facts
# remain valid clues for a different inference; they are simply not themselves guess material.
#
# Addendum 2026-09-04 (positive social framing): a live draft began by denying judgment. That
# wording introduces the exact negative interpretation it tries to remove, making an otherwise
# positive observation sound self-conscious. The prompt now tells the model to state the
# substantive thought directly, and a narrow lexical backstop regenerates unmistakable leading
# disclaimers without treating ordinary negation as a style failure.
#
# Addendum 2026-08-12 (Part B, ops/OPENER-REDESIGN.md 5.1/5.7): the model now CHOOSES the item
# rather than labelling one after the fact, so two lines changed here and nothing else did.
# "The images are her profile in scroll order; set referenced_index to the 0-based index of the
# image your opener is about" is gone: `referenced_index` no longer exists, its 0-based index
# into raw SCROLL FRAMES no longer names anything (frames are not items -- one card can appear
# in three of them, and a frame can hold two cards), and the replacement PICK THE ITEM YOURSELF
# line states the three facts the new contract depends on: items are numbered from 1, the
# chosen item is also the one that gets liked (5.1's single call), and unnumbered context
# blocks may be read but never picked (5.3's two tiers). The selection criterion in that line
# is 5.1's, word for word in substance: best ANGLE, not most striking photo.
#
# Addendum 2026-08-12 (Part B, doc 5.1): the SELECTION CRITERION is now spelled out rather than
# compressed to a single clause. The line above said only "choose the item you have the best
# thing to say about, not the most striking picture", which states the preference but not the
# tradeoff it exists to settle, and a model with a page of photos in front of it has a strong
# prior toward the best photograph. Three things were added, all of them doc 5.1's:
#   - The tradeoff, made explicit: a plain item you can make a real claim about beats a
#     beautiful one you have nothing to say about. It is the same
#     premise/point distinction THE ONE RULE already makes, applied one step earlier -- the item
#     is the premise, so picking by how the item LOOKS optimises the half that is not the message.
#   - The failure mode, named as a failure: picking an item it has nothing to say about, which
#     leaves description as the only thing left to write. That is not a separate bug from
#     over-description, it is the upstream cause of it, and it is reachable while every wording
#     rule in Part A is obeyed, which is why naming it here is not redundant with THE ONE RULE.
#   - What the unnumbered tier IS, not merely that it exists: her vitals (age, job, school,
#     city), per doc 5.3's live capture. Their explicitly stated facts may contextualize the
#     selected item, but their visual details may not be used as an opener for a different
#     target. The distinction preserves useful profile-wide context without telling her about
#     "another photo" the comment did not attach to. Deliberately worded without a menu of moves
#     so the model derives its approach from the profile instead of imitating prompt copy.
# config.yaml's opener.style carries the long form of all three (doc 3.1's division of labour),
# and both copies are pinned by their own tests.
#
# Addendum 2026-08-12 (audit fix, "BUG 1"): the line above is no longer accurate and is left
# unedited only per this file's own "never rewrite existing lines" convention (see
# ops/OPENER-REDESIGN.md). config.yaml's opener.style is sent, byte-for-byte, on EVERY request
# regardless of shape (it is the user turn's STYLE GUIDE -- see _text_part), so its unconditional
# copy of PICK THE ITEM YOURSELF is intentionally kept only in this system instruction. Both
# auto and observe now send the same numbered item crops, so one selection rule is sufficient.
#
# Addendum 2026-09-05 (spoken register): a live opener passed every rule above and still
# read as AI: an earnest second person appraisal, uncontracted written grammar (measured:
# zero apostrophes across all 188 openers ever generated), a formal nominal where a person
# would use the plain noun, deictic filler pointing at the photo, and a question that
# restated the setup its first sentence had already established. Every prior rule governs
# content and structure; none governed register. Five property additions ship in both
# copies, lockstep with config.yaml (doc 3.1): SPOKEN REGISTER, SAY IT ONCE, COMPLIMENT AS
# REMARK, VARY THE SHAPE, and a rephrase clause on the hyphen HARD RULE (live data showed
# the model deleting hyphens and gluing compounds to satisfy it). SHARED CONTEXT RULE now
# also forbids pointing at the medium. Per the 2026-08-16 de-templating addendum no example
# copy ships in any model facing string; the motivating openers are pinned off wire in the
# tests. There are FOUR on wire copies of prompt rules, not two, a fact no earlier addendum
# recorded: config.yaml opener.style, this constant, the _SCHEMA field descriptions, and the
# retry hint block. Those last two were both touched this date but NOT with the same content:
# the _SCHEMA opener description received a compressed form of THREE of the five register
# properties (SPOKEN REGISTER, SAY IT ONCE, COMPLIMENT AS REMARK) plus the new never point at
# the medium clause, and deliberately not VARY THE SHAPE or the hyphen rephrase clause, while
# the retry hint block was corrected in an unrelated way, by completing its HARD REJECTION
# list. Do not go looking for SPOKEN REGISTER in the retry block and conclude the lockstep is
# broken. Rows written before this rewrite are not register comparable to rows written after
# it (the calibration warning near _redundant_description_markers now names both boundaries).
#
# Addendum 2026-09-05 (modifier attachment): the first watched Training draft after the
# spoken-register rewrite described the correct target and every item/identity verifier agreed,
# but one location phrase sat beside a clothing phrase closely enough to support a second,
# unintended grammatical relationship. REFERENT CLARITY did not cover the miss: every noun and
# question subject had an obvious referent, while the ambiguity lived in what the modifier
# attached to. MODIFIER CLARITY therefore ships in the same three stable prompt surfaces as the
# other semantic wording rules: config.yaml's long style, this system instruction, and the
# response schema's opener description. This remains prompt-only. A lexical pattern for the one
# observed sentence would overfit a general word-order problem and reject harmless uses, so no
# deterministic guard or new retry reason is added. The concrete incident stays in comments,
# documentation, and tests only; no model-facing string receives example copy.
#
# Addendum 2026-09-05 (visual-location turn boundary): CONFIRMATION BOUNDARY's prompt-only fix
# failed twice more in live Training. Both drafts hedged an image-derived proper-name location,
# then immediately asked about an activity or experience there as though the guess had already
# been confirmed. A narrow post-sanitize guard now rejects that exact semantic shape before the
# composer and lets OpenerService retry it. Direct confirmation/correction and a standalone guess
# remain legal. The prompt mirrors now make the location guess the whole conversational move,
# explicitly subordinate SAY IT ONCE, REPLY COMFORT, and APPLICATION RULE to that boundary, and
# remove the literal high-frequency structures VARY THE SHAPE had been putting on wire.
#
# Addendum 2026-09-06 (no grading): owner feedback on a watched Training draft, the New Year's
# sushi opener "Starting the new year with a massive spread of sushi is an elite move. Was that
# the main event for the night or just the appetizer?" COMPLIMENT AS REMARK produced this. That
# rule located the defect in the verdict being ON HER and being EMPHATIC, so the model moved the
# verdict onto the thing and lowered the volume, satisfying every clause. The real defect is the
# verdict itself: she chose to post the item, so a judgment of how good it is transfers no
# information, which is why it reads the same as a compliment on her appearance. Measurement: 95
# unique openers recovered from data/hinge_debug/*/actions.jsonl; "elite move" verbatim twice;
# roughly 20 of 95 carry the same grade-shaped predicate (bold move, iconic weekend energy, top
# tier, is unmatched, is a masterpiece, is unreal, a great look x3, the perfect x4, an
# incredible/amazing experience x2, is pretty fantastic x2 -- named here, off wire, per the
# de-templating rule below). This is a partial LOCAL sample only: BigQuery openers/
# opener_rejections is the system of record and was not queried. A phrase blocklist was rejected:
# it would have caught 2 of about 20, because the defect is a predicate shape, not a vocabulary.
# The structural cause worth recording: CONVERSATIONAL VALUE, INFORMATION GAIN, MINIMUM
# INVENTION, and NEVER INVENT THE SENDER together leave the taste verdict as the only predicate
# requiring zero facts about either person, so it is the path of least resistance rather than a
# stylistic tic. WHAT SHIPS: NO GRADING in the same three stable semantic surfaces as MODIFIER
# CLARITY (config.yaml's opener.style, this constant, and the response schema, whose angle field
# carries the self-check). Prompt-only, per the 2026-09-05 owner decision that prompt-only comes
# first and the 2026-08-11 scaffolding-defense decision is not reopened. DELIBERATELY NOT
# CHANGED: no deterministic guard and no new retry reason, so the retry hint block (copy 4 of 4)
# is untouched -- do not go looking for NO GRADING there and conclude the lockstep is broken.
# Escalate to a guard only if a watched batch still grades. THE CRITICAL DESIGN CONSTRAINT: the
# 2026-08-16 minimum-invention addendum records that making falsifiability unconditional
# pressured the model to invent motives merely to produce a claim. NO GRADING therefore routes
# its failure case to a SHORTER message (cut the beat, let one specific question be the whole
# message) and never to a substitute claim. If a future edit removes that escape hatch, it
# re-creates the 2026-08-16 bug. No example verdict vocabulary ships in any model-facing string,
# per the 2026-08-16 de-templating addendum; the concrete regressions are pinned off wire in the
# tests.
#
# Addendum 2026-09-06 (b) (no grading refinements): an adversarial review of the addendum above
# found two real conflicts, and a separate audit found a live guard collision, all three fixed
# together. (1) PLAYFUL HYPERBOLE carve-out: SUBSTITUTION TEST's swap check is meant to catch a
# literal verdict on quality, but an anchored nonliteral exaggeration can be just as portable in
# that sense without being a grade, and PLAYFUL HYPERBOLE is measured at 13.7% of first beats on
# the local 95-opener corpus, one of the few non-degenerate moves the model still reaches for --
# narrowing it was never the intent. NO GRADING now states the swap targets a literal quality
# verdict and that an unmistakably nonliteral PLAYFUL HYPERBOLE survives it. (2) VARY THE SHAPE
# subordination: nothing said the one-question fallback NO GRADING routes its failure case to
# was itself subject to VARY THE SHAPE, so reaching for that same single-sentence construction
# message after message could become its own template. A short subordinating clause fixes this.
# MEASURED CONTEXT, deliberately NOT put on the wire so as not to overweight a guard against a
# problem that has not happened yet: the local corpus is 94.7% exactly two sentences and 98.9%
# question-final, so today routing MORE messages to the one-sentence fallback INCREASES shape
# diversity rather than collapsing it. (3) The "HERE'S" scaffolding collision this same addendum
# recorded a residual for (2026-09-05 entry, ops/OPENER-REDESIGN.md) is fixed in the guard, not
# the prompt -- see the addendum above _SCAFFOLD_LEADING_PHRASES below for that change. Full
# record: ops/OPENER-REDESIGN.md, Addendum -- 2026-09-06 (b).
#
# Addendum 2026-09-06 (c) (compliment as remark, fold the qualification in): an adversarial
# review of the two addenda above found a THIRD real defect, this time in how COMPLIMENT AS
# REMARK and NO GRADING sit next to each other rather than in either rule alone: COMPLIMENT AS
# REMARK still affirmatively taught the exact technique NO GRADING goes on to call
# insufficient (move the praise onto the thing, keep it understated), and the qualification
# that rescues it lived only in NO GRADING's later "narrows COMPLIMENT AS REMARK" sentence.
# Under minimal thinking a model reads and can satisfy the first, affirmative sentence before
# it ever reaches the second, which is precisely how the motivating "... is an elite move"
# regression got past a rule that was supposed to forbid it. OWNER DECISION: NARROW COMPLIMENT
# AS REMARK, do not retire it. Rationale on record: the same audit measured this rule set at
# roughly 69% prohibition and mechanical instruction against only 21% positive specification,
# with the corpus at 94.7% exactly two sentences, 98.9% question final, and 82% of first beats
# deletable -- the space of legal positive moves is already collapsed, so retiring the one
# remaining compliment would be the wrong direction; narrowing it in place is correct. WHAT
# SHIPPED (2026-09-06 (c), corrected 2026-09-06 (d) below): COMPLIMENT AS REMARK originally
# stated, in its own sentence and ahead of the technique it gates, that moving the praise off
# her and onto the thing is not by itself enough -- the one permitted compliment must stay
# inseparable from a specific observation about that item, unable to survive being detached
# from what was actually noticed. This shipped in all three on-wire copies (config.yaml
# opener.style, this constant, and the _SCHEMA opener description below) by replacing words
# inside COMPLIMENT AS REMARK's existing sentence rather than only appending a new one. NO
# GRADING's own subordination sentence was deliberately left UNCHANGED: it still does real
# work stating the precedence explicitly (NO GRADING narrows COMPLIMENT AS REMARK rather than
# competing with it), and keeping it is what the owner decision called for. What changed is
# that the two rules no longer read as an affirmative technique and a later, separate fix for
# it: the technique's own sentence carried its own limit, so a minimal-thinking read of
# COMPLIMENT AS REMARK alone could no longer satisfy it while still producing a grade. No
# example verdict vocabulary was added anywhere, per the 2026-08-16 de-templating rule. This
# resolved the LOW severity finding from the 2026-09-06 adversarial review of the NO GRADING
# addenda above; see ops/OPENER-REDESIGN.md for the design's fuller context.
#
# CORRECTION (this comment previously overclaimed, fixed in place 2026-09-06 (d)): the
# sentence above used to read "...so every distinct property already on wire (thing as
# sentence's subject, praise landing sideways the way a friend would mention it in passing,
# understated over emphatic, shaping a compliment that occurs rather than requiring one)
# survives the fold unchanged", implying parity across all three surfaces. Checked against the
# actual text, that is true only of config.yaml's long form, where all four properties were
# genuinely on wire before and after. This constant never carried "praise landing sideways ...
# in passing" or a separate "shapes a compliment that occurs" sentence at any point -- only
# "thing as the sentence's subject" and "understated over emphatic" ever were -- and the
# _SCHEMA opener description never carried any of the four. Nothing was lost by the fold in
# either compressed surface: these properties were never present there to lose, consistent
# with the deliberate compression gap the 2026-09-05 register rewrite already documented (the
# schema copy receives a compressed subset of properties, not the full set). See
# ops/OPENER-REDESIGN.md, Addendum -- 2026-09-06 (e) for the full record of this correction.
#
# Addendum 2026-09-06 (d) (compliment as remark defers to the substitution test): a further
# adversarial review found the (c) fold above had created a NEW contradiction rather than only
# closing one. (c) gave COMPLIMENT AS REMARK its own portability test ("unable to survive
# being detached from what was actually noticed") with no exception, while NO GRADING's
# SUBSTITUTION TEST explicitly carves out an unmistakably nonliteral PLAYFUL HYPERBOLE. An
# anchored hyperbolic compliment whose predicate could transfer to a different woman's photo
# therefore passed NO GRADING (hyperbole exception) but failed COMPLIMENT AS REMARK's own
# restated test (no exception) -- opposite verdicts on the same sentence under minimal
# thinking. THE FIX: COMPLIMENT AS REMARK no longer restates a second, subtly different
# portability test; it now points at NO GRADING's SUBSTITUTION TEST directly (this constant:
# "it must also pass NO GRADING's SUBSTITUTION TEST below"; _SCHEMA opener description below:
# "it must also pass the SUBSTITUTION TEST below", with a matching SUBSTITUTION TEST label
# added at the schema's own grade-check sentence so the forward reference lands somewhere
# named). One shared test, with its one exception, now lives in exactly one place per surface
# and cannot drift out of sync with a restated copy again. This also removes words rather than
# adding a second hyperbole exception, which matters because this rule set already runs
# roughly 10,000 tokens per request. Shipped in all three on-wire surfaces (config.yaml
# opener.style, this constant, and the _SCHEMA opener description). No example verdict
# vocabulary was added anywhere, per the 2026-08-16 de-templating rule. Full record:
# ops/OPENER-REDESIGN.md, Addendum -- 2026-09-06 (e).
#
# Addendum 2026-09-06 (g) (compliment as remark's magnitude clause joins the same exception): a
# fourth adversarial review found (d) above closed only HALF the contradiction it named.
# COMPLIMENT AS REMARK carries two independent requirements, not one: a PORTABILITY test ((d)
# pointed this at NO GRADING's SUBSTITUTION TEST, whose PLAYFUL HYPERBOLE exception already
# covers it) and a separate MAGNITUDE requirement ("understated over emphatic"), which (d) left
# untouched and which carried no exception of its own. An unmistakably nonliteral PLAYFUL
# HYPERBOLE is by nature emphatic, so a strong hyperbolic compliment passed the portability test
# via (d)'s fix while still failing the magnitude clause -- opposite verdicts on the same
# sentence again, under the same minimal-thinking read. THE FIX: the magnitude clause now points
# at that same test's PLAYFUL HYPERBOLE exception ("whose PLAYFUL HYPERBOLE exception covers
# tone too") instead of gaining its own separate carve-out, so ONE portability test and ONE
# nonliteral exception stay stated in one place, per (d)'s original goal. Shipped in both
# surfaces that carried the magnitude clause (config.yaml opener.style and this constant); the
# _SCHEMA opener description never carried it (established by the 2026-09-06 (e) correction
# above), so it is unchanged. No example verdict vocabulary was added anywhere, per the
# 2026-08-16 de-templating rule. Full record: ops/OPENER-REDESIGN.md, Addendum -- 2026-09-06 (g).
_SYSTEM = (
    "You write the opening message a man sends a woman on a dating app. Use the dating and "
    "conversational principles associated with Coach Corey Wayne's 'How to Be a 3% Man', without "
    "copying his wording. Be a charming gentleman first: relaxed, confident, playful, genuinely "
    "curious, and direct without pressure. Carry the spirit of his 90/10 framework across the "
    "interaction: default to sincere interest and easy confidence, and reserve light teasing or "
    "cheeky humor for the occasional profile where it arises naturally. Do not force teasing into "
    "every opener. SHARED CONTEXT RULE: your message is displayed directly under the exact photo "
    "or prompt it attaches to, and she is looking at that item while she reads your words. "
    "Never point at the medium itself: a word gesturing at the photo, the screen, or the "
    "profile as an object adds nothing she cannot see; a bare demonstrative does that work. "
    "PRIMARY ITEM RULE: after choosing a numbered item, it is the clear main subject of "
    "referenced, angle, and opener, including its header. The message must feel natural directly "
    "under that item, never like a reason to discuss another image. Your opener must justify why "
    "the selected item was liked: it supplies the reason, subject, or payoff. You may use explicit "
    "profile text or an unnumbered CONTEXT image as supporting context only when it sharpens a "
    "connection back to the selected item. Do not let another image replace the selected item as "
    "the reason for the Like, subject, or payoff, point her away from it, or use context without "
    "an unmistakable selected-item anchor. "
    "PHOTO HEADER RULE: any title, caption, or prompt printed with a photo is part of that same "
    "item and defines how the photo is meant to be read. Interpret the visible scene through "
    "that text before choosing an angle. The angle and opener must respect the combined meaning, "
    "never contradict, reverse, or ignore the header's framing. "
    "PROFILE TEXT FACT CHECK: the HER PROFILE TEXT section contains facts she has already told you. "
    "Before writing any question, check it against every stated bio and prompt answer. Never ask for "
    "a fact, preference, activity, place, or opinion that the profile text already answers. Use those "
    "facts as context for a fresh, forward-moving angle instead. For example, if she says she likes "
    "apples, do not ask whether she likes apples. "
    "CONVERSATIONAL VALUE RULE: do more than label what she can already see. A visible detail "
    "may be named and may be the subject, premise, or setup. The opener fails only when its "
    "final conversational point is merely that description. Use the detail to add a perspective, "
    "grounded interpretation, playful framing, connection, or natural question. The message "
    "must be profile-specific, but it does not have to contain a guess or a claim that could be "
    "wrong. CLAIMS ONLY WHEN NATURAL: a correctable inference is one available move, not a "
    "requirement. When no natural inference exists, prefer a grounded observation or specific "
    "question over a forced guess. If you make a claim, write the whole proposition rather than "
    "a shorthand answer to an imagined question, and choose the least speculative interpretation "
    "that fits ordinary explanations of the scene. MINIMUM INVENTION: never invent a purpose, "
    "motive, cause, plan, sequence, effort, goal, or unseen circumstance just to create a claim. "
    "Do not assign why she chose or did something unless her profile states it. Any literal "
    "premise must be plainly visible in the item or explicitly stated in her profile. Never use "
    "one invented fact as the premise for another. A setting or destination does not establish "
    "how she arrived, what effort it took, or whether she pursued a goal. A hedge does not rescue "
    "a far-fetched premise. Never infer ownership, employment, a routine, a responsibility, or a "
    "relationship merely from proximity in one photo. TRACEABILITY TEST: for any inference, she "
    "should instantly see which visible or stated clue led there. If that path needs an "
    "explanation, the inference is too remote. INFORMATION GAIN TEST: the conclusion of a guess "
    "must not itself be directly visible or explicitly stated in the selected item, its header, "
    "a sign, or elsewhere in her profile. Visible and stated facts may support a different "
    "inference, but they are clues, not guessed conclusions. If an ordinary viewer can read or "
    "see the conclusion directly without inference, treat it as known context and choose a "
    "different angle. PLAYFUL HYPERBOLE: unmistakably nonliteral "
    "exaggeration is allowed when its visible anchor is immediate. It adds playful framing; it "
    "does not license presenting an invented motive, circumstance, or event as literal fact. "
    "SAFETY AND DIGNITY: never infer, tease, or pose a forced choice about self harm, suicide, "
    "death, injury, an accident, a dangerous stunt, or courage around danger. A bridge, height, "
    "water, travel scene, or recognized location is never evidence that she jumped or considered "
    "jumping. Mention a risky activity only when her profile explicitly states it, and then keep "
    "it matter of fact rather than sensational. "
    "Make one clear, positive, profile-specific bid, then leave room for her reply. A natural "
    "claim she can correct can be effective, but it is not mandatory. One specific, easy, "
    "positive question may be the whole message when that is the strongest natural angle. "
    "SETUP PAYOFF CONTINUITY: every visible detail you name must be necessary to, and used by, "
    "the conversational move. If removing a descriptive clause leaves the later point or "
    "question unchanged, cut it. QUESTION COHERENCE: ask one coherent thing at a time. An 'or' "
    "is allowed only for parallel, genuinely contrasting answers to that same underlying "
    "question, never to join unrelated dimensions. CASUAL OR PUNCTUATION: never put a comma "
    "immediately before 'or', even where formal grammar would allow one. Write it the way a "
    "person would text casually. SPOKEN REGISTER: write it the way the sender would say it "
    "out loud: use the contractions a relaxed speaker would use, prefer plain everyday words "
    "over formal noun phrases, and allow natural spoken elision, but never internet slang, "
    "chat abbreviations, meme phrasing, or borrowed caption labels. This licenses wording "
    "only; spelling, capitalization and every punctuation rule here stand unchanged. "
    "One intensifier is plenty. "
    "Flawless written grammar reads as an essay, not a text. SAY IT ONCE: the second beat "
    "inherits the first beat's topic; never restate a connection or context the message "
    "already established, and strip each question to the one clause a person would text. "
    "CONFIRMATION BOUNDARY overrides this inheritance rule: never let a later beat inherit an "
    "unconfirmed claim as fact. "
    "REFERENT CLARITY: every pronoun, shorthand "
    "noun, and question subject must have one immediately obvious referent. Across two beats, "
    "keep the same referent unless the transition to a new one is explicit and immediately "
    "clear. Do not make the reader choose between different ordinary meanings of the same word. "
    "MODIFIER CLARITY: every modifying phrase must have only one natural attachment on first "
    "reading. If the phrase's placement permits a plausible unintended meaning, reorder or "
    "rephrase the line. "
    "ROLE CONSISTENCY: if you give a visible subject a playful role or rank, preserve that role "
    "across every beat. Do not give the same subject an incompatible role later. If you mean a "
    "different subject, name it explicitly. "
    "REPLY COMFORT: make the most natural honest reply feel good to give. Invite her to share a "
    "preference, perspective, inspiration, or experience, not self justification. Never test her "
    "intelligence, sincerity, knowledge, effort, or whether a visible detail is genuine. Do not "
    "offer a forced choice whose honest answers make her defend, diminish, or embarrass herself. "
    "If a visible detail could have more than one explanation, respond to its visible effect or "
    "choose another angle rather than asking her to verify its status. Asking only whether an "
    "inferred location itself is right is the narrow confirmation required by CONFIRMATION "
    "BOUNDARY, not permission to build on the location before she answers. "
    "POSITIVE SOCIAL FRAMING: state the intended positive observation, question, or invitation "
    "directly. Never begin by naming an insulting, judgmental, awkward, pressuring, creepy, or "
    "offensive interpretation of your own message and then denying it. That denial introduces "
    "the negative interpretation even when the sentence says it is absent. Remove the disclaimer "
    "and make the substantive thought stand on its own. "
    "RECIPROCITY BEFORE FUTURE: an opener is one bid to a stranger, not an audition for a role "
    "described in her profile. Never answer one of her preferences by advertising the sender as "
    "the man who satisfies it, promising what he will do for her, or declaring how he will "
    "perform. Do not assume that a match, date, relationship, or shared future already exists, "
    "including with possessive language about a first date, place, trip, or other future together. "
    "A direct low-pressure invitation may propose a specific get-together, but must leave her "
    "acceptance completely open; a proposal is not an established shared plan. "
    "CONFIRMATION BOUNDARY: a guess remains unconfirmed until she replies. Never follow a guess "
    "with a statement, question, compliment, or invitation that assumes it is correct. Word the "
    "guess so her natural next move is to confirm or correct it. If you add a second beat, it may "
    "only invite that confirmation or correction without presupposing the answer. Never ask about "
    "an experience, preference, or consequence that only makes sense if the guess is true. "
    "Otherwise stop after the guess. Questions should invite positive, fun conversation, not form "
    "an interview. VISUAL LOCATION TURN BOUNDARY: when a place is inferred from an image rather "
    "than stated in her profile text, that location guess is the only conversational move before "
    "she replies. End after it or ask only whether the location itself is right; the confirmation "
    "is the conversational payoff. Do not add a statement or question about an activity, reason, "
    "preference, feeling, experience, or consequence at that place before she confirms it. "
    "Any teasing must be clearly good-natured and never belittling, arrogant, condescending, or "
    "mean. Mild innuendo is eligible only when her own profile clearly invites that playful tone; "
    "never force it. A brief greeting is optional but cannot substitute for profile-specific "
    "substance. At most one authentic, specific compliment is allowed; never pile on flattery or "
    "seek approval. COMPLIMENT AS REMARK: shape that one compliment, when it occurs, as an "
    "offhand remark about the visible thing, not an earnest verdict on her. Moving the praise "
    "onto the thing is not enough alone: it must also pass NO GRADING's SUBSTITUTION TEST "
    "below, whose PLAYFUL HYPERBOLE exception carries here and governs every condition that "
    "follows, covering both portability and volume: phrased with the thing as the sentence's "
    "subject, landing sideways in passing, and understated over emphatic, keeping peer footing "
    "instead of grading her or seeking her approval. Landing sideways survives that exception "
    "because it is placement rather than volume, so even an exaggeration stays an aside and "
    "never becomes the point. "
    "NO GRADING: a score is not an observation. Never assign the visible thing a grade, "
    "rank, level, or verdict on how good it is, however casually worded, and whether the subject "
    "is her or what she did. She chose to post it, so she already knows it is good and a sentence "
    "whose whole point is that judgment gives her nothing; rating her choices also puts the sender "
    "above her as a judge. This narrows COMPLIMENT AS REMARK rather than competing with it: moving "
    "praise off her and onto the thing does not by itself make it a remark, so the one permitted "
    "compliment must be inseparable from a specific observation about that item. SUBSTITUTION "
    "TEST: if the predicate would still fit unchanged under a different woman's different photo, "
    "it is a grade and it fails, except an unmistakably nonliteral PLAYFUL HYPERBOLE, which stays "
    "playful framing rather than an assessment of quality even where its wording could transfer. "
    "When the best thing you can say about an item is how good it is, "
    "do not manufacture a claim to fill the sentence: cut that beat and let one specific question "
    "be the whole message, still subject to VARY THE SHAPE so the same single sentence "
    "construction never becomes its own template. MINIMUM INVENTION outranks the wish for a first "
    "sentence. "
    "Do not act as if intimacy or romantic interest already exists. Keep the tone "
    "non-needy and do not demand that she chase. HEDGE THE CLAIM, NEVER YOURSELF: when a claim "
    "is uncertain, express that uncertainty naturally in wording that fits the specific item. "
    "The goal is calibrated uncertainty, not a particular lead in. This is not a phrase menu: "
    "choose the construction from context, and never use a hedge as a substitute for the complete "
    "self contained claim. VARY THE OPENING: let the specific item and angle determine the wording "
    "and sentence shape; do not rotate or recycle a fixed stock hedge, and never open every "
    "message the same way. VARY THE SHAPE: let the specific item determine the whole message's "
    "sentence count, clause pattern, and question form. Choose a structure only because the item "
    "calls for it, never because it is the easy mold. No structure is the default; a shape that "
    "repeats message after message is a template even when the words change. "
    "Never apologise for writing, never ask permission, and "
    "never call your own question dumb. GUESS "
    "THE WORLD, NOT HER IDENTITY: name a country, a region, or a park the way a well travelled "
    "friend would, never a street, a neighbourhood, a hotel, a specific venue, or anywhere that "
    "could be where she lives, and never guess her employer, her school, or her age, or identify "
    "anyone else in the photo. When a place comes from recognizing the image rather than from her "
    "profile text, present it only as an inference and obey VISUAL LOCATION TURN BOUNDARY: end "
    "after the guess or ask only whether the location itself is right. Do not build on an inferred "
    "location as though it were correct, state it as shared experience, or turn it into a generic "
    "compliment. NEVER INVENT THE "
    "SENDER: you may not claim he has been somewhere, "
    "done something, or likes something, because you do not know his history and he has to live "
    "with whatever you write. PICK THE ITEM YOURSELF: the numbered images are her profile photos, "
    "numbered from 1 in the order they are given, and you choose which one to write about. Choose "
    "the item you have the best conversational angle on, not the most striking picture: a plain "
    "photo that supports a natural observation, question, connection, or playful framing beats "
    "a beautiful one you have nothing to say about. The item supplies the material and the "
    "conversational payoff is the message. Written prompts are deliberately "
    "absent from the numbered choices and item_index must never refer to a prompt. THE FAILURE TO "
    "AVOID IS PICKING A PHOTO YOU HAVE NOTHING TO SAY ABOUT, because then all that is left to write is what it "
    "looks like, which is never enough as the final point. Read all of the numbered items "
    "first, find the one that hands you the most natural conversational angle, and pick that one even when "
    "another item is the better picture. Set item_index "
    "to that item's number; it is also the item that gets liked, so your message and the like "
    "always land on the same thing. Any image given WITHOUT a number is context, usually her "
    "vitals: her age, her job, her school, her city, or another unnumbered profile image. You may "
    "use what it shows to support a connection, but the selected numbered item must remain the "
    "opener's clear main subject and every context connection must lead back to it. Never make an "
    "unnumbered image the opener's main premise, reason for the Like, subject, or payoff, or "
    "point her away from the item receiving the Like. It simply carries no number, so you can "
    "never pick it and item_index can never refer to it. "
    "APPLICATION RULE: TWO sentences is the absolute maximum, and within "
    "that ceiling be as short as the angle allows: spend no word merely repeating what she can "
    "already see and none on padding, but never cut necessary setup or the conversational payoff. "
    "Subject to CONFIRMATION BOUNDARY, a second sentence may be "
    "one easy positive question "
    "or a direct low-pressure invitation. Do not try to build a text relationship in the opener. "
    "HARD RULE: never use an em dash or any hyphen; use commas or periods instead and spell out "
    "hyphenated abbreviations. When a compound would need a hyphen, rephrase so no hyphen is "
    "needed; never just delete the hyphen and glue the words together. "
    "HARD RULE: write the opener in plain ASCII letters and punctuation "
    "only; use no emoji and transliterate accented or non-English letters to plain ASCII. A "
    "terminal plain text smiley, :) is allowed only when it meaningfully makes an otherwise "
    "potentially misread playful or teasing line clearly good natured; never add it by default. "
    "Fill item_index, referenced, angle and item_description before you write the opener: "
    "referenced is the "
    "full description of what you are reacting to and is never sent to her, so put the literal "
    "inventory there; the message may use only the setup it needs and must add a conversational "
    "payoff. Angle is your own short wording "
    "for what your opener is doing, and item_description says in a few words what the item you "
    "picked is, a photo and what it shows. "
    "The opener field must contain only the bare message itself, "
    "with no "
    "preamble, label, or surrounding quotes. Follow the style guide. Output only the structured result."
)

# ---------------------------------------------------------------------------------------
# THE ITEM-CROP REQUEST SHAPE (ops/OPENER-REDESIGN.md 5.2 and 5.7)
#
# What the model sees stops being her raw scroll frames and becomes one cropped image per
# profile item, numbered, plus the unnumbered context crops. Doc 5.2's argument is NOT about
# legibility or size -- it is about who owns the numbering:
#
#   "If we send overlapping full frames, the model must derive its own independent enumeration
#    and the two must agree by luck. With crops, image 3 in the request IS item 3. Agreement by
#    construction."
#
# Two more reasons from the same section, both of which this shape has to preserve rather than
# merely allow. (1) DUPLICATION BIAS: a card straddling a scroll seam appears in two or three
# frames, and now that the model is CHOOSING among items rather than writing about whatever
# caught its eye, repetition reads as salience -- we would bias selection by our own scroll
# cadence. One crop per item removes the duplicate entirely. (2) The crops are needed anyway,
# because doc 5.6's post-tap verification is a signature match against the stored crop.
#
# Sent (doc 5.7): her name as text, items 1..N each as ONE cropped image in order, the context
# blocks cropped and unnumbered, a truncation flag when the capture hit its ceiling, and the
# Part A style guide unchanged. NOT sent: full screenshots, scroll frames, and the
# endorsement blocks (doc 2.4 -- a tease built on a friend's line is the worst possible
# ammunition, so they are excluded upstream and never reach this module at all).
#
# HER NAME IS PASSED BACK AS TEXT because cropping loses it (doc 5.2: "Her name is lost by
# cropping and must be passed back as text. It is already extracted"). It is the one piece of
# information the frames carried in their sticky header that no crop can carry, so passing it
# restores parity with the old shape rather than adding a new input.
# ---------------------------------------------------------------------------------------

# Stated ONCE, before the first image, rather than repeated on every label: the convention is
# what needs stating, not the instruction. Repeating "set item_index to k" next to each image
# would put a fresh imperative immediately before the model writes, N times, with the LAST one
# read the freshest -- the same recency mechanism that made _SYSTEM's old two-hedge list
# produce 5/5 "I bet" openers on a live dry run (see _SYSTEM's 2026-08-11 addendum). The labels
# themselves stay bare identifiers for that reason.
_ITEM_PREAMBLE = (
    "=== HER PROFILE, ONE IMAGE PER ITEM ===\n"
    "Every image below is one item cropped from her profile, and each image is immediately "
    "preceded by its own label. A label reading ITEM k means the image directly after it IS "
    "item k, so there is nothing here for you to count and no order for you to work out. "
    "A numbered crop may contain a title, caption, or prompt above its photo; that text and "
    "the photo are one compound item and must be read together."
)

# Appended to _ITEM_PREAMBLE only when context crops are actually being sent. Explaining a
# label that does not appear in the request would be describing something that is not there,
# which would otherwise be a small lie about the request.
_ITEM_PREAMBLE_CONTEXT = (
    " A label reading CONTEXT means the image directly after it has no number: read it and "
    "use what it shows as supporting context for a numbered item. You can never pick it, and "
    "the numbered item must remain the opener's clear main subject and reason for the Like, "
    "not a reason to discuss this CONTEXT image."
)

# Placed immediately BEFORE the image it names: Gemini reads parts as one ordered sequence, and "the
# next image" is only unambiguous when the pointer text sits adjacent to what it points at.
# Adjacency is what turns doc 5.2's "image k IS item k" from a fact about how we built the
# request into a fact the model can read off the request.
_ITEM_LABEL = "=== ITEM {number} ==="

# Deliberately spells out the prohibition in the label itself rather than only in the preamble.
# A context block is the one thing in the request that looks exactly like a selectable item
# (it is a crop of her profile, sitting in the same list) and differs only by not having a
# number, so the difference is stated where it cannot be missed. Doc 5.3: context blocks are
# "sent, read as supporting context, never selectable".  A labelled context crop can support
# a connection, but the selected numbered item must stay the message's primary anchor.
_CONTEXT_LABEL = "=== CONTEXT, NOT NUMBERED, CANNOT BE PICKED ==="

# What HER NAME renders as when the driver's OCR did not read one. An explicit "we did not read
# it" rather than an empty line or a silently omitted section: the model is being told what it
# has, and a blank field reads as a name that is blank.
_NAME_UNAVAILABLE = "(not read)"


@dataclass(frozen=True)
class ItemRequest:
    """One profile's items as the MODEL sees them: the payload half of doc 5.7's request shape.

    Deliberately a plain value type over bytes and strings, holding no vision objects, no
    segmentation, and no signatures -- this module must be able to build (and a test must be
    able to pin) a request without the vision extras installed, and without importing anything
    from `operation_love.drivers`. The producer side is `drivers.item_crops.ItemPayload`, whose
    `items`/`context`/`truncated` map onto the fields here one for one, deliberately with the
    SAME names so the adapter that will build this from a payload (a LATER workflow -- see this
    module's `generate` docstring for the seam) is transcription rather than translation.

    `items` is the numbered list, in model order: `items[k - 1]` is item k, 1-based per
    FIRST_ITEM_INDEX. `context` is the unnumbered tier, sent AFTER every numbered item, which
    the schema's `item_index` description promises to the model in as many words ("Some blocks
    are shown WITHOUT a number, for context only ... you can never pick one"). Sending a
    context crop among the numbered ones, or numbering it, would make that description a lie.

    `name` is her first name as text (doc 5.2: cropping loses the sticky-header name, so it is
    passed back). `truncated` is True when the capture hit its ceiling, i.e. these are only the
    items we managed to read.

    AN EMPTY `items` IS REFUSED, LOUDLY. Doc 5.1's contract is that the model picks an item, so
    a request offering zero of them cannot be answered honestly -- the only reply available is
    ITEM_INDEX_ABSENT, and paying for a billed API call to be told what we already knew is
    worse than raising. The zero-image branch that DOES exist in _text_part is for the legacy
    frame shape, where the model still had a profile to write about even with no photos; this
    shape has nothing at all. Doc 5.3's "treat a missing table as a hard stop, never as a reason
    to fall back to a fixed coordinate" is the same instinct one layer up.
    """

    # `items` is the one field with no default, and that ordering is the point: there is no such
    # thing as an item request without items, so it cannot be omitted by accident. Everything
    # else degrades honestly -- a name the OCR did not read renders as "(not read)", a profile
    # with no vitals block simply sends no context, and an untruncated capture says nothing
    # about truncation.
    items: tuple[bytes, ...]
    name: str = ""
    context: tuple[bytes, ...] = ()
    truncated: bool = False

    def __post_init__(self) -> None:
        # Normalize to tuples so a caller passing a list cannot mutate the request after it was
        # built (frozen is only shallow), and so `images` below is cheap and order-stable.
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "context", tuple(self.context))
        object.__setattr__(self, "truncated", bool(self.truncated))
        if not self.items:
            raise ValueError(
                "ItemRequest requires at least one numbered item: the model's job on this "
                "request shape is to CHOOSE an item (ops/OPENER-REDESIGN.md 5.1), and a "
                "request carrying none can only be answered with ITEM_INDEX_ABSENT. Hard stop "
                "upstream instead of paying for a call that cannot succeed.")
        for position, image in enumerate(self.images):
            if not isinstance(image, (bytes, bytearray)) or not image:
                raise ValueError(
                    f"ItemRequest: {self.describe_image(position)} is not usable image bytes "
                    f"({type(image).__name__}, {len(image) if hasattr(image, '__len__') else '?'} "
                    "bytes). Every image in this request is one item's crop and the numbering "
                    "is positional, so a missing or empty one would silently renumber every "
                    "item after it.")

    @classmethod
    def from_profile(cls, profile: Profile) -> "ItemRequest":
        """THE ADAPTER: one capture's item payload as the model's request (doc 5.7).

        Transcription, not translation -- `Profile.items` / `item_context` / `name` /
        `items_truncated` were shaped to map onto this class's four fields one for one (see
        this class's docstring and `perception.capture.Profile`), so this method can be read
        end to end and nothing here can renumber, reorder or drop a crop.

        It lives here rather than on `Profile` for the dependency direction: this module
        already imports `Profile`, while `perception.capture` must stay importable by the
        ranker and the stores without dragging the opener in.

        RAISES (via `__post_init__`) when the profile carries no numbered items. That is
        deliberate and it is not this method's job to soften: a capture that could not
        enumerate says so in `Profile.items_unavailable`, and the caller's correct move is to
        stop with that sentence, never to fall back to `profile.photos` -- doc 5.2's whole
        point is that raw scroll frames cannot carry an item number (one card appears in
        several frames, one frame can hold two cards), so substituting them would hand the
        model a numbering nobody can act on.
        """
        return cls(
            items=tuple(getattr(profile, "items", ()) or ()),
            name=str(getattr(profile, "name", "") or ""),
            context=tuple(getattr(profile, "item_context", ()) or ()),
            truncated=bool(getattr(profile, "items_truncated", False)),
        )

    @property
    def item_count(self) -> int:
        """N: the size of the list the model is offered. It may only answer with 1..N."""
        return len(self.items)

    @property
    def context_count(self) -> int:
        return len(self.context)

    @property
    def images(self) -> tuple[bytes, ...]:
        """THE REQUEST'S IMAGE LIST, in wire order: numbered items first, then context.

        Identical in construction and in order to `ItemPayload.images`, which is what a caller
        will hand us. Position is the whole contract here -- `images[k - 1]` is item k -- so
        every other method on this class indexes into this one list rather than re-deriving the
        split, and _assemble_parts labels by position against it.
        """
        return self.items + self.context

    @property
    def image_count(self) -> int:
        return len(self.items) + len(self.context)

    def label_for(self, position: int) -> str:
        """The text part that must sit immediately before `images[position]`.

        Numbered items are labelled with their 1-based number; anything past the numbered items
        is a context crop and gets the unnumbered label. Positional by construction, so a crop
        can never be labelled with a number that disagrees with where it actually sits in the
        list -- which is the entire point of doc 5.2's "agreement by construction".
        """
        if not 0 <= position < self.image_count:
            raise ValueError(
                f"image position {position} is outside 0..{self.image_count - 1}")
        if position < self.item_count:
            return _ITEM_LABEL.format(number=position + FIRST_ITEM_INDEX)
        return _CONTEXT_LABEL

    def describe_image(self, position: int) -> str:
        """Operator-facing name for `images[position]`, for error messages only.

        Never "photo index N": these are not her profile photos in capture order, and telling
        an operator to go look at photo 3 of a profile when the failure is in item 3's CROP
        sends them to the wrong place entirely.
        """
        if position < self.item_count:
            return f"item {position + FIRST_ITEM_INDEX}'s crop"
        return f"context crop {position - self.item_count + 1} (unnumbered)"


_COMMON_ABBREVIATION_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Jr|Sr|vs|etc)\.", re.IGNORECASE)
_PLACE_PREFIX_ABBREVIATION_RE = re.compile(
    r"\b(?:St|Mt)\.(?=\s+(?!(?:Am|Is|Are|Was|Were|Do|Does|Did|Have|Has|Had|"
    r"Can|Could|Would|Will|What|Where|When|Why|How|Which|Who)\b)[A-Z][A-Za-z'])"
)
_DOTTED_ABBREVIATION_BEFORE_LOWER_RE = re.compile(r"\b(?:[A-Z]\.){2,}(?=\s+[a-z])")
_SENTENCE_END_RE = re.compile(r"(?:[!?]+|\.+)(?=(?:[\"'”’)]*)?(?:\s+|$))")
_COMMA_BEFORE_OR_RE = re.compile(r",(?=\s+or\b)", re.IGNORECASE)

# A narrow, deterministic backstop for the reported harmful-inference shape. Prompt guidance
# does the general semantic work because only the model sees the image and profile evidence;
# this guard catches wording that turns an ordinary travel/bridge scene into a question about
# jumping or courage. It deliberately does not ban every occurrence of "jump" or every risky
# sport: an explicitly stated skydiving or bungee experience can be a normal profile topic.
# These patterns instead identify a direct, unsupported implication of jumping from a height or
# self-harm, which should always be regenerated before it can reach the comment box.
_SENSITIVE_INFERENCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("courage to jump", re.compile(
        r"\b(?:work(?:ed)?\s+up|find|found|had)\s+(?:the\s+)?courage\s+to\s+jump\b",
        re.IGNORECASE)),
    ("asking whether she jumped", re.compile(
        r"\b(?:did|do|would|were|are)\s+you\s+(?:ever\s+)?(?:going\s+to\s+)?jump\b",
        re.IGNORECASE)),
    ("jumping from a height", re.compile(
        r"\bjump(?:ed|ing)?\s+(?:off|from)\s+(?:the\s+)?"
        r"(?:bridge|cliff|ledge|balcony|roof|height)\b", re.IGNORECASE)),
    ("self harm or suicide", re.compile(
        r"\b(?:self[\s-]?harm|suicid(?:e|al))\b", re.IGNORECASE)),
)

# A location inferred from pixels is not established context. These deliberately case-aware
# cues cover an explicit visual guess whose conclusion begins with a proper name (Lake Louise,
# Norway, an Iceland tour, and so on) without treating every ordinary or figurative "looks
# like" as a location inference. The model-facing prompt owns the general semantic rule; this
# is the high-precision backstop for the recurring shape found in live Training drafts.
#
# Only the cue words are case-insensitive. The final lookahead is intentionally case-sensitive:
# a lower-case complement such as "looks like serious dedication" is an interpretation, not a
# proper-place identification, and must not burn a retry.
_UNCONFIRMED_LOCATION_CUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i:\b(?:look(?:s|ed)?|seem(?:s|ed)?)(?:\s+(?:a\s+(?:whole\s+)?lot|"
        r"quite|very|really|almost|exactly|unmistakably))?\s+like\s+"
        r"(?:an?\s+|the\s+)?)"
        r"(?=(?!I\b)[A-Z][A-Za-z'])"
    ),
    re.compile(
        r"(?i:\bmy\s+(?:official\s+|best\s+)?guess"
        r"(?:\s+(?:for|about|on)\s+[^,.?!]{1,80}?)?\s+(?:is|would\s+be)\s+"
        r"(?:an?\s+|the\s+)?)"
        r"(?=(?!I\b)[A-Z][A-Za-z'])"
    ),
    re.compile(
        r"(?i:\bi(?:'m|\s+am|'d|\s+would)\s+(?:going\s+to\s+)?guess(?:ing)?"
        r"(?:\s+(?:that|this|it)(?:'s|\s+is|\s+was))?\s+(?:an?\s+|the\s+)?)"
        r"(?=(?!I\b)[A-Z][A-Za-z'])"
    ),
    re.compile(
        r"(?i:\b(?:that|this|it)\s+(?:(?:has|had)\s+to|must|might|could)\s+be\s+"
        r"(?:an?\s+|the\s+)?)"
        r"(?=(?!I\b)[A-Z][A-Za-z'])"
    ),
    re.compile(
        r"(?i:\b(?:is|was|could|might|can|would)\s+(?:that|this|it)\s+"
        r"(?:(?:be|in|at|near)\s+)?(?:an?\s+|the\s+)?)"
        r"(?=(?!I\b)[A-Z][A-Za-z'])"
    ),
)

_LOCATION_FOLLOWUP_COMMA_RE = re.compile(
    r"[,;:]\s*(?=(?:(?:but|and|or|so)\s+)?"
    r"(?:am|is|are|was|were|do|does|did|have|has|had|can|could|would|will|"
    r"what|where|when|why|how|which|who|right|correct|you|she|he|they|we|i|"
    r"that|this|it|must|might|probably|definitely|surely|clearly)\b)",
    re.IGNORECASE,
)
_LOCATION_FOLLOWUP_CONJUNCTION_RE = re.compile(
    r"\s+(?=(?:but|and|or|so|then)\s+"
    r"(?:am|is|are|was|were|do|does|did|have|has|had|can|could|would|will|"
    r"what|where|when|why|how|which|who|you|she|he|they|we|i|that|this|it|"
    r"must|might|probably|definitely|surely|clearly)\b)",
    re.IGNORECASE,
)
_LOCATION_CONTEXT_NOUN_RE = re.compile(
    r"\b(?:scenery|backdrop|street|architecture|landscape|lake|mountain|tower|country|"
    r"region|park|city|village|coast|beach|bridge|glacier|tour|skyline|ridge|island)\b",
    re.IGNORECASE,
)
_LOCATION_CONTEXT_TRAVEL_RE = re.compile(
    r"\b(?:visit(?:ing)?|trip|trips?|destination|travel|"
    r"stay|stayed|living in|hike(?:ing)?|ski(?:ing)?|skate(?:ing|d)?|"
    r"glacier|highlands|vacation|journey)\b",
    re.IGNORECASE,
)

# A later beat is safe only when it still asks about the identification itself. These are
# grammatical confirmation/correction shapes, not place-name lists, so the detector remains
# useful for locations it has never seen. Anything else after a proper-name inference is the
# unsafe direction: the system cannot prove from outgoing text alone that an activity or
# experience question did not take the guessed place as fact.
_DIRECT_LOCATION_CONFIRMATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:am|was)\s+i\s+(?:even\s+)?"
               r"(?:close|right|correct|(?:way\s+)?off)"
               r"(?:\s+(?:about|on)\s+(?:that|this|the|my)\s+"
               r"(?:guess|place|location|country|region|park|area))?\?+$",
               re.IGNORECASE),
    re.compile(r"^how\s+(?:close|far\s+off)\s+(?:am|was)\s+i\?+$", re.IGNORECASE),
    re.compile(r"^did\s+i\s+(?:get|guess|call)\s+"
               r"(?:it|that|this|the\s+(?:place|location))\s+"
               r"(?:right|correctly)\?+$",
               re.IGNORECASE),
    re.compile(r"^(?:is|was|could)\s+(?:that|this|it)\s+(?:be\s+)?"
               r"(?:right|correct|close|(?:way\s+)?off)\?+$", re.IGNORECASE),
    re.compile(r"^(?:is|was)\s+my\s+(?:guess|call)\s+"
               r"(?:right|correct|close|(?:way\s+)?off)\?+$", re.IGNORECASE),
    re.compile(r"^where\s+(?:is|was)\s+(?:that|this|it)"
               r"(?:\s+(?:taken|shot|filmed))?\?+$", re.IGNORECASE),
    re.compile(r"^(?:right|correct)\?+$", re.IGNORECASE),
)
_DIRECT_LOCATION_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)^(?:is|was|could|might|can|would)\s+(?:that|this|it)\s+"
        r"(?:be\s+)?(?P<location>.+?)\?+$"
    ),
    re.compile(
        r"(?i)^(?:is|was)\s+(?:that|this|it)\s+(?:taken|shot|filmed)\s+"
        r"(?P<location>.+?)\?+$"
    ),
)
_PROPER_PLACE_WORD = r"(?:[A-Z][A-Za-z']*\.?|(?:d|l)'[A-Z][A-Za-z']*)"
_PROPER_PLACE_RE = re.compile(
    rf"^(?:the\s+)?{_PROPER_PLACE_WORD}"
    rf"(?:(?:\s+|,\s*){_PROPER_PLACE_WORD}|"
    rf"\s+(?:and|of|the|de|del|la|las|los|le|du|des|van|von|in|on|upon|at)"
    rf"\s+{_PROPER_PLACE_WORD})*$"
)

# Cap on how much of a malformed model-output value gets echoed into an error message --
# long enough to be diagnostic, short enough that a huge/garbage payload can't blow up a
# log line or the hub's stop-reason display.
_MAX_ERROR_REPR_LEN = 200


def _truncated_repr(value: Any) -> str:
    """repr() a model-output value for an operator-facing error message, truncated so a
    pathological payload (e.g. a multi-KB string where a short opener was expected)
    doesn't dominate the message."""
    text = repr(value)
    if len(text) > _MAX_ERROR_REPR_LEN:
        text = text[:_MAX_ERROR_REPR_LEN] + "...(truncated)"
    return text


def _image_media_type(data: bytes) -> str:
    """Sniff the real image format from magic bytes. Gemini 400s if the declared mimeType
    doesn't match the actual bytes, so we can't just hardcode one. Defaults to PNG since
    every current device or reference browser capture path produces PNG."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _sanitize(text: str) -> str:
    """The single fold applied to every model-written opener before it becomes the recorded,
    sent OpenerResult.opener -- delegates character folding to
    operation_love.typography.fold_to_ascii, then removes a comma immediately before ``or``
    so even formally correct model punctuation reads like casual texting. The text this
    function returns is BYTE-FOR-BYTE what Adb.text() will actually type
    (fold_to_ascii is also what drivers.adb._clean_text_for_input calls; see typography.py's
    module docstring). This keeps the owner's no-dash rule intact (every dash-like codepoint
    -- em/en dash, hyphen, and their lookalikes -- folds to a comma or space via the
    canonical DASH_FOLD table inside fold_to_ascii) while also folding curly quotes, exotic
    spaces, ligatures, and accents (an accented name survives as its plain ASCII spelling,
    not a silently mangled or dropped one) -- OWNER HARD RULE: no em dashes, no hyphens of
    any kind, it's the single biggest AI-written tell. Whatever fold_to_ascii cannot reduce
    to printable ASCII survives here untouched; _parse's undeliverable_chars() check right
    after this call is what turns that into a loud, retried failure instead of a silent send
    of unrenderable text."""
    folded = fold_to_ascii(str(text))
    return _COMMA_BEFORE_OR_RE.sub("", folded)


def _strip_wrapping_quotes(text: str) -> str:
    """Repair an opener the model wrapped in its own matching pair of quotes (e.g. the
    opener field holding '"Nice antlers."' instead of the bare Nice antlers.) -- a pure
    formatting artifact with an unambiguous repair, so it is fixed silently here rather than
    routed through _scaffolding_markers to burn a retry. Called from _parse AFTER _sanitize,
    so curly quotes have already folded to straight ASCII ones (see typography.fold_to_ascii's
    QUOTE_FOLD table) and this only ever has to consider a straight " or ' pair.

    Only strips when the wrapping is unambiguous: the first and last characters are the same
    quote character, and every interior occurrence of that character is INTRA-WORD, i.e. has a
    letter on both sides. An intra-word apostrophe is an apostrophe, not a quote: it cannot be
    the other half of the leading ' , so removing the outer pair around 'That's a nice mug.'
    leaves a correct, balanced string. Any interior occurrence that is NOT intra-word (a
    trailing possessive as in 'Grams' pie', an inner quoted phrase, a stray ") means the
    leading/trailing characters are not a clean matched wrapping pair, and stripping them
    would leave a broken, unbalanced string, so the text is returned untouched. Anything else
    -- unquoted text, mismatched quote characters, an empty/1-char string -- is likewise
    returned unchanged.

    2026-09-05: the intra-word allowance is what keeps this repair alive under the SPOKEN
    REGISTER rules. Those rules make contractions the norm (before them the model produced
    ZERO apostrophes across 188 openers, so a wrapped single-quote opener always had a clean
    interior and always repaired). Rejecting on any interior ' would have made the common case
    -- a contraction inside a model-quoted message -- unrepairable, and nothing downstream
    catches it: _scaffolding_markers does not match a wholly quoted message, so the literal
    leading and trailing apostrophes would have been typed into her comment box.

    The open residual this same change MANUFACTURES is the word-final elision apostrophe:
    _is_intra_word demands a letter on BOTH sides, so the elision SPOKEN REGISTER explicitly
    licenses is not intra-word, _strip_wrapping_quotes("'Nothin' fancy about that mug.'")
    still refuses, and those wrapping quotes ship -- unlike the trailing possessive and the
    nested quote above, this is a shape the register rules actively invite. The pre-named fix
    (a scaffolding rule for a message still wholly wrapped in a matched pair after this helper
    declines) is deliberately deferred to the first watched batch, per the prompt-only owner
    decision recorded in the 2026-09-05 addendum of ops/OPENER-REDESIGN.md.
    """
    if len(text) < 2:
        return text
    first, last = text[0], text[-1]
    if first != last or first not in ("\"", "'"):
        return text
    inner = text[1:-1]
    if any(ch == first and not _is_intra_word(inner, i) for i, ch in enumerate(inner)):
        return text
    return inner


def _is_intra_word(text: str, index: int) -> bool:
    """True when ``text[index]`` sits between two letters, i.e. it is punctuation inside a
    word (the apostrophe of a contraction) rather than a delimiter around one. Pure and
    deterministic; a position at either end of the string is never intra-word."""
    return (0 < index < len(text) - 1
            and text[index - 1].isalpha()
            and text[index + 1].isalpha())


# Keywords that make a leading "<clause>:" read as a self-describing label rather than
# ordinary opener text (rule 1 of _scaffolding_markers below). "option"/"options"/"version"
# are deliberately NOT here -- see that function's docstring.
_SCAFFOLD_LABEL_KEYWORDS_RE = re.compile(
    r"\b(?:here|opener|response|message|suggestion|reply|draft|output|result|answer|example)\b",
    re.IGNORECASE,
)

# Addendum 2026-09-06 (b) ("HERE'S" collision, ops/OPENER-REDESIGN.md residuals section): "here
# is"/"here's" used to sit in this tuple as bare leading phrases, so ANY opener starting with
# either one was rejected -- including a perfectly natural spoken opening like "Here's hoping
# that trail's as steep as it looks." SPOKEN REGISTER (2026-09-05) raises the odds a model
# writes exactly that, so the collision was live, burning the same max_attempts=5 retry budget
# a run cannot spare. Genuine scaffolding "here's"/"here is" introduces the message AS AN
# OBJECT -- a determiner plus a meta noun naming the output ("the response", "an option", "a
# version", "my take", almost always followed by a colon) -- while a natural spoken "here's" is
# followed by a gerund or an ordinary concrete noun and never names the artifact. Both bare
# phrases are removed below; _SCAFFOLD_HERE_IS_OBJECT_RE narrows rule 2 to that object shape.
# Leading interjections a model uses to preface its answer instead of just answering (rule 2).
_SCAFFOLD_LEADING_PHRASES = (
    "sure!", "sure,", "certainly", "of course,", "absolutely!", "got it",
    "here you go",
)

# "here's"/"here is" scaffolding, narrowed (rule 2, continued): matches only when the meta noun
# evidence is present, i.e. a determiner immediately followed by a word naming the output
# itself. "Here's hoping ...", "Here's to ...", and "Here's the thing, ..." do NOT match --
# "hoping", "to", and "thing" are never the artifact -- so a real spoken opening is never
# rejected on this rule alone.
_SCAFFOLD_HERE_IS_OBJECT_RE = re.compile(
    r"^here(?:'s|\s+is)\s+(?:the|an?|my)\s+"
    r"(?:response|reply|suggestion|draft|output|result|answer|example|opener|message|version|"
    r"option|options|attempt|take)\b",
    re.IGNORECASE,
)

# Meta self-reference phrases beyond the bare word "opener" (rule 3). These are assistant
# framing that no human message ever contains, so they are matched unconditionally.
_SCAFFOLD_META_PHRASE_RE = re.compile(
    r"\bas an ai\b|\bas a language model\b",
    re.IGNORECASE,
)

# Refusal framing (rule 3, continued). "I cannot" / "I'm unable" CANNOT be matched on their
# own: "I cannot believe you skied that line." and "I cannot get over that dog's face." are
# natural, in-style openers, and flagging them would burn a retry apiece -- five in a row
# stops the whole run (service.py). So the phrase only counts as scaffolding when a refusal
# verb follows it. "help" carries a negative lookahead for "but", because "I cannot help but
# notice the antlers" is the idiom, not a refusal.
_SCAFFOLD_REFUSAL_RE = re.compile(
    r"\b(?:i cannot|i can not|i'm unable to|i am unable to)\s+"
    r"(?:help(?!\s+but)|assist|provide|generate|create|write|produce|comply|fulfil|fulfill|"
    r"do that|answer that|respond to)\b",
    re.IGNORECASE,
)

_SCAFFOLD_OPENER_WORD_RE = re.compile(r"\bopener\b", re.IGNORECASE)

_SCAFFOLD_LEADING_MARKUP_CHARS = ("#", "*", "`", "{", "[")


def _scaffolding_markers(text: str) -> list[str]:
    """Deterministic (NOT a second LLM call) detector for scaffolding/preamble text that
    leaked INSIDE the opener string despite the JSON schema forbidding free text outside it
    -- e.g. {"opener": "Here's the response: Great ocean, where was this taken?"}. Returns a
    list of short human-readable descriptions of every pattern found; an empty list means
    clean. Feeds _parse's OpenerParseError, which service.py's retry loop turns into a
    retry_hint the model can act on.

    Precision matters far more than recall: a false positive burns a retry, and 5
    consecutive rejections stop the whole run (service.py). Every rule below is therefore
    narrow and literal rather than a broad "sounds like AI" heuristic. Newlines are already
    collapsed to spaces and dashes already folded by the time this runs (fold_to_ascii, via
    _sanitize), so no rule here needs to account for either.

    Rules (exactly these, no more):
      1. Leading label/interjection clause: a colon in the first 40 characters whose
         preceding clause (<=40 chars, guaranteed by the 40-char window) contains one of
         the label words above, e.g. "Here's the response: ..." or "Opener: ...". Excludes
         option/options/version so "Two options: skiing or the beach?" is not rejected.
      2. Leading interjection: text starts with one of the interjection phrases above
         ("Sure!", "Certainly", "Here you go", ...), or with "here's"/"here is" followed by a
         determiner and a meta noun naming the output ("Here's the response: ...", "Here's an
         option: ..."). A bare "Here's"/"Here is" with no such object following (a natural
         spoken opening, e.g. "Here's hoping ...") is NOT matched -- see
         _SCAFFOLD_HERE_IS_OBJECT_RE's own comment for why.
      3. Meta self-reference anywhere: the word "opener" (a genuine opener about her ski
         photo will never contain the word "opener"), "as an AI"/"as a language model", or
         refusal framing ("I cannot" and friends followed by a refusal verb -- see
         _SCAFFOLD_REFUSAL_RE for why the bare phrase alone must not count).
      4. Markup/structural artifacts: text starts with #, *, `, {, or [; or contains a
         backtick, **, or the literal substring "opener" (a leaked JSON key).
    """
    markers: list[str] = []
    stripped = text.strip()

    # Rule 1: leading label/interjection clause.
    head = stripped[:40]
    for i, ch in enumerate(head):
        if ch != ":":
            continue
        clause = stripped[:i]
        if _SCAFFOLD_LABEL_KEYWORDS_RE.search(clause):
            markers.append(f'leading label clause "{stripped[:i + 1]}"')
            break

    # Rule 2: leading interjection.
    lowered = stripped.lower()
    for phrase in _SCAFFOLD_LEADING_PHRASES:
        if lowered.startswith(phrase):
            markers.append(f'leading interjection "{stripped[:len(phrase)]}"')
            break
    here_is_match = _SCAFFOLD_HERE_IS_OBJECT_RE.match(stripped)
    if here_is_match:
        markers.append(f'leading interjection "{here_is_match.group(0)}"')

    # Rule 3: meta self-reference anywhere.
    if _SCAFFOLD_OPENER_WORD_RE.search(text):
        markers.append('meta self-reference: the word "opener"')
    meta_match = _SCAFFOLD_META_PHRASE_RE.search(text)
    if meta_match:
        markers.append(f'meta self-reference "{meta_match.group(0)}"')
    refusal_match = _SCAFFOLD_REFUSAL_RE.search(text)
    if refusal_match:
        markers.append(f'refusal framing "{refusal_match.group(0)}"')

    # Rule 4: markup/structural artifacts.
    if stripped[:1] in _SCAFFOLD_LEADING_MARKUP_CHARS:
        markers.append(f'markup artifact: starts with "{stripped[:1]}"')
    if "`" in text:
        markers.append("markup artifact: backtick")
    if "**" in text:
        markers.append("markup artifact: double asterisk")
    if '"opener"' in text.lower():
        markers.append('markup artifact: literal "opener" key')

    return markers


_PREEMPTIVE_DISCLAIMER_PREFIX = r"^\s*(?:(?:hey|hi)\b[!,.\s]*)?"
_PREEMPTIVE_DISCLAIMER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "denied judgment",
        re.compile(
            _PREEMPTIVE_DISCLAIMER_PREFIX
            + r"(?:no|zero)\s+judg(?:e?ment|ing)\b(?:\s+(?:here|from\s+me|at\s+all))?\s*[,;:.!]",
            re.IGNORECASE,
        ),
    ),
    (
        "denied offense",
        re.compile(
            _PREEMPTIVE_DISCLAIMER_PREFIX
            + r"no\s+offen[cs]e\b(?:\s+(?:intended|meant|here|at\s+all))?\s*[,;:.!]",
            re.IGNORECASE,
        ),
    ),
    (
        "denied pressure",
        re.compile(
            _PREEMPTIVE_DISCLAIMER_PREFIX
            + r"(?:no|zero)\s+pressure\b(?:\s+(?:here|from\s+me|at\s+all))?\s*[,;:.!]",
            re.IGNORECASE,
        ),
    ),
    (
        "requested that a negative interpretation be ignored",
        re.compile(
            _PREEMPTIVE_DISCLAIMER_PREFIX
            + r"(?:don't|do\s+not)\s+(?:judge\s+me|take\s+(?:this|it)\s+the\s+wrong\s+way|"
              r"get\s+me\s+wrong|be\s+offended)\b\s*[,;:.!]",
            re.IGNORECASE,
        ),
    ),
    (
        "denied an awkward or insulting tone",
        re.compile(
            _PREEMPTIVE_DISCLAIMER_PREFIX
            + r"(?:(?:not\s+to|not\s+trying\s+to|i(?:'m|\s+am)\s+not\s+"
              r"(?:trying|meaning)\s+to|i\s+(?:don't|do\s+not)\s+(?:want|mean)\s+to)\s+"
              r"(?:be|sound|seem|come\s+across\s+as)\s+"
              r"(?:weird|creepy|rude|judg(?:e?mental)|pushy|forward|awkward|offensive|nosy|"
              r"shallow|mean|dramatic))\b\s*[,;:.!]",
            re.IGNORECASE,
        ),
    ),
    (
        "announced an awkward or insulting tone",
        re.compile(
            _PREEMPTIVE_DISCLAIMER_PREFIX
            + r"(?:this|that)\s+(?:(?:may|might|will)\s+sound|is\s+going\s+to\s+sound)\s+"
              r"(?:weird|creepy|rude|judg(?:e?mental)|pushy|forward|awkward|offensive|nosy|"
              r"shallow|mean|dramatic)\b\s*[,;:.!]",
            re.IGNORECASE,
        ),
    ),
)


def _preemptive_disclaimer_markers(text: str) -> list[str]:
    """Return unmistakable negative-framing disclaimers at the start of an opener.

    These phrases name an undesirable social interpretation and then deny or apologize for it,
    making that interpretation salient before the actual message begins. The multimodal prompt
    owns the broad semantic rule. This deterministic backstop is deliberately anchored at the
    beginning and requires narrow wording plus punctuation, so ordinary negation in the body of
    a message does not spend a retry.
    """
    return [label for label, pattern in _PREEMPTIVE_DISCLAIMER_PATTERNS
            if pattern.search(text)]


_PREMATURE_SHARED_FUTURE_PATTERNS = (
    (
        "assumed shared first date or outing",
        re.compile(
            r"\bour\s+(?:very\s+)?first\s+"
            r"(?:date|spot|place|trip|vacation|getaway|adventure|destination|outing|"
            r"dinner|drink|drinks|weekend|night)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "promise of future performance",
        re.compile(
            r"\b(?:i\s+will|i'll)\s+(?:personally\s+)?"
            r"(?:make\s+sure|ensure|guarantee)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "self-advertised dating role",
        re.compile(
            r"\b(?:i'm|i\s+am)\s+(?:(?:clearly|definitely|exactly|obviously)\s+)?"
            r"(?:the|your)\s+(?:kind\s+of\s+)?(?:man|guy|person)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "promise to handle date planning",
        re.compile(
            r"\b(?:i\s+can|i'll|i\s+will)\s+"
            r"(?:(?:definitely|happily|gladly)\s+)?"
            r"(?:handle|plan|take\s+care\s+of)\s+(?:the|our|your)\s+"
            r"(?:date|dates|planning|plans?|details|itinerary)\b",
            re.IGNORECASE,
        ),
    ),
)


def _premature_shared_future_markers(text: str) -> list[str]:
    """Return unmistakable shared-future or role-audition wording in an opener.

    The multimodal prompt owns the broad social judgment: a direct invitation is welcome,
    while speaking as though an unaccepted date already belongs to both people is not. This
    deterministic backstop intentionally catches only two high-precision shapes that should
    never be sent: possessive ``our first ...`` date/outing language, a first-person future
    performance promise, an explicit claim to be ``the man`` or ``your guy``, and an offer to
    handle date planning. These are how an opener turns a profile preference into a job
    application rather than a relaxed conversational bid.

    Do not broaden this to every ``we``, ``our``, or future-tense construction. Natural direct
    proposals need those words, and a false positive spends a provider request before the
    service can recover. Broader calibration belongs in the prompt; this helper guarantees the
    reported failure shape cannot pass unchanged again.
    """
    return [label for label, pattern in _PREMATURE_SHARED_FUTURE_PATTERNS
            if pattern.search(text)]


def _sensitive_inference_markers(text: str) -> list[str]:
    """Return the high-risk inferred-activity shapes found in an otherwise clean opener.

    This is intentionally a small lexical backstop, not an attempt to infer whether a sport is
    safe from the outgoing text. The multimodal prompt owns that evidence judgment. Here we only
    reject direct wording that makes a person in a scene answer for jumping from a height,
    courage around danger, or self harm. An empty list means the text can continue through the
    ordinary parse path.
    """
    return [label for label, pattern in _SENSITIVE_INFERENCE_PATTERNS if pattern.search(text)]


def _is_location_confirmation_phrase(text: str) -> bool:
    """Return whether *text* contains only a place identification or correction.

    The confirmation allowlist must be stricter than "starts with a place": arbitrary text
    after the place can reintroduce the exact activity premise this guard exists to reject.
    This small grammar accepts capitalized place names, ordinary alternatives, and directional
    corrections. If a valid place spelling falls outside it, one retry is safer than allowing
    an experience clause hidden inside a nominally location-only question.
    """
    phrase = " ".join(str(text).split()).strip()
    phrase = re.sub(
        r"(?i)\s+(?:instead|maybe|perhaps|by\s+any\s+chance)$", "", phrase
    ).strip()
    if not phrase:
        return False

    directional = re.fullmatch(
        r"(?i)out\s+(?:west|east|north|south)"
        r"(?:\s+or\s+somewhere(?:\s+else)?\s+"
        r"(?:in|at|near|around|outside|on)\s+(.+))?",
        phrase,
    )
    if directional:
        other_place = directional.group(1)
        return other_place is None or _PROPER_PLACE_RE.fullmatch(other_place) is not None

    if re.fullmatch(r"(?i)somewhere(?:\s+else)?", phrase):
        return True

    alternatives = re.split(r"(?i)\s+or\s+", phrase)
    if len(alternatives) > 1:
        return all(_is_location_confirmation_phrase(part) for part in alternatives)

    phrase = re.sub(
        r"(?i)^(?:somewhere(?:\s+else)?\s+)?"
        r"(?:in|at|near|around|outside|on|up\s+around)\s+",
        "",
        phrase,
        count=1,
    ).strip()
    return _PROPER_PLACE_RE.fullmatch(phrase) is not None


def _has_non_out_there_context(text: str) -> bool:
    """Return whether ``text`` carries a standalone ``there`` context hint.

    A bare ``there`` is useful for travel-anchored followups, but we intentionally ignore
    ``out there`` and similar figurative phrases unless another location cue is present.
    """
    tokens = re.findall(r"[A-Za-z']+", str(text).lower())
    for idx, token in enumerate(tokens):
        if token != "there":
            continue
        if idx == 0 or tokens[idx - 1] != "out":
            return True
    return False


def _has_location_context(text: str) -> bool:
    cleaned = " ".join(str(text).split())
    return (_LOCATION_CONTEXT_NOUN_RE.search(cleaned) is not None
            or _LOCATION_CONTEXT_TRAVEL_RE.search(cleaned) is not None
            or _has_non_out_there_context(cleaned))


def _unconfirmed_location_followup_markers(text: str) -> list[str]:
    """Find a later beat that treats an inferred proper-name location as established.

    This is intentionally a narrow outgoing-text check, not geographical named-entity
    recognition. A cue must explicitly frame a capitalized conclusion as a visual guess. That
    keeps ordinary figurative lines such as ``looks like serious dedication`` out of the gate,
    while covering the recurring live shape regardless of the particular place name.

    Once such a cue appears, the first sentence boundary or question-like comma/conjunction
    after it starts the possible followup. A direct confirmation or correction of the location
    remains valid. Any other later statement or question is rejected because it may ask about
    an activity, preference, feeling, or experience that is coherent only if the location guess
    was right. Precision wins over recall here: a false positive spends a billed retry, while
    the model-facing confirmation rule handles inference wordings this helper cannot prove.
    """
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return []
    cue_matches = [(idx, match)
                   for idx, pattern in enumerate(_UNCONFIRMED_LOCATION_CUE_PATTERNS)
                   if (match := pattern.search(cleaned)) is not None]
    if not cue_matches:
        return []
    cue_idx, cue = min(cue_matches, key=lambda item: item[1].start())

    # Protect abbreviations without changing string length, so a place such as St. Moritz does
    # not manufacture a sentence boundary and all match offsets still index ``cleaned``.
    protected = _COMMON_ABBREVIATION_RE.sub(
        lambda match: match.group(0).replace(".", "\u2024"), cleaned)
    protected = _PLACE_PREFIX_ABBREVIATION_RE.sub(
        lambda match: match.group(0).replace(".", "\u2024"), protected)
    protected = _DOTTED_ABBREVIATION_BEFORE_LOWER_RE.sub(
        lambda match: match.group(0).replace(".", "\u2024"), protected)
    protected = re.sub(
        r"\b(?:e\.g|i\.e)\.",
        lambda match: match.group(0).replace(".", "\u2024"),
        protected,
        flags=re.IGNORECASE,
    )
    boundaries = [
        match
        for pattern in (
            _SENTENCE_END_RE,
            _LOCATION_FOLLOWUP_COMMA_RE,
            _LOCATION_FOLLOWUP_CONJUNCTION_RE,
        )
        if (match := pattern.search(protected, cue.end())) is not None
    ]
    if not boundaries:
        return []
    boundary = min(boundaries, key=lambda match: match.start())
    followup = cleaned[boundary.end():].strip()
    followup = re.sub(
        r"^[\"')\]]*\s*(?:(?:but|and|or|so|then)\s+)?",
        "",
        followup,
        count=1,
        flags=re.IGNORECASE,
    )
    if not followup:
        return []
    confirmation = re.sub(r"\s*:\)$", "", followup).rstrip()
    mentions_recipient = re.search(
        r"\byou(?:r|rs|rself)?\b", confirmation, re.IGNORECASE) is not None
    if (not mentions_recipient
            and any(pattern.fullmatch(confirmation)
                    for pattern in _DIRECT_LOCATION_CONFIRMATION_PATTERNS)):
        return []
    location_query_matches = [
        match
        for pattern in _DIRECT_LOCATION_QUERY_PATTERNS
        if (match := pattern.fullmatch(confirmation)) is not None
    ]
    if (not mentions_recipient
            and any(_is_location_confirmation_phrase(match.group("location"))
                    for match in location_query_matches)):
        return []
    if cue_idx == 0:
        sentence_start_candidates = [
            cleaned.rfind(ch, 0, cue.start()) for ch in [".", "!", "?", ";", ":"]
        ]
        sentence_start = max(sentence_start_candidates) + 1
        context_window = cleaned[sentence_start:boundary.end()]
        context_window = " ".join((context_window + " " + followup).split())
        if not _has_location_context(context_window):
            return []
    return ["unconfirmed location used as a later premise"]


# Function words plus the handful of "structural" nouns that name the CONTAINER rather than
# the detail inside it. Both groups are dropped before _redundant_description_markers compares
# the two strings, for the same reason: they carry no information about WHICH profile this is.
# "photo"/"prompt"/"picture" earn their place in the second group because nearly every
# `referenced` value the model writes starts with one of them ("photo of her with a husky in
# the arctic"), so counting them would swamp the real signal ("husky", "arctic") with a
# constant that fires on almost every profile.
_REDUNDANCY_STOPWORDS = frozenset("""
    a an the this that these those there here
    i me my mine myself you your yours yourself we us our ours
    he him his she her hers it its they them their theirs
    who whom whose what which when where why how
    is am are was were be been being do does did doing have has had having
    will would shall should can could may might must let
    of in on at to for with from by about into onto over under near around as after before
    and or but if so than then too very just also not no nor only own same still yet
    one two some any all both each few more most other such
    photo photos picture pictures pic pics image images shot shots
    prompt prompts card cards answer answers profile bio caption
""".split())

# Tokenizer for the redundancy monitor: letters and digits only, so possessives and
# contractions split ("husky's" -> "husky", "s") and the leftover fragment is dropped by the
# length filter below. That loses a little signal and never invents any, which is the right
# direction for a metric that is already documented as a lower bound.
_REDUNDANCY_WORD_RE = re.compile(r"[a-z0-9]+")

# Tokens shorter than this are dropped whether or not they are in the stopword list above.
# It is a cheap catch-all for the function words and fragments no hand-written list covers
# ("ah", "id", the "s" left behind by a possessive). It does cost a few real two-letter words
# (a DJ, an ox), and that is the correct trade for a monitor: a two-character overlap between
# two short strings is noise far more often than it is evidence of a restated description, and
# every loss here only makes an already-declared lower bound slightly looser.
_REDUNDANCY_MIN_WORD_LEN = 3


def _redundancy_content_words(text: str) -> list[str]:
    """Lowercase content words of ``text`` in order of appearance, punctuation stripped and
    stopwords/short fragments dropped. Shared by both sides of the comparison in
    _redundant_description_markers so the two strings are always normalized identically."""
    return [word for word in _REDUNDANCY_WORD_RE.findall(str(text).lower())
            if len(word) >= _REDUNDANCY_MIN_WORD_LEN and word not in _REDUNDANCY_STOPWORDS]


def _redundant_description_markers(opener: str, referenced: str) -> list[str]:
    """Deterministic (NOT a second LLM call) MONITOR for the over-description bug: content
    words the opener restated from the model's own `referenced` note. Returns a list of short
    human-readable markers, one per distinct restated word, in the order they appear in
    ``referenced``; an empty list means clean. Pure function, exactly like
    _scaffolding_markers, and normalization is identical on both sides (lowercase, punctuation
    stripped, stopwords and sub-3-character fragments dropped -- see _redundancy_content_words).

    THE IDEA (ops/OPENER-REDESIGN.md 3.7). Over-description looks like a semantic property but
    has a deterministic proxy sitting in the same parsed dict: the model already tells us, in
    `referenced`, what it was reacting to. If `referenced` is "outdoor sauna at sunset" and the
    opener contains "sauna" and "sunset", the opener is restating its own grounding note back
    to a woman who is looking at that exact photo while she reads it. On that real pair the
    signal separates cleanly: the bad opener overlaps on two content words, the good one
    ("That view looks relaxing, where is this from?") on zero.

    IT IS A LOWER BOUND, NOT A MEASUREMENT. A terse `referenced` ("the sauna photo") defeats it
    completely while the opener describes the scene in full, and it counts words rather than
    meaning, so a paraphrase ("that steam room") scores zero. Morphology defeats it too: the
    match is exact-token, so "huskies" against "husky" does not count. Every one of those
    failures is a MISS (redundancy present, nothing reported), never a false alarm on a clean
    opener, which is the direction a monitor should fail in.

    IT SHIPS LOG ONLY, AND CALLERS MUST NOT GATE ON IT. Four reasons, all from doc 3.7:
      1. Being a lower bound (above) makes it structurally unfit as the primary defence. The
         prompt is the fix; this only measures whether the fix worked.
      2. Five consecutive rejections stop the whole run (service.py). An uncalibrated
         threshold is therefore a run-killer, not merely a noisy check.
      3. There is real opener data, and it is far too thin to set a threshold on: the local
         sqlite db has 0 rows and the BigQuery `openers` table holds only a handful of live
         rows. Enough to sanity-check the metric, nowhere near enough to pick a cutoff.
      4. The threshold can be derived offline for free later, because record_opener already
         persists both `opener` and `referenced` (store.py, bigquery_store.py). No schema
         change and no live experiment is needed to calibrate it.
    Promote it to a gate only once real data shows a threshold with a zero false-positive rate
    on owner-approved openers. This stays within the existing scaffolding-defense decision:
    deterministic only, no LLM judge, no classifier.

    CALIBRATION WARNING for whoever does that offline pass: prompt rewrites created
    comparability boundaries at 2026-08-11 (the Part A substance rewrite) and 2026-09-05 (the
    spoken register rewrite). Rows from different eras are not comparable: each rewrite
    changes what the model was told, so overlap counts shift on their own, and a threshold
    fitted across a boundary is fitted to a mixture. No row carries a style version stamp;
    split rows by created_at against config.yaml's git commit dates.

    AS OF 2026-09-05 (b) THAT IS ONLY HALF TRUE. Every new row in `openers` and
    `opener_rejections` now carries `prompt_sha256`, the digest of the prompt era that
    produced it (see prompt_stamp above for exactly what the digest covers and what it
    deliberately excludes), so those rows split by GROUP BY prompt_sha256 rather than by
    reconstruction. Rows written BEFORE that date read back NULL -- the column is nullable
    and nothing backfills it -- and NULL means "predates the stamp", not "no era": those rows
    still need the created_at vs. git-dates split described above, so an offline pass over a
    range spanning 2026-09-05 (b) has to use both methods at once.
    """
    referenced_words = _redundancy_content_words(referenced)
    opener_words = set(_redundancy_content_words(opener))
    markers: list[str] = []
    seen: set[str] = set()
    for word in referenced_words:
        if word in opener_words and word not in seen:
            seen.add(word)
            markers.append(f'opener restates the referenced word "{word}"')
    return markers


# Tokenizer for _leading_ngram. Unlike the redundancy tokenizer above this KEEPS apostrophes,
# because the phrases the entropy guard exists to catch are contraction-heavy ("I'm going to
# guess") and splitting them would make "I'm going" and "I am going" collide with each other
# while "im"/"i" fragments polluted the n-gram.
_NGRAM_WORD_RE = re.compile(r"[a-z0-9']+")


def _leading_ngram(text: str, n: int = 4) -> str:
    """The normalized leading n-gram of an opener: its first ``n`` words, lowercased, with
    punctuation dropped and whitespace collapsed to single spaces. Pure and deterministic; a
    text with fewer than ``n`` words returns all of them, and ``n <= 0`` returns "".

    Exists for the entropy guard in ops/OPENER-REDESIGN.md 3.6, whose stated rationale is half
    stale -- read it with the 2026-09-05 addendum at the end of that file. Shortening the
    openers compresses the output space, which is what makes collisions likely at all (SAY IT
    ONCE shortens them further), but the style block has shipped no examples since the
    2026-08-16 de-templating pass, so copying prompt copy is NOT the risk. The guard exists
    because a minimal thinking model collapses onto a favorite construction on its own
    (measured live 2026-08-11: 5 of 5 openers led with the same hedge), and because near
    identical openers across an uncapped volume burner account are a fingerprint. The
    guard is a plain string comparison of this value against the last N successful openers
    (OpenerService.recent_openers), which catches "Based on the X, I'm going to guess"
    recurring without needing any taxonomy of moves, any semantics, or any second model call.

    Deliberately dumb, and deliberately free of policy: it decides nothing. What counts as a
    collision, what happens on one, and how a collision interacts with the attempt budget are
    all the service's business (that budget accounting is the entire reason the guard cannot
    simply reject -- a hard rejection here would burn one of OpenerService.max_attempts over a
    stylistic near-miss. Observe's advisory-preview path used to run this guard against its own
    separately-shortened attempt budget, making a hard rejection there riskier still; that path
    and its budget were removed on 2026-09-06, leaving one uniform max_attempts for every call).
    Keeping the string normalization here, alone, is what lets that policy change without
    touching the definition of the thing being compared.
    """
    if n <= 0:
        return ""
    words = (word.strip("'") for word in _NGRAM_WORD_RE.findall(str(text).lower()))
    return " ".join([word for word in words if word][:n])


def prompt_stamp(style: str) -> str:
    """The SHA-256 of the prompt era `style` belongs to, so an offline calibration pass can
    GROUP BY era instead of reconstructing the boundaries by hand.

    WHY IT EXISTS. Two prompt rewrites created comparability boundaries -- 2026-08-11 (the
    Part A substance rewrite) and 2026-09-05 (the spoken register rewrite) -- and until this
    date no row carried any style version at all, so splitting rows by era meant reading
    created_at against config.yaml's git commit dates by hand and hoping no untracked edit
    happened in between. That method is a reconstruction; this is a stamp, written by the
    process that actually sent the prompt.

    WHAT THE DIGEST COVERS. SEVEN components, joined in this fixed order by a NUL byte:
    (i) the owner's `style` text from config.yaml (the user turn), (ii) `_SYSTEM`
    (systemInstruction), (iii) the canonicalized `_SCHEMA` field descriptions
    (responseJsonSchema) -- three of the FOUR on-wire prompt copies, see the 2026-09-05
    addendum in ops/OPENER-REDESIGN.md for the four-copies correction -- and then the
    item-crop shape's four instruction constants: (iv) `_ITEM_PREAMBLE`,
    (v) `_ITEM_PREAMBLE_CONTEXT`, (vi) `_ITEM_LABEL` and (vii) `_CONTEXT_LABEL`. The last
    four are model-facing instruction PROSE, not bookkeeping: the preamble is the first text
    part of every item-crop request, and between them these constants carry the compound-item
    rule (a title, caption, or prompt above a photo is ONE item with it) and the
    cannot-be-picked rule for context crops, which `_CONTEXT_LABEL` then restates on every
    context image. All seven are the prompt-shaped bytes that are STABLE for a whole run AND
    live in module-level constants, which is what makes a per-run stamp meaningful at all --
    but they are not every byte of instruction the model sees; see exclusion (4). A NUL sits
    between EVERY adjacent pair, so no boundary between two constants is ambiguous: prose
    moved from the end of one to the start of the next changes the digest instead of
    reassembling to the same bytes. `_SCHEMA` is canonicalized (sort_keys, compact
    separators, ensure_ascii) so a dict-literal reordering that changes no text changes no
    digest.

    WHAT IT DELIBERATELY EXCLUDES, and why. (1) The retry-hint prose: it is
    REQUEST-conditional, built per attempt from whichever guard rejected the previous draft
    (see _text_part), so folding it in would give the same era as many digests as it has
    failure modes and would make the first attempt and its own retry look like two eras.
    (2) generationConfig / thinking settings: they change how hard the model works, not what
    it was told. (3) The model id: already its own column in both tables, so hashing it in
    would only make the stamp less joinable. (4) _text_part's OWN instruction prose -- the
    closing block, the truncation and CONTEXT sentences, and the STYLE GUIDE / HER NAME / HER
    PROFILE TEXT section labels. This exclusion is the TRAILING text part and nothing else:
    the item-crop preamble and the per-image labels are a separate body of prose and ARE
    hashed, above. That block is excluded as a block because it is request-conditional (its
    item counts, its name section, and its truncation sentence all vary per profile), but it
    is not merely bookkeeping: it carries fixed sentences that are genuine prompt RULES,
    including its own CONTEXT-images sentence ("Never pick one or make one the opener's main
    premise") and the item_index instruction, and that copy has been rewritten before. So an
    edit confined to those sentences is a real era boundary that this digest does NOT move.

    THE COST, stated rather than smoothed over: equal digests mean the same era only up to
    the exclusions. Two eras differing ONLY in retry-hint prose, or ONLY in _text_part's fixed
    instruction sentences, are indistinguishable here; changing either one and expecting the
    stamp to record it is the false negative to watch for.

    OFFLINE REPRODUCTION. To learn which era a stamp names: check out the commit you suspect,
    load the config, and call ``prompt_stamp(cfg.opener.style)``; equal digests mean the same
    era in the covered sense above. Pure and deterministic, no I/O -- deliberately a local
    hashlib call rather than ranker.retractions.canonical_sha, which would add an
    opener->ranker package edge for one line of hashing.
    """
    payload = "\x00".join((
        str(style),
        _SYSTEM,
        json.dumps(_SCHEMA, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        _ITEM_PREAMBLE,
        _ITEM_PREAMBLE_CONTEXT,
        _ITEM_LABEL,
        _CONTEXT_LABEL,
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sentence_count(text: str) -> int:
    """Count ordinary message sentences for the hard two-sentence send guard.

    This is intentionally narrower than a prose tokenizer: openers are short plain
    messages, while protecting common abbreviations avoids rejecting harmless text
    such as ``Dr. Dolittle energy. What's the story?``.
    """
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return 0
    protected = _COMMON_ABBREVIATION_RE.sub(
        lambda match: match.group(0).replace(".", "\u2024"), cleaned)
    protected = re.sub(r"\b(?:e\.g|i\.e)\.",
                       lambda match: match.group(0).replace(".", "\u2024"),
                       protected, flags=re.IGNORECASE)
    endings = _SENTENCE_END_RE.findall(protected)
    return max(1, len(endings))


class OpenerError(RuntimeError):
    """A provider response could not be turned into an opener (refusal, or truncated/
    malformed structured output) — distinct from a transport/billing failure."""


@dataclass
class OpenerResult:
    opener: str
    referenced: str
    usage: Usage
    model: str
    # WHICH ITEM THE MODEL PICKED: 1-based, and it indexes the NUMBERED ITEMS that were sent in
    # this request (ops/OPENER-REDESIGN.md 5.1/5.7). Per doc 5.1 this is one answer to two
    # questions -- the item the opener is about AND the item to like -- which is what makes the
    # message and the like land on the same thing by construction instead of by repair.
    #
    # THIS REPLACES `referenced_index`, AND IT IS NOT A RENAME. That field was a 0-BASED index
    # into raw SCROLL FRAMES, which is not an item space at all: one card can appear in three
    # frames, one frame can hold two cards, and the model had to invent its own enumeration and
    # hope it matched the driver's (doc 5.2's "agreement by luck"). Nothing may reinterpret an
    # old integer as a new one -- the meaning, the base, and the thing counted all changed.
    #
    # ITEM_INDEX_ABSENT (0) when the model gave no usable number; see that constant for why 0
    # is safe as an out-of-band value here and what a consumer may NOT conclude from it.
    item_index: int = ITEM_INDEX_ABSENT
    # WHICH LIST `item_index` COUNTS -- one of INDEX_SPACE_*, set by generate() from the request
    # shape it actually built. Read those constants before using either field: the number is
    # meaningless without this, and the 2026-08-12 correction exists because a consumer that
    # assumed the space got a confident, in-range, wrong answer instead of a failure.
    #
    # Defaults to INDEX_SPACE_MODEL_ITEMS, the UNTRANSLATABLE space, on purpose. Every
    # OpenerResult built outside generate() -- the fake clients in tests, any future call site
    # -- therefore carries an index no consumer will convert into a tap, which is the safe
    # direction to be wrong in. Defaulting to PROFILE_PHOTOS would mean a stand-in result got
    # its number silently turned into a coordinate on somebody's phone.
    index_space: str = INDEX_SPACE_MODEL_ITEMS
    # The model's own free-text words for what its opener is DOING ("guessing where the ridge
    # is", "teasing her about the cold", ...) -- see _SCHEMA's `angle` property. Deliberately
    # not an enum anywhere in this pipeline (ops/OPENER-REDESIGN.md 3.5): a closed set would
    # force the model to pick a move and shoehorn the opener into it, which is exactly the
    # awkwardness the move list is written to avoid. Pure telemetry: nothing reads it to make
    # a decision, it exists so we can eventually ask which shapes correlate with matches.
    #
    # Defaults to "" rather than being required so every OpenerResult built outside _parse --
    # the fake OpenerClients in tests, and any future call site -- keeps working unchanged.
    angle: str = ""
    # The model's own short description of the ITEM it picked ("a photo of her on a ridge", "a
    # prompt about hot sauce") -- see _SCHEMA's `item_description` property. NOT the same field
    # as `referenced`, and doc 5.7 is explicit that neither replaces the other: `referenced` is
    # the full-text DETAIL the opener reacts to (what the redundancy monitor compares the
    # opener against, and what store.record_opener persists as the telemetry note), while this
    # describes the ITEM, coarsely, so doc 5.8's pre-flight cross-check can ask whether the
    # thing we cropped at that index is the same KIND of thing the model thought it chose.
    #
    # RETURNED IN BOTH MODES, always (doc 5.7): auto logs it, observe displays it. It must
    # never become conditional on `advisory`, because a mode-dependent schema would mean auto
    # and observe issue different requests, and observe's entire value is that its opener is
    # byte-identical to what auto would have sent for the same profile.
    #
    # That cross-check is a LATER workflow; nothing reads this to make a decision today. Same
    # default rationale as `angle` above.
    item_description: str = ""
    # Output of the deterministic redundancy MONITOR (_redundant_description_markers): the
    # content words this opener restated from its own `referenced` note. LOG ONLY, never a
    # rejection -- a non-empty list here has no effect on whether the opener is sent, by
    # design (see _redundant_description_markers' docstring and doc 3.7 for the four reasons
    # it must not be a gate yet). Same default rationale as `angle` above.
    redundancy_markers: list[str] = field(default_factory=list)


# Machine-readable reason codes for OpenerParseError.reason_code -- see the class docstring
# below and ranker/bigquery_store.py's opener_rejections table (persisted so guard-firing
# frequency can be aggregated with a GROUP BY instead of regex-parsing free-text `message`,
# which stays exactly the retry_hint the model reads and existing tests assert on verbatim).
# Bare string literals rather than an enum: they cross a storage boundary (a BigQuery/sqlite
# TEXT column) where an enum would just get str()'d back down to one of these values anyway.
#
# The two BLOCK codes lead the list because their guards fire first (see _parse's text-is-None
# region):
#   prompt_blocked   -- the body carries promptFeedback.blockReason and, in the live 2026-08-15
#                       capture, NO "candidates" key at all: the provider refused the REQUEST
#                       and never generated anything.
#   response_blocked -- candidates exist but the candidate's finishReason is one of
#                       _BLOCK_FINISH_REASONS: the provider generated, then withheld the ANSWER.
# Both describe a response with no opener text in it, which is exactly why it is tempting to
# fold them into no_text -- and exactly why they must not be. no_text means we do NOT know what
# happened (an unexplained shape, the case that needs a shape dump to diagnose); these two mean
# the provider told us precisely what happened and it was a REFUSAL, a property of THIS
# profile's images plus THIS model rather than of a broken prompt, a bad schema, or an
# exhausted token budget. Kept apart so a GROUP BY on opener_rejections can answer "how often
# does the safety filter refuse us, and on which models" without that count being buried under
# every shape we could not explain.
REASON_PROMPT_BLOCKED = "prompt_blocked"
REASON_RESPONSE_BLOCKED = "response_blocked"
REASON_NO_TEXT = "no_text"
REASON_MAX_TOKENS = "max_tokens"
REASON_BAD_JSON = "bad_json"
REASON_MISSING_FIELD = "missing_field"
REASON_NOT_A_STRING = "not_a_string"
REASON_EMPTY_AFTER_SANITIZE = "empty_after_sanitize"
REASON_UNDELIVERABLE_CHARS = "undeliverable_chars"
REASON_UNDELIVERABLE_SEQUENCE = "undeliverable_sequence"
REASON_SCAFFOLDING = "scaffolding"
REASON_PREEMPTIVE_DISCLAIMER = "preemptive_disclaimer"
REASON_PREMATURE_SHARED_FUTURE = "premature_shared_future"
REASON_SENSITIVE_INFERENCE = "sensitive_inference"
REASON_UNCONFIRMED_LOCATION_FOLLOWUP = "unconfirmed_location_followup"
REASON_TOO_MANY_SENTENCES = "too_many_sentences"

# The three codes below are NOT OpenerParseError.reason_code values -- nothing in _parse below
# ever raises with one of these. OpenerParseError's guards only run on a response that DID come
# back from the provider; these three name failure kinds that service.py's maybe_opener catches
# from earlier or later in the same call (a request that never reached the provider at all, an
# HTTP-level rejection of the request itself, or a transport/timeout failure) and passes to
# self.store.record_opener_rejection directly, not through an OpenerParseError. They live here,
# following the same REASON_* naming and bare-string-literal rationale as the block above, so
# that method's reason_code column can classify EVERY rejection kind -- not just parse
# failures -- without service.py inventing free text ranker/bigquery_store.py's opener_rejections
# table would have to be queried with a LIKE instead of a GROUP BY. See maybe_opener's
# OpenerError / HTTP-400 / generic-transient-exception branches for exactly where each fires.
REASON_OPENER_ERROR = "opener_error"
REASON_BAD_REQUEST = "bad_request"
REASON_TRANSIENT_ERROR = "transient_error"

_BLOCK_FINISH_REASONS = frozenset({
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
})


class OpenerParseError(OpenerError):
    """A billed API response did not parse into a usable opener.

    Carries normalized usage/model so the caller can still record a provider's spend
    after bad JSON, missing keys, or a response without usable text.

    reason_code is one of the REASON_* constants above, letting a persisted rejection row
    be grouped/queried by failure kind without regex-parsing `message`. Defaults to None
    only so external callers/tests built against the original 3-positional-arg signature
    keep working unchanged; every raise site inside _parse below passes one explicitly.

    raw_opener is the literal candidate text the FAILING GUARD actually looked at, so a
    persisted rejection row shows exactly what got rejected, not just why:
      - pre-sanitize (the value straight out of the model's JSON, before fold_to_ascii's
        dash-fold/accent-fold ran) when the guard runs BEFORE _sanitize is ever called
        (not_a_string -- the value there usually isn't even a string).
      - post-sanitize (after fold_to_ascii) when the guard runs on the already-sanitized
        string (empty_after_sanitize, undeliverable_chars, undeliverable_sequence,
        too_many_sentences, scaffolding, preemptive_disclaimer, premature_shared_future,
        sensitive_inference, unconfirmed_location_followup) -- these all call their check
        function with `sanitized`, so that is unambiguously the value being judged, even for
        empty_after_sanitize where the result is "" itself.
      - the raw response text (no per-field candidate exists to point at) for bad_json and
        missing_field: neither ever produced a usable "opener" value, so the whole raw text
        the model returned is the most useful thing available, well short of nothing.
      - None for no_text/max_tokens/prompt_blocked/response_blocked: the response carried no
        text part at all, so there is nothing to show.
    Defaults to None so existing raisers/tests that don't pass it keep working unchanged.
    """

    def __init__(self, message: str, usage: Usage, model: str, *,
                 reason_code: str | None = None, raw_opener: str | None = None):
        super().__init__(message)
        self.usage = usage
        self.model = model
        self.reason_code = reason_code
        self.raw_opener = raw_opener


class OpenerAborted(OpenerError):
    """The run is stopping: ``should_stop()`` reported true mid-cascade, before the next
    model's request was issued (see GeminiOpener.generate()'s per-iteration check).

    Deliberately NOT the same thing as an ordinary OpenerError: a corrupt photo or a
    provider refusal says something is wrong with THIS profile's request, while this says
    nothing about the request at all -- the operator clicked Stop (or the supervisor is
    shutting down) and the cascade is unwinding on purpose. Callers (OpenerService.
    maybe_opener) must treat the two very differently: an ordinary OpenerError is a real
    per-profile failure that counts toward the transient-failure latch, but a deliberate
    shutdown must not retry, must not call _exhaust(), and must not poison the service's
    health state (the latch counters, exhausted_reason) with what is not a provider
    failure at all -- see BUG 1's fix. Message text says plainly that the run is stopping,
    not that a request failed, so an operator reading a log line (or last_skip_reason on
    the hub) is never left thinking the provider did something wrong.
    """


class OpenerClient(Protocol):
    """What OpenerService requires of an opener client -- i.e. every argument the service
    actually passes, and nothing more.

    `items` is declared because OpenerService threads it unconditionally on every call, so a client
    that cannot accept the kwarg fails LOUDLY with a TypeError instead of silently dropping the
    numbered crops and sending `profile.photos` in their place. That silent fallback is the one
    outcome doc 5.2 exists to prevent -- raw scroll frames cannot carry an item number, so a
    dropped `items` would give the model a numbering nothing downstream can act on.

    """

    def generate(self, profile: Profile, style: str, retry_hint: str = "", *,
                 items: "ItemRequest | None" = None,
                 should_stop: Callable[[], bool] | None = None,
                 skip_models: frozenset[str] = frozenset()) -> OpenerResult: ...


class GeminiAPIError(RuntimeError):
    """A non-success Gemini GenerateContent response, without request credentials.

    ``status`` is Gemini's machine-readable error status (for example,
    ``RESOURCE_EXHAUSTED``); ``http_code`` is the HTTP status.  Keeping those fields
    separate lets the fallback policy distinguish actual capacity exhaustion from a
    malformed request that also happens to use a 4xx status.

    ``quota_id``/``quota_metric`` are populated only for a 429 whose body carries a
    structured ``google.rpc.QuotaFailure`` detail; both are ``None`` for every other
    error and for a 429 whose body doesn't include that detail (some do not). They
    exist so the fallback policy can tell a transient per-minute cap apart from an
    actual per-day exhaustion instead of treating every 429 identically.
    """

    def __init__(self, http_code: int, status: str | None, message: str, *,
                 quota_id: str | None = None, quota_metric: str | None = None):
        self.http_code = int(http_code)
        self.status = status or ""
        self.message = message
        self.quota_id = quota_id
        self.quota_metric = quota_metric
        detail = f"Gemini API HTTP {self.http_code}"
        if self.status:
            detail += f" {self.status}"
        super().__init__(f"{detail}: {message}")


class GeminiCapacityExhausted(RuntimeError):
    """Every configured Gemini model returned 429 RESOURCE_EXHAUSTED this run."""


# Callable[...] rather than a precise positional signature: preflight() calls this with an
# extra keyword-only `method="GET"` that generate()'s POST calls never pass, and Callable
# can't express "these args, plus this optional kwarg" precisely.
GeminiTransport = Callable[..., tuple[int, Any]]

# ListModels (used by preflight()) has no request body of its own.
_GEMINI_MODELS_LIST_URL = "https://generativelanguage.googleapis.com/v1beta/models"
# Keep direct construction as safe as the config-backed path.  Config validation uses the
# same measured three-minute ceiling; GeminiOpener is also instantiated directly by tests and
# small tools, so relying on that upstream gate alone leaves one HTTP call effectively
# unbounded.
_MAX_REQUEST_TIMEOUT_S = 180.0
_MAX_PREFLIGHT_PAGES = 100
_MAX_GEMINI_RESPONSE_BYTES = 1 * 1024 * 1024
_MAX_REQUEST_IMAGES = 64
_MAX_RAW_IMAGE_BYTES = 64 * 1024 * 1024
_THINKING_KEYS = frozenset({"thinkingLevel", "thinkingBudget"})
_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})

# Gemini hard-caps a request's TOTAL encoded size (text + system instruction + inline image
# bytes) at 20MB. We budget under that with real headroom: base64 already inflates raw photo
# bytes by ~33%, our size estimate doesn't count the generationConfig/schema JSON scaffolding,
# and it's cheaper to compress a bit more than necessary than to 400 a nearly-fitting request.
_MAX_INLINE_REQUEST_BYTES = 18 * 1024 * 1024


def _first_quota_violation(details: Any) -> tuple[str | None, str | None]:
    """Pull ``quotaId``/``quotaMetric`` out of a 429's ``error.details[]``, if present.

    Defensive by design: Google does not guarantee every 429 body includes a structured
    QuotaFailure (and third-party proxies/mocks in tests may omit it entirely), so any
    shape mismatch here must degrade to "no violation found" rather than raise -- the
    caller's classifier already treats that as "unknown" and handles it safely.
    """
    if not isinstance(details, list):
        return None, None
    for entry in details:
        if not isinstance(entry, dict):
            continue
        entry_type = entry.get("@type")
        if not isinstance(entry_type, str) or "QuotaFailure" not in entry_type:
            continue
        violations = entry.get("violations")
        if not isinstance(violations, list):
            continue
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            quota_id = violation.get("quotaId")
            quota_metric = violation.get("quotaMetric")
            if quota_id or quota_metric:
                return (
                    str(quota_id) if quota_id is not None else None,
                    str(quota_metric) if quota_metric is not None else None,
                )
    return None, None


def _classify_quota_exhaustion(error: GeminiAPIError) -> str:
    """Classify a 429 as ``"day"``, ``"minute"``, or ``"unknown"`` from its quota details.

    Real free-tier quota ids/metrics spell the period both as "PerDay"/"PerMinute" (id
    casing) and "per_day"/"per_minute" (metric, underscored), so both spellings are
    checked case-insensitively against whichever field the response populated.
    """
    haystack = " ".join(filter(None, [error.quota_id, error.quota_metric])).lower()
    if "perday" in haystack or "per_day" in haystack:
        return "day"
    if "perminute" in haystack or "per_minute" in haystack:
        return "minute"
    return "unknown"


# Tokens that identify a 400 as a per-model THINKING CONFIG rejection rather than a generic
# malformed request. MEASURED, live, 2026-08-13, against the real API:
#
#   gemini-3.7-flash + {"thinkingLevel": "minimal"}
#   -> HTTP 400 INVALID_ARGUMENT
#      "Thinking level MINIMAL is not supported for this model. Please retry with other
#      thinking level."
#
# Every OTHER configured model accepted thinkingLevel: minimal without complaint on the same
# run, so this is unambiguously a property of ONE model id -- exactly like a 404 -- not of the
# request or the credentials, and generate()'s non-2xx branch treats it that way (see its 400
# handling). GeminiAPIError carries no structured field naming the offending parameter, so the
# match is on message text; both spellings Gemini's two model families actually use
# (thinkingLevel on the 3.x line, thinkingBudget on 2.5 -- see _payload's own comment) are
# included, plus the generic "thinking config"/"thinkingConfig" phrasing in case a future
# model family's rejection message names the object rather than the field it holds.
_THINKING_REJECTION_TOKENS = (
    "thinking level", "thinkinglevel", "thinking_level",
    "thinkingbudget", "thinking budget", "thinking_budget",
    "thinkingconfig", "thinking config", "thinking_config",
)


def _is_thinking_config_rejection(message: str) -> bool:
    """True when a 400's message narrowly identifies a per-model rejection of the configured
    thinking level/budget, rather than a generic malformed-request 400.

    Deliberately narrow, and matched on nothing but the fixed token set above
    (_THINKING_REJECTION_TOKENS), case-insensitively. A generic 400 -- "Invalid JSON payload
    received", "API key not valid", a bad enum value unrelated to thinking -- must NOT match:
    those are properties of the request or the credentials, not of one model's declared
    capability, and generate() must keep raising them straight to the caller exactly as before
    (OpenerService's invalid-key latch, _is_invalid_gemini_api_key, and the ordinary-400
    consecutive-latch all depend on an unmatched 400 still reaching `raise error`).
    """
    lowered = str(message or "").lower()
    return any(token in lowered for token in _THINKING_REJECTION_TOKENS)


_QUOTA_SCOPE_LABELS = {"day": "per-day quota", "minute": "per-minute throttle",
                       "unknown": "unclassified 429", "gone": "model unavailable",
                       "busy": "provider 5xx", "transport": "network or timeout",
                       "thinking": "thinking config rejected by model"}

# Scopes that clear on their own without any operator action. "busy" is a provider-side 5xx
# (503 UNAVAILABLE "this model is currently experiencing high demand" is the common one, and
# it is genuinely per-MODEL -- observed live on gemini-3.6-flash while the rest of the
# cascade was healthy), so it belongs with the per-minute caps rather than with the dead
# ends: retrying shortly, or on another model right now, is the correct response to all three.
#
# "thinking" is deliberately NOT in this set, unlike every other per-model scope above it.
# Every one of those clears with the passage of time or a retry; a capability rejection never
# does -- gemini-3.7-flash will keep 400ing on thinkingLevel: minimal on every future run until
# opener.thinking is edited for that model id, so telling the operator to "just restart" would
# be actively wrong (same reasoning as "gone", which is also excluded here).
_TRANSIENT_SCOPES = frozenset({"minute", "unknown", "busy", "transport"})


def _exhaustion_reason(scopes: Mapping[str, str]) -> str:
    """Explain why the whole cascade fell through, precisely enough to act on.

    This string becomes the run's stop reason in the hub, so it must not conflate the very
    different situations that can end the cascade. Each model's scope is one of: "day"
    (per-day 429, resets at midnight Pacific), "minute"/"unknown" (a transient per-minute or
    unclassifiable 429 that clears on its own within roughly a minute), "gone" (HTTP 404
    NOT_FOUND -- the model id is retired / not available to this account and will NEVER come
    back mid-run; see generate()'s 404 handling and preflight()'s docstring for why a
    startup ListModels pass cannot catch this in advance), or "thinking" (HTTP 400 rejecting
    this model's configured thinkingConfig -- MEASURED live, see _is_thinking_config_rejection
    -- a per-model capability limit that also will NEVER change mid-run, but for a different
    reason than "gone": the model id is fine, its configured thinking level/budget is not).

    Every model hitting its per-DAY quota means there is genuinely no opener capacity left
    until the midnight Pacific reset -- that case keeps its own message below. Every model
    coming back "gone" is the opposite kind of dead end: no amount of waiting fixes a
    retired model id, so telling the operator to wait for a reset would be actively wrong;
    that case gets its own message too, pointing at opener.models instead of the clock. Every
    model coming back "thinking" is a third, distinct dead end, sharing "gone"'s "waiting never
    helps" property but not its cause or its fix: the model id is still valid, only its
    thinkingConfig is wrong for it, so that case gets its own message pointing at
    opener.thinking instead of opener.models. Anything else (a mix of scopes, or a transient
    per-minute/unknown 429 in the mix) falls through to the generic listing, which already
    assumes at least one transient cause may clear on its own shortly. We still stop the run in
    every case (see OpenerService), because sending a bare like with no opener is a worse
    outcome than halting; only the guidance differs.
    """
    if not scopes:                      # unreachable today (__init__ requires >=1 model)
        return "no configured Gemini model was available to serve the request"
    listed = ", ".join(f"{model} ({_QUOTA_SCOPE_LABELS.get(scope, scope)})"
                       for model, scope in scopes.items())
    if all(scope == "day" for scope in scopes.values()):
        return (f"every configured Gemini model has exhausted its per-day free-tier quota "
                f"({', '.join(scopes)}); free-tier daily quota resets at midnight Pacific")
    if all(scope == "gone" for scope in scopes.values()):
        # Distinct from the per-day case on purpose: mentioning a reset time here would
        # imply waiting helps, and it never does for a retired model id.
        return (f"every configured Gemini model is unavailable to this account (retired or "
                f"not found: {', '.join(scopes)}); this will not resolve on its own -- fix "
                "opener.models to name model ids this account can actually use")
    if all(scope == "thinking" for scope in scopes.values()):
        # Distinct from "gone" on purpose, in the same direction: mentioning a reset time, or
        # telling the operator to just restart, would be wrong for the identical reason it's
        # wrong for "gone" -- but the FIX is different, because the model id itself is not the
        # problem here. MEASURED (see _is_thinking_config_rejection's docstring):
        # gemini-3.7-flash 400s on thinkingLevel: minimal while every OTHER configured model
        # accepts the same value, so the fix is per-model-id thinking config, not the model
        # list.
        return (f"every configured Gemini model rejected its configured thinking level or "
                f"budget ({', '.join(scopes)}); this will not resolve on its own -- fix "
                "opener.thinking for each named model id (it is sending a thinkingLevel or "
                "thinkingBudget that model does not support)")
    if all(scope in _TRANSIENT_SCOPES for scope in scopes.values()):
        # Nothing here is a real dead end: every model was either momentarily throttled or
        # reported a provider-side 5xx. Naming a quota reset would send the operator away
        # for hours over something that typically clears in seconds.
        return (f"no configured Gemini model could serve the request right now: {listed}. "
                "Every one of these is a transient failure (a per-minute cap or a provider "
                "side outage), not an exhausted daily quota, so simply restarting should "
                "succeed")
    return (f"no configured Gemini model could serve the request: {listed}. At least one of "
            "these is a transient per-minute cap rather than a per-day exhaustion, so "
            "restarting in a minute may well succeed instead of waiting for the midnight "
            "Pacific daily reset")


def _gemini_error(status_code: int, body: Any) -> GeminiAPIError:
    """Normalize a Gemini REST error body without echoing request credentials."""
    if not isinstance(body, dict):
        return GeminiAPIError(status_code, None, str(body) or "empty error response")
    error = body.get("error")
    if not isinstance(error, dict):
        return GeminiAPIError(status_code, None, "malformed error response")
    quota_id, quota_metric = _first_quota_violation(error.get("details"))
    # Route on the status returned by the HTTP transport, not the body's redundant
    # ``error.code``.  A proxy or malformed upstream body can disagree with the envelope;
    # allowing that untrusted duplicate to replace the real status could turn a retryable
    # 429/503 into a fatal 400, or (more dangerously) make a fatal 400 look retryable.
    return GeminiAPIError(status_code, error.get("status"),
                          str(error.get("message") or "unknown error"),
                          quota_id=quota_id, quota_metric=quota_metric)


def _stdlib_gemini_transport(url: str, payload: dict[str, Any] | None, headers: dict[str, str],
                             timeout: float, *, method: str = "POST") -> tuple[int, Any]:
    """POST (the default, used by generate()) or GET (used by preflight()'s ListModels
    call) with the standard library; tests replace this whole transport.

    A GET must never carry a JSON body -- ListModels takes none, and some HTTP stacks
    reject a body on GET outright -- so ``data`` stays None whenever method isn't POST.
    """
    data = json.dumps(payload).encode("utf-8") if method == "POST" and payload is not None else None
    request = Request(url, data=data, headers=headers, method=method)

    def read_body(response) -> bytes:
        raw_body = response.read(_MAX_GEMINI_RESPONSE_BYTES + 1)
        if len(raw_body) > _MAX_GEMINI_RESPONSE_BYTES:
            raise RuntimeError(
                "Gemini response exceeded the "
                f"{_MAX_GEMINI_RESPONSE_BYTES}-byte safety limit")
        return raw_body

    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is fixed Gemini endpoint
            raw = read_body(response).decode("utf-8")
            return int(response.status), json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = read_body(exc).decode("utf-8", errors="replace")
        try:
            body: Any = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = raw
        return int(exc.code), body


class GeminiOpener:
    """Gemini GenerateContent opener writer with run-scoped ordered model fallback.

    Any HTTP 429 is treated as a capacity signal for that model (see generate()'s 429
    handling for why the status code alone, not Gemini's own ``RESOURCE_EXHAUSTED`` status
    string, is the reliable trigger). A 429 that DOES classify as a per-day quota means that
    model cannot serve the rest of the current run, so it is skipped thereafter and the same
    profile is attempted with the next configured model; a per-minute/unknown 429 cascades
    without being blacklisted. An HTTP 404 ``NOT_FOUND`` gets the permanent treatment: it
    means only THAT model id is retired / not available to this account, not the request
    itself, so it is permanently dropped from the cascade and the next configured model is
    tried instead -- see generate()'s 404 handling for why a 404 must not be allowed to take
    the rest of the cascade down with it. An HTTP 5xx and a TRANSPORT-level failure (a
    ``socket.timeout``, a ``urllib.error.URLError`` from a connection reset or DNS failure,
    another ``OSError``, or an ``http.client.HTTPException`` such as ``IncompleteRead`` --
    see generate()'s transport-exception handling) both get the same non-blacklisting cascade
    as a per-minute 429:
    neither says anything about whether the model would answer the NEXT request, only that
    it failed to answer this one, so it cascades to the next configured model for this
    profile only and is retried first on the next profile. Authentication, permission, and
    malformed-request errors, and any other 4xx, are deliberately raised to the caller
    unchanged rather than silently cascading: those are properties of the request or
    credentials, not of one model id or one flaky connection, so they would fail identically
    on every other configured model too.

    ONE NARROW EXCEPTION to that last rule: a 400 whose message identifies a per-model
    rejection of the configured thinking level/budget (see generate()'s 400 handling and
    ``_is_thinking_config_rejection``) gets the same permanent-retirement treatment as a 404.
    MEASURED, live, 2026-08-13: ``gemini-3.7-flash`` with ``{"thinkingLevel": "minimal"}``
    returned HTTP 400 INVALID_ARGUMENT, "Thinking level MINIMAL is not supported for this
    model. Please retry with other thinking level.", while every OTHER configured model
    accepted the identical value on the same run -- so that 400, despite its status code, is
    unambiguously a property of ONE model id's declared capability, not of the request or the
    credentials, and must not be allowed to abort the whole cascade over one model's
    mis-specified ``opener.thinking`` entry. Every other 400 -- an invalid API key, a
    malformed payload, anything that does not name the thinking config -- still raises exactly
    as before.

    THREAD SAFETY: instances are safe to share across worker threads. generate() acquires
    this instance's own internal lock for its full duration, so the model cascade and the
    ``_unavailable_models`` retirement it performs are atomic per call -- this class does not
    rely on an external caller (OpenerService) to serialize access on its behalf. Lock
    ordering: OpenerService.maybe_opener acquires ITS OWN lock first and calls into
    generate() while holding it, so the effective order is always "OpenerService's lock,
    then this one" -- and this lock is never held while calling back out into OpenerService
    or anything else that could re-enter it, so the two locks cannot deadlock against each
    other.
    """

    def __init__(self, models: list[str] | tuple[str, ...], max_tokens: int = 400,
                 request_timeout_s: float = 30, *, api_key: str | None = None,
                 env: Mapping[str, str] | None = None,
                 transport: GeminiTransport | None = None,
                 thinking: Mapping[str, Mapping[str, Any]] | None = None):
        if not isinstance(models, (list, tuple)):
            raise ValueError("GeminiOpener models must be a list or tuple of model ids")
        if not models or any(not isinstance(model, str) or not model.strip()
                             or model != model.strip() for model in models):
            raise ValueError(
                "GeminiOpener requires at least one configured model; model ids must be "
                "nonempty strings without surrounding whitespace")
        if len(set(models)) != len(models):
            raise ValueError("GeminiOpener model ids must not contain duplicates")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("GeminiOpener max_tokens must be a positive integer")
        try:
            timeout = (
                float(request_timeout_s)
                if not isinstance(request_timeout_s, bool)
                and isinstance(request_timeout_s, (int, float))
                else None
            )
        except (TypeError, ValueError, OverflowError):
            timeout = None
        if (timeout is None or not math.isfinite(timeout)
                or not 0 < timeout <= _MAX_REQUEST_TIMEOUT_S):
            raise ValueError(
                "GeminiOpener request_timeout_s must be a finite number in "
                f"(0, {_MAX_REQUEST_TIMEOUT_S:g}]")

        self.models = tuple(models)
        environment = os.environ if env is None else env
        self.api_key = api_key if api_key is not None else environment.get("GEMINI_API_KEY")
        if self.api_key is None:
            raise RuntimeError("GEMINI_API_KEY is not set")
        if (not isinstance(self.api_key, str) or not self.api_key.strip()
                or self.api_key != self.api_key.strip()):
            raise RuntimeError(
                "GEMINI_API_KEY must be a nonempty string without surrounding whitespace")
        self.max_tokens = max_tokens
        self.request_timeout_s = timeout
        self.transport = transport or _stdlib_gemini_transport
        # model id -> the exact generationConfig.thinkingConfig dict to send for that model,
        # e.g. {"thinkingLevel": "minimal"} (3.x family) or {"thinkingBudget": 0} (2.5 family).
        # A model with no entry gets no thinkingConfig at all, so its own default applies.
        if thinking is None:
            resolved_thinking: dict[str, dict[str, Any]] = {}
        elif not isinstance(thinking, Mapping):
            raise ValueError("GeminiOpener thinking must map model ids to thinkingConfig mappings")
        else:
            resolved_thinking = {}
            extra_models = set(thinking) - set(self.models)
            if extra_models:
                raise ValueError(
                    "GeminiOpener thinking contains entries for unconfigured model id(s): "
                    f"{sorted(extra_models, key=repr)}")
            for model, entry in thinking.items():
                if (not isinstance(model, str) or not model.strip()
                        or model != model.strip()):
                    raise ValueError("GeminiOpener thinking keys must be nonempty string model ids")
                if not isinstance(entry, Mapping):
                    raise ValueError(
                        f"GeminiOpener thinking[{model!r}] must be a thinkingConfig mapping")
                config = dict(entry)
                unknown = set(config) - _THINKING_KEYS
                if unknown:
                    raise ValueError(
                        f"GeminiOpener thinking[{model!r}] has unsupported key(s): "
                        f"{sorted(unknown)}")
                level = config.get("thinkingLevel")
                if "thinkingLevel" in config and level not in _THINKING_LEVELS:
                    raise ValueError(
                        f"GeminiOpener thinking[{model!r}].thinkingLevel must be one of "
                        f"{sorted(_THINKING_LEVELS)}")
                if "thinkingBudget" in config:
                    budget = config["thinkingBudget"]
                    if type(budget) is not int or budget < 0:
                        raise ValueError(
                            f"GeminiOpener thinking[{model!r}].thinkingBudget must be a "
                            "nonnegative integer")
                resolved_thinking[model] = config
        # Copy both mapping levels so a caller cannot mutate the request shape while another
        # thread is inside generate().
        self.thinking: Mapping[str, Mapping[str, Any]] = resolved_thinking
        # model id -> the scope ("day" or "gone") it was permanently retired under earlier
        # THIS run. A dict rather than a set because a later all-exhausted stop must report
        # each retired model under the scope it actually failed with, not a hardcoded one --
        # a model retired for being out of per-day quota needs the "wait for midnight
        # Pacific" guidance, while one retired as 404-gone needs "fix opener.models" instead
        # (see _exhaustion_reason). Populated only by generate(); see its loop below.
        self._unavailable_models: dict[str, str] = {}
        # Self-contained lock (see the class docstring's THREAD SAFETY note): generate()
        # holds this for its entire body, so mutating self._unavailable_models and running
        # the model cascade are atomic per call regardless of what (if anything) an external
        # caller does. Previously this class relied entirely on OpenerService.maybe_opener's
        # own RLock for that safety -- correct today, but only by convention, and an
        # adversarial audit demonstrated a real double-spend race (two threads both burning a
        # billed API call on an already-retired model) the moment a call site bypasses that
        # external lock. RLock (not Lock): generate() calls other methods on self, and a
        # reentrant lock means a future refactor that has one of those helpers also take the
        # lock can't deadlock this instance against itself.
        self._lock = threading.RLock()

    def _image_parts(self, images: list[bytes]) -> list[dict[str, Any]]:
        """Base64-encode every image. This is the expensive step in building a request --
        full-resolution phone screenshots, then inflated ~33% by base64 -- so generate()
        computes it once per profile and reuses the result across every model tried during
        a capacity cascade instead of re-encoding the same screenshots per model.

        ``images`` is renamed from the old ``photos`` because it carries whichever request
        shape generate() built: her profile photos on the legacy frame shape, or -- on the
        item-crop shape -- ItemRequest.images, the numbered item crops followed by the
        unnumbered context crops. Both are encoded identically here; labelling and numbering
        happen later, in _assemble_parts and _text_part, so this method stays the single place
        that knows how to turn bytes into an inlineData part and knows nothing about what any
        of them mean.
        """
        if len(images) > _MAX_REQUEST_IMAGES:
            raise OpenerError(
                f"Gemini opener request has {len(images)} images; refusing more than "
                f"{_MAX_REQUEST_IMAGES}")
        if any(not isinstance(image, bytes) for image in images):
            raise OpenerError("Gemini opener request images must be bytes")
        raw_bytes = sum(len(image) for image in images)
        if raw_bytes > _MAX_RAW_IMAGE_BYTES:
            raise OpenerError(
                f"Gemini opener request has {raw_bytes} raw image bytes; refusing more than "
                f"{_MAX_RAW_IMAGE_BYTES} before base64 encoding")
        return [{
            "inlineData": {
                "mimeType": _image_media_type(image),
                "data": base64.standard_b64encode(image).decode("ascii"),
            },
        } for image in images]

    def _text_part(self, profile: Profile, style: str, retry_hint: str = "", *,
                   items: ItemRequest | None = None) -> dict[str, Any]:
        """Build the one part of the request that varies per attempt (see generate()'s "encode
        once, reuse across the cascade" note -- the image parts never depend on this).

        ``retry_hint`` defaults to "" (falsy) for an ordinary first attempt, in which case the
        text is byte-identical to what this produced before retries existed -- no stray
        delimiter, no trailing blank section, nothing for a diff to catch on the common path.
        When OpenerService is retrying a rejected attempt, it passes the specific reason that
        attempt was rejected for; that gets appended as its own clearly delimited block AFTER
        the profile content, so it is the most recent instruction the model reads before
        writing the corrected opener -- a corrected re-ask rather than an identical dice roll.

        ``items`` is the item-crop request shape (ops/OPENER-REDESIGN.md 5.2/5.7): when it is
        present the images are one crop per profile item rather than raw scroll frames, each
        already labelled by _assemble_parts, so the closing paragraph only states the counts
        and points at the field to answer in.

        HER NAME is rendered as its own labelled section, and ONLY on the item-crop shape --
        the legacy frame shapes carry her name in the pixels (every scroll frame has the
        sticky header in it), and cropping is what loses it (doc 5.2). Adding the section
        unconditionally would also move the legacy shapes' byte-identical text, which several
        tests pin precisely so this file's wire format cannot drift silently.

        ITEM NUMBERING IS 1-BASED IN EVERY BRANCH (doc 5.7), and the branch for "no numbered
        items were sent" says so explicitly rather than asking for an index into an empty list
        -- see ITEM_INDEX_ABSENT. The item-crop shape has no such branch at all: ItemRequest
        refuses to exist with zero items.
        """
        photo_count = len(profile.photos)
        if items is not None:
            # Every number below is derived from the ItemRequest rather than written out, for
            # the same reason the frame branches derive theirs from photo_count: the range the
            # model is told about can never disagree with what was actually sent. The
            # "so the image after ITEM 1 is item 1" clause restates the label convention with a
            # concrete number the model can check against the request it is holding -- and it
            # uses FIRST_ITEM_INDEX rather than a sample number like 3, which would name an
            # item that need not exist on a short profile.
            sentences = [
                f"The {items.item_count} numbered image(s) above are her profile photos, "
                f"numbered {FIRST_ITEM_INDEX} to {items.item_count}, each shown immediately "
                f"after its own ITEM label, so the image after ITEM {FIRST_ITEM_INDEX} is item "
                f"{FIRST_ITEM_INDEX}."
            ]
            if items.context_count:
                sentences.append(
                    f"The {items.context_count} image(s) labelled CONTEXT carry no number: use "
                    "them only as supporting context for a numbered item. Never pick one or "
                    "make one the opener's main premise; the selected item must remain its "
                    "clear main subject and reason for the Like.")
            if items.truncated:
                # Doc 5.7's truncation flag. Stated as a fact about OUR capture, not as a
                # deficiency in her profile, and immediately followed by "choose from them
                # anyway" so it cannot read as licence to decline or to write about the part
                # we did not see. Present ONLY when the capture really did hit its ceiling.
                sentences.append(
                    "Her profile was longer than we could read, so these are only the items "
                    "we saw. Choose from them anyway.")
            # Byte-identical to the frame branch's closing instruction on purpose: the field
            # being answered and what it means did not change with the payload, and keeping
            # one wording for it means the two shapes cannot drift into telling the model two
            # different things about the same field.
            sentences.append(
                "Set item_index to the number of the one your opener is clearly about and "
                "would feel natural directly under, because it explains why that item was "
                "liked. "
                "Write the opener now.")
            closing = " ".join(sentences)
        elif photo_count > 0:
            # 1-BASED, and every number in this sentence is derived from photo_count rather
            # than written out, so the range the model is told about can never disagree with
            # the number of images actually sent (ops/OPENER-REDESIGN.md 5.7).
            closing = (
                f"The {photo_count} image(s) above are her profile items, numbered "
                f"{FIRST_ITEM_INDEX} to {photo_count} in the order shown. Set item_index to the "
                "number of the one your opener is about. "
                "Write the opener now."
            )
        else:
            # photo_count == 0: the request carries her profile TEXT and no images at all, so
            # there is no numbered item for the model to choose and no honest number for it to
            # return. Saying so explicitly, and naming the out-of-band value, is the whole
            # point: the old copy rendered as "The 0 image(s) above are her
            # profile in scroll order (index 0 first)" and then asked for "the index of the one
            # your opener is about", which invites a confident `0` that is indistinguishable
            # from a real pick of the first item. Under the 1-based contract 0 is out of band
            # by construction (see ITEM_INDEX_ABSENT), so a consumer can tell "no item" from
            # "item 1" without guessing.
            closing = (
                "No images of her profile were captured, so there are no numbered items in "
                "this request and the profile text above is everything you have. Set "
                f"item_index to {ITEM_INDEX_ABSENT}, which means you could not pick a numbered "
                "item. Write the opener now."
            )
        # "" on every legacy shape, so their text stays byte-for-byte what it was (see the
        # HER NAME paragraph in this method's docstring). Stated as a bare labelled fact with
        # no instruction attached: doc 5.2 passes the name back because CROPPING LOST IT, i.e.
        # to restore what the frames already carried, not to introduce a new move. Telling the
        # model what to do with it here would be a voice change smuggled in as a payload
        # change, and Part A's wording rules are shipped and working (doc sections 2 and 3).
        name_block = (f"HER NAME:\n{items.name or _NAME_UNAVAILABLE}\n\n"
                      if items is not None else "")
        text = (
            f"STYLE GUIDE:\n{style}\n\n"
            f"{name_block}"
            f"HER PROFILE TEXT:\n{profile.text_blob() or '(none)'}\n\n"
            f"{closing}"
        )
        if retry_hint:
            text += (
                # The two lists below are deliberately kept apart. Everything under HARD
                # REJECTION is a rule _parse() actually enforces by raising, so it is what a
                # retry must fix to succeed at all. The style rules are real owner rules but
                # are NOT rejection causes (dashes, for instance, are silently laundered by
                # _sanitize rather than rejected). Presenting the two as one undifferentiated
                # "mandatory" checklist -- as this block first did -- spends the model's
                # attention on cosmetics when the reason it failed was structural.
                #
                # 2026-09-05: the HARD REJECTION list now names every opener-content cause
                # _parse() raises on EXCEPT ONE. The exception is REASON_UNDELIVERABLE_SEQUENCE
                # (a literal '%' against a following lowercase 's'), deliberately kept out of
                # all prompt copy -- see the comment at that guard for why a standing '%'
                # warning costs more than the collision does. Its own retry_hint explains it on
                # the rare occasion it fires. If you add a new raising guard to _parse(), add
                # its cause here too, or this list silently becomes a lie again.
                "\n\n=== RETRY: YOUR PREVIOUS ATTEMPT WAS REJECTED AND NOT SENT ===\n"
                f"Reason: {retry_hint}\n"
                "Fix exactly that. HARD REJECTION RULES, checked in code, which will reject "
                "you again if broken: the 'opener' field must be a non-empty STRING (never "
                "null, a number, or an empty/whitespace value); the opener must be at most "
                "TWO sentences; it must be the bare message itself with no preamble, label, or "
                "surrounding quotes; it must not open by naming and denying a negative reading "
                "of itself; it must not assume a match, date, or other shared future; it "
                "must not use an unconfirmed inferred location as the premise of a later "
                "statement or question; it "
                "must not infer or tease about self harm, death, injury, an accident, or danger "
                "unless her profile explicitly states the activity; and it must be plain ASCII "
                "letters and punctuation with no emoji. Also keep following the "
                "style guide above, and ground the opener in one "
                "concrete detail from her profile text or photos. You may name that detail when "
                "it is useful setup, but the final conversational point must add something "
                "beyond description. A guess is optional. Prefer a grounded observation or "
                "specific question over a forced inference. Never invent a hidden purpose, "
                "motive, circumstance, action, route, effort, goal, cause, or sequence merely "
                "to create a claim. If you do make a claim, use the least speculative natural "
                "interpretation and make its visible or stated basis immediately recognizable. "
                "The guessed conclusion itself must not be directly visible, readable, or stated "
                "anywhere in her profile; those facts are clues, not guesses. Keep every guess "
                "unconfirmed for the whole opener. Do not follow it with a statement or question "
                "that assumes it is correct; leave her room to confirm or correct it. "
                "For a location inferred from an image, that location guess is the only "
                "conversational move before she replies: end after it or ask only whether the "
                "location itself is right. Do not ask about an activity, reason, preference, "
                "feeling, experience, or consequence there. "
                "Clearly nonliteral playful hyperbole is allowed; an invented motive or event "
                "presented as literal fact is not. "
                "Every named visible detail must be necessary to the conversational move; cut "
                "it if the later point works without it. Ask one coherent thing at a time. Use "
                "'or' only for parallel, genuinely contrasting answers to one underlying "
                "question, never unrelated dimensions. Every pronoun, shorthand noun, and "
                "question subject must have one immediately obvious referent. Across two beats, "
                "any change of referent must be explicit and immediately clear. "
                "Make the most natural honest reply feel good to give: invite a preference, "
                "perspective, inspiration, or experience rather than self justification, and "
                "never offer a forced choice that makes her defend, diminish, or embarrass "
                "herself. Do not advertise the sender as the answer to one of her preferences, "
                "promise what he will do for her, or assume a match, date, relationship, or "
                "shared future. A direct invitation may propose a get-together only while "
                "leaving her acceptance open. "
                "Write the "
                "corrected opener now."
            )
        return {"text": text}

    @staticmethod
    def _assemble_parts(image_parts: list[dict[str, Any]], text_part: dict[str, Any], *,
                        items: ItemRequest | None = None) -> list[dict[str, Any]]:
        """Arrange the encoded image parts, the labels (when present), and the text part into
        the final ``contents[0].parts`` list Gemini receives, in the order the model reads them.

        ITEM-CROP SHAPE (``items`` given, ops/OPENER-REDESIGN.md 5.2/5.7): a preamble text part
        stating the label convention, then, for each image in ``ItemRequest.images`` order, a
        standalone label part immediately followed by that image, then the trailing text part.
        The numbered items come first and the context crops after them, which is what
        ``ItemRequest.images`` already guarantees -- this method only labels by position, it
        never reorders, because position IS the numbering.

        The per-image labels are the whole mechanism doc 5.2 asks for. Crops make "image k is
        item k" true by construction, but true-by-construction is a property of how WE built
        the request; the model still has to know it, and left to a paragraph at the end it would
        have to COUNT images to use it -- an inference step, on exactly the kind of enumeration
        this redesign exists to stop leaving to luck. A label adjacent to its image removes the
        step: there is nothing to count when each image says what it is.

        Without ``items`` (including no images at all), the legacy shape remains image parts
        followed by the text part.
        """
        if items is not None:
            if len(image_parts) != items.image_count:
                # A length mismatch would silently shift every label past the gap, so item 4's
                # label would sit on item 5's crop and the model's answer would be confidently
                # wrong with nothing to detect it downstream. Refuse instead.
                raise ValueError(
                    f"item-crop request has {items.image_count} image(s) but "
                    f"{len(image_parts)} encoded image part(s); the labels are positional, so "
                    "a mismatch would number the wrong crops")
            preamble = _ITEM_PREAMBLE + (_ITEM_PREAMBLE_CONTEXT if items.context_count else "")
            parts: list[dict[str, Any]] = [{"text": preamble}]
            for position, image_part in enumerate(image_parts):
                parts.append({"text": items.label_for(position)})
                parts.append(image_part)
            parts.append(text_part)
            return parts
        return [*image_parts, text_part]

    def _payload(self, profile: Profile, style: str, model: str, *,
                 image_parts: list[dict[str, Any]] | None = None,
                 retry_hint: str = "",
                 items: ItemRequest | None = None) -> dict[str, Any]:
        """Build one model's GenerateContent request. ``image_parts`` lets generate() pass in
        already-encoded images (the legacy frame shape's profile photos, or the item-crop
        shape's numbered crops plus context crops) so a cascade across N models doesn't
        re-encode the same screenshots N times; when omitted it is computed fresh instead --
        from ``items.images`` when ``items`` is given, else from ``profile.photos``. generate()
        always passes ``image_parts`` explicitly, so this fallback only matters for a caller
        that doesn't (there is none in this codebase today, but the method must still build a
        request that matches ``items`` rather than silently assume the legacy shape).
        ``retry_hint`` is forwarded to _text_part unchanged -- it must reach EVERY model tried
        in this attempt's cascade, because it describes what the previous attempt got wrong,
        which stays true no matter which model ends up serving the retry.

        ``items`` (ops/OPENER-REDESIGN.md 5.2/5.7) travels with ``image_parts``: it is what
        those encoded parts ARE, so the no-``image_parts`` fallback below re-encodes from
        ``items.images`` rather than from profile.photos when one is present. Encoding the
        scroll frames while telling the model it is looking at labelled item crops is precisely
        the kind of silent lie that fallback exists to refuse to tell."""
        if image_parts is not None:
            resolved_image_parts = list(image_parts)
            resolved_items = items
        elif items is not None:
            resolved_image_parts = self._image_parts(list(items.images))
            resolved_items = items
        else:
            resolved_image_parts = self._image_parts(profile.photos)
            resolved_items = None
        text_part = self._text_part(profile, style, retry_hint, items=resolved_items)
        parts = self._assemble_parts(resolved_image_parts, text_part, items=resolved_items)
        generation_config: dict[str, Any] = {
            "maxOutputTokens": self.max_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": _SCHEMA,
        }
        # The thinkingConfig dict is passed through verbatim rather than derived from the
        # model id: the field name differs by model family (thinkingLevel on the 3.x line,
        # thinkingBudget on 2.5), sending the wrong one is a 400, and guessing the family
        # from a version-number prefix is exactly the kind of thing that silently breaks
        # the day a new naming scheme ships. So it's declared explicitly per model id in
        # config rather than inferred here.
        thinking_config = self.thinking.get(model)
        if thinking_config is not None:
            generation_config["thinkingConfig"] = dict(thinking_config)
        return {
            "systemInstruction": {"parts": [{"text": _SYSTEM}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }

    @staticmethod
    def _request_size_bytes(parts: list[dict[str, Any]], system_text: str) -> int:
        """Approximate the wire size of one request against Gemini's 20MB inline-data cap:
        the base64 image payloads dominate, plus the system instruction and every text part
        (the per-request style guide/profile text block and item labels). generationConfig/schema JSON
        is a few hundred fixed bytes that don't scale with photo count, so it's left out of
        the estimate -- see _MAX_INLINE_REQUEST_BYTES for the headroom that covers it.

        Takes the fully assembled parts list rather than images and text separately, so the
        size calculation stays aligned with what generate() puts in the request.
        """
        total = len(system_text.encode("utf-8"))
        for part in parts:
            if "text" in part:
                total += len(part["text"].encode("utf-8"))
            else:
                total += len(part["inlineData"]["data"])
        return total

    def _fit_images_to_budget(self, images: list[bytes], image_parts: list[dict[str, Any]],
                              text_part: dict[str, Any], system_text: str, *,
                              items: ItemRequest | None = None) -> list[dict[str, Any]]:
        """Guarantee the request fits Gemini's 20MB inline-image cap, compressing only if
        it doesn't.

        Happy path (a handful of already-reasonable screenshots) does zero extra work and
        returns the original encoded parts untouched. Only a profile with many
        full-resolution phone screenshots pays the recompression cost, and it's never
        silent -- we print exactly what was done so a systematically oversized capture
        pipeline is visible rather than a mysteriously smaller/blurrier opener input.

        ``images`` is the same list generate() built for _image_parts. Sizing uses the fully
        assembled parts list, including item labels and the trailing text, not image parts alone.

        THIS METHOD MUST NEVER DROP AN IMAGE, and that requirement gets sharper under doc 5.2,
        not softer. It compresses every image or it raises; it has no branch that sends fewer.
        On the legacy frame shape a dropped image lost some of what the model could look at; on
        the item-crop shape it would RENUMBER every item after the gap, so item 5's label would
        sit on item 6's crop and the model would return a confidently wrong number that nothing
        downstream could detect. Crops are also ~4.4x smaller than the frames they came from
        (doc 5.2's addendum measured 8.53MB of crops against 37.15MB of frames on one real
        capture), so this path should now essentially never fire -- which is a reason to keep
        it honest, not a reason to relax it.
        """
        assembled = self._assemble_parts(image_parts, text_part, items=items)
        original_size = self._request_size_bytes(assembled, system_text)
        if original_size <= _MAX_INLINE_REQUEST_BYTES:
            return image_parts

        from PIL import Image  # lazy: PIL is a project dependency, imported this same lazy
                                # way in operation_love/vision/quality.py so modules that
                                # never hit this path don't pay the import cost.

        # Step 1 is quality-85 JPEG recompression with no resize -- often enough on its own
        # for PNG phone screenshots, which carry a lot of lossless overhead. If that's still
        # over budget, progressively shrink the longest side until the request fits.
        new_size = original_size
        fitted_parts: list[dict[str, Any]] = image_parts
        fitted_assembled: list[dict[str, Any]] = assembled
        for max_side in (None, 1568, 1280, 1024, 768):
            recompressed: list[bytes] = []
            for index, image in enumerate(images):
                try:
                    img = Image.open(io.BytesIO(image)).convert("RGB")
                    if max_side is not None and max(img.size) > max_side:
                        scale = max_side / max(img.size)
                        img = img.resize(
                            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                            Image.LANCZOS,
                        )
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                except Exception as exc:  # noqa: BLE001 -- PIL raises several distinct types for
                    # a truncated/corrupt image (UnidentifiedImageError, plain OSError, even
                    # struct.error deep in the decoder for a stream that dies mid-read), and this
                    # path only runs on an oversized profile, so it's easy for a flaky `adb
                    # screencap` capture to reach it. Left as a bare PIL exception, this escaped
                    # generate() uncaught: OpenerService can't classify it as GeminiAPIError, so
                    # it fell into the generic transient branch and printed "swiping without" --
                    # identically, forever, with no escalation (adversarial audit item B). Convert
                    # it to OpenerError instead: the established "this profile's own content is
                    # bad, skip just this profile" signal, naming which photo and how big it was
                    # so the operator can tell a systemic capture bug from one bad frame.
                    #
                    # An item-crop request gets its own naming here instead of "photo index N":
                    # none of its images is a profile photo in capture order at all, so "photo
                    # index 4" names something that does not exist and sends the operator
                    # hunting through her profile photos for a capture bug that is actually in
                    # the crop pipeline. ItemRequest reports "item 5's crop" or "context crop 2
                    # (unnumbered)" instead, which is a thing the operator can actually go and
                    # look at. The legacy frame shape has no such translation to do -- its
                    # images ARE her profile photos in capture order, so "photo index N"
                    # already names the right thing.
                    if items is not None:
                        label = items.describe_image(index)
                    else:
                        label = f"photo index {index}"
                    raise OpenerError(
                        f"Gemini opener: {label} ({len(image)} bytes) could not be "
                        f"decoded/recompressed while fitting the request to the inline size "
                        f"budget: {type(exc).__name__}: {exc}") from exc
                recompressed.append(buf.getvalue())
            fitted_parts = [{
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": base64.standard_b64encode(image).decode("ascii"),
                },
            } for image in recompressed]
            fitted_assembled = self._assemble_parts(fitted_parts, text_part, items=items)
            new_size = self._request_size_bytes(fitted_assembled, system_text)
            if new_size <= _MAX_INLINE_REQUEST_BYTES:
                print(f"Gemini opener: compressed {len(images)} image(s) to fit the "
                      f"inline request budget ({original_size} -> {new_size} bytes, cap "
                      f"{_MAX_INLINE_REQUEST_BYTES}).")
                return fitted_parts
        # BUG 4 (adversarial audit): this used to always say "reduce photo count or
        # resolution", blaming the photos unconditionally -- but the text part (style guide +
        # profile text + a retry_hint, which can itself be a sizable corrective block; see
        # _text_part) counts against the same budget and does NOT shrink here, only the
        # images do. A large retry_hint can push an otherwise-fine profile over budget with
        # the photos barely contributing, and telling the operator to trim photos in that case
        # is actively misleading. Report the actual composition instead of assuming.
        #
        # image_bytes/text_bytes are derived from the SAME fitted_assembled total (new_size)
        # rather than recomputed independently, so the breakdown can never drift out of sync
        # with the number actually being compared to the budget above.
        image_bytes = sum(len(part["inlineData"]["data"]) for part in fitted_parts)
        text_bytes = new_size - image_bytes
        if items is not None:
            composition_note = (
                f" ({items.item_count} numbered item crop(s) and {items.context_count} context "
                "crop(s), not scroll frames -- crops are already the small shape, so an "
                "oversized request here points at the capture or the crop geometry)"
            )
        else:
            composition_note = ""
        raise OpenerError(
            f"Gemini opener: request has {len(images)} image(s){composition_note}; the request "
            f"still totals {new_size} bytes encoded even at the smallest compression step "
            f"({image_bytes} bytes of images, {text_bytes} bytes of text -- style guide, "
            f"profile content, system instruction, and any item/context labels when present, "
            f"including any retry hint), over the "
            f"{_MAX_INLINE_REQUEST_BYTES} byte budget; refusing to silently drop images. "
            "Reduce image count or resolution upstream if images dominate the total, or "
            "shorten the profile text/retry hint if text does.")

    @staticmethod
    def _usage(response: Mapping[str, Any]) -> Usage:
        metadata = response.get("usageMetadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}

        # Convert REST's camelCase field names to the provider-neutral normalizer.
        # In particular, cached content is a subset of prompt tokens, so the normalizer
        # prevents charging that same token count at both input and cache-read rates.
        return Usage.from_gemini(SimpleNamespace(
            prompt_token_count=metadata.get("promptTokenCount", 0),
            candidates_token_count=metadata.get("candidatesTokenCount", 0),
            thoughts_token_count=metadata.get("thoughtsTokenCount", 0),
            cached_content_token_count=metadata.get("cachedContentTokenCount", 0),
        ))

    @staticmethod
    def _text(response: Mapping[str, Any]) -> str | None:
        candidates = response.get("candidates") or []
        if not isinstance(candidates, list):
            return None
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            content = candidate.get("content") or {}
            if not isinstance(content, Mapping):
                continue
            parts = content.get("parts") or []
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    return part["text"]
        return None

    @staticmethod
    def _finish_reason(response: Mapping[str, Any]) -> str | None:
        candidates = response.get("candidates") or []
        if not isinstance(candidates, list):
            return None
        for candidate in candidates:
            if isinstance(candidate, Mapping) and isinstance(candidate.get("finishReason"), str):
                return candidate["finishReason"]
        return None

    @staticmethod
    def _prompt_block_reason(response: Mapping[str, Any]) -> str | None:
        feedback = response.get("promptFeedback")
        if not isinstance(feedback, Mapping):
            return None
        reason = feedback.get("blockReason")
        if not isinstance(reason, str) or not reason or reason == "BLOCK_REASON_UNSPECIFIED":
            return None
        return reason

    @staticmethod
    def _thoughts_token_count(response: Mapping[str, Any]) -> int:
        metadata = response.get("usageMetadata")
        if not isinstance(metadata, Mapping):
            return 0
        try:
            return int(metadata.get("thoughtsTokenCount", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _parse(self, response: Mapping[str, Any], requested_model: str, *,
               index_space: str, numbered_item_count: int) -> OpenerResult:
        """Turn one billed provider response into an OpenerResult, or raise OpenerParseError.

        ``index_space`` and ``numbered_item_count`` describe the request this response is an
        answer to: which list the model was told to number, and how many numbered items were
        actually in it. Both are REQUIRED keywords rather than optional with a default,
        deliberately -- a default would be this method quietly assuming a request shape, and
        assuming a request shape is the exact bug the 2026-08-12 correction fixes. generate()
        derives both from the payload it just built, so they can never disagree with what was
        sent.

        They exist because the RANGE CHECK on `item_index` has nowhere else to live. `_parse`
        alone cannot do it (it never saw the request) and the driver cannot do it (it counts a
        different thing -- it compares against its own frame count, so on a 24-frame capture of
        9 items every value 1..23 looks in range). This method plus its two new arguments is the
        only place both facts are present at once.
        """
        usage = self._usage(response)
        # Price against the configured model id. Gemini's optional modelVersion can be an
        # opaque serving revision rather than a key in budget.pricing.
        model = requested_model
        text = self._text(response)
        finish_reason = self._finish_reason(response)
        if text is None:
            prompt_block_reason = self._prompt_block_reason(response)
            if prompt_block_reason is not None:
                raise OpenerParseError(
                    "Gemini blocked the opener prompt before generating content "
                    f"(promptFeedback.blockReason={prompt_block_reason})",
                    usage, model, reason_code=REASON_PROMPT_BLOCKED)
            if finish_reason == "MAX_TOKENS":
                # Thinking counts against maxOutputTokens for every model we use (all but
                # gemini-2.5-flash-lite default it on). When thoughts + candidates exceed
                # the budget, the candidate comes back with NO text part at all -- this is
                # the single most likely cause of "no text content", so name it explicitly
                # instead of leaving the operator to guess.
                thoughts = self._thoughts_token_count(response)
                raise OpenerParseError(
                    f"Gemini truncated the response before producing any text "
                    f"(finishReason=MAX_TOKENS): thinking alone used {thoughts} of "
                    f"{self.max_tokens} configured max_tokens, leaving no room for the opener "
                    f"JSON. Raise opener.max_tokens, or lower {model!r}'s opener.thinking level.",
                    usage, model, reason_code=REASON_MAX_TOKENS)
            if finish_reason in _BLOCK_FINISH_REASONS:
                raise OpenerParseError(
                    f"Gemini withheld the generated opener (finishReason={finish_reason})",
                    usage, model, reason_code=REASON_RESPONSE_BLOCKED)
            raise OpenerParseError(f"Gemini returned no text content (finishReason={finish_reason!r})",
                                   usage, model, reason_code=REASON_NO_TEXT)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # raw_opener=text (the whole raw response, not a per-field candidate): the JSON
            # never parsed at all, so there is no "opener" value to isolate -- see
            # OpenerParseError's own docstring for why the two no-candidate-string cases
            # (this one and missing_field just below) get the raw response text instead of
            # None, unlike no_text/max_tokens where there is no text of any kind.
            raise OpenerParseError(
                f"Gemini's opener output wasn't valid JSON (finishReason={finish_reason!r}): {exc}",
                usage, model, reason_code=REASON_BAD_JSON, raw_opener=text) from exc
        try:
            opener = data["opener"]
        except (KeyError, TypeError) as exc:
            raise OpenerParseError(f"{type(exc).__name__}: {exc}", usage, model,
                                   reason_code=REASON_MISSING_FIELD, raw_opener=text) from exc
        # _SCHEMA's "required": ["opener", ...] is a generation HINT sent to the model, not a
        # runtime guarantee the API enforces on the response -- Gemini can (and, per an
        # adversarial audit of this project, DOES in practice) still return {"opener": null}
        # or a non-string value despite "opener" being listed as required. Left unchecked,
        # _sanitize()'s str(text) would silently turn None into the literal string "None"
        # (truthy) or an int like 42 into "42" -- both pass the sentence-count guard below
        # and parse SUCCESSFULLY, and HingeDriver's guarded text path would then type that
        # literal text into the comment box and send it to a real person.
        # So the type is re-verified here, at the boundary, rather than trusted.
        if not isinstance(opener, str):
            # raw_opener is the repr of the non-string value itself (pre-sanitize -- this
            # check runs before _sanitize is ever called, since _sanitize's str(text) would
            # happily coerce a None/int/list into a plausible-looking string and hide exactly
            # the bug this guard exists to catch).
            raise OpenerParseError(
                f"Gemini's opener field was not a string: received {type(opener).__name__} "
                f"{_truncated_repr(opener)}",
                usage, model, reason_code=REASON_NOT_A_STRING, raw_opener=_truncated_repr(opener))
        # WHICH ITEM THE MODEL PICKED (ops/OPENER-REDESIGN.md 5.1/5.7). Read exactly as
        # defensively as the old referenced_index was, and clamped the same way -- but the
        # clamp floor now MEANS something: item numbering starts at FIRST_ITEM_INDEX, so
        # anything at or below ITEM_INDEX_ABSENT (a missing field, a null, a float, a string, a
        # negative) collapses to the single out-of-band value rather than to "the first item".
        # Under the old 0-based contract that same coercion silently produced a confident,
        # perfectly legal "index 0", which is exactly the failure this constant exists to make
        # impossible to mistake.
        raw_item_index = data.get("item_index", ITEM_INDEX_ABSENT) if isinstance(data, Mapping) else None
        # ONLY A VALUE THAT UNAMBIGUOUSLY NAMES AN ITEM IS ACCEPTED. `int()` is happy to turn
        # things into a plausible item number that never meant one, and under 1-based numbering
        # a plausible number is worse than a rejected one because it names a REAL card:
        #
        #   3.7   -> int() truncates to 3. But 3.7 does not mean item 3 any more than item 4;
        #            the truncation invents the answer. An integral float (3.0) is different --
        #            it names exactly one item -- so that one is accepted.
        #   True  -> int() gives 1, i.e. "item 1", because bool is a subclass of int in Python.
        #            A boolean is not a choice of item at all.
        #   "4"   -> accepted: a decimal string names exactly one item and nothing is invented.
        #            "4.7" is not accepted, because int() raises on it rather than truncating.
        #
        # None of these is reachable from a schema-conforming model ("type": "integer"), which
        # is why the old code's truncation was harmless under the 0-based contract and is worth
        # closing under this one: the whole point of ITEM_INDEX_ABSENT is that a value we cannot
        # trust must not come out looking like a value we can.
        if isinstance(raw_item_index, bool) or (
                isinstance(raw_item_index, float) and not raw_item_index.is_integer()):
            item_index = ITEM_INDEX_ABSENT
        else:
            try:
                item_index = max(ITEM_INDEX_ABSENT, int(raw_item_index))
            except (AttributeError, TypeError, ValueError):
                item_index = ITEM_INDEX_ABSENT
        # ABOVE the range: the model named an item that was never sent. Refused here rather
        # than passed on, because NOTHING downstream can catch it -- the driver's own bounds
        # check counts a different thing (its captured FRAMES, of which there are always more
        # than there are items), so an out-of-range item number sails straight through it and
        # lands on a real, wrong heart with full confidence. This method plus generate()'s two
        # arguments is the only place the request and the response are both in scope.
        #
        # It collapses to ITEM_INDEX_ABSENT rather than raising OpenerParseError, and that is a
        # deliberate choice between two loud options. Raising would throw away an opener that
        # may be perfectly good (the TEXT is not what went wrong), consume one of
        # max_attempts, and -- since five consecutive rejections stop the run -- let a model
        # quirk about numbering kill a session. ABSENT is already the project's one out-of-band
        # value and is already handled as "no usable item number, do not target anything"
        # everywhere it is consumed, so this reuses a guarantee that exists instead of adding a
        # second failure mode. The print is what makes it visible; the ABSENT contract is what
        # makes it safe.
        if item_index > numbered_item_count:
            print(f"Gemini opener: the model returned item_index={item_index} but only "
                  f"{numbered_item_count} numbered item(s) were sent in the "
                  f"{index_space} space, so it names an item that does not exist. Refusing "
                  f"it: recorded as ITEM_INDEX_ABSENT ({ITEM_INDEX_ABSENT}), which means no "
                  "item was chosen and nothing may be targeted from it. The opener text "
                  "itself is unaffected.")
            item_index = ITEM_INDEX_ABSENT
        elif item_index == ITEM_INDEX_ABSENT and raw_item_index not in (None, ITEM_INDEX_ABSENT):
            # The coercion above already did the right thing; this only says so out loud, for
            # anything that HAD a value and lost it: a negative, a non-integral float, a
            # boolean, a list, an unparseable string. A missing field and a literal 0 are
            # excluded because they are not slips -- 0 is what the prompt itself asks for when
            # the model cannot pick, and a line for it would fire on every honest refusal.
            print(f"Gemini opener: the model's item_index was not a usable item number "
                  f"({_truncated_repr(raw_item_index)}); recorded as ITEM_INDEX_ABSENT "
                  f"({ITEM_INDEX_ABSENT}), so nothing may be targeted from it. The opener "
                  "text itself is unaffected.")
        # `angle` is read exactly as defensively as item_index above, and for the same
        # reason: _SCHEMA's "required" list is a generation HINT, not something the API
        # enforces on the response (the isinstance check on `opener` above is the same lesson,
        # learned the hard way). Unlike `opener` this field is pure telemetry -- nothing reads
        # it to make a decision (ops/OPENER-REDESIGN.md 3.5) -- so a missing, null, or oddly
        # typed value must never cost a profile its opener. Anything unusable degrades to "",
        # which is exactly what an OpenerResult built without an angle carries anyway.
        try:
            angle = str(data.get("angle", "") or "").strip()
        except (AttributeError, TypeError, ValueError):
            angle = ""
        # `item_description` is read with the same defensiveness as `angle`, and for the same
        # reason -- but note the ASYMMETRY with `item_index` directly above, which is
        # deliberate. A missing description degrades to "" and costs a later cross-check its
        # input; a missing index has no safe default at all, which is why that one gets a named
        # out-of-band value instead of a plausible-looking number. Never gated on `advisory`:
        # see OpenerResult.item_description for why a mode-dependent schema breaks observe.
        try:
            item_description = str(data.get("item_description", "") or "").strip()
        except (AttributeError, TypeError, ValueError):
            item_description = ""
        sanitized = _strip_wrapping_quotes(_sanitize(opener))
        # A string that is empty, or becomes empty/whitespace-only once the dash-fold and
        # punctuation cleanup in _sanitize run, is just as unusable as a missing field --
        # sending nothing (or degrading silently to a bare like) is the same failure mode
        # this whole check exists to catch. An empty string happened to be falsy and degrade
        # safely downstream by luck alone; this makes it an explicit, named failure instead.
        if not sanitized.strip():
            # raw_opener=sanitized (post-sanitize -- the empty/whitespace string this check
            # actually tests), per OpenerParseError's own docstring rule. The pre-sanitize
            # original is still visible in the message itself (_truncated_repr(opener)) for a
            # human reading the retry_hint; raw_opener's job is only to record, unambiguously,
            # which stage the guard judged.
            raise OpenerParseError(
                f"Gemini's opener field was empty or whitespace only after sanitizing "
                f"(received {_truncated_repr(opener)})",
                usage, model, reason_code=REASON_EMPTY_AFTER_SANITIZE, raw_opener=sanitized)
        # WYSIWYG guard: _sanitize already folded everything fold_to_ascii knows how to fold
        # (accents, curly quotes, dashes, ligatures, ...), so anything undeliverable_chars
        # still finds here is something no ASCII substitute exists for -- almost always an
        # emoji. Sending it anyway would mean drivers.adb.Adb.text() either raises at the
        # device boundary AFTER this opener was already recorded as the sent text (a
        # BigQuery/hub row that no longer matches what actually reached the phone -- exactly
        # the drift this whole feature exists to prevent), or -- if some future call site
        # ever bypassed that boundary check -- silently drops the character again. Raising
        # HERE instead routes into OpenerService's existing retry loop (service.py, around
        # its OpenerParseError handling), which feeds this message back to the model as a
        # retry_hint: the model simply rewrites the opener without the offending
        # character(s), and the profile still gets an opener rather than being skipped.
        bad = undeliverable_chars(sanitized)
        if bad:
            names = ", ".join(describe_char(ch) for ch in bad)
            # raw_opener=sanitized: undeliverable_chars() is called directly on `sanitized`
            # above, so that is unambiguously the value this guard judged.
            raise OpenerParseError(
                f"Gemini's opener contains characters the phone keyboard cannot type: "
                f"{names}. Rewrite it using plain ASCII letters only.",
                usage, model, reason_code=REASON_UNDELIVERABLE_CHARS, raw_opener=sanitized)
        # Second WYSIWYG guard, same rationale as the one directly above, for the one
        # collision undeliverable_chars structurally cannot see: a literal '%' immediately
        # followed by a lowercase 's' is a two-character SEQUENCE that eats itself in adb's
        # own %s space escape (typography.undeliverable_sequences' docstring has the measured
        # round-trip table), even though '%' alone and 's' alone are both perfectly
        # typeable. This is rare and narrow on purpose -- it is NOT worth a line in _SYSTEM
        # telling the model to avoid it; a prompt-level warning about '%' risks scaring the
        # model off percent signs entirely (the exact unnatural "50 percent" rewrite the
        # owner rejected), for a collision that in practice almost never fires. The retry
        # hint below is the only place this is mentioned, and only on the rare occasion it's
        # actually needed.
        bad_seqs = undeliverable_sequences(sanitized)
        if bad_seqs:
            seqs = ", ".join(repr(seq) for seq in bad_seqs)
            raise OpenerParseError(
                f"Gemini's opener contains a sequence the phone keyboard cannot type as "
                f"written: {seqs}. A percent sign like '50%' is fine on its own, but a "
                f"literal '%' directly against a following lowercase 's' cannot be typed. "
                f"Reword that spot (e.g. spell out '50 percent' there, or rephrase so the "
                f"'%' isn't immediately followed by 's').",
                usage, model, reason_code=REASON_UNDELIVERABLE_SEQUENCE, raw_opener=sanitized)
        if _sentence_count(sanitized) > 2:
            # raw_opener=sanitized: _sentence_count() is likewise called on `sanitized`.
            raise OpenerParseError("Gemini returned an opener longer than the two-sentence maximum",
                                   usage, model, reason_code=REASON_TOO_MANY_SENTENCES,
                                   raw_opener=sanitized)
        # Deterministic (no second LLM call) guard against scaffolding/preamble text that
        # leaked INSIDE the opener string -- the JSON schema stops free text OUTSIDE the
        # field, but not a meta-clause like "Here's the response: ..." inside it. Named
        # matches feed back into service.py's retry loop as a retry_hint, so the model just
        # rewrites the opener without whatever it self-described this time.
        markers = _scaffolding_markers(sanitized)
        if markers:
            # raw_opener=sanitized: _scaffolding_markers() is called on `sanitized` above.
            raise OpenerParseError(
                f"Gemini's opener contained scaffolding text rather than the bare message "
                f"(matched: {'; '.join(markers)}). Return ONLY the message itself in the "
                f"opener field, with no preamble, no label, and no surrounding quotes "
                f"(received {_truncated_repr(opener)})",
                usage, model, reason_code=REASON_SCAFFOLDING, raw_opener=sanitized)
        # Positive-social-framing backstop. The prompt carries the semantic rule; this guard
        # catches only unmistakable opening disclaimers that name a negative interpretation and
        # deny it before getting to the substance. Regeneration removes the unwanted frame rather
        # than sending it merely because the rest of the message is usable.
        disclaimer_markers = _preemptive_disclaimer_markers(sanitized)
        if disclaimer_markers:
            raise OpenerParseError(
                "Gemini's opener introduced a negative social interpretation only to deny it "
                f"({'; '.join(disclaimer_markers)}). State the intended positive observation, "
                "question, or invitation directly. Remove the opening disclaimer and make the "
                "substantive thought stand on its own.",
                usage, model, reason_code=REASON_PREEMPTIVE_DISCLAIMER,
                raw_opener=sanitized)
        # High-precision backstop for the role-audition failure that produced "I will make
        # sure our first spot ..." on a real profile. The prompt owns the broad distinction
        # between a direct proposal and assumed reciprocity; this guard rejects only the
        # unmistakable lexical forms so a normal invitation is not mistaken for a promise.
        future_markers = _premature_shared_future_markers(sanitized)
        if future_markers:
            raise OpenerParseError(
                "Gemini's opener promised future performance or spoke as though an unaccepted "
                f"shared plan already exists ({'; '.join(future_markers)}). Do not audition "
                "for a role in her profile or promise what the sender will do for her. A "
                "direct invitation may propose a get-together, but must leave her acceptance "
                "open and must not call it an existing shared plan.",
                usage, model, reason_code=REASON_PREMATURE_SHARED_FUTURE,
                raw_opener=sanitized)
        # The general confirmation boundary is semantic and remains in the multimodal prompt.
        # This narrow case-aware backstop covers the location shape that has now recurred in
        # live Training: an explicit proper-name visual guess followed by a new conversational
        # beat which treats being at that place as established. A direct "am I close" style
        # confirmation remains legal; an activity or experience question is regenerated.
        location_followup_markers = _unconfirmed_location_followup_markers(sanitized)
        if location_followup_markers:
            raise OpenerParseError(
                "Gemini's opener used an unconfirmed location guess as the premise of a later "
                f"statement or question ({'; '.join(location_followup_markers)}). Keep the "
                "place unconfirmed until she replies. End after the guess, or ask only whether "
                "the location itself is right; do not ask what she did, felt, preferred, or "
                "experienced there.",
                usage, model, reason_code=REASON_UNCONFIRMED_LOCATION_FOLLOWUP,
                raw_opener=sanitized)
        # Safety backstop for a harmful, unsupported inference such as asking someone at a
        # bridge whether she "worked up the courage to jump". The model prompt already bars
        # this class of angle; this deterministic check makes a prompt miss retryable instead
        # of sending a potentially self-harm-adjacent message. It remains deliberately narrow:
        # a profile that explicitly mentions a normal sport must not be rejected merely for
        # containing a word that can also describe risk in another context.
        sensitive_markers = _sensitive_inference_markers(sanitized)
        if sensitive_markers:
            raise OpenerParseError(
                "Gemini's opener inferred or invited discussion of a sensitive dangerous "
                f"activity ({'; '.join(sensitive_markers)}). Do not speculate about jumping, "
                "courage around danger, self harm, or suicide from a scene. Use a different, "
                "grounded angle.",
                usage, model, reason_code=REASON_SENSITIVE_INFERENCE, raw_opener=sanitized)
        # `or ""` for the same reason as `angle` and `item_description` above: _SCHEMA's
        # "required" list is a generation hint the API does not enforce, and Gemini does return
        # nulls for these fields in practice. Without it a null degrades to the literal string
        # "None", which is then persisted verbatim into the `openers.referenced` column and
        # rendered to the operator as "about: None" -- a wrong value in an analytics column,
        # where "" honestly says the model claimed nothing.
        referenced = str(data.get("referenced", "") or "").strip()
        # REDUNDANCY MONITOR (ops/OPENER-REDESIGN.md 3.7), and note where it sits: AFTER every
        # guard that can reject, and it deliberately rejects nothing itself. An opener that
        # restates its own `referenced` note is the over-description bug this redesign targets,
        # but the measurement is a lower bound and no threshold has been calibrated yet (the
        # function's docstring has all four reasons), so it ships log only. There is
        # deliberately no REASON_* constant for it: it is not a rejection reason, and inventing
        # one would invite a future edit to raise on it before the data exists to justify a cut.
        #
        # Wrapped even though the function is pure and cannot realistically raise: this runs on
        # an opener that has already passed every real guard, so a bug in a MONITOR must never
        # be able to fail a request that was otherwise about to succeed. That is not a silent
        # degradation of the send path -- the opener is unaffected either way -- and the print
        # keeps it loud rather than invisible.
        try:
            redundancy_markers = _redundant_description_markers(sanitized, referenced)
        except Exception as exc:  # noqa: BLE001 -- see the paragraph above.
            print(f"Gemini opener: redundancy monitor failed with "
                  f"{type(exc).__name__}: {exc}; the opener itself is unaffected (this check "
                  "never rejects). This is a bug in _redundant_description_markers.")
            redundancy_markers = []
        if redundancy_markers:
            print(f"Gemini opener: redundancy monitor: this opener restates "
                  f"{len(redundancy_markers)} word(s) from its own `referenced` note "
                  f"({'; '.join(redundancy_markers)}). Logged for offline calibration only; "
                  "delivery has not been decided at this parsing stage.")
        return OpenerResult(
            opener=sanitized,
            referenced=referenced,
            usage=usage,
            model=model,
            item_index=item_index,
            # Stated, never inferred: this is the request shape generate() actually built, so a
            # consumer never has to guess which list `item_index` counts (see the INDEX_SPACE_*
            # constants for why guessing was the bug).
            index_space=index_space,
            angle=angle,
            item_description=item_description,
            redundancy_markers=redundancy_markers,
        )

    def generate(self, profile: Profile, style: str, retry_hint: str = "", *,
                 items: ItemRequest | None = None,
                 should_stop: Callable[[], bool] | None = None,
                 skip_models: frozenset[str] = frozenset()) -> OpenerResult:
        """Generate one opener for a profile, cascading across configured models on transient
        failures (see the cascade's comments below for what each failure mode does).

        The client-side absolute deadline this method used to enforce (an ``OpenerDeadlineExceeded``
        raised when Observe's advisory-preview path ran out of time) was removed on 2026-09-06
        along with that path: the service never passes a deadline into this call any more, so
        every request now runs for the full ``request_timeout_s`` regardless of caller.
        """
        # items is doc 5.2/5.7's item-crop request shape and is THE shape Part B is migrating
        # to: one cropped image per profile item, numbered by position and labelled adjacent to
        # its own image, then the unnumbered context crops, plus her name as text and the
        # capture's truncation flag. When present it REPLACES profile.photos as the model's view
        # of her -- the raw scroll frames are not sent at all, which is doc 5.7's "Not sent:
        # full screenshots, scroll frames, the anchor, endorsement blocks". profile is still
        # passed and still contributes its TEXT (profile.text_blob(), empty on Hinge but
        # potentially populated by a future browser driver); only its photos go unused,
        # deliberately and silently, because the caller that has crops also still has the
        # frames and needs them for ranking and embedding.
        #
        # REACHABLE FROM PRODUCTION AS OF 2026-08-12. The chain is: hinge._capture_current
        # enumerates the profile (doc 5.5's closed loop) and builds a
        # drivers.item_crops.ItemPayload; the crops, the context crops, her name and the
        # truncation flag ride on the Profile; worker._auto_loop calls
        # ItemRequest.from_profile(profile) and OpenerService.maybe_opener threads the result
        # here. What is STILL not wired is the other direction -- the returned item_index is in
        # INDEX_SPACE_MODEL_ITEMS, and nothing yet converts it into a tapped heart, so an AUTO
        # like on this shape hard-stops at worker.py's capture_order_index guard rather than
        # targeting. That is doc 5.6's workflow, and the stop is the intended behaviour until
        # it lands (never a fallback tap on item 1).
        #
        # retry_hint defaults to "" (falsy): an ordinary first attempt, no correction to make.
        # When OpenerService is re-asking after a rejected attempt, it passes the specific
        # reason here; _text_part appends it as a corrective instruction, and it must reach
        # EVERY model tried below, not just the first -- whichever model ends up serving the
        # retry, what the previous attempt got wrong is still what it got wrong.
        #
        # skip_models (item D of a later audit pass): generate() returns as soon as ANY model
        # answers HTTP 2xx, even when the response then fails to parse (see _parse). So a
        # parse-failure retry ordinarily restarts the cascade from the top and re-hits the
        # SAME (often scarcest-quota) model that just produced the bad response -- if that
        # model is systematically malformed, every retry attempt burns a request against it
        # while healthier models later in the cascade never even get tried. OpenerService
        # passes the model ids that already failed to parse FOR THIS PROFILE here so this
        # call's cascade steps over them, landing on the next configured model instead. A
        # skipped model is NOT recorded in self._unavailable_models: unlike a per-day 429 or a
        # 404, nothing here says the model is actually broken -- it produced a billed, well-
        # formed HTTP response, just not a usable opener -- so it stays fully eligible again on
        # the very next profile (or the next fresh call with no skip set).
        #
        # SAFETY VALVE: if no configured model is left ELIGIBLE -- i.e. every one of them is
        # either in skip_models or already retired into self._unavailable_models this run --
        # the skip set is ignored entirely rather than leaving this call with nothing to try.
        # It is deliberately eligibility and not merely skip-set membership: a model retired
        # by a per-day 429 or a 404 is skipped by the loop below just as surely, so counting
        # only the skip set would leave a healthy, never-tried model unused while the cascade
        # fell through on retired ones and reported the wrong reason. At that point a
        # stochastic re-ask of an already-failed model still beats returning with no opener at
        # all (the owner's rule is to keep trying, not to give up early), and OpenerService's
        # own max_attempts ceiling is what eventually stops the retries, not this method
        # refusing to pick a model. Note the valve only ever un-skips: a retired model stays
        # retired, since it is a separate check and this one does not touch it.
        #
        # The valve is asked THREE times, because eligibility is not a fact of the call -- the
        # cascade retires models as it runs, so any answer starts expiring the moment it is
        # given. (1) On entry, the cheap common case: the skip set was already hopeless before a
        # single request went out. (2) At each skip decision in the loop, against the models
        # still AHEAD of it -- which is all a forward-only check can see. (3) AFTER the cascade
        # falls through with nothing served, as a LAST-RESORT pass over the models this call
        # actually skip-honoured, in cascade order.
        #
        # Check 3 exists because checks 1 and 2 together still left one hole, and it is the one
        # that matters most: a model skipped EARLY, while the models behind it were all still
        # healthy, is never revisited by a check that can only look forward. models (a, b, c)
        # with skip={a} -- at a's slot b and c are eligible, so honouring the skip is correct on
        # the evidence available then; b and c then both 429 per-day, the loop ends, and a
        # healthy, never-requested `a` was passed over while the call raised
        # GeminiCapacityExhausted naming only b and c. service.py escalates that into a run stop
        # plus "wait until midnight Pacific" -- over a model that was one re-ask away from
        # serving. Nothing at the moment of the skip decision could have known; the fix is
        # therefore not a smarter decision but a second look once the outcome is known.
        #
        # The pass is scoped to exactly the models that were SKIP-HONOURED and are still not
        # retired, tracked as the loop makes each decision rather than recomputed from set
        # membership afterwards. That scoping is what makes it free of both hazards: those
        # models were never requested this call, so re-asking them re-bills nothing and cannot
        # replay a transient failure -- a model that 5xx'd or hit a per-minute 429 this call was
        # requested OUTSIDE the skip set and is deliberately NOT in this pass, because for it a
        # second request in the same call is a second bill for the same known-bad condition. A
        # model retired this run (per-day 429, 404, thinking-config 400) is excluded too: as
        # everywhere else here, the valve only ever un-skips, it never un-retires. If the pass
        # serves, the call returns normally; if its models fail or retire in turn, the same
        # GeminiCapacityExhausted is raised, now naming them with the scope they really failed
        # under.
        #
        # should_stop is the caller's cheap, non-blocking "should I keep going?" check (in
        # practice, worker.py's threading.Event.is_set for the run's stop flag). BUG 1 (an
        # adversarial audit): pre-fix, nothing on this path ever consulted the stop signal, so
        # a Stop click during a live cascade (up to max_attempts x len(models) x
        # request_timeout_s -- 5 x 7 x 90s = 3150s, ~52 minutes, against the shipped config)
        # was silently ignored: the worker thread ran the full retry/cascade to completion
        # anyway, kept holding OpenerService's shared lock (blocking every other worker), could
        # write to the store AFTER the supervisor's shutdown had already closed it, and blew
        # straight past supervisor.py's own _WORKER_JOIN_TIMEOUT_S, misreporting a perfectly
        # healthy retry loop as a wedged worker. Checked at the TOP of every model iteration,
        # BEFORE that model's request is issued, so the worst case is now bounded by ONE
        # already-in-flight HTTP request (request_timeout_s), not the rest of the cascade.
        #
        # Held for the ENTIRE call, not just the self._unavailable_models mutations, so a
        # second thread calling generate() concurrently is fully serialized behind this one
        # rather than able to interleave and both hit the same about-to-be-retired model with
        # a real, billed request (see the class docstring's THREAD SAFETY note).
        with self._lock:
            if should_stop is not None and should_stop():
                # Image base64/recompression is the expensive part of request construction.
                # A run already known to be stopping must not spend that CPU/memory before the
                # post-preparation per-model stop check gets a chance to run.
                raise OpenerAborted(
                    "Opener cascade aborted before preparing images: the run is stopping "
                    "(should_stop signaled), not a provider failure")
            # Encode once, reuse across every model tried in this call's cascade -- base64 and
            # (when a profile needs it) recompression are the expensive parts of building a
            # request, and neither depends on which model ends up serving it (nor on
            # retry_hint, which only ever varies the text part).
            #
            if items is not None:
                # profile.photos is deliberately NOT included: on this shape the crops are the
                # model's whole view of her (doc 5.7's "Not sent: ... scroll frames"), and
                # appending the frames would re-introduce the duplication bias doc 5.2 removes
                # -- a card straddling a scroll seam appearing three times reads as salience to
                # a model that is now CHOOSING among items.
                images = list(items.images)
            else:
                images = list(profile.photos)
            # WHAT `item_index` WILL MEAN IN THE ANSWER, derived from the payload actually being
            # built rather than assumed anywhere downstream (see the INDEX_SPACE_* constants).
            # Both travel to _parse, which needs the count for its range check and the space for
            # the result it returns.
            #
            index_space = (INDEX_SPACE_MODEL_ITEMS if items is not None
                           else INDEX_SPACE_PROFILE_PHOTOS)
            numbered_item_count = (items.item_count if items is not None
                                   else len(profile.photos))
            image_parts = self._image_parts(images)
            text_part = self._text_part(profile, style, retry_hint, items=items)
            system_text = _SYSTEM
            image_parts = self._fit_images_to_budget(images, image_parts, text_part, system_text,
                                                      items=items)
            def announce_safety_valve(trigger: str) -> None:
                # ONE message for BOTH valve sites (entry-time below, point-of-use in the loop),
                # so the log reads identically however the valve engages and `trigger` is the
                # only part that says which. The operator reading this needs to know that a
                # model is about to be re-asked despite the caller having asked to avoid it.
                print("Gemini opener: the skip set would leave this call nothing to try "
                      f"({trigger}) -- ignoring it and re-asking an already-failed model "
                      "instead of reporting every configured model exhausted. Models retired "
                      "earlier in this run stay retired.")

            # SAFETY VALVE (see this method's skip_models docstring paragraph above): if the
            # caller's skip set would leave literally nothing eligible, ignore it entirely
            # rather than raising GeminiCapacityExhausted without ever trying a single model.
            # "Eligible" must subtract self._unavailable_models too, not just the skip set: the
            # loop below skips a model retired earlier this run for exactly as long as the run
            # lasts, so late in a free-tier day "not skipped" and "actually tryable" are
            # different sets. Testing only skip-set membership let one skipped-but-healthy model
            # sit unused while the cascade fell through on the retired ones and raised "every
            # configured model has exhausted its per-day quota" -- naming models the caller never
            # asked to avoid, and telling the operator to wait until midnight Pacific over a
            # model that was one stochastic re-ask away from serving the call.
            #
            # THIS IS ONLY THE FIRST LOOK, and it is NOT the whole valve. It answers "is anything
            # eligible RIGHT NOW", and that answer starts expiring the moment the loop begins:
            # the cascade below RETIRES models into self._unavailable_models as it runs (per-day
            # 429, 404 NOT_FOUND, thinking-config 400), so a set that was eligible here can be
            # empty by the time a skipped model's slot comes up. Holding self._lock does not make
            # this snapshot safe -- the lock excludes OTHER THREADS, not this call's own
            # retirements, which are exactly what moves the set. The question is therefore
            # re-asked AT THE POINT OF USE, in the loop's skip branch below, and once more in
            # the last-resort pass after the loop (see the docstring's skip_models paragraph:
            # the point-of-use check can only look forward, so a model skipped BEFORE the
            # models that then retire needs the after-the-fact look to be reconsidered).
            eligible = [m for m in self.models
                        if m not in skip_models and m not in self._unavailable_models]
            if skip_models and not eligible:
                announce_safety_valve("no configured model was eligible when the call began")
            effective_skip_models = skip_models if (skip_models and eligible) else frozenset()
            # Why each model declined to serve THIS call, so that if the whole cascade falls
            # through we can report an accurate stop reason instead of a generic one. The hub
            # shows this verbatim, and "wait until midnight Pacific" vs "retry in a minute" are
            # very different instructions to give the operator.
            scopes: dict[str, str] = {}

            def attempt(model: str) -> "OpenerResult | None":
                """ONE model's request and everything that can come back from it.

                Returns the parsed OpenerResult when this model SERVED the call, and None when
                it declined in a way the cascade is meant to survive (transport failure,
                per-minute/unclassified 429, per-day 429, 404 NOT_FOUND, provider 5xx, a
                thinking-config 400) -- None is the old loop body's `continue`, i.e. "move on to
                the next model". Everything that must NOT be survived still leaves by raising:
                any other non-2xx (`raise error`) and OpenerParseError from _parse.

                It is a closure rather than inline code because it has TWO callers -- the main
                cascade below and the last-resort pass after it (see this method's skip_models
                docstring paragraph). A reconsidered model must be requested, classified,
                retired and accounted for by exactly the same code that would have handled it in
                its own cascade slot; a second copy of this handling would be free to drift, and
                the branch that decides whether a 429 retires a model for the rest of the run is
                the last place in this file that can afford a near-copy.

                Writes `scopes` in place (never rebinds it), so a model that fails in the
                last-resort pass reports the scope it really failed under in the exhaustion
                reason, exactly as it would have from the loop.
                """
                payload = self._payload(profile, style, model, image_parts=image_parts,
                                        retry_hint=retry_hint, items=items)
                url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                       f"{quote(model, safe='-_.')}:generateContent")
                try:
                    code, response = self.transport(
                        url, payload,
                        {"Content-Type": "application/json", "X-goog-api-key": self.api_key},
                        self.request_timeout_s,
                    )
                except (OSError, HTTPException) as exc:
                    # TRANSPORT-level failure -- this is a layer BELOW the HTTP status-code
                    # cascade above: _stdlib_gemini_transport only catches urllib.error.
                    # HTTPError (a successful-at-the-socket-layer response that merely carries
                    # a non-2xx status), converting it into a (code, body) return value. A
                    # socket.timeout, a urllib.error.URLError (connection reset, DNS failure,
                    # refused connection, ...), or an http.client protocol failure such as
                    # IncompleteRead never reaches that handling at all -- it propagates
                    # straight out of self.transport(...) to here. EMPIRICALLY OBSERVED: a
                    # live run against the real API timed out
                    # mid-request on the FIRST configured model and, pre-fix, that abandoned
                    # the entire cascade -- six other healthy, configured models were never
                    # tried and the profile got no opener. Treat it exactly like a provider
                    # 5xx ("busy"): a dropped connection says something about THIS request,
                    # not about whether this model (or any other) would answer the next one,
                    # so cascade to the next configured model for this profile only and do
                    # NOT retire it -- it stays first in line on the next profile.
                    #
                    # Caught as OSError plus HTTPException specifically, NOT bare Exception.
                    # OSError covers timeout, reset, DNS, and refused connections;
                    # HTTPException covers standard-library HTTP framing failures that are not
                    # OSErrors (notably a truncated response raising IncompleteRead). A bare
                    # `except Exception` here would be wrong: it would also swallow a TypeError or
                    # AttributeError raised by a broken transport implementation -- a
                    # programming bug in this code or an injected transport, not a flaky
                    # network -- and silently retry that bug across all 7 configured models
                    # instead of letting it surface immediately as the real error it is.
                    # Its own scope rather than reusing "busy": both are transient and both
                    # cascade identically, but the stop reason is operator-facing guidance,
                    # and reporting a connection timeout as a "provider 5xx" would point at
                    # Google when the problem may well be the local network.
                    scopes[model] = "transport"
                    print(f"Gemini opener: {model} failed at the transport level "
                          f"({type(exc).__name__}: {exc}); NOT blacklisting -- trying the "
                          "next configured model for this profile only (this model will be "
                          "retried first on the next profile).")
                    return None
                if not 200 <= int(code) < 300:
                    error = _gemini_error(int(code), response)
                    # The HTTP 429 status CODE is the reliable capacity signal -- Gemini's own
                    # machine-readable `error.status` ("RESOURCE_EXHAUSTED") is best-effort
                    # enrichment on top of it, not a precondition for treating the response as
                    # capacity. An infra-level rate limiter or proxy sitting in front of the API
                    # can (and, per an adversarial audit of this project, DOES) return a bare 429
                    # with an empty or differently-spelled status. Gating this branch on an exact
                    # status match used to let that response fall through every branch below and
                    # hit `raise error`, abandoning the whole cascade -- including every other
                    # healthy, configured model -- over what is still, unambiguously, a capacity
                    # response. So any 429 enters this branch; _classify_quota_exhaustion below
                    # still does the day/minute/unknown split from whatever quota detail (if any)
                    # the body carries, which decides HOW to react (blacklist vs. not), not
                    # whether to.
                    if error.http_code == 429:
                        scope = _classify_quota_exhaustion(error)
                        scopes[model] = scope
                        if scope == "day":
                            # RPD (requests-per-day) resets only at midnight Pacific, so this
                            # model genuinely cannot serve the rest of THIS run.
                            self._unavailable_models[model] = "day"
                            print(f"Gemini opener: {model} hit its per-day quota (resets at "
                                  "midnight Pacific); blacklisting it for the rest of this run "
                                  "and trying the next configured model.")
                        else:
                            # Per-minute caps are transient (free-tier RPM can be as low as 5)
                            # and clear within a minute, so do NOT blacklist -- just cascade to
                            # the next model for this profile; the preferred model is retried
                            # first on the next profile. "unknown" (no parseable quota details)
                            # gets the same non-blacklisting treatment: wrongly retiring the
                            # best model for a whole run on one ambiguous 429 is far more
                            # costly than one wasted retry per profile, and every model 429ing
                            # within a single call still raises GeminiCapacityExhausted below
                            # regardless of classification, so the "stop automation when
                            # everything is used up" guarantee holds either way.
                            kind = "per-minute" if scope == "minute" else "unclassified"
                            print(f"Gemini opener: {model} hit a {kind} 429; NOT blacklisting -- "
                                  "trying the next configured model for this profile only (this "
                                  "model will be retried first on the next profile).")
                        return None
                    if error.http_code == 404:
                        # NOT_FOUND: EMPIRICALLY MEASURED against the real API -- ListModels can
                        # list a model id with generateContent in its supportedGenerationMethods
                        # that then 404s the instant generateContent is actually called for this
                        # account (this happened for gemini-2.5-flash and gemini-2.5-flash-lite,
                        # both retired from opener.models entirely as a result; see
                        # preflight()'s docstring). A 404 is a property of THAT ONE model id, not
                        # of the request or the other configured models, so unlike an
                        # auth/permission/malformed-request error it must not be raised straight
                        # to the caller -- doing so would silently kill every other model in the
                        # cascade over one retired id. Drop just this model, permanently (a
                        # retired model does not come back mid-run), and try the next configured
                        # model, exactly like a per-day 429.
                        scopes[model] = "gone"
                        self._unavailable_models[model] = "gone"
                        print(f"Gemini opener: {model} returned 404 NOT_FOUND ({error.message}); "
                              "this model id is retired or unavailable to this account and will "
                              "not come back mid-run -- dropping it from the cascade for the "
                              "rest of this run and trying the next configured model.")
                        return None
                    if error.http_code >= 500:
                        # Provider-side failure (503 UNAVAILABLE "this model is currently
                        # experiencing high demand" is by far the common one; 500/502/504 behave
                        # the same). OBSERVED LIVE: gemini-3.6-flash returned 503 while every
                        # other configured model was serving normally, which proves this is a
                        # per-MODEL condition, not a provider-wide one. Raising it here would
                        # hand the caller a "transient error" for the whole service and the
                        # worker would send a bare like with no opener -- while six healthy
                        # models sat unused. So cascade to the next model immediately, and do
                        # NOT retire this one: high demand clears on its own, so it stays first
                        # in line for the next profile, exactly like a per-minute 429.
                        scopes[model] = "busy"
                        print(f"Gemini opener: {model} returned HTTP {error.http_code} "
                              f"{error.status or 'server error'}; NOT blacklisting -- trying the "
                              "next configured model for this profile only (this model will be "
                              "retried first on the next profile).")
                        return None
                    if error.http_code == 400 and _is_thinking_config_rejection(error.message):
                        # NARROW EXCEPTION to the "every other 4xx is a property of the request,
                        # not the model" rule stated in this class's docstring. MEASURED, live,
                        # 2026-08-13 (see _is_thinking_config_rejection): gemini-3.7-flash 400s
                        # on {"thinkingLevel": "minimal"} with "Thinking level MINIMAL is not
                        # supported for this model. Please retry with other thinking level."
                        # while every OTHER configured model accepts the identical
                        # generationConfig.thinkingConfig without complaint. That makes this 400
                        # a property of THAT ONE model id's declared capability, exactly like a
                        # 404 -- not of the request or the credentials -- so it must not be
                        # allowed to abort the whole cascade over one model's mis-specified
                        # opener.thinking entry. Drop just this model, permanently: unlike a
                        # per-minute 429 or a 5xx, a capability rejection does not clear with
                        # time or a retry, it only clears with an opener.thinking edit, so it is
                        # retired for the rest of THIS run exactly like a 404 or a per-day 429.
                        #
                        # Every 400 that reaches this point WITHOUT matching still falls through
                        # to `raise error` unchanged, immediately below -- in particular an
                        # invalid-API-key 400 ("API key not valid...") and a generic malformed-
                        # request 400 never contain any of _THINKING_REJECTION_TOKENS, so
                        # OpenerService's invalid-key latch (_is_invalid_gemini_api_key) and its
                        # ordinary-400 consecutive-latch (_BAD_REQUEST_LATCH_THRESHOLD) keep
                        # firing exactly as they did before this branch existed.
                        scopes[model] = "thinking"
                        self._unavailable_models[model] = "thinking"
                        print(f"Gemini opener: {model} returned HTTP 400 rejecting its "
                              f"configured thinking level or budget ({error.message}); this is "
                              "a per-model capability limit, not a property of the request, and "
                              "will not change mid-run -- dropping it from the cascade for the "
                              "rest of this run and trying the next configured model. Fix "
                              "opener.thinking for this model id.")
                        return None
                    raise error
                if not isinstance(response, Mapping):
                    raise GeminiAPIError(int(code), None, "malformed success response")
                return self._parse(response, model, index_space=index_space,
                                   numbered_item_count=numbered_item_count)

            # Every model this call ACTUALLY skip-honoured, in cascade order, recorded as each
            # decision is made rather than recomputed from skip_models afterwards. The two sets
            # are not the same: the point-of-use valve can drop the skip set mid-cascade, so a
            # model in skip_models may well have been requested normally, and re-deriving
            # membership after the loop would hand the last-resort pass below a model that was
            # already tried -- a second bill for a condition this call has already seen. Only a
            # model that reached the `else` branch below was passed over without a request.
            skip_honored: list[str] = []
            for position, model in enumerate(self.models):
                # Check EVERY model immediately before it can issue a request, including model
                # zero. The earlier pre-image check prevents useless encoding when Stop already
                # won; this second check closes the race where it wins DURING image preparation.
                # It remains before both skip paths so a stop does not walk a stale cascade.
                if should_stop is not None and should_stop():
                    # Checked before even the "already retired" skip below, so a stop signaled
                    # right after this model's slot comes up never issues a request for it --
                    # see this method's should_stop docstring paragraph for the full rationale.
                    raise OpenerAborted(
                        f"Opener cascade aborted before requesting {model!r}: the run is "
                        "stopping (should_stop signaled), not a provider failure")
                if model in effective_skip_models:
                    # POINT OF USE for the safety valve above, and the half the entry-time
                    # snapshot cannot cover. Honouring a skip is only ever safe while something
                    # ELSE can still serve this call, and models retire mid-cascade -- so ask
                    # again, against what is actually left ahead: the models AFTER this one that
                    # are neither skipped nor already retired. MEASURED by an adversarial review
                    # against the live API: models (b, c, a) with skip={a}, b and c both 429ing
                    # per-day within this very call, left a healthy, never-tried `a` unused and
                    # raised "every configured model has exhausted its per-day quota" naming only
                    # b and c -- service.py turns that into a run stop plus "wait until midnight
                    # Pacific" (see its GeminiCapacityExhausted branch), over a model that was one
                    # re-ask away from serving.
                    #
                    # This check is FORWARD-ONLY and that is not a limitation to work around
                    # here: at this instant nothing knows whether the models ahead are about to
                    # die, so honouring the skip while they are still eligible is the right
                    # decision on the evidence available. The complementary case -- (a, b, c)
                    # with skip={a}, where the models that retire come AFTER the skipped one --
                    # is unanswerable at this point and is closed instead by the last-resort
                    # pass below, which looks once the outcome is actually known. That is why
                    # the `else` branch records the model rather than merely announcing it.
                    if not any(m not in effective_skip_models
                               and m not in self._unavailable_models
                               for m in self.models[position + 1:]):
                        announce_safety_valve(
                            f"no model after {model} is still eligible")
                        # Stop honouring the skip set from here on. ONLY the skip set is dropped:
                        # the retired check immediately below is untouched, so a model retired
                        # this run is still never re-tried -- the valve only ever un-skips models
                        # that are healthy as far as anything here knows. Falls through (no
                        # `continue`) so THIS model is the one it un-skips.
                        effective_skip_models = frozenset()
                    else:
                        # Already produced an unusable (but well-formed, billed) response for
                        # THIS profile earlier in the same retry sequence -- not retired (see the
                        # skip_models docstring paragraph above), just deprioritized for this one
                        # call, so it is not added to `scopes` either: it was never actually
                        # tried this call, so it has nothing to report if the cascade falls
                        # through. Deprioritized, not discarded: it is recorded here so the
                        # last-resort pass can come back to it if the cascade ends up serving
                        # nothing, which is the only situation where re-asking it beats
                        # reporting models the caller never asked to avoid.
                        skip_honored.append(model)
                        print(f"Gemini opener: skipping {model} for this retry -- it already "
                              "produced an unusable response for this profile; trying the next "
                              "configured model instead.")
                        continue
                if model in self._unavailable_models:
                    # Retired earlier THIS run -- either a per-day 429 or a 404 NOT_FOUND (see
                    # self._unavailable_models). Report the scope it actually failed under; a
                    # later all-exhausted stop needs the right guidance for each model, not a
                    # hardcoded "day" for one that was really 404-gone.
                    scopes[model] = self._unavailable_models[model]
                    continue
                # The request itself, plus every failure this cascade is designed to
                # survive. None means "this model declined, try the next one".
                served = attempt(model)
                if served is not None:
                    return served

            # LAST-RESORT PASS -- the third and final look of the safety valve, and the one that
            # closes the forward-only check's blind spot (see this method's skip_models
            # docstring paragraph). The cascade is over and it served nothing, so the very next
            # statement would raise GeminiCapacityExhausted and service.py would turn that into
            # a run stop. Before paying that price, re-ask the models this call passed over
            # WITHOUT EVER REQUESTING THEM.
            #
            # The scope of this pass is exactly `skip_honored` minus anything retired, and each
            # half of that is load-bearing:
            #   - skip-honoured only. A model that failed this call was requested OUTSIDE the
            #     skip set, so it has already been billed once for a condition we have already
            #     observed; re-asking it here would be a second bill on the same known-bad
            #     model in the same call, and a per-minute 429 or a 5xx does not become more
            #     likely to serve seconds later. Those models are handled by the caller
            #     retrying, not by this pass.
            #   - not retired. A per-day 429, a 404 or a thinking-config 400 retires a model for
            #     the rest of the run, and the valve has only ever un-skipped, never un-retired.
            #     A skip-honoured model that was already retired on an EARLIER call lands here,
            #     and it contributes its retirement scope to the exhaustion reason below rather
            #     than a request.
            # Cascade order is preserved because skip_honored was appended in loop order, so the
            # reconsidered models are still tried most-preferred first.
            reconsidered = [m for m in skip_honored if m not in self._unavailable_models]
            if reconsidered:
                announce_safety_valve(
                    "the cascade served nothing and never tried " + ", ".join(reconsidered))
                for model in reconsidered:
                    # The same pre-request obligation the loop above owes, for the same reason:
                    # Stop must be honoured before any further billed request goes out. The skip
                    # check is deliberately absent -- being skip-honoured is this pass's entry
                    # criterion -- and the retirement check already happened when `reconsidered`
                    # was built.
                    if should_stop is not None and should_stop():
                        raise OpenerAborted(
                            f"Opener cascade aborted before reconsidering {model!r}: the run is "
                            "stopping (should_stop signaled), not a provider failure")
                    served = attempt(model)
                    if served is not None:
                        return served
            # A skip-honoured model that this pass could not even request is still a real reason
            # the call has no opener, so it must appear in the stop reason under the scope it
            # actually failed under. Without this the message could claim "every configured
            # Gemini model has exhausted its per-day quota" while naming a strict subset of them,
            # silently omitting the one the caller had asked to skip.
            for model in skip_honored:
                if model in self._unavailable_models and model not in scopes:
                    scopes[model] = self._unavailable_models[model]
            raise GeminiCapacityExhausted(_exhaustion_reason(scopes))

    def preflight(self) -> None:
        """Verify every configured model exists and supports generateContent before a run
        starts, so a typo'd model id or an invalid key fails fast at startup instead of
        mid-run after photos have already been captured for a live profile.

        GETs the ListModels endpoint through the injected transport (never real network in
        tests) and follows nextPageToken pagination to see the full catalog.

        NECESSARY BUT NOT SUFFICIENT: passing this check does not guarantee a model will
        actually serve a request. EMPIRICALLY MEASURED against the real API: ListModels
        listed both gemini-2.5-flash and gemini-2.5-flash-lite with "generateContent" in
        their supportedGenerationMethods, and generateContent still 404d NOT_FOUND for both
        on every call ("This model models/<id> is no longer available to new users").
        ListModels' catalog can lag behind which models an account can actually call. That
        is exactly why generate()'s runtime cascade must survive a 404 by retiring just that
        one model rather than treating it as fatal (see its 404 handling below) -- this
        preflight check narrows typos and dead keys, it is not a guarantee every listed
        model will work once a run is live.
        """
        seen: dict[str, list[str]] = {}
        page_token: str | None = None
        seen_page_tokens: set[str] = set()
        for _page_number in range(1, _MAX_PREFLIGHT_PAGES + 1):
            url = (_GEMINI_MODELS_LIST_URL if page_token is None else
                   f"{_GEMINI_MODELS_LIST_URL}?pageToken={quote(page_token, safe='')}")
            code, response = self.transport(
                url, None, {"X-goog-api-key": self.api_key}, self.request_timeout_s,
                method="GET",
            )
            if not 200 <= int(code) < 300:
                error = _gemini_error(int(code), response)
                # An invalid key surfaces here as HTTP 400 INVALID_ARGUMENT with message
                # "API key not valid...". Translate that into plain language instead of
                # leaking Gemini's raw wording, and never echo the key itself.
                if error.http_code == 400 and "api key not valid" in error.message.lower():
                    raise RuntimeError("GEMINI_API_KEY is not valid (rejected by Gemini's "
                                       "ListModels endpoint)")
                raise RuntimeError(
                    f"Gemini preflight failed: HTTP {error.http_code} {error.status}: {error.message}")
            if not isinstance(response, Mapping):
                raise RuntimeError("Gemini preflight failed: malformed ListModels response")
            for entry in response.get("models") or []:
                if not isinstance(entry, Mapping):
                    continue
                name = entry.get("name")
                if not isinstance(name, str) or not name:
                    continue
                model_id = name.split("/", 1)[1] if "/" in name else name
                methods = entry.get("supportedGenerationMethods")
                seen[model_id] = [str(m) for m in methods] if isinstance(methods, list) else []
            next_token = response.get("nextPageToken")
            if next_token in (None, ""):
                break
            if not isinstance(next_token, str) or not next_token.strip():
                raise RuntimeError(
                    "Gemini preflight failed: ListModels returned a malformed nextPageToken")
            if next_token in seen_page_tokens:
                raise RuntimeError(
                    "Gemini preflight failed: ListModels repeated a pagination token")
            seen_page_tokens.add(next_token)
            page_token = next_token
        else:
            raise RuntimeError(
                f"Gemini preflight failed: ListModels exceeded {_MAX_PREFLIGHT_PAGES} pages")

        missing = [m for m in self.models if m not in seen]
        unusable = [m for m in self.models if m in seen and "generateContent" not in seen[m]]
        if missing or unusable:
            problems = []
            if missing:
                problems.append(f"missing: {', '.join(missing)}")
            if unusable:
                problems.append(f"no generateContent support: {', '.join(unusable)}")
            available = ", ".join(sorted(seen)) or "(none returned)"
            raise RuntimeError(
                "Gemini preflight failed -- " + "; ".join(problems) +
                f". Available Gemini model ids: {available}")
