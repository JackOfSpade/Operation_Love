"""The profile data structure passed between layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Profile:
    """Everything captured from one profile.

    photos: raw image bytes (PNG/JPEG) for each photo on the profile.
    prompts: list of (question, answer) for app prompt cards (Hinge/Bumble).
    bio: free-text bio / "about me" if present.

    THE ITEM PAYLOAD (the five trailing fields) is ops/OPENER-REDESIGN.md 5.2/5.7's request
    shape, carried here because the driver is the only layer that can build it and the opener
    is the layer that has to send it. It is deliberately PLAIN VALUES -- bytes, str, bool --
    and not `drivers.item_crops.ItemPayload`: this module is imported by the ranker, the
    worker, the stores and the opener, none of which may acquire a dependency on cv2 or on
    `operation_love.drivers`, and `opener.ItemRequest.from_profile` is a transcription of
    exactly these fields for the same reason.

    A driver that does not enumerate items (every non-Hinge driver today) leaves all five at
    their defaults, and a consumer reads that as "this capture has no item space at all",
    which is a different thing from `items_unavailable` -- see that field.
    """
    photos: list[bytes] = field(default_factory=list)
    bio: str = ""
    prompts: list[tuple[str, str]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    # Her first name as text. Cropping loses the sticky-header name the model would otherwise
    # read off a frame (doc 5.2), so it is passed back separately; "" when OCR did not read one.
    name: str = ""
    # THE NUMBERED LIST, in model order: `items[k - 1]` is the model's item k, 1-based per
    # opener.FIRST_ITEM_INDEX. One cropped image per selectable (heart-bearing) item. Position
    # is the whole contract -- doc 5.2's "image k IS item k" -- so nothing may reorder, filter
    # or append to this tuple after the driver built it.
    items: tuple[bytes, ...] = ()
    # The unnumbered context tier (doc 5.3's heartless vitals block): sent, read and freely
    # referenced, never selectable. Kept in its own field rather than mixed into `items`,
    # because a context crop among the numbered ones would renumber every item after it.
    item_context: tuple[bytes, ...] = ()
    # Whether the enumeration covers the whole profile. False means the capture demonstrably
    # started at a confirmed scroll top AND reached the end; True means these are only the
    # items we managed to read. Doc 5.7's truncation flag, distinct from
    # meta["capture_truncated"], which is the older per-frame read's own ceiling report.
    items_truncated: bool = False
    # A driver that enumerates leaves THIS PROFILE in exactly one of three states, and these two
    # fields plus `items` tell them apart (found+fixed 2026-08-22: a docstring here and a comment
    # at hinge.py:5317 both used to claim only two states -- "produced a list" or "stated why it
    # could not" -- which is false the moment enumeration runs to completion and legitimately
    # numbers nothing, e.g. a profile whose cards are all video):
    #
    #   1. `items` non-empty, both string fields "": enumeration produced a numbered list.
    #   2. `items` empty, `items_unavailable` a one-sentence reason: enumeration could not
    #      produce a payload at all -- an index that contradicts itself, a capture that could not
    #      be fingerprinted for identity, a dependency that raised. This is the one worker.py's
    #      auto loop treats as a HARD STOP regardless of decision (see there): sending the raw
    #      scroll frames instead of the crops doc 5.2 exists to send would hand the model a
    #      numbering nothing downstream can act on, so the loop stops rather than degrades.
    #   3. `items` empty, `items_unnumbered` a one-sentence reason: enumeration RAN TO COMPLETION
    #      -- the index was sound, identity was confirmed, crops were built -- and every
    #      candidate was legitimately excluded (still-photo evidence never covered it, policy
    #      demoted it, and so on). This is deliberately NOT `items_unavailable`: a profile of
    #      videos, or one where the dwell only ever reached one of many cards, is a normal
    #      outcome and must never halt an auto run on a PASS decision (see
    #      ops/STILL-PHOTO-DISCRIMINATOR.md 5d, which measured exactly this on a live run -- 15
    #      blocks indexed, 0 numbered). A LIKE decision is the one exception, and it is not this
    #      state's own -- worker.py's auto loop hard-stops a LIKE the same as case 2, just with
    #      its own reason, because a like still needs a verifiable item to attach an opener to
    #      and this state has none to offer (found+fixed 2026-08-22, second pass, same day).
    #
    # `items_unavailable` and `items_unnumbered` are never both non-empty -- they answer
    # different questions ("did enumeration finish?" vs "given that it finished, did anything
    # survive?") and only one question is ever the live one for a given capture.
    items_unavailable: str = ""
    # See the state-3 case above. One operator-readable sentence saying why nothing was numbered
    # despite enumeration completing, or "" when that is not this capture's state. Read by
    # worker.py's observe-mode hub warning, and, since a second pass the same day closed a gap
    # the first one left open (found+fixed 2026-08-22), also by the auto loop itself -- but only
    # on a LIKE decision: a PASS decision never reaches that opener code at all, so it still
    # sails past a zero-item profile exactly as before. A LIKE decision hard-stops on this field
    # for the same reason it hard-stops on `items_unavailable`: an opener must name a verifiable
    # item, and with nothing numbered there is none to name. Populated by the driver from the
    # actual per-item refusal reasons it recorded -- see hinge.py's `_index_captured_items`,
    # never a fixed sentence, per this repo's standing rule that guidance must derive from the
    # condition it describes rather than outlive it.
    items_unnumbered: str = ""
    # Machine-readable subtype for `items_unavailable`. Empty preserves the generic refusal
    # contract; `targeting_calibration` means the capture deliberately skipped enumeration
    # because the installed targeting calibration was absent or rejected for this live app.
    # It remains separate from the human-readable reason so downstream status can act on a
    # stable value without parsing diagnostics.
    items_unavailable_kind: str = ""

    def text_blob(self) -> str:
        parts = [self.bio.strip()] if self.bio else []
        parts += [f"{q.strip()}: {a.strip()}" for q, a in self.prompts if a.strip()]
        return "\n".join(p for p in parts if p)
