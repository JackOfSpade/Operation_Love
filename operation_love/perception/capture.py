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
    # WHY there is no item payload, in one operator-readable sentence, or "" when there is one.
    # Never both: a capture either produced a numbered list or stated why it could not.
    #
    # This exists so the refusal is a RESULT rather than an absence. Doc 5.2 replaces raw
    # scroll frames with crops precisely so the model's item number means something; falling
    # back to the frames when the crops are missing would silently reintroduce the ambiguity,
    # so the consumer's only correct move is to stop and say this sentence (see worker.py's
    # auto loop). A driver that enumerates MUST set exactly one of `items` / this.
    items_unavailable: str = ""

    def text_blob(self) -> str:
        parts = [self.bio.strip()] if self.bio else []
        parts += [f"{q.strip()}: {a.strip()}" for q, a in self.prompts if a.strip()]
        return "\n".join(p for p in parts if p)
