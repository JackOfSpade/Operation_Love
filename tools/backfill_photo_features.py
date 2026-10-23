"""Backfill the whole-photo feature (`photo_features`) for every label that lacks it.

    python -m tools.backfill_photo_features --config config.yaml
    python -m tools.backfill_photo_features --config config.yaml --dry-run --limit 5
    python -m tools.backfill_photo_features --config config.yaml --workers 2

WHY. `ranker.feature_set: face_crop_whole_photo_v1` scores the face-crop vector PLUS a pooled
whole-photo CLIP vector, and in that mode a label without a `photo_features` row makes the
ranker NOT READY (it is never zero-filled or dropped). Labels made before the feature existed
have no row, so this tool computes one per label from the frames that label was decided on.

TRAIN/SERVE PARITY. The vector is computed by the LIVE function, `Embedder.embed_whole_photos`,
on the SAME bytes the live path saw: the full-screen frames `record_profile` uploaded to Cloud
Storage (`profile_photos`, manual labels) or `record_auto_pass` archived (`auto_pass_photos`,
`auto_review_*` answers), all of them, in `photo_index` order. Every frame is checked against
its stored sha256 before it is used, and the embedder's `frames_sha256` must equal the digest of
the stored per-frame digests; either mismatch refuses that label (a row computed from bytes the
label was not decided on would be a silent train/serve skew).

WHAT IT TOUCHES.
* Reads: `store.labels_missing_photo_features("whole_photo_v1")` and the listed frames, from the
  configured private bucket only (a URI outside it is refused). Frames stay in memory for the one
  label being embedded and are never written to disk; at most `--workers` labels' frames are held
  at once.
* Writes: one `add_photo_features(..., source="backfill", best_effort=True)` row per computed
  label, unless `--dry-run`: each row is inserted at once and never buffered, so one the store
  rejects is that label's failure and cannot fail the next label's write or the final flush.
  The store is built with `make_store(cfg, ensure=True)` only when writing; a dry run uses
  `ensure=False` and creates nothing.
* Resumable and idempotent: only labels with no usable row are listed (none, or a zero-card row
  an earlier build stored; labels sharing one (app, profile_id) share one row and are listed
  once), so a re-run continues where an interrupted one stopped and a finished one has nothing
  to do.
* Nothing is ever written as a zero-card row. Frames in which the extractor finds no photo card
  are an extraction failure (`NoPhotoCards`, category "no photo card found"), and a label whose
  frames the store cannot supply (empty frame lists: SQLite archives none, a pruned AUTO pass, no
  profile_id, an inconsistent archive) had nothing measured; both are refused.
* The summary counts the cards in the 0.5..0.8 white band and how each of the extractor's two
  pixel cues decided them (vision/photo_cards.py "Photo or text"): the one place the live
  extraction departs from the experiment's CLIP rule, measured on the corpus.

Exit status 0 when every listed label was computed (and, unless a dry run, written and flushed
with nothing dropped); 1 when any label failed or the final flush failed. Failures name the
label (run_id/app/profile_id, never a person's name) and the reason, and the summary is printed
whatever happened.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

BACKFILL_SOURCE = "backfill"
DEFAULT_WORKERS = 2
# "A few profiles' frames at once": each worker holds one label's frames (up to a dozen
# full-resolution PNGs, ~2.4 MB) while it embeds them.
MAX_WORKERS = 8
_GCS_TIMEOUT_S = 60.0
_PROGRESS_EVERY = 25
_LABEL_KEYS = ("run_id", "app", "profile_id", "source", "frame_uris", "frame_sha256s")

# Failure categories, in the order the summary prints them.
NO_FRAMES = "frames unavailable"
NO_PHOTO_CARDS = "no photo card found"
MANIFEST = "malformed frame manifest"
OUTSIDE_BUCKET = "frame outside the configured bucket"
MISSING_OBJECT = "frame object missing"
SHA_MISMATCH = "frame sha256 mismatch"
DIGEST_MISMATCH = "frames_sha256 mismatch"
BAD_EMBEDDING = "malformed embedding result"
STORE_REFUSED = "store refused the row"
ERROR = "error"
_CATEGORY_ORDER = (
    NO_FRAMES,
    NO_PHOTO_CARDS,
    MANIFEST,
    OUTSIDE_BUCKET,
    MISSING_OBJECT,
    SHA_MISMATCH,
    DIGEST_MISMATCH,
    BAD_EMBEDDING,
    STORE_REFUSED,
    ERROR,
)


class LabelRefused(Exception):
    """This label cannot produce a trustworthy row; ``category`` is one of the constants."""

    def __init__(self, category: str, reason: str):
        super().__init__(reason)
        self.category = category
        self.reason = reason


def frames_digest(frame_sha256s: Sequence[str]) -> str:
    """sha256 hex of the concatenation of each frame's sha256 hex digest, in order.

    The `frames_sha256` formula the embedder stamps on every whole-photo vector, recomputed
    here from the STORED per-frame digests so a row can be tied to the exact archived frames.
    """
    return hashlib.sha256("".join(frame_sha256s).encode("ascii")).hexdigest()


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


class FrameSource:
    """Read-only access to one label's archived frames, each checked against its sha256.

    Frames are read only from the store's configured private bucket (``gs://<bucket>/...``);
    any other URI is refused, as ``BigQueryStore.auto_pass_photo`` refuses it.
    """

    def __init__(self, store, *, storage_client=None, bucket_name: str | None = None):
        if bucket_name is None:
            bucket_name = str(getattr(store, "photo_bucket_name", "") or "")
        if storage_client is None:
            storage_client = getattr(store, "storage_client", None)
        self._bucket_name = bucket_name
        self._prefix = f"gs://{bucket_name}/" if bucket_name else ""
        self._bucket = (
            storage_client.bucket(bucket_name)
            if storage_client is not None and bucket_name
            else None
        )

    def frames(self, label: dict) -> list[bytes]:
        uris, digests = label["frame_uris"], label["frame_sha256s"]
        if not isinstance(uris, (list, tuple)) or not isinstance(
            digests, (list, tuple)
        ):
            raise LabelRefused(MANIFEST, "frame_uris / frame_sha256s are not lists")
        if not uris:
            raise LabelRefused(
                NO_FRAMES,
                "the store cannot supply this label's frames (none archived, a pruned "
                "AUTO pass, no profile_id, or an inconsistent archive); nothing to embed",
            )
        if len(uris) != len(digests):
            raise LabelRefused(
                MANIFEST,
                f"{len(uris)} frame URIs but {len(digests)} stored sha256 digests",
            )
        out: list[bytes] = []
        for index, (uri, expected) in enumerate(zip(uris, digests, strict=True)):
            if not _is_sha256_hex(expected):
                raise LabelRefused(
                    MANIFEST, f"frame {index} has no valid stored sha256 to verify"
                )
            data = self._read(index, uri)
            if hashlib.sha256(data).hexdigest() != expected:
                raise LabelRefused(
                    SHA_MISMATCH,
                    f"frame {index} does not match its stored sha256; refusing the label",
                )
            out.append(data)
        return out

    def _read(self, index: int, uri: object) -> bytes:
        if not isinstance(uri, str) or not uri:
            raise LabelRefused(MANIFEST, f"frame {index} has no object URI")
        if not self._prefix or not uri.startswith(self._prefix):
            raise LabelRefused(
                OUTSIDE_BUCKET,
                f"frame {index} ({uri}) is outside the configured private bucket "
                f"{self._bucket_name!r}",
            )
        if self._bucket is None:
            raise LabelRefused(
                OUTSIDE_BUCKET, "this store has no Cloud Storage client to read frames"
            )
        blob = self._bucket.blob(uri[len(self._prefix) :])
        try:
            data = blob.download_as_bytes(timeout=_GCS_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - NotFound is a refusal; the rest propagate
            if type(exc).__name__ == "NotFound":
                raise LabelRefused(
                    MISSING_OBJECT, f"frame {index} ({uri}) is missing"
                ) from exc
            raise
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise LabelRefused(MISSING_OBJECT, f"frame {index} ({uri}) is empty")
        return bytes(data)


@dataclass(frozen=True)
class Computed:
    """One label's whole-photo row, validated and ready to write."""

    label: dict
    embedding: list[float]
    card_count: int
    frames_sha256: str
    feature_version: str
    band_cards: tuple[str, ...] = ()


def compute_label(
    label: dict,
    *,
    embedder,
    read_frames: Callable[[dict], list[bytes]],
    feature_version: str,
    dims: int,
) -> Computed:
    """Download, verify and embed ONE label's frames.  Raises ``LabelRefused`` or any error."""
    from operation_love.vision.photo_cards import NoPhotoCards

    frames = read_frames(label)
    try:
        result = embedder.embed_whole_photos(frames)
    except NoPhotoCards as exc:
        raise LabelRefused(
            NO_PHOTO_CARDS,
            f"{exc}; zero photo cards is an extraction failure, never written as a row",
        ) from exc
    expected_digest = frames_digest(label["frame_sha256s"])
    if getattr(result, "frames_sha256", None) != expected_digest:
        raise LabelRefused(
            DIGEST_MISMATCH,
            "the embedder's frames_sha256 is not the digest of the stored frame digests",
        )
    version = getattr(result, "feature_version", None)
    if version != feature_version:
        raise LabelRefused(
            BAD_EMBEDDING,
            f"embedder produced feature_version {version!r}, expected {feature_version!r}",
        )
    card_count = getattr(result, "card_count", None)
    if type(card_count) is not int or card_count < 1:
        raise LabelRefused(
            BAD_EMBEDDING, f"card_count {card_count!r} is not a positive count"
        )
    vector = getattr(result, "vector", None)
    if vector is None:
        raise LabelRefused(BAD_EMBEDDING, f"no vector but card_count is {card_count}")
    embedding = [float(v) for v in vector]
    if len(embedding) != dims:
        raise LabelRefused(
            BAD_EMBEDDING, f"vector has {len(embedding)} dims, expected {dims}"
        )
    if not all(math.isfinite(v) for v in embedding):
        raise LabelRefused(BAD_EMBEDDING, "vector has a non-finite value")
    return Computed(
        label=label,
        embedding=embedding,
        card_count=card_count,
        frames_sha256=expected_digest,
        feature_version=feature_version,
        band_cards=tuple(getattr(result, "band_cards", ()) or ()),
    )


@dataclass
class BackfillReport:
    feature_version: str
    dry_run: bool
    listed: int = 0
    attempted: int = 0
    computed: int = 0
    written: int = 0
    dropped: int = 0
    card_counts: Counter = field(default_factory=Counter)
    # photo_cards.BAND_* verdict -> count, over every computed label's band cards
    band_cards: Counter = field(default_factory=Counter)
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    flush_error: str | None = None
    coverage_before: dict | None = None
    coverage_after: dict | None = None

    @property
    def ok(self) -> bool:
        return not self.failures and self.dropped == 0 and self.flush_error is None


def _label_name(label: dict) -> str:
    return f"{label['app']}/{label['profile_id']} (run {label['run_id']})"


def _validated_labels(listed: object) -> list[dict]:
    if not isinstance(listed, (list, tuple)):
        raise RuntimeError(
            "labels_missing_photo_features must return a list "
            f"(got {type(listed).__name__})"
        )
    out = []
    for ordinal, label in enumerate(listed):
        if not isinstance(label, dict):
            raise RuntimeError(f"missing-feature entry {ordinal} is not a dict")
        absent = [key for key in _LABEL_KEYS if key not in label]
        if absent:
            raise RuntimeError(
                f"missing-feature entry {ordinal} lacks {', '.join(absent)}"
            )
        out.append(label)
    return out


def format_coverage(coverage: dict | None) -> str:
    if coverage is None:
        return "unavailable"
    return (
        f"{coverage['with_features']} of {coverage['labels']} labels have features "
        f"({coverage['zero_cards']} zero-card), {coverage['missing']} missing"
    )


def _coverage(store, feature_version: str, log) -> dict | None:
    try:
        return dict(store.photo_feature_coverage(feature_version))
    except Exception as exc:  # noqa: BLE001 - a summary line must not hide the real work
        log(f"Coverage unavailable: {type(exc).__name__}: {exc}")
        return None


def backfill(
    store,
    embedder,
    read_frames: Callable[[dict], list[bytes]],
    *,
    feature_version: str,
    dims: int,
    dry_run: bool = False,
    limit: int | None = None,
    workers: int = DEFAULT_WORKERS,
    log: Callable[[str], None] = print,
) -> BackfillReport:
    """Compute (and unless ``dry_run``, write) every missing row.  Never raises per label."""
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be an integer in [1, {MAX_WORKERS}]")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    report = BackfillReport(feature_version=feature_version, dry_run=dry_run)
    report.coverage_before = _coverage(store, feature_version, log)
    labels = _validated_labels(store.labels_missing_photo_features(feature_version))
    report.listed = len(labels)
    todo = labels if limit is None else labels[:limit]
    report.attempted = len(todo)
    mode = "DRY RUN, nothing is written" if dry_run else "writing source=backfill"
    log(
        f"Whole-photo backfill ({feature_version}; {mode}): {report.listed} label(s) "
        f"missing a row, {report.attempted} to process."
    )
    log(f"  before: {format_coverage(report.coverage_before)}")
    if not todo:
        report.coverage_after = report.coverage_before
        return report

    warmup = getattr(embedder, "warmup", None)
    if callable(warmup):
        warmup()  # a broken model fails here, before any frame is downloaded

    def run_one(label: dict) -> Computed:
        return compute_label(
            label,
            embedder=embedder,
            read_frames=read_frames,
            feature_version=feature_version,
            dims=dims,
        )

    done = 0
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(run_one, label): label for label in todo}
        for future in as_completed(futures):
            label = futures[future]
            done += 1
            try:
                computed = future.result()
            except LabelRefused as exc:
                report.failures.append((_label_name(label), exc.category, exc.reason))
                log(f"  REFUSED {_label_name(label)}: {exc.reason}")
                continue
            except Exception as exc:  # noqa: BLE001 - one label's failure is reported, not fatal
                reason = f"{type(exc).__name__}: {exc}"
                report.failures.append((_label_name(label), ERROR, reason))
                log(f"  FAILED {_label_name(label)}: {reason}")
                continue
            report.computed += 1
            report.card_counts[computed.card_count] += 1
            report.band_cards.update(computed.band_cards)
            if not dry_run:
                try:
                    written = store.add_photo_features(
                        run_id=label["run_id"],
                        app=label["app"],
                        profile_id=label["profile_id"],
                        feature_version=computed.feature_version,
                        embedding=computed.embedding,
                        card_count=computed.card_count,
                        frames_sha256=computed.frames_sha256,
                        source=BACKFILL_SOURCE,
                        best_effort=True,
                    )
                except Exception as exc:  # noqa: BLE001 - this label's failure, reported
                    reason = f"{type(exc).__name__}: {exc}"
                    report.failures.append((_label_name(label), STORE_REFUSED, reason))
                    log(
                        f"  FAILED {_label_name(label)}: the store refused the row: {reason}"
                    )
                    continue
                if written is True:
                    report.written += 1
                else:
                    report.failures.append(
                        (
                            _label_name(label),
                            STORE_REFUSED,
                            f"add_photo_features returned {written!r}",
                        )
                    )
                    log(f"  FAILED {_label_name(label)}: the store refused the row")
            if done % _PROGRESS_EVERY == 0:
                log(f"  {done}/{report.attempted} processed")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if not dry_run:
            # Rows already written are valid on their own; flush them even when interrupted so
            # a re-run resumes after them.  A flush failure is reported on the summary (and
            # fails the exit status) instead of replacing it.
            try:
                store.flush()
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                report.flush_error = f"{type(exc).__name__}: {exc}"
                log(f"  FLUSH FAILED: {report.flush_error}")
            dropped = getattr(store, "dropped_rows", None)
            if callable(dropped):
                report.dropped = int(dict(dropped()).get("photo_features", 0))
    report.coverage_after = (
        report.coverage_before if dry_run else _coverage(store, feature_version, log)
    )
    return report


def format_report(report: BackfillReport) -> str:
    lines = [
        f"Processed {report.attempted}: computed {report.computed}, "
        + ("wrote nothing (dry run)" if report.dry_run else f"wrote {report.written}")
        + f", failed {len(report.failures)}."
    ]
    if report.card_counts:
        dist = ", ".join(
            f"{count} card{'s' if count != 1 else ''} x{n}"
            for count, n in sorted(report.card_counts.items())
        )
        lines.append(f"  card counts: {dist}")
    band = _format_band_cards(report.band_cards)
    if band:
        lines.append(band)
    if report.failures:
        by_category = Counter(category for _, category, _ in report.failures)
        lines.append(
            "  failures: "
            + ", ".join(
                f"{category} x{by_category[category]}"
                for category in _CATEGORY_ORDER
                if by_category[category]
            )
        )
        if by_category[NO_FRAMES]:
            lines.append(
                "  Labels whose frames the store cannot supply can never be backfilled; while "
                "they exist the face_crop_whole_photo_v1 feature set stays NOT READY."
            )
        if by_category[NO_PHOTO_CARDS]:
            lines.append(
                "  Frames in which the extractor found no photo card are an extraction "
                "failure (a Hinge layout vision/photo_cards.py does not recognise); while "
                "they exist the face_crop_whole_photo_v1 feature set stays NOT READY."
            )
    if report.flush_error is not None:
        lines.append(f"  FLUSH FAILED: {report.flush_error}")
    if report.dropped:
        lines.append(
            f"  DATA LOSS: the store permanently dropped {report.dropped} photo_features "
            "row(s); re-run to recompute them."
        )
    lines.append(f"  after: {format_coverage(report.coverage_after)}")
    lines.append(
        "Backfill complete."
        if report.ok
        else "Backfill INCOMPLETE: re-run to retry (only missing rows are processed)."
    )
    return "\n".join(lines)


def _format_band_cards(counts: Counter) -> str | None:
    """The band-card line of the summary, or None when no computed label had one."""
    from operation_love.vision.photo_cards import (
        BAND_BOTH,
        BAND_CONTIGUOUS,
        BAND_EDGE_MEDIA,
        BAND_TEXT,
    )

    total = sum(counts.values())
    if not total:
        return None
    return (
        f"  band cards (white share 0.5..0.8): {total} seen; kept by edge media only "
        f"{counts[BAND_EDGE_MEDIA]}, by contiguous content only {counts[BAND_CONTIGUOUS]}, "
        f"by both {counts[BAND_BOTH]}; dropped as text {counts[BAND_TEXT]}"
    )


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="backfill_photo_features",
        description="Compute the whole-photo feature row for every label that lacks one.",
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="download, verify and embed, but write nothing (store opened with ensure=False)",
    )
    ap.add_argument(
        "--limit", type=_positive_int, default=None, help="process at most N labels"
    )
    ap.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        help=f"labels processed concurrently (1-{MAX_WORKERS}, default {DEFAULT_WORKERS})",
    )
    args = ap.parse_args(argv)
    if args.workers > MAX_WORKERS:
        ap.error(f"--workers must be at most {MAX_WORKERS}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    from operation_love import config as cfg_mod
    from operation_love.ranker import make_store
    from operation_love.vision.embed import Embedder
    from operation_love.vision.photo_cards import (
        WHOLE_PHOTO_DIMS,
        WHOLE_PHOTO_FEATURE_VERSION,
    )

    cfg = cfg_mod.load(args.config)
    store = make_store(cfg, ensure=not args.dry_run)
    close_error = None
    try:
        report = backfill(
            store,
            Embedder(),
            FrameSource(store).frames,
            feature_version=WHOLE_PHOTO_FEATURE_VERSION,
            dims=WHOLE_PHOTO_DIMS,
            dry_run=args.dry_run,
            limit=args.limit,
            workers=args.workers,
        )
    finally:
        try:
            store.close()
        except Exception as exc:  # noqa: BLE001 - reported below, after the summary
            close_error = f"{type(exc).__name__}: {exc}"
    print(format_report(report))
    if close_error is not None:
        print(f"Closing the store FAILED: {close_error}")
        return 1
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
