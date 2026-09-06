"""ranker.profile_key: the stable, cross-time attribution key -- pure hashing, no I/O.

See operation_love/ranker/profile_key.py's own docstring for the full design rationale (why a
hash rather than the raw fingerprint, why the grid rides inside the hashed payload, why no name
is mixed in). These tests pin the observable CONTRACT: deterministic, sensitive to both inputs
that matter, a 64-character hex digest, and the None-propagation rule for unknown identities.
"""
import hashlib
import json

from operation_love.drivers.item_identity import ProfileIdentity
from operation_love.ranker.profile_key import compute_profile_key, profile_key_from_identity


def test_compute_profile_key_matches_the_documented_canonical_encoding():
    """Pins the EXACT encoding the module docstring promises: a JSON object with only `v`,
    `grid`, `fingerprint`, dict keys sorted, separators tightened, ASCII-only. A silent change
    to any of these (a renamed/dropped field, sort_keys turned off, different separators) would
    still produce *a* hash -- this is what catches it being the WRONG one, independently of
    calling back into the module's own implementation."""
    fingerprint = (10, 20, 30, 255, 0)
    grid = (64, 16)
    expected_payload = json.dumps(
        {"v": 1, "grid": [64, 16], "fingerprint": [10, 20, 30, 255, 0]},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    expected = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()

    assert compute_profile_key(fingerprint, grid) == expected


def test_compute_profile_key_returns_a_64_character_lowercase_hex_digest():
    key = compute_profile_key((1, 2, 3), (64, 16))
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_compute_profile_key_is_deterministic():
    assert compute_profile_key((1, 2, 3), (64, 16)) == compute_profile_key((1, 2, 3), (64, 16))


def test_compute_profile_key_changes_when_the_fingerprint_changes():
    assert compute_profile_key((1, 2, 3), (64, 16)) != compute_profile_key((1, 2, 4), (64, 16))


def test_compute_profile_key_changes_when_the_grid_changes():
    """The grid rides inside the hashed payload (module docstring: 'WHY THE GRID RIDES INSIDE
    THE HASHED PAYLOAD') precisely so a recalibration that changes the grid opens a new key
    namespace rather than silently colliding with, or silently diverging from, the old one."""
    assert compute_profile_key((1, 2, 3), (64, 16)) != compute_profile_key((1, 2, 3), (128, 32))


def test_compute_profile_key_accepts_any_integer_sequence_not_only_tuples():
    """`ProfileIdentity.fingerprint` is documented as a plain tuple, but nothing about the hash
    should depend on the container type -- only the values and their order."""
    assert compute_profile_key([1, 2, 3], (64, 16)) == compute_profile_key((1, 2, 3), (64, 16))


def test_profile_key_from_identity_returns_none_for_no_identity():
    assert profile_key_from_identity(None) is None


def test_profile_key_from_identity_returns_none_when_fingerprint_is_unknown():
    unknown = ProfileIdentity(fingerprint=None, band=None, grid=(64, 16), frame_index=None,
                              scroll_top_distance=None, reason="no identity band declared")
    assert unknown.known is False

    assert profile_key_from_identity(unknown) is None


def test_profile_key_from_identity_hashes_the_known_fingerprint_and_grid():
    identity = ProfileIdentity(fingerprint=(10, 20, 30), band=(0.0, 0.0, 1.0, 1.0),
                               grid=(64, 16), frame_index=2, scroll_top_distance=0.0,
                               reason="settled header", agreeing_frames=5)
    assert identity.known is True

    assert profile_key_from_identity(identity) == compute_profile_key((10, 20, 30), (64, 16))


def test_profile_key_from_identity_never_returns_the_raw_fingerprint():
    """The one property this whole module exists to guarantee: what comes OUT is a hash, not
    a rendering of the pixels that went in. A raw fingerprint tuple (or its str()) would
    neither be 64 hex characters nor equal the SHA-256 of the canonical payload -- this pins
    both, plus the positive case (it IS that exact digest)."""
    fingerprint = tuple(range(0, 250, 5))  # a realistic-length, varied fingerprint
    identity = ProfileIdentity(fingerprint=fingerprint, band=(0.0, 0.0, 1.0, 1.0), grid=(64, 16),
                               frame_index=0, scroll_top_distance=0.0, reason="settled header")

    key = profile_key_from_identity(identity)

    assert key != str(fingerprint)
    assert key != fingerprint
    assert len(key) == 64 and all(c in "0123456789abcdef" for c in key)
    assert key == compute_profile_key(fingerprint, (64, 16))
