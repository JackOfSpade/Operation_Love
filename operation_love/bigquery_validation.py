"""Shared lexical constraints for BigQuery identifiers interpolated into SQL."""
from __future__ import annotations

import re


BIGQUERY_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")
BIGQUERY_DATASET_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BIGQUERY_LOCATION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")

BIGQUERY_IDENTIFIER_PATTERNS = {
    "project_id": BIGQUERY_PROJECT_ID_RE,
    "dataset": BIGQUERY_DATASET_RE,
    "location": BIGQUERY_LOCATION_RE,
}


def validate_bigquery_identifier(value: object, name: str) -> str:
    """Return one exact safe SQL identifier/location or raise ``ValueError``."""
    pattern = BIGQUERY_IDENTIFIER_PATTERNS.get(name)
    if pattern is None:
        raise ValueError(f"unknown BigQuery identifier kind {name!r}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"BigQuery {name} must be a non-empty string (got {value!r})")
    if pattern.fullmatch(value) is None:
        raise ValueError(f"BigQuery {name} contains unsupported characters (got {value!r})")
    return value


def validate_bigquery_photo_bucket(value: object) -> str:
    """Reject missing or whitespace-padded GCS bucket names before SDK/network setup."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"BigQuery photo_bucket must be a non-empty string (got {value!r})")
    if value != value.strip():
        raise ValueError(
            f"BigQuery photo_bucket must not have surrounding whitespace (got {value!r})")
    return value
