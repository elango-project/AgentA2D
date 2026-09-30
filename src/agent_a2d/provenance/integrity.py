"""Serialization integrity — HMAC computation and verification.

Provides deterministic canonical serialization and HMAC-SHA256-based
integrity verification for MemoryObject fields.

Key management contract:
    seed     → experiment randomness (reproducibility parameter, public)
    hmac_key → integrity secret (supplied externally, NEVER serialized
               with objects or included in experiment config)

This is an integrity mechanism for the experimental harness,
not a production cryptographic architecture.
"""

import hashlib
import hmac as _hmac
import json
from typing import Sequence


# ---------------------------------------------------------------------------
# Fixed field order for canonical serialization.
# This order MUST NOT change within a schema_version.
# ---------------------------------------------------------------------------

CANONICAL_FIELD_ORDER: tuple[str, ...] = (
    "object_id",
    "object_version",
    "supersedes_object_id",
    "content",
    "content_hash",
    "trust_label",
    "trust_basis_ref",
    "parent_refs",
    "ancestor_refs",
    "transformation",
    "session_id",
    "created_event_id",
    "status",
    "schema_version",
)


def compute_content_hash(content: str) -> str:
    """Compute canonical content hash: SHA-256 of UTF-8 encoded content.

    >>> compute_content_hash("hello")
    '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def canonical_serialize(
    *,
    object_id: str,
    object_version: int,
    supersedes_object_id: str | None,
    content: str,
    content_hash: str,
    trust_label: str,
    trust_basis_ref: str | None,
    parent_refs: Sequence[str],
    ancestor_refs: Sequence[str],
    transformation: str,
    session_id: str,
    created_event_id: str,
    status: str,
    schema_version: int,
) -> bytes:
    """Produce a deterministic canonical byte representation.

    Guarantees
    ----------
    - Fixed field ordering (defined by CANONICAL_FIELD_ORDER).
    - Sequences serialized in their given order (order is meaningful).
    - JSON with explicit separators, no whitespace.
    - ensure_ascii=True for byte-level determinism across platforms.
    - UTF-8 encoding.
    - None represented as JSON null.

    This function is the **single source of truth** for the HMAC payload.
    """
    payload: list[tuple[str, object]] = [
        ("object_id", object_id),
        ("object_version", object_version),
        ("supersedes_object_id", supersedes_object_id),
        ("content", content),
        ("content_hash", content_hash),
        ("trust_label", trust_label),
        ("trust_basis_ref", trust_basis_ref),
        ("parent_refs", list(parent_refs)),
        ("ancestor_refs", list(ancestor_refs)),
        ("transformation", transformation),
        ("session_id", session_id),
        ("created_event_id", created_event_id),
        ("status", status),
        ("schema_version", schema_version),
    ]
    json_str = json.dumps(
        payload,
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return json_str.encode("utf-8")


def compute_integrity_proof(
    *,
    hmac_key: bytes,
    object_id: str,
    object_version: int,
    supersedes_object_id: str | None,
    content: str,
    content_hash: str,
    trust_label: str,
    trust_basis_ref: str | None,
    parent_refs: Sequence[str],
    ancestor_refs: Sequence[str],
    transformation: str,
    session_id: str,
    created_event_id: str,
    status: str,
    schema_version: int,
) -> str:
    """Compute HMAC-SHA256 integrity proof over canonical payload.

    Returns hex-encoded HMAC digest string.
    """
    payload_bytes = canonical_serialize(
        object_id=object_id,
        object_version=object_version,
        supersedes_object_id=supersedes_object_id,
        content=content,
        content_hash=content_hash,
        trust_label=trust_label,
        trust_basis_ref=trust_basis_ref,
        parent_refs=parent_refs,
        ancestor_refs=ancestor_refs,
        transformation=transformation,
        session_id=session_id,
        created_event_id=created_event_id,
        status=status,
        schema_version=schema_version,
    )
    return _hmac.new(hmac_key, payload_bytes, hashlib.sha256).hexdigest()


def verify_integrity_proof(
    *,
    hmac_key: bytes,
    integrity_proof: str,
    object_id: str,
    object_version: int,
    supersedes_object_id: str | None,
    content: str,
    content_hash: str,
    trust_label: str,
    trust_basis_ref: str | None,
    parent_refs: Sequence[str],
    ancestor_refs: Sequence[str],
    transformation: str,
    session_id: str,
    created_event_id: str,
    status: str,
    schema_version: int,
) -> bool:
    """Verify HMAC-SHA256 integrity proof.

    Uses ``hmac.compare_digest`` for timing-safe comparison.
    Returns True if proof matches, False otherwise.
    """
    expected = compute_integrity_proof(
        hmac_key=hmac_key,
        object_id=object_id,
        object_version=object_version,
        supersedes_object_id=supersedes_object_id,
        content=content,
        content_hash=content_hash,
        trust_label=trust_label,
        trust_basis_ref=trust_basis_ref,
        parent_refs=parent_refs,
        ancestor_refs=ancestor_refs,
        transformation=transformation,
        session_id=session_id,
        created_event_id=created_event_id,
        status=status,
        schema_version=schema_version,
    )
    return _hmac.compare_digest(integrity_proof, expected)
