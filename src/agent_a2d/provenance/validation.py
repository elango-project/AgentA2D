"""AgentA2D serialization and object validation.

Fail-closed validation at every boundary.
"""

from typing import Mapping

from agent_a2d.core.enums import TrustLevel
from agent_a2d.core.errors import (
    BrokenTraceReferenceError,
    ContentIntegrityError,
    DuplicateObjectError,
    ForgedTrustError,
    IntegrityProofError,
    InvalidStatusTransitionError,
    SchemaVersionError,
)
from agent_a2d.core.types import MemoryObject
from agent_a2d.provenance.integrity import (
    compute_content_hash,
    verify_integrity_proof,
)
from agent_a2d.provenance.lineage import validate_lineage

EXPECTED_SCHEMA_VERSION = 1


def validate_memory_object(
    obj: MemoryObject,
    object_store: Mapping[str, MemoryObject],
    hmac_key: bytes,
    trace_events: Mapping[str, object] | None = None,
) -> None:
    """Perform full defense-in-depth validation of a MemoryObject.

    Fail-closed: raises an AgentA2DError subclass on any validation failure.
    """
    # 1. Schema version
    if obj.schema_version != EXPECTED_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"Unsupported schema version: {obj.schema_version}. "
            f"Expected {EXPECTED_SCHEMA_VERSION}"
        )

    # 2. Duplicate object ID with different content/status
    if obj.object_id in object_store:
        existing = object_store[obj.object_id]
        if existing != obj:
            raise DuplicateObjectError(
                f"Object {obj.object_id} already exists with different payload."
            )

    # 3. Content Integrity
    if obj.content_hash != compute_content_hash(obj.content):
        raise ContentIntegrityError("content_hash does not match H(content)")

    # 4. Integrity Proof (HMAC covers ALL fields including status)
    is_valid = verify_integrity_proof(
        hmac_key=hmac_key,
        object_id=obj.object_id,
        object_version=obj.object_version,
        supersedes_object_id=obj.supersedes_object_id,
        content=obj.content,
        content_hash=obj.content_hash,
        trust_label=obj.trust_label.name,
        trust_basis_ref=obj.trust_basis_ref,
        parent_refs=obj.parent_refs,
        ancestor_refs=obj.ancestor_refs,
        transformation=obj.transformation,
        session_id=obj.session_id,
        created_event_id=obj.created_event_id,
        status=obj.status.name,
        schema_version=obj.schema_version,
        integrity_proof=obj.integrity_proof,
    )
    if not is_valid:
        raise IntegrityProofError("HMAC integrity proof verification failed")

    # 5. Trust basis
    if obj.trust_label == TrustLevel.TRUSTED and not obj.trust_basis_ref:
        raise ForgedTrustError("trust_label=TRUSTED requires trust_basis_ref")

    # 6. Trace reference
    if trace_events is not None and obj.created_event_id not in trace_events:
        raise BrokenTraceReferenceError(
            f"TraceEvent {obj.created_event_id} does not exist."
        )

    # 7. Lineage validation (LIN-01, LIN-02, LIN-05, LIN-06)
    validate_lineage(obj, object_store)

    # 8. Supersession chain
    if obj.object_version > 1:
        if not obj.supersedes_object_id:
            raise InvalidStatusTransitionError("Version > 1 missing supersedes_object_id")
        if obj.supersedes_object_id not in object_store:
            raise InvalidStatusTransitionError(
                f"Superseded object {obj.supersedes_object_id} not found."
            )
        prior = object_store[obj.supersedes_object_id]
        if prior.object_version != obj.object_version - 1:
            raise InvalidStatusTransitionError("Non-sequential object_version.")
        if prior.content_hash != obj.content_hash:
            raise InvalidStatusTransitionError(
                "Status transition altered content_hash."
            )
    else:
        if obj.supersedes_object_id is not None:
            raise InvalidStatusTransitionError("Version 1 cannot supersede.")
