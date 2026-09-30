"""AgentA2D provenance middleware.

Enforces ℓ=0 on external data, agent outputs, and LLM derivations.
Handles creation of MemoryObjects with correct provenance stamps.
"""

from typing import Mapping, Sequence

from agent_a2d.core.enums import ObjectStatus, TrustLevel
from agent_a2d.core.types import MemoryObject
from agent_a2d.provenance.integrity import (
    compute_content_hash,
    compute_integrity_proof,
)
from agent_a2d.provenance.lineage import compute_ancestor_closure


def create_memory_object(
    *,
    object_id: str,
    content: str,
    trust_label: TrustLevel,
    trust_basis_ref: str | None,
    parent_refs: Sequence[str],
    transformation: str,
    session_id: str,
    created_event_id: str,
    object_store: Mapping[str, MemoryObject],
    hmac_key: bytes,
) -> MemoryObject:
    """Internal factory to construct and sign a MemoryObject."""
    content_hash = compute_content_hash(content)
    ancestor_refs = compute_ancestor_closure(parent_refs, object_store)

    proof = compute_integrity_proof(
        hmac_key=hmac_key,
        object_id=object_id,
        object_version=1,
        supersedes_object_id=None,
        content=content,
        content_hash=content_hash,
        trust_label=trust_label.name,
        trust_basis_ref=trust_basis_ref,
        parent_refs=tuple(parent_refs),
        ancestor_refs=ancestor_refs,
        transformation=transformation,
        session_id=session_id,
        created_event_id=created_event_id,
        status=ObjectStatus.ACTIVE.name,
        schema_version=1,
    )

    return MemoryObject(
        object_id=object_id,
        object_version=1,
        supersedes_object_id=None,
        content=content,
        content_hash=content_hash,
        trust_label=trust_label,
        trust_basis_ref=trust_basis_ref,
        parent_refs=tuple(parent_refs),
        ancestor_refs=ancestor_refs,
        transformation=transformation,
        session_id=session_id,
        created_event_id=created_event_id,
        status=ObjectStatus.ACTIVE,
        schema_version=1,
        integrity_proof=proof,
    )


def stamp_external_data(
    *,
    object_id: str,
    content: str,
    session_id: str,
    created_event_id: str,
    object_store: Mapping[str, MemoryObject],
    hmac_key: bytes,
) -> MemoryObject:
    """Stamp external/raw data (e.g., incoming email). Always ℓ=0."""
    return create_memory_object(
        object_id=object_id,
        content=content,
        trust_label=TrustLevel.UNTRUSTED,
        trust_basis_ref=None,
        parent_refs=(),
        transformation="ingestion_parse",
        session_id=session_id,
        created_event_id=created_event_id,
        object_store=object_store,
        hmac_key=hmac_key,
    )


def stamp_agent_output(
    *,
    object_id: str,
    content: str,
    parent_refs: Sequence[str],
    transformation: str,
    session_id: str,
    created_event_id: str,
    object_store: Mapping[str, MemoryObject],
    hmac_key: bytes,
) -> MemoryObject:
    """Stamp agent-generated text or derivation. ALWAYS ℓ=0.
    
    If the LLM attempts to output 'TRUSTED', the middleware explicitly
    ignores it and sets TrustLevel.UNTRUSTED.
    """
    return create_memory_object(
        object_id=object_id,
        content=content,
        trust_label=TrustLevel.UNTRUSTED,
        trust_basis_ref=None,
        parent_refs=parent_refs,
        transformation=transformation,
        session_id=session_id,
        created_event_id=created_event_id,
        object_store=object_store,
        hmac_key=hmac_key,
    )


def stamp_protected_assertion(
    *,
    object_id: str,
    content: str,
    trust_basis_ref: str,
    parent_refs: Sequence[str],
    transformation: str,
    session_id: str,
    created_event_id: str,
    object_store: Mapping[str, MemoryObject],
    hmac_key: bytes,
) -> MemoryObject:
    """Stamp a protected system assertion or human authorization. ℓ=1.
    
    Must supply a valid trust_basis_ref.
    """
    return create_memory_object(
        object_id=object_id,
        content=content,
        trust_label=TrustLevel.TRUSTED,
        trust_basis_ref=trust_basis_ref,
        parent_refs=parent_refs,
        transformation=transformation,
        session_id=session_id,
        created_event_id=created_event_id,
        object_store=object_store,
        hmac_key=hmac_key,
    )


def transition_status(
    *,
    prior_version: MemoryObject,
    new_object_id: str,
    new_status: ObjectStatus,
    created_event_id: str,
    hmac_key: bytes,
) -> MemoryObject:
    """Create a new MemoryObject version reflecting a status change.
    
    Copies all lineage, trust, and content from prior version, increments
    object_version, updates status, and issues a fresh integrity proof.
    """
    proof = compute_integrity_proof(
        hmac_key=hmac_key,
        object_id=new_object_id,
        object_version=prior_version.object_version + 1,
        supersedes_object_id=prior_version.object_id,
        content=prior_version.content,
        content_hash=prior_version.content_hash,
        trust_label=prior_version.trust_label.name,
        trust_basis_ref=prior_version.trust_basis_ref,
        parent_refs=prior_version.parent_refs,
        ancestor_refs=prior_version.ancestor_refs,
        transformation="status_transition",
        session_id=prior_version.session_id,
        created_event_id=created_event_id,
        status=new_status.name,
        schema_version=prior_version.schema_version,
    )

    return MemoryObject(
        object_id=new_object_id,
        object_version=prior_version.object_version + 1,
        supersedes_object_id=prior_version.object_id,
        content=prior_version.content,
        content_hash=prior_version.content_hash,
        trust_label=prior_version.trust_label,
        trust_basis_ref=prior_version.trust_basis_ref,
        parent_refs=prior_version.parent_refs,
        ancestor_refs=prior_version.ancestor_refs,
        transformation="status_transition",
        session_id=prior_version.session_id,
        created_event_id=created_event_id,
        status=new_status,
        schema_version=prior_version.schema_version,
        integrity_proof=proof,
    )
