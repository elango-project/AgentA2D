"""Tests for core types and integrity mechanisms."""

import json
from dataclasses import FrozenInstanceError

import pytest

from agent_a2d.core.enums import ObjectStatus, TrustLevel
from agent_a2d.core.types import MemoryObject
from agent_a2d.provenance.integrity import (
    canonical_serialize,
    compute_content_hash,
    compute_integrity_proof,
    verify_integrity_proof,
)


def test_memory_object_immutability(hmac_key: bytes):
    """Test full immutability of MemoryObject (INV-12)."""
    content = "Hello, world!"
    content_hash = compute_content_hash(content)

    proof = compute_integrity_proof(
        hmac_key=hmac_key,
        object_id="obj_1",
        object_version=1,
        supersedes_object_id=None,
        content=content,
        content_hash=content_hash,
        trust_label=TrustLevel.UNTRUSTED.name,
        trust_basis_ref=None,
        parent_refs=("src_1", "src_2"),
        ancestor_refs=("src_1", "src_2", "src_0"),
        transformation="llm_summary",
        session_id="sess_1",
        created_event_id="evt_1",
        status=ObjectStatus.ACTIVE.name,
        schema_version=1,
    )

    obj = MemoryObject(
        object_id="obj_1",
        object_version=1,
        supersedes_object_id=None,
        content=content,
        content_hash=content_hash,
        trust_label=TrustLevel.UNTRUSTED,
        trust_basis_ref=None,
        parent_refs=("src_1", "src_2"),
        ancestor_refs=("src_1", "src_2", "src_0"),
        transformation="llm_summary",
        session_id="sess_1",
        created_event_id="evt_1",
        status=ObjectStatus.ACTIVE,
        schema_version=1,
        integrity_proof=proof,
    )

    # Cannot modify fields (FrozenInstanceError)
    with pytest.raises(FrozenInstanceError):
        obj.status = ObjectStatus.QUARANTINED  # type: ignore

    with pytest.raises(FrozenInstanceError):
        obj.content = "hacked"  # type: ignore

    # Tuple prevents in-place mutation of parent_refs
    assert not hasattr(obj.parent_refs, "append")


def test_status_included_in_integrity(hmac_key: bytes):
    """Verify that changing status changes the expected HMAC (covers INV-17)."""
    content = "content"
    content_hash = compute_content_hash(content)
    
    proof_active = compute_integrity_proof(
        hmac_key=hmac_key,
        object_id="obj_1",
        object_version=1,
        supersedes_object_id=None,
        content=content,
        content_hash=content_hash,
        trust_label=TrustLevel.UNTRUSTED.name,
        trust_basis_ref=None,
        parent_refs=("src_1",),
        ancestor_refs=("src_1",),
        transformation="parse",
        session_id="sess_1",
        created_event_id="evt_1",
        status=ObjectStatus.ACTIVE.name,
        schema_version=1,
    )

    proof_quarantined = compute_integrity_proof(
        hmac_key=hmac_key,
        object_id="obj_1",
        object_version=1,
        supersedes_object_id=None,
        content=content,
        content_hash=content_hash,
        trust_label=TrustLevel.UNTRUSTED.name,
        trust_basis_ref=None,
        parent_refs=("src_1",),
        ancestor_refs=("src_1",),
        transformation="parse",
        session_id="sess_1",
        created_event_id="evt_1",
        status=ObjectStatus.QUARANTINED.name,
        schema_version=1,
    )

    assert proof_active != proof_quarantined

    # verify_integrity_proof should fail if status was tampered with
    assert not verify_integrity_proof(
        hmac_key=hmac_key,
        object_id="obj_1",
        object_version=1,
        supersedes_object_id=None,
        content=content,
        content_hash=content_hash,
        trust_label=TrustLevel.UNTRUSTED.name,
        trust_basis_ref=None,
        parent_refs=("src_1",),
        ancestor_refs=("src_1",),
        transformation="parse",
        session_id="sess_1",
        created_event_id="evt_1",
        status=ObjectStatus.QUARANTINED.name,  # Altered status
        schema_version=1,
        integrity_proof=proof_active,  # Proof computed for ACTIVE
    )


def test_canonical_serialize_deterministic():
    """Verify that canonical serialize produces exactly the same bytes for same input."""
    data = dict(
        object_id="obj_1",
        object_version=1,
        supersedes_object_id=None,
        content="content",
        content_hash="hash",
        trust_label="UNTRUSTED",
        trust_basis_ref=None,
        parent_refs=("a", "b"),
        ancestor_refs=("a", "b", "c"),
        transformation="tx",
        session_id="sess",
        created_event_id="evt",
        status="ACTIVE",
        schema_version=1,
    )

    bytes1 = canonical_serialize(**data)
    bytes2 = canonical_serialize(**data)
    
    assert bytes1 == bytes2
    # Ensure it's valid JSON
    decoded = json.loads(bytes1.decode("utf-8"))
    
    # Check that tuple was converted to list in JSON
    assert decoded[7] == ["parent_refs", ["a", "b"]]

from agent_a2d.core.errors import ForgedTrustError, InvalidStatusTransitionError

def test_trust_basis_validation():
    with pytest.raises(ForgedTrustError, match='without a valid trust_basis_ref'):
        MemoryObject(
            object_id='1', object_version=1, supersedes_object_id=None,
            content='c', content_hash='h', trust_label=TrustLevel.TRUSTED,
            trust_basis_ref=None, parent_refs=(), ancestor_refs=(),
            transformation='tx', session_id='s', created_event_id='e',
            status=ObjectStatus.ACTIVE, schema_version=1, integrity_proof='p'
        )
        
    with pytest.raises(ForgedTrustError, match='but trust_basis_ref is provided'):
        MemoryObject(
            object_id='1', object_version=1, supersedes_object_id=None,
            content='c', content_hash='h', trust_label=TrustLevel.UNTRUSTED,
            trust_basis_ref='auth_1', parent_refs=(), ancestor_refs=(),
            transformation='tx', session_id='s', created_event_id='e',
            status=ObjectStatus.ACTIVE, schema_version=1, integrity_proof='p'
        )

def test_version_validation():
    with pytest.raises(InvalidStatusTransitionError, match='requires supersedes_object_id'):
        MemoryObject(
            object_id='2', object_version=2, supersedes_object_id=None,
            content='c', content_hash='h', trust_label=TrustLevel.UNTRUSTED,
            trust_basis_ref=None, parent_refs=(), ancestor_refs=(),
            transformation='tx', session_id='s', created_event_id='e',
            status=ObjectStatus.QUARANTINED, schema_version=1, integrity_proof='p'
        )
        
    with pytest.raises(InvalidStatusTransitionError, match='cannot have supersedes_object_id'):
        MemoryObject(
            object_id='1', object_version=1, supersedes_object_id='0',
            content='c', content_hash='h', trust_label=TrustLevel.UNTRUSTED,
            trust_basis_ref=None, parent_refs=(), ancestor_refs=(),
            transformation='tx', session_id='s', created_event_id='e',
            status=ObjectStatus.ACTIVE, schema_version=1, integrity_proof='p'
        )

