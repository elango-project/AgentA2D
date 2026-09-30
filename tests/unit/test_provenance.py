"""Tests for provenance middleware and full object validation."""

import pytest

from agent_a2d.core.enums import ObjectStatus, TrustLevel
from agent_a2d.core.errors import (
    BrokenTraceReferenceError,
    ContentIntegrityError,
    DuplicateObjectError,
    ForgedTrustError,
    IntegrityProofError,
    InvalidStatusTransitionError,
    SchemaVersionError,
)
from agent_a2d.provenance.middleware import (
    stamp_agent_output,
    stamp_external_data,
    stamp_protected_assertion,
    transition_status,
)
from agent_a2d.provenance.validation import validate_memory_object


def test_middleware_enforces_untrusted(hmac_key: bytes):
    """Verify middleware assigns ℓ=0 to external and agent output."""
    store = {}
    ext_obj = stamp_external_data(
        object_id="1", content="mail", session_id="s",
        created_event_id="e", object_store=store, hmac_key=hmac_key
    )
    assert ext_obj.trust_label == TrustLevel.UNTRUSTED
    store["1"] = ext_obj

    agent_obj = stamp_agent_output(
        object_id="2", content="summary", parent_refs=["1"],
        transformation="llm", session_id="s", created_event_id="e",
        object_store=store, hmac_key=hmac_key
    )
    assert agent_obj.trust_label == TrustLevel.UNTRUSTED


def test_middleware_trusted_assertion(hmac_key: bytes):
    """Verify middleware correctly assigns ℓ=1 for protected assertions."""
    auth_obj = stamp_protected_assertion(
        object_id="1", content="auth", trust_basis_ref="sys_record_1",
        parent_refs=[], transformation="auth", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    assert auth_obj.trust_label == TrustLevel.TRUSTED
    assert auth_obj.trust_basis_ref == "sys_record_1"


def test_status_transition(hmac_key: bytes):
    """Verify status transitions create immutable valid new versions."""
    store = {}
    obj1 = stamp_external_data(
        object_id="1", content="mail", session_id="s",
        created_event_id="e", object_store=store, hmac_key=hmac_key
    )
    store["1"] = obj1
    
    obj2 = transition_status(
        prior_version=obj1, new_object_id="2", new_status=ObjectStatus.QUARANTINED,
        created_event_id="e2", hmac_key=hmac_key
    )
    
    assert obj2.object_version == 2
    assert obj2.supersedes_object_id == "1"
    assert obj2.status == ObjectStatus.QUARANTINED
    assert obj2.content == obj1.content
    
    # Validation should pass
    store["2"] = obj2
    validate_memory_object(obj2, store, hmac_key)


def test_validation_fail_closed_schema(hmac_key: bytes):
    obj = stamp_external_data(
        object_id="1", content="x", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    import copy
    bad = copy.copy(obj)
    object.__setattr__(bad, "schema_version", 999)
    with pytest.raises(SchemaVersionError):
        validate_memory_object(bad, {"1": bad}, hmac_key)


def test_validation_fail_closed_duplicate(hmac_key: bytes):
    obj = stamp_external_data(
        object_id="1", content="x", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    import copy
    different = copy.copy(obj)
    object.__setattr__(different, "content", "y")
    store = {"1": obj}
    
    with pytest.raises(DuplicateObjectError):
        validate_memory_object(different, store, hmac_key)


def test_validation_fail_closed_tampered_content(hmac_key: bytes):
    obj = stamp_external_data(
        object_id="1", content="x", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    import copy
    bad = copy.copy(obj)
    object.__setattr__(bad, "content", "y")
    # content hash will fail
    with pytest.raises(ContentIntegrityError):
        validate_memory_object(bad, {}, hmac_key)


def test_validation_fail_closed_hmac(hmac_key: bytes):
    obj = stamp_external_data(
        object_id="1", content="x", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    import copy
    bad = copy.copy(obj)
    object.__setattr__(bad, "integrity_proof", "bad_hmac")
    with pytest.raises(IntegrityProofError):
        validate_memory_object(bad, {}, hmac_key)


def test_validation_fail_closed_trace_reference(hmac_key: bytes):
    obj = stamp_external_data(
        object_id="1", content="x", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    # trace_events dict doesn't contain 'e'
    with pytest.raises(BrokenTraceReferenceError):
        validate_memory_object(obj, {}, hmac_key, trace_events={"other": object()})
