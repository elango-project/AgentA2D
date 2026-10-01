"""Security negative tests covering section 13."""

import copy

import pytest

from agent_a2d.core.enums import TrustLevel
from agent_a2d.core.errors import ContentIntegrityError, ForgedTrustError, BrokenLineageError, IntegrityProofError
from agent_a2d.core.types import MemoryObject
from agent_a2d.provenance.middleware import stamp_external_data
from agent_a2d.provenance.validation import validate_memory_object


def test_13_1_tampered_content(hmac_key: bytes):
    """\u00a713.1: Content modification invalidates integrity proof."""
    obj = stamp_external_data(
        object_id="1", content="data", session_id="s", created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    
    # Tamper with content bypassing frozen restrictions
    bad_obj = copy.copy(obj)
    object.__setattr__(bad_obj, "content", "malicious")
    
    with pytest.raises(ContentIntegrityError):
        validate_memory_object(bad_obj, {}, hmac_key)


def test_13_2_forged_trust_label(hmac_key: bytes):
    """\u00a713.2: Forging \u2113=1 without a valid basis raises ForgedTrustError."""
    obj = stamp_external_data(
        object_id="1", content="data", session_id="s", created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    
    bad_obj = copy.copy(obj)
    object.__setattr__(bad_obj, "trust_label", TrustLevel.TRUSTED)
    
    # HMAC will fail first, but if we assume HMAC was somehow re-computed (by an attacker without the key)
    # the __post_init__ or validation catches forged trust logic.
    with pytest.raises(ForgedTrustError):
        MemoryObject(
            object_id=bad_obj.object_id,
            object_version=bad_obj.object_version,
            supersedes_object_id=bad_obj.supersedes_object_id,
            content=bad_obj.content,
            content_hash=bad_obj.content_hash,
            trust_label=TrustLevel.TRUSTED,
            trust_basis_ref=None,
            parent_refs=bad_obj.parent_refs,
            ancestor_refs=bad_obj.ancestor_refs,
            transformation=bad_obj.transformation,
            session_id=bad_obj.session_id,
            created_event_id=bad_obj.created_event_id,
            status=bad_obj.status,
            schema_version=bad_obj.schema_version,
            integrity_proof="forged"
        )


def test_13_3_broken_lineage(hmac_key: bytes):
    """\u00a713.3: Referencing non-existent parents fails validation."""
    # Create object claiming parent that doesn't exist
    from agent_a2d.provenance.middleware import stamp_agent_output
    
    with pytest.raises(BrokenLineageError):
        stamp_agent_output(
            object_id="2",
            content="summary",
            parent_refs=["missing_parent"],
            transformation="llm",
            session_id="s",
            created_event_id="e",
            object_store={},
            hmac_key=hmac_key
        )


def test_13_4_wrong_hmac_key(hmac_key: bytes):
    """\u00a713.4: Validating with the wrong experiment HMAC key fails."""
    obj = stamp_external_data(
        object_id="1", content="data", session_id="s", created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    wrong_key = b"wrong_secret"
    
    with pytest.raises(IntegrityProofError):
        validate_memory_object(obj, {}, wrong_key)
