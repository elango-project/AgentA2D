"""Tests for deterministic ingestion."""

import pytest

from agent_a2d.core.enums import TrustLevel
from agent_a2d.ingestion.parser import parse_external_message


def test_ingestion_always_untrusted(hmac_key: bytes):
    """Raw external input always becomes \u2113=0."""
    obj = parse_external_message(
        raw_payload="Hello world",
        message_id="msg_1",
        session_id="sess_1",
        created_event_id="evt_1",
        object_store={},
        hmac_key=hmac_key,
    )
    
    assert obj.trust_label == TrustLevel.UNTRUSTED
    assert obj.trust_basis_ref is None
    assert obj.transformation == "ingestion_parse"
    assert not obj.parent_refs
    assert not obj.ancestor_refs


def test_ingestion_deterministic(hmac_key: bytes):
    """Deterministic same-input -> same-output behavior."""
    payload = "Important data"
    
    obj1 = parse_external_message(
        raw_payload=payload,
        message_id="msg_1",
        session_id="sess_1",
        created_event_id="evt_1",
        object_store={},
        hmac_key=hmac_key,
    )
    
    obj2 = parse_external_message(
        raw_payload=payload,
        message_id="msg_1",
        session_id="sess_1",
        created_event_id="evt_1",
        object_store={},
        hmac_key=hmac_key,
    )
    
    assert obj1 == obj2
    assert obj1.integrity_proof == obj2.integrity_proof
