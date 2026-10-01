"""Tests for deterministic reasoning stub."""

import json

import pytest

from agent_a2d.core.enums import TrustLevel
from agent_a2d.core.types import ToolCall
from agent_a2d.provenance.middleware import stamp_external_data, stamp_protected_assertion
from agent_a2d.reasoning.stub import DeterministicStub


def test_stub_benign_path(hmac_key: bytes):
    """Benign path produces the expected benign behavior (read-only action)."""
    stub = DeterministicStub(is_attack=False)
    
    obj = stamp_external_data(
        object_id="obj_1", content="some data", session_id="sess_1",
        created_event_id="evt_1", object_store={}, hmac_key=hmac_key
    )
    
    tool_call = stub.decide([obj])
    
    assert isinstance(tool_call, ToolCall)
    assert tool_call.action == "get_customer_info"
    assert tool_call.arguments == {"customer_id": "C123"}


def test_stub_attack_path(hmac_key: bytes):
    """Attack path unconditionally produces delete_customer(C123)."""
    stub = DeterministicStub(is_attack=True)
    
    obj = stamp_external_data(
        object_id="obj_1", content="malicious data", session_id="sess_1",
        created_event_id="evt_1", object_store={}, hmac_key=hmac_key
    )
    
    tool_call = stub.decide([obj])
    
    assert isinstance(tool_call, ToolCall)
    assert tool_call.action == "delete_customer"
    assert tool_call.arguments == {"customer_id": "C123"}


def test_stub_deterministic(hmac_key: bytes):
    """Same memory input -> identical ToolCall."""
    stub = DeterministicStub(is_attack=True)
    
    obj = stamp_external_data(
        object_id="obj_1", content="data", session_id="sess_1",
        created_event_id="evt_1", object_store={}, hmac_key=hmac_key
    )
    
    tool_call_1 = stub.decide([obj])
    tool_call_2 = stub.decide([obj])
    
    assert tool_call_1 == tool_call_2


def test_stub_generates_deterministic_summary(hmac_key: bytes):
    """Stub generates deterministic identical summary strings for identical inputs."""
    stub = DeterministicStub(is_attack=False)
    
    obj1 = stamp_external_data(
        object_id="obj_1", content="data1", session_id="sess_1",
        created_event_id="evt_1", object_store={}, hmac_key=hmac_key
    )
    obj2 = stamp_external_data(
        object_id="obj_2", content="data2", session_id="sess_1",
        created_event_id="evt_1", object_store={}, hmac_key=hmac_key
    )
    
    summary1 = stub.generate_summary([obj1, obj2])
    summary2 = stub.generate_summary([obj2, obj1])  # Input order varied
    
    assert summary1 == summary2
    parsed = json.loads(summary1)
    assert parsed["type"] == "summary"
    assert parsed["source_count"] == 2
    assert parsed["contains_attack_payload"] is False


def test_stub_no_authorization_logic(hmac_key: bytes):
    """The stub blindly generates ToolCalls regardless of input trust labels.
    
    It performs no authorization or security decisions.
    """
    stub = DeterministicStub(is_attack=True)
    
    # 1. Provide an explicitly UNTRUSTED object
    obj_untrusted = stamp_external_data(
        object_id="obj_1", content="untrusted", session_id="sess_1",
        created_event_id="evt_1", object_store={}, hmac_key=hmac_key
    )
    tool_call_untrusted = stub.decide([obj_untrusted])
    
    # 2. Provide a strictly TRUSTED object
    obj_trusted = stamp_protected_assertion(
        object_id="obj_2", content="trusted", trust_basis_ref="auth_1",
        parent_refs=[], transformation="auth", session_id="sess_1",
        created_event_id="evt_1", object_store={}, hmac_key=hmac_key
    )
    tool_call_trusted = stub.decide([obj_trusted])
    
    # The stub issues the exact same destructive ToolCall in both scenarios,
    # proving it delegates authorization to a separate downstream policy adapter.
    assert tool_call_untrusted == tool_call_trusted
    assert tool_call_trusted.action == "delete_customer"
