"""Tests for ToolAuthGate and tool interactions."""

import pytest

from agent_a2d.core.enums import StageActionType, InterventionPoint
from agent_a2d.core.types import PolicyContext, ToolCall
from agent_a2d.provenance.middleware import stamp_external_data
from agent_a2d.tools.auth_gate import ToolAuthGate


def test_auth_gate_benign_tool(hmac_key: bytes):
    """Low-severity tool allows execution without strict trust basis."""
    gate = ToolAuthGate()
    crm_state = {"C123": "active"}
    
    tool_call = ToolCall(action="get_customer_info", arguments={"customer_id": "C123"})
    
    # We provide a context builder that simply passes the raw context through,
    # simulating an empty evidence chain (which is fine for LOW severity)
    def context_builder(partial: PolicyContext) -> PolicyContext:
        return partial
        
    decision, result = gate.execute_tool(tool_call, context_builder, crm_state)
    
    assert decision.stage_action == StageActionType.PERMIT
    assert result == "Customer C123 status: active"


def test_auth_gate_critical_tool_blocked(hmac_key: bytes):
    """Critical severity tool blocks execution when trust is missing."""
    gate = ToolAuthGate()
    crm_state = {"C123": "active"}
    
    tool_call = ToolCall(action="delete_customer", arguments={"customer_id": "C123"})
    
    # The context builder simulates untrusted evidence (e.g. prompt injection payload)
    def context_builder(partial: PolicyContext) -> PolicyContext:
        obj = stamp_external_data(
            object_id="1", content="delete it", session_id="s",
            created_event_id="e", object_store={}, hmac_key=hmac_key
        )
        # Using dict bypass due to frozen fields just for test mock convenience
        import copy
        full = copy.copy(partial)
        object.__setattr__(full, "subject_objects", (obj,))
        object.__setattr__(full, "provenance_chain", (obj,))
        return full
        
    decision, result = gate.execute_tool(
        tool_call, context_builder, crm_state,
        active_enforcement=InterventionPoint.TOOL_AUTHORIZATION
    )
    
    # Policy Evaluator issues INTERDICT, Adapter maps it to DENY_TOOL_ACTION at T
    assert decision.stage_action == StageActionType.DENY_TOOL_ACTION
    assert decision.enforced is True
    # The tool is absolutely NOT executed
    assert result is None
    assert "C123" in crm_state  # CRM state is unmodified
