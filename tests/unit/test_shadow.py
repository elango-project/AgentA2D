"""Tests for shadow execution and downstream NOT_REACHED semantics."""

import pytest

from agent_a2d.core.enums import InterventionPoint, StageActionType
from agent_a2d.core.types import PolicyContext
from agent_a2d.policy.shadow import ShadowPolicyEvaluator
from agent_a2d.tools.registry import DELETE_CUSTOMER_PROFILE


def test_shadow_reference_trace():
    """Reference trace evaluates all stages completely."""
    shadow = ShadowPolicyEvaluator()
    
    # Simulate a run with no active enforcement, so all stages provide a context
    empty_context = PolicyContext(
        subject_objects=(),
        provenance_chain=(),
        action_anchor=DELETE_CUSTOMER_PROFILE,
        tool_call=None,
        session_id="s",
        trace_id="t"
    )
    
    contexts = {
        InterventionPoint.INGESTION: empty_context,
        InterventionPoint.MEMORY_WRITE: empty_context,
        InterventionPoint.RETRIEVAL: empty_context,
        InterventionPoint.TOOL_AUTHORIZATION: empty_context,
    }
    
    results = shadow.evaluate_shadow(contexts)
    
    # All stages produce a genuine EnforcedDecision
    assert results[InterventionPoint.INGESTION] is not None
    assert results[InterventionPoint.MEMORY_WRITE] is not None
    assert results[InterventionPoint.RETRIEVAL] is not None
    assert results[InterventionPoint.TOOL_AUTHORIZATION] is not None


def test_shadow_downstream_not_reached():
    """Active early intervention causes downstream stages to be NOT_REACHED."""
    shadow = ShadowPolicyEvaluator()
    
    empty_context = PolicyContext(
        subject_objects=(),
        provenance_chain=(),
        action_anchor=DELETE_CUSTOMER_PROFILE,
        tool_call=None,
        session_id="s",
        trace_id="t"
    )
    
    # Simulate an active run that quarantined input at I-stage.
    # Therefore, M, R, and T were never reached and produce no context.
    contexts = {
        InterventionPoint.INGESTION: empty_context,
    }
    
    results = shadow.evaluate_shadow(contexts)
    
    assert results[InterventionPoint.INGESTION] is not None
    assert results[InterventionPoint.INGESTION].stage_action == StageActionType.QUARANTINE_INPUT
    
    # Downstream stages explicitly return None (NOT_REACHED)
    assert results[InterventionPoint.MEMORY_WRITE] is None
    assert results[InterventionPoint.RETRIEVAL] is None
    assert results[InterventionPoint.TOOL_AUTHORIZATION] is None
