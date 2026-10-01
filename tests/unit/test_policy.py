"""Tests for policy evaluation and stage enforcement adapter."""

import pytest

from agent_a2d.core.enums import InterventionPoint, StageActionType, TrustLevel, Verdict
from agent_a2d.core.types import PolicyContext, ToolCall
from agent_a2d.policy.adapter import StageEnforcementAdapter
from agent_a2d.policy.evaluator import PolicyEvaluator
from agent_a2d.provenance.middleware import stamp_external_data, stamp_protected_assertion
from agent_a2d.tools.registry import DELETE_CUSTOMER_PROFILE, GET_CUSTOMER_INFO_PROFILE


@pytest.fixture
def evaluator() -> PolicyEvaluator:
    return PolicyEvaluator()


@pytest.fixture
def adapter() -> StageEnforcementAdapter:
    return StageEnforcementAdapter()


def test_evaluator_stage_independent(evaluator: PolicyEvaluator, hmac_key: bytes):
    """PolicyEvaluator produces the identical verdict for the same context,
    proving it is stage-blind (INV-19).
    """
    obj = stamp_external_data(
        object_id="1", content="test", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    
    context = PolicyContext(
        subject_objects=(obj,),
        provenance_chain=(obj,),
        action_anchor=DELETE_CUSTOMER_PROFILE,
        tool_call=None,
        session_id="s",
        trace_id="t"
    )
    
    # Evaluate produces stage-independent PolicyDecision
    decision = evaluator.evaluate(context)
    assert decision.verdict == Verdict.INTERDICT


def test_adapter_mappings(adapter: StageEnforcementAdapter, evaluator: PolicyEvaluator, hmac_key: bytes):
    """Adapter correctly maps INTERDICT to I, M, R, T stage actions."""
    obj = stamp_external_data(
        object_id="1", content="test", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    
    context = PolicyContext(
        subject_objects=(obj,),
        provenance_chain=(obj,),
        action_anchor=DELETE_CUSTOMER_PROFILE,
        tool_call=None,
        session_id="s",
        trace_id="t"
    )
    
    decision = evaluator.evaluate(context)
    assert decision.verdict == Verdict.INTERDICT
    
    # I-stage
    enf_i = adapter.enforce(decision, InterventionPoint.INGESTION)
    assert enf_i.stage_action == StageActionType.QUARANTINE_INPUT
    assert enf_i.enforced is True
    
    # M-stage
    enf_m = adapter.enforce(decision, InterventionPoint.MEMORY_WRITE)
    assert enf_m.stage_action == StageActionType.PREVENT_ACTIVATION
    
    # R-stage
    enf_r = adapter.enforce(decision, InterventionPoint.RETRIEVAL)
    assert enf_r.stage_action == StageActionType.EXCLUDE_FROM_REASONING
    
    # T-stage
    enf_t = adapter.enforce(decision, InterventionPoint.TOOL_AUTHORIZATION)
    assert enf_t.stage_action == StageActionType.DENY_TOOL_ACTION


def test_policy_positive_case(evaluator: PolicyEvaluator, adapter: StageEnforcementAdapter, hmac_key: bytes):
    """Valid authority produces ALLOW and PERMIT."""
    auth_obj = stamp_protected_assertion(
        object_id="2", content="auth", trust_basis_ref="sys_1",
        parent_refs=[], transformation="auth", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    
    context = PolicyContext(
        subject_objects=(auth_obj,),
        provenance_chain=(auth_obj,),
        action_anchor=DELETE_CUSTOMER_PROFILE,
        tool_call=ToolCall("delete_customer", {}),
        session_id="s",
        trace_id="t"
    )
    
    decision = evaluator.evaluate(context)
    assert decision.verdict == Verdict.ALLOW
    
    enf_t = adapter.enforce(decision, InterventionPoint.TOOL_AUTHORIZATION)
    assert enf_t.stage_action == StageActionType.PERMIT
    assert enf_t.enforced is False


def test_oracle_action_anchor(evaluator: PolicyEvaluator, hmac_key: bytes):
    """Missing tool_call relies entirely on oracle action_anchor."""
    obj = stamp_external_data(
        object_id="1", content="test", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    
    # Context lacking tool_call but providing oracle action_anchor
    context = PolicyContext(
        subject_objects=(obj,),
        provenance_chain=(obj,),
        action_anchor=DELETE_CUSTOMER_PROFILE,
        tool_call=None,
        session_id="s",
        trace_id="t"
    )
    
    decision = evaluator.evaluate(context)
    assert decision.verdict == Verdict.INTERDICT


def test_invalid_provenance_mapping(adapter: StageEnforcementAdapter, evaluator: PolicyEvaluator, hmac_key: bytes):
    """Forged trust fails closed with REJECT."""
    import copy
    
    obj = stamp_external_data(
        object_id="1", content="test", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    # Forge trust
    bad_obj = copy.copy(obj)
    object.__setattr__(bad_obj, "trust_label", TrustLevel.TRUSTED)
    
    context = PolicyContext(
        subject_objects=(bad_obj,),
        provenance_chain=(bad_obj,),
        action_anchor=DELETE_CUSTOMER_PROFILE,
        tool_call=None,
        session_id="s",
        trace_id="t"
    )
    
    decision = evaluator.evaluate(context)
    assert decision.verdict == Verdict.INVALID_PROVENANCE
    
    # INVALID_PROVENANCE maps to REJECT universally
    for point in InterventionPoint:
        enf = adapter.enforce(decision, point)
        assert enf.stage_action == StageActionType.REJECT
        assert enf.enforced is True
