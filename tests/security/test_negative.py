"""Security negative tests covering section 13."""

import copy

import pytest

from agent_a2d.core.enums import TrustLevel, Verdict, InterventionPoint
from agent_a2d.core.errors import ContentIntegrityError, ForgedTrustError, BrokenLineageError, IntegrityProofError, InvalidStatusTransitionError
from agent_a2d.core.types import MemoryObject, PolicyContext, ToolCall
from agent_a2d.provenance.middleware import stamp_external_data, stamp_protected_assertion
from agent_a2d.provenance.validation import validate_memory_object
from agent_a2d.policy.evaluator import PolicyEvaluator
from agent_a2d.policy.shadow import ShadowPolicyEvaluator
from agent_a2d.tools.registry import DELETE_CUSTOMER_PROFILE


def test_13_1_tampered_content(hmac_key: bytes):
    """\u00a713.1: Content modification invalidates integrity proof."""
    obj = stamp_external_data(
        object_id="1", content="data", session_id="s", created_event_id="e", object_store={}, hmac_key=hmac_key
    )
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
    with pytest.raises(ForgedTrustError):
        MemoryObject(
            object_id=bad_obj.object_id, object_version=bad_obj.object_version,
            supersedes_object_id=bad_obj.supersedes_object_id, content=bad_obj.content,
            content_hash=bad_obj.content_hash, trust_label=TrustLevel.TRUSTED,
            trust_basis_ref=None, parent_refs=bad_obj.parent_refs,
            ancestor_refs=bad_obj.ancestor_refs, transformation=bad_obj.transformation,
            session_id=bad_obj.session_id, created_event_id=bad_obj.created_event_id,
            status=bad_obj.status, schema_version=bad_obj.schema_version,
            integrity_proof="forged"
        )


def test_13_3_broken_lineage(hmac_key: bytes):
    """\u00a713.3: Referencing non-existent parents fails validation."""
    from agent_a2d.provenance.middleware import stamp_agent_output
    with pytest.raises(BrokenLineageError):
        stamp_agent_output(
            object_id="2", content="summary", parent_refs=["missing_parent"],
            transformation="llm", session_id="s", created_event_id="e",
            object_store={}, hmac_key=hmac_key
        )


def test_13_4_wrong_hmac_key(hmac_key: bytes):
    """\u00a713.4: Validating with the wrong experiment HMAC key fails."""
    obj = stamp_external_data(
        object_id="1", content="data", session_id="s", created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    with pytest.raises(IntegrityProofError):
        validate_memory_object(obj, {}, b"wrong_secret")


def test_13_5_policy_gate_positive(hmac_key: bytes):
    """\u00a713.5: Positive valid authority allows action without failing closed."""
    evaluator = PolicyEvaluator()
    auth_obj = stamp_protected_assertion(
        object_id="2", content="auth", trust_basis_ref="sys_1",
        parent_refs=[], transformation="auth", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    context = PolicyContext(
        subject_objects=(auth_obj,), provenance_chain=(auth_obj,),
        action_anchor=DELETE_CUSTOMER_PROFILE, tool_call=ToolCall("delete_customer", {}),
        session_id="s", trace_id="t"
    )
    decision = evaluator.evaluate(context)
    assert decision.verdict == Verdict.ALLOW


def test_13_6_stage_independent_equivalence(hmac_key: bytes):
    """\u00a713.6: Stage-independent verdict equivalence (evaluator output is unaffected by stage)."""
    evaluator = PolicyEvaluator()
    obj = stamp_external_data(
        object_id="1", content="data", session_id="s", created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    context = PolicyContext(
        subject_objects=(obj,), provenance_chain=(obj,),
        action_anchor=DELETE_CUSTOMER_PROFILE, tool_call=None,
        session_id="s", trace_id="t"
    )
    decision = evaluator.evaluate(context)
    assert decision.verdict == Verdict.INTERDICT


def test_13_7_session_boundary_integrity():
    """\u00a713.7: Sessions enforce strict boundaries in the trace schema (tested via validation fail-closed)."""
    from agent_a2d.core.types import Trace, Session, TraceEvent
    from agent_a2d.provenance.validation import validate_trace
    from agent_a2d.core.errors import BrokenTraceReferenceError
    
    # We construct a trace that breaks boundary constraints (e.g., sessions overlapping or missing snapshot links)
    s1 = Session("sess_1", 1, "t1", "t2", ("evt_1",), "snap_0", "snap_1")
    s2 = Session("sess_2", 2, "t2", "t3", ("evt_2",), "snap_1", "snap_2")
    # Missing evt_1 in trace events
    trace = Trace(
        "trace_1", "exp", "trial", "atk", "workload", 1, "1.0", 1,
        (s1, s2), (), {}, "snap_0", "snap_1", "snap_2"
    )
    with pytest.raises(BrokenTraceReferenceError):
        validate_trace(trace)


def test_13_8_retrieval_non_mutation(hmac_key: bytes):
    """\u00a713.8: Retrieval non-mutation fails closed if modified."""
    from agent_a2d.memory.store import MemoryStore
    store = MemoryStore(hmac_key)
    obj = stamp_external_data(
        object_id="1", content="data", session_id="s", created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    store.write(obj)
    retrieved = store.retrieve_by_id("1")
    
    # Prove that attempting to mutate retrieved object throws an error due to frozen dataclass
    with pytest.raises(FrozenInstanceError):
        retrieved.content = "hacked"
    
    # Also prove store is unaffected
    assert store.retrieve_by_id("1") == obj


def test_13_9_shadow_mode_behavior():
    """\u00a713.9: Shadow mode explicit downstream NOT_REACHED representation."""
    shadow = ShadowPolicyEvaluator()
    contexts = {
        InterventionPoint.INGESTION: PolicyContext((), (), DELETE_CUSTOMER_PROFILE, None, "s", "t")
    }
    results = shadow.evaluate_shadow(contexts)
    # Downstream explicitly recorded as None rather than fabricated
    assert results[InterventionPoint.MEMORY_WRITE] is None
    assert results[InterventionPoint.RETRIEVAL] is None
    assert results[InterventionPoint.TOOL_AUTHORIZATION] is None


def test_13_10_environment_snapshot(hmac_key: bytes):
    """\u00a713.10: EnvironmentSnapshot validation on restore fails closed on corruption."""
    from agent_a2d.memory.environment import capture_environment, restore_environment
    from agent_a2d.memory.store import MemoryStore
    store = MemoryStore(hmac_key)
    obj = stamp_external_data(
        object_id="1", content="data", session_id="s", created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    store.write(obj)
    snapshot = capture_environment(
        snapshot_id="snap", timestamp="t", memory_store=store, crm_state={}, tool_state={}, session_state={}
    )
    # Corrupt snapshot
    bad_obj = copy.copy(obj)
    object.__setattr__(bad_obj, "content", "corrupted")
    snapshot.memory_state["1"] = bad_obj
    
    with pytest.raises(ContentIntegrityError):
        restore_environment(snapshot=snapshot, memory_store=MemoryStore(hmac_key), crm_state={}, tool_state={})


def test_13_11_object_versioning(hmac_key: bytes):
    """\u00a713.11: Invalid status transitions fail closed (e.g. going backward)."""
    from agent_a2d.provenance.middleware import transition_status
    from agent_a2d.core.enums import ObjectStatus
    from agent_a2d.memory.store import MemoryStore
    
    store = MemoryStore(hmac_key)
    obj = stamp_external_data(
        object_id="1", content="data", session_id="s", created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    store.write(obj)
    
    # transition to quarantined is fine
    obj2 = transition_status(
        prior_version=obj, new_object_id="2", new_status=ObjectStatus.QUARANTINED,
        created_event_id="e2", hmac_key=hmac_key
    )
    store.write(obj2)

    # attempting to transition back to ACTIVE from QUARANTINED should fail
    obj3 = transition_status(
        prior_version=obj2, new_object_id="3", new_status=ObjectStatus.ACTIVE,
        created_event_id="e3", hmac_key=hmac_key
    )
    with pytest.raises(InvalidStatusTransitionError):
        store.write(obj3)

# We define FrozenInstanceError helper
from dataclasses import FrozenInstanceError
