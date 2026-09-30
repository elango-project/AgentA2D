"""Tests for trace structures and validation."""

import copy

import pytest

from agent_a2d.core.enums import Stage
from agent_a2d.core.errors import BrokenLineageError, BrokenTraceReferenceError
from agent_a2d.core.types import Session, Trace, TraceEvent
from agent_a2d.provenance.validation import validate_trace


@pytest.fixture
def valid_trace() -> Trace:
    evt1 = TraceEvent(
        event_id="evt_1", session_id="sess_1", stage=Stage.INGESTION,
        timestamp="2026-10-01T00:00:00Z", input_refs=(), output_refs=("obj_1",),
        enforced_decision=None, metadata={}
    )
    evt2 = TraceEvent(
        event_id="evt_2", session_id="sess_2", stage=Stage.RETRIEVAL,
        timestamp="2026-10-01T00:05:00Z", input_refs=("obj_1",), output_refs=(),
        enforced_decision=None, metadata={}
    )
    
    session_1 = Session(
        session_id="sess_1", session_index=1, started_at="2026-10-01T00:00:00Z",
        ended_at="2026-10-01T00:01:00Z", events=("evt_1",),
        pre_snapshot_id="snap_0", post_snapshot_id="snap_1"
    )
    
    session_2 = Session(
        session_id="sess_2", session_index=2, started_at="2026-10-01T00:04:00Z",
        ended_at="2026-10-01T00:06:00Z", events=("evt_2",),
        pre_snapshot_id="snap_1", post_snapshot_id="snap_2"
    )
    
    # Mocking a MemoryObject as just an object that acts like one for this test
    # validate_trace just checks if the key exists in trace.objects
    mock_objects = {"obj_1": object()}
    
    return Trace(
        trace_id="trace_1", experiment_id="exp_1", trial_id="trial_1",
        attack_id="atk_1", workload_id="work_1", seed=42,
        policy_version="1.0", schema_version=1,
        sessions=(session_1, session_2), events=(evt1, evt2),
        objects=mock_objects,  # type: ignore
        initial_snapshot_id="snap_0",
        session_boundary_snapshot_id="snap_1",
        final_snapshot_id="snap_2"
    )


def test_validate_trace_valid(valid_trace: Trace):
    """Happy path validation."""
    validate_trace(valid_trace)


def test_validate_trace_missing_event(valid_trace: Trace):
    """Session.events references a missing TraceEvent."""
    s1 = valid_trace.sessions[0]
    bad_s1 = copy.copy(s1)
    object.__setattr__(bad_s1, "events", ("missing_evt",))
    
    bad_trace = copy.copy(valid_trace)
    object.__setattr__(bad_trace, "sessions", (bad_s1, valid_trace.sessions[1]))
    
    with pytest.raises(BrokenTraceReferenceError, match="references missing event missing_evt"):
        validate_trace(bad_trace)


def test_validate_trace_session_mismatch(valid_trace: Trace):
    """TraceEvent.session_id does not match the owning session."""
    evt1 = valid_trace.events[0]
    bad_evt1 = copy.copy(evt1)
    object.__setattr__(bad_evt1, "session_id", "wrong_session")
    
    bad_trace = copy.copy(valid_trace)
    object.__setattr__(bad_trace, "events", (bad_evt1, valid_trace.events[1]))
    
    with pytest.raises(BrokenTraceReferenceError, match="belongs to wrong_session but referenced by sess_1"):
        validate_trace(bad_trace)


def test_validate_trace_missing_object(valid_trace: Trace):
    """TraceEvent references an object not in Trace.objects."""
    evt1 = valid_trace.events[0]
    bad_evt1 = copy.copy(evt1)
    object.__setattr__(bad_evt1, "output_refs", ("missing_obj",))
    
    bad_trace = copy.copy(valid_trace)
    object.__setattr__(bad_trace, "events", (bad_evt1, valid_trace.events[1]))
    
    with pytest.raises(BrokenLineageError, match="references missing output object missing_obj"):
        validate_trace(bad_trace)


def test_validate_trace_two_session_structure(valid_trace: Trace):
    """V1 must have exactly 2 sessions with correct indices."""
    bad_trace = copy.copy(valid_trace)
    object.__setattr__(bad_trace, "sessions", (valid_trace.sessions[0],))
    with pytest.raises(BrokenTraceReferenceError, match="exactly 2 sessions"):
        validate_trace(bad_trace)
        
    s2 = valid_trace.sessions[1]
    bad_s2 = copy.copy(s2)
    object.__setattr__(bad_s2, "session_index", 3)
    bad_trace2 = copy.copy(valid_trace)
    object.__setattr__(bad_trace2, "sessions", (valid_trace.sessions[0], bad_s2))
    with pytest.raises(BrokenTraceReferenceError, match="index 1 and 2"):
        validate_trace(bad_trace2)


def test_validate_trace_snapshot_consistency(valid_trace: Trace):
    """Snapshot references must be internally consistent."""
    bad_trace = copy.copy(valid_trace)
    object.__setattr__(bad_trace, "initial_snapshot_id", "snap_wrong")
    with pytest.raises(BrokenTraceReferenceError, match="initial snapshot != Session 1 pre"):
        validate_trace(bad_trace)
        
    bad_trace2 = copy.copy(valid_trace)
    object.__setattr__(bad_trace2, "session_boundary_snapshot_id", "snap_wrong")
    with pytest.raises(BrokenTraceReferenceError, match="boundary snapshot != Session 1 post"):
        validate_trace(bad_trace2)


def test_cross_session_persistence(valid_trace: Trace):
    """Test cross-session boundary preservation logic."""
    # This is validated implicitly by ensuring that obj_1 created in Session 1
    # is available and referenceable in Session 2. 
    # Our valid_trace fixture does exactly this: evt_1 (s1) outputs obj_1, evt_2 (s2) inputs obj_1.
    validate_trace(valid_trace)
    assert valid_trace.events[0].output_refs[0] == "obj_1"
    assert valid_trace.events[1].input_refs[0] == "obj_1"
    assert "obj_1" in valid_trace.objects
