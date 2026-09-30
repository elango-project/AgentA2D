"""Tests for trace structures."""

import pytest

from agent_a2d.core.enums import Stage
from agent_a2d.core.types import Session, Trace, TraceEvent


def test_trace_construction():
    event = TraceEvent(
        event_id="evt_1",
        session_id="sess_1",
        stage=Stage.INGESTION,
        timestamp="2026-10-01T00:00:00Z",
        input_refs=(),
        output_refs=("obj_1",),
        enforced_decision=None,
        metadata={}
    )
    
    session_1 = Session(
        session_id="sess_1",
        session_index=1,
        started_at="2026-10-01T00:00:00Z",
        ended_at="2026-10-01T00:01:00Z",
        events=("evt_1",),
        pre_snapshot_id="snap_0",
        post_snapshot_id="snap_1"
    )
    
    trace = Trace(
        trace_id="trace_1",
        experiment_id="exp_1",
        trial_id="trial_1",
        attack_id="atk_1",
        workload_id="work_1",
        seed=42,
        policy_version="1.0",
        schema_version=1,
        sessions=(session_1,),
        events=(event,),
        objects={},
        initial_snapshot_id="snap_0",
        session_boundary_snapshot_id="snap_1",
        final_snapshot_id="snap_1"
    )
    
    assert trace.trace_id == "trace_1"
    assert trace.sessions[0].session_id == "sess_1"
    assert trace.events[0].stage == Stage.INGESTION
