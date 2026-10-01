import pytest
from agent_a2d.experiment.analyzer import TraceAnalyzer
from agent_a2d.core.types import Trace, Session, TraceEvent
from agent_a2d.core.enums import Stage

def test_trace_metric_extraction_and_malformed():
    analyzer = TraceAnalyzer(target_action="delete_customer", target_arg="C123")
    
    # 1. Empty/malformed trace
    empty_trace = Trace(
        trace_id="t1", experiment_id="e1", trial_id="tr1", attack_id="a1", workload_id="w1", seed=42,
        policy_version="v4", schema_version=1,
        initial_snapshot_id="s1", session_boundary_snapshot_id="s1", final_snapshot_id="s1",
        sessions=(), events=(), objects={}
    )
    
    metrics = analyzer.extract_metrics(empty_trace)
    assert not metrics["storage_contamination"]
    assert not metrics["unauthorized_action_executed"]
    assert metrics["recovery_state"] == "benign_completion" # Safely defaults when nothing happens
    
    # 2. Trace with a halt metadata
    halt_event = TraceEvent(
        event_id="e1", session_id="sess1", stage=Stage.INGESTION, timestamp="0", 
        input_refs=(), output_refs=(), enforced_decision=None,
        metadata={"halted": True}
    )
    halt_trace = Trace(
        trace_id="t2", experiment_id="e1", trial_id="tr2", attack_id="a1", workload_id="w1", seed=42,
        policy_version="v4", schema_version=1,
        initial_snapshot_id="s1", session_boundary_snapshot_id="s1", final_snapshot_id="s1",
        sessions=(Session(session_id="sess1", session_index=0, started_at="0", ended_at="1", events=("e1",), pre_snapshot_id="s1", post_snapshot_id="s1"),), 
        events=(halt_event,), objects={}
    )
    metrics_halt = analyzer.extract_metrics(halt_trace)
    assert metrics_halt["recovery_state"] == "interdicted"
