"""AgentA2D core deterministic pipeline.

Executes the two-session vertical slice without policy enforcement.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent_a2d.core.enums import Stage
from agent_a2d.core.types import Session, Trace, TraceEvent, ToolCall
from agent_a2d.ingestion.parser import parse_external_message
from agent_a2d.memory.environment import capture_environment
from agent_a2d.memory.store import MemoryStore
from agent_a2d.provenance.middleware import stamp_agent_output
from agent_a2d.reasoning.stub import DeterministicStub


def _now() -> str:
    """Returns deterministic-compatible timestamp format."""
    return datetime.now(timezone.utc).isoformat()


def run_deterministic_vertical_slice(
    experiment_id: str,
    trial_id: str,
    is_attack: bool,
    raw_payload: str,
    hmac_key: bytes,
    crm_state: Dict[str, Any],
    tool_state: Dict[str, Any],
) -> Trace:
    """Execute the full Session 1 -> Boundary -> Session 2 vertical slice.
    
    This is the P3 deterministic boundary. It tracks events and provenance
    but does NOT evaluate policy.
    """
    memory_store = MemoryStore(hmac_key)
    events: List[TraceEvent] = []
    
    # ---------------------------------------------------------
    # TRACE INITIALIZATION & SESSION 1 START
    # ---------------------------------------------------------
    initial_snapshot = capture_environment(
        snapshot_id=f"{trial_id}_snap_0",
        timestamp=_now(),
        memory_store=memory_store,
        crm_state=crm_state,
        tool_state=tool_state,
        session_state={"session_id": "sess_1"}
    )
    
    # 1. Ingestion Stage
    evt_ingest_id = f"{trial_id}_evt_1"
    raw_obj = parse_external_message(
        raw_payload=raw_payload,
        message_id=f"{trial_id}_obj_raw",
        session_id="sess_1",
        created_event_id=evt_ingest_id,
        object_store=memory_store.raw_storage,
        hmac_key=hmac_key,
    )
    memory_store.write(raw_obj)
    
    evt_ingest = TraceEvent(
        event_id=evt_ingest_id,
        session_id="sess_1",
        stage=Stage.INGESTION,
        timestamp=_now(),
        input_refs=(),
        output_refs=(raw_obj.object_id,),
        enforced_decision=None,
        metadata={}
    )
    events.append(evt_ingest)
    
    # 2. Agent Summary Stage (Memory Write)
    stub = DeterministicStub(is_attack=is_attack)
    summary_content = stub.generate_summary([raw_obj])
    
    evt_memory_id = f"{trial_id}_evt_2"
    summary_obj = stamp_agent_output(
        object_id=f"{trial_id}_obj_summary",
        content=summary_content,
        parent_refs=[raw_obj.object_id],
        transformation="llm_summary",
        session_id="sess_1",
        created_event_id=evt_memory_id,
        object_store=memory_store.raw_storage,
        hmac_key=hmac_key,
    )
    memory_store.write(summary_obj)
    
    evt_memory = TraceEvent(
        event_id=evt_memory_id,
        session_id="sess_1",
        stage=Stage.MEMORY_WRITE,
        timestamp=_now(),
        input_refs=(raw_obj.object_id,),
        output_refs=(summary_obj.object_id,),
        enforced_decision=None,
        metadata={}
    )
    events.append(evt_memory)

    # SESSION 1 END & BOUNDARY
    boundary_snapshot = capture_environment(
        snapshot_id=f"{trial_id}_snap_1",
        timestamp=_now(),
        memory_store=memory_store,
        crm_state=crm_state,
        tool_state=tool_state,
        session_state={"session_id": "sess_2"}
    )
    
    session_1 = Session(
        session_id="sess_1",
        session_index=1,
        started_at=initial_snapshot.timestamp,
        ended_at=boundary_snapshot.timestamp,
        events=(evt_ingest.event_id, evt_memory.event_id),
        pre_snapshot_id=initial_snapshot.snapshot_id,
        post_snapshot_id=boundary_snapshot.snapshot_id,
    )
    
    # ---------------------------------------------------------
    # SESSION 2 START
    # ---------------------------------------------------------
    
    # 3. Retrieval Stage
    retrieved = list(memory_store.retrieve_all_current())
    retrieved_ids = tuple(o.object_id for o in retrieved)
    
    evt_retrieve_id = f"{trial_id}_evt_3"
    evt_retrieve = TraceEvent(
        event_id=evt_retrieve_id,
        session_id="sess_2",
        stage=Stage.RETRIEVAL,
        timestamp=_now(),
        input_refs=retrieved_ids,
        output_refs=retrieved_ids,  # Retrieval passes them through
        enforced_decision=None,
        metadata={}
    )
    events.append(evt_retrieve)
    
    # 4. Reasoning / Tool Authorization Stage
    tool_call = stub.decide(retrieved)
    
    evt_tool_id = f"{trial_id}_evt_4"
    evt_tool = TraceEvent(
        event_id=evt_tool_id,
        session_id="sess_2",
        stage=Stage.TOOL_AUTHORIZATION,
        timestamp=_now(),
        input_refs=retrieved_ids,
        output_refs=(),
        enforced_decision=None,
        metadata={"tool_name": tool_call.action}
    )
    events.append(evt_tool)
    
    # SESSION 2 END
    final_snapshot = capture_environment(
        snapshot_id=f"{trial_id}_snap_2",
        timestamp=_now(),
        memory_store=memory_store,
        crm_state=crm_state,
        tool_state=tool_state,
        session_state=None
    )
    
    session_2 = Session(
        session_id="sess_2",
        session_index=2,
        started_at=boundary_snapshot.timestamp,
        ended_at=final_snapshot.timestamp,
        events=(evt_retrieve.event_id, evt_tool.event_id),
        pre_snapshot_id=boundary_snapshot.snapshot_id,
        post_snapshot_id=final_snapshot.snapshot_id,
    )

    # ---------------------------------------------------------
    # TRACE CONSTRUCTION
    # ---------------------------------------------------------
    trace = Trace(
        trace_id=f"{experiment_id}_{trial_id}",
        experiment_id=experiment_id,
        trial_id=trial_id,
        attack_id="atk_1" if is_attack else "none",
        workload_id="workload_1",
        seed=0,  # Fixed for now, P4 will handle experiment configuration
        policy_version="none",
        schema_version=1,
        sessions=(session_1, session_2),
        events=tuple(events),
        objects=memory_store.raw_storage,
        initial_snapshot_id=initial_snapshot.snapshot_id,
        session_boundary_snapshot_id=boundary_snapshot.snapshot_id,
        final_snapshot_id=final_snapshot.snapshot_id,
    )
    
    return trace
