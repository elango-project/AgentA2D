"""AgentA2D core deterministic pipeline.

Executes the two-session vertical slice WITH pluggable policy enforcement.
"""

from typing import Any, Dict, List, Optional, Tuple

from agent_a2d.core.enums import InterventionPoint, Stage, StageActionType
from agent_a2d.core.types import PolicyContext, Session, Trace, TraceEvent, ToolCall
from agent_a2d.experiment.clock import DeterministicClock
from agent_a2d.experiment.config import ExperimentConfig
from agent_a2d.ingestion.parser import parse_external_message
from agent_a2d.memory.environment import capture_environment
from agent_a2d.memory.store import MemoryStore
from agent_a2d.policy.adapter import StageEnforcementAdapter
from agent_a2d.policy.evaluator import PolicyEvaluator
from agent_a2d.provenance.middleware import stamp_agent_output
from agent_a2d.reasoning.stub import DeterministicStub
from agent_a2d.tools.auth_gate import ToolAuthGate


def run_deterministic_vertical_slice(
    config: ExperimentConfig,
    crm_state: Dict[str, Any],
    tool_state: Dict[str, Any],
) -> Trace:
    """Execute the full Session 1 -> Boundary -> Session 2 vertical slice.
    
    Applies active_enforcement if configured. Early enforcement halts the pipeline,
    leaving downstream events intentionally un-generated (NOT_REACHED).
    """
    clock = DeterministicClock(config.seed)
    memory_store = MemoryStore(config.hmac_key)
    events: List[TraceEvent] = []
    
    evaluator = PolicyEvaluator()
    adapter = StageEnforcementAdapter()
    
    def _evaluate_and_enforce(context: PolicyContext, point: InterventionPoint) -> Tuple[Any, bool]:
        """Evaluates policy and returns (EnforcedDecision, should_halt)."""
        decision = evaluator.evaluate(context)
        enf_decision = adapter.enforce(decision, point)
        
        # Halt ONLY if this is the active enforcement point AND the policy denied it
        should_halt = (
            config.active_enforcement == point and 
            enf_decision.stage_action != StageActionType.PERMIT
        )
        return enf_decision, should_halt

    # ---------------------------------------------------------
    # TRACE INITIALIZATION & SESSION 1 START
    # ---------------------------------------------------------
    initial_snapshot = capture_environment(
        snapshot_id=f"{config.trial_id}_snap_0",
        timestamp=clock.now(),
        memory_store=memory_store,
        crm_state=crm_state,
        tool_state=tool_state,
        session_state={"session_id": "sess_1"}
    )
    
    # === STAGE 1: INGESTION ===
    evt_ingest_id = f"{config.trial_id}_evt_1"
    raw_obj = parse_external_message(
        raw_payload=config.raw_payload,
        message_id=f"{config.trial_id}_obj_raw",
        session_id="sess_1",
        created_event_id=evt_ingest_id,
        object_store=memory_store.raw_storage,
        hmac_key=config.hmac_key,
    )
    
    ctx_ingest = PolicyContext(
        subject_objects=(raw_obj,),
        provenance_chain=(raw_obj,),
        action_anchor=config.oracle_action_anchor,
        tool_call=None,
        session_id="sess_1",
        trace_id=config.experiment_id
    )
    enf_ingest, halt_ingest = _evaluate_and_enforce(ctx_ingest, InterventionPoint.INGESTION)
    
    if not halt_ingest:
        memory_store.write(raw_obj)
        
    evt_ingest = TraceEvent(
        event_id=evt_ingest_id,
        session_id="sess_1",
        stage=Stage.INGESTION,
        timestamp=clock.now(),
        input_refs=(),
        output_refs=(raw_obj.object_id,) if not halt_ingest else (),
        enforced_decision=enf_ingest,
        metadata={"halted": halt_ingest}
    )
    events.append(evt_ingest)
    
    # === STAGE 2: MEMORY WRITE ===
    halt_memory = False
    evt_memory_id = f"{config.trial_id}_evt_2"
    if not halt_ingest:
        stub = config.reasoning_strategy or DeterministicStub(is_attack=config.is_attack)
        summary_content = stub.generate_summary([raw_obj])
        
        summary_obj = stamp_agent_output(
            object_id=f"{config.trial_id}_obj_summary",
            content=summary_content,
            parent_refs=[raw_obj.object_id],
            transformation="llm_summary",
            session_id="sess_1",
            created_event_id=evt_memory_id,
            object_store=memory_store.raw_storage,
            hmac_key=config.hmac_key,
        )
        
        ctx_memory = PolicyContext(
            subject_objects=(summary_obj,),
            provenance_chain=(summary_obj, raw_obj),
            action_anchor=config.oracle_action_anchor,
            tool_call=None,
            session_id="sess_1",
            trace_id=config.experiment_id
        )
        enf_memory, halt_memory = _evaluate_and_enforce(ctx_memory, InterventionPoint.MEMORY_WRITE)
        
        if not halt_memory:
            memory_store.write(summary_obj)
            
        evt_memory = TraceEvent(
            event_id=evt_memory_id,
            session_id="sess_1",
            stage=Stage.MEMORY_WRITE,
            timestamp=clock.now(),
            input_refs=(raw_obj.object_id,),
            output_refs=(summary_obj.object_id,) if not halt_memory else (),
            enforced_decision=enf_memory,
            metadata={"halted": halt_memory}
        )
        events.append(evt_memory)

    # SESSION 1 END & BOUNDARY
    boundary_snapshot = capture_environment(
        snapshot_id=f"{config.trial_id}_snap_1",
        timestamp=clock.now(),
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
        events=tuple(e.event_id for e in events if e.session_id == "sess_1"),
        pre_snapshot_id=initial_snapshot.snapshot_id,
        post_snapshot_id=boundary_snapshot.snapshot_id,
    )
    
    # ---------------------------------------------------------
    # SESSION 2 START
    # ---------------------------------------------------------
    
    # === STAGE 3: RETRIEVAL ===
    halt_retrieve = False
    evt_retrieve_id = f"{config.trial_id}_evt_3"
    
    if not halt_ingest and not halt_memory:
        retrieved = list(memory_store.retrieve_all_current())
        retrieved_ids = tuple(o.object_id for o in retrieved)
        
        # Build provenance chain for retrieved items
        prov_chain = list(retrieved)
        # Note: in a deep lineage scenario we'd recursively gather ancestors. 
        # For our simple slice, retrieved items + their immediate ancestors tracked via lineage works.
        # But for PolicyContext, we just provide the subject objects for now, as rules will scan them.
        
        ctx_retrieve = PolicyContext(
            subject_objects=tuple(retrieved),
            provenance_chain=tuple(retrieved), 
            action_anchor=config.oracle_action_anchor,
            tool_call=None,
            session_id="sess_2",
            trace_id=config.experiment_id
        )
        enf_retrieve, halt_retrieve = _evaluate_and_enforce(ctx_retrieve, InterventionPoint.RETRIEVAL)
        
        evt_retrieve = TraceEvent(
            event_id=evt_retrieve_id,
            session_id="sess_2",
            stage=Stage.RETRIEVAL,
            timestamp=clock.now(),
            input_refs=retrieved_ids,
            output_refs=retrieved_ids if not halt_retrieve else (),
            enforced_decision=enf_retrieve,
            metadata={"halted": halt_retrieve}
        )
        events.append(evt_retrieve)
        
        # === STAGE 4: TOOL AUTHORIZATION ===
        evt_tool_id = f"{config.trial_id}_evt_4"
        if not halt_retrieve:
            stub = config.reasoning_strategy or DeterministicStub(is_attack=config.is_attack)
            tool_call = stub.decide(retrieved, probe_instruction=config.probe_instruction)
            
            gate = ToolAuthGate()
            
            def ctx_builder(partial: PolicyContext) -> PolicyContext:
                return PolicyContext(
                    subject_objects=tuple(retrieved),
                    provenance_chain=tuple(retrieved),
                    action_anchor=partial.action_anchor,
                    tool_call=partial.tool_call,
                    session_id="sess_2",
                    trace_id=config.experiment_id
                )
            
            enf_tool, result = gate.execute_tool(tool_call, ctx_builder, crm_state)
            halt_tool = (config.active_enforcement == InterventionPoint.TOOL_AUTHORIZATION and 
                         enf_tool.stage_action != StageActionType.PERMIT)
            
            evt_tool = TraceEvent(
                event_id=evt_tool_id,
                session_id="sess_2",
                stage=Stage.TOOL_AUTHORIZATION,
                timestamp=clock.now(),
                input_refs=retrieved_ids,
                output_refs=(),
                enforced_decision=enf_tool,
                metadata={"tool_name": tool_call.action, "halted": halt_tool, "result": result}
            )
            events.append(evt_tool)
    
    # SESSION 2 END
    final_snapshot = capture_environment(
        snapshot_id=f"{config.trial_id}_snap_2",
        timestamp=clock.now(),
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
        events=tuple(e.event_id for e in events if e.session_id == "sess_2"),
        pre_snapshot_id=boundary_snapshot.snapshot_id,
        post_snapshot_id=final_snapshot.snapshot_id,
    )

    # ---------------------------------------------------------
    # TRACE CONSTRUCTION
    # ---------------------------------------------------------
    trace = Trace(
        trace_id=config.experiment_id,
        experiment_id=config.experiment_id,
        trial_id=config.trial_id,
        attack_id="atk_1" if config.is_attack else "none",
        workload_id="workload_1",
        seed=config.seed,
        policy_version=config.policy_version,
        schema_version=config.schema_version,
        sessions=(session_1, session_2),
        events=tuple(events),
        objects=memory_store.raw_storage,
        initial_snapshot_id=initial_snapshot.snapshot_id,
        session_boundary_snapshot_id=boundary_snapshot.snapshot_id,
        final_snapshot_id=final_snapshot.snapshot_id,
    )
    
    return trace
