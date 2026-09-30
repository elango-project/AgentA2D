"""AgentA2D core types.

All types are strictly frozen dataclasses to enforce immutability.
Collections use tuple and Mapping instead of list and dict to
prevent in-place mutation.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from agent_a2d.core.enums import (
    ActionSeverity,
    InterventionPoint,
    ObjectStatus,
    Stage,
    StageActionType,
    TrustLevel,
    Verdict,
)


@dataclass(frozen=True)
class MemoryObject:
    """Fully immutable, versioned memory object."""

    object_id: str
    object_version: int
    supersedes_object_id: str | None
    content: str
    content_hash: str
    trust_label: TrustLevel
    trust_basis_ref: str | None
    parent_refs: tuple[str, ...]
    ancestor_refs: tuple[str, ...]
    transformation: str
    session_id: str
    created_event_id: str
    status: ObjectStatus
    schema_version: int
    integrity_proof: str

    def __post_init__(self) -> None:
        if self.trust_label == TrustLevel.TRUSTED and not self.trust_basis_ref:
            from agent_a2d.core.errors import ForgedTrustError
            raise ForgedTrustError("trust_label is TRUSTED without a valid trust_basis_ref")
        if self.trust_label == TrustLevel.UNTRUSTED and self.trust_basis_ref:
            from agent_a2d.core.errors import ForgedTrustError
            raise ForgedTrustError("trust_label is UNTRUSTED but trust_basis_ref is provided")
        if self.object_version > 1 and not self.supersedes_object_id:
            from agent_a2d.core.errors import InvalidStatusTransitionError
            raise InvalidStatusTransitionError("object_version > 1 requires supersedes_object_id")
        if self.object_version == 1 and self.supersedes_object_id:
            from agent_a2d.core.errors import InvalidStatusTransitionError
            raise InvalidStatusTransitionError("object_version == 1 cannot have supersedes_object_id")

@dataclass(frozen=True)
class ToolCall:
    """A requested tool action."""

    action: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ProtectedActionProfile:
    """Oracle Action Anchor for the deterministic V1 experiment.

    Provides the known future action required for early intervention
    checkpoints (I, M, R) to evaluate sufficiency of authority.
    """

    action: str
    severity: ActionSeverity
    subject: str
    required_authority: TrustLevel


@dataclass(frozen=True)
class AuthorizationRecord:
    """A protected system assertion or explicit human authorization."""

    record_id: str
    action: str
    subject: str
    timestamp: str
    human_approver: str | None
    system_approver: str | None


@dataclass(frozen=True)
class PolicyDecision:
    """Stage-independent policy verdict.

    Does NOT contain intervention point or stage action.
    """

    verdict: Verdict
    reason: str
    provenance_chain_snapshot: tuple[str, ...]


@dataclass(frozen=True)
class EnforcedDecision:
    """Stage-specific enforcement outcome mapped from a PolicyDecision."""

    verdict: Verdict
    stage_action: StageActionType
    intervention_point: InterventionPoint
    enforced: bool
    reason: str
    provenance_chain_snapshot: tuple[str, ...]


@dataclass(frozen=True)
class PolicyContext:
    """Context provided to PolicyEvaluator.

    Crucially, does NOT contain intervention_point. Structurally guarantees
    that PolicyEvaluator is stage-blind.

    Contains tool_call to explicitly represent the requested action at
    the T stage without smuggling it via intervention_point.
    """

    subject_objects: tuple[MemoryObject, ...]
    provenance_chain: tuple[MemoryObject, ...]
    action_anchor: ProtectedActionProfile | None
    tool_call: ToolCall | None
    session_id: str
    trace_id: str


@dataclass(frozen=True)
class TraceEvent:
    """A single deterministic event in the system trace."""

    event_id: str
    session_id: str
    stage: Stage
    timestamp: str
    input_refs: tuple[str, ...]
    output_refs: tuple[str, ...]
    enforced_decision: EnforcedDecision | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class Session:
    """A logical session boundary container."""

    session_id: str
    session_index: int
    started_at: str
    ended_at: str
    events: tuple[str, ...]
    pre_snapshot_id: str
    post_snapshot_id: str


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Snapshot of mutable environment state for deterministic replay."""

    snapshot_id: str
    timestamp: str
    memory_state: Mapping[str, MemoryObject]
    crm_state: Mapping[str, Any]
    tool_state: Mapping[str, Any]
    session_state: Mapping[str, Any] | None
    schema_version: int


@dataclass(frozen=True)
class Trace:
    """Canonical event trace of an experiment trial."""

    trace_id: str
    experiment_id: str
    trial_id: str
    attack_id: str
    workload_id: str
    seed: int
    policy_version: str
    schema_version: int
    sessions: tuple[Session, ...]
    events: tuple[TraceEvent, ...]
    objects: Mapping[str, MemoryObject]
    initial_snapshot_id: str
    session_boundary_snapshot_id: str
    final_snapshot_id: str
