"""AgentA2D core enumerations.

Defines all enumeration types used across the deterministic core.
No external dependencies — stdlib only.
"""

from enum import Enum, auto


class TrustLevel(Enum):
    """Binary trust level for V1.

    UNTRUSTED (ℓ=0): Default for all agent-generated or external content.
    TRUSTED   (ℓ=1): Only assignable by protected system assertion or
                      explicit human authorization.  Must carry a valid
                      trust_basis_ref.
    """

    UNTRUSTED = 0
    TRUSTED = 1


class InterventionPoint(Enum):
    """Research intervention points in the vertical slice.

    These are the four checkpoints where policy can be enforced.
    The experimental variable is WHICH point is actively enforced.
    """

    INGESTION = auto()
    MEMORY_WRITE = auto()
    RETRIEVAL = auto()
    TOOL_AUTHORIZATION = auto()


class Stage(Enum):
    """Stages in the vertical slice trace.

    Every stage in the two-session flow, including non-intervention stages.
    """

    INGESTION = auto()
    LLM_TRANSFORMATION = auto()
    MEMORY_WRITE = auto()
    SESSION_BOUNDARY = auto()
    RETRIEVAL = auto()
    REASONING = auto()
    TOOL_REQUEST = auto()
    TOOL_AUTHORIZATION = auto()
    ACTION_OUTCOME = auto()


class Verdict(Enum):
    """Stage-independent policy verdict.

    Produced by PolicyEvaluator.  Does NOT contain stage information.
    The evaluator never receives the intervention point; therefore
    these verdicts are structurally stage-blind (INV-18, INV-19).
    """

    ALLOW = auto()
    INTERDICT = auto()
    INVALID_PROVENANCE = auto()


class ObjectStatus(Enum):
    """Status of a MemoryObject version.

    Fully immutable per version — status transitions create new object
    versions with fresh integrity proofs.

    Permitted transitions: ACTIVE → QUARANTINED, ACTIVE → REJECTED.
    No reverse transitions in V1.
    """

    ACTIVE = auto()
    QUARANTINED = auto()
    REJECTED = auto()


class ActionSeverity(Enum):
    """Severity classification for tool actions."""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class StageActionType(Enum):
    """Physical enforcement action mapped from a Verdict.

    Produced by StageEnforcementAdapter — this is where stage-specific
    behaviour lives, NOT in PolicyEvaluator.
    """

    QUARANTINE_INPUT = auto()        # INTERDICT at I
    PREVENT_ACTIVATION = auto()      # INTERDICT at M
    EXCLUDE_FROM_REASONING = auto()  # INTERDICT at R
    DENY_TOOL_ACTION = auto()        # INTERDICT at T
    REJECT = auto()                  # INVALID_PROVENANCE at any point
    PERMIT = auto()                  # ALLOW at any point


# --- Valid status transitions (V1) ---

VALID_STATUS_TRANSITIONS: frozenset[tuple[ObjectStatus, ObjectStatus]] = frozenset(
    {
        (ObjectStatus.ACTIVE, ObjectStatus.QUARANTINED),
        (ObjectStatus.ACTIVE, ObjectStatus.REJECTED),
    }
)
