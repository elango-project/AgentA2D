"""AgentA2D domain exceptions.

All exceptions inherit from AgentA2DError.
No external dependencies — stdlib only.
"""


class AgentA2DError(Exception):
    """Base exception for all AgentA2D domain errors."""


class ForgedTrustError(AgentA2DError):
    """trust_label is TRUSTED without a valid trust_basis_ref.

    Raised when a MemoryObject claims ℓ=1 but has no resolvable reference
    to a protected-system assertion or human-authorization record.
    """


class BrokenLineageError(AgentA2DError):
    """Lineage validation failure.

    Raised when parent_refs or ancestor_refs contain unknown references,
    the ancestor set is incomplete (not the full transitive closure), or
    ancestry has been removed.
    """


class CyclicLineageError(AgentA2DError):
    """Object appears in its own ancestor_refs (direct or transitive)."""


class DuplicateObjectError(AgentA2DError):
    """An object_id already exists in the store with different content."""


class BrokenTraceReferenceError(AgentA2DError):
    """created_event_id references a non-existent or mismatched trace event.

    This is distinct from ForgedAuthorizationError — it covers any broken
    reference into the trace event log, not specifically forged authority.
    """


class SchemaVersionError(AgentA2DError):
    """Unknown or unsupported schema_version."""


class ContentIntegrityError(AgentA2DError):
    """content_hash does not match H(content)."""


class IntegrityProofError(AgentA2DError):
    """HMAC verification of integrity_proof fails."""


class InvalidTrustBasisError(AgentA2DError):
    """trust_basis_ref resolves but target is not a valid authority record.

    The referenced object exists but is not a protected-system assertion
    or human-authorization record.
    """


class ImmutabilityViolationError(AgentA2DError):
    """Attempt to modify any field of a frozen MemoryObject."""


class InvalidStatusTransitionError(AgentA2DError):
    """Disallowed status transition or invalid supersession chain.

    Raised on reverse transitions (e.g. QUARANTINED → ACTIVE), broken
    supersedes_object_id chains, non-sequential object_version, or
    content mismatch between versions.
    """
