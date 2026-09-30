"""AgentA2D lineage calculation and validation (LIN-01 through LIN-06)."""

from typing import Mapping, Sequence

from agent_a2d.core.enums import TrustLevel
from agent_a2d.core.errors import BrokenLineageError, CyclicLineageError
from agent_a2d.core.types import MemoryObject


def compute_ancestor_closure(
    parent_refs: Sequence[str], object_store: Mapping[str, MemoryObject]
) -> tuple[str, ...]:
    """Compute the full transitive closure of ancestors.

    Raises BrokenLineageError if any parent is missing from the store.
    """
    ancestors: set[str] = set()
    queue = list(parent_refs)

    while queue:
        current_id = queue.pop(0)
        if current_id not in object_store:
            raise BrokenLineageError(f"Missing parent or ancestor: {current_id}")

        if current_id not in ancestors:
            ancestors.add(current_id)
            current_obj = object_store[current_id]
            queue.extend(current_obj.parent_refs)

    return tuple(sorted(ancestors))


def validate_lineage(
    obj: MemoryObject, object_store: Mapping[str, MemoryObject]
) -> None:
    """Validate all lineage invariants for a given MemoryObject.

    LIN-01: All referenced parents exist
    LIN-02: Lineage is acyclic
    LIN-05: Ancestor chain is complete
    LIN-06: Derivation never increases trust
    """
    # LIN-02: Acyclic
    if obj.object_id in obj.parent_refs or obj.object_id in obj.ancestor_refs:
        raise CyclicLineageError(
            f"Object {obj.object_id} appears in its own ancestry."
        )

    # LIN-01 & LIN-05: verify closure
    expected_ancestors = compute_ancestor_closure(obj.parent_refs, object_store)
    
    # Check if sets match. We don't rely on order here, just the exact elements.
    if set(expected_ancestors) != set(obj.ancestor_refs):
        raise BrokenLineageError(
            f"Ancestor closure mismatch. Expected {expected_ancestors}, "
            f"got {obj.ancestor_refs}"
        )

    # LIN-06: Derivation never increases trust.
    if obj.trust_label == TrustLevel.TRUSTED:
        if not obj.trust_basis_ref:
            from agent_a2d.core.errors import ForgedTrustError
            raise ForgedTrustError("TRUSTED object lacks trust_basis_ref")
