"""AgentA2D environment snapshotting.

Captures and restores full deterministic mutable state.
"""

import copy
from typing import Any, Dict, Optional

from agent_a2d.core.types import EnvironmentSnapshot
from agent_a2d.memory.store import MemoryStore


def capture_environment(
    *,
    snapshot_id: str,
    timestamp: str,
    memory_store: MemoryStore,
    crm_state: Dict[str, Any],
    tool_state: Dict[str, Any],
    session_state: Optional[Dict[str, Any]],
    schema_version: int = 1,
) -> EnvironmentSnapshot:
    """Capture a full deterministic snapshot of all mutable state.

    Deep-copies CRM, tool, and session state. MemoryObjects are immutable
    dataclasses, so shallow dict copy is sufficient for deep isolation.
    """
    return EnvironmentSnapshot(
        snapshot_id=snapshot_id,
        timestamp=timestamp,
        memory_state=memory_store.raw_storage,
        crm_state=copy.deepcopy(crm_state),
        tool_state=copy.deepcopy(tool_state),
        session_state=copy.deepcopy(session_state) if session_state else None,
        schema_version=schema_version,
    )


def restore_environment(
    *,
    snapshot: EnvironmentSnapshot,
    memory_store: MemoryStore,
    crm_state: Dict[str, Any],
    tool_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Restore environment state from a snapshot.

    Replaces memory_store, crm_state, and tool_state in-place.
    Returns the restored session_state, if any.
    
    Validates all MemoryObjects during restore (fail-closed integrity).
    """
    # Restore MemoryStore (implicitly revalidates all objects via write())
    memory_store._replace_from_snapshot(snapshot.memory_state)
    
    # Restore CRM State
    crm_state.clear()
    crm_state.update(copy.deepcopy(snapshot.crm_state))
    
    # Restore Tool State
    tool_state.clear()
    tool_state.update(copy.deepcopy(snapshot.tool_state))
    
    # Return Session State
    if snapshot.session_state is not None:
        return copy.deepcopy(snapshot.session_state)
    return None
