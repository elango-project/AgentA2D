"""AgentA2D persistent memory store.

Provides versioned storage and retrieval for MemoryObjects.
Retrieval returns complete objects and NEVER mutates persistent storage.
"""

from typing import Dict, Iterable, Optional

from agent_a2d.core.enums import ObjectStatus
from agent_a2d.core.errors import DuplicateObjectError
from agent_a2d.core.types import MemoryObject
from agent_a2d.provenance.validation import validate_memory_object


class MemoryStore:
    """In-process versioned memory store for deterministic core.

    Tracks all object versions. Retrieval yields the active/current
    version in the supersession chain.
    """

    def __init__(self, hmac_key: bytes) -> None:
        self._hmac_key = hmac_key
        # internal storage: object_id -> MemoryObject
        self._objects: Dict[str, MemoryObject] = {}
        # logical_id -> current object_id
        # For version 1, logical_id == object_id. For version > 1, 
        # logical_id remains the original version 1 object_id.
        self._logical_to_current: Dict[str, str] = {}
        # Reverse mapping: object_id -> logical_id
        self._id_to_logical: Dict[str, str] = {}

    def _get_logical_id(self, obj: MemoryObject) -> str:
        """Find the root version 1 object_id for a given versioned object."""
        if obj.object_version == 1:
            return obj.object_id
        if not obj.supersedes_object_id:
            # Caught by validation, but just in case
            return obj.object_id
        
        # Traverse supersession chain to find root
        current_id = obj.supersedes_object_id
        while current_id in self._objects:
            prior = self._objects[current_id]
            if prior.object_version == 1:
                return prior.object_id
            if prior.supersedes_object_id:
                current_id = prior.supersedes_object_id
            else:
                break
        
        # Fallback if prior versions are somehow missing (though validation prevents this)
        return obj.supersedes_object_id

    def write(self, obj: MemoryObject) -> None:
        """Write a MemoryObject version to the store.
        
        Validates the object fully before storing.
        Updates the logical supersession chain.
        """
        # Validate fully (including HMAC, lineage, supersession chain)
        validate_memory_object(obj, self._objects, self._hmac_key)

        if obj.object_id in self._objects:
            # We know from validation it's exactly the same payload if it didn't raise
            # DuplicateObjectError. We can safely ignore or overwrite.
            return

        # Store object
        self._objects[obj.object_id] = obj

        # Update logical chain tracking
        logical_id = self._get_logical_id(obj)
        self._id_to_logical[obj.object_id] = logical_id
        
        # Only update current pointer if this is the highest version seen
        current_active_id = self._logical_to_current.get(logical_id)
        if current_active_id:
            current_active_obj = self._objects[current_active_id]
            if obj.object_version > current_active_obj.object_version:
                self._logical_to_current[logical_id] = obj.object_id
        else:
            self._logical_to_current[logical_id] = obj.object_id

    def retrieve_all_current(self) -> Iterable[MemoryObject]:
        """Retrieve all current versions of all logical objects.
        
        Returns the MemoryObject instances (never bare strings).
        NEVER mutates the stored objects.
        """
        for current_id in self._logical_to_current.values():
            obj = self._objects[current_id]
            # V1: Only ACTIVE and QUARANTINED objects are eligible for retrieval.
            # REJECTED objects are stored for audit only.
            if obj.status in (ObjectStatus.ACTIVE, ObjectStatus.QUARANTINED):
                yield obj

    def retrieve_by_id(self, object_id: str) -> Optional[MemoryObject]:
        """Retrieve a specific object version by exact ID."""
        return self._objects.get(object_id)

    def retrieve_current_by_id(self, object_id: str) -> Optional[MemoryObject]:
        """Retrieve the current version of the logical chain containing object_id."""
        logical_id = self._id_to_logical.get(object_id)
        if not logical_id:
            return None
        current_id = self._logical_to_current.get(logical_id)
        if not current_id:
            return None
        return self._objects.get(current_id)

    @property
    def raw_storage(self) -> Dict[str, MemoryObject]:
        """Provide read-only access to raw internal storage for snapshots."""
        return dict(self._objects)

    def _replace_from_snapshot(self, memory_state: Dict[str, MemoryObject]) -> None:
        """Internal method for EnvironmentSnapshot restoration."""
        self._objects.clear()
        self._logical_to_current.clear()
        self._id_to_logical.clear()
        
        # Sort objects by version so we process version 1 before version 2
        sorted_objs = sorted(memory_state.values(), key=lambda o: o.object_version)
        for obj in sorted_objs:
            self.write(obj)
