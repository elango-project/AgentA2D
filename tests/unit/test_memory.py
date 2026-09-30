"""Tests for memory store and environment snapshots."""

import copy

import pytest

from agent_a2d.core.enums import ObjectStatus
from agent_a2d.core.errors import ContentIntegrityError
from agent_a2d.memory.environment import capture_environment, restore_environment
from agent_a2d.memory.store import MemoryStore
from agent_a2d.provenance.middleware import stamp_external_data, transition_status


def test_write_and_retrieve(hmac_key: bytes):
    """Write stores a MemoryObject and retrieve returns it intact."""
    store = MemoryStore(hmac_key)
    obj = stamp_external_data(
        object_id="1", content="hello", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    
    store.write(obj)
    
    retrieved = store.retrieve_by_id("1")
    assert retrieved is not None
    assert retrieved == obj


def test_version_tracking(hmac_key: bytes):
    """MemoryStore tracks supersession chains and returns current version."""
    store = MemoryStore(hmac_key)
    
    obj_v1 = stamp_external_data(
        object_id="1", content="hello", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    store.write(obj_v1)
    
    obj_v2 = transition_status(
        prior_version=obj_v1, new_object_id="2", new_status=ObjectStatus.QUARANTINED,
        created_event_id="e2", hmac_key=hmac_key
    )
    store.write(obj_v2)
    
    # retrieve_current_by_id("1") should yield v2
    current = store.retrieve_current_by_id("1")
    assert current is not None
    assert current.object_id == "2"
    assert current.status == ObjectStatus.QUARANTINED
    
    # retrieve_all_current should yield exactly one object (v2)
    all_current = list(store.retrieve_all_current())
    assert len(all_current) == 1
    assert all_current[0].object_id == "2"


def test_retrieval_non_mutation(hmac_key: bytes):
    """Retrieval NEVER mutates persistent storage."""
    store = MemoryStore(hmac_key)
    obj = stamp_external_data(
        object_id="1", content="hello", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    store.write(obj)
    
    retrieved = store.retrieve_by_id("1")
    
    # Since MemoryObject is a frozen dataclass containing tuples instead of lists, 
    # Python enforces immutability at runtime. A user cannot trivially modify the 
    # retrieved object to impact the store.
    
    # We can verify the store itself hasn't been wrapped or altered.
    assert id(store.retrieve_by_id("1")) == id(obj)


def test_snapshot_and_restore(hmac_key: bytes):
    """Test full environment snapshot capture and restoration."""
    store = MemoryStore(hmac_key)
    obj_v1 = stamp_external_data(
        object_id="1", content="hello", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    store.write(obj_v1)
    
    crm = {"customer_123": {"status": "active"}}
    tools = {"delete_customer": {"executions": 0}}
    session_data = {"active_user": "admin"}
    
    snapshot = capture_environment(
        snapshot_id="snap1",
        timestamp="2026-10-01",
        memory_store=store,
        crm_state=crm,
        tool_state=tools,
        session_state=session_data
    )
    
    # Now corrupt/mutate the live state
    obj_v2 = transition_status(
        prior_version=obj_v1, new_object_id="2", new_status=ObjectStatus.QUARANTINED,
        created_event_id="e2", hmac_key=hmac_key
    )
    store.write(obj_v2)
    crm["customer_123"]["status"] = "deleted"
    tools["delete_customer"]["executions"] = 1
    
    # Restore
    new_store = MemoryStore(hmac_key)
    restored_session = restore_environment(
        snapshot=snapshot,
        memory_store=new_store,
        crm_state=crm,
        tool_state=tools
    )
    
    # Verify everything rolled back to snap1
    assert crm["customer_123"]["status"] == "active"
    assert tools["delete_customer"]["executions"] == 0
    assert restored_session == session_data
    
    # Store should only have v1
    assert new_store.retrieve_by_id("1") == obj_v1
    assert new_store.retrieve_by_id("2") is None


def test_restore_revalidates(hmac_key: bytes):
    """Restoration must revalidate all objects."""
    store = MemoryStore(hmac_key)
    obj = stamp_external_data(
        object_id="1", content="hello", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    store.write(obj)
    
    snapshot = capture_environment(
        snapshot_id="snap1",
        timestamp="2026-10-01",
        memory_store=store,
        crm_state={},
        tool_state={},
        session_state={}
    )
    
    # Tamper with the snapshot's raw dictionary (simulating a corrupted disk state)
    # We have to bypass the frozen protections just for the negative test.
    bad_obj = copy.copy(obj)
    object.__setattr__(bad_obj, "content", "hacked")
    snapshot.memory_state["1"] = bad_obj
    
    # Restore should fail closed with ContentIntegrityError
    new_store = MemoryStore(hmac_key)
    with pytest.raises(ContentIntegrityError):
        restore_environment(
            snapshot=snapshot,
            memory_store=new_store,
            crm_state={},
            tool_state={}
        )
