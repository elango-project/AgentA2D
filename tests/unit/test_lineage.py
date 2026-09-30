"""Tests for lineage invariants (LIN-01 through LIN-06)."""

import pytest

from agent_a2d.core.enums import TrustLevel
from agent_a2d.core.errors import BrokenLineageError, CyclicLineageError, ForgedTrustError
from agent_a2d.provenance.lineage import compute_ancestor_closure, validate_lineage
from agent_a2d.provenance.middleware import stamp_external_data, stamp_agent_output


def test_lin01_missing_parent(hmac_key: bytes):
    """LIN-01: All referenced parents must exist."""
    # We attempt to compute closure with a missing parent
    with pytest.raises(BrokenLineageError, match="Missing parent or ancestor: missing"):
        compute_ancestor_closure(["missing"], {})


def test_lin02_acyclic(hmac_key: bytes):
    """LIN-02: Lineage is acyclic."""
    obj1 = stamp_external_data(
        object_id="1", content="a", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    
    # We have to manually forge a cyclic object because middleware prevents it
    # We'll use a trick by mutating the object store and validating
    store = {"1": obj1}
    
    # Forge a bad object
    # Actually, we can just create an object whose ancestor_refs contains its own ID.
    import copy
    bad_obj = copy.copy(obj1)
    object.__setattr__(bad_obj, "ancestor_refs", ("1",))
    
    with pytest.raises(CyclicLineageError, match="appears in its own ancestry"):
        validate_lineage(bad_obj, store)


def test_lin05_ancestor_closure(hmac_key: bytes):
    """LIN-05: Ancestor chain is complete."""
    obj1 = stamp_external_data(
        object_id="1", content="a", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    store = {"1": obj1}
    
    obj2 = stamp_agent_output(
        object_id="2", content="b", parent_refs=["1"],
        transformation="tx", session_id="s", created_event_id="e",
        object_store=store, hmac_key=hmac_key
    )
    store["2"] = obj2
    
    obj3 = stamp_agent_output(
        object_id="3", content="c", parent_refs=["2"],
        transformation="tx", session_id="s", created_event_id="e",
        object_store=store, hmac_key=hmac_key
    )
    store["3"] = obj3
    
    # Obj3 should have ancestors 1 and 2
    assert set(obj3.ancestor_refs) == {"1", "2"}
    validate_lineage(obj3, store)
    
    # Forge incomplete ancestors
    import copy
    bad_obj3 = copy.copy(obj3)
    object.__setattr__(bad_obj3, "ancestor_refs", ("2",))  # missing 1
    
    with pytest.raises(BrokenLineageError, match="Ancestor closure mismatch"):
        validate_lineage(bad_obj3, store)


def test_lin06_derivation_never_increases_trust(hmac_key: bytes):
    """LIN-06: Derivation never increases trust."""
    obj1 = stamp_external_data(
        object_id="1", content="a", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    store = {"1": obj1}
    
    obj2 = stamp_agent_output(
        object_id="2", content="b", parent_refs=["1"],
        transformation="tx", session_id="s", created_event_id="e",
        object_store=store, hmac_key=hmac_key
    )
    
    assert obj2.trust_label == TrustLevel.UNTRUSTED
    
    # Try to manually assert TRUSTED without trust_basis_ref
    import copy
    bad_obj2 = copy.copy(obj2)
    object.__setattr__(bad_obj2, "trust_label", TrustLevel.TRUSTED)
    
    with pytest.raises(ForgedTrustError, match="TRUSTED object lacks trust_basis_ref"):
        validate_lineage(bad_obj2, store)


def test_multiple_parents(hmac_key: bytes):
    """Test valid multiple parents logic."""
    obj1 = stamp_external_data(
        object_id="1", content="a", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    obj2 = stamp_external_data(
        object_id="2", content="b", session_id="s",
        created_event_id="e", object_store={}, hmac_key=hmac_key
    )
    store = {"1": obj1, "2": obj2}
    
    obj3 = stamp_agent_output(
        object_id="3", content="c", parent_refs=["1", "2"],
        transformation="tx", session_id="s", created_event_id="e",
        object_store=store, hmac_key=hmac_key
    )
    
    assert set(obj3.ancestor_refs) == {"1", "2"}
    validate_lineage(obj3, store)
