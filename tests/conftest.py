"""Shared test fixtures for AgentA2D."""

import pytest


@pytest.fixture
def hmac_key() -> bytes:
    """Fixed external test key, separate from any experiment seed.
    
    This fulfills the requirement that the HMAC key is supplied externally
    and is independent of the experiment seed used for reproducibility.
    """
    return b"test_only_fixed_hmac_key_for_integrity_verification"
