"""AgentA2D deterministic ingestion.

Parses raw external input and applies initial ℓ=0 provenance stamps.
"""

from typing import Dict, Mapping

from agent_a2d.core.types import MemoryObject
from agent_a2d.provenance.middleware import stamp_external_data


def parse_external_message(
    raw_payload: str,
    message_id: str,
    session_id: str,
    created_event_id: str,
    object_store: Mapping[str, MemoryObject],
    hmac_key: bytes,
) -> MemoryObject:
    """Parse raw external data and assign initial provenance.
    
    Always applies TrustLevel.UNTRUSTED via the middleware.
    """
    # In a real system, this would parse JSON/MIME.
    # For the deterministic stub, we treat the raw_payload directly as content.
    return stamp_external_data(
        object_id=message_id,
        content=raw_payload,
        session_id=session_id,
        created_event_id=created_event_id,
        object_store=object_store,
        hmac_key=hmac_key,
    )
