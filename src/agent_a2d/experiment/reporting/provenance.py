import json
import hashlib
from typing import Any

class CanonicalEncoder(json.JSONEncoder):
    """Encodes objects deterministically into JSON."""
    def default(self, o: Any) -> Any:
        import dataclasses
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if hasattr(o, "name") and hasattr(o, "value"): # Enums
            return o.value
        return super().default(o)

def canonical_serialize(obj: Any) -> bytes:
    """
    Deterministically serialize an object to JSON for hashing.
    Requirements met:
    - UTF-8
    - Canonical JSON representation
    - Sorted keys
    - Stable separators
    - Deterministic collection handling
    """
    return json.dumps(
        obj, 
        sort_keys=True, 
        separators=(',', ':'), 
        ensure_ascii=False, 
        cls=CanonicalEncoder
    ).encode('utf-8')

def compute_digest(obj: Any) -> str:
    """Compute SHA-256 canonical content digest over the serialized bytes."""
    serialized = canonical_serialize(obj)
    return hashlib.sha256(serialized).hexdigest()

def digest_manifest(manifest_dict: dict) -> str:
    """
    Generate a digest for the manifest excluding the generation timestamp.
    """
    safe_dict = manifest_dict.copy()
    if "generation_timestamp" in safe_dict:
        del safe_dict["generation_timestamp"]
    if "canonical_content_digest" in safe_dict:
        del safe_dict["canonical_content_digest"]
        
    return compute_digest(safe_dict)
