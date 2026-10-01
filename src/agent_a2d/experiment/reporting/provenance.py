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
    Explicit Numeric Serialization Rules:
    - int: standard decimal string representation
    - float: finite floats only (allow_nan=False), uses Python's standard Dtoa algorithm which is deterministic across Python 3.8+.
    - None: 'null'
    - bool: 'true' / 'false'
    - tuple/list: standard JSON array
    - dict: sorted JSON object keys
    - frozen dataclasses: recursively converted to dictionaries
    - Enum: underlying value string/int
    """
    return json.dumps(
        obj, 
        sort_keys=True, 
        separators=(',', ':'), 
        ensure_ascii=False,
        allow_nan=False,
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
