import enum
from dataclasses import dataclass
from typing import Dict, Any, Optional

class ProviderFailure(str, enum.Enum):
    SUCCESS = "SUCCESS"
    AUTH_FAILURE = "AUTH_FAILURE"
    QUOTA_FAILURE = "QUOTA_FAILURE"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    SDK_FAILURE = "SDK_FAILURE"
    OTHER_PROVIDER_FAILURE = "OTHER_PROVIDER_FAILURE"

class ReferenceStatus(str, enum.Enum):
    REFERENCE_SUCCESSFUL = "REFERENCE_SUCCESSFUL"
    REFERENCE_FAILED = "REFERENCE_FAILED"

@dataclass(frozen=True)
class ReproducibilityManifest:
    model: str
    provider: str
    api_version: str
    sdk_version: str
    prompt_config_version: str
    policy_version: str
    schema_version: int
    seed: Optional[int]
    workload_version: str
    git_commit: str

@dataclass(frozen=True)
class CompletedExperimentResult:
    experiment_id: str
    trial_id: str
    trace_id: str
    attack_id: str
    workload_id: str
    seed: Optional[int]
    provider: str
    model: str
    sdk_version: str
    api_version: str
    arm: str
    reference_status: ReferenceStatus
    provider_status: ProviderFailure
    metrics: Dict[str, Any]
    timestamps: Dict[str, float]
    git_commit: str
