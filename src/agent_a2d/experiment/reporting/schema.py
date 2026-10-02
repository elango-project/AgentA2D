import enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# Re-export base enums/schemas for convenience or rely on the ones from P6-A
from agent_a2d.experiment.schema import CompletedExperimentResult, ProviderFailure, ReferenceStatus

class ExclusionCategory(str, enum.Enum):
    VALIDATION_QUARANTINE = "VALIDATION_QUARANTINE"
    ANALYTICAL_INELIGIBILITY = "ANALYTICAL_INELIGIBILITY"

@dataclass(frozen=True)
class ExclusionRecord:
    trial_id: str
    category: ExclusionCategory
    reason: str
    details: Dict[str, Any]

@dataclass(frozen=True)
class DerivedResultProvenance:
    analysis_id: str
    rq: str
    metric: str
    eligibility_rule_version: str
    source_trial_ids: List[str]
    ineligible_trial_ids: List[str]
    provider_failure_trial_ids: List[str]
    quarantined_trial_ids: List[str]
    source_result_hash: str
    analyzer_version: str
    reporting_version: str
    analysis_code_commit: str

@dataclass(frozen=True)
class RQSummaryTable:
    numerator: float
    denominator: float
    eligible_count: int
    ineligible_count: int
    quarantined_count: int
    provider_failure_count: int
    eligibility_rule_version: str
    raw_data: Dict[str, Any]
    provenance: DerivedResultProvenance

@dataclass(frozen=True)
class ValidatedDataset:
    valid_results: List[CompletedExperimentResult]
    quarantine_records: List[ExclusionRecord]

@dataclass(frozen=True)
class EvidencePackageManifest:
    experiment_execution_commit: str
    analysis_code_commit: str
    package_generation_commit: str
    experiment_execution_tag: Optional[str]
    p6_protocol_version: str
    schema_version: int
    analyzer_version: str
    reporting_version: str
    canonical_content_digest: str
    generation_timestamp: str  # Informational only, excluded from digest math

class ArtifactStatus(str, enum.Enum):
    GENERATED = "GENERATED"
    RENDERER_UNAVAILABLE = "RENDERER_UNAVAILABLE"
    RENDER_FAILED = "RENDER_FAILED"

@dataclass(frozen=True)
class AccountingRecord:
    total_input: int
    valid: int
    quarantine: int
    provider_failures: int
    rq_eligible: Dict[str, int]
    rq_ineligible: Dict[str, int]

@dataclass(frozen=True)
class ScientificExecutionContext:
    """The stable identity defining the scientific experiment boundaries.
    Excludes all generation timestamps, paths, and rendering variables."""
    experiment_id: str
    experiment_execution_commit: str
    experiment_execution_ref: str
    p6_protocol_version: str
    schema_version: int
    provider: str
    model: str
    prompt_config_version: str
    policy_version: str
    workload_version: str
    seed: Optional[int]

@dataclass(frozen=True)
class PackageManifest:
    package_id: str
    experiment_id: str
    experiment_execution_commit: str
    experiment_execution_ref: str
    analysis_code_commit: str
    package_generation_commit: str
    p6_protocol_version: str
    schema_version: int
    analyzer_version: str
    reporting_version: str
    accounting: AccountingRecord
    artifact_statuses: Dict[str, ArtifactStatus]
    scientific_content_digest: str
    archive_digest: Optional[str] = None
    
class PackagingIntegrityError(Exception):
    """Raised when cross-layer validation fails before packaging."""
    pass
