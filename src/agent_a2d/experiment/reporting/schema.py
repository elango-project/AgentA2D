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
