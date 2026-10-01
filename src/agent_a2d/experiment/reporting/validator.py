from typing import List, Dict, Any, Set
from agent_a2d.experiment.schema import CompletedExperimentResult, ProviderFailure, ReferenceStatus
from agent_a2d.experiment.reporting.schema import ExclusionRecord, ExclusionCategory, ValidatedDataset

class DataValidationGate:
    def __init__(self, expected_schema_version: int = 1, expected_policy_version: str = "v4"):
        self.expected_schema_version = expected_schema_version
        self.expected_policy_version = expected_policy_version
        
        # Exact canonical metrics expected
        self.expected_metrics = {
            "storage_contamination", "retrieval_contamination", "behavioral_contamination", 
            "malicious_tool_request", "unauthorized_action_executed", "recovery_state"
        }
        self.allowed_recovery_states = {"benign_completion", "interdicted", "compromised", "unknown"}
        self.allowed_arms = {"CLEAN", "REFERENCE_ATTACK", "DEFENSE-I", "DEFENSE-M", "DEFENSE-R", "DEFENSE-T"}

    def validate(self, results: List[CompletedExperimentResult]) -> ValidatedDataset:
        valid_results = []
        quarantine = []
        
        seen_trials: Set[str] = set()
        seen_traces: Set[str] = set()
        
        for r in results:
            # 1. Unique IDs
            if r.trial_id in seen_trials:
                quarantine.append(ExclusionRecord(r.trial_id, ExclusionCategory.VALIDATION_QUARANTINE, "Duplicate trial_id", {}))
                continue
            if r.trace_id in seen_traces:
                quarantine.append(ExclusionRecord(r.trial_id, ExclusionCategory.VALIDATION_QUARANTINE, "Duplicate trace_id", {}))
                continue
                
            # 2. Required fields and strict types/enums
            if not isinstance(r.provider_status, ProviderFailure):
                quarantine.append(ExclusionRecord(r.trial_id, ExclusionCategory.VALIDATION_QUARANTINE, "Invalid provider_status enum", {"value": str(r.provider_status)}))
                continue
            if not isinstance(r.reference_status, ReferenceStatus):
                quarantine.append(ExclusionRecord(r.trial_id, ExclusionCategory.VALIDATION_QUARANTINE, "Invalid reference_status enum", {"value": str(r.reference_status)}))
                continue
            if not isinstance(r.arm, str) or r.arm not in self.allowed_arms:
                quarantine.append(ExclusionRecord(r.trial_id, ExclusionCategory.VALIDATION_QUARANTINE, "Invalid arm name", {"arm": str(r.arm)}))
                continue
                
            # 3. Required Metadata
            if not r.git_commit or not isinstance(r.git_commit, str):
                quarantine.append(ExclusionRecord(r.trial_id, ExclusionCategory.VALIDATION_QUARANTINE, "Missing or invalid git_commit", {}))
                continue
            if not r.model or not isinstance(r.model, str):
                quarantine.append(ExclusionRecord(r.trial_id, ExclusionCategory.VALIDATION_QUARANTINE, "Missing or invalid model", {}))
                continue
                
            # 4. Metric bounds
            metrics = r.metrics
            if not isinstance(metrics, dict):
                quarantine.append(ExclusionRecord(r.trial_id, ExclusionCategory.VALIDATION_QUARANTINE, "Metrics is not a dictionary", {}))
                continue
                
            if not self.expected_metrics.issubset(metrics.keys()):
                quarantine.append(ExclusionRecord(r.trial_id, ExclusionCategory.VALIDATION_QUARANTINE, "Missing required metric keys", {"missing": list(self.expected_metrics - metrics.keys())}))
                continue
                
            type_fail = False
            for bm in (self.expected_metrics - {"recovery_state"}):
                if not isinstance(metrics[bm], bool):
                    type_fail = True
                    break
            if type_fail:
                quarantine.append(ExclusionRecord(r.trial_id, ExclusionCategory.VALIDATION_QUARANTINE, "Metric type bound violation (expected bool)", {}))
                continue
                
            if metrics["recovery_state"] not in self.allowed_recovery_states:
                quarantine.append(ExclusionRecord(r.trial_id, ExclusionCategory.VALIDATION_QUARANTINE, "Invalid recovery_state domain", {"value": metrics["recovery_state"]}))
                continue
                
            # Passing all rules
            seen_trials.add(r.trial_id)
            seen_traces.add(r.trace_id)
            valid_results.append(r)
            
        return ValidatedDataset(valid_results=valid_results, quarantine_records=quarantine)
