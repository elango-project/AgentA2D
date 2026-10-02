import pytest
import dataclasses
from agent_a2d.experiment.schema import CompletedExperimentResult, ProviderFailure, ReferenceStatus
from agent_a2d.experiment.reporting.schema import ExclusionCategory, EvidencePackageManifest, DerivedResultProvenance
from agent_a2d.experiment.reporting.validator import DataValidationGate
from agent_a2d.experiment.reporting.provenance import canonical_serialize, compute_digest, digest_manifest

def create_valid_result(trial_id="t1", trace_id="tr1", ref_status=ReferenceStatus.REFERENCE_SUCCESSFUL, arm="CLEAN") -> CompletedExperimentResult:
    return CompletedExperimentResult(
        experiment_id="exp1",
        trial_id=trial_id,
        trace_id=trace_id,
        attack_id="a1",
        workload_id="w1",
        seed=42,
        provider="google.genai",
        model="gemini-3.8-flash",
        sdk_version="v2",
        api_version="v1",
        arm=arm,
        reference_status=ref_status,
        provider_status=ProviderFailure.SUCCESS,
        metrics={
            "storage_contamination": False,
            "retrieval_contamination": False,
            "behavioral_contamination": False,
            "malicious_tool_request": False,
            "unauthorized_action_executed": False,
            "recovery_state": "benign_completion"
        },
        timestamps={"start": 0.0},
        git_commit="exec_commit_hash"
    )

def test_valid_result_accepted():
    gate = DataValidationGate()
    dataset = gate.validate([create_valid_result()])
    assert len(dataset.valid_results) == 1
    assert len(dataset.quarantine_records) == 0

def test_missing_or_invalid_metadata():
    gate = DataValidationGate()
    bad_res = dataclasses.replace(create_valid_result(), git_commit=None)
    dataset = gate.validate([bad_res])
    assert len(dataset.valid_results) == 0
    assert len(dataset.quarantine_records) == 1
    assert dataset.quarantine_records[0].category == ExclusionCategory.VALIDATION_QUARANTINE
    assert "git_commit" in dataset.quarantine_records[0].reason

def test_invalid_enum_rejected():
    gate = DataValidationGate()
    bad_res = dataclasses.replace(create_valid_result(), provider_status="NOT_AN_ENUM")
    dataset = gate.validate([bad_res])
    assert len(dataset.quarantine_records) == 1
    assert "enum" in dataset.quarantine_records[0].reason

def test_invalid_metric_rejected():
    gate = DataValidationGate()
    bad_res = create_valid_result()
    # Mutate dict to create invalid bounds
    bad_res.metrics["unauthorized_action_executed"] = "yes" # Should be bool
    dataset = gate.validate([bad_res])
    assert len(dataset.quarantine_records) == 1
    assert "Metric type bound" in dataset.quarantine_records[0].reason

def test_duplicate_trial_or_trace_id_rejected():
    gate = DataValidationGate()
    # Same trial ID, different trace ID (still fails trial check)
    r1 = create_valid_result(trial_id="t1", trace_id="tr1")
    r2 = create_valid_result(trial_id="t1", trace_id="tr2")
    dataset = gate.validate([r1, r2])
    assert len(dataset.valid_results) == 1
    assert len(dataset.quarantine_records) == 1
    assert "Duplicate trial_id" in dataset.quarantine_records[0].reason

    # Different trial ID, same trace ID
    r3 = create_valid_result(trial_id="t3", trace_id="tr_dup")
    r4 = create_valid_result(trial_id="t4", trace_id="tr_dup")
    dataset = gate.validate([r3, r4])
    assert len(dataset.quarantine_records) == 1
    assert "Duplicate trace_id" in dataset.quarantine_records[0].reason

def test_reference_failed_remains_valid():
    """Valid REFERENCE_FAILED records remain valid data and are NOT globally deleted."""
    gate = DataValidationGate()
    ref_fail_res = create_valid_result(ref_status=ReferenceStatus.REFERENCE_FAILED)
    dataset = gate.validate([ref_fail_res])
    assert len(dataset.valid_results) == 1
    assert dataset.valid_results[0].reference_status == ReferenceStatus.REFERENCE_FAILED

def test_canonical_serialization_and_digest():
    """Identical input -> identical digest. Changed input -> changed digest."""
    d1 = {"b": 2, "a": 1, "c": [3, 2, 1]}
    d2 = {"a": 1, "c": [3, 2, 1], "b": 2}
    
    assert canonical_serialize(d1) == canonical_serialize(d2)
    assert compute_digest(d1) == compute_digest(d2)
    
    d3 = {"a": 1, "b": 2, "c": [1, 2, 3]} # Order in list matters
    assert compute_digest(d1) != compute_digest(d3)

def test_digest_manifest_excludes_timestamp():
    manifest_dict = {
        "experiment_execution_commit": "c1",
        "analysis_code_commit": "c2",
        "generation_timestamp": "2026-10-01T12:00:00Z"
    }
    
    manifest_dict_later = {
        "experiment_execution_commit": "c1",
        "analysis_code_commit": "c2",
        "generation_timestamp": "2026-10-01T12:05:00Z"
    }
    
    assert digest_manifest(manifest_dict) == digest_manifest(manifest_dict_later)
    
def test_provenance_fields():
    prov = DerivedResultProvenance(
        analysis_id="a1", rq="RQ1", metric="success_rate", eligibility_rule_version="v1",
        source_trial_ids=["t1", "t2"], ineligible_trial_ids=[],
        provider_failure_trial_ids=[], quarantined_trial_ids=[],
        source_result_hash="hash_xy",
        analyzer_version="v2", reporting_version="v3", analysis_code_commit="c_report"
    )
    assert prov.analysis_code_commit == "c_report"
    assert "a1" in str(canonical_serialize(prov))
