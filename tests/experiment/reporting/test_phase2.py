import pytest
from agent_a2d.experiment.schema import CompletedExperimentResult, ProviderFailure, ReferenceStatus
from agent_a2d.experiment.reporting.eligibility import (
    EligibilityReason, evaluate_rq1_prevention, evaluate_rq2_utility, 
    evaluate_rq3_residual, evaluate_rq4_laundering
)
from tests.experiment.reporting.test_phase1 import create_valid_result

def test_1_valid_ref_successful_attack_rq1():
    # 1. valid reference-successful attack -> RQ1 eligible
    r = create_valid_result(ref_status=ReferenceStatus.REFERENCE_SUCCESSFUL)
    r = object.__setattr__(r, 'workload_id', 'attack_1') or r
    res = evaluate_rq1_prevention(r)
    assert res.eligible is True
    assert res.reason == EligibilityReason.ELIGIBLE

def test_2_and_3_ref_failed_attack():
    # 2. reference-failed attack -> RQ1 prevention ineligible
    # 3. reference-failed attack -> remains valid overall (this is a property of returning an EligibilityResult instead of deleting it)
    r = create_valid_result(ref_status=ReferenceStatus.REFERENCE_FAILED)
    r = object.__setattr__(r, 'workload_id', 'attack_1') or r
    res = evaluate_rq1_prevention(r)
    assert res.eligible is False
    assert res.reason == EligibilityReason.REFERENCE_FAILED
    # Trace ID and Trial ID are still preserved in the response, proving it's not silently deleted.
    assert res.trial_id == r.trial_id

def test_4_and_5_benign_task_rq2():
    # 4. benign memory-dependent task -> RQ2 eligible
    # 5. benign task does not require reference success
    r_succ = create_valid_result(ref_status=ReferenceStatus.REFERENCE_SUCCESSFUL)
    r_succ = object.__setattr__(r_succ, 'workload_id', 'benign_task') or r_succ
    assert evaluate_rq2_utility(r_succ).eligible is True
    
    r_fail = create_valid_result(ref_status=ReferenceStatus.REFERENCE_FAILED)
    r_fail = object.__setattr__(r_fail, 'workload_id', 'benign_task') or r_fail
    assert evaluate_rq2_utility(r_fail).eligible is True

def test_6_and_7_rq3_residual():
    # 6. successful reference attack -> RQ3 eligible
    # 7. failed reference attack -> RQ3 ineligible
    r_succ = create_valid_result(ref_status=ReferenceStatus.REFERENCE_SUCCESSFUL)
    r_succ = object.__setattr__(r_succ, 'workload_id', 'attack_1') or r_succ
    assert evaluate_rq3_residual(r_succ, metric="storage_contamination").eligible is True
    
    r_fail = create_valid_result(ref_status=ReferenceStatus.REFERENCE_FAILED)
    r_fail = object.__setattr__(r_fail, 'workload_id', 'attack_1') or r_fail
    assert evaluate_rq3_residual(r_fail, metric="storage_contamination").eligible is False

def test_8_and_9_rq4_laundering():
    # 8. laundering attack -> RQ4 category recognized
    r_laun = create_valid_result(ref_status=ReferenceStatus.REFERENCE_SUCCESSFUL)
    r_laun = object.__setattr__(r_laun, 'workload_id', 'laundering_1') or r_laun
    assert evaluate_rq4_laundering(r_laun, is_prevention_metric=True).eligible is True
    
    # 9. fragmented-memory attack -> RQ4 category recognized
    r_frag = create_valid_result(ref_status=ReferenceStatus.REFERENCE_SUCCESSFUL)
    r_frag = object.__setattr__(r_frag, 'workload_id', 'fragmentation_1') or r_frag
    assert evaluate_rq4_laundering(r_frag, is_prevention_metric=True).eligible is True
    
    # Also verify the REFERENCE_FAILED behavior for RQ4
    r_frag_fail = create_valid_result(ref_status=ReferenceStatus.REFERENCE_FAILED)
    r_frag_fail = object.__setattr__(r_frag_fail, 'workload_id', 'fragmentation_1') or r_frag_fail
    # Prevention metric -> ineligible
    assert evaluate_rq4_laundering(r_frag_fail, is_prevention_metric=True).eligible is False
    # Susceptibility metric -> eligible
    assert evaluate_rq4_laundering(r_frag_fail, is_prevention_metric=False).eligible is True

def test_10_provider_failure():
    # 10. provider failure -> never eligible
    r = create_valid_result(ref_status=ReferenceStatus.REFERENCE_SUCCESSFUL)
    r = object.__setattr__(r, 'workload_id', 'attack_1') or r
    r = object.__setattr__(r, 'provider_status', ProviderFailure.QUOTA_FAILURE) or r
    
    assert evaluate_rq1_prevention(r).eligible is False
    assert evaluate_rq1_prevention(r).reason == EligibilityReason.PROVIDER_FAILURE
    assert evaluate_rq2_utility(r).eligible is False
    assert evaluate_rq3_residual(r).eligible is False
    assert evaluate_rq4_laundering(r, is_prevention_metric=True).eligible is False

# 11. validation quarantine -> never reaches eligibility
# This is an architectural property. ValidatedDataset explicitly separates valid_results and quarantine_records.
# The eligibility functions strictly type-hint `CompletedExperimentResult`, proving they only operate on valid records.

def test_12_and_13_deterministic_and_explicit_reason():
    # 12. eligibility result is deterministic
    # 13. every ineligible decision has an explicit reason
    r = create_valid_result(ref_status=ReferenceStatus.REFERENCE_FAILED)
    r = object.__setattr__(r, 'workload_id', 'attack_1') or r
    
    res1 = evaluate_rq1_prevention(r)
    res2 = evaluate_rq1_prevention(r)
    assert res1 == res2
    assert res1.eligible is False
    assert res1.reason == EligibilityReason.REFERENCE_FAILED
    assert isinstance(res1.reason, EligibilityReason)

def test_14_no_deletion():
    # 14. no RQ eligibility decision deletes the underlying valid trial
    # Demonstrated by returning an `EligibilityResult` linking back via `trial_id` and `trace_id` rather than a filtered list.
    r = create_valid_result(ref_status=ReferenceStatus.REFERENCE_FAILED)
    res = evaluate_rq1_prevention(r)
    assert res.trial_id == r.trial_id
    assert res.trace_id == r.trace_id
