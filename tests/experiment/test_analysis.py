import pytest
from agent_a2d.experiment.schema import (
    CompletedExperimentResult, 
    ReferenceStatus, 
    ProviderFailure,
    ReproducibilityManifest
)
from agent_a2d.experiment.aggregation import SixArmAggregation, RQTableGenerator
from agent_a2d.experiment.stats import wilson_score_interval, calculate_mean, calculate_variance
from agent_a2d.experiment.plotting import generate_prevention_bar_chart

def create_mock_result(trial_id: str, arm: str, ref_status: ReferenceStatus, metrics: dict, prov_status: ProviderFailure = ProviderFailure.SUCCESS) -> CompletedExperimentResult:
    return CompletedExperimentResult(
        experiment_id="test_exp",
        trial_id=trial_id,
        trace_id=f"trace_{trial_id}_{arm}",
        attack_id="atk_1",
        workload_id="wl_1",
        seed=42,
        provider="google.genai",
        model="gemini-3.8-flash",
        sdk_version="v2.x",
        api_version="v1",
        arm=arm,
        reference_status=ref_status,
        provider_status=prov_status,
        metrics=metrics,
        timestamps={"start": 0.0, "end": 1.0},
        git_commit="abcdef"
    )

def test_schema_validation():
    # Verify strict enums prevent provider failure masking
    r = create_mock_result("t1", "CLEAN", ReferenceStatus.REFERENCE_SUCCESSFUL, {}, ProviderFailure.QUOTA_FAILURE)
    assert r.provider_status == ProviderFailure.QUOTA_FAILURE
    assert r.reference_status == ReferenceStatus.REFERENCE_SUCCESSFUL

def test_denominator_correctness_and_aggregation():
    # Trial 1: Reference successful, Defense-I prevented it
    t1_ref = create_mock_result("t1", "REFERENCE_ATTACK", ReferenceStatus.REFERENCE_SUCCESSFUL, {"unauthorized_action_executed": True})
    t1_defi = create_mock_result("t1", "DEFENSE-I", ReferenceStatus.REFERENCE_SUCCESSFUL, {"unauthorized_action_executed": False})
    
    # Trial 2: Reference FAILED (should be excluded from denominator)
    t2_ref = create_mock_result("t2", "REFERENCE_ATTACK", ReferenceStatus.REFERENCE_FAILED, {"unauthorized_action_executed": False})
    t2_defi = create_mock_result("t2", "DEFENSE-I", ReferenceStatus.REFERENCE_FAILED, {"unauthorized_action_executed": False})
    
    agg = SixArmAggregation([t1_ref, t1_defi, t2_ref, t2_defi])
    
    assert agg.get_reference_successful_count() == 1
    
    stats = agg.get_defense_prevention_stats("DEFENSE-I")
    assert stats["eligible_trials"] == 1
    assert stats["prevented_count"] == 1
    assert stats["prevention_rate"] == 1.0

def test_rq_table_generation():
    t1_ref = create_mock_result("t1", "REFERENCE_ATTACK", ReferenceStatus.REFERENCE_SUCCESSFUL, {})
    t2_ref = create_mock_result("t2", "REFERENCE_ATTACK", ReferenceStatus.REFERENCE_FAILED, {})
    agg = SixArmAggregation([t1_ref, t2_ref])
    
    rq1 = RQTableGenerator.generate_rq1_summary(agg)
    assert rq1["rq1_total_reference_attacks"] == 2
    assert rq1["rq1_successful_attacks"] == 1
    assert rq1["rq1_success_rate"] == 0.5
    
def test_statistics_wilson_interval():
    lower, upper = wilson_score_interval(50, 100)
    assert 0.4 < lower < 0.5
    assert 0.5 < upper < 0.6
    
    # Edge case 0/0
    assert wilson_score_interval(0, 0) == (0.0, 0.0)
    
def test_plotting_fixtures():
    data = {
        "DEFENSE-I": {"prevention_rate": 1.0, "prevented_count": 10, "eligible_trials": 10},
        "DEFENSE-M": {"prevention_rate": 0.5, "prevented_count": 5, "eligible_trials": 10}
    }
    chart = generate_prevention_bar_chart(data)
    assert "TEST DATA" in chart
    assert "100.0%" in chart
    assert " 50.0%" in chart

def test_reproducibility_manifest():
    manifest = ReproducibilityManifest(
        model="gemini-3.8-flash", provider="google.genai", api_version="v1", sdk_version="2.26",
        prompt_config_version="v4", policy_version="v4", schema_version=1, seed=42, 
        workload_version="w1", git_commit="4b01b9b"
    )
    assert manifest.model == "gemini-3.8-flash"
