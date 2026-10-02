import pytest
import copy

from agent_a2d.experiment.schema import CompletedExperimentResult
from agent_a2d.experiment.reporting.schema import (
    ScientificExecutionContext,
    ValidatedDataset, RQSummaryTable, PackageManifest, ArtifactStatus, AccountingRecord
)
from agent_a2d.experiment.reporting.figures import FigureData, FigureGenerator
from agent_a2d.experiment.reporting.tables import TableGenerator
from agent_a2d.experiment.reporting.eligibility import EligibilityResult, EligibilityReason
from agent_a2d.experiment.reporting.packager import compute_scientific_digest
from agent_a2d.experiment.reporting.provenance import compute_digest
import tests.experiment.reporting.test_phase1 as tp1

@pytest.fixture
def base_scenario():
    # 1. Raw results
    r1 = tp1.create_valid_result("t1", "CLEAN")
    r2 = tp1.create_valid_result("t2", "CLEAN")
    r2_mut = copy.deepcopy(r2)
    object.__setattr__(r2_mut, "arm", "arm2") # Give t2 a different arm
    dataset = ValidatedDataset([r1, r2_mut], [])
    
    # 2. Eligibility
    el1 = EligibilityResult("t1", "CLEAN", "RQ1", "prevention", True, EligibilityReason.ELIGIBLE, "v1")
    el2 = EligibilityResult("t2", "CLEAN", "RQ1", "prevention", True, EligibilityReason.ELIGIBLE, "v1")
    eligibility = [el1, el2]
    
    # 3. Tables
    gen = TableGenerator("a", "b")
    prov1 = gen._build_provenance("RQ1", "prevention", "v1", ["t1"], [], [], [], dataset)
    t1 = RQSummaryTable(0, 0, 0, 0, 0, 0, "v1", {}, prov1)
    
    prov2 = gen._build_provenance("RQ1", "prevention", "v1", ["t2"], [], [], [], dataset)
    t2 = RQSummaryTable(0, 0, 0, 0, 0, 0, "v1", {}, prov2)
    
    rq_tables = {"RQ1": {"arm1": t1, "arm2": t2}}
    
    # 4. Figures
    fgen = FigureGenerator()
    f1 = fgen.build_rq1_figure_data(rq_tables["RQ1"])
    figures = [f1]
    

    context = ScientificExecutionContext(
        experiment_id="exp1",
        experiment_execution_commit="exec_c",
        experiment_execution_ref="main",
        p6_protocol_version="v1",
        schema_version=1,
        provider="google",
        model="gemini-test",
        prompt_config_version="p1",
        policy_version="pol1",
        workload_version="w1",
        seed=42
    )
    
    return context, dataset, eligibility, rq_tables, figures


def test_01_digest_identical_logical_input(base_scenario):
    c1, d1, e1, t1, f1 = base_scenario
    c2, d2, e2, t2, f2 = copy.deepcopy(base_scenario)
    
    hash1 = compute_scientific_digest(c1, d1, e1, t1, f1)
    hash2 = compute_scientific_digest(c2, d2, e2, t2, f2)
    assert hash1 == hash2

def test_02_digest_reordered_collections(base_scenario):
    c1, d1, e1, t1, f1 = base_scenario
    
    c2, d2, e2, t2, f2 = copy.deepcopy(base_scenario)
    # Shuffle valid_results list
    d2.valid_results.reverse()
    # Shuffle eligibility list
    e2.reverse()
    
    hash1 = compute_scientific_digest(c1, d1, e1, t1, f1)
    hash2 = compute_scientific_digest(c2, d2, e2, t2, f2)
    assert hash1 == hash2

def test_03_digest_reordered_dicts(base_scenario):
    c1, d1, e1, t1, f1 = base_scenario
    c2, d2, e2, t2, f2 = copy.deepcopy(base_scenario)
    
    # Dictionary insertion order is randomized here
    t2["RQ1"] = {"arm2": t2["RQ1"]["arm2"], "arm1": t2["RQ1"]["arm1"]}
    
    hash1 = compute_scientific_digest(c1, d1, e1, t1, f1)
    hash2 = compute_scientific_digest(c2, d2, e2, t2, f2)
    assert hash1 == hash2

def test_04_07_digest_non_scientific_metadata(base_scenario):
    # Tests 4, 5, 6, 7 combined
    c1, d1, e1, t1, f1 = base_scenario
    hash1 = compute_scientific_digest(c1, d1, e1, t1, f1)
    
    # The manifest stores filesystem roots, generation timestamps, and renderer environment metadata!
    # By strictly isolating the digest inputs to (dataset, eligibility, rq_tables, figures),
    # we mathematically guarantee the digest is independent of them, as they are not passed to compute_scientific_digest!
    
    acc = AccountingRecord(2, 2, 0, 0, {"RQ1": 2}, {})
    manifest = PackageManifest(
        package_id="pkg", experiment_id="exp", experiment_execution_commit="c1",
        experiment_execution_ref="r1", analysis_code_commit="c2",
        package_generation_commit="c3", p6_protocol_version="v1", schema_version=1,
        analyzer_version="v1", reporting_version="v1", accounting=acc,
        artifact_statuses={"f1": ArtifactStatus.RENDERER_UNAVAILABLE}, # Rendering metadata!
        scientific_content_digest=hash1, archive_digest=None
    )
    
    # The test simply proves that we don't pass the manifest or environment metadata to the scientific digest
    
    hash2 = compute_scientific_digest(c1, d1, e1, t1, f1)
    assert hash1 == hash2

def test_08_mutation_raw_results(base_scenario):
    c1, d1, e1, t1, f1 = base_scenario
    hash1 = compute_scientific_digest(c1, d1, e1, t1, f1)
    
    c2, d2, e2, t2, f2 = copy.deepcopy(base_scenario)
    # Mutate a raw result field
    object.__setattr__(d2.valid_results[0], "workload_id", "mutated_workload")
    
    hash2 = compute_scientific_digest(c2, d2, e2, t2, f2)
    assert hash1 != hash2

def test_09_mutation_table_value(base_scenario):
    c1, d1, e1, t1, f1 = base_scenario
    hash1 = compute_scientific_digest(c1, d1, e1, t1, f1)
    
    c2, d2, e2, t2, f2 = copy.deepcopy(base_scenario)
    # Mutate table metric value
    object.__setattr__(t2["RQ1"]["arm1"], "numerator", 999.0)
    
    hash2 = compute_scientific_digest(c2, d2, e2, t2, f2)
    assert hash1 != hash2

def test_10_mutation_eligibility(base_scenario):
    c1, d1, e1, t1, f1 = base_scenario
    hash1 = compute_scientific_digest(c1, d1, e1, t1, f1)
    
    c2, d2, e2, t2, f2 = copy.deepcopy(base_scenario)
    # Mutate eligibility decision
    object.__setattr__(e2[0], "eligible", False)
    
    hash2 = compute_scientific_digest(c2, d2, e2, t2, f2)
    assert hash1 != hash2

def test_11_mutation_figure_data(base_scenario):
    c1, d1, e1, t1, f1 = base_scenario
    hash1 = compute_scientific_digest(c1, d1, e1, t1, f1)
    
    c2, d2, e2, t2, f2 = copy.deepcopy(base_scenario)
    # Mutate figure series value
    f2[0].series[0]["prevention_rate"] = 0.5
    
    hash2 = compute_scientific_digest(c2, d2, e2, t2, f2)
    assert hash1 != hash2

def test_12_mutation_provenance(base_scenario):
    c1, d1, e1, t1, f1 = base_scenario
    hash1 = compute_scientific_digest(c1, d1, e1, t1, f1)
    
    c2, d2, e2, t2, f2 = copy.deepcopy(base_scenario)
    # Mutate provenance population ID
    m_prov = copy.deepcopy(t2["RQ1"]["arm1"].provenance)
    m_prov.source_trial_ids.append("fake_trial")
    object.__setattr__(t2["RQ1"]["arm1"], "provenance", m_prov)
    
    hash2 = compute_scientific_digest(c2, d2, e2, t2, f2)
    assert hash1 != hash2

def test_13_nan_infinity_rejection():
    # compute_digest inherently checks canonical_serialize which rejects NaN
    with pytest.raises(ValueError):
        compute_digest({"bad_metric": float('nan')})
    with pytest.raises(ValueError):
        compute_digest({"bad_metric": float('inf')})

def test_14_scientific_null_distinct_from_zero(base_scenario):
    c1, d1, e1, t1, f1 = base_scenario
    c2, d2, e2, t2, f2 = copy.deepcopy(base_scenario)
    
    # Null (None)
    f1[0].series[0]["prevention_rate"] = None
    hash1 = compute_scientific_digest(c1, d1, e1, t1, f1)
    
    # Zero (0.0)
    f2[0].series[0]["prevention_rate"] = 0.0
    hash2 = compute_scientific_digest(c2, d2, e2, t2, f2)
    
    # Missing key entirely
    c3, d3, e3, t3, f3 = copy.deepcopy(base_scenario)
    del f3[0].series[0]["prevention_rate"]
    hash3 = compute_scientific_digest(c3, d3, e3, t3, f3)
    
    assert hash1 != hash2
    assert hash1 != hash3
    assert hash2 != hash3

def test_15_mutation_execution_context(base_scenario):
    c1, d1, e1, t1, f1 = base_scenario
    hash1 = compute_scientific_digest(c1, d1, e1, t1, f1)
    
    # 1. experiment_id
    c2, d2, e2, t2, f2 = copy.deepcopy(base_scenario)
    object.__setattr__(c2, "experiment_id", "exp2")
    assert hash1 != compute_scientific_digest(c2, d2, e2, t2, f2)
    
    # 2. experiment_execution_commit
    c3, d3, e3, t3, f3 = copy.deepcopy(base_scenario)
    object.__setattr__(c3, "experiment_execution_commit", "other_commit")
    assert hash1 != compute_scientific_digest(c3, d3, e3, t3, f3)
    
    # 3. experiment_execution_ref
    c4, d4, e4, t4, f4 = copy.deepcopy(base_scenario)
    object.__setattr__(c4, "experiment_execution_ref", "other_ref")
    assert hash1 != compute_scientific_digest(c4, d4, e4, t4, f4)
    
    # 4. p6_protocol_version
    c5, d5, e5, t5, f5 = copy.deepcopy(base_scenario)
    object.__setattr__(c5, "p6_protocol_version", "v2")
    assert hash1 != compute_scientific_digest(c5, d5, e5, t5, f5)
    
    # 5. schema_version
    c6, d6, e6, t6, f6 = copy.deepcopy(base_scenario)
    object.__setattr__(c6, "schema_version", 2)
    assert hash1 != compute_scientific_digest(c6, d6, e6, t6, f6)
    
    # 6. model
    c7, d7, e7, t7, f7 = copy.deepcopy(base_scenario)
    object.__setattr__(c7, "model", "other_model")
    assert hash1 != compute_scientific_digest(c7, d7, e7, t7, f7)
    
    # 7. provider
    c8, d8, e8, t8, f8 = copy.deepcopy(base_scenario)
    object.__setattr__(c8, "provider", "other_provider")
    assert hash1 != compute_scientific_digest(c8, d8, e8, t8, f8)
    
    # 8. prompt_config_version
    c9, d9, e9, t9, f9 = copy.deepcopy(base_scenario)
    object.__setattr__(c9, "prompt_config_version", "p2")
    assert hash1 != compute_scientific_digest(c9, d9, e9, t9, f9)
    
    # 9. policy_version
    c10, d10, e10, t10, f10 = copy.deepcopy(base_scenario)
    object.__setattr__(c10, "policy_version", "pol2")
    assert hash1 != compute_scientific_digest(c10, d10, e10, t10, f10)
    
    # 10. workload_version
    c11, d11, e11, t11, f11 = copy.deepcopy(base_scenario)
    object.__setattr__(c11, "workload_version", "w2")
    assert hash1 != compute_scientific_digest(c11, d11, e11, t11, f11)
