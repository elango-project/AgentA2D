import pytest
import os
import json
from agent_a2d.experiment.schema import CompletedExperimentResult, ProviderFailure, ReferenceStatus
from agent_a2d.experiment.reporting.schema import ValidatedDataset
from agent_a2d.experiment.reporting.eligibility import (
    evaluate_rq1_prevention, evaluate_rq2_utility, evaluate_rq3_residual, evaluate_rq4_laundering
)
from agent_a2d.experiment.reporting.tables import TableGenerator
from agent_a2d.experiment.reporting.figures import FigureGenerator, FigureData
from agent_a2d.experiment.reporting.provenance import canonical_serialize, compute_digest
from tests.experiment.reporting.test_phase1 import create_valid_result

@pytest.fixture
def sample_tables():
    # Construct an integration-style dataset to feed into tables, then to figures
    r1 = create_valid_result("t1", arm="CLEAN")
    r1 = object.__setattr__(r1, 'workload_id', 'attack_1') or r1
    r1.metrics["unauthorized_action_executed"] = False
    
    r2 = create_valid_result("t2", arm="DEFENSE-I")
    r2 = object.__setattr__(r2, 'workload_id', 'benign_1') or r2
    r2.metrics["recovery_state"] = "benign_completion"
    
    # zero denominator for RQ1 because it's benign workload -> ineligible
    
    r3 = create_valid_result("t3", arm="DEFENSE-M", ref_status=ReferenceStatus.REFERENCE_SUCCESSFUL)
    r3 = object.__setattr__(r3, 'workload_id', 'attack_1') or r3
    r3.metrics["unauthorized_action_executed"] = True
    r3.metrics["storage_contamination"] = True
    r3.metrics["recovery_state"] = "compromised"
    
    r4 = create_valid_result("t4", arm="DEFENSE-R")
    r4 = object.__setattr__(r4, 'workload_id', 'laundering_1') or r4
    r4.metrics["unauthorized_action_executed"] = False
    
    dataset = ValidatedDataset(valid_results=[r1, r2, r3, r4], quarantine_records=[])
    
    el1 = [evaluate_rq1_prevention(r) for r in [r1, r2, r3, r4]]
    el2 = [evaluate_rq2_utility(r) for r in [r1, r2, r3, r4]]
    
    # 6 metrics for rq3
    el3 = []
    for m in ["storage_contamination", "retrieval_contamination", "behavioral_contamination", 
              "malicious_tool_request", "unauthorized_action_executed", "recovery_state"]:
        el3.extend([evaluate_rq3_residual(r, m) for r in [r1, r2, r3, r4]])
        
    el4 = []
    for m in [True, False]:
        el4.extend([evaluate_rq4_laundering(r, m) for r in [r1, r2, r3, r4]])
        
    gen = TableGenerator("a1", "c1")
    return {
        "rq1": gen.generate_rq1_table(dataset, el1),
        "rq2": gen.generate_rq2_table(dataset, el2),
        "rq3": gen.generate_rq3_table(dataset, el3),
        "rq4": gen.generate_rq4_table(dataset, el4)
    }

def test_1_and_2_consumes_outputs_no_filtering(sample_tables):
    # 1. consumes phase 3 outputs (type logic)
    # 2. no independent filtering
    fg = FigureGenerator()
    fig1 = fg.build_rq1_figure_data(sample_tables["rq1"])
    # Should include all arms that were in the tables, even if zero denominator
    arms_in_tables = set(sample_tables["rq1"].keys())
    arms_in_fig = {s["arm"] for s in fig1.series}
    assert arms_in_tables == arms_in_fig

def test_3_deterministic_category_ordering(sample_tables):
    # 3. deterministic category ordering
    fg = FigureGenerator()
    fig1 = fg.build_rq1_figure_data(sample_tables["rq1"])
    arms = [s["arm"] for s in fig1.series]
    assert arms == sorted(arms)

def test_4_and_5_zero_denominators_and_numerators(sample_tables):
    # 4. zero denominator
    # 5. zero numerator
    fg = FigureGenerator()
    fig1 = fg.build_rq1_figure_data(sample_tables["rq1"])
    
    # t2 (DEFENSE-I) is benign, so for RQ1 it is ineligible -> zero denominator
    defense_i_series = next(s for s in fig1.series if s["arm"] == "DEFENSE-I")
    assert defense_i_series["total_eligible"] == 0
    assert defense_i_series["prevention_rate"] is None
    
    # t3 (DEFENSE-M) is an attack that succeeded (unauthorized_action_executed = True), so prevention numerator is 0
    defense_m_series = next(s for s in fig1.series if s["arm"] == "DEFENSE-M")
    assert defense_m_series["total_eligible"] == 1
    assert defense_m_series["prevented_count"] == 0
    assert defense_m_series["prevention_rate"] == 0.0

def test_6_and_7_rq3_separation_and_categorical(sample_tables):
    # 6. RQ3 six-metric separation
    # 7. recovery-state categorical plotting
    fg = FigureGenerator()
    figs = fg.build_rq3_figure_data(sample_tables["rq3"])
    
    assert len(figs) == 6
    metrics = {f.metric for f in figs}
    assert "recovery_state" in metrics
    assert "storage_contamination" in metrics
    
    rec_fig = next(f for f in figs if f.metric == "recovery_state")
    
    # Check categorical structure
    assert "distribution" in rec_fig.series[0]

def test_8_rq4_separation(sample_tables):
    # 8. RQ4 separation
    fg = FigureGenerator()
    figs = fg.build_rq4_figure_data(sample_tables["rq4"])
    # 1 category (laundering_1) * 2 metrics = 2 figures
    # (fragmentation was not in sample_tables)
    # wait, TableGenerator loops over known categories "laundering", "fragmentation" 
    # even if empty, it creates them. 
    # Let's check length. 2 categories * 2 metrics = 4 figures
    assert len(figs) == 4
    for f in figs:
        assert f.rq == "RQ4"
        assert f.metric in ["laundering_prevention", "laundering_susceptibility", "fragmentation_prevention", "fragmentation_susceptibility"]

def test_9_synthetic_labeling(sample_tables):
    # 9. synthetic-data labeling
    fg = FigureGenerator(is_synthetic=True)
    fig1 = fg.build_rq1_figure_data(sample_tables["rq1"])
    assert "[SYNTHETIC TEST DATA]" in fig1.title
    assert fig1.synthetic_test_data is True

def test_10_provenance_propagation(sample_tables):
    # 10. provenance propagation
    fg = FigureGenerator()
    fig1 = fg.build_rq1_figure_data(sample_tables["rq1"])
    prov = fig1.merged_provenance
    assert prov.analysis_id == "a1"
    assert prov.analysis_code_commit == "c1"
    # Should include all trials from constituent tables
    assert "t1" in prov.source_trial_ids
    assert "t3" in prov.source_trial_ids

def test_11_deterministic_serialization(sample_tables):
    # 11. deterministic figure-data serialization
    fg = FigureGenerator()
    fig1 = fg.build_rq1_figure_data(sample_tables["rq1"])
    fig1_copy = fg.build_rq1_figure_data(sample_tables["rq1"])
    
    # Must be perfectly byte identical
    assert compute_digest(fig1) == compute_digest(fig1_copy)

def test_12_and_13_and_14_neutrality_and_rq2_terminology(sample_tables):
    # 12. no conclusion/ranking text
    # 13. completion_proxy_rate terminology
    # 14. limitation metadata for RQ2
    fg = FigureGenerator()
    fig2 = fg.build_rq2_figure_data(sample_tables["rq2"])
    
    assert "completion_proxy" in fig2.metric
    assert fig2.limitation_note is not None
    assert "semantic quality" in fig2.limitation_note
    
    for s in fig2.series:
        assert "completion_proxy_rate" in s
        
    s_bytes = canonical_serialize(fig2)
    s_str = s_bytes.decode('utf-8').lower()
    for bad_word in ["winner", "best", "ranking", "significance"]:
        assert bad_word not in s_str

def test_15_provenance_source_hash_binds_to_content():
    # Phase 3 provenance defect fix: Check that modifying a result changes the hash.
    from agent_a2d.experiment.schema import CompletedExperimentResult
    r1 = create_valid_result("t1", arm="DEFENSE-M", ref_status=ReferenceStatus.REFERENCE_SUCCESSFUL)
    r1 = object.__setattr__(r1, 'workload_id', 'attack_1') or r1
    r1.metrics["unauthorized_action_executed"] = False
    
    r1_mod = create_valid_result("t1", arm="DEFENSE-M", ref_status=ReferenceStatus.REFERENCE_SUCCESSFUL)
    r1_mod = object.__setattr__(r1_mod, 'workload_id', 'attack_1') or r1_mod
    r1_mod.metrics["unauthorized_action_executed"] = True
    
    dataset1 = ValidatedDataset(valid_results=[r1], quarantine_records=[])
    dataset2 = ValidatedDataset(valid_results=[r1_mod], quarantine_records=[])
    
    el1 = [evaluate_rq1_prevention(r1)]
    el2 = [evaluate_rq1_prevention(r1_mod)]
    
    gen = TableGenerator("a", "b")
    t1 = gen.generate_rq1_table(dataset1, el1)["DEFENSE-M"]
    t2 = gen.generate_rq1_table(dataset2, el2)["DEFENSE-M"]
    
    # Hash MUST change if the underlying data changed
    assert t1.provenance.source_result_hash != t2.provenance.source_result_hash

def test_16_composite_provenance_preserves_hashes(sample_tables):
    # Phase 4 provenance defect fix: Composition hash.
    fg = FigureGenerator()
    fig1 = fg.build_rq1_figure_data(sample_tables["rq1"])
    
    # If we modify a constituent hash, the composite hash MUST change
    mod_tables = dict(sample_tables["rq1"])
    t_clean = mod_tables["CLEAN"]
    
    import dataclasses
    from agent_a2d.experiment.reporting.schema import DerivedResultProvenance
    # Create a forged table with a different source hash
    mod_prov = dataclasses.replace(t_clean.provenance, source_result_hash="forged_hash_123")
    mod_table = dataclasses.replace(t_clean, provenance=mod_prov)
    mod_tables["CLEAN"] = mod_table
    
    fig_mod = fg.build_rq1_figure_data(mod_tables)
    
    assert fig1.merged_provenance.source_result_hash != fig_mod.merged_provenance.source_result_hash

def test_17_figure_data_environment_independence(sample_tables):
    fg1 = FigureGenerator()
    fig1 = fg1.build_rq1_figure_data(sample_tables["rq1"])
    
    # RenderMetadata should be accessible on fg1, but not in FigureData
    assert not hasattr(fig1, "env_metadata")
    assert fg1.render_metadata.python_version is not None
    
    # Serialized form must not contain 'matplotlib' version
    serialized = canonical_serialize(fig1).decode('utf-8')
    assert "matplotlib" not in serialized

def test_18_none_rendering_distinction(sample_tables, tmp_path):
    # None/zero rendering defect fix: Ensure None is distinct from 0.0
    fg = FigureGenerator()
    fig1 = fg.build_rq1_figure_data(sample_tables["rq1"])
    
    # Render the figure to a temporary path
    out_path = tmp_path / "test_fig.png"
    fg.render_figure(fig1, str(out_path))
    
    # We just ensure the data correctly distinguishes None from 0.0
    # In rq1, DEFENSE-I is ineligible, so prevention_rate is None
    defense_i_series = next(s for s in fig1.series if s["arm"] == "DEFENSE-I")
    assert defense_i_series["prevention_rate"] is None
    
    # DEFENSE-M has 0 prevented attacks out of 1
    defense_m_series = next(s for s in fig1.series if s["arm"] == "DEFENSE-M")
    assert defense_m_series["prevention_rate"] == 0.0
    
    # The actual plot logic in render_figure maps None to N/A text and doesn't mutate the data
    assert defense_i_series["prevention_rate"] is not defense_m_series["prevention_rate"]

