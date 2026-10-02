import pytest
from agent_a2d.experiment.schema import CompletedExperimentResult
from agent_a2d.experiment.reporting.schema import ValidatedDataset, RQSummaryTable, PackageManifest, AccountingRecord
from agent_a2d.experiment.reporting.packager import PackagingValidationGate
from agent_a2d.experiment.reporting.figures import FigureGenerator, FigureData
from agent_a2d.experiment.reporting.tables import TableGenerator
from agent_a2d.experiment.reporting.schema import DerivedResultProvenance
from agent_a2d.experiment.reporting.provenance import compute_digest
import copy
import tests.experiment.reporting.test_phase1 as tp1

def get_base_manifest():
    acc = AccountingRecord(2, 2, 0, 0, {"RQ1": 2}, {})
    return PackageManifest("pkg", "exp", "c1", "r1", "c2", "c3", "v1", 1, "v1", "v1", acc, {}, "")

def test_defect1_source_result_hash_validation():
    gate = PackagingValidationGate()
    
    # 1. Identical source content -> validation passes
    r1 = tp1.create_valid_result("t1", "CLEAN")
    r2 = tp1.create_valid_result("t2", "CLEAN")
    dataset = ValidatedDataset([r1, r2], [])
    
    # Build provenance manually or with TableGenerator
    from agent_a2d.experiment.reporting.tables import TableGenerator
    gen = TableGenerator("a", "b")
    prov = gen._build_provenance("RQ1", "prevention", "v1", ["t1", "t2"], [], [], [], dataset)
    t = RQSummaryTable(0, 0, 0, 0, 0, 0, "v1", {}, prov)
    
    # Passes
    gate.validate_inputs(dataset, {"arm": t}, [], get_base_manifest())
    
    # 2. Reorder source results -> validation still passes
    dataset_reordered = ValidatedDataset([r2, r1], [])
    gate.validate_inputs(dataset_reordered, {"arm": t}, [], get_base_manifest())
    
    # 3. Change a source result field -> validation fails
    r1_mutated = copy.deepcopy(r1)
    object.__setattr__(r1_mutated, "arm", "MUTATED")
    dataset_mutated = ValidatedDataset([r1_mutated, r2], [])
    from agent_a2d.experiment.reporting.schema import PackagingIntegrityError
    with pytest.raises(PackagingIntegrityError, match="Source hash mismatch"):
        gate.validate_inputs(dataset_mutated, {"arm": t}, [], get_base_manifest())
        
    # 4. Alter content while keeping the same trial ID -> validation fails
    r1_mutated_content = copy.deepcopy(r1)
    m = dict(r1_mutated_content.metrics)
    m["new_metric"] = 42
    object.__setattr__(r1_mutated_content, "metrics", m)
    dataset_mutated_content = ValidatedDataset([r1_mutated_content, r2], [])
    with pytest.raises(PackagingIntegrityError, match="Source hash mismatch"):
        gate.validate_inputs(dataset_mutated_content, {"arm": t}, [], get_base_manifest())
        
    # 5. Alter only the stored source hash -> validation fails
    prov_bad = copy.deepcopy(prov)
    object.__setattr__(prov_bad, "source_result_hash", "bad_hash")
    t_bad = RQSummaryTable(0, 0, 0, 0, 0, 0, "v1", {}, prov_bad)
    with pytest.raises(PackagingIntegrityError, match="Source hash mismatch"):
        gate.validate_inputs(dataset, {"arm": t_bad}, [], get_base_manifest())

def test_defect2_figure_composite_hash_validation():
    gate = PackagingValidationGate()
    r1 = tp1.create_valid_result("t1", "CLEAN")
    dataset = ValidatedDataset([r1], [])
    
    gen = TableGenerator("a", "b")
    prov1 = gen._build_provenance("RQ1", "prevention", "v1", ["t1"], [], [], [], dataset)
    t1 = RQSummaryTable(0, 0, 0, 0, 0, 0, "v1", {}, prov1)
    
    prov2 = gen._build_provenance("RQ2", "utility", "v1", ["t1"], [], [], [], dataset)
    t2 = RQSummaryTable(0, 0, 0, 0, 0, 0, "v1", {}, prov2)
    
    rq_tables = {"RQ1": {"arm1": t1}, "RQ2": {"arm1": t2}}
    
    # Generate figure using FigureGenerator
    fgen = FigureGenerator()
    f1 = fgen.build_rq1_figure_data({"arm1": t1})
    
    # 1. Valid figure + matching tables -> passes
    gate.validate_inputs(dataset, rq_tables, [f1], get_base_manifest())
    
    # 2. Modify one constituent table source hash -> fails
    # If the table has a different hash, the figure's constituent list won't match anything
    t1_bad = copy.deepcopy(t1)
    object.__setattr__(t1_bad.provenance, "source_result_hash", "bad")
    from agent_a2d.experiment.reporting.schema import PackagingIntegrityError
    with pytest.raises(PackagingIntegrityError, match="Source hash mismatch"):
        gate.validate_inputs(dataset, {"RQ1": {"arm1": t1_bad}}, [f1], get_base_manifest())
        
    f1_bad = copy.deepcopy(f1)
    object.__setattr__(f1_bad, "constituent_table_identities", [{"rq": "bad", "metric": "bad", "source_result_hash": "bad"}])
    with pytest.raises(PackagingIntegrityError, match="references unknown constituent table"):
        gate.validate_inputs(dataset, rq_tables, [f1_bad], get_base_manifest())
        
    # 3. Remove a constituent table -> fails
    with pytest.raises(PackagingIntegrityError, match="references unknown constituent table"):
        gate.validate_inputs(dataset, {"RQ2": {"arm1": t2}}, [f1], get_base_manifest())
        
    # 4. Replace a table with an unrelated table having overlapping trial IDs -> fails
    r2 = tp1.create_valid_result("t2", "CLEAN")
    dataset_2 = ValidatedDataset([r1, r2], [])
    prov3 = gen._build_provenance("RQ1", "prevention", "v1", ["t1", "t2"], [], [], [], dataset_2)
    t3 = RQSummaryTable(0, 0, 0, 0, 0, 0, "v1", {}, prov3)
    with pytest.raises(PackagingIntegrityError, match="unknown constituent table"):
        gate.validate_inputs(dataset_2, {"RQ1": {"arm1": t3}}, [f1], get_base_manifest())
