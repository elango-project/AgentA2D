import pytest
from typing import Dict
from agent_a2d.experiment.schema import CompletedExperimentResult, ProviderFailure
from agent_a2d.experiment.reporting.schema import (
    PackageManifest, AccountingRecord, ArtifactStatus,
    ValidatedDataset, DerivedResultProvenance, PackagingIntegrityError
)
from agent_a2d.experiment.reporting.packager import PackagingValidationGate
from agent_a2d.experiment.reporting.provenance import canonical_serialize

def test_01_manifest_and_enums():
    """Test schema, execution ref separation, three commits, enums."""
    acc = AccountingRecord(
        total_input=10, valid=8, quarantine=2, provider_failures=1,
        rq_eligible={"RQ1": 5}, rq_ineligible={"RQ1": 2}
    )
    
    # 3-commit provenance + separate execution ref
    manifest = PackageManifest(
        package_id="pkg-1",
        experiment_id="exp-1",
        experiment_execution_commit="c1",
        experiment_execution_ref="branch-run-1",
        analysis_code_commit="c2",
        package_generation_commit="c3",
        p6_protocol_version="v1",
        schema_version=1,
        analyzer_version="1.0",
        reporting_version="1.0",
        accounting=acc,
        artifact_statuses={"fig1": ArtifactStatus.RENDERER_UNAVAILABLE},
        scientific_content_digest="hash123",
        archive_digest=None
    )
    
    assert manifest.experiment_execution_ref == "branch-run-1"
    assert manifest.artifact_statuses["fig1"] == ArtifactStatus.RENDERER_UNAVAILABLE

def test_02_validation_gate_success():
    pass

def test_03_unknown_trial_rejection():
    gate = PackagingValidationGate()
    dataset = ValidatedDataset(valid_results=[], quarantine_records=[])
    
    # Create a table with unknown trial
    from agent_a2d.experiment.reporting.schema import RQSummaryTable
    prov = DerivedResultProvenance(
        analysis_id="a1", rq="RQ1", metric="prevention", eligibility_rule_version="v1",
        source_trial_ids=["fake_trial_id"], ineligible_trial_ids=[],
        provider_failure_trial_ids=[], quarantined_trial_ids=[],
        source_result_hash="abc", analyzer_version="v1", reporting_version="v1", analysis_code_commit="c"
    )
    t = RQSummaryTable(0, 0, 0, 0, 0, 0, "v1", {}, prov)
    
    acc = AccountingRecord(0, 0, 0, 0, {}, {})
    manifest = PackageManifest("", "", "", "", "", "", "v1", 1, "", "", acc, {}, "")
    
    with pytest.raises(PackagingIntegrityError, match="unknown trial IDs"):
        gate.validate_inputs(dataset, {"arm1": t}, [], manifest)
        
def test_04_orphan_derived_artifact_rejection():
    gate = PackagingValidationGate()
    dataset = ValidatedDataset(valid_results=[], quarantine_records=[])
    
    from agent_a2d.experiment.reporting.figures import FigureData
    prov = DerivedResultProvenance(
        analysis_id="a1", rq="RQ1", metric="prevention", eligibility_rule_version="v1",
        source_trial_ids=["t1"], ineligible_trial_ids=[],
        provider_failure_trial_ids=[], quarantined_trial_ids=[],
        source_result_hash="abc", analyzer_version="v1", reporting_version="v1", analysis_code_commit="c"
    )
    
    # Add trial to valid_results so it's not "unknown trial"
    import tests.experiment.reporting.test_phase1 as tp1
    t1 = tp1.create_valid_result("t1", "CLEAN")
    dataset = ValidatedDataset(valid_results=[t1], quarantine_records=[])
    
    f = FigureData("f1", "title", "RQ1", "prevention", False, [], prov, None)
    acc = AccountingRecord(1, 1, 0, 0, {}, {})
    manifest = PackageManifest("", "", "", "", "", "", "v1", 1, "", "", acc, {}, "")
    
    # Empty tables, but figure exists -> orphan figure
    with pytest.raises(PackagingIntegrityError, match="orphan trial IDs"):
        gate.validate_inputs(dataset, {}, [f], manifest)
        
def test_05_metadata_accounting_rejections():
    gate = PackagingValidationGate()
    dataset = ValidatedDataset(valid_results=[], quarantine_records=[])
    
    # Accounting mismatch: total != valid + quarantine
    acc = AccountingRecord(10, 5, 2, 0, {}, {}) # 5+2 != 10
    manifest = PackageManifest("", "", "", "", "", "", "v1", 1, "", "", acc, {}, "")
    with pytest.raises(PackagingIntegrityError, match=r"total_input != valid \+ quarantine"):
        gate.validate_inputs(dataset, {}, [], manifest)
        
    # Bad protocol
    acc2 = AccountingRecord(10, 8, 2, 0, {}, {})
    manifest2 = PackageManifest("", "", "", "", "", "", "v2", 1, "", "", acc2, {}, "")
    with pytest.raises(PackagingIntegrityError, match="Invalid p6 protocol version"):
        gate.validate_inputs(dataset, {}, [], manifest2)
        
    # Bad words in manifest
    manifest3 = PackageManifest("winner", "", "", "", "", "", "v1", 1, "", "", acc2, {}, "")
    with pytest.raises(PackagingIntegrityError, match="Forbidden conclusion term"):
        gate.validate_inputs(dataset, {}, [], manifest3)

def test_06_explicit_numeric_canonicalization():
    """Test standard canonicalization rules explicitly."""
    # 1. Finite float
    assert canonical_serialize({"val": 3.14159}) == b'{"val":3.14159}'
    
    # 2. Integer
    assert canonical_serialize({"val": 42}) == b'{"val":42}'
    
    # 3. None
    assert canonical_serialize({"val": None}) == b'{"val":null}'
    
    # 4. Booleans
    assert canonical_serialize({"val": True, "b": False}) == b'{"b":false,"val":true}'
    
    # 5. Enums
    assert canonical_serialize({"val": ArtifactStatus.GENERATED}) == b'{"val":"GENERATED"}'
    
    # 6. NaN/Inf should raise error because allow_nan=False
    with pytest.raises(ValueError):
        canonical_serialize({"val": float('inf')})
    with pytest.raises(ValueError):
        canonical_serialize({"val": float('nan')})
