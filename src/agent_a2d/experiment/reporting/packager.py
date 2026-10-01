from typing import Dict, List, Any, Optional
from agent_a2d.experiment.reporting.schema import (
    ValidatedDataset, RQSummaryTable, PackageManifest,
    PackagingIntegrityError, DerivedResultProvenance
)
from agent_a2d.experiment.reporting.figures import FigureData

class PackagingValidationGate:
    """Performs cross-layer validation before generating the evidence package."""
    
    def validate_inputs(
        self,
        dataset: ValidatedDataset,
        rq_tables: Dict[str, Any], # nested dicts depending on RQ
        figures: List[FigureData],
        manifest: PackageManifest
    ) -> None:
        
        # 1. Collect all known raw trial IDs
        known_raw_trials = {r.trial_id for r in dataset.valid_results}
        known_raw_trials.update({q.trial_id for q in dataset.quarantine_records})
        
        # 2. Extract derived trial IDs from tables
        all_tables = []
        def _extract_tables(d):
            for k, v in d.items():
                if isinstance(v, RQSummaryTable):
                    all_tables.append(v)
                elif isinstance(v, dict):
                    _extract_tables(v)
        _extract_tables(rq_tables)
        
        table_provs = []
        all_table_trials = set()
        for t in all_tables:
            table_provs.append(t.provenance)
            all_table_trials.update(t.provenance.source_trial_ids)
            all_table_trials.update(t.provenance.ineligible_trial_ids)
            all_table_trials.update(t.provenance.provider_failure_trial_ids)
            all_table_trials.update(t.provenance.quarantined_trial_ids)
            self._validate_provenance_populations(t.provenance, known_raw_trials, "Table")
            
        # 3. Extract derived trial IDs from figures
        for f in figures:
            self._validate_provenance_populations(f.merged_provenance, known_raw_trials, "Figure")
            
            # Orphan check: figure must map to existing tables
            fig_trials = set(f.merged_provenance.source_trial_ids) | set(f.merged_provenance.ineligible_trial_ids) | set(f.merged_provenance.provider_failure_trial_ids) | set(f.merged_provenance.quarantined_trial_ids)
            if not fig_trials.issubset(all_table_trials):
                raise PackagingIntegrityError(f"Figure {f.figure_id} contains orphan trial IDs not present in upstream tables.")
            
        # 4. Check for forbidden scientific conclusion fields in manifest
        bad_words = ["winner", "best", "ranking", "significance", "semantic utility"]
        import json
        import dataclasses
        manifest_str = json.dumps(dataclasses.asdict(manifest)).lower()
        for bw in bad_words:
            if bw in manifest_str:
                raise PackagingIntegrityError(f"Forbidden conclusion term found in manifest: {bw}")
                
        # 5. Metadata consistency
        if manifest.p6_protocol_version != "v1":
            raise PackagingIntegrityError(f"Invalid p6 protocol version: {manifest.p6_protocol_version}")
            
        if manifest.schema_version < 1:
            raise PackagingIntegrityError("Invalid schema version.")
            
        # 6. Accounting consistency
        acc = manifest.accounting
        if acc.total_input != acc.valid + acc.quarantine:
            raise PackagingIntegrityError("Accounting mismatch: total_input != valid + quarantine")
            
        # We can loosely verify valid == eligible + ineligible + provider_failures for at least one RQ.
        for rq, eligible in acc.rq_eligible.items():
            ineligible = acc.rq_ineligible.get(rq, 0)
            if eligible + ineligible + acc.provider_failures != acc.valid:
                raise PackagingIntegrityError(f"Accounting mismatch for {rq}: populations do not sum to valid total.")

    def _validate_provenance_populations(self, prov: DerivedResultProvenance, known_trials: set, source_type: str):
        all_refs = set(prov.source_trial_ids) | set(prov.ineligible_trial_ids) | set(prov.provider_failure_trial_ids) | set(prov.quarantined_trial_ids)
        unknown = all_refs - known_trials
        if unknown:
            raise PackagingIntegrityError(f"{source_type} references unknown trial IDs: {unknown}")
