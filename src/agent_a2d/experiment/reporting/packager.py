from typing import Dict, List, Any, Optional
from agent_a2d.experiment.reporting.schema import (
    ValidatedDataset, RQSummaryTable, PackageManifest,
    PackagingIntegrityError, DerivedResultProvenance
)
from agent_a2d.experiment.reporting.figures import FigureData
from agent_a2d.experiment.reporting.provenance import compute_digest, canonical_serialize
from agent_a2d.experiment.reporting.eligibility import EligibilityResult

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
        
        # Map table hashes to tables for figure verification
        table_hash_map = {}
        for t in all_tables:
            # DEFECT 1 Fix: Verify table source_result_hash cryptographically binds to actual source content
            self._verify_table_provenance(t.provenance, dataset, known_raw_trials)
            table_key = (t.provenance.rq, t.provenance.metric, t.provenance.source_result_hash)
            table_hash_map[table_key] = t
            
        # 3. Extract derived trial IDs from figures
        for f in figures:
            # DEFECT 2 Fix: Verify figure provenance against constituent tables
            self._verify_figure_provenance(f, table_hash_map)
            # Also verify figure populations aren't orphan (subsumed by table check, but we do it anyway)
            all_refs = set(f.merged_provenance.source_trial_ids) | set(f.merged_provenance.ineligible_trial_ids) | set(f.merged_provenance.provider_failure_trial_ids) | set(f.merged_provenance.quarantined_trial_ids)
            unknown = all_refs - known_raw_trials
            if unknown:
                raise PackagingIntegrityError(f"Figure references unknown trial IDs: {unknown}")
            
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

    def _verify_table_provenance(self, prov: DerivedResultProvenance, dataset: ValidatedDataset, known_trials: set):
        """Cryptographically verifies source_result_hash against the canonical source result content."""
        all_refs = set(prov.source_trial_ids) | set(prov.ineligible_trial_ids) | set(prov.provider_failure_trial_ids) | set(prov.quarantined_trial_ids)
        unknown = all_refs - known_trials
        if unknown:
            raise PackagingIntegrityError(f"Table references unknown trial IDs: {unknown}")
            
        # Extract the actual source result objects referenced by the eligible source_trial_ids
        sorted_trials = sorted(prov.source_trial_ids)
        results_to_hash = [r for r in dataset.valid_results if r.trial_id in sorted_trials]
        
        # We MUST ensure no IDs are missing from valid_results if they were in source_trial_ids
        if len(results_to_hash) != len(sorted_trials):
            raise PackagingIntegrityError("Mismatch between source_trial_ids and actual extracted results")
            
        # Sort deterministically
        results_to_hash.sort(key=lambda r: r.trial_id)
        
        # Compute expected hash
        expected_hash = compute_digest(results_to_hash)
        if expected_hash != prov.source_result_hash:
            raise PackagingIntegrityError(f"Source hash mismatch. Expected {expected_hash}, got {prov.source_result_hash}")

    def _verify_figure_provenance(self, figure: FigureData, table_hash_map: Dict[str, RQSummaryTable]):
        """Verifies the figure's merged composite hash against its exact constituent tables."""
        composition = []
        for identity in figure.constituent_table_identities:
            # We created a composite key in table_hash_map using (rq, metric, source_result_hash)
            table_key = (identity["rq"], identity["metric"], identity["source_result_hash"])
            if table_key not in table_hash_map:
                raise PackagingIntegrityError(f"Figure {figure.figure_id} references unknown constituent table: {identity}")
            
            t = table_hash_map[table_key]
            composition.append({
                "rq": t.provenance.rq,
                "metric": t.provenance.metric,
                "source_trial_ids": sorted(t.provenance.source_trial_ids),
                "ineligible_trial_ids": sorted(t.provenance.ineligible_trial_ids),
                "provider_failure_trial_ids": sorted(t.provenance.provider_failure_trial_ids),
                "quarantined_trial_ids": sorted(t.provenance.quarantined_trial_ids),
                "source_result_hash": t.provenance.source_result_hash
            })
            
        composition.sort(key=lambda x: x["source_result_hash"])
        expected_hash = compute_digest(composition)
        
        if expected_hash != figure.merged_provenance.source_result_hash:
            raise PackagingIntegrityError(f"Figure composite hash mismatch for {figure.figure_id}. Expected {expected_hash}, got {figure.merged_provenance.source_result_hash}")


def compute_scientific_digest(
    dataset: ValidatedDataset,
    eligibility_decisions: List[EligibilityResult],
    rq_tables: Dict[str, Any],
    figures: List[FigureData]
) -> str:
    """
    Computes the canonical scientific digest over the logical scientific contents.
    The digest is independent of generation timestamps, filesystem paths, OS details,
    rendering environment metadata, and archive metadata.
    
    Structure:
    canonical scientific component -> component SHA-256 -> sorted (logical_name, component_hash) pairs -> final SHA-256
    """
    components = []
    
    # 1. Raw results
    sorted_raw = sorted(dataset.valid_results, key=lambda r: r.trial_id)
    components.append(("raw_results", compute_digest(sorted_raw)))
    
    # 2. Quarantine records
    sorted_quarantine = sorted(dataset.quarantine_records, key=lambda q: q.trial_id)
    components.append(("quarantine_records", compute_digest(sorted_quarantine)))
    
    # 3. Eligibility decisions
    sorted_el = sorted(eligibility_decisions, key=lambda e: (e.rq, e.metric, e.trial_id))
    components.append(("eligibility_decisions", compute_digest(sorted_el)))
    
    # 4. RQ tables
    all_tables = []
    def _extract_tables(d, path):
        for k, v in d.items():
            if isinstance(v, RQSummaryTable):
                all_tables.append(("table:" + ":".join(path + [str(k)]), v))
            elif isinstance(v, dict):
                _extract_tables(v, path + [str(k)])
    
    _extract_tables(rq_tables, [])
    # Sort by the logical path to guarantee deterministic order
    all_tables.sort(key=lambda x: x[0])
    for logical_name, t in all_tables:
        components.append((logical_name, compute_digest(t)))
        
    # 5. FigureData
    # Sort figures by figure_id
    sorted_figures = sorted(figures, key=lambda f: f.figure_id)
    for f in sorted_figures:
        components.append((f"figure:{f.figure_id}", compute_digest(f)))
        
    # Final sort of all (logical_name, hash) pairs
    components.sort(key=lambda x: x[0])
    
    return compute_digest(components)
