import sys
from dataclasses import dataclass

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_VERSION = matplotlib.__version__
    MATPLOTLIB_BACKEND = matplotlib.get_backend()
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    MATPLOTLIB_VERSION = "unavailable"
    MATPLOTLIB_BACKEND = "unavailable"

from typing import List, Dict, Any, Optional
from agent_a2d.experiment.reporting.schema import RQSummaryTable, DerivedResultProvenance
from agent_a2d.experiment.reporting.provenance import compute_digest, canonical_serialize

@dataclass(frozen=True)
class RenderMetadata:
    matplotlib_version: str
    backend: str
    python_version: str

@dataclass(frozen=True)
class FigureData:
    figure_id: str
    title: str
    rq: str
    metric: str
    synthetic_test_data: bool
    series: List[Dict[str, Any]]
    merged_provenance: DerivedResultProvenance
    limitation_note: Optional[str]

class FigureGenerator:
    """Generates reproducible figures exclusively from Phase 3 validated tables."""
    
    def __init__(self, is_synthetic: bool = False):
        self.is_synthetic = is_synthetic
        self.render_metadata = RenderMetadata(
            matplotlib_version=MATPLOTLIB_VERSION,
            backend=MATPLOTLIB_BACKEND,
            python_version=sys.version.split(" ")[0]
        )
        
    def _merge_provenance(self, tables: List[RQSummaryTable], rq: str, metric: str) -> DerivedResultProvenance:
        if not tables:
            raise ValueError("Cannot merge provenance from empty table list.")
            
        base = tables[0].provenance
        all_trials = set()
        composition = []
        
        for t in tables:
            all_trials.update(t.provenance.source_trial_ids)
            composition.append({
                "source_trial_ids": sorted(t.provenance.source_trial_ids),
                "source_result_hash": t.provenance.source_result_hash
            })
            
        # We recompute the source result hash for the merged set by composing the constituent hashes
        composition.sort(key=lambda x: x["source_result_hash"])
        merged_hash = compute_digest(composition)
        
        sorted_trials = sorted(list(all_trials))
        
        return DerivedResultProvenance(
            analysis_id=base.analysis_id,
            rq=rq,
            metric=metric,
            eligibility_rule_version=base.eligibility_rule_version,
            source_trial_ids=sorted_trials,
            source_result_hash=merged_hash,
            analyzer_version=base.analyzer_version,
            reporting_version=base.reporting_version,
            analysis_code_commit=base.analysis_code_commit
        )
        
    def _format_title(self, base_title: str) -> str:
        if self.is_synthetic:
            return f"[SYNTHETIC TEST DATA] {base_title}"
        return base_title
        
    def build_rq1_figure_data(self, rq1_tables: Dict[str, RQSummaryTable]) -> FigureData:
        """RQ1 Intervention-point attack containment."""
        series = []
        # Deterministic ordering
        for arm in sorted(rq1_tables.keys()):
            t = rq1_tables[arm]
            series.append({
                "arm": arm,
                "prevention_rate": t.raw_data.get("prevention_rate"),
                "prevented_count": t.raw_data.get("prevented_count", 0),
                "total_eligible": t.eligible_count,
                "provider_failure_count": t.provider_failure_count,
                "ineligible_count": t.ineligible_count,
                "quarantined_count": t.quarantined_count
            })
            
        prov = self._merge_provenance(list(rq1_tables.values()), "RQ1", "prevention")
        
        return FigureData(
            figure_id="fig_rq1_containment",
            title=self._format_title("RQ1: Persistent Attack Containment"),
            rq="RQ1",
            metric="prevention",
            synthetic_test_data=self.is_synthetic,
            series=series,
            merged_provenance=prov,
            limitation_note=None
        )

    def build_rq2_figure_data(self, rq2_tables: Dict[str, RQSummaryTable]) -> FigureData:
        """RQ2 Benign memory-dependent task completion proxy."""
        series = []
        for arm in sorted(rq2_tables.keys()):
            t = rq2_tables[arm]
            series.append({
                "arm": arm,
                "completion_proxy_rate": t.raw_data.get("completion_proxy_rate"),
                "task_completion_count": t.raw_data.get("task_completion_count", 0),
                "total_eligible": t.eligible_count,
                "provider_failure_count": t.provider_failure_count
            })
            
        prov = self._merge_provenance(list(rq2_tables.values()), "RQ2", "utility_proxy")
        
        limitation = (
            "completion/outcome proxy, not semantic quality. "
            "Does not measure semantic utility, answer quality, correctness, or intelligence."
        )
        
        return FigureData(
            figure_id="fig_rq2_utility_proxy",
            title=self._format_title("RQ2: Legitimate Task Completion Proxy"),
            rq="RQ2",
            metric="completion_proxy",
            synthetic_test_data=self.is_synthetic,
            series=series,
            merged_provenance=prov,
            limitation_note=limitation
        )

    def build_rq3_figure_data(self, rq3_tables: Dict[str, Dict[str, RQSummaryTable]]) -> List[FigureData]:
        """RQ3 Residual Contamination. Returns 6 separate FigureData objects."""
        figures = []
        boolean_metrics = [
            "storage_contamination", "retrieval_contamination", "behavioral_contamination",
            "malicious_tool_request", "unauthorized_action_executed"
        ]
        
        # 1. Boolean metrics
        for metric in sorted(boolean_metrics):
            if metric not in rq3_tables:
                continue
            arm_tables = rq3_tables[metric]
            series = []
            for arm in sorted(arm_tables.keys()):
                t = arm_tables[arm]
                series.append({
                    "arm": arm,
                    "contamination_rate": t.raw_data.get("contamination_rate"),
                    "contaminated_count": t.raw_data.get("contaminated_count", 0),
                    "total_eligible": t.eligible_count
                })
            
            prov = self._merge_provenance(list(arm_tables.values()), "RQ3", metric)
            figures.append(FigureData(
                figure_id=f"fig_rq3_{metric}",
                title=self._format_title(f"RQ3: {metric}"),
                rq="RQ3",
                metric=metric,
                synthetic_test_data=self.is_synthetic,
                series=series,
                merged_provenance=prov,
                limitation_note=None
            ))
            
        # 2. Categorical metric: recovery_state
        recovery_keys = [k for k in rq3_tables.keys() if k.startswith("recovery_state:")]
        if recovery_keys:
            arms = set()
            for k in recovery_keys:
                arms.update(rq3_tables[k].keys())
                
            series = []
            all_tables = []
            for arm in sorted(list(arms)):
                dist = {}
                eligible_count = 0
                for k in sorted(recovery_keys):
                    if arm in rq3_tables[k]:
                        t = rq3_tables[k][arm]
                        all_tables.append(t)
                        state = t.raw_data.get("state")
                        dist[state] = t.raw_data.get("state_count", 0)
                        eligible_count = t.eligible_count
                
                series.append({
                    "arm": arm,
                    "distribution": dist,
                    "total_eligible": eligible_count
                })
                
            prov = self._merge_provenance(all_tables, "RQ3", "recovery_state")
            figures.append(FigureData(
                figure_id="fig_rq3_recovery_state",
                title=self._format_title("RQ3: Recovery State Distribution"),
                rq="RQ3",
                metric="recovery_state",
                synthetic_test_data=self.is_synthetic,
                series=series,
                merged_provenance=prov,
                limitation_note=None
            ))
            
        return figures

    def build_rq4_figure_data(self, rq4_tables: Dict[str, Dict[str, RQSummaryTable]]) -> List[FigureData]:
        """RQ4 Laundering/Fragmentation. Returns FigureData objects for prevention/susceptibility."""
        figures = []
        for category in sorted(rq4_tables.keys()):
            for metric in sorted(rq4_tables[category].keys()):
                t = rq4_tables[category][metric]
                series = [{
                    "category": category,
                    "metric_type": metric,
                    "rate": t.raw_data.get("rate"),
                    "numerator_count": t.raw_data.get("numerator_count", 0),
                    "total_eligible": t.eligible_count
                }]
                
                prov = self._merge_provenance([t], "RQ4", f"{category}_{metric}")
                figures.append(FigureData(
                    figure_id=f"fig_rq4_{category}_{metric}",
                    title=self._format_title(f"RQ4: {category.capitalize()} ({metric.capitalize()})"),
                    rq="RQ4",
                    metric=f"{category}_{metric}",
                    synthetic_test_data=self.is_synthetic,
                    series=series,
                    merged_provenance=prov,
                    limitation_note=None
                ))
        return figures

    def render_figure(self, figure_data: FigureData, output_path: str) -> None:
        """Render the deterministic figure_data to a file using non-interactive backend.
        Handles zero-denominators by rendering them explicitly (or avoiding division by zero).
        """
        if not HAS_MATPLOTLIB:
            # Fallback for environments where matplotlib is unavailable
            with open(output_path, 'w') as f:
                f.write("Matplotlib not available. Render skipped.\n")
            return
            
        plt.figure(figsize=(10, 6))
        
        arms = []
        rates = []
        na_indices = []
        
        if figure_data.metric == "recovery_state":
            for i, s in enumerate(figure_data.series):
                arms.append(s["arm"])
                if s["total_eligible"] == 0:
                    rates.append(0.0)
                    na_indices.append(i)
                else:
                    rates.append(sum(s["distribution"].values()))
        elif figure_data.rq == "RQ4":
            for i, s in enumerate(figure_data.series):
                arms.append(s["category"])
                val = s.get("rate")
                if val is None:
                    rates.append(0.0)
                    na_indices.append(i)
                else:
                    rates.append(val)
        else:
            for i, s in enumerate(figure_data.series):
                arms.append(s["arm"])
                rate_keys = [k for k in s.keys() if "rate" in k]
                rate_key = rate_keys[0] if rate_keys else None
                val = s.get(rate_key) if rate_key else None
                
                if val is None:
                    rates.append(0.0)
                    na_indices.append(i)
                else:
                    rates.append(val)
                    
        plt.bar(arms, rates, color='gray')
        
        # Explicitly annotate N/A for unavailable data (e.g. zero denominator)
        for idx in na_indices:
            plt.text(idx, 0.02, "N/A", ha='center', va='bottom', color='red', fontweight='bold')
            
        plt.title(figure_data.title)
        plt.xlabel("Category / Arm")
        plt.ylabel("Value (Rate or Count)")
        
        if figure_data.limitation_note:
            plt.figtext(0.5, 0.01, figure_data.limitation_note, wrap=True, horizontalalignment='center', fontsize=8)
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close() # Clean up state
