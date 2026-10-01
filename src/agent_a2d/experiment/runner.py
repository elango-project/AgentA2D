"""AgentA2D experiment runner.

Executes a paired reference trace and active trace to establish the canonical
result for a deterministic trial.
"""

import copy
from typing import Any, Dict

from agent_a2d.experiment.config import ExperimentConfig
from agent_a2d.experiment.result import ExperimentResult
from agent_a2d.pipeline import run_deterministic_vertical_slice


class ExperimentRunner:
    """Runs dual traces (reference + active) from identical environments."""

    def run_trial(
        self,
        config: ExperimentConfig,
        initial_crm_state: Dict[str, Any],
        initial_tool_state: Dict[str, Any],
    ) -> ExperimentResult:
        """Execute the trial.
        
        1. Run the reference trace with active_enforcement=None.
        2. If the config specifies an active_enforcement, re-run with it active.
        """
        # Run Reference Trace (always None for active_enforcement to record all shadow decisions)
        ref_config = copy.copy(config)
        object.__setattr__(ref_config, "active_enforcement", None)
        
        ref_crm = copy.deepcopy(initial_crm_state)
        ref_tool = copy.deepcopy(initial_tool_state)
        
        ref_trace = run_deterministic_vertical_slice(
            config=ref_config,
            crm_state=ref_crm,
            tool_state=ref_tool,
        )
        
        active_trace = None
        if config.active_enforcement is not None:
            active_crm = copy.deepcopy(initial_crm_state)
            active_tool = copy.deepcopy(initial_tool_state)
            
            active_trace = run_deterministic_vertical_slice(
                config=config,
                crm_state=active_crm,
                tool_state=active_tool,
            )
            
        return ExperimentResult(
            config=config,
            reference_trace=ref_trace,
            active_trace=active_trace,
        )
