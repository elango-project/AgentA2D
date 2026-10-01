"""AgentA2D experiment result structures."""

from dataclasses import dataclass
from typing import Optional

from agent_a2d.core.types import Trace
from agent_a2d.experiment.config import ExperimentConfig


@dataclass(frozen=True)
class ExperimentResult:
    """The result of a single experimental trial.
    
    Contains the reference trace (run with active_enforcement=None)
    and the active trace (run with the targeted active_enforcement),
    if active_enforcement was configured.
    """
    
    config: ExperimentConfig
    reference_trace: Trace
    active_trace: Optional[Trace]
