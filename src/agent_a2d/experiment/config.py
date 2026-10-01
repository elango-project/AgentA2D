"""AgentA2D experiment configuration.

Defines the parameters for a single deterministic trial.
"""

from dataclasses import dataclass
from typing import Optional

from agent_a2d.core.enums import InterventionPoint
from agent_a2d.core.types import ProtectedActionProfile
from agent_a2d.reasoning.strategy import ReasoningStrategy


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single experimental trial."""
    
    experiment_id: str
    trial_id: str
    is_attack: bool
    raw_payload: str
    seed: int
    hmac_key: bytes
    
    # Policy configuration
    policy_version: str
    schema_version: int
    oracle_action_anchor: Optional[ProtectedActionProfile]
    
    # Enforcement location (if None, pure reference trace)
    active_enforcement: Optional[InterventionPoint]
    
    # Optional real LLM integration for P6+
    # If None, defaults to DeterministicStub(is_attack)
    reasoning_strategy: Optional["ReasoningStrategy"] = None
    probe_instruction: str = ""
