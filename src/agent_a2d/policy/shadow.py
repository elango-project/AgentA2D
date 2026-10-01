"""AgentA2D shadow execution mapping.

Simulates the complete trace evaluation for shadow-mode tracking,
distinguishing explicitly reached states from NOT_REACHED states.
"""

from typing import Dict

from agent_a2d.core.enums import InterventionPoint
from agent_a2d.core.types import EnforcedDecision, PolicyContext
from agent_a2d.policy.adapter import StageEnforcementAdapter
from agent_a2d.policy.evaluator import PolicyEvaluator


class ShadowPolicyEvaluator:
    """Computes shadow decisions for all intervention points.
    
    If an active enforcement (e.g., at I) halts the pipeline, downstream stages
    (M, R, T) are strictly recorded as NOT_REACHED rather than fabricating a
    hypothetical policy evaluation.
    """

    def __init__(self) -> None:
        self.evaluator = PolicyEvaluator()
        self.adapter = StageEnforcementAdapter()

    def evaluate_shadow(
        self,
        contexts_by_stage: Dict[InterventionPoint, PolicyContext | None]
    ) -> Dict[InterventionPoint, EnforcedDecision | None]:
        """Compute the shadow decisions based on the provided contexts.
        
        If a context is None, it means the pipeline never reached that stage
        (due to early active enforcement). The result will explicitly be None
        to signal NOT_REACHED.
        """
        results: Dict[InterventionPoint, EnforcedDecision | None] = {}
        
        for point in InterventionPoint:
            context = contexts_by_stage.get(point)
            if context is None:
                # Downstream NOT_REACHED
                results[point] = None
            else:
                decision = self.evaluator.evaluate(context)
                results[point] = self.adapter.enforce(decision, point)
                
        return results
