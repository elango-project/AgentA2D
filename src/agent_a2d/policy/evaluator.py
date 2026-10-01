"""AgentA2D policy evaluator.

Produces a purely stage-independent PolicyDecision.
"""

from agent_a2d.core.types import PolicyContext, PolicyDecision
from agent_a2d.policy.rules import evaluate_rules


class PolicyEvaluator:
    """Stage-blind evaluator of provenance and evidence.
    
    Guarantees identical verdicts for identical inputs, completely
    ignoring what stage of the pipeline invoked it.
    """

    def evaluate(self, context: PolicyContext) -> PolicyDecision:
        """Evaluate the context and return a PolicyDecision."""
        verdict, reason = evaluate_rules(context)
        
        # Capture the snapshot of the provenance chain for deterministic audit
        chain_snapshot = tuple(obj.object_id for obj in context.provenance_chain)
        
        return PolicyDecision(
            verdict=verdict,
            reason=reason,
            provenance_chain_snapshot=chain_snapshot
        )
