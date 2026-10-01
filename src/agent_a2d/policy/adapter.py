"""AgentA2D Stage Enforcement Adapter.

Maps stage-independent PolicyDecisions to physical enforcement actions.
This is the ONLY place where InterventionPoint enters the logic.
"""

from agent_a2d.core.enums import InterventionPoint, StageActionType, Verdict
from agent_a2d.core.types import EnforcedDecision, PolicyDecision


class StageEnforcementAdapter:
    """Adapts PolicyDecisions to stage-specific physical actions."""

    def enforce(
        self,
        decision: PolicyDecision,
        intervention_point: InterventionPoint,
    ) -> EnforcedDecision:
        """Map the verdict to a physical action based on the intervention point.
        
        Returns an EnforcedDecision reflecting both the theoretical verdict
        and the concrete action taken at this specific pipeline stage.
        """
        stage_action: StageActionType
        enforced: bool = True
        
        if decision.verdict == Verdict.INVALID_PROVENANCE:
            stage_action = StageActionType.REJECT
        elif decision.verdict == Verdict.ALLOW:
            stage_action = StageActionType.PERMIT
            enforced = False
        elif decision.verdict == Verdict.INTERDICT:
            # Map INTERDICT to the stage-specific physical action
            if intervention_point == InterventionPoint.INGESTION:
                stage_action = StageActionType.QUARANTINE_INPUT
            elif intervention_point == InterventionPoint.MEMORY_WRITE:
                stage_action = StageActionType.PREVENT_ACTIVATION
            elif intervention_point == InterventionPoint.RETRIEVAL:
                stage_action = StageActionType.EXCLUDE_FROM_REASONING
            elif intervention_point == InterventionPoint.TOOL_AUTHORIZATION:
                stage_action = StageActionType.DENY_TOOL_ACTION
            else:
                raise ValueError(f"Unknown intervention point: {intervention_point}")
        else:
            raise ValueError(f"Unknown verdict: {decision.verdict}")

        return EnforcedDecision(
            verdict=decision.verdict,
            stage_action=stage_action,
            intervention_point=intervention_point,
            enforced=enforced,
            reason=decision.reason,
            provenance_chain_snapshot=decision.provenance_chain_snapshot,
        )
