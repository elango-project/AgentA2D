"""AgentA2D tool authorization gateway.

Uses PolicyEvaluator and StageEnforcementAdapter to gate ToolCalls.
"""

from typing import Any, Callable, Dict, Tuple

from agent_a2d.core.enums import InterventionPoint, StageActionType
from agent_a2d.core.types import EnforcedDecision, PolicyContext, ToolCall
from agent_a2d.policy.adapter import StageEnforcementAdapter
from agent_a2d.policy.evaluator import PolicyEvaluator
from agent_a2d.tools.fake_tools import delete_customer, get_customer_info
from agent_a2d.tools.registry import ToolRegistry


class ToolAuthGate:
    """Tool execution gateway enforcing T-stage policy."""

    def __init__(self) -> None:
        self.registry = ToolRegistry()
        self.evaluator = PolicyEvaluator()
        self.adapter = StageEnforcementAdapter()
        self._implementations: Dict[str, Callable[..., str]] = {
            "delete_customer": delete_customer,
            "get_customer_info": get_customer_info,
        }

    def execute_tool(
        self,
        tool_call: ToolCall,
        context_builder: Callable[[PolicyContext], PolicyContext],
        crm_state: Dict[str, Any]
    ) -> Tuple[EnforcedDecision, str | None]:
        """Attempt to execute a ToolCall, gated by policy.
        
        The context_builder is used to fully assemble the PolicyContext
        (with provenance_chain, etc.) since the auth gate only knows the
        ToolCall itself.
        """
        # Look up profile
        profile = self.registry.get_profile(tool_call.action)
        
        # Build context
        # We pass a partial context, the caller completes it with evidence
        partial_context = PolicyContext(
            subject_objects=(),
            provenance_chain=(),
            action_anchor=profile,
            tool_call=tool_call,
            session_id="",
            trace_id=""
        )
        full_context = context_builder(partial_context)
        
        # Evaluate
        decision = self.evaluator.evaluate(full_context)
        enforced_decision = self.adapter.enforce(decision, InterventionPoint.TOOL_AUTHORIZATION)
        
        if enforced_decision.stage_action != StageActionType.PERMIT:
            return enforced_decision, None
            
        # Execute
        impl = self._implementations.get(tool_call.action)
        if not impl:
            return enforced_decision, f"Tool {tool_call.action} not implemented."
            
        result = impl(tool_call.arguments, crm_state)
        return enforced_decision, result
