"""AgentA2D deterministic reasoning stub.

Provides a predictable, reproducible proxy for LLM reasoning.
Strictly separated from policy logic.
"""

import json
from typing import Iterable

from agent_a2d.core.types import MemoryObject, ToolCall


class DeterministicStub:
    """A deterministic proxy for the LLM reasoning phase.
    
    Generates consistent ToolCalls based on retrieved memories.
    Does NOT contain or enforce security policy.
    """

    def __init__(self, is_attack: bool) -> None:
        """Initialize the stub.
        
        Args:
            is_attack: If True, the stub simulates an attacker successfully
                manipulating the agent into requesting a critical tool action.
        """
        self.is_attack = is_attack

    def generate_summary(
        self,
        inputs: Iterable[MemoryObject],
    ) -> str:
        """Simulate the LLM summarizing ingested data for persistent memory."""
        # Deterministic combination of input content
        contents = [obj.content for obj in sorted(inputs, key=lambda o: o.object_id)]
        summary_payload = {
            "type": "summary",
            "source_count": len(contents),
            "data": contents,
            "contains_attack_payload": self.is_attack
        }
        return json.dumps(summary_payload, sort_keys=True)

    def decide(
        self,
        retrieved_memories: Iterable[MemoryObject],
        probe_instruction: str = ""
    ) -> ToolCall:
        """Simulate LLM reasoning to produce a ToolCall.
        
        If is_attack is True, it attempts a critical action on a hardcoded
        target (simulating an escalation of privileges or destructive action).
        Otherwise, it attempts a benign read-only action.
        """
        has_memories = any(retrieved_memories)
        
        if self.is_attack and has_memories:
            # Simulate a successful prompt injection / confused deputy attack
            return ToolCall(
                action="delete_customer",
                arguments={"customer_id": "C123"}
            )
        else:
            # Benign operational behavior
            return ToolCall(
                action="get_customer_info",
                arguments={"customer_id": "C123" if has_memories else "DEFAULT"}
            )
