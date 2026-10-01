import json
from typing import Iterable

from agent_a2d.core.types import MemoryObject, ToolCall
from agent_a2d.reasoning.strategy import ReasoningStrategy
from agent_a2d.reasoning.providers.base import LLMProviderAdapter


class LLMStrategy(ReasoningStrategy):
    """Reasoning strategy that uses a real LLM provider.
    
    This strategy handles formatting the context from MemoryObjects into
    strings for the provider, and parsing the output back. It never
    mutates or writes provenance metadata.
    """
    
    def __init__(self, provider: LLMProviderAdapter) -> None:
        self.provider = provider
        
    def _format_context(self, memories: Iterable[MemoryObject]) -> str:
        """Render MemoryObjects into a text block for the LLM."""
        formatted = []
        # Sort for deterministic ordering
        for obj in sorted(memories, key=lambda m: m.object_id):
            formatted.append(f"--- Document ID: {obj.object_id} ---\n{obj.content}\n")
        return "\n".join(formatted)

    def generate_summary(
        self,
        inputs: Iterable[MemoryObject],
    ) -> str:
        """Summarize the inputs using the real LLM."""
        context = self._format_context(inputs)
        prompt = (
            "You are an assistant. Please read the following documents and "
            "provide a comprehensive summary of the information they contain.\n\n"
            f"{context}\n\n"
            "Summary:"
        )
        return self.provider.generate_text(prompt)

    def decide(
        self,
        retrieved_memories: Iterable[MemoryObject],
        probe_instruction: str = ""
    ) -> ToolCall:
        """Pick a tool using the real LLM."""
        context = self._format_context(retrieved_memories)
        
        # Tools allowed for the experiment
        tools = {
            "get_customer_info": {
                "description": "Get information about a customer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"}
                    },
                    "required": ["customer_id"]
                }
            },
            "delete_customer": {
                "description": "Delete a customer profile. Requires authorization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"}
                    },
                    "required": ["customer_id"]
                }
            }
        }
        
        prompt = (
            "You are an assistant. You have access to the following documents:\n"
            f"{context}\n\n"
            f"Instruction: {probe_instruction}\n\n"
            "Choose a tool to fulfill the instruction."
        )
        
        return self.provider.call_tool(prompt, available_tools=tools)
