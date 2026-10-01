from typing import Protocol, Iterable

from agent_a2d.core.types import MemoryObject, ToolCall


class ReasoningStrategy(Protocol):
    """Abstract interface for agent reasoning implementations.
    
    This abstracts away whether the agent is using a deterministic stub
    or a real LLM provider, ensuring the pipeline itself remains
    provider-agnostic.
    """

    def generate_summary(
        self,
        inputs: Iterable[MemoryObject],
    ) -> str:
        """Simulate or execute summarizing ingested data for persistent memory.
        
        Args:
            inputs: The memory objects to summarize.
            
        Returns:
            The raw text summary (to be wrapped in provenance by middleware).
        """
        ...

    def decide(
        self,
        retrieved_memories: Iterable[MemoryObject],
        probe_instruction: str = ""
    ) -> ToolCall:
        """Simulate or execute reasoning to produce a ToolCall.
        
        Args:
            retrieved_memories: The context available for the decision.
            probe_instruction: The instruction given to the agent for this session.
            
        Returns:
            The requested ToolCall.
        """
        ...
