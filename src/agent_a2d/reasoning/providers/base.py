from typing import Protocol, Mapping, Any
from agent_a2d.core.types import ToolCall

class LLMProviderAdapter(Protocol):
    """Minimal LLM integration adapter.
    
    Receives rendered context/prompt and returns text or a ToolCall.
    Has no access to MemoryObjects or provenance metadata.
    """

    def generate_text(self, prompt: str) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: The full rendered prompt including context.
            
        Returns:
            The raw text output from the LLM.
        """
        ...

    def call_tool(self, prompt: str, available_tools: Mapping[str, Any]) -> ToolCall:
        """Select a tool to call based on the prompt.
        
        Args:
            prompt: The full rendered prompt including context.
            available_tools: Schemas for the tools the LLM can call.
            
        Returns:
            A ToolCall object indicating the chosen action and arguments.
        """
        ...
