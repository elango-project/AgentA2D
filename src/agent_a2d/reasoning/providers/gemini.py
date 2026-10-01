import json
import logging
from typing import Mapping, Any, Optional

from google import genai
from google.genai import types

from agent_a2d.core.types import ToolCall
from agent_a2d.reasoning.providers.base import LLMProviderAdapter

logger = logging.getLogger(__name__)

class GeminiProviderAdapter(LLMProviderAdapter):
    """Adapter for Google's Gemini models."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        seed: Optional[int] = 42,
    ):
        """Initialize the Gemini adapter with fixed parameters for reproducibility.
        
        Args:
            api_key: Google Gemini API key.
            model_name: The precise model identifier.
            temperature: Sampling temperature (0.0 for deterministic).
            seed: Optional seed for reproducibility.
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            # We enforce deterministic configuration
        )
        # Store for manifest
        self.manifest_data = {
            "provider": "google.genai",
            "model": model_name,
            "temperature": temperature,
            "seed": seed,
        }

    def generate_text(self, prompt: str) -> str:
        """Generate a text completion."""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.config
            )
            return response.text or ""
        except Exception as e:
            logger.error(f"Provider error during generate_text: {e}")
            raise

    def call_tool(self, prompt: str, available_tools: Mapping[str, Any]) -> ToolCall:
        """Use Gemini's function calling capability to pick a tool."""
        
        # Convert our available_tools schema to Gemini Tool declarations
        gemini_tools = []
        for tool_name, schema in available_tools.items():
            func_decl = types.FunctionDeclaration(
                name=tool_name,
                description=schema.get("description", ""),
                # Pass the properties schema explicitly
            )
            # Basic conversion for P6, assumes simple string properties
            props = {}
            required = schema.get("parameters", {}).get("required", [])
            for prop_name, prop_details in schema.get("parameters", {}).get("properties", {}).items():
                props[prop_name] = types.Schema(
                    type=types.Type.STRING,
                    description=prop_details.get("description", "")
                )
            
            func_decl.parameters = types.Schema(
                type=types.Type.OBJECT,
                properties=props,
                required=required
            )
            
            gemini_tools.append(types.Tool(function_declarations=[func_decl]))
            
        config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            seed=self.config.seed,
            tools=gemini_tools,
            # Force tool usage if needed, but for now allow auto
            # response_modalities=["TEXT"], 
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            # Look for function call
            if response.function_calls:
                call = response.function_calls[0]
                # Convert args back to a dict
                args = {}
                if call.args:
                    for key, val in call.args.items():
                        args[key] = val
                        
                return ToolCall(
                    action=call.name,
                    arguments=args
                )
            else:
                # Fallback if the model didn't call a tool
                # We can just return a dummy tool or parse text
                return ToolCall(action="fallback_no_tool", arguments={"text": response.text or ""})
                
        except Exception as e:
            logger.error(f"Provider error during call_tool: {e}")
            raise
