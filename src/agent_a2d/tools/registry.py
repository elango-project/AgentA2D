"""AgentA2D Tool Registry.

Contains definitions of ProtectedActionProfiles and fake tools.
"""

from typing import Dict, Optional

from agent_a2d.core.enums import ActionSeverity, TrustLevel
from agent_a2d.core.types import ProtectedActionProfile

# Action profile for deleting a customer
DELETE_CUSTOMER_PROFILE = ProtectedActionProfile(
    action="delete_customer",
    severity=ActionSeverity.CRITICAL,
    subject="customer_record",
    required_authority=TrustLevel.TRUSTED,
)

# Harmless read action
GET_CUSTOMER_INFO_PROFILE = ProtectedActionProfile(
    action="get_customer_info",
    severity=ActionSeverity.LOW,
    subject="customer_record",
    required_authority=TrustLevel.UNTRUSTED,
)


class ToolRegistry:
    """Registry mapping tool names to their protected action profiles."""

    def __init__(self) -> None:
        self._profiles: Dict[str, ProtectedActionProfile] = {
            "delete_customer": DELETE_CUSTOMER_PROFILE,
            "get_customer_info": GET_CUSTOMER_INFO_PROFILE,
        }

    def get_profile(self, tool_name: str) -> Optional[ProtectedActionProfile]:
        """Get the protected action profile for a tool, if it exists."""
        return self._profiles.get(tool_name)
