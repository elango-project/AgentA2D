"""AgentA2D policy rules.

Evaluates evidence against a required action profile.
"""

from agent_a2d.core.enums import TrustLevel, Verdict
from agent_a2d.core.types import PolicyContext


def evaluate_rules(context: PolicyContext) -> tuple[Verdict, str]:
    """Evaluate context using core deterministic rules.
    
    Returns a stage-independent Verdict and a reason string.
    """
    # Determine the target profile from tool_call or oracle anchor
    target_profile = None
    if context.tool_call:
        # If tool_call is provided, it dictates the required authority
        # In a real system, we'd lookup the profile from registry
        # For simplicity here, we assume the context builder populated action_anchor
        # if there was one, or we check it directly. 
        # Actually, let's use the provided action_anchor which should match the tool_call.
        target_profile = context.action_anchor
    else:
        # At early stages (I, M, R), tool_call is None, so we rely entirely on action_anchor
        target_profile = context.action_anchor

    if not target_profile:
        # If there's no protected profile targeted, default to ALLOW
        return Verdict.ALLOW, "No protected action targeted"

    if target_profile.required_authority == TrustLevel.UNTRUSTED:
        return Verdict.ALLOW, "Targeted action does not require trusted authority"

    # We need TRUSTED authority. Scan the provenance chain for valid endorsements.
    # An endorsement is valid if an object is TRUSTED and has a trust_basis_ref.
    has_valid_endorsement = False
    for obj in context.provenance_chain:
        # If any object in the chain is magically TRUSTED without a ref, that's invalid provenance
        if obj.trust_label == TrustLevel.TRUSTED and not obj.trust_basis_ref:
            return Verdict.INVALID_PROVENANCE, f"Object {obj.object_id} claims TRUSTED without basis"
            
        if obj.trust_label == TrustLevel.TRUSTED and obj.trust_basis_ref:
            has_valid_endorsement = True

    if has_valid_endorsement:
        return Verdict.ALLOW, "Valid authority found in provenance chain"
    
    return Verdict.INTERDICT, f"Missing required {target_profile.required_authority.name} authority"
