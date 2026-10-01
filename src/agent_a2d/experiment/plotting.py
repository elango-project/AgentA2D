from typing import Dict, Any, List

def generate_prevention_bar_chart(data: Dict[str, Dict[str, Any]]) -> str:
    """Generate a synthetic text-based bar chart for defense prevention rates.
    
    WARNING: THIS USES SYNTHETIC FIXTURES OR PROVIDED TEST DATA ONLY.
    """
    lines = ["--- DEFENSE PREVENTION RATES (TEST DATA) ---"]
    for arm, stats in data.items():
        rate = stats.get("prevention_rate", 0.0)
        bar_length = int(rate * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        lines.append(f"{arm:10} | {bar} | {rate*100:5.1f}% ({stats.get('prevented_count', 0)}/{stats.get('eligible_trials', 0)})")
    
    lines.append("--------------------------------------------")
    return "\n".join(lines)
