import math
from typing import List, Tuple, Sequence

def calculate_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)

def calculate_variance(values: Sequence[float], mean: float = None) -> float:
    if len(values) < 2:
        return 0.0
    if mean is None:
        mean = calculate_mean(values)
    return sum((x - mean) ** 2 for x in values) / (len(values) - 1)

def wilson_score_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate the Wilson score confidence interval for a proportion.
    
    Default z=1.96 corresponds to a 95% confidence interval.
    """
    if n == 0:
        return (0.0, 0.0)
        
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denominator
    
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return (lower, upper)
