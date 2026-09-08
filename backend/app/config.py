import os
from pydantic import BaseModel
from typing import Dict, List, Any

class SystemConfig(BaseModel):
    # Epsilon presets
    EPSILON_PRESETS: Dict[str, float] = {
        "controlled": 0.02,
        "moderate": 0.05,
        "strong": 0.08,
    }
    MIN_EPSILON: float = 0.001
    MAX_EPSILON: float = 0.15

    # Iteration presets
    ITERATION_PRESETS: List[int] = [5, 10, 20]
    MIN_ITERATIONS: int = 1
    MAX_ITERATIONS: int = 50

    # Benchmark target classes (ImageNet class indices)
    APPROVED_TARGET_CLASSES: Dict[int, str] = {
        9: "ostrich",
        292: "lion, Panthera leo",
        954: "banana",
    }
    DEFAULT_TARGET_CLASS: int = 9  # Ostrich

    # Risk Engine thresholds
    RISK_THRESHOLD_LOW: float = 30.0
    RISK_THRESHOLD_MEDIUM: float = 65.0

    # Saturated confidence & margin risk signals
    # Saturated raw confidence (>99%) is a known adversarial fingerprint
    SATURATED_CONFIDENCE_THRESHOLD: float = 0.99
    # Large top-1 minus top-2 margin (>0.90) indicates an overconfident prediction spike
    HIGH_MARGIN_THRESHOLD: float = 0.90
    # Penalty weight applied when saturated raw confidence/margin occurs with cross-view instability
    SATURATED_OVERCONFIDENCE_PENALTY_WEIGHT: float = 35.0

    # Softmax entropy risk signal
    # Extremely low entropy (<0.10) indicates a degenerate spiked probability distribution
    LOW_ENTROPY_THRESHOLD: float = 0.10
    # Penalty weight applied for abnormally low softmax entropy under view disagreement
    LOW_ENTROPY_PENALTY_WEIGHT: float = 15.0

    # Compounding risk multiplier config
    # Minimum distinct alert categories required to trigger compounding multiplier bonus
    COMPOUNDING_SIGNALS_THRESHOLD: int = 3
    # Additional risk score bonus added when 3+ independent adversarial signals co-occur
    COMPOUNDING_SIGNALS_BONUS: float = 25.0

    # Expanded recovery pass parameters
    # Transform strength used in second-stage expanded pass specifically for consensus recovery
    EXPANDED_PASS_STRENGTH: float = 0.85

    # Transformations & Multi-view config
    DEFAULT_TRANSFORMS: List[str] = [
        "blur",
        "jpeg",
        "resize",
        "brightness",
        "contrast",
        "noise",
        "sharpen",
    ]

    # Randomized Smoothing parameters
    # Multi-level Gaussian noise strengths for randomized smoothing
    RANDOMIZED_SMOOTHING_NOISE_LEVELS: List[float] = [0.03, 0.06, 0.10]
    # Number of randomized Gaussian noise samples per strength level
    RANDOMIZED_SMOOTHING_REPEATS: int = 3  # 3 repeats * 3 levels = 9 smoothing views

config = SystemConfig()
