from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class TrainingHistory:
    losses: List[float] = field(default_factory=list)
    weights: List[np.ndarray] = field(default_factory=list)
    biases: List[float] = field(default_factory=list)
    gradients: List[np.ndarray] = field(default_factory=list)
    learning_rate: float = 0.0
    epochs_run: int = 0
    converged: bool = False

    def is_empty(self) -> bool:
        return len(self.losses) == 0
