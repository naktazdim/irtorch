from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass()
class GRMShapes:
    n_items: int
    n_persons: int
    n_grades: int
    n_responses: int
    n_levels: Optional[np.ndarray] = None


@dataclass()
class GRMInputs:
    response_array: np.ndarray
    shapes: GRMShapes
    level_array: Optional[np.ndarray] = None


@dataclass()
class GRMOutputs:
    a_array: np.ndarray
    b_array: np.ndarray
    t_array: np.ndarray
    level_mean_array: Optional[np.ndarray] = None
    level_std_array: Optional[np.ndarray] = None
