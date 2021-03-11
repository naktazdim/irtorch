from dataclasses import dataclass

import numpy as np


@dataclass()
class GRMShapes:
    n_items: int
    n_persons: int
    n_grades: int
    n_responses: int
    n_levels: int


@dataclass()
class GRMInputs:
    response_array: np.ndarray
    shapes: GRMShapes
    level_array: np.ndarray


@dataclass()
class GRMOutputs:
    a_array: np.ndarray
    b_array: np.ndarray
    t_array: np.ndarray
    level_mean_array: np.ndarray
    level_std_array: np.ndarray
