import os

import pandas as pd
import numpy as np


def resource_path(*paths: str) -> str:
    return os.path.join(os.path.dirname(__file__), "resources", "converter", *paths)


def df(*paths: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(resource_path(*paths), **kwargs)


def array(*paths: str, **kwargs) -> np.ndarray:
    return np.loadtxt(resource_path(*paths), delimiter=",", **kwargs)

