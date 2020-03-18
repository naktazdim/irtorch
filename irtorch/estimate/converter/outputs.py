from itertools import product
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

from .meta import GRMMeta


@dataclass()
class GRMOutputs:
    meta: GRMMeta
    a_array: np.ndarray
    b_array: np.ndarray
    t_array: np.ndarray
    level_mean_array: Optional[np.ndarray] = None
    level_std_array: Optional[np.ndarray] = None

    def __post_init__(self):
        assert self.a_array.shape == (self.meta.n_items,)
        assert self.b_array.shape == (self.meta.n_items, self.meta.n_grades - 1)
        assert self.t_array.shape == (self.meta.n_persons,)
        if self.level_mean_array is not None:
            assert self.level_mean_array.shape == (self.meta.n_levels, self.meta.n_grades - 1)
            assert self.level_std_array.shape == (self.meta.n_levels, self.meta.n_grades - 1)

    def make_a_df(self) -> pd.DataFrame:
        """

        :return: columns=(item, a)
        """
        return pd.DataFrame().assign(item=self.meta.item_category.categories, a=self.a_array)

    def make_b_df(self) -> pd.DataFrame:
        """
        |item|grade| b |
        |foo |  2  |   |
        |foo |  3  |   |
        |foo |  4  |   |
        |bar |  2  |   |
        |bar |  3  |   |
        |bar |  4  |   |
        ...

        :return: columns=(item, grade, b)
        """
        return pd.DataFrame(
            product(self.meta.item_category.categories,
                    np.arange(2, self.meta.n_grades + 1)),
            columns=["item", "grade"])\
            .assign(b=self.b_array.flatten())

    def make_t_df(self) -> pd.DataFrame:
        """

        :return: columns=(person, t)
        """
        return pd.DataFrame().assign(person=self.meta.person_category.categories, t=self.t_array)

    def make_level_df(self) -> pd.DataFrame:
        """
        |level|grade| b |
        | foo |  2  |   |
        | foo |  3  |   |
        | foo |  4  |   |
        | bar |  2  |   |
        | bar |  3  |   |
        | bar |  4  |   |
        ...

        :return: columns=(level, grade, mean, std)
        """
        return pd.DataFrame(
            product(self.meta.level_category.categories,
                    np.arange(2, self.meta.n_grades + 1)),
            columns=["level", "grade"]) \
            .assign(mean=self.level_mean_array.flatten(),
                    std=self.level_std_array.flatten())

    def to_csvs(self, dir_path: str):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        self.make_a_df().to_csv(dir_path / "a.csv", index=False)
        self.make_b_df().to_csv(dir_path / "b.csv", index=False)
        self.make_t_df().to_csv(dir_path / "t.csv", index=False)
        if self.level_mean_array is not None:
            self.make_level_df().to_csv(dir_path / "b_prior.csv", index=False)
