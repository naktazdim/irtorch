from itertools import product
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

from .meta import GRMMeta


@dataclass()
class GRMOutputs:
    a_array: np.ndarray
    b_array: np.ndarray
    t_array: np.ndarray
    level_mean_array: Optional[np.ndarray] = None
    level_std_array: Optional[np.ndarray] = None

    def make_a_df(self, meta: GRMMeta) -> pd.DataFrame:
        """

        :return: columns=(item, a)
        """
        return pd.DataFrame().assign(item=meta.item_category.categories, a=self.a_array)

    def make_b_df(self, meta: GRMMeta) -> pd.DataFrame:
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
            product(meta.item_category.categories,
                    np.arange(2, meta.n_grades + 1)),
            columns=["item", "grade"])\
            .assign(b=self.b_array.flatten())

    def make_t_df(self, meta: GRMMeta) -> pd.DataFrame:
        """

        :return: columns=(person, t)
        """
        return pd.DataFrame().assign(person=meta.person_category.categories, t=self.t_array)

    def make_level_df(self, meta: GRMMeta) -> pd.DataFrame:
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
            product(meta.level_category.categories,
                    np.arange(2, meta.n_grades + 1)),
            columns=["level", "grade"]) \
            .assign(mean=self.level_mean_array.flatten(),
                    std=self.level_std_array.flatten())

    def to_csvs(self, dir_path: str, meta: GRMMeta):
        assert self.a_array.shape == (meta.n_items,)
        assert self.b_array.shape == (meta.n_items, meta.n_grades - 1)
        assert self.t_array.shape == (meta.n_persons,)
        if self.level_mean_array is not None:
            assert self.level_mean_array.shape == (meta.n_levels, meta.n_grades - 1)
            assert self.level_std_array.shape == (meta.n_levels, meta.n_grades - 1)

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        self.make_a_df(meta).to_csv(dir_path / "a.csv", index=False)
        self.make_b_df(meta).to_csv(dir_path / "b.csv", index=False)
        self.make_t_df(meta).to_csv(dir_path / "t.csv", index=False)
        if self.level_mean_array is not None:
            self.make_level_df(meta).to_csv(dir_path / "b_prior.csv", index=False)
