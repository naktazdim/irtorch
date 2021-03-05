from itertools import product
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .meta import GRMMeta
from irtorch.estimate.model import GRMOutputs


def make_a_df(outputs: GRMOutputs, meta: GRMMeta) -> pd.DataFrame:
    """

    :return: columns=(item, a)
    """
    return pd.DataFrame().assign(item=meta.item_category.categories, a=outputs.a_array)


def make_b_df(outputs: GRMOutputs, meta: GRMMeta) -> pd.DataFrame:
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
        columns=["item", "grade"]) \
        .assign(b=outputs.b_array.flatten())


def make_t_df(outputs: GRMOutputs, meta: GRMMeta) -> pd.DataFrame:
    """

    :return: columns=(person, t)
    """
    return pd.DataFrame().assign(person=meta.person_category.categories, t=outputs.t_array)


def make_level_df(outputs: GRMOutputs, meta: GRMMeta) -> pd.DataFrame:
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
        .assign(mean=outputs.level_mean_array.flatten(),
                std=outputs.level_std_array.flatten())


@dataclass()
class OutputDFs:
    a: pd.DataFrame
    b: pd.DataFrame
    t: pd.DataFrame
    level: Optional[pd.DataFrame] = None

    def to_csvs(self, dir_path: str):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.a.to_csv(dir_path / "a.csv", index=False)
        self.b.to_csv(dir_path / "b.csv", index=False)
        self.t.to_csv(dir_path / "t.csv", index=False)
        if self.level is not None:
            self.level.to_csv(dir_path / "b_prior.csv", index=False)


def make_output_dfs(outputs: GRMOutputs, meta: GRMMeta) -> OutputDFs:
    assert outputs.a_array.shape == (meta.n_items,)
    assert outputs.b_array.shape == (meta.n_items, meta.n_grades - 1)
    assert outputs.t_array.shape == (meta.n_persons,)
    if outputs.level_mean_array is not None:
        assert outputs.level_mean_array.shape == (meta.n_levels, meta.n_grades - 1)
        assert outputs.level_std_array.shape == (meta.n_levels, meta.n_grades - 1)

    output_dfs = OutputDFs(
        make_a_df(outputs, meta),
        make_b_df(outputs, meta),
        make_t_df(outputs, meta)
    )
    if outputs.level_mean_array is not None:
        output_dfs.level = make_level_df(outputs, meta)
    return output_dfs


def to_csvs(outputs: GRMOutputs, dir_path: str, meta: GRMMeta):
    make_output_dfs(outputs, meta).to_csvs(dir_path)
