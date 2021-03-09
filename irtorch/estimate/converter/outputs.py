from itertools import product
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .meta import GRMMeta
from irtorch.estimate.model import GRMOutputs


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
    assert outputs.level_mean_array.shape == (meta.n_levels, meta.n_grades - 1)
    assert outputs.level_std_array.shape == (meta.n_levels, meta.n_grades - 1)

    a_df = pd.DataFrame().assign(item=meta.item_category.categories, a=outputs.a_array)
    b_df = pd.DataFrame(
        product(meta.item_category.categories,
                np.arange(2, meta.n_grades + 1)),
        columns=["item", "grade"]) \
        .assign(b=outputs.b_array.flatten())
    t_df = pd.DataFrame().assign(person=meta.person_category.categories, t=outputs.t_array)
    level_df = pd.DataFrame(
        product(meta.level_category.categories,
                np.arange(2, meta.n_grades + 1)),
        columns=["level", "grade"]) \
        .assign(mean=outputs.level_mean_array.flatten(),
                std=outputs.level_std_array.flatten())

    return OutputDFs(a_df, b_df, t_df, level_df)


def to_csvs(outputs: GRMOutputs, dir_path: str, meta: GRMMeta):
    make_output_dfs(outputs, meta).to_csvs(dir_path)
