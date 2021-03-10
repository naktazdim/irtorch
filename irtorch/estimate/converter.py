from typing import Optional
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from irtorch.estimate.entities import InputDFs, OutputDFs
from irtorch.estimate.model.data import GRMInputs, GRMOutputs, GRMShapes


@dataclass()
class GRMMeta:
    item_category: pd.Categorical
    person_category: pd.Categorical
    n_grades: int
    level_category: Optional[pd.Categorical] = None

    @property
    def n_items(self) -> int:
        return len(self.item_category.categories)

    @property
    def n_persons(self) -> int:
        return len(self.person_category.categories)

    @property
    def n_levels(self) -> Optional[int]:
        return None if self.level_category is None else len(self.level_category.categories)


class Converter(object):
    def __init__(self):
        self.meta = None  # type: Optional[GRMMeta]

    def inputs_from_dfs(self, input_dfs: InputDFs) -> GRMInputs:
        response_df, level_df = input_dfs.response_df, input_dfs.level_df
        if level_df is None:
            level_df = pd.DataFrame(columns=["item", "level"])

        assert "item" in response_df.columns
        assert "person" in response_df.columns
        assert "response" in response_df.columns

        assert np.issubdtype(response_df.response.dtype, np.integer)
        assert response_df.response.min() >= 1

        assert "item" in level_df.columns
        assert "level" in level_df.columns
        assert level_df.item.unique().all()

        response_df = response_df.astype({"item": "category", "person": "category"})
        item_category = response_df.item.dtype
        person_category = response_df.person.dtype

        level_df = pd.merge(
            item_category.categories.to_frame(name="item"),
            level_df
                .drop_duplicates(subset="item", keep="first")
                .astype({"item": item_category}),
            how="left"
        )
        level_df["level"] = level_df.level.fillna("_unknown").astype({"level": "category"})
        level_category = level_df.level.dtype

        self.meta = GRMMeta(
            item_category,
            person_category,
            response_df.response.max(),
            level_category
        )
        return GRMInputs(
            np.c_[response_df.item.cat.codes.values,
                  response_df.person.cat.codes.values,
                  response_df.response.values],
            GRMShapes(
                len(item_category.categories),
                len(person_category.categories),
                self.meta.n_grades,
                len(response_df),
                len(level_category.categories)
            ),
            level_df.level.cat.codes.values
        )

    def outputs_to_dfs(self, outputs: GRMOutputs) -> OutputDFs:
        assert outputs.a_array.shape == (self.meta.n_items,)
        assert outputs.b_array.shape == (self.meta.n_items, self.meta.n_grades - 1)
        assert outputs.t_array.shape == (self.meta.n_persons,)
        assert outputs.level_mean_array.shape == (self.meta.n_levels, self.meta.n_grades - 1)
        assert outputs.level_std_array.shape == (self.meta.n_levels, self.meta.n_grades - 1)

        a_df = pd.DataFrame().assign(item=self.meta.item_category.categories, a=outputs.a_array)
        b_df = pd.DataFrame(
            product(self.meta.item_category.categories,
                    np.arange(2, self.meta.n_grades + 1)),
            columns=["item", "grade"]) \
            .assign(b=outputs.b_array.flatten())
        t_df = pd.DataFrame().assign(person=self.meta.person_category.categories, t=outputs.t_array)
        level_df = pd.DataFrame(
            product(self.meta.level_category.categories,
                    np.arange(2, self.meta.n_grades + 1)),
            columns=["level", "grade"]) \
            .assign(mean=outputs.level_mean_array.flatten(),
                    std=outputs.level_std_array.flatten())

        return OutputDFs(a_df, b_df, t_df, level_df)