from typing import Tuple
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from irtorch.dataset.entities import Dataset, Predictions
from irtorch.model.data import GRMInputs, GRMOutputs, GRMShapes


@dataclass()
class GRMMeta:
    item_category: pd.Categorical
    person_category: pd.Categorical
    level_category: pd.Categorical
    n_items: int
    n_persons: int
    n_grades: int
    n_levels: int


def inputs_from_dfs(dataset: Dataset) -> Tuple[GRMMeta, GRMInputs]:
    response_df, level_df = dataset.response_df, dataset.level_df
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

    n_items = len(item_category.categories)
    n_persons = len(person_category.categories)
    n_responses = len(response_df)
    n_grades = response_df.response.max()
    n_levels = len(level_category.categories)

    meta = GRMMeta(
        item_category,
        person_category,
        level_category,
        n_items, n_persons, n_grades, n_levels
    )
    inputs = GRMInputs(
        np.c_[response_df.item.cat.codes.values,
              response_df.person.cat.codes.values,
              response_df.response.values],
        GRMShapes(n_items, n_persons, n_grades, n_responses, n_levels),
        level_df.level.cat.codes.values
    )
    return meta, inputs


def outputs_to_dfs (meta: GRMMeta, outputs: GRMOutputs) -> Predictions:
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

    return Predictions(a_df, b_df, t_df, level_df)