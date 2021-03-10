from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .meta import GRMMeta
from irtorch.estimate.model import GRMInputs, GRMShapes


@dataclass()
class InputDFs:
    """
    response_df: columns=["item", "person", "response"]
    level_df: columns=["item", "level"]
    """
    response_df: pd.DataFrame
    level_df: Optional[pd.DataFrame] = None


def inputs_from_df(input_dfs: InputDFs) -> Tuple[GRMMeta, GRMInputs]:
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

    meta = GRMMeta(
        item_category,
        person_category,
        response_df.response.max(),
        level_category
    )
    inputs = GRMInputs(
        np.c_[response_df.item.cat.codes.values,
              response_df.person.cat.codes.values,
              response_df.response.values],
        GRMShapes(
            len(item_category.categories),
            len(person_category.categories),
            meta.n_grades,
            len(response_df),
            len(level_category.categories)
        ),
        level_df.level.cat.codes.values
    )

    return meta, inputs
