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

    assert "item" in response_df.columns
    assert "person" in response_df.columns
    assert "response" in response_df.columns

    assert np.issubdtype(response_df.response.dtype, np.integer)
    assert response_df.response.min() >= 1

    response_df = response_df.astype({"item": "category", "person": "category"})
    meta = GRMMeta(
        response_df.item.dtype,
        response_df.person.dtype,
        response_df.response.max()
    )
    inputs = GRMInputs(
        np.c_[response_df.item.cat.codes.values,
              response_df.person.cat.codes.values,
              response_df.response.values],
        GRMShapes(
            len(meta.item_category.categories),
            len(meta.person_category.categories),
            meta.n_grades,
            len(response_df)
        )
    )

    if level_df is None:
        level_df = pd.DataFrame(columns=["item", "level"])

    assert "item" in level_df.columns
    assert "level" in level_df.columns
    assert level_df.item.unique().all()

    level_df = pd.merge(
        meta.item_category.categories.to_frame(name="item"),
        level_df
            .drop_duplicates(subset="item")
            .astype({"item": meta.item_category}),
        how="left"
    )
    level_df["level"] = level_df.level.fillna("_unknown").astype({"level": "category"})

    meta.level_category = level_df.level.dtype
    inputs.level_array = level_df.level.cat.codes.values
    inputs.shapes.n_levels = len(meta.level_category.categories)

    return meta, inputs
