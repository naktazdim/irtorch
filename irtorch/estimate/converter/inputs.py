from typing import Tuple

import numpy as np
import pandas as pd

from .meta import GRMMeta
from irtorch.estimate.model import GRMInputs


def inputs_from_df(response_df: pd.DataFrame,
                   level_df: pd.DataFrame = None) -> Tuple[GRMMeta, GRMInputs]:
    """

    :param response_df: columns=["item", "person", "response"]
    :param level_df: columns=["item", "level"]
    """
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
              response_df.response.values]
    )

    if level_df is not None:
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

    return meta, inputs
