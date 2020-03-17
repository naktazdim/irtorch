from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .meta import GRMMeta


@dataclass()
class GRMInputs:
    meta: GRMMeta
    response_array: np.ndarray
    level_array: Optional[np.ndarray] = None

    @classmethod
    def from_df(cls,
                response_df: pd.DataFrame,
                level_df: pd.DataFrame = None):
        """

        :param response_df: columns=["item", "person", "response"]
        :param level_df: columns=["item", "level"]
        """
        ret = cls._from_response_df(response_df)
        if level_df is not None:
            ret._add_level(level_df)
        return ret

    @classmethod
    def _from_response_df(cls, response_df: pd.DataFrame):
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
        response_array = np.c_[
            response_df.item.cat.codes.values,
            response_df.person.cat.codes.values,
            response_df.response.values
        ]
        return GRMInputs(meta, response_array)

    def _add_level(self, level_df: pd.DataFrame):
        assert "item" in level_df.columns
        assert "level" in level_df.columns
        assert level_df.item.unique().all()

        level_df = pd.merge(
            self.meta.item_category.categories.to_frame(name="item"),
            level_df
                .drop_duplicates(subset="item")
                .astype({"item": self.meta.item_category}),
            how="left"
        )
        level_df["level"] = level_df.level.fillna("_unknown").astype({"level": "category"})

        self.meta.level_category = level_df.level.dtype
        self.level_array = level_df.level.cat.codes.values
