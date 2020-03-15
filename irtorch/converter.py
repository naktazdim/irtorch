import numpy as np
import pandas as pd


class GRMDataConverter(object):
    def __init__(self,
                 response_df: pd.DataFrame,
                 level_df: pd.DataFrame,
                 n_grades: int = None):
        """

        :param response_df: columns=["item", "person", "response"]
        :param level_df: columns=["item", "level"]
        :param n_grades: 項目数。指定しない場合は response の最大値
        """
        assert "item" in response_df.columns
        assert "person" in response_df.columns
        assert "response" in response_df.columns

        assert "item" in level_df.columns
        assert "level" in level_df.columns

        assert np.issubdtype(response_df.response.dtype, np.integer)
        assert response_df.response.min() >= 1

        self.response_df = response_df.copy()
        self.response_df["item"] = self.response_df["item"].astype("category")
        self.response_df["person"] = self.response_df["person"].astype("category")

        self.item_category = self.response_df.item.dtype
        self.person_category = self.response_df.person.dtype

        self.level_df = (
            level_df[level_df["item"].isin(self.item_category.categories)]
            .astype({"item": self.item_category, "level": "category"})
        )

        self.level_category = self.level_df.level.dtype

        self.n_items = len(self.item_category.categories)
        self.n_persons = len(self.person_category.categories)
        self.n_responses = len(response_df)
        self.n_grades = n_grades or response_df.response.max()
        self.n_levels = len(self.level_category.categories)

    def make_response_array(self) -> np.ndarray:
        """

        :return: shape=(n_responses, 3)
        """
        return np.c_[
            self.response_df.item.cat.codes.values,
            self.response_df.person.cat.codes.values,
            self.response_df.response.values
        ]

    def make_level_array(self) -> np.ndarray:
        """

        :return: shape=(*, 2)
        """
        return np.c_[
            self.level_df.item.cat.codes.values,
            self.level_df.level.cat.codes.values
        ]

    def _make_a_df_base(self) -> pd.DataFrame:
        return pd.DataFrame(
            pd.Categorical.from_codes(
                codes=np.arange(self.n_items),
                dtype=self.item_category
            ),
            columns=["item"]
        )

    def _make_b_df_base(self) -> pd.DataFrame:
        # item|grade
        # foo |  2
        # foo |  3
        # foo |  4
        # bar |  2
        # bar |  3
        # bar |  4
        # ...
        ret = pd.concat(
            [pd.DataFrame(
                {"item": item,
                 "grade": np.arange(2, self.n_grades + 1)},
            ) for item in range(self.n_items)]
        )
        ret["item"] = pd.Categorical.from_codes(ret["item"], dtype=self.item_category)
        return ret.reset_index(drop=True)

    def _make_t_df_base(self) -> pd.DataFrame:
        return pd.DataFrame(
            pd.Categorical.from_codes(
                codes=np.arange(self.n_persons),
                dtype=self.person_category
            ), columns=["person"]
        )

    def make_a_df(self, a_array: np.ndarray) -> pd.DataFrame:
        """

        :param a_array: shape=(n_items,)
        :return: columns=(item, a)
        """
        assert a_array.shape == (self.n_items,)
        return self._make_a_df_base().assign(a=a_array)

    def make_b_df(self, b_array: np.ndarray) -> pd.DataFrame:
        """

        :param b_array: shape=(n_items, n_grades - 1)
        :return: columns=(item, grade, b)
        """
        assert b_array.shape == (self.n_items, self.n_grades - 1)
        return self._make_b_df_base().assign(b=b_array.flatten())

    def make_t_df(self, t_array: np.ndarray) -> pd.DataFrame:
        """

        :param t_array: shape=(n_persons,)
        :return: columns=(person, t)
        """
        assert t_array.shape == (self.n_persons,)
        return self._make_t_df_base().assign(t=t_array.flatten())
