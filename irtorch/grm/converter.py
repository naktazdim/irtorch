import numpy as np
import pandas as pd

from irtorch.grm.prior import GaussianPrior


class GRMDataConverter(object):
    def __init__(self, response_df: pd.DataFrame, n_grades: int = None):
        """

        :param response_df: columns=["item", "person", "response"]
        :param n_grades: 項目数。指定しない場合は response の最大値
        """
        assert "item" in response_df.columns
        assert "person" in response_df.columns
        assert "response" in response_df.columns

        assert np.issubdtype(response_df.response.dtype, np.integer)
        assert response_df.response.min() >= 1

        self.response_df = response_df.copy()
        self.response_df["item"] = self.response_df["item"].astype("category")
        self.response_df["person"] = self.response_df["person"].astype("category")

        self.item_category = self.response_df.item.dtype
        self.person_category = self.response_df.person.dtype

        self.n_items = len(self.item_category.categories)
        self.n_persons = len(self.person_category.categories)
        self.n_grades = n_grades or response_df.response.max()

    def make_item_array(self) -> np.ndarray:
        """

        :return: shape=(n_responses,)
        """
        return self.response_df.item.cat.codes.values

    def make_person_array(self) -> np.ndarray:
        """

        :return: shape=(n_responses,)
        """
        return self.response_df.person.cat.codes.values

    def make_response_array(self) -> np.ndarray:
        """

        :return: shape=(n_responses,)
        """
        return self.response_df.response.values

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

    def make_a_prior(self, a_prior_df: pd.DataFrame = None) -> GaussianPrior:
        """

        :param a_prior_df: columns=(item, mean, std)
        :return: shape=(n_items,)
        """
        if a_prior_df is None:
            a_prior_df = pd.DataFrame(columns=["item", "mean", "std"])
        else:
            assert "item" in a_prior_df.columns
            assert "mean" in a_prior_df.columns
            assert "std" in a_prior_df.columns
        a_prior_df = a_prior_df.astype({"item": self.item_category})

        a_prior_df_ordered = (
            self._make_a_df_base()
                .merge(a_prior_df, how="left")
                .fillna({"mean": 0.0, "std": 1.0})
        )

        return GaussianPrior(
            a_prior_df_ordered["mean"].values,
            a_prior_df_ordered["std"].values
        )

    def make_b_prior(self, b_prior_df: pd.DataFrame = None) -> GaussianPrior:
        """

        :param b_prior_df: columns=(item, grade, mean, std)
        :return: shape=(n_items, n_grades - 1)
        """
        if b_prior_df is None:
            b_prior_df = pd.DataFrame(columns=["item", "grade", "mean", "std"])
        else:
            assert "item" in b_prior_df.columns
            assert "grade" in b_prior_df.columns
            assert "mean" in b_prior_df.columns
            assert "std" in b_prior_df.columns
        b_prior_df = b_prior_df.astype({"item": self.item_category})

        b_prior_df_ordered = (
            self._make_b_df_base()
                .merge(b_prior_df, how="left")
                .fillna({"mean": 0.0, "std": 1.0})
        )

        return GaussianPrior(
            b_prior_df_ordered["mean"].values,
            b_prior_df_ordered["std"].values
        ).reshape(self.n_items, self.n_grades - 1)

    def make_t_prior(self, t_prior_df: pd.DataFrame = None) -> GaussianPrior:
        """

        :param t_prior_df: columns=(person, mean, std)
        :return: shape=(n_persons,)
        """
        if t_prior_df is None:
            t_prior_df = pd.DataFrame(columns=["person", "mean", "std"])
        else:
            assert "person" in t_prior_df.columns
            assert "mean" in t_prior_df.columns
            assert "std" in t_prior_df.columns
        t_prior_df = t_prior_df.astype({"person": self.person_category})

        t_prior_df_ordered = (
            self._make_t_df_base()
                .merge(t_prior_df, how="left")
                .fillna({"mean": 0.0, "std": 1.0})
        )

        return GaussianPrior(
            t_prior_df_ordered["mean"].values,
            t_prior_df_ordered["std"].values
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
