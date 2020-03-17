from itertools import product

import numpy as np
import pandas as pd


class GRMDataConverter(object):
    def __init__(self,
                 response_df: pd.DataFrame,
                 level_df: pd.DataFrame = None):
        """

        :param response_df: columns=["item", "person", "response"]
        :param level_df: columns=["item", "level"]
        """
        assert "item" in response_df.columns
        assert "person" in response_df.columns
        assert "response" in response_df.columns

        assert np.issubdtype(response_df.response.dtype, np.integer)
        assert response_df.response.min() >= 1

        self.response_df = response_df.astype({"item": "category", "person": "category"})

        self.item_category = self.response_df.item.dtype
        self.person_category = self.response_df.person.dtype

        self.n_items = len(self.item_category.categories)
        self.n_persons = len(self.person_category.categories)
        self.n_responses = len(response_df)
        self.n_grades = response_df.response.max()

        if level_df is None:
            self.is_hierarchical = False
        else:
            self.is_hierarchical = True

            assert "item" in level_df.columns
            assert "level" in level_df.columns
            assert level_df.item.unique().all()

            self.level_df = pd.merge(
                self.item_category.categories.to_frame(name="item"),
                level_df
                    .drop_duplicates(subset="item")
                    .astype({"item": self.item_category}),
                how="left"
            )
            self.level_df["level"] = self.level_df.level.fillna("_unknown").astype({"level": "category"})
            self.level_category = self.level_df.level.dtype
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

        :return: shape=(n_items,)
        """
        return self.level_df.level.cat.codes.values if self.is_hierarchical else None

    def make_a_df(self, a_array: np.ndarray) -> pd.DataFrame:
        """

        :param a_array: shape=(n_items,)
        :return: columns=(item, a)
        """
        assert a_array.shape == (self.n_items,)
        return pd.DataFrame().assign(item=self.item_category.categories, a=a_array)

    def make_b_df(self, b_array: np.ndarray) -> pd.DataFrame:
        """
        |item|grade| b |
        |foo |  2  |   |
        |foo |  3  |   |
        |foo |  4  |   |
        |bar |  2  |   |
        |bar |  3  |   |
        |bar |  4  |   |
        ...

        :param b_array: shape=(n_items, n_grades - 1)
        :return: columns=(item, grade, b)
        """
        assert b_array.shape == (self.n_items, self.n_grades - 1)
        return pd.DataFrame(
            product(self.item_category.categories,
                    np.arange(2, self.n_grades + 1)),
            columns=["item", "grade"])\
            .assign(b=b_array.flatten())

    def make_t_df(self, t_array: np.ndarray) -> pd.DataFrame:
        """

        :param t_array: shape=(n_persons,)
        :return: columns=(person, t)
        """
        assert t_array.shape == (self.n_persons,)
        return pd.DataFrame().assign(person=self.person_category.categories, t=t_array)

    def make_level_df(self, level_mean_array: np.ndarray, level_std_array: np.ndarray) -> pd.DataFrame:
        """
        |level|grade| b |
        | foo |  2  |   |
        | foo |  3  |   |
        | foo |  4  |   |
        | bar |  2  |   |
        | bar |  3  |   |
        | bar |  4  |   |
        ...

        :param level_mean_array: shape=(n_levels, n_grades - 1)
        :param level_std_array: shape=(n_levels, n_grades - 1)
        :return: columns=(level, grade, mean, std)
        """
        assert level_mean_array.shape == (self.n_levels, self.n_grades - 1)
        assert level_std_array.shape == (self.n_levels, self.n_grades - 1)
        return pd.DataFrame(
            product(self.level_category.categories,
                    np.arange(2, self.n_grades + 1)),
            columns=["item", "grade"]) \
            .assign(mean=level_mean_array.flatten(),
                    std=level_std_array.flatten())
