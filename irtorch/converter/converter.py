from itertools import product

import numpy as np
import pandas as pd

from .inputs import GRMInputs


class GRMDataConverter(object):
    def __init__(self,
                 response_df: pd.DataFrame,
                 level_df: pd.DataFrame = None):
        """

        :param response_df: columns=["item", "person", "response"]
        :param level_df: columns=["item", "level"]
        """
        self.is_hierarchical = (level_df is not None)
        self._inputs = GRMInputs.from_df(response_df, level_df)
        self.meta = self._inputs.meta

    def make_response_array(self) -> np.ndarray:
        """

        :return: shape=(n_responses, 3)
        """
        return self._inputs.response_array

    def make_level_array(self) -> np.ndarray:
        """

        :return: shape=(n_items,)
        """
        return self._inputs.level_array

    def make_a_df(self, a_array: np.ndarray) -> pd.DataFrame:
        """

        :param a_array: shape=(n_items,)
        :return: columns=(item, a)
        """
        assert a_array.shape == (self.meta.n_items,)
        return pd.DataFrame().assign(item=self.meta.item_category.categories, a=a_array)

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
        assert b_array.shape == (self.meta.n_items, self.meta.n_grades - 1)
        return pd.DataFrame(
            product(self.meta.item_category.categories,
                    np.arange(2, self.meta.n_grades + 1)),
            columns=["item", "grade"])\
            .assign(b=b_array.flatten())

    def make_t_df(self, t_array: np.ndarray) -> pd.DataFrame:
        """

        :param t_array: shape=(n_persons,)
        :return: columns=(person, t)
        """
        assert t_array.shape == (self.meta.n_persons,)
        return pd.DataFrame().assign(person=self.meta.person_category.categories, t=t_array)

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
        assert level_mean_array.shape == (self.meta.n_levels, self.meta.n_grades - 1)
        assert level_std_array.shape == (self.meta.n_levels, self.meta.n_grades - 1)
        return pd.DataFrame(
            product(self.meta.level_category.categories,
                    np.arange(2, self.meta.n_grades + 1)),
            columns=["item", "grade"]) \
            .assign(mean=level_mean_array.flatten(),
                    std=level_std_array.flatten())
