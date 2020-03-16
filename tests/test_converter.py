import pytest
import os

import numpy as np
import numpy.testing
import pandas as pd

from irtorch.converter import GRMDataConverter


def resource_path(*paths: str) -> str:
    return os.path.join(os.path.dirname(__file__), "resources", "converter", *paths)


def df(*paths: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(resource_path(*paths), **kwargs)


def array(*paths: str, **kwargs) -> np.ndarray:
    return np.loadtxt(resource_path(*paths), delimiter=",", **kwargs)


@pytest.fixture()
def converter() -> GRMDataConverter:
    return GRMDataConverter(df("input", "response.csv"), df("input", "level.csv"))


def test_make_response_array(converter):
    numpy.testing.assert_array_equal(
        converter.make_response_array(),
        array("output", "response_array.csv", dtype=int)
    )


def test_make_a_df(converter):
    pd.testing.assert_frame_equal(
        converter.make_a_df(array("input", "a_array.csv")),
        df("output", "a.csv"),
        check_dtype=False,
        check_categorical=False
    )


def test_make_b_df(converter):
    pd.testing.assert_frame_equal(
        converter.make_b_df(array("input", "b_array.csv")),
        df("output", "b.csv"),
        check_dtype=False,
        check_categorical=False
    )


def test_make_t_df(converter):
    pd.testing.assert_frame_equal(
        converter.make_t_df(array("input", "t_array.csv")),
        df("output", "t.csv"),
        check_dtype=False,
        check_categorical=False
    )

