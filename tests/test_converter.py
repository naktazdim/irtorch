import pytest

import numpy as np
import pandas as pd

from irtorch.converter import GRMDataConverter
from tests.util import df, array


@pytest.fixture()
def converter() -> GRMDataConverter:
    return GRMDataConverter(df("input", "response.csv"), df("input", "level.csv"))


def test_make_response_array(converter):
    np.testing.assert_array_equal(
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

