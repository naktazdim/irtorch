import pytest

import numpy as np
import pandas as pd

from irtorch.estimate.converter import GRMInputs, GRMOutputs
from tests.util import df, array


@pytest.fixture()
def inputs() -> GRMInputs:
    return GRMInputs.from_df(df("input", "response.csv"), df("input", "level.csv"))


@pytest.fixture()
def outputs() -> GRMOutputs:
    return GRMOutputs(
        array("input", "a_array.csv"),
        array("input", "b_array.csv"),
        array("input", "t_array.csv")
    )


def test_make_response_array(inputs):
    np.testing.assert_array_equal(
        inputs.response_array,
        array("output", "response_array.csv", dtype=int)
    )


def test_make_a_df(inputs, outputs):
    pd.testing.assert_frame_equal(
        outputs.make_a_df(inputs.meta),
        df("output", "a.csv"),
        check_dtype=False,
        check_categorical=False
    )


def test_make_b_df(inputs, outputs):
    pd.testing.assert_frame_equal(
        outputs.make_b_df(inputs.meta),
        df("output", "b.csv"),
        check_dtype=False,
        check_categorical=False
    )


def test_make_t_df(inputs, outputs):
    pd.testing.assert_frame_equal(
        outputs.make_t_df(inputs.meta),
        df("output", "t.csv"),
        check_dtype=False,
        check_categorical=False
    )
