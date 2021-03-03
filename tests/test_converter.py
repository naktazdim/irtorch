from typing import Tuple

import pytest

import numpy as np
import pandas as pd

from irtorch.estimate.converter import inputs_from_df, GRMMeta
from irtorch.estimate.model import GRMInputs, GRMOutputs
from irtorch.estimate.converter.outputs import make_a_df, make_b_df, make_t_df, to_csvs
from tests.util import df, array


@pytest.fixture()
def meta_and_inputs() -> Tuple[GRMMeta, GRMInputs]:
    return inputs_from_df(df("input", "response.csv"), df("input", "level.csv"))


@pytest.fixture()
def outputs() -> GRMOutputs:
    return GRMOutputs(
        array("input", "a_array.csv"),
        array("input", "b_array.csv"),
        array("input", "t_array.csv")
    )


def test_make_response_array(meta_and_inputs):
    _, inputs = meta_and_inputs
    np.testing.assert_array_equal(
        inputs.response_array,
        array("output", "response_array.csv", dtype=int)
    )


def test_make_a_df(meta_and_inputs, outputs):
    meta, _ = meta_and_inputs
    pd.testing.assert_frame_equal(
        make_a_df(outputs, meta),
        df("output", "a.csv"),
        check_dtype=False,
        check_categorical=False
    )


def test_make_b_df(meta_and_inputs, outputs):
    meta, _ = meta_and_inputs
    pd.testing.assert_frame_equal(
        make_b_df(outputs, meta),
        df("output", "b.csv"),
        check_dtype=False,
        check_categorical=False
    )


def test_make_t_df(meta_and_inputs, outputs):
    meta, _ = meta_and_inputs
    pd.testing.assert_frame_equal(
        make_t_df(outputs, meta),
        df("output", "t.csv"),
        check_dtype=False,
        check_categorical=False
    )
