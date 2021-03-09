from typing import Tuple

import pytest

import numpy as np
import pandas as pd

from irtorch.estimate.converter import inputs_from_df, GRMMeta, InputDFs
from irtorch.estimate.model import GRMInputs, GRMOutputs
from irtorch.estimate.converter.outputs import make_output_dfs
from tests.util import df, array


@pytest.fixture()
def meta_and_inputs() -> Tuple[GRMMeta, GRMInputs]:
    input_dfs = InputDFs(df("input", "response.csv"), df("input", "level.csv"))
    return inputs_from_df(input_dfs)


def test_make_response_array(meta_and_inputs):
    _, inputs = meta_and_inputs
    np.testing.assert_array_equal(
        inputs.response_array,
        array("array", "response_array.csv", dtype=int)
    )


def test_make_output_dfs(meta_and_inputs):
    meta, _ = meta_and_inputs
    outputs = GRMOutputs(
        array("array", "a_array.csv"),
        array("array", "b_array.csv"),
        array("array", "t_array.csv"),
        array("array", "level_mean_array.csv"),
        array("array", "level_mean_array.csv")
    )

    output_dfs = make_output_dfs(outputs, meta)
    pd.testing.assert_frame_equal(
        output_dfs.a,
        df("output", "a.csv"),
        check_dtype=False,
        check_categorical=False
    )
    pd.testing.assert_frame_equal(
        output_dfs.b,
        df("output", "b.csv"),
        check_dtype=False,
        check_categorical=False
    )
    pd.testing.assert_frame_equal(
        output_dfs.t,
        df("output", "t.csv"),
        check_dtype=False,
        check_categorical=False
    )
    pd.testing.assert_frame_equal(
        output_dfs.level,
        df("output", "b_prior.csv"),
        check_dtype=False,
        check_categorical=False
    )
