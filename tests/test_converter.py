import numpy as np
import pandas as pd

from irtorch.converter import Converter
from irtorch.entities import Dataset
from irtorch.model.data import GRMOutputs
from tests.util import df, array


def test_converter():
    converter = Converter()
    input_dfs = Dataset(
        df("input", "response.csv"),
        df("input", "level.csv"),
    )
    grm_inputs = converter.inputs_from_dfs(input_dfs)

    np.testing.assert_array_equal(
        grm_inputs.response_array,
        array("array", "response_array.csv")
    )

    grm_outputs = GRMOutputs(
        array("array", "a_array.csv"),
        array("array", "b_array.csv"),
        array("array", "t_array.csv"),
        array("array", "level_mean_array.csv"),
        array("array", "level_mean_array.csv")
    )
    output_dfs = converter.outputs_to_dfs(grm_outputs)

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
