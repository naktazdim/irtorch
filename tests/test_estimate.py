import tempfile

import pytest
import pandas as pd

from irtorch.estimate import estimate
from tests.util import df


@pytest.fixture()
def response_df() -> pd.DataFrame:
    return df("input", "response.csv")


@pytest.fixture()
def level_df() -> pd.DataFrame:
    return df("input", "level.csv")


def test_estimate(response_df, level_df):
    # とりあえずこけずに動くことだけ確認する
    with tempfile.TemporaryDirectory() as temp_dir:
        estimate(
            response_df,
            n_iter=2,
            batch_size=1,
            out_dir=temp_dir,
            log_dir=temp_dir,
            level_df=level_df
        )