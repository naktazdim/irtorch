import tempfile

from irtorch.estimate import estimate
from irtorch.estimate.entities import Dataset
from tests.util import df


def test_estimate():
    # とりあえずこけずに動くことだけ確認する
    with tempfile.TemporaryDirectory() as temp_dir:
        estimate(
            Dataset(
                df("input", "response.csv"),
                level_df=df("input", "level.csv")
            ),
            n_iter=2,
            batch_size=1,
            out_dir=temp_dir,
            log_dir=temp_dir,
        )