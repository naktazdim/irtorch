import tempfile

from irtorch.estimate import estimate
from tests.util import df


def test_estimate():
    # とりあえずこけずに動くことだけ確認する
    with tempfile.TemporaryDirectory() as temp_dir:
        estimate(
            df("input", "response.csv"),
            n_iter=2,
            batch_size=1,
            out_dir=temp_dir,
            log_dir=temp_dir,
            level_df=df("input", "level.csv")
        )