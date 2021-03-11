import fire

from irtorch import estimate
from irtorch.entities import Dataset


def main(
        response: str,
        level: str = None,
        n_iter: int = 1000,
        batch_size: int = 1000,
        patience: int = None,
        out_dir: str = ".",
        log_dir: str = "."
):
    estimate(
        Dataset.from_csvs(response, level),
        n_iter=n_iter,
        batch_size=batch_size,
        patience=patience,
        out_dir=out_dir,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    fire.Fire(main)
