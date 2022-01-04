import fire

from irtorch.dataset import Dataset, Converter
import irtorch.model.estimator


def estimate(
        dataset: Dataset,
        out_dir: str,
        log_dir: str,
        n_iter: int,
        batch_size: int,
        patience: int = None,
):
    converter = Converter()
    grm_inputs = converter.inputs_from_dfs(dataset)
    irtorch.model.estimator.estimate(grm_inputs,
                                     log_dir,
                                     n_iter,
                                     batch_size,
                                     lambda grm_outputs: converter.outputs_to_dfs(grm_outputs).to_csvs(out_dir),
                                     patience)


def estimate_cli(
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
    fire.Fire(estimate_cli)
