import argparse

import pandas as pd

from irtorch import estimate
from irtorch.estimate.entities import Dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("response", type=str)
    p.add_argument("--level", type=str)
    p.add_argument("-n", "--n-iter", type=int, default=1000)
    p.add_argument("-b", "--batch-size", type=int, default=1000)
    p.add_argument("-p", "--patience", type=int)
    p.add_argument("-o", "--out-dir", type=str, default=".")
    p.add_argument("-l", "--log-dir", type=str, default=".")
    args = p.parse_args()

    dataset = Dataset(
        pd.read_csv(args.response),
        pd.read_csv(args.level) if args.level else None,
    )
    estimate(
        dataset,
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        patience=args.patience,
        out_dir=args.out_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
