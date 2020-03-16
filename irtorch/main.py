import argparse

import pandas as pd

from irtorch.estimate import estimate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("response", type=str)
    p.add_argument("--level", type=str)
    p.add_argument("-n", "--n-iter", type=int, default=1000)
    p.add_argument("-b", "--batch-size", type=int, default=1000)
    p.add_argument("-o", "--out-dir", type=str, default=".")
    p.add_argument("-l", "--log-dir", type=str, default=".")
    args = p.parse_args()

    estimate(
        pd.read_csv(args.response),
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        log_dir=args.log_dir,
        level_df=pd.read_csv(args.level) if args.level else None,
    )


if __name__ == "__main__":
    main()
