import argparse
from typing import Optional

import pandas as pd

from irtorch.estimate import estimate


def read_csv_or_none(path: Optional[str]):
    return None if path is None else pd.read_csv(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("response", type=str)
    p.add_argument("--a-prior", type=str)
    p.add_argument("--b-prior", type=str)
    p.add_argument("--t-prior", type=str)
    p.add_argument("-n", "--n-iter", type=int, default=1000)
    p.add_argument("-o", "--out-dir", type=str, default=".")
    p.add_argument("-l", "--log-dir", type=str, default=".")
    args = p.parse_args()

    estimate(
        pd.read_csv(args.response),
        a_prior_df=read_csv_or_none(args.a_prior),
        b_prior_df=read_csv_or_none(args.b_prior),
        t_prior_df=read_csv_or_none(args.t_prior),
        n_iter=args.n_iter,
        out_dir=args.out_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
