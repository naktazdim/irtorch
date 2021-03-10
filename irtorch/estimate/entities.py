from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass()
class InputDFs:
    """
    response_df: columns=["item", "person", "response"]
    level_df: columns=["item", "level"]
    """
    response_df: pd.DataFrame
    level_df: Optional[pd.DataFrame] = None


@dataclass()
class OutputDFs:
    a: pd.DataFrame
    b: pd.DataFrame
    t: pd.DataFrame
    level: Optional[pd.DataFrame] = None

    def to_csvs(self, dir_path: str):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.a.to_csv(dir_path / "a.csv", index=False)
        self.b.to_csv(dir_path / "b.csv", index=False)
        self.t.to_csv(dir_path / "t.csv", index=False)
        if self.level is not None:
            self.level.to_csv(dir_path / "b_prior.csv", index=False)
