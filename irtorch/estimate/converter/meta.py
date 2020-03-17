from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass()
class GRMMeta:
    item_category: pd.Categorical
    person_category: pd.Categorical
    n_grades: int
    level_category: Optional[pd.Categorical] = None

    @property
    def n_items(self) -> int:
        return len(self.item_category.categories)

    @property
    def n_persons(self) -> int:
        return len(self.person_category.categories)

    @property
    def n_levels(self) -> Optional[int]:
        return None if self.level_category is None else len(self.level_category.categories)
