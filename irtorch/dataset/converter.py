from typing import Optional

from irtorch.dataset.entities import Dataset, Predictions
from irtorch.model.data import GRMInputs, GRMOutputs
from .converter_impl import GRMMeta, inputs_from_dfs, outputs_to_dfs


class Converter(object):
    def __init__(self):
        self.meta: Optional[GRMMeta] = None

    def inputs_from_dfs(self, dataset: Dataset) -> GRMInputs:
        self.meta, inputs = inputs_from_dfs(dataset)
        return inputs

    def outputs_to_dfs(self, outputs: GRMOutputs) -> Predictions:
        return outputs_to_dfs(self.meta, outputs)
