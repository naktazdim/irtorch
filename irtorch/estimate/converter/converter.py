from typing import Optional

from irtorch.estimate.entities import InputDFs, OutputDFs
from .meta import GRMMeta
from .inputs import inputs_from_df
from .outputs import make_output_dfs
from irtorch.estimate.model.data import GRMInputs, GRMOutputs


class Converter(object):
    def __init__(self):
        self.meta = None  # type: Optional[GRMMeta]

    def inputs_from_dfs(self, input_dfs: InputDFs) -> GRMInputs:
        meta, inputs = inputs_from_df(input_dfs)
        self.meta = meta
        return inputs

    def outputs_to_dfs(self, outputs: GRMOutputs) -> OutputDFs:
        return make_output_dfs(outputs, self.meta)
