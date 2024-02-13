from cells.blanket_cell import BlanketCell
from pymdp import utils
import numpy as np


class SensoryCell(BlanketCell):

    def __init__(
        self,
        node_idx,
        internal_cells,
        external_cells,
        internal_and_external_states,
    ):
        super().__init__(
            self,
            node_idx,
            internal_cells,
            external_cells,
            internal_and_external_states,
        )
        self.cell_type = "sensory"

        self.actions_received = {
            n: 0 for n in external_cells
        }  # keep track of what you received and from who

        self.actions_sent = {n: 0 for n in internal_cells}

        self.setup(
            self.internal_and_external_states,
            hidden_state_indices=self.external_neighbor_indices,
            control_state_indices=self.internal_neighbor_indices,
        )
