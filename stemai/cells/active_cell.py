from cells.blanket_cell import BlanketCell
from pymdp import utils
import numpy as np


class ActiveCell(BlanketCell):

    def __init__(
        self,
        node_idx,
        internal_cells,
        external_cells,
        states,
    ):

        super().__init__(
            node_idx,
            internal_cells,
            external_cells,
            states,
        )

        self.cell_type = "active"

        self.actions_received = {
            n: 0 for n in internal_cells
        }  # keep track of what you received and from who

        self.actions_sent = {n: 0 for n in external_cells}

        self.setup(
            self.states,
            hidden_state_indices=self.internal_neighbor_indices,
            control_state_indices=self.external_neighbor_indices,
        )
        self.build_generative_model()


#000 001, 011, 

#011, 110, 101, 111 