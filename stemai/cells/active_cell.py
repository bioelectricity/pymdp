from cells.blanket_cell import BlanketCell
from pymdp import utils
import numpy as np


class ActiveCell(BlanketCell):
    """Class representing an active cell
    which inherits from blanket cell

    The active cell's hidden state space will be the internal neighbors
    and it's control state space will be external neighbor indices

    #TODO: also need to connect active cells to sensory cells"""

    def __init__(
        self,
        node_idx,
        internal_and_sensory_cells,
        external_and_sensory_cells,
        states,
    ):

        super().__init__(
            node_idx=node_idx,
            incoming_neighbors=internal_and_sensory_cells,
            outgoing_neighbors=external_and_sensory_cells,
            states=states,
        )

        self.cell_type = "active"

        self.actions_received = {}
        self.actions_sent = {}

        self.setup(
            self.states,
            hidden_state_indices=self.incoming_neighbor_indices,
            control_state_indices=self.outgoing_neighbor_indices,
        )

        print(f"Active cell state names :{self.state_names}")
        self.build_generative_model()
