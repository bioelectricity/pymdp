from cells.blanket_cell import BlanketCell
from pymdp import utils
import numpy as np


class SensoryCell(BlanketCell):
    """Class representing a sensory cell
    which inherits from blanket cell

    The sensory cell's hidden state space will be the external neighbors
    and it's control state space will be internal neighbors

    #TODO: also need to connect sensory cells to active cells"""

    def __init__(
        self,
        node_idx,
        external_and_active_cells,
        internal_and_active_cells,
        states,
    ):
        super().__init__(
            node_idx=node_idx,
            incoming_neighbors=external_and_active_cells,
            outgoing_neighbors=internal_and_active_cells,
            states=states,
        )
        self.cell_type = "sensory"

        self.setup(
            self.states,
            hidden_state_indices=self.incoming_neighbor_indices,
            control_state_indices=self.outgoing_neighbor_indices,
        )
        self.build_generative_model()

    # def act(self, external_obs):
