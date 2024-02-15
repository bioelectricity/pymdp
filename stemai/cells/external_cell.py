from cells.cell import Cell
from pymdp import utils
import numpy as np


class ExternalCell(Cell):
    """A class representing an external cell
    The external cell will have a fixed random B matrix
    and it will not update B after state inference.

    The hidden state space of the external cell will be the external cell neighbors and the active cells
    and the control state space will be the external cell neighbors and the sensory cells"""

    def __init__(
        self,
        node_idx,
        neighbors,
        external_cell_indices,
        active_cell_indices,
        sensory_cell_indices,
        states,
    ):

        super().__init__(node_idx)

        self.num_neighbors = len(neighbors)

        self.neighbors = neighbors  # list of neighboring nodes
        self.neighbor_indices = external_cell_indices
        self.sensory_cell_indices = sensory_cell_indices
        self.active_cell_indices = active_cell_indices

        self.state_neighbors = neighbors + active_cell_indices
        self.action_neighbors = neighbors + sensory_cell_indices

        self.states = states

        self.cell_type = "external"

        self.actions_received = {n: 0 for n in self.neighbors + active_cell_indices}

        self.actions_sent = {n: 0 for n in self.neighbors + sensory_cell_indices}

        print(f"Setting up external cell {node_idx}")
        self.setup(
            self.states,
            hidden_state_indices=self.neighbor_indices + self.active_cell_indices,
            control_state_indices=self.neighbor_indices + self.sensory_cell_indices,
        )

        self.build_generative_model()

    def build_B(self):
        return self.build_fixed_random_B()

    def build_C(self):
        return self.build_uniform_C()

    def build_D(self):
        return self.build_uniform_D()
