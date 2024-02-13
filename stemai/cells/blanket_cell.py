from cells.cell import Cell
from pymdp import utils
import numpy as np


class BlanketCell(Cell):
    """A class that inherits from pymdp agent that represents a cell in our networks

    We include the node index, the number of neighbors, the indices of the neighbors, and the global states
    in order to create a list of local states for this particular cell given its neighbors and the global states
    """

    def __init__(
        self,
        node_idx,
        internal_neighbors,
        external_neighbors,
        internal_and_external_states,
    ):
        """
        node_idx: the index of the node in the network
        internal_neighbors: the indices of the internal cell neighbors of the blanket cell
        external_neighbors: the indices of the external cell neighbors of the blanket cell
        internal_network_states: the global states of the internal network (internal + blanket)
        all_states: the global states of the entire network (internal + blanket + external)

        """

        super().__init__(node_idx)

        self.num_internal_neighbors = len(internal_neighbors)
        self.num_external_neighbors = len(external_neighbors)
        self.internal_neighbors = internal_neighbors  # list of neighboring nodes
        self.external_neighbors = external_neighbors

        self.internal_neighbor_indices = [idx for idx, _ in enumerate(internal_neighbors)]
        self.external_neighbor_indices = [idx for idx, _ in enumerate(external_neighbors)]

        self.internal_and_external_states = internal_and_external_states

    def setup(self):
        """This will depend on whether its active or sensory"""

        pass

    def build_B(self):
        """What kind of B matrix should the active cell have?"""
        B = utils.obj_array(self.num_factors)

        for i in range(self.num_factors):  # generate a randomized (deterministic) B

            B_i = np.zeros((self.num_states[i], self.num_states[i], self.num_actions[i]))
            for action in range(self.num_actions[i]):
                B_i[:, :, action] = np.full(
                    (self.num_states[i], self.num_states[i]), 1 / self.num_states[i]
                )
            B[i] = B_i

        return B

    def build_C(self):
        return self.build_uniform_C(self.num_obs)

    def build_D(self):
        return self.build_uniform_D(self.num_states)
