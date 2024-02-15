from cells.cell import Cell
from pymdp import utils
import numpy as np

# should blanket cells be learning B?


class BlanketCell(Cell):
    """A class that represents a Blanket cell, from which
    sensory and active cells will inherit

    Blanket cells will have uniform B matrices which are learned over time
    so we will overwrite the build_B method to create a uniform B matrix
    and we will overwrite the act method to update the B matrix after every state inference
    """

    def __init__(
        self,
        node_idx,
        incoming_neighbors,
        outgoing_neighbors,
        states,
    ):
        """
        node_idx: the index of the node in the network
        incoming_neighbors: the indices of the cells that send incoming signals to the blanket cell
        outgoing_neighbors: the indices of the cells that receive outgoing signals from the blanket cell
        states: the global states of the entire network (internal + blanket + external)

        """

        super().__init__(node_idx)

        self.num_incoming_neighbors = len(incoming_neighbors)
        self.num_outgoing_neighbors = len(outgoing_neighbors)
        self.incoming_neighbors = incoming_neighbors  # list of neighboring nodes
        self.outgoing_neighbors = outgoing_neighbors

        self.incoming_neighbor_indices = [idx for idx, _ in enumerate(incoming_neighbors)]
        self.outgoing_neighbor_indices = [idx for idx, _ in enumerate(outgoing_neighbors)]

        self.states = states

    def build_B(self):
        """Blanket cells will have uniform B matrices which will be learned over time"""
        return self.build_uniform_B()

    def build_C(self):
        return self.build_uniform_C()

    def build_D(self):
        return self.build_uniform_D()

    def act(self, obs: int) -> str:
        """Here we overwrite the abstract act() class
        for blanket cells, because blanket cells
        will update their transition likelihoods after every state inference"""

        if self.qs is not None:
            self.qs_prev = self.qs
        self.infer_states([obs])

        self.infer_policies()
        self.action_signal = int(self.sample_action()[0])
        self.action_string = self.action_names[self.action_signal]

        # update B
        if self.qs_prev is not None:
            self.update_B(self.qs_prev)

        return self.action_string
