from cells.cell import Cell
from pymdp import utils
import numpy as np


class InternalCell(Cell):
    """
    Internal cells will be connected to:
    - other internal cells
    - sensory cells
    - active cells

    They will receive observations from their neighbors and the sensory cells
    And they will send actions to their neighbors and the active cells

    """

    def __init__(
        self,
        node_idx,
        internal_neighbors,
        sensory_cell_indices,
        active_cell_indices,
        states,
    ):


        super().__init__(node_idx)

        self.num_internal_neighbors = len(internal_neighbors)

        self.internal_neighbors = internal_neighbors  # list of neighboring nodes
        self.internal_neighbor_indices = [idx for idx, _ in enumerate(internal_neighbors)]
        self.sensory_cell_indices = sensory_cell_indices
        self.active_cell_indices = active_cell_indices

        self.state_neighbors = internal_neighbors + sensory_cell_indices
        self.action_neighbors = internal_neighbors + active_cell_indices

        self.states = states

        self.cell_type = "internal"

        self.actions_received = {
            n: 0 for n in self.internal_neighbors #+ sensory_cell_indices
        }  # keep track of what you received and from who

        self.actions_sent = {n: 0 for n in self.internal_neighbors + active_cell_indices}

        self.setup(
            self.states,
            hidden_state_indices=self.internal_neighbor_indices + self.sensory_cell_indices,
            control_state_indices=self.internal_neighbor_indices + self.active_cell_indices,
        )
        self.build_generative_model()

    def build_B(self):
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
    
    def act(self, obs):
        if self.qs is not None:
            self.qs_prev = self.qs
        qs = self.infer_states([obs])

        self.infer_policies()
        self.action_signal = int(self.sample_action()[0])
        self.action_string = self.action_names[self.action_signal]

        if self.qs_prev is not None:
            self.update_B(self.qs_prev)

        return self.action_string