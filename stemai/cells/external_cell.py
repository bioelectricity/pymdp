from cells.cell import Cell
from pymdp import utils
import numpy as np


class ExternalCell(Cell):


    def __init__(self, node_idx, neighbors, sensory_cell_indices, active_cell_indices, external_and_blanket_states):

        super().__init__(node_idx)

        self.num_neighbors = len(neighbors)

        self.neighbors = neighbors  # list of neighboring nodes
        self.neighbor_indices = [idx for idx, _ in enumerate(neighbors)]
        self.sensory_cell_indices = sensory_cell_indices
        self.active_cell_indices = active_cell_indices

        self.state_neighbors = neighbors + active_cell_indices
        self.action_neighbors = neighbors + sensory_cell_indices

        self.external_and_blanket_states = external_and_blanket_states

        self.cell_type = "external"

        self.actions_received = {
            n: 0 for n in self.neighbors + active_cell_indices
        }

        self.actions_sent = {n: 0 for n in self.neighbors + sensory_cell_indices}


        self.setup(self.external_and_blanket_states, hidden_state_indices=self.neighbor_indices + self.active_cell_indices, control_state_indices=self.neighbor_indices + self.sensory_cell_indices)
        
        self.build_generative_model()

      
    def build_B(self):
        B = utils.obj_array(self.num_factors)

        for i in range(self.num_factors):  # generate a randomized (deterministic) B

            B_i = np.zeros((self.num_states[i], self.num_states[i], self.num_actions[i]))
            for action in range(self.num_actions[i]):
                for state in range(self.num_states[i]):
                    random_state = np.random.choice(list(range(self.num_states[i])))
                    B_i[random_state, state, action] = 1
            B[i] = B_i

        return B

    def build_C(self):
        return self.build_uniform_C(self.num_obs)

    def build_D(self):
        return self.build_uniform_D(self.num_states)

    def disconnect_from(self, neighbor):
        raise Exception("Cannot change connections of an external cell")

    def connect_to(self, neighbor):
        raise Exception("Cannot change connections of an external cell")
