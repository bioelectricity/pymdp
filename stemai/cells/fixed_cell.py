

from cells.cell import Cell
from pymdp import utils
import numpy as np

class FixedCell(Cell):

    """A class that inherits from pymdp agent that represents a cell in our networks
    
    We include the node index, the number of neighbors, the indices of the neighbors, and the global states
    in order to create a list of local states for this particular cell given its neighbors and the global states"""
    
    def __init__(self, node_idx, num_neighbors, neighbor_indices, global_states):

        super().__init__(node_idx, num_neighbors, neighbor_indices, global_states)

        self.build_generative_model()

    def build_A(self):
        return self.build_identity_A(self.num_obs)
    
    def build_B(self):
        B = utils.obj_array(self.num_factors)

        for i in range(self.num_factors): #generate a randomized (deterministic) B

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
    

