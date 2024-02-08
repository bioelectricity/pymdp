

from pymdp.agent import Agent 
from pymdp import utils
import numpy as np


from cells.cell import Cell
from pymdp import utils
import numpy as np

class StemCell(Cell):

    """A class that inherits from pymdp agent that represents a cell in our networks
    
    We include the node index, the number of neighbors, the indices of the neighbors, and the global states
    in order to create a list of local states for this particular cell given its neighbors and the global states"""
    
    def __init__(self, node_idx, num_neighbors, neighbor_indices, global_states):

        super().__init__(node_idx, num_neighbors, neighbor_indices, global_states)

        self.build_generative_model()

    def build_A(self, num_obs):
        return self.build_identity_A(num_obs)
    
    def build_B(self, num_states, action_zero_states, action_one_states):
        B = utils.obj_array(self.num_factors)
        
        for i in range(self.num_factors): #generate a randomized (deterministic) B

            B_i = np.zeros((num_states[i], num_states[i], self.num_actions[i]))
            for action in range(self.num_actions[i]):
                possible_states_for_this_action = [action_zero_states, action_one_states][action]
                B_i[possible_states_for_this_action, :, action] = np.full((len(possible_states_for_this_action), num_states[i]), 1 / len(possible_states_for_this_action))
            B[i] = B_i

        return B
    
    def build_C(self, num_obs):
        return self.build_uniform_C(num_obs)
    
    def build_D(self, num_states):
        return self.build_uniform_D(num_states)
    
