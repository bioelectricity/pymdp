

from pymdp.agent import Agent 
from pymdp import utils
import numpy as np


from cells.cell import Cell
from pymdp import utils
import numpy as np
from division_and_death import remove_neighbor_from_B, remove_neighbor_from_C, remove_neighbor_from_D

class StemCell(Cell):

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
                B_i[:, :, action] = np.full((self.num_states[i], self.num_states[i]), 1 / self.num_states[i])
            B[i] = B_i

        return B
    
    def build_C(self, num_obs):
        return self.build_uniform_C(num_obs)
    
    def build_D(self, num_states):
        return self.build_uniform_D(num_states)
    

    
    def disconnect_from(self, neighbor):

        assert neighbor in self.global_neighbor_indices, f"Neighbor {neighbor} not in {self.global_neighbor_indices}"
        self.global_neighbor_indices.remove(neighbor)
        self.num_neighbors -= 1

        old_state_names = self.state_names.copy()

        self.setup() 

        num_states, _, _ = self.B[0].shape
        new_num_states = num_states // 2


        #use the neighbor index to find the state, actions, (and obs) we need to marginalize over
        states_and_actions_to_marginalize = {}
        for state_idx in range(new_num_states):
            new_state = self.state_names[state_idx]
            states_and_actions_to_marginalize[state_idx] = [s_idx for s_idx, state in enumerate(old_state_names) if state[:neighbor] + state[neighbor+1:]== new_state]


        self.B = remove_neighbor_from_B(self.B, states_and_actions_to_marginalize)
        self.pB = remove_neighbor_from_B(self.pB,  states_and_actions_to_marginalize)

        self.A = self.build_A() #if we are learning A, we have to change this
        self.pA = self.build_A()

        self.C = remove_neighbor_from_C(self.C, states_and_actions_to_marginalize)
        self.pC = remove_neighbor_from_C(self.pC, states_and_actions_to_marginalize)

        self.D = remove_neighbor_from_D(self.D, states_and_actions_to_marginalize)
        self.pD = remove_neighbor_from_D(self.pD, states_and_actions_to_marginalize)

        


        




