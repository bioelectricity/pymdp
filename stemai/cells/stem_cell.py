

from pymdp import utils
import numpy as np


from cells.cell import Cell
from pymdp import utils
import numpy as np
from stemai.network_modulation.disconnecting import remove_neighbor_from_pB, remove_neighbor_from_C, remove_neighbor_from_D
from stemai.network_modulation.connecting import add_neighbor_to_C, add_neighbor_to_D, add_neighbor_to_pB
class StemCell(Cell):

    """A class that inherits from pymdp agent that represents a cell in our networks
    
    We include the node index, the number of neighbors, the indices of the neighbors, and the global states
    in order to create a list of local states for this particular cell given its neighbors and the global states"""
    
    def __init__(self, node_idx, num_neighbors, neighbors, global_states, is_blanket_node = False, env_node_indices = None):

        super().__init__(node_idx, num_neighbors, neighbors, global_states, is_blanket_node, env_node_indices)

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
    
    def build_C(self):
        return self.build_uniform_C(self.num_obs)
    
    def build_D(self):
        return self.build_uniform_D(self.num_states)
    

    
    def disconnect_from(self, neighbor):

        assert neighbor in self.neighbors, f"Neighbor {neighbor} not in {self.neighbors}"

        self.neighbors.remove(neighbor)
        self.neighbor_indices = [idx for idx, _ in enumerate(self.neighbors)]

        self.num_neighbors -= 1

        old_state_names = self.state_names.copy()

        self.setup() 

        old_num_states, _, _ = self.B[0].shape
        new_num_states = old_num_states // 2


        #use the neighbor index to find the state, actions, (and obs) we need to marginalize over
        states_and_actions_to_marginalize = {}
        for state_idx in range(new_num_states):
            new_state = self.state_names[state_idx]
            states_and_actions_to_marginalize[state_idx] = [s_idx for s_idx, state in enumerate(old_state_names) if state[:neighbor] + state[neighbor+1:]== new_state]

        self.pB = remove_neighbor_from_pB(self.pB,  states_and_actions_to_marginalize)
        self.B = utils.norm_dist_obj_arr(self.pB)

        self.A = self.build_A() #if we are learning A, we have to change this

        self.C = remove_neighbor_from_C(self.C, states_and_actions_to_marginalize)
        #self.pC = remove_neighbor_from_C(self.pC, states_and_actions_to_marginalize)

        self.D = remove_neighbor_from_D(self.D, states_and_actions_to_marginalize)
        #self.pD = remove_neighbor_from_D(self.pD, states_and_actions_to_marginalize)

    def connect_to(self, neighbor):
        """Add a new connection to the given neighbor
        
        Currently the new neighbor becomes the first neighbor in the list of neighbors
        in order to preserve indexing. 
        The probabilities in the B, C, D matrices with respect to states 
        of the new neighbor will be distributed equally from the old state probabilities"""
        
        assert neighbor not in self.neighbors, "Neighbor already in neighbors"

        self.neighbors.insert(0, neighbor)
        self.num_neighbors += 1
        self.neighbor_indices = [idx for idx, _ in enumerate(self.neighbors)]

        old_state_names = self.state_names.copy()

        self.setup() 


        #get a mapping between the old states and the new states that we need to distribute
        #the values of the old states over 
        states_and_actions_to_distribute = {}
        for idx, state in enumerate(old_state_names):
            states_to_distribute = [s_idx for s_idx, s in enumerate(self.state_names) if s[1:] == state] #there will be two states to distribute over 
            states_and_actions_to_distribute[idx] = states_to_distribute

        self.pB = add_neighbor_to_pB(self.pB,  states_and_actions_to_distribute)
        self.B = utils.norm_dist_obj_arr(self.pB)


        self.A = self.build_A() #if we are learning A, we have to change this

        self.C = add_neighbor_to_C(self.C, states_and_actions_to_distribute)
        #self.pC = remove_neighbor_from_C(self.pC, states_and_actions_to_marginalize)

        self.D = add_neighbor_to_D(self.D, states_and_actions_to_distribute)
        #self.pD = remove_neighbor_from_D(self.pD, states_and_actions_to_marginalize)




        




