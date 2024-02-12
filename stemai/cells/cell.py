

from pymdp.agent import Agent 
from pymdp import utils
import numpy as np

class Cell(Agent):

    """A class that inherits from pymdp agent that represents a cell in our networks
    
    We include the node index, the number of neighbors, the indices of the neighbors, and the global states
    in order to create a list of local states for this particular cell given its neighbors and the global states"""
    
    def __init__(self, node_idx, num_neighbors, neighbors, global_states):

        self.node_idx = node_idx 
        self.num_neighbors = num_neighbors
        self.neighbors = neighbors #list of neighboring nodes
        self.neighbor_indices = [idx for idx, _ in enumerate(neighbors)]
        self.global_states = global_states
        self.other_idx = -1

        self._action = None
        self.binary_action = None

        self.num_modalities = 1
        self.num_factors = 1

        self.actions_received = {n:0 for n in neighbors} #keep track of what you received and from who

    def setup(self):

        self.num_states = [2**(self.num_neighbors + 1)] #neighbors, and other agent 
        self.num_obs = [2**(self.num_neighbors + 1)] 
        self.num_actions = [2**(self.num_neighbors + 1)]
        

        state_names = []


        for state in self.global_states:
            other_agent = int(state[-1])
            values = [int(state[index]) for index in self.neighbor_indices] + [other_agent] 
            state_name = "".join(map(str, values))
            if state_name not in state_names:
                state_names.append(state_name)

        assert len(state_names) == 2**(self.num_neighbors + 1)

        self.state_names = state_names

    
    def build_identity_A(self, num_obs):
        A = utils.obj_array(self.num_modalities)

        for m in range(self.num_modalities):
            A[m] = np.eye(num_obs[m])

        return A
    
    def build_uniform_C(self, num_obs):

        C = utils.obj_array(self.num_modalities)
        for m in range(self.num_modalities):
            C[m] = np.zeros(num_obs[m])
        return C
    
    def build_uniform_D(self, num_states):
        D = utils.obj_array(self.num_factors)
        for f in range(self.num_factors):

            D[f] = np.random.uniform(0,1,size = num_states[f])
            D[f] /= D[0].sum()
        return D

    
    def build_A(self, num_obs):
        pass
    
    def build_B(self, num_states, action_zero_states, action_one_states):
        """This will depend on what kind of cell this is"""
        pass

    def build_C(self):
        pass 

    def build_D(self):
        pass 


    def build_generative_model(self):

        self.setup()

        A = self.build_A()
        
        pB = self.build_B()
        B = utils.norm_dist_obj_arr(pB)

        C = self.build_C()

        D = self.build_D()

        super().__init__(A=A, pB=pB, B = B, C=C, D=D)

    def signal_to_index(self, signals):
        """
        Convert a list of signals from each neighbor + self signal + other signal
        into an index into the local state space of this cell """

        state = "".join(map(str, signals))

        return self.state_names.index(state)



