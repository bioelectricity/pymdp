

from pymdp.agent import Agent 
from pymdp import utils
import numpy as np

class Cell(Agent):

    """A class that inherits from pymdp agent that represents a cell in our networks
    
    We include the node index, the number of neighbors, the indices of the neighbors, and the global states
    in order to create a list of local states for this particular cell given its neighbors and the global states"""
    
    def __init__(self, node_idx, num_neighbors, neighbor_indices, global_states):

        self.node_idx = node_idx 
        self.num_neighbors = num_neighbors
        self.global_neighbor_indices = neighbor_indices
        self.global_states = global_states
        self.self_idx = -2
        self.other_idx = -1

        self._action = None
        self.binary_action = None

        self.num_modalities = 1
        self.num_factors = 1
        self.num_actions = [2]

    def setup(self):
        action_zero_states_global = [idx for idx, state in enumerate(self.global_states) if state[self.node_idx] == "0"]
        action_one_states_global = [idx for idx, state in enumerate(self.global_states) if state[self.node_idx] == "1"]

        action_zero_states = []
        action_one_states = []
        num_states = [2**(self.num_neighbors + 2)] #neighbors, self, and other agent 
        num_obs = [2**(self.num_neighbors + 2)] 

        state_names = []

        local_idx = 0

        for global_idx, state in enumerate(self.global_states):
            other_agent = int(state[-1])
            self_state = int(state[self.node_idx])
            values = [int(state[index]) for index in self.global_neighbor_indices] + [self_state, other_agent] 
            state_name = "".join(map(str, values))
            if state_name not in state_names:
                state_names.append(state_name)
                if global_idx in action_zero_states_global:
                    action_zero_states.append(local_idx)
                elif global_idx in action_one_states_global:
                    action_one_states.append(local_idx)
                else:
                    raise Exception("Every state should either be globally 0 or 1 for this agent")
                local_idx += 1

        assert len(state_names) == 2**(self.num_neighbors + 2)

        self.state_names = state_names

        return num_states, num_obs, action_zero_states, action_one_states
    
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

        num_states, num_obs, action_zero_states, action_one_states = self.setup()

        print(f"Num states: {num_states}")
        A = self.build_A(num_obs)
        
        B = self.build_B(num_states, action_zero_states, action_one_states)

        C = self.build_C(num_obs)

        D = self.build_D(num_states)

        super().__init__(A=A, B=B, C=C, D=D)

    def signal_to_index(self, signals):
        """
        Convert a list of signals from each neighbor + self signal + other signal
        into an index into the local state space of this cell """

        state = "".join(map(str, signals))

        return self.state_names.index(state)


