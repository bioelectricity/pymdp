from pymdp.agent import Agent
from pymdp import utils
from pymdp import maths
import numpy as np
import tqdm 

#with and without sliding window 

class NeuronalCell(Agent):
    """A class that inherits from pymdp agent that represents an abstract cell in a network"""

    def __init__(self, node_idx, neighbors, gamma_A_scalar = 1.0, gamma_B_scalar = 10.0):
        """node_idx will be the index of the cell in the overall network"""

        self.node_idx = node_idx

        self.num_factors = 1  # currently we only have one state factor
        self.neighbors = neighbors
        self.num_neighbors = len(neighbors)
        self.num_modalities = len(neighbors)  # currently we only have one observation modality

        gamma_A = utils.obj_array(self.num_modalities)
        gamma_B = utils.obj_array(self.num_factors)

        for m in range(self.num_modalities):
            # if m == self.num_modalities - 1:
            #     gamma_A[m] = np.array([0.5,0.5])
            # else:
            gamma_A[m] = gamma_A_scalar
            #gamma_A[m] = np.array([gamma_A_scalar, gamma_A_scalar])
       
        for f in range(self.num_factors):
            gamma_B[f] = np.array([gamma_B_scalar + np.random.uniform(0,0.1), gamma_B_scalar + np.random.uniform(0,0.1)])
        self.gamma_A =gamma_A
        self.gamma_B = gamma_B
        self.observation_history = []
        self.actions_received = {}
        self.actions_sent = {}

        print(f"Gamma A: {self.gamma_A}")
        print(f"Gamma B: {self.gamma_B}")



        self.setup(self.num_neighbors)

        super().__init__(A = self.A, B = self.B, pA = self.A, beta_zeta_prior = self.gamma_A, beta_omega_prior = self.gamma_B)

    def setup(self, num_neighbors):

        self.num_states = [2]
        self.num_obs = [2]*num_neighbors
        self.num_actions = 1
        #actions: take the posterior distribution and a

        self.build_A()
        self.build_B()

    
    def build_A(self):

        A = utils.obj_array(self.num_modalities)

        for neighbor in range(self.num_modalities):
            A[neighbor] = np.eye(self.num_states[0])
            
        self.A = A

        assert utils.is_normalized(self.A), "A matrix is not normalized (i.e. A[m].sum(axis = 0) must all equal 1.0 for all modalities)"
    
    def build_B(self):
        B = utils.obj_array(self.num_factors)

        for f in range(self.num_factors):
            B[f] = np.eye(self.num_states[0]).reshape((self.num_states[0], self.num_states[0], 1))
        self.B = B 

  
    def disconnect_from(self, neighbor_node):
        self.num_neighbors -= 1
        self.num_modalities -= 1

        print(f"Neighbor node: {neighbor_node}")
        print(f"Neighbors: {self.neighbors}")

        
        neighbor_idx = list(self.neighbors).index(neighbor_node)

        print(f"Neighbor idx: {neighbor_idx}")

        self.num_obs.remove(self.num_obs[neighbor_idx])
        old_A = np.copy(self.A)
        old_base_A = np.copy(self.base_A)
        old_beta_zeta_prior = np.copy(self.beta_zeta_prior)

        print(f"Old beta zeta: {len(old_beta_zeta_prior)}")
        self.build_B()
        mapping = {}
        neighbor_idx = list(self.neighbors).index(neighbor_node)
        for o in range(self.num_neighbors + 1):
            if o == neighbor_idx:
                continue
            elif o < neighbor_idx:
                mapping[o] = o 
            else:
                mapping[o] = o - 1
        new_A = utils.obj_array(self.num_modalities)
        new_base_A = utils.obj_array(self.num_modalities)
        new_beta_zeta_prior = utils.obj_array(self.num_modalities)
        for old_m in range(self.num_neighbors+1):
            if old_m not in mapping:
                continue
            new_m = mapping[old_m]
            new_base_A[new_m] = old_base_A[old_m]
            
            new_A[new_m] = old_A[old_m]
            new_beta_zeta_prior[new_m] = old_beta_zeta_prior[old_m]
        self.A = new_A
        self.base_A = new_base_A
        self.beta_zeta_prior = new_beta_zeta_prior
        self.beta_zeta = new_beta_zeta_prior
        self.neighbors.remove(self.neighbors[neighbor_idx])
        print(f"New beta zeta: {len(self.beta_zeta_prior)}")

    def act(self, obs):
        """
        For a neuronal cell, the observation is a 0 or 1 signal
        for each neighbor, and then the agent performs state inference
        and the action it performs is sampled directly from the posterior over states"""

        self.observation_history.append(obs)

        qs = self.infer_states(obs)
        action = utils.sample(qs[0])

        print(f"Qs : {qs}")
        #self.update_zeta(obs)



        return action
    

    def update_after_trial(self):
        # update gamma_A
        for t in range(len(self.observation_history)):
            self.update_zeta(self.observation_history[t])
           # self.update_A(self.observation_history[t])
        
        #overwrite the sensory ones
        self.observation_history = []