from pymdp.agent import Agent
from pymdp import utils
from pymdp import maths
import numpy as np
import tqdm

# with and without sliding window


class NeuronalCell(Agent):
    """A class that inherits from pymdp agent that represents an abstract cell in a network"""

    def __init__(self, node_idx, neighbors, gamma_A, gamma_B_scalar=0.01, alpha = 16):
        """node_idx will be the index of the cell in the overall network"""

        self.node_idx = node_idx

        self.num_factors = 1  # currently we only have one state factor
        self.neighbors = neighbors
        self.num_neighbors = len(neighbors)
        self.num_modalities = len(neighbors)  # currently we only have one observation modality

        gamma_B = utils.obj_array(self.num_factors)

        for f in range(self.num_factors):
            gamma_B[f] = np.array(
                [
                    gamma_B_scalar + np.random.uniform(0, 0.1),
                    gamma_B_scalar + np.random.uniform(0, 0.1),
                ]
            )
        self.gamma_A = gamma_A
        self.gamma_B = gamma_B
        self.observation_history = []
        self.qs_over_time = []
        self.actions_received = {}
        self.actions_sent = {}
        self.num_obs = [2] * self.num_modalities
        self.num_states = [2]
        self.logging = False

        if self.logging: print(f"Gamma A: {self.gamma_A}")
        if self.logging: print(f"Gamma B: {self.gamma_B}")

        self.setup(self.num_neighbors)

        print(f"Initial gamma A: {self.gamma_A}")
        

        super().__init__(A = self.A, B = self.B, C = self.C, D = self.D, pA = self.A, gamma_A_prior = self.gamma_A, gamma_B_prior = self.gamma_B, alpha = alpha)
    def setup(self, num_neighbors):

        self.num_states = [2]
        self.num_obs = [2] * num_neighbors
        self.num_actions = 1
        # actions: take the posterior distribution and a

        self.build_A()
        self.build_B()
        self.D = self.build_uniform_D()
        
        self.C = self.build_uniform_C()

    def build_A(self):

        A = utils.obj_array(self.num_modalities)

        for neighbor in range(self.num_modalities):

            A[neighbor] = np.eye(self.num_states[0])

        self.A = A

        assert utils.is_normalized(
            self.A
        ), "A matrix is not normalized (i.e. A[m].sum(axis = 0) must all equal 1.0 for all modalities)"

    def build_B(self):
        B = utils.obj_array(self.num_factors)

        for f in range(self.num_factors):
            B[f] = np.eye(self.num_states[0]).reshape((self.num_states[0], self.num_states[0], 1))
        self.B = B

    def build_uniform_C(self):
        """Construts a uniform C vector, meaning the cell has
        no preference for any particular observation."""
        C = utils.obj_array(self.num_modalities)
        for m in range(self.num_modalities):
            C[m] = np.zeros(self.num_obs[m])
        return C

    def build_uniform_D(self):
        """Constructs a uniform state prior"""
        D = utils.obj_array(self.num_factors)
        for f in range(self.num_factors):

            D[f] = np.random.uniform(0, 1, size=self.num_states[f])
            D[f] /= D[0].sum()
        return D

    def rebuild_A_factor_list(self):
        self.A_factor_list = self.num_modalities * [list(range(self.num_factors))] # defaults to having all modalities depend on all factors

    def rebuild_A_factor_list(self):
        self.A_factor_list = self.num_modalities * [list(range(self.num_factors))] # defaults to having all modalities depend on all factors

    def disconnect_from(self, neighbor_node):

        self.num_neighbors -= 1
        self.num_modalities -= 1

        if self.logging: print(f"Neighbor node: {neighbor_node}")
        if self.logging: print(f"Neighbors: {self.neighbors}")

        neighbor_idx = list(self.neighbors).index(neighbor_node)

        if self.logging: print(f"Neighbor idx: {neighbor_idx}")

        self.num_obs.remove(self.num_obs[neighbor_idx])


        if self.num_modalities == 0:
            self.gamma_A = []
            self.gamma_A_prior = []
            self.A = utils.obj_array(0)
            self.neighbors.remove(neighbor_node)
            return
        old_base_A = np.copy(self.base_A)
        old_gamma_A_prior  = np.copy(self.gamma_A_prior)
        old_gamma_A = np.copy(self.gamma_A)

        #self.build_B()
        mapping = {} #mapping from old modality to new modality 
        neighbor_idx = list(self.neighbors).index(neighbor_node)
        for o in range(self.num_neighbors + 1):
            if o == neighbor_idx:
                continue
            elif o < neighbor_idx:
                mapping[o] = o
            else:
                mapping[o] = o - 1
        new_base_A = utils.obj_array(self.num_modalities)
        new_gamma_A = utils.obj_array(self.num_modalities)
        new_gamma_A_prior = utils.obj_array(self.num_modalities)
        for old_m in range(self.num_neighbors + 1):
            if old_m not in mapping:
                continue
            new_m = mapping[old_m]
            new_base_A[new_m] = old_base_A[old_m]
            new_gamma_A[new_m] = old_gamma_A[old_m]

            new_gamma_A_prior[new_m] = old_gamma_A_prior[old_m]
        self.base_A = new_base_A
        self.gamma_A_prior = new_gamma_A_prior
        self.gamma_A = new_gamma_A
        self.neighbors.remove(neighbor_node)
    
        self.A = utils.scale_A_with_gamma(self.base_A, self.gamma_A)

        self.actions_received.pop(neighbor_node)
        self.rebuild_A_factor_list()
        self.qs_over_time = []
        self.observation_history = []

    def reset_cell(self):
        self.curr_timestep = 0
        self.qs = self.D 
        self.qs_prev = None
        self.action = None
    
    def check_connect_to(self, neighbor_node):
        if neighbor_node in self.neighbors:
            return False
        return True

    def check_disconnect_from(self, neighbor_node):
        if neighbor_node not in self.neighbors:            
            return False
        return True

        
    def connect_to(self, neighbor_node):
        self.num_neighbors += 1
        self.num_modalities += 1
        self.num_obs.append(2)
        old_base_A = np.copy(self.base_A)
        old_gamma_A = np.copy(self.gamma_A)
        old_gamma_A_prior = np.copy(self.gamma_A_prior)

        #self.build_B()

        new_base_A = utils.obj_array(self.num_modalities)
        new_gamma_A = utils.obj_array(self.num_modalities)
        new_gamma_A_prior = utils.obj_array(self.num_modalities)
        
        for m in range(1, self.num_modalities):
            new_base_A[m] = old_base_A[m-1]
            new_gamma_A[m] = old_gamma_A[m-1]
            new_gamma_A_prior[m] = old_gamma_A_prior[m-1]
        new_base_A[0] = np.eye(self.num_states[0])
        new_gamma_A[0] = 0.1
        new_gamma_A_prior[0] = 0.1
        self.base_A = new_base_A
        self.gamma_A_prior = new_gamma_A_prior
        self.gamma_A = new_gamma_A
        self.A = utils.scale_A_with_gamma(self.base_A, self.gamma_A)
        # print(f"Neighbors: {self.neighbors}")
        self.neighbors = [neighbor_node] + self.neighbors
        # print(f"New neighbors: {self.neighbors}")
        
        self.rebuild_A_factor_list()
 

        self.qs_over_time = []
        self.observation_history = []
        self.actions_received[neighbor_node] = np.random.choice([0, 1])


    def act(self, obs, distance_to_reward=None):
        """
        For a neuronal cell, the observation is a 0 or 1 signal
        for each neighbor, and then the agent performs state inference
        and the action it performs is sampled directly from the posterior over states"""

        self.observation_history.append(obs)
        qs = self.infer_states(obs)
        # self.D = self.qs
        
        self.qs_over_time.append(qs)

        action = utils.sample(maths.softmax(3.0 * qs[0]))

        self.neuronal_action = action

        return action

    def update_after_trial(self, modalities_to_omit):
        # update gamma_A
        for t in range(len(self.observation_history)):
            if self.cell_type == "internal":
                modalities = list(range(self.num_modalities - modalities_to_omit))
                self.update_gamma_A(
                    self.observation_history[t], self.qs_over_time[t], modalities=modalities
                )
            else:
                self.update_gamma_A(self.observation_history[t], self.qs_over_time[t])
            # self.update_A(self.observation_history[t])

        # overwrite the sensory ones
        self.observation_history = []
        self.qs_over_time = []

    def build_uniform_C(self):
        """Construts a uniform C vector, meaning the cell has
        no preference for any particular observation."""
        C = utils.obj_array(self.num_modalities)
        for m in range(self.num_modalities):
            C[m] = np.zeros(self.num_obs[m])
        return C

    def build_uniform_D(self):
        """Constructs a uniform state prior"""
        D = utils.obj_array(self.num_factors)
        for f in range(self.num_factors):

            D[f] = np.random.uniform(0, 1, size=self.num_states[f])
            D[f] /= D[0].sum()
        return D
