from pymdp.agent import Agent
from pymdp import utils
from pymdp import maths
import numpy as np
import tqdm

# with and without sliding window


class NeuronalCell(Agent):
    """A class that inherits from pymdp agent that represents an abstract cell in a network"""

    def __init__(self, node_idx, neighbors, gamma_A, gamma_B_scalar=0.01):
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

        if self.logging: print(f"Gamma A: {self.gamma_A}")
        if self.logging: print(f"Gamma B: {self.gamma_B}")

        C = self.build_uniform_C()
        D = self.build_uniform_D()

        self.setup(self.num_neighbors)

        super().__init__(
            A=self.A,
            B=self.B,
            pA=self.A,
            A_factor_list=self.A_factor_list,
            C=C,
            D=D,
            beta_zeta_prior=self.gamma_A,
            beta_omega_prior=self.gamma_B,
        )

    def setup(self, num_neighbors):

        self.num_states = [2, 10] #binary state of the world, battery state
        self.num_obs = [2] * num_neighbors + [10] #binary observation of the world, battery level
        self.num_actions_for_state = 2 #fire / don't fire
        self.num_actions_for_battery = 2 #increase / decrease
        # actions: take the posterior distribution and a

        self.build_A()
        self.build_B()

    def build_A(self):

        A = utils.obj_array(self.num_modalities)

        for neighbor in range(self.num_modalities -1):
            A[neighbor] = np.eye(self.num_states[0])
        A[-1] = np.eye(self.num_states[1])

        self.A_factor_list = [0]*(self.num_modalities-1) + [1]

        self.A = A

        assert utils.is_normalized(
            self.A
        ), "A matrix is not normalized (i.e. A[m].sum(axis = 0) must all equal 1.0 for all modalities)"

    def build_B(self):
        B = utils.obj_array(self.num_factors)


        #uniform B matrix for the world state
        B[0] = np.zeros(self.num_states[0], self.num_states[0], self.num_actions_for_state)
        B[0][:, :, 0] = np.full(self.num_states[0], self.num_states[0], 1 / self.num_states[0])
        B[0][:, :, 1] = np.full(self.num_states[0], self.num_states[0], 1 / self.num_states[0])

        #uniform B matrix for the battery state
        B[1] = np.zeros(self.num_states[1], self.num_states[1], self.num_actions_for_battery)
        B[1][:, :, 0] = np.full(self.num_states[1], self.num_states[1], 1 / self.num_states[1])
        B[1][:, :, 1] = np.full(self.num_states[1], self.num_states[1], 1 / self.num_states[1])

        self.B = B

    def disconnect_from(self, neighbor_node):
        self.num_neighbors -= 1
        self.num_modalities -= 1

        if self.logging: print(f"Neighbor node: {neighbor_node}")
        if self.logging: print(f"Neighbors: {self.neighbors}")

        neighbor_idx = list(self.neighbors).index(neighbor_node)

        if self.logging: print(f"Neighbor idx: {neighbor_idx}")

        self.num_obs.remove(self.num_obs[neighbor_idx])
        old_base_A = np.copy(self.base_A)
        old_beta_zeta = np.copy(self.beta_zeta)
        old_beta_zeta_prior = np.copy(self.beta_zeta_prior)

        if self.logging: print(f"Old beta zeta: {len(old_beta_zeta_prior)}")
        self.build_B()
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
        new_beta_zeta = utils.obj_array(self.num_modalities)
        new_beta_zeta_prior = utils.obj_array(self.num_modalities)
        for old_m in range(self.num_neighbors + 1):
            if old_m not in mapping:
                continue
            new_m = mapping[old_m]
            new_base_A[new_m] = old_base_A[old_m]
            new_beta_zeta[new_m] = old_beta_zeta[old_m]

            new_beta_zeta_prior[new_m] = old_beta_zeta_prior[old_m]
        
        self.base_A = new_base_A        
        self.beta_zeta_prior = new_beta_zeta_prior
        self.beta_zeta = new_beta_zeta
        self.A = utils.scale_A_with_zeta(self.A, self.beta_zeta)
        self.neighbors.remove(self.neighbors[neighbor_idx])
        if self.logging: print(f"New beta zeta: {len(self.beta_zeta_prior)}")

    def connect_to(self, neighbor_node):
        self.num_neighbors += 1
        self.num_modalities += 1
        self.num_obs.append(2)
        old_base_A = np.copy(self.base_A)
        old_beta_zeta = np.copy(self.beta_zeta)
        old_beta_zeta_prior = np.copy(self.beta_zeta_prior)
        self.build_B()

        new_base_A = utils.obj_array(self.num_modalities)
        new_beta_zeta = utils.obj_array(self.num_modalities)
        new_beta_zeta_prior = utils.obj_array(self.num_modalities)
        for m in range(1, self.num_modalities):
            new_base_A[m] = old_base_A[m]
            new_beta_zeta[m] = old_beta_zeta[m]
            new_beta_zeta_prior[m] = old_beta_zeta_prior[m]
        new_base_A[0] = np.eye(self.num_states[0])
        new_beta_zeta[0] = 0.1
        new_beta_zeta_prior[0] = 0.1
        self.base_A = new_base_A
        self.beta_zeta_prior = new_beta_zeta_prior
        self.beta_zeta = new_beta_zeta
        self.A = utils.scale_A_with_zeta(self.base_A, self.beta_zeta)
        self.neighbors.append(neighbor_node)


    def act(self, obs, distance_to_reward=None):
        """
        For a neuronal cell, the observation is a 0 or 1 signal
        for each neighbor, and then the agent performs state inference
        and the action it performs is sampled directly from the posterior over states"""

        self.observation_history.append(obs)

        qs = self.infer_states(obs)
        # self.D = self.qs
        self.infer_policies()
        action = self.sample_action()
        battery_action = action[1]
        if battery_action == 0: #decrease policy length
           self.policy_length = min(1, self.policy_length - 1)
        elif battery_action == 1: #increase policy length
            self.policy_length = min(10, self.policy_length + 1)

        self.neuronal_action = action[0]

        self.update_B(self.qs_over_time[-1])
        self.qs_over_time.append(qs)

        return action

    def update_after_trial(self, modalities_to_omit):
        # update gamma_A
        for t in range(len(self.observation_history)):
            if self.cell_type == "internal":
                modalities = list(range(self.num_modalities - modalities_to_omit))
                self.update_zeta(
                    self.observation_history[t], self.qs_over_time[t], modalities=modalities
                )
            else:
                self.update_zeta(self.observation_history[t], self.qs_over_time[t])
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
