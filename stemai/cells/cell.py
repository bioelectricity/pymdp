from pymdp.agent import Agent
from pymdp import utils
import numpy as np
from stemai.network_modulation.disconnecting import (
    remove_neighbor_from_pB,
    remove_neighbor_from_C,
    remove_neighbor_from_D,
)
from stemai.network_modulation.connecting import (
    add_neighbor_to_C,
    add_neighbor_to_D,
    add_neighbor_to_pB,
)


class Cell(Agent):
    """A class that inherits from pymdp agent that represents a cell in our networks

    We include the node index, the number of neighbors, the indices of the neighbors, and the global states
    in order to create a list of local states for this particular cell given its neighbors and the global states
    """

    def __init__(self, node_idx):

        self.node_idx = node_idx

        self.num_modalities = 1
        self.num_factors = 1

    def setup(self, states_and_actions, hidden_state_indices, control_state_indices):
        """
        Sets up the state and action names given
        the entire state and action space of teh cell
        as well as which of the indices in each state corresponds to
        a hidden state or a control state
        """

        self.num_states = [2 ** len(hidden_state_indices)]  # neighbors
        self.num_obs = [2 ** len(hidden_state_indices)]
        self.num_actions = [2 ** len(control_state_indices)]

        state_names = []

        for state in states_and_actions:  # hidden state space is the internal states of the network
            values = [int(state[index]) for index in hidden_state_indices]
            state_name = "".join(map(str, values))
            if state_name not in state_names:
                state_names.append(state_name)

        self.state_names = state_names

        action_names = []

        for (
            action
        ) in (
            self.internal_and_external_states
        ):  # hidden state space is the internal states of the network
            values = [int(action[index]) for index in control_state_indices]
            action_name = "".join(map(str, values))
            if action_name not in action_names:
                action_names.append(action_name)

        self.action_names = action_names

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

            D[f] = np.random.uniform(0, 1, size=num_states[f])
            D[f] /= D[0].sum()
        return D

    def build_A(self):
        """All cells currently have identity A matrices"""
        return self.build_identity_A(self.num_obs)

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

        super().__init__(A=A, pB=pB, B=B, C=C, D=D)

    def state_signal_to_index(self, signals):
        """
        Convert a list of signals from each neighbor + self signal + other signal
        into an index into the local state space of this cell"""

        state = "".join(map(str, signals))

        return self.state_names.index(state)

    def action_signal_to_index(self, signals):
        """
        Convert a list of signals from each neighbor + self signal + other signal
        into an index into the local state space of this cell"""

        action = "".join(map(str, signals))

        return self.action_names.index(action)
    
    def act(self, obs):
        self.infer_states([obs])
        self.infer_policies()
        self.action_signal = int(self.sample_action()[0])
        self.action_string = self.action_names[self.action_signal]

        return self.action_string

    def disconnect_from(self, neighbor):

        assert neighbor in self.neighbors, f"Neighbor {neighbor} not in {self.neighbors}"

        self.neighbors.remove(neighbor)
        self.neighbor_indices = [idx for idx, _ in enumerate(self.neighbors)]

        self.num_neighbors -= 1

        old_state_names = self.state_names.copy()

        self.setup()

        old_num_states, _, _ = self.B[0].shape
        new_num_states = old_num_states // 2

        # use the neighbor index to find the state, actions, (and obs) we need to marginalize over
        states_and_actions_to_marginalize = {}
        for state_idx in range(new_num_states):
            new_state = self.state_names[state_idx]
            states_and_actions_to_marginalize[state_idx] = [
                s_idx
                for s_idx, state in enumerate(old_state_names)
                if state[:neighbor] + state[neighbor + 1 :] == new_state
            ]

        self.pB = remove_neighbor_from_pB(self.pB, states_and_actions_to_marginalize)
        self.B = utils.norm_dist_obj_arr(self.pB)

        self.A = self.build_A()  # if we are learning A, we have to change this

        self.C = remove_neighbor_from_C(self.C, states_and_actions_to_marginalize)
        # self.pC = remove_neighbor_from_C(self.pC, states_and_actions_to_marginalize)

        self.D = remove_neighbor_from_D(self.D, states_and_actions_to_marginalize)
        # self.pD = remove_neighbor_from_D(self.pD, states_and_actions_to_marginalize)

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

        # get a mapping between the old states and the new states that we need to distribute
        # the values of the old states over
        states_and_actions_to_distribute = {}
        for idx, state in enumerate(old_state_names):
            states_to_distribute = [
                s_idx for s_idx, s in enumerate(self.state_names) if s[1:] == state
            ]  # there will be two states to distribute over
            states_and_actions_to_distribute[idx] = states_to_distribute

        self.pB = add_neighbor_to_pB(self.pB, states_and_actions_to_distribute)
        self.B = utils.norm_dist_obj_arr(self.pB)

        self.A = self.build_A()  # if we are learning A, we have to change this

        self.C = add_neighbor_to_C(self.C, states_and_actions_to_distribute)
        # self.pC = remove_neighbor_from_C(self.pC, states_and_actions_to_marginalize)

        self.D = add_neighbor_to_D(self.D, states_and_actions_to_distribute)
        # self.pD = remove_neighbor_from_D(self.pD, states_and_actions_to_marginalize)
