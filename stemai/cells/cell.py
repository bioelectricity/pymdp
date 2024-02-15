from pymdp.agent import Agent
from pymdp import utils
import numpy as np


class Cell(Agent):
    """A class that inherits from pymdp agent that represents an abstract cell in a network
    """

    def __init__(self, node_idx):
        """node_idx will be the index of the cell in the overall network"""

        self.node_idx = node_idx

        self.num_modalities = 1 #currently we only have one observation modality 
        self.num_factors = 1 #currently we only have one state factor

    def setup(self, states_and_actions, hidden_state_indices, control_state_indices):
        """
        Sets up the state and action names given
        the entire state and action space of the cell. 

        states_and_actions: a list of all possible states and actions in the entire network 
        hidden_state_indices: the indices of the states in the states_and_actions list that correspond to hidden states of this cell 
        control_state_indices: the indices of the states in the states_and_actions list that correspond to control states of this cell 
        """

        self.num_states = [2 ** len(hidden_state_indices)] 
        self.num_obs = [2 ** len(hidden_state_indices)]
        self.num_actions = [2 ** len(control_state_indices)]

        state_names = []

        for state in states_and_actions:  # hidden state space is the internal states of the network
            values = [int(state[index]) for index in hidden_state_indices]
            state_name = "".join(map(str, values))
            if state_name not in state_names:
                state_names.append(state_name)

        self.state_names = state_names

        assert len(self.state_names) == self.num_states[0], "Number of states does not match the number of state names"

        action_names = []

        for (
            action
        ) in (
            states_and_actions
        ):  # hidden state space is the internal states of the network
            values = [int(action[index]) for index in control_state_indices]
            action_name = "".join(map(str, values))
            if action_name not in action_names:
                action_names.append(action_name)

        self.action_names = action_names

        assert len(self.action_names) == self.num_actions[0], "Number of actions does not match the number of action names"


    def build_identity_A(self):
        """Builds an observation likelihood for each observation modality
        that is a direct identity mapping between states and observations"""
        A = utils.obj_array(self.num_modalities)

        for m in range(self.num_modalities):
            A[m] = np.eye(self.num_obs[m])

        return A
    
    def build_uniform_B(self):
        B = utils.obj_array(self.num_factors)

        for i in range(self.num_factors):  # generate a randomized (deterministic) B

            B_i = np.zeros((self.num_states[i], self.num_states[i], self.num_actions[i]))
            for action in range(self.num_actions[i]):
                B_i[:, :, action] = np.full(
                    (self.num_states[i], self.num_states[i]), 1 / self.num_states[i]
                )
            B[i] = B_i

        return B
    
    def build_fixed_random_B(self):
        B = utils.obj_array(self.num_factors)

        for i in range(self.num_factors):  # generate a randomized (deterministic) B

            B_i = np.zeros((self.num_states[i], self.num_states[i], self.num_actions[i]))
            for action in range(self.num_actions[i]):
                for state in range(self.num_states[i]):
                    random_state = np.random.choice(list(range(self.num_states[i])))
                    B_i[random_state, state, action] = 1
            B[i] = B_i
        return B

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

    def build_A(self):
        """All cells currently have identity A matrices"""
        return self.build_identity_A()

    def build_B(self, num_states, action_zero_states, action_one_states):
        """This will depend on what kind of cell this is"""
        pass

    def build_C(self):
        """Abstract method to be called in build_generative_model for constructing
        the observation preferences"""
        pass

    def build_D(self):
        """Abstract method to be called in build_generative_model for constructing
        the prior over states"""
        pass

    def build_generative_model(self):
        """Build the generative model of this cell
        and then initalize the pymdp Agent"""

        A = self.build_A()

        pB = self.build_B()
        B = utils.norm_dist_obj_arr(pB)

        C = self.build_C()

        D = self.build_D()

        super().__init__(A=A, pB=pB, B=B, C=C, D=D)

    def state_signal_to_index(self, signals: list) -> int:
        """
        Convert a list of signals from each observable neighbor
        into an index into the state space of the network
        
        signals: a list of signals from each observable neighbor
        
        returns: an index into the state space of the network"""

        state = "".join(map(str, signals))

        return self.state_names.index(state)

    def action_signal_to_index(self, signals: list) -> int:
        """
        Convert a list of signals to each outgoing actionable neighbor
        into an index into the local action space of this cell
        
        signals: a list of signals to each actionable neighbor
        
        returns: an index into the action space of the network"""

        action = "".join(map(str, signals))

        return self.action_names.index(action)
    
    def act(self, obs: int) -> str:
        """Perform state and action inference, return the action string
        which includes the action signal for each actionable neighbor
        of this cell
        
        obs: the observation signal index from the observable neighbors
        """
        self.infer_states([obs])
        self.infer_policies()
        self.action_signal = int(self.sample_action()[0])
        self.action_string = self.action_names[self.action_signal]

        return self.action_string
