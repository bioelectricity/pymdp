from stemai.cells.cell import Cell
from pymdp import utils
from pymdp import learning
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


class InternalCell(Cell):
    """
    A class for internal cells in the network
    In this class, we form connections between the internal cells and its internal neighbors

    These cells will eventually also be connected to active and sensory cells, but this will happen
    from within a System object, not within the cells themselves

    Internal cells have the ability to form new connections and disconnections with other internal neighbors

    Upon action, internal cells will receive observations from their internal neighbors and the sensory cells
    And they will send actions to their internal neighbors and the active cells
    """

    def __init__(
        self,
        node_idx,
        internal_neighbors,
        internal_neighbor_indices,
        sensory_cells,
        active_cells,
        states,
    ):
        """
        node_idx: the index of the node in the Network
        internal_neighbors: the internal neighbors of the cell in the network
        sensory_cell_indices: the indices of the sensory cells in the network
        active_cell_indices: the indices of the active cells in the network
        states: the global states of the network"""

        super().__init__(node_idx)

        self.num_internal_neighbors = len(internal_neighbors)

        self.internal_neighbors = internal_neighbors  # list of neighboring nodes
        self.internal_neighbor_indices = internal_neighbor_indices
        self.sensory_cell_indices = sensory_cells
        self.active_cell_indices = active_cells

        self.state_neighbors = internal_neighbors + sensory_cells
        self.action_neighbors = internal_neighbors + active_cells

        self.states = states

        self.cell_type = "internal"

        self.qs_over_time = []
        self.actions_over_time = []


        # initialize the actions received from other internal neighbors
        self.actions_received = {
            n: 0 for n in self.internal_neighbors
        }  # keep track of what you received and from who

        # initialize the actions sent to other internal neighbors
        self.actions_sent = {n: 0 for n in self.internal_neighbors}

        # set up the state space of this cell, where hidden states correspond to
        # internal cell neighbors and sensory cells, and control states correspond to
        # internal cells and active cells

        self.setup(
            self.states,
            hidden_state_indices=self.internal_neighbor_indices + self.sensory_cell_indices,
            control_state_indices=self.internal_neighbor_indices + self.active_cell_indices,
        )

        # build the generative model of the cell
        self.build_generative_model()

    def _reset(self):
        self.qs_over_time = []
        self.actions_over_time = []
        self.curr_timestep = 0

    def build_B(self) -> np.ndarray:
        """Internal cells will have uniform transition likelihoods
        meaning they are initialized without any information about how the actions they perform
        (i.e. the signals they send to their internal neighbors and active cells) will influence
        the signals they receive from their internal neighbors and the sensory cells"""
        return self.build_uniform_B()

    def build_C(self) -> np.ndarray:
        """Internal cells have uniform preferences over observations"""
        return self.build_uniform_C()

    def build_D(self) -> np.ndarray:
        """Internal cells have uniform priors over states"""
        return self.build_uniform_D()

    def act(self, obs: int, update=True, accumulate = True) -> str:
        """Here we overwrite the abstract act() class
        for internal cells, because internal cells
        will update their transition likelihoods after every state inference"""

        if self.qs is not None:
            self.qs_prev = self.qs
            if accumulate:
                self.qs_over_time.append(self.qs)

        #the first entry in self.qs_over_time will be the second state inferred 
        #which is the qs_previous for the first action 
        self.infer_states([obs])

        self.infer_policies()
        self.action_signal = int(self.sample_action()[0])
        if accumulate:
            self.actions_over_time.append(self.action)

        #the first entry in self.actions_over_time will be the first action inferred
        #when qs_prev is None and qs is not None
        self.action_string = self.action_names[self.action_signal]

        # # update B
        if update:
            if self.qs_prev is not None:
                self.update_B(self.qs_prev)

        return self.action_string
    
    def update_B_after_trial(self):
        # update B
        for t in range(len(self.qs_over_time) - 1):
            qB = learning.update_state_likelihood_dirichlet_interactions(
                self.pB,
                self.B,
                self.actions_over_time[t+1],
                self.qs_over_time[t+1],
                self.qs_over_time[t],
                self.B_factor_list,
                self.lr_pB,
                self.factors_to_learn
            )

            self.pB = qB # set new prior to posterior
            self.B = utils.norm_dist_obj_arr(qB)  # take expected value of posterior Dirichlet parameters to calculate posterior over B array

    def disconnect_from(self, neighbor):
        """Disconnect this cell from the given neighbor

        Currently this neighbor must be an internal neighbor cell, not an active or sensory cell

        Disconnection from active and sensory cells must occur in the System, not within the internal cells

        This removes the connection from the network and then updates
        the generative model of this cell to reflect the new state and action space"""

        assert (
            neighbor in self.internal_neighbors
        ), f"Trying to remove an internal neighbor: {neighbor}, that is not in cell neighborhood: {self.internal_neighbors}"

        neighbor_idx = self.internal_neighbors.index(neighbor).copy()
        self.internal_neighbors.remove(neighbor)

        self.internal_neighbor_indices.remove(self.internal_neighbor_indices[neighbor_idx])

        self.num_internal_neighbors -= 1

        old_state_names = self.state_names.copy()

        self.setup(
            self.states,
            hidden_state_indices=self.internal_neighbor_indices + self.sensory_cell_indices,
            control_state_indices=self.internal_neighbor_indices + self.active_cell_indices,
        )
        old_num_states, _, _ = self.B[0].shape

        new_num_states = old_num_states // 2

        # use the neighbor index to find the state, actions, (and obs) we need to marginalize over
        states_and_actions_to_marginalize = {}

        # this works because the internal neighbors always come first in the indexing into state strings
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

        Currently this neighbor must be an internal neighbor cell, not an active or sensory cell

        Connection to a new active and sensory cell must occur in the System, not within the internal cells

        Currently the new neighbor becomes the first neighbor in the list of neighbors
        in order to preserve indexing.
        The probabilities in the B, C, D matrices with respect to states
        of the new neighbor will be distributed equally from the old state probabilities"""

        assert neighbor not in self.internal_neighbors, "Neighbor already in neighbors"

        self.internal_neighbors.insert(0, neighbor)
        self.num_internal_neighbors += 1
        self.internal_neighbor_indices = [idx for idx, _ in enumerate(self.internal_neighbors)]

        old_state_names = self.state_names.copy()

        self.setup(
            self.states,
            hidden_state_indices=self.internal_neighbor_indices + self.sensory_cell_indices,
            control_state_indices=self.internal_neighbor_indices + self.active_cell_indices,
        )
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
