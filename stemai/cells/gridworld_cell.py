from stemai.cells.external_cell import ExternalCell
from pymdp.envs import TMazeEnv
from pymdp import utils
from pymdp import maths
import itertools
import numpy as np
from scipy import stats


class GridWorldCell(ExternalCell):
    """A class representing an external cell
    The external cell will have a fixed random B matrix
    and it will not update B after state inference.

    The hidden state space of the external cell will be the external cell neighbors and the active cells
    and the control state space will be the external cell neighbors and the sensory cells"""

    def __init__(
        self,
        node,
        neighbors,
        external_cell_indices,
        active_cell_indices,
        sensory_cell_indices,
        states,
        reward_probs=None,
    ):

        # super().__init__(node, neighbors, external_cell_indices, active_cell_indices, sensory_cell_indices, states)

        self.neighbors = neighbors
        self.cell_type = "external"
        self.observation_history = []

        print(f"External neighbors: {self.neighbors}")
        # self.neighbor_indices = external_cell_indices
        # self.sensory_cell_indices = sensory_cell_indices
        # self.active_cell_indices = active_cell_indices
        self.action_names = ["RIGHT", "LEFT", "DOWN", "UP"]

        self.actions_received = {
            n: np.random.choice([0, 1]) for n in self.neighbors + active_cell_indices
        }

    def setup(self, num_neighbors):

        self.num_states = [2]
        self.num_obs = [2] * num_neighbors
        self.num_actions = 1

    def act(self, qs_obs: int) -> str:
        """Perform state and action inference, return the action string
        which includes the action signal for each actionable neighbor
        of this cell

        obs: the observation signal index from the observable neighbors


        Here for the tmaze cell, the observation will be the location of the rat
        one out of four possible observations coming from the active cells
        """

        obs = np.random.choice(np.flatnonzero(qs_obs == np.array(qs_obs).max()))

        print(f"Action: {self.action_names[obs]}")

        self.observation_history.append(obs)

        if obs == 0:
            if self.agent_location[1] < self.grid_size - 1:
                self.agent_location = (self.agent_location[0], self.agent_location[1] + 1)
        elif obs == 1:
            if self.agent_location[1] > 0:
                self.agent_location = (self.agent_location[0], self.agent_location[1] - 1)
        elif obs == 3:
            if self.agent_location[0] > 0:
                self.agent_location = (self.agent_location[0] - 1, self.agent_location[1])
        elif obs == 2:
            if self.agent_location[0] < self.grid_size - 1:
                self.agent_location = (self.agent_location[0] + 1, self.agent_location[1])

        # manhattan distance
        distance_to_reward_location = abs(self.agent_location[0] - self.reward_location[0]) + abs(
            self.agent_location[1] - self.reward_location[1]
        )

        print(f"Agent location: {self.agent_location}")
        print(f"Distance to reward location: {distance_to_reward_location}")
        probabilities = [0.5, 0.5]
        if distance_to_reward_location == 0:  # on the point
            signal = 1
        elif distance_to_reward_location == self.grid_size * 2:
            # completely rnadom
            signal = np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            # sampling randomly from a distance across 0 and 1
            # the probabilities of the reward depend on the distance
            probabilities = np.array(
                [
                    0.5 - ((20 - distance_to_reward_location) / 20) / 2,
                    0.5 + ((20 - distance_to_reward_location) / 20) / 2,
                ]
            )

            # TODO can be a new parameter to make this a non-linear probability distribution

            print(f"Probabilities for external agent action sampling: {probabilities}")
            signal = np.random.choice([0, 1], p=probabilities)
            print(f"Environmental signal : {signal}")

        return signal, self.agent_location, distance_to_reward_location, probabilities

    def act_accumulated(self, qs_obs: int, outgoing_nodes) -> str:
        """Perform state and action inference, return the action string
        which includes the action signal for each actionable neighbor
        of this cell

        obs: the observation signal index from the observable neighbors


        Here for the tmaze cell, the observation will be the location of the rat
        one out of four possible observations coming from the active cells
        """
        obs = np.random.choice(np.flatnonzero(qs_obs == np.array(qs_obs).max()))

        print(f"Action: {self.action_names[obs]}")

        self.observation_history.append(obs)

        print(f"History: {self.observation_history[-self.action_time_horizon:]} ")

        obs = int(stats.mode(self.observation_history[-self.action_time_horizon :])[0])

        print(f"Averaged observation: {obs}")

        if obs == 0:
            if self.agent_location[1] < self.grid_size - 1:
                self.agent_location = (self.agent_location[0], self.agent_location[1] + 1)
        elif obs == 1:
            if self.agent_location[1] > 0:
                self.agent_location = (self.agent_location[0], self.agent_location[1] - 1)
        elif obs == 3:
            if self.agent_location[0] > 0:
                self.agent_location = (self.agent_location[0] - 1, self.agent_location[1])
        elif obs == 2:
            if self.agent_location[0] < self.grid_size - 1:
                self.agent_location = (self.agent_location[0] + 1, self.agent_location[1])

        # manhattan distance
        distance_to_reward_location = abs(self.agent_location[0] - self.reward_location[0]) + abs(
            self.agent_location[1] - self.reward_location[1]
        )

        print(f"Agent location: {self.agent_location}")
        print(f"Distance to reward location: {distance_to_reward_location}")
        probabilities = [0.5, 0.5]

        if distance_to_reward_location == 0:  # on the point
            probabilities = [0, 1]
        elif distance_to_reward_location == self.grid_size * 2:
            # completely rnadom
            probabilities = [0.5, 0.5]
        # signal = np.random.choice([0,1], p=[0.5, 0.5])
        else:
            # sampling randomly from a distance across 0 and 1
            # the probabilities of the reward depend on the distance
            probabilities = np.array(
                [
                    0.5 - ((20 - distance_to_reward_location) / 20) / 2,
                    0.5 + ((20 - distance_to_reward_location) / 20) / 2,
                ]
            )

        each_sensory_cell_signal = []
        for outgoing_node in outgoing_nodes:
            signal = np.random.choice([0, 1], p=probabilities)
            each_sensory_cell_signal.append(signal)

        return (
            each_sensory_cell_signal,
            self.agent_location,
            distance_to_reward_location,
            probabilities,
        )
