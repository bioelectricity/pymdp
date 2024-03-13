# %%
import pathlib
import sys
import os
from stemai.networks.network import Network
from pymdp.envs import Env

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
import numpy as np


class StemCellSystem(Network):
    """A class representing a system of interacting networks"""

    def __init__(
        self,
        internal_network: Network,
        external_network: Network,
        sensory_network: Network,
        active_network: Network,
    ):
        """
        internal_network: a Network of internal cells
        external_network: a Network of external cells
        sensory_network: a Network of sensory cells
        active_network: a Network of active cells

        This class will compose the networks into one big system-level network
        and is responsible for forming connections between internal, sensory, active and external cells
        within each network

        """

        self.internal_network = internal_network
        self.external_network = external_network
        self.sensory_network = sensory_network
        self.active_network = active_network

        self.num_internal_cells = internal_network.num_cells
        self.num_external_cells = external_network.num_cells
        self.num_sensory_cells = sensory_network.num_cells
        self.num_active_cells = active_network.num_cells

        self.num_cells = (
            self.num_internal_cells
            + self.num_external_cells
            + self.num_sensory_cells
            + self.num_active_cells
        )

        self.internal_cell_indices = list(range(self.num_internal_cells))
        self.sensory_cell_indices = list(
            range(self.num_internal_cells, self.num_internal_cells + self.num_sensory_cells)
        )
        self.active_cell_indices = list(
            range(
                self.num_internal_cells + self.num_sensory_cells,
                self.num_internal_cells + self.num_sensory_cells + self.num_active_cells,
            )
        )
        self.external_cell_indices = list(
            range(
                self.num_internal_cells + self.num_sensory_cells + self.num_active_cells,
                self.num_cells,
            )
        )
        self.reward_interval = 0
        self.in_consistent_interval = False

        print(f"Internal cell indices: {self.internal_cell_indices}")
        print(f"Sensory cell indices: {self.sensory_cell_indices}")
        print(f"Active cell indices: {self.active_cell_indices}")
        print(f"External cell indices: {self.external_cell_indices}")

        self.set_states()

        self.internal_network.create_agents(
            incoming_cells=self.sensory_cell_indices,
            outgoing_cells=self.active_cell_indices,
            global_states=self.states,
        )
        self.external_network.create_agents(
            incoming_cells=self.active_cell_indices,
            outgoing_cells=self.sensory_cell_indices,
            global_states=self.states,
        )
        self.sensory_network.create_agents(
            incoming_cells=self.external_cell_indices + self.active_cell_indices,
            outgoing_cells=self.internal_cell_indices + self.active_cell_indices,
            global_states=self.states,
        )
        self.active_network.create_agents(
            incoming_cells=self.internal_cell_indices + self.sensory_cell_indices,
            outgoing_cells=self.external_cell_indices + self.sensory_cell_indices,
            global_states=self.states,
        )

        # compose all the networks into one system network
        system = networkx.compose(internal_network.network, sensory_network.network)
        system = networkx.compose(system, active_network.network)
        self.system = networkx.compose(system, external_network.network)

        # self.external_obs = np.random.choice(
        #     [0, 1], size=self.num_external_cells + self.num_active_cells
        # )

        self.t = 0

        for internal_node in self.internal_network.network.nodes:
            # add edges between all internal nodes and active nodes

            for active_node in self.active_network.network.nodes:
                self.system.add_edge(internal_node, active_node)
            # add edges between all internal nodes and sensory nodes

            for sensory_node in self.sensory_network.network.nodes:
                self.system.add_edge(internal_node, sensory_node)

        for external_node in self.external_network.network.nodes:
            # add edges between all external nodes and active nodes
            for active_node in self.active_network.network.nodes:
                self.system.add_edge(external_node, active_node)

            # add edges between all external nodes and sensory nodes
            for sensory_node in self.sensory_network.network.nodes:
                self.system.add_edge(external_node, sensory_node)

        # also need to add edges between sensory and active nodes
        for sensory_node in self.sensory_network.network.nodes:
            for active_node in self.active_network.network.nodes:
                self.system.add_edge(sensory_node, active_node)

    def update_observations(self, node, action_string, neighbors):
        for idx, neighbor in enumerate(neighbors):
            neighbor["agent"].actions_received[node] = int(action_string[idx])

    def update_observations_external(self, node, action, neighbors, qs=None):

        for idx, neighbor in enumerate(neighbors):
            neighbor["agent"].actions_received[node] = int(action)

    def update_reward_location(self, reward_location):
        self.reward_location = reward_location
        for node in self.external_network.network.nodes:
            self.external_network.network.nodes[node][
                "agent"
            ].reward_location = self.reward_location

    def sensory_act(self, node, update=True, accumulate=True, logging=False):

        sensory_neighbors = list(networkx.neighbors(self.sensory_network.network, node))

        # nodes that send signals to the sensory cells
        incoming_nodes_names = (
            sensory_neighbors
            + list(self.external_network.network.nodes)
            + list(self.active_network.network.nodes)
        )
        # nodes that receive signals from the sensory cells
        outgoing_node_names = (
            sensory_neighbors
            + list(self.internal_network.network.nodes)
            + list(self.active_network.network.nodes)
        )

        outgoing_nodes = (
            [self.sensory_network.nodes[node] for node in sensory_neighbors]
            + [self.internal_network.nodes[node] for node in self.internal_network.nodes]
            + [self.active_network.nodes[node] for node in self.active_network.nodes]
        )

        # print(f"Sensory agent incoming nodes: {incoming_nodes_names}")
        # print(f"Sensory agent outgoing nodes: {outgoing_node_names}")

        sensory_agent = self.sensory_network.nodes[node]["agent"]

        if self.t == 0:
            sensory_agent.actions_received = {
                n: np.random.choice([0, 1]) for n in incoming_nodes_names
            }
            sensory_agent.actions_sent = {n: np.random.choice([0, 1]) for n in outgoing_node_names}
            signals = [
                np.random.choice([0, 1]) for i in range(len(incoming_nodes_names))
            ]  # a list of signals from each external node
        else:
            signals = [sensory_agent.actions_received[i] for i in incoming_nodes_names]
        if logging:
            print(f"Environment signal to sensory agent: {signals}")
        sensory_obs = sensory_agent.state_signal_to_index(signals)
        if logging:
            print(f"Sensory observation: {sensory_obs}")

        action_string = sensory_agent.act(sensory_obs, update=update, accumulate=accumulate)

        if logging:
            print(f"Sensory action: {action_string}")

        self.update_observations(node, action_string, outgoing_nodes)

    def internal_act(self, node, update=True, accumulate=True, logging=False):

        internal_neighbors = list(networkx.neighbors(self.internal_network.network, node))

        # nodes that send signals to the internal cells
        incoming_nodes = internal_neighbors + list(self.sensory_network.network.nodes)
        # nodes that receive signals from the internal cells
        outgoing_nodes = [
            self.internal_network.network.nodes[node] for node in internal_neighbors
        ] + [self.active_network.network.nodes[node] for node in self.active_network.network.nodes]

        internal_agent = self.internal_network.network.nodes[node]["agent"]

        signals = [internal_agent.actions_received[i] for i in incoming_nodes]
        if logging:
            print(f"Signal to internal agent {node}: {signals}")
        internal_obs = internal_agent.state_signal_to_index(signals)
        if logging:
            print(f"Internal agent {node} observation: {internal_obs}")
        action_string = internal_agent.act(internal_obs, update=update, accumulate=accumulate)
        if logging:
            print(f"Internal agent {node} action: {action_string}")
        self.update_observations(node, action_string, outgoing_nodes)

    def active_act(self, node, update=True, accumulate=True, logging=False):

        active_neighbors = list(networkx.neighbors(self.active_network.network, node))

        # nodes that send signals to the active cells
        incoming_nodes = (
            active_neighbors + list(self.internal_network.nodes) + list(self.sensory_network.nodes)
        )

        # nodes that receive signals from the active cells
        outgoing_nodes = (
            [self.active_network.nodes[node] for node in active_neighbors]
            + [self.external_network.nodes[node] for node in self.external_network.nodes]
            + [self.sensory_network.nodes[node] for node in self.sensory_network.nodes]
        )

        active_agent = self.active_network.nodes[node]["agent"]

        if self.t == 0:
            active_agent.actions_received = {n: 0 for n in incoming_nodes}
        #     active_agent.actions_sent = {n: 0 for n in list(self.external_network.nodes) + list(self.sensory_network.nodes)}

        incoming_signals = [active_agent.actions_received[i] for i in incoming_nodes]
        if logging:
            print(f"Signal to active agent from internal agent: {incoming_signals}")
        active_obs = active_agent.state_signal_to_index(incoming_signals)
        if logging:
            print(f"Active observation: {active_obs}")
        action_string = active_agent.act(active_obs, update=update, accumulate=accumulate)
        if logging:
            print(f"Active action: {action_string}")
        self.update_observations(node, action_string, outgoing_nodes)

        return action_string

    def external_act(self, node, logging=False):
        print(f"EXTERNAL ACT FOR NODE {node}")
        external_neighbors = list(networkx.neighbors(self.external_network.network, node))
        external_nodes = [self.external_network.network.nodes[node] for node in external_neighbors]

        incoming_nodes = external_neighbors + list(self.active_network.nodes)
        outgoing_nodes = external_nodes + [
            self.sensory_network.network.nodes[node] for node in self.sensory_network.nodes
        ]

        external_agent = self.external_network.nodes[node]["agent"]

        signals = [external_agent.actions_received[i] for i in incoming_nodes]
        if logging:
            print(f"Signal to external agent: {signals}")

        action, self.agent_location, self.distance_to_reward, self.probabilities = (
            external_agent.act(signals)
        )

        self.external_signal = action

        self.update_observations_external(node, action, outgoing_nodes)

        return action, self.agent_location, self.distance_to_reward, self.probabilities

    def step(self, logging=False):

        # first : we take the external observation, and we pass it to the sensory network

        # first the sensory cells act in response to the previous external observation
        for sensory_node in self.sensory_network.nodes:
            self.sensory_act(sensory_node, update=True, logging=logging)

        # then, the internal cells act
        for internal_node in self.internal_network.nodes:
            self.internal_act(internal_node, update=True, logging=logging)

        # then, the active cells act
        for active_node in self.active_network.nodes:
            self.active_act(active_node, update=True, logging=logging)

        # finally, the external nodes act
        for external_node in self.external_network.nodes:
            action, agent_location, distance, probabilities = self.external_act(
                external_node, logging=logging
            )

        self.t += 1

        return action, agent_location, distance, probabilities

    def _reset(self):
        for node in self.internal_network.nodes:
            self.internal_network.nodes[node]["agent"].curr_timestep = 0
        for node in self.sensory_network.nodes:
            self.sensory_network.nodes[node]["agent"].curr_timestep = 0

        for node in self.active_network.nodes:
            self.active_network.nodes[node]["agent"].curr_timestep = 0
        for node in self.external_network.nodes:
            self.external_network.nodes[node]["agent"].agent_location = self.agent_location
            self.external_network.nodes[node]["agent"].reward_location = self.reward_location

    def update_gamma_A(self):
        for node in self.internal_network.nodes:
            self.internal_network.nodes[node]["agent"].update_after_trial()
            # self.internal_network.nodes[node]["agent"].curr_timestep = 0
        for node in self.sensory_network.nodes:
            self.sensory_network.nodes[node]["agent"].update_after_trial()
            # self.sensory_network.nodes[node]["agent"].curr_timestep = 0

        for node in self.active_network.nodes:
            self.active_network.nodes[node]["agent"].update_after_trial()
            # self.active_network.nodes[node]["agent"].curr_timestep = 0

    def prune(self):
        for node in self.internal_network.nodes:
            agent = self.internal_network.nodes[node]["agent"]
            neighbors = list(networkx.neighbors(self.internal_network.network, node))
            neighbor_idx = 0
            for neighbor in neighbors:
                if "i" not in neighbor:
                    neighbor_idx += 1
                    continue

                # neighbor_node = self.internal_network.nodes[neighbors[neighbor_idx]]
                gamma_A_m = agent.beta_zeta[neighbor_idx]
                print(f"Gamma: {gamma_A_m}")
                print(f"Checking if I should prune: {1 / gamma_A_m[1]}")
                if 1 / gamma_A_m[1] < 0.2:
                    print(
                        f"PRUNING CONNECTION for node {node} and neighbor {neighbor} with precision {1 / gamma_A_m} and A value {agent.A[neighbor_idx]}"
                    )
                    # prune this connection
                    self.internal_network.nodes[node]["agent"].disconnect_from(neighbor)
                    self.internal_network.nodes[neighbor]["agent"].disconnect_from(node)

                    node_neighbors = list(networkx.neighbors(self.internal_network.network, node))
                    if neighbor in node_neighbors:
                        self.internal_network.network.remove_edge(node, neighbor)
                    # Update neighbor index after pruning
                    neighbor_idx -= 1
                neighbor_idx += 1

    def update_after_trial(self):
        print("Updating transition matrices")
        for node in self.internal_network.nodes:
            self.internal_network.nodes[node]["agent"].update_B_after_trial()
            self.internal_network.nodes[node]["agent"]._reset()
        for node in self.sensory_network.nodes:
            self.sensory_network.nodes[node]["agent"].update_B_after_trial()
            self.sensory_network.nodes[node]["agent"]._reset()

        for node in self.active_network.nodes:
            self.active_network.nodes[node]["agent"].update_B_after_trial()
            self.active_network.nodes[node]["agent"]._reset()
