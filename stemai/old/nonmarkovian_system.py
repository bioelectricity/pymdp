# %%
import pathlib
import sys
import os
from stemai.networks.network import Network

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
import numpy as np


class NonMarkovianSystem(Network):
    """A class representing a system of interacting networks

    This system does not have active or sensory networks"""

    def __init__(
        self,
        internal_network: Network,
        external_network: Network,
    ):
        """
        internal_network: a Network of internal cells
        external_network: a Network of external cells

        This class will compose the networks into one big system-level network
        and is responsible for forming connections between internal and external cells
        within each network

        """

        self.internal_network = internal_network
        self.external_network = external_network

        self.num_internal_cells = internal_network.num_cells
        self.num_external_cells = external_network.num_cells

        self.num_cells = self.num_internal_cells + self.num_external_cells

        self.internal_cell_indices = list(range(self.num_internal_cells))
        self.external_cell_indices = list(range(self.num_internal_cells, self.num_cells))

        self.internal_seed_cell_idx = self.internal_cell_indices[0]
        self.external_seed_cell_idx = self.external_cell_indices[0]

        self.set_states()

        self.internal_network.create_agents(
            incoming_cells=[self.external_seed_cell_idx],
            outgoing_cells=[self.external_seed_cell_idx],
            global_states=self.states,
            seed_node=0,
        )
        self.external_network.create_agents(
            incoming_cells=[self.internal_seed_cell_idx],
            outgoing_cells=[self.internal_seed_cell_idx],
            global_states=self.states,
            seed_node=0,
        )
        # compose all the networks into one system network
        self.system = networkx.compose(internal_network.network, external_network.network)

        self.t = 0

        internal_seed_node = list(self.internal_network.network.nodes)[0]
        external_seed_node = list(self.external_network.network.nodes)[0]

        print(f"Adding edge between {internal_seed_node} and {external_seed_node}")

        self.system.add_edge(internal_seed_node, external_seed_node)

    def update_observations(self, node, action_string, neighbors):
        for idx, neighbor in enumerate(neighbors):
            neighbor["agent"].actions_received[node] = int(action_string[idx])

    def internal_act(self, node, logging=False):

        incoming_nodes = list(networkx.neighbors(self.system, node))

        print(f"Node {node} system neighbors: {incoming_nodes}")

        # nodes that receive signals from the internal cells
        outgoing_nodes = [self.system.nodes[node] for node in incoming_nodes]

        internal_agent = self.system.nodes[node]["agent"]

        if self.t == 0:
            signals = np.random.choice([0, 1], size=len(incoming_nodes))
        else:
            signals = [internal_agent.actions_received[i] for i in incoming_nodes]
        if logging:
            print(f"Signal to internal agent {node}: {signals}")
        internal_obs = internal_agent.state_signal_to_index(signals)
        if logging:
            print(f"Internal agent {node} observation: {internal_obs}")
        action_string = internal_agent.act(internal_obs)
        if logging:
            print(f"Internal agent {node} action: {action_string}")
        self.update_observations(node, action_string, outgoing_nodes)

    def external_act(self, node, logging=False):

        incoming_nodes = list(networkx.neighbors(self.system, node))
        print(f"Incoming neighbors for external node: {incoming_nodes}")
        outgoing_nodes = [self.system.nodes[node] for node in incoming_nodes]

        external_agent = self.system.nodes[node]["agent"]
        print(f"State space: {external_agent.state_names}")

        signals = [external_agent.actions_received[i] for i in incoming_nodes]
        if logging:
            print(f"Signal to external agent {node}: {signals}")
        external_obs = external_agent.state_signal_to_index(signals)
        if logging:
            print(f"External observation: {external_obs}")
        action_string = external_agent.act(external_obs)
        if logging:
            print(f"External action: {action_string}")
        self.update_observations(node, action_string, outgoing_nodes)

    def step(self, logging=False):

        # first : we take the external observation, and we pass it to the sensory network

        # first the sensory cells act in response to the previous external observation
        # then, the internal cells act
        for internal_node in self.internal_network.nodes:
            self.internal_act(internal_node, logging=logging)

        # finally, the external nodes act
        for external_node in self.external_network.nodes:
            self.external_act(external_node, logging=logging)

        self.t += 1
