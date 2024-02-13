# %%
import pathlib
import sys
import os

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
import numpy as np


class System:

    def __init__(
        self,
        internal_network, 
        external_network, 
        sensory_network, 
        active_network
    ):


        self.internal_network = internal_network
        self.external_network = external_network
        self.sensory_network = sensory_network
        self.active_network = active_network
        
        self.num_internal_cells = internal_network.num_cells
        self.num_external_cells = external_network.num_cells
        self.num_sensory_cells = sensory_network.num_cells
        self.num_active_cells = active_network.num_cells

        self.num_cells = self.num_internal_cells + self.num_external_cells + self.num_sensory_cells + self.num_active_cells

        self.internal_cell_indices = list(range(self.num_internal_cells))
        self.sensory_cell_indices = list(range(self.num_internal_cells, self.num_internal_cells + self.num_sensory_cells))
        self.active_cell_indices = list(range(self.num_internal_cells + self.num_sensory_cells, self.num_internal_cells + self.num_sensory_cells + self.num_active_cells))
        self.external_cell_indices = list(range(self.num_internal_cells + self.num_sensory_cells + self.num_active_cells, self.num_cells))

        self.system = networkx.compose(internal_network.network, external_network.network, sensory_network.network, active_network.network)

        self.external_obs = np.random.choice([0, 1], size=self.num_external_cells)

        #need to be sure the indices here are correct 
        for internal_node in self.internal_network.network.nodes:
            for active_node in self.active_network.network.nodes:
                self.system.add_edge(internal_node, active_node)

            for sensory_node in self.sensory_network.network.nodes:
                self.system.add_edge(internal_node, sensory_node)

        for external_node in self.external_network.network.nodes:
            for active_node in self.active_network.network.nodes:
                self.system.add_edge(external_node, active_node)

            for sensory_node in self.sensory_network.network.nodes:
                self.system.add_edge(external_node, sensory_node)

    def update_observations(self, node, action_string, neighbors):
        for idx, neighbor in enumerate(neighbors):
            neighbor["agent"].actions_received[node] = int(action_string[idx])


    def sensory_act(self, node):

        env_signals = self.external_obs # a list of signals from each external node

        internal_nodes = self.internal_network.nodes

        sensory_agent = self.sensory_network.nodes[node]["agent"]

        sensory_obs = sensory_agent.state_signal_to_index(env_signals)

        action_string = sensory_agent.act(sensory_obs)

        self.update_observations(node, action_string, internal_nodes)

    def internal_act(self, node):

        internal_neighbors = list(networkx.neighbors(self.internal_network, node)) 
        sensory_neighbors = list(self.sensory_network.nodes)
        active_neighbors = list(self.active_network.nodes)

        internal_agent = self.network.nodes[node]["agent"]

        signals = [internal_agent.actions_received[i] for i in internal_neighbors + sensory_neighbors]

        internal_obs = internal_agent.state_signal_to_index(signals)

        action_string = internal_agent.act(internal_obs)

        self.update_observations(node, action_string, internal_neighbors + active_neighbors)

    def active_act(self, node):
        internal_nodes = self.internal_network.nodes
        external_nodes = self.external_network.nodes

        active_agent = self.active_network.nodes[node]["agent"]

        internal_signals = [active_agent.actions_received[i] for i in internal_nodes]

        active_obs = active_agent.state_signal_to_index(internal_signals)

        action_string = active_agent.act(active_obs)

        self.update_observations(node, action_string, external_nodes)

    def external_act(self, node):

        external_neighbors = list(networkx.neighbors(self.external_network, node))
        active_neighbors = list(self.active_network.nodes)
        sensory_neighbors = list(self.sensory_network.nodes)

        external_agent = self.external_network.nodes[node]["agent"]

        signals = [external_agent.actions_received[i] for i in external_neighbors + active_neighbors]

        external_obs = external_agent.state_signal_to_index(signals)

        action_string = external_agent.act(external_obs)

        self.update_observations(node, action_string, external_neighbors + sensory_neighbors)

    def step(self):

        #first : we take the external observation, and we pass it to the sensory network

        # first the sensory cells act in response to the previous external observation
        for sensory_node in self.sensory_network.nodes:
            self.sensory_act(sensory_node)

        #then, the internal cells act
        for internal_node in self.internal_network.nodes:
            self.internal_act(internal_node)

        #then, the active cells act 
        for active_node in self.active_network.nodes:
            self.active_act(active_node)

        #finally, the external nodes act
        for external_node in self.external_network.nodes:
            self.external_act(external_node)

