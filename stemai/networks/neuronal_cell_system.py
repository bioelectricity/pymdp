# %%
import pathlib
import sys
import os
from stemai.networks.network import Network
import time 
path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
import numpy as np


class System(Network):
    """A class representing a system of interacting networks

    This system does not have active or sensory networks"""

    def __init__(
        self,
        internal_network: Network,
        external_network: Network,
        sensory_network: Network, 
        active_network: Network
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

        self.sensory_network = sensory_network
        self.active_network = active_network

        self.num_internal_cells = internal_network.num_cells
        self.num_external_cells = external_network.num_cells
        self.num_sensory_cells = sensory_network.num_cells
        self.num_active_cells = active_network.num_cells

        self.action_names = ["RIGHT", "LEFT", "DOWN", "UP"]

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

        self.sensory_cells = list(self.sensory_network.nodes)
        self.active_cells = list(self.active_network.nodes)
        self.internal_cells = list(self.internal_network.nodes)
        self.external_cells = list(self.external_network.nodes)

        self.set_states()

        self.internal_network.create_agents(
            incoming_cells=self.sensory_cells,
            outgoing_cells=self.active_cells,
        )
        self.active_network.create_agents(
            incoming_cells=self.internal_cells + self.sensory_cells,
            outgoing_cells=self.external_cells + self.sensory_cells,
        )
        self.sensory_network.create_agents(
            incoming_cells=self.external_cells + self.active_cells,
            outgoing_cells=self.internal_cells + self.active_cells,
        )
        self.external_network.create_agents(
            incoming_cells=self.active_cells,
            outgoing_cells=self.sensory_cells,
            global_states=self.states,
        )
        # compose all the networks into one system network
        system = networkx.compose(internal_network.network, sensory_network.network)
        system = networkx.compose(system, active_network.network)
        self.system = networkx.compose(system, external_network.network)


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

    def update_observations(self, node, action, neighbors):
        for idx, neighbor in enumerate(neighbors):
            neighbor["agent"].actions_received[node] = int(action)

    def sensory_act(self, node, logging=True):
        print(F"SENSORY ACT FOR NODE {node}")
        sensory_neighbors = list(networkx.neighbors(self.sensory_network.network, node))

        # nodes that send signals to the sensory cells
        incoming_nodes_names = sensory_neighbors + list(self.external_network.network.nodes) + list(
            self.active_network.network.nodes
        )
        # nodes that receive signals from the sensory cells
        outgoing_node_names = sensory_neighbors + list(self.internal_network.network.nodes) + list(
            self.active_network.network.nodes
        )

        outgoing_nodes = [self.sensory_network.nodes[node] for node in sensory_neighbors] + [
            self.internal_network.nodes[node] for node in self.internal_network.nodes
        ] + [self.active_network.nodes[node] for node in self.active_network.nodes]

        # print(f"Sensory agent incoming nodes: {incoming_nodes_names}")
        # print(f"Sensory agent outgoing nodes: {outgoing_node_names}")

        sensory_agent = self.sensory_network.nodes[node]["agent"]

        if self.t == 0:
            sensory_agent.actions_received = {n: np.random.choice([0,1]) for n in incoming_nodes_names}
            sensory_agent.actions_sent = {n: np.random.choice([0,1]) for n in outgoing_node_names}
            signals = [np.random.choice([0,1]) for i in range(len(incoming_nodes_names))]  # a list of signals from each external node
        else:
            signals = [sensory_agent.actions_received[i] for i in incoming_nodes_names]

        signals_dict = {n: sensory_agent.actions_received[n] for n in incoming_nodes_names}
        print(f"Signal to sensory agent: {signals_dict}")

        #here, the observation is the signal from each modality, each neighbor
        #rather than the converted signal index into the space of all possible observations
        action = sensory_agent.act(signals)

        print(f"Sensory action {node}: {action}")

        self.update_observations(node, action, outgoing_nodes)
    
    
    def internal_act(self, node, update=True, accumulate = True, logging=False):
        print(f"INTERNAL ACT FOR NODE {node}")
        internal_neighbors = list(networkx.neighbors(self.internal_network.network, node))

        # nodes that send signals to the internal cells
        incoming_nodes = internal_neighbors + list(self.sensory_network.network.nodes)
        # nodes that receive signals from the internal cells

        print(f"Active nodes: {self.active_network.network.nodes}")
        outgoing_nodes = [
            self.internal_network.network.nodes[node] for node in internal_neighbors
        ] + [self.active_network.network.nodes[node] for node in self.active_network.network.nodes]

        internal_agent = self.internal_network.network.nodes[node]["agent"]

        signals = [internal_agent.actions_received[i] for i in incoming_nodes]
        signals_dict = {n: internal_agent.actions_received[n] for n in incoming_nodes}
        print(f"Signal to internal agent: {signals_dict}")
        
        action = internal_agent.act(signals)

        print(f"Internal action {node}: {action}")
        self.update_observations(node, action, outgoing_nodes)


    def active_act(self, node, logging=False):
        print(f"ACTIVE ACT FOR NODE {node}")

        active_neighbors = list(networkx.neighbors(self.active_network.network, node))

        # nodes that send signals to the active cells
        incoming_nodes = list(self.internal_network.nodes) + active_neighbors + list(self.sensory_network.nodes)

        # nodes that receive signals from the active cells
        outgoing_nodes =  [
            self.external_network.nodes[node] for node in self.external_network.nodes
        ] + [self.active_network.nodes[node] for node in active_neighbors] + [self.sensory_network.nodes[node] for node in self.sensory_network.nodes]

        active_agent = self.active_network.nodes[node]["agent"]

        #     active_agent.actions_sent = {n: 0 for n in list(self.external_network.nodes) + list(self.sensory_network.nodes)}
        incoming_signals = [active_agent.actions_received[i] for i in incoming_nodes]
        signals_dict = {n: active_agent.actions_received[n] for n in incoming_nodes}
        print(f"Signal to active agent: {signals_dict}")
        
        action = active_agent.act(incoming_signals)
        print(f"Active action {node}: {action}")
        self.update_observations(node, action, outgoing_nodes)

        return action


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

        action, self.agent_location, self.distance_to_reward, self.probabilities = external_agent.act(signals)
        self.external_signal = action
        self.update_observations(node, action, outgoing_nodes)

        return action, self.agent_location, self.distance_to_reward, self.probabilities


    def step(self, logging=False):

        # first : we take the external observation, and we pass it to the sensory network

        # first the sensory cells act in response to the previous external observation
        for sensory_node in self.sensory_network.nodes:
            self.sensory_act(sensory_node, logging=logging)

        # then, the internal cells act
        for internal_node in self.internal_network.nodes:
            self.internal_act(internal_node, logging=logging)

        # then, the active cells act
        for active_node in self.active_network.nodes:

            self.active_act(active_node, logging=logging)

        # finally, the external nodes act
        for external_node in self.external_network.nodes:
            action, agent_location, distance, probabilities = self.external_act(external_node, logging=logging)

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
            #self.internal_network.nodes[node]["agent"].curr_timestep = 0
        for node in self.sensory_network.nodes:
            self.sensory_network.nodes[node]["agent"].update_after_trial()
            #self.sensory_network.nodes[node]["agent"].curr_timestep = 0

        for node in self.active_network.nodes:
            self.active_network.nodes[node]["agent"].update_after_trial()
            #self.active_network.nodes[node]["agent"].curr_timestep = 0

    def prune(self):
        for node in self.internal_network.nodes:
            agent = self.internal_network.nodes[node]["agent"]
            neighbors = list(networkx.neighbors(self.internal_network.network, node))
            neighbor_idx = 0
            for neighbor in neighbors:
                if 'i' not in neighbor:
                    neighbor_idx += 1
                    continue

                #neighbor_node = self.internal_network.nodes[neighbors[neighbor_idx]]
                gamma_A_m = agent.beta_zeta[neighbor_idx] 
                print(f"Gamma: {gamma_A_m*10}")
                if gamma_A_m*10 < 0.2:
                    print(f"PRUNING CONNECTION for node {node} and neighbor {neighbor} with precision {gamma_A_m} and A value {agent.A[neighbor_idx]}")
                    #prune this connection 
                    self.internal_network.nodes[node]["agent"].disconnect_from(neighbor)
                    self.internal_network.nodes[neighbor]["agent"].disconnect_from(node)

                    node_neighbors = list(networkx.neighbors(self.internal_network.network, node))
                    if neighbor in node_neighbors:
                        self.internal_network.network.remove_edge(node, neighbor)
                    # Update neighbor index after pruning
                    neighbor_idx -= 1

                neighbor_idx += 1

