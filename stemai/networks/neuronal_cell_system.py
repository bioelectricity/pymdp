# %%
import pathlib
import sys
import os
from stemai.networks.network import Network
import time
import pdb

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
import numpy as np
from pymdp import utils


class System(Network):
    """A class representing a system of interacting networks

    This system does not have active or sensory networks"""

    def __init__(
        self,
        internal_network: Network,
        external_network: Network,
        sensory_network: Network,
        active_network: Network,
        active_connectivity_proportion=0.6,
        sensory_connectivity_proportion=0.3,
        action_time_horizon=10,
        precision_threshold=0.5,
        add_connections = False, 
        prune_connections = True,
        new_connection_node_percentage = 0.1, 
        new_connection_probability = 0.1,
        logging = False,
        default = False
    ):
        """
        internal_network: a Network of internal cells
        external_network: a Network of external cells

        This class will compose the networks into one big system-level network
        and is responsible for forming connections between internal and external cells
        within each network

        """

        self.connect_sensory_to_active = False

        self.internal_network = internal_network
        self.external_network = external_network

        self.sensory_network = sensory_network
        self.active_network = active_network
        self.external_act_over_time = True
        self.action_time_horizon = action_time_horizon
        self.precision_threshold = precision_threshold

        self.num_internal_cells = internal_network.num_cells
        self.num_external_cells = external_network.num_cells
        self.num_sensory_cells = sensory_network.num_cells
        self.num_active_cells = active_network.num_cells
        self.active_connectivity_proportion = active_connectivity_proportion
        self.sensory_connectivity_proportion = sensory_connectivity_proportion

        self.internal_correlation_matrix = np.zeros(
            (self.num_internal_cells, self.num_internal_cells)
        )

        self.action_names = ["RIGHT", "LEFT", "DOWN", "UP"]
        self.action_time_horizon = action_time_horizon

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

        self.add_connections = add_connections
        self.prune_connections = prune_connections
        self.new_connection_node_percentage = new_connection_node_percentage
        self.new_connection_probability = new_connection_probability
        self.logging = logging

        self.compose(default = default)

        self.configure()

        self.set_agents_in_system()

        self.t = 0

        # also need to add edges between sensory and active nodes

    def compose(self, default = False):
        # compose all the networks into one system network
        system = networkx.compose(self.internal_network.network, self.sensory_network.network)
        system = networkx.compose(system, self.active_network.network)
        self.system = networkx.compose(system, self.external_network.network)
        if default and not os.path.exists("internal_network.pickle"):
            import pickle
            pickle.dump(self.internal_network.network, open("internal_network.pickle", "wb"))
            pickle.dump(self.sensory_network.network, open("sensory_network.pickle", "wb"))
            pickle.dump(self.active_network.network, open("active_network.pickle", "wb"))
            pickle.dump(self.external_network.network, open("external_network.pickle", "wb"))


    def set_agents_in_system(self):
        for node in self.internal_network.nodes:
            self.system.nodes[node]["agent"] = self.internal_network.nodes[node]["agent"]
        for node in self.sensory_network.nodes:
            self.system.nodes[node]["agent"] = self.sensory_network.nodes[node]["agent"]
        for node in self.active_network.nodes:
            self.system.nodes[node]["agent"] = self.active_network.nodes[node]["agent"]
        for node in self.external_network.nodes:
            self.system.nodes[node]["agent"] = self.external_network.nodes[node]["agent"]
            self.system.nodes[node]["agent"].action_time_horizon = self.action_time_horizon

            self.system.nodes[node]["agent"].action_time_horizon = self.action_time_horizon
        # different subsets of internal and active cells

    def configure(self):
        first_half_of_internal = list(self.internal_network.network.nodes)[
            : int(len(self.internal_cells) / 3)
        ]
        second_half_of_internal = list(self.internal_network.network.nodes)[
            int(len(self.internal_cells) / 3) :
        ]

        self.internal_network.incoming_nodes = {
            internal_node: [] for internal_node in self.internal_network.network.nodes
        }
        self.internal_network.outgoing_nodes = {
            internal_node: [] for internal_node in self.internal_network.network.nodes
        }

        self.active_network.incoming_nodes = {
            active_node: [] for active_node in self.active_network.network.nodes
        }
        self.active_network.outgoing_nodes = {
            active_node: self.external_cells for active_node in self.active_network.network.nodes
        }

        self.sensory_network.outgoing_nodes = {
            sensory_node: [] for sensory_node in self.sensory_network.network.nodes
        }
        self.sensory_network.incoming_nodes = {
            sensory_node: self.external_cells for sensory_node in self.sensory_network.network.nodes
        }

        # external_incoming = {external_node: self.active_cells for external_node in self.external_network.network.nodes}
        # external_outgoing = {external_node: self.sensory_cells for external_node in self.external_network.network.nodes}

        # modify the code to randomly sample internal nodes and sensory nodes to connect to active nodes

        for active_node in self.active_network.network.nodes:
            internal_nodes_to_connect_to_active = np.random.choice(
                first_half_of_internal,
                size=int(len(first_half_of_internal) * self.active_connectivity_proportion),
                replace=False,
            )
            for internal_node in internal_nodes_to_connect_to_active:
                self.system.add_edge(internal_node, active_node)
                self.internal_network.outgoing_nodes[internal_node].append(active_node)
                self.active_network.incoming_nodes[active_node].append(internal_node)

        for sensory_node in self.sensory_network.network.nodes:
            internal_nodes_to_connect_to_sensory = np.random.choice(
                second_half_of_internal,
                size=int(len(second_half_of_internal) * self.sensory_connectivity_proportion),
                replace=False,
            )

            for internal_node in internal_nodes_to_connect_to_sensory:
                self.system.add_edge(sensory_node, internal_node)
                self.internal_network.incoming_nodes[internal_node].append(sensory_node)
                self.sensory_network.outgoing_nodes[sensory_node].append(internal_node)

        for external_node in self.external_network.network.nodes:
            # add edges between all external nodes and active nodes
            for active_node in self.active_network.network.nodes:
                self.system.add_edge(external_node, active_node)

            # add edges between all external nodes and sensory nodes
            for sensory_node in self.sensory_network.network.nodes:
                self.system.add_edge(external_node, sensory_node)

        if self.connect_sensory_to_active:
            for sensory_node in self.sensory_network.network.nodes:
                for active_node in self.active_network.network.nodes:
                    self.system.add_edge(sensory_node, active_node)

        self.internal_network.create_agents(
            incoming_cells=self.internal_network.incoming_nodes,
            outgoing_cells=self.internal_network.outgoing_nodes,
            cell_type="internal",
        )

        if self.connect_sensory_to_active:
            for active_cell in self.active_network.incoming_nodes:
                self.active_network.incoming_nodes[active_cell] += self.sensory_cells
                self.active_network.outgoing_nodes[active_cell] += self.sensory_cells

            for sensory_cell in self.sensory_network.outgoing_nodes:
                self.sensory_network.outgoing_nodes[sensory_cell] += self.active_cells
                self.sensory_network.incoming_nodes[sensory_cell] += self.active_cells

        self.active_network.create_agents(
            incoming_cells=self.active_network.incoming_nodes,  # + self.sensory_cells,
            outgoing_cells=self.active_network.outgoing_nodes,  # + self.sensory_cells,
            cell_type="active",
        )

        self.sensory_network.create_agents(
            incoming_cells=self.sensory_network.incoming_nodes,  # + self.active_cells,
            outgoing_cells=self.sensory_network.outgoing_nodes,  # + self.active_cells,
            cell_type="sensory",
        )
        self.external_network.create_agents(
            incoming_cells=self.active_cells,
            outgoing_cells=self.sensory_cells,
            global_states=[],
        )

    def update_grid_locations(self, reward_location, agent_location):
        self.reward_location = reward_location
        self.agent_location = agent_location
        for node in self.external_network.network.nodes:
            self.external_network.network.nodes[node][
                "agent"
            ].reward_location = self.reward_location
        for node in self.external_network.network.nodes:
            self.external_network.network.nodes[node]["agent"].agent_location = self.agent_location

        self.system.distance_to_reward = abs(
            self.agent_location[0] - self.reward_location[0]
        ) + abs(self.agent_location[1] - self.reward_location[1])

    def update_observations(self, node, action, neighbors, qs=None):

        for idx, neighbor in enumerate(neighbors):
            if qs is not None and neighbor["agent"].cell_type == "external":
                neighbor["agent"].actions_received[node] = qs[0][1]
            else:
                neighbor["agent"].actions_received[node] = int(action)

    def update_sensory_observatinos(self, node, actions_per_cell, neighbors):

        for idx, neighbor in enumerate(neighbors):
            neighbor["agent"].actions_received[node] = int(actions_per_cell[idx])

    def sensory_act(self, node, logging=True):
        if logging:
            print(f"SENSORY ACT FOR NODE {node}")
        sensory_neighbors = list(networkx.neighbors(self.sensory_network.network, node))

        # nodes that send signals to the sensory cells
        incoming_nodes_names = sensory_neighbors + self.sensory_network.incoming_nodes[node]

        # print(f"Sensory agent incoming nodes: {incoming_nodes_names}")

        # nodes that receive signals from the sensory cells
        outgoing_node_names = self.sensory_network.outgoing_nodes[node]

        # print(f"Sensory agent outgoing nodes: {outgoing_node_names}")

        outgoing_nodes = [self.sensory_network.nodes[node] for node in sensory_neighbors] + [
            self.internal_network.nodes[node] for node in outgoing_node_names
        ]  # + [self.active_network.nodes[node] for node in self.active_network.nodes]

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

        signals_dict = {n: sensory_agent.actions_received[n] for n in incoming_nodes_names}
        if logging:
            print(f"Signal to sensory agent: {signals_dict}")

        # here, the observation is the signal from each modality, each neighbor
        # rather than the converted signal index into the space of all possible observations
        action = sensory_agent.act(signals)

        if logging: print(f"Sensory action {node}: {action}")

        self.update_observations(node, action, outgoing_nodes)

    
    def internal_act(self, node, update=True, accumulate = True, logging=False):
        internal_agent = self.internal_network.network.nodes[node]["agent"]
        internal_neighbors = [n for n in internal_agent.neighbors if 'i' in n]
        # nodes that send signals to the internal cells
        incoming_nodes = internal_neighbors + self.internal_network.incoming_nodes[node]

        # nodes that receive signals from the internal cells

        if logging: print(f"Incoming nodes for internal agent: {incoming_nodes}")
        outgoing_nodes = [
            self.internal_network.network.nodes[node] for node in internal_neighbors
        ] + [
            self.active_network.network.nodes[node]
            for node in self.internal_network.outgoing_nodes[node]
        ]

        signals = [internal_agent.actions_received[i] for i in incoming_nodes]
        if len(signals) == 0:
            return 
        signals_dict = {n: internal_agent.actions_received[n] for n in incoming_nodes}
        if logging: print(f"Signal to internal agent: {signals_dict}")

        assert len(signals) == internal_agent.num_modalities, f"Signals: {signals} don't match num_modalities : {internal_agent.num_modalities} for node {node}"

        action = internal_agent.act(signals, self.distance_to_reward)

        if logging: print(f"Internal action {node}: {action}")
        self.update_observations(node, action, outgoing_nodes)

    def active_act(self, node, logging=False):
        if logging: print(f"ACTIVE ACT FOR NODE {node}")

        active_neighbors = list(networkx.neighbors(self.active_network.network, node))

        # nodes that send signals to the active cells
        incoming_nodes = (
            active_neighbors + self.active_network.incoming_nodes[node]
        )  # + list(self.sensory_network.nodes)

        # nodes that receive signals from the active cells
        outgoing_nodes = [self.active_network.nodes[node] for node in active_neighbors] + [
            self.external_network.nodes[node] for node in self.active_network.outgoing_nodes[node]
        ]

        active_agent = self.active_network.nodes[node]["agent"]
        # print(f"Gamma_A active: {active_agent.beta_zeta}")

        #     active_agent.actions_sent = {n: 0 for n in list(self.external_network.nodes) + list(self.sensory_network.nodes)}
        incoming_signals = [active_agent.actions_received[i] for i in incoming_nodes]
        signals_dict = {n: active_agent.actions_received[n] for n in incoming_nodes}
        if logging: print(f"Signal to active agent: {signals_dict}")

        action = active_agent.act(incoming_signals)
        if logging: print(f"Active action {node}: {action}")
        if logging: print(f"Active qs: {active_agent.qs}")

        self.update_observations(
            node, action, outgoing_nodes, [np.array([0, np.sum(incoming_signals)])]
        )

        return action

    def external_act(self, node, logging=False):
        if logging: print(f"EXTERNAL ACT FOR NODE {node}")
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

        if self.external_act_over_time:
            (
                each_sensory_cell_signal,
                self.agent_location,
                self.distance_to_reward,
                self.probabilities,
            ) = external_agent.act_accumulated(signals, outgoing_nodes)
        else:

            (
                each_sensory_cell_signal,
                self.agent_location,
                self.distance_to_reward,
                self.probabilities,
            ) = external_agent.act(signals)
        self.external_signal = each_sensory_cell_signal
        # self.update_observations(node, action, outgoing_nodes)

        self.update_sensory_observatinos(node, each_sensory_cell_signal, outgoing_nodes)

        return (
            each_sensory_cell_signal,
            self.agent_location,
            self.distance_to_reward,
            self.probabilities,
        )

    def renormalize_precisions(self):
        for node in self.internal_network.nodes:
            modalities_to_omit = len(self.internal_network.incoming_nodes[node])
            if self.logging: print(f"Node : {node}")
            if self.logging: print(f"Network incoming nodes: {self.internal_network.incoming_nodes}")
            if self.logging: print(f"Node incoming: {self.internal_network.incoming_nodes[node]}")
            if self.logging: print(f"Modalities to omit: {modalities_to_omit}")

            if len(self.internal_network.nodes[node]["agent"].gamma_A) > modalities_to_omit + 1:
                print(self.internal_network.nodes[node]["agent"].gamma_A)

                print(self.internal_network.nodes[node]["agent"].gamma_A[:-modalities_to_omit])

                if modalities_to_omit > 0:
                    gamma = self.internal_network.nodes[node]["agent"].gamma_A[:-modalities_to_omit]
                else:
                    gamma = self.internal_network.nodes[node]["agent"].gamma_A
                
                #old_beta_zeta = np.copy(self.internal_network.nodes[node]["agent"].gamma_A)

                #print(f"old gamma A: {old_beta_zeta}")

                normalized_beta_zeta = np.zeros((len(gamma), 2))

                for state in range(2):
                    gamma_per_state = np.array([g[state] for g in gamma])

                    #print(f"To normalize: {gamma_per_state}")
                    
                    max_distance = max(gamma_per_state)
                    min_distance = min(gamma_per_state)

                    spread = max_distance - min_distance

                    #print(f"Spread: {spread}")

                    if spread == 0:
                        continue


                    normalized_beta_zeta_state = (
                        (gamma_per_state - min_distance) / (spread * 10)
                    ) + min_distance


                    #print(f"normalized gamma a state {state}:   {normalized_beta_zeta_state}")

                    normalized_beta_zeta[:,state] = normalized_beta_zeta_state             
                
                #print(f"normalized gamma a: {normalized_beta_zeta}")
                
                
                if np.nan in normalized_beta_zeta:
                    normalized_beta_zeta = np.nan_to_num(normalized_beta_zeta) + 0.0001

                normalized_beta_zeta = [x for x in normalized_beta_zeta]

                #print(f"OLD GAMMA A : {self.internal_network.nodes[node]['agent'].gamma_A}")

                if modalities_to_omit > 0:
                    self.internal_network.nodes[node]["agent"].gamma_A[:-modalities_to_omit] = (
                        normalized_beta_zeta
                    )
                    self.internal_network.nodes[node]["agent"].gamma_A_prior[
                        :-modalities_to_omit
                    ] = normalized_beta_zeta

                else:
                    self.internal_network.nodes[node]["agent"].gamma_A = normalized_beta_zeta
                    self.internal_network.nodes[node][
                        "agent"
                    ].gamma_A_prior = normalized_beta_zeta

                #print(f"NEW GAMMA A : {self.internal_network.nodes[node]['agent'].gamma_A}")

                self.internal_network.nodes[node]["agent"].A = utils.scale_A_with_gamma(
                    np.copy(self.internal_network.nodes[node]["agent"].base_A),
                    self.internal_network.nodes[node]["agent"].gamma_A,
                )

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
            action, agent_location, distance, probabilities = self.external_act(
                external_node, logging=logging
            )

        self.t += 1

        return action, agent_location, distance, probabilities

    def _reset(self):
        self.t = 0
        for node in self.internal_network.nodes:
            self.internal_network.nodes[node]["agent"].reset_cell()

        for node in self.sensory_network.nodes:
            self.sensory_network.nodes[node]["agent"].reset_cell()

        for node in self.active_network.nodes:
            self.active_network.nodes[node]["agent"].reset_cell()
        for node in self.external_network.nodes:
            self.external_network.nodes[node]["agent"].agent_location = self.agent_location
            self.external_network.nodes[node]["agent"].reward_location = self.reward_location

    def update_gamma_A(self):
        for node in self.internal_network.nodes:
            if len(self.internal_network.nodes[node]["agent"].gamma_A) > 2:

                self.internal_network.nodes[node]["agent"].update_after_trial(len(self.internal_network.incoming_nodes[node]))
            self.internal_network.nodes[node]["agent"].curr_timestep = 0
        # for node in self.sensory_network.nodes:
        #     if len(self.sensory_network.nodes[node]["agent"].beta_zeta) > 2:
        #         self.sensory_network.nodes[node]["agent"].update_after_trial()
        #     #self.sensory_network.nodes[node]["agent"].curr_timestep = 0

        # for node in self.active_network.nodes:
        #     if len(self.active_network.nodes[node]["agent"].beta_zeta) > 2:
        #         self.active_network.nodes[node]["agent"].update_after_trial()
        #     #self.active_network.nodes[node]["agent"].curr_timestep = 0

    def add_new_connections(self):
        nodes = list(self.internal_network.nodes)

        nodes_to_add_new_connections_for = np.random.choice(
            list(self.internal_network.nodes), int(self.new_connection_probability * len(self.internal_network.nodes))
        )


        for node in nodes_to_add_new_connections_for:
            agent = self.internal_network.nodes[node]["agent"]

            neighbors = list(networkx.neighbors(self.internal_network.network, node))
            nodes_that_arent_neighbors = [n for n in nodes if n not in neighbors and n!=node]
            nodes_to_connect_to = np.random.choice(
                nodes_that_arent_neighbors, int(self.new_connection_node_percentage * len(nodes_that_arent_neighbors))
            )
            for new_node in nodes_to_connect_to:
                new_agent = self.internal_network.nodes[new_node]["agent"]
                if not new_agent.check_connect_to(node) or not agent.check_connect_to(new_node):
                    continue

                agent.connect_to(new_node)
                new_agent.connect_to(node)
                self.system.add_edge(node, new_node)
                self.internal_network.network.add_edge(node, new_node)
                assert node in list(networkx.neighbors(self.internal_network.network, new_node))
                assert new_node in list(networkx.neighbors(self.internal_network.network, node))

    def collect_precisions(self):
        nodes = list(self.internal_network.nodes)
        gamma_dict = {}

        for node_idx in range(len(nodes)):
            node = nodes[node_idx]

            agent = self.internal_network.nodes[node]["agent"]
            neighbors = agent.neighbors

            gamma_dict[node] = {n: (g[0],g[1]) for n, g in zip(neighbors, agent.gamma_A)}
        return gamma_dict


    def prune(self):
        node_idx = 0
        nodes = list(self.internal_network.nodes)

        for node_idx in range(len(nodes)):
            node = nodes[node_idx]

            agent = self.internal_network.nodes[node]["agent"]

            neighbors = agent.neighbors

            internal_neighbors = [n for n in neighbors if "i" in n]

            internal_neighbor_indices = [neighbors.index(n) for n in internal_neighbors]

            if len(internal_neighbors) == 0:
                continue

            other_neighbors = [n for n in neighbors if "i" not in n]

            other_neighbor_indices = [neighbors.index(n) for n in other_neighbors]

            #internal_precisions = [np.max(agent.A[neighbor_idx]) for neighbor_idx in internal_neighbor_indices]
            #all_precisions = internal_precisions + [np.max(agent.A[neighbor_idx]) for neighbor_idx in other_neighbor_indices]

            all_precisions = [np.sum(p) for p in agent.gamma_A]

            internal_precisions = [p for idx, p in enumerate(all_precisions) if idx in internal_neighbor_indices]

            #internal_precisions = [np.max(agent.A[neighbor_idx]) for neighbor_idx in internal_neighbor_indices]
            #all_precisions = internal_precisions + [np.max(agent.A[neighbor_idx]) for neighbor_idx in other_neighbor_indices]

            print(f"all precisions: {all_precisions}")
            
            minimum_precision_neighbor = np.argmin(internal_precisions)
            precision = internal_precisions[minimum_precision_neighbor]

            neighbor = internal_neighbors[minimum_precision_neighbor]
            assert "i" in neighbor

            print(f"Precision: {precision}")

            assert len(all_precisions) == len(neighbors), f"Length of all precisions: {all_precisions} doesn't match length of neighbors: {agent.neighbors}"

            #if precision < 0.5 + self.precision_threshold and precision > 0.5 - self.precision_threshold:
            if precision < 0.35:
                print(f"Pruning edge between {node} and {neighbor}")
                agent.disconnect_from(neighbor)
                #new_agent.disconnect_from(node)
                #node_neighbors = list(networkx.neighbors(self.internal_network.network, node))
                if node not in list(networkx.neighbors(self.internal_network.network, neighbor)): #only remove if you aren't their neighbor
                    print("Removing connection")
                    self.internal_network.network.remove_edge(node, neighbor)
                    self.system.remove_edge(node, neighbor)
            
            #if self.t % 10 == 0:
            maximum_precision_neighbor = np.argmax(internal_precisions)
            precision = internal_precisions[maximum_precision_neighbor]
            neighbor = internal_neighbors[maximum_precision_neighbor]
            if precision == 1:
                agent.disconnect_from(neighbor)
                if node in list(networkx.neighbors(self.internal_network.network, neighbor)):
                    self.internal_network.network.remove_edge(node, neighbor)
                    self.system.remove_edge(node, neighbor)


            # if (maximum_precision ==1):
            #     new_agent = self.internal_network.nodes[maximum_precision_neighbor]["agent"]
            #     if not new_agent.check_disconnect_from(node) or not agent.check_disconnect_from(maximum_precision_neighbor):
            #         pdb.set_trace()
            #         continue
            #     agent.disconnect_from(maximum_precision_neighbor)
            #     new_agent.disconnect_from(node)
            #     node_neighbors = list(networkx.neighbors(self.internal_network.network, node))
            #     if maximum_precision_neighbor in node_neighbors:
            #         self.internal_network.network.remove_edge(node, maximum_precision_neighbor)
            #         self.system.remove_edge(node, maximum_precision_neighbor)   

             
