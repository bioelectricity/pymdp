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
import pymdp

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

        self.connect_sensory_to_active = False

        self.internal_network = internal_network
        self.external_network = external_network

        self.sensory_network = sensory_network
        self.active_network = active_network
        self.external_act_over_time = False

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
        


        print("Creating internal agents...")
        self.internal_network.create_agents(
            incoming_cells=self.sensory_cells,
            outgoing_cells=self.active_cells,
            cell_type = "internal"

        )

        if self.connect_sensory_to_active:
            incoming_cells = self.internal_cells + self.sensory_cells 
            outgoing_cells = self.external_cells + self.sensory_cells 
        else:
            incoming_cells = self.internal_cells 
            outgoing_cells = self.external_cells
        self.active_network.create_agents(
            incoming_cells=incoming_cells,# + self.sensory_cells,
            outgoing_cells=outgoing_cells,# + self.sensory_cells,
            cell_type = "active"

        )

        if self.connect_sensory_to_active:
            sensory_incoming_cells = self.external_cells + self.active_cells 
            sensory_outgoing_cells = self.internal_cells + self.active_cells 
        else:
            sensory_incoming_cells = self.external_cells 
            sensory_outgoing_cells = self.internal_cells

        self.sensory_network.create_agents(
            incoming_cells=sensory_incoming_cells,# + self.active_cells,
            outgoing_cells=sensory_outgoing_cells,# + self.active_cells,
            cell_type = "sensory"

        )
        self.external_network.create_agents(
            incoming_cells=self.active_cells,
            outgoing_cells=self.sensory_cells,
            global_states=[],
        )

        self.compose()

        self.t = 0



        # also need to add edges between sensory and active nodes


    def compose(self):
        # compose all the networks into one system network
        system = networkx.compose(self.internal_network.network, self.sensory_network.network)
        system = networkx.compose(system, self.active_network.network)
        self.system = networkx.compose(system, self.external_network.network)

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
        if self.connect_sensory_to_active:
            for sensory_node in self.sensory_network.network.nodes:
                for active_node in self.active_network.network.nodes:
                    self.system.add_edge(sensory_node, active_node)

    def update_reward_location(self, reward_location):
        self.reward_location = reward_location
        for node in self.external_network.network.nodes:
            self.external_network.network.nodes[node]["agent"].reward_location = self.reward_location


    def update_observations(self, node, action, neighbors, qs = None):
        if qs is not None:
            qs = pymdp.maths.softmax(0.001*np.array(qs[0]))

            print(f"softmaxed qs: {qs}")

        for idx, neighbor in enumerate(neighbors):
            if qs is not None and neighbor["agent"].cell_type == "external":
                neighbor["agent"].actions_received[node] = qs[1]
            else:
                neighbor["agent"].actions_received[node] = int(action)

    def sensory_act(self, node, logging=True):
        print(F"SENSORY ACT FOR NODE {node}")
        sensory_neighbors = list(networkx.neighbors(self.sensory_network.network, node))

        # nodes that send signals to the sensory cells
        incoming_nodes_names = sensory_neighbors + list(self.external_network.network.nodes)# + list(
        

        # nodes that receive signals from the sensory cells
        outgoing_node_names = sensory_neighbors + list(self.internal_network.network.nodes)# + list(

        if self.connect_sensory_to_active:
            incoming_nodes_names += self.active_network.network.nodes 
            outgoing_node_names += self.active_network.network.nodes

        outgoing_nodes = [self.sensory_network.nodes[node] for node in sensory_neighbors] + [
            self.internal_network.nodes[node] for node in self.internal_network.nodes
        ]# + [self.active_network.nodes[node] for node in self.active_network.nodes]

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
        print(f"gamma_A: {self.internal_network.network.nodes[node]['agent'].beta_zeta}")
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
        
        action = internal_agent.act(signals, self.distance_to_reward)

        print(f"Internal action {node}: {action}")
        self.update_observations(node, action, outgoing_nodes)


    def active_act(self, node, logging=False):
        print(f"ACTIVE ACT FOR NODE {node}")

        active_neighbors = list(networkx.neighbors(self.active_network.network, node))

        # nodes that send signals to the active cells
        incoming_nodes = list(self.internal_network.nodes) + active_neighbors# + list(self.sensory_network.nodes)

        # nodes that receive signals from the active cells
        outgoing_nodes =  [
            self.external_network.nodes[node] for node in self.external_network.nodes] + [self.active_network.nodes[node] for node in active_neighbors] 
        
        if self.connect_sensory_to_active:
            incoming_nodes += list(self.sensory_network.nodes)
            outgoing_nodes +=  [self.sensory_network.nodes[node] for node in self.sensory_network.nodes]
        active_agent = self.active_network.nodes[node]["agent"]

        #     active_agent.actions_sent = {n: 0 for n in list(self.external_network.nodes) + list(self.sensory_network.nodes)}
        incoming_signals = [active_agent.actions_received[i] for i in incoming_nodes]
        signals_dict = {n: active_agent.actions_received[n] for n in incoming_nodes}
        print(f"Signal to active agent: {signals_dict}")
        
        action = active_agent.act(incoming_signals)
        print(f"Active action {node}: {action}")
        print(f"Active qs: {active_agent.qs}")

        self.update_observations(node, action, outgoing_nodes)#, qs=active_agent.qs)

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

        if self.external_act_over_time:
            action, self.agent_location, self.distance_to_reward, self.probabilities = external_agent.act_accumulated(signals)
        else:

            action, self.agent_location, self.distance_to_reward, self.probabilities = external_agent.act(signals)
        self.external_signal = action
        self.update_observations(node, action, outgoing_nodes)

        return action, self.agent_location, self.distance_to_reward, self.probabilities
    

    def renormalize_precisions(self):
        for node in self.internal_network.nodes:
            if len(self.internal_network.nodes[node]["agent"].beta_zeta) > 1:
                max_distance = max(self.internal_network.nodes[node]["agent"].beta_zeta[:-1])
                min_distance = min(self.internal_network.nodes[node]["agent"].beta_zeta[:-1])

                spread = max_distance - min_distance


                old_beta_zeta = np.copy(self.internal_network.nodes[node]["agent"].beta_zeta)
                normalized_beta_zeta = (old_beta_zeta - min_distance) / spread * 0.1 + min_distance
                self.internal_network.nodes[node]["agent"].beta_zeta[:-1] = normalized_beta_zeta[:-1]
                
                print(f"old beta zeta: {old_beta_zeta}")

                print(f"normalized beta zeta: {self.internal_network.nodes[node]['agent'].beta_zeta}")
        for node in self.active_network.nodes:
            if len(self.active_network.nodes[node]["agent"].beta_zeta) > 1:
                max_distance = max(self.active_network.nodes[node]["agent"].beta_zeta[:-1])
                min_distance = min(self.active_network.nodes[node]["agent"].beta_zeta[:-1])

                spread = max_distance - min_distance


                old_beta_zeta = np.copy(self.active_network.nodes[node]["agent"].beta_zeta)
                normalized_beta_zeta = (old_beta_zeta - min_distance) / spread * 0.1 + min_distance
                self.active_network.nodes[node]["agent"].beta_zeta[:-1] = normalized_beta_zeta[:-1]
                
                print(f"old beta zeta: {old_beta_zeta}")

                print(f"normalized beta zeta: {self.active_network.nodes[node]['agent'].beta_zeta}")
        for node in self.sensory_network.nodes:
            if len(self.sensory_network.nodes[node]["agent"].beta_zeta) > 2:
                max_distance = max(self.sensory_network.nodes[node]["agent"].beta_zeta[:-1])
                min_distance = min(self.sensory_network.nodes[node]["agent"].beta_zeta[:-1])

                spread = max_distance - min_distance

                old_beta_zeta = np.copy(self.sensory_network.nodes[node]["agent"].beta_zeta)
                print(f"denominator: {((spread * 0.1) + min_distance)}")
                normalized_beta_zeta = (old_beta_zeta - min_distance) / ((spread * 0.1) + min_distance)
                if np.nan in normalized_beta_zeta:
                    continue
                else:
                    self.sensory_network.nodes[node]["agent"].beta_zeta[:-1] = normalized_beta_zeta[:-1]
                
                print(f"old beta zeta: {old_beta_zeta}")

                print(f"normalized beta zeta: {self.sensory_network.nodes[node]['agent'].beta_zeta}")




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
            if len(self.internal_network.nodes[node]["agent"].beta_zeta) > 2:
                self.internal_network.nodes[node]["agent"].update_after_trial()
            #self.internal_network.nodes[node]["agent"].curr_timestep = 0
        for node in self.sensory_network.nodes:
            if len(self.sensory_network.nodes[node]["agent"].beta_zeta) > 2:
                self.sensory_network.nodes[node]["agent"].update_after_trial()
            #self.sensory_network.nodes[node]["agent"].curr_timestep = 0

        for node in self.active_network.nodes:
            if len(self.active_network.nodes[node]["agent"].beta_zeta) > 2:
                self.active_network.nodes[node]["agent"].update_after_trial()
            #self.active_network.nodes[node]["agent"].curr_timestep = 0

    def prune(self):
        node_idx = 0
        nodes = list(self.internal_network.nodes)

        for node_idx in range(len(nodes)):
            node = nodes[node_idx]
            agent = self.internal_network.nodes[node]["agent"]
            neighbors = list(networkx.neighbors(self.internal_network.network, node))
            neighbor_idx = 0

            internal_neighbors = [n for n in neighbors if 'i' in n]

            internal_neighbor_indices = [neighbors.index(n) for n in internal_neighbors]

            precisions = [agent.beta_zeta[neighbor_idx] for neighbor_idx in internal_neighbor_indices]
            
            if np.nan in precisions:
                raise

            if len(precisions) < 2:
                continue

            log_precisions = np.log(precisions)

            minimum_precision_neighbor = np.argmin(log_precisions)


            neighbor = internal_neighbors[minimum_precision_neighbor]
            assert 'i' in neighbor

            if log_precisions[minimum_precision_neighbor]*10 < -0.7:

                print(f"PRUNING CONNECTION for node {node} and neighbor {neighbor} with precision {precisions[minimum_precision_neighbor]} and A value {agent.A[neighbor_idx]}")
                #prune this connection 
                self.internal_network.nodes[node]["agent"].disconnect_from(neighbor)
                self.internal_network.nodes[neighbor]["agent"].disconnect_from(node)
                
                node_neighbors = list(networkx.neighbors(self.internal_network.network, node))
                if neighbor in node_neighbors:
                    self.internal_network.network.remove_edge(node, neighbor)
                    #self.system.remove_edge(node, neighbor)

                # if len(list(networkx.neighbors(self.internal_network.network, node))) == 0:
                #     self.internal_network.network.remove_node(node)
                #     self.system.remove_node(node)
                #     node_idx -=1
                
                self.compose()
                
