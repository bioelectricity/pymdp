#%%
import pathlib
import sys
import os 
path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)
import networkx 
import numpy as np
from utils import generate_binary_numbers

class Network:
    """Abstract Network class that will be inherited by GenerativeModel and GenerativeProcess"""

    def __init__(self, num_cells, connectivity, num_env_nodes = 1, blanket_node = 0):
        """
        num_cells: number of cells in the network 
        connectivity: float between 0 and 1, probability of connection between any two cells
        num_env_nodes: the number of environment nodes that the blanket node will interact with 
        blanket_node: the index of the blanket node in the network
        """

        self.num_cells = num_cells
        self.connectivity = connectivity
        self.blanket_node = blanket_node
        self.num_env_nodes = num_env_nodes
        self.env_node_indices = list(range(num_cells, num_cells + num_env_nodes))

        self.network = networkx.fast_gnp_random_graph(num_cells, connectivity)
        #connect the blanket cell to the environment nodes 
        self.connect_blanket_to_env()

        self.actions = np.random.choice([0,1], size = num_cells)
        self.set_global_states()

        print(f"Global_states : {self.global_states}")

    def connect_blanket_to_env(self):
        """Connects the blanket cell to the environment cell"""
        for env_node in self.env_node_indices:
            self.network.add_node(env_node)
            self.network.add_edge(self.blanket_node, env_node)

    def set_global_states(self):
        """The global state names for all the cell signals in the network, plus the environment nodes"""
        self.global_states = [x[::-1] for x in generate_binary_numbers(self.num_cells+self.num_env_nodes, 2**(self.num_cells+self.num_env_nodes))]

    def create_agents(self):
        """Creates active inference agents 
        for each node in the network"""

        for node in self.network.nodes:
            if node not in self.env_node_indices: #environment nodes aren't agents
                self.create_agent(node)

        return self.network
    

    def generate_observations(self, agent, neighbors, env_obs = None):
        """Generates the observation for a given agent given its neighbors
        
        if the agent is the blanket node, it receives the environment observation as well"""
        signals = [agent.actions_received[i] for i in neighbors if i not in self.env_node_indices]

        if env_obs is not None:
            signals += env_obs# a list of zero or 1 for each neighbor
        index = agent.signal_to_index(signals)
        return index

    
    def act(self, env_observation):
        

        for node in self.network.nodes:
            if node in self.env_node_indices:
                #environment nodes don't act
                continue
            agent = self.network.nodes[node]["agent"]
            neighbors = list(networkx.neighbors(self.network, node))
            if node == self.blanket_node:
                env_obs = env_observation
            else:
                env_obs = None
            obs = self.generate_observations(agent,neighbors, env_obs)
            agent.infer_states([obs])
            agent.infer_policies()
            agent.action_signal = int(agent.sample_action()[0])

            agent.action_string = agent.state_names[agent.action_signal]
            #this is the action sent to each neighbor + action sent to env

            for idx, neighbor_idx in enumerate(neighbors):

                neighbor = self.network.nodes[neighbor_idx]
                if neighbor_idx not in self.env_node_indices:
                    neighbor["agent"].actions_received[node] = int(agent.action_string[idx])
            if node == self.blanket_node:
                env_neighbors = [i for i, n in enumerate(list(networkx.neighbors(self.network, node))) if n in self.env_node_indices]
                blanket_node_signals_to_env = [int(agent.action_string[i]) for i in env_neighbors]
        return blanket_node_signals_to_env

#%%
