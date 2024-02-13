#%%
import pathlib
import sys
import os 
path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)
import networkx 
from networks.network import Network
from cells.fixed_cell import FixedCell

class GenerativeProcess(Network):

    def __init__(self, num_cells, connectivity, num_env_nodes = 1):
        """ We start assuming our ABB can be modeled as an Erdos Renyi graph 
        with num_cells nodes and connectivity probability connectivity. 
        
        See https://networkx.org/documentation/stable/reference/generators.html
        for other kinds of random graphs"""

        super().__init__(num_cells, connectivity, num_env_nodes)

        
        self.create_agents()

    def create_agent(self, node):
        """Creates an active inference agent for a given node in the network"""
        neighbors = list(networkx.neighbors(self.network, node))

        num_neighbors = len(neighbors)
        agent = FixedCell(node, num_neighbors, neighbors, self.global_states, is_blanket_node = node == self.blanket_node, env_node_indices = self.env_node_indices)
        agent._action = self.actions[node]
        networkx.set_node_attributes(self.network, {node:agent}, "agent")

