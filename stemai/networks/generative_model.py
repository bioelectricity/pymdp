#%%
import pathlib
import sys
import os 
path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)
import networkx 
from networks.network import Network
from cells.stem_cell import StemCell

class GenerativeModel(Network):

    def __init__(self, num_cells, connectivity, initial_action = None):
        """ We start assuming our ABB can be modeled as an Erdos Renyi graph 
        with num_cells nodes and connectivity probability connectivity. 
        
        See https://networkx.org/documentation/stable/reference/generators.html
        for other kinds of random graphs"""

        super().__init__(num_cells, connectivity,initial_action)

        
        self.create_agents()


    def create_agents(self):

        agent_dict = {}

        for node in self.network.nodes:
            neighbor_indices = list(networkx.neighbors(self.network, node))

            num_neighbors = len(neighbor_indices)
            agent = StemCell(node, num_neighbors, neighbor_indices, self.global_states)
            agent._action = self.actions[node]
            agent_dict[node] = agent
        networkx.set_node_attributes(self.network, agent_dict, "agent")

        return self.network
    