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
import numpy as np

class GenerativeModel(Network):

    def __init__(self, num_cells, connectivity, initial_action = None):
        """ We start assuming our ABB can be modeled as an Erdos Renyi graph 
        with num_cells nodes and connectivity probability connectivity. 
        
        See https://networkx.org/documentation/stable/reference/generators.html
        for other kinds of random graphs"""

        super().__init__(num_cells, connectivity,initial_action)

        
        self.create_agents()

    def create_agent(self, node):
        neighbors = list(networkx.neighbors(self.network, node))

        num_neighbors = len(neighbors)
        agent = StemCell(node, num_neighbors, neighbors, self.global_states)
        agent._action = self.actions[node]
        networkx.set_node_attributes(self.network, {node:agent}, "agent")

    def create_agents(self):

        for node in self.network.nodes:
            self.create_agent(node)

        return self.network
    
    def disconnect_cells(self, node1_index, node2_index):
        """Removes a connection in the network"""

        node1, node2 = self.network.nodes[node1_index], self.network.nodes[node2_index]
        self.network.remove_edge(node1_index, node2_index)
        print(f"Disconnecting {node1_index} from {node2_index}")

        node1["agent"].disconnect_from(node2_index) 
        print(f"Disconnecting {node2_index} from {node1_index}")

        node2["agent"].disconnect_from(node1_index)
        
        return self.network
    
    def connect_cells(self, node1_index, node2_index):
        """Adds a connection in the network"""
        node1, node2 = self.network.nodes[node1_index], self.network.nodes[node2_index]

        self.network.add_edge(node1_index, node2_index)
        node1["agent"].connect_to(node2_index)
        node2["agent"].connect_to(node1_index)

        return self.network

    def kill_cell(self, node):
        """Removes a cell from the network"""

        neighbors = list(self.network.neighbors(node)).copy()
        neighbors.sort(reverse=True) #so that we don't have index errors after removal
        for neighbor in neighbors:
            print(f"Disconnecting {node} from {neighbor}")
            self.disconnect_cells(node, neighbor)

        self.network.remove_node(node)

        return self.network

    def divide_cell(self, parent_node, connect_to_neighbors = None):
        """Adds a cell to the network
        
        If connect_to_neighbors is None, then the child 
         will only be connected to its parent. 
          
        If connect_to_neighbors is "all" then the child 
         will be connected to all of the neighbors of the parent (and the parent)
          
        If connect_to_neighbors is "half", then the child wil be 
        connected to half of the parents neighbors (and the parent)"""

        if connect_to_neighbors is not None:
            assert connect_to_neighbors in ["all", "half"], "connect_to_neighbors must be all or half"

        #make a new node and connect it to the parent
        child_node = self.num_cells
        self.network.add_node(child_node)
        self.num_cells +=1 
        self.actions = np.append(self.actions, np.random.choice([0,1]))
        self.set_global_states()
        self.create_agent(child_node)
        self.network.add_edge(parent_node, child_node)
        self.connect_cells(parent_node, child_node)

        if connect_to_neighbors == "all":
            for neighbor in self.network.neighbors(parent_node):
                if neighbor == child_node:
                    continue
                self.connect_cells(neighbor, child_node)
                self.network.add_edge(neighbor, child_node)
        elif connect_to_neighbors == "half":
            neighbors = list(self.network.neighbors(parent_node))
            for neighbor in neighbors[:len(neighbors)//2]:
                self.connect_cells(neighbor, child_node)
                self.network.add_edge(neighbor, child_node)

        return self.network