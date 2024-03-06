# %%
import pathlib
import sys
import os

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
import numpy as np
from stemai.cells.neuronal_cell import NeuronalCell
from stemai.utils import generate_binary_numbers


class NeuronalNetwork:
    """Abstract Network class that will be inherited by GenerativeModel and GenerativeProcess"""

    def __init__(self, num_cells, connectivity, node_labels, celltype=NeuronalCell, gamma_A = 1.0, gamma_B = 1.0):
        """
        num_cells: number of cells in the network
        connectivity: float between 0 and 1, probability of connection between any two cells
        node_labels: list of strings, the names of the nodes in the network
        """

        self.num_cells = num_cells
        self.connectivity = connectivity

        self.network = networkx.fast_gnp_random_graph(num_cells, connectivity)

        self.network = networkx.relabel_nodes(
            self.network, dict(zip(self.network.nodes, node_labels))
        )
        self.nodes = self.network.nodes

        self.actions = {n: np.random.choice([0, 1]) for n in node_labels}

        self.gamma_A = gamma_A

        self.gamma_B = gamma_B

        self.color = "lightblue"  # for plotting

        self.celltype = celltype

        
    def create_agents(self, incoming_cells, outgoing_cells, cell_type=None):
        """Creates active inference agents for each node in the network

        incoming_cells: list of indices of cells that send signals to the current cell
        outgoing_cells: list of indices of cells that receive signals from the current cell

        This function will be called from within the global system that has multiple composed networks,
        and here, global states represents the entire state space of the global system"""
        print("HERE")
        print("Creating agents")
        
        for idx, node in enumerate(self.network.nodes):
            neighbors = list(networkx.neighbors(self.network, node)) + incoming_cells 

            
            agent = self.celltype(idx, neighbors, self.gamma_A, self.gamma_B)
            networkx.set_node_attributes(self.network, {node: agent}, "agent")

            print("CREATING AGENT ")

            print(f"Node {node}")
            print(f"Neighbors: {neighbors}")
            print(f"A matrix: {agent.A}")
            print(f"B matrix: {agent.B}")
            print(f"Gamma A: {agent.gamma_A}")
            print()
            agent.cell_type = cell_type
            
            # initialize the actions received from other internal neighbors
            agent.actions_received = {
                n: np.random.choice([0, 1]) for n in neighbors
            }  # keep track of what you received and from who

            # initialize the actions sent to other internal neighbors
            agent.actions_sent = {n: np.random.choice([0, 1]) for n in neighbors + outgoing_cells}
 