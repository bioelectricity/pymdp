# %%
import pathlib
import sys
import os

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
import numpy as np
import pickle
from stemai.utils import generate_binary_numbers


class Network:
    """Abstract Network class that will be inherited by GenerativeModel and GenerativeProcess"""

    def __init__(self, num_cells, connectivity, node_labels, celltype, file):
        """
        num_cells: number of cells in the network
        connectivity: float between 0 and 1, probability of connection between any two cells
        node_labels: list of strings, the names of the nodes in the network
        """

        self.num_cells = num_cells
        self.connectivity = connectivity

        if file is not None:
            self.network = pickle.load(open(file, "rb"))
        else:

            self.network = networkx.fast_gnp_random_graph(num_cells, connectivity)

            self.network = networkx.relabel_nodes(
                self.network, dict(zip(self.network.nodes, node_labels))
            )
        self.nodes = self.network.nodes
        self.celltype = celltype

        self.actions = {n: np.random.choice([0, 1]) for n in node_labels}

        self.set_states()

    def set_states(self):
        """The global state names for all the cell signals in the network"""

        self.states = [x[::-1] for x in generate_binary_numbers(self.num_cells, 2**self.num_cells)]

    def create_agents(
        self, incoming_cells, outgoing_cells, global_states, seed_node=None, cell_type=None
    ):
        """Creates active inference agents for each node in the network

        incoming_cells: list of indices of cells that send signals to the current cell
        outgoing_cells: list of indices of cells that receive signals from the current cell

        This function will be called from within the global system that has multiple composed networks,
        and here, global states represents the entire state space of the global system"""

        self.global_states = global_states

        print("Netwrok class agent creation")

        for idx, node in enumerate(self.network.nodes):
            if seed_node is not None and idx != seed_node:
                self.create_agent(node, [], [], global_states)
            else:
                self.create_agent(node, incoming_cells, outgoing_cells, global_states)

        return self.network
