# %%
import pathlib
import sys
import os

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
import numpy as np
from utils import generate_binary_numbers


class Network:
    """Abstract Network class that will be inherited by GenerativeModel and GenerativeProcess"""

    def __init__(
        self,
        num_cells,
        connectivity,
    ):
        """
        num_cells: number of cells in the network
        connectivity: float between 0 and 1, probability of connection between any two cells
        num_env_nodes: the number of environment nodes that the blanket node will interact with
        blanket_node: the index of the blanket node in the network
        """

        self.num_cells = num_cells
        self.connectivity = connectivity
        self.network = networkx.fast_gnp_random_graph(num_cells, connectivity)
        self.nodes = self.network.nodes

        self.set_states()


    def set_states(self):
        """The global state names for all the cell signals in the network, plus the environment nodes"""

        self.states = [
            x[::-1]
            for x in generate_binary_numbers(self.num_cells, 2**self.num_cells)
        ]


