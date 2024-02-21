# %%
import pathlib
import sys
import os

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
from networks.network import Network
from stemai.cells.tmaze_cell import TMazeCell


class TMazeNetwork(Network):
    """A network object representing a network of external cells"""

    def __init__(self, num_external_cells, connectivity, cells):

        self.color = "lightblue"

        super().__init__(num_external_cells, connectivity, cells)

    def create_agent(self, node, active_cell_indices, sensory_cell_indices, states) -> TMazeCell:
        """Creates an active inference agent for a given node in the network"""
        neighbors = list(networkx.neighbors(self.network, node))

        external_cell_indices = [
            int(node.replace("e", "")) for node in neighbors if node.startswith("e")
        ]

        agent = TMazeCell(node,
            neighbors,
            external_cell_indices,
            active_cell_indices,
            sensory_cell_indices,
            states)
        agent._action = self.actions[node]
        networkx.set_node_attributes(self.network, {node: agent}, "agent")

        return agent
