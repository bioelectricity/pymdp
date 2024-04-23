# %%
import pathlib
import sys
import os

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
from stemai.networks.network import Network
from stemai.cells.external_cell import ExternalCell


class ExternalNetwork(Network):
    """A network object representing a network of external cells"""

    def __init__(self, num_external_cells, connectivity, cells, celltype, file):

        self.color = "lightblue"

        super().__init__(num_external_cells, connectivity, cells, celltype, file)

    def create_agent(self, node, active_cell_indices, sensory_cell_indices, states) -> ExternalCell:
        """Creates an active inference agent for a given node in the network"""
        neighbors = list(networkx.neighbors(self.network, node))

        external_cell_indices = [
            int(node.replace("e", "")) for node in neighbors if node.startswith("e")
        ]

        agent = self.celltype(
            node,
            neighbors,
            external_cell_indices,
            active_cell_indices,
            sensory_cell_indices,
            states,
        )
        agent._action = self.actions[node]
        networkx.set_node_attributes(self.network, {node: agent}, "agent")

        return agent
