# %%
import pathlib
import sys
import os

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
from stemai.networks.network import Network
from stemai.cells.active_cell import ActiveCell


class ActiveNetwork(Network):
    """A network object representing a network of active cells"""

    def __init__(self, num_active_cells, connectivity, cells, celltype=ActiveCell):

        self.color = "indianred"  # for plotting

        super().__init__(num_active_cells, connectivity, cells, celltype)

    def create_agent(
        self, node, internal_and_sensory_cells, external_and_sensory_cells, states
    ) -> ActiveCell:
        """Creates an active inference agent for a given node in the network"""

        active_neighbors = list(networkx.neighbors(self.network, node))

        agent = self.celltype(
            node,
            active_neighbors + internal_and_sensory_cells,
            active_neighbors + external_and_sensory_cells,
            states,
        )
        agent._action = self.actions[node]
        networkx.set_node_attributes(self.network, {node: agent}, "agent")

        return agent
