# %%
import pathlib
import sys
import os

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
from networks.network import Network
from stemai.cells.sensory_cell import SensoryCell


class SensoryNetwork(Network):

    def __init__(self, num_active_cells, connectivity, cells):
        self.color = "lightgrey"
        super().__init__(num_active_cells, connectivity, cells)

    def create_agent(
        self, node, external_and_active_cells, internal_and_active_cells, states
    ) -> SensoryCell:
        """Creates an active inference agent for a given node in the network"""
        sensory_neighbors = list(networkx.neighbors(self.network, node))

        agent = SensoryCell(
            node,
            sensory_neighbors + external_and_active_cells,
            sensory_neighbors + internal_and_active_cells,
            states,
        )
        agent._action = self.actions[node]
        networkx.set_node_attributes(self.network, {node: agent}, "agent")

        agent.actions_received = {n: 0 for n in sensory_neighbors}
        agent.actions_sent = {n: 0 for n in sensory_neighbors}
        return agent
