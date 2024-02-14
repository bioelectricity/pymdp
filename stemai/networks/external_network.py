# %%
import pathlib
import sys
import os

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
from networks.network import Network
from stemai.cells.external_cell import ExternalCell


class ExternalNetwork(Network):

    def __init__(self, num_external_cells, connectivity, cells):

        self.color = "blue"

        super().__init__(num_external_cells, connectivity, cells)

    def create_agent(self, node, sensory_cell_indices, active_cell_indices, states) -> ExternalCell:
        """Creates an active inference agent for a given node in the network"""
        neighbors = list(networkx.neighbors(self.network, node))

        agent = ExternalCell(
            node,
            neighbors,
            sensory_cell_indices,
            active_cell_indices,
            states,
        )
        agent._action = self.actions[node]
        networkx.set_node_attributes(self.network, {node: agent}, "agent")

        return agent

    def create_agents(self, sensory_cell_indices, active_cell_indices, external_and_blanket_states):
        """Creates active inference agents
        for each node in the network"""

        for node in self.network.nodes:
            self.create_agent(node, sensory_cell_indices, active_cell_indices, external_and_blanket_states)

        return self.network