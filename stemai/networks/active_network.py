# %%
import pathlib
import sys
import os

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + "/"
sys.path.append(module_path)
import networkx
from networks.network import Network
from stemai.cells.active_cell import ActiveCell


class ActiveNetwork(Network):

    def __init__(self, num_active_cells, connectivity, cells):

        self.color = "red"

        super().__init__(num_active_cells, connectivity, cells)

    def create_agent(self, node, internal_cells, external_cells, states) -> ActiveCell:
        """Creates an active inference agent for a given node in the network"""

        agent = ActiveCell(
            node,
            internal_cells,
            external_cells,
            states,
        )
        agent._action = self.actions[node]
        networkx.set_node_attributes(self.network, {node: agent}, "agent")

        return agent

    def create_agents(self, internal_cells, external_cells, internal_and_external_states):
        """Creates active inference agents
        for each node in the network"""

        for node in self.network.nodes:
            self.create_agent(node, internal_cells, external_cells, internal_and_external_states)

        return self.network