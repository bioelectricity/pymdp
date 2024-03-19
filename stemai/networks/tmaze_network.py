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
from stemai.networks import MarkovianSystem


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

        agent = TMazeCell(
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


class TMazeSystem(MarkovianSystem):
    """A class representing a system of interacting networks"""

    def __init__(
        self,
        internal_network: Network,
        external_network: Network,
        sensory_network: Network,
        active_network: Network,
    ):
        """
        internal_network: a Network of internal cells
        external_network: a Network of external cells
        sensory_network: a Network of sensory cells
        active_network: a Network of active cells

        This class will compose the networks into one big system-level network
        and is responsible for forming connections between internal, sensory, active and external cells
        within each network

        """

        super().__init__(internal_network, external_network, sensory_network, active_network)

    def external_act(self, node, logging=False):

        external_neighbors = list(networkx.neighbors(self.external_network.network, node))
        external_nodes = [self.external_network.network.nodes[node] for node in external_neighbors]

        incoming_nodes = external_neighbors + list(self.active_network.nodes)
        outgoing_nodes = external_nodes + [
            self.sensory_network.network.nodes[node] for node in self.sensory_network.nodes
        ]

        external_agent = self.external_network.nodes[node]["agent"]

        signals = [external_agent.actions_received[i] for i in incoming_nodes]
        if logging:
            print(f"Signal to external agent: {signals}")

        external_obs = external_agent.state_signal_to_index(signals)
        if logging:
            print(f"External observation: {external_obs}")

        action_string, reward, location, cue = external_agent.act(
            external_obs, self.in_consistent_interval
        )
        if int(reward) == 1 or (
            self.in_consistent_interval and self.reward_interval < 10
        ):  # if we got rewarded and we are still in the reward interval
            self.in_consistent_interval = True
            print("IN REWARD INTERVAL")
            self.reward_interval += 1
            print(f"Reward interval: {self.reward_interval}")
        if (
            self.reward_interval >= 10 and int(reward) != 1
        ):  # if we hit the end of the reward interval
            print(f"Received reward: {reward}, leaving reward interval")
            self.in_consistent_interval = False
            self.reward_interval = 0
        if logging:
            print(f"External action: {action_string}")
        self.update_observations(node, action_string, outgoing_nodes)

        return reward, self.in_consistent_interval, location, cue

    def step(self, logging=False):

        # first : we take the external observation, and we pass it to the sensory network
        accumulate = not self.in_consistent_interval

        # first the sensory cells act in response to the previous external observation
        for sensory_node in self.sensory_network.nodes:
            self.sensory_act(sensory_node, update=False, accumulate=accumulate, logging=logging)

        # then, the internal cells act
        for internal_node in self.internal_network.nodes:
            self.internal_act(internal_node, update=False, accumulate=accumulate, logging=logging)

        # then, the active cells act
        for active_node in self.active_network.nodes:
            action = self.active_act(
                active_node, update=False, accumulate=accumulate, logging=logging
            )

        # finally, the external nodes act
        for external_node in self.external_network.nodes:
            reward, self.in_consistent_interval, location, cue = self.external_act(
                external_node, logging=logging
            )

        self.t += 1
        return reward, location, cue
