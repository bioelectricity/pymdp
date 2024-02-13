# %%

from stemai.networks.agent_network import GenerativeModel
from utils import draw_network
from tests.disconnecting import *
from tests.connecting import *

num_agents = 8
connectivity = 0.6

generative_model = GenerativeModel(num_agents, connectivity)

original_network = generative_model.network.copy()
draw_network(generative_model.network, "Original network")
# test disconnecting
nodes = generative_model.network.nodes
node1 = nodes[0]
original_B = node1["agent"].B


node1_neighbors = node1["agent"].neighbors.copy()
print(f"Disconnecting node 0 from node {node1_neighbors[0]}")
generative_model.disconnect_cells(0, node1_neighbors[0])

new_node = generative_model.network.nodes[0]
draw_network(generative_model.network, f"Disconnected node 0 from node {node1_neighbors[0]}")
disconnected_B = new_node["agent"].B

test_disconnecting_B(original_B, disconnected_B)

print(f"Reconnecting node 0 to node {node1_neighbors[0]}")

generative_model.connect_cells(0, node1_neighbors[0])
draw_network(generative_model.network, f"Reconnected node 0 to node {node1_neighbors[0]}")
reconnected_B = new_node["agent"].B
test_connecting_B(disconnected_B, reconnected_B)


print("Killing cell 0")

generative_model.kill_cell(0)
draw_network(generative_model.network, "Killed cell 0")

print(f"Dividing Cell 1 with parent connection")
generative_model.divide_cell(1)
draw_network(generative_model.network, "Divided cell 1 - only connected to parent")


print(f"Dividing Cell 1 with all neighbor connection")
generative_model.divide_cell(2, connect_to_neighbors="all")
draw_network(generative_model.network, "Divided cell 2 - all parent neighbors")

print(f"Dividing Cell 3 with all half connection")
generative_model.divide_cell(3, connect_to_neighbors="half")
draw_network(generative_model.network, "Divided cell 3 - half of parent neighbors")
