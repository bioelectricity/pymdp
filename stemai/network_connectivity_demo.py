#%%

from networks.generative_model import GenerativeModel
from utils import draw_network
from tests.disconnecting import *
from tests.connecting import *

num_agents = 10
connectivity = 0.5
generative_model = GenerativeModel(num_agents, connectivity)

original_network = generative_model.network.copy()
draw_network(generative_model.network, "Original network")
#test disconnecting 
nodes = generative_model.network.nodes
node1 = nodes[0]
original_B = node1['agent'].B


node1_neighbors = node1["agent"].neighbors.copy()
print(f"Disconnecting node 0 from node {node1_neighbors[0]}")
generative_model.disconnect_cells(0, node1_neighbors[0])

new_node = generative_model.network.nodes[0]
draw_network(generative_model.network, f"Disconnected node 0 from node {node1_neighbors[0]}")
disconnected_B = new_node['agent'].B

test_disconnecting_B(original_B, disconnected_B)

print(f"Reconnecting node 0 to node {node1_neighbors[0]}")

generative_model.connect_cells(0, node1_neighbors[0])
draw_network(generative_model.network, f"Reconnected node 0 to node {node1_neighbors[0]}")
reconnected_B = new_node['agent'].B
test_connecting_B(disconnected_B, reconnected_B)