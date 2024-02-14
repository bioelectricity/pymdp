#%%
from networks.system import System 
from networks.internal_network import InternalNetwork
from networks.external_network import ExternalNetwork
from networks.sensory_network import SensoryNetwork
from networks.active_network import ActiveNetwork
from utils import draw_network

num_internal_cells = 2
num_external_cells = 3

num_active_cells = 1
num_sensory_cells = 1


internal_cells = [f"i{i}" for i in range(num_internal_cells)]


sensory_cells = [f"s{i}" for i in range(num_sensory_cells)]

active_cells = [f"a{i}" for i in range(num_active_cells)]

external_cells = [f"e{i}" for i in range(num_external_cells)]

internal_network = InternalNetwork(num_internal_cells, 1, internal_cells)

sensory_network = SensoryNetwork(num_sensory_cells, 1, sensory_cells,)

active_network = ActiveNetwork(num_active_cells, 1, active_cells)

external_network = ExternalNetwork(num_external_cells, 1, external_cells)

colors = {}
for network in [internal_network, sensory_network, active_network, external_network]:
    for node in network.network.nodes:
        colors[node] = network.color

draw_network(internal_network.network, colors, title = "Internal Network")
draw_network(sensory_network.network, colors,title = "Sensory Network")
draw_network(active_network.network, colors,title = "Active Network")
draw_network(external_network.network, colors,title = "External Network")


system = System(internal_network, external_network, sensory_network, active_network)
draw_network(system.system, colors, title = "System Network")
system.step(logging = True)

# %%
