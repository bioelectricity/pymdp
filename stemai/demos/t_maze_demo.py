#%%
import os
os.chdir("../")
from networks.internal_network import InternalNetwork
from networks.tmaze_network import TMazeNetwork
from stemai.networks import SensoryNetwork, ActiveNetwork, MarkovianSystem
from utils import draw_network
import networkx

num_internal_cells = 4 #arbitrary choice 

num_active_cells = 2 #four possible actions: 00,01,10,11
num_sensory_cells = 5 #24 possible actions (this includes 32 possibilities, but 6 of them will never occur from env)
num_external_cells = 1 #externall cell act function is just the tmaze step function given the action from the active cells 

internal_cells = [f"i{i}" for i in range(num_internal_cells)]

sensory_cells = [f"s{i}" for i in range(num_internal_cells, num_internal_cells + num_sensory_cells)]

active_cells = [
    f"a{i}"
    for i in range(
        num_internal_cells + num_sensory_cells,
        num_internal_cells + num_sensory_cells + num_active_cells,
    )
]

external_cells = [
    f"e{i}"
    for i in range(
        num_internal_cells + num_sensory_cells + num_active_cells,
        num_internal_cells + num_sensory_cells + num_active_cells + num_external_cells,
    )
]

print(f"internal_cells: {internal_cells}")
print(f"sensory_cells: {sensory_cells}")
print(f"active_cells: {active_cells}")
print(f"external_cells: {external_cells}")

internal_network = InternalNetwork(num_internal_cells, 1, internal_cells)

sensory_network = SensoryNetwork(
    num_sensory_cells,
    1,
    sensory_cells,
)

active_network = ActiveNetwork(num_active_cells, 1, active_cells)

tmaze = TMazeNetwork(num_external_cells, 1, external_cells)

colors = {}
for network in [internal_network, sensory_network, active_network, tmaze]:
    for node in network.network.nodes:
        colors[node] = network.color


system = MarkovianSystem(internal_network, tmaze, sensory_network, active_network)

pos = networkx.spring_layout(system.system)
images = []
import matplotlib.pyplot as plt

filenames = []
import imageio
import os

for t in range(50):
    system.step(logging=False)

    temp_file_name = draw_network(
        system.system,
        colors,
        t=t,
        title="System Network",
        pos=pos,
        _draw_neighboring_pairs=True,
        save=True,
    )

    images.append(imageio.imread(temp_file_name))
    filenames.append(temp_file_name)
    plt.close()


# Create a GIF from the images
gif_path = f"tmaze-simulation:{num_internal_cells}-{num_external_cells}.gif"
imageio.mimsave(gif_path, images, fps=1)


# Delete the temporary image files after creating the GIF
for temp_file_name in filenames:
    os.remove(temp_file_name)
