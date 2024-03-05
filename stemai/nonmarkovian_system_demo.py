# %%
from stemai.networks.nonmarkovian_system import NonMarkovianSystem
from networks.internal_network import InternalNetwork
from networks.external_network import ExternalNetwork
from utils import draw_network
import networkx

num_internal_cells = 2
num_external_cells = 3

internal_cells = [f"i{i}" for i in range(num_internal_cells)]

external_cells = [
    f"e{i}" for i in range(num_internal_cells, num_internal_cells + num_external_cells)
]

internal_network = InternalNetwork(num_internal_cells, 1, internal_cells)

external_cell_indices = list(range(num_internal_cells, num_internal_cells + num_external_cells))
external_network = ExternalNetwork(num_external_cells, 1, external_cells)

colors = {}
for network in [internal_network, external_network]:
    for node in network.network.nodes:
        colors[node] = network.color


system = NonMarkovianSystem(internal_network, external_network)
pos = networkx.spring_layout(system.system)
images = []
import matplotlib.pyplot as plt

filenames = []
import imageio
import os

for t in range(50):
    system.step(logging=True)

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
gif_path = f"inactive-simulation:{num_internal_cells}-{num_external_cells}.gif"
imageio.mimsave(gif_path, images, fps=1)


# Delete the temporary image files after creating the GIF
for temp_file_name in filenames:
    os.remove(temp_file_name)
