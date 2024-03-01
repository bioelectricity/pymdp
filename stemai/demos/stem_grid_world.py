# %%
import os 
import matplotlib.pyplot as plt 
import imageio
os.chdir('/Users/daphne/Desktop/stemai/pymdp')

from stemai.networks.markovian_system import MarkovianSystem
from stemai.networks.internal_network import InternalNetwork
from stemai.networks.external_network import ExternalNetwork
from stemai.networks.sensory_network import SensoryNetwork
from stemai.networks.active_network import ActiveNetwork
from stemai.cells.gridworld_cell import GridWorldCell
from stemai.networks.neuronal_cell_system import System

from stemai.utils import draw_network
import networkx
import numpy as np

num_internal_cells = 5 #lars thinks 20, start fuly connected 

#CONNECTIVITY FOR NEURONAL CELLS 
#likelihood precision updating
#then you prune connections 

#CONNECTIVITY FOR STEM CELLS 
#start with fewer cells (3 or 4, fully connected)



num_external_cells = 1

num_active_cells = 2
num_sensory_cells = 1

#Define the grid world with a desired location 

REWARD_LOCATION = (0,0)
AGENT_LOCATION = (4,4)
GRID_SIZE = 10

internal_cells = [f"i{i}" for i in range(num_internal_cells)]


sensory_cells = [f"s{i}" for i in range(num_internal_cells, num_internal_cells + num_sensory_cells)]

active_cells = [
    f"a{i}"
    for i in range(
        num_internal_cells + num_sensory_cells,
        num_internal_cells + num_sensory_cells + num_active_cells,
    )
]

external_node_labels = [f"e{i}" for i in range(num_external_cells)]

print(f"internal_cells: {internal_cells}")
print(f"sensory_cells: {sensory_cells}")
print(f"active_cells: {active_cells}")
print(f"external_cells: {external_node_labels}")

internal_network = InternalNetwork(num_internal_cells, 1, internal_cells)

sensory_network = SensoryNetwork(
    num_sensory_cells,
    1,
    sensory_cells,
)

active_network = ActiveNetwork(num_active_cells, 1, active_cells)

external_network = ExternalNetwork(num_external_cells, 1, external_node_labels, celltype = GridWorldCell)
system = System(internal_network, external_network, sensory_network, active_network)

for node in external_network.network.nodes:
    node = external_network.network.nodes[node]
    node["agent"].reward_location = REWARD_LOCATION

    node["agent"].agent_location = AGENT_LOCATION

    node["agent"].grid_size = GRID_SIZE

colors = {}
for network in [internal_network, sensory_network, active_network, external_network]:
    for node in network.network.nodes:
        colors[node] = network.color


num_trials = 10
import numpy as np
grid = np.zeros((GRID_SIZE, GRID_SIZE))
grid[REWARD_LOCATION] = 1
grid[AGENT_LOCATION] = 2

agent_location = AGENT_LOCATION

plt.imshow(grid)
plt.title("Initial Grid")
plt.show()



grid_images = []
grid_filenames = []
network_filenames = []
network_images = []
trial = 0

time_to_reward = []
overall_t = 0

all_reward_locations = [(0,0),(9,9),(0,9),(9,0)]

while agent_location != REWARD_LOCATION and trial < num_trials:
    print(f"Trial {trial}, T :{system.t}")
    print()

    action, agent_location, distance, probabilities = system.step(logging=False)
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    print()
    REWARD_LOCATION = all_reward_locations[trial % 4]
    grid[REWARD_LOCATION] = 1
    grid[system.agent_location] = 2


    #fig = plt.figure(figsize = (6,6))
    plt.title(f"Trial: {trial}, timestep :{system.t}, distance_to_reward: {system.distance_to_reward}, signal: {system.external_signal}, probabilities: {(round(system.probabilities[0],2), round(system.probabilities[1],2))}", color='black')
    # if system.t % 10 == 0:
    #     print(f"Updating gamma_A")
    #     system.update_gamma_A()
    #     plt.text(2, 2, "UPDATING GAMMA_A", fontsize=12, color='white')
    plt.imshow(grid)

    plt.savefig(f"grid_at_{trial}_{system.t}.png", facecolor='w')
    image = imageio.imread(f"grid_at_{trial}_{system.t}.png")
    if len(grid_images) > 0 and image.shape == grid_images[-1].shape:
        grid_images.append(image)
        grid_filenames.append(f"grid_at_{trial}_{system.t}.png")
    #plt.show()
    plt.clf()


    if agent_location == REWARD_LOCATION or system.t > 300:
        time_to_reward.append(system.t)
        system.t = 0
        system.agent_location = (0,0)
        system._reset()
        trial += 1
        agent_location = (0,0)


    temp_file_name = draw_network(
    system.system,
    colors,
    t=f"{trial}_{system.t}",
    title=f"Trial: {trial}, timestep :{system.t}, distance_to_reward: {system.distance_to_reward}, signal: {system.external_signal}, probabilities: {(round(system.probabilities[0],2), round(system.probabilities[1],2))}",
    pos=pos,
    _draw_neighboring_pairs=True,
    save=True,
    show=False
    )
    plt.title(f"Trial: {trial}, timestep :{system.t}, distance_to_reward: {system.distance_to_reward}, signal: {system.external_signal}, probabilities: {(round(system.probabilities[0],2), round(system.probabilities[1],2))}", color='black')
    plt.clf()
    network_images.append(imageio.imread(temp_file_name))
    network_filenames.append(temp_file_name)
    overall_t += 1
    if overall_t % 100 == 0:
        print(f"Pruning")
        system.prune()





print(f"Generating GIFS")
# Create a GIF from the images
gif_path = f"gridworld-network-simulation.gif"
imageio.mimsave(gif_path, network_images, fps=1)


for temp_file_name in network_filenames:
    os.remove(temp_file_name)

# Create a GIF from the images
gif_path = f"gridworld-grid-simulation-{num_internal_cells}.gif"
imageio.mimsave(gif_path, grid_images, fps=1)


# Delete the temporary image files after creating the GIF
for temp_file_name in grid_filenames:
    os.remove(temp_file_name)


plt.plot(time_to_reward)
plt.xlabel("Trials")
plt.ylabel("Number of timesteps to reward")
plt.title(f"Number of timesteps to reward over trials, num_cells = {num_internal_cells}")
plt.savefig(f"time_to_reward_{num_internal_cells}.png")
plt.show()