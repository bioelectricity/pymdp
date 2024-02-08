#%%
from networks.generative_process import GenerativeProcess
import networkx 
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os

num_agents = 10
connectivity = 0.2
T = 50

abb = GenerativeProcess(num_agents, connectivity)
networkx.draw(abb.network)

# Initialize a list to store the images
images = []
filenames = []
# Generate fixed positions for the nodes
pos = networkx.spring_layout(abb.network)
# Use the 'pos' variable to ensure the positions of the nodes are fixed in the GIF

# Generate and save the network images for each timestep
for t in range(T):
    plt.figure(figsize=(10, 8))

    agent_observation = np.random.choice([0,1])
    # Add an extra node for displaying the agent_observation
    # Plot an additional circle for displaying the agent_observation
    observation_color = 'mediumseagreen' if agent_observation == 0 else 'lightblue'
    plt.scatter([1.1], [0.5], s=1000, c=observation_color)  # Position the observation circle slightly outside the main network
    plt.text(1.1, 0.5, 'Observation', horizontalalignment='center', verticalalignment='center')
    print("Acting...")
    abb_action = abb.act(agent_observation)
    #print(f"ABB action: {abb_action}")
    print()

    print("Drawing the network...")

    abb_agent_actions = abb.action 

    # Define two colors for the states 0 and 1
    colors = {0: 'mediumseagreen', 1: 'lightblue'}

    # Create a color map based on the agent actions
    color_map = [colors[action] for action in abb_agent_actions]

    # Draw the network with the specified node colors
    networkx.draw(abb.network, pos=pos,node_color=color_map, with_labels=True)
    plt.title(f"Simulation at timestep {t}")

    # Save the current figure to a temporary file and add it to the images list
    temp_file_name = f"temp_image_{t}.png"
    plt.savefig(temp_file_name)
    images.append(imageio.imread(temp_file_name))
    filenames.append(temp_file_name)
    plt.close()

# Create a GIF from the images
gif_path = f"simulation:{num_agents}-{connectivity}.gif"
imageio.mimsave(gif_path, images, fps=1)


# Delete the temporary image files after creating the GIF
for temp_file_name in filenames:
    os.remove(temp_file_name)


