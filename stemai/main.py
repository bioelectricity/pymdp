#%%
from networks.generative_process import GenerativeProcess
import networkx 
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os

num_agents = 8
connectivity = 0.4
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
    abb.act(agent_observation)
    #print(f"ABB action: {abb_action}")
    print()

    print("Drawing the network...")

    # Define two colors for the states 0 and 1
    colors = {0: 'mediumseagreen', 1: 'lightblue'}

    actions_received = {}

    for node in abb.network.nodes:
        agent = abb.network.nodes[node]["agent"]
        actions_received[node] = {i : agent.actions_received[i] for i in agent.global_neighbor_indices}

    # color_map = [colors[action] for action in abb_agent_actions]
    print(f"Actions received: {actions_received}")
    # # Draw the network with the specified node colors
    networkx.draw(abb.network, pos=pos,node_color='black', with_labels=True, edge_color = 'white')

    neighboring_pairs = []
    for node in abb.network.nodes:
        neighbors = list(abb.network.neighbors(node))
        for neighbor in neighbors:
            if (neighbor, node) not in neighboring_pairs:  # Ensure each pair is added only once
                neighboring_pairs.append((node, neighbor))
    # To avoid overwriting, we'll draw bidirectional edges with distinct colors for each direction
    # To optimize the iteration, we can draw edges in pairs (receiver to sender and sender to receiver) in one go
    # This avoids iterating over the network twice for each pair of agents
    for (receiver, sender) in neighboring_pairs:
               # Define edge color based on the action received
        edge_color_received = 'mediumseagreen' if int(actions_received[receiver][sender]) == 0 else 'lightblue'
        # Define edge color based on the action sent, which requires accessing the sender's actions_received
        edge_color_sent = 'mediumseagreen' if int(actions_received[sender][receiver]) == 0 else 'lightblue'
        # Draw the edge for the action received
        networkx.draw_networkx_edges(abb.network, pos,
                                     edgelist=[(sender, receiver)],
                                     edge_color=edge_color_received,
                                     arrows=True,
                                     arrowstyle= '-|>',
                                     style='dashed')  # Dashed line for received action
        # Draw the edge for the action sent, slightly offset to avoid overlap
        networkx.draw_networkx_edges(abb.network, pos,
                                     edgelist=[(receiver, sender)],
                                     edge_color=edge_color_sent,
                                     arrows=True,
                                     arrowstyle= '-|>',
                                     connectionstyle='arc3,rad=0.3')  # Curved line for sent action
    
    # Create a color map based on the agent actions
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


