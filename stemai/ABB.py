#%%
import pathlib
import sys
path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)
import networkx 
import numpy as np
from pymdp import utils 
from pymdp.agent import Agent
from utils import signals_to_index, extract_agent_action

class ABB:

    def __init__(self, num_cells, connectivity):
        """ We start assuming our ABB can be modeled as an Erdos Renyi graph 
        with num_cells nodes and connectivity probability connectivity. 
        
        See https://networkx.org/documentation/stable/reference/generators.html
        for other kinds of random graphs"""

        self.num_cells = num_cells
        self.connectivity = connectivity
        self.network = networkx.fast_gnp_random_graph(num_cells, connectivity)

        self.action = np.random.choice([0,1], size = num_cells)
        print(f"ABB action: {self.action}")
        self.create_agents()

    def build_generative_model(self, num_neighbors):
        num_factors = 1
        num_states = [2**(num_neighbors + 2)] #neighbors, self, and other agent 
        num_actions = [2**(num_neighbors + 2)] 
        num_obs = [2**(num_neighbors + 2)] 
        num_modalities = 1

        A = utils.obj_array(num_modalities)

        for i in range(num_modalities):
            A[i] = np.eye(num_obs[i])
        
        B = utils.obj_array(num_factors)
        
        for i in range(num_factors): #generate a randomized (deterministic) B

            B_i = np.zeros((num_states[i], num_states[i], num_actions[i]))
            for action in range(num_actions[i]):
                for state in range(num_states[i]):
                    random_state = np.random.choice(num_states[i])
                    B_i[random_state, state, action] = 1
            B[i] = B_i

        C = utils.obj_array(num_modalities)

        C[0] = np.zeros(num_obs[0])
        
        D = utils.obj_array(num_factors)
        D[0] = np.random.uniform(0,1,size =num_states[0])
        D[0] /= D[0].sum()

        return A, B, C, D


    def create_agents(self):

        agent_dict = {}

        for node in self.network.nodes:
            num_neighbors = len(list(networkx.neighbors(self.network, node)))
            print(f"Agent {node} has {num_neighbors} neighbors")
            A, B, C, D = self.build_generative_model(num_neighbors)
            agent = Agent(A=A, B=B, C=C, D=D)
            agent._action = 0
            agent.binary_action = 0
            agent_dict[node] = agent
        networkx.set_node_attributes(self.network, agent_dict, "agent")

        return self.network
    
    def generate_observations(self, obs, agent, neighbors):

        signals = list(self.action[neighbors]) # a list of zero or 1 for each neighbor
        signals.append(agent.binary_action) #self signal
        signals.append(obs) #other agent signal
        print(f"All signals: {signals}")
        index = signals_to_index(signals)
        #print(f"State index: {index}")
        return index

    
    def act(self, agent_observation):

        abb_action = []

        for node in self.network.nodes:
            agent = self.network.nodes[node]["agent"]
            neighbors = list(networkx.neighbors(self.network, node))

            obs = self.generate_observations(agent_observation,agent,neighbors)
            agent.infer_states([obs])
            agent.infer_policies()
            agent._action = int(agent.sample_action()[0])

            print(f"Agent action: {agent._action}")
            binary_action = extract_agent_action(agent._action, len(neighbors))
            agent.binary_action = binary_action
            assert binary_action in [0,1]
            print(f"Binary action: {binary_action}")
            abb_action.append(binary_action)
        print(f"ABB action: {abb_action}")
        self.action = np.array(abb_action)

        return abb_action


#%%

abb = ABB(3, 1)
networkx.draw(abb.network)
# %%
T = 50
import matplotlib.pyplot as plt

import imageio

# Initialize a list to store the images
images = []
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
    print(f"ABB action: {abb_action}")
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
    plt.close()

# Create a GIF from the images
gif_path = "simulation1.gif"
imageio.mimsave(gif_path, images, fps=1)

