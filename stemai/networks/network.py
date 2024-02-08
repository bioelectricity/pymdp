#%%
import pathlib
import sys
import os 
path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)
import networkx 
import numpy as np
from utils import generate_binary_numbers

class Network:

    def __init__(self, num_cells, connectivity, initial_action = None):

        self.num_cells = num_cells
        self.connectivity = connectivity
        self.network = networkx.fast_gnp_random_graph(num_cells, connectivity)

        if initial_action is None:
            initial_action = np.random.choice([0,1], size = num_cells)
        self.actions = initial_action
        self.global_states = [x[::-1] for x in generate_binary_numbers(num_cells+1, 2**(num_cells+1))]

        print(f"Global_states : {self.global_states}")

    
    def generate_observations(self, obs, agent, neighbors):

        signals = list(self.actions[neighbors]) + [agent._action, obs]# a list of zero or 1 for each neighbor
        print(f"All signals: {signals}")
        index = agent.signal_to_index(signals)
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

            #print(f"Agent action: {agent._action}")
            assert agent._action in [0,1]
            abb_action.append(agent._action)
        #print(f"ABB action: {abb_action}")
        self.action = np.array(abb_action)

        return abb_action

    def create_agents(self):
        """This function creates agents and will differ depending on 
        whether the network is a generative process or a generative model"""

        pass
#%%
