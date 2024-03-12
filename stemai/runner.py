from stemai.networks.neuronal_network import NeuronalNetwork
from stemai.cells.gridworld_cell import GridWorldCell
from stemai.networks.external_network import ExternalNetwork
from stemai.networks.neuronal_cell_system import System
from stemai.utils import draw_network

import numpy as np
import matplotlib.pyplot as plt 
import networkx 
import os 
import imageio

class Runner:

    def __init__(self, num_trials, num_internal_cells, num_external_cells, num_active_cells, num_sensory_cells, internal_connectivity, sensory_connectivity, external_connectivity, reward_location, agent_location, grid_size, active_connectivity_proportion, sensory_connectivity_proportion, action_time_threshold, precision_threshold, precision_update_frequency, index, logging = True):

        if not os.path.exists(f'out/{index}/0/networks'): 
            os.makedirs(f'out/{index}/0/networks')
            os.makedirs(f'out/{index}/0/grids')
        self.num_trials = num_trials
        self.num_internal_cells = num_internal_cells
        self.num_external_cells = num_external_cells
        self.num_active_cells = num_active_cells
        self.num_sensory_cells = num_sensory_cells
        self.internal_connectivity = internal_connectivity
        self.active_connectivity = 1
        self.sensory_connectivity = sensory_connectivity
        self.external_connectivity = external_connectivity
        self.reward_location = reward_location
        self.agent_location = agent_location
        self.grid_size = grid_size
        self.active_connectivity_proportion = active_connectivity_proportion
        self.sensory_connectivity_proportion = sensory_connectivity_proportion
        self.action_time_threshold = action_time_threshold
        self.precision_threshold = precision_threshold
        self.precision_update_frequency = precision_update_frequency
        self.logging = logging
        self.index = index

        import yaml

        params = {
            'num_trials': self.num_trials,
            'num_internal_cells': self.num_internal_cells,
            'num_external_cells': self.num_external_cells,
            'num_active_cells': self.num_active_cells,
            'num_sensory_cells': self.num_sensory_cells,
            'internal_connectivity': self.internal_connectivity,
            'sensory_connectivity': self.sensory_connectivity,
            'external_connectivity': self.external_connectivity,
            'reward_location': self.reward_location,
            'agent_location': self.agent_location,
            'grid_size': self.grid_size,
            'active_connectivity_proportion': self.active_connectivity_proportion,
            'sensory_connectivity_proportion': self.sensory_connectivity_proportion,
            'action_time_threshold': self.action_time_threshold,
            'precision_threshold': self.precision_threshold,
            'precision_update_frequency': self.precision_update_frequency,
            'logging': self.logging
        }

        with open(f'out/{self.index}/params.yaml', 'w') as file:
            yaml.dump(params, file)


        self.construct_system()
        self.set_locations()

    def construct_system(self):

        internal_node_labels = [f"i{i}" for i in range(self.num_internal_cells)]

        active_node_labels = [f"a{i}" for i in range(self.num_active_cells)]
        sensory_node_labels = [f"s{i}" for i in range(self.num_sensory_cells)]

        external_node_labels = [f"e{i}" for i in range(self.num_external_cells)]

        self.internal_network = NeuronalNetwork(self.num_internal_cells, self.internal_connectivity, node_labels=internal_node_labels, color = "mediumseagreen")

        print("Created internal network")

        self.active_network = NeuronalNetwork(self.num_active_cells, self.active_connectivity, node_labels=active_node_labels, color = "indianred")


        self.sensory_network = NeuronalNetwork(self.num_sensory_cells, self.sensory_connectivity, node_labels=sensory_node_labels, color = "lightgrey")

        self.external_network = ExternalNetwork(self.num_external_cells, self.external_connectivity, external_node_labels, celltype = GridWorldCell)

        print("Created all networks")
        #now connect them together 
        # compose all the networks into one system network
        self.system = System(self.internal_network, self.external_network, self.sensory_network, self.active_network, active_connectivity_proportion=self.active_connectivity_proportion, sensory_connectivity_proportion=self.sensory_connectivity_proportion, action_time_horizon=self.action_time_threshold, precision_threshold=self.precision_threshold)

        #set the reward states of external cells 
        for node in self.external_network.network.nodes:
            node = self.external_network.network.nodes[node]
            node["agent"].reward_location = self.reward_location

            node["agent"].agent_location = self.agent_location

            node["agent"].grid_size = self.grid_size

            
        self.system.reward_location = self.reward_location
        self.system.agent_location = self.agent_location
        self.system.distance_to_reward = abs(self.system.agent_location[0] - self.system.reward_location[0]) + abs(self.system.agent_location[1] - self.system.reward_location[1])

        self.pos = networkx.spring_layout(self.system.system)

        self.colors = {}
        for network in [self.internal_network, self.sensory_network, self.active_network, self.external_network]:
            for node in network.network.nodes:
                self.colors[node] = network.color

    def construct_grid(self):

        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.reward_location] = 1
        grid[self.agent_location] = 2

        return grid

    def set_locations(self):
        self.all_reward_locations = [self.reward_location] * self.num_trials
        self.all_agent_locations = [self.agent_location] * self.num_trials
        self.system.update_grid_locations(self.all_reward_locations[0], self.all_agent_locations[0])

    def draw(self, trial, t):
        title=f"Trial: {trial}, timestep :{t}, distance_to_reward: {self.system.distance_to_reward}, signal: {self.system.external_signal}, probabilities: {(round(self.system.probabilities[0],2), round(self.system.probabilities[1],2))}",

        draw_network(
        self.system.system,
        self.colors,
        t=f"{trial}_{t}",
        title=title,
        pos=self.pos,
        _draw_neighboring_pairs=True,
        save=True,
        show=False,
        temp_file_name=f"out/{self.index}/{trial}/networks/{t}.png"
        )
        plt.title(title, color='black')
        plt.clf()

    def plot_grids_for_trial(self, trial):

        for t, grid in enumerate(self.grids_over_time):
            plt.imshow(grid)
            title=f"Trial: {trial}, timestep :{t}, distance_to_reward: {self.distances_over_time[trial][t]}, signal: {self.signals_over_time[trial][t]}"
            if t in self.gamma_update_times:
                plt.text(0, 0, "Gamma update", color='white')
            plt.savefig(f"out/{self.index}/{trial}/grids/{t}.png", facecolor='w')
            plt.title(title, color='black')
            plt.clf()
        self.grids_over_time = []
        self.gamma_update_times = []
    
    def plot_time_to_reward(self):
        plt.plot(self.time_to_reward_per_trial)
        plt.xlabel("Trials")
        plt.ylabel("Number of timesteps to reward")
        plt.title(f"Number of timesteps to reward over trials")
        plt.savefig(f"out/{self.index}/time_to_reward.png")
        plt.clf()

    def plot_distances_over_time(self):
        average = []
        for trial, distances in self.distances_over_time.item():
            plt.plot(distances)
            average.append(np.mean(distances))
            plt.xlabel("Timesteps")
            plt.ylabel("Distance to reward")
            plt.title(f"Distance to reward over time, trial: {trial}")
            plt.savefig(f"out/{self.index}/{trial}/distances_over_time.png")
            plt.clf()
        plt.plot(average)
        plt.xlabel("Trials")
        plt.ylabel("Average distance to reward")
        plt.title(f"Average distance to reward over trials")
        plt.savefig(f"out/{self.index}/average_distance_to_reward.png")
        plt.clf()

    def write_data(self):
        with open(f"out/{self.index}/time_to_reward.txt", "w") as file:
            for time in self.time_to_reward_per_trial:
                file.write(f"{time}\n")

        with open(f"out/{self.index}/distances_over_time.txt", "w") as file:
            for trial, distances in self.distances_over_time.items():
                file.write(f"{trial}: {distances}\n")

        with open(f"out/{self.index}/connectivities.txt", "w") as file:
            for trial, connectivity in self.connectivities.items():
                file.write(f"{trial}: {connectivity}\n")

    def generate_gifs(self):
        network_images = []
        grid_images = []
        for i in range(self.num_trials):            
            network_fns = [f"out/{self.index}/{i}/networks/{j}.png" for j in range(len([x for x in os.listdir(f"out/{self.index}/{i}/networks/") if x.endswith('.png')]))]
            network_images+= [imageio.imread(f) for f in network_fns ]
            grid_fns = [f"out/{self.index}/{i}/grids/{j}.png" for j in range(len([x for x in os.listdir(f"out/{self.index}/{i}/grids/") if x.endswith('.png')]))]
            grid_images += [imageio.imread(f) for f in grid_fns]
        gif_path = f"out/{self.index}/network-simulation.gif"
        imageio.mimsave(gif_path, network_images, fps=5)
        # for temp_file in network_fns:
        #     os.remove(temp_file)
        gif_path = f"out/{self.index}/grid-simulation.gif"
        imageio.mimsave(gif_path, grid_images, fps=5)
        # for temp_file in grid_fns:
        #     os.remove(temp_file)


    def run(self):
        agent_location = self.system.agent_location
        trial = 0

        self.distances_over_time = {}
        self.signals_over_time = {}
        self.time_to_reward_per_trial = []
        self.connectivities = {}
        self.grids_over_time= []
        self.gamma_update_times = []
        self.distances_over_time[trial] = []
        self.signals_over_time[trial] = []
        self.connectivities[trial] = []
        while agent_location != self.system.reward_location and trial < self.num_trials:

            if self.logging: print(f"Trial: {trial}, T :{self.system.t}" + "\n")
            grid = self.construct_grid()
            self.grids_over_time.append(grid)
            _, agent_location, distance, _ = self.system.step(logging = self.logging)
            self.agent_location = agent_location
            self.signals_over_time[trial].append(self.system.external_signal)

            self.distances_over_time[trial].append(distance)
            total_edges = self.system.system.number_of_edges()
            self.connectivities[trial].append(total_edges)
            
            if self.system.t > 0 and self.system.t % self.precision_update_frequency == 0:
                self.system.update_gamma_A()
                self.gamma_update_times.append(self.system.t)

            if agent_location == self.system.reward_location:
                self.time_to_reward_per_trial.append(self.system.t)
                self.plot_grids_for_trial(trial)

                self.system._reset()

                self.system.renormalize_precisions()

                self.system.prune()
                trial += 1

                self.grids_over_time= []
                self.gamma_update_times = []
                self.distances_over_time[trial] = []
                self.signals_over_time[trial] = []
                self.connectivities[trial] = []

                os.makedirs(f"out/{self.index}/{trial}/networks")
                os.makedirs(f"out/{self.index}/{trial}/grids")

                reward_location = self.all_reward_locations[trial]
                agent_location = self.all_agent_locations[trial]
                self.system.update_grid_locations(reward_location, agent_location)

            
        
            self.draw(trial, self.system.t)

        self.plot_time_to_reward()
        self.plot_distances_over_time()
        self.generate_gifs()






            

