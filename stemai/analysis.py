
#%%
import os 
import matplotlib.pyplot as plt 
import pickle
import numpy as np
import imageio
import networkx
os.chdir('../')
#plots to make 
from stemai.demos.ngw_params import all_parameter_combinations, params_to_sweep, param_to_index_mapping
from stemai.utils import draw_network


class TrialAnalysis:

    def __init__(
        self,
        num_trials,
        num_internal_cells,
        num_external_cells,
        num_active_cells,
        num_sensory_cells,
        internal_connectivity,
        sensory_connectivity,
        external_connectivity,
        reward_location,
        agent_location,
        grid_size,
        active_connectivity_proportion,
        sensory_connectivity_proportion,
        action_time_threshold,
        precision_threshold,
        precision_update_frequency,
        prune_connections, 
        add_connections,
        index,
        dir = "out",
        logging=False,
    ):
        self.num_trials = num_trials - 1
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
        self.prune_connections = prune_connections
        self.add_connections = add_connections

        self.path = f"{dir}/{index}"

        self.trial_paths = [f"{self.path}/{i}" for i in range(self.num_trials) if os.path.exists(f"{self.path}/{i}")]


    def generate_network_gif(self):
   
        network_images = []
        network_files = []
        for trial_idx, trial_path in enumerate(self.trial_paths):
            network_files = [f"{trial_path}/networks/" + f for f in os.listdir(f"{trial_path}/networks") if f.endswith('.pickle')]


            first_network = network_files[0]
            with open(first_network, "rb") as f:
                network = pickle.load(f)
            pos = networkx.spring_layout(network)
            for network_file in network_files:
                with open(network_file, "rb") as f:
                    network = pickle.load(f)

                for t in range(self.num_trials):
                    title = (
                        f"Trial: {trial_idx}, timestep :{t}",
                    )
                    fn = f"{trial_path}/networks/network_{t}.png"
                    print(f"Drawing network and saving to {fn}")

                    draw_network(
                        network,
                        t=f"{trial_idx}_{t}",
                        title=title,
                        pos=pos,
                        _draw_neighboring_pairs=True,
                        save=True,
                        show=False,
                        temp_file_name=fn,
                    )
                    network_files.append(fn)
                    network_images.append(imageio.imread(fn))
        print(f"Saving gif to : {self.path}/network-simulation.gif")
        gif_path = f"{self.path}/network-simulation.gif"
        imageio.mimsave(gif_path, network_images, fps=5)
        for temp_file in network_files:
            os.remove(temp_file)

    def generate_grid_gif(self):
        grids = np.load(self.path + "/grids.npy")
        grid_images = []
        for i in range(len(grids)):
            plt.imshow(grids[i])
            fn = f"{self.path}/grids/grid_{i}.png"
            plt.savefig(fn)
            grid_images.append(imageio.imread(f"{self.path}/grids/grid_{i}.png"))
            os.remove(fn)
        gif_path = f"{self.path}/grid-simulation.gif"
        print(f"Saving gif to : {self.path}/grid-simulation.gif")

        imageio.mimsave(gif_path, grid_images, fps=5)

    def plot_time_to_reward(self):
        fn = f"{self.path}/time_to_reward.txt"
        time_to_reward = []
        with open(fn, "r") as f:
            lines = f.readlines()
            for line in lines:
                time_to_reward.append(float(line.replace("\n","")))
        plt.plot(time_to_reward)
        plt.xlabel("Trials")
        plt.ylabel("Number of timesteps to reward")
        plt.title(f"Number of timesteps to reward over trials")
        plt.savefig(f"{self.path}/time_to_reward.png")
        plt.clf()

        self.time_to_reward = time_to_reward

    def plot_distances_over_time(self):
        fn = f"{self.path}/distances_over_time.txt"
        distances_over_time = {}
        with open(fn, "r") as f:
            lines = f.readlines()
            for trial, line in enumerate(lines):
                distances_over_time[trial] = [float(l) for l in line.replace("\n","").replace(']','').split(",")[1:]]

        average = []
        for trial, distances in distances_over_time.items():
            plt.plot(distances)
            average.append(np.mean(distances))
            plt.xlabel("Timesteps")
            plt.ylabel("Distance to reward")
            plt.title(f"Distance to reward over time, trial: {trial}")
            plt.savefig(f"{self.path}/{trial}/distances_over_time.png")
            plt.clf()
        plt.plot(average)
        plt.xlabel("Trials")
        plt.ylabel("Average distance to reward")
        plt.title(f"Average distance to reward over trials")
        plt.savefig(f"{self.path}/average_distance_to_reward.png")
        plt.clf()

        self.distances_over_time = distances_over_time
        self.average_distances = average

    def plot_connections_by_time(self):

        fn = f"{self.path}/connectivities.txt"
        connectivities = []
        with open(fn, "r") as f:
            lines = f.readlines()
            for trial, line in enumerate(lines):
                c_list = [float(l) for l in line.replace("\n","").replace(']','').split(",")[1:]]
                if len(c_list) > 0:
                    connectivities.append(c_list[0])

        # plt.plot(connectivities[:len(self.time_to_reward)], self.time_to_reward)
        
        # plt.xlabel("Number of connections")
        # plt.ylabel("Time to reach reward")
        # plt.title("Number of connections against time to reach reward")
        # plt.savefig(f"{self.path}/connections_by_time.png")
        # plt.clf()
        self.connectivities = connectivities

    def generate_plots(self, gifs = False):
        self.plot_time_to_reward()
        self.plot_distances_over_time()
        self.plot_connections_by_time()
        if gifs:
            self.generate_network_gif()
            self.generate_grid_gif()



#%%

# idx = 0
# param_to_index_mapping = {}
# for param_to_sweep, values in params_to_sweep.items():
#     param_to_index_mapping[param_to_sweep] = {}
#     for v in values:
#         param_to_index_mapping[param_to_sweep][idx] = v 
#         idx += 1

# param_to_index_mapping = {'add_connections': {26: True, 27: False},
#  'prune_connections': {28: True, 29: False}}



run_dirs = ["out", "out-1", "out-2", "out-3"]

param_results = {}

for param_to_sweep, index_mapping in param_to_index_mapping.items():

    print(f"Plotting for {param_to_sweep}, values: {index_mapping.values()}")

    all_average_times = []
    all_last_times = []
    all_param_values = []
    all_distances = []
    all_connectivities = []
    all_times = []

    for run in run_dirs:
        average_times = []
        last_times = []
        _times = []
        param_values = []
        distances = []
        connectivities = []
            
        for index, p_value in index_mapping.items():
            if not os.path.exists(f"{run}/{index}"):
                continue
            trial_analysis = TrialAnalysis(**all_parameter_combinations[index], index=index, dir =run)
            trial_analysis.generate_plots()
            output = f"out/{index}"
            times = trial_analysis.time_to_reward
            if len(times) == 0:
                continue
            _times.append(times)
            average_time = sum(times)/len(times)
            last_time = times[-1]
            average_times.append(average_time)
            last_times.append(last_time)
            param_values.append(p_value)
            distances.append(trial_analysis.average_distances)
            connectivities.append(trial_analysis.connectivities)
        all_average_times.append(average_times)
        all_last_times.append(last_times)
        all_param_values.append(param_values)
        all_distances.append(distances)
        all_connectivities.append(connectivities)
        all_times.append(_times)
    
    avg_avg = np.mean(all_average_times, axis = 0)
    last_avg = np.mean(all_last_times, axis = 0)

    full_distances = []
    full_connectivities = []
    full_times = []

    for param_idx, param in enumerate(param_values):
        distances_per_param = [all_distances[run_idx][param_idx] for run_idx in range(len(run_dirs))]
        max_length = max(len(lst) for lst in distances_per_param)
        distances_per_param = [lst + [np.nan] * (max_length - len(lst)) for lst in distances_per_param]

        connectivities_per_param = [all_connectivities[run_idx][param_idx] for run_idx in range(len(run_dirs))]
        max_length = max(len(lst) for lst in connectivities_per_param)
        connectivities_per_param = [lst + [np.nan] * (max_length - len(lst)) for lst in connectivities_per_param]

        times_per_param = [all_times[run_idx][param_idx] for run_idx in range(len(run_dirs))]
        max_length = max(len(lst) for lst in times_per_param)
        times_per_param = [lst + [np.nan] * (max_length - len(lst)) for lst in times_per_param]

        full_distances.append(distances_per_param)
        full_connectivities.append(connectivities_per_param)
        full_times.append(times_per_param)


    distance_avg = [np.mean(d, axis = 0) for d in full_distances]
    connectivity_avg = [np.mean(c, axis = 0) for c in full_connectivities]
    avg_times = [np.mean(t, axis = 0) for t in full_times]

    distance_std = [  np.std(d, axis = 0) for d in full_distances]
    connectivity_std = [np.std(c, axis = 0) for c in full_connectivities]
    avg_times_std = [np.std(t, axis = 0) for t in full_times]


    plt.bar(np.arange(len(param_values)), avg_avg, 0.4, yerr=np.std(all_average_times, axis=0), capsize=5, label="Average time to reach reward")
    plt.bar(np.arange(len(param_values)) + 0.4, last_avg, 0.4, yerr=np.std(all_last_times, axis=0), capsize=5, label="Final time to reach reward")
    plt.xlabel(param_to_sweep)
    plt.xticks(np.arange(len(param_values)), param_values)
    plt.legend()
    plt.savefig(f"out/{param_to_sweep}.png")
    plt.show()

    plt.clf()

    for param_idx, param in enumerate(param_values):
        plt.plot(distance_avg[param_idx], label=f"{param_to_sweep} = {param}")
        plt.fill_between(np.arange(len(distance_avg[param_idx])), distance_avg[param_idx] - distance_std[param_idx], distance_avg[param_idx] + distance_std[param_idx], alpha=0.2)
        
    plt.ylabel("Distance to reward")
    plt.xlabel("Trials")

    # plt.xticks(np.arange(len(distance_avg[param_idx])),distance_avg[param_idx])
    plt.legend()
    plt.title(f"Distance to reward {param_to_sweep}")
    plt.savefig(f"out/{param_to_sweep}_connections.png")
    plt.show()
    plt.clf()

    for param_idx, param in enumerate(param_values):
        plt.plot(avg_times[param_idx], label=f"{param_to_sweep} = {param}")
        plt.fill_between(np.arange(len(avg_times[param_idx])), avg_times[param_idx] - avg_times_std[param_idx], avg_times[param_idx] + avg_times_std[param_idx], alpha=0.2)
    plt.ylabel("Time to reach reward")
    plt.xlabel("Trials")
 #   plt.xticks(np.arange(len(avg_times[param_idx])), avg_times[param_idx])
    plt.legend()
    plt.title(f"Time to reach reward for {param_to_sweep}")
    plt.savefig(f"out/{param_to_sweep}_average_time.png")
    plt.show()
    plt.clf()

    # for param_idx, param in enumerate(param_values):
    #     plt.plot(connectivity_avg[param_idx], avg_times[param_idx], label=f"Number of connections per time to reach reward: {param_to_sweep} = {param}")
    #     #plt.fill_between(np.arange(len(param_values)), connectivity_avg[param_idx] - connectivity_std[param_idx], connectivity_avg[param_idx] + connectivity_std[param_idx], alpha=0.2)
    # plt.xlabel(param_to_sweep)
    # plt.xticks(np.arange(len(param_values)), param_values)
    # plt.legend()
    # plt.title(f"Average distance to reward for {param_to_sweep}")
    # plt.savefig(f"out/{param_to_sweep}_distances.png")
    # plt.show()
    # plt.clf()


# # %%

# %%
