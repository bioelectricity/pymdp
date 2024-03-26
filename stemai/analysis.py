
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
        print(f"Loading time to reward from {fn}")
        time_to_reward = []
        with open(fn, "r") as f:
            lines = f.readlines()
            for line in lines:
                time_to_reward.append(float(line.replace("\n","")))
        plt.plot(time_to_reward)
        plt.xlabel("Trials")
        plt.ylabel("Number of timesteps to reward")
        plt.title(f"Number of timesteps to reward over trials")
        print(f"Saving plot to {self.path}/time_to_reward.png")
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
            plt.savefig(f"{self.path}/distances_over_time_{trial}.png")
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


#%%


def analyze_parameter(dir):
    import yaml 


    with open(f"{dir}/params.yaml", "r") as file:
        default_params = yaml.load(file, Loader=yaml.FullLoader)

    num_runs = [f for f in os.listdir(dir) if 'DS' not in f and not  f.endswith('yaml') and not f.endswith('png')]
    if len(num_runs) == 0:
        return 

    average_times = []
    last_times = []
    _times = []
    distances = []
    connectivities = []

    print(f"Parameter dir : {dir}, num runs: {num_runs}")


    for run in num_runs:
        index = int(run)
        trial_analysis = TrialAnalysis(**default_params, index=index, dir =dir)
        trial_analysis.generate_plots()
        times = trial_analysis.time_to_reward
        if len(times) == 0:
            continue
        _times.append(times)
        average_time = sum(times)/len(times) #average time taken to reach reward for this run, averaged over trials 
        last_time = times[-1] #the time taken to reach the reward in the final trial of this run 
        average_times.append(average_time)
        last_times.append(last_time)
        distances.append(trial_analysis.average_distances)
        connectivities.append(trial_analysis.connectivities)
    
    if len(_times) == 0:
        return None, None
    print(f"Times: {_times}")

    max_length = max(len(times) for times in _times)
    for times in _times:
        times.extend([np.nan] * (max_length - len(times)))

    average_times_over_runs = np.nanmean(_times, axis = 0)
    std_times_over_runs = np.nanstd(_times, axis = 0)

    plt.plot(average_times_over_runs, label="Average time to reach reward")
    plt.fill_between(np.arange(len(average_times_over_runs)), average_times_over_runs - std_times_over_runs, average_times_over_runs + std_times_over_runs, alpha=0.2)
    plt.title("Average time to reach reward over runs")
    plt.savefig(f"{dir}/average_time_to_reward.png")
    plt.clf()

    max_dist = max(len(_d) for _d in distances)
    for d in distances:
        d.extend([np.nan] * (max_dist - len(d)))
    average_distances_over_runs = np.nanmean(distances, axis=0)
    std_distances_over_runs = np.nanstd(distances, axis=0)

    plt.plot(average_distances_over_runs, label="Average distance to reward")
    plt.fill_between(np.arange(len(average_distances_over_runs)), average_distances_over_runs - std_distances_over_runs, average_distances_over_runs + std_distances_over_runs, alpha=0.2)
    plt.title("Average distance to reward over runs")
    plt.savefig(f"{dir}/average_distance_to_reward.png")
    plt.clf()
    max_connect = max(len(_c) for _c in connectivities)
    for c in connectivities:
        c.extend([np.nan] * (max_connect - len(c)))
    average_connectivities_over_runs = np.nanmean(connectivities, axis=0)
    std_connectivities_over_runs = np.nanstd(connectivities, axis=0)

    plt.plot(average_connectivities_over_runs, label="Average connectivity to reward")
    plt.fill_between(np.arange(len(average_connectivities_over_runs)), average_connectivities_over_runs - std_connectivities_over_runs, average_connectivities_over_runs + std_connectivities_over_runs, alpha=0.2)
    plt.title("Average connectivity over runs")
    plt.savefig(f"{dir}/average_connectivity_to_reward.png")

    return average_times, last_times



#%%

default_dir = "new-gamma-update/param_0"

default_average_times, default_last_times = analyze_parameter(default_dir)

mean_default_avg_time = np.mean(default_average_times)
std_default_avg_time = np.std(default_average_times)

mean_default_last_time = np.mean(default_last_times)
std_default_last_time = np.std(default_last_times)

defaults = {}

defaults["new-gamma-update"] = {"mean_default_avg_time": mean_default_avg_time, "std_default_avg_time": std_default_avg_time, "mean_default_last_time": mean_default_last_time, "std_default_last_time": std_default_last_time}
default_dir = "output/param_0"

default_average_times, default_last_times = analyze_parameter(default_dir)

mean_default_avg_time = np.mean(default_average_times)
std_default_avg_time = np.std(default_average_times)

mean_default_last_time = np.mean(default_last_times)
std_default_last_time = np.std(default_last_times)

defaults["output"] = {"mean_default_avg_time": mean_default_avg_time, "std_default_avg_time": std_default_avg_time, "mean_default_last_time": mean_default_last_time, "std_default_last_time": std_default_last_time}


#%%

sweep_param_dirs_1 = [f'output/{f}' for f in os.listdir('output') if 'param' in f and f != 'param_0']
times_per_param = {}

sweep_param_dirs = [f'new-gamma-update/{f}' for f in os.listdir('new-gamma-update') if 'param' in f and f != 'param_0']
sweep_param_dirs += sweep_param_dirs_1
for param_dir in sweep_param_dirs:
    average_times, last_times = analyze_parameter(param_dir)
    if average_times is not None:
        times_per_param[param_dir] = (average_times, last_times)



#%%

directories = [ "output", "new-gamma-update"]

for param, param_values_dict in param_to_index_mapping.items():

    param_values = list(param_values_dict.values())

    param_values += [f"{p}*" for p in param_values]

    avg_avg = []
    last_avg = []

    avg_std = []
    last_std = []

    for directory in directories:


        for idx, value in param_values_dict.items():
            dir = f"{directory}/param_{idx}"
            if idx == 0:
                mean_avg_time = defaults[directory]["mean_default_avg_time"]
                std_avg_time = defaults[directory]["std_default_avg_time"]

                mean_last_time = defaults[directory]["mean_default_last_time"]
                std_last_time = defaults[directory]["std_default_last_time"]
            else:
                if dir in times_per_param:
                    average_times, last_times = times_per_param[dir]
                    mean_avg_time = np.mean(average_times)
                    std_avg_time = np.std(average_times)
                    mean_last_time = np.mean(last_times)
                    std_last_time = np.std(last_times)
            avg_avg.append(mean_avg_time)
            last_avg.append(mean_last_time)
            avg_std.append(std_avg_time)
            last_std.append(std_last_time)

    print(f"param values: {param_values}")

    plt.bar(np.arange(len(param_values)), avg_avg, 0.4, yerr=avg_std, capsize=5, label="Average time to reach reward")
    plt.bar(np.arange(len(param_values)) + 0.4, last_avg, 0.4, yerr=last_std, capsize=5, label="Final time to reach reward")
    plt.xlabel(param)
    plt.xticks(np.arange(len(param_values)), param_values)
    plt.legend()
    plt.savefig(f"bar-charts/{param}.png")
    plt.show()

    plt.clf()

# %%
