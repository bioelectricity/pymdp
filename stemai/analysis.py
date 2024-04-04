
#%%
import os 
import matplotlib.pyplot as plt 
import pickle
import numpy as np
import imageio
import networkx
import pdb
os.chdir('../')
#plots to make 
from stemai.demos.ngw_params import all_parameter_combinations, params_to_sweep, param_to_index_mapping
from stemai.utils import draw_network
import json
def flatten(original_list):
    flattened_list = [item for sublist in original_list for item in sublist]
    return flattened_list
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
        prune_interval,
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
        self.prune_interval = prune_interval

        self.path = f"{dir}/{index}"

        self.trial_paths = [f"{self.path}/{i}" for i in range(self.num_trials) if os.path.exists(f"{self.path}/{i}")]

    def get_gammas(self):

        gamma_files = os.listdir(f"{self.path}/gammas")
        gamma_data = {}

        for trial, gamma_file_idx in enumerate(range(1, len(gamma_files)+1)):
            gamma_file = f"{gamma_file_idx}.pickle"
            alldata = pickle.load(open(f"{self.path}/gammas/{gamma_file}", "rb"))
            for t, data in enumerate(alldata[list(alldata.keys())[0]].values()):
                gamma_data[(trial+1)*t] = data
        self.gamma_data = gamma_data

    def get_precisions(self):
        precision_files = os.listdir(f"{self.path}/precisions")
        precision_data = {}

        for t, precision_file_idx in enumerate(range(1, len(precision_files)+1)):
            precision_file = f"{precision_file_idx}.pickle"
            alldata = pickle.load(open(f"{self.path}/precisions/{precision_file}", "rb"))
            for t, data in alldata[list(alldata.keys())[0]].items():
                precision_data[t] = data
        self.precision_data = precision_data

    def draw_networks_from_precisions(self):

        #for trial_path in self.trial_paths:

        precision_files = os.listdir(f"{self.path}/precisions")
        pos = None
        network_images = []
        network_files = []
        precision_data = {}

        print(f"Precision files: {precision_files}")

        for t, precision_file_idx in enumerate(range(1, len(precision_files)+1)):
            precision_file = f"{precision_file_idx}.pickle"
            alldata = pickle.load(open(f"{self.path}/precisions/{precision_file}", "rb"))
            for t, data in alldata[list(alldata.keys())[0]].items():
                precision_data[t] = data

                # Initialize a directed graph
                G = networkx.Graph()

                # Add nodes to the graph
                for node, neighbors in data.items():
                    for neighbor, weight in neighbors.items():
                        G.add_edge(node, neighbor, weight=weight)

                edge_colors = [weight for weight in networkx.get_edge_attributes(G, 'weight').values()]

                #print(f"Weights: {edge_colors}")
                edge_colors = ["darkblue" if color > 0.84 else "lightblue" if color < 0.68 else "blue" for color in edge_colors]  # Convert to gradient of blue based on edge color
                # print(f"Colors: {edge_colors}")
                
                # Draw the graph
                if pos is None:
                    pos = networkx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm


                if not os.path.exists(f"{self.path}/networks"):
                    os.makedirs(f"{self.path}/networks")
                fn =  f"{self.path}/networks/network_{t}.png"
                draw_network(G, pos=pos, save=True, show=False, edge_colors = edge_colors, temp_file_name=f"{self.path}/networks/network_{t}.png")
                network_files.append(fn)
                network_images.append(imageio.imread(fn))
        gif_path = f"{self.path}/network-simulation.gif"
        imageio.mimsave(gif_path, network_images, fps=5)
        # for temp_file in network_files:
        #     os.remove(temp_file)
        self.precision_data = precision_data

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
        grid_images = []
        for trial_idx, trial_path in enumerate(self.trial_paths):
            grids = np.load(f"{trial_path}/grids.npy")
            if not os.path.exists(f"{self.path}/grids"):
                os.makedirs(f"{self.path}/grids")
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

        for i in range(0, int(len(time_to_reward)), 10):
            print(f"Line at {i}")
            plt.axvline(x=i, color='black', linestyle='--')
    
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
        #     plt.plot(distances)
             average.append(np.mean(distances))
        #     plt.xlabel("Timesteps")
        #     plt.ylabel("Distance to reward")
        #     plt.title(f"Distance to reward over time, trial: {trial}")
        #     plt.savefig(f"{self.path}/distances_over_time_{trial}.png")
        #     plt.clf()
        # plt.plot(average)
        # plt.xlabel("Trials")
        # plt.ylabel("Average distance to reward")
        # plt.title(f"Average distance to reward over trials")
        # plt.savefig(f"{self.path}/average_distance_to_reward.png")
        # plt.clf()

        self.distances_over_time = distances_over_time
        self.average_distances = average

    def plot_connections_by_time(self):

        file_path = f"{self.path}/connectivities.txt"


        import json

        # Dictionary to store data from each line
        connectivities = []
        c_per_node = []

        # Open the file and read each line 
        with open(file_path, 'r') as file:
            for line in file:
                

                line = line.split(': ', 1)[1].replace("'", '"')
                print(line)

                c = json.loads(line)
                if len(c) == 0:
                    continue
                c_per_node.append(c)
                connectivities.append(sum(list(c.values())))

        
        self.all_connectivities_per_node = c_per_node
        plt.scatter(connectivities[:len(self.time_to_reward)], self.time_to_reward)
        
        plt.xlabel("Number of connections")
        plt.ylabel("Time to reach reward")
        plt.title("Number of connections against time to reach reward")
        plt.savefig(f"{self.path}/connections_by_time.png")
        plt.clf()
        self.connectivities = connectivities
        plt.plot(connectivities)
        
        plt.ylabel("Number of connections")
        plt.xlabel("Trials")
        plt.title("Number of connections over time")
        plt.savefig(f"{self.path}/connections.png")
        plt.clf()
    def generate_plots(self, gifs = False):
        # if not os.path.exists(f"{self.path}/network-simulation.gif") and self.prune_connections:
        #     self.draw_networks_from_precisions()
        self.plot_time_to_reward()
        self.plot_distances_over_time()
        self.plot_connections_by_time()
        if gifs:
            #self.generate_network_gif()
            self.generate_grid_gif()

    def analyze_precisions(self):
        self.get_precisions()
        node_precisions = {}
        for t, data in self.precision_data.items():
            for node, neighbor_precision in data.items():
                if node not in node_precisions:
                    node_precisions[node] = {}
                for neighbor, precision in neighbor_precision.items():
                    if not neighbor in node_precisions[node]:
                        node_precisions[node][neighbor] = [precision]
                    else:
                        node_precisions[node][neighbor].append(precision)
        self.node_precisions = node_precisions


    def analyze_gammas(self):
        self.get_gammas()
        gammas = {}
        for t, data in self.gamma_data.items():
            for node, neighbor_gamma in data.items():
                if node not in gammas:
                    gammas[node] = {}
                for neighbor, precision in neighbor_gamma.items():
                    if not neighbor in gammas[node]:
                        gammas[node][neighbor] = {t: precision}
                    else:
                        gammas[node][neighbor][t] = precision
        self.gammas = gammas

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
    all_distances = []

    print(f"Parameter dir : {dir}, num runs: {num_runs}")

    trial_objs = []

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
        all_distances.append(trial_analysis.distances_over_time)
        trial_objs.append(trial_analysis)
    
    if len(_times) == 0:
        return None, None
    print(f"Times: {_times}")

    max_length = max(len(times) for times in _times)
    for times in _times:
        times.extend([np.nan] * (max_length - len(times)))

    average_times_over_runs = np.nanmean(_times, axis = 0)
    std_times_over_runs = np.nanstd(_times, axis = 0)

    plt.plot(average_times_over_runs, label="Average time to reach reward")
    for i in range(0, len(average_times_over_runs), 10):
        print(f"Line at {i}")
        plt.axvline(x=i, color='black', linestyle='--')
    
    for idx, time in enumerate(_times):
        plt.plot(time, alpha=0.2, label=f"Run {idx}")
    plt.fill_between(np.arange(len(average_times_over_runs)), average_times_over_runs - std_times_over_runs, average_times_over_runs + std_times_over_runs, alpha=0.2)
    plt.title("Average time to reach reward over runs")
    plt.savefig(f"{dir}/average_time_to_reward.png")
    plt.clf()

    max_dist = max(len(_d) for _d in distances)
    for d in distances:
        d.extend([np.nan] * (max_dist - len(d)))
    for idx, d in enumerate(distances):
        plt.plot(d, alpha=0.2, label=f"Run {idx}")
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

    plt.plot(average_connectivities_over_runs, label="Average connectivity")
    plt.fill_between(np.arange(len(average_connectivities_over_runs)), average_connectivities_over_runs - std_connectivities_over_runs, average_connectivities_over_runs + std_connectivities_over_runs, alpha=0.2)
    plt.title("Average connectivity over runs")
    plt.savefig(f"{dir}/average_connectivity.png")
    plt.clf()


    # plt.scatter(flatten(_times), flatten(connectivities))
    # plt.xlabel("Time to reward")
    # plt.ylabel("Connectivity")
    # plt.title("Connectivity against time to reward")
    # plt.savefig(f"{dir}/connectivity_vs_time.png")
    # plt.clf()

    return trial_objs, average_times, last_times, connectivities, _times, all_distances




 
default_dir = "default-run"

trial_objs, default_average_times, default_last_times, default_connectivities, default_times, default_distances = analyze_parameter(default_dir)

mean_default_avg_time = np.mean(default_average_times)
std_default_avg_time = np.std(default_average_times)

mean_default_last_time = np.mean(default_last_times)
std_default_last_time = np.std(default_last_times)
 #%%
for t_idx, trial in enumerate(trial_objs):
    trial.analyze_gammas()

    for node in trial.gammas:

        neighbors = [n for n in trial.gammas[node] if 's' not in n]
        node_precision_array = np.zeros((len(neighbors), 427))
        maxlen = 0
        for idx, neighbor_node in enumerate(neighbors):
            if len(trial.gammas[node][neighbor_node].values()) > maxlen:
                maxlen = len(trial.gammas[node][neighbor_node].values())
        node_precision_array = np.zeros((len(neighbors), maxlen))
        for idx, neighbor_node in enumerate(neighbors):

            node_precision_array[idx][:len(trial.gammas[node][neighbor_node].values())] = [np.sum(p) for p in trial.gammas[node][neighbor_node].values()]
            node_precision_array[idx][len(trial.gammas[node][neighbor_node].values()):] = np.nan

        fig = plt.figure(figsize = (20,10))
        import matplotlib

        array_to_plot = node_precision_array[:, ::5]
        masked_array = np.ma.array (array_to_plot, mask=np.isnan(array_to_plot))
        cmap = matplotlib.cm.jet
        cmap.set_bad('white',1.)
        plt.imshow(masked_array, cmap = cmap)
        plt.colorbar()
        plt.xlabel("Trials")
        plt.ylabel("Neighbor nodes")
        plt.yticks(range(len(neighbors)), labels = neighbors)
       # plt.xticks(range(len(list(trial.gamma_data.keys())[::10])), list(trial.gamma_data.keys())[::10], rotation = 45)
        plt.title(f"Precision over time for node {node}")
        plt.savefig(f"{default_dir}/{t_idx + 1}/precision_over_time_{node}.png")
        plt.clf()
        #%%
#%%
sweep_param_dirs = [f'output/{f}' for f in os.listdir('output') if 'param' in f and f != 'param_1']
times_per_param = {}

for param_dir in sweep_param_dirs:
    average_times, last_times, _, _, _ = analyze_parameter(param_dir)
    if average_times is not None:
        times_per_param[param_dir] = (average_times, last_times)



#%%

directories = [ "output"]

for param, param_values_dict in param_to_index_mapping.items():

    param_values = list(param_values_dict.values())

    # param_values += [f"{p}*" for p in param_values]

    avg_avg = []
    last_avg = []

    avg_std = []
    last_std = []

    for directory in directories:


        for idx, value in param_values_dict.items():
            dir = f"{directory}/param_{idx}"
            if idx == 1:
                mean_avg_time = mean_default_avg_time
                std_avg_time = std_default_avg_time
                mean_last_time = mean_default_last_time
                std_last_time = std_default_last_time
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
