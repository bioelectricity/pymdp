
#%%
import os 
import matplotlib.pyplot as plt 
import itertools
import numpy as np
os.chdir('../')
#plots to make 


# plot the number of connections against the time taken to reach the reward location for each trial 

def plot_connections_by_time():
    runs = os.listdir("out")
    for run in runs:
        if run == ".DS_Store":
            continue

        connectivities = []
                
        with open(f"out/{run}/connectivities.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                c_list = line.split(",")
                print(f"Connectivity list: {c_list}")
                c_list = [float(c.replace('[','').replace(']','')) for c in c_list[1:]][0]
                connectivities.append(c_list)
        print(f"Connectivities: {connectivities}")
        times = []

        with open(f"out/{run}/time_to_reward.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                times.append(float(line.replace("\n","")))
        print(f"Times: {times}")
        plt.plot(connectivities[:len(times)], times)
        plt.xlabel("Number of connections")
        plt.ylabel("Time to reach reward")
        plt.title("Number of connections against time to reach reward")
        plt.savefig(f"out/{run}/connections_by_time.png")
        plt.clf()
#plot_connections_by_time()



all_num_internal_cells = [50,100,150]
all_num_sensory_cells = [4,6,8,10]
all_internal_connectivity = [0.1,0.3,0.5]
all_sensory_connectivity = [0.3,0.5,0.7]
all_active_connectivity_proportion = [0.3,0.6,0.8]
all_sensory_connectivity_proportion = [0.2,0.4,0.6]
all_action_time_threshold = [5,10,15]
all_precision_threshold = [0.1,0.3,0.5]
all_precision_update_frequency = [10,20,30]

params_to_sweep = {'internal_cells':all_num_internal_cells, 'sensory_cells': all_num_sensory_cells, 'internal_connectivity':all_internal_connectivity, 'sensory_connectivity':all_sensory_connectivity, 'active_connectivity':all_active_connectivity_proportion, 'sensory_connectivity_proportion': all_sensory_connectivity_proportion, 'action_time_threshold': all_action_time_threshold, 'precision_threshold': all_precision_threshold, 'precision_update_frequency':all_precision_update_frequency}

default_num_trials = 15
default_num_internal_cells = 50
default_num_external_cells = 1
default_num_active_cells = 4
default_num_sensory_cells = 6
default_internal_connectivity = 0.3
default_active_connectivity = 0
default_sensory_connectivity = 0.5
default_external_connectivity = 1
default_reward_location = (9,9)
default_agent_location = (0,0)
default_grid_size = 10
default_active_connectivity_proportion = 0.6
default_sensory_connectivity_proportion = 0.4
default_action_time_threshold = 10
default_precision_threshold = 0.4
default_precision_update_frequency =10
defaults = {'internal_cells':default_num_internal_cells, 'sensory_cells': default_num_sensory_cells, 'internal_connectivity':default_internal_connectivity, 'sensory_connectivity':default_sensory_connectivity, 'active_connectivity':default_active_connectivity_proportion, 'sensory_connectivity_proportion': default_sensory_connectivity_proportion, 'action_time_threshold': default_action_time_threshold, 'precision_threshold': default_precision_threshold, 'precision_update_frequency':default_precision_update_frequency}


all_parameter_combinations = []
for name, sweep_params in params_to_sweep.items():
    for param in sweep_params:
        parameters = defaults.copy()
        parameters[name] = param

        all_parameter_combinations.append(parameters)
#all_parameter_combinations = list(itertools.product(all_num_trials, all_num_internal_cells, all_num_external_cells, all_num_active_cells, all_num_sensory_cells, all_internal_connectivity, all_sensory_connectivity, all_external_connectivity, all_reward_location, all_agent_location, all_grid_size, all_active_connectivity_proportion, all_sensory_connectivity_proportion, all_action_time_threshold, all_precision_threshold, all_precision_update_frequency))
#all_parameter_combinations = [dict(zip(['num_trials', 'num_internal_cells', 'num_external_cells', 'num_active_cells', 'num_sensory_cells', 'internal_connectivity', 'sensory_connectivity', 'external_connectivity', 'reward_location', 'agent_location', 'grid_size', 'active_connectivity_proportion', 'sensory_connectivity_proportion', 'action_time_threshold', 'precision_threshold', 'precision_update_frequency'], param)) for param in all_parameter_combinations]
#%%
#plot bar charts for precision update frequency 

precision_update_frequency_indices = {0: 10, 1: 20, 2: 30}


average_times = []
last_times = []
update_frequencies = []
for index, update_freq in precision_update_frequency_indices.items():
    output = f"out/{index}"
    times = []
    with open(f"{output}/time_to_reward.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            times.append(float(line.replace("\n","")))
    average_time = sum(times)/len(times)
    last_time = times[-1]
    average_times.append(average_time)
    last_times.append(last_time)
    update_frequencies.append(update_freq)

fig, ax = plt.subplots(layout='constrained')



plt.bar(update_frequencies , average_times, 1,label = "Average time to reach reward")
plt.bar(np.array(update_frequencies) +1, last_times,1, label = "Last time to reach reward")
plt.xlabel("Precision update frequency")
plt.xticks([10,20,30])

plt.legend()
plt.show()


    
