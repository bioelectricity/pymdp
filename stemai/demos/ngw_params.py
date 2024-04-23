num_internal_cells = 50

num_external_cells = 1
num_active_cells = 4
num_sensory_cells = 6

internal_connectivity = 0.6
active_connectivity = 0
sensory_connectivity = 0.6
external_connectivity = 1

<<<<<<< HEAD
REWARD_LOCATION = (0, 9)
=======
REWARD_LOCATION = (9, 9)
>>>>>>> 8188d5581b557d5995f45caaf938d6c97c8b0116
AGENT_LOCATION = (4, 4)
GRID_SIZE = 10

active_connectivity_proportion = 0.6
sensory_connectivity_proportion = 0.2
action_time_horizon = 10
precision_threshold = 0.5
precision_update_frequency = 10



all_num_internal_cells = [50, 100]
<<<<<<< HEAD
all_num_sensory_cells = [5, 10]
=======
all_num_sensory_cells = [2, 4, 6]
>>>>>>> 8188d5581b557d5995f45caaf938d6c97c8b0116
all_internal_connectivity = [0.1, 0.3, 0.5]
all_sensory_connectivity = [0.3, 0.5, 0.7]
all_active_connectivity_proportion = [0.3, 0.6, 0.8]
all_sensory_connectivity_proportion = [0.2, 0.4, 0.6]
all_action_time_threshold = [1, 5, 10]
<<<<<<< HEAD
all_precision_threshold = [0.05,0.1,0.2]
all_precision_update_frequency = [10, 20, 30]
all_prune_connections = [True, False] #need to rerun with False 
all_prune_intervals = [2, 5, 10]
=======
all_precision_threshold = [0.1, 0.3, 0.5]
all_precision_update_frequency = [10, 20, 30]
all_add_connections = [True, False] #need to rerun with True
all_prune_connections = [True, False] #need to rerun with False 
>>>>>>> 8188d5581b557d5995f45caaf938d6c97c8b0116

params_to_sweep = {
    "num_internal_cells": all_num_internal_cells,
    "num_sensory_cells": all_num_sensory_cells,
    "internal_connectivity": all_internal_connectivity,
    "sensory_connectivity": all_sensory_connectivity,
    "active_connectivity_proportion": all_active_connectivity_proportion,
    "sensory_connectivity_proportion": all_sensory_connectivity_proportion,
    "action_time_threshold": all_action_time_threshold,
    "precision_threshold": all_precision_threshold,

    "precision_update_frequency": all_precision_update_frequency,
<<<<<<< HEAD
    "prune_interval": all_prune_intervals,
=======
    "add_connections": all_add_connections,
>>>>>>> 8188d5581b557d5995f45caaf938d6c97c8b0116
    "prune_connections": all_prune_connections,

}

<<<<<<< HEAD
default_num_trials = 100
default_num_internal_cells = 50
default_num_external_cells = 1
default_num_active_cells = 6
default_num_sensory_cells = 12
default_internal_connectivity = 0.3
default_active_connectivity = 0
default_sensory_connectivity = 0.2
=======
default_num_trials = 21
default_num_internal_cells = 50
default_num_external_cells = 1
default_num_active_cells = 4
default_num_sensory_cells = 6
default_internal_connectivity = 0.3
default_active_connectivity = 0
default_sensory_connectivity = 0.5
>>>>>>> 8188d5581b557d5995f45caaf938d6c97c8b0116
default_external_connectivity = 1
default_reward_location = (9, 9)
default_agent_location = (0, 0)
default_grid_size = 10
default_active_connectivity_proportion = 0.6
<<<<<<< HEAD
default_sensory_connectivity_proportion = 0.6
default_action_time_threshold = 5
default_precision_threshold = 0.06 #how far away from 0.5 you are in each direction
default_precision_update_frequency = 10 #timesteps 
default_prune_interval = 5 #trials
=======
default_sensory_connectivity_proportion = 0.4
default_action_time_threshold = 10
default_precision_threshold = 0.4
default_precision_update_frequency = 10
default_add_connections = False 
>>>>>>> 8188d5581b557d5995f45caaf938d6c97c8b0116
default_prune_connections = True
defaults = {
    "logging": False,
    "num_trials": default_num_trials,
    "num_internal_cells": default_num_internal_cells,
    "num_sensory_cells": default_num_sensory_cells,
    "internal_connectivity": default_internal_connectivity,
    "sensory_connectivity": default_sensory_connectivity,
    "active_connectivity_proportion": default_active_connectivity_proportion,
    "sensory_connectivity_proportion": default_sensory_connectivity_proportion,
    "action_time_threshold": default_action_time_threshold,
    "precision_threshold": default_precision_threshold,
    "precision_update_frequency": default_precision_update_frequency,
    "num_external_cells": default_num_external_cells,
    "num_active_cells": default_num_active_cells,
    "external_connectivity": default_external_connectivity,
    "reward_location": default_reward_location,
    "agent_location": default_agent_location,
    "grid_size": default_grid_size,
<<<<<<< HEAD
    "prune_interval": default_prune_interval,
=======
    "add_connections": default_add_connections,
>>>>>>> 8188d5581b557d5995f45caaf938d6c97c8b0116
    "prune_connections": default_prune_connections,
}
did_default=False
all_parameter_combinations = []
<<<<<<< HEAD
idx = 1
=======
idx = 0
>>>>>>> 8188d5581b557d5995f45caaf938d6c97c8b0116
param_to_index_mapping = {}

for name, sweep_params in params_to_sweep.items():
    param_to_index_mapping[name] = {}
    for param in sweep_params:
        parameters = defaults.copy()
        if parameters[name] == param and did_default is True:
            param_to_index_mapping[name][default_index] = param
            continue
        elif parameters[name] == param:
            did_default = True
            default_index = idx
        parameters[name] = param

        all_parameter_combinations.append(parameters)  # %%
        param_to_index_mapping[name][idx] = param
        idx += 1
