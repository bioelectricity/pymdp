#%%
from stemai.runner import Runner 
import os 
#%%
#comment these 


all_num_internal_cells = [50,100,150]
all_num_sensory_cells = [4,6,8,10]
all_internal_connectivity = [0.1,0.3,0.5]
all_sensory_connectivity = [0.3,0.5,0.7]
all_active_connectivity_proportion = [0.3,0.6,0.8]
all_sensory_connectivity_proportion = [0.2,0.4,0.6]
all_action_time_threshold = [5,10,15]
all_precision_threshold = [0.1,0.3,0.5]
all_precision_update_frequency = [10,20,30]

params_to_sweep = {'num_internal_cells':all_num_internal_cells, 'num_sensory_cells': all_num_sensory_cells, 'internal_connectivity':all_internal_connectivity, 'sensory_connectivity':all_sensory_connectivity, 'active_connectivity_proportion':all_active_connectivity_proportion, 'sensory_connectivity_proportion': all_sensory_connectivity_proportion, 'action_time_threshold': all_action_time_threshold, 'precision_threshold': all_precision_threshold, 'precision_update_frequency':all_precision_update_frequency}

default_num_trials = 30
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
defaults = {'logging': False, 'num_trials': default_num_trials, 'num_internal_cells':default_num_internal_cells, 'num_sensory_cells': default_num_sensory_cells, 'internal_connectivity':default_internal_connectivity, 'sensory_connectivity':default_sensory_connectivity, 'active_connectivity_proportion':default_active_connectivity_proportion, 'sensory_connectivity_proportion': default_sensory_connectivity_proportion, 'action_time_threshold': default_action_time_threshold, 'precision_threshold': default_precision_threshold, 'precision_update_frequency':default_precision_update_frequency,'num_external_cells': default_num_external_cells, 'num_active_cells': default_num_active_cells, 'external_connectivity': default_external_connectivity, 'reward_location': default_reward_location, 'agent_location': default_agent_location, 'grid_size': default_grid_size}

all_parameter_combinations = []
for name, sweep_params in params_to_sweep.items():
    for param in sweep_params:
        parameters = defaults.copy()
        parameters[name] = param

        all_parameter_combinations.append(parameters)#%%

import concurrent.futures

# Start Generation Here
import concurrent.futures
import shutil 
def run_simulation(index, param):

    if os.path.exists(f'out/{index}/connectivities.txt'):
        print(f"Simulation {index} already exists")
        return
    elif os.path.exists(f'out/{index}'):
        print(f"Simulation {index} already exists but not complete, removing")
        shutil.rmtree(f'out/{index}')

    os.makedirs(f'out/{index}/0/networks')
    os.makedirs(f'out/{index}/0/grids')



    print(f"Running simulation {index}")
    runner = Runner(**param, index=index)
    runner.run()
    print(f"Finished simulation {index}")

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(run_simulation, range(1, len(all_parameter_combinations)), all_parameter_combinations[1:])


# import asyncio
# import os

# async def run_simulation(index, param):
#     if not os.path.exists(f'out/{index}/0/networks'): 
#         os.makedirs(f'out/{index}/0/networks')
#         os.makedirs(f'out/{index}/0/grids')

#     print(f"Running simulation {index}")
#     runner = Runner(**param, index=index)
#     await runner.run()

#     print(f"Finished simulation {index}")

# async def main():
#     tasks = [asyncio.create_task(run_simulation(index, param)) for index, param in enumerate(all_parameter_combinations)]
#     await asyncio.gather(*tasks)

# asyncio.run(main())

# index = 0
# for param in all_parameter_combinations:
#     if not os.path.exists(f'out/{index}/0/networks'): 
#         os.makedirs(f'out/{index}/0/networks')
#         os.makedirs(f'out/{index}/0/grids')

#     print(f"Running simulation {index}")
#     runner = Runner(**param, index=index)
#     runner.run()
#     index += 1
#     print(f"Finished simulation {index}")
# %%
