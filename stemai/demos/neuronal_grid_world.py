#%%
from stemai.runner import Runner 
import os 
#%%
all_num_trials = [1]
all_num_internal_cells = [50,100,150]
all_num_external_cells = [1]
all_num_active_cells = [4]
all_num_sensory_cells = [4,6,8,10]
all_internal_connectivity = [0.1,0.3,0.5]
all_active_connectivity = [0]
all_sensory_connectivity = [0.3,0.5,0.7]
all_external_connectivity = [1]
all_reward_location = [(9,9)]
all_agent_location = [(0,0)]
all_grid_size = [10]
all_active_connectivity_proportion = [0.3,0.6,0.8]
all_sensory_connectivity_proportion = [0.2,0.4,0.6]
all_action_time_threshold = [5,10,15]
all_precision_threshold = [0.1,0.3,0.5]
all_precision_update_frequency = [10,20,30]

import itertools

all_parameter_combinations = list(itertools.product(all_num_trials, all_num_internal_cells, all_num_external_cells, all_num_active_cells, all_num_sensory_cells, all_internal_connectivity, all_sensory_connectivity, all_external_connectivity, all_reward_location, all_agent_location, all_grid_size, all_active_connectivity_proportion, all_sensory_connectivity_proportion, all_action_time_threshold, all_precision_threshold, all_precision_update_frequency))
all_parameter_combinations = [dict(zip(['num_trials', 'num_internal_cells', 'num_external_cells', 'num_active_cells', 'num_sensory_cells', 'internal_connectivity', 'sensory_connectivity', 'external_connectivity', 'reward_location', 'agent_location', 'grid_size', 'active_connectivity_proportion', 'sensory_connectivity_proportion', 'action_time_threshold', 'precision_threshold', 'precision_update_frequency'], param)) for param in all_parameter_combinations]
#%%

import concurrent.futures

# Start Generation Here
import asyncio
import os

async def run_simulation(index, param):
    if not os.path.exists(f'out/{index}/0/networks'): 
        os.makedirs(f'out/{index}/0/networks')
        os.makedirs(f'out/{index}/0/grids')

    print(f"Running simulation {index}")
    runner = Runner(**param, index=index)
    await runner.run()
    await runner.plot_time_to_reward()
    await runner.plot_distances_over_time()
    await runner.generate_gifs()

    print(f"Finished simulation {index}")

async def main():
    tasks = [asyncio.create_task(run_simulation(index, param)) for index, param in enumerate(all_parameter_combinations)]
    await asyncio.gather(*tasks)

asyncio.run(main())


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
