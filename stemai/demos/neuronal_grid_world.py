# %%
from stemai.runner import Runner
import os
from stemai.demos.ngw_params import all_parameter_combinations

# %%
# comment these

import concurrent.futures

# Start Generation Here
import concurrent.futures
import shutil 
def run_simulation(index, param, dir  = 'out'):

    if os.path.exists(f'{dir}/{index}/14'):
        print(f"Simulation {index} already exists")
        return
    elif os.path.exists(f'{dir}/{index}'):
        print(f"Simulation {index} already exists but not complete, removing")
        shutil.rmtree(f'{dir}/{index}')

    if not os.path.exists(f'{dir}/{index}'):
        os.makedirs(f'{dir}/{index}')



    print(f"Running simulation {index}")
    runner = Runner(**param, index=index, dir = dir)
    runner.run()
    print(f"Finished simulation {index}")

idx = 0
# while True:
#     dir = f"out-{idx}"
    
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         executor.map(
#             run_simulation, range(len(all_parameter_combinations)), all_parameter_combinations, dir
#         )
#     print(f"Finished batch for dir {dir}")

run_simulation(19, all_parameter_combinations[19], dir = "out-0")

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
