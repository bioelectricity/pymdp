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
    executor.map(
        run_simulation, range(len(all_parameter_combinations)), all_parameter_combinations
    )

# run_simulation(1, all_parameter_combinations[1])

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
