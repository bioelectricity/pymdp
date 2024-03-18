# %%
from stemai.runner import Runner
import os
from stemai.demos.ngw_params import all_parameter_combinations
import tqdm 
# %%
# comment these

import yaml

# Start Generation Here
import concurrent.futures
import shutil 
def run_simulation(param):

    dirs_to_search = os.listdir('output')
    new_dir = None
    print(f"Searching directories: {dirs_to_search} for this parameter combination")
    for d in dirs_to_search:
        if os.path.exists(f"output/{d}/params.yaml"):
            print(f"Checking directory {d} for params.yaml file")
            file = open(f"output/{d}/params.yaml", "r")
            if file.read() == yaml.dump(param):
                print(f"Found directory for this parameter combination: {d}")
                new_dir_idx = max([int(f) for f in os.listdir(f'output/{d}') if not f.endswith('yaml') and not f.endswith('.png') and 'DS' not in f], default=0) + 1
                new_dir = f'output/{d}/{new_dir_idx}'
    assert new_dir is not None

    print(f"Will save results to : {new_dir}")
    os.makedirs(new_dir)

    runner = Runner(**param, dir = new_dir)
    runner.run()


while True:

    for param in tqdm.tqdm(all_parameter_combinations[19:]):
        run_simulation(param)
    
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = []
    #     for idx, param in enumerate(all_parameter_combinations):
    #         futures.append(executor.submit(run_simulation, idx, param, dir=dir))
    #     for future in concurrent.futures.as_completed(futures):
    #         future.result()

    print(f"Finished batch for dir {dir}")

# run_simulation(19, all_parameter_combinations[19], dir = "out-0")

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
