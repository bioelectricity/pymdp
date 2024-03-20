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
def run_simulation_and_save(param, dir):

    dirs_to_search = os.listdir(dir)
    new_dir = None
    print(f"Searching directories: {dirs_to_search} for this parameter combination")
    for d in dirs_to_search:
        if os.path.exists(f"{dir}/{d}/params.yaml"):
            print(f"Checking directory {d} for params.yaml file")
            file = open(f"{dir}/{d}/params.yaml", "r")
            if file.read() == yaml.dump(param):
                print(f"Found directory for this parameter combination: {d}")
                new_dir_idx = max([int(f) for f in os.listdir(f'{dir}/{d}') if not f.endswith('yaml') and not f.endswith('.png') and 'DS' not in f], default=0) + 1
                new_dir = f'{dir}/{d}/{new_dir_idx}'
                print(f"New dir: {new_dir}")
    if new_dir is None:
        d = max([int(f[-1]) for f in os.listdir(dir) if not f.endswith('yaml') and not f.endswith('.png') and 'DS' not in f], default=0) + 1
        new_dir = f'{dir}/param_{d}/0'
    assert new_dir is not None

    print(f"Will save results to : {new_dir}")
    os.makedirs(new_dir)

    run_simulation(param, new_dir)

def run_simulation(param, dir, save_grids=False, save_networks=False):
    runner = Runner(**param, dir = dir)
    runner.run(save_grids=save_grids, save_networks=save_networks)


def sweep(dir):

    while True:

        for param in tqdm.tqdm(all_parameter_combinations):
            run_simulation_and_save(param, dir)
        
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = []
        #     for idx, param in enumerate(all_parameter_combinations):
        #         futures.append(executor.submit(run_simulation, idx, param, dir=dir))
        #     for future in concurrent.futures.as_completed(futures):
        #         future.result()

        print(f"Finished batch for dir {dir}")
 

def run_default():
    param = all_parameter_combinations[0]
    dir = 'default-run'
    run_simulation(param, dir, save_grids=True, save_networks=True)

if __name__ == "__main__":
    #run_default()
    sweep('new-gamma-update')
