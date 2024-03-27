# %%
import os
# os.chdir('/Users/daphne/Desktop/stemai/pymdp')
from stemai.runner import Runner

from stemai.demos.ngw_params import all_parameter_combinations, defaults
import tqdm 
# %%
# comment these

import yaml

# Start Generation Here
import concurrent.futures
import shutil 
import time 
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
        d = max([int(f.replace('param_','')) for f in os.listdir(dir) if not f.endswith('yaml') and not f.endswith('.png') and 'DS' not in f], default=0) + 1
        new_dir = f'{dir}/param_{d}/0'
        param_yaml = yaml.dump(param)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        with open(f"{dir}/param_{d}/params.yaml", "w") as file:
            file.write(param_yaml)
    assert new_dir is not None

    print(f"Will save results to : {new_dir}")
    time.sleep(5)
    
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    run_simulation(param, new_dir)

def run_simulation(param, dir, save_grids=False, save_networks=False, default = False):
    runner = Runner(**param, dir = dir, default = default)
    runner.run(save_grids=save_grids, save_networks=save_networks)


def sweep(dir):

    while True:

        for idx, param in enumerate(tqdm.tqdm(all_parameter_combinations)):
            print(f"Param: {param}")
            run_simulation_and_save(param, dir)
            
        
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = []
        #     for idx, param in enumerate(all_parameter_combinations):
        #         futures.append(executor.submit(run_simulation, idx, param, dir=dir))
        #     for future in concurrent.futures.as_completed(futures):
        #         future.result()

        print(f"Finished batch for dir {dir}")
 

def run_default():
    param = defaults

    dir = 'default-run'
    run_simulation(param, dir, save_grids=True, save_networks=True, default=True)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run neuronal grid world simulations.")
    parser.add_argument("--mode", choices=["sweep", "run_default"], required=True, help="Choose 'sweep' to run parameter sweep or 'run_default' to run with default parameters.")
    args = parser.parse_args()

    if args.mode == "run_default":
        run_default()
    elif args.mode == "sweep":
        sweep('output')
   #sweep('new-gamma-update')
