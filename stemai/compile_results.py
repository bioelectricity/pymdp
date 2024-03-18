#%%
import os 
import yaml
os.chdir("../")

from stemai.demos.ngw_params import all_parameter_combinations, params_to_sweep, param_to_index_mapping


#compile results from all out directories that have the same params.yaml file so we end up 
#with one directory per params.yaml file and in each directory we have all files for all the results
#for runs on that parameter set 

#we also want to add a flag that is for whether or not the simulation completed to the end of the time
#which we will mark for now as 14 trials and cut that off at the end of the time
#%%
for idx, param in enumerate(all_parameter_combinations):
    param_results_dir = f'output/param_{idx}'
    os.makedirs(param_results_dir, exist_ok=True)
    with open(f"{param_results_dir}/params.yaml", "w") as file:
        yaml.dump(param, file)
    dirs_to_search = [f"OLD-results/{d}" for d in os.listdir('OLD-results') if 'out' in d]
    for d in dirs_to_search:
        matching_dir_idx =1
        for subdir in os.listdir(d):
            if os.path.exists(f"{d}/{subdir}/params.yaml"):
                file = open(f"{d}/{subdir}/params.yaml", "r")
                if file.read() == yaml.dump(param):
                    # Found matching yaml file
                    matching_yaml_file = f"{d}/{subdir}/params.yaml"
                    print("Found matching yaml file at ", matching_yaml_file)

                    os.makedirs(f"{param_results_dir}/{matching_dir_idx}", exist_ok=True)
                    contents = os.listdir(f"{d}/{subdir}")
                    print(f"Contents of {d}/{subdir} are {contents}")
                    for c in contents:
                        if c != 'params.yaml':
                            print(f"Moving {d}/{subdir}/{c} to {param_results_dir}/{matching_dir_idx}")
                            os.rename(f"{d}/{subdir}/{c}", f"{param_results_dir}/{matching_dir_idx}/{c}")
                    matching_dir_idx += 1
                file.close()
# %%
