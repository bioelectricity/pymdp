#%%

from stemai.demos.ngw_params import all_parameter_combinations, params_to_sweep, param_to_index_mapping


#compile results from all out directories that have the same params.yaml file so we end up 
#with one directory per params.yaml file and in each directory we have all files for all the results
#for runs on that parameter set 

#we also want to add a flag that is for whether or not the simulation completed to the end of the time
#which we will mark for now as 14 trials and cut that off at the end of the time
