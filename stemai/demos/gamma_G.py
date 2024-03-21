#%%
import os 
os.chdir('/Users/daphne/Desktop/stemai/pymdp')
import numpy as np 
from pymdp import utils
import matplotlib.pyplot as plt
from pymdp import maths
from pymdp.agent import Agent 
import copy

from pymdp.envs import TMazeEnv
reward_probabilities = [0.98, 0.02] # probabilities used in the original SPM T-maze demo
env = TMazeEnv(reward_probs = reward_probabilities)

B_gp = env.get_transition_dist()
A_gp = env.get_likelihood_dist()
B_gm = copy.deepcopy(B_gp)
A_gm = copy.deepcopy(A_gp)

# %%
import os

os.chdir("/Users/daphne/Desktop/stemai/pymdp")
import numpy as np
from pymdp import utils
from pymdp.agent import Agent

#generative model 

num_modalities = 3 #location, reward, cue
num_factors = 2 #location (4 locations: center, left, right, down), context (2 contexts: left, right)

#actions: (center, left, right, down) -> change location state 

#2 cues : can be ambiguous, left or right 

#reward: 4, -6, 0

gamma_G = 1.0
pA = utils.dirichlet_like(A_gp, scale = 1e16)

pA[1][1:,1:3,:] = 1.0

A_gm = utils.norm_dist_obj_arr(pA)      
controllable_indices = [0] # this is a list of the indices of the hidden state factors that are controllable
learnable_modalities = [1] # this is a list of the modalities that you want to be learn-able 

agent = Agent(A = A_gm, pA = pA, B = B_gm, gamma=gamma_G, modalities_to_learn=learnable_modalities,lr_pA = 0.25, use_param_info_gain=True, inference_algo="MMP", inference_horizon=2, policy_len=1, policy_sep_prior=True)
agent.C[1][1] = 4.0
agent.C[1][2] = -6.0
agent.D[0] = utils.onehot(0, agent.num_states[0])

T = 25 # number of timesteps

obs = env.reset() # reset the environment and get an initial observation

# these are useful for displaying read-outs during the loop over time
reward_conditions = ["Right", "Left"]
location_observations = ['CENTER','RIGHT ARM','LEFT ARM','CUE LOCATION']
reward_observations = ['No reward','Reward!','Loss!']
cue_observations = ['Cue Right','Cue Left']
msg = """ === Starting experiment === \n Reward condition: {}, Observation: [{}, {}, {}]"""
print(msg.format(reward_conditions[env.reward_condition], location_observations[obs[0]], reward_observations[obs[1]], cue_observations[obs[2]]))

num_trials = 10

for t in range(T):

    print(f"observation: {obs}")

    qx = agent.infer_states(obs)

    q_pi, efe = agent.infer_policies()

    print(f"inferred policy: {q_pi}")
    print(f"efe: {efe}")
    action = agent.sample_action()


    print("updating E")
    agent.update_E()

    print(f"ACTION: {action}")
    print(f"Update gamma")
            
    agent.update_gamma()

    msg = """[Step {}] Action: [Move to {}]"""
    print(msg.format(t, location_observations[int(action[0])]))

    obs = env.step(action)

    msg = """[Step {}] Observation: [{},  {}, {}]"""
    print(msg.format(t, location_observations[obs[0]], reward_observations[obs[1]], cue_observations[obs[2]]))

    if obs[1] == 1:
        print("Reward obtained!")