#%%
import os 
os.chdir('/Users/daphne/Desktop/stemai/pymdp')
import numpy as np 
from pymdp import utils
import matplotlib.pyplot as plt
from pymdp import maths
from pymdp.agent import Agent 
import copy

from pymdp.envs import TMazeEnvNullOutcome
reward_probabilities = [0.98, 0.02] # probabilities used in the original SPM T-maze demo
env = TMazeEnvNullOutcome(reward_probs = reward_probabilities)

B_gp = env.get_transition_dist()
A_gp = env.get_likelihood_dist()
A_gm = copy.deepcopy(A_gp)

B_gm = utils.obj_array(len(B_gp))

A_gm[1] = np.zeros_like(A_gp[1])

A_gm[1][:,0,0] = [1,0,0]
A_gm[1][:,1,0] = [0, 0.98, 0.02]
A_gm[1][:,2,0] = [0, 0.02,0.98]
A_gm[1][:,3,0] = [1,0,0]

A_gm[1][:,0,1] = [1,0,0]
A_gm[1][:,1,1] = [0, 0.02,0.98]
A_gm[1][:,2,1] = [0, 0.98, 0.02]
A_gm[1][:,3,1] = [1,0,0]

A_gm[2] = np.zeros_like(A_gp[2])

A_gm[2][:,0,0] = [1,0,0]
A_gm[2][:,1,0] = [1,0,0]
A_gm[2][:,2,0] = [1,0,0]
A_gm[2][:,3,0] = [0,1,0]

A_gm[2][:,0,1] = [1,0,0]
A_gm[2][:,1,1] = [1,0,0]
A_gm[2][:,2,1] = [1,0,0]
A_gm[2][:,3,1] = [0,0,1]

B_gm[0] = np.zeros_like(B_gp[0])

B_gm[0][:,0,0,] = [1,0,0,0]
B_gm[0][:,0,1] = [0,1,0,0]
B_gm[0][:,0,2] = [0,0,1,0]
B_gm[0][:,0,3] = [1,0,0,0]

B_gm[0][:,1,0] = [0,1,0,0]
B_gm[0][:,1,1] = [0,1,0,0]
B_gm[0][:,1,2] = [0,0,1,0]
B_gm[0][:,1,3] = [0,1,0,0]

B_gm[0][:,2,0] = [0,0,1,0]
B_gm[0][:,2,1] = [0,1,0,0]
B_gm[0][:,2,2] = [0,0,1,0]
B_gm[0][:,2,3] = [0,0,1,0]

B_gm[0][:,3,0] = [0,0,0,1]
B_gm[0][:,3,1] = [0,1,0,0]
B_gm[0][:,3,2] = [0,0,1,0]
B_gm[0][:,3,3] = [0,0,0,1]

B_gm[1] = copy.deepcopy(B_gp[1])

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

controllable_indices = [0] # this is a list of the indices of the hidden state factors that are controllable
learnable_modalities = [1] # this is a list of the modalities that you want to be learn-able 

pE = np.array([2.3]*16).astype(float)
agent = Agent(A = A_gm,B = B_gm, gamma=gamma_G, pE = pE, modalities_to_learn=learnable_modalities,lr_pA = 0.25, use_param_info_gain=True, inference_algo="MMP", inference_horizon=2, policy_len=2, policy_sep_prior=True)


agent.C[1][1] = 4.0
agent.C[1][2] = -6.0
agent.D[0] = [0.976,0.008,0.008,0.008]

T = 25 # number of timesteps

initial_obs = env.reset() # reset the environment and get an initial observation

# these are useful for displaying read-outs during the loop over time
reward_conditions = ["Right", "Left"]
location_observations = ['CENTER','RIGHT ARM','LEFT ARM','CUE LOCATION']
reward_observations = ['No reward','Reward!','Loss!']
cue_observations = ['Cue Right','Cue Left']

num_trials = 32
import time
gamma_G_over_trials = [[]]*num_trials

obs = initial_obs
assert env.reward_condition == 1
for trial in range(num_trials):
    reward_achieved = False
    agent.qs = agent.D
    agent.qs_prev = None
    gamma_G_over_trials[trial] = []
    wait_time = 0
    while not reward_achieved:

        print(f"observation: {obs}")

        qx = agent.infer_states(obs)

        q_pi, efe = agent.infer_policies()

        print(f"inferred policy: {q_pi}")
        print(f"efe: {efe}")
        action = agent.sample_action()


        print("updating E")
        #agent.update_E()
        policy = np.argmax(q_pi)
        agent.pE[policy] += 1
        agent.E = utils.norm_dist(agent.pE)
        
        

        print(f"ACTION: {action}")
        print(f"Update gamma")
                
        agent.update_gamma()
        gamma_G_over_trials[trial].append(agent.affective_charge)

    #  msg = """[Step {}] Action: [Move to {}]"""
        #print(msg.format(t, location_observations[int(action[0])]))

        obs = env.step(action)

        # msg = """[Step {}] Observation: [{},  {}, {}]"""
        # print(msg.format(t, location_observations[obs[0]], reward_observations[obs[1]], cue_observations[obs[2]]))

        if obs[1] == 1:
            if wait_time < 10:
                wait_time +=1 
            else:
                print("Reward obtained!")
                obs = env.reset()

                reward_achieved = True

# %%

flattened_gamma_G = [item for sublist in gamma_G_over_trials for item in sublist]
plt.plot(flattened_gamma_G)
plt.ylabel("Affective charge")
plt.title("Affective charge within time-steps of each trial")
plt.xlabel("Time-step")