#%%
import os 
os.chdir('/Users/daphne/Desktop/stemai/pymdp')
import numpy as np 
from pymdp import utils
import matplotlib.pyplot as plt
from pymdp import maths
from pymdp.agent import Agent 
import copy

from pymdp.envs import TMaze
reward_probabilities = [0.98, 0.02] # probabilities used in the original SPM T-maze demo
env = TMaze(reward_probs = reward_probabilities)
env._reward_condition = 0

B_gp = env.get_transition_dist()
A_gp = env.get_likelihood_dist()
A_gm = copy.deepcopy(A_gp)

A_gm[0] = A_gp[0][:,:,0]

B_gm = copy.deepcopy(B_gp)

A_factor_list = []


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

gamma_G = 0.95
pA = utils.dirichlet_like(A_gp, scale = 1e16)

pA[1][1:,1:3,:] = 1.0

controllable_indices = [0] # this is a list of the indices of the hidden state factors that are controllable
learnable_modalities = [1] # this is a list of the modalities that you want to be learn-able 

pE = np.array([2.3]*10).astype(float)
from numpy import array
policies = [array([[0, 0],
        [0, 0]]),
 array([[0, 0],
        [1, 0]]),
 array([[0, 0],
        [2, 0]]),
 array([[0, 0],
        [3, 0]]),
 array([[1, 0],
        [1, 0]]),
 array([[2, 0],
        [2, 0]]),
 array([[3, 0],
        [0, 0]]),
 array([[3, 0],
        [1, 0]]),
 array([[3, 0],
        [2, 0]]),
 array([[3, 0],
        [3, 0]])]


pD = [np.array([0.976,0.008,0.008,0.008]), np.array([1,1]).astype(float)]


A_factor_list = [[0],[0,1],[0,1] ]
agent = Agent(A = A_gm,A_factor_list=A_factor_list,B = B_gm, gamma=gamma_G, pE = pE, pD = pD, lr_pD = 0.01, lr_pE = 0.9, use_param_info_gain=True, inference_algo="MMP", inference_horizon=1, policy_len=2, policy_sep_prior=True, policies = policies, factors_to_learn=[1], control_fac_idx=[0])

names = ['center', 'right', 'left', 'cue']
policy_names = []

agent.C[1][1] = 4.0
agent.C[1][2] = -6.0
agent.D[0] = np.array([0.976,0.008,0.008,0.008])

initial_D = copy.deepcopy(agent.D)
#%%
T = 25 # number of timesteps

num_modalities = 3 #location, reward, cue
num_factors = 2 #location (4 locations: center, left, right, down), context (2 contexts: left, right)

# these are useful for displaying read-outs during the loop over time
reward_conditions = ["Right", "Left"]
location_observations = ['CENTER','RIGHT ARM','LEFT ARM','CUE LOCATION']
reward_observations = ['No reward','Reward!','Loss!']
cue_observations = ['Null','Cue Right','Cue Left']

num_trials = 64

#num_trials =32
import time
gamma_G_over_trials = [agent.gamma]
gamma_G_over_timesteps = [agent.gamma]
affective_charge_over_trials = []
affective_charge_per_timestep = []
D_over_trials = []

strongest_prior_belief_about_policies = []

q_pi_over_time = []

qs_over_time = []

policies_over_time = []

observations_over_time = []
REWARD_CONDITION = 0

names = ['center', 'right', 'left', 'cue']
named_policies = []
for p in agent.policies:
    p_loc = p[:,0]
    name = [names[p_loc[0]], names[p_loc[1]]]
    named_policies.append(name)
obs = env.reset(reward_condition=REWARD_CONDITION) # reset the environment and get an initial observation

for trial in range(num_trials):
    timestep = 0

    selected_policy = []

    if trial == 32:
        REWARD_CONDITION = 1

    observations_per_trial = []    
    qs_per_trial = []   
    policy_names_per_trial = []    

    if trial != 0:
        #agent.update_gamma()
        affective_charge_over_trials.append(agent.affective_charge)
        gamma_G_over_trials.append(agent.gamma)
        D_over_trials.append(agent.D[1])
        strongest_prior_belief_about_policies.append(np.max(agent.q_pi))

       #  new_D = copy.deepcopy(agent.D)

       #  new_D[1] = agent.qs.mean(axis=0).mean(axis=0)[1]
       # # agent.reset()
       #  agent.D = new_D
       #  agent.update_D()
       #  agent.update_E()
        agent.update_D()

        print(f"Updated D: {agent.D}")
        
        obs = env.reset(reward_condition=REWARD_CONDITION)
        # initial_qs = agent.D
        # agent.qs = initial_qs 
        # agent.qs_prev = None
        # agent.action = None
        # agent.curr_timestep = 0
        policies_over_time.append(selected_policy)
        observations_over_time.append(observations_per_trial)
        qs_over_time.append(qs_per_trial)
        agent.qs = copy.deepcopy(initial_D)
        agent.action = None
        agent.qs_prev = copy.deepcopy(initial_D)
        agent.prev_obs = []
        agent.reset()
        agent.prev_actions = None
        policy_names.append(policy_names_per_trial)
       #  agent.update_E()

        #

       # agent.set_latest_beliefs(last_belief = agent.D)
    # action = [0,0]
    # env.step(action)

    for timestep in range(3):
        print()

        print(f"trial: {trial}, timestep: {timestep}")

        print(f"observation: {location_observations[obs[0]]}, {reward_observations[obs[1]]}, {cue_observations[obs[2]]}")

        observations_per_trial.append(obs)

        qx = agent.infer_states(obs)
        print(f"qs: {np.mean(np.mean(qx, axis = 0),axis=0)}")

        print(f"F pi :{ [(name, round(agent.F[idx],2)) for idx, name in enumerate(named_policies)]}")

        qs_per_trial.append(qx )
        q_pi, efe = agent.infer_policies()

        print(f"Q pi :{ [(name, round(agent.q_pi[idx],2)) for idx, name in enumerate(named_policies)]}")
        print(f"G :{ [(name, round(agent.G[idx],2)) for idx, name in enumerate(named_policies)]}")
        print(f"E :{ [(name, round(agent.E[idx],2)) for idx, name in enumerate(named_policies)]}")
        print(f"Gamma: {agent.gamma}")
        q_pi_over_time.append(q_pi)
        agent.update_E()

       #  print(f"inferred policy: {q_pi}")
       #  print(f"efe: {efe}")
        action = agent.sample_action()
        if len(policy_names_per_trial) > 0 and policy_names_per_trial[-1] == "right":
            policy_names_per_trial.append('right')
        else:
              policy_names_per_trial.append(names[int(action[0])])
        selected_policy.append(action[0])
        agent.update_gamma()
        gamma_G_over_timesteps.append(agent.gamma)
        affective_charge_per_timestep.append(agent.affective_charge)

       
        print(f"ACTION: {location_observations[int(action[0])]}")
        # print(f"Update gamma")
        # agent.update_gamma()
        # gamma_G_over_trials[trial].append(agent.affective_charge)
    #  msg = """[Step {}] Action: [Move to {}]"""
        #print(msg.format(t, location_observations[int(action[0])]))

        obs = env.step(action)

        # msg = """[Step {}] Observation: [{},  {}, {}]"""
        # print(msg.format(t, location_observations[obs[0]], reward_observations[obs[1]], cue_observations[obs[2]]))

        # if obs[1] == 1:
        #     if wait_time < 10:
        #         wait_time +=1 
        #     else:
        #         print("Reward obtained!")
        #         obs = env.reset()

        #         reward_achieved = True

# %%

plt.plot(gamma_G_over_trials)
plt.ylabel("Gamma G")
plt.title("Gamma G over trials")
plt.xlabel("Trials")
plt.show()
plt.clf()

plt.plot(affective_charge_over_trials)
plt.ylabel("Affective charge")
plt.title("Affective charge over trials")
plt.xlabel("Trials")
plt.show()

plt.plot(affective_charge_per_timestep)
plt.ylabel("Affective charge")
plt.title("Affective charge per time-step")
plt.xlabel("Timestep")
plt.show()


D_to_plot = [D[0] for D in D_over_trials]
plt.plot(D_to_plot)
plt.ylabel("D[0]")
plt.title("D[0] over trials")
plt.xlabel("Trials")
plt.show()

plt.plot(strongest_prior_belief_about_policies)
plt.ylabel("Strongest prior belief about policies")
plt.title("Strongest prior belief about policies over trials")
plt.xlabel("Trials")
plt.show()


unique_policies = np.unique(policy_names, axis =0 )
unique_policies = [list(p) for p in unique_policies]
policy_indices = [unique_policies.index(p) for p in policy_names]
plt.plot(policy_indices)
plt.yticks([0,1, 2, 3], labels =["CL", "CR", "LL", "RR"])
# plt.yticklabels(["CR", "RR"])
plt.ylabel("Policy")
plt.title("Policy over trials")
plt.xlabel("Trials")