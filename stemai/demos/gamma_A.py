<<<<<<< HEAD
#%%

#TODO 
#move bounds of precision to 0.1 and 10


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

=======
# %%
import os

os.chdir("/Users/daphne/Desktop/stemai/pymdp")
import numpy as np
from pymdp import utils
from pymdp.agent import Agent
>>>>>>> 8188d5581b557d5995f45caaf938d6c97c8b0116

#generative model 

num_modalities = 3 #location, reward, cue
num_factors = 2 #location, context


#%%
A = utils.obj_array(2)

for modality in range(2):
    A[modality] = maths.softmax(3.0*np.eye(2))


gamma_A = utils.obj_array(2)

# flat distribution is like 0.005
for modality in range(2):
    gamma_A[modality] = np.array([1.0,1.0])

gamma_A_prior = np.copy(gamma_A)



B = utils.obj_array(1)
<<<<<<< HEAD
for f in range(1):
    B[f] = np.eye(2).reshape((2, 2, 1))

D = utils.obj_array(1)
D[0] = np.array([0.5,0.5])

print(f"D: {D}")
agent = Agent(A = A, B = B,D=D, gamma_A = gamma_A, gamma_A_prior = gamma_A_prior )
agent.qs_over_time = []
print(f"Original A: {agent.A}")
#%%
plt.imshow(agent.A[1], cmap = 'grey')
plt.imshow(agent.A[1], cmap = 'grey')
plt.show()
#%%
print(f"Gamma A prior: {agent.gamma_A_prior}")

print(f"Gamma A: {agent.gamma_A}")
agent.observation_history = []
obs = [0,1]


=======
B[0] = np.zeros((2, 2, 2))
for action in range(2):
    B[0][:, :, action] = np.full((2, 2), 1 / 2)

D = utils.obj_array(1)
D[0] = np.array([1, 0])
agent = Agent(A=A, B=B, D=D, beta_zeta=gamma_A, beta_zeta_prior=gamma_A_prior)
agent.qs_over_time = []
print(f"Original A: {agent.A}")
print(agent.A[0][0, :])
print(f"Gamma A: {agent.beta_zeta}")
agent.observation_history = []
obs = [0, 1]
>>>>>>> 8188d5581b557d5995f45caaf938d6c97c8b0116
agent.observation_history.append(obs)


print(f"Obs: {obs}")

agent.infer_states(obs)
# agent.D = agent.qs
print(f"QS: {agent.qs}")
print(f"Inferred action: {agent.action}")
agent.qs_over_time.append(agent.qs)
#agent.update_gamma_A(obs, agent.qs)

print(f"Gamma A prior: {agent.gamma_A_prior}")

print(f"Gamma A: {agent.gamma_A}")
print(f"New A: {agent.A}")

obs = [0,1]

print(f"NEXT OBS: {obs}")
agent.observation_history.append(obs)

agent.infer_states(obs)
# agent.D = agent.qs

print(f"QS: {agent.qs}")
print(f"Inferred action: {agent.action}")

agent.qs_over_time.append(agent.qs)
agent.update_gamma_A(agent.observation_history[-1], agent.qs)

print(f"Gamma A prior: {agent.gamma_A_prior}")

print(f"Gamma A: {agent.gamma_A}")
print(f"New A: {agent.A}")


obs = [1,0]
print(f"NEXT OBS : {obs}")

agent.observation_history.append(obs)

agent.infer_states(obs)
# agent.D = agent.qs

print(f"QS: {agent.qs}")
print(f"Inferred action: {agent.action}")

agent.qs_over_time.append(agent.qs)
agent.update_gamma_A(obs, agent.qs)

print(f"New A: {agent.A}")
print(agent.A[0][0,:])
print(f"Gamma A prior: {agent.gamma_A_prior}")

print(f"Gamma A: {agent.gamma_A}")


obs = [0,0]
print(f"NEXT OBS : {obs}")

agent.observation_history.append(obs)

agent.infer_states(obs)
agent.infer_policies()
agent.sample_action()
print(f"QS: {agent.qs}")
print(f"Inferred action: {agent.action}")

agent.qs_over_time.append(agent.qs)
agent.update_gamma_A(obs, agent.qs)

print(f"New A: {agent.A}")
print(agent.A[0][0,:])
print(f"Gamma A prior: {agent.gamma_A_prior}")

print(f"Gamma A: {agent.gamma_A}")


obs = [1,1]
print(f"NEXT OBS : {obs}")

agent.observation_history.append(obs)

agent.infer_states(obs)
agent.infer_policies()
agent.sample_action()
print(f"QS: {agent.qs}")
print(f"Inferred action: {agent.action}")

agent.qs_over_time.append(agent.qs)
agent.update_gamma_A(obs, agent.qs)

print(f"New A: {agent.A}")
print(agent.A[0][0,:])
print(f"Gamma A prior: {agent.gamma_A_prior}")

print(f"Gamma A: {agent.gamma_A}")


obs = [1,0]
print(f"NEXT OBS : {obs}")

agent.observation_history.append(obs)

agent.infer_states(obs)
# agent.D = agent.qs

print(f"QS: {agent.qs}")
print(f"Inferred action: {agent.action}")

agent.qs_over_time.append(agent.qs)
agent.update_gamma_A(obs, agent.qs)

print(f"New A: {agent.A}")
<<<<<<< HEAD
print(agent.A[0][0,:])
print(f"Gamma A prior: {agent.gamma_A_prior}")

print(f"Gamma A: {agent.gamma_A}")
# %%
=======
print(agent.A[0][0, :])
print(f"Gamma A: {agent.beta_zeta}")
>>>>>>> 8188d5581b557d5995f45caaf938d6c97c8b0116
