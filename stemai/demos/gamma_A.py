#%%
import os 
os.chdir('/Users/daphne/Desktop/stemai/pymdp')
import numpy as np 
from pymdp import utils
import matplotlib.pyplot as plt
from pymdp.agent import Agent 

A = utils.obj_array(2)

for modality in range(2):
    A[modality] = np.eye(2)

gamma_A = utils.obj_array(2)

#flat distribution is like 0.005
for modality in range(2):
    gamma_A[modality] = np.array([1.0,1.0])

gamma_A_prior = np.copy(gamma_A)

B = utils.obj_array(1)
for f in range(1):
    B[f] = np.eye(2).reshape((2, 2, 1))

D = utils.obj_array(1)
D[0] = np.array([0.5,0.5])

print(f"D: {D}")
agent = Agent(A = A, B = B,D=D, gamma_A = gamma_A, gamma_A_prior = gamma_A_prior )
agent.qs_over_time = []
print(f"Original A: {agent.A}")

plt.imshow(agent.A[1], cmap = 'grey')
plt.imshow(agent.A[1], cmap = 'grey')
plt.show()
#%%
print(f"Gamma A prior: {agent.gamma_A_prior}")

print(f"Gamma A: {agent.gamma_A}")
agent.observation_history = []
obs = [0,1]


agent.observation_history.append(obs)


print(f"Obs: {obs}")

agent.infer_states(obs)
# agent.D = agent.qs
print(f"QS: {agent.qs}")
print(f"Inferred action: {agent.action}")
agent.qs_over_time.append(agent.qs)
#agent.update_zeta(obs, agent.qs)

print(f"Gamma A prior: {agent.gamma_A_prior}")

print(f"Gamma A: {agent.gamma_A}")
print(f"New A: {agent.A}")

obs = [0,0]

print(f"NEXT OBS: {obs}")
agent.observation_history.append(obs)

agent.infer_states(obs)
# agent.D = agent.qs

print(f"QS: {agent.qs}")
print(f"Inferred action: {agent.action}")

agent.qs_over_time.append(agent.qs)
#agent.update_zeta(agent.observation_history[-1], agent.qs)

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
agent.update_zeta(obs, agent.qs)

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
agent.update_zeta(obs, agent.qs)

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
agent.update_zeta(obs, agent.qs)

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
agent.update_zeta(obs, agent.qs)

print(f"New A: {agent.A}")
print(agent.A[0][0,:])
print(f"Gamma A prior: {agent.gamma_A_prior}")

print(f"Gamma A: {agent.gamma_A}")