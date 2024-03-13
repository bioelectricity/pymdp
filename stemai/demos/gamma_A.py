# %%
import os

os.chdir("/Users/daphne/Desktop/stemai/pymdp")
import numpy as np
from pymdp import utils
from pymdp.agent import Agent

A = utils.obj_array(2)

for modality in range(2):
    A[modality] = np.eye(2)

gamma_A = utils.obj_array(2)

# flat distribution is like 0.005
for modality in range(2):
    gamma_A[modality] = 1.0

gamma_A_prior = np.copy(gamma_A)

B = utils.obj_array(1)
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
agent.observation_history.append(obs)

agent.infer_states(obs)
print(f"QS: {agent.qs}")
agent.qs_over_time.append(agent.qs)
agent.update_zeta(agent.observation_history[0], agent.qs)

print(f"New A: {agent.A}")
print(agent.A[0][0, :])
print(f"Gamma A: {agent.beta_zeta}")
