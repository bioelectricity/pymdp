#%%


import os 
os.chdir('/Users/daphne/Desktop/stemai/pymdp')
import numpy as np 
from pymdp import utils
import matplotlib.pyplot as plt
from pymdp import maths
from pymdp.agent import Agent 
from pymdp.precision_updates import update_gamma_A
from precision_testing.precision_utils import *
import copy




num_obs = [2]
num_states = [2]
num_actions = [1]

A = build_uniform_A_matrix(num_obs, num_states)
B = build_uniform_B_matrix(num_states, num_actions)

def precision_learning(gamma_value):

    gamma_A = np.ones((1,2))
    gamma_A[:] = gamma_value
    gamma_A_prior = copy.deepcopy(gamma_A)

    observation = [1]

    A_for_observation = A[0][0]

    current_s_for_observation = np.argmax(A_for_observation)


    A_factor_list = [[0]]
    qs = utils.obj_array(len(num_states))

    qs_values = np.linspace(0,1.0,9)

    all_As = [A]
    all_ps = []
    all_gammas = []
    all_qs = []
    for qs_v in qs_values:
        qs[0] = np.array([0.0,0.0])
        qs[0][current_s_for_observation] = qs_v
        qs[0][np.abs(1-current_s_for_observation)] = 1 - qs_v
        all_qs.append(qs[0])


        gamma_A_posterior, new_gamma_A_prior = update_gamma_A(observation, copy.deepcopy(A), gamma_A, qs, gamma_A_prior, A_factor_list, update_prior = False, modalities = None)

        # print(f"Previous gamma_A: {gamma_A}")
        # print(f"Previous A: {scaled_A}")
        # print("New gamma_A: ", gamma_A_posterior)

        new_A = utils.scale_A_with_gamma(copy.deepcopy(A), gamma_A_posterior)
        p = measure_one_hotness(new_A)
        all_ps.append(p)
        all_gammas.append(gamma_A_posterior)

        #print(f"New A: {new_A}")

        all_As.append(new_A)
    return all_As, all_ps, all_gammas, all_qs, qs_values, current_s_for_observation




gamma_values = np.linspace(0.5, 1.5, 9)
fig, axes = plt.subplots(1,2, figsize = (15,6))

for gamma_value in gamma_values:
    all_As, all_ps, all_gammas, all_qs, qs_values, current_s_for_observation = precision_learning(gamma_value)



    axes[0].scatter(np.array(all_qs)[:,0], [x[0] for x in np.array(all_gammas)[:,0]], label = 'gamma: {}'.format(gamma_value.round(2)))
    axes[1].scatter(np.array(all_qs)[:,0], [x[1] for x in np.array(all_gammas)[:,0]], label = 'gamma: {}'.format(gamma_value.round(2)))
    axes[0].set_title("State 0")
    axes[1].set_title("State 1")
    axes[0].set_xlabel("Qs - state 0")
    axes[0].set_ylabel("Gamma_A per state")
    axes[1].set_xlabel("Qs - state 0")
    axes[1].set_ylabel("Gamma_A per state")

plt.legend()
plt.suptitle(f"Precision update for observation with evidence for state {current_s_for_observation}")


#%%
num_plots = len(all_As)
num_cols = 2  # Number of columns in the subplot grid
num_rows = -(-num_plots // num_cols)  # Calculate the number of rows (ceiling division)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

qs_values = np.array([0] + list(qs_values))
for idx, (a, q) in enumerate(zip(all_As, qs_values)):
    row_idx = idx // num_cols  # Calculate the row index
    col_idx = idx % num_cols  # Calculate the column index
    ax = axes[row_idx, col_idx] if num_plots > 1 else axes  # Handle single subplot case
    
    # Plot the A matrix
    ax.imshow(a[0], vmin=0, vmax=1, cmap='gray')
    if idx > 0:
        ax.set_title(f'qs: {q.round(2)}')
    else:
        ax.set_title('Original A')
    ax.axis('off')
# plt.suptitle("A matrix increasing precision over second column")
# plt.savefig('precision_demo/A_matrices_array_gamma_2.png')


# %%

#TODO 
#test update_gamma_A_MMP
#test all of the above given the agent class and make sure its working