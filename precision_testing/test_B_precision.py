#%%

import os 
os.chdir('/Users/daphne/Desktop/stemai/pymdp')
import numpy as np 
from pymdp import utils
import matplotlib.pyplot as plt
from pymdp import maths
from pymdp.agent import Agent 
from precision_testing.precision_utils import *

import copy



num_obs = [2]
num_states = [2]
num_actions = [2]


B = build_uniform_B_matrix(num_states, num_actions)
base_p = measure_one_hotness(B)


precision_differences = []
gammas = np.linspace(0.01, 10.0, 9)
Bs = [copy.deepcopy(B)]

for gamma_B in gammas:
    scaled_B = utils.scale_B_with_gamma(copy.deepcopy(B), gamma_B)
    assert utils.is_normalized(scaled_B)
    p = measure_one_hotness(scaled_B)
    precision_difference = np.array(base_p) - np.array(p)
    precision_differences.append(precision_difference[0].sum())
    Bs.append(scaled_B)

import matplotlib.pyplot as plt
plt.scatter(gammas, precision_differences)
plt.xlabel("Gamma B")
plt.ylabel("Precision Difference")
plt.hlines(y = 0, xmin = 0, xmax = max(gammas), color='black', linestyle='--')
plt.xticks(gammas)
plt.savefig('precision_demo/precision_differences_scalar_gamma_B.png')
#%%
# Set up subplots
num_plots = len(Bs)
num_cols = 2  # Number of columns in the subplot grid
num_rows = -(-num_plots // num_cols)  # Calculate the number of rows (ceiling division)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

gammas = np.array([0] + list(gammas))
for idx, (b, gamma) in enumerate(zip(Bs, gammas)):
    row_idx = idx // num_cols  # Calculate the row index
    col_idx = idx % num_cols  # Calculate the column index
    ax = axes[row_idx, col_idx] if num_plots > 1 else axes  # Handle single subplot case
    
    # Plot the A matrix
    ax.imshow(b[0], vmin=0, vmax=1, cmap='gray')
    if idx > 0:
        ax.set_title(f'B with precision: {gamma.round(2)}')
    else:
        ax.set_title('Original B')
    ax.axis('off')
plt.suptitle("B matrix increasing precision over whole array")
plt.savefig('precision_demo/B_matrices_scalar_gamma.png')


# %%

gamma_B = np.ones((2,2, 2))
Bs = [copy.deepcopy(B)]
precision_differences = []
gammas = np.linspace(0.01, 2.0, 9)
gammas_2 = np.linspace(0.05, 4.0, 9)
for idx in range(len(gammas)):
    g1 = gammas[idx]
    g2 = gammas_2[idx]
    gamma_B[:,1,0] = [g1, g1]
    gamma_B[:,1,1] = [g2, g2]

    scaled_A = utils.scale_A_with_gamma(copy.deepcopy(B), gamma_B)
    p = measure_one_hotness(scaled_A)
    precision_difference = np.array(base_p) - np.array(p)
    precision_differences.append(precision_difference[0].sum())
    Bs.append(scaled_A)


import matplotlib.pyplot as plt
plt.scatter(gammas, precision_differences)
plt.xlabel("Gamma")
plt.ylabel("Precision Difference")
plt.hlines(y = 0, xmin = 0, xmax = 2, color='black', linestyle='--')
plt.xticks(gammas)
plt.title("B matrix increasing precision over second column")

plt.savefig('precision_demo/precision_differences_array_gamma_B_2.png')
#%%
# Set up subplots
num_plots = len(Bs)
num_cols = 2  # Number of columns in the subplot grid
num_rows = -(-num_plots // num_cols)  # Calculate the number of rows (ceiling division)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

gammas = np.array([0] + list(gammas))
for idx, (b, gamma) in enumerate(zip(Bs, gammas)):
    row_idx = idx // num_cols  # Calculate the row index
    col_idx = idx % num_cols  # Calculate the column index
    ax = axes[row_idx, col_idx] if num_plots > 1 else axes  # Handle single subplot case
    
    # Plot the A matrix
    ax.imshow(b[0][:,:,0], vmin=0, vmax=1, cmap='gray')
    if idx > 0:
        ax.set_title(f'B with precision: {gamma.round(2)}')
    else:
        ax.set_title('Original B')
    ax.axis('off')
plt.suptitle("B matrix increasing precision over second column, first action")
plt.savefig('precision_demo/B_matrices_array_gamma_2_action_1.png')

# %%
num_plots = len(Bs)
num_cols = 2  # Number of columns in the subplot grid
num_rows = -(-num_plots // num_cols)  # Calculate the number of rows (ceiling division)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

gammas = np.array([0] + list(gammas))
for idx, (b, gamma) in enumerate(zip(Bs, gammas)):
    row_idx = idx // num_cols  # Calculate the row index
    col_idx = idx % num_cols  # Calculate the column index
    ax = axes[row_idx, col_idx] if num_plots > 1 else axes  # Handle single subplot case
    
    # Plot the A matrix
    ax.imshow(b[0][:,:,1], vmin=0, vmax=1, cmap='gray')
    if idx > 0:
        ax.set_title(f'B with precision: {gamma.round(2)}')
    else:
        ax.set_title('Original B')
    ax.axis('off')
plt.suptitle("B matrix increasing precision over second column, second action")
plt.savefig('precision_demo/B_matrices_array_gamma_2_action_2.png')

# %%
