#%%


import os 
#os.chdir('/Users/daphne/Desktop/stemai/pymdp')
import numpy as np 
from pymdp import utils
import matplotlib.pyplot as plt
from pymdp import maths
from pymdp.agent import Agent 
from precision_testing.precision_utils import *
import copy


"""
First we test gamma_A

gamma_A can take on three different forms

1) a scalar 
2) a vector of length num_modalities 
3) a list/collection of np.ndarray of len num_modalities, where the m-th element will have shape (num_states[m], num_states[n], num_states[k]) aka A.shape[1:], where
    m, n, k are the indices of the state factors that modality [m] depends on


"""




num_obs = [2]
num_states = [2]
num_actions = [1]

A = build_uniform_A_matrix(num_obs, num_states)

base_p = measure_one_hotness(A)

precision_differences = []
gammas = np.linspace(0.01, 2.0, 9)
As = [copy.deepcopy(A)]

for gamma_A in gammas:
    scaled_A = utils.scale_A_with_gamma(copy.deepcopy(A), [gamma_A])
    assert utils.is_normalized(scaled_A)
    p = measure_one_hotness(scaled_A)
    precision_difference = np.array(base_p) - np.array(p)
    precision_differences.append(precision_difference[0].sum())
    As.append(scaled_A)

import matplotlib.pyplot as plt
plt.scatter(gammas, precision_differences)
plt.xlabel("Gamma")
plt.ylabel("Precision Difference")
plt.hlines(y = 0, xmin = 0, xmax = 2, color='black', linestyle='--')
plt.xticks(gammas)
plt.savefig('precision_demo/precision_differences_scalar_gamma_A.png')
#%%
# Set up subplots
num_plots = len(As)
num_cols = 2  # Number of columns in the subplot grid
num_rows = -(-num_plots // num_cols)  # Calculate the number of rows (ceiling division)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

gammas = np.array([0] + list(gammas))
for idx, (a, gamma) in enumerate(zip(As, gammas)):
    row_idx = idx // num_cols  # Calculate the row index
    col_idx = idx % num_cols  # Calculate the column index
    ax = axes[row_idx, col_idx] if num_plots > 1 else axes  # Handle single subplot case
    
    # Plot the A matrix
    ax.imshow(a[0], vmin=0, vmax=1, cmap='gray')
    if idx > 0:
        ax.set_title(f'A with precision: {gamma.round(2)}')
    else:
        ax.set_title('Original A')
    ax.axis('off')
plt.suptitle("A matrix increasing precision over whole array")
plt.savefig('precision_demo/A_matrices_scalar_gamma.png')


# %%

gamma_A = np.ones((2,2))
As = [copy.deepcopy(A)]
precision_differences = []
gammas = np.linspace(0.01, 2.0, 9)
for gamma in gammas:
    gamma_A[:,0] = [gamma, gamma]

    scaled_A = utils.scale_A_with_gamma(copy.deepcopy(A), [gamma_A])
    p = measure_one_hotness(scaled_A)
    precision_difference = np.array(base_p) - np.array(p)
    precision_differences.append(precision_difference[0].sum())
    As.append(scaled_A)


import matplotlib.pyplot as plt
plt.scatter(gammas, precision_differences)
plt.xlabel("Gamma")
plt.ylabel("Precision Difference")
plt.hlines(y = 0, xmin = 0, xmax = 2, color='black', linestyle='--')
plt.xticks(gammas)
plt.title("A matrix increasing precision over first column")

plt.savefig('precision_demo/precision_differences_array_gamma_A.png')
#%%
# Set up subplots
num_plots = len(As)
num_cols = 2  # Number of columns in the subplot grid
num_rows = -(-num_plots // num_cols)  # Calculate the number of rows (ceiling division)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

gammas = np.array([0] + list(gammas))
for idx, (a, gamma) in enumerate(zip(As, gammas)):
    row_idx = idx // num_cols  # Calculate the row index
    col_idx = idx % num_cols  # Calculate the column index
    ax = axes[row_idx, col_idx] if num_plots > 1 else axes  # Handle single subplot case
    
    # Plot the A matrix
    ax.imshow(a[0], vmin=0, vmax=1, cmap='gray')
    if idx > 0:
        ax.set_title(f'A with precision: {gamma.round(2)}')
    else:
        ax.set_title('Original A')
    ax.axis('off')
plt.suptitle("A matrix increasing precision over first column")
plt.savefig('precision_demo/A_matrices_array_gamma.png')

# %%

gamma_A = np.ones((2,2))
As = [copy.deepcopy(A)]
precision_differences = []
gammas = np.linspace(0.01, 2.0, 9)
for gamma in gammas:
    gamma_A[:,1] = [gamma, gamma]

    scaled_A = utils.scale_A_with_gamma(copy.deepcopy(A), gamma_A)
    p = measure_one_hotness(scaled_A)
    precision_difference = np.array(base_p) - np.array(p)
    precision_differences.append(precision_difference[0].sum())
    As.append(scaled_A)


import matplotlib.pyplot as plt
plt.scatter(gammas, precision_differences)
plt.xlabel("Gamma")
plt.ylabel("Precision Difference")
plt.hlines(y = 0, xmin = 0, xmax = 2, color='black', linestyle='--')
plt.xticks(gammas)
plt.title("A matrix increasing precision over second column")

plt.savefig('precision_demo/precision_differences_array_gamma_A_2.png')
#%%
# Set up subplots
num_plots = len(As)
num_cols = 2  # Number of columns in the subplot grid
num_rows = -(-num_plots // num_cols)  # Calculate the number of rows (ceiling division)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

gammas = np.array([0] + list(gammas))
for idx, (a, gamma) in enumerate(zip(As, gammas)):
    row_idx = idx // num_cols  # Calculate the row index
    col_idx = idx % num_cols  # Calculate the column index
    ax = axes[row_idx, col_idx] if num_plots > 1 else axes  # Handle single subplot case
    
    # Plot the A matrix
    ax.imshow(a[0], vmin=0, vmax=1, cmap='gray')
    if idx > 0:
        ax.set_title(f'A with precision: {gamma.round(2)}')
    else:
        ax.set_title('Original A')
    ax.axis('off')
plt.suptitle("A matrix increasing precision over second column")
plt.savefig('precision_demo/A_matrices_array_gamma_2.png')

# %%
