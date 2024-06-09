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

# %%
gamma_B = np.ones((2,2, 2))
