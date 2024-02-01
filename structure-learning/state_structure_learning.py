#%%
"""Here we will try to mimic the model for MNIST"""
import os 
import sys 
import pathlib 
path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)
import pymdp 
from pymdp.agent import Agent
from pymdp import utils 
from pymdp import maths
import numpy as np
#initial model 

def calc_delta_F(agent_pA, expanded_agent_pA, agent_pA_previous, expanded_agent_pA_previous, observation_discrete):

    num_modalities = len(agent_pA)

    reduced_beta_posterior = np.array([agent_pA[m].sum(axis = 0)[observation_discrete] for m in range(num_modalities)])

    reduced_beta_prior = np.array([agent_pA_previous[m].sum(axis = 0)[observation_discrete] for m in range(num_modalities)])

    expanded_beta_posterior = np.array([expanded_agent_pA[m].sum(axis=0)[observation_discrete] for m in range(num_modalities)])
    
    expanded_beta_prior = np.array([expanded_agent_pA_previous[m].sum(axis = 0)[observation_discrete] for m in range(num_modalities)])

    delta_F = maths.spm_log_single(expanded_beta_posterior) + maths.spm_log_single(reduced_beta_prior) - maths.spm_log_single(expanded_beta_prior) - maths.spm_log_single(reduced_beta_posterior) 

    print(f"Delta F : {delta_F}")
    return delta_F


def add_state(agent, observation):
    last_factor_index = agent.num_factors - 1
    new_num_states = agent.num_states #[num states in factor1, num states in factor2]
    new_num_states[last_factor_index] += 1 #new state in the last factor 
    agent.num_states = new_num_states 


    new_pB = utils.obj_array(agent.num_factors)
    for f in range(agent.num_factors):
        new_pB[f] = np.zeros((new_num_states[f], new_num_states[f], 1))
        if f == last_factor_index:
            new_pB[f][:,:,0] = np.eye(new_num_states[f])
        else:
            new_pB[f] = agent.pB[f]
    
    agent.pB = new_pB
    agent.B = utils.norm_dist_obj_arr(new_pB)

    new_pA = utils.obj_array(agent.num_modalities) #dirichlet prior over A
    #likelihood 
    for m in range(agent.num_modalities):
        new_pA[m] = np.copy(agent.pA[m])

        #initialize new pA to have a new state with 1/16 
        new_pA[m] = np.insert(new_pA[m], last_factor_index, 1/16, axis=-1)
        
        #add the observation value to each observation in the new pA
        obs_idx = 0 if observation[m][0] < 0.5 else 1
            
        new_pA[m][obs_idx,-1] += observation[m][0]
    
    agent.pA = new_pA
    agent.A = utils.norm_dist_obj_arr(new_pA)

    # New state is the initial state
    agent.D = utils.obj_array_uniform(new_num_states)


    agent.policies = agent._construct_policies() #do we need to update the pymdp num_controls variable?
    agent.E = agent._construct_E_prior()

    return agent

def mutual_information(matrix):
    # Convert the matrix to a joint probability distribution
    joint_prob = matrix / np.sum(matrix)
    
    # Calculate the marginal probabilities
    marginal_x = np.sum(joint_prob, axis=1)
    marginal_y = np.sum(joint_prob, axis=0)
    
    # Compute the mutual information
    mi = 0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (marginal_x[i] * marginal_y[j]))
    
    return mi

import numpy as np

# def mutual_information_1(matrix):
#     """
#     Calculate the mutual information for a matrix where each dimension
#     represents a different random variable.

#     :param matrix: A 2D numpy array representing joint frequencies or probabilities.
#     :return: Mutual information value.
#     """
#     # Ensure the matrix represents a probability distribution
#     if np.sum(matrix) != 1:
#         matrix = matrix / np.sum(matrix)

#     # Calculate the marginal probabilities
#     marginal_x = np.sum(matrix, axis=1)
#     marginal_y = np.sum(matrix, axis=0)

#     # Compute the mutual information
#     mi = 0
#     rows, cols = matrix.shape
#     for i in range(rows):
#         for j in range(cols):
#             if matrix[i, j] > 0:  # To avoid log(0)
#                 mi += matrix[i, j] * np.log(matrix[i, j] / (marginal_x[i] * marginal_y[j]))

#     return mi





def compare_F_and_G_orginal_expanded(original_agent, expanded_agent, observation_discrete):#, original_pA_previous, expanded_pA_previous):
    """Compute delta F (Log Bayes Factor) between original and expanded model. 
    If the expanded model has lower F, compute Delta G betwen original and expanded. 
    If expanded model has lower G, then return expanded model as the best model
    """


    #marginal likelihood of original model for the new observation
    ML_original_model = np.array([original_agent.pA[m].sum(axis=-1)[observation_discrete[m]] for m in range(len(A))])

    #print(f"Original ML: {ML_original_model[0]}")


    #marginal likelihood of expanded model for the new observation
    ML_expanded_model = np.array([expanded_agent.pA[m].sum(axis=-1)[observation_discrete[m]] for m in range(len(A))])
    #print(f"Expanded ML: {ML_expanded_model[0]}")

    delta_F = ML_original_model - ML_expanded_model

    #delta_F = calc_delta_F(original_agent.pA, expanded_agent.pA, )

    print(f"Delta_F: {delta_F}")

    #compare marginal likelihood of original and expanded model 
    if delta_F.sum() < 0:
        
        #Compare expected free energy 

        MI_original_model = np.array([mutual_information(original_agent.A[m]) for m in range(agent.num_modalities) ] ).sum()  # mutual information for original model 

        MI_expanded_model = np.array([mutual_information(expanded_agent.A[m]) for m in range(agent.num_modalities) ] ).sum()  # mutual information for original model 

        delta_G = MI_original_model - MI_expanded_model

        print(f"Delta G: {delta_G}")
        
        if delta_G > 0:
            
            return True

    
    return False 
        
def run(observation_stream, discrete_observation_stream, agent):


    qs = agent.infer_states(discrete_observation_stream) #infer the hidden states given the observations

    inferred_state = np.argmax(qs[0])
    print(f"Inferred state: {np.argmax(qs[0])}")


    #Option 1: add a new state to the state space 
    expanded_agent = Agent(A=np.copy(agent.A), B=np.copy(agent.B), D=np.copy(agent.D), pA = np.copy(agent.pA))
    expanded_agent = add_state(expanded_agent, observation_stream)

    #Option 2: Original model: add (s,o) pair to existing state 
    #update the pA assuming that this is the correct state 
    print(f"Old pA: {agent.pA[0][:10]}")
    for m in range(agent.num_modalities):
        o = discrete_observation_stream[m][0]
        agent.pA[m][o,inferred_state] += observation_stream[m][0]

    print(f"New pA: {agent.pA[0][:10]}")

    agent.A = utils.norm_dist_obj_arr(agent.pA)


    expand_model = compare_F_and_G_orginal_expanded(agent, expanded_agent, discrete_observation_stream)

    print(f"Expand model? {expand_model}")

    if expand_model:
        agent = expanded_agent

    return agent

    
num_factors = 1
num_states = [1]

num_modalities = 784
num_obs = [2] * 784

num_paths = [1]

pA = utils.obj_array(num_modalities) #dirichlet prior over A

#initialize our A matrix with a set number of observation modalities (and observations) and one state per state factor (an identity mapping from observations to states)
for m in range(num_modalities):
    pA[m] = np.zeros((num_obs[m], num_states[0]))
    pA[m] += 1/16 #populate each dirichlet count for each observation with 1/16
print(f"Initial pA (subset): {pA[0][:5]}")

A = utils.norm_dist_obj_arr(pA) #generate the categorical A out of pA



#initialize a precise B with only one path 
B = utils.obj_array(num_factors)
B[0] = np.zeros((num_states[0], num_states[0], num_paths[0]))
B[0][0,0,0] = 1

D = utils.obj_array_uniform(num_states)

agent = Agent(A=A, B=B, D=D, pA = pA, pB = B)

#round the observation to 0 or 1

for t in range(100):
    print(f"Timestep: {t}")


    print(agent.pA[0].shape)
    print(agent.A[0].shape)
    print(f"D: {agent.D}")
    observation_stream = []
    for i in range(784):
        observation_stream.append([np.random.uniform(0,1)]) #random numbers between 0 and 1 for each pixel

    discrete_observation_stream = [[0] if obs[0] < 0.5 else [1] for obs in observation_stream] #discretize the observation stream

    agent = run(observation_stream, discrete_observation_stream, agent)

    print(agent.pA[0].shape)
    print(agent.A[0].shape)
# %%
