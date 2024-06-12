import numpy as np
from pymdp import utils, maths
import copy


def update_gamma_A_MMP(observation, base_A, gamma_A, qs, gamma_A_prior, A_factor_list, update_prior = False, modalities = None, distr_obs = False):
    """
    When using the marginal message passing scheme, the posterior over states is expanded over policies
    and future time steps. 
    So to perform the update, we have to perform the update for each policy separately, and each time point
    and then sum over policies and time points within policies to recover the right form of gamma_A
    
    
    """
    if update_prior:
        new_gamma_A_prior = gamma_A
    else:
        new_gamma_A_prior = gamma_A_prior
    
    expected_A = utils.scale_A_with_gamma(base_A, gamma_A)


    get_factors = lambda q, factor_list: [q[f_idx] for f_idx in factor_list]
    prediction_errors_policy = utils.obj_array(len(qs))
    
    #have to iterate over policies in order to calculate the prediction errors 
    for policy, qs_policy in enumerate(qs):

        prediction_errors_policy_t = utils.obj_array(len(qs_policy))

        for timepoint in range(len(qs_policy)):

            qs_relevant = np.array([get_factors(qs_policy[timepoint], factor_list) for factor_list in A_factor_list], dtype = 'object')

            bold_o_per_modality = utils.obj_array_from_list([maths.spm_dot(expected_A[m], qs_relevant[m]) for m in range(len(base_A))])

            if distr_obs:
                observation_array = utils.obj_array_from_list([observation[m] for m in range(len(base_A))])
            else:
                observation_array = utils.obj_array_from_list([utils.onehot(observation[m], base_A[m].shape[0]) for m in range(len(base_A))])

            prediction_errors = np.array(observation_array) - np.array(bold_o_per_modality)


            prediction_errors_policy_t[timepoint] = prediction_errors
        
        prediction_errors_policy[policy] = np.sum(prediction_errors_policy_t,axis =0)
        
    
    prediction_errors = np.sum(prediction_errors_policy, axis = 0)

    lnA = maths.spm_log_obj_array(base_A)

    # do checking here to make sure pzeta is broadcast-consistent
    
    if modalities is None:
        modalities = range(len(base_A))
    
    if np.isscalar(gamma_A_prior):
        gamma_A_prior = np.array([gamma_A_prior] * len(base_A))
    
    gamma_A_full = utils.obj_array(len(base_A))
    
    for m in range(len(base_A)):
        beta_A_prior = gamma_A_prior[m]

        beta_A_full = copy.deepcopy(beta_A_prior)

        if m not in modalities:

            gamma_A_full[m] =gamma_A_prior[m]
        else:
            #for policy_idx, prediction_errors in enumerate(prediction_errors_policy):
            beta_A_prior = gamma_A_prior[m]
            beta_update_term = (prediction_errors[m] * lnA[m]).sum(axis=0) 

            beta_A_full = beta_A_prior  + 0.3*beta_update_term  

            for idx, s in enumerate(beta_A_full):
                if s < 0.5:
                    beta_A_full[idx] = 0.5 - 10**-5 #set this as a parameter
                if s > 100:
                    beta_A_full[idx] = 100  - 10**-5 #set this as a parameter

            gamma_A_full[m] = np.array(beta_A_full) 
      

    if np.isscalar(gamma_A):
        gamma_A_posterior = sum([gamma_A_m.sum() for gamma_A_m in gamma_A_full])
    elif np.isscalar(gamma_A[0]):
        gamma_A_posterior = np.array([gamma_A_m.sum() for gamma_A_m in gamma_A_full])
    else:
        gamma_A_posterior = gamma_A_full
    
    return np.array(gamma_A_posterior), np.array(new_gamma_A_prior)




def update_gamma_A(observation, base_A, gamma_A, qs, gamma_A_prior, A_factor_list, update_prior = False, modalities = None):

    """
    gamma_A can be:
    - a scalar 
    - a vector of length num_modalities 
    - a list/collection of np.ndarray of len num_modalities, where the m-th element will have shape (num_states[m], num_states[n], num_states[k]) aka A.shape[1:], where
    m, n, k are the indices of the state factors that modality [m] depends on
    """

    if update_prior:
        new_gamma_A_prior = gamma_A
    else:
        new_gamma_A_prior = gamma_A_prior
    
    expected_A = utils.scale_A_with_gamma(base_A, gamma_A)


    get_factors = lambda q, factor_list: [q[f_idx] for f_idx in factor_list]
    qs_relevant = np.array([get_factors(qs, factor_list) for factor_list in A_factor_list], dtype = 'object')

    bold_o_per_modality = utils.obj_array_from_list([maths.spm_dot(expected_A[m], qs_relevant[m]) for m in range(len(base_A))])

    observation_array = utils.obj_array_from_list([utils.onehot(observation[m], base_A[m].shape[0]) for m in range(len(base_A))])

    prediction_errors = np.array(observation_array) - np.array(bold_o_per_modality)

    lnA = maths.spm_log_obj_array(base_A)

    # do checking here to make sure pzeta is broadcast-consistent
    
    if modalities is None:
        modalities = range(len(base_A))

    gamma_A_full = utils.obj_array(len(base_A))
    
    if np.isscalar(gamma_A_prior):
        gamma_A_prior = np.array([gamma_A_prior] * len(base_A))

    for m in range(len(base_A)):
        if m not in modalities:

            gamma_A_full[m] =gamma_A_prior[m]
        else:

            beta_A_prior = gamma_A_prior[m]
            beta_update_term = (prediction_errors[m] * lnA[m]).sum(axis=0) 

            beta_A_full = beta_A_prior  + 0.3*beta_update_term  

            for idx, s in enumerate(beta_A_full):
                if s < 0.5:
                    beta_A_full[idx] = 0.5 - 10**-5 #set this as a parameter
                if s > 100:
                    beta_A_full[idx] = 100  - 10**-5 #set this as a parameter

            gamma_A_full[m] = np.array(beta_A_full) 
    print(f"Gamma A full :{gamma_A_full}")

    if np.isscalar(gamma_A):
        gamma_A_posterior = sum([gamma_A_m.sum() for gamma_A_m in gamma_A_full])
    elif np.isscalar(gamma_A[0]):
        gamma_A_posterior = np.array([gamma_A_m.sum() for gamma_A_m in gamma_A_full])
    else:
        gamma_A_posterior = np.array(gamma_A_full)

    return gamma_A_posterior, new_gamma_A_prior


# E_{Q(s_{t-1, m, n, k}}[P(s_{t,f}|s_{t-1, m}, s_{t-1, n}, ... s_{t-1, k})] # this is what's computed by get_expected_states_with_interactions
# ==> Q(s_{t,f})

def update_gamma_B( q_pi, qs_pi, qs_pi_previous, B, gamma_B, gamma_B_prior, policies, B_factor_list, update_prior=False):
    """
    q_pi: a probability distribution over the actions that i can take

    qs_pi: an object array of length num_policies, 
        where each element in the object array is a list of length policy_horizon, 
        where each element in the list is an object array of shape (num_states[f],) for each factor f

    qs_pi_previous: an object array of length num_policies, 
        where each element in the object array is a list of length policy_horizon, 
        where each element in the list is an object array of shape (num_states[f],) for each factor f

    """

    
    get_factors = lambda q, factor_list: [q[f_idx] for f_idx in factor_list]
    
    # check whether B is ln[E_Q[]
    expected_B = utils.scale_B_with_omega(B, gamma_B)
    
    lnB = maths.spm_log_obj_array(B)
    # do checking here to make sure pzeta is broadcast-consistent
    if np.isscalar(gamma_B_prior):
        gamma_B_prior = np.array([gamma_B_prior] * len(B))
    

    omega_per_policy = utils.obj_array(len(policies))

    for idx, policy in enumerate(policies):

        omega_per_policy_and_time_horizon = utils.obj_array(len(policy))

        for t in range(len(policy)): #iterate over the time horizon of the policy
        
            #right now i am indexing qs_pi_previous[idx][0] but for policy_len > 1 maybe we want to sum over all qs_pi_previous[idx]?
            qs_pi_previous_relevant_factors = np.array([get_factors(qs_pi_previous[idx][t], factor_list) for factor_list in B_factor_list], dtype = 'object')

            omega_per_policy_and_time_horizon[t] = utils.obj_array(len(B))
            for f in range(len(B)):
                s_omega_pi_f = maths.spm_dot(expected_B[f][...,int(policy[0,f])], qs_pi_previous_relevant_factors[f][0][...,None])
                lnB_s_omega_pi_f = maths.spm_dot(lnB[f][...,int(policy[0,f])], qs_pi_previous_relevant_factors[f][0][...,None])
                prediction_errors_f = s_omega_pi_f - qs_pi[idx][0][f][0]
                omega_per_policy_and_time_horizon[t][f] = q_pi[idx] * (prediction_errors_f[...,None] * lnB_s_omega_pi_f[...,None] + gamma_B_prior[f])

        omega_per_policy[idx] = omega_per_policy_and_time_horizon.sum(axis=0) #sum over the time horizon

    gamma_B_summed_over_policies = omega_per_policy.sum(axis=0)

    # how do we contract gamma_A_full in order to be consistent with the original shape of gamma_A and gamma_A_p
    if np.isscalar(gamma_B):

        gamma_B_posterior = sum([beta_omege_f.sum() for beta_omege_f in gamma_B_summed_over_policies])
    elif np.isscalar(gamma_B[0]):
        gamma_B_posterior = [beta_omege_f.sum() for beta_omege_f in gamma_B_summed_over_policies]
    else:
        gamma_B_posterior = gamma_B_summed_over_policies


    if update_prior:
        gamma_B_prior = gamma_B_posterior

    return gamma_B_posterior, gamma_B_prior



def update_gamma_G(G, gamma, q_pi, q_pi_bar, policies):

    affective_charge = 0

    affective_charge = (q_pi - q_pi_bar).dot(G)

    new_beta = (1/gamma) - 10*affective_charge

    return 1 / new_beta, affective_charge

