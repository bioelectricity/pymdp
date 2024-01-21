def update_omega( q_pi, qs_pi, qs_pi_previous, B, beta_omega, beta_omega_prior, policies, B_factor_list, update_prior=False):
    """
    q_pi: a probability distribution over the actions that i can take

    qs_pi: an object array of length num_policies, 
        where each element in the object array is a list of length policy_horizon, 
        where each element in the list is an object array of shape (num_states[f],) for each factor f

    qs_pi_previous: an object array of length num_policies, 
        where each element in the object array is a list of length policy_horizon, 
        where each element in the list is an object array of shape (num_states[f],) for each factor f

    """

    
    if B_factor_list is not None:
        get_factors = lambda q, factor_list: [q[f_idx] for f_idx in factor_list]
    
    # check whether B is ln[E_Q[]
    expected_B = utils.scale_B_with_omega(B, beta_omega)
    
    lnB = maths.spm_log_obj_array(B)
    # do checking here to make sure pzeta is broadcast-consistent
    if np.isscalar(beta_omega_prior):
        beta_omega_prior = np.array([beta_omega_prior] * len(B))
    

    omega_per_policy = utils.obj_array(len(policies))

    for idx, policy in enumerate(policies):
        qs_pi_previous_relevant_factors = np.array([get_factors(qs_pi_previous[idx][0], factor_list) for factor_list in B_factor_list], dtype = 'object')

        omega_per_policy[idx] = utils.obj_array(len(B))
        for f in range(len(B)):
            s_omega_pi_f = maths.spm_dot(expected_B[f][...,int(policy[0,f])], qs_pi_previous_relevant_factors[f][0][...,None])
            lnB_s_omega_pi_f = maths.spm_dot(lnB[f][...,int(policy[0,f])], qs_pi_previous_relevant_factors[f][0][...,None])
            prediction_errors_f = s_omega_pi_f - qs_pi[idx][0][f][0]
            omega_per_policy[idx][f] = q_pi[idx] * (prediction_errors_f[...,None] * lnB_s_omega_pi_f[...,None] + beta_omega_prior[f])

    beta_omega_summed_over_policies = omega_per_policy.sum(axis=0)

    # how do we contract beta_zeta_full in order to be consistent with the original shape of beta_zeta and beta_zeta_p
    if np.isscalar(beta_omega):

        beta_omega_posterior = sum([beta_omege_f.sum() for beta_omege_f in beta_omega_summed_over_policies])
    elif np.isscalar(beta_omega[0]):
        beta_omega_posterior = [beta_omege_f.sum() for beta_omege_f in beta_omega_summed_over_policies]
    else:
        beta_omega_posterior = beta_omega_summed_over_policies

    if update_prior:
        beta_omega_prior = beta_omega_posterior

    return beta_omega_posterior, beta_omega_prior