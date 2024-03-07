#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

import numpy as np
from pymdp import utils, maths
import copy

def update_obs_likelihood_dirichlet(pA, A, obs, qs, lr=1.0, modalities="all"):
    """ 
    Update Dirichlet parameters of the observation likelihood distribution.

    Parameters
    -----------
    pA: ``numpy.ndarray`` of dtype object
        Prior Dirichlet parameters over observation model (same shape as ``A``)
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    obs: 1D ``numpy.ndarray``, ``numpy.ndarray`` of dtype object, ``int`` or ``tuple``
        The observation (generated by the environment). If single modality, this can be a 1D ``numpy.ndarray``
        (one-hot vector representation) or an ``int`` (observation index)
        If multi-modality, this can be ``numpy.ndarray`` of dtype object whose entries are 1D one-hot vectors,
        or a ``tuple`` (of ``int``)
    qs: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object, default None
        Marginal posterior beliefs over hidden states at current timepoint.
    lr: float, default 1.0
        Learning rate, scale of the Dirichlet pseudo-count update.
    modalities: ``list``, default "all"
        Indices (ranging from 0 to ``n_modalities - 1``) of the observation modalities to include 
        in learning. Defaults to "all", meaning that modality-specific sub-arrays of ``pA``
        are all updated using the corresponding observations.
    
    Returns
    -----------
    qA: ``numpy.ndarray`` of dtype object
        Posterior Dirichlet parameters over observation model (same shape as ``A``), after having updated it with observations.
    """


    num_modalities = len(pA)
    num_observations = [pA[modality].shape[0] for modality in range(num_modalities)]

    obs_processed = utils.process_observation(obs, num_modalities, num_observations)
    obs = utils.to_obj_array(obs_processed)

    if modalities == "all":
        modalities = list(range(num_modalities))

    qA = copy.deepcopy(pA)
        
    for modality in modalities:
        dfda = maths.spm_cross(obs[modality], qs)
        dfda = dfda * (A[modality] > 0).astype("float")
        qA[modality] = qA[modality] + (lr * dfda)

    return qA

def update_obs_likelihood_dirichlet_factorized(pA, A, obs, qs, A_factor_list, lr=1.0, modalities="all"):
    """ 
    Update Dirichlet parameters of the observation likelihood distribution, in a case where the observation model is reduced (factorized) and only represents
    the conditional dependencies between the observation modalities and particular hidden state factors (whose indices are specified in each modality-specific entry of ``A_factor_list``)

    Parameters
    -----------
    pA: ``numpy.ndarray`` of dtype object
        Prior Dirichlet parameters over observation model (same shape as ``A``)
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    obs: 1D ``numpy.ndarray``, ``numpy.ndarray`` of dtype object, ``int`` or ``tuple``
        The observation (generated by the environment). If single modality, this can be a 1D ``numpy.ndarray``
        (one-hot vector representation) or an ``int`` (observation index)
        If multi-modality, this can be ``numpy.ndarray`` of dtype object whose entries are 1D one-hot vectors,
        or a ``tuple`` (of ``int``)
    qs: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object, default None
        Marginal posterior beliefs over hidden states at current timepoint.
    A_factor_list: ``list`` of ``list`` of ``int``
        List of lists, where each list with index `m` contains the indices of the hidden states that observation modality `m` depends on.
    lr: float, default 1.0
        Learning rate, scale of the Dirichlet pseudo-count update.
    modalities: ``list``, default "all"
        Indices (ranging from 0 to ``n_modalities - 1``) of the observation modalities to include 
        in learning. Defaults to "all", meaning that modality-specific sub-arrays of ``pA``
        are all updated using the corresponding observations.
    
    Returns
    -----------
    qA: ``numpy.ndarray`` of dtype object
        Posterior Dirichlet parameters over observation model (same shape as ``A``), after having updated it with observations.
    """

    num_modalities = len(pA)
    num_observations = [pA[modality].shape[0] for modality in range(num_modalities)]

    obs_processed = utils.process_observation(obs, num_modalities, num_observations)
    obs = utils.to_obj_array(obs_processed)

    if modalities == "all":
        modalities = list(range(num_modalities))

    qA = copy.deepcopy(pA)
        
    for modality in modalities:
        dfda = maths.spm_cross(obs[modality], qs[A_factor_list[modality]])
        dfda = dfda * (A[modality] > 0).astype("float")
        qA[modality] = qA[modality] + (lr * dfda)

    return qA

def update_state_likelihood_dirichlet(
    pB, B, actions, qs, qs_prev, lr=1.0, factors="all"
):
    """
    Update Dirichlet parameters of the transition distribution. 

    Parameters
    -----------
    pB: ``numpy.ndarray`` of dtype object
        Prior Dirichlet parameters over transition model (same shape as ``B``)
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    actions: 1D ``numpy.ndarray``
        A vector with length equal to the number of control factors, where each element contains the index of the action (for that control factor) performed at 
        a given timestep.
    qs: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
        Marginal posterior beliefs over hidden states at current timepoint.
    qs_prev: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
        Marginal posterior beliefs over hidden states at previous timepoint.
    lr: float, default ``1.0``
        Learning rate, scale of the Dirichlet pseudo-count update.
    factors: ``list``, default "all"
        Indices (ranging from 0 to ``n_factors - 1``) of the hidden state factors to include 
        in learning. Defaults to "all", meaning that factor-specific sub-arrays of ``pB``
        are all updated using the corresponding hidden state distributions and actions.

    Returns
    -----------
    qB: ``numpy.ndarray`` of dtype object
        Posterior Dirichlet parameters over transition model (same shape as ``B``), after having updated it with state beliefs and actions.
    """

    num_factors = len(pB)

    qB = copy.deepcopy(pB)
   
    if factors == "all":
        factors = list(range(num_factors))

    for factor in factors:
        dfdb = maths.spm_cross(qs[factor], qs_prev[factor])
        dfdb *= (B[factor][:, :, int(actions[factor])] > 0).astype("float")
        qB[factor][:,:,int(actions[factor])] += (lr*dfdb)

    return qB

def update_state_likelihood_dirichlet_interactions(
    pB, B, actions, qs, qs_prev, B_factor_list, lr=1.0, factors="all"
):
    """
    Update Dirichlet parameters of the transition distribution, in the case when 'interacting' hidden state factors are present, i.e.
    the dynamics of a given hidden state factor `f` are no longer independent of the dynamics of other hidden state factors.

    Parameters
    -----------
    pB: ``numpy.ndarray`` of dtype object
        Prior Dirichlet parameters over transition model (same shape as ``B``)
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    actions: 1D ``numpy.ndarray``
        A vector with length equal to the number of control factors, where each element contains the index of the action (for that control factor) performed at 
        a given timestep.
    qs: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
        Marginal posterior beliefs over hidden states at current timepoint.
    qs_prev: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
        Marginal posterior beliefs over hidden states at previous timepoint.
    B_factor_list: ``list`` of ``list`` of ``int``
        A list of lists, where each element ``B_factor_list[f]`` is a list of indices of hidden state factors that that are needed to predict the dynamics of hidden state factor ``f``.
    lr: float, default ``1.0``
        Learning rate, scale of the Dirichlet pseudo-count update.
    factors: ``list``, default "all"
        Indices (ranging from 0 to ``n_factors - 1``) of the hidden state factors to include 
        in learning. Defaults to "all", meaning that factor-specific sub-arrays of ``pB``
        are all updated using the corresponding hidden state distributions and actions.

    Returns
    -----------
    qB: ``numpy.ndarray`` of dtype object
        Posterior Dirichlet parameters over transition model (same shape as ``B``), after having updated it with state beliefs and actions.
    """

    num_factors = len(pB)

    qB = copy.deepcopy(pB)
   
    if factors == "all":
        factors = list(range(num_factors))

    for factor in factors:
        dfdb = maths.spm_cross(qs[factor], qs_prev[B_factor_list[factor]])
        dfdb *= (B[factor][...,int(actions[factor])] > 0).astype("float")
        qB[factor][...,int(actions[factor])] += (lr*dfdb)

    return qB

def update_state_prior_dirichlet(
    pD, qs, lr=1.0, factors="all"
):
    """
    Update Dirichlet parameters of the initial hidden state distribution 
    (prior beliefs about hidden states at the beginning of the inference window).

    Parameters
    -----------
    pD: ``numpy.ndarray`` of dtype object
        Prior Dirichlet parameters over initial hidden state prior (same shape as ``qs``)
    qs: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
        Marginal posterior beliefs over hidden states at current timepoint
    lr: float, default ``1.0``
        Learning rate, scale of the Dirichlet pseudo-count update.
    factors: ``list``, default "all"
        Indices (ranging from 0 to ``n_factors - 1``) of the hidden state factors to include 
        in learning. Defaults to "all", meaning that factor-specific sub-vectors of ``pD``
        are all updated using the corresponding hidden state distributions.
    
    Returns
    -----------
    qD: ``numpy.ndarray`` of dtype object
        Posterior Dirichlet parameters over initial hidden state prior (same shape as ``qs``), after having updated it with state beliefs.
    """

    num_factors = len(pD)

    qD = copy.deepcopy(pD)
   
    if factors == "all":
        factors = list(range(num_factors))

    for factor in factors:
        idx = pD[factor] > 0 # only update those state level indices that have some prior probability
        qD[factor][idx] += (lr * qs[factor][idx])
       
    return qD

def _prune_prior(prior, levels_to_remove, dirichlet = False):
    """
    Function for pruning a prior Categorical distribution (e.g. C, D)

    Parameters
    -----------
    prior: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
        The vector(s) containing the priors over hidden states of a generative model, e.g. the prior over hidden states (``D`` vector). 
    levels_to_remove: ``list`` of ``int``, ``list`` of ``list``
        A ``list`` of the levels (indices of the support) to remove. If the prior in question has multiple hidden state factors / multiple observation modalities, 
        then this will be a ``list`` of ``list``, where each sub-list within ``levels_to_remove`` will contain the levels to prune for a particular hidden state factor or modality 
    dirichlet: ``Bool``, default ``False``
        A Boolean flag indicating whether the input vector(s) is/are a Dirichlet distribution, and therefore should not be normalized at the end. 
        @TODO: Instead, the dirichlet parameters from the pruned levels should somehow be re-distributed among the remaining levels

    Returns
    -----------
    reduced_prior: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
        The prior vector(s), after pruning, that lacks the hidden state or modality levels indexed by ``levels_to_remove``
    """

    if utils.is_obj_array(prior): # in case of multiple hidden state factors

        assert all([type(levels) == list for levels in levels_to_remove])

        num_factors = len(prior)

        reduced_prior = utils.obj_array(num_factors)

        factors_to_remove = []
        for f, s_i in enumerate(prior): # loop over factors (or modalities)
            
            ns = len(s_i)
            levels_to_keep = list(set(range(ns)) - set(levels_to_remove[f]))
            if len(levels_to_keep) == 0:
                print(f'Warning... removing ALL levels of factor {f} - i.e. the whole hidden state factor is being removed\n')
                factors_to_remove.append(f)
            else:
                if not dirichlet:
                    reduced_prior[f] = utils.norm_dist(s_i[levels_to_keep])
                else:
                    raise(NotImplementedError("Need to figure out how to re-distribute concentration parameters from pruned levels, across remaining levels"))


        if len(factors_to_remove) > 0:
            factors_to_keep = list(set(range(num_factors)) - set(factors_to_remove))
            reduced_prior = reduced_prior[factors_to_keep]

    else: # in case of one hidden state factor

        assert all([type(level_i) == int for level_i in levels_to_remove])

        ns = len(prior)
        levels_to_keep = list(set(range(ns)) - set(levels_to_remove))

        if not dirichlet:
            reduced_prior = utils.norm_dist(prior[levels_to_keep])
        else:
            raise(NotImplementedError("Need to figure out how to re-distribute concentration parameters from pruned levels, across remaining levels"))

    return reduced_prior

def _prune_A(A, obs_levels_to_prune, state_levels_to_prune, dirichlet = False):
    """
    Function for pruning a observation likelihood model (with potentially multiple hidden state factors)
    :meta private:
    Parameters
    -----------
    A: ``numpy.ndarray`` with ``ndim >= 2``, or ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    obs_levels_to_prune: ``list`` of int or ``list`` of ``list``: 
        A ``list`` of the observation levels to remove. If the likelihood in question has multiple observation modalities, 
        then this will be a ``list`` of ``list``, where each sub-list within ``obs_levels_to_prune`` will contain the observation levels 
        to remove for a particular observation modality 
    state_levels_to_prune: ``list`` of ``int``
        A ``list`` of the hidden state levels to remove (this will be the same across modalities)
    dirichlet: ``Bool``, default ``False``
        A Boolean flag indicating whether the input array(s) is/are a Dirichlet distribution, and therefore should not be normalized at the end. 
        @TODO: Instead, the dirichlet parameters from the pruned columns should somehow be re-distributed among the remaining columns

    Returns
    -----------
    reduced_A: ``numpy.ndarray`` with ndim >= 2, or ``numpy.ndarray ``of dtype object
        The observation model, after pruning, which lacks the observation or hidden state levels given by the arguments ``obs_levels_to_prune`` and ``state_levels_to_prune``
    """

    columns_to_keep_list = []
    if utils.is_obj_array(A):
        num_states = A[0].shape[1:]
        for f, ns in enumerate(num_states):
            indices_f = np.array( list(set(range(ns)) - set(state_levels_to_prune[f])), dtype = np.intp)
            columns_to_keep_list.append(indices_f)
    else:
        num_states = A.shape[1]
        indices = np.array( list(set(range(num_states)) - set(state_levels_to_prune)), dtype = np.intp )
        columns_to_keep_list.append(indices)

    if utils.is_obj_array(A): # in case of multiple observation modality

        assert all([type(o_m_levels) == list for o_m_levels in obs_levels_to_prune])

        num_modalities = len(A)

        reduced_A = utils.obj_array(num_modalities)
        
        for m, A_i in enumerate(A): # loop over modalities
            
            no = A_i.shape[0]
            rows_to_keep = np.array(list(set(range(no)) - set(obs_levels_to_prune[m])), dtype = np.intp)
            
            reduced_A[m] = A_i[np.ix_(rows_to_keep, *columns_to_keep_list)]
        if not dirichlet:    
            reduced_A = utils.norm_dist_obj_arr(reduced_A)
        else:
            raise(NotImplementedError("Need to figure out how to re-distribute concentration parameters from pruned rows/columns, across remaining rows/columns"))
    else: # in case of one observation modality

        assert all([type(o_levels_i) == int for o_levels_i in obs_levels_to_prune])

        no = A.shape[0]
        rows_to_keep = np.array(list(set(range(no)) - set(obs_levels_to_prune)), dtype = np.intp)
            
        reduced_A = A[np.ix_(rows_to_keep, *columns_to_keep_list)]

        if not dirichlet:
            reduced_A = utils.norm_dist(reduced_A)
        else:
            raise(NotImplementedError("Need to figure out how to re-distribute concentration parameters from pruned rows/columns, across remaining rows/columns"))

    return reduced_A

def _prune_B(B, state_levels_to_prune, action_levels_to_prune, dirichlet = False):
    """
    Function for pruning a transition likelihood model (with potentially multiple hidden state factors)

    Parameters
    -----------
    B: ``numpy.ndarray`` of ``ndim == 3`` or ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at `t` to hidden states at `t+1`, given some control state `u`.
        Each element B[f] of this object array stores a 3-D tensor for hidden state factor `f`, whose entries `B[f][s, v, u] store the probability
        of hidden state level `s` at the current time, given hidden state level `v` and action `u` at the previous time.
    state_levels_to_prune: ``list`` of ``int`` or ``list`` of ``list`` 
        A ``list`` of the state levels to remove. If the likelihood in question has multiple hidden state factors, 
        then this will be a ``list`` of ``list``, where each sub-list within ``state_levels_to_prune`` will contain the state levels 
        to remove for a particular hidden state factor 
    action_levels_to_prune: ``list`` of ``int`` or ``list`` of ``list`` 
        A ``list`` of the control state or action levels to remove. If the likelihood in question has multiple control state factors, 
        then this will be a ``list`` of ``list``, where each sub-list within ``action_levels_to_prune`` will contain the control state levels 
        to remove for a particular control state factor 
    dirichlet: ``Bool``, default ``False``
        A Boolean flag indicating whether the input array(s) is/are a Dirichlet distribution, and therefore should not be normalized at the end. 
        @TODO: Instead, the dirichlet parameters from the pruned rows/columns should somehow be re-distributed among the remaining rows/columns

    Returns
    -----------
    reduced_B: ``numpy.ndarray`` of `ndim == 3` or ``numpy.ndarray`` of dtype object
        The transition model, after pruning, which lacks the hidden state levels/action levels given by the arguments ``state_levels_to_prune`` and ``action_levels_to_prune``
    """

    slices_to_keep_list = []

    if utils.is_obj_array(B):

        num_controls = [B_arr.shape[2] for _, B_arr in enumerate(B)]

        for c, nc in enumerate(num_controls):
            indices_c = np.array( list(set(range(nc)) - set(action_levels_to_prune[c])), dtype = np.intp)
            slices_to_keep_list.append(indices_c)
    else:
        num_controls = B.shape[2]
        slices_to_keep = np.array( list(set(range(num_controls)) - set(action_levels_to_prune)), dtype = np.intp )

    if utils.is_obj_array(B): # in case of multiple hidden state factors

        assert all([type(ns_f_levels) == list for ns_f_levels in state_levels_to_prune])

        num_factors = len(B)

        reduced_B = utils.obj_array(num_factors)
        
        for f, B_f in enumerate(B): # loop over modalities
            
            ns = B_f.shape[0]
            states_to_keep = np.array(list(set(range(ns)) - set(state_levels_to_prune[f])), dtype = np.intp)
            
            reduced_B[f] = B_f[np.ix_(states_to_keep, states_to_keep, slices_to_keep_list[f])]

        if not dirichlet:    
            reduced_B = utils.norm_dist_obj_arr(reduced_B)
        else:
            raise(NotImplementedError("Need to figure out how to re-distribute concentration parameters from pruned rows/columns, across remaining rows/columns"))

    else: # in case of one hidden state factor

        assert all([type(state_level_i) == int for state_level_i in state_levels_to_prune])

        ns = B.shape[0]
        states_to_keep = np.array(list(set(range(ns)) - set(state_levels_to_prune)), dtype = np.intp)
            
        reduced_B = B[np.ix_(states_to_keep, states_to_keep, slices_to_keep)]

        if not dirichlet:
            reduced_B = utils.norm_dist(reduced_B)
        else:
            raise(NotImplementedError("Need to figure out how to re-distribute concentration parameters from pruned rows/columns, across remaining rows/columns"))

    return reduced_B




def update_beta_zeta(observation, A, beta_zeta, qs, beta_zeta_prior, A_factor_list, update_prior = False, modalities = None):

    """
    beta_zeta can be:
    - a scalar 
    - a vector of length num_modalities 
    - a list/collection of np.ndarray of len num_modalities, where the m-th element will have shape (num_states[m], num_states[n], num_states[k]) aka A.shape[1:], where
    m, n, k are the indices of the state factors that modality [m] depends on
    """

    # print(f"Observation: {observation}")

    #Do we want to do empirical bayes where we update pzeta? 
    if update_prior:
        new_beta_zeta_prior = beta_zeta
    else:
        new_beta_zeta_prior = beta_zeta_prior
    
    expected_A = utils.scale_A_with_zeta(A, beta_zeta)


  #  beta_zeta = 1/ np.array(beta_zeta)
   # beta_zeta_prior = 1/ np.array(beta_zeta_prior)

    # in case A_factor_list is non-trivial, you have to sub-select qs[relevant_factor_idx]
    get_factors = lambda q, factor_list: [q[f_idx] for f_idx in factor_list]
    qs_relevant = np.array([get_factors(qs, factor_list) for factor_list in A_factor_list], dtype = 'object')
    # print(f"Expected a : {expected_A}")
    # print(f"Qs relevant : {qs_relevant}")
    # print(f"Qs relevant : {qs_relevant}")
    bold_o_per_modality = utils.obj_array_from_list([maths.spm_dot(expected_A[m], qs_relevant[m]) for m in range(len(A))])

    observation_array = utils.obj_array_from_list([utils.onehot(observation[m], A[m].shape[0]) for m in range(len(A))])

    # print(f"Observations under gamma_A: {bold_o_per_modality}")
    # print(f"Observations: {observation_array}")

    prediction_errors = np.absolute(np.array(bold_o_per_modality) - np.array(observation_array))

    # print(f"Beta zeta prior: {beta_zeta_prior}")
    # print(f"Inverse beta zeta prior: {1/beta_zeta_prior}")


    #lnA = maths.spm_log_obj_array(A)

    # do checking here to make sure pzeta is broadcast-consistent
    
    if modalities is None:
        modalities = range(len(A))

    beta_zeta_full = utils.obj_array(len(A))
    
    if np.isscalar(beta_zeta_prior):
        beta_zeta_prior = np.array([beta_zeta_prior] * len(A))

    #print(f"beta zeta prior :{beta_zeta_prior}")

    for m in range(len(A)):
        if m not in modalities:

            beta_zeta_full[m] =np.array([beta_zeta_prior[m]]*2)
        else:

            # print(f"MODALITY : {m}, prediction error: {prediction_errors[m]}")
            prediction_errors_expanded = prediction_errors[m]
            for _ in range(A[m].ndim - 1):
                prediction_errors_expanded = prediction_errors_expanded[..., np.newaxis]
            #print(f"Zeta prior : {beta_zeta_prior[m]}")

            # print(f"pred error times A[m]: {(prediction_errors_expanded * A[m]).sum(axis=0)}")

            #beta_zeta_full_m = (prediction_errors_expanded * lnA[m]).sum(axis=0) + (1/beta_zeta_prior[m])
            beta_zeta_full_m = (prediction_errors_expanded * A[m]).sum(axis=0) + (1/beta_zeta_prior[m])

            # print(f"Beta zeta full m: { 1 / (np.array(beta_zeta_full_m) + 1e-16)}")

            beta_zeta_full[m] = 1 / (np.array(beta_zeta_full_m) + 1e-16)

            # print(f"Beta zeta full m: {beta_zeta_full[m]}")

       # beta_zeta_full[m] = np.array([np.minimum(x,100) for x in beta_zeta_full_m])
    # how do we contract beta_zeta_full in order to be consistent with the original shape of beta_zeta and beta_zeta_p
    #print(f"Beta zeta full: {beta_zeta_full}")
    if np.isscalar(beta_zeta):
        beta_zeta_posterior = sum([beta_zeta_m.sum() for beta_zeta_m in beta_zeta_full])
    elif np.isscalar(beta_zeta[0]):
        beta_zeta_posterior = np.array([beta_zeta_m.sum() / 2 for beta_zeta_m in beta_zeta_full])
    # #     print(f"Gamma_A posterior : {beta_zeta_posterior}")
    else:
        beta_zeta_posterior = beta_zeta_full

    if np.nan in beta_zeta_posterior:
        beta_zeta_posterior = np.nan_to_num(beta_zeta_posterior) + 0.0001

        
    return np.array(beta_zeta_posterior), np.array(new_beta_zeta_prior)


# E_{Q(s_{t-1, m, n, k}}[P(s_{t,f}|s_{t-1, m}, s_{t-1, n}, ... s_{t-1, k})] # this is what's computed by get_expected_states_with_interactions
# ==> Q(s_{t,f})

def update_beta_omega( q_pi, qs_pi, qs_pi_previous, B, beta_omega, beta_omega_prior, policies, B_factor_list, update_prior=False):
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
    expected_B = utils.scale_B_with_omega(B, beta_omega)
    
    lnB = maths.spm_log_obj_array(B)
    # do checking here to make sure pzeta is broadcast-consistent
    if np.isscalar(beta_omega_prior):
        beta_omega_prior = np.array([beta_omega_prior] * len(B))
    

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
                omega_per_policy_and_time_horizon[t][f] = q_pi[idx] * (prediction_errors_f[...,None] * lnB_s_omega_pi_f[...,None] + beta_omega_prior[f])

        omega_per_policy[idx] = omega_per_policy_and_time_horizon.sum(axis=0) #sum over the time horizon

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



def update_beta_gamma(G, gamma, q_pi, policies):
    pi_0 = maths.softmax(-gamma * G) #should we index into policies? 

    for idx  in range(len(policies)):
        gamma += (q_pi[idx] - pi_0).dot(G)

    return gamma