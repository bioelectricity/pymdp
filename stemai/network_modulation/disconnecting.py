import numpy as np 
from pymdp import utils
from network_modulation.utils import marginalize

    
def remove_neighbor_from_pB(pB, states_and_actions_to_marginalize):
    """
    Remove a neighbor from the pB matrix.
    
    Parameters:
    - pB: np.ndarray. The pB matrix (the B prior)
    - states_and_actions_to_marginalize: dict. a mapping from old states
    to the new states that we need to marginalize over
    
    Returns:
    - np.ndarray. The pB matrix with the neighbor removed.
    """

    new_pB = utils.obj_array(len(pB))

    num_states, _, num_actions = pB[0].shape
    new_num_states = num_states // 2
    new_num_actions = num_actions // 2  

    #marginalize over previous state
    pB_marginalized = marginalize(pB[0], new_num_states, states_and_actions_to_marginalize, 1)
    #marginalize over next state
    pB_marginalized = marginalize(pB_marginalized, new_num_states, states_and_actions_to_marginalize, 0)

    #marginalize over actions 
    pB_marginalized = marginalize(pB_marginalized, new_num_actions, states_and_actions_to_marginalize, -1)

    new_pB[0] = pB_marginalized
    return new_pB

def remove_neighbor_from_C(C, states_and_actions_to_marginalize):
    """
    Remove a neighbor from the C matrix.
    
    Parameters:
    - C: np.ndarray. The C matrix (can also be pC)
    - neighbor: int. The index of the neighbor to remove.
    
    Returns:
    - np.ndarray. The C matrix with the neighbor removed.
    """
    new_C = utils.obj_array(len(C))
    num_obs = C[0].shape[0]
    new_num_obs = num_obs // 2

    C_marginalized = marginalize(C[0], new_num_obs, states_and_actions_to_marginalize, 0)

    new_C[0] = C_marginalized
    return new_C


def remove_neighbor_from_D(D, states_and_actions_to_marginalize):
    """
    Remove a neighbor from the D matrix.
    
    """
    new_D = utils.obj_array(len(D))
    num_states = D[0].shape[0]
    new_num_states = num_states // 2

    D_marginalized = marginalize(D[0], new_num_states, states_and_actions_to_marginalize, 0)
    new_D[0] = D_marginalized
    return new_D