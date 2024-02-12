import numpy as np 
from pymdp import utils
from network_modulation.utils import distribute

def add_neighbor_to_pB(pB, states_and_actions_to_distribute):
    """
    Add a neighbor to the pB matrix.
    
    Parameters:
    - pB: np.ndarray. The B matrix (can also be pB)
    - states_and_actions_to_distribute: dict. A
    dictionary that maps an old state in the B matrix to the states to distribute
    over for the new neighbor
    
    Returns:
    - np.ndarray. The pB matrix with the neighbor added.
    """

    new_pB = utils.obj_array(len(pB))

    old_num_states, _, old_num_actions = pB[0].shape
    new_num_states = old_num_states * 2
    new_num_actions = old_num_actions * 2  

    pB_distributed = distribute(pB[0], old_num_states, new_num_states, states_and_actions_to_distribute, 1)
    pB_distributed = distribute(pB_distributed, old_num_states, new_num_states, states_and_actions_to_distribute, 0)
    pB_distributed = distribute(pB_distributed, old_num_actions, new_num_actions, states_and_actions_to_distribute, -1)

    new_pB[0] = pB_distributed
    return new_pB


def add_neighbor_to_C(C, states_and_actions_to_distribute):
    """
    Add a neighbor to the C vector
    """
    new_C = utils.obj_array(len(C))
    old_num_obs = C[0].shape[0]
    new_num_obs = old_num_obs * 2

    C = np.zeros((new_num_obs))

    new_C[0] = distribute(C, old_num_obs, new_num_obs, states_and_actions_to_distribute, 0)

    return new_C


def add_neighbor_to_D(D, states_and_actions_to_distribute):
    """
    Add a neighbor to the D matrix.
    
    """
    new_D = utils.obj_array(len(D))
    old_num_states = D[0].shape[0]
    new_num_states = old_num_states * 2
    new_D[0] = distribute(D[0], old_num_states, new_num_states, states_and_actions_to_distribute, 0)

    return new_D