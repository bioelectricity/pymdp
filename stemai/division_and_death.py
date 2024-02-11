import numpy as np 



def remove_neighbor_from_B(B, states_and_actions_to_marginalize):
    """
    Remove a neighbor from the B matrix.
    
    Parameters:
    - B: np.ndarray. The B matrix (can also be pB)
    - neighbor: int. The index of the neighbor to remove.
    
    Returns:
    - np.ndarray. The B matrix with the neighbor removed.
    """

    num_states, _, num_actions = B[0].shape
    new_num_states = num_states // 2
    new_num_actions = num_actions // 2  

    B_marginalized_over_states = np.zeros((new_num_states, new_num_states, num_actions))

    #first marginalize over states
    for action in range(num_actions):
        for state_idx in range(new_num_states):
            for neighbor_state_idx in states_and_actions_to_marginalize[state_idx]:
                #the probability of going to any next state given this state is the marginalized probability given the old state at the neighbors signals
                B_marginalized_over_states[:, state_idx, action] += B[0][:, neighbor_state_idx, action]

    B_marginalized_over_states_and_actions = np.zeros((new_num_states, new_num_states, new_num_actions))

    #then marginalize over actions
    for action_idx in range(new_num_actions):
        for state_idx in range(new_num_states):
            for neighbor_action_idx in states_and_actions_to_marginalize[state_idx]:
                #the probability of going to this new state given any previous state is the marginalized probability given the old action at the neighbors signals
                B_marginalized_over_states_and_actions[state_idx, :, action_idx] += B_marginalized_over_states[state_idx, :, neighbor_action_idx]


    return B_marginalized_over_states_and_actions

def remove_neighbor_from_C(C, states_and_actions_to_marginalize):
    """
    Remove a neighbor from the C matrix.
    
    Parameters:
    - C: np.ndarray. The C matrix (can also be pC)
    - neighbor: int. The index of the neighbor to remove.
    
    Returns:
    - np.ndarray. The C matrix with the neighbor removed.
    """
    num_obs = C[0].shape[0]
    new_num_obs = num_obs // 2

    C_marginalized = np.zeros((new_num_obs))

    for obs_idx in range(new_num_obs):

        for neighbor_obs_idx in states_and_actions_to_marginalize[obs_idx]:
            C_marginalized[obs_idx] += C[0][neighbor_obs_idx]

    return C_marginalized


def remove_neighbor_from_D(D, states_and_actions_to_marginalize):
    """
    Remove a neighbor from the C matrix.
    
    Parameters:
    - C: np.ndarray. The C matrix (can also be pC)
    - neighbor: int. The index of the neighbor to remove.
    
    Returns:
    - np.ndarray. The C matrix with the neighbor removed.
    """
    num_states = D[0].shape[0]
    new_num_states = num_states // 2

    D_marginalized = np.zeros((new_num_states))

    for state_idx in range(new_num_states):

        for neighbor_obs_idx in states_and_actions_to_marginalize[state_idx]:
            D_marginalized[state_idx] += D[0][neighbor_obs_idx]

    return D_marginalized