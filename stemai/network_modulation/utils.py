import numpy as np

def marginalize(arr, num_states, to_marginalize, axis):
    """Marginalize arr over the given axis using the given states to marginalize over"""
    marginalized_arr_shape = list(arr.shape)
    marginalized_arr_shape[axis] = num_states 
    marginalized_arr = np.zeros(tuple(marginalized_arr_shape))

    for state_idx in range(num_states):
        states_to_marginalize = to_marginalize[state_idx]
        selected_indices = [slice(None)] * arr.ndim
        selected_indices[axis] = state_idx

        selected_indices_to_marg = [slice(None)] * arr.ndim
        selected_indices_to_marg[axis] = states_to_marginalize

        marginalized_arr[tuple(selected_indices)] = np.sum(arr[tuple(selected_indices_to_marg)], axis=axis)
    return marginalized_arr

def distribute(arr, old_num_states, new_num_states, to_distribute, axis):
    """Distribute arr over the given axis using the given states to distribute over"""
    distributed_arr_shape = list(arr.shape)
    distributed_arr_shape[axis] = new_num_states 
    distributed_arr = np.zeros(tuple(distributed_arr_shape))
    for old_state_idx in range(old_num_states):
        selected_indices = [slice(None)] * arr.ndim
        selected_indices[axis] = old_state_idx

        for s_d in to_distribute[old_state_idx]:
            selected_indices_to_dist = [slice(None)] * arr.ndim
            selected_indices_to_dist[axis] = s_d
            distributed_arr[tuple(selected_indices_to_dist)] = arr[tuple(selected_indices)] / 2
            
    return distributed_arr
