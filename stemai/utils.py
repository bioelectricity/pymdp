
import numpy as np 

def generate_binary_numbers(N, num_states):
    binary_numbers = []
    for i in range(num_states):
        # Convert to binary, remove the '0b' prefix, and pad with leading zeros
        binary_str = bin(i)[2:].zfill(N)
        binary_numbers.append(binary_str)
    return binary_numbers



def signals_to_index(signals):
    """
    Convert a list of signals (0s and 1s) from neighbors into an index.
    
    Parameters:
    - signals: List[int]. A list of signals (0 or 1) from each neighbor.
    
    Returns:
    - int. The index corresponding to the state.
    """
    index = 0
    for signal in signals:
        index = (index << 1) | signal
    return index

def extract_agent_action(action, N):
    """
    Determine the given agent's action (0 or 1) from a number.
    
    Parameters:
    - action: int. The action number, in the range from 0 to 2^N - 1.
    - N: int. The total number of options or states, including the agent's action.
    
    Returns:
    - int. The action (0 or 1) of the given agent.
    """

    # Convert to binary and pad with zeros to ensure it has length N
    binary_action = bin(action)[2:].zfill(N + 2)

    print(f"Binary action {binary_action}")
    # Extract the -2nd bit (second from the right)
    return binary_action

def remove_neighbor_from_B(B, neighbor, new_state_names, old_state_names):
    """
    Remove a neighbor from the B matrix.
    
    Parameters:
    - B: np.ndarray. The B matrix.
    - neighbor: int. The index of the neighbor to remove.
    
    Returns:
    - np.ndarray. The B matrix with the neighbor removed.
    """

    num_states, _, num_actions = B[0].shape
    new_num_states = num_states // 2
    new_num_actions = num_actions // 2  

    states_and_actions_to_marginalize = {}
    for state_idx in range(new_num_states):
        new_state = new_state_names[state_idx]
        states_and_actions_to_marginalize[state_idx] = [s_idx for s_idx, state in enumerate(old_state_names) if state[:neighbor] + state[neighbor+1:]== new_state]

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

# Example usage:
# Assuming B is your original B matrix and you want to remove the neighbor at index 1
# new_B = remove_neighbor(B, 1)

