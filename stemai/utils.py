
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
    # Extract the -2nd bit (second from the right)
    agent_action = int(binary_action[-2])
    return agent_action