
import numpy as np 
import networkx 


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

# Example usage:
# Assuming B is your original B matrix and you want to remove the neighbor at index 1
# new_B = remove_neighbor(B, 1)

def draw_network(network, title = None):
    """
    Draw a network using networkx and matplotlib.
    
    Parameters:
    - network: networkx.Graph. The network to draw.
    """
    import matplotlib.pyplot as plt
    networkx.draw(network, with_labels=True, font_weight='bold')
    if title is not None:
        plt.title(title)
    plt.show()