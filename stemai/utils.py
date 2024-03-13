import numpy as np
import networkx
import matplotlib.pyplot as plt


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
def stemness(B):
    max_possible_entry = 1.0 / len(B[0])
    min_actual_entry = np.min(B[0])
    return min_actual_entry / max_possible_entry


def draw_network(
    network, colors, title=None, pos=None, t=None, _draw_neighboring_pairs=False, save=False, show = False, temp_file_name=None
):
    """
    Draw a network using networkx and matplotlib.

    Parameters:
    - network: networkx.Graph. The network to draw.
    """
    #fig = plt.figure(figsize=(12, 6))

    node_colors = [colors[node] for node in network.nodes]
    # shift position a little bit
    shift = [-0.05, -0.05]
    shifted_pos = {node: node_pos + shift for node, node_pos in pos.items()}

    GS_labels = {}
    for node in network.nodes:
        if hasattr(network.nodes[node]["agent"], "G"):
            G = network.nodes[node]["agent"].G.sum().round(2) * -1
            S = stemness(network.nodes[node]["agent"].B).round(2)
            GS_labels[node] = f"G: {G}, S: {S}"
    if _draw_neighboring_pairs:

        networkx.draw(
            network,
            with_labels=True,
            node_color=node_colors,
            pos=pos,
            font_weight="bold",
            edge_color="white",
        )
        networkx.draw_networkx_labels(
            network, shifted_pos, labels=GS_labels, horizontalalignment="left"
        )

        draw_neighboring_pairs(network, pos)
    else:

        networkx.draw(
            network, with_labels=True, node_color=node_colors, pos=pos, font_weight="bold"
        )

    if title is not None:
        plt.title(title)

    if save:
        # Save the current figure to a temporary file and add it to the images list

        plt.savefig(temp_file_name)
    if show:
        plt.show()
    return temp_file_name


def draw_neighboring_pairs(network, pos):

    neighboring_pairs = []
    for node in network.nodes:
        neighbors = list(network.neighbors(node))
        for neighbor in neighbors:
            if (node, neighbor) not in neighboring_pairs:  # Ensure each pair is added only once
                neighboring_pairs.append((node, neighbor))

    actions_received = {}

    for node in network.nodes:
        agent = network.nodes[node]["agent"]
        actions_received[node] = agent.actions_received

    done_already = []

    for receiver, sender in neighboring_pairs:
        # Define edge color based on the action received

        if (sender, receiver) in neighboring_pairs and (sender, receiver) not in done_already:
            # Define edge color based on the action sent, which requires accessing the sender's actions_received
            if receiver in actions_received[sender]:
                edge_color_received = (
                    "lightcoral"
                    if int(actions_received[sender][receiver]) == 0
                    else "darkslategray"
                )
                # Draw the edge for the action received
                networkx.draw_networkx_edges(
                    network,
                    pos,
                    edgelist=[(receiver, sender)],
                    edge_color=edge_color_received,
                    arrows=True,
                    arrowstyle="-|>",
                    style="dashed",
                )  # Dashed line for received action
                # Draw the edge for the action sent, slightly offset to avoid overlap
                done_already.append((sender, receiver))
        if (receiver, sender) in neighboring_pairs and (receiver, sender) not in done_already:

            if sender in actions_received[receiver]:
                edge_color_sent = (
                    "lightcoral"
                    if int(actions_received[receiver][sender]) == 0
                    else "darkslategray"
                )

                networkx.draw_networkx_edges(
                    network,
                    pos,
                    edgelist=[(sender, receiver)],
                    edge_color=edge_color_sent,
                    arrows=True,
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.3",
                )  # Curved line for sent action
                done_already.append((receiver, sender))
