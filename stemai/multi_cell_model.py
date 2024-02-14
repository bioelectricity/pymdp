# %%
from stemai.networks.external_network import GenerativeProcess
from stemai.networks.agent_network import GenerativeModel
import numpy as np


print(f"Building generative process")
generative_process = GenerativeProcess(5, 0.6, 1)
print()

print(f"Building generative model")
generative_model = GenerativeModel(5, 0.6, 1)
print()

# %%
T = 50

# now we need to model them interacting

# Generate and save the network images for each timestep

genmodel_signal = [np.random.choice([0, 1])]
for t in range(T):
    # Add an extra node for displaying the agent_observation
    # Plot an additional circle for displaying the agent_observation
    genprocess_signal = generative_process.act(genmodel_signal)

    print(f"Generative process signal : {genprocess_signal}")

    # now we need to convert this list of actions into one action for the generative model

    genmodel_signal = generative_model.act(genprocess_signal)

    print(f"Generative model signal : {genmodel_signal}")
