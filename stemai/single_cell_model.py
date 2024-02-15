#%%
from networks.generative_process import GenerativeProcess
from networks.generative_model import GenerativeModel
import numpy as np


print(f"Building generative process")
generative_process = GenerativeProcess(1, 1)
print()

print(f"Building generative model")
generative_model = GenerativeModel(1, 1)
print()
T = 50

#now we need to model them interacting 

# Generate and save the network images for each timestep

initial_gp_observation = np.random.choice([0,1])
for t in range(T):
    # Add an extra node for displaying the agent_observation
    # Plot an additional circle for displaying the agent_observation
    genprocess_signal = generative_process.act(initial_gp_observation)
   

    print(f"Generative process signal : {genprocess_signal}")

    #now we need to convert this list of actions into one action for the generative model

    genmodel_signal = generative_model.act(genprocess_signal[0])

    print(f"Generative model signal : {genmodel_signal}")

