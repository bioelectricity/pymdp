#%%
import numpy as np
import os 
try:
    from stemai.cells.agent_cell import NeuronalCell
except:
    os.chdir('../../')
    from stemai.cells.agent_cell import NeuronalCell
import copy
from pymdp import utils

grid_size = (1,31)
grid = np.zeros(grid_size)
import time
reward_location = (0, int(grid_size[1]-1))
agent_location = (0, int((grid_size[1]-1)/2))

grid[reward_location] = 1
grid[agent_location] = 2


#%%
def make_B_gif(arrays, fn):
    images = []
    for arr in arrays:

        arr = arr[0]
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        cax0 = axs[0].imshow(arr[:,:,0], cmap='gray', vmin=0, vmax=1)
        axs[0].set_xticks([0, 1])
        axs[0].set_xticklabels(['Fire', 'Not Fire'])
        axs[0].set_xlabel("s_t")
        axs[0].set_yticks([0, 1])
        axs[0].set_yticklabels(['Fire', 'Not Fire'])
        axs[0].set_ylabel("s_{t+1}")
        axs[0].set_title('Fire')
        fig.colorbar(cax0, ax=axs[0], fraction=0.046, pad=0.04)

        cax1 = axs[1].imshow(arr[:,:,1], cmap='gray', vmin=0, vmax=1)
        axs[1].set_title('No Fire')
        axs[1].set_xticks([0, 1])
        axs[1].set_xticklabels(['Fire', 'Not Fire'])
        axs[1].set_xlabel("s_t")
        axs[1].set_yticks([0, 1])
        axs[1].set_yticklabels(['Fire', 'Not Fire'])
        axs[1].set_ylabel("s_{t+1}")
        fig.colorbar(cax1, ax=axs[1], fraction=0.046, pad=0.04)

        plt.savefig("temp_B.png")
        images.append(imageio.imread("temp_B.png"))
        #os.remove("temp.png")
        plt.clf()
    imageio.mimsave(fn, images)


def make_pB_gif(arrays, fn):
    images = []
    for arr in arrays:

        arr = arr[0]
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        cax0 = axs[0].imshow(arr[:,:,0], cmap='gray')
        axs[0].set_xticks([0, 1])
        axs[0].set_xticklabels(['Fire', 'Not Fire'])
        axs[0].set_xlabel("s_t")
        axs[0].set_yticks([0, 1])
        axs[0].set_yticklabels(['Fire', 'Not Fire'])
        axs[0].set_ylabel("s_{t+1}")
        axs[0].set_title('Fire')
        fig.colorbar(cax0, ax=axs[0], fraction=0.046, pad=0.04)

        cax1 = axs[1].imshow(arr[:,:,1], cmap='gray')
        axs[1].set_title('No Fire')
        axs[1].set_xticks([0, 1])
        axs[1].set_xticklabels(['Fire', 'Not Fire'])
        axs[1].set_xlabel("s_t")
        axs[1].set_yticks([0, 1])
        axs[1].set_yticklabels(['Fire', 'Not Fire'])
        axs[1].set_ylabel("s_{t+1}")
        fig.colorbar(cax1, ax=axs[1], fraction=0.046, pad=0.04)

        plt.savefig("temp_pB.png")
        images.append(imageio.imread("temp_pB.png"))
        #os.remove("temp.png")
        plt.clf()
    imageio.mimsave(fn, images)

def get_grid(reward_location, agent_location, signal):
    grid = np.zeros(grid_size)
    if signal == 0:
        grid[reward_location] = 0.5
    else:
        grid[reward_location] = 1.5
    grid[agent_location] = 2
    return grid

#%%

class GridWorldCell:

    def __init__(self, reward_location, grid_size, agent_location):
        self.id = 'e1' 
        self.agent_location = agent_location
        self.reward_location = reward_location
        self.observation_history = []
        self.grid_size = grid_size

    def set_locations(self, reward_location, agent_location):
        self.reward_location = reward_location
        self.agent_location = agent_location

    def act(self, obs: int) -> str:
        #print(f"Observation: {qs_obs}")
        #obs = np.random.choice(np.flatnonzero(qs_obs == np.array(qs_obs).max()))
        self.observation_history.append(obs)

        if obs == 0: #observation 0 (fire) means move to the right
            if self.agent_location[1] < self.grid_size[1] - 1:
                self.agent_location = (self.agent_location[0], self.agent_location[1] + 1)
        elif obs == 1: #observation 1 means move to the left 
            if self.agent_location[1] > 0:
                self.agent_location = (self.agent_location[0], self.agent_location[1] - 1)

        distance_to_reward_location = np.sqrt((self.agent_location[0] - self.reward_location[0])**2 + (self.agent_location[1] - self.reward_location[1])**2)
        
        maximum_distance = np.sqrt((self.grid_size[0] - 1)**2 + (self.grid_size[1] - 1)**2)
    

        print(f"Agent location: {self.agent_location}")
        print(f"Distance to reward location: {distance_to_reward_location}")

        probabilities = [0.5, 0.5]
        if distance_to_reward_location == 0:  # on the point
            signal = 0
        else:
            # sampling randomly from a distance across 0 and 1
            # the probabilities of the reward depend on the distance


            probabilities = np.array(
                [
                    0.5 - ((maximum_distance - distance_to_reward_location) / maximum_distance) / 2,
                    0.5 + ((maximum_distance - distance_to_reward_location) / maximum_distance) / 2,
                ]
            )

            # TODO can be a new parameter to make this a non-linear probability distribution

            print(f"Probabilities for external agent action sampling: {probabilities}")
            signal = np.random.choice([1, 0], p=probabilities)
        print(f"Environmental signal : {signal}")

        return signal

#%%

policy_len = 1
pE = np.array([1]*2*policy_len).astype(float)
lr_pE = 0.0
gamma_G = 0.95

pD = utils.obj_array(1)
pD[0] =  np.array([0.5, 0.5])

def plot_grid(grid, trial):
    plt.imshow(grid)
    fn = f"temp/{trial}_{t}.png"
    plt.savefig(fn)
    grid_images.append(imageio.imread(fn))
    os.remove(fn)
    #plt.show()
    plt.clf()

external_cells = []

num_external_cells = 5

for e in range(num_external_cells):
    external_cells.append(GridWorldCell(reward_location, grid_size, agent_location))

sensory_cell = NeuronalCell(0, [e.id for e in external_cells], [1.0], alpha = 0.1, action_sampling="deterministic", lr_pE = 0, use_utility = True, inference_algo = "VANILLA", pD = pD, lr_pD = 0.01, lr_pB = 0.5,save_belief_hist = True) #,policy_sep_prior=True ) #, inference_horizon = 2, pD = pD, lr_pD = 0.01,policy_len = policy_len, gamma=gamma_G, pE = pE, lr_pE = lr_pE) #, use_param_info_gain = True)#, alpha = 0.5)#, inference_algo = "VANILLA")#,  gamma = gamma_G)

sensory_cell = NeuronalCell(0, [e.id for e in external_cells], [1.0], alpha = 16, action_sampling="deterministic",  inference_algo = "MMP", inference_horizon = 2, use_utility = True, lr_pB = 0.5) #, )#, alpha = 0.5)#, inference_algo = "VANILLA")#,  gamma = gamma_G)

#%%
import matplotlib.pyplot as plt 
import imageio 
grid = get_grid(reward_location, agent_location, 1)
obs = 1

time_taken_per_trial = []
grid_images = []
B_over_time = []
pB_over_time = []
q_pi_over_time = []
actions_over_time = []
gamma_over_time = [sensory_cell.gamma]
efe_over_time = []
qs_over_time = []

observations_over_time = []

num_trials = 20

initial_D = np.copy(sensory_cell.D)

solved = []

utilities = []
info_gains = []


timesteps_per_trial = []
overall_t = 0
for i in range(1):
    if i == 0:
        REWARD_LOCATION = (0,int(grid_size[1]-1))
    else:
        REWARD_LOCATION = (0,0)

    reward_location = REWARD_LOCATION
    for e in external_cells:
        e.set_locations(reward_location, agent_location)
    sensory_cell.build_B()
    for trial in range(num_trials):
        timesteps_per_trial.append(overall_t)
        t = 0
        sensory_cell.qs = copy.deepcopy(initial_D)
        sensory_cell.qs_prev = copy.deepcopy(initial_D)
        sensory_cell.action = None
        sensory_cell.prev_obs = []
        sensory_cell.prev_actions = None
        # if trial > 0:
        #     sensory_cell.update_D()


        # if trial > 0 and sensory_cell.inference_algo == "MMP":

        #     sensory_cell.update_gamma()
        #     gamma_over_time.append(sensory_cell.gamma)

        while external_cells[0].agent_location != external_cells[0].reward_location:
            plot_grid(grid, trial)
            
            observation_signal = []

            for e in external_cells:
                signal = e.act(obs)
                observation_signal.append(signal)

            observations_over_time.append(observation_signal)
            if trial > 1:
                update_B = True
            else:
                update_B = False
            
            obs = sensory_cell.act(observation_signal, update_B = update_B)
            print()
            print(f"agent signal: {obs}")
            print(f"Qs: {sensory_cell.qs}")
            print(f"F: {sensory_cell.F}")
            print(f"Q pi: {sensory_cell.q_pi}")
            print(f"B: {sensory_cell.B}")
            print(f"G: {sensory_cell.G}")

            qs_over_time.append(sensory_cell.qs)
            
            B_over_time.append(sensory_cell.B)
            pB_over_time.append(sensory_cell.pB)
            q_pi_over_time.append(sensory_cell.q_pi[0]) #0 -> move right, 1-> move left
            grid = get_grid(external_cells[0].reward_location, external_cells[0].agent_location, signal)
            actions_over_time.append(obs)
            efe_over_time.append(sensory_cell.G)
            utilities.append(sensory_cell.utilities)
            info_gains.append(sensory_cell.info_gains)
            t+=1 
            overall_t += 1
            if t == 75:
                break

        print(f"Trial over, resetting locations")

        reward_location = REWARD_LOCATION
        agent_location = (0, int((grid_size[1]-1)/2))
        for e in external_cells:
            e.set_locations(reward_location, agent_location)

        time_taken_per_trial.append(t)


print("Plotting time taken")

plt.plot(time_taken_per_trial)
plt.xlabel("Trial")
plt.ylabel("Time taken")
plt.savefig("time_taken_per_trial.png")
plt.show()
plt.clf()

print("Plot probability of firing")
print(q_pi_over_time)



plt.plot(q_pi_over_time)
plt.xlabel("Timesteps")
plt.ylabel("Probability of firing")
plt.savefig("q_pi_over_time.png")
plt.ylim(0,1)
plt.show()

plt.clf()

print("Plotting actions over time")

plt.plot(actions_over_time)
for t in timesteps_per_trial:
    plt.axvline(x=t, color='black', linestyle='--', alpha = 0.5)
plt.xlabel("Timesteps")
plt.ylabel("Action")
plt.yticks([0,1], ['Fire', 'No Fire'])
plt.savefig("action_over_time.png")
plt.show()
plt.clf()



plt.plot(gamma_over_time)
plt.xlabel("Trials")
plt.ylabel("Gamma_G")
plt.savefig("gamma_over_time.png")
plt.show()
plt.clf()

print("Plotting EFE")

plt.plot([-1*efe[0] for efe in efe_over_time], label = "G for fire", alpha = 0.5)
plt.plot([-1*efe[1] for efe in efe_over_time], label = "G for not fire", alpha = 0.5)

for t in timesteps_per_trial:
    plt.axvline(x=t, color='black', linestyle='--', alpha = 0.5)
plt.xlabel("Timesteps")
plt.ylabel("Expected free energy")
plt.legend()

plt.savefig("G_over_time.png")
# plt.ylim(0,1)
plt.show()


print("Plotting EFE")

plt.plot([u[0] for u in utilities], color = 'lightblue', label = "Utility for fire", alpha = 0.5)
plt.plot([u[1] for u in utilities], color = 'darkblue', label = "Utility for no fire", alpha = 0.5)

for t in timesteps_per_trial:
    plt.axvline(x=t, color='black', linestyle='--', alpha = 0.5)
plt.xlabel("Timesteps")
plt.ylabel("Utility")
plt.legend()

plt.savefig("U_over_time.png")
plt.show()

plt.plot([u[0] for u in info_gains], color = 'darkblue', label = "Info gain for fire", alpha = 0.5)
plt.plot([u[1] for u in info_gains], color = 'darkgreen', label = "Info gain for no fire", alpha = 0.5)

for t in timesteps_per_trial:
    plt.axvline(x=t, color='black', linestyle='--', alpha = 0.5)
plt.xlabel("Timesteps")
plt.ylabel("Information gain")
plt.legend()

plt.savefig("I_over_time.png")
# plt.ylim(0,1)
plt.show()

gif_path = f"1d-grid-simulation.gif"

imageio.mimsave(gif_path, grid_images, fps=5)


print("Making B GIF")

make_B_gif(B_over_time[::10], "B_over_time.gif")
make_pB_gif(pB_over_time[::10], "pB_over_time.gif")


"""
What it learns in B is what to do in response to the environmental signal
which is inherently noisy at first 
and so i guess what it should learn is that no matter what it should go right 
but the only way to learn that is to keep going right no matter what

"""

# %%
