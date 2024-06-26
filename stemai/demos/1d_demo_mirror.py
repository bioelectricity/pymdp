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
import matplotlib.pyplot as plt 
import imageio 
import time

# Previous obsection: [[[0.24074074074074076 0.7592592592592593]]]
# Lh_seq: [array([array([0.3116776459424052, 0.6883223540575949], dtype=object)],
#        dtype=object)       

# Previous obsection: [array([array([1., 0.])], dtype=object)]
# Lh_seq: [array([array([0.86319311, 0.13680689])], dtype=object)]

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

def make_A_gif(arrays, fn):
    images = []
    for arr in arrays:

        arr = arr[0]
        fig = plt.figure()

        plt.imshow(arr, cmap='gray', vmin=0, vmax=1)
        plt.xticks([0, 1], labels = ['Obs: Fire', 'Obs: Not Fire'])
       # plt.xticklabels(['Obs: Fire', 'Obs: Not Fire'])
        plt.xlabel("o_t")
        plt.yticks([0, 1], labels = ['State: Fire', 'State: Not Fire'])
       # axs[0].set_yticklabels(['State: Fire', 'State: Not Fire'])
        plt.ylabel("s_{t}")
        plt.title('Fire')
        plt.colorbar()

        plt.savefig("temp_A.png")
        images.append(imageio.imread("temp_A.png"))
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

def plot_grid(grid_images, grid, trial, t):
    plt.imshow(grid)
    fn = f"temp/{trial}_{t}.png"
    plt.savefig(fn)
    grid_images.append(imageio.imread(fn))
    os.remove(fn)
    #plt.show()
    plt.clf()

def get_grid(reward_location, agent_location, signal, grid_size):
    grid = np.zeros(grid_size)
    if signal == 0:
        grid[reward_location] = 0.5
    else:
        grid[reward_location] = 1.5
    grid[agent_location] = 2
    return grid



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
            else:
                self.agent_location = (0,0)
        elif obs == 1: #observation 1 means move to the left 
            if self.agent_location[1] > 0:
                self.agent_location = (self.agent_location[0], self.agent_location[1] - 1)
            else:
                self.agent_location = (self.grid_size[0], self.grid_size[1])

        distance_to_reward_location = np.sqrt((self.agent_location[0] - self.reward_location[0])**2 + (self.agent_location[1] - self.reward_location[1])**2)
        
        maximum_distance = np.sqrt((self.grid_size[0] - 1)**2 + (self.grid_size[1] - 1)**2) - 3
    

        print(f"Agent location: {self.agent_location}")
        print(f"Distance to reward location: {distance_to_reward_location}")

        if agent_location[1] < 10: #between 0 and 10: fire 
            probabilities = [1,0]
        elif agent_location[1] < 20: #between 10 and 20 -> probabilities as distance to 10
            maximum_distance = 10
            checkpoint = (1,10)
            distance = np.sqrt((self.agent_location[0] - checkpoint)**2 + (self.agent_location[1] - checkpoint)**2)

            probabilities = np.array(
                [
                    ((maximum_distance - (distance-1)) / maximum_distance) / 2,
                    1 - ((maximum_distance - (distance -1) ) / maximum_distance) / 2,

                ]
            )
        elif agent_location[1] < 30: #between 20 and 30 : no fire 
            probabilities = [0,1]
        elif agent_location[1] < 40: #between 30 and 40: probabilities as distance to 40
            checkpoint = (1,40)
            distance = np.sqrt((self.agent_location[0] - checkpoint)**2 + (self.agent_location[1] - checkpoint)**2)

            probabilities = np.array(
                [
                    ((maximum_distance - (distance-1)) / maximum_distance) / 2,
                    1 - ((maximum_distance - (distance -1) ) / maximum_distance) / 2,

                ]
            )
        signal = np.random.choice([0,1], p=probabilities)

        print(f"Probabilities: {probabilities}")
        print(f"Signal: {signal}")


       # return probabilities
        return probabilities, signal

#%%
grid_size = (1,51)
grid = np.zeros(grid_size)
import time
reward_location = (0, int(grid_size[1]-1))
agent_location = (0, int((grid_size[1]-1)/2))

grid[reward_location] = 1
grid[agent_location] = 2

policy_len = 1
pE = np.array([1]*2*policy_len).astype(float)
lr_pE = 0.0
gamma_G = 0.95

pD = utils.obj_array(1)
pD[0] =  np.array([0.5, 0.5])

pC = utils.obj_array(1)
pC[0] =  np.array([0.5, 0.5])

distr_obs = False
use_param_info_gain = True

lr_pC = 0.1
GAMMA_A_UPDATE = True
lr_pB = 0.01
lr_pE = 0
#sensory_cell = NeuronalCell(0, [e.id for e in external_cell s], np.array([[0.05,0.05]]*num_external_cells + [[0.05,0.05]]), alpha = 0.1, action_sampling="deterministic", lr_pE = 0, use_utility = True, inference_algo = "VANILLA", pD = pD, lr_pD = 0.01, lr_pB = 0.5,save_belief_hist = True) #,policy_sep_prior=True ) #, inference_horizon = 2, pD = pD, lr_pD = 0.01,policy_len = policy_len, gamma=gamma_G, pE = pE, lr_pE = lr_pE) #, use_param_info_gain = True)#, alpha = 0.5)#, inference_algo = "VANILLA")#,  gamma = gamma_G)

#sensory_cell.D = np.array([[0.5,0.5], [0.5,0.5]])
#%%



def run(num_trials, reward_location, agent_location):
    external_cells = []

    num_external_cells = 1

    for e in range(num_external_cells):
        external_cells.append(GridWorldCell(reward_location, grid_size, agent_location))


    sensory_cell = NeuronalCell(0, [e.id for e in external_cells], np.array([[0.05,0.05]]*num_external_cells),action_sampling="deterministic",  inference_algo = "MMP",  lr_pE = lr_pE, inference_horizon = 2, policy_len = policy_len, use_utility = True, use_param_info_gain = use_param_info_gain, pD = pD, pC=pC, lr_pC = lr_pC, lr_pB = lr_pB, distr_obs = distr_obs) #, )#, alpha = 0.5)#, inference_algo = "VANILLA")#,  gamma = gamma_G)

    grid = get_grid(reward_location, agent_location, 1, grid_size)
    sensory_action = 1

    time_taken_per_trial = []
    grid_images = []
    B_over_time = []
    pB_over_time = []
    q_pi_over_time = []
    actions_over_time = []
    gamma_over_time = [sensory_cell.gamma]
    efe_over_time = []
    qs_over_time = []
    gamma_A_over_trials = []
    C_over_trials = []

    observations_over_time = []

    A_over_trials = []

    initial_D = np.copy(sensory_cell.D)

    solved = []

    utilities = []
    info_gains = []
    param_info_gains = []

    modality_precisions = []


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
            print(f"TRIAL : {trial}")
            timesteps_per_trial.append(overall_t)
            t = 0
            sensory_cell.qs = copy.deepcopy(initial_D)
            sensory_cell.qs_prev = copy.deepcopy(initial_D)
            sensory_cell.action = None
            sensory_cell.prev_obs = []
            sensory_cell.prev_actions = None

            print(f"A: { sensory_cell.A}")
            A_over_trials.append(sensory_cell.A)
            C_over_trials.append(sensory_cell.C)
            gamma_A_over_trials.append(sensory_cell.gamma_A[0])
        
            if trial > 0:
                sensory_cell.update_after_trial(gamma_A_update = GAMMA_A_UPDATE)
                #sensory_cell.update_D()
                # sensory_cell.lr_pC *= 1.2


            if trial > 0 and sensory_cell.inference_algo == "MMP":

               # sensory_cell.update_gamma()

                gamma_over_time.append(sensory_cell.gamma)

            while external_cells[0].agent_location != external_cells[0].reward_location:
                plot_grid(grid_images, grid, trial, t)
                
                observation_signal = []
                observation_distribution = []

                for e in external_cells:
                    probabilities, signal = e.act(sensory_action)
                    observation_distribution.append(probabilities)
                    observation_signal.append(signal)

                print(f"External signal: {observation_signal}")
                print(f"External distribution: {observation_distribution}")
                print(f"Distr obs: {sensory_cell.distr_obs}")
                #time.sleep(5)

                observations_over_time.append(np.array(observation_signal))
                # if trial > 1:
                #     update_B = True
                # else:
                #     update_B = False

                update_B = True

                A_modality_precision =  sensory_cell.gamma_A[0][0] + sensory_cell.gamma_A[0][1]
                modality_precisions.append(A_modality_precision)

                if distr_obs:
                    sensory_action = sensory_cell.act(np.array(observation_distribution), update_B = update_B)
                else:
                    sensory_action = sensory_cell.act(np.array(observation_signal), update_B = update_B)
                print()
                print(f"agent signal action: {sensory_action}")
                print(f"Qs: {sensory_cell.qs}")
                print(f"F: {sensory_cell.F}")
                print(f"Q pi: {sensory_cell.q_pi}")
                print(f"B: {sensory_cell.B}")
                print(f"G: {sensory_cell.G}")
                print(f"Gamma A: {sensory_cell.gamma_A}")
                # if trial > 0:
                #     sensory_cell.update_gamma_A(np.array(observation_signal), sensory_cell.qs_over_time[-1], modalities = None)
                qs_over_time.append(sensory_cell.qs)
                
                B_over_time.append(sensory_cell.B)
                pB_over_time.append(sensory_cell.pB)
                q_pi_over_time.append(sensory_cell.q_pi[0]) #0 -> move right, 1-> move left
                grid = get_grid(external_cells[0].reward_location, external_cells[0].agent_location, signal, grid_size)
                actions_over_time.append(sensory_action)
                efe_over_time.append(sensory_cell.G)
                utilities.append(sensory_cell.utilities)
                info_gains.append(sensory_cell.info_gains)
                param_info_gains.append(sensory_cell.param_info_gains)
                t+=1 
                overall_t += 1
                
                
                #sensory_cell.update_gamma_A(observation_signal, sensory_cell.qs, modalities = None )

                if t == 500:
                    solved.append(False)
                    break


            print(f"Trial over, resetting locations")

            reward_location = REWARD_LOCATION
            agent_location = (0, int((grid_size[1]-1)/2))
            for e in external_cells:
                e.set_locations(reward_location, agent_location)

            time_taken_per_trial.append(t)
            #gamma_A_over_trials.append(sensory_cell.gamma_A)

            if t < 500:
                solved.append(True)

        
    return solved, sensory_cell, time_taken_per_trial, grid_images, B_over_time, pB_over_time, q_pi_over_time, actions_over_time, gamma_over_time, efe_over_time, qs_over_time, gamma_A_over_trials, observations_over_time, A_over_trials, utilities, info_gains, timesteps_per_trial, C_over_trials, modality_precisions, param_info_gains
 

def plot(dir):

    plt.plot(time_taken_per_trial)

    plt.xlabel("Trial")
    plt.ylabel("Time taken")
    plt.savefig(f"{dir}/time_taken_per_trial.png")
    plt.clf()
    with open(f"{dir}/time_taken_per_trial.txt", "w") as file:
        for time in time_taken_per_trial:
            file.write(f"{time}\n")

    print("Plot probability of firing")
    print(q_pi_over_time)

    plt.plot(q_pi_over_time)
    plt.xlabel("Timesteps")
    plt.ylabel("Probability of firing")
    plt.savefig(f"{dir}/q_pi_over_time.png")
    plt.ylim(0,1)
    plt.clf()
    np.save(f"{dir}/q_pi_over_time.npy", q_pi_over_time)

    print("Plotting actions over time")

    plt.plot(actions_over_time)
    for t in timesteps_per_trial:
        plt.axvline(x=t, color='black', linestyle='--', alpha = 0.5)
    plt.xlabel("Timesteps")
    plt.ylabel("Action")
    plt.yticks([0,1], ['Fire', 'No Fire'])
    plt.savefig(f"{dir}/action_over_time.png")
    plt.clf()

    np.save(f"{dir}/action_over_time.npy", actions_over_time)


    plt.plot(gamma_over_time)
    plt.xlabel("Trials")
    plt.ylabel("Gamma_G")
    plt.savefig(f"{dir}/gamma_over_time.png")
    plt.clf()
    np.save(f"{dir}/gamma_over_trials.npy", gamma_over_time)




    plt.plot([g[0] for g in gamma_A_over_trials], label = "gamma_A Fire")
    plt.plot([g[1] for g in gamma_A_over_trials], label = "gamma_A No Fire")

    plt.xlabel("Trials")
    plt.ylabel("Gamma_A")
    plt.legend()
    plt.savefig(f"{dir}/gamma_A_over_time.png")
    plt.clf()
    np.save(f"{dir}/gamma_A_over_trials.npy", gamma_A_over_trials)

    plt.plot( modality_precisions)

    plt.xlabel("Trials")
    plt.ylabel("Gamma_A_modality")
    plt.savefig(f"{dir}/gamma_A_modality_over_time.png")
    plt.clf()

    plt.plot([-1*efe[0] for efe in efe_over_time], label = "G for fire", alpha = 0.5)
    plt.plot([-1*efe[1] for efe in efe_over_time], label = "G for not fire", alpha = 0.5)

    for t in timesteps_per_trial:
        plt.axvline(x=t, color='black', linestyle='--', alpha = 0.5)
    plt.xlabel("Timesteps")
    plt.ylabel("Expected free energy")
    plt.legend()

    plt.savefig(f"{dir}/G_over_time.png")
    # plt.ylim(0,1)
    plt.clf()

    np.save(f"{dir}/efe_over_time.npy", efe_over_time)

    plt.plot([p[0] for p in param_info_gains], label = "pB Info gain for fire", alpha = 0.5)
    plt.plot([p[1] for p in param_info_gains], label = "pB info gain for not fire", alpha = 0.5)

    for t in timesteps_per_trial:
        plt.axvline(x=t, color='black', linestyle='--', alpha = 0.5)
    plt.xlabel("Timesteps")
    plt.ylabel("pB info gain")
    plt.legend()

    plt.savefig(f"{dir}/pB_info_gain_over_time.png")
    # plt.ylim(0,1)
    plt.clf()

    np.save(f"{dir}/param_info_gains.npy", param_info_gains)

    plt.plot([(efe[0])-efe[1] for efe in efe_over_time])

    for t in timesteps_per_trial:
        plt.axvline(x=t, color='black', linestyle='--', alpha = 0.5)
    plt.xlabel("Timesteps")
    plt.ylabel("Difference in expected free energy for policies")
    plt.legend()

    plt.savefig(f"{dir}/difference_G_over_time.png")
    # plt.ylim(0,1)
    plt.clf()


    plt.plot([u[0] for u in utilities], color = 'lightblue', label = "Utility for fire", alpha = 0.5)
    plt.plot([u[1] for u in utilities], color = 'darkblue', label = "Utility for no fire", alpha = 0.5)

    for t in timesteps_per_trial:
        plt.axvline(x=t, color='black', linestyle='--', alpha = 0.5)
    plt.xlabel("Timesteps")
    plt.ylabel("Utility")
    plt.legend()

    plt.savefig(f"{dir}/U_over_time.png")
    plt.clf()
    np.save(f"{dir}/utilities.npy", utilities)



    plt.plot([u[0]-u[1] for u in utilities], color = 'darkblue')

    for t in timesteps_per_trial:
        plt.axvline(x=t, color='black', linestyle='--')
    plt.xlabel("Timesteps")
    plt.ylabel("Difference in utility for policies")

    plt.savefig(f"{dir}/difference_U_over_time.png")
    plt.clf()
    plt.plot([u[0] for u in info_gains], color = 'darkblue', label = "Info gain for fire", alpha = 0.5)
    plt.plot([u[1] for u in info_gains], color = 'darkgreen', label = "Info gain for no fire", alpha = 0.5)

    for t in timesteps_per_trial:
        plt.axvline(x=t, color='black', linestyle='--', alpha = 0.5)
    plt.xlabel("Timesteps")
    plt.ylabel("Information gain")
    plt.legend()
    plt.savefig(f"{dir}/I_over_time.png")
    # plt.ylim(0,1)
    plt.clf()
    np.save(f"{dir}/info_gains.npy", info_gains)

    plt.plot([u[0]-u[1] for u in info_gains])

    for t in timesteps_per_trial:
        plt.axvline(x=t, color='black', linestyle='--', alpha = 0.5)
    plt.xlabel("Timesteps")
    plt.ylabel("Difference in info gain for policies over time")

    plt.savefig(f"{dir}/difference_I_over_time.png")
    # plt.ylim(0,1)
    plt.clf()

    plt.plot([C[0][0] for C in C_over_trials], color = 'darkblue', label = "Preference for fire", alpha = 0.5)
    plt.plot([C[0][1] for C in C_over_trials], color = 'darkgreen', label = "Preference for no fire", alpha = 0.5)

    plt.xlabel("Timesteps")
    plt.ylabel("Preferences")
    plt.legend()

    plt.savefig(f"{dir}/C_over_time.png")
    plt.clf()

    np.save(f"{dir}/C.npy", C_over_trials)

    gif_path = f"{dir}/1d-grid-simulation.gif"

    imageio.mimsave(gif_path, grid_images, fps=5)


    print("Making B GIF")
    np.save(f"{dir}/B_over_time.npy", B_over_time)
    np.save(f"{dir}/A_over_trials.npy", A_over_trials)

    make_B_gif(B_over_time[::10], f"{dir}/B_over_time.gif")
    # #make_pB_gif(pB_over_time[::10], "pB_over_time.gif")
    make_A_gif(A_over_trials, f"{dir}/A_over_trials.gif")

    """
    What it learns in B is what to do in response to the environmental signal
    which is inherently noisy at first 
    and so i guess what it should learn is that no matter what it should go right 
    but the only way to learn that is to keep going right no matter what

    """

# %%


num_runs = 1

num_trials = 1

for r in range(num_runs):
    existing_files = [int(f.split("-")[0]) for f in os.listdir("out-mirror") if "DS" not in f]
    if len(existing_files) == 0:
        run_dir_num = 0
    else:
        run_dir_num = max(existing_files) + 1
    
    solved, sensory_cell, time_taken_per_trial, grid_images, B_over_time, pB_over_time, q_pi_over_time, actions_over_time, gamma_over_time, efe_over_time, qs_over_time, gamma_A_over_trials, observations_over_time, A_over_trials, utilities, info_gains, timesteps_per_trial, C_over_trials, modality_precisions, param_info_gains = run(num_trials, reward_location, agent_location)
    if solved[-1]:
        dir = f"out-mirror/{run_dir_num}-right"
    else:
        dir = f"out-mirror/{run_dir_num}-left"
    os.makedirs(dir)


    sensory_cell_data = {
        'D': sensory_cell.D.tolist(),
        'B': sensory_cell.B.tolist(),
        'pB': sensory_cell.pB.tolist(),
        'A': sensory_cell.A.tolist(),
        'C': sensory_cell.C.tolist(),
        'gamma': sensory_cell.gamma,
        'alpha': sensory_cell.alpha,
        'inference_algo': sensory_cell.inference_algo,
        'lr_pE': sensory_cell.lr_pE,
        'lr_pD': sensory_cell.lr_pD,
        'lr_pC': sensory_cell.lr_pC,
        'lr_pB': sensory_cell.lr_pB,
        'use_utility': sensory_cell.use_utility,
        'inference_horizon': sensory_cell.inference_horizon
    }

    with open(f"{dir}/sensory_cell_data.txt", 'w') as file:
        for param, value in sensory_cell_data.items():
            file.write(f"{param} : {value}" + "\n")

    plot(dir)

raise

# %%
def plot_averages(dir, filename, quantity, negative = False):
    Gs = []
    for subdir in os.listdir(dir):
        if "DS" in subdir or "png" in subdir:
            continue
        G = np.load(f"{dir}/{subdir}/{filename}", allow_pickle=True)
        Gs.append(G)

    max_length = max(len(G) for G in Gs)
    Gs_padded = [np.pad(G, ((0, max_length - G.shape[0]), (0, 0)), 'constant', constant_values=np.nan) for G in Gs]

    Gs = Gs_padded

    Gs = np.array(Gs)

    mean_G = np.mean(Gs,axis = 0)
    std_G = np.std(Gs, axis=0)

    mean_G_0 = mean_G[:,0]
    std_G_0 = std_G[:,0]
    mean_G_1 = mean_G[:,1]
    std_G_1 = std_G[:,1]
    plt.figure(figsize=(10, 6))

    if negative:
        mean_G_0 *= -1
        mean_G_1 *= -1

    # Plotting mean_G with standard deviation as shaded area
    plt.plot(mean_G_0, color = 'darkblue', label=f'Mean {quantity} for fire')
    plt.plot(mean_G_1, color = 'mediumseagreen', label=f'Mean {quantity} for no fire')

    plt.fill_between(range(len(mean_G_0)), mean_G_0 - std_G_0, mean_G_0 + std_G_0, color='lightblue', alpha=0.2)
    plt.fill_between(range(len(mean_G_1)), mean_G_1 - std_G_1, mean_G_1 + std_G_1, color='lightgreen', alpha=0.2)

    plt.xlabel('Time', fontsize = 12)
    plt.ylabel(f'Mean {quantity}', fontsize = 12)
    plt.legend(fontsize = 12)
    plt.title(f"Mean {quantity}")
    plt.savefig(f"figs/mean_{quantity}_with_std.png")
    plt.show()

# %%
plot_averages("out", "efe_over_time.npy", "EFE", negative = True)
# %%

# %%
plot_averages("out","info_gains.npy", "Info gain")
# %%
plot_averages("out","utilities.npy", "Utility")
#%%
plot_averages("out","gamma_A_over_trials.npy", "Gamma A")




# %%
def plot_times(dir):
    times = []
    for subdir in os.listdir(dir):
        if "DS" in subdir or "png" in subdir:
            continue
        filename = f"{dir}/{subdir}/time_taken_per_trial.txt"
        t = []
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                t.append(float(line.strip("\n")))
        times.append(np.array(t))

    times = np.array(times)

    mean_G = np.mean(times,axis = 0)
    std_G = np.std(times, axis=0)

    plt.figure(figsize=(10, 6))


    # Plotting mean_G with standard deviation as shaded area
    for t in times:
        plt.plot(t, alpha = 0.5, color = 'lightblue')
    plt.plot(mean_G, color = 'darkblue', label=f'Mean time taken for fire')

    plt.fill_between(range(len(mean_G)), mean_G - std_G, mean_G + std_G, color='lightblue', alpha=0.2)

    plt.xlabel('Trials', fontsize = 12)
    plt.ylabel(f'Mean time taken', fontsize = 12)
    plt.title(f"Mean time taken over trials")
    plt.savefig(f"figs/mean_time_with_std.png")
    plt.show()

plot_times("out")
# %%

dir = "out"
filename = "C.npy"
Cs = []
for subdir in os.listdir(dir):
    if "DS" in subdir or "png" in subdir:
        continue
    G = np.load(f"{dir}/{subdir}/{filename}", allow_pickle=True)
    Cs.append(np.array(np.array([g for g in G[:,0]],dtype=np.float64), dtype = np.float64))
# %%

Cs = np.array(Cs, dtype = np.float64)
mean_G = np.mean(Cs,axis = 0)
std_G = np.std(Cs, axis=0)
# %%
mean_G_0 = mean_G[:,0]
std_G_0 = std_G[:,0]
mean_G_1 = mean_G[:,1]
std_G_1 = std_G[:,1]
plt.figure(figsize=(10, 6))


# Plotting mean_G with standard deviation as shaded area
plt.plot(mean_G_0, color = 'darkblue', label=f'Mean preferences for fire')
plt.plot(mean_G_1, color = 'mediumseagreen', label=f'Mean preferences for no fire')

plt.fill_between(range(len(mean_G_0)), mean_G_0 - std_G_0, mean_G_0 + std_G_0, color='lightblue', alpha=0.2)
plt.fill_between(range(len(mean_G_1)), mean_G_1 - std_G_1, mean_G_1 + std_G_1, color='lightgreen', alpha=0.2)

plt.xlabel('Time', fontsize = 12)
plt.ylabel(f'Mean preferences', fontsize = 12)
plt.legend(fontsize = 12)
plt.title(f"Mean preferences")
plt.savefig(f"figs/mean_preferences_with_std.png")
plt.show()

# %%
