#%%
import numpy as np
import os 
import copy
try:
    from stemai.cells.agent_cell import NeuronalCell
except:
    os.chdir('../../')
    from stemai.cells.agent_cell import NeuronalCell


grid_size = (21,21)
grid = np.zeros(grid_size)

reward_location = (0, 20)
agent_location = (10, 10)

grid[reward_location] = 1
grid[agent_location] = 2

def get_grid(reward_location, agent_location, signal):
    grid_size = (21,21)
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

    def act(self, obs1: int, obs2: int) -> str:
        #print(f"Observation: {qs_obs}")
        #obs = np.random.choice(np.flatnonzero(qs_obs == np.array(qs_obs).max()))
        print(f"Obs: {(obs1, obs2)}")

        if obs1 == 0:
            if self.agent_location[1] < self.grid_size[1] - 1:
                self.agent_location = (self.agent_location[0], self.agent_location[1] + 1)
        elif obs1 == 1:
            if self.agent_location[1] > 0:
                self.agent_location = (self.agent_location[0], self.agent_location[1] - 1)

        if obs2 == 0:
            if self.agent_location[0] < self.grid_size[0] - 1:
                self.agent_location = (self.agent_location[0] + 1, self.agent_location[1])
        elif obs2 == 1:
            if self.agent_location[0] > 0:
                self.agent_location = (self.agent_location[0] - 1, self.agent_location[1])


        #maximum_distance = np.sqrt((self.grid_size[0] - 1)**2 + (self.grid_size[1] - 1)**2)
        # manhattan distance
        distance_to_reward_location = abs(self.agent_location[0] - self.reward_location[0]) + abs(
            self.agent_location[1] - self.reward_location[1]
        )
        maximum_distance = abs((self.grid_size[0] - 1) - self.reward_location[0]) + abs(
            (self.grid_size[1] - 1) - self.reward_location[1]
        )

        #distance_to_reward_location = np.sqrt((self.agent_location[0] - self.reward_location[0])**2 + (self.agent_location[1] - self.reward_location[1])**2)
        print(f"Agent location: {self.agent_location}")
        print(f"Distance to reward location: {distance_to_reward_location}")
        probabilities = [0.5, 0.5]
        if distance_to_reward_location == 0:  # on the point
            signal = 1
        elif distance_to_reward_location >= np.floor(maximum_distance):
            # completely rnadom
            signal = np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            # sampling randomly from a distance across 0 and 1
            # the probabilities of the reward depend on the distance
            
            # sampling randomly from a distance across 0 and 1
            # the probabilities of the reward depend on the distance
            # steepness = 1 / (distance_to_reward_location + 1)  # Adjust the steepness based on the distance
            # probabilities = np.array(
            #     [
            #         0.5 - steepness / 2,
            #         0.5 + steepness / 2,
            #     ]
            # )

            probabilities = np.array(
                [
                    0.5 - ((maximum_distance - distance_to_reward_location) / maximum_distance) / 2,
                    0.5 + ((maximum_distance - distance_to_reward_location) / maximum_distance) / 2,
                ]
            )
            
        

            # TODO can be a new parameter to make this a non-linear probability distribution

            print(f"Probabilities for external agent action sampling: {probabilities}")
            signal = np.random.choice([0, 1], p=probabilities)
            print(f"Environmental signal : {signal}")

        return signal

#%%

pE = np.array([2.3]*2).astype(float)
lr_pE = 0.9
gamma_G = 0.95 
external_cell  = GridWorldCell(reward_location, grid_size,  agent_location)
sensory_cell_1 = NeuronalCell(0, [external_cell.id], [1.0], action_sampling = 'deterministic', pE = pE, lr_pE = lr_pE, gamma = gamma_G)
sensory_cell_2 = NeuronalCell(0, [external_cell.id], [1.0], action_sampling = 'deterministic', pE = pE, lr_pE = lr_pE, gamma = gamma_G)
#%%
import matplotlib.pyplot as plt 
import imageio 
grid = get_grid(reward_location, agent_location, 1)
obs1 = 1
obs2 = 0

initial_D = copy.deepcopy(sensory_cell_1.D)

time_taken_per_trial = []
grid_images = []
gammas_over_time = {sensory_cell_1.node_idx: [], sensory_cell_2.node_idx: []}
for i in range(2):
    if i == 0:
        REWARD_LOCATION = (0,20)
    else:
        REWARD_LOCATION = (0,0)

    reward_location = REWARD_LOCATION
    external_cell.set_locations(reward_location, agent_location)
    sensory_cell_1.build_B()
    sensory_cell_2.build_B()
    for trial in range(15):
        t = 0
        sensory_cell_1.qs = copy.deepcopy(initial_D)
        sensory_cell_2.qs = copy.deepcopy(initial_D)
        sensory_cell_1.qs_prev = copy.deepcopy(initial_D)
        sensory_cell_2.qs_prev = copy.deepcopy(initial_D)
        sensory_cell_1.action = None
        sensory_cell_2.action = None
        sensory_cell_1.prev_obs = []
        sensory_cell_1.reset()
        sensory_cell_1.prev_actions = None
        sensory_cell_2.prev_obs = []
        sensory_cell_2.reset()
        sensory_cell_2.prev_actions = None
        if trial > 0:
            sensory_cell_1.update_gamma()
            sensory_cell_1.update_gamma()

        gammas_over_time[sensory_cell_1.node_idx].append(sensory_cell_1.gamma)
        gammas_over_time[sensory_cell_2.node_idx].append(sensory_cell_2.gamma)


        while external_cell.agent_location != external_cell.reward_location:

            print(f"Trial: {trial}, Time: {t}")
            plt.imshow(grid)
            fn = f"temp/{trial}_{t}.png"
            plt.savefig(fn)
            grid_images.append(imageio.imread(fn))
            os.remove(fn)

            #plt.show()
            plt.clf()
            signal = external_cell.act(obs1, obs2)
            print(f"Environment signal: {signal}")
            obs1 = sensory_cell_1.act([signal])
            obs2 = sensory_cell_2.act([signal])


            print(f"Sensory signal: {(obs1, obs2)}")

            print(sensory_cell_1.q_pi)
            print(sensory_cell_1.B)
            grid = get_grid(external_cell.reward_location, external_cell.agent_location, signal)
            t+=1 

            if t == 300:
                break

        print(f"Trial over, resetting locations")

        reward_location = REWARD_LOCATION
        agent_location = (10,10)
        external_cell.set_locations(reward_location, agent_location)

        time_taken_per_trial.append(t)


gif_path = f"2d-grid-simulation.gif"

imageio.mimsave(gif_path, grid_images, fps=5)

"""
What it learns in B is what to do in response to the environmental signal
which is inherently noisy at first 
and so i guess what it should learn is that no matter what it should go right 
but the only way to learn that is to keep going right no matter what

"""

# %%
