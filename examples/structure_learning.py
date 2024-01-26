"""This routine returns a generative model (mdp) in the form of an MDP,
given a sequence of outcomes. If the outcomes are not available, then
they can be generated automatically from a generative process, specified
with an MDP structure (see spm_MDP_structure_teaching). The generative
model learns from successive epochs of data generated under the first
level of each factor of the process. By exploring different extensions to
the model (using Bayesian model comparison) successive epochs are
assimilated under a model structure that accommodates paths. This routine
makes certain assumptions about the structural form of generative models
at any given level of a hierarchical model. These are minimal
assumptions:

i) Dynamics are conditionally independent of outcomes. This means that
the generative model can be factorised into a likelihood mapping (A) and
transition probabilities over latent states (B)

(ii) Latent states can be partitioned into factors, whose dynamics are
conditionally independent 

(iii) The dynamics for each factor can be partitioned into discrete
paths.

This leads to a generic form for any level of a hierarchical (deep)
 Markov decision process in which the likelihood mapping (for each
modality) is a tensor whose trailing dimensions correspond to the
dimensions of each factor. The (transition) priors are tensors, with a
probability transition matrix for each path. In addition, the initial
state and path of each factor are specified with D and E. 

With this form, structure learning can simply consider the addition of a latent state, a
latent path or a new factor.

It is assumed that the first path of any factor has no dynamics and
corresponds to an identity operator. Subsequent paths can have any form.
Because outcomes are assumed to be generated under the first level of
each factor, they generate the same outcome. In other words, the
likelihood mapping is shared by the first state of every factor. In turn,
this means that adding a factor entails adding a second state to the
implicit first state of the new factor.
"""

import numpy as np
from pymdp import Agent
from pymdp import utils 

num_factors = 1

num_states = [1]

num_modalities = 4
num_obs = [2,2,2,2]

num_paths = [1]


A = utils.obj_array(num_modalities)

#initialize our A matrix with a set number of observation modalities (and observations) and one state per state factor (an identity mapping from observations to states)
for m in range(num_modalities):
    A[m] = np.zeros((num_obs[m], num_states[0]))
    A[m][0,0] = 1

#initialized with stationary path 
B = utils.obj_array(num_factors)
B[0] = np.zeros((num_states[0], num_states[0], num_paths[0]))
B[0][0,0,0] = 1
#[[[1]]]

D = utils.obj_array_uniform(num_factors)

agent = Agent(A=A, B=B, D=D)

agent.num_paths = num_paths



def spm_expand(agent, observations, add_factor = False, add_state = False, add_path = False):
    """

    Function that expands an agent's GM by adding either: 1) additional state factor ; 2) additional state to a factor; 3) additional path 
    Augment with an additional path
    :param mdp: MDP structure
    :param add_factor : add a new state factor (with 2 states)
    :param add_state: add a state to the last factor
    :param add_path: add a path
    :param OPTIONS: Options for expansion
    :return: Updated MDP structure
    """
    # # Priors over initial states and control (first state and control)
    # for f in range(1, agent.num_factors + 1):
    #     if agent.D[f] is None:  # Pre-specified
    #         agent.D[f] = np.zeros((agent.num_paths[f - 1], 1), dtype=float)  # Initial state
    #         agent.D[f] = np.zeros((agent.num_paths[f - 1], 1), dtype=float)  # Initial path

    
    #adding an additional state factor to the agent's GM
    if add_factor: # Add factor: with 2 states, because the first is shared by all factors

        new_num_factors = agent.num_factors + 1
        new_pB = utils.obj_array(new_num_factors)

        for f in range(agent.num_factors):
            new_pB[f] = agent.pB[f]

        new_num_states = agent.num_states + [2] #updating it
        new_num_paths = agent.num_paths + [1]

        #New state factor 1 path: an identity mapping from each state to itself. Eg. if in state 1 / 2 at time t, will be in state 1 / 2 at time t+1 
        new_pB[agent.num_factors] = np.eye(new_num_states[-1], new_num_states[-1], new_num_paths[-1])
        #(2,2,1)

        agent.num_states = new_num_states
        agent.num_paths = new_num_paths 
        agent.num_factors = new_num_factors
        agent.pB = new_pB 

        agent.B = utils.norm_dist_obj_arr(new_pB) 

        agent.num_controls = new_num_paths

        #Initializing a new A matrix 
        new_pA = utils.obj_array(agent.num_modalities) 

        #for every modality, the mapping between the observations and the first state in every state factor is the same

        #A[0][:,0] = A[1][:,0] = A[2][:,0] = A[3][:,0]

        #For the new state factor: 
            #First state is mapped to the observations in the same way that the first state of all other state factors is mapped (TODO: how is this mapped?). 
            #The second state is 1-hot mapped directly to the observation you have received 

        for m in range(agent.num_modalities):
            new_pA[m] = np.zeros(([agent.num_obs[m]] + agent.num_states))
            #(num_obs[m], 2, 2)
            for f in range(agent.num_factors - 1):
                new_pA[m][:,f] = agent.A[m][f] #num_obs[m], 2
            new_pA[m][:, agent.num_factors + 1][:,0] = agent.A[m][agent.num_factors][:,0] #num_obs[m], 1
            new_pA[m][:, agent.num_factors + 1][:,1] = observations[m] #maybe we want to add some precision?          
        
        agent.pA = new_pA
        agent.A = utils.norm_dist_obj_arr(new_pA)


        new_D = utils.obj_array(agent.num_factors)
        for f in range(agent.num_factors-1):
            new_D[f] = agent.D[f]
        new_D[-1] = np.array([0,1]) #or maybe this is [0,1]? 

        agent.policies = agent._construct_policies() #do we need to update the pymdp num_controls variable?
        agent.E = agent._construct_E_prior()

    #adding a state to an agent's state factor 
    elif add_state:

        # Add latent state
        # Augment priors: the first path is a precise identity mapping

        last_factor_index = agent.num_factors - 1
        new_num_states = agent.num_states #[num states in factor1, num states in factor2]
        new_num_states[last_factor_index] += 1 #new state in the last factor 
        agent.num_states = new_num_states 

        new_pB = utils.obj_array(agent.num_factors)
        for f in range(agent.num_factors):
            if f == last_factor_index:
                new_pB[f] = np.eye((new_num_states[f], new_num_states[f], agent.num_paths[0]))
            else:
                new_pB[f] = agent.pB[f]
        
        agent.pB = new_pB
        agent.B = utils.norm_dist_obj_arr(new_pB)
        
        #likelihood 
        for m in range(agent.num_modalities):
            new_pA[m] = np.copy(agent.new_pA[m])
            new_pA[m] = np.insert(new_pA[m], last_factor_index, 1/agent.num_obs[m], axis=-1)
        
        agent.pA = new_pA
        agent.A = utils.norm_dist_obj_arr(new_pA)
    
        # New state is the initial state
        new_D = agent.D

        new_D[-1] = np.zeros((agent.num_states[-1]))
        new_D[-1][-1] = 1

        agent.D = new_D

        agent.policies = agent._construct_policies() #do we need to update the pymdp num_controls variable?
        agent.E = agent._construct_E_prior()

    #adding a path between states in a state factor 
    elif add_path:

        last_factor_index = agent.num_factors -1 
        new_pB = np.insert(agent.pB[last_factor_index], -1, 1/agent.num_states[last_factor_index], axis=-1)
        agent.pB[last_factor_index] = new_pB

        agent.B = utils.norm_dist_obj_arr(new_pB)

        agent.num_controls[-1] += 1

        agent.policies = agent._construct_policies() #do we need to update the pymdp num_controls variable?
        agent.E = agent._construct_E_prior()
    
