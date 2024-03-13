from cells.external_cell import ExternalCell
from pymdp.envs import TMazeEnv
from pymdp import utils
import itertools
import numpy as np

LOCATION_FACTOR_ID = 0
TRIAL_FACTOR_ID = 1

LOCATION_MODALITY_ID = 0
REWARD_MODALITY_ID = 1
CUE_MODALITY_ID = 2

REWARD_IDX = 1
LOSS_IDX = 2


class TMazeCell(ExternalCell):
    """A class representing an external cell
    The external cell will have a fixed random B matrix
    and it will not update B after state inference.

    The hidden state space of the external cell will be the external cell neighbors and the active cells
    and the control state space will be the external cell neighbors and the sensory cells"""

    def __init__(
        self,
        node,
        neighbors,
        external_cell_indices,
        active_cell_indices,
        sensory_cell_indices,
        states,
        reward_probs=None,
    ):

        super().__init__(
            node,
            neighbors,
            external_cell_indices,
            active_cell_indices,
            sensory_cell_indices,
            states,
        )

        self.tmaze = TMazeEnv(reward_probs=reward_probs)

        self.tmaze.reset()

        self.actions = list(itertools.product(list(range(len(self.active_cell_indices))), repeat=2))

    def act(self, obs: int, in_consistent_interval=False) -> str:
        """Perform state and action inference, return the action string
        which includes the action signal for each actionable neighbor
        of this cell

        obs: the observation signal index from the observable neighbors


        Here for the tmaze cell, the observation will be the location of the rat
        one out of four possible observations coming from the active cells
        """
        actions_per_factor = [obs, 0]
        env_step = self.tmaze.step(actions_per_factor)
        # sensory_states = list(itertools.product((0,1), repeat = 5))[:24]

        # actions = list(itertools.product(range(4), range(3), range(2)))
        # sensory_mapping = {a: s for a,s in zip(actions, sensory_states)}

        # TODO: how does this feed into the sensory observations?

        # we don't need 6 cells per se we just need enough for all combinations of observations

        # which is 4 locations, 3 rewards, 2 cues = 24
        # so how many cells do we need for 24 possible observations?
        # we need 5 cells to represent 32 possible observations
        # so this is defo worth discussing with the team...

        location = env_step[0]
        reward = env_step[1]  # ['No reward','Reward!','Loss!']

        cue = env_step[2]

        if int(reward) == 1 or in_consistent_interval is True:  # reward
            action_string = "0011"  # predictable reward that wouldn't occur otherwise
        else:
            action_string = ""

            # locations: ['CENTER','RIGHT ARM','LEFT ARM','CUE LOCATION']

            if location == 0:  # center
                action_string += "00"
            elif location == 1:  # right arm
                action_string += "01"
            elif location == 2:  # left arm
                action_string += "10"
            elif location == 3:  # cue location
                action_string += "11"

            if location == 3:  # cue location
                if cue == 0:
                    action_string += "00"
                elif cue == 1:
                    action_string += "01"
            else:
                action_string += "10"

        self.action_string = action_string

        return self.action_string, int(reward), int(location), int(cue)

    def _reset(self, state=None):
        return TMazeEnv.reset(self, state=state)
