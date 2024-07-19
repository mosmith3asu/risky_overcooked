import numpy as np
from risky_overcooked_rl.utils.deep_models import ReplayMemory,DQN_vector_feature,device,SelfPlay_QRE_OSA,SelfPlay_QRE_OSA_CPT
from risky_overcooked_rl.utils.rl_logger import RLLogger,TrajectoryVisualizer
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,OvercookedState,SoupState, ObjectState
from risky_overcooked_py.mdp.actions import Action,Direction
from itertools import product,count
import torch
import torch.optim as optim
import math
from datetime import datetime
import imageio
class GIF_maker():
    def __init__(self,LAYOUT,HORIZON=400):
        self.mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    def init_env(self,HORIZON):
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=HORIZON)
        self.env.reset()
        self.state_history = [self.env.state.deepcopy()]
        self.visualizer = TrajectoryVisualizer(self.env)
    def load_trajectory(self,p1_traj,p2_traj,prob_slips):
        self.init_env(len(p1_traj)+1)
        for a1, a2,p in zip(p1_traj, p2_traj,prob_slips):
            self.move_agents(a1, a2,p)
    def move_agents(self,action1,action2,prob_slip):
        action1 = {'N':Direction.NORTH,
                   'S':Direction.SOUTH,
                   'E':Direction.EAST,
                   'W':Direction.WEST,
                   'X':Action.STAY,
                   'I':Action.INTERACT}[action1]
        action2 = {'N': Direction.NORTH,
                   'S': Direction.SOUTH,
                   'E': Direction.EAST,
                   'W': Direction.WEST,
                   'X': Action.STAY,
                   'I': Action.INTERACT}[action2]
        joint_action = (action1,action2)
        self.mdp.p_slip = float(prob_slip)
        old_objects = [self.env.state.players[i].held_object for i in range(2)]

        next_state, reward, done, info = self.env.step(joint_action)
        next_state = next_state.deepcopy()
        self.state_history.append(next_state)

        if float(prob_slip) ==1:
            for i in range(2):
                in_puddle = self.env.mdp.is_water(next_state.players[i].position)
                lost_object = old_objects[i] is not None and next_state.players[i].held_object is None
                if in_puddle and lost_object:
                    # next_state = next_state.deepcopy()
                    next_state.players[i].held_object = old_objects[i]
                    next_state.players[i].orientation = Direction.FALL
                    self.state_history.append(next_state)
    def preview(self):
        self.visualizer.preview_trajectory(self.state_history)

    def make_gif(self,fname):
        imgs = self.visualizer.get_images(self.state_history)
        imageio.mimsave(f'{fname}.gif', imgs,loop=0,fps=5)


def extend_directions():
    """ Necessary for adding fall animation but breaks MDP in action selection"""
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    FALL = (0,0)

    Direction.FALL = FALL
    Direction.ALL_DIRECTIONS = INDEX_TO_DIRECTION = [NORTH, SOUTH, EAST, WEST,FALL]
    Direction.DIRECTION_TO_INDEX = {a: i for i, a in enumerate(INDEX_TO_DIRECTION)}
    Direction.OPPOSITE_DIRECTIONS = {NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST}
    Direction.DIRECTION_TO_NAME = {
        d: name
        for d, name in zip(
            [NORTH, SOUTH, EAST, WEST,FALL], ["NORTH", "SOUTH", "EAST", "WEST",'FALL']
        )
    }

def main():
    extend_directions()
    LAYOUT = 'risky_cramped_room'
    # LAYOUT = 'risky_coordination_ring'
    # LAYOUT = 'risky_walkaround'
    # LAYOUT = 'risky_roundabout'
    # LAYOUT = 'risky_forced_coordination'
    # LAYOUT = 'risky_forced_discoordination'
    LAYOUT = 'risky_multipath'
    # LAYOUT = 'risky_middlepuddle'
    S,W,N,E,X,I = 'S','W','N','E','X','I'
    gifer = GIF_maker(LAYOUT,HORIZON=200)
    seeking_joint_traj = [
    [W, S, 0],
    [W, I, 0], # P2 PICK ONION
    [S, E, 0],
    [S, E, 0],
    [I, N, 0], # P1 PICK ONION
    [E, N, 0],
    [E, I, 1], # P2 PLACE ONION 1 | P1 DROP ONION
    [W, W, 1],
    [W, W, 0],
    [I, S, 0], # P1 PICK ONION
    [E, S, 0],
    [E, I, 0], # P2 PICK ONION
    [N, N, 0],
    [N, N, 0],
    [I, E, 0],  # P1 PLACE ONION 2
    [S, E, 0],
    [S, N, 0],
    [W, I, 0], # P2 PLACE ONION 3
    [W, W, 0],
    [I, W, 0], # P1 PICK ONION
    [E, S, 0],
    [E,S, 0],
    [N,W, 0],
    [N,I, 0],
    [E,E, 0],
    [I,E, 1], # P2 DROP ONION 1 | P1 PALCE ONION 1
    [S,W, 0],
    [S,W, 0],
    [W,I, 0], #  P2 PICK ONION
    [W,N, 0],
    [I,N, 0], #  P1 PICK ONION
    [E,E, 0],
    [E,E, 0],
    [N,I, 0], #  P2 PLACE ONION 2
    [N,W, 0],
    [E,W, 0],
    [I,S, 0], #  P1 PLACE ONION 3
    [S,W, 0],
    [S,I, 0], #  P2 PICK DISH
    [W,N, 0],
    [W,N, 0],
    [N,E, 0],
    [W,E, 0],
    [I,N, 0], #  P1 PICK DISH
    [N,I, 0],#  P1 PICK SOUP
    [E,S, 0],
    [E,S, 1], #  P2 DROP SOUP
    [I,W, 0],
    [X,W, 0],
    [X,I, 0], #  P2 PICK ONION
    [X,N, 0],
    [X,N, 0],
    [S,E, 0],
    [X,E, 0],
    [X, N, 0],
    [X, I, 0],#  P2 PLACE ONION 1
    [N,W, 0],
    [E,W, 0],
    [I,S, 0], # P1 PICK SOUP
    [S,S, 0],
    [S,I, 0], #  P2 PICK ONION
    [W,N, 0],
    [S,N, 1], # P2 DROP ONION
    [I,S, 0],
    [X,X, 0],
    ]
    averse_joint_traj = [
        [S, W, 1],
        [W, I, 1],# P2 PICK UP ONION
        [X, E, 1],
        [X, N, 1],
        [X, I, 1], # P2 SETS DOWN ONION
        [I, W, 1], # P1 PICK UP ONION
        [N, I, 1], # P2 PICK UP ONION
        [I, N, 1], # P1 DELIVERS ONION 1
        [W, E, 1],
        [S, I, 1], # P2 SETS DOWN ONION
        [I, S, 1], # P1 PICK UP ONION
        [E, I, 1], # P2 PICK UP ONION
        [N, N, 1],
        [I, E, 1], # P1 DELIVERS ONION 2
        [S, I, 1], # P2 SETS DOWN ONION
        [W, S, 1],
        [I, I, 1], # P1 PICK UP ONION | P2 PICK UP ONION
        [N, N, 1],
        [I, E, 1], # P1 DELIVERS ONION 3
        [S, I, 1], # P2 SETS DOWN ONION
        [W, S, 1],
        [I, W, 1], # P1 PICK UP ONION
        [N, I, 1], # P2 PICK UP ONION
        [E, N, 1],
        [I, E, 1], # P1 DELIVERS ONION 1
        [W, I, 1], # P2 SETS DOWN ONION
        [S, S, 1],
        [I, I, 1], # P1 PICK UP ONION | P2 PICK UP ONION
        [E, E, 1],
        [I, N, 1], # P1 DELIVERS ONION 2
        [S, I, 1],  # P2 SETS DOWN ONION
        [W, W, 1],
        [I, N, 1], # P1 PICK UP ONION
        [N, W, 1],
        [E, I, 1], # P2 PICK UP DISH
        [I, E, 1], # P1 DELIVERS ONION 3
        [S, I, 1], # P2 SET DOWN DISH
        [W, S, 1],
        [I, I, 1], # P1 PICK UP DISH
        [N, N, 1],
        [I, N, 1], # P1 PICK UP SOUP | P2 DROP ONION
        [W, S, 0],
        [S, S, 0],
        [I, E, 0], # P1 SET DOWN SOUP
        [W, N, 0],
        [S, I, 0],# P2 PICK UP SOUP
        [W, S, 0],
        [I, I, 0], # P2 DELIVER SOUP | P1 PICK UP DISH
        [N, W, 0],
        [E, I, 0],
        [E, E, 0],
        [X, E, 1], # P2 DROP ONION
        [X, W, 1],
        [X, W, 1],
        [X, I, 1],
        [I, E, 1], # P1 PICK UP SOUP
        [S, E, 1], # P2 DROP ONION
        [W, W, 0],
        [I, N, 0],# P1 SET DOWN SOUP
        [N, I, 0],
        [W, S, 0],
        [E, I, 0], # P2 DELIVER SOUP
        [N, W, 0],  # P2 DELIVER SOUP
    ]


    # joint_traj = np.array(seeking_joint_traj)
    joint_traj = np.array(averse_joint_traj)
    gifer.load_trajectory(joint_traj[:,0], joint_traj[:,1],joint_traj[:,2])

    gifer.preview()
    # gifer.make_gif('risky_coordination_ring_averse')
    # gifer.make_gif('risky_coordination_ring_seeking')




if __name__=='__main__':
    main()