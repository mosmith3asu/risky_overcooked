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
import random
from datetime import datetime
from risky_overcooked_rl.utils.trainer import Trainer
debug = False

class Validation(Trainer):
    def __init__(self,model_object,config):
        super().__init__(model_object,config)

        self._rshape_scale = 1.0


    ################################################################
    # Train/Test Rollouts   ########################################
    ################################################################
    def trajectory_rollout(self,it, joint_trajectory):
        self.model.rationality = self._rationality
        self.env.reset()

        # Random start state if specified
        if it / self.ITERATIONS < self.perc_random_start:
            self.env.state = self.random_start_state()

        losses = []
        cum_reward = 0
        cum_shaped_reward = np.zeros(2)
        for t in count():
            # TODO: Verify if observing correctly

            action1,action2, prob_slip = joint_trajectory[t]
            self.env.mdp.p_slip = prob_slip

            action1 = {'N': Direction.NORTH,
                       'S': Direction.SOUTH,
                       'E': Direction.EAST,
                       'W': Direction.WEST,
                       'X': Action.STAY,
                       'I': Action.INTERACT}[action1]
            action2 = {'N': Direction.NORTH,
                       'S': Direction.SOUTH,
                       'E': Direction.EAST,
                       'W': Direction.WEST,
                       'X': Action.STAY,
                       'I': Action.INTERACT}[action2]
            joint_action = (action1, action2)

            next_state, reward, done, info = self.env.step(joint_action)

            # Track reward traces
            shaped_rewards = self._rshape_scale * np.array(info["shaped_r_by_agent"])
            if self.shared_rew: shaped_rewards = np.mean(shaped_rewards)*np.ones(2)
            total_rewards =  np.array([reward + shaped_rewards]).flatten()
            cum_reward += reward
            cum_shaped_reward += shaped_rewards

            print(reward)

            # Store in memory ----------------
            # self.model.memory_double_push(state=obs,
            #                             action=joint_action_idx,
            #                             rewards = total_rewards,
            #                             next_prospects=next_state_prospects,
            #                             done = done)
            # Update model ----------------
            # loss = self.model.update()
            # if loss is not None: losses.append(loss)
            # if done:  break
            self.env.state = next_state
        # mean_loss = np.mean(losses)
        return cum_reward, cum_shaped_reward#, mean_loss


def main():

    config = {
        'ALGORITHM': 'Boltzmann_QRE-DDQN-OSA',
        'Date': datetime.now().strftime("%m/%d/%Y, %H:%M"),

        # Env Params ----------------
        # 'LAYOUT': "risky_coordination_ring", 'HORIZON': 200, 'ITERATIONS': 15_000,
        'LAYOUT': "coordination_ring_CLDE", 'HORIZON': 200, 'ITERATIONS': 15_000,

        # 'LAYOUT': "risky_cramped_room_CLCE", 'HORIZON': 200, 'ITERATIONS': 20_000,
        # 'LAYOUT': "cramped_room_CLCE", 'HORIZON': 200, 'ITERATIONS': 20_000,
        # 'LAYOUT': "super_cramped_room", 'HORIZON': 200, 'ITERATIONS': 10_000,
        # 'LAYOUT': "risky_super_cramped_room", 'HORIZON': 200, 'ITERATIONS': 10_000,
        'AGENT': None,                  # name of agent object (computed dynamically)
        "obs_shape": None,                  # computed dynamically based on layout
        "perc_random_start": 0.9,          # percentage of ITERATIONS with random start states
        "shared_rew": False,                # shared reward for both agents
        # Learning Params ----------------
        'epsilon_sched': [0.1,0.1,5000],         # epsilon-greedy range (start,end)
        'rshape_sched': [1,0,5_000],     # rationality level range (start,end)
        'rationality_sched': [0.0,5,5000],
        'lr_sched': [1e-2,1e-4,5_000],
        'test_rationality': 5,          # rationality level for testing
        'gamma': 0.95,                      # discount factor
        'tau': 0.005,                       # soft update weight of target network
        "num_hidden_layers": 3,             # MLP params
        "size_hidden_layers": 256,#32,      # MLP params
        "device": device,
        "minibatch_size":256,          # size of mini-batches
        "replay_memory_size": 30_000,   # size of replay memory
        'clip_grad': 100,

    }
    config['replay_memory_size'] = 30_000
    config['epsilon_sched'] = [1.0, 0.15, 10_000]
    config['rshape_sched'] = [1, 0, 10_000]
    config['rationality_sched'] = [5.0, 5.0, 10_000]
    config['lr_sched'] = [1e-2, 1e-4, 5_000]
    config['perc_random_start'] = 0.9
    config['test_rationality'] = config['rationality_sched'][1]
    config['tau'] = 0.01
    config['num_hidden_layers'] = 5
    config['size_hidden_layers'] = 128
    config['shared_rew'] = False
    config['gamma'] = 0.95
    config['note'] = 'increased gamma'

    S,W,N,E,X,I = 'S','W','N','E','X','I'
    validator = Validation(SelfPlay_QRE_OSA,config)
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
    validator.trajectory_rollout(0, averse_joint_traj)







if __name__ == "__main__":
    main()