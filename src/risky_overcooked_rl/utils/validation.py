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
        self._rationality = 5


    ################################################################
    # Train/Test Rollouts   ########################################
    ################################################################
    def trajectory_rollout(self,it, joint_trajectory):
        self.model.rationality = self._rationality
        self.env.reset()
        joint_actions = []
        joint_action_idxs = []
        p_slips = []

        for player in self.env.state.players:
            self.add_held_obj(player,'onion')

        for action1, action2, prob_slip in joint_trajectory:
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
            joint_action_idx = Action.ALL_JOINT_ACTIONS.index(joint_action)
            joint_actions.append(joint_action)
            joint_action_idxs.append(joint_action_idx)
            p_slips.append(prob_slip)



        # Random start state if specified
        # if it / self.ITERATIONS < self.perc_random_start:
        #     self.env.state = self.random_start_state()

        losses = []
        cum_reward = 0
        cum_shaped_reward = np.zeros(2)
        for t in count():
            self.env.mdp.p_slip = p_slips[t]
            joint_action = joint_actions[t]
            joint_action_idx = joint_action_idxs[t]

            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state,device=device).unsqueeze(0)
            # joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs, epsilon=self._epsilon)
            next_state_prospects = self.mdp.one_step_lookahead(self.env.state.deepcopy(),
                                                               joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx],
                                                               as_tensor=True, device=device)
            next_state, reward, done, info = self.env.step(joint_action)

            # Track reward traces
            shaped_rewards = self._rshape_scale * np.array(info["shaped_r_by_agent"])
            if self.shared_rew: shaped_rewards = np.mean(shaped_rewards)*np.ones(2)
            total_rewards =  np.array([reward + shaped_rewards]).flatten()
            cum_reward += reward
            cum_shaped_reward += shaped_rewards

            print(reward)

            # Store in memory ----------------
            self.model.memory_double_push(state=obs,
                                        action=joint_action_idx,
                                        rewards = total_rewards,
                                        next_prospects=next_state_prospects,
                                        done = done)
            # Update model ----------------
            loss = self.model.update()
            if loss is not None: losses.append(loss)
            if done:  break
            self.env.state = next_state
        # mean_loss = np.mean(losses)
        return cum_reward, cum_shaped_reward#, mean_loss
    def OSA_test(self,it,joint_trajectory):
        self.model.rationality = self._rationality
        self.env.reset()
        joint_actions = []
        joint_action_idxs = []
        p_slips = []

        for player in self.env.state.players:
            self.add_held_obj(player, 'onion')

        for action1, action2, prob_slip in joint_trajectory:
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
            joint_action_idx = Action.ALL_JOINT_ACTIONS.index(joint_action)
            joint_actions.append(joint_action)
            joint_action_idxs.append(joint_action_idx)
            p_slips.append(prob_slip)

        # Random start state if specified
        # if it / self.ITERATIONS < self.perc_random_start:
        #     self.env.state = self.random_start_state()

        losses = []
        cum_reward = 0
        cum_shaped_reward = np.zeros(2)
        for t in count():
            self.env.mdp.p_slip = p_slips[t]
            joint_action = joint_actions[t]
            joint_action_idx = joint_action_idxs[t]
            old_state = self.env.state.deepcopy()
            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state, device=device).unsqueeze(0)
            # joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs, epsilon=self._epsilon)
            next_state_prospects = self.mdp.one_step_lookahead(self.env.state,
                                                               joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx],
                                                               as_tensor=True, device=device)
            next_state_prospects_state = self.mdp.one_step_lookahead(self.env.state,
                                                               joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx],
                                                               encoded=False)
            next_state, reward, done, info = self.env.step(joint_action)

            next_obs = self.mdp.get_lossless_encoding_vector_astensor(next_state, device=device).unsqueeze(0)
            assert np.any([torch.all(next_obs == prospect[1]).item() for prospect in next_state_prospects]), 'encoded not seen'
            for _,_next_state,_p_next_state in next_state_prospects_state:
                _obs = self.mdp.get_lossless_encoding_vector_astensor(_next_state, device=device).unsqueeze(0)
                assert np.any([torch.all(_obs==prospect[1]).item() for prospect in next_state_prospects]),'encoded not seen'

            # for _, _next_state, _p_next_state in next_state_prospects_state:
            #     if next_state == _next_state:
            #         assert _p_next_state == 1.0, 'next state not seen'
            # assert [np.where([(next_state==prospect[1]) for prospect in next_state_prospects_state])[0]]
            assert next_state in [prospect[1] for prospect in next_state_prospects_state], 'next state not in prospects'


            expected_prospects = 0
            for ip, player in enumerate(next_state.players):
                if self.mdp.is_water(player.position) and old_state.players[ip].has_object():
                    expected_prospects += 2
            if expected_prospects == 0: expected_prospects=1
            assert expected_prospects == len(next_state_prospects), 'prospects not as expected'

            # Track reward traces
            shaped_rewards = self._rshape_scale * np.array(info["shaped_r_by_agent"])
            if self.shared_rew: shaped_rewards = np.mean(shaped_rewards) * np.ones(2)
            total_rewards = np.array([reward + shaped_rewards]).flatten()
            cum_reward += reward
            cum_shaped_reward += shaped_rewards

            print(reward)

            # Store in memory ----------------
            self.model.memory_double_push(state=obs,
                                          action=joint_action_idx,
                                          rewards=total_rewards,
                                          next_prospects=next_state_prospects,
                                          done=done)
            # Update model ----------------
            loss = self.model.update()
            if loss is not None: losses.append(loss)
            if done:  break
            self.env.state = next_state
        # mean_loss = np.mean(losses)
        return cum_reward, cum_shaped_reward  # , mean_loss


def averse1(config):
    config['LAYOUT'] = "coordination_ring"
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

def slip_test(config):
    config['LAYOUT'] = "risky_slip_test"


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

    averse_joint_traj = [
        [E, X, 0.01],
        [E, W, 0.01],
        [S, N, 0.01],
        [S, N, 0.01],
        [W, E, 0.01],
        [W, E, 0.01],
        [N, S, 0.01],
        [N, S, 0.01],
    ]
    config["HORIZON"] = len(averse_joint_traj)-1

    validator = Validation(SelfPlay_QRE_OSA, config)
    for _ in range(100000):
        # validator.trajectory_rollout(0, averse_joint_traj)
        validator.OSA_test(0, averse_joint_traj)


def main():
    config = {
        'ALGORITHM': 'Boltzmann_QRE-DDQN-OSA',
        'Date': datetime.now().strftime("%m/%d/%Y, %H:%M"),

        # Env Params ----------------
        'LAYOUT': "coordination_ring_CLDE", 'HORIZON': 200, 'ITERATIONS': 15_000,
        'AGENT': None,  # name of agent object (computed dynamically)
        "obs_shape": None,  # computed dynamically based on layout
        "perc_random_start": 0.9,  # percentage of ITERATIONS with random start states
        # "shared_rew": False,                # shared reward for both agents
        "p_slip": 0.25,
        # Learning Params ----------------
        'epsilon_sched': [0.1, 0.1, 5000],  # epsilon-greedy range (start,end)
        'rshape_sched': [1, 0, 5_000],  # rationality level range (start,end)
        'rationality_sched': [0.0, 5, 5000],
        'lr_sched': [1e-2, 1e-4, 3_000],
        # 'test_rationality': 5,          # rationality level for testing
        'gamma': 0.95,  # discount factor
        'tau': 0.005,  # soft update weight of target network
        "num_hidden_layers": 3,  # MLP params
        "size_hidden_layers": 256,  # 32,      # MLP params
        "device": device,
        "minibatch_size": 256,  # size of mini-batches
        "replay_memory_size": 30_000,  # size of replay memory
        'clip_grad': 100,

    }
    slip_test(config)







if __name__ == "__main__":
    main()