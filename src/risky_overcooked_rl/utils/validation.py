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
from risky_overcooked_rl.utils.cirriculum import Curriculum

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

        # for player in self.env.state.players:
        #     self.add_held_obj(player,'onion')

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


        rollout_info = {
            'state_history': [],
            'onion_risked': np.zeros([0,2]),
            'onion_pickup': np.zeros([0, 2]),
            'onion_drop': np.zeros([0, 2]),
            'dish_risked': np.zeros([0, 2]),
            'dish_pickup': np.zeros([0, 2]),
            'dish_drop': np.zeros([0, 2]),
            'soup_pickup': np.zeros([0, 2]),
            'soup_delivery': np.zeros([0, 2]),

            'soup_risked':  np.zeros([0,2]),
            'onion_slip':  np.zeros([0,2]),
            'dish_slip':   np.zeros([0,2]),
            'soup_slip':   np.zeros([0,2]),
            'soup_drop': np.zeros([0, 2]),
            'onion_handoff':np.zeros([0,2]),
            'dish_handoff': np.zeros([0,2]),
            'soup_handoff': np.zeros([0,2]),
            'shaped_reward_hist': np.zeros([0,2]),
            'reward_hist': np.zeros([0,1]),
            'mean_loss': 0
        }

        # Random start state if specified
        # if it / self.ITERATIONS < self.perc_random_start:
        #     self.env.state = self.random_start_state()

        losses = []
        cum_reward = 0
        cum_shaped_reward = np.zeros(2)
        rollout_info['state_history'].append(self.env.state)
        for t in count():
            self.env.mdp.p_slip = p_slips[t]
            joint_action = joint_actions[t]
            joint_action_idx = joint_action_idxs[t]
            old_state = self.env.state.deepcopy()

            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state,device=device).unsqueeze(0)

            # joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs, epsilon=self._epsilon)
            # next_state_prospects = self.mdp.one_step_lookahead(self.env.state.deepcopy(),
            #                                                    joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx],
            #                                                    as_tensor=True, device=device)
            next_state, reward, done, info = self.env.step(joint_action,get_mdp_info=True)

            next_state_prospects = self.mdp.one_step_lookahead(old_state,
                                                               joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx],
                                                               as_tensor=True, device=device)

            rollout_info['state_history'].append(next_state)
            for key in rollout_info.keys():
                if key not in ['state_history','mean_loss','reward_hist','shaped_reward_hist']:
                    rollout_info[key] = np.vstack([rollout_info[key], np.array(info['mdp_info']['event_infos'][key])])

            # rollout_info['onion_slips'] = np.vstack([rollout_info['onion_slips'], np.array(info['mdp_info']['event_infos']['onion_slip'])])
            # rollout_info['onion_pickup'] = np.vstack([rollout_info['onion_pickup'], np.array(info['mdp_info']['event_infos']['onion_pickup'])])
            # rollout_info['onion_drop'] = np.vstack([rollout_info['onion_drop'], np.array(info['mdp_info']['event_infos']['onion_drop'])])
            # rollout_info['onion_handoff'] = np.vstack([rollout_info['onion_handoff'], np.array(info['mdp_info']['event_infos']['onion_handoff'])])
            # rollout_info['dish_slips'] = np.vstack([rollout_info['dish_slips'], np.array(info['mdp_info']['event_infos']['dish_slip'])])
            # rollout_info['soup_slips'] = np.vstack([rollout_info['soup_slips'], np.array(info['mdp_info']['event_infos']['soup_slip'])])
            # rollout_info['onion_risked'] = np.vstack([rollout_info['onion_risked'], np.array(info['mdp_info']['event_infos']['onion_risked'])])
            # rollout_info['dish_risked'] = np.vstack([rollout_info['dish_risked'], np.array(info['mdp_info']['event_infos']['dish_risked'])])
            # rollout_info['soup_risked'] = np.vstack([rollout_info['soup_risked'], np.array(info['mdp_info']['event_infos']['soup_risked'])])
            # rollout_info['dish_handoff'] = np.vstack([rollout_info['dish_handoff'], np.array(info['mdp_info']['event_infos']['dish_handoff'])])
            # rollout_info['soup_handoff'] = np.vstack([rollout_info['soup_handoff'], np.array(info['mdp_info']['event_infos']['soup_handoff'])])
            rollout_info['shaped_reward_hist'] = np.vstack([rollout_info['shaped_reward_hist'], np.array(info["shaped_r_by_agent"])])
            rollout_info['reward_hist'] = np.vstack([rollout_info['reward_hist'], np.array(reward)])

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
            # # Update model ----------------
            # loss = self.model.update()
            # if loss is not None: losses.append(loss)
            if done:  break
            # self.env.state = next_state
        # mean_loss = np.mean(losses)
        return cum_reward, cum_shaped_reward,rollout_info
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

def heatmap_test(config):
    from risky_overcooked_rl.utils.rl_logger import TrajectoryHeatmap


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
    config["HORIZON"] = len(averse_joint_traj)-1
    validator = Validation(SelfPlay_QRE_OSA,config)
    cum_reward, cum_shaped_reward, rollout_info = validator.trajectory_rollout(0, averse_joint_traj)
    state_history = rollout_info['state_history']
    HM = TrajectoryHeatmap(validator.env)
    HM.que_trajectory(state_history)
    HM.preview()

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


def handoff_test(config):
    config['LAYOUT'] = "sanity_check"
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
        [W, X, 0.01],
        [I, X, 0.01],
        [E, X, 0.01],
        [I, W, 0.01], # p1 drops onion
        [W, I, 0.01], # p2 picks up onion [4]
        [I, E, 0.01],
        [E, I, 0.01], # p2 pots onion [6]==> both recieve shaped reward
        [I, W, 0.01],
        [W, I, 0.01],
        [I, E, 0.01],
        [E, I, 0.01],
        [I, W, 0.01],
        [S, I, 0.01],
        [I, E, 0.01],
        [E, I, 0.01], # ALL THREE ONIONS POTTED
        [I, W, 0.01],
        [X, I, 0.01],
        [X, E, 0.01],
    ]
    wait_for_cook = [[X, X, 0.01] for i in range(21)]
    averse_joint_traj += wait_for_cook
    # averse_joint_traj +=[
    #     [X, I, 0.01],
    #     [X, S, 0.01],
    #     [X, I, 0.01], # SOUP DELIVERED
    #     [X, X, 0.01],
    # ]
    averse_joint_traj += [
        [X, I, 0.01], # SOUP PICKUP
        [X, W, 0.01],
        [E, I, 0.01],  # SOUP DROPPED
        [I, X, 0.01],
        [I, X, 0.01],
        [X, I, 0.01],
        [X, I, 0.01],
        [I, X, 0.01],
        [I, X, 0.01],
        [X, I, 0.01],
        [X, I, 0.01],
        [X, X, 0.01],

    ]


    config["HORIZON"] = len(averse_joint_traj)-1

    validator = Validation(SelfPlay_QRE_OSA, config)

    for it in range(1):
        cum_reward, cum_shaped_reward,rollout_info = validator.trajectory_rollout(it, averse_joint_traj)
        print(f'Shaped Reward: | Onion handoff')


        for t in range(len(averse_joint_traj)-1):
            disp_str = f'{averse_joint_traj[t][0:2]} \t|\t'
            disp_str += f'{rollout_info["shaped_reward_hist"][t,:]+rollout_info["reward_hist"][t]}'
            # disp_str += f'| {rollout_info["onion_handoff"][t]}'
            # helds = [player.held_object for player in rollout_info["state_history"][t].players]
            # disp_str += f'{helds}'

            for obj in ['onion','dish','soup']:
                if np.any(rollout_info[f"{obj}_handoff"][t]):
                    disp_str += f'\t {obj} handoff'
                if np.any(rollout_info[f"{obj}_pickup"][t]):
                    disp_str += f'\t {obj} pickup'
                if np.any(rollout_info[f"{obj}_drop"][t]):
                    disp_str += f'\t {obj} drop'


            # if np.any(rollout_info["onion_handoff"][t]):
            #     disp_str += '\t onion handoff'
            # if np.any(rollout_info["onion_pickup"][t]):
            #     disp_str += '\t onion pickup'
            # if np.any(rollout_info["onion_drop"][t]):
            #     disp_str += '\t onion drop'
            #
            # if np.any(rollout_info["dish_handoff"][t]):
            #     disp_str += '\t dish handoff'
            # if np.any(rollout_info["dish_pickup"][t]):
            #     disp_str += '\t dish pickup'
            # if np.any(rollout_info["dish_drop"][t]):
            #     disp_str += '\t dish drop'
            #
            # if np.any(rollout_info["soup_pickup"][t]):
            #     disp_str += '\t soup pickup'
            # if np.any(rollout_info["soup_handoff"][t]):
            #     disp_str += '\t soup handoff'
            print(disp_str)

        validator.traj_visualizer.que_trajectory(rollout_info['state_history'])
        validator.traj_visualizer.preview_qued_trajectory()
        # validator.OSA_test(0, averse_joint_traj)

def cirriculum_test(config):
    config['LAYOUT'] = "forced_coordination_sanity_check"
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
        [W, X, 0.01],
        [I, X, 0.01],
        [E, X, 0.01],
        [I, W, 0.01], # p1 drops onion
        [S, I, 0.01], # p2 picks up onion [4]
        [I, E, 0.01],
        [E, I, 0.01], # p2 pots onion [6]==> both recieve shaped reward
        [I, X, 0.01], # p1 drops dish
        [X, X, 0.01],
        [X, W, 0.01],
        [X, X, 0.01],
        [X, I, 0.01], # p2 picks up dish
        [X, E, 0.01],
        [X, I, 0.01], # p1 plates soup
        [X, S, 0.01],
        [X, I, 0.01],# p2 delivers soup
        [X, X, 0.01],
        [X, X, 0.01],
    ]
    # averse_joint_traj = [
    #     [W, X, 0.01],
    #     [I, X, 0.01],
    #     [E, X, 0.01],
    #     [I, W, 0.01],  # p1 drops onion
    #     [W, I, 0.01],  # p2 picks up onion [4]
    #     [I, E, 0.01],
    #     [E, I, 0.01],  # p2 pots onion [6]==> both recieve shaped reward
    #     [I, W, 0.01],
    #     [W, I, 0.01],
    #     [I, E, 0.01],
    #     [E, I, 0.01],
    #     [I, W, 0.01],
    #     [S, I, 0.01],
    #     [I, E, 0.01],
    #     [E, I, 0.01],  # ALL THREE ONIONS POTTED
    #     [I, W, 0.01],
    #     [X, I, 0.01],
    #     [X, E, 0.01],
    # ]
    # wait_for_cook = [[X, X, 0.01] for i in range(21)]
    # averse_joint_traj += wait_for_cook
    # averse_joint_traj += [
    #     [X, I, 0.01],
    #     [X, S, 0.01],
    #     [X, I, 0.01],  # SOUP DELIVERED
    #     [X, X, 0.01],
    # ]


    config["HORIZON"] = len(averse_joint_traj)-1

    validator = Validation(SelfPlay_QRE_OSA, config)
    cirriculum = Curriculum(validator.env)

    cirriculum.set_cirriculum(n_onions=1,cook_time=2)
    for it in range(1):
        cum_reward, cum_shaped_reward,rollout_info = validator.trajectory_rollout(it, averse_joint_traj)


        print(f'Shaped Reward: | Onion handoff')
        for t in range(len(averse_joint_traj)-1):
            disp_str = f'{averse_joint_traj[t][0:2]} \t|\t'
            disp_str += f'{rollout_info["shaped_reward_hist"][t,:]+rollout_info["reward_hist"][t]}'
            # disp_str += f'| {rollout_info["onion_handoff"][t]}'
            # helds = [player.held_object for player in rollout_info["state_history"][t].players]
            # disp_str += f'{helds}'

            # if rollout_info["shaped_reward_hist"][t-2,1]==3:
            #     disp_str += '\t onion potted'


            if np.any(rollout_info["onion_handoff"][t]):
                disp_str += '\t onion handoff'
            if np.any(rollout_info["onion_pickup"][t]):
                disp_str += '\t onion pickup'
            if np.any(rollout_info["onion_drop"][t]):
                disp_str += '\t onion drop'

            if np.any(rollout_info["dish_handoff"][t]):
                disp_str += '\t dish handoff'
            if np.any(rollout_info["dish_pickup"][t]):
                disp_str += '\t dish pickup'
            if np.any(rollout_info["dish_drop"][t]):
                disp_str += '\t dish drop'

            if np.any(rollout_info["soup_pickup"][t]):
                disp_str += '\t soup pickup'
            if np.any(rollout_info["soup_delivery"][t]):
                disp_str += '\t soup delivery'
            print(disp_str)


        # validator.OSA_test(0, averse_joint_traj)


def main():
    config = {
        'ALGORITHM': 'Boltzmann_QRE-DDQN-OSA',
        'Date': datetime.now().strftime("%m_%d_%Y-%H_%M"),

        # Env Params ----------------
        'LAYOUT': "risky_coordination_ring",
        # 'LAYOUT': "risky_multipath",
        # 'LAYOUT': "forced_coordination",
        'HORIZON': 200,
        'ITERATIONS': 30_000,
        'AGENT': None,  # name of agent object (computed dynamically)
        "obs_shape": None,  # computed dynamically based on layout
        "p_slip": 0.1,

        # Learning Params ----------------
        "rand_start_sched": [0.0, 0.0, 10_000],  # percentage of ITERATIONS with random start states
        'epsilon_sched': [1.0, 0.15, 5000],  # epsilon-greedy range (start,end)
        'rshape_sched': [1, 0, 10_000],  # rationality level range (start,end)
        'rationality_sched': [5, 5, 10_000],
        'lr_sched': [1e-2, 1e-4, 3_000],
        # 'test_rationality': 5,          # rationality level for testing
        'gamma': 0.97,  # discount factor
        'tau': 0.01,  # soft update weight of target network
        "num_hidden_layers": 5,  # MLP params
        "size_hidden_layers": 256,  # 32,      # MLP params
        "device": device,
        "minibatch_size": 256,  # size of mini-batches
        "replay_memory_size": 30_000,  # size of replay memory
        'clip_grad': 100,
        'monte_carlo': False
    }
    # slip_test(config)

    # handoff_test(config)
    # cirriculum_test(config)
    heatmap_test(config)




if __name__ == "__main__":
    main()