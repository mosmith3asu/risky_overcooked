import os
import sys
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

import numpy as np
import matplotlib.pyplot as plt
from risky_overcooked_rl.utils.model_manager import get_default_config, parse_args #get_argparser
from risky_overcooked_rl.utils.trainer import Trainer
from risky_overcooked_rl.utils.deep_models import SelfPlay_QRE_OSA_CPT
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,OvercookedState,SoupState, ObjectState
from risky_overcooked_rl.utils.state_utils import StartStateManager
import itertools
import torch

class Discriminability():
    def __init__(self,layout,joint_policies,N_samples = 5000,discount=0.75,debug=False):
        self.N_samples = N_samples
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policies = joint_policies
        self.agents = [1] # which agents to compare
        self.debug = debug
        self.discount = discount

        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=200)
        self.env.reset()
        self.state_manager = StartStateManager(self.mdp)
        self.comparison_idxs = list(itertools.combinations(range(len(self.policies)),2))

        self.kl_rel2first = [[] for _ in range(len(self.policies)-1)]
    def jensen_shannon_divergence(self,P,Q):
        # https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        # Jensen–Shannon divergence (Symmetrized KL divergence)
        M = 0.5 * (P + Q)
        return 0.5*self.KL_divergence(P,M) + 0.5*self.KL_divergence(Q,M)

    def KL_divergence(self,P,Q,epsilon=1e-10):
        # return np.sum(P*np.log(P/Q + 1e-10))
        P += epsilon
        Q += epsilon
        return np.sum(P*np.log(P/Q))

    # def mutual_distance_metric(self,dists,discount=0.5):
    #    """Does not work between 0-1"""
    #     dists = np.array(dists)
    #     d_min = np.min(dists)
    #     return np.sum(dists-d_min)**discount + d_min
    def mutual_distance_metric(self,dists):
        dists = np.array(dists)
        d_min = np.min(dists)
        return np.sum(np.log(self.discount*(dists-d_min)+1)) + d_min

    def run(self):
        distances = np.zeros(self.N_samples)
        for i in range(self.N_samples):
            state = self.get_random_state()
            joint_actions_idxs, action_probs = self.get_actions(state)
            KL_diverences = []

            for k in self.agents:
                for idx1,idx2 in self.comparison_idxs:
                    pA1 = action_probs[idx1][0, k, :].detach().cpu().numpy()
                    pA2 = action_probs[idx2][0, k, :].detach().cpu().numpy()

                    # if self.debug and idx:

                    kl = self.jensen_shannon_divergence(pA1,pA2)
                    KL_diverences.append(kl)

                    if idx1==0:
                        KL = self.KL_divergence(pA1,pA2)
                        self.kl_rel2first[idx2-1].append(KL)
            distances[i]=self.mutual_distance_metric(KL_diverences)

            assert not np.isnan(distances[i]), 'Nan KL divergence detected'
        self.kl_rel2first= [np.mean(kls) for kls in  self.kl_rel2first]
        return np.mean(distances)

    def get_random_state(self):
        return self.state_manager.assign(self.env.state,
                                          random_loc = True,
                                          random_pot = 0.5,# True,
                                         random_held= 0.7,#True,
                                         with_soup=False)

    def get_actions(self,state):
        obs = self.mdp.get_lossless_encoding_vector_astensor(state, device=self.device).unsqueeze(0)
        action_probs = []
        joint_actions_idxs = []
        for pi in self.policies:
            _, iJA,pA = pi.choose_joint_action(obs, epsilon=0)
            joint_actions_idxs.append(iJA)
            action_probs.append(pA)
        return joint_actions_idxs,action_probs




class CoordinationFluency():

    def __init__(self,state_history):
        self.state_history = state_history

        self.subtask_codes = {
            "TBD": -1,
            "Pick up onion": 0,
            "Deliver onion": 1,
            "Pick up dish": 2,
            "Pick up soup": 3,
            "Deliver soup": 4
        }
        self.subtask_history = [[-1], [-1]]  # agent i,j
        for state, next_state in zip(state_history[:-2], state_history[1:]):
            self.append_subtask_history(state,next_state)


    # Reward Measures
    # def get_total_reward(self):
    #     pass


    ###################################
    # SUBTASK PARSER ##################
    def append_subtask_history(self,state,next_state):
        """
        Subtasks:
        -1: TBD: agent does not have object and will be filled with "Pick Up X" once agent does
        0: Pick up onion: Starts with no object, ends when onion is picked up
        1: Deliver onion: Starts with onion, ends when onion is placed in pot/placed on counter/dropped (no held obj)
        2: Pick up dish: Starts with no object, ends when dish is picked up
        3: Pick up soup: Starts with dish/no object, ends when dish is placed on counter/soup picked up/dropped (no held obj/soup held)
        4: Deliver soup: Starts with soup, ends when soup is placed on counter/delivered/dropped (no held obj)
        :return:
        """

        player_obj_trans = self.held_object_transition(state,next_state)

        for i,obj_trans in enumerate(player_obj_trans):

            # Pick up TBD -----------------------------------
            if np.all(obj_trans == [None,None]):
                name = 'TBD'
                self.subtask_history[i].append(self.subtask_codes[name])

            # Pick up onion -----------------------------------
            elif np.all(obj_trans == [None,'onion']):
                name = 'Pick up onion'
                self.subtask_history[i].append(self.subtask_codes[name])
                self.subtask_history[i] = self.replace_tbd(self.subtask_history[i], self.subtask_codes[name])

            # Deliver onion -----------------------------------
            elif (np.all(obj_trans == ['onion',None])
                  or np.all(obj_trans == ['onion','onion'])):
                name = 'Deliver onion'
                self.subtask_history[i].append(self.subtask_codes[name])

            # Pick up dish -----------------------------------
            elif np.all(obj_trans == [None, 'dish']):
                name ='Pick up dish'
                self.subtask_history[i].append(self.subtask_codes[name])
                self.subtask_history[i] = self.replace_tbd(self.subtask_history[i], self.subtask_codes[name])

            # Pick up soup (with plate/ to handoff) -----------------------------------
            elif (np.all(obj_trans == ['dish','soup'])
                  or np.all(obj_trans == ['dish',None])
                  or np.all(obj_trans == ['dish', 'dish'])
            ):
                name = 'Pick up soup'
                self.subtask_history[i].append(self.subtask_codes[name])

            # Pick up soup (from handoff) -----------------------------------
            elif np.all(obj_trans == np.array([None, 'soup'])):
                name = 'Pick up soup'
                self.subtask_history[i].append(self.subtask_codes[name])
                self.subtask_history[i] = self.replace_tbd(self.subtask_history[i], self.subtask_codes[name])

            # Deliver soup -----------------------------------
            elif (np.all(obj_trans == ['soup',None])
                  or np.all(obj_trans == ['soup','soup'])
            ):
                name = 'Deliver soup'
                self.subtask_history[i].append(self.subtask_codes[name])

            # Same subtask as previous -----------------------------------
            # elif obj_trans[0] == obj_trans[1]:
            #     self.subtask_history[i].append(self.subtask_history[i][-1])
            else:
                raise ValueError(f"Unrecognized object transition player {i}: {obj_trans}")

    def replace_tbd(self,history,subtask_code):
        """ Go back and replace all TBDs with subtask_code"""


        for t in range(2,len(history)+1):
            if history[-t] == self.subtask_codes['TBD']:
                history[-t] = subtask_code
            else: break
        return history
    def held_object_transition(self,state,next_state):
        """

        :param state: prev OvercookedState
        :param next_state: next OvercookedState
        :return: object transition [prev_obj,next_obj] or None
        """
        changed = [None,None]
        for i in range(len(state.players)):
            obj = state.players[i].held_object.name if state.players[i].has_object() else None
            next_obj = next_state.players[i].held_object.name if next_state.players[i].has_object() else None
            changed[i] = [obj,next_obj]
        return changed


    ###################################
    # Fluency Measures ################


    def get_subtask_slices(self,subtask_history):
        player_subtask_slices = []
        for i, history in enumerate(subtask_history):
            subtask_slices = []
            tstart = 0
            for t in range(len(history)):
                if history[tstart] != history[t]:
                    subtask_slices.append([tstart,t])
                    tstart = t
            subtask_slices.append([tstart,len(history)])
            player_subtask_slices.append(subtask_slices)

        return player_subtask_slices

    def get_inactivity(self):
        subtask_history = self.subtask_history
        state_history = self.state_history
        player_inactivity = [[],[]]
        player_subtask_slices = self.get_subtask_slices(subtask_history)
        for i, subtask_slices in enumerate(player_subtask_slices):
            for tstart,tend in subtask_slices:
                subtask_state_history = state_history[tstart:tend]
                inactivity = self.find_subtask_inactivity(subtask_state_history,i)
                player_inactivity[i] += inactivity.astype(int).tolist()

        return player_inactivity


    def find_subtask_inactivity(self,subtask_states,player):
        inactive_states = np.zeros(len(subtask_states))
        player_subtasks_states = [state.players[player] for state in subtask_states ]
        for t in range(len(player_subtasks_states)-1):
            next_state = player_subtasks_states[t+1]
            repeat_states = [s == next_state for s in player_subtasks_states[:t]]
            if any(repeat_states):
                earliest_revisited_state_idx = np.where(repeat_states)[0][0]
                inactive_states[earliest_revisited_state_idx:t+1] = 1
        # print(f"Player {player} repeated state at t={t}")
        # print(inactive_states)
        return inactive_states

    def measures(self,iR=0,iH=1):
        inactivity = self.get_inactivity()
        T = len(inactivity[iH])

        # Trim undertermined subtasks at end of time horizon
        # itrim = T
        # if -1 in inactivity[iR]: itrim = np.where(inactivity[iR]==-1)[0][0]
        # if -1 in inactivity[iH]: itrim = np.min(itrim,np.where(inactivity[iH]==-1)[0][0])
        # inactivity[iH] = inactivity[iH][:itrim]
        # inactivity[iR] = inactivity[iR][:itrim]
        # T = itrim


        R_IDLE = np.sum(inactivity[iR]) / T # % robot idle time
        H_IDLE = np.sum(inactivity[iH]) / T  # % human idle time
        joint_inactivity = np.array(inactivity[iR])+np.array(inactivity[iH]) # % time where both are active
        C_ACT = len(np.where(joint_inactivity == 0)[0])/T # % time where both are active
        res = {
            'R_IDLE': R_IDLE,
            'H_IDLE': H_IDLE,
            'C_ACT': C_ACT
        }
        return res

if __name__ == "__main__":
    config = get_default_config()
    # config[
    #     'loads'] = 'risky_coordination_ring_pslip04__rational__10_09_2024-13_44'

    # config[
    #     'loads'] ='risky_coordination_ring_pslip025__b00_lam05_etap10_etan088_deltap061_deltan069__10_21_2024-11_31'
    config[
        'loads'] ='risky_coordination_ring_pslip025__b00_lam225_etap088_etan10_deltap061_deltan069__10_21_2024-11_31'
    # config['cpt_params'] = {'b': 0, 'lam': 2.25,
    #                         'eta_p': 1, 'eta_n': 0.88,
    #                         'delta_p': 1, 'delta_n': 1}
    # config['time_cost'] = 0.0
    config['p_slip'] = 0.25
    config = parse_args(config)
    config["ALGORITHM"] = 'Evaluate-' + config['ALGORITHM']
    trainer = Trainer(SelfPlay_QRE_OSA_CPT, config)

    N_tests = 1
    stats = {
        'test_rewards': [],
        'test_shaped_rewards': [],
        'onion_risked': np.zeros([1, 2]),
        'dish_risked': np.zeros([1, 2]),
        'soup_risked': np.zeros([1, 2]),
        'onion_handoff': np.zeros([1, 2]),
        'dish_handoff': np.zeros([1, 2]),
        'soup_handoff': np.zeros([1, 2]),
    }

    # for i in range(N_tests):
    #     test_reward, test_shaped_reward, state_history, action_history, aprob_history, info = \
    #         trainer.test_rollout(rationality=5, get_info=True)

    # EVAL = EvalTracker()
    _, _, state_history, _, _, _ = trainer.test_rollout(rationality=10, get_info=True)
    EVAL = CoordinationFluency(state_history)
    # for state,next_state in zip(state_history[:-2],state_history[1:]):
    #     EVAL.append_subtask_history(state,next_state)
    # for player_subtasks in zip(EVAL.subtask_history):
    #     print(player_subtasks)

    # subtask_slices = EVAL.get_subtask_slices(EVAL.subtask_history)
    # find where P1 is waiting to pick up soup
    # player = 0
    # pick_up_soup0 = np.where(np.array(EVAL.subtask_history[player])==3)[0][0]
    # pick_up_soup1 = pick_up_soup0
    # for t in range(pick_up_soup0,len(EVAL.subtask_history[player])):
    #     if EVAL.subtask_history[player][t] != 3:
    #         pick_up_soup1 = t
    #         break
    # print('\n\n')
    # print(EVAL.subtask_history[player][pick_up_soup0-1:pick_up_soup1+1])
    # EVAL.find_subtask_inactivity(state_history[pick_up_soup0:pick_up_soup1],player=player)
    # print(EVAL.get_inactivity()[0])
    print(EVAL.measures())
    trainer.traj_visualizer.que_trajectory(state_history)
    trainer.logger.log(test_reward=[0, 1],  train_reward=[0, 1],
                    loss=[0, 1])
    trainer.logger.draw()
    trainer.logger.wait_for_close()
    # print(EVAL.subtask_history)
        # stats['test_rewards'].append(test_reward)
        # stats['test_shaped_rewards'].append(test_shaped_reward)
        # stats['onion_risked'] += info['onion_risked'] / N_tests
        # stats['dish_risked'] += info['dish_risked'] / N_tests
        # stats['soup_risked'] += info['soup_risked'] / N_tests
        # stats['onion_handoff'] += info['onion_handoff'] / N_tests
        # stats['dish_handoff'] += info['dish_handoff'] / N_tests
        # stats['soup_handoff'] += info['soup_handoff'] / N_tests

    # print(stats)
    # fig, axs = plt.subplots(1, 2)
    # handoff_keys = ['onion_handoff', 'dish_handoff', 'soup_handoff']
    # risked_keys = ['onion_risked', 'dish_risked', 'soup_risked']
    # risked_values = [np.mean(stats[k]) for k in risked_keys]
    # handoff_values = [np.mean(stats[k]) for k in handoff_keys]
    # axs[0].bar(risked_keys, risked_values, color='maroon', width=0.4)
    # axs[1].bar(handoff_keys, handoff_values, color='blue', width=0.4)
    # for ax in axs:
    #     ax.set_ylim([0, 10])
    #
    # print('Average test reward:', np.mean(stats['test_rewards']))
    # plt.ioff()
    # plt.show()
