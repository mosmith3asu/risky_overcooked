import os
import sys
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

import numpy as np
import matplotlib.pyplot as plt
import torch
import random
# from risky_overcooked_rl.utils.model_manager import get_default_config
from risky_overcooked_rl.algorithms.DDQN import get_default_config
from risky_overcooked_rl.utils.deep_models import SelfPlay_QRE_OSA_CPT
from risky_overcooked_rl.utils.belief_update import BayesianBeliefUpdate
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from itertools import count

# LAYOUT = 'risky_coordination_ring'; P_SLIP = 0.3
# LAYOUT = 'risky_coordination_ring'; P_SLIP = 0.25
LAYOUT = 'risky_coordination_ring'; P_SLIP = 0.4
# LAYOUT = 'risky_multipath'; P_SLIP = 0.25
# LAYOUT = 'risky_multipath'; P_SLIP = 0.1
RATIONALITY = 10
BELIEF_CAPACITY = 200
N_TESTS = 10
SEED = None
# SEED = 42
ALPHA = 0.99

def simulate_episode(env, true_agent, belief_updater):
    if SEED is not None:
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
    else:
        torch.seed()
        random.seed()
        np.random.seed()
    device = true_agent.device

    obs_history = []
    action_history = []
    aprob_history = []
    is_most_likely = []
    env.reset()

    for t in count():
        obs = env.mdp.get_lossless_encoding_vector_astensor(env.state, device=device).unsqueeze(0)
        joint_action, joint_action_idx, action_probs = true_agent.choose_joint_action(obs, epsilon=0)

        belief_updater.update_belief(obs, joint_action_idx)

        obs_history.append(obs)
        action_history.append(joint_action_idx)
        aprob_history.append(action_probs)


        next_state, reward, done, info = env.step(joint_action)


        if done:  break
    # print(f'Is Most Likely: {np.mean(is_most_likely)}')
    return obs_history, action_history, aprob_history

def get_belief_history(partner_type, N_tests=10, rationality=10,with_rational=True):
    print('\n\nPartner Type:', partner_type,'\n\n')


    config = get_default_config()
    if LAYOUT == 'risky_coordination_ring' and P_SLIP == 0.3:
        config['LAYOUT'] = 'risky_coordination_ring'
        config['p_slip'] = 0.25
        seeking_fname = 'risky_coordination_ring_pslip03__b00_lam044_etap10_etan088_deltap061_deltan069__12_10_2024-05_10'
        averse_fname = 'risky_coordination_ring_pslip03__b00_lam225_etap088_etan10_deltap061_deltan069__12_10_2024-05_10'
        rational_fname = 'risky_coordination_ring_pslip03__rational__12_10_2024-05_10'
    # # CONFIG 1: #######################
    elif LAYOUT == 'risky_coordination_ring' and P_SLIP == 0.25:
        config['LAYOUT'] = 'risky_coordination_ring'
        config['p_slip'] = 0.25
        averse_fname = 'risky_coordination_ring_pslip025__b00_lam225_etap088_etan10_deltap061_deltan069__10_21_2024-11_31'
        seeking_fname = 'risky_coordination_ring_pslip025__b00_lam05_etap10_etan088_deltap061_deltan069__10_21_2024-11_31'
        rational_fname = 'risky_coordination_ring_pslip025__rational__10_29_2024-10_18'

    # # CONFIG 2: #######################
    elif LAYOUT == 'risky_coordination_ring' and P_SLIP == 0.4:
        config['LAYOUT'] = 'risky_coordination_ring'
        config['p_slip'] = 0.4
        # averse_fname = 'risky_coordination_ring_pslip04__b00_lam225_etap088_etan10_deltap061_deltan069__10_22_2024-11_35'
        # seeking_fname =  'risky_coordination_ring_pslip04__b00_lam05_etap10_etan088_deltap061_deltan069__10_22_2024-11_36'
        # rational_fname = 'risky_coordination_ring_pslip04__rational__10_09_2024-13_44'
        seeking_fname = 'risky_coordination_ring_pslip03__b00_lam044_etap10_etan088_deltap061_deltan069'
        averse_fname = 'risky_coordination_ring_pslip03__b00_lam225_etap088_etan10_deltap061_deltan069'
        rational_fname = 'risky_coordination_ring_pslip03__rational'

    # CONFIG 3: #######################
    elif LAYOUT == 'risky_multipath' and P_SLIP == 0.1:
        config['LAYOUT'] = 'risky_multipath'
        config['p_slip'] = 0.1
        averse_fname = 'risky_multipath_pslip01__b00_lam225_etap088_etan10_deltap061_deltan069__10_21_2024-21_32'
        seeking_fname = 'risky_multipath_pslip01__b00_lam05_etap10_etan088_deltap061_deltan069__10_21_2024-21_32'
        rational_fname = 'risky_multipath_pslip01__rational__10_11_2024-12_20'

    elif LAYOUT == 'risky_multipath' and P_SLIP == 0.25:
        config['LAYOUT'] = 'risky_multipath'
        config['p_slip'] = 0.25
        averse_fname  = 'risky_multipath_pslip025__b00_lam225_etap088_etan10_deltap10_deltan10__10_17_2024-06_12'
        seeking_fname = 'risky_multipath_pslip025__b00_lam05_etap10_etan088_deltap10_deltan10__10_17_2024-06_12'
        rational_fname  = 'risky_multipath_pslip025__rational__10_28_2024-16_26'
    else:
        raise ValueError('Invalid Configuration')



    # LOAD ENV
    config["ALGORITHM"] = 'Evaluate-' + config['ALGORITHM']
    config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
    mdp = OvercookedGridworld.from_layout_name(config['LAYOUT'])
    mdp.p_slip = config['p_slip']
    obs_shape = mdp.get_lossless_encoding_vector_shape()
    n_actions = 36
    env = OvercookedEnv.from_mdp(mdp, horizon=config['HORIZON'], time_cost=config['time_cost'])

    # Load Agents
    seeking_agent = SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config, seeking_fname)
    averse_agent = SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config, averse_fname)
    rational_agent = SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config, rational_fname)
    true_agent = SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config,
                                                averse_fname if partner_type == 'Averse' else seeking_fname)


    seeking_agent.rationality = rationality
    averse_agent.rationality = rationality
    true_agent.rationality = rationality
    rational_agent.rationality = rationality

    seeking_agent.model.eval()
    averse_agent.model.eval()
    true_agent.model.eval()
    rational_agent.model.eval()


    hist = []
    for i in range(N_tests):


        belief_updater = BayesianBeliefUpdate([seeking_agent, averse_agent, rational_agent],
                                                  [seeking_agent, averse_agent, rational_agent],
                                                  names=['Seeking', 'Rational', 'Averse'],
                                                  title=f'Belief | {partner_type} Partner',
                                              capacity=BELIEF_CAPACITY, alpha=ALPHA)

        simulate_episode(env, true_agent, belief_updater)

        hist.append(belief_updater.belief_history)

    return hist

if __name__ == "__main__":
    plt.ioff()



    # colors = [(127/255, 0, 219/255),(255 / 255, 154 / 255, 0),'black']
    colors = [(127/255, 0, 219/255),(255 / 255, 154 / 255, 0),'black']

    # clors = ['red','blue']
    agents = ['Seeking', 'Averse']

    agent_canidates = ['Seeking', 'Averse', 'Rational']
    # agent_canidates = ['Seeking', 'Averse']

    variances = {'Seeking':None, 'Averse':None}
    means = {'Seeking': None, 'Averse': None}

    # for i, partner_type in enumerate(agents):
    #     hist = np.array(get_belief_history(partner_type, N_tests=N_tests))
    #     mean = np.mean(hist, axis=0)
    #     variance = np.var(hist, axis=0)
    #     means[partner_type] = mean
    #     variances[partner_type] = variance

    plt.close('all')
    fig, axs = plt.subplots(1, 2,figsize=(10,5))
    for i, partner_type in enumerate(agents):
        hist = np.array(get_belief_history(partner_type, N_tests=N_TESTS,rationality=RATIONALITY))

        # mean = means[partner_type]
        # variance = variances[partner_type]
        # x = np.arange(mean.shape[0])

        # for j in range(mean.shape[1]):
        for j, canidate in enumerate(agent_canidates):
            mean = np.mean(hist[:, :, j], axis=0)
            std = np.std(hist[:, :, j], axis=0)
            x = np.arange(mean.shape[0])
            c = colors[j]
            axs[i].plot(x, mean, color=c, label=canidate)
            axs[i].fill_between(x, mean - std, mean + std,  color=c,alpha=0.2)
            # axs[i].plot(x, mean[:, j], color=c, label=agent_canidates[j])
            # axs[i].fill_between(x, mean[:, j] - np.sqrt(variance[:, j]), mean[:, j] + np.sqrt(variance[:, j]),
            #                     color=c,
            #                     alpha=0.2)
        axs[i].set_xlabel("Episode")
        axs[i].set_ylabel("Belief")
        axs[i].set_title(f"{partner_type} Partner")
        if i == len(agents) - 1:
            axs[i].legend(loc='center right')
        axs[i].set_ylim([-0.05,1.05])
    fig.suptitle(f'{N_TESTS} Tests [{LAYOUT} | $pslip={P_SLIP}$]', fontsize=16)
    # fig.suptitle(f'{N_tests} Tests [{"risky_multipath"} | $pslip={0.1}$]', fontsize=16)

    plt.show()
    # for i,partner_type in enumerate(agents):
    #
    #
    #     hist = np.array(get_belief_history(partner_type,N_tests=N_tests))
    #
    #     mean = np.mean(hist, axis=0)
    #     variance = np.var(hist, axis=0)
    #     x = np.arange(mean.shape[0])
    #
    #     for j in range(mean.shape[1]):
    #         c = colors[j]
    #         axs[i].plot(x, mean[:,j], color=c, label=agents[j])
    #         axs[i].fill_between(x,mean[:,j] - np.sqrt(variance[:,j]), mean[:,j]  + np.sqrt(variance[:,j]), color=c, alpha=0.2)
    #     axs[i].set_xlabel("Episode")
    #     axs[i].set_ylabel("Belief")
    #     axs[i].set_title(f"Belief | {partner_type} Partner")
    #     axs[i].legend()
    # plt.show()
    # hist_averse = get_belief_history('Averse', N_tests=N_tests)
    # mean = np.mean(hist_seeking, axis=0)
    # variance = np.var(hist_seeking, axis=0)
    # ax.plot(mean_seeking, color='blue', label='Averse')
    # plt.fill_between(mean - np.sqrt(variance), mean + np.sqrt(variance), color='red', alpha=0.2, label='Variance')

    # Plot each line


    # plt.figure(figsize=(10, 6))
    # for line in data:
    #     ax.plot(line, alpha=0.3, color='blue')
    #
    # hist_averse = get_belief_history('Averse', N_tests=10)
    # # hist = np.mean(np.array(hist_seeking))
    # hist = get_belief_history('Averse')
    # hist = np.mean(hist)
