import os
import sys
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from study_1 import get_default_config,get_absolute_save_dir,set_config_value
from risky_overcooked_rl.algorithms.DDQN.utils.agents import SelfPlay_QRE_OSA_CPT
from risky_overcooked_rl.utils.belief_update import BayesianBeliefUpdate
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from itertools import count


# LAYOUT = 'risky_coordination_ring'; P_SLIP = 0.4
LAYOUT = 'risky_multipath'; P_SLIP = 0.15
RATIONALITY = 10
BELIEF_CAPACITY = 200
N_TESTS = 10
SEED = None
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

def get_belief_history(partner_type, N_tests=10, rationality=10):
    print('\n\nPartner Type:', partner_type,'\n\n')
    config = get_default_config()
    config['env']['LAYOUT'] = LAYOUT
    config['env']['p_slip'] = P_SLIP
    config['agents']['model']["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    rational_fname =  f'{LAYOUT}_pslip0{int(P_SLIP * 10)}__rational'
    averse_fname = f'{LAYOUT}_pslip0{int(P_SLIP * 10)}__b00_lam225_etap088_etan10_deltap061_deltan069'
    seeking_fname = f'{LAYOUT}_pslip0{int(P_SLIP * 10)}__b00_lam044_etap10_etan088_deltap061_deltan069'

    # LOAD ENV
    config["ALGORITHM"] = 'Evaluate-' + config['ALGORITHM']
    mdp = OvercookedGridworld.from_layout_name(config['env']['LAYOUT'])
    mdp.p_slip = config['env']['p_slip']
    obs_shape = mdp.get_lossless_encoding_vector_shape()
    n_actions = 36
    env = OvercookedEnv.from_mdp(mdp, horizon=config['env']['HORIZON'],
                                 time_cost=config['env']['time_cost'])

    # Load Agents
    seeking_agent = SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config['agents'], seeking_fname, save_dir=get_absolute_save_dir())
    averse_agent = SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config['agents'], averse_fname, save_dir=get_absolute_save_dir())
    rational_agent = SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config['agents'], rational_fname, save_dir=get_absolute_save_dir())
    true_fname =  averse_fname if partner_type == 'Averse' else seeking_fname
    true_agent = SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config['agents'],true_fname, save_dir=get_absolute_save_dir())


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

    fsz = 11
    colors = [(127/255, 0, 219/255),(255 / 255, 154 / 255, 0),'black']
    agents = ['Seeking', 'Averse']
    agent_canidates = ['Seeking', 'Averse','Rational']
    variances = {'Seeking':None, 'Averse':None}
    means = {'Seeking': None, 'Averse': None}

    plt.close('all')
    fig, axs = plt.subplots(2, 1,figsize=(3,3.5))
    for i, partner_type in enumerate(agents):
        hist = np.array(get_belief_history(partner_type, N_tests=N_TESTS,rationality=RATIONALITY))

        # for j in range(mean.shape[1]):
        for j, canidate in enumerate(agent_canidates):
            mean = np.mean(hist[:, :, j], axis=0)
            std = np.std(hist[:, :, j], axis=0)
            x = np.arange(mean.shape[0])
            c = colors[j]

            can_label = {
                "Seeking": "$b(\pi_S\mid\mathcal{O})$", #"$\pi_S$",
                "Rational":"$b(\pi_0\mid\mathcal{O})$", #"$\pi_0$",
                "Averse":  "$b(\pi_A\mid\mathcal{O})$" #"$\pi_A$"
            }
            axs[i].plot(x, mean, color=c, label=can_label[canidate])#canidate)
            axs[i].fill_between(x, mean - std, mean + std,  color=c,alpha=0.2)
            # axs[i].plot(x, mean[:, j], color=c, label=agent_canidates[j])
            # axs[i].fill_between(x, mean[:, j] - np.sqrt(variance[:, j]), mean[:, j] + np.sqrt(variance[:, j]),
            #                     color=c,
            #                     alpha=0.2)
        if i==0:
            axs[i].set_title(f'{"RCR" if LAYOUT == "risky_coordination_ring" else "RMP"}')
            axs[i].set_xticks([])
            # for minor ticks
            axs[i].set_xticks([], minor=True)
        else:
            axs[i].set_xlabel("Timestep ($t$)",fontsize=fsz)

        # axs[i].set_xlabel("Episode")
        if  LAYOUT == "risky_coordination_ring":
            pi_label = '$\pi_A$' if partner_type == 'Averse' else '$\pi_S$'
            axs[i].set_ylabel(f"{partner_type} ({pi_label})",fontsize=fsz)
            # axs[i].set_ylabel(f"{partner_type} Partner") #+ "\n($b(\\boldsymbol{\pi}\mid\\mathcal{O}$)")
        # axs[i].set_title(f"{partner_type} Partner")
        else:
            axs[i].set_yticks([])
            # for minor ticks
            axs[i].set_yticks([], minor=True)
            if i == len(agents) - 1:
                axs[i].legend(loc='center right')
        axs[i].set_ylim([-0.05,1.05])

    fig.tight_layout()
    plt.subplots_adjust(left=0.19)
    plt.savefig(f'Fig_Belief_{"RCR" if LAYOUT == "risky_coordination_ring" else "RMP"}.svg', bbox_inches='tight')
    plt.show()
