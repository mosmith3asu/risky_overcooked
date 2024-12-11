import os
import sys
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from risky_overcooked_rl.utils.model_manager import get_default_config
from risky_overcooked_rl.utils.deep_models import SelfPlay_QRE_OSA_CPT
from risky_overcooked_rl.utils.belief_update import BayesianBeliefUpdate
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from itertools import count
from src.risky_overcooked_py.mdp.actions import Action

def predictability(pA_k,a_k,discrete=True):
    if discrete: return int(np.argmax(pA_k) == a_k)
    else: return pA_k[a_k]



def KL_divergence(p, q):
    p = np.array(p)/np.sum(p)
    q = np.array(q)/np.sum(q)
    return np.log(np.sum(p * np.log(p / q)))
def simulate_oracle_episode(env, partner_policy):
    iego,ipartner = 0,1
    device = partner_policy.device

    obs_history = []
    action_history = []
    aprob_history = []
    predictabilityH = []
    predictabilityR = []
    actionprobH = []
    actionprobR = []
    kl_divergenceH = []
    kl_divergenceR = []
    cum_reward = 0

    rollout_info = {
        'onion_risked': np.zeros([1, 2]),
        'onion_pickup': np.zeros([1, 2]),
        'onion_drop': np.zeros([1, 2]),
        'dish_risked': np.zeros([1, 2]),
        'dish_pickup': np.zeros([1, 2]),
        'dish_drop': np.zeros([1, 2]),
        'soup_pickup': np.zeros([1, 2]),
        'soup_delivery': np.zeros([1, 2]),

        'soup_risked': np.zeros([1, 2]),
        'onion_slip': np.zeros([1, 2]),
        'dish_slip': np.zeros([1, 2]),
        'soup_slip': np.zeros([1, 2]),
        'onion_handoff': np.zeros([1, 2]),
        'dish_handoff': np.zeros([1, 2]),
        'soup_handoff': np.zeros([1, 2]),

    }

    env.reset()

    for t in count():
        obs = env.mdp.get_lossless_encoding_vector_astensor(env.state, device=device).unsqueeze(0)

        # CHOOSE ACTIONS ---------------------------------------------------------

        # Choose Partner (Human) Action
        joint_action, joint_action_idx, pA = partner_policy.choose_joint_action(obs, epsilon=0)
        partner_iA = joint_action_idx % 6
        ego_iA = joint_action_idx // 6

        # Calc Predictability
        pA = pA.detach().cpu().numpy()
        pa_R = pA[0, iego, ego_iA]
        pa_H = pA[0, ipartner, partner_iA]
        pa_hat_R = predictability(pA[0, iego],ego_iA)
        pa_hat_H = predictability(pA[0, ipartner],partner_iA)
        # pa_hat_R = pA[0, iego, ego_iA]  # prob of ego action given partner inference
        # pa_hat_H = pA[0, ipartner, partner_iA]  # prob of partner action given ego inference
        predictabilityH.append(pa_hat_H)
        predictabilityR.append(pa_hat_R)

        actionprobR.append(pa_R)
        actionprobH.append(pa_H)

        # kl_divergenceR.append(KL_divergence(pA[0, iego], pA[0, iego]))
        # kl_divergenceH.append(KL_divergence(pA[0, ipartner], pA[0, ipartner]))



        # STEP ---------------------------------------------------------
        next_state, reward, done, info = env.step(joint_action)

        # LOG ---------------------------------------------------------
        obs_history.append(obs)
        action_history.append(joint_action_idx)
        # aprob_history.append(action_probs)
        cum_reward += reward
        for key in rollout_info.keys():
            rollout_info[key] += np.array(info['mdp_info']['event_infos'][key])

        if done:  break

    stats = {
        'risks_taken': np.sum(rollout_info['onion_risked']) + np.sum(rollout_info['dish_risked']) + np.sum(rollout_info['soup_risked']),
        'reward': cum_reward,
        'predictabilityR': np.mean(predictabilityR),
        'predictabilityH': np.mean(predictabilityH),
        'kl_divergenceH': KL_divergence(actionprobH, predictabilityH),
        'kl_divergenceR': KL_divergence(actionprobR, predictabilityR),
        # 'kl_divergenceH': np.mean(kl_divergenceH),
        # 'kl_divergenceR': np.mean(kl_divergenceR)#

    }

    return stats
def simulate_episode(env, partner_policy, belief):
    iego,ipartner = 0,1
    device = partner_policy.device

    obs_history = []
    action_history = []
    aprob_history = []
    predictabilityH = []
    predictabilityR = []
    actionprobH = []
    actionprobR = []
    kl_divergenceH = []
    kl_divergenceR = []
    cum_reward = 0

    rollout_info = {
        'onion_risked': np.zeros([1, 2]),
        'onion_pickup': np.zeros([1, 2]),
        'onion_drop': np.zeros([1, 2]),
        'dish_risked': np.zeros([1, 2]),
        'dish_pickup': np.zeros([1, 2]),
        'dish_drop': np.zeros([1, 2]),
        'soup_pickup': np.zeros([1, 2]),
        'soup_delivery': np.zeros([1, 2]),

        'soup_risked': np.zeros([1, 2]),
        'onion_slip': np.zeros([1, 2]),
        'dish_slip': np.zeros([1, 2]),
        'soup_slip': np.zeros([1, 2]),
        'onion_handoff': np.zeros([1, 2]),
        'dish_handoff': np.zeros([1, 2]),
        'soup_handoff': np.zeros([1, 2]),

    }

    env.reset()

    for t in count():
        obs = env.mdp.get_lossless_encoding_vector_astensor(env.state, device=device).unsqueeze(0)

        # CHOOSE ACTIONS ---------------------------------------------------------

        # Choose Partner (Human) Action
        _, partner_iJA, partner_pA = partner_policy.choose_joint_action(obs, epsilon=0)
        partner_iA = partner_iJA % 6

        # Choose Ego Action
        ego_policy = belief.best_response
        _, ego_iJA, ego_pA = ego_policy.choose_joint_action(obs, epsilon=0)
        ego_iA = ego_iJA // 6


        # Calc Predictability
        # ego_pA = ego_pA.detach().cpu().numpy()
        # partner_pA = partner_pA.detach().cpu().numpy()
        #
        #
        # pa_R = ego_pA[0,iego,ego_iA]
        # pa_hat_H = ego_pA[0, ipartner, partner_iA]  # prob of partner action given ego inference
        # # kl_divergenceR.append(KL_divergence(ego_pA[0, ipartner], partner_pA[0, ipartner]))
        #
        # pa_H = partner_pA[0, ipartner,partner_iA]
        # pa_hat_R = partner_pA[0, iego, ego_iA]  # prob of ego action given partner inference
        # # kl_divergenceR.append(KL_divergence(partner_pA[0, iego], ego_pA[0, iego]))

        ego_pA = ego_pA.detach().cpu().numpy()
        partner_pA = partner_pA.detach().cpu().numpy()


        pa_R = ego_pA[0,iego,ego_iA]
        pa_hat_H = predictability(ego_pA[0, ipartner],partner_iA)

        pa_H = partner_pA[0, ipartner,partner_iA]
        pa_hat_R = predictability(partner_pA[0, iego],ego_iA)
        # pa_hat_R = partner_pA[0, iego, ego_iA]  # prob of ego action given partner inference
        # kl_divergenceR.append(KL_divergence(partner_pA[0, iego], ego_pA[0, iego]))

        actionprobR.append(pa_R)
        actionprobH.append(pa_H)



        predictabilityH.append(pa_hat_H)
        predictabilityR.append(pa_hat_R)

        kl_divergenceH.append(KL_divergence([pa_H, 1-pa_H], [pa_hat_H, 1-pa_hat_H]))


        # Calc Joint Action
        action_idxs = (ego_iA, partner_iA)
        joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(action_idxs)
        joint_action = (Action.ALL_ACTIONS[ego_iA], Action.INDEX_TO_ACTION[partner_iA])

        # UPDATE BELIEF ---------------------------------------------------------
        belief.update_belief(obs, joint_action_idx)

        # STEP ---------------------------------------------------------
        next_state, reward, done, info = env.step(joint_action)

        # LOG ---------------------------------------------------------
        obs_history.append(obs)
        action_history.append(joint_action_idx)
        # aprob_history.append(action_probs)
        cum_reward += reward
        for key in rollout_info.keys():
            rollout_info[key] += np.array(info['mdp_info']['event_infos'][key])

        if done:  break

    stats = {
        'risks_taken': np.sum(rollout_info['onion_risked']) + np.sum(rollout_info['dish_risked']) + np.sum(rollout_info['soup_risked']),
        'reward': cum_reward,
        'predictabilityR': np.mean(predictabilityR),
        'predictabilityH': np.mean(predictabilityH),
        'kl_divergenceH': KL_divergence(actionprobH, predictabilityH),
        'kl_divergenceR': KL_divergence(actionprobR, predictabilityR),
        # 'kl_divergenceH': np.mean(kl_divergenceH),
        # 'kl_divergenceR': np.mean(kl_divergenceR)#
    }

    return stats

def run_tests(partner_type, config,inference_type, N_tests=10, rationality=10):
    print('\n\nPartner Type:', partner_type,'\n\n')
    # partner_type = 'Averse'
    # partner_type = 'Seeking'


    # CONFIG 1: #######################


    averse_fname = config['averse_fname']
    seeking_fname = config['seeking_fname']
    rational_fname = config['rational_fname']

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
    true_agent = SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config,averse_fname if partner_type == 'Averse' else seeking_fname)


    seeking_agent.rationality = rationality
    averse_agent.rationality = rationality
    true_agent.rationality = rationality
    seeking_agent.model.eval()
    averse_agent.model.eval()
    true_agent.model.eval()

    if rational_fname is not None:
        rational_agent = SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config, rational_fname)
        rational_agent.rationality = rationality
        rational_agent.model.eval()

    stat_samples = {
        'risks_taken': [],
        'reward': [],
        'predictabilityH': [],
        'predictabilityR': [],
        'kl_divergenceH': [],
        'kl_divergenceR': [],

    }

    for i in range(N_tests):
        torch.seed()
        random.seed()
        np.random.seed()

        if inference_type == 'oracle':
            stats = simulate_oracle_episode(env, true_agent)
        else:
            models = []
            names = []

            if 'seeking' in inference_type:
                models.append(seeking_agent)
                names.append('Seeking')
            if 'averse' in inference_type:
                models.append(averse_agent)
                names.append('Averse')
            if 'rational' in inference_type and rational_fname is not None:
                models.append(rational_agent)
                names.append('Rational')
            belief_updater = BayesianBeliefUpdate(models, models, names=names, title=f'Belief | {partner_type} Partner')
            stats = simulate_episode(env, true_agent, belief_updater)
        for key in stat_samples.keys():
            stat_samples[key].append(stats[key])
        # hist.append(belief_updater.belief_history)

    return stat_samples

if __name__ == "__main__":
    N_tests = 500
    reward_offset = 40

    config = get_default_config()

    # CONFIG 1: #######################
    # config['LAYOUT'] = 'risky_coordination_ring'
    # config['p_slip'] = 0.25
    # config['averse_fname'] = 'risky_coordination_ring_pslip025__b00_lam225_etap088_etan10_deltap061_deltan069__10_21_2024-11_31'
    # config['seeking_fname'] = 'risky_coordination_ring_pslip025__b00_lam05_etap10_etan088_deltap061_deltan069__10_21_2024-11_31'
    # config['rational_fname'] = 'risky_coordination_ring_pslip025__rational__10_29_2024-10_18'

    # # # CONFIG 2: #######################
    # config['LAYOUT'] = 'risky_coordination_ring'
    # config['p_slip'] = 0.4
    # config['averse_fname'] = 'risky_coordination_ring_pslip04__b00_lam225_etap088_etan10_deltap061_deltan069__10_22_2024-11_35'
    # config['seeking_fname'] =  'risky_coordination_ring_pslip04__b00_lam05_etap10_etan088_deltap061_deltan069__10_22_2024-11_36'
    # config['rational_fname'] = 'risky_coordination_ring_pslip04__rational__10_09_2024-13_44'
    # config['rational_fname'] ='risky_coordination_ring_pslip04__rational__10_28_2024-12_51'

    # # # CONFIG 3: #######################
    # config['LAYOUT'] = 'risky_multipath'
    # config['p_slip'] = 0.1
    # config['averse_fname'] = 'risky_multipath_pslip01__b00_lam225_etap088_etan10_deltap061_deltan069__10_21_2024-21_32'
    # config['seeking_fname'] = 'risky_multipath_pslip01__b00_lam05_etap10_etan088_deltap061_deltan069__10_21_2024-21_32'
    # config['rational_fname'] = 'risky_multipath_pslip01__rational__10_11_2024-12_20'

    # # # CONFIG 4: #######################
    config['LAYOUT'] = 'risky_multipath'
    config['p_slip'] = 0.25
    config['averse_fname'] = 'risky_multipath_pslip025__b00_lam225_etap088_etan10_deltap10_deltan10__10_17_2024-06_12'
    config['seeking_fname'] = 'risky_multipath_pslip025__b00_lam05_etap10_etan088_deltap10_deltan10__10_17_2024-06_12'
    config['rational_fname'] = 'risky_multipath_pslip025__rational__10_28_2024-16_26'

    ###########################################################################
    ###########################################################################
    ###########################################################################

    plt.ioff()
    # fig, axs = plt.subplots(1, 4, figsize=(13, 3), constrained_layout=True)
    # fig, axs = plt.subplots(1, 4, figsize=(13, 3))
    fig, axs = plt.subplots(1, 4, figsize=(13, 3))


    # colors = ['red','blue']
    colors = {'Oracle':tuple([50/255 for _ in range(3)]), 'RS-ToM':(255/255,154/255,0), 'Rational':(255/255,90/255,0)}
    agents = ['Seeking', 'Averse']

    variances = {'Seeking':None, 'Averse':None}
    means = {'Seeking': None, 'Averse': None}


    data = {
        'Reward': {'Seeking': {'Oracle':[],'RS-ToM':[],'Rational':[]},
                   'Averse': {'Oracle':[],'RS-ToM':[],'Rational':[]}, },
        'Risks Taken': {'Seeking': {'Oracle':[],'RS-ToM': [], 'Rational': []},
                        'Averse': {'Oracle':[],'RS-ToM': [], 'Rational': []}, },
        'Robot Predictability': {'Seeking': {'Oracle':[],'RS-ToM': [], 'Rational': []},
                            'Averse': {'Oracle':[],'RS-ToM': [], 'Rational': []}, },
        'Human Predictability': {'Seeking': {'Oracle': [], 'RS-ToM': [], 'Rational': []},
                           'Averse': {'Oracle': [], 'RS-ToM': [], 'Rational': []}, },
    }
    ylims = {
        'Reward': [-40 + reward_offset,50 + reward_offset],
        'Risks Taken': [0,50],
        'Robot Predictability': [0,1],
        'Human Predictability': [0, 1],
    }


    for i, partner_type in enumerate(agents):
        stat_samples = run_tests(partner_type, inference_type=['seeking','averse','rational'], config=config, N_tests=N_tests)
        data['Reward'][partner_type]['RS-ToM'] = np.array(stat_samples['reward']) + reward_offset
        data['Risks Taken'][partner_type]['RS-ToM'] = stat_samples['risks_taken']
        data['Robot Predictability'][partner_type]['RS-ToM'] = stat_samples['predictabilityR']
        data['Human Predictability'][partner_type]['RS-ToM'] = stat_samples['predictabilityH']
        # data['Robot Predictability'][partner_type]['RS-ToM'] =  stat_samples['kl_divergenceR']
        # data['Human Predictability'][partner_type]['RS-ToM'] =  stat_samples['kl_divergenceH']
    for i, partner_type in enumerate(agents):
        stat_samples = run_tests(partner_type,inference_type='oracle',config=config, N_tests=N_tests)
        data['Reward'][partner_type]['Oracle'] = np.array(stat_samples['reward']) + reward_offset
        data['Risks Taken'][partner_type]['Oracle'] = stat_samples['risks_taken']
        data['Robot Predictability'][partner_type]['Oracle'] =stat_samples['predictabilityR']
        data['Human Predictability'][partner_type]['Oracle'] =stat_samples['predictabilityH']
        # data['Robot Predictability'][partner_type]['Oracle'] = stat_samples['kl_divergenceR']
        # data['Human Predictability'][partner_type]['Oracle'] = stat_samples['kl_divergenceH']
    for i, partner_type in enumerate(agents):
        if config['rational_fname'] is not None:
            stat_samples = run_tests(partner_type,inference_type=['rational'],config=config, N_tests=N_tests)
            data['Reward'][partner_type]['Rational'] = np.array(stat_samples['reward']) + reward_offset
            data['Risks Taken'][partner_type]['Rational'] = stat_samples['risks_taken']
            data['Robot Predictability'][partner_type]['Rational'] = stat_samples['predictabilityR']
            data['Human Predictability'][partner_type]['Rational'] = stat_samples['predictabilityH']
            # data['Robot Predictability'][partner_type]['Rational'] = stat_samples['kl_divergenceR']
            # data['Human Predictability'][partner_type]['Rational'] = stat_samples['kl_divergenceH']
        else:
            dum_val = 0
            data['Reward'][partner_type]['Rational'] = [dum_val]
            data['Risks Taken'][partner_type]['Rational'] = [dum_val]
            data['Robot Predictability'][partner_type]['Rational'] = [dum_val/10]
            data['Human Predictability'][partner_type]['Rational'] = [dum_val / 10]
    print(data)
    width = 0.35
    features = ['Reward','Risks Taken','Robot Predictability','Human Predictability']
    for i, feature in enumerate(features):
        d = data[feature]
        x = np.arange(len(agents))

        axs[i].bar(x - width / 2, [np.mean(d[a]['Oracle']) for a in agents],
                   width/2, yerr=[np.std(d[a]['Oracle']) for a in agents],
                   label='Oracle', facecolor=colors['Oracle'],capsize=5)

        axs[i].bar(x, [np.mean(d[a]['RS-ToM']) for a in agents],
                   width/2, yerr=[np.std(d[a]['RS-ToM']) for a in agents],
                   label='RS-ToM', facecolor=colors['RS-ToM'],capsize=5)

        axs[i].bar(x + width / 2, [np.mean(d[a]['Rational']) for a in agents],
                   width/2, yerr=[np.std(d[a]['Rational']) for a in agents],
                   label='Rational', facecolor=colors['Rational'],capsize=5)


        axs[i].set_xticks(x)
        axs[i].set_xticklabels([f'Risk-{name}\n Partner' for name in agents])
        axs[i].set_ylabel(f'{feature}' + (' (no time-cost)' if feature == 'Reward' else ''))
        axs[i].set_ylim(ylims[feature])
        if i == len(features)-1:
            # axs[i].legend(bbox_to_anchor=(1.05, 1, 1.06, 0), loc='upper left', borderaxespad=0.)


            fig.tight_layout()
            fig.subplots_adjust(top=0.75)
            left = axs[0].get_position().x0
            right = 0.942  # for 4 plats
            right = 0.693  # for 3 plats

            # right = axs[1].get_position().x1 - 0.046
            bottom = axs[-1].get_position().y1 * 1.05
            top = 1
            bbox = (left, bottom, right, top)
            plt.legend(*axs[-1].get_legend_handles_labels(), loc='lower center', ncols=4, bbox_to_anchor=bbox,
                       bbox_transform=plt.gcf().transFigure, mode='expand', borderaxespad=0.0,fontsize=12)




    fig.suptitle(f'{config["LAYOUT"]}  [pslip={config["p_slip"]} | tests={N_tests}]\n ')


    plt.show()
