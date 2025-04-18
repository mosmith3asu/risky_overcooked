import os
import sys
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

import numpy as np
import matplotlib.pyplot as plt
import torch
from risky_overcooked_rl.utils.deep_models import SelfPlay_QRE_OSA_CPT
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from risky_overcooked_rl.utils.evaluation_tools import Discriminability
import risky_overcooked_rl.algorithms.DDQN as Algorithm

def run_discriminant(layout,fnames,p_slip,debug=False):
    """
    Calculates the discriminability of a set of policies
    - uses Jensen–Shannon divergence for discriminability
    - lower value ==> more similar (0=> identical)
    - want to maximize discriminability score
    :param layout: overcooked layout name
    :param fnames: list of policy file names
    :param p_slip: probability os slipping
    :return: discriminability score
    """
    # LOAD ENV
    config = Algorithm.get_default_config()
    config['LAYOUT'] = layout
    config['p_slip'] = p_slip
    config["ALGORITHM"] = 'Discriminate-' + config['ALGORITHM']
    config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
    mdp = OvercookedGridworld.from_layout_name(config['LAYOUT'])
    mdp.p_slip = config['p_slip']
    obs_shape = mdp.get_lossless_encoding_vector_shape()
    n_actions = 36

    policies = [SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config, fname)
                for fname in fnames]
    disc = Discriminability(layout, policies,discount=1)
    mutual_score = disc.run()
    print(f'Score:{mutual_score}')
    ind_scores = disc.kl_rel2first
    return mutual_score, ind_scores

def main():
    # layout = 'risky_multipath'
    # p_slip = 0.1
    # fnames = [
    #     'risky_multipath_pslip01__b00_lam225_etap088_etan10_deltap061_deltan069__10_21_2024-21_32',
    #     'risky_multipath_pslip01__b00_lam05_etap10_etan088_deltap061_deltan069__10_21_2024-21_32',
    #     'risky_multipath_pslip01__rational__10_11_2024-12_20'
    # ]
    # run_discriminant(layout,fnames,p_slip)


    # p_slip = 0.1
    # layout = 'risky_coordination_ring'
    # fnames = [
    #     'risky_coordination_ring_pslip04__b00_lam225_etap088_etan10_deltap061_deltan069__10_22_2024-11_35',
    #     'risky_coordination_ring_pslip04__b00_lam05_etap10_etan088_deltap061_deltan069__10_22_2024-11_36',
    #     'risky_coordination_ring_pslip04__rational__10_09_2024-13_44'
    # ]
    # run_discriminant(layout, fnames, p_slip)

    ###################################
    scores = []
    RA_scores = []
    RS_scores = []

    # p_slip = 0.1
    # layout = 'risky_multipath'
    # policy_batches = [
    #     #  risk-averse + risk-seeking + rational
    #     ['risky_multipath_pslip01__b00_lam225_etap088_etan10_deltap061_deltan069__10_21_2024-21_32',
    #      'risky_multipath_pslip01__b00_lam05_etap10_etan088_deltap061_deltan069__10_21_2024-21_32',
    #      'risky_multipath_pslip01__rational__10_11_2024-12_20'],
    #
    #     # Indentical averse + rational
    #     ['risky_multipath_pslip01__b00_lam225_etap088_etan10_deltap061_deltan069__10_21_2024-21_32',
    #      'risky_multipath_pslip01__b00_lam225_etap088_etan10_deltap061_deltan069__10_21_2024-21_32',
    #      'risky_multipath_pslip01__rational__10_11_2024-12_20'],
    #
    #     # Indentical seeking + rational
    #     ['risky_multipath_pslip01__b00_lam05_etap10_etan088_deltap061_deltan069__10_21_2024-21_32',
    #      'risky_multipath_pslip01__b00_lam05_etap10_etan088_deltap061_deltan069__10_21_2024-21_32',
    #      'risky_multipath_pslip01__rational__10_11_2024-12_20'],
    # ]
    # p_slip = 0.4
    # layout = 'risky_coordination_ring'
    # policy_batches = [
    #     #  risk-averse + risk-seeking + rational
    #     [ 'risky_coordination_ring_pslip04__b00_lam225_etap088_etan10_deltap061_deltan069__10_22_2024-11_35',
    #       'risky_coordination_ring_pslip04__b00_lam05_etap10_etan088_deltap061_deltan069__10_22_2024-11_36',
    #       'risky_coordination_ring_pslip04__rational__10_09_2024-13_44'],
    #
    #     # ['risky_coordination_ring_pslip04__rational__10_09_2024-13_44',
    #     #  'risky_coordination_ring_pslip04__rational__10_09_2024-13_44',
    #     #  'risky_coordination_ring_pslip04__rational__10_09_2024-13_44'],
    #
    #     # Similar risk-averse policies + rational
    #     ['risky_coordination_ring_pslip04__b00_lam225_etap088_etan10_deltap061_deltan069__10_22_2024-11_35',
    #      'risky_coordination_ring_pslip04__b00_lam225_etap088_etan10_deltap10_deltan10__10_09_2024-13_43',
    #      'risky_coordination_ring_pslip04__rational__10_09_2024-13_44'],
    #
    #     # Indentical risk-averse policies + rational
    #     ['risky_coordination_ring_pslip04__b00_lam225_etap088_etan10_deltap061_deltan069__10_22_2024-11_35',
    #      'risky_coordination_ring_pslip04__b00_lam225_etap088_etan10_deltap061_deltan069__10_22_2024-11_35',
    #      'risky_coordination_ring_pslip04__rational__10_09_2024-13_44'],
    #
    #     # Similar risk-seeking policies + rational
    #     ['risky_coordination_ring_pslip04__b00_lam05_etap10_etan088_deltap10_deltan10__10_09_2024-13_44',
    #      'risky_coordination_ring_pslip04__b00_lam05_etap10_etan088_deltap061_deltan069__10_22_2024-11_36',
    #      'risky_coordination_ring_pslip04__rational__10_09_2024-13_44'],
    #
    #     # Indentical risk-seeking policies + rational
    #     ['risky_coordination_ring_pslip04__b00_lam05_etap10_etan088_deltap061_deltan069__10_22_2024-11_36',
    #      'risky_coordination_ring_pslip04__b00_lam05_etap10_etan088_deltap061_deltan069__10_22_2024-11_36',
    #      'risky_coordination_ring_pslip04__rational__10_09_2024-13_44'],
    #
    #     # ['risky_coordination_ring_pslip04__b00_lam225_etap088_etan10_deltap10_deltan10__10_09_2024-13_43',
    #     #  'risky_coordination_ring_pslip04__b00_lam05_etap10_etan088_deltap061_deltan069__10_22_2024-11_36',
    #     #  'risky_coordination_ring_pslip04__rational__10_09_2024-13_44'],
    #
    #
    # ]

    # bar_color = (255 / 255, 154 / 255, 0)
    fig, axs = plt.subplots(1, 2, figsize=(10, 2))
    ylim = [0,0.6]

    layout = 'risky_coordination_ring'
    PSLIPS = [0.2,0.3,0.4, 0.5,0.6, 0.7]

    # layout = 'risky_multipath'
    # PSLIPS = [0.1, 0.15, 0.2, 0.25, 0.3]
    pltnum=0
    DATA = [['risky_coordination_ring', [0.2,0.3,0.4, 0.5,0.6, 0.7]],
            ['risky_multipath', [0.1, 0.15, 0.2, 0.25, 0.3]]]
    for layout, PSLIPS in DATA:
        scores = []
        sel_bar_color = (255 / 255, 90 / 255, 0)
        bar_color = [tuple([50 / 255 for _ in range(3)]) for _ in range(len(PSLIPS))]

        policy_batches = [
            [
                f'{layout}_pslip0{int(p_slip*10)}__rational',
                # f'{layout}_pslip0{int(p_slip*10)}__b00_lam225_etap088_etan10_deltap061_deltan069',
                # f'{layout}_pslip0{int(p_slip*10)}__b00_lam044_etap10_etan088_deltap061_deltan069'
                f'{layout}_pslip0{int(p_slip * 10)}__b00_lam225_etap088_etan10_deltap061_deltan069',
                f'{layout}_pslip0{int(p_slip * 10)}__b00_lam044_etap10_etan088_deltap061_deltan069'
                ]
            for p_slip in PSLIPS
        ]

        for i,fnames in enumerate(policy_batches):
            p_slip = PSLIPS[i]
            mutual_score, ind_scores = run_discriminant(layout, fnames, p_slip)
            scores.append(mutual_score)
            RA_scores.append(ind_scores[0])
            RS_scores.append(ind_scores[1])

        # make a barchart of scores with PSLIPS as labels
        # fig, axs = plt.subplots(2,1)
        # fig, axs = plt.subplots(1, 1,figsize=(5,2))
        plt.ioff()
        # ax = axs[0]
        # ax=axs
        ax = axs[pltnum]
        bar_color[np.argmax(scores)] = sel_bar_color
        ax.bar([f'{p_slip}' for p_slip in PSLIPS], scores,color=bar_color)
        ax.set_xlabel('$p_\\rho$')
        ax.set_ylabel('$\sigma$')
        ax.set_title(f'{"RCR" if layout=="risky_coordination_ring" else "RMP"}')
        ax.set_ylim(ylim)
        pltnum+=1

    # ax.text(np.argmax(scores), max(scores), '*', fontsize=12,ha='center')
    fig.tight_layout()
    plt.savefig(f"results/Fig_Discriminability.svg", bbox_inches='tight')

    # grouped barchart of RA_scores and RS_scores
    # ax = axs[1]
    # width = 0.40
    # x = np.arange(len(PSLIPS))
    # ax.bar(x - width/2, RA_scores, width,label='Averse')
    # ax.bar(x + width/2, RS_scores, width,label='Seeking')
    # ax.legend()
    # # ax.bar([f'{p_slip}' for p_slip in PSLIPS], scores)
    # ax.set_xticks(x, [f'{p_slip}' for p_slip in PSLIPS])
    # ax.set_xlabel('$p_\\rho$')
    # ax.set_ylabel('KL-Divergence')
    # # ax.set_title('Discriminability of policies')
    #
    plt.show()
    # # print(scores)
    # # assert np.argmax(scores)==0,'Discriminant test failed'
if __name__ == "__main__":
    main()