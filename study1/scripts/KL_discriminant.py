"""
Code is largely deprecated, but kept for reference.
This was origonally intended to search over p_slips to maximize discriminability of policies.
However, alternative models (not chosen) were removed since study was published.
Turns out to be easier to qualitatively evaluate discriminability after each training
"""


import os
import sys
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

import numpy as np
import matplotlib.pyplot as plt
import torch
from risky_overcooked_rl.algorithms.DDQN.utils.agents import SelfPlay_QRE_OSA_CPT
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from risky_overcooked_rl.utils.evaluation_tools import Discriminability
import risky_overcooked_rl.algorithms.DDQN as Algorithm
from study1 import get_absolute_save_dir


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
    config['env']['LAYOUT'] = layout
    config['env']['p_slip'] = p_slip
    config["ALGORITHM"] = 'Discriminate-' + config['ALGORITHM']
    config['agents']['model']["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    mdp = OvercookedGridworld.from_layout_name(config['env']['LAYOUT'])
    mdp.p_slip = config['env']['p_slip']
    obs_shape = mdp.get_lossless_encoding_vector_shape()
    n_actions = 36

    policies = [SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config['agents'], fname,save_dir=get_absolute_save_dir())
                for fname in fnames]
    disc = Discriminability(layout, policies,discount=1)
    mutual_score = disc.run()
    print(f'Score:{mutual_score}')
    ind_scores = disc.kl_rel2first
    return mutual_score, ind_scores

def main():
    RA_scores = []
    RS_scores = []
    fig, axs = plt.subplots(1, 2, figsize=(10, 2))
    ylim = [0,0.6]

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
        plt.ioff()
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


    plt.show()

if __name__ == "__main__":
    main()