import os
import sys
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from risky_overcooked_rl.utils.model_manager import get_default_config, parse_args #get_argparser
from risky_overcooked_rl.utils.trainer import Trainer
from risky_overcooked_rl.utils.deep_models import SelfPlay_QRE_OSA_CPT
from risky_overcooked_rl.utils.belief_update import BayesianBeliefUpdate
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from itertools import count
from src.risky_overcooked_py.mdp.actions import Action
from risky_overcooked_rl.utils.evaluation_tools import Discriminability
from itertools import product
from copy import deepcopy

def run_discriminant(layout,fnames,p_slip):
    # LOAD ENV
    config = get_default_config()
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

    score = Discriminability(layout, policies).run()
    print(f'Score:{score}')

def main():
    layout = 'risky_multipath'
    p_slip = 0.1
    fnames = [
        'risky_multipath_pslip01__b00_lam225_etap088_etan10_deltap061_deltan069__10_21_2024-21_32',
        'risky_multipath_pslip01__b00_lam05_etap10_etan088_deltap061_deltan069__10_21_2024-21_32',
        'risky_multipath_pslip01__rational__10_11_2024-12_20'
    ]
    run_discriminant(layout,fnames,p_slip)

    fnames = [
        'risky_multipath_pslip01__b00_lam225_etap088_etan10_deltap061_deltan069__10_21_2024-21_32',
        'risky_multipath_pslip01__b00_lam05_etap10_etan088_deltap061_deltan069__10_21_2024-21_32',
        'risky_multipath_pslip01__rational__10_11_2024-12_20'
    ]
    run_discriminant(layout, fnames, p_slip)
if __name__ == "__main__":
    main()