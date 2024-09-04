import numpy as np
from risky_overcooked_rl.utils.cirriculum import CirriculumTrainer
from risky_overcooked_rl.utils.cirriculum import Curriculum
from risky_overcooked_rl.utils.deep_models import device,SelfPlay_QRE_OSA,SelfPlay_QRE_OSA_CPT
from risky_overcooked_rl.utils.rl_logger import RLLogger,TrajectoryVisualizer
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,OvercookedState,SoupState, ObjectState
from risky_overcooked_py.mdp.actions import Action
from itertools import product,count
import torch
import torch.optim as optim
import math
import random
from datetime import datetime
debug = False
from collections import deque

# noinspection PyDictCreation
def main():
    config = {
        'ALGORITHM': 'Boltzmann_QRE-DDQN-OSA',
        'Date': datetime.now().strftime("%m_%d_%Y-%H_%M"),

        # Env Params ----------------
        # 'LAYOUT': "risky_multipath",
        'LAYOUT': "risky_coordination_ring",
        'HORIZON': 200,
        'ITERATIONS': 30_000,
        'AGENT': None,                  # name of agent object (computed dynamically)
        "obs_shape": None,                  # computed dynamically based on layout
        "p_slip": 0.1,

        # Learning Params ----------------
        "rand_start_sched": [1.0, 0.25, 10_000],  # percentage of ITERATIONS with random start states
        'epsilon_sched': [1.0,0.15,2000],         # epsilon-greedy range (start,end)
        'rshape_sched': [1,0,5_000],     # rationality level range (start,end)
        'rationality_sched': [5,5,10_000],
        'lr_sched': [1e-4,1e-4,1_000],
        # 'test_rationality': 5,          # rationality level for testing
        'gamma': 0.95,                      # discount factor
        'tau': 0.005,                       # soft update weight of target network
        "num_hidden_layers": 5,             # MLP params
        "size_hidden_layers": 256,#32,      # MLP params
        "device": device,
        "minibatch_size":256,          # size of mini-batches
        "replay_memory_size": 20_000,   # size of replay memory
        'clip_grad': 100,
    }

    # ----------------------------------------
    # config['note'] = 'Standard OSA'
    # config['note'] = 'Standard OSA Curriculum + rationality'
    # config['note'] = 'OSA Curriculum + scaling_QRE + rationality + layer'
    config['note'] = 'OSA Curriculum + scaling_QRE + rationality '
    # config['note'] = 'OSA Curriculum + rationality + slip'
    # config['p_slip'] = 0.5
    # config['num_hidden_layers'] = 7
    config['rationality_sched']= [10,10,10_000]
    CirriculumTrainer(SelfPlay_QRE_OSA, config).run()

    # ----------------------------------------
    # config['cpt_params']= {'b': 0.0, 'lam': 1.0,
    #                'eta_p': 1., 'eta_n': 1.,
    #                'delta_p': 1., 'delta_n': 1.}
    # config['lr_sched'] = [1e-2,1e-5,1_000]
    # config['tau'] = 0.005
    # config['note'] = 'Rational CPT + lr/tau'
    # CirriculumTrainer(SelfPlay_QRE_OSA_CPT, config).run()

    # ----------------------------------------

    # config['cpt_params']= {'b': 0.0, 'lam': 1.0,
    #                'eta_p': 0.95, 'eta_n': 0.95,
    #                'delta_p': 0.95, 'delta_n': 0.95}
    # config['note'] = 'Minimal CPT'
    # CirriculumTrainer(SelfPlay_QRE_OSA_CPT,config).run()

    # ----------------------------------------
    # config['note'] = 'Normal CPT'
    # config['cpt_params']= {'b': 'mean', 'lam': 2.25,
    #                'eta_p': 0.88, 'eta_n': 0.88,
    #                'delta_p': 0.61, 'delta_n': 0.69}
    # CirriculumTrainer(SelfPlay_QRE_OSA_CPT,config).run()




if __name__ == "__main__":
    main()