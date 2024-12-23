import os
import sys
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

import numpy as np
import matplotlib.pyplot as plt
from risky_overcooked_rl.utils.model_manager import get_default_config, parse_args #get_argparser
from risky_overcooked_rl.utils.trainer import Trainer
from risky_overcooked_rl.utils.deep_models import SelfPlay_QRE_OSA_CPT


if __name__ == "__main__":
    config = get_default_config()
    config['loads'] = 'risky_coordination_ring_pslip025__b00_lam225_etap10_etan10_deltap088_deltan10__10_11_2024-12_46'
    config['cpt_params'] = {'b': 0, 'lam': 2.25,
                  'eta_p':1,'eta_n':0.88,
                  'delta_p':1,'delta_n':1}
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

    for i in range(N_tests):
        test_reward, test_shaped_reward, state_history, action_history, aprob_history, info =\
            trainer.test_rollout(rationality=5, get_info=True)
        stats['test_rewards'].append(test_reward)
        stats['test_shaped_rewards'].append(test_shaped_reward)
        stats['onion_risked'] += info['onion_risked']/N_tests
        stats['dish_risked'] += info['dish_risked']/N_tests
        stats['soup_risked'] += info['soup_risked']/N_tests
        stats['onion_handoff'] += info['onion_handoff']/N_tests
        stats['dish_handoff'] += info['dish_handoff']/N_tests
        stats['soup_handoff'] += info['soup_handoff']/N_tests

    print(stats)
    fig,axs = plt.subplots(1, 2)
    handoff_keys = ['onion_handoff', 'dish_handoff', 'soup_handoff']
    risked_keys = ['onion_risked','dish_risked','soup_risked']
    risked_values = [np.mean(stats[k]) for k in risked_keys]
    handoff_values = [np.mean(stats[k]) for k in handoff_keys]
    axs[0].bar(risked_keys, risked_values, color='maroon', width=0.4)
    axs[1].bar(handoff_keys, handoff_values, color='blue', width=0.4)
    for ax in axs:
        ax.set_ylim([0,10])

    print('Average test reward:', np.mean(stats['test_rewards']))
    plt.ioff()
    plt.show()
