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
    config['loads'] = 'rational'
    config['time_cost'] = 0.0
    config['p_slip'] = 0.1
    config = parse_args(config)
    config["ALGORITHM"] = 'Retrain-' + config['ALGORITHM']
    trainer = Trainer(SelfPlay_QRE_OSA_CPT, config)


    N_tests = 10
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
