import sys
import os

print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))
from risky_overcooked_rl.algorithms.DDQN.curriculum import CirriculumTrainer
from risky_overcooked_rl.algorithms.DDQN.agents import SelfPlay_QRE_OSA_CPT
from risky_overcooked_rl.utils.model_manager import parse_args#, ModelManager
import risky_overcooked_rl.algorithms.DDQN as Algorithm

# noinspection PyDictCreation
def main():

    # config_1 = get_default_config()
    # config_2 = get_default_config(path = '\\risky_overcooked_rl\\algorithms\\DDQN\\_config.yaml')
    #
    # for key, val in config_1.items():
    #     if config_2[key] != val:
    #         print(key, val, config_2[key])
    config = Algorithm.get_default_config()
    config['p_slip'] = 0.4

    # config['epsilon_sched']= [1.0, 0.15, 4000]
    config = parse_args(config)
    config["ALGORITHM"] = 'Curriculum-' + config['ALGORITHM']

    # Run Curriculum learning
    for key, val in config['cpt_params'].items():
        if isinstance(val,int): config['cpt_params'][key] = float(val)

    # config['cpt_params'] = {'b': 0.0, 'lam': 2.25, 'eta_p': 0.88, 'eta_n': 1.0, 'delta_p': 0.61, 'delta_n': 0.69}  # risk-averse

    CirriculumTrainer(SelfPlay_QRE_OSA_CPT, config).run()


if __name__ == "__main__":
    main()