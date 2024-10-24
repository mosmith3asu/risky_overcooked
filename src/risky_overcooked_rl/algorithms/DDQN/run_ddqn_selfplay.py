import sys
import os

print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))
from risky_overcooked_rl.algorithms.DDQN.curriculum import CirriculumTrainer
from risky_overcooked_rl.algorithms.DDQN.agents import SelfPlay_QRE_OSA_CPT
from risky_overcooked_rl.utils.model_manager import parse_args,get_default_config#, ModelManager


# noinspection PyDictCreation
def main():
    config = get_default_config()

    # config['epsilon_sched']= [1.0, 0.15, 4000]
    config = parse_args(config)
    config["ALGORITHM"] = 'Curriculum-' + config['ALGORITHM']

    # Run Curriculum learning
    for key, val in config['cpt_params'].items():
        if isinstance(val,int): config['cpt_params'][key] = float(val)

    CirriculumTrainer(SelfPlay_QRE_OSA_CPT, config).run()


if __name__ == "__main__":
    main()