import sys
import os

print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))
# from risky_overcooked_rl.algorithms.DDQN.curriculum import CirriculumTrainer
from risky_overcooked_rl.algorithms.DDQN.trainer import ResponseTrainer
from risky_overcooked_rl.algorithms.DDQN.agents import SelfPlay_QRE_OSA_CPT,ResponseAgent
from risky_overcooked_rl.utils.model_manager import parse_args,get_default_config#, ModelManager



# noinspection PyDictCreation
def main():
    config = get_default_config()

    config['p_slip']= 0.4
    config['epsilon_sched'][-1] *= 2
    config['rshape_sched'][-1] *= 2
    config = parse_args(config)
    config["ALGORITHM"] = 'Response-' + config['ALGORITHM']

    # Run Curriculum learning
    for key, val in config['cpt_params'].items():
        if isinstance(val,int): config['cpt_params'][key] = float(val)

    ResponseTrainer(SelfPlay_QRE_OSA_CPT, config).run()


if __name__ == "__main__":
    main()