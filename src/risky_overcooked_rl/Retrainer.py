import os
import sys
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from risky_overcooked_rl.utils.model_manager import get_default_config, parse_args #get_argparser
from risky_overcooked_rl.utils.trainer import Trainer
from risky_overcooked_rl.utils.deep_models import SelfPlay_QRE_OSA_CPT


if __name__ == "__main__":
    config = get_default_config()
    config['epsilon_sched'] = [0.5,0.1, 5_000] # start at lower epsilon
    config['rshape_sched'] = [0.0, 0.0, 5_000]  # start at lower epsilon
    # config['lr_sched'] = [1e-5, 1e-5, 10_000] # turn down learning rate at start
    config = parse_args(config)
    config["ALGORITHM"] = 'Retrain-' + config['ALGORITHM']
    trainer = Trainer(SelfPlay_QRE_OSA_CPT, config)
    trainer.N_tests = 2
    trainer.run()
