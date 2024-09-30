import os
import sys
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from src.risky_overcooked_rl.utils.model_manager import get_default_config, parse_args #get_argparser
from src.risky_overcooked_rl.utils.trainer import Trainer
from src.risky_overcooked_rl.utils.deep_models import SelfPlay_QRE_OSA_CPT


if __name__ == "__main__":
    config = get_default_config()
    config = parse_args(config)
    trainer = Trainer(SelfPlay_QRE_OSA_CPT, config)
    trainer.N_tests = 2
    trainer.run()
