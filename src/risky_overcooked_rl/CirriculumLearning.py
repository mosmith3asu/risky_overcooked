import sys
import os

print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))
from risky_overcooked_rl.utils.cirriculum import CirriculumTrainer
from risky_overcooked_rl.utils.deep_models import device, SelfPlay_QRE_OSA_CPT # SelfPlay_QRE_OSA
from risky_overcooked_rl.utils.model_manager import get_argparser
debug = False

# noinspection PyDictCreation
def main():
    config = {}
    parser = get_argparser()
    args = parser.parse_args()
    config.update(vars(args))

    # Run Curriculum learning
    for key, val in config['cpt_params'].items():
        if isinstance(val,int): config['cpt_params'][key] = float(val)
    CirriculumTrainer(SelfPlay_QRE_OSA_CPT, config).run()


if __name__ == "__main__":
    main()