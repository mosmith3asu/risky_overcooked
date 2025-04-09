
from risky_overcooked_rl.algorithms.DDQN.utils.curriculum import CirriculumTrainer
from risky_overcooked_rl.algorithms.DDQN.utils.agents import SelfPlay_QRE_OSA_CPT
import risky_overcooked_rl.algorithms.DDQN as Algorithm

# noinspection PyDictCreation
def main():
    config = Algorithm.get_default_config()
    config["ALGORITHM"] = 'Curriculum-' + config['ALGORITHM'] # Add Curriculum to Algorithm name
    CirriculumTrainer(SelfPlay_QRE_OSA_CPT, config).run()


if __name__ == "__main__":
    main()