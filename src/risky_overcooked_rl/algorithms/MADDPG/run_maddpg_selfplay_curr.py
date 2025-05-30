# import sys
# import os
# # print('\\'.join(os.getcwd().split('\\')[:-1]))
# sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))
from src.risky_overcooked_rl.algorithms.MADDPG.curriculum import CirriculumTrainer
from src.risky_overcooked_rl.algorithms.MADDPG.utils import parse_config
import torch



def run() -> None:
    cfg = parse_config()
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Trainer(cfg).run()
    CirriculumTrainer(cfg).run()


if __name__ == "__main__":
    run()
