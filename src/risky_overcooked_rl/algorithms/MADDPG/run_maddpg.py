# import sys
# import os
# # print('\\'.join(os.getcwd().split('\\')[:-1]))
# sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))



import torch
from src.risky_overcooked_rl.algorithms.MADDPG.trainer import Trainer
from src.risky_overcooked_rl.algorithms.MADDPG.agents import MADDPG
from src.risky_overcooked_rl.algorithms.MADDPG.utils import parse_config
import torch
class Config():
    class agent:
        name = 'maddpg'
        _target_ = MADDPG

        class params:
            obs_dim = None
            action_dim= None  # to be specified later
            action_range= None  # to be specified later
            agent_index= None  # Different by environments
            hidden_dim= 256
            device= 'cuda' if torch.cuda.is_available() else 'cpu'
            discrete_action_space= True
            batch_size= 256
            # lr = 1e-4  # CHANGED!!!!!
            # tau = 0.0005  # CHANGED!!!!!
            lr= 1e-5
            tau=0.001
            gamma= 0.95
            class critic:
                input_dim = None




    data = 'local'
    layout = 'risky_coordination_ring'
    p_slip = 0.0

    discrete_action = True
    discrete_action_space = True
    num_warmup_episodes = 25
    episode_length= 400
    num_episodes = 2000#1250
    num_train_steps = 400*2000#500_000  # 40000

    experiment= 'vanilla'
    seed= 0
    # exploration
    # ou_init_scale =  0.3   # max exploration  # CHANGED!!!!!
    # ou_exponent_decay = 1   # 1=linear decay  # CHANGED!!!!!
    # ou_final_scale = 0.0    # min exploration  # CHANGED!!!!!
    ou_init_scale =  0.75    # max exploration
    ou_exponent_decay = 4   # 1=linear decay
    ou_final_scale = 0.0    # min exploration



    # eval_frequency= 5000
    eval_frequency = 10 # episodes
    num_eval_episodes= 3

    common_reward= False#True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Logging Settings
    log_frequency= 1000
    log_save_tb= False
    save_video= False

    replay_buffer_capacity =1e6


# class Config():
#     class agent:
#         name = 'maddpg'
#         _target_ = MADDPG
#
#         class params:
#             obs_dim = None
#             action_dim= None  # to be specified later
#             action_range= None  # to be specified later
#             agent_index= None  # Different by environments
#             hidden_dim= 256
#             device= 'cuda' if torch.cuda.is_available() else 'cpu'
#             discrete_action_space= True
#             batch_size= 256
#             lr= 0.0001
#             tau=0.0005
#             gamma= 0.95
#             class critic:
#                 input_dim = None
#
#
#
#
#     data = 'local'
#     layout = 'risky_coordination_ring'
#     p_slip = 0.0
#
#     discrete_action = True
#     discrete_action_space = True
#     num_warmup_episodes = 25
#     episode_length= 400
#     num_episodes = 2000#1250
#     num_train_steps = 400*2000#500_000  # 40000
#
#     experiment= 'vanilla'
#     seed= 0
#     # exploration
#     # ou_init_scale =  0.3    # max exploration
#     # ou_exponent_decay = 2   # 1=linear decay
#     # ou_final_scale = 0.0    # min exploration
#     ou_init_scale =  0.75    # max exploration
#     ou_exponent_decay = 4   # 1=linear decay
#     ou_final_scale = 0.0    # min exploration
#
#
#
#     eval_frequency= 5000
#     num_eval_episodes= 3
#
#     common_reward= False#True
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     # Logging Settings
#     log_frequency= 1000
#     log_save_tb= False
#     save_video= False
#
#     replay_buffer_capacity =5e4


def run() -> None:
    cfg = parse_config()
    # cfg = Config()
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(cfg)
    trainer.run()

if __name__ == "__main__":
    run()
