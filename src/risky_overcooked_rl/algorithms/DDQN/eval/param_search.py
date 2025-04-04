from multiprocessing import Pool
import sys
import os
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))
from risky_overcooked_rl.algorithms.DDQN.utils.curriculum import CirriculumTrainer
from risky_overcooked_rl.algorithms.DDQN.utils.agents import SelfPlay_QRE_OSA_CPT
import risky_overcooked_rl.algorithms.DDQN as Algorithm
from itertools import product
from copy import deepcopy

def train_worker(config):
    CirriculumTrainer(SelfPlay_QRE_OSA_CPT, config).run()
    # print(config)

def get_config_search_list(search_dict,layout='risky_tree',p_slip=0.2):
    search_list = []
    def_config = Algorithm.get_default_config()
    # def_config["ITERATIONS"] = 15_000
    def_config["LAYOUT"] = layout
    def_config["p_slip"] = p_slip
    # def_config["ITERATIONS"] = 20
    # def_config["epsilon_sched"] = [1,1,2]
    # def_config["rshape_sched"] = [1,1,2]
    def_config["enable_report"] = False

    def_config["auto_save"] = True
    def_config['save_dir'] = '\\risky_overcooked_rl\\algorithms\\DDQN\\eval\\param_search\\'

    def_config["wait_for_close"] = False
    def_config['enable_report'] =False

    key_idxs = list(search_dict.keys())
    search_dict_vals = list(product(*list(search_dict.values())))

    for k,params in enumerate(search_dict_vals):
        config = deepcopy(def_config)
        config['fname_ext'] = f'SEARCH_{k}_'
        for i, key in enumerate(key_idxs):
            if key=='lr':
                config['lr_sched'] = [ params[i], params[i],1_000]
            else:
                config[key] = params[i]
        search_list.append(config)
    return search_list
def main(reversed = True):
    # PARAMETER SEARCH
    N_workers = 10
    # search_space = {
    #     'lr': [0.001],  # learning rate
    #     'gamma': [0.95],  # discount factor
    #     'tau': [0.005],  # soft update weight of target network
    #     "num_hidden_layers": [5],  # MLP params
    #     "size_hidden_layers": [256],  # MLP params
    #     "minibatch_size": [256],  # size of mini-batches
    #     "replay_memory_size": [20_000],  # size of replay memory
    # }
    search_space = {
        'lr': [0.0001,0.0005],  # learning rate
        'gamma': [0.95,0.98],  # discount factor
        'tau': [0.005, 0.001],  # soft update weight of target network
        "num_hidden_layers": [7,9],  # MLP params
        "size_hidden_layers": [128,256],  # MLP params
        "minibatch_size": [256],  # size of mini-batches
        "replay_memory_size": [100_000,200_000],  # size of replay memory
    }

    config_list = get_config_search_list(search_space)
    if reversed:
        config_list = config_list[::-1]
    print('\n############################################')
    print(f'Running {len(config_list)} configurations')
    print('\n############################################')

    # BATCH PROCESSING
    pool = Pool(processes=N_workers)

    res = pool.map(train_worker, config_list)



if __name__ == "__main__":
    main()