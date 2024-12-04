import sys
import os
import multiprocessing as mp

print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))
from risky_overcooked_rl.algorithms.DDQN.curriculum import CirriculumTrainer
from risky_overcooked_rl.algorithms.DDQN.agents import SelfPlay_QRE_OSA_CPT
from risky_overcooked_rl.utils.model_manager import parse_args,get_default_config#, ModelManager
from copy import deepcopy
def train_worker(config):
    CirriculumTrainer(SelfPlay_QRE_OSA_CPT, config).run()
    # print(config)

def get_config_search_list(LAYOUTS, PSLIPS, CPT_PARAMS):
    config_lst = []
    def_config = get_default_config()
    def_config["enable_report"] = False
    def_config["auto_save"] = True
    def_config["wait_for_close"] = False
    def_config['enable_report'] =False

    for lk, layout in LAYOUTS:
        for i, pslip in enumerate(PSLIPS):
            for j, cpt_params in enumerate(CPT_PARAMS):
                config = deepcopy(def_config)
                config['cpt_params'] = cpt_params
                config['pslip'] = pslip
                config['LAYOUT'] = layout
                config_lst.append(config)
    return config_lst


def main(N_workers = 4):
    N_workers = min(N_workers, mp.cpu_count()-1)

    LAYOUTS = ['risky_coordination_ring']
    PSLIPS = [0.2,0.3,0.4,0.5,0.7,0.9]
    CPT_PARAMS = [
        {'b': 0.0, 'lam': 2.25, 'eta_p': 0.88, 'eta_n': 1.0, 'delta_p': 0.61,'delta_n': 0.69}, # risk-averse
        {'b': 0.0, 'lam': 1.0, 'eta_p': 1.0, 'eta_n': 1.0, 'delta_p': 1.0,'delta_n': 1.0}, # risk-neutral
        {'b': 0.0, 'lam': 0.44, 'eta_p': 1.0, 'eta_n': 0.88, 'delta_p': 0.61,'delta_n': 0.69}, # risk-seeking
    ]
    config_list = get_config_search_list(LAYOUTS, PSLIPS, CPT_PARAMS)
    pool = mp.Pool(processes=N_workers)
    res = pool.map(train_worker, config_list)



    # config = get_default_config()

    # config['epsilon_sched']= [1.0, 0.15, 4000]
    # config = parse_args(config)
    # config["ALGORITHM"] = 'Curriculum-' + config['ALGORITHM']

    # Run Curriculum learning
    # for key, val in config['cpt_params'].items():
    #     if isinstance(val,int): config['cpt_params'][key] = float(val)

    # config_list = get_config_search_list(search_space)
    # CirriculumTrainer(SelfPlay_QRE_OSA_CPT, config).run()


if __name__ == "__main__":
    main()