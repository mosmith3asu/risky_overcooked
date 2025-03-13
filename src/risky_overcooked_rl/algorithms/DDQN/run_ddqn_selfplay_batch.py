import sys
import os
import multiprocessing as mp
import warnings

print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))
from risky_overcooked_rl.algorithms.DDQN.utils.curriculum import CirriculumTrainer
from risky_overcooked_rl.algorithms.DDQN.utils.agents import SelfPlay_QRE_OSA_CPT
# from risky_overcooked_rl.utils.model_manager import parse_args,get_default_config#, ModelManager
import risky_overcooked_rl.algorithms.DDQN as Algorithm


from copy import deepcopy
def train_worker(config):
    print(f'Running config: {config["ibatch"]}')
    CirriculumTrainer(SelfPlay_QRE_OSA_CPT, config).run()
    # print(config)

def get_config_search_list():
    warnings.warn(f'Only running odd batches?')
    config_lst = []
    def_config = Algorithm.get_default_config()
    def_config["enable_report"] = False
    def_config["auto_save"] = True
    def_config["wait_for_close"] = False
    def_config['enable_report'] =False

    # def_config['ITERATIONS'] = 20
    # def_config['rshape_sched'] = [1, 0, 20]
    # def_config['epsilon_sched'] = [1, 0, 20]


    batching = Algorithm.get_default_batching()

    # Get Meta information
    workers = batching['workers']
    if workers > mp.cpu_count() -1:
        workers = min(workers, mp.cpu_count() - 1)
        warnings.warn(f'Num workers > CPU count... reducing to {workers}')

    istart = batching['istart']

    # Count number of jobs in _batching.yaml
    batch_size = 0
    for _, job_configs in batching['jobs'].items():
        for _, pslip in enumerate(job_configs['p_slips']):
            for _, agent_type in enumerate(job_configs['agents']):
                batch_size += 1

    # Parse All Specified Permutations of _batching.yaml
    for layout, job_configs in batching['jobs'].items():
        for i, pslip in enumerate(job_configs['p_slips']):
            for j, agent_type in enumerate(job_configs['agents']):
                        config = deepcopy(def_config)
                        config['cpt_params'] = batching['CPT_PARAMS'][agent_type]
                        config['p_slip'] = pslip
                        config['LAYOUT'] = layout
                        config['ibatch']= f'{len(config_lst)+1}/{batch_size}'
                        config_lst.append(config)
                        print(f'Loading Config: {len(config_lst)}/{batch_size}')
    # config_lst = config_lst[istart:]
    return workers, config_lst


def main():
    N_workers, config_list = get_config_search_list()


    print(f'Running {len(config_list)} configurations on {N_workers} workers')
    pool = mp.Pool(processes=N_workers)
    res = pool.map(train_worker, config_list)

if __name__ == "__main__":
    main()