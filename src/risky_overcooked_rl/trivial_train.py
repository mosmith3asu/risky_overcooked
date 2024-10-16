import os
import sys
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from risky_overcooked_rl.utils.model_manager import get_default_config, parse_args #get_argparser
from risky_overcooked_rl.utils.trainer import Trainer
from risky_overcooked_rl.utils.deep_models import SelfPlay_QRE_OSA_CPT


if __name__ == "__main__":
    config = get_default_config()
    config['loads'] = 'risky_coordination_ring_pslip025__b00_lam225_etap10_etan10_deltap088_deltan10__10_11_2024-12_46'
    config['cpt_params'] = {'b': 0, 'lam': 2.25,
                  'eta_p':1,'eta_n':0.88,
                  'delta_p':1,'delta_n':1}
    # config['time_cost'] = 0.0
    config['p_slip'] = 0.25

    config['epsilon_sched'] = [1e-5,1e-5, 5_000] # start at lower epsilon
    config['rshape_sched'] = [0.0, 0.0, 5_000]  # start at lower epsilon
    config['lr_sched'] = [1e-25, 1e-25, 10_000] # turn down learning rate at start
    config = parse_args(config)
    config["ALGORITHM"] = 'Trivial-' + config['ALGORITHM']
    trainer = Trainer(SelfPlay_QRE_OSA_CPT, config)
    trainer.N_tests = 1
    trainer.run()
