from risky_overcooked_rl.utils.trainer import Trainer
from risky_overcooked_rl.utils.deep_models import device,SelfPlay_QRE_OSA,SelfPlay_QRE_OSA_CPT
from datetime import datetime

def main():
    import yaml
    with open('utils/_default_config.yaml') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    print(data)
def main2():

    config = {
        'ALGORITHM': 'Boltzmann_QRE-DDQN-OSA',
        'Date': datetime.now().strftime("%m_%d_%Y-%H_%M"),

        # Env Params ----------------
        'LAYOUT': "coordination_ring_CLDE", 'HORIZON': 200, 'ITERATIONS': 15_000,
        'AGENT': None,                  # name of agent object (computed dynamically)
        "obs_shape": None,                  # computed dynamically based on layout
        # "shared_rew": False,                # shared reward for both agents
        "p_slip": 0.25,
        # Learning Params ----------------
        "rand_start_sched": [1.0, 0.5, 10_000],  # percentage of ITERATIONS with random start states
        'epsilon_sched': [0.1,0.1,5000],         # epsilon-greedy range (start,end)
        'rshape_sched': [1,0,5_000],     # rationality level range (start,end)
        'rationality_sched': [0.0,5,5000],
        'lr_sched': [1e-2,1e-4,3_000],
        # 'test_rationality': 5,          # rationality level for testing
        'gamma': 0.95,                      # discount factor
        'tau': 0.005,                       # soft update weight of target network
        "num_hidden_layers": 3,             # MLP params
        "size_hidden_layers": 256,#32,      # MLP params
        "device": device,
        "minibatch_size":256,          # size of mini-batches
        "replay_memory_size": 30_000,   # size of replay memory
        'clip_grad': 100,

    }
    # config['LAYOUT'] = "cramped_room_CLCE"

    # BEST ##########################
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 3_000]
    # config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['shared_rew'] = False
    # config['gamma'] = 0.95
    ###############################

    # Top Left
    config['ITERATIONS'] = 30_000
    config['LAYOUT'] = "risky_coordination_ring"
    config['replay_memory_size'] = 30_000
    config['epsilon_sched'] = [1.0, 0.1, 15_000]
    config['rshape_sched'] = [1, 0, 10_000]
    config['rationality_sched'] = [5.0, 5.0, 10_000]
    config['lr_sched'] = [1e-2, 1e-4, 3_000]
    config["rand_start_sched"]= [1.0, 0.75, 10_000]  # percentage of ITERATIONS with random start states
    config['tau'] = 0.01
    config['num_hidden_layers'] = 5
    config['size_hidden_layers'] = 256
    config['gamma'] = 0.95
    config['p_slip'] = 0.25
    config['note'] = 'medium risk + random chance start'
    Trainer(SelfPlay_QRE_OSA, config).run()



    # # Bottom Left
    # config['LAYOUT'] = "risky_coordination_ring"
    # config['ITERATIONS'] = 30_000
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 3_000]
    # config["rand_start_sched"]= [1.0, 0.5, 15_000] #config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['gamma'] = 0.97
    # config['p_slip'] = 0.1
    # config['note'] = 'minimal risk + chance start + increased gamma + lower LR'
    # Trainer(SelfPlay_QRE_OSA, config).run()
    # config['LAYOUT'] = "risky_coordination_ring"
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 3_000]
    # config["rand_start_sched"]= [1.0, 0.75, 10_000]  # percentage of ITERATIONS with random start states # config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['gamma'] = 0.95
    # config['p_slip'] = 0.1
    # config['note'] = 'minimal risk + trivial cpt+ chance start'
    # config['cpt_params'] = {'b': 0.0, 'lam': 1.0,
    #                         'eta_p': 1., 'eta_n': 1.,
    #                         'delta_p': 1., 'delta_n': 1.}
    # # config['cpt_params']= {'b': 0.4, 'lam': 1.0,
    # #                'eta_p': 0.88, 'eta_n': 0.88,
    # #                'delta_p': 0.61, 'delta_n': 0.69}
    # Trainer(SelfPlay_QRE_OSA_CPT, config).run()

    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.2, 8_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 3_000]
    # config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['shared_rew'] = False
    # config['gamma'] = 0.95
    # config['note'] = 'added collab reward shaping'
    # Trainer(SelfPlay_QRE_OSA, config).run()
    # # config['replay_memory_size'] = 30_000
    # # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # # config['rshape_sched'] = [1, 0, 10_000]
    # # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # # config['lr_sched'] = [1e-2, 1e-4, 5_000]
    # # config['perc_random_start'] = 0.9
    # # config['test_rationality'] = config['rationality_sched'][1]
    # # config['tau'] = 0.01
    # # config['num_hidden_layers'] = 6
    # # config['size_hidden_layers'] = 128
    # # config['shared_rew'] = False
    # # config['gamma'] = 0.95
    # # config['note'] = 'increased depth'
    # # Trainer(SelfPlay_QRE_OSA, config).run()


    # # Top Right
    # config['LAYOUT'] = "risky_coordination_ring"
    # config['ITERATIONS'] = 30_000
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 4_000]
    # config["rand_start_sched"] = [0.0, 0.00, 15_000]  # config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['gamma'] = 0.97
    # config['p_slip'] = 0.99
    # config['note'] = 'max risk + (test handoff)'
    # Trainer(SelfPlay_QRE_OSA, config).run()

    # bottom Right
    # config['LAYOUT'] = "risky_coordination_ring"
    # config['ITERATIONS'] = 30_000
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 3_000]
    # config["rand_start_sched"] = [1.5, 0.05, 15_000]  # config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['gamma'] = 0.95
    # config['p_slip'] = 0.1
    # config['note'] = 'minimal risk + chance start'
    # Trainer(SelfPlay_QRE_OSA, config).run()
    # config['LAYOUT'] = "risky_coordination_ring"
    # config['ITERATIONS'] = 30_000
    # config['replay_memory_size'] = 30_000
    # config['epsilon_sched'] = [1.0, 0.15, 10_000]
    # config['rshape_sched'] = [1, 0, 10_000]
    # config['rationality_sched'] = [5.0, 5.0, 10_000]
    # config['lr_sched'] = [1e-2, 1e-4, 2_000]
    # config['perc_random_start'] = 0.9
    # config['tau'] = 0.01
    # config['num_hidden_layers'] = 5
    # config['size_hidden_layers'] = 256
    # config['gamma'] = 0.95
    # config['p_slip'] = 0.1
    # config['note'] = 'minimal risk + 90% chance start'
    # Trainer(SelfPlay_QRE_OSA, config).run()

    # config['cpt_params']= {'b': 0.0, 'lam': 1.0,
    #                'eta_p': 1., 'eta_n': 1.,
    #                'delta_p': 1., 'delta_n': 1.}
    # # config['cpt_params']= {'b': 0.4, 'lam': 1.0,
    # #                'eta_p': 0.88, 'eta_n': 0.88,
    # #                'delta_p': 0.61, 'delta_n': 0.69}
    # Trainer(SelfPlay_QRE_OSA_CPT,config).run()

if __name__ == '__main__':
    main()