"""
Utilities for managing saving, logging, and loading of models.
"""
import os
import yaml
import argparse

def get_absolute_save_dir():
    dirs = os.getcwd().split('\\')
    src_idx = dirs.index('src') # find index of src directory
    return '\\'.join(dirs[:src_idx+1]) + '\\risky_overcooked_rl\\models\\'

def get_src_dir():
    dirs = os.getcwd().split('\\')
    src_idx = dirs.index('src') # find index of src directory
    return '\\'.join(dirs[:src_idx+1])

def get_default_config():
    src_dir = get_src_dir()
    with open(f'{src_dir}\\risky_overcooked_rl\\utils\\_default_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def parse_args(config):
    # parser_args = { # !!! Redefines default config file. DANGEROUS !!!
    #     'cpt_params':{'b': 0, 'lam': 1.0, 'eta_p': 1.,
    #                   'eta_n': 1., 'delta_p': 1., 'delta_n': 1.},
    #     'LAYOUT':'risky_coordination_ring',
    #     'p_slip':0.1,
    #     'loads': '',
    #     'note': ''
    # }
    parser = argparse.ArgumentParser()
    for key,val in config.items():#parser_args.items():
        if 'cpt_params' == key:
            parser.add_argument('--' + 'cpt', dest=str(key), nargs=6,
                                action=type('', (argparse.Action,),
                                            dict(__call__=lambda a, p, n, v, o: getattr(n, a.dest).update(
                                                dict([[vi.split('=')[0], float(vi.split('=')[1])
                                                if vi.split('=')[1].replace('.', '').isnumeric()
                                                else vi.split('=')[1]
                                                       ] for vi in v])
                                            ))),
                                default={'b': 0, 'lam': 1.0, 'eta_p': 1., 'eta_n': 1., 'delta_p': 1., 'delta_n': 1.})
        else:
            parser.add_argument('--' + str(key), dest=str(key), type=type(val), default=val)
    # return parser
    args = parser.parse_args()
    config.update(vars(args))

    for key, val in config['cpt_params'].items():
        if isinstance(val, int):
            config['cpt_params'][key] = float(val)
    return config

# def main():
#     from risky_overcooked_rl.utils.deep_models import DQN_vector_feature
#     save_dir = 'models/'
#     filename = 'model.pth'
#     sm = ModelManager()
#
#     model = DQN_vector_feature(obs_shape=[41], n_actions=6, num_hidden_layers=5,
#                                size_hidden_layers=256)  # torch.tensor([0, 1, 2, 3, 4])
#     # model_info = {'fname':'test', 'timestamp': 'test',  'layout': 'test', 'p_slip': 0.0,
#     #  'b': 0.0, 'lam': 0.0, 'eta_p': 0.0, 'eta_n': 0.0, 'delta_p': 0.0, 'delta_n': 0.0}
#     fname = 'test2'
#     model_info = {'timestamp': 'tstamp', 'layout': 'test_layout', 'p_slip': 1.0,
#                   'b': 1.0, 'lam': 1.0, 'eta_p': 1.0, 'eta_n': 1.0, 'delta_p': 1.0, 'delta_n': 1.0}
#     # sm.save(model, model_info, fname)
#     # # model_info['layout'] = 'na_layout'
#     # sm.load(model_info)
#     model_info_hash = hash((model_info[key] for key in sm.search_headers))
#     fname = f'{timestamp}_{model_info_hash} '
#     print()
#
# if __name__ == '__main__':
#    main()