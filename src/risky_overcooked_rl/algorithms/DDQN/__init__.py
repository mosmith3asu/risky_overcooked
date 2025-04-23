import sys
import os
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))




def get_default_config(path = '\\risky_overcooked_rl\\algorithms\\DDQN\\_config.yaml'):
    import yaml
    import os

    dirs = os.getcwd().split('\\')
    src_idx = dirs.index('src')  # find index of src directory
    src_dir = '\\'.join(dirs[:src_idx+1])
    with open(f'{src_dir}{path}') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def get_default_batching(path = '\\risky_overcooked_rl\\algorithms\\DDQN\\_batching.yaml'):
    import yaml
    import os

    dirs = os.getcwd().split('\\')
    src_idx = dirs.index('src')  # find index of src directory
    src_dir = '\\'.join(dirs[:src_idx+1])
    with open(f'{src_dir}{path}') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def get_save_dir():
    # TODO: implement this is save confing to generalize to other algs
    return '\\risky_overcooked_rl\\models'