import sys
import os

print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))


def search_config_value(config, target_key, level=0):
    found = None
    if isinstance(config, dict):
        for key, val in config.items():
            if key == target_key:
                assert found is None, f"Key '{target_key}' found multiple times in the configuration."
                found = config[key]
            else:
                _found = search_config_value(val, target_key, level=level + 1)
                if _found is not None:
                    assert found is None, f"Key '{target_key}' found multiple times in the configuration."
                    found = _found
    if level == 0:
        assert found is not None, f"Key '{target_key}' not found in the configuration."
    return found

def set_config_value(config, target_key, new_value, level=0):
    """
    Recursively searches for target_key in a nested dictionary d and sets its value to new_value.
    Parameters:
    - config (dict): The dictionary to search.
    - target_key (str): The key to search for.
    - new_value: The value to set when the key is found.
    """
    was_found = False
    if isinstance(config, dict):
        for key,val in config.items():
            if key == target_key:
                config[key] = new_value
                assert was_found == False, f"Key '{target_key}' found multiple times in the configuration."
                was_found = True
            else:
                _found = set_config_value(val, target_key, new_value,level=level+1)
                if _found:
                    assert was_found == False, f"Key '{target_key}' found multiple times in the configuration."
                    was_found = True
    if level == 0:
        assert was_found, f"Key '{target_key}' not found in the configuration."
    return was_found

def get_default_config(path = '\\risky_overcooked_rl\\algorithms\\DDQN\\_config.yaml'):
    import yaml
    import os

    # dirs = os.getcwd().split('\\')
    # src_idx = dirs.index('src')  # find index of src directory
    # src_dir = '\\'.join(dirs[:src_idx+1])
    src_dir = get_src_dir()
    fpath = f'{src_dir}{path}'.replace('\\', '/')
    with open(fpath) as f:
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

def get_absolute_save_dir(path = '\\risky_overcooked_rl\\algorithms\\DDQN\\models\\'):
    # dirs = os.getcwd().split('\\')
    # try:
    #     dirs = os.path.abspath(__file__)
    #     dirs = dirs.replace('/', '\\').split('\\')
    #     src_idx = dirs.index('src') # find index of src directory
    # except ValueError:
    #     raise Exception(f"The 'src' directory was not found in {os.path.abspath(__file__)}.")
    # return '\\'.join(dirs[:src_idx+1]) + path
    return (get_src_dir() + path).replace('\\', '/')

def get_src_dir():
    try:
        dirs = os.path.abspath(__file__)
        dirs = dirs.replace('/', '\\').split('\\')
        src_idx = dirs.index('src')  # find index of src directory
    except ValueError:
        raise Exception(f"The 'src' directory was not found in {os.path.abspath(__file__)}.")
    return '\\'.join(dirs[:src_idx + 1])



def get_save_dir():
    # TODO: implement this is save confing to generalize to other algs
    return '\\risky_overcooked_rl\\models'



SAVE_DIR = get_absolute_save_dir()
CONFIG = get_default_config()
