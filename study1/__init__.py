import sys
import os

print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))
ROOT_DIR = "\\study1\\"

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

def get_default_config(path = ROOT_DIR+ '_config.yaml'):
    import yaml
    import os

    # dirs = os.getcwd().split('\\')
    # src_idx = dirs.index('src')  # find index of src directory
    # src_dir = '\\'.join(dirs[:src_idx+1])
    # src_dir = get_src_dir()
    # if path.startswith('\\'):
    #     path = '\\'.join(path.split('\\')[1:])
    dirs = os.getcwd().split('\\')
    src_idx = dirs.index('study1')  # find index of src directory
    src_dir = '\\'.join(dirs[:src_idx])


    with open(f'{src_dir}{path}') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def get_src_dir():
    dirs = os.getcwd().split('\\')
    try:
        src_idx = dirs.index('src') # find index of src directory
        return '\\'.join(dirs[:src_idx + 1]) + '\\'
    except:
        root_idx = dirs.index('risky_overcooked')
        return '\\'.join(dirs[:root_idx + 1]) + "\\src\\"

def get_absolute_save_dir(path = ROOT_DIR+'models\\'):
    dirs = os.getcwd().split('\\')
    study1_idx = dirs.index('study1') # find index of study_1 directory
    return '\\'.join(dirs[:study1_idx]) + path
    # try:
    #     src_idx = dirs.index('src') # find index of src directory
    #     return '\\'.join(dirs[:src_idx + 1]) + path
    # except:
    #     root_idx = dirs.index('risky_overcooked')
    #     return '\\'.join(dirs[:root_idx + 1]) + "\\src" + path




def get_save_dir():
    # TODO: implement this is save confing to generalize to other algs
    return ROOT_DIR+'models'



SAVE_DIR = get_absolute_save_dir()
CONFIG = get_default_config()

