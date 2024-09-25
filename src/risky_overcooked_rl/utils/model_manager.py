"""
Utilities for managing saving, logging, and loading of models.
"""
import os
import csv
import torch
import warnings
import yaml
import argparse

class ModelManager:
    def __init__(self):
        self.save_dir = self.get_absolute_save_dir()
        self.logger_fname = '_model_logs.csv'
        self.headers = ['fname','timestamp', 'layout', 'p_slip',
                        'b','lam','eta_p','eta_n', 'delta_p', 'delta_n']
        self.model_info_headers = self.headers[1:]
        self.search_headers = self.headers[2:]
        self.pt_ext = '.pt'
        self.check_logger_file()

    ###############################
    # Saving Functions ###########
    def save(self, model, model_info, fname):
        model_path = self.save_dir + fname + self.pt_ext
        torch.save(model.state_dict(), model_path)
        self.log_model(model_info, fname)

    def log_model(self, model_info,fname):
        logger_path = self.save_dir + self.logger_fname
        row_data = [fname] + self.model_info_dict2list(model_info)
        assert os.path.isfile(logger_path), f'No model_logs.csv file found at {logger_path}'
        with open(logger_path, mode='a') as f:
            csv.writer(f, lineterminator='\n').writerow(row_data)

    def model_info_dict2list(self, model_info,search=False):
        if search: return [model_info[key] for key in self.search_headers]
        else: return [model_info[key] for key in self.headers[1:]]

    ###############################
    # Loading Functions ###########
    def load(self,model_info):
        # Find what fname
        model_path = self.save_dir + self.find_model_fname(model_info)
        return torch.load(model_path, weights_only=True)

    def find_model_fname(self, model_info):
        """ Iterate through logger file to find model with matching model_info """
        model_fname = None
        logger_path = self.save_dir + self.logger_fname
        with open(logger_path, mode='r') as f:
            saved_info_lsts = list(csv.reader(f))[1:] # skips header
            model_info_lst = self.model_info_dict2list(model_info,search=True)
            for saved_info in saved_info_lsts:
                # convert numeric strings into float
                saved_info = [float(info) if info.replace('.','').isnumeric() else info for info in saved_info]
                # check if info matches and store fname if does
                if model_info_lst == saved_info[2:]: # skip fname and timestamp
                    if model_fname is not None:
                        warnings.warn(f'Multiple models found with same model_info: {model_info}. \nUsing most recent [{saved_info[1]}]...')
                    model_fname = saved_info[0]
        assert model_fname is not None, f'No model found with model_info: {model_info}' # check if model found
        return model_fname

    def get_default_config(self):
        with open('utils/_default_config.yaml') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config
    ###############################
    # Navigation Functions ########
    def get_absolute_save_dir(self):
        dirs = os.getcwd().split('\\')
        src_idx = dirs.index('src') # find index of src directory
        return '\\'.join(dirs[:src_idx+1]) + '\\risky_overcooked_rl\\models\\'

    def check_logger_file(self):
        # check if file "model_logs.csv" exists. If not, create it and write the header
        logger_path = self.save_dir + self.logger_fname
        if not os.path.isfile(logger_path):
            warnings.warn(f'No model_logs.csv file found. Creating new file at {logger_path}')
            with open(logger_path, mode='w') as f:
                csv.writer(f, lineterminator='\n').writerow(self.headers)
        else:
            # check if headers are correct
            with open(logger_path, mode='r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                assert headers == self.headers, f'Headers in {logger_path} do not match expected headers.'

# noinspection PyDictCreation
def get_argparser():
    parser_args = { # !!! Redefines default config file. DANGEROUS !!!
        'cpt_params':{'b': 0, 'lam': 1.0, 'eta_p': 1.,
                      'eta_n': 1., 'delta_p': 1., 'delta_n': 1.},
        'LAYOUT':'risky_coordination_ring',
        'p_slip':0.1
    }
    parser = argparse.ArgumentParser()
    for key,val in parser_args.items():
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
    return parser

def main():
    from risky_overcooked_rl.utils.deep_models import DQN_vector_feature
    save_dir = 'models/'
    filename = 'model.pth'
    sm = ModelManager()

    model = DQN_vector_feature(obs_shape=[41], n_actions=6, num_hidden_layers=5,
                               size_hidden_layers=256)  # torch.tensor([0, 1, 2, 3, 4])
    # model_info = {'fname':'test', 'timestamp': 'test',  'layout': 'test', 'p_slip': 0.0,
    #  'b': 0.0, 'lam': 0.0, 'eta_p': 0.0, 'eta_n': 0.0, 'delta_p': 0.0, 'delta_n': 0.0}
    fname = 'test2'
    model_info = {'timestamp': 'tstamp', 'layout': 'test_layout', 'p_slip': 1.0,
                  'b': 1.0, 'lam': 1.0, 'eta_p': 1.0, 'eta_n': 1.0, 'delta_p': 1.0, 'delta_n': 1.0}
    # sm.save(model, model_info, fname)
    # # model_info['layout'] = 'na_layout'
    # sm.load(model_info)
    model_info_hash = hash((model_info[key] for key in sm.search_headers))
    fname = f'{timestamp}_{model_info_hash} '
    print()

if __name__ == '__main__':
   main()