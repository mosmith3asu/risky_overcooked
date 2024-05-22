import numpy as np
import time

class FunctionTimer(object):
    def __init__(self):
        self.time_dicts = {}
        self.sig_digs = 4

    def __call__(self, fn_name, fn, *args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        if fn_name not in self.time_dicts.keys(): self.time_dicts[fn_name] = [end - start]
        else: self.time_dicts[fn_name].append(end - start)
        return result

    def report(self):
        # report_str = 'Times:' + ''.join([f'{key}: t({np.mean(val)}) n({len(val)}) | 'for key, val in self.time_dicts.items()])
        report_str = 'Times:' + ''.join([f'\t{key}:T({np.sum(val).round(self.sig_digs)}) t({np.mean(val).round(self.sig_digs)}) n({len(val)}) | 'for key, val in self.time_dicts.items()])

        print(report_str)
    def clear(self):
        self.time_dicts = {}