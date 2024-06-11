import numpy as np
import time
import matplotlib.pyplot as plt
# import pandas as pd
import textwrap


class RLLogger(object):
    def __init__(self,rows,cols,lw = 0.5,figsize=(10,5)):
        self.logs = {}
        self.sig_digs = 4
        self.filter_widows = {}
        self.xlabels = {}
        self.ylabels = {}
        self.display_raws = {}
        self.lines = {}
        self.filtered_lines = {}
        self.axs = {}
        self.iax = 0 # index counter for plot
        self.rows = rows
        self.cols = cols
        plt.ion()
        self.fig = plt.figure(figsize=figsize, constrained_layout=True)
        self.fig.canvas.manager.set_window_title('RLLoger V1.0')

        self.raw_settings = {'c':'k','lw':lw}
        self.filtered_settings = {'c':'r','lw':lw}


    def log(self, **data):
        for key, value in data.items():
            assert len(value) == 2, f'Value must be a tuple of length 2. Got {key}:{value}'
            self.logs[key] = np.vstack([self.logs[key], np.array(value)])

    def filter(self, x,window):
        """ apply a window filter to the values. """

        # np.convolve(x, np.ones(window), 'valid') / window
        if len(x)>window:
            f = []
            for i in range(len(x)):
                w = min(i,window)
                f.append(np.mean(x[i-w:i]))
            return f
            # return np.convolve(x, np.ones(w) / w, 'same')
        else: return x
        # return np.ma.average(x)
        # window = min(window,len(vals))
        # return np.convolve(vals.flatten(), np.ones(window), 'valid') / window
        # return np.ones_like(vals)

    # def add_table(self,key,data_dict, loc = None,ncols=2,
    #                 fnt_size=12,sig_digs=4,
    #                 x_position = 1.05,#x_position = 1.05,  # Position for the first column
    #                 y_position = 1.0,  # Start from the top of the plot
    #                 plt_adjust = 0.6,
    #                 row_spacing = 0.1):
    #
    #     if loc is not None: self.axs[key] = self.fig.add_subplot(self.rows, self.cols, self.cols * loc[0] + loc[1])
    #     else: self.axs[key] = self.fig.add_subplot(self.rows, self.cols, self.iax)
    #     items =  list(data_dict.items())
    #     data = []
    #     nrows = len(data_dict)//ncols
    #     for r in range(nrows):
    #         row = []
    #         for c in range(ncols):
    #             i = r*ncols + c
    #             k, v = items[i]
    #             row.append(f'{k}: {v}')
    #         data.append(row)
    #     df = pd.DataFrame(data)
    #     df.index.name = None
    #     pd.plotting.table(self.axs[key],df,loc='center',colWidths=[0.5]*len(data_dict),cellLoc='center',fontsize=fnt_size)
    #
    #     self.axs[key].axis('off')
    def add_table(self,key,data_dict, loc = None,ncols=2,
                    fnt_size=8,sig_digs=4,
                    x_position = 1.05,#x_position = 1.05,  # Position for the first column
                    y_position = 1.0,  # Start from the top of the plot
                    plt_adjust = 0.6,
                    row_spacing = 0.1):

        plt.gcf().canvas.draw()
        row_offset = 0
        open_width = 1-plt_adjust
        # self.axs[key].axis('off')
        key = list(self.axs.keys())[0]
        for i, (k, v) in enumerate(data_dict.items()):
            col_index = (i+row_offset) % ncols
            row_index = (i+row_offset) // ncols
            x = x_position + col_index * 2*(open_width/ncols)  # Adjust the spacing between columns
            y = y_position - row_index * row_spacing  # Adjust the spacing between rows
            text = self.axs[key].text(x, y, f'{k}: {v}', fontsize=fnt_size, transform=self.axs[key].transAxes,verticalalignment='top',horizontalalignment='left')

            bbox = text.get_window_extent().transformed(self.axs[key].transAxes.inverted())
            if bbox.x1 > (1+col_index)+1.2*open_width/ncols:
                row_offset+=1

        plt.subplots_adjust(right=plt_adjust)

    def add_lineplot(self, key,
                     xlabel='',ylabel='',title='',
                     loc= None,
                     filter_window=None,display_raw=True):

        self.iax += 1
        self.logs[key] = np.empty((0,2))
        self.xlabels[key] = xlabel
        self.ylabels[key] = ylabel
        self.filter_widows[key] = filter_window
        self.display_raws[key] = display_raw
        if loc is not None:  self.axs[key] = self.fig.add_subplot(self.rows, self.cols, self.cols*loc[0] + loc[1])
        else:  self.axs[key] = self.fig.add_subplot(self.rows,self.cols,self.iax)
        self.lines[key] = self.axs[key].plot([], [], **self.raw_settings)[0]
        if filter_window is not None:
            self.filtered_lines[key] = self.axs[key].plot([], [], **self.filtered_settings)[0]
        self.axs[key].set_xlabel(xlabel)
        self.axs[key].set_ylabel(ylabel)
        self.axs[key].set_title(title)

    def draw(self):
        for key,data in self.logs.items():
            x = data[:,0]
            y = data[:,1]
            self.axs[key].set_xlim([np.min(x), np.max(x)])
            self.axs[key].set_ylim([np.min(y), np.max(y)])
            # self.lines[key].set_xdata(x)
            self.lines[key].set_data(x,y)

        for key,_ in self.filtered_lines.items():
            data = self.logs[key]
            x = data[:,0]
            y = data[:,1]
            y = self.filter(y,window=self.filter_widows[key])
            self.filtered_lines[key].set_data(x,y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    def spin(self):
        self.fig.canvas.flush_events()

class FunctionTimer(object):
    """ Used for timing functions to see how long they take to run."""
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

def test_logger():
    config = {'p1':1,'p2':2}

    data = []
    logger = RLLogger(rows = 2,cols = 2)
    logger.add_lineplot('test_reward',xlabel='iter',ylabel='$R_{test}$',filter_window=10,display_raw=True,loc = (0,1))
    logger.add_lineplot('train_reward', xlabel='iter', ylabel='$R_{train}$', filter_window=10, display_raw=True,loc=(1,1))
    logger.add_table('Params',config)
    T = 1000
    for i in range(T):
        d = np.random.randint(0,10)
        logger.log(test_reward=[i,d],train_reward=[i, d+2])
        logger.draw()
        # data.append(d)
        # ln.set_ydata(d)
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        time.sleep(0.1)
        print(i)

if __name__ == '__main__':
    test_logger()