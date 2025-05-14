import warnings

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections import deque
from risky_overcooked_py.visualization.state_visualizer import StateVisualizer
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Button, Slider



def filter_data( x, window):
    """ apply a window filter to the values. """
    if len(x) > window:
        f = np.zeros_like(x)
        std = np.zeros_like(x)
        for i in range(1,len(x)):
            w = min(i, window)
            f[i] = np.mean(x[i - w:i])
            std[i] = np.std(x[i - w:i])
        f[0] = f[1]
        std[0] =std[1]
        return f, std
    else:
        return x, np.zeros_like(x)

class StatusItems:

    def __init__(self,id,ax,font_sz=12,**kwargs):
        self.id = id
        self.ax = ax
        self.font_sz = font_sz

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.axis('off')
        self.def_val = '-'

        self.data = [[]]
        self.key_locs = {}

        self.tbl = None

        # self.tbl = self.ax.table(cellText=self.data, fontsize=font_sz, loc='center', cellLoc='left')
        # self.tbl.auto_set_font_size(True)
        # self.tbl.auto_set_column_width([0, 1])

    def add_item(self, key):
        txt = f'{key}: {self.def_val}'
        loc = (0, len(self.data[0]))
        self.data[0].append(txt)
        self.key_locs[key] = loc


        if self.tbl is None:
            self.tbl = self.ax.table(cellText=self.data, fontsize=self.font_sz, loc='center', cellLoc='left')
        else:
            height = self.tbl.get_celld()[(0, 0)].get_height()
            self.tbl.add_cell(loc[0],loc[1], text=txt,height=height,width=1)

        for key, cell in self.tbl.get_celld().items():
            cell.set_linewidth(0)

        # self.tbl.auto_set_row_height([i for i in range(len(self.data))])
        self.tbl.auto_set_column_width([i for i in range(len(self.data[0]))])
        self.tbl.auto_set_font_size(True)
    def update_item(self,**kwargs):
        r = 0
        for key,val in kwargs.items():
            loc = self.key_locs[key]
            self.tbl.get_celld()[loc].get_text().set_text(f'{key}: {val}')

        self.tbl.auto_set_column_width([i for i in range(len(self.data))])
        self.tbl.auto_set_font_size(True)

    def draw(self):
        """ Does not need draw call"""
        pass

    @property
    def keys(self):
        return list(self.key_locs.keys())


class LinePlotItem:
    def __init__(self, id, ax,title='', xlabel='', ylabel='',filter_window=10,xtick=True):
        self.id = id
        self.std_settings = {'color':'r','alpha':0.2}
        self.line_settings = {'color':'k'}

        # Initialize the LinePlotLogger
        self.ax = ax
        self.title = title
        self.line = None # do not draw till sufficient data
        self.patches = None

        # Set annotations
        self.ax.set_title(self.title)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        # remove x ticks
        if not xtick:
            self.ax.set_xticks([])

        # instantiate dummy data
        self.data = np.empty((0, 2), dtype=float) # x, y  data
        self.patches_data = np.empty((0, 3), dtype=float) # x, y+std,y-std data
        self.filtered_data = np.empty((0, 2), dtype=float) # x, y filtered data
        self.filter_window = filter_window
        # self.tics_between_filter = 1 # reduces computational load from filter
        # self.tic = 0

    def add_data(self, x, y):
        """Append new data to the existing data"""
        self.data = np.vstack([self.data, np.array([[x, y]])])
        if self.filtered_data.shape[0]>self.filter_window:
            i = np.shape(self.data)[0]
            window_data = self.data[i-self.filter_window:i,1]
        else:
            window_data = self.data[:, 1]
        y_filt = np.mean(window_data)  # update last value
        std = np.std(window_data)
        self.filtered_data = np.vstack([self.filtered_data, np.array([[x, y_filt]])])
        self.patches_data = np.vstack([self.patches_data, np.hstack([x, y_filt - std, y_filt + std])])

    def draw(self):


        # Update line plot
        self.update_line()
        self.update_patches()

        # Resize ax
        self.ax.relim()
        self.ax.autoscale_view()

    def update_patches(self):
        """" Update stdv region """

        if np.shape(self.patches_data)[0] < 2:
            return
        x = self.patches_data[:,0]
        lower_bound = self.patches_data[:,1]
        upper_bound = self.patches_data[:,2]
        if self.patches is not None:
            self.patches.remove()
        self.patches = self.ax.fill_between(x, lower_bound, upper_bound, **self.std_settings)

    def update_line(self):
        """ Update filtered/raw line"""
        # Do not update if insufficient data
        if np.shape(self.filtered_data)[0] < 2 and np.shape(self.data)[0] < 2:
            return False
        elif self.line is None:
            # Create a new line plot
            self.line, = self.ax.plot(self.data[:, 0], self.data[:, 1],**self.line_settings)
        else:
            n_filter = self.filtered_data.shape[0]
            xy = np.vstack([self.filtered_data, self.data[n_filter:, :]])


            # Else update with existing data
            # x = self.filtered_data[:, 0] if self.filtered_data.shape[0] >= 2 else self.data[:,0]
            # y = self.filtered_data[:, 1] if self.filtered_data.shape[0] >= 2 else self.data[:,1]
            self.line.set_data(xy[:,0], xy[:,1])
        return True

    def update_checkpoint(self,x):
        pass


class IterationTimer:
    def __init__(self,num_iters=None, buffer_length=10,sig_dig=1):
        self.iteration_data = deque(maxlen=buffer_length)
        self.start_time = time.time()
        self.iter_count = 0 # used for remaining time calc
        self.num_iters = num_iters
        self.sig_dig = sig_dig
        self.per_iteration_key = 's/it'
        self.remaining_key = 'Rem'
        self.fallback_val = '-'

    def tic(self):
        dur = time.time() - self.start_time
        self.start_time = time.time()
        self.iteration_data.append(dur)
        self.iter_count += 1

    @property
    def per_iteration(self):
        return  np.round(np.mean(self.iteration_data),self.sig_dig) if len(self.iteration_data)>1 else self.fallback_val

    @property
    def remaining(self):
        """ Remaining time of Alg (if specified)"""
        per_iter = self.per_iteration
        if self.num_iters is None or per_iter ==self.fallback_val: return {'-'}
        rem_time = (self.num_iters - self.iter_count) * per_iter # seconds
        rem_time = rem_time/3600
        return round(rem_time, self.sig_dig)

class RL_Logger:
    def __init__(self, num_iters = 0 , figsize=(10, 5)):
        plt.ion()

        self.root_fig = plt.figure(figsize=figsize, constrained_layout=True)
        self.root_fig.canvas.manager.set_window_title('RLLoger V2.1')
        self.fig_number = self.root_fig.number

        ######## DIVIDE FIGURES INTO GROUPS ##################################
        self.left_group_fig,self.right_group_fig = self.root_fig.subfigures(1, 2, wspace=0.00, width_ratios=[1.5, 1.])
        self.lineplot_fig, self.interface_fig = self.left_group_fig.subfigures(2, 1, wspace=0.00, height_ratios=[10, 1])
        self.settings_fig, self.status_fig = self.right_group_fig.subfigures(2, 1, wspace=0.00, height_ratios=[10, 1])

        self.dynamic_figs = [self.root_fig,self.lineplot_fig,self.status_fig]
        ######### DATA ###############################
        self.lineplots = []
        self.interfaces = []
        self.status = None
        self.iter_timer = IterationTimer(num_iters=num_iters,buffer_length=10)
        self.add_status(self.iter_timer.per_iteration_key,self.iter_timer.remaining_key) # always add iteration timer to status


    ###############################################
    ####### INITIALIZATION METHODS ################
    def append_ax(self,fig,loc):
        """Change geometry of existing ax"""
        axs = fig.axes
        if loc == 'bottom':
            row = len(axs) + 1
            gs = gridspec.GridSpec(row, 1)
            for i, ax in enumerate(axs):
                ax.set_position(gs[i].get_position(fig))
                ax.set_subplotspec(gs[i])
            n_plots = len(axs)
            new_ax = fig.add_subplot(n_plots + 1, 1, n_plots + 1)
        elif loc == 'right':
            col = len(axs) + 1
            gs = gridspec.GridSpec(1, col)
            for i, ax in enumerate(axs):
                ax.set_position(gs[i].get_position(fig))
                ax.set_subplotspec(gs[i])
            n_plots = len(axs)
            new_ax = fig.add_subplot(1, n_plots + 1, n_plots + 1)
        else:
            raise NotImplementedError(f"Unknown location: {loc}. Supported locations are 'bottom' or 'right'.")

        return new_ax
    def add_lineplot(self,id,**kwargs):
        # dynamically add new ax self.lineplot_fig
        if len(self.lineplots) == 0:
            ax = self.lineplot_fig.add_subplot(111)
        else:
            ax = self.append_ax(self.lineplot_fig,loc='bottom')
        self.lineplots.append(LinePlotItem(id,ax,**kwargs))

    def add_status(self,*args,**kwargs):
        # always add iter_timer
        for id in args:
            print(id)
            if self.status is None:
                ax = self.status_fig.add_subplot(111)
                self.status = StatusItems(id, ax, **kwargs)

            self.status.add_item(id)

    def add_settings(self):
        pass

    ###############################################
    ############# RUNTIME METHODS #################

    def log(self,**kwargs):
        for key,val in kwargs.items():
            is_found = 0
            # Update Line Data
            for lineplot in self.lineplots:
                if lineplot.id == key:
                    lineplot.add_data(val[0],val[1])
                    is_found += 1

            # Update status data
            for status_key in self.status.keys:
                if key == status_key:
                    self.status.update_item(**{key: val})
                    is_found += 1

            self.status.update_item(**{self.iter_timer.per_iteration_key: self.iter_timer.per_iteration})
            self.status.update_item(**{self.iter_timer.remaining_key: self.iter_timer.remaining})
            # Check uniqueness and available
            assert is_found > 0, f"Lineplot with id {key} not found in RL_logger.log(id=val)"
            assert is_found <= 1, f"Multiple id {key}  found in RL_logger.log(id=val)"

    def update(self):
        """ Updates drawing of linplots """
        for lineplot in self.lineplots:
            lineplot.draw()
        self.status.draw()

        self.root_fig.canvas.draw()
        self.spin()

    def spin(self):
        """ Call this periodically to unfreeze plot window"""
        self.root_fig.canvas.flush_events()

    def wait_for_close(self,enable=True):
        """Stops the program to wait for user input (i.e. save model, save plot, close, ect..)"""
        if enable and not self.is_closed:
            print('\nWaiting for plot to close...')
            while plt.fignum_exists(self.fig_number):
                self.spin()
                time.sleep(0.1)

    def save_fig(self, PATH):
        self.root_fig.savefig(PATH)

    def start_iteration(self):
        """ Put this at beginning of iteration loop for timing"""
        self.iter_timer.tic()

    ###############################################
    ############# STATUS METHODS ##################

    @property
    def is_closed(self):
        return not plt.fignum_exists(self.fig_number)


    def current_val(self,id):
         return None

def main():
    # Example usage
    N = 100
    logger = RL_Logger(num_iters=N)
    logger.add_lineplot('test_reward', xlabel='', ylabel='$R_{test}$', filter_window=10,xtick=False)
    logger.add_lineplot('train_reward', xlabel='iter', ylabel='$R_{train}$', filter_window=1)
    logger.add_status('eps')
    # logger.add_status('train_reward', xlabel='iter', ylabel='$R_{train}$', filter_window=1)

    for i in range(N):
        # print(i)
        logger.start_iteration()

        logger.log(test_reward = [i, np.sin(i/10)+np.random.rand()/4],
                   train_reward = [i, np.sin(i/10)+np.random.rand()/4],
                   eps = i
                   )
        logger.update()
        time.sleep(0.01)

    logger.wait_for_close()
    plt.show()
if __name__=="__main__":
    main()