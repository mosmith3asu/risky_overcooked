import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections import deque
from risky_overcooked_py.visualization.state_visualizer import StateVisualizer
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import pygame
import cv2
import copy

class TrajectoryVisualizer(object):
    def __init__(self,env,blocking=True):
        self.env = env
        self.blocking = blocking
        self.slider_xpad = 0.2
        self.qued_trajector = []

        # self.spawn_figure()

    def spawn_figure(self):
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        # self.fig.close()
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.imgs = []
        self.visualizer = StateVisualizer()
        self.env.reset()
        tmp_image = self.render_image(self.env.state)
        self.img = self.ax.imshow(tmp_image)
        axfreq = self.fig.add_axes([self.slider_xpad, 0.1, 0.9 - self.slider_xpad, 0.03])
        self.time_slider = Slider(
            ax=axfreq,
            label='t',
            valmin=0,
            valmax=self.env.horizon - 1,
            valinit=0,
        )
        self.fig_number = self.fig.number
    def render_image(self,state):
        image = self.visualizer.render_state(state=state, grid=self.env.mdp.terrain_mtx)
        buffer = pygame.surfarray.array3d(image)
        image = copy.deepcopy(buffer)
        image = np.flip(np.rot90(image, 3), 1)
        image = cv2.resize(image, (2 * 528, 2 * 464))
        return image

    def update_slider(self,val):
        t = int(self.time_slider.val)
        infos = ["", ""]
        # onion_slips = trajs['ep_infos'][0][99]['episode']['ep_game_stats']['onion_slip']
        # empty_slips = trajs['ep_infos'][0][99]['episode']['ep_game_stats']['empty_slip']
        #
        # for player_idx in range(2):
        #     if t in onion_slips[player_idx]:
        #         infos[player_idx] += "Onion Slip"
        #     elif t in empty_slips[player_idx]:
        #         infos[player_idx] += "Empty Slip"
        self.ax.set_title(infos[0] + " | " + infos[1])
        self.img.set_data(self.imgs[t])
        # line.set_ydata(f(t, amp_slider.val, freq_slider.val))
        # self.fig.canvas.draw_idle()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def que_trajectory(self,state_history):
        self.qued_trajectory = state_history

    def preview_qued_trajectory(self,*args):
        self.preview_trajectory(self.qued_trajectory)

    def preview_trajectory(self,state_history):
        if self.blocking: self.spawn_figure()
        self.imgs = []
        for state in state_history:
            self.imgs.append(self.render_image(state))
        self.time_slider.on_changed(self.update_slider)

        self.fig.show()
        if self.blocking:
            while plt.fignum_exists(self.fig_number):
                self.fig.canvas.flush_events()
                time.sleep(0.1)

class RLLogger(object):
    def __init__(self, rows, cols,num_iterations=None, lw=0.5, figsize=(10, 5)):
        self.logs = {}
        self.sig_digs = 4
        self.filter_widows = {}
        self.xlabels = {}
        self.ylabels = {}
        self.display_raws = {}
        self.lines = {}
        self.filtered_lines = {}
        self.trackers = {}
        self.interfaces = {}
        self.status = {}
        self.axs = {}
        self.iax = 0  # index counter for plot
        self.rows = rows
        self.cols = cols
        self.num_iterations = num_iterations
        self.iter_count = 0 # number of iterations in loop
        plt.ion()

        self.last_iteration_time = None

        self.root_fig = plt.figure(figsize=figsize, constrained_layout=True)

        # self.subfigs = self.root_fig.subfigures(1, 2, wspace=0.00, width_ratios=[1.5, 1.])
        # self.plot_fig = self.subfigs[0]
        # self.settings_fig = self.subfigs[1]

        # self.subfigs = self.root_fig.subfigures(2, 2, wspace=0.07, width_ratios=[1.5, 1.],height_ratios=[5,1])
        # self.plot_fig = self.subfigs[0,0]
        # self.settings_fig = self.subfigs[0,1]
        # self.status_fig = self.subfigs[1, 0]
        # self.interface_fig = self.subfigs[1, 1]

        self.subfigs = self.root_fig.subfigures(1, 2, wspace=0.00, width_ratios=[1.5, 1.])
        self.plot_fig,self.interface_fig = self.subfigs[0].subfigures(2, 1, wspace=0.00, height_ratios=[10, 1])
        self.subfigs =  self.subfigs[1].subfigures(2, 1, wspace=0.00, height_ratios=[10, 1])
        self.settings_fig = self.subfigs[0]
        self.status_fig = self.subfigs[1]





        # axs0 = self.subfigs[0].subplots(2, 2)
        # self.interface_fig.set_facecolor('lightgray')
        # self.status_fig.set_facecolor('lightgray')
        # self.status_fig.suptitle('Status')
        self.settings_fig.set_facecolor('lightgray')
        self.settings_fig.suptitle('Settings')
        # self.subfigs[0].suptitle('subfigs[0]\nLeft side')
        # self.subfigs[0].supxlabel('xlabel for subfigs[0]')

        self.root_fig.canvas.manager.set_window_title('RLLoger V1.1')

        self.raw_settings = {'c': 'k', 'lw': lw}
        self.filtered_settings = {'c': 'r', 'lw': lw}

    def log(self, **data):
        for key, value in data.items():
            assert len(value) == 2, f'Value must be a tuple of length 2. Got {key}:{value}'
            self.logs[key] = np.vstack([self.logs[key], np.array(value)])

    def filter(self, x, window):
        """ apply a window filter to the values. """

        # np.convolve(x, np.ones(window), 'valid') / window
        if len(x) > window:
            f = []
            for i in range(len(x)):
                w = min(i, window)
                f.append(np.mean(x[i - w:i]))
            return f
            # return np.convolve(x, np.ones(w) / w, 'same')
        else:
            return x

    ###########################################################
    # Status Methods ############################################
    def add_status(self,key='status', iteration_timer=True,progress=True,remaining_time=True,ncols=2,fnt_size=6):
        assert self.num_iterations is not None, 'Number of iterations must be set to use status'
        self.status['Prog:'] = 0
        self.status['S/iter'] = deque([time.time()], maxlen=10)
        self.status['Time Left'] = 0
        self.axs[key] = self.status_fig.add_subplot(1, 1, 1)
        # self.axs[key].set_title('Status')
        data_dict = {'Prog':0,'S/iter':0,'Time Left':0}
        items = list(data_dict.items())
        data = []
        nrows = len(data_dict) // ncols
        ncols = np.sum([iteration_timer,progress,remaining_time])
        for r in range(nrows):
            row = []
            for c in range(ncols):
                i = r * ncols + c
                k, v = items[i]
                row.append(f'{k}: {v}')
            data.append(row)
        tbl = self.axs[key].table(cellText=data, fontsize=fnt_size, loc='center', cellLoc='left')
        tbl.auto_set_font_size(False)

        self.axs[key].axis('off')
        self.status['tbl'] = tbl

        for key, cell in tbl.get_celld().items():
            cell.set_linewidth(0)
        tbl.auto_set_column_width([0, 1])

    def start_iteration(self):
        self.last_iteration_time = time.time()
    def end_iteration(self,advance_iteration =True):
        dt = time.time() - self.last_iteration_time
        self.status['S/iter'].append(dt)
        if advance_iteration: self.iter_count += 1
    def draw_status(self):
        tbl = self.status['tbl']
        s_per_iter = np.mean(self.status["S/iter"])
        rem_time = (self.num_iterations-self.iter_count)* s_per_iter
        # reformat rem_time into hours, minutes, seconds
        hours = rem_time // 3600
        rem_time = rem_time % 3600
        minutes = rem_time // 60
        rem_time = rem_time % 60
        seconds = rem_time
        rem_time = f'{int(hours)}:{int(minutes)}:{int(seconds)}'
        tbl.get_celld()[(0, 0)].get_text().set_text(f'Progress: {np.round(100*self.iter_count/self.num_iterations,2)}%')
        tbl.get_celld()[(0, 1)].get_text().set_text(f'S/iter: {np.round(s_per_iter,2)}')
        tbl.get_celld()[(0, 2)].get_text().set_text(f'Time Left: {rem_time}')



    ###########################################################
    # Construction ############################################
    def add_tracker(self, key, vals, max_iterations,
                    xlabel='', ylabel='', title='',
                    loc=None, ):
        """Used to track paramters over time like exploration rate"""
        self.iax += 1
        self.xlabels[key] = xlabel
        self.ylabels[key] = ylabel
        if loc is not None:
            self.axs[key] = self.plot_fig.add_subplot(self.rows, self.cols, self.cols * loc[0] + loc[1])
        else:
            self.axs[key] = self.plot_fig.add_subplot(self.rows, self.cols, self.iax)

        xvals = np.linspace(0, max_iterations, len(vals))
        self.trackers[key] = self.axs[key].plot(xvals, vals)[0]
        self.axs[key].set_xlabel(xlabel)
        self.axs[key].set_ylabel(ylabel)
        self.axs[key].set_title(title)

        raise NotImplementedError

    def add_table(self, key, data_dict, loc=None, ncols=1,
                  fnt_size=9, sig_digs=4):

        # if loc is not None:  self.axs[key] = self.fig.add_subplot(self.rows, self.cols, self.cols * loc[0] + loc[1])
        # else:  self.axs[key] = self.fig.add_subplot(self.rows, self.cols, self.iax)
        self.axs[key] = self.settings_fig.add_subplot(1, 1, 1)
        items = list(data_dict.items())
        data = []
        nrows = len(data_dict) // ncols
        for r in range(nrows):
            row = []
            for c in range(ncols):
                i = r * ncols + c
                k, v = items[i]
                row.append(f'{k}: {v}')
            data.append(row)
        tbl = self.axs[key].table(cellText=data, fontsize=fnt_size, loc='center', cellLoc='left')
        # df = pd.DataFrame(data)
        # df.index.name = None
        # pd.plotting.table(self.axs[key], df, loc='center', colWidths=[0.5] * len(data_dict), cellLoc='center',
        #                   fontsize=fnt_size)
        tbl.auto_set_font_size(False)

        self.axs[key].axis('off')


    def add_lineplot(self, key,
                     xlabel='', ylabel='', title='',
                     loc=None,
                     filter_window=None, display_raw=True):

        self.iax += 1
        self.logs[key] = np.empty((0, 2))
        self.xlabels[key] = xlabel
        self.ylabels[key] = ylabel
        self.filter_widows[key] = filter_window
        self.display_raws[key] = display_raw
        if loc is not None:
            self.axs[key] = self.plot_fig.add_subplot(self.rows, self.cols, self.cols * loc[0] + loc[1])
        else:
            self.axs[key] = self.plot_fig.add_subplot(self.rows, self.cols, self.iax)
        self.lines[key] = self.axs[key].plot([], [], **self.raw_settings)[0]
        if filter_window is not None:
            self.filtered_lines[key] = self.axs[key].plot([], [], **self.filtered_settings)[0]
        self.axs[key].set_xlabel(xlabel)
        self.axs[key].set_ylabel(ylabel)
        self.axs[key].set_title(title)

    def add_button(self,label, callback, group='buttons'):
        bname = f'{group}_{label}'
        # axprev = self.interface_fig.add_axes([0.7, 0.05, 0.1, 0.075])
        n_buttons = len(self.interfaces)
        # self.axs[bname] = self.interface_fig.add_subplot(1, n_buttons+1, n_buttons+1)
        # self.axs[bname] = self.interface_fig.add_axes([])
        self.axs[bname] = self.interface_fig.add_axes([0.1, 0.1, 0.25, 0.9])
        self.interfaces[bname] = Button(self.axs[bname], label)
        self.interfaces[bname].on_clicked(callback)

    ###########################################################
    # Render Utils ############################################
    def draw(self):
        for key, data in self.logs.items():
            x = data[:, 0]
            y = data[:, 1]
            self.axs[key].set_xlim([np.min(x), np.max(x)])
            self.axs[key].set_ylim([np.min(y), np.max(y)])
            # self.lines[key].set_xdata(x)
            self.lines[key].set_data(x, y)

        for key, _ in self.filtered_lines.items():
            data = self.logs[key]
            x = data[:, 0]
            y = data[:, 1]
            y = self.filter(y, window=self.filter_widows[key])
            self.filtered_lines[key].set_data(x, y)
        self.plot_fig.canvas.draw()
        self.plot_fig.canvas.flush_events()
        self.draw_status()

    def spin(self):
        self.plot_fig.canvas.flush_events()

    def wait_for_close(self,enable=True):
        """Stops the program to wait for user input (i.e. save model, save plot, close, ect..)"""
        if enable:
            plt.ioff()
            plt.show()

def test_logger():
    config = {'p1':1,'p2':2}

    def callback(event):
        print('clicked')

    T = 1000
    data = []
    logger = RLLogger(rows = 2,cols = 1,num_iterations=T)
    logger.add_lineplot('test_reward',xlabel='iter',ylabel='$R_{test}$',filter_window=10,display_raw=True,loc = (0,1))
    logger.add_lineplot('train_reward', xlabel='iter', ylabel='$R_{train}$', filter_window=10, display_raw=True,loc=(1,1))
    logger.add_table('Params',config)
    logger.add_status()
    # logger.add_button('Save',callback)
    # logger.add_button('Close', callback)

    for i in range(T):
        logger.start_iteration()
        d = np.random.randint(0,10)
        logger.log(test_reward=[i,d],train_reward=[i, d+2])
        logger.draw()
        # data.append(d)
        # ln.set_ydata(d)
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        time.sleep(0.1)

        logger.end_iteration()
        print(i)
    logger.wait_for_close(enable=True)

#
# class RLLogger(object):
#     def __init__(self,rows,cols,lw = 0.5,figsize=(10,5)):
#         self.logs = {}
#         self.sig_digs = 4
#         self.filter_widows = {}
#         self.xlabels = {}
#         self.ylabels = {}
#         self.display_raws = {}
#         self.lines = {}
#         self.filtered_lines = {}
#         self.trackers = {}
#         self.axs = {}
#         self.iax = 0 # index counter for plot
#         self.rows = rows
#         self.cols = cols
#         plt.ion()
#
#
#         self.root_fig = plt.figure(figsize=figsize, constrained_layout=True)
#
#         self.subfigs = self.root_fig.subfigures(1, 2, wspace=0.07, width_ratios=[1.5, 1.])
#         self.plot_fig = self.subfigs[0]
#         self.settings_fig = self.subfigs[1]
#
#         # axs0 = self.subfigs[0].subplots(2, 2)
#         self.settings_fig.set_facecolor('lightgray')
#         self.settings_fig.suptitle('Settings')
#         # self.subfigs[0].suptitle('subfigs[0]\nLeft side')
#         # self.subfigs[0].supxlabel('xlabel for subfigs[0]')
#
#         self.root_fig.canvas.manager.set_window_title('RLLoger V1.1')
#
#         self.raw_settings = {'c':'k','lw':lw}
#         self.filtered_settings = {'c':'r','lw':lw}
#
#
#     def log(self, **data):
#         for key, value in data.items():
#             assert len(value) == 2, f'Value must be a tuple of length 2. Got {key}:{value}'
#             self.logs[key] = np.vstack([self.logs[key], np.array(value)])
#
#     def filter(self, x,window):
#         """ apply a window filter to the values. """
#
#         # np.convolve(x, np.ones(window), 'valid') / window
#         if len(x)>window:
#             f = []
#             for i in range(len(x)):
#                 w = min(i,window)
#                 f.append(np.mean(x[i-w:i]))
#             return f
#             # return np.convolve(x, np.ones(w) / w, 'same')
#         else: return x
#
#
#
#     def add_tracker(self, key, vals,max_iterations,
#                     xlabel='',ylabel='',title='',
#                      loc= None,):
#         """Used to track paramters over time like exploration rate"""
#         self.iax += 1
#         self.xlabels[key] = xlabel
#         self.ylabels[key] = ylabel
#         if loc is not None:  self.axs[key] = self.plot_fig.add_subplot(self.rows, self.cols, self.cols * loc[0] + loc[1])
#         else:   self.axs[key] = self.plot_fig.add_subplot(self.rows, self.cols, self.iax)
#
#         xvals = np.linspace(0,max_iterations,len(vals))
#         self.trackers[key] = self.axs[key].plot(xvals, vals)[0]
#         self.axs[key].set_xlabel(xlabel)
#         self.axs[key].set_ylabel(ylabel)
#         self.axs[key].set_title(title)
#
#         raise NotImplementedError
#
#
#     def add_table(self, key, data_dict, loc=None, ncols=1,
#                   fnt_size=9, sig_digs=4):
#
#         # if loc is not None:  self.axs[key] = self.fig.add_subplot(self.rows, self.cols, self.cols * loc[0] + loc[1])
#         # else:  self.axs[key] = self.fig.add_subplot(self.rows, self.cols, self.iax)
#         self.axs[key] = self.settings_fig.add_subplot(1, 1, 1)
#         items = list(data_dict.items())
#         data = []
#         nrows = len(data_dict) // ncols
#         for r in range(nrows):
#             row = []
#             for c in range(ncols):
#                 i = r * ncols + c
#                 k, v = items[i]
#                 row.append(f'{k}: {v}')
#             data.append(row)
#         tbl = self.axs[key].table(cellText = data,fontsize=fnt_size,loc='center',cellLoc='left')
#         # df = pd.DataFrame(data)
#         # df.index.name = None
#         # pd.plotting.table(self.axs[key], df, loc='center', colWidths=[0.5] * len(data_dict), cellLoc='center',
#         #                   fontsize=fnt_size)
#         tbl.auto_set_font_size(False)
#
#         self.axs[key].axis('off')
#
#
#     def add_lineplot(self, key,
#                      xlabel='',ylabel='',title='',
#                      loc= None,
#                      filter_window=None,display_raw=True):
#
#         self.iax += 1
#         self.logs[key] = np.empty((0,2))
#         self.xlabels[key] = xlabel
#         self.ylabels[key] = ylabel
#         self.filter_widows[key] = filter_window
#         self.display_raws[key] = display_raw
#         if loc is not None:  self.axs[key] = self.plot_fig.add_subplot(self.rows, self.cols, self.cols*loc[0] + loc[1])
#         else:  self.axs[key] = self.plot_fig.add_subplot(self.rows,self.cols,self.iax)
#         self.lines[key] = self.axs[key].plot([], [], **self.raw_settings)[0]
#         if filter_window is not None:
#             self.filtered_lines[key] = self.axs[key].plot([], [], **self.filtered_settings)[0]
#         self.axs[key].set_xlabel(xlabel)
#         self.axs[key].set_ylabel(ylabel)
#         self.axs[key].set_title(title)
#
#     def draw(self):
#         for key,data in self.logs.items():
#             x = data[:,0]
#             y = data[:,1]
#             self.axs[key].set_xlim([np.min(x), np.max(x)])
#             self.axs[key].set_ylim([np.min(y), np.max(y)])
#             # self.lines[key].set_xdata(x)
#             self.lines[key].set_data(x,y)
#
#         for key,_ in self.filtered_lines.items():
#             data = self.logs[key]
#             x = data[:,0]
#             y = data[:,1]
#             y = self.filter(y,window=self.filter_widows[key])
#             self.filtered_lines[key].set_data(x,y)
#         self.plot_fig.canvas.draw()
#         self.plot_fig.canvas.flush_events()
#     def spin(self):
#         self.plot_fig.canvas.flush_events()
#
#     def halt(self):
#         """Stops the program to wait for user input (i.e. save model, save plot, close, ect..)"""
#         plt.ioff()
#         plt.show()

# class RLLogger(object):
#     def __init__(self,rows,cols,lw = 0.5,figsize=(10,5)):
#         self.logs = {}
#         self.sig_digs = 4
#         self.filter_widows = {}
#         self.xlabels = {}
#         self.ylabels = {}
#         self.display_raws = {}
#         self.lines = {}
#         self.filtered_lines = {}
#         self.axs = {}
#         self.iax = 0 # index counter for plot
#         self.rows = rows
#         self.cols = cols
#         plt.ion()
#         self.fig = plt.figure(figsize=figsize, constrained_layout=True)
#         self.fig.canvas.manager.set_window_title('RLLoger V1.0')
#
#         self.raw_settings = {'c':'k','lw':lw}
#         self.filtered_settings = {'c':'r','lw':lw}
#
#
#     def log(self, **data):
#         for key, value in data.items():
#             assert len(value) == 2, f'Value must be a tuple of length 2. Got {key}:{value}'
#             self.logs[key] = np.vstack([self.logs[key], np.array(value)])
#
#     def filter(self, x,window):
#         """ apply a window filter to the values. """
#
#         # np.convolve(x, np.ones(window), 'valid') / window
#         if len(x)>window:
#             f = []
#             for i in range(len(x)):
#                 w = min(i,window)
#                 f.append(np.mean(x[i-w:i]))
#             return f
#             # return np.convolve(x, np.ones(w) / w, 'same')
#         else: return x
#         # return np.ma.average(x)
#         # window = min(window,len(vals))
#         # return np.convolve(vals.flatten(), np.ones(window), 'valid') / window
#         # return np.ones_like(vals)
#
#     # def add_table(self,key,data_dict, loc = None,ncols=2,
#     #                 fnt_size=12,sig_digs=4,
#     #                 x_position = 1.05,#x_position = 1.05,  # Position for the first column
#     #                 y_position = 1.0,  # Start from the top of the plot
#     #                 plt_adjust = 0.6,
#     #                 row_spacing = 0.1):
#     #
#     #     if loc is not None: self.axs[key] = self.fig.add_subplot(self.rows, self.cols, self.cols * loc[0] + loc[1])
#     #     else: self.axs[key] = self.fig.add_subplot(self.rows, self.cols, self.iax)
#     #     items =  list(data_dict.items())
#     #     data = []
#     #     nrows = len(data_dict)//ncols
#     #     for r in range(nrows):
#     #         row = []
#     #         for c in range(ncols):
#     #             i = r*ncols + c
#     #             k, v = items[i]
#     #             row.append(f'{k}: {v}')
#     #         data.append(row)
#     #     df = pd.DataFrame(data)
#     #     df.index.name = None
#     #     pd.plotting.table(self.axs[key],df,loc='center',colWidths=[0.5]*len(data_dict),cellLoc='center',fontsize=fnt_size)
#     #
#     #     self.axs[key].axis('off')
#     def add_table(self,key,data_dict, loc = None,ncols=2,
#                     fnt_size=8,sig_digs=4,
#                     x_position = 1.05,#x_position = 1.05,  # Position for the first column
#                     y_position = 1.0,  # Start from the top of the plot
#                     plt_adjust = 0.6,
#                     row_spacing = 0.1):
#
#         plt.gcf().canvas.draw()
#         row_offset = 0
#         open_width = 1-plt_adjust
#         # self.axs[key].axis('off')
#         key = list(self.axs.keys())[0]
#         for i, (k, v) in enumerate(data_dict.items()):
#             col_index = (i+row_offset) % ncols
#             row_index = (i+row_offset) // ncols
#             x = x_position + col_index * 2*(open_width/ncols)  # Adjust the spacing between columns
#             y = y_position - row_index * row_spacing  # Adjust the spacing between rows
#             text = self.axs[key].text(x, y, f'{k}: {v}', fontsize=fnt_size, transform=self.axs[key].transAxes,verticalalignment='top',horizontalalignment='left')
#
#             bbox = text.get_window_extent().transformed(self.axs[key].transAxes.inverted())
#             if bbox.x1 > (1+col_index)+1.2*open_width/ncols:
#                 row_offset+=1
#
#         plt.subplots_adjust(right=plt_adjust)
#
#     def add_lineplot(self, key,
#                      xlabel='',ylabel='',title='',
#                      loc= None,
#                      filter_window=None,display_raw=True):
#
#         self.iax += 1
#         self.logs[key] = np.empty((0,2))
#         self.xlabels[key] = xlabel
#         self.ylabels[key] = ylabel
#         self.filter_widows[key] = filter_window
#         self.display_raws[key] = display_raw
#         if loc is not None:  self.axs[key] = self.fig.add_subplot(self.rows, self.cols, self.cols*loc[0] + loc[1])
#         else:  self.axs[key] = self.fig.add_subplot(self.rows,self.cols,self.iax)
#         self.lines[key] = self.axs[key].plot([], [], **self.raw_settings)[0]
#         if filter_window is not None:
#             self.filtered_lines[key] = self.axs[key].plot([], [], **self.filtered_settings)[0]
#         self.axs[key].set_xlabel(xlabel)
#         self.axs[key].set_ylabel(ylabel)
#         self.axs[key].set_title(title)
#
#     def draw(self):
#         for key,data in self.logs.items():
#             x = data[:,0]
#             y = data[:,1]
#             self.axs[key].set_xlim([np.min(x), np.max(x)])
#             self.axs[key].set_ylim([np.min(y), np.max(y)])
#             # self.lines[key].set_xdata(x)
#             self.lines[key].set_data(x,y)
#
#         for key,_ in self.filtered_lines.items():
#             data = self.logs[key]
#             x = data[:,0]
#             y = data[:,1]
#             y = self.filter(y,window=self.filter_widows[key])
#             self.filtered_lines[key].set_data(x,y)
#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()
#     def spin(self):
#         self.fig.canvas.flush_events()

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


if __name__ == '__main__':
    test_logger()