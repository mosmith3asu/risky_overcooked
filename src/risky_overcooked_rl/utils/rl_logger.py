import warnings

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
from scipy.stats import multivariate_normal
from matplotlib.widgets import CheckButtons


class TrajectoryVisualizer(object):
    def __init__(self,env,blocking=True):
        self.env = env
        self.blocking = blocking
        self.slider_xpad = 0.2
        self.qued_trajector = []
        self.visualizer = StateVisualizer()
        # self.spawn_figure()

    def spawn_figure(self):
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        # self.fig.close()
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.imgs = []
        # self.visualizer = StateVisualizer()
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
    def get_images(self,state_history):
        imgs = [self.render_image(state) for state in state_history]
        return imgs
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


class TrajectoryHeatmap(object):
    def __init__(self,env,blocking=True):
        self.env = env
        self.blocking = blocking
        self.qued_trajector = []
        self.visualizer = StateVisualizer()
        self.grid_sz = np.shape(self.env.mdp.terrain_mtx)

        self.checkboxes = {
            'Player 1': True,
            'Player 2': True,
            'Unowned': True,
            'locations': True,
            'onion': False,
            'dish': False,
            'soup': False,
        }
    def draw_backgrounds(self,axs=None):
        state = self.qued_trajectory[0]
        # image = self.visualizer.render_state(state=state, grid=self.env.mdp.terrain_mtx)
        image = self.visualizer.render_state_nochefs(state=state, grid=self.env.mdp.terrain_mtx)
        buffer = pygame.surfarray.array3d(image)
        image = copy.deepcopy(buffer)
        image = np.flip(np.rot90(image, 3), 1)
        image = cv2.resize(image, (2 * 528, 2 * 464))


        for ax in (self.axs if axs is None else axs):
            ax.imshow(image)
        return image


    def spawn_figure(self):
        nrows,ncols = 2, 3
        self.fig, self.axs = plt.subplots(nrows,ncols,constrained_layout=True)
        self.fig_number = self.fig.number

        # Define individual axes
        self.ax_dict = {
            'all_locs': self.axs[0, 0],
            'p1_locs': self.axs[0, 1],
            'p2_locs': self.axs[0, 2],
            'onion': self.axs[1, 0],
            'dish': self.axs[1, 1],
            'soup': self.axs[1, 2]
        }
        self.ax_dict['all_locs'].set_title('All Locations')
        self.ax_dict['p1_locs'].set_title('P1 Locations')
        self.ax_dict['p2_locs'].set_title('P2 Locations')
        self.ax_dict['onion'].set_title('Onions')
        self.ax_dict['dish'].set_title('Dishes')
        self.ax_dict['soup'].set_title('Soups')


        self.axs = self.axs.flatten()
        for ax in self.axs:
            ax.set_xticks([])
            ax.set_yticks([])


    def que_trajectory(self, state_history):
        self.qued_trajectory = state_history

    def preview(self,*args):
        self.spawn_figure()
        self.img = self.draw_backgrounds()
        masks = self.calc_masks()
        for key,mask in masks.items():
            self.draw_heatmap(self.ax_dict[key],mask)
        # self.draw_heatmap(self.ax_dict['soup'],masks['soup'])

        if self.blocking:
            while plt.fignum_exists(self.fig_number):
                self.fig.canvas.flush_events()
                time.sleep(0.1)
    def calc_masks(self):
        masks = {
            'all_locs': np.zeros(np.flip(self.grid_sz)),
            'p1_locs': np.zeros(np.flip(self.grid_sz)),
            'p2_locs': np.zeros(np.flip(self.grid_sz)),
            'onion': np.zeros(np.flip(self.grid_sz)),
            'dish': np.zeros(np.flip(self.grid_sz)),
            'soup': np.zeros(np.flip(self.grid_sz)),
        }

        for t, state in enumerate(self.qued_trajectory):
            # Increment player locations #################
            for ip,player in enumerate(state.players):
                if t > 0:
                    prev_state = self.qued_trajectory[t-1]
                    prev_pos = prev_state.players[ip].position
                    if not np.all(player.position == prev_pos):
                        masks[f'p{ip+1}_locs'][player.position] += 1
                        masks[f'all_locs'][player.position] += 1
                else:
                    masks[f'p{ip + 1}_locs'][player.position] += 1
                    masks[f'all_locs'][player.position] += 1

            # Increment objects ##################################
            for obj_name,obj_lst in state.all_objects_by_type.items():
                if t > 0:
                    prev_obj_lst = self.qued_trajectory[t - 1].all_objects_by_type[obj_name]
                    prev_locs = [_obj.position for _obj in prev_obj_lst]
                else: prev_locs = []

                for obj in obj_lst:
                    # If the object was not their last turn
                    if not (obj.position in prev_locs):
                        masks[f'{obj_name}'][obj.position] += 1

        return masks

    def draw_heatmap(self,ax,mask):

        """
        https://python-graph-gallery.com/2d-density-plot/
        https://python-graph-gallery.com/bubble-plot/
        :param mask:
        :return:
        """
        mask = np.flip(mask.T,axis=0)

        # draw the heatmap ontop of overcooked map
        from matplotlib.colors import ListedColormap
        import matplotlib.colors as colors

        max_alpha = 0.75
        # cmap = plt.cm.plasma
        cmap = plt.cm.jet
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, max_alpha, cmap.N)
        my_cmap = ListedColormap(my_cmap)

        ny,nx = self.grid_sz
        img_shape = np.shape(self.img)[:2]
        szy,szx = int(img_shape[0]/ny), int(img_shape[1]/nx)
        # center = [0.5 * (szy - 1), 0.5 * (szx - 1)]
        center = [0.5 * (szy), 0.5 * (szx)]
        scale_mask = np.zeros([szy, szx])
        for iy in range(szy):
            for ix in range(szx):
                scale_mask[iy, ix] = multivariate_normal.pdf([ix, iy], mean=center, cov=3*szx)

        for iy in range(ny):
            for ix in range(nx):
                x = np.linspace(ix, ix + 1, szx) * szx
                y = szy - np.linspace(iy, iy + 1, szy) * szy
                val = scale_mask * mask[iy, ix]
                X, Y = np.meshgrid(x, y)
                ax.pcolor(X, Y + szy * (ny - 1), val,
                          norm=colors.Normalize(vmin=0, vmax=np.max(scale_mask) * np.max(mask)),
                          cmap=my_cmap)

    def draw_heatmap2(self,ax,mask):

        """
        https://python-graph-gallery.com/2d-density-plot/
        https://python-graph-gallery.com/bubble-plot/
        :param mask:
        :return:
        """
        mask = np.flip(mask.T,axis=0)

        # draw the heatmap ontop of overcooked map
        from matplotlib.colors import ListedColormap
        import matplotlib.patches as patches
        import matplotlib.colors as colors

        max_alpha = 0.5
        # cmap = plt.cm.plasma
        # cmap = plt.cm.jet
        cmap = plt.cm.autumn
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, max_alpha, cmap.N)
        my_cmap = ListedColormap(my_cmap)

        ny,nx = self.grid_sz
        img_shape = np.shape(self.img)[:2]
        szy,szx = int(img_shape[0]/ny), int(img_shape[1]/nx)
        # center = [0.5 * (szy - 1), 0.5 * (szx - 1)]
        center = [0.5 * (szy), 0.5 * (szx)]
        scale_mask = np.zeros([szy, szx])
        for iy in range(szy):
            for ix in range(szx):
                scale_mask[iy, ix] = multivariate_normal.pdf([ix, iy], mean=center, cov=3*szx)

        for iy in range(ny):
            for ix in range(nx):
                # x = np.linspace(ix, ix + 1, szx) * szx
                # y = szy - np.linspace(iy, iy + 1, szy) * szy
                # val = scale_mask * mask[iy, ix]
                # X, Y = np.meshgrid(x, y)
                # ax.pcolor(X, Y + szy * (ny - 1), val,
                #           norm=colors.Normalize(vmin=0, vmax=np.max(scale_mask) * np.max(mask)),
                #           cmap=my_cmap)
                val = mask[iy, ix]/np.max(mask)
                x = ix * szx
                y = iy * szy
                if val>0:

                    rect = patches.Rectangle((x, y), szx, szy, facecolor=my_cmap(val), alpha=max_alpha*val)
                    ax.add_patch(rect)


class RLLogger(object):
    def __init__(self, rows, cols,num_iterations=None, lw=0.5, figsize=(10, 5)):
        self.max_log_size = 1000

        self.logs = {}
        self.sig_digs = 4
        self.filter_widows = {}
        self.xlabels = {}
        self.ylabels = {}
        self.display_raws = {}
        self.display_stds = {}
        self.lines = {}
        self.filtered_lines = {}
        self.checkpoint_lines = {}
        self.std_patches = {}
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
        self.subfigs = self.root_fig.subfigures(1, 2, wspace=0.00, width_ratios=[1.5, 1.])
        self.plot_fig,self.interface_fig = self.subfigs[0].subfigures(2, 1, wspace=0.00, height_ratios=[10, 1])
        self.subfigs =  self.subfigs[1].subfigures(2, 1, wspace=0.00, height_ratios=[10, 1])
        self.settings_fig = self.subfigs[0]
        self.status_fig = self.subfigs[1]

        self.fig_number = self.root_fig.number

        self.settings_fig.set_facecolor('lightgray')
        self.settings_fig.suptitle('Settings')
        self.root_fig.canvas.manager.set_window_title('RLLoger V1.2')

        self.std_settings = {'color':'r','alpha':0.2}
        self.raw_settings = {'c': 'gray', 'lw': lw}
        self.filtered_settings = {'c': 'k', 'lw': lw}
    def close_plots(self):
        plt.close(self.root_fig)
    def log(self, **data):
        for key, value in data.items():
            assert len(value) == 2, f'Value must be a tuple of length 2. Got {key}:{value}'
            self.logs[key] = np.vstack([self.logs[key], np.array(value)])

    def filter(self, x, window):
        """ apply a window filter to the values. """

        # np.convolve(x, np.ones(window), 'valid') / window
        if len(x) > window:
            f = []
            std = []
            for i in range(len(x)):
                w = min(i, window)
                f.append(np.mean(x[i - w:i]))
                std.append(np.std(x[i - w:i]))
            return f, std
            # return np.convolve(x, np.ones(w) / w, 'same')
        else:
            return x, None

    def down_sample(self):
        for key, data in self.logs.items():
            if len(data) >= self.max_log_size:
                d1 = data[::2]
                d2 = data[1::2]
                if len(d1) == len(d2):
                    self.logs[key] = (d1+d2)/2
                else:
                    self.logs[key] = np.vstack([(d1[:-1,:] + d2) / 2, d1[-1,:]])
                # print(f'Down sampling {key} from {len(data)} to {len(self.logs[key])} samples')


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

    def add_checkpoint_line(self):
        for key, data in self.logs.items():
            self.checkpoint_lines[key] = self.axs[key].axvline(x=0, color='g', linestyle='--',lw=1, label='TBD')

    def add_lineplot(self, key,
                     xlabel='', ylabel='', title='',
                     loc=None,
                     filter_window=None,
                     display_raw=False,
                     display_std = True
                     ):

        self.iax += 1
        self.logs[key] = np.empty((0, 2))
        self.xlabels[key] = xlabel
        self.ylabels[key] = ylabel
        self.filter_widows[key] = filter_window
        self.display_raws[key] = display_raw
        self.display_stds[key] = display_std
        if loc is not None: self.axs[key] = self.plot_fig.add_subplot(self.rows, self.cols, self.cols * loc[0] + loc[1])
        else: self.axs[key] = self.plot_fig.add_subplot(self.rows, self.cols, self.iax)

        if display_raw: self.lines[key] = self.axs[key].plot([], [], **self.raw_settings)[0] # raw data line

        if filter_window is not None:
            self.filtered_lines[key] = self.axs[key].plot([], [], **self.filtered_settings)[0]

        if display_std:

            x = []
            lower_bound = []
            upper_bound = []
            self.std_patches[key] = self.axs[key].fill_between(x, lower_bound, upper_bound, **self.std_settings)
        self.axs[key].set_xlabel(xlabel)
        self.axs[key].set_ylabel(ylabel)
        self.axs[key].set_title(title)

    def add_button(self,label, callback, group='buttons'):
        bname = f'{group}_{label}'
        # axprev = self.interface_fig.add_axes([0.7, 0.05, 0.1, 0.075])
        n_buttons = len(self.interfaces)
        # self.axs[bname] = self.interface_fig.add_subplot(1, n_buttons+1, n_buttons+1)
        # self.axs[bname] = self.interface_fig.add_axes([])
        self.axs[bname] = self.interface_fig.add_axes([0.05+0.27*n_buttons, 0.1, 0.25, 0.9])
        self.interfaces[bname] = Button(self.axs[bname], label)
        self.interfaces[bname].on_clicked(callback)

    ###########################################################
    # Render Utils ############################################
    def remove_outliers(self, data, window=0.95):
        """Remove the top and bottom 5% of the data"""
        if len(data)>3:
            mean = np.mean(data)
            sd = np.std(data)
            final_list = [x for x in data if (x > mean - 2 * sd) and (x < mean + 2 * sd)]
            # final_list = [x for x in data if (x > mean - 2 * sd)]
            # final_list = [x for x in final_list if (x < mean + 2 * sd)]
            return final_list
        else:
            return data

    def update_checkpiont_line(self,iteration):
        try:
            for key, data in self.checkpoint_lines.items():
                self.checkpoint_lines[key].set_xdata(iteration)
                data = self.logs[key]
                icp =  np.where(data[:, 0]==iteration)[0][0]

                window = data[icp-5:icp+5, 1]
                m = np.mean(window).round(1)
                std = np.std(window).round(1)
                txt = f'R = {m}$\pm${std}'

                self.checkpoint_lines[key].set_label(txt)
                self.axs[key].legend(loc='upper right', fontsize=6)
        except:
            pass
            # print(f'\n\nRLLogger.update_checkpoint_line() exception...')
    def draw(self):
        if self.iter_count % 10 == 0:
            self.down_sample()

        # Raw Data Lines
        for key, _ in self.lines.items():
            data = self.logs[key]
            x = data[:, 0]
            y = data[:, 1]

            self.lines[key].set_data(x, y)
            if len(x) > 1:
                self.axs[key].set_xlim([np.min(x), 1.1 * np.max(x)])

                if 'loss' in key.lower():
                    _y = y
                    # _y = self.remove_outliers(y)
                    if np.size(_y) > 2:
                        self.axs[key].set_ylim([0, 1.1 * np.max(_y + 1e-6)])
                else:
                    self.axs[key].set_ylim([np.min(y), np.max(y) + 0.1])

        # Filter Data Lines
        for key, _ in self.filtered_lines.items():
            data = self.logs[key]
            x = data[:, 0]
            y = data[:, 1]
            y, std = self.filter(y, window=self.filter_widows[key])
            self.filtered_lines[key].set_data(x, y)

            if key in self.std_patches and std is not None:
                y = np.array(y)
                std = np.array(std)
                self.std_patches[key].remove()
                self.std_patches[key] = self.axs[key].fill_between(x, y - std, y + std, **self.std_settings)


        self.plot_fig.canvas.draw()
        self.plot_fig.canvas.flush_events()

        self.draw_status()

    def spin(self):
        self.plot_fig.canvas.flush_events()


    def wait_for_close(self,enable=True):
        """Stops the program to wait for user input (i.e. save model, save plot, close, ect..)"""
        if enable and not self.is_closed:
            print('\nWaiting for plot to close...')
            while plt.fignum_exists(self.fig_number):
                self.spin()
                time.sleep(0.1)
    @property
    def is_closed(self):
        return not plt.fignum_exists(self.fig_number)

    def save_fig(self,PATH):
        self.root_fig.savefig(PATH)

def test_logger():
    config = {'p1':1,'p2':2}

    def callback(event):
        print('clicked')

    T = 200
    data = []
    logger = RLLogger(rows = 2,cols = 1,num_iterations=T)
    logger.add_lineplot('test_reward',xlabel='iter',ylabel='$R_{test}$',filter_window=10,display_raw=True,display_std=True,loc = (0,1))
    logger.add_lineplot('train_reward', xlabel='iter', ylabel='$R_{train}$', filter_window=10, display_raw=True,display_std=True,loc=(1,1))
    logger.add_table('Params',config)
    logger.add_status()
    logger.add_button('Preview',callback)
    logger.add_button('Close', callback)
    logger.add_checkpoint_line()

    for i in range(T):
        logger.start_iteration()
        d = np.random.randint(0,10)
        noise = np.exp(-2*i/T)*np.random.normal(0,100)
        d += noise
        logger.log(test_reward=[i,d],train_reward=[i, d+2])
        logger.draw()

        if i % 10 == 0:
            logger.update_checkpiont_line(i)
            time.sleep(0.1)

        logger.end_iteration()
        print(i)
    logger.wait_for_close(enable=True)


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


# class RLLogger(object):
#     def __init__(self, rows, cols,num_iterations=None, lw=0.5, figsize=(10, 5)):
#         self.logs = {}
#         self.sig_digs = 4
#         self.filter_widows = {}
#         self.xlabels = {}
#         self.ylabels = {}
#         self.display_raws = {}
#         self.lines = {}
#         self.filtered_lines = {}
#         self.checkpoint_lines = {}
#         self.trackers = {}
#         self.interfaces = {}
#         self.status = {}
#         self.axs = {}
#         self.iax = 0  # index counter for plot
#         self.rows = rows
#         self.cols = cols
#         self.num_iterations = num_iterations
#         self.iter_count = 0 # number of iterations in loop
#         plt.ion()
#
#         self.last_iteration_time = None
#
#         self.root_fig = plt.figure(figsize=figsize, constrained_layout=True)
#         self.subfigs = self.root_fig.subfigures(1, 2, wspace=0.00, width_ratios=[1.5, 1.])
#         self.plot_fig,self.interface_fig = self.subfigs[0].subfigures(2, 1, wspace=0.00, height_ratios=[10, 1])
#         self.subfigs =  self.subfigs[1].subfigures(2, 1, wspace=0.00, height_ratios=[10, 1])
#         self.settings_fig = self.subfigs[0]
#         self.status_fig = self.subfigs[1]
#
#         self.fig_number = self.root_fig.number
#
#         self.settings_fig.set_facecolor('lightgray')
#         self.settings_fig.suptitle('Settings')
#         self.root_fig.canvas.manager.set_window_title('RLLoger V1.1')
#
#         self.raw_settings = {'c': 'k', 'lw': lw}
#         self.filtered_settings = {'c': 'r', 'lw': lw}
#     def close_plots(self):
#         plt.close(self.root_fig)
#     def log(self, **data):
#         for key, value in data.items():
#             assert len(value) == 2, f'Value must be a tuple of length 2. Got {key}:{value}'
#             self.logs[key] = np.vstack([self.logs[key], np.array(value)])
#
#     def filter(self, x, window):
#         """ apply a window filter to the values. """
#
#         # np.convolve(x, np.ones(window), 'valid') / window
#         if len(x) > window:
#             f = []
#             for i in range(len(x)):
#                 w = min(i, window)
#                 f.append(np.mean(x[i - w:i]))
#             return f
#             # return np.convolve(x, np.ones(w) / w, 'same')
#         else:
#             return x
#
#     ###########################################################
#     # Status Methods ############################################
#     def add_status(self,key='status', iteration_timer=True,progress=True,remaining_time=True,ncols=2,fnt_size=6):
#         assert self.num_iterations is not None, 'Number of iterations must be set to use status'
#         self.status['Prog:'] = 0
#         self.status['S/iter'] = deque([time.time()], maxlen=10)
#         self.status['Time Left'] = 0
#         self.axs[key] = self.status_fig.add_subplot(1, 1, 1)
#         # self.axs[key].set_title('Status')
#         data_dict = {'Prog':0,'S/iter':0,'Time Left':0}
#         items = list(data_dict.items())
#         data = []
#         nrows = len(data_dict) // ncols
#         ncols = np.sum([iteration_timer,progress,remaining_time])
#         for r in range(nrows):
#             row = []
#             for c in range(ncols):
#                 i = r * ncols + c
#                 k, v = items[i]
#                 row.append(f'{k}: {v}')
#             data.append(row)
#         tbl = self.axs[key].table(cellText=data, fontsize=fnt_size, loc='center', cellLoc='left')
#         tbl.auto_set_font_size(False)
#
#         self.axs[key].axis('off')
#         self.status['tbl'] = tbl
#
#         for key, cell in tbl.get_celld().items():
#             cell.set_linewidth(0)
#         tbl.auto_set_column_width([0, 1])
#
#     def start_iteration(self):
#         self.last_iteration_time = time.time()
#     def end_iteration(self,advance_iteration =True):
#         dt = time.time() - self.last_iteration_time
#         self.status['S/iter'].append(dt)
#         if advance_iteration: self.iter_count += 1
#     def draw_status(self):
#         tbl = self.status['tbl']
#         s_per_iter = np.mean(self.status["S/iter"])
#         rem_time = (self.num_iterations-self.iter_count)* s_per_iter
#         # reformat rem_time into hours, minutes, seconds
#         hours = rem_time // 3600
#         rem_time = rem_time % 3600
#         minutes = rem_time // 60
#         rem_time = rem_time % 60
#         seconds = rem_time
#         rem_time = f'{int(hours)}:{int(minutes)}:{int(seconds)}'
#         tbl.get_celld()[(0, 0)].get_text().set_text(f'Progress: {np.round(100*self.iter_count/self.num_iterations,2)}%')
#         tbl.get_celld()[(0, 1)].get_text().set_text(f'S/iter: {np.round(s_per_iter,2)}')
#         tbl.get_celld()[(0, 2)].get_text().set_text(f'Time Left: {rem_time}')
#
#
#
#     ###########################################################
#     # Construction ############################################
#     def add_tracker(self, key, vals, max_iterations,
#                     xlabel='', ylabel='', title='',
#                     loc=None, ):
#         """Used to track paramters over time like exploration rate"""
#         self.iax += 1
#         self.xlabels[key] = xlabel
#         self.ylabels[key] = ylabel
#         if loc is not None:
#             self.axs[key] = self.plot_fig.add_subplot(self.rows, self.cols, self.cols * loc[0] + loc[1])
#         else:
#             self.axs[key] = self.plot_fig.add_subplot(self.rows, self.cols, self.iax)
#
#         xvals = np.linspace(0, max_iterations, len(vals))
#         self.trackers[key] = self.axs[key].plot(xvals, vals)[0]
#         self.axs[key].set_xlabel(xlabel)
#         self.axs[key].set_ylabel(ylabel)
#         self.axs[key].set_title(title)
#
#         raise NotImplementedError
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
#         tbl = self.axs[key].table(cellText=data, fontsize=fnt_size, loc='center', cellLoc='left')
#         # df = pd.DataFrame(data)
#         # df.index.name = None
#         # pd.plotting.table(self.axs[key], df, loc='center', colWidths=[0.5] * len(data_dict), cellLoc='center',
#         #                   fontsize=fnt_size)
#         tbl.auto_set_font_size(False)
#
#         self.axs[key].axis('off')
#
#     def add_checkpoint_line(self):
#         for key, data in self.logs.items():
#             self.checkpoint_lines[key] = self.axs[key].axvline(x=0, color='g', linestyle='--',lw=1)
#
#     def add_lineplot(self, key,
#                      xlabel='', ylabel='', title='',
#                      loc=None,
#                      filter_window=None, display_raw=True):
#
#         self.iax += 1
#         self.logs[key] = np.empty((0, 2))
#         self.xlabels[key] = xlabel
#         self.ylabels[key] = ylabel
#         self.filter_widows[key] = filter_window
#         self.display_raws[key] = display_raw
#         if loc is not None:
#             self.axs[key] = self.plot_fig.add_subplot(self.rows, self.cols, self.cols * loc[0] + loc[1])
#         else:
#             self.axs[key] = self.plot_fig.add_subplot(self.rows, self.cols, self.iax)
#         self.lines[key] = self.axs[key].plot([], [], **self.raw_settings)[0]
#         if filter_window is not None:
#             self.filtered_lines[key] = self.axs[key].plot([], [], **self.filtered_settings)[0]
#         self.axs[key].set_xlabel(xlabel)
#         self.axs[key].set_ylabel(ylabel)
#         self.axs[key].set_title(title)
#
#     def add_button(self,label, callback, group='buttons'):
#         bname = f'{group}_{label}'
#         # axprev = self.interface_fig.add_axes([0.7, 0.05, 0.1, 0.075])
#         n_buttons = len(self.interfaces)
#         # self.axs[bname] = self.interface_fig.add_subplot(1, n_buttons+1, n_buttons+1)
#         # self.axs[bname] = self.interface_fig.add_axes([])
#         self.axs[bname] = self.interface_fig.add_axes([0.05+0.27*n_buttons, 0.1, 0.25, 0.9])
#         self.interfaces[bname] = Button(self.axs[bname], label)
#         self.interfaces[bname].on_clicked(callback)
#
#     ###########################################################
#     # Render Utils ############################################
#     def remove_outliers(self, data, window=0.95):
#         """Remove the top and bottom 5% of the data"""
#         if len(data)>3:
#             mean = np.mean(data)
#             sd = np.std(data)
#             final_list = [x for x in data if (x > mean - 2 * sd) and (x < mean + 2 * sd)]
#             # final_list = [x for x in data if (x > mean - 2 * sd)]
#             # final_list = [x for x in final_list if (x < mean + 2 * sd)]
#             return final_list
#         else:
#             return data
#
#     def update_checkpiont_line(self,iteration):
#         try:
#             for key, data in self.checkpoint_lines.items():
#                 self.checkpoint_lines[key].set_xdata(iteration)
#         except:
#             print(f'\n\nRLLogger.update_checkpoint_line() exception...')
#     def draw(self):
#         for key, data in self.logs.items():
#             x = data[:, 0]
#             y = data[:, 1]
#
#             self.lines[key].set_data(x, y)
#             if len(x) > 1:
#                 self.axs[key].set_xlim([np.min(x), 1.1 * np.max(x)])
#
#                 if 'loss' in key.lower():
#                     _y = y
#                     # _y = self.remove_outliers(y)
#                     if np.size(_y) > 2:
#                         self.axs[key].set_ylim([0, 1.1 * np.max(_y + 1e-6)])
#                 else:
#                     self.axs[key].set_ylim([np.min(y), np.max(y) + 0.1])
#             # self.lines[key].set_data(x, y)
#             # if len(x) > 1:
#             #     self.axs[key].set_xlim([np.min(x), 1.1*np.max(x)])
#             #     # self.axs[key].set_ylim([0, np.max(y)])
#             #     self.axs[key].set_ylim([np.min(y), np.max(y)+0.1])
#             #
#             #
#             # if 'loss' in key.lower():
#             #     # self.lines[key].set_xdata(x)
#             #     _y = self.remove_outliers(y)
#             #     # self.axs[key].set_ylim([np.min(_y), np.max(_y)])
#             #     if np.size(_y) > 2:
#             #         self.axs[key].set_ylim([0, 1.1*np.max(_y+1e-6)])
#
#         for key, _ in self.filtered_lines.items():
#             data = self.logs[key]
#             x = data[:, 0]
#             y = data[:, 1]
#             y = self.filter(y, window=self.filter_widows[key])
#             # if 'loss' in key.lower():
#             #     self.axs[key].set_xlim([np.min(x), np.max(x)])
#             #     self.axs[key].set_ylim([np.min(y), np.max(y)])
#
#             self.filtered_lines[key].set_data(x, y)
#         self.plot_fig.canvas.draw()
#         self.plot_fig.canvas.flush_events()
#
#         self.draw_status()
#
#     def spin(self):
#         self.plot_fig.canvas.flush_events()
#
#     def wait_for_close(self,enable=True):
#         """Stops the program to wait for user input (i.e. save model, save plot, close, ect..)"""
#         if enable and not self.is_closed:
#             print('\nWaiting for plot to close...')
#             while plt.fignum_exists(self.fig_number):
#                 self.spin()
#                 time.sleep(0.1)
#     @property
#     def is_closed(self):
#         return not plt.fignum_exists(self.fig_number)
#
#
#     def save_fig(self,PATH):
#         self.root_fig.savefig(PATH)
