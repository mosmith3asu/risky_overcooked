import warnings

import numpy as np
import time

from collections import deque
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Button
import matplotlib.patches as patches


"""
More widgets at https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20List.html

"""

class Item:
    def __init__(self,id,ax,hide_axes=False):
        self.id = id
        self.ax = ax
        if hide_axes:
            self.hide_axes()

    def hide_axes(self):
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.axis('off')

    def draw(self):
        """ Draw item"""
        pass

    def update(self):
        NotImplementedError(f"Update method not implemented for this item {self.id}")


class Checkpoint:
    def __init__(self,id, axs, checkpoint_item, callback=None,**kwargs):
        self.id = id
        self.axs = axs
        self.callback = callback
        self.cp_val = None
        self.cp_iter = None
        self.cp_init_val = None
        self.cp_item = checkpoint_item

        self.style = {
            'color': kwargs.get('color', 'g'),
            'linestyle': kwargs.get('linestyle', '--')
        }

        self.cp_lines = [None for _ in range(len(axs))]

    def draw(self,enable=False):
        if enable:  self.check()

    def check(self):
        """ Check if checkpoint is reached"""
        _val = self.cp_item.recent_val
        if _val is not None:

            if self.cp_val is None:
                self.cp_iter, self.cp_val = _val
                self.cp_init_val = self.cp_val

            elif  _val[1] >= self.cp_val and _val[1] > self.cp_init_val:
                self.cp_iter, self.cp_val = _val
                self.update(self.cp_iter)
                if self.callback is not None:
                    self.callback()

    def update(self,it):
        """ Update checkpoint line on designated axes"""
        for i,ax in enumerate(self.axs):
            if self.cp_lines[i] is not None:
                try:
                    self.cp_lines[i].set_xdata(it)
                except:
                    self.cp_lines[i].set_xdata([it])
            else:
                self.cp_lines[i] = ax.axvline(it, **self.style)

class ButtonItem(Item):
    def __init__(self, id, ax, callback,
                 txt_color='white',color=(0.2,0.2,0.2),hovercolor='blue',
                 **kwargs):
        """ Button item for RL logger"""
        super().__init__(id, ax)
        label = kwargs.pop('label', id)
        self.button = Button(self.ax, label, color=color,hovercolor=hovercolor,  **kwargs)

        self.callback = callback
        if callback is not None: self.button.on_clicked(self.callback)
        self.button.label.set_color(txt_color)


class ToggleButtonItem(ButtonItem):
    def __init__(self, id, ax, callback,
                 txt_color='white',color=(0.2,0.2,0.2),
                 **kwargs):
        """ Toggle button item for RL logger"""

        self.on_txt_color = kwargs.pop('on_txt_color', txt_color)
        self.on_color = kwargs.pop('on_txt_color', 'grey')
        self.off_color = color
        self.off_txt_color = txt_color
        self.state = kwargs.pop('state', False)

        kwargs['color'] = self.on_color if self.state else self.off_color
        kwargs['txt_color'] = self.on_txt_color if self.state else self.off_txt_color

        super().__init__(id, ax, callback=None, **kwargs)
        # self.state = kwargs.get('state',False)
        self.callback = callback
        self.button.on_clicked(self.on_click)


    def on_click(self,event):
        self.toggle()
        self.callback(self.id,self.state)

    def toggle(self,change_state=True):
        if change_state:
            self.state = not self.state
        if self.state:
            self.button.label.set_color(self.on_txt_color)
            self.button.color = self.on_color
            self.button.hovercolor = self.off_color
        else:
            self.button.label.set_color(self.off_txt_color)
            self.button.color = self.off_color
            self.button.hovercolor = self.on_color


class SettingItem(Item):
    def __init__(self,id,ax,data_dict,ncols = 1,**kwargs):
        super().__init__(id, ax,hide_axes=True)
        self.data_dict = data_dict
        self.fnt_size = kwargs.get('fontsize', 9)

        data = []
        items = list(data_dict.items())
        nrows = len(data_dict) // ncols
        for r in range(nrows):
            row = []
            for c in range(ncols):
                i = r * ncols + c
                k, v = items[i]
                row.append(f'{k}: {v}')
            data.append(row)
        self.tbl = self.ax.table(cellText=data,fontsize=self.fnt_size, loc='center',
                                 cellLoc='left',bbox=[0,0,1,1])
        self.tbl.auto_set_font_size(False)
        self.tbl.set_fontsize(self.fnt_size)
    def draw(self):
        self.tbl.scale(1,1)


class StatusItems(Item):

    def __init__(self,id,ax,font_sz=12,**kwargs):
        super().__init__(id, ax, hide_axes=True)
        self.id = id
        self.ax = ax
        self.font_sz = font_sz
        self.def_val = '-'
        self.cellColour = kwargs.get('cellColour', 'k')
        self.txtColour = kwargs.get('txtColour', 'w')

        self.data = [[]]
        self.key_locs = {}
        self.sig_digs = {}
        self.value_callbacks = {}

        self.tbl = None

    def add_item(self, key, callback=None, **kwargs):
        self.sig_digs[key]= kwargs.get('sig_dig', 2)
        txt = f'{key}: {self.def_val}'
        loc = (0, len(self.data[0]))
        self.data[0].append(txt)
        self.key_locs[key] = loc

        if callback is not None:
            self.value_callbacks[key] = callback



        if self.tbl is None:
            self.tbl = self.ax.table(cellText=self.data, fontsize=self.font_sz,
                                     loc='center', cellLoc='left',cellColours=[self.cellColour])
        else:
            height = self.tbl.get_celld()[(0, 0)].get_height()
            self.tbl.add_cell(loc[0],loc[1], text=txt,height=height,width=1,facecolor=self.cellColour)

        for key, cell in self.tbl.get_celld().items():
            cell.set_linewidth(0)

        # self.tbl.auto_set_row_height([i for i in range(len(self.data))])
        self.tbl.auto_set_column_width([i for i in range(len(self.data[0]))])
        self.tbl.auto_set_font_size(True)
    def update_item(self,**kwargs):
        for key, callback in self.value_callbacks.items():
            val = callback() if callable(callback) else callback
            loc = self.key_locs[key]
            if isinstance(val, int) or isinstance(val, float):
                val = round(val, self.sig_digs[key])
            elif isinstance(val, np.ndarray):
                val = np.round(val, self.sig_digs[key])
            self.tbl.get_celld()[loc].get_text().set_text(f'{key}: {val}')
            self.tbl.get_celld()[loc].get_text().set_color(self.txtColour)


        for key,val in kwargs.items():
            loc = self.key_locs[key]
            if isinstance(val, int) or isinstance(val, float):
                val = round(val, self.sig_digs[key])
            elif isinstance(val, np.ndarray):
                val = np.round(val, self.sig_digs[key])
            self.tbl.get_celld()[loc].get_text().set_text(f'{key}: {val}')
            self.tbl.get_celld()[loc].get_text().set_color(self.txtColour)
        self.tbl.auto_set_column_width([i for i in range(len(self.data))])
        self.tbl.auto_set_font_size(False)
        self.tbl.scale(1, 1)


    @property
    def keys(self):
        return list(self.key_locs.keys())

class LinePlotItem:
    def __init__(self, id, ax,title='', xlabel='', ylabel='',filter_window=10,xtick=True,**line_kwargs):
        self.id = id
        self.std_settings = {'color':'b','alpha':0.2}
        self.line_settings = {'color':'k', 'linestyle':'-','linewidth':1}
        for key,val in line_kwargs.items(): self.line_settings[key] = val

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

    @property
    def recent_val(self):
        return self.filtered_data[-1, :] if self.filtered_data.shape[0] > 0 else None

import multiprocessing as mp
class CustomCallbackItem:
    def __init__(self, id, ax, callback=None, **kwargs):
        self.id = id
        self.ax = ax
        self.callback = callback
        self.update_every_draw = kwargs.get('update_every_draw', False)
        self.update_first_draw = kwargs.get('first_draw', True)
        self.first_draw = False

    def draw(self):
        if self.update_every_draw:
            # print(f'Updating on draw')
            self.update()
        elif self.update_first_draw and not self.first_draw:
            # print(f'Updating on first draw')
            self.update()
            self.first_draw = True


    def update(self):
        # mp.Process(target=self.callback, args=(self.id, self.ax)).start()
        self.callback(self.id, self.ax)
        self.ax.autoscale_view()



class IterationTimer:
    def __init__(self,num_iters=None, buffer_length=10,sig_dig=1):
        self.iteration_data = deque(maxlen=buffer_length)
        self.start_time = time.time()
        self.iter_count = 0 # used for remaining time calc
        self.num_iters = num_iters
        self.sig_dig = sig_dig
        self.per_iteration_key = '$s/it$'
        self.remaining_key = '%Hr'
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

class AnnotationItem:
    def __init__(self,ax,text_or_callback,xy=(0.01,0.99),xytext=(0,0),**kwargs):
        self.text_or_callback = text_or_callback
        self.xy = xy
        self.xytext = xytext
        self.kwargs = kwargs
        self.ax = ax
        self.annotation = None
        self.va = kwargs.pop('va', 'top')
        self.ha = kwargs.pop('va', 'left')


    def draw(self):
        if not isinstance(self.text_or_callback, str):
            txt = self.text_or_callback() if callable(self.text_or_callback) else str(self.text_or_callback)
        else:
            txt = self.text_or_callback

        # bbox = self.ax.get_position()
        # x0,y1 = (bbox.xmin, bbox.ymin)
        # xy = self.ax.transAxes.transform(self.xy)
        # self.xy = (0,1)
        xy = self.xy
        if self.annotation is None:


            self.annotation = self.ax.annotate(txt, xy=xy, xycoords='axes fraction',
                                               ha=self.ha, va = self.va, **self.kwargs)
        else:
            self.annotation.set_text(txt)
            self.annotation.set_position(xy)
            # self.annotation.set_xytext(self.xytext)


class RLLogger_V2:
    def __init__(self, num_iters = 0 , figsize=(10, 5),**kwargs):
        plt.ion()

        self.root_fig = plt.figure(figsize=figsize)
        self.root_fig.canvas.manager.set_window_title('RLLoger V2.1')
        self.fig_number = self.root_fig.number
        self.with_heatmap = kwargs.get('with_heatmap', False)  # if True, will add a heatmap figure


        vratios = kwargs.get('vratios', [1, 10])  # vertical ratios for top and bottom group
        top_hratios = kwargs.get('top_hratios', [1, 1])  # vertical ratios for status and settings group


        ######## DIVIDE FIGURES INTO GROUPS ##################################
        self.top_group_fig, self.bottom_group_fig = self.root_fig.subfigures(2, 1, height_ratios=vratios)
        self.interface_fig, self.status_fig = self.top_group_fig.subfigures(1, 2, width_ratios=top_hratios)
        #
        if self.with_heatmap:
            bottom_hratios = kwargs.get('bottom_hratios', [1.3,0.5, 1])  # vertical ratios for status and settings group
            self.lineplot_fig, self.custom_fig, self.settings_fig = self.bottom_group_fig.subfigures(1, 3,
                                                                                                     width_ratios=bottom_hratios)
        else:
            bottom_hratios = kwargs.get('bottom_hratios', [1.3, 1])  # vertical ratios for status and settings group
            self.lineplot_fig, self.settings_fig = self.bottom_group_fig.subfigures(1, 2, width_ratios=bottom_hratios)

        # self.left_group_fig,self.right_group_fig = self.root_fig.subfigures(1, 2, width_ratios=[1.5, 1.])
        # self.interface_fig, self.lineplot_fig = self.left_group_fig.subfigures(2, 1, height_ratios=[1, 10])
        # self.status_fig, self.settings_fig = self.right_group_fig.subfigures(2, 1, height_ratios=[1, 10])

        # Plot spacing configs
        self._lineplot_fig_adjust = {'hspace':0.05, 'wspace':0.05,'top':0.99,'bottom':0.1,'right':0.99,'left':0.13}
        self._settings_fig_adjust = {'hspace':0.05, 'wspace':0.05,'top':0.99,'bottom':0.01,'right':0.99,'left':0.01}
        self._interface_fig_adjust = {'hspace': 0.05, 'wspace': 0.1, 'top': 0.9, 'bottom': 0.1, 'right': 0.99, 'left': 0.01}
        self._status_fig_adjust = {'hspace': 0.05, 'wspace': 0.05, 'top': 0.99, 'bottom': 0.01, 'right': 0.99, 'left': 0.01}
        self._custom_fig_adjust = {'hspace':0.05, 'wspace':0.05,'top':0.99,'bottom':0.01,'right':0.99,'left':0.01}
        self.status_fig.set_facecolor('k')
        self.interface_fig.set_facecolor('k')
        # self.all_figs = [self.root_fig,self.left_group_fig,self.right_group_fig,self.lineplot_fig,self.interface_fig,self.settings_fig,self.status_fig]
        self.all_figs = [self.root_fig,self.top_group_fig,self.bottom_group_fig,self.lineplot_fig,
                         self.interface_fig,self.settings_fig,self.status_fig]
        if self.with_heatmap:
            self.all_figs.append(self.custom_fig)

        self.dynamic_figs = [self.root_fig,self.lineplot_fig,self.status_fig]

        ######### DATA ###############################
        self.ids = []
        self.lineplots = []
        self.interfaces = []
        self.items = {}
        self.status = None
        self.settings = None
        self.checkpoint = None
        self.is_drawn = False
        self.wait_for_close = kwargs.get('wait_for_close', False)
        self.iter_timer = IterationTimer(num_iters=num_iters,buffer_length=10)
        self.add_status(self.iter_timer.per_iteration_key,self.iter_timer.remaining_key) # always add iteration timer to status
        self._enable_checkpointing = False

    ###############################################
    ####### INITIALIZATION METHODS ################
    def _check_id(self,id):
        if id in self.ids:
            raise ValueError(f"ID {id} already exists. Please use a different ID.")

    def _append_ax(self,fig,loc):
        """Change geometry of existing ax"""
        # First Axis in Figure
        if len(fig.axes) == 0:
            return fig.add_subplot(111)

        # Dynamically append axes
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

    # Lineplot Instantiates -------------------
    def add_lineplot(self,id,**kwargs):
        self._check_id(id)
        ax = self._append_ax(self.lineplot_fig, loc='bottom')
        this_item = LinePlotItem(id,ax,**kwargs)
        self.lineplots.append(this_item)
        self.ids.append(id)
        self.items[id] = this_item

    def add_checkpoint_watcher(self,id,draw_on,callback=None):
        """ Can add a callback function to be called when the checkpoint is reached (e.g. save checkpoint)"""
        assert id in self.items.keys(), 'Checkpoint id not found in RL_logger.watch_checkpoint_on(id)'
        axs = [self.items[name].ax for name in draw_on]
        self.checkpoint = Checkpoint(id='checkpointer',checkpoint_item=self.items[id],
                                     axs=axs,callback=callback)

    # Interface Instantiates -------------------
    def add_button(self,id,callback,**kwargs):
        self._check_id(id)
        ax = self._append_ax(self.interface_fig, loc='right')
        this_item = ButtonItem(id,ax , callback=callback,**kwargs)
        self.interfaces.append(this_item)
        self.ids.append(id)
        self.items[id] = this_item

    def add_toggle_button(self,id,callback=None,**kwargs):
        if callback is None:
            assert id in self.__dict__.keys(), f"Checkbox requires callback unless setting attribute internal to RLLogger." \
                                               f"Attribute [RLLoger.{id}] not found, use RL_logger.add_checkbox(id,callback)"
            callback = self.set_val
            kwargs['state'] = self.__dict__[id]
            # self._check_id(id)
        ax = self._append_ax(self.interface_fig, loc='right')
        this_item = ToggleButtonItem(id, ax, callback, **kwargs)
        self.interfaces.append(this_item)
        self.ids.append(id)
        self.items[id] = this_item

    def add_checkbox(self,id,callback=None,**kwargs):
        raise NotImplementedError
        # if callback is None:
        #     assert id in self.__dict__.keys(), f"Checkbox requires callback unless setting attribute internal to RLLogger." \
        #                                        f"Attribute [RLLoger.{id}] not found, use RL_logger.add_checkbox(id,callback)"
        #     callback = self.set_val
        #     kwargs['checked'] = self.__dict__[id]
        # # self._check_id(id)
        # ax = self._append_ax(self.interface_fig, loc='right')
        # this_item = CheckboxItem(id,ax,callback,**kwargs)
        # self.interfaces.append(this_item)
        # self.ids.append(id)
        # self.items[id] = this_item

    def set_val(self,key,val):
        self.__dict__[key] = val

    # Status Instantiates -------------------
    def add_status(self,*args,**kwargs):
        callback = kwargs.get('callback', None)
        if callback is not None:
            assert len(args)== 1, f"Status callback requires a single id argument. {len(args)} were given."
        for id in args:
            self._check_id(id)
            if self.status is None:
                ax = self.status_fig.add_subplot(111)
                self.status = StatusItems(id, ax, **kwargs)
                self.items[id] = self.status


            self.status.add_item(id,**kwargs)
            self.ids.append(id)

    # Settings Instantiates -------------------
    def add_settings(self,data_dict,id='settings',**kwargs):
        self._check_id(id)
        self.settings = SettingItem(id=id, ax=self.settings_fig.add_subplot(111),
                                    data_dict=data_dict,**kwargs)
        self.ids.append(id)
        self.items[id] = self.settings

    def add_annotation(self, id, text_or_callback,**kwargs):
        assert id in self.ids, f"Annotation id {id} not found in RLLogger.add_annotation(id, text_or_callback)"
        ax = self.items[id].ax
        this_item = AnnotationItem(ax,text_or_callback,**kwargs)
        self.items[id + "_annotation"] = this_item

    def add_callback_item(self, id, callback, **kwargs):
        """ Add a custom callback item that can be used to update the figure"""
        self._check_id(id)
        ax = self._append_ax(self.custom_fig, loc='bottom')
        this_item = CustomCallbackItem(id, ax, callback=callback, **kwargs)
        self.items[id] = this_item
        self.ids.append(id)
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

    def draw(self):
        """ Updates drawing of linplots """
        if not self.is_drawn:
            self.lineplot_fig.subplots_adjust(**self._lineplot_fig_adjust)
            self.settings_fig.subplots_adjust(**self._settings_fig_adjust)
            self.interface_fig.subplots_adjust(**self._interface_fig_adjust)
            self.status_fig.subplots_adjust(**self._status_fig_adjust)
            self.status_fig.subplots_adjust(**self._custom_fig_adjust)
        self.is_drawn = True

        for lineplot in self.lineplots:
            lineplot.draw()
        self.status.draw()
        if self.checkpoint is not None: self.checkpoint.draw(enable=self._enable_checkpointing)
        self.settings.draw()
        for item in self.items.values():
            item.draw()
        self.root_fig.canvas.draw()
        self.spin()

    def spin(self):
        """ Call this periodically to unfreeze plot window"""
        self.root_fig.canvas.flush_events()

    def halt(self,**kwargs):
        """Stops the program to wait for user input (i.e. save model, save plot, close, ect..)"""
        if self.wait_for_close and not self.is_closed:
            print('\nWaiting for plot to close...')
            while plt.fignum_exists(self.fig_number):
                self.spin()
                time.sleep(0.1)

    def save_fig(self, PATH):
        self.root_fig.savefig(PATH)

    def loop_iteration(self):
        """ Put this at beginning of iteration loop for timing"""
        self.iter_timer.tic()

    def close_plots(self):
        plt.close(self.root_fig)

    ###############################################
    ############# STATUS METHODS ##################
    def enable_checkpointing(self,val):
        assert isinstance(val,bool), f"Checkpointing must be a boolean value. {val} is not a boolean."
        self._enable_checkpointing = val
        return self._enable_checkpointing


    @property
    def is_closed(self):
        return not plt.fignum_exists(self.fig_number)

    @property
    def checkpoint_val(self):
         return None

def example_usage():
    def button_callback(event):
        print("Button clicked!")
    def checkpoint_callback(*args):
        print("Checkpoing Callback!")
    def status_callback():
        return np.random.rand()

    def annotation_callback():
        return np.random.rand()

    # Example usage
    N = 100
    settings = {'a': 1, 'b': 2, 'c': 3}

    logger = RLLogger_V2(num_iters=N, wait_for_close=True)
    logger.add_lineplot('test_reward', xlabel='', ylabel='$R_{test}$', filter_window=10,xtick=False)
    logger.add_lineplot('train_reward', xlabel='iter', ylabel='$R_{train}$', filter_window=1)
    logger.add_annotation('test_reward',annotation_callback)
    logger.add_status('eps')
    logger.add_status('call', callback=status_callback)
    logger.add_settings(settings)
    logger.add_button('Preview', button_callback)
    logger.add_button('Heatmap', button_callback)
    logger.add_button('Save', button_callback)
    logger.add_toggle_button('wait_for_close', label='Wait For Close')
    logger.add_checkpoint_watcher('test_reward',draw_on=['test_reward','train_reward'],callback=checkpoint_callback)
    # logger.add_status('train_reward', xlabel='iter', ylabel='$R_{train}$', filter_window=1)
    logger.enable_checkpointing(True)
    for i in range(N):
        # print(i)
        logger.loop_iteration() # begin every iteration with this command

        logger.log(test_reward = [i, np.sin(i/10)+np.random.rand()/4+i/100],
                   train_reward = [i, np.sin(i/10)+np.random.rand()/4+i/100],
                   eps = i+0.555555
                   ) # logg lineplot and status variables
        logger.draw()
        time.sleep(0.01)


    logger.halt()
    plt.show()

if __name__=="__main__":
    example_usage()



###################################################################
# BROKEN ##########################################################
###################################################################
# class CheckboxItem(Item):
#     """
#     TODO: Known text overflow error with longer labels
#     """
#     def __init__(self,id,ax,callback,**kwargs):
#         super().__init__(id,ax,hide_axes=True)
#         self.label = kwargs.get('label', id)
#         self.color = kwargs.get('color', 'white')
#         self.checked = kwargs.get('checked', False)
#         self.padding = kwargs.get('padding', 0.05)
#         self.checkbox_size = kwargs.get('checkbox_size', 0.4)
#         self.checkbox_patch = None
#         self.checkmark_line = None
#         self.callback = callback
#
#         self.checkbox_style = {
#             'linewidth': kwargs.get('linewidth', 1),
#             'edgecolor': kwargs.get('edgecolor', 'black'),
#             'facecolor': kwargs.get('facecolor', 'white')
#         }
#         self.label_style = {
#             'color': kwargs.get('labelcolor', 'k'),
#             'fontsize': kwargs.get('fontsize', 10),
#             'va': kwargs.get('va', 'center'),
#             'ha': kwargs.get('ha', 'left'),
#             'clip_on': kwargs.get('clip_on', False)
#         }
#
#         self.draw_checkbox()
#         self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click) #connect to callback
#
#     def draw_checkbox(self):
#         self.ax.set_xticks([])
#         self.ax.set_yticks([])
#         self.ax.set_xlim(0, 1)
#         self.ax.set_ylim(0, 1)
#         self.ax.set_aspect('equal')
#         self.ax.axis('off')
#
#         # Checkbox in Axes coordinates
#         box_size = 0.5
#         self.checkbox_x = self.padding
#         self.checkbox_y = 0.5 - box_size / 2
#         self.checkbox_box_size = box_size
#
#         # Draw checkbox
#         self.checkbox_patch = patches.Rectangle(
#             (self.checkbox_x, self.checkbox_y),
#             box_size, box_size,
#             linewidth=1,
#             edgecolor='black',
#             facecolor='white',
#             transform=self.ax.transAxes,
#             zorder=2
#         )
#         self.ax.add_patch(self.checkbox_patch)
#
#         # Draw label — allow it to extend outside axes
#         label = self.ax.text(self.checkbox_x + box_size + self.padding, 0.5, self.label,
#                      transform=self.ax.transAxes, **self.label_style)
#
#         if self.checked:
#             self.draw_checkmark()
#
#         self.ax.figure.canvas.draw()
#
#     def draw_checkmark(self):
#         x0 = self.checkbox_x
#         y0 = self.checkbox_y
#         s = self.checkbox_box_size
#         # Define checkmark line
#         self.checkmark_line, = self.ax.plot(
#             [x0 + 0.2 * s, x0 + 0.4 * s, x0 + 0.8 * s],
#             [y0 + 0.5 * s, y0 + 0.1 * s, y0 + 0.9 * s],
#             color='black', linewidth=2
#         )
#
#     def remove_checkmark(self):
#         if self.checkmark_line:
#             self.checkmark_line.remove()
#             self.checkmark_line = None
#
#     def toggle(self):
#         self.checked = not self.checked
#         if self.checked:
#             self.draw_checkmark()
#         else:
#             self.remove_checkmark()
#         self.ax.figure.canvas.draw()
#         if self.callback is not None:
#             self.callback(self.id,self.checked)
#
#     def on_click(self, event):
#         if event.inaxes != self.ax:
#             return
#
#         # Check if click is within checkbox
#         if (self.checkbox_x <= event.xdata <= self.checkbox_x + self.checkbox_box_size and
#                 self.checkbox_y <= event.ydata <= self.checkbox_y + self.checkbox_box_size):
#             self.toggle()
