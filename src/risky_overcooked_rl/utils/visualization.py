import warnings




import numpy as np
import time
from risky_overcooked_py.visualization.state_visualizer import StateVisualizer
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import pygame
import cv2
import copy
from scipy.stats import multivariate_normal
from risky_overcooked_rl.utils.state_utils import StartStateManager

from risky_overcooked_py.mdp.actions import Action,Direction
from risky_overcooked_py.visualization.pygame_utils import (
    MultiFramePygameImage,
    blit_on_new_surface_of_size,
    run_static_resizeable_window,
    scale_surface_by_factor,
    vstack_surfaces,
)
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,OvercookedState,SoupState, ObjectState




class TrajectoryVisualizer(object):
    def __init__(self,env,blocking=True):
        self.env = env
        self.blocking = blocking
        self.slider_xpad = 0.2
        self.qued_trajectory = None
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
    def _approve_callback(self,event):
        plt.close(self.fig)
        self.approved = True

    def _reject_callback(self,event):
        plt.close(self.fig)
        self.approved = False
    def preview_approve_trajectory(self,state_history, title=None):
        if self.blocking:
            self.spawn_figure()
        if title is not None:
            self.fig.suptitle(title)

        self.imgs = []
        for state in state_history:
            self.imgs.append(self.render_image(state))
        self.time_slider.on_changed(self.update_slider)

        xstart,xend = 0.8,0.99
        ystart,yend = 0.2,0.8

        plt.subplots_adjust(right=xstart)

        # Add approve button
        left = xstart
        bottom = 0.5 * (yend - ystart) + ystart + 0.05
        width = xend - xstart
        height = 0.5 * (yend - ystart) - 0.05
        rect = [left, bottom, width, height]
        approve_ax = plt.axes(rect)
        approve_button = Button(approve_ax, 'Approve', color='green', hovercolor='0.975')
        approve_button.on_clicked(self._approve_callback)

        # Add reject button
        left = xstart
        bottom = ystart
        width = xend - xstart
        height = 0.5 * (yend - ystart) - 0.05
        rect = [left, bottom, width, height]
        reject_ax = plt.axes(rect)
        reject_button = Button(reject_ax, 'Reject', color='red', hovercolor='0.975')
        reject_button.on_clicked(self._reject_callback)

        self.fig.show()
        if self.blocking:
            while plt.fignum_exists(self.fig_number):
                self.fig.canvas.flush_events()
                time.sleep(0.1)
        return self.approved

def parallel_trajectory_heatmap(traj_heatmap,ax_dict=None):
    if ax_dict is not None:
        traj_heatmap.custom_preview(ax_dict)
        for key, ax in ax_dict.items():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(key)
        traj_heatmap.preview()
        return
    traj_heatmap.preview()
    return
class TrajectoryHeatmap(object):
    def __init__(self,env,blocking=True):
        self.env = env
        self.blocking = blocking
        self.qued_trajectory = None
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


    def custom_preview(self,ax_dict):
        self.img = self.draw_backgrounds(ax_dict.values())
        masks = self.calc_masks()
        for key, mask in masks.items():
            if key in ax_dict.keys():
                self.draw_heatmap(ax_dict[key], mask)

    def preview(self,*args,show=True):
        self.spawn_figure()
        self.img = self.draw_backgrounds()
        masks = self.calc_masks()
        for key,mask in masks.items():
            self.draw_heatmap(self.ax_dict[key],mask)
        # self.draw_heatmap(self.ax_dict['soup'],masks['soup'])

        if self.blocking and show:
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

    def draw_heatmap(self,ax,mask,max_alpha = 0.6):

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


class ChronophotographicVisualization(StateVisualizer):

    @classmethod
    def from_custom_trajectory(cls, layout, trajectory,skip_n=2,init_orientation=("W","W"),
                               start_locs = False, held_objs=False,rand_pots=True

                               ):
        joint_traj = np.array(trajectory)
        p1_traj, p2_traj, prob_slips = joint_traj[:,0], joint_traj[:,1],joint_traj[:,2]
        horizon = len(p1_traj)+1
        mdp = OvercookedGridworld.from_layout_name(layout)
        env = OvercookedEnv.from_mdp(mdp, horizon=horizon)

        SSM = StartStateManager(mdp)
        # SSM.set(loc=((1,1),(1,2)))
        env.reset()

        # --------------------------------------
        dirs = {'N': Direction.NORTH,
                   'S': Direction.SOUTH,
                   'E': Direction.EAST,
                   'W': Direction.WEST,
                   'X': Action.STAY,
                   'I': Action.INTERACT}
        env.state.players[0].orientation= dirs[init_orientation[0]]
        env.state.players[1].orientation = dirs[init_orientation[1]]
        if start_locs:
            env.state.players[0].position = start_locs[0]
            env.state.players[1].position = start_locs[1]
        if held_objs:
            for i,obj in enumerate(held_objs):
                player = env.state.players[i]
                if obj == "soup":
                    soup = SoupState.get_soup(player.position, num_onions=3, num_tomatoes=0, finished=True)
                    player.set_object(soup)
                else:
                    player.set_object(ObjectState(obj, player.position))
        if rand_pots:
            env.state = SSM.random_pot_contents(env.state)
        state_history = [env.state.deepcopy()]
        for a1, a2,p in zip(p1_traj, p2_traj,prob_slips):
            joint_action = (dirs[a1], dirs[a2])
            mdp.p_slip = float(p)
            next_state, reward, done, info = env.step(joint_action)
            state_history.append(next_state.deepcopy())

        return cls(layout, state_history[skip_n:], HORIZON=horizon)

    def __init__(self,layout, state_history, HORIZON=200):
        """
        :param state_history: list of OvercookedStates
        """
        super().__init__()
        self.state_history = state_history
        self.layout = layout
        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=HORIZON)
        self.grid = self.env.mdp.terrain_mtx
        self.alpha_range = [0.1,1]
        # self.env.reset()
        # self.visualizer = StateVisualizer()


    def render_layout(self,state, grid):
        pygame.init()
        grid = grid or self.grid
        assert grid
        grid_surface = pygame.surface.Surface(
            self._unscaled_grid_pixel_size(grid)
        )
        self._render_grid(grid_surface, grid)
        # self._render_players(grid_surface, state.players)
        # self._render_objects(grid_surface, state.objects, grid)

        return grid_surface
    def render_rest(self,state, grid_surface,hud_data=None, action_probs=None):

        if self.scale_by_factor != 1:
            grid_surface = scale_surface_by_factor(
                grid_surface, self.scale_by_factor
            )

        # render text after rescaling as text looks bad when is rendered small resolution and then rescalled to bigger one
        if self.is_rendering_cooking_timer:
            self._render_cooking_timers(grid_surface, state.objects, self.grid)

        # arrows does not seem good when rendered in very small resolution
        if self.is_rendering_action_probs and action_probs is not None:
            self._render_actions_probs(
                grid_surface, state.players, action_probs
            )

        if self.is_rendering_hud and hud_data:
            hud_width = self.width or grid_surface.get_width()
            hud_surface = pygame.surface.Surface(
                (hud_width, self._calculate_hud_height(hud_data))
            )
            hud_surface.fill(self.background_color)
            self._render_hud_data(hud_surface, hud_data)
            rendered_surface = vstack_surfaces(
                [hud_surface, grid_surface], self.background_color
            )
        else:
            hud_width = None
            rendered_surface = grid_surface

        result_surface_size = (
            self.width or rendered_surface.get_width(),
            self.height or rendered_surface.get_height(),
        )

        if result_surface_size != rendered_surface.get_size():
            result_surface = blit_on_new_surface_of_size(
                rendered_surface,
                result_surface_size,
                background_color=self.background_color,
            )
        else:
            result_surface = rendered_surface

        return result_surface

    def render_chef_trajectory(self,  state_history, grid_surface):
        alphas = np.linspace(self.alpha_range[0], self.alpha_range[1], len(state_history))
        for alpha, state in zip(alphas,state_history):
            self._render_players(grid_surface, state.players, alpha=alpha)
        return grid_surface

    def render_object_trajectory(self,  state_history, grid_surface):
        alphas = np.linspace(self.alpha_range[0], self.alpha_range[1], len(state_history))
        for alpha, state in zip(alphas,state_history):
            self._render_objects(grid_surface, state.objects, self.grid, alpha=alpha)
        return grid_surface

    def surface2img(self,surface):
        buffer = pygame.surfarray.array3d(surface)
        image = copy.deepcopy(buffer)
        image = np.flip(np.rot90(image, 3), 1)
        image = cv2.resize(image, (2 * 528, 2 * 464))
        return image
    def preview(self):
        plt.ioff()
        fig,ax = plt.subplots()
        # surface = self.render_state(self.state_history[0], self.grid)

        surface = self.render_layout(self.state_history[0], self.grid)
        surface = self.render_chef_trajectory( self.state_history,surface)
        surface = self.render_object_trajectory(self.state_history, surface)
        surface = self.render_rest(self.state_history[0], surface)
        img = self.surface2img(surface)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)
        # ax.imshow(chefs)
        plt.show()


if __name__ == '__main__':
    pass
