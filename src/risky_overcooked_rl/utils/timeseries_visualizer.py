import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import  Slider
import torch
from risky_overcooked_rl.utils.deep_models import SelfPlay_QRE_OSA_CPT
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
import risky_overcooked_rl.algorithms.DDQN as Algorithm
from src.risky_overcooked_py.mdp.actions import Action
from itertools import count
import pickle
import pandas as pd
import pygame
import cv2
import copy
import time
from risky_overcooked_py.visualization.state_visualizer import StateVisualizer
from risky_overcooked_py.visualization.pygame_utils import (
    MultiFramePygameImage,
    blit_on_new_surface_of_size,
    run_static_resizeable_window,
    scale_surface_by_factor,
    vstack_surfaces,
)


class TimeseriesChronograph(StateVisualizer):
    def __init__(self,layout,p_slip,rationality=10,horizon = 400):
        super().__init__()
        self.layout = layout
        self.p_slip = p_slip
        self.horizon = horizon
        self.state_history = None


        # Parse Config ---------------------------------------------------------
        config = Algorithm.get_default_config()
        config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
        config["LAYOUT"] = layout
        config["ALGORITHM"] = 'Evaluate-' + config['ALGORITHM']
        config['p_slip'] = p_slip
        config['HORIZON'] = horizon
        # config['time_cost'] = time_cost
        self.device = config['device']

        # set up env ---------------------------------------------------------
        mdp = OvercookedGridworld.from_layout_name(config['LAYOUT'])
        mdp.p_slip = config['p_slip']
        obs_shape = mdp.get_lossless_encoding_vector_shape()
        n_actions = 36
        self.env = OvercookedEnv.from_mdp(mdp, horizon=config['HORIZON'])
        self.grid = self.env.mdp.terrain_mtx
        self.alpha_range = [0.1, 1]
        # load policies ---------------------------------------------------------
        policy_fnames = {
            'Rational': f'{layout}_pslip0{int(p_slip * 10)}__rational',
            'Averse':f'{layout}_pslip0{int(p_slip * 10)}__b00_lam225_etap088_etan10_deltap061_deltan069',
            'Seeking':f'{layout}_pslip0{int(p_slip * 10)}__b00_lam044_etap10_etan088_deltap061_deltan069'
            # 'Rational': f'{layout}_pslip{f"{p_slip}".replace(".", "")}__rational',
            # 'Averse': f'{layout}_pslip{f"{p_slip}".replace(".", "")}__b-02_lam225_etap088_etan10_deltap061_deltan069',
            # 'Seeking': f'{layout}_pslip{f"{p_slip}".replace(".", "")}__b-02_lam044_etap10_etan088_deltap061_deltan069'

        }
        self.policies = {
            'Rational': SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config, policy_fnames['Rational']),
            'Averse': SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config, policy_fnames['Averse']),
            'Seeking': SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config, policy_fnames['Seeking'])
        }
        for p in self.policies.keys():
            self.policies['Rational'].rationality = rationality
            self.policies[p].model.eval()

        # Testing Params ---------------------------------------------------------
        self.human_policies = ['Seeking', 'Averse']
        self.robot_conditions = ['Oracle', 'RS-ToM', 'Rational']


    def simulate(self,ego_policy_name,partner_policy_name):
        ego_policy = self.policies[ego_policy_name]
        partner_policy = self.policies[partner_policy_name]

        state_history = []
        self.env.reset()
        state_history.append(self.env.state.deepcopy())

        for t in count():
            obs = self.env.mdp.get_lossless_encoding_vector_astensor(self.env.state, device=self.device).unsqueeze(0)

            # CHOOSE ACTIONS ---------------------------------------------------------
            _, partner_iJA, partner_pA = partner_policy.choose_joint_action(obs, epsilon=0)
            _, ego_iJA, ego_pA = ego_policy.choose_joint_action(obs, epsilon=0)

            partner_iA = partner_iJA % 6
            ego_iA = ego_iJA // 6

            action_idxs = (ego_iA, partner_iA)
            # joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(action_idxs)
            joint_action = (Action.ALL_ACTIONS[ego_iA], Action.INDEX_TO_ACTION[partner_iA])

            # STEP ---------------------------------------------------------
            next_state, reward, done, info = self.env.step(joint_action)
            state_history.append(next_state.deepcopy())

            if done:  break
        self.state_history = state_history
        return self.state_history


    def render_layout(self,state, grid):
        pygame.init()
        grid = grid or self.grid
        assert grid
        grid_surface = pygame.surface.Surface(self._unscaled_grid_pixel_size(grid))
        self._render_grid(grid_surface, grid)
        # self._render_players(grid_surface, state.players)
        # self._render_objects(grid_surface, state.objects, grid)
        return grid_surface

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

    def surface2img(self,surface):
        buffer = pygame.surfarray.array3d(surface)
        image = copy.deepcopy(buffer)
        image = np.flip(np.rot90(image, 3), 1)
        image = cv2.resize(image, (2 * 528, 2 * 464))
        return image

    def draw(self,ax,istate,npast):

        preview_states = self.state_history[istate-npast:istate]

        # plt.ioff()

        # surface = self.render_state(self.state_history[0], self.grid)

        surface = self.render_layout(preview_states[0], self.grid)
        surface = self.render_chef_trajectory(preview_states, surface)
        surface = self.render_object_trajectory(preview_states, surface)
        surface = self.render_rest(preview_states[0], surface)
        img = self.surface2img(surface)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)
        # ax.imshow(chefs)
        # plt.show()

    def preview(self,istate = 5, npast = 5):
        fig, self.ax = plt.subplots(figsize=(10,10))
        # Add sliders
        fig.subplots_adjust(bottom=0.25)

        ax_istate = fig.add_axes([0.25, 0.15, 0.65, 0.03])
        ax_npast = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self.istate_slider = Slider(
            ax=ax_istate,
            label='iState',
            valmin=5,
            valmax=self.horizon,
            valinit=istate,
            valfmt='%0.0f'
        )
        self.npast_slider = Slider(
            ax=ax_npast,
            label='nPast',
            valmin=1,
            valmax=20,
            valinit=npast,
            valfmt='%0.0f'
        )

        self.istate_slider.on_changed(self.update_slider)
        self.npast_slider.on_changed(self.update_slider)


        # Draw
        self.draw(self.ax,istate=istate,npast=npast)
        while plt.fignum_exists(fig.number):
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.1)


    def update_slider(self,val):
        self.ax.clear()
        istate = int(self.istate_slider.val)
        npast = int(self.npast_slider.val)
        self.draw( self.ax,istate=istate,npast=npast)



def main():
    # Settings
    # layout = 'risky_coordination_ring'; p_slip = 0.4
    layout = 'risky_multipath'; p_slip = 0.15

    strategies = ['Averse','Averse'] # ego, partner policy
    # strategies = ['Seeking', 'Seeking']  # ego, partner policy
    # strategies = ['Seeking', 'Averse']  # ego, partner policy
    # strategies = ['Rational','Seeking']  # ego, partner policy
    # strategies = ['Rational', 'Averse']  # ego, partner policy

    # Initialize & Simulate
    visualizer = TimeseriesChronograph(layout=layout,p_slip=p_slip)
    visualizer.simulate(*strategies)
    visualizer.preview()

if __name__ == "__main__":
    main()