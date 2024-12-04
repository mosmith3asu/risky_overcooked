
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from risky_overcooked_py.mdp.actions import Action,Direction
from risky_overcooked_py.visualization.state_visualizer import StateVisualizer
import matplotlib.pyplot as plt
import pygame
import cv2
import copy
from risky_overcooked_py.visualization.pygame_utils import (
    MultiFramePygameImage,
    blit_on_new_surface_of_size,
    run_static_resizeable_window,
    scale_surface_by_factor,
    vstack_surfaces,
)
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,OvercookedState,SoupState, ObjectState

import os
import sys
print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from risky_overcooked_rl.utils.model_manager import get_default_config, parse_args #get_argparser
from risky_overcooked_rl.utils.trainer import Trainer
from risky_overcooked_rl.utils.deep_models import SelfPlay_QRE_OSA_CPT


class ChronophotographicVisualization(StateVisualizer):

    @classmethod
    def from_custom_trajectory(cls, layout, trajectory,skip_n=2,init_orientation="W"):
        joint_traj = np.array(trajectory)
        p1_traj, p2_traj, prob_slips = joint_traj[:,0], joint_traj[:,1],joint_traj[:,2]
        horizon = len(p1_traj)+1
        mdp = OvercookedGridworld.from_layout_name(layout)
        env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
        env.reset()

        # --------------------------------------
        dirs = {'N': Direction.NORTH,
                   'S': Direction.SOUTH,
                   'E': Direction.EAST,
                   'W': Direction.WEST,
                   'X': Action.STAY,
                   'I': Action.INTERACT}
        env.state.players[0].orientation= dirs[init_orientation]
        env.state.players[1].orientation = dirs[init_orientation]
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

def main():
    # N_steps = 5
    # config = get_default_config()
    # config['loads'] = 'risky_coordination_ring_pslip04__rational__10_09_2024-13_44'
    # config['p_slip'] = 0.4
    # config = parse_args(config)
    # config["ALGORITHM"] = 'Evaluate-' + config['ALGORITHM']
    # trainer = Trainer(SelfPlay_QRE_OSA_CPT, config)
    #
    # _, _, state_history, _, _, _ = trainer.test_rollout(rationality=10, get_info=True)
    #
    # ChronophotographicVisualization('risky_coordination_ring', state_history[:N_steps]).preview()

    LAYOUT = 'forced_coordination'

    S, W, N, E, X, I = 'S', 'W', 'N', 'E', 'X', 'I'
    seeking_joint_traj = [
        [X, S, 0],
        [X, I, 0],  # P2 PICK ONION
        [W, E, 0],
        [W, E, 0],
        [S, N, 0],  # P1 PICK ONION
    ]
    averse_joint_traj = [
        [X, W, 1],
        [X, S, 1],  # P2 PICK UP ONION
        [X, I, 1],
        # START
        [W, N, 1],
        [S, E, 1],  # P2 SETS DOWN ONION
        [I, I, 1],  # P1 PICK UP ONION
        [E, X, 1],  # P2 PICK UP ONION
        [I, X, 1],  # P1 DELIVERS ONION 1
        [S, X, 1],
        # [X, X, 1],
        # [X, X, 1],
        # [X, X, 1],
        # [X, X, 1],
        # [W, E, 1],
        # [S, I, 1],  # P2 SETS DOWN ONION
        # [I, S, 1],  # P1 PICK UP ONION
        # [E, I, 1],  # P2 PICK UP ONION
        # [N, N, 1],
        # [I, E, 1],  # P1 DELIVERS ONION 2
        # [S, I, 1],  # P2 SETS DOWN ONION
        # [W, S, 1],
        # [I, I, 1],  # P1 PICK UP ONION | P2 PICK UP ONION
        # [N, N, 1],
        # [I, E, 1],  # P1 DELIVERS ONION 3
        # [S, I, 1],  # P2 SETS DOWN ONION
        # [W, S, 1],
        # [I, W, 1],  # P1 PICK UP ONION
        # [N, I, 1],  # P2 PICK UP ONION
        # [E, N, 1],
        # [I, E, 1],  # P1 DELIVERS ONION 1
        # [W, I, 1],  # P2 SETS DOWN ONION
        # [S, S, 1],
        # [I, I, 1],  # P1 PICK UP ONION | P2 PICK UP ONION
        # [E, E, 1],
        # [I, N, 1],  # P1 DELIVERS ONION 2
        # [S, I, 1],  # P2 SETS DOWN ONION
        # [W, W, 1],
        # [I, N, 1],  # P1 PICK UP ONION
        # [N, W, 1],
        # [E, I, 1],  # P2 PICK UP DISH
        # [I, E, 1],  # P1 DELIVERS ONION 3
        # [S, I, 1],  # P2 SET DOWN DISH
        # [W, S, 1],
        # [I, I, 1],  # P1 PICK UP DISH
        # [N, N, 1],
        # [I, N, 1],  # P1 PICK UP SOUP | P2 DROP ONION
        # [W, S, 0],
        # [S, S, 0],
        # [I, E, 0],  # P1 SET DOWN SOUP
        # [W, N, 0],
        # [S, I, 0],  # P2 PICK UP SOUP
        # [W, S, 0],
        # [I, I, 0],  # P2 DELIVER SOUP | P1 PICK UP DISH
        # [N, W, 0],
        # [E, I, 0],
        # [E, E, 0],
        # [X, E, 1],  # P2 DROP ONION
        # [X, W, 1],
        # [X, W, 1],
        # [X, I, 1],
        # [I, E, 1],  # P1 PICK UP SOUP
        # [S, E, 1],  # P2 DROP ONION
        # [W, W, 0],
        # [I, N, 0],  # P1 SET DOWN SOUP
        # [N, I, 0],
        # [W, S, 0],
        # [E, I, 0],  # P2 DELIVER SOUP
        # [N, W, 0],  # P2 DELIVER SOUP
    ]
    # ChronophotographicVisualization.from_custom_trajectory('risky_coordination_ring_EXAMPLE', seeking_joint_traj).preview()
    # ChronophotographicVisualization.from_custom_trajectory('risky_coordination_ring_EXAMPLE', averse_joint_traj).preview()
    #[blue,green]
    seeking_joint_traj = [
        [W, X, 0],
        [S, X, 0],
        [W, X, 0],
        [W, X, 0],
        [S, X, 0],
        [I, X, 0],
        [E, X, 0],
        [E, W, 0],
        [N, W, 0],
        [E, W, 0],
        [S, W, 0],
        [E, S, 0],
        [X, I, 0],

        [X, E, 0],
        [X, E, 0],
        [X, W, 0],
        # START
        [N, W, 0],
        [N, N, 0],
        [N, N, 0],
        [N, N, 0],
        [W, N, 0],
    ]
    # averse_joint_traj = [
    #     [W, X, 0],
    #     [S, X, 0],
    #     [W, X, 0],
    #     [W, X, 0],
    #     [S, X, 0],
    #     [I, X, 0],
    #     [E, X, 0],
    #     [E, W, 0],
    #     [N, W, 0],
    #     [E, W, 0],
    #     [S, W, 0],
    #     [E, S, 0],
    #     [X, I, 0],
    #
    #     [X, E, 0],
    #     [X, E, 0],
    #     [X, X, 0],
    #     # START
    #     [N, N, 0],
    #     [N, N, 0],
    #     [N, N, 0],
    #     [N, W, 0],
    #     [W, W, 0],
    # ]
    averse_joint_traj = [
        [W, X, 0],
        [S, X, 0],
        [W, X, 0],
        [W, X, 0],
        [S, X, 0],
        [I, X, 0],
        [E, X, 0],
        [E, W, 0],
        [N, W, 0],
        [E, W, 0],
        [S, W, 0],
        [E, S, 0],
        [X, I, 0],

        [X, X, 0],
        [N, X, 0],
        [N, X, 0],
        # START
        [N, E, 0],
        [N, E, 0],
        [W, E, 0],
        [W, E, 0],
        [W, N, 0],
        [W, N, 0],
    ]
    ChronophotographicVisualization.from_custom_trajectory('risky_multipath', averse_joint_traj,skip_n=18).preview()
if __name__ == "__main__":
    main()