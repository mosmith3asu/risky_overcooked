import itertools

import numpy as np
import torch
import copy
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,OvercookedState,SoupState, ObjectState

class FeasibleActionManager(object):
    def __init__(self, env):
        self.env = env
    def is_feasible_move(self,player, action):
        facing = player.orientation
        adj_pos = Action.move_in_direction(player.position, action)
        terr = self.env.mdp.get_terrain_type_at_pos(adj_pos)

        # move into counter already facing
        same_dir = (facing == action)
        is_counter = (terr == "X")
        if same_dir and is_counter: return False

        # move into resource without prerequisite object
        # ==> Not valid if player is just moving by in constrained space
        if not player.has_object() and terr in ['P', 'S', 'D']: return False
        return True

    def is_feasible_interact(self, player, action='Interact'):
        facing = player.orientation
        adj_pos = Action.move_in_direction(player.position, facing)
        terr = self.env.mdp.get_terrain_type_at_pos(adj_pos)

        # Invalid cases
        if terr in [" ","W"]: return False
        if not player.has_object() and terr in ['P', 'S']: return False
        elif player.has_object() and terr in ["D","O","T"]: return False

        # Valid cases
        elif player.has_object() and terr in ["P","S","X"]: return True
        elif not player.has_object() and terr in ["D","O","T","X"]: return True
        else: raise ValueError(f"Unknown interact case {terr} + {player.has_object()} [{player.held_object}]")

    def is_feasible_action(self,player,action):
        # Movement actions
        if action in Direction.INDEX_TO_DIRECTION:
            return self.is_feasible_move(player, action)

        # Pointless interact action
        elif action == 'interact':
            return self.is_feasible_interact(player, action)

        elif action == (0, 0): return True
        else: raise ValueError(f"Uknown Action {action}")
    def get_feasible_joint_actions(self, state, as_joint_idx = False):
        """
        Actions that are not useful:
        - moving into counter you are already facing
        - moving to face a {counter, service, pot} without held object
        - attempting to interact with anything but ingredient resource w/o held object
        :param state:
        :param as_idx: returns array of feasible actions
        :return:
        """
        feasible_actions = np.ones([2,len(Action.ALL_ACTIONS)])
        for ip, player in enumerate(state.players):
            for ia, action in enumerate(Action.INDEX_TO_ACTION):
                feasible_actions[ip, ia] = int(self.is_feasible_action(player, action))

        # Convert to joint index if specified
        if as_joint_idx:
            # feasible_actions = itertools.product(*feasible_actions)
            feasible_actions = np.array(list(itertools.product(*feasible_actions)))
            feasible_actions = (feasible_actions[:,0]*feasible_actions[:,1]).flatten()

        return feasible_actions


def invert_obs(obs,N_PLAYER_FEAT = 9):
    if isinstance(obs, np.ndarray):
        _obs = np.concatenate([obs[N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
                               obs[:N_PLAYER_FEAT],
                               obs[2 * N_PLAYER_FEAT:]])
    elif isinstance(obs, torch.Tensor):
        n_dim = len(obs.shape)
        if n_dim == 1:
            _obs = torch.cat([obs[N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
                               obs[:N_PLAYER_FEAT],
                               obs[2 * N_PLAYER_FEAT:]])
        elif n_dim == 2:
            _obs = torch.cat([obs[:, N_PLAYER_FEAT:2 * N_PLAYER_FEAT],
                                   obs[:, :N_PLAYER_FEAT],
                                   obs[:, 2 * N_PLAYER_FEAT:]], dim=1)
        else: raise ValueError("Invalid obs dimension")
    else: raise ValueError("Invalid obs type")
    return _obs



def invert_joint_action(joint_action_batch):
    if isinstance(joint_action_batch, int):
        return Action.reverse_joint_action_index(joint_action_batch)
    elif isinstance(joint_action_batch, np.ndarray):
        BATCH_SIZE = joint_action_batch.shape[0]
        action_batch = np.array([Action.reverse_joint_action_index(joint_action_batch[i]) for i in range(BATCH_SIZE)])
    elif isinstance(joint_action_batch, torch.Tensor):
        BATCH_SIZE = joint_action_batch.shape[0]
        # action_batch = torch.tensor([Action.reverse_joint_action_index(joint_action_batch[i]) for i in range(BATCH_SIZE)]).unsqueeze(1)
        action_batch = torch.tensor([Action.reverse_joint_action_index(joint_action_batch[i]) for i in range(BATCH_SIZE)],
                                    device=joint_action_batch.device).unsqueeze(1)
    else:  raise ValueError("Invalid joint_action dim")
    return action_batch

    # BATCH_SIZE = action_batch.shape[0]
    # action_batch = torch.tensor(
    #     [Action.reverse_joint_action_index(action_batch[i]) for i in range(BATCH_SIZE)]).unsqueeze(1)
    # return action_batch


def invert_prospect(prospects):
     _prospects = copy.deepcopy(prospects)
     for i,prospect in enumerate(_prospects):
         _prospects[i][1] = invert_obs(prospect[1])
     return _prospects


class StartStateManager:
    def __init__(self,mdp):
        self.mdp = mdp
        # self.with_soup = with_soup
        # self.random_loc = random_loc
        # self.random_pot = random_pot
        # self.random_held = random_held

    def assign(self,state,random_loc=False, with_soup=False, random_pot = False,random_held=False):
        assert not (with_soup and random_held), "Cannot start with both with_soup and random_held"
        if random_loc:   state = self.random_start_loc(state)
        if random_pot:   state = self.random_pot_contents(state)
        if random_held:  state = self.random_held_objects(state)
        if with_soup:    state = self.start_with_soup(state)
        return state

    def start_with_soup(self,state):
        # player = state.players[player_idx]
        # soup = SoupState.get_soup( player.position, num_onions=3, num_tomatoes=0, finished=True)
        # player.set_object(soup)
        for player in state.players:
            soup = SoupState.get_soup(player.position, num_onions=3, num_tomatoes=0, finished=True)
            player.set_object(soup)
        return state

    def random_start_loc(self,state):
        state = self.mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)()
        return state

    def random_pot_contents(self,state,rnd_obj_prob_thresh=0.25):
        pots = self.mdp.get_pot_states(state)["empty"]
        for pot_loc in pots:
            p = np.random.rand()
            if p < rnd_obj_prob_thresh:
                n = int(np.random.randint(low=1, high=3))
                cooking_tick = 0 if n == 3 else -1
                state.objects[pot_loc] = SoupState.get_soup(pot_loc,num_onions=n,num_tomatoes=0,cooking_tick=cooking_tick)
        return state

    def random_held_objects(self,state,rnd_obj_prob_thresh=0.25):
        for player in state.players:
            p = np.random.rand()
            if p < rnd_obj_prob_thresh:
                # Different objects have different probabilities
                obj = np.random.choice(["onion","dish", "soup"], p=[0.6, 0.2, 0.2])
                n = int(np.random.randint(low=1, high=4))
                if obj == "soup":
                    soup = SoupState.get_soup(player.position,num_onions=n,num_tomatoes=0,finished=True)
                    player.set_object(soup)
                else:
                    player.set_object(ObjectState(obj, player.position))
        return state

    # def random_start_state(self,rnd_obj_prob_thresh=0.25):
        # Random position
        # random_state = self.mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)()
        # random_state.players[0].position = mdp.start_player_positions[1]
        # If there are two players, make sure no overlapp
        # while np.all(np.array(random_state.players[1].position) == np.array(random_state.players[0].position)):
        #     random_state = mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)()
        #     random_state.players[1].position = mdp.start_player_positions[1]
        # env.state = random_state

        # Arbitrary hard-coding for randomization of objects
        # For each pot, add a random amount of onions and tomatoes with prob rnd_obj_prob_thresh
        # Begin the soup cooking with probability rnd_obj_prob_thresh
        # pots = self.mdp.get_pot_states(random_state)["empty"]
        # for pot_loc in pots:
        #     p = np.random.rand()
        #     if p < rnd_obj_prob_thresh:
        #         n = int(np.random.randint(low=1, high=3))
        #         q = np.random.rand()
        #         # cooking_tick = np.random.randint(0, 20) if n == 3 else -1
        #         cooking_tick = 0 if n == 3 else -1
        #         random_state.objects[pot_loc] = SoupState.get_soup(
        #             pot_loc,
        #             num_onions=n,
        #             num_tomatoes=0,
        #             cooking_tick=cooking_tick,
        #         )

        # # For each player, add a random object with prob rnd_obj_prob_thresh
        # for player in random_state.players:
        #     p = np.random.rand()
        #     if p < rnd_obj_prob_thresh:
        #         # Different objects have different probabilities
        #         obj = np.random.choice(
        #             ["dish", "onion", "soup"], p=[0.2, 0.6, 0.2]
        #         )
        #         n = int(np.random.randint(low=1, high=4))
        #         if obj == "soup":
        #             player.set_object(
        #                 SoupState.get_soup(
        #                     player.position,
        #                     num_onions=n,
        #                     num_tomatoes=0,
        #                     finished=True,
        #                 )
        #             )
        #         else:
        #             player.set_object(ObjectState(obj, player.position))
        # # random_state = mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.25)()
        # return random_state