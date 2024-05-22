import numpy as np
# import matplotlib.pyplot as plt

from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,OvercookedState,SoupState, ObjectState

class StartStateManager:
    def __init__(self,mdp):
        self.mdp = mdp

    def start_with_soup(self,state):
        player = state.players[0]
        soup = SoupState.get_soup(
            player.position,
            num_onions=3,
            num_tomatoes=0,
            finished=True,
        )
        player.set_object(soup)
        return state

    def random_start_state(self,rnd_obj_prob_thresh=0.25):
        # Random position
        random_state = self.mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)()
        # random_state.players[0].position = mdp.start_player_positions[1]
        # If there are two players, make sure no overlapp
        # while np.all(np.array(random_state.players[1].position) == np.array(random_state.players[0].position)):
        #     random_state = mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)()
        #     random_state.players[1].position = mdp.start_player_positions[1]
        # env.state = random_state

        # Arbitrary hard-coding for randomization of objects
        # For each pot, add a random amount of onions and tomatoes with prob rnd_obj_prob_thresh
        # Begin the soup cooking with probability rnd_obj_prob_thresh
        pots = self.mdp.get_pot_states(random_state)["empty"]
        for pot_loc in pots:
            p = np.random.rand()
            if p < rnd_obj_prob_thresh:
                n = int(np.random.randint(low=1, high=3))
                q = np.random.rand()
                # cooking_tick = np.random.randint(0, 20) if n == 3 else -1
                cooking_tick = 0 if n == 3 else -1
                random_state.objects[pot_loc] = SoupState.get_soup(
                    pot_loc,
                    num_onions=n,
                    num_tomatoes=0,
                    cooking_tick=cooking_tick,
                )

        # For each player, add a random object with prob rnd_obj_prob_thresh
        for player in random_state.players:
            p = np.random.rand()
            if p < rnd_obj_prob_thresh:
                # Different objects have different probabilities
                obj = np.random.choice(
                    ["dish", "onion", "soup"], p=[0.2, 0.6, 0.2]
                )
                n = int(np.random.randint(low=1, high=4))
                if obj == "soup":
                    player.set_object(
                        SoupState.get_soup(
                            player.position,
                            num_onions=n,
                            num_tomatoes=0,
                            finished=True,
                        )
                    )
                else:
                    player.set_object(ObjectState(obj, player.position))
        # random_state = mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.25)()
        return random_state