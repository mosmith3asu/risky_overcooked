import json
import os
import pickle
import random
from abc import ABC, abstractmethod
from queue import Empty, Full, LifoQueue, Queue
from collections import deque
from threading import Lock, Thread
from time import time

import numpy as np
import torch
try:
    from utils import DOCKER_VOLUME, create_dirs
except:
    from risky_overcooked_webserver.server.utils import DOCKER_VOLUME, create_dirs

from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from risky_overcooked_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MotionPlanner,
)
from risky_overcooked_rl.utils.belief_update import BayesianBeliefUpdate
from risky_overcooked_rl.algorithms.DDQN.utils.agents import DQN_vector_feature
from risky_overcooked_rl.algorithms.DDQN.utils.game_thoery import QuantalResponse_torch
# Relative path to where all static pre-trained agents are stored on server
AGENT_DIR = None

# Maximum allowable game time (in seconds)
MAX_GAME_TIME = None


def _configure(max_game_time, agent_dir):
    global AGENT_DIR, MAX_GAME_TIME
    MAX_GAME_TIME = max_game_time
    AGENT_DIR = agent_dir


def fix_bc_path(path):
    """
    Loading a PPO agent trained with a BC agent requires loading the BC model as well when restoring the trainer, even though the BC model is not used in game
    For now the solution is to include the saved BC model and fix the relative path to the model in the config.pkl file
    """

    import dill

    # the path is the agents/Rllib.*/agent directory
    agent_path = os.path.dirname(path)
    with open(os.path.join(agent_path, "config.pkl"), "rb") as f:
        data = dill.load(f)
    bc_model_dir = data["bc_params"]["bc_config"]["model_dir"]
    last_dir = os.path.basename(bc_model_dir)
    bc_model_dir = os.path.join(agent_path, "bc_params", last_dir)
    data["bc_params"]["bc_config"]["model_dir"] = bc_model_dir
    with open(os.path.join(agent_path, "config.pkl"), "wb") as f:
        dill.dump(data, f)


class Game(ABC):

    """
    Class representing a game object. Coordinates the simultaneous actions of arbitrary
    number of players. Override this base class in order to use.

    Players can post actions to a `pending_actions` queue, and driver code can call `tick` to apply these actions.


    It should be noted that most operations in this class are not on their own thread safe. Thus, client code should
    acquire `self.lock` before making any modifications to the instance.

    One important exception to the above rule is `enqueue_actions` which is thread safe out of the box
    """

    # Possible TODO: create a static list of IDs used by the class so far to verify id uniqueness
    # This would need to be serialized, however, which might cause too great a performance hit to
    # be worth it

    EMPTY = "EMPTY"

    class Status:
        DONE = "done"
        ACTIVE = "active"
        RESET = "reset"
        INACTIVE = "inactive"
        ERROR = "error"

    def __init__(self, *args, **kwargs):
        """
        players (list): List of IDs of players currently in the game
        spectators (set): Collection of IDs of players that are not allowed to enqueue actions but are currently watching the game
        id (int):   Unique identifier for this game
        pending_actions List[(Queue)]: Buffer of (player_id, action) pairs have submitted that haven't been commited yet
        lock (Lock):    Used to serialize updates to the game state
        is_active(bool): Whether the game is currently being played or not
        """
        self.players = []
        self.spectators = set()
        self.pending_actions = []
        self.id = kwargs.get("id", id(self))
        self.lock = Lock()
        self._is_active = False


    @abstractmethod
    def is_full(self):
        """
        Returns whether there is room for additional players to join or not
        """
        pass

    @abstractmethod
    def apply_action(self, player_idx, action):
        """
        Updates the game state by applying a single (player_idx, action) tuple. Subclasses should try to override this method
        if possible
        """
        pass

    @abstractmethod
    def is_finished(self):
        """
        Returns whether the game has concluded or not
        """
        pass

    def is_ready(self):
        """
        Returns whether the game can be started. Defaults to having enough players
        """
        return self.is_full()

    @property
    def is_active(self):
        """
        Whether the game is currently being played
        """
        return self._is_active

    @property
    def reset_timeout(self):
        """
        Number of milliseconds to pause game on reset
        """
        return 3000

    def apply_actions(self):
        """
        Updates the game state by applying each of the pending actions in the buffer. Is called by the tick method. Subclasses
        should override this method if joint actions are necessary. If actions can be serialized, overriding `apply_action` is
        preferred
        """
        for i in range(len(self.players)):
            try:
                while True:
                    action = self.pending_actions[i].get(block=False)
                    self.apply_action(i, action)
            except Empty:
                pass

    def activate(self):
        """
        Activates the game to let server know real-time updates should start. Provides little functionality but useful as
        a check for debugging
        """
        self._is_active = True

    def deactivate(self):
        """
        Deactives the game such that subsequent calls to `tick` will be no-ops. Used to handle case where game ends but
        there is still a buffer of client pings to handle
        """
        self._is_active = False

    def reset(self):
        """
        Restarts the game while keeping all active players by resetting game stats and temporarily disabling `tick`
        """

        if not self.is_active:
            raise ValueError("Inactive Games cannot be reset")
        if self.is_finished():
            return self.Status.DONE
        self.deactivate()
        self.activate()
        return self.Status.RESET

    def needs_reset(self):
        """
        Returns whether the game should be reset on the next call to `tick`
        """
        return False

    def tick(self):
        """
        Updates the game state by applying each of the pending actions. This is done so that players cannot directly modify
        the game state, offering an additional level of safety and thread security.

        One can think of "enqueue_action" like calling "git add" and "tick" like calling "git commit"

        Subclasses should try to override `apply_actions` if possible. Only override this method if necessary
        """
        if not self.is_active:
            return self.Status.INACTIVE
        if self.needs_reset():
            self.reset()
            return self.Status.RESET

        self.apply_actions()
        return self.Status.DONE if self.is_finished() else self.Status.ACTIVE

    def enqueue_action(self, player_id, action):
        """
        Add (player_id, action) pair to the pending action queue, without modifying underlying game state

        Note: This function IS thread safe
        """
        if not self.is_active:
            # Could run into issues with is_active not being thread safe
            return
        if player_id not in self.players:
            # Only players actively in game are allowed to enqueue actions
            return
        try:
            player_idx = self.players.index(player_id)
            self.pending_actions[player_idx].put(action)
        except Full:
            pass

    def get_state(self):
        """
        Return a JSON compatible serialized state of the game. Note that this should be as minimalistic as possible
        as the size of the game state will be the most important factor in game performance. This is sent to the client
        every frame update.
        """
        return {"players": self.players}

    def to_json(self):
        """
        Return a JSON compatible serialized state of the game. Contains all information about the game, does not need to
        be minimalistic. This is sent to the client only once, upon game creation
        """
        return self.get_state()

    def is_empty(self):
        """
        Return whether it is safe to garbage collect this game instance
        """
        return not self.num_players

    def add_player(self, player_id, idx=None, buff_size=-1):
        """
        Add player_id to the game
        """
        if self.is_full():
            raise ValueError("Cannot add players to full game")
        if self.is_active:
            raise ValueError("Cannot add players to active games")
        if not idx and self.EMPTY in self.players:
            idx = self.players.index(self.EMPTY)
        elif not idx:
            idx = len(self.players)

        padding = max(0, idx - len(self.players) + 1)
        for _ in range(padding):
            self.players.append(self.EMPTY)
            self.pending_actions.append(self.EMPTY)

        self.players[idx] = player_id
        self.pending_actions[idx] = Queue(maxsize=buff_size)
        # self.pending_actions[idx] = deque(maxlen=buff_size)

    def add_spectator(self, spectator_id):
        """
        Add spectator_id to list of spectators for this game
        """
        if spectator_id in self.players:
            raise ValueError("Cannot spectate and play at same time")
        self.spectators.add(spectator_id)

    def remove_player(self, player_id):
        """
        Remove player_id from the game
        """
        try:
            idx = self.players.index(player_id)
            self.players[idx] = self.EMPTY
            self.pending_actions[idx] = self.EMPTY
        except ValueError:
            return False
        else:
            return True

    def remove_spectator(self, spectator_id):
        """
        Removes spectator_id if they are in list of spectators. Returns True if spectator successfully removed, False otherwise
        """
        try:
            self.spectators.remove(spectator_id)
        except ValueError:
            return False
        else:
            return True

    def clear_pending_actions(self):
        """
        Remove all queued actions for all players
        """
        for i, player in enumerate(self.players):
            if player != self.EMPTY:
                queue = self.pending_actions[i]
                queue.queue.clear()

    @property
    def num_players(self):
        return len([player for player in self.players if player != self.EMPTY])

    def get_data(self):
        """
        Return any game metadata to server driver.
        """
        return {}


class DummyGame(Game):

    """
    Standin class used to test basic server logic
    """

    def __init__(self, **kwargs):
        super(DummyGame, self).__init__(**kwargs)
        self.counter = 0

    def is_full(self):
        return self.num_players == 2

    def apply_action(self, idx, action):
        pass

    def apply_actions(self):
        self.counter += 1

    def is_finished(self):
        return self.counter >= 100

    def get_state(self):
        state = super(DummyGame, self).get_state()
        state["count"] = self.counter
        return state


class DummyInteractiveGame(Game):

    """
    Standing class used to test interactive components of the server logic
    """

    def __init__(self, **kwargs):
        super(DummyInteractiveGame, self).__init__(**kwargs)
        self.max_players = int(
            kwargs.get("playerZero", "human") == "human"
        ) + int(kwargs.get("playerOne", "human") == "human")
        self.max_count = kwargs.get("max_count", 30)
        self.counter = 0
        self.counts = [0] * self.max_players

    def is_full(self):
        return self.num_players == self.max_players

    def is_finished(self):
        return max(self.counts) >= self.max_count

    def apply_action(self, player_idx, action):
        if action.upper() == Direction.NORTH:
            self.counts[player_idx] += 1
        if action.upper() == Direction.SOUTH:
            self.counts[player_idx] -= 1

    def apply_actions(self):
        super(DummyInteractiveGame, self).apply_actions()
        self.counter += 1

    def get_state(self):
        state = super(DummyInteractiveGame, self).get_state()
        state["count"] = self.counter
        for i in range(self.num_players):
            state["player_{}_count".format(i)] = self.counts[i]
        return state


class OvercookedGame(Game):
    """
    Class for bridging the gap between Overcooked_Env and the Game interface

    Instance variable:
        - max_players (int): Maximum number of players that can be in the game at once
        - mdp (OvercookedGridworld): Controls the underlying Overcooked game logic
        - score (int): Current reward acheived by all players
        - max_time (int): Number of seconds the game should last
        - npc_policies (dict): Maps user_id to policy (Agent) for each AI player
        - npc_state_queues (dict): Mapping of NPC user_ids to LIFO queues for the policy to process
        - curr_tick (int): How many times the game server has called this instance's `tick` method
        - ticker_per_ai_action (int): How many frames should pass in between NPC policy forward passes.
            Note that this is a lower bound; if the policy is computationally expensive the actual frames
            per forward pass can be higher
        - action_to_overcooked_action (dict): Maps action names returned by client to action names used by OvercookedGridworld
            Note that this is an instance variable and not a static variable for efficiency reasons
        - human_players (set(str)): Collection of all player IDs that correspond to humans
        - npc_players (set(str)): Collection of all player IDs that correspond to AI
        - randomized (boolean): Whether the order of the layouts should be randomized

    Methods:
        - npc_policy_consumer: Background process that asynchronously computes NPC policy forward passes. One thread
            spawned for each NPC
        - _curr_game_over: Determines whether the game on the current mdp has ended
    """

    def __init__(
        self,
        layouts=["cramped_room"],
        mdp_params={},
        num_players=2,
        gameTime=60,
        playerZero="human",
        playerOne="human",
        showPotential=False,
        randomized=False,
        ticks_per_ai_action=1,
        **kwargs
    ):
        super(OvercookedGame, self).__init__(**kwargs)
        self.show_potential = showPotential
        self.mdp_params = mdp_params
        self.layouts = layouts
        self.max_players = int(num_players)
        self.mdp = None

        self.debug = True
        self.is_frozen = False

        # self.mdp_params["p_slip"] = kwargs.get("p_slips", [0.0])[0]
        # self.mdp = OvercookedGridworld.from_layout_name(
        #     layouts[0], **self.mdp_params
        # )


        self.mp = None
        self.score = 0
        self.phi = 0
        self.max_time = min(int(gameTime), kwargs.get("max_game_time", MAX_GAME_TIME))
        self.npc_policies = {}
        self.npc_state_queues = {}
        self.action_to_overcooked_action = {
            "STAY": Action.STAY,
            "UP": Direction.NORTH,
            "DOWN": Direction.SOUTH,
            "LEFT": Direction.WEST,
            "RIGHT": Direction.EAST,
            "SPACE": Action.INTERACT,
        }
        self.ticks_per_ai_action = ticks_per_ai_action
        self.curr_tick = 0
        self.human_players = set()
        self.npc_players = set()

        self.client_ready = True

        if randomized:
            random.shuffle(self.layouts)

        if playerZero != "human":
            player_zero_id = playerZero + "_0"
            self.add_player(player_zero_id, idx=0, buff_size=1, is_human=False)
            self.npc_policies[player_zero_id] = self.get_policy(
                playerZero, idx=0
            )
            self.npc_state_queues[player_zero_id] = LifoQueue()

        if playerOne != "human":
            player_one_id = playerOne + "_1"
            self.add_player(player_one_id, idx=1, buff_size=1, is_human=False)
            self.npc_policies[player_one_id] = self.get_policy(
                playerOne, idx=1
            )
            self.npc_state_queues[player_one_id] = LifoQueue()


        # Always kill ray after loading agent, otherwise, ray will crash once process exits
        # Only kill ray after loading both agents to avoid having to restart ray during loading
        # if ray.is_initialized():
        #     ray.shutdown()

        # if kwargs["dataCollection"]:
        #     self.write_data = True
        #     self.write_config = kwargs["collection_config"]
        # else:
        #     self.write_data = False

        self.trajectory = []

    def _curr_game_over(self):
        return time() - self.start_time >= self.max_time

    def needs_reset(self):
        # return False
        return self._curr_game_over() and not self.is_finished()

    def add_player(self, player_id, idx=None, buff_size=-1, is_human=True):
        super(OvercookedGame, self).add_player(
            player_id, idx=idx, buff_size=buff_size
        )
        if is_human:
            self.human_players.add(player_id)
        else:
            self.npc_players.add(player_id)

    def remove_player(self, player_id):
        removed = super(OvercookedGame, self).remove_player(player_id)
        if removed:
            if player_id in self.human_players:
                self.human_players.remove(player_id)
            elif player_id in self.npc_players:
                self.npc_players.remove(player_id)
            else:
                raise ValueError("Inconsistent state")

    def npc_policy_consumer(self, policy_id):
        queue = self.npc_state_queues[policy_id]
        policy = self.npc_policies[policy_id]
        while self._is_active:
            state = queue.get()
            npc_action, _ = policy.action(state)
            super(OvercookedGame, self).enqueue_action(policy_id, npc_action)

    def is_full(self):
        return self.num_players >= self.max_players

    def is_finished(self):
        val = not self.layouts and self._curr_game_over()
        return val

    def is_empty(self):
        """
        Game is considered safe to scrap if there are no active players or if there are no humans (spectating or playing)
        """
        return (
            super(OvercookedGame, self).is_empty()
            or not self.spectators
            and not self.human_players
        )

    def is_ready(self):
        """
        Game is ready to be activated if there are a sufficient number of players and at least one human (spectator or player)
        """
        # server_ready = super(OvercookedGame, self).is_ready() and not self.is_empty()
        # return  server_ready and self.client_ready
        return super(OvercookedGame, self).is_ready() and not self.is_empty()

    def apply_action(self, player_id, action):
        pass

    def apply_actions(self):
        # Default joint action, as NPC policies and clients probably don't enqueue actions fast
        # enough to produce one at every tick
        joint_action = [Action.STAY] * len(self.players)

        # Synchronize individual player actions into a joint-action as required by overcooked logic
        for i in range(len(self.players)):
            # if this is a human, don't block and inject
            if self.players[i] in self.human_players:
                try:
                    # we don't block here in case humans want to Stay
                    joint_action[i] = self.pending_actions[i].get(block=False)
                except Empty:
                    pass
            else:
                # we block on agent actions to ensure that the agent gets to do one action per state
                joint_action[i] = self.pending_actions[i].get(block=True)

        # Apply overcooked game logic to get state transition
        prev_state = self.state
        self.state, info = self.mdp.get_state_transition(
            prev_state, joint_action
        )
        if self.show_potential:
            self.phi = self.mdp.potential_function(
                prev_state, self.mp, gamma=0.99
            )

        # Send next state to all background consumers if needed
        if self.curr_tick % self.ticks_per_ai_action == 0:
            for npc_id in self.npc_policies:
                self.npc_state_queues[npc_id].put(self.state, block=False)

        # Update score based on soup deliveries that might have occured
        curr_reward = sum(info["sparse_reward_by_agent"])
        self.score += curr_reward

        transition = {
            "state": json.dumps(prev_state.to_dict()),
            "joint_action": json.dumps(joint_action),
            "reward": curr_reward,
            "time_left": max(self.max_time - (time() - self.start_time), 0),
            "score": self.score,
            "time_elapsed": time() - self.start_time,
            "cur_gameloop": self.curr_tick,
            "layout": json.dumps(self.mdp.terrain_mtx),
            "layout_name": self.curr_layout,
            "trial_id": str(self.start_time),
            "player_0_id": self.players[0],
            "player_1_id": self.players[1],
            "player_0_is_human": self.players[0] in self.human_players,
            "player_1_is_human": self.players[1] in self.human_players,
        }

        self.trajectory.append(transition)


        # Return about the current transition
        return prev_state, joint_action, info

    def enqueue_action(self, player_id, action):
        overcooked_action = self.action_to_overcooked_action[action]
        super(OvercookedGame, self).enqueue_action(
            player_id, overcooked_action
        )

    def reset(self):
        status = super(OvercookedGame, self).reset()
        if status == self.Status.RESET:
            # Hacky way of making sure game timer doesn't "start" until after reset timeout has passed
            self.start_time += self.reset_timeout / 1000

    def tick(self):
        self.curr_tick += 1
        return super(OvercookedGame, self).tick()

    def activate(self):
        super(OvercookedGame, self).activate()

        # Sanity check at start of each game
        if not self.npc_players.union(self.human_players) == set(self.players):
            raise ValueError("Inconsistent State")

        self.curr_layout = self.layouts.pop()
        self.mdp_params['p_slip'] = self.p_slip
        self.mdp = OvercookedGridworld.from_layout_name(
            self.curr_layout, **self.mdp_params
        )
        for key, val in self.npc_policies.items():
            if isinstance(val, ToMAI):
                # If the policy is a DQN vector feature policy, we need to activate it
                self.npc_policies[key].activate(self.mdp)

        if self.debug:
            print(f'\n\nActivating OvercookedGame...')
            print("\tLayout: {}".format(self.curr_layout))
            print("\tp_slip: {}".format(self.mdp.p_slip))
            print("\tWrite data: {}".format(self.write_data))

        for key,val in self.npc_policies.items():
            if isinstance(val, ToMAI):
                self.npc_policies[key].activate(self.mdp)

                if self.debug:
                    print("\tCanidates...")
                    for can_fname in self.npc_policies[key].candidate_fnames:
                        print("\t\t{}".format(can_fname))



        if self.show_potential:
            self.mp = MotionPlanner.from_pickle_or_compute(
                self.mdp, counter_goals=NO_COUNTERS_PARAMS
            )
        self.state = self.mdp.get_standard_start_state()
        if self.show_potential:
            self.phi = self.mdp.potential_function(
                self.state, self.mp, gamma=0.99
            )
        self.start_time = time()
        self.curr_tick = 0
        self.score = 0
        self.threads = []
        if not self.is_frozen:

            for npc_policy in self.npc_policies:
                self.npc_policies[npc_policy].reset()
                self.npc_state_queues[npc_policy].put(self.state)
                t = Thread(target=self.npc_policy_consumer, args=(npc_policy,))
                self.threads.append(t)
                t.start()

    def deactivate(self):
        super(OvercookedGame, self).deactivate()
        # Ensure the background consumers do not hang
        for npc_policy in self.npc_policies:
            self.npc_state_queues[npc_policy].put(self.state)

        # Wait for all background threads to exit
        for t in self.threads:
            t.join()

        # Clear all action queues
        self.clear_pending_actions()
        # self.client_ready = False

    def get_state(self):
        state_dict = {}
        state_dict["potential"] = self.phi if self.show_potential else None
        state_dict["state"] = self.state.to_dict()
        state_dict["score"] = self.score
        state_dict["time_left"] = max(
            self.max_time - (time() - self.start_time), 0
        )
        return state_dict

    def to_json(self):
        obj_dict = {}
        obj_dict["terrain"] = self.mdp.terrain_mtx if self._is_active else None
        obj_dict["state"] = self.get_state() if self._is_active else None

        return obj_dict

    def get_policy(self, npc_id, idx=0):
        # print("Loading agent {}".format(npc_id))
        if npc_id.lower() == "rs-tom":
            # print("Loading agent {}".format(npc_id))
            # Load the RS-TOM agent
            try:
                agent = ToMAI(['averse', 'neutral', 'seeking'])
                agent.activate(self.mdp)
                return agent
            except Exception as e:
                raise IOError("Error loading Rllib Agent\n{}".format(e.__repr__()))
        elif npc_id.lower() == "rational":
            try:
                agent = ToMAI(['neutral'])
                agent.activate(self.mdp)
                return agent

            except Exception as e:
                raise IOError("Error loading Rllib Agent\n{}".format(e.__repr__()))

        # elif npc_id.lower().startswith("rllib"):
        #     try:
        #         # Loading rllib agents requires additional helpers
        #         fpath = os.path.join(AGENT_DIR, npc_id, "agent")
        #         fix_bc_path(fpath)
        #         agent = load_agent(fpath, agent_index=idx)
        #         return agent
        #     except Exception as e:
        #         raise IOError(
        #             "Error loading Rllib Agent\n{}".format(e.__repr__())
        #         )
        else:
            try:
                fpath = os.path.join(AGENT_DIR, npc_id, "agent.pickle")
                with open(fpath, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                raise IOError("Error loading agent\n{}".format(e.__repr__()))

    def get_data(self,clear_trajectory=True):
        """
        Returns and then clears the accumulated trajectory
        """
        data = {
            "uid": str(time()),
            "trajectory": self.trajectory,
        }
        if clear_trajectory: self.trajectory = []
        # if we want to store the data and there is data to store
        if self.write_data and len(data["trajectory"]) > 0:
            configs = self.write_config
            # create necessary dirs
            data_path = create_dirs(configs, self.curr_layout)
            # the 3-layer-directory structure should be able to uniquely define any experiment
            print("Writing data to {}".format(data_path))
            with open(os.path.join(data_path, "result.pkl"), "wb") as f:
                pickle.dump(data, f)
        return data


class OvercookedTutorial(OvercookedGame):

    """
    Wrapper on OvercookedGame that includes additional data for tutorial mechanics, most notably the introduction of tutorial "phases"

    Instance Variables:
        - curr_phase (int): Indicates what tutorial phase we are currently on
        - phase_two_score (float): The exact sparse reward the user must obtain to advance past phase 2
    """

    def __init__(
        self,
        layouts=["tutorial_0"],
        mdp_params={},
        playerZero="human",
        playerOne="AI",
        phaseTwoScore=15,
        **kwargs
    ):
        super(OvercookedTutorial, self).__init__(
            layouts=layouts,
            mdp_params=mdp_params,
            playerZero=playerZero,
            playerOne=playerOne,
            showPotential=False,
            **kwargs
        )
        self.phase_two_score = phaseTwoScore
        self.phase_two_finished = False
        self.max_time = 0
        self.max_players = 2
        self.ticks_per_ai_action = 1
        self.curr_phase = 0
        # we don't collect tutorial data
        self.write_data = False



    @property
    def reset_timeout(self):
        return 1

    def needs_reset(self):
        if self.curr_phase == 0:
            return self.score > 0
        elif self.curr_phase == 1:
            return self.score > 0
        elif self.curr_phase == 2:
            return self.phase_two_finished
        return False

    def is_finished(self):
        # return not self.layouts and self.score >= float("inf")
        return self.score > 0

    def reset(self):
        super(OvercookedTutorial, self).reset()
        self.curr_phase += 1

    def get_policy(self, *args, **kwargs):
        tutorial_num = int(self.layouts[0].split("_")[-1])
        return TutorialAI(tutorial_phase = tutorial_num)

    def apply_actions(self):
        """
        Apply regular MDP logic with retroactive score adjustment tutorial purposes
        """
        _, _, info = super(OvercookedTutorial, self).apply_actions()

        human_reward, ai_reward = info["sparse_reward_by_agent"]

        # We only want to keep track of the human's score in the tutorial
        self.score -= ai_reward

        # Phase two requires a specific reward to complete
        if self.curr_phase == 2:
            self.score = 0
            if human_reward == self.phase_two_score:
                self.phase_two_finished = True


class DummyOvercookedGame(OvercookedGame):
    """
    Class that hardcodes the AI to be random. Used for debugging
    """

    def __init__(self, layouts=["cramped_room"], **kwargs):
        super(DummyOvercookedGame, self).__init__(layouts, **kwargs)

    def get_policy(self, *args, **kwargs):
        return DummyAI()


class DummyAI:
    """
    Randomly samples actions. Used for debugging
    """

    def action(self, state):
        [action] = random.sample(
            [
                Action.STAY,
                Direction.NORTH,
                Direction.SOUTH,
                Direction.WEST,
                Direction.EAST,
                Action.INTERACT,
            ],
            1,
        )
        return action, None

    def reset(self):
        pass


class DummyComputeAI(DummyAI):
    """
    Performs simulated compute before randomly sampling actions. Used for debugging
    """

    def __init__(self, compute_unit_iters=1e5):
        """
        compute_unit_iters (int): Number of for loop cycles in one "unit" of compute. Number of
                                    units performed each time is randomly sampled
        """
        super(DummyComputeAI, self).__init__()
        self.compute_unit_iters = int(compute_unit_iters)

    def action(self, state):
        # Randomly sample amount of time to busy wait
        iters = random.randint(1, 10) * self.compute_unit_iters

        # Actually compute something (can't sleep) to avoid scheduling optimizations
        val = 0
        for i in range(iters):
            # Avoid branch prediction optimizations
            if i % 2 == 0:
                val += 1
            else:
                val += 2

        # Return randomly sampled action
        return super(DummyComputeAI, self).action(state)


class StayAI:
    """
    Always returns "stay" action. Used for debugging
    """

    def action(self, state):
        return Action.STAY, None

    def reset(self):
        pass


class TutorialAI:
    def __init__(self,tutorial_phase):
        self.tutorial_phase = tutorial_phase
        self.curr_phase = -1
        self.curr_tick = -1

        # Routing Commands
        self.start2onion = [Direction.WEST for _ in range(1)]  # start to onion
        # self.obj2pot = [Direction.EAST for _ in range(6)] + [Direction.NORTH]  # both onion and dish
        self.obj2pot = [Direction.EAST for _ in range(5)] + [Direction.NORTH]  # both onion and dish

        self.pot2onion = [Direction.WEST for _ in range(5)]
        self.onion2dish = [Direction.SOUTH for _ in range(1)]
        self.pot2service = [Direction.EAST for _ in range(1)]
        self.service2start = [Direction.WEST for _ in range(5)]

        self.puddle2onion = [Action.STAY] + [Direction.WEST for _ in range(4)]
        self.puddle2dish = [Action.STAY] + [Direction.WEST for _ in range(4)] + [Direction.SOUTH]

        self.obj2pot_detour = [Direction.EAST for _ in range(2)] + \
                              [Direction.SOUTH,Direction.EAST,Direction.EAST,Direction.NORTH ]+ \
                              [Direction.EAST for _ in range(2)] + [Direction.NORTH]  # both onion and dish


        # self.pot2onion_detour = self.obj2pot_detour = [Direction.WEST for _ in range(2)] + \
        #                       [Direction.SOUTH,Direction.WEST,Direction.WEST,Direction.NORTH ]+ \
        #                       [Direction.WEST for _ in range(2)] + [Direction.NORTH]  # both onion and dish
        # self.service2start_detour = self.obj2pot_detour = [Direction.WEST for _ in range(1)] + \
        #                       [Direction.SOUTH,Direction.WEST,Direction.WEST,Direction.NORTH ]+ \
        #                       [Direction.WEST for _ in range(2)] + [Direction.NORTH]  # both onion and dish
        # Routing Sequences



        self.start2pass = [Direction.WEST,Direction.WEST,Direction.WEST,Direction.SOUTH ]  # start to passing
        self.pass2pot = [Direction.NORTH,Direction.EAST,Direction.EAST,Direction.EAST,Direction.NORTH]  # start to passing
        self.pot2pass = [Direction.WEST,Direction.WEST,Direction.WEST,Direction.SOUTH]  # start to passing
        self.pass2dump = [Action.STAY,Action.INTERACT,Direction.NORTH] + [Direction.WEST for _ in range(2)]  + \
                         [Direction.EAST for _ in range(2)]  +[Direction.SOUTH] # start to passing

        # Select Policy
        if tutorial_phase == 0:
            self.policy = self.policy_0()
        elif tutorial_phase == 1:
            self.route_sequence = [self.start2onion]
            for _ in range(3):  # bring 3 onions
                self.route_sequence.append([Action.INTERACT] + self.obj2pot + [Action.INTERACT] + self.pot2onion)
            self.route_sequence.append(
                self.onion2dish + [Action.INTERACT] + self.obj2pot)  # +[Action.INTERACT] calced based on soup tick
            self.route_sequence.append(self.pot2service + [Action.INTERACT])
            self.route_sequence.append(self.service2start)

            self.route_idx = 0
            self.checkpoint_tick = 0
            self.slipped_route = None
            self.policy = self.policy_1
        elif tutorial_phase == 2:
            self.policy = self.policy_2()


        elif tutorial_phase == 3:
            self.curr_route = [Action.STAY for _ in range(3)] + self.start2pass[1:]


            self.route_sequence = []
            self.route_sequence.append([Action.STAY for _ in range(3)])
            self.route_sequence.append(self.start2pass)
            for _ in range(3):
                self.route_sequence.append(['wait4onion'])
                self.route_sequence.append([Action.STAY for i in range(2)]+[Action.INTERACT] + self.pass2pot + [Action.INTERACT] + self.pot2pass)
            self.route_sequence.append(['wait4dish'])
            self.route_sequence.append([Action.STAY for i in range(2)] + [Action.INTERACT] + self.pass2pot)
            self.route_sequence.append(['wait4soup'])
            self.route_sequence.append([Direction.WEST] + self.pot2pass + [Action.INTERACT]) # unknown error requires adding extra west
            self.route_sequence.append(['stay'])
            self.dumping_route = None

            self.route_idx = 0
            self.checkpoint_tick = 0
            self.policy = self.policy_3
        else:
            raise ValueError("Invalid tutorial phase")

    def action(self, state):
        self.curr_tick += 1
        if self.tutorial_phase == 0:
            return (self.policy[self.curr_tick % len(self.policy)], None)
        elif self.tutorial_phase == 1:
            return (self.policy(state), None)
        elif self.tutorial_phase == 2:
            return (self.policy[self.curr_tick % len(self.policy)], None)
        elif self.tutorial_phase == 3:
            return (self.policy(state), None)

        return Action.STAY, None

    def get_AI_soup(self, state):
        """ Returns the number of onions in the AI's pot """
        all_soups = state.unowned_objects_by_type['soup']
        if len(all_soups) == 0:
            # raise ValueError("IWas expecting soup")
            return None
        elif len(all_soups) == 1:
            AI_soup = all_soups[0]
            return AI_soup
        elif len(all_soups) > 1:
            y_coords = [soup.position[1] for soup in all_soups]
            idx = int(y_coords[0]>y_coords[1]) #index of AI soup
            AI_soup = all_soups[idx]
            return AI_soup
        else:
            raise ValueError("Invalid number of soups in pot")

    @property
    def route_tick(self):
        return self.curr_tick - self.checkpoint_tick

    def policy_1(self,state):
        route = self.route_sequence[self.route_idx]


        AI = state.players[1]  # get AI player

        # Advance Route
        if self.route_tick >= len(route):
            if AI.has_object():
                if AI.get_object().name == 'dish':
                    soup = self.get_AI_soup(state)
                    if soup.is_ready: return Action.INTERACT
                    else: return Action.STAY

            self.route_idx += 1
            self.route_idx %= len(self.route_sequence)  # loops routes
            self.checkpoint_tick = self.curr_tick
            route = self.route_sequence[self.route_idx]

        # check if slipped

        if AI.dropped_obj == 'onion':
            self.checkpoint_tick = self.curr_tick
            self.slipped_route = self.puddle2onion + [Action.INTERACT]
        elif AI.dropped_obj == 'dish':
            self.checkpoint_tick = self.curr_tick
            self.slipped_route = self.puddle2dish + [Action.INTERACT]
        # elif AI.dropped_obj != 'none':
        #     print(f'Dropped unknown object: {AI.dropped_obj}')

        # Reroute if slipped
        if self.slipped_route is not None:
            action = self.slipped_route[self.route_tick]
            if self.route_tick >= len(self.slipped_route)-1: # slipped_rout complete
                self.slipped_route = None
                self.checkpoint_tick = self.curr_tick

        # Otherwise continue on main route
        else:
            action = route[self.route_tick]

        return action

    def policy_2(self):
        # Form Main Loop
        loop = []
        loop += self.start2onion

        for _ in range(3):  # bring 3 onions
            loop += [Action.INTERACT]
            loop += self.obj2pot_detour
            loop += [Action.INTERACT]
            loop += self.pot2onion + [self.pot2onion[-2]]
        loop += self.onion2dish
        loop += [Action.INTERACT]
        loop += self.obj2pot_detour
        loop += [Action.STAY for _ in range(4)]  # plates soup
        loop += [Action.INTERACT]  # plates soup
        loop += self.pot2service
        loop += [Action.INTERACT]  # deliver soup
        loop += self.service2start

        return loop

    def policy_0(self):
        # Form Main Loop
        loop = []
        loop += self.start2onion

        for _ in range(3): # bring 3 onions
            loop += [Action.INTERACT]
            loop += self.obj2pot
            loop += [Action.INTERACT]
            loop += self.pot2onion
        loop += self.onion2dish
        loop += [Action.INTERACT]
        loop += self.obj2pot
        loop += [Action.STAY for _ in range(7)] # plates soup
        loop += [Action.INTERACT] # plates soup
        loop += self.pot2service
        loop += [Action.INTERACT] # deliver soup
        loop += self.service2start

        return loop

    def policy_3(self, state):
        if len(self.curr_route)>0:
            action = self.curr_route.pop(0)

            if action == 'wait4soup':
                soup = self.get_AI_soup(state)
                if soup.is_ready:
                    return Action.INTERACT
                else:
                    self.curr_route.append('wait4soup')
                    return Action.STAY

            elif action == 'wait4service':
                for key, objs in state.all_objects_by_type.items():
                    if key == 'soup' and len(objs)>0:
                        self.curr_route.append('wait4service')
                    return Action.STAY

                action = Action.STAY # no objects
            return action

        else:
            AI = state.players[1]  # get AI player
            pass_pos = (AI.position[0] + Direction.SOUTH[0], AI.position[1] + Direction.SOUTH[1])
            soup = self.get_AI_soup(state)
            if soup is None or len(soup.ingredients) <3 : desired_obj = 'onion'
            elif soup.is_cooking or soup.is_ready : desired_obj = 'dish'
            else:  raise ValueError("Unknown pot state")

            if AI.has_object():
                if AI.get_object().name == 'soup':
                    self.curr_route = self.pot2pass + [Action.STAY,Action.STAY, Action.INTERACT ,'wait4service']
                elif AI.get_object().name == desired_obj and AI.get_object().name == 'onion':
                    self.curr_route = self.pass2pot + [Action.STAY,Action.INTERACT] + self.pot2pass
                elif AI.get_object().name == desired_obj and AI.get_object().name == 'dish':
                    self.curr_route = self.pass2pot + [Action.STAY,'wait4soup']
                else:
                    self.curr_route = self.pass2dump[2:]
                    # raise ValueError(f"Unknown object {AI.get_object().name} in AI's possession")


            elif state.has_object(pass_pos):
                counter_obj = state.get_object(pass_pos)
                if counter_obj.name == 'soup':
                    self.curr_route = [Action.STAY]
                else:
                    self.curr_route =  [Action.STAY,Action.STAY, Action.INTERACT]
            else:
                self.curr_route = [Action.STAY]

            return self.curr_route.pop(0)


        # route = self.route_sequence[self.route_idx]
        # AI = state.players[1]  # get AI player
        #
        # if self.dumping_route is not None:
        #     # print('dumping route')
        #     action = self.dumping_route[self.route_tick]
        #     if self.route_tick >= len(self.dumping_route) - 1:  # slipped_rout complete
        #         self.dumping_route = None
        #         self.checkpoint_tick = self.curr_tick
        #     return action
        # elif 'stay' in route:
        #     if AI.has_object():
        #         if AI.get_object().name == 'soup':
        #             return Action.INTERACT
        #
        #     soup_exists = False
        #     for players in state.players:
        #         if players.has_object():
        #             if players.get_object().name == 'soup':
        #                 soup_exists = True
        #     if not soup_exists:
        #         # print('Soup does not exist')
        #         self.route_idx = 0
        #     return Action.STAY
        # elif 'wait4onion' in route:
        #     # print('wait4onion route')
        #     pass_pos = (AI.position[0] + Direction.SOUTH[0], AI.position[1] + Direction.SOUTH[1])
        #     # print(pass_pos,state.has_object(pass_pos))
        #     # if state.has_object(pass_pos):
        #     #     counter_obj = state.get_object(pass_pos)
        #     #     if counter_obj.name == 'onion':
        #     #         self.route_idx += 1
        #     #         self.route_idx %= len(self.route_sequence)
        #     #         self.checkpoint_tick = self.curr_tick
        #     #         route = self.route_sequence[self.route_idx]
        #     #         action = route[self.route_tick]
        #     #         return action
        #     #     elif counter_obj.name == 'dish':
        #     #         print('dumping dish')
        #     #         self.dumping_route =  self.pass2dump
        #     #         self.checkpoint_tick = self.curr_tick
        #     #         return Action.STAY
        #     # else: return Action.STAY
        #     if state.has_object(pass_pos):
        #         if not AI.has_object():
        #             return Action.INTERACT
        #     elif AI.has_object():
        #         _obj = AI.get_object()
        #         if _obj.name == 'onion':
        #             self.route_idx += 1
        #             self.route_idx %= len(self.route_sequence)
        #             self.checkpoint_tick = self.curr_tick
        #             route = self.route_sequence[self.route_idx][3:]
        #             action = route[self.route_tick]
        #             return action
        #         elif _obj.name == 'dish':
        #             print('dumping dish')
        #             self.dumping_route =  self.pass2dump[1:]
        #             self.checkpoint_tick = self.curr_tick
        #             return Action.STAY
        #         else: return Action.STAY
        #     else: return Action.STAY
        #
        # elif 'wait4dish' in route:
        #     # print('wait4dish route')
        #     pass_pos = (AI.position[0] + Direction.SOUTH[0], AI.position[1] + Direction.SOUTH[1])
        #     if state.has_object(pass_pos):
        #         counter_obj = state.get_object(pass_pos)
        #         if not AI.has_object():
        #             return Action.INTERACT
        #
        #     elif AI.has_object():
        #         _obj = AI.get_object()
        #         if _obj.name == 'dish':
        #             self.route_idx += 1
        #             self.route_idx %= len(self.route_sequence)
        #             self.checkpoint_tick = self.curr_tick
        #             route = self.route_sequence[self.route_idx][3:]
        #             action = route[self.route_tick]
        #             return action
        #         elif _obj.name == 'onion':
        #             self.dumping_route = self.pass2dump[1:]
        #             self.checkpoint_tick = self.curr_tick
        #             return Action.STAY
        #         else:
        #             return Action.STAY
        #     else: return Action.STAY
        # elif 'wait4soup' in route:
        #     # print('wait4soup route')
        #     if AI.has_object():
        #         if AI.get_object().name == 'dish':
        #             soup = self.get_AI_soup(state)
        #             if soup.is_ready:
        #                 self.route_idx += 1
        #                 self.route_idx %= len(self.route_sequence)
        #                 self.checkpoint_tick = self.curr_tick
        #
        #                 print(self.route_sequence[self.route_idx],self.route_tick)
        #                 return Action.INTERACT
        #             else:
        #                 return Action.STAY
        #     else:
        #         raise ValueError("Expecting dish in hand")
        # else:
        #     # print('standard route')
        #     # Advance Route
        #     if self.route_tick >= len(route):
        #         self.route_idx += 1
        #         self.route_idx %= len(self.route_sequence)  # loops routes
        #         self.checkpoint_tick = self.curr_tick
        #         # route = self.route_sequence[self.route_idx]
        #         return Action.STAY
        #     action = route[self.route_tick]
        #
        #     # soup_exists = False
        #     # for players in state.players:
        #     #     if players.has_object():
        #     #         if players.get_object().name == 'soup':
        #     #             soup_exists = True
        #     # if not soup_exists:
        #     #     # print('Soup does not exist')
        #     #     self.route_idx = 0
        #     return action

    def reset(self):
        self.curr_tick = -1
        self.curr_phase += 1



class ToMAI:
    """
    AI that is used for the ToM agent. It is a simple AI that follows a set of instructions to complete the game.
    It is used to test the ToM agent and its ability to predict the actions of the AI.
    """

    def __init__(self, candidates, device = 'cpu' ):
        """
        :param layout: layout for which to load model
        :param candidates: used as fname extension while loading eg: RS-ToM=['averse','neutral','rational']
        """
        # Parse args
        self.mdp = None
        self.layout = None
        self.p_slip = None
        self.device = device

        # Load Policies
        self.candidates = candidates
        self.candidate_fnames = None
        self.policies = None
        self.n_candidates = None

        # Instantiate Belief Updater
        self.belief = None




    def activate(self,mdp):
        """ Activates when the game is activated. This is used to load the policies and instantiate the belief updater."""
        self.mdp = mdp
        self.layout = mdp.layout_name
        self.p_slip = mdp.p_slip

        # Load Policies
        self.candidate_fnames = [f'{self.layout}_pslip{str(self.p_slip).replace(".", "")}__{candidate}.pt' for candidate
                                 in self.candidates]
        self.policies = self.load_policies(self.candidate_fnames)
        self.n_candidates = len(self.policies)

        # Instantiate Belief Updater
        self.belief = BayesianBeliefUpdate(self.policies, self.policies, names=self.candidates,iego=1,ipartner=0)
        self.belief.reset_prior()


    def load_policies(self, candidate_fnames):
        """
        Loads the policies for the AI. This is used to test the ToM agent and its ability to predict the actions of the AI.
        """
        # print(f"All fnames {candidate_fnames}")
        policies = []
        for fname in candidate_fnames:
            PATH = AGENT_DIR + f'/RiskSensitiveAI/{fname}'
            if not os.path.exists(PATH):
                raise ValueError(f"Policy file {PATH} does not exist")
            try:
                policies.append(TorchPolicy(PATH,self.device))
            except Exception as e:
                raise print(f"Error loading policy from {PATH}\n{e}")
        return policies


    def observe(self,state,human_action,ai_action):
        """
        :param obs: encoded state vector compliant with the ToM agent's models
        :param joint_action:
        :return:
        """
        is_trivial = human_action == "Stay" or human_action == (0,0)
        if self.n_candidates >1 and not is_trivial:
            obs = self.mdp.get_lossless_encoding_vector_astensor(state, device=self.device).unsqueeze(0)
            human_iA = Action.ACTION_TO_INDEX[human_action]
            self.belief.update_belief(obs, human_iA, is_only_partner_action=True)

    def action(self, state):
        ego_policy = self.belief.best_response if self.n_candidates > 1 else self.policies[0]
        obs = self.mdp.get_lossless_encoding_vector_astensor(state, device=self.device).unsqueeze(0)
        action = ego_policy.action(obs)
        return action, None

    def reset(self):
        pass

class TorchPolicy:
    """Handles Torch.NN interface and action selection.
    Is lightweight version of SelfPlay_QRE_OSA agent
    """
    def __init__(self,PATH, device, action_selection='softmax',ego_idx = 1,
                 sophistication=8,belief_trick=True
                 ):
        # torch.cuda.set_device(device)
        # print(f"TorchPolicy: Loading policy from {PATH}")
        self.PATH = PATH
        self.device = device
        assert action_selection in ['greedy', 'softmax'], "Action selection must be either greedy or softmax"
        self.action_selection = action_selection
        self.ego_idx = ego_idx
        # self.num_hidden_layers = num_hidden_layers
        # self.size_hidden_layers = size_hidden_layers


        self.player_action_dim = len(Action.ALL_ACTIONS)
        self.joint_action_dim = len(Action.ALL_JOINT_ACTIONS)
        self.joint_action_space = Action.ALL_JOINT_ACTIONS
        self.num_agents = 2
        # self.rationality = 20
        self.rationality = 10
        self.model = self.load_model(PATH)

        self.QRE = QuantalResponse_torch(rationality=self.rationality,belief_trick=belief_trick,
                                         sophistication=sophistication,joint_action_space=self.joint_action_space,
                                         device=self.device)

        # self.frozen_count = 0




    def load_model(self, PATH):
        loaded_model = torch.load(PATH, weights_only=True, map_location=self.device)
        # print(f'Geting model data')
        # obs_shape = (loaded_model['layer1.weight'].size()[1],)
        # size_hidden_layers = loaded_model['layer1.weight'].shape[0]
        obs_shape = (loaded_model['layers.0.weight'].size()[1],)
        size_hidden_layers = loaded_model['layers.0.weight'].shape[0]
        num_hidden_layers = int(len(loaded_model.keys()) / 2)
        # size_hidden_layers = loaded_model['layer1.weight'].size()[0]
        # for key,val in loaded_model.items():
        #     print(key, "\t", val.size())

        n_actions = self.joint_action_dim
        # model = DQN_vector_feature(obs_shape, n_actions, self.num_hidden_layers, self.size_hidden_layers).to(self.device)
        model = DQN_vector_feature(obs_shape, n_actions, num_hidden_layers, size_hidden_layers).to(self.device)
        model.load_state_dict(loaded_model)
        return model

    def action(self, obs):
        with torch.no_grad():
            _,_,joint_pA = self.choose_joint_action(obs)
            ego_pA = joint_pA[0,self.ego_idx]
            if self.action_selection == 'greedy':
                ego_action = Action.INDEX_TO_ACTION[np.argmax(ego_pA)]
            elif self.action_selection == 'softmax':
                ia = np.random.choice(np.arange(len(ego_pA)), p=ego_pA)
                ego_action = Action.INDEX_TO_ACTION[ia]
        return ego_action

    def choose_joint_action(self, obs,epsilon=0):
        with torch.no_grad():
            NF_Game = self.get_normal_form_game(obs)
            joint_action, joint_action_idx, action_probs = self.QRE.choose_actions(NF_Game)
        return joint_action, joint_action_idx, action_probs

    def compute_EQ(self, NF_Games):
        NF_Games = NF_Games.reshape(-1, self.num_agents, self.player_action_dim, self.player_action_dim)
        all_dists = self.QRE.level_k_qunatal(NF_Games)
        return all_dists


    def get_normal_form_game(self, obs):
        """ Batch compute the NF games for each observation"""
        batch_size = obs.shape[0]
        all_games = torch.zeros([batch_size, self.num_agents, self.player_action_dim, self.player_action_dim],
                                device=self.device)
        for i in range(self.num_agents):
            if i == 1: obs = self.invert_obs(obs)
            q_values = self.model(obs).detach()
            q_values = q_values.reshape(batch_size, self.player_action_dim, self.player_action_dim)
            all_games[:, i, :, :] = q_values if i == 0 else torch.transpose(q_values, -1, -2)
        return all_games

    def invert_obs(self,obs,N_PLAYER_FEAT = 9):
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
            else:
                raise ValueError("Invalid obs dimension")
        else:
            raise ValueError("Invalid obs type")
        return _obs



#
#
# class TutorialAI:
#     COOK_SOUP_LOOP = [
#         # Grab first onion
#         Direction.WEST,
#         Direction.WEST,
#         Direction.WEST,
#         Action.INTERACT,
#         # Place onion in pot
#         Direction.EAST,
#         Direction.NORTH,
#         Action.INTERACT,
#         # Grab second onion
#         Direction.WEST,
#         Action.INTERACT,
#         # Place onion in pot
#         Direction.EAST,
#         Direction.NORTH,
#         Action.INTERACT,
#         # Grab third onion
#         Direction.WEST,
#         Action.INTERACT,
#         # Place onion in pot
#         Direction.EAST,
#         Direction.NORTH,
#         Action.INTERACT,
#         # Cook soup
#         Action.INTERACT,
#         # Grab plate
#         Direction.EAST,
#         Direction.SOUTH,
#         Action.INTERACT,
#         Direction.WEST,
#         Direction.NORTH,
#         # Deliver soup
#         Action.INTERACT,
#         Direction.EAST,
#         Direction.EAST,
#         Direction.EAST,
#         Action.INTERACT,
#         Direction.WEST,
#     ]
#
#     COOK_SOUP_COOP_LOOP = [
#         # Grab first onion
#         Direction.WEST,
#         Direction.WEST,
#         Direction.WEST,
#         Action.INTERACT,
#         # Place onion in pot
#         Direction.EAST,
#         Direction.SOUTH,
#         Action.INTERACT,
#         # Move to start so this loops
#         Direction.EAST,
#         Direction.EAST,
#         # Pause to make cooperation more real time
#         Action.STAY,
#         Action.STAY,
#         Action.STAY,
#         Action.STAY,
#         Action.STAY,
#         Action.STAY,
#         Action.STAY,
#         Action.STAY,
#         Action.STAY,
#     ]
#
#     def __init__(self):
#         self.curr_phase = -1
#         self.curr_tick = -1
#
#     def action(self, state):
#         self.curr_tick += 1
#         if self.curr_phase == 0:
#             return (
#                 self.COOK_SOUP_LOOP[self.curr_tick % len(self.COOK_SOUP_LOOP)],
#                 None,
#             )
#         elif self.curr_phase == 2:
#             return (
#                 self.COOK_SOUP_COOP_LOOP[
#                     self.curr_tick % len(self.COOK_SOUP_COOP_LOOP)
#                 ],
#                 None,
#             )
#         return Action.STAY, None
#
#     def reset(self):
#         self.curr_tick = -1
#         self.curr_phase += 1
