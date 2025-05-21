import copy

import numpy as np
import os
import sys
import pickle
import atexit
import json
import logging
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from threading import Lock, Thread
from time import time

import queue
from queue import Empty, Full, LifoQueue, Queue
# from .data_logging import ExperimentData
from data_logging import DummyData, DemographicData,SurveyData, InteractionData
from utils import ThreadSafeDict, ThreadSafeSet
from datetime import datetime
from game import ToMAI,TutorialAI, Game, OvercookedGame
import game as GAME
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from risky_overcooked_rl.utils.belief_update import BayesianBeliefUpdate
from risky_overcooked_rl.algorithms.DDQN.utils.agents import DQN_vector_feature
#########################################################################################
#### GLOBALS ############################################################################
#########################################################################################
DEBUG = True

if True:
    # Read in global config
    CONF_PATH = os.getenv("CONF_PATH", "config.json")
    with open(CONF_PATH, "r") as f:
        CONFIG = json.load(f)

    LOGFILE = CONFIG["logfile"] # Where errors will be logged
    LAYOUTS = CONFIG["layouts"] # Available layout names
    LAYOUT_GLOBALS = CONFIG["layout_globals"]     # Values that are standard across layouts
    MAX_GAME_LENGTH = CONFIG["MAX_GAME_LENGTH"]    # Maximum allowable game length (in seconds)
    AGENT_DIR = CONFIG["AGENT_DIR"]    # Path to where pre-trained agents will be stored on server
    MAX_EXPERIMENTS = CONFIG["MAX_EXPERIMENTS"]  # Maximum number of games that can run concurrently. Contrained by available memory and CPU
    MAX_FPS = CONFIG["MAX_FPS"] # Frames per second cap for serving to client
    PREDEFINED_CONFIG = json.dumps(CONFIG["predefined"])     # Default configuration for predefined experiment
    TUTORIAL_CONFIG = json.dumps(CONFIG["tutorial"])     # Default configuration for tutorial


    FREE_IDS = queue.Queue(maxsize=MAX_EXPERIMENTS) # Global queue of available IDs. This is how we synch game creation and keep track of how many games are in memory
    FREE_MAP = ThreadSafeDict()  # Bitmap that indicates whether ID is currently in use. Game with ID=i is "freed" by setting FREE_MAP[i] = True
    EXPERIMENT_MAP = ThreadSafeDict()
    for i in range(MAX_EXPERIMENTS): # Initialize our ID tracking data
        FREE_IDS.put(i)
        FREE_MAP[i] = True
        EXPERIMENT_MAP[i] = None

    EXPERIMENTS = ThreadSafeDict()     # Mapping of game-id to game objects
    ACTIVE_EXPERIMENTS = ThreadSafeSet()     # Set of games IDs that are currently being played
    # WAITING_EXPERIMENTS = queue.Queue()  # Queue of games IDs that are waiting for additional players to join. Note that some of these IDs might be stale (i.e. if FREE_MAP[id] = True)
    USERS = ThreadSafeDict()  # Mapping of users to locks associated with the ID. Enforces user-level serialization
    USER_ROOMS = ThreadSafeDict()     # Mapping of user id's to the current game (room) they are in

    # Mapping of string game names to corresponding classes
    # GAME_NAME_TO_CLS = {
    #     # "overcooked": CustomOvercookedGame,
    #     "overcooked": OvercookedGame,
    #     "tutorial": OvercookedTutorial,
    # }
    #
    GAME._configure(MAX_GAME_LENGTH, AGENT_DIR)


#########################################################################################
#### FLASK CONFIG #######################################################################
#########################################################################################
if True:
    # Create and configure flask app
    app = Flask(__name__, template_folder=os.path.join("static", "templates"))
    app.config["DEBUG"] = os.getenv("FLASK_ENV", "production") == "development"
    socketio = SocketIO(app, cors_allowed_origins="*", logger=app.config["DEBUG"])


    # Attach handler for logging errors to file
    handler = logging.FileHandler(LOGFILE)
    handler.setLevel(logging.ERROR)
    app.logger.addHandler(handler)

#########################################################################################
#### Global Coordination Functions ######################################################
#########################################################################################
def try_create_experiment(user_id,*args,**kwargs):
    """
       Tries to create a brand new Game object based on parameters in `kwargs`

       Returns (Game, Error) that represent a pointer to a game object, and error that occured
       during creation, if any. In case of error, `Game` returned in None. In case of sucess,
       `Error` returned is None

       Possible Errors:
           - Runtime error if server is at max game capacity
           - Propogate any error that occured in game __init__ function
       """

    try:
        curr_id = FREE_IDS.get(block=False)
        assert FREE_MAP[curr_id], "Current id is already in use"
        print("Creating experiment with id {}".format(curr_id) if DEBUG else "")
        experiment = Experiment(curr_id, **kwargs)
        EXPERIMENT_MAP[user_id] = curr_id
        # EXPERIMENTS[experiment.id] = experiment
        # FREE_MAP[experiment.id] = False
        # return experiment, None
    except queue.Empty:
        err = RuntimeError("Server at max capacity")
        return None, err
    except Exception as e:
        raise IOError("Error creating experiment\n{}".format(e.__repr__()))
        return None, e
    else:
        EXPERIMENTS[experiment.id] = experiment
        FREE_MAP[experiment.id] = False
        return experiment, None


def cleanup_experiment(experiment):
    if FREE_MAP[experiment.id]:
        raise ValueError("Double free on a game")

    # # User tracking
    # for user_id in experiment.players:
    #     leave_curr_room(user_id)

    # Socketio tracking
    socketio.close_room(experiment.id) #TODO: is room opened somewhere?

    # Game tracking
    FREE_MAP[experiment.id] = True
    FREE_IDS.put(experiment.id)
    del EXPERIMENTS[experiment.id]

    if experiment.id in ACTIVE_EXPERIMENTS:
        ACTIVE_EXPERIMENTS.remove(experiment.id)

def get_experiment(experiment_id):
    return EXPERIMENTS.get(experiment_id, None)


def get_curr_experiment(user_id):
    experiment_id = EXPERIMENT_MAP.get(user_id, None)
    return get_experiment(experiment_id)
    # return get_experiment(get_curr_room(user_id))


def cleanup_game(game_id):
    pass
    # if FREE_MAP[game.id]:
    #     raise ValueError("Double free on a game")


def set_curr_room(user_id, room_id):
    USER_ROOMS[user_id] = room_id


# def get_curr_room(user_id):
#     return USER_ROOMS.get(user_id, None)
#
# def leave_curr_room(user_id):
#     del USER_ROOMS[user_id]


#########################################################################################
# Socket Handler Helpers ################################################################
#########################################################################################
def _create_user(user_id):
    if user_id in USERS:
        return
    USERS[user_id] = Lock()
    if DEBUG: print(f'Created User: {user_id}')

def _leave_game(user_id):
    raise NotImplementedError("Deprecated.")

def _create_experiment(user_id, params={}):
    experiment, err = try_create_experiment(user_id, **params)
    if not experiment:
        raise IOError("Error creating experiment: _create_experiment()")
        emit("creation_failed", {"error": err.__repr__()})
        return

    with experiment.lock:
        join_room(experiment.id)
    set_curr_room(user_id, experiment.id)


def _ensure_consistent_state():
    """
    Simple sanity checks of invariants on global state data

    Let ACTIVE be the set of all active game IDs, GAMES be the set of all existing
    game IDs, and WAITING be the set of all waiting (non-stale) game IDs. Note that
    a game could be in the WAITING_GAMES queue but no longer exist (indicated by
    the FREE_MAP)

    - Intersection of WAITING and ACTIVE games must be empty set
    - Union of WAITING and ACTIVE must be equal to GAMES
    - id \in FREE_IDS => FREE_MAP[id]
    - id \in ACTIVE_GAMES => Game in active state
    - id \in WAITING_GAMES => Game in inactive state
    """
    raise NotImplementedError("Deprecated.")

# def _close_hanging_games(curr_experiment):
#     with curr_experiment.lock:
#         if curr_experiment.game is not None:
#             data = {}
#             socketio.emit(
#                 "end_game", {"status": 'close', "data": data}, room=curr_experiment.game.id
#             )
#             curr_experiment.game.deactivate()
#             cleanup_game(curr_experiment.game.id)
#########################################################################################
# Application routes ####################################################################
#########################################################################################
@app.route("/")
def index():
    # TODO: pass all of necessary config
    return render_template("experiment.html",  tutorial_config=TUTORIAL_CONFIG)


@app.route("/debug")
def debug():
    resp = {}
    games = []
    active_games = []
    waiting_games = []
    users = []
    free_ids = []
    free_map = {}
    for experiment_id in ACTIVE_EXPERIMENTS:
        experiment = get_experiment(experiment_id)
        active_games.append({"id": experiment_id, "state": experiment.game.to_json()})

    # for experiment_id in list(WAITING_GAMES.queue):
    #     experiment = get_experiment(game_id)
    #     experiment_state = None if FREE_MAP[experiment_id] else experiment.to_json()
    #     waiting_games.append({"id": experiment_id, "state": experiment_state})

    for experiment_id in EXPERIMENTS:
        games.append(experiment_id)

    # for user_id in USER_ROOMS:
    #     users.append({user_id: get_curr_room(user_id)})

    for experiment_id in list(FREE_IDS.queue):
        free_ids.append(experiment_id)

    for experiment_id in FREE_MAP:
        free_map[experiment_id] = FREE_MAP[experiment_id]

    resp["active_games"] = active_games
    resp["waiting_games"] = waiting_games
    resp["all_games"] = games
    resp["users"] = users
    resp["free_ids"] = free_ids
    resp["free_map"] = free_map
    return jsonify(resp)


#########################
# Socket Event Handlers #
#########################

# Asynchronous handling of client-side socket events. Note that the socket persists even after the
# event has been handled. This allows for more rapid data communication, as a handshake only has to
# happen once at the beginning. Thus, socket events are used for all game updates, where more rapid
# communication is needed

def creation_params(params):
    # """
    #    This function extracts the dataCollection and oldDynamics settings from the input and
    #    process them before sending them to game creation
    #    """
    # this params file should be a dictionary that can have these keys:
    # playerZero: human/Rllib*agent
    # playerOne: human/Rllib*agent
    # layout: one of the layouts in the config file, I don't think this one is used
    # gameTime: time in seconds
    # oldDynamics: on/off
    # dataCollection: on/off
    # layouts: [layout in the config file], this one determines which layout to use, and if there is more than one layout, a series of game is run back to back
    #

    use_old = False
    if "oldDynamics" in params and params["oldDynamics"] == "on":
        params["mdp_params"] = {"old_dynamics": True}
        use_old = True

    if "dataCollection" in params and params["dataCollection"] == "on":
        # config the necessary setting to properly save data
        params["dataCollection"] = True
        mapping = {"human": "H"}
        # gameType is either HH, HA, AH, AA depending on the config
        gameType = "{}{}".format(
            mapping.get(params["playerZero"], "A"),
            mapping.get(params["playerOne"], "A"),
        )
        params["collection_config"] = {
            "time": datetime.today().strftime("%Y-%m-%d_%H-%M-%S"),
            "type": gameType,
        }
        if use_old:
            params["collection_config"]["old_dynamics"] = "Old"
        else:
            params["collection_config"]["old_dynamics"] = "New"

    else:
        params["dataCollection"] = False
    # raise NotImplementedError("Deprecated.")

@socketio.on("connect")
def on_connect():
    user_id = request.sid

    if user_id in USERS:
        return

    USERS[user_id] = Lock()
    print(user_id)

@socketio.on("join_experiment")
def on_join_experiment(data):
    raise NotImplementedError("Deprecated.")
    # """ Participants clicks [Join] button on landing page"""
    #
    # if DEBUG: print("join triggered")
    #
    # user_id = request.sid
    # _create_user(user_id)
    #
    # # create experiment
    # with USERS[user_id]:
    #     if get_curr_experiment(user_id):
    #         return  # TODO: Check if user is in a experiment that was closed out of
    #
    #     # Create game if not previously in one
    #     params = data.get("params", {})
    #     if DEBUG: print("\t| params", params)
    #
    #     creation_params(params)
    #     _create_experiment(user_id, params)
    #     return

@socketio.on("create_game")
def on_create_game(data):
    """ Participants clicks [Begin Game] button"""

    if DEBUG: print(f"create_game triggered: {data}")

    user_id = request.sid

    with USERS[user_id]:
        curr_experiment = get_curr_experiment(user_id)

        if curr_experiment.game is None:
            curr_experiment.open_game(**data)

        curr_game = curr_experiment.game
        curr_game.client_ready = True

        if curr_game:
            if curr_game.is_ready():
                curr_game.activate()
                emit(
                    "start_game",
                    {"start_info": curr_game.to_json()},
                    room=curr_game.id,
                )
                socketio.start_background_task(play_game, curr_game, fps=6)

            elif curr_game.is_frozen: # just an image of priming stage
                # curr_game.activate()
                render_info = curr_game.render()
                emit(
                    "start_game",
                    {"start_info": render_info},
                    room=curr_game.id,
                )
                # with curr_game.lock:
                #     curr_game.deactivate()
                #     cleanup_game(curr_game.id)

            else:
                print('Game not ready')
        else:
            print('User {} not in game'.format(user_id))

@socketio.on("survey_response")
def on_survey_response(data):
    """ Participants clicks [Join] button on landing page"""
    if DEBUG: print("survey_response triggered")
    user_id = request.sid
    with USERS[user_id]:
        curr_experiment = get_curr_experiment(user_id)

        if curr_experiment.game is not None:
            curr_experiment.close_game()

        for key,responses in data.items():

            with curr_experiment.lock:
                curr_experiment.log_survey(key, responses)
                curr_experiment.update_stage(key, True)




@socketio.on("update_stage")
def on_update_stage(data):
    if DEBUG: print("on_complete_stage triggered", data)

    # Joining Experiment
    for key,val in data.items():
        if key == 'skip':
            return
        if key == 'join':
            """ Participants clicks [Join] button on landing page"""
            user_id = request.sid
            _create_user(user_id)

            # create experiment
            with USERS[user_id]:

                if get_curr_experiment(user_id):
                    raise IOError("User has in experiment")
                    return  # TODO: Check if user is in a experiment that was closed out of

                # Create game if not previously in one
                params = data.get("params", {})
                if DEBUG: print("\t| params", params)

                creation_params(params)
                _create_experiment(user_id, params)
                return
        else:
            # All other updates
            user_id = request.sid
            with USERS[user_id]:
                curr_experiment = get_curr_experiment(user_id)
                if DEBUG: print(user_id,curr_experiment)

                with curr_experiment.lock:

                    for key,val in data.items():
                        curr_experiment.update_stage(key,val)


@socketio.on("action")
def on_action(data):

    user_id = request.sid
    action = data["action"]

    experiment = get_curr_experiment(user_id)
    if not experiment.game:
        raise IOError("Action took but no game found")
        return

    # experiment.game.enqueue_action(user_id, action)
    human_id = 'human' + "_0"
    experiment.game.enqueue_action(human_id, action)

@socketio.on("disconnect")
def on_disconnect():
    print("disonnect triggered", file=sys.stderr)
    # Ensure game data is properly cleaned-up in case of unexpected disconnect
    user_id = request.sid
    if user_id not in USERS:
        return
    with USERS[user_id]:
        _leave_game(user_id)

    del USERS[user_id]

@socketio.on("leave")
def on_leave(data):
    raise NotImplementedError("Deprecated.")

@socketio.on("close_game")
def on_close_game(data):
    if DEBUG: print(f'Closing game...')
    user_id = request.sid

    with USERS[user_id]:
        curr_experiment = get_curr_experiment(user_id)
        with curr_experiment.lock:
            # _close_hanging_games(curr_experiment)
            curr_experiment.close_game()
    # raise NotImplementedError("Deprecated.")

@socketio.on("client_ready")
def on_client_ready(data):
    # user_id = request.sid
    # print(f"[{user_id}] client joined", file=sys.stderr)
    raise NotImplementedError("Deprecated.")

# Exit handler for server
def on_exit():
    # Force-terminate all games on server termination
    for game_id in EXPERIMENTS:
        socketio.emit(
            "end_game",
            {
                "status": Game.Status.INACTIVE,
                "data": get_experiment(game_id).game.get_data(),
            },
            room=game_id,
        )


#########################################################################################
#### EXPERIMENT MANAGER #################################################################
#########################################################################################
AGENT_DIR = CONFIG["AGENT_DIR"]
MAX_GAME_TIME = CONFIG["MAX_GAME_LENGTH"] # Maximum allowable game time (in seconds)
class RiskyOvercookedGame(OvercookedGame):
    """
    RiskyOvercookedGame is a subclass of OvercookedGame that implements the game logic for the
    Risky Overcooked experiment. It handles the game state, player actions, and scoring.
    """

    def __init__(self,
                 layout,
                 partner_name,
                 **kwargs
                 ):

        self.layout = layout
        self.curr_layout = layout
        kwargs['layouts'] = [layout]
        super(RiskyOvercookedGame, self).__init__(playerOne=partner_name, max_game_time = 200,**kwargs)
        self.mdp_params = kwargs.get("mdp_params", {'neglect_boarders':True})
        self.layout = layout
        self.curr_layout = layout
        self.id = kwargs.get("id", id(self))

        # Add human as player 0
        self.ihuman = 0
        human_id = 'human' + "_0"
        self.add_player(human_id, idx=0, is_human=True)

        self.write_data = False
        self.is_tutorial = 'tutorial' in layout.lower()
        self.is_frozen = False # Whether the game is frozen (i.e. no actions are being applied)

    def is_ready(self):
        """
        Game is ready to be activated if there are a sufficient number of players and at least one human (spectator or player)
        """
        # server_ready = super(OvercookedGame, self).is_ready() and not self.is_empty()
        # return  server_ready and self.client_ready
        return super(OvercookedGame, self).is_ready() and not self.is_empty() and not self.is_frozen

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
                    # raise IOError("No action found in queue for HUMAN")
                    pass
            else:
                # we block on agent actions to ensure that the agent gets to do one action per state
                joint_action[i] = self.pending_actions[i].get(block=True)


        # print(joint_action, file=sys.stderr)
        # Apply overcooked game logic to get state transition
        prev_state = self.state
        self.state, info = self.mdp.get_state_transition(
            prev_state, joint_action
        )

        # Send next state to all background consumers if needed
        if self.curr_tick % self.ticks_per_ai_action == 0:
            for npc_id in self.npc_policies:
                self.npc_state_queues[npc_id].put(self.state, block=False)

        # Update score based on soup deliveries that might have occured
        curr_reward = sum(info["sparse_reward_by_agent"]) if not self.is_tutorial else info["sparse_reward_by_agent"][self.ihuman]
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
            "layout_name": self.layout,
            "trial_id": str(self.start_time),
            "player_0_id": self.players[0],
            "player_1_id": self.players[1],
            "player_0_is_human": self.players[0] in self.human_players,
            "player_1_is_human": self.players[1] in self.human_players,
        }

        self.trajectory.append(transition)

        # Return about the current transition
        return prev_state, joint_action, info

    def get_policy(self,npc_id, idx=0):
        if npc_id.lower() == "rs-tom":
            try:
                agent = ToMAI(['averse', 'neutral', 'seeking'])
                # agent.activate(self.mdp)
                return agent
            except Exception as e:
                raise IOError("Error loading Agent\n{}".format(e.__repr__()))


        elif npc_id.lower() == "rational":
            try:
                agent = ToMAI(['neutral'])
                # agent.activate(self.mdp)
                return agent
            except Exception as e:
                raise IOError("Error loading Agent\n{}".format(e.__repr__()))

        # TUTORIALS
        elif  npc_id.lower() == "tutorialai":
            try:
                tut_num = int(self.layout[-1])
                return TutorialAI(tutorial_phase = tut_num)
            except Exception as e:
                raise IOError("Error loading Agent\n{}".format(e.__repr__()))
        else:
            raise IOError("Error loading UKNOWN Agent={}".format(npc_id))

    def reset(self):
        raise NotImplementedError("Deprecated.")
        # status = super(OvercookedGame, self).reset()
        # if status == self.Status.RESET:
        #     # Hacky way of making sure game timer doesn't "start" until after reset timeout has passed
        #     self.start_time += self.reset_timeout / 1000
    def needs_reset(self):
        return False
        # raise NotImplementedError("Deprecated.")
        # return False
        # return self._curr_game_over() and not self.is_/finished()

    def is_finished(self):
        if self.is_tutorial:
            # Tutorial games are never finished
            return self.score > 0
        return time() - self.start_time >= self.max_time

    def render(self):
        """
        Called instead of self.activate for rendering image of game
        """
        super(OvercookedGame, self).activate()

        # Sanity check at start of each game
        if not self.npc_players.union(self.human_players) == set(self.players):
            raise ValueError("Inconsistent State")

        self.curr_layout = self.layouts.pop()
        self.mdp = OvercookedGridworld.from_layout_name(
            self.curr_layout, **self.mdp_params
        )

        if self.debug:
            print(f'\n\nActivating OvercookedGame...')
            print("\tLayout: {}".format(self.curr_layout))
            print("\tp_slip: {}".format(self.mdp.p_slip))
            print("\tWrite data: {}".format(self.write_data))

        for key, val in self.npc_policies.items():
            if isinstance(val, ToMAI):
                self.npc_policies[key].activate(self.mdp)

        self.state = self.mdp.get_standard_start_state()

        self.start_time = time()
        self.curr_tick = 0
        self.score = 0
        self.threads = []

        # obj_dict = {}
        # obj_dict["terrain"] = self.mdp.terrain_mtx if self._is_active else None
        # obj_dict["state"] = self.get_state() if self._is_active else None
        return self.to_json()

    def view_action(self):
        joint_action = [Action.STAY] * len(self.players)

        # Synchronize individual player actions into a joint-action as required by overcooked logic
        for i in range(len(self.players)):
            # if this is a human, don't block and inject
            try:
                # we don't block here in case humans want to Stay
                if not self.pending_actions[i].empty():
                    joint_action[i] = self.pending_actions[i].queue[0]
            except Empty:
                pass

            return joint_action

    def tick(self):
        for key, val in self.npc_policies.items():
            if isinstance(val, ToMAI):
                human_action,ai_action = self.view_action()
                self.npc_policies[key].observe(self.state,human_action,ai_action)

        return super(RiskyOvercookedGame, self).tick()
class Experiment:
    #TODO: Switch to predefined config

    TRIALS = [
        # {'layout': 'risky_coordination_ring','p_slip': 0.4},
        {'layout': 'risky_multipath', 'p_slip': 0.4},
        {'layout': 'risky_multipath', 'p_slip': 0.4}
    ]

    CONDITIONS = {
        0: ['rs-tom', 'rational'],  # infers risk-sensitivity vs assumes they are rational
        1: ['rational','rs-tom']  # infers risk-sensitivity vs assumes they are rational
    }


    #### INITIALIZATION #####################################################################

    def __init__(self, uid, **kwargs):
        if DEBUG:
            print(f"Creating Experiment:")
            print(f"\t| ID: {uid}")
            for key, val in kwargs.items():
                print(f"\t| {key}: {val}")
        self.id = uid # used for internal client id
        self.prolific_id = kwargs.get('prolific_id','None' ) # the id used by prolific
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # timestamp of experiment start

        self.layouts, self.p_slips, self.partners = self.sample_condition()
        self.trial_params =list(zip(self.layouts, self.p_slips, self.partners))
        self.n_trials = len(self.layouts) # number of trials in the experiment


        self.stages = self.init_stage_data() # initialize data logging
        self.condition = None # assigned when sampling condition
        self.robot = None # loaded on trial begin
        self.game = None # loads when game is created
        self.lock = Lock()

        # TODO: Implement these attributes
        self.data_collection = kwargs.get("dataCollection", True)  # whether to collect data

    def sample_condition(self,all_games=True):
        """ Randomly sample a trial from the TRIALS list. """
        layouts = []
        p_slips = []
        partner_types = []

        if all_games:
            """ Each partner will play all layouts."""
            icond = np.random.randint(0,2) # 0 or 1
            self.condition = self.CONDITIONS[icond]
            for partner in self.condition:
                trials = copy.deepcopy(self.TRIALS)
                np.random.shuffle(trials)
                for trial in trials:
                    layouts.append(trial['layout'])
                    p_slips.append(trial['p_slip'])
                    partner_types.append(partner)
        else: raise NotImplementedError("Only all_games=True is implemented.")

        return layouts,p_slips,partner_types

    def init_stage_data(self):
        STAGES ={}
        tutorial_p_slip = 0.0
        partner = 'Tutorial'
        # Pretrial --------------
        STAGES['consent'] = DummyData()
        STAGES['participant_information'] = DemographicData()
        # STAGES['demographic'] = DemographicData()
        STAGES['demographic'] = SurveyData('demographic')
        STAGES['risk_propensity'] = SurveyData('risk_propensity')
        STAGES['instructions'] = DummyData()
        STAGES['risky_tutorial_0'] = InteractionData('tutorial0',tutorial_p_slip,partner)
        STAGES['risky_tutorial_1'] = InteractionData('tutorial1',tutorial_p_slip,partner)
        STAGES['risky_tutorial_2'] = InteractionData('tutorial2',tutorial_p_slip,partner)
        STAGES['risky_tutorial_3'] = InteractionData('tutorial3',tutorial_p_slip,partner)

        # Trials --------------
        for i in range(int(self.n_trials/2)):
            STAGES[f'priming{i}'] = SurveyData(f'priming{i}')
            STAGES[f'game{i}'] = InteractionData(self.layouts[i],self.p_slips[i],self.partners[i])
            STAGES[f'trust_survey{i}'] =  SurveyData(f'trust_survey{i}')

        STAGES[f'washout'] = DummyData()

        for i in range(int(self.n_trials/2),self.n_trials):
            STAGES[f'priming{i}'] = SurveyData(f'priming{i}')
            STAGES[f'game{i}'] = InteractionData(self.layouts[i], self.p_slips[i], self.partners[i])
            STAGES[f'trust_survey{i}'] = SurveyData(f'trust_survey{i}')

        # Posttrial --------------
        STAGES['relative_trust_survey'] =  SurveyData(f'relative_trust_survey')
        STAGES['debriefing'] = DummyData()
        STAGES['redirected'] = DummyData()

        return STAGES


    #### INTERFACE ##########################################################################
    def log_transition(self):
        pass

    def log_survey(self,name, response_dict):
        assert name in self.stages.keys(), f"Survey Stage {name} not found in stages."
        self.stages[name].set_responses(response_dict)
        self.update_stage(name, True)
        if DEBUG:
            print(f"Logging {name} data:")
            for key, val in response_dict.items():
                print(f"\t| {key}: {val}")
        # if name == 'demographic':
        #     self.data.pretrial.set_demographic(**response_dict)
        # elif name == 'risk_propensity':
        #     self.data.pretrial.set_risk_propensity(response_dict)


    def update_stage(self,stage,val):
        """ Update the current stage of the experiment. """
        if stage in self.stages.keys(): self.stages[stage].complete = val
        else:  raise ValueError(f"Stage {stage} not found in stages.")

    def verify(self):
        """ Verify that all stages are complete. """
        for key,val in self.stages.items():
            if not val:
                raise ValueError(f"Stage {key} is not complete.")

        self.data.verify()

    #### GAMESTATE MANAGEMENT ###############################################################
    def open_game(self,**kwargs):
        assert self.game is None, 'Experiment already exists'
        if DEBUG: print("Opening game with params: ", kwargs)
        # try:
        name = kwargs.get('name', 'Was not defined...')

        if 'tutorial' in name:
            self.game = RiskyOvercookedGame(name,'TutorialAI',**kwargs)
            # self.game = self.load_tutorial()
        elif 'game' in name:
            trial_idx = int(kwargs['name'].split('game')[1])
            layout, p_slip, partner_type = self.trial_params[trial_idx]
            self.game = RiskyOvercookedGame(layout,partner_type,**kwargs)
        elif 'priming' in name:
            trial_idx = int(kwargs['name'].split('priming')[1])
            layout, p_slip, partner_type = self.trial_params[trial_idx]
            self.game = RiskyOvercookedGame(layout, partner_type, **kwargs)
            self.game.is_frozen = True


        self.game.id = self.id  # Set the game id to the experiment id
        return self.game

    def close_game(self):
        assert self.game is not None, 'Attempting to close a nonexistant game'
        with self.game.lock:
            data = self.game.get_data()
            socketio.emit(
                "end_game", {"status": 'closing', "data": data}, room=self.game.id
            )

            # if status != Game.Status.INACTIVE:
            #     game.deactivate()
            self.game.deactivate()
            cleanup_game(self.game.id)

            del self.game
            self.game = None
        # with self.lock:
        #     socketio.emit(
        #         "end_game", {"status": 'close', "data": {}}, room=self.game.id
        #     )
        #     self.game.deactivate()
        #     del self.game
        #     self.game = None


    def close_experiment(self):

        # if FREE_MAP[game.id]:
        #     raise ValueError("Double free on a game")
        #
        #     # User tracking
        # for user_id in game.players:
        #     leave_curr_room(user_id)
        #
        #     # Socketio tracking
        # socketio.close_room(game.id)
        # # Game tracking
        # FREE_MAP[game.id] = True
        # FREE_IDS.put(game.id)
        # del GAMES[game.id]
        #
        # if game.id in ACTIVE_GAMES:
        #     ACTIVE_GAMES.remove(game.id)
        pass
    #### SAVE/LOAD ##########################################################################
    def load_model(self,layout,p_slip,partner_type):
        """ Load the model for the given layout, p_slip, and partner_type. """
        # This is a placeholder function. You need to implement the actual loading logic.
        pass

    def save_data(self):
        """ Save the experiment data to a file. """
        # This is a placeholder function. You need to implement the actual saving logic.
        pass


    #### PROPERTIES AND SUCH ################################################################

    @property
    def current_stage(self):
        for key,val in self.stages.items():
            if not val.complete:
                return key
    @property
    def current_game(self):
        raise NotImplementedError("Not implemented yet.")

def play_game(game: RiskyOvercookedGame, fps=6):
    """
    Asynchronously apply real-time game updates and broadcast state to all clients currently active
    in the game. Note that this loop must be initiated by a parallel thread for each active game

    game (Game object):     Stores relevant game state. Note that the game id is the same as to socketio
                            room id for all clients connected to this game
    fps (int):              Number of game ticks that should happen every second
    """
    # if DEBUG: print('Starting Play')
    # # status = Game.Status.ACTIVE
    # # while not game.is_finished:
    # while not game.is_finished():
    #     # Advance Game
    #     with game.lock:
    #         status = game.tick()
    #
    #     # if DEBUG: print(f'\t |Playing game {status}')
    #     socketio.emit(
    #         "state_pong", {"state": game.get_state()},
    #         room=game.id
    #     )
    #     socketio.sleep(1 / fps)
    #
    # if DEBUG: print('\t | Game finished...')
    # with game.lock:
    #     data = game.get_data()
    #     socketio.emit(
    #         "end_game", {"status": status, "data": data},
    #         room=game.id
    #     )
    #
    #     if status != Game.Status.INACTIVE:
    #         game.deactivate()
    #     # cleanup_game(game)

########################
    status = Game.Status.ACTIVE
    if DEBUG: print('Starting Play')
    while status != Game.Status.DONE and status != Game.Status.INACTIVE:

        # if DEBUG: print(f'\t |Playing game {status}')
        with game.lock:
            status = game.tick()

        socketio.emit(
            "state_pong", {"state": game.get_state()}, room=game.id
        )
        socketio.sleep(1 / fps)

    # if DEBUG: print('\t | Game finished...')
    with game.lock:
        data = game.get_data()
        socketio.emit(
            "end_game", {"status": status, "data": data}, room=game.id
        )

        # if status != Game.Status.INACTIVE:
        #     game.deactivate()
        game.deactivate()
        cleanup_game(game.id)


if __name__ == "__main__":
    # Dynamically parse host and port from environment variables (set by docker build)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 80))

    # Attach exit handler to ensure graceful shutdown
    atexit.register(on_exit)

    # https://localhost:80 is external facing address regardless of build environment
    socketio.run(app, host=host, port=port, log_output=app.config["DEBUG"])
