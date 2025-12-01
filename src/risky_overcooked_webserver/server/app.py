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
from game import ToMAI,TutorialAI,StayAI, Game, OvercookedGame
import game as GAME
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from risky_overcooked_rl.utils.belief_update import BayesianBeliefUpdate
from risky_overcooked_rl.algorithms.DDQN.utils.agents import DQN_vector_feature
#########################################################################################
#### GLOBALS ############################################################################
#########################################################################################
DEBUG = False
# PRINTEND = '\n'
PRINTEND = ''

if True:
    # Read in global config
    CONF_PATH = os.getenv("CONF_PATH", "config.json")
    with open(CONF_PATH, "r") as f:
        CONFIG = json.load(f)

    LOGFILE = CONFIG["logfile"] # Where errors will be logged
    LAYOUTS = CONFIG["layouts"] # Available layout names
    AGENTS = CONFIG["AGENTS"]  # Path to where pre-trained agents will be stored on server
    EXPERIMENT_CONFIG = CONFIG["EXPERIMENT_CONFIG"]  # Path to where pre-trained agents will be stored on server

    LAYOUT_GLOBALS = CONFIG["layout_globals"]     # Values that are standard across layouts
    MAX_GAME_LENGTH = CONFIG["MAX_GAME_LENGTH"]    # Maximum allowable game length (in seconds)
    AGENT_DIR = CONFIG["AGENT_DIR"]    # Path to where pre-trained agents will be stored on server
    MAX_EXPERIMENTS = CONFIG["MAX_EXPERIMENTS"]  # Maximum number of games that can run concurrently. Contrained by available memory and CPU
    MAX_FPS = CONFIG["MAX_FPS"] # Frames per second cap for serving to client
    PREDEFINED_CONFIG = json.dumps(CONFIG["predefined"])     # Default configuration for predefined experiment
    TUTORIAL_CONFIG = json.dumps(CONFIG["tutorial"])     # Default configuration for tutorial

    EXPERIMENT_CONFIG = CONFIG["EXPERIMENT_CONFIG"]  # Default configuration for tutorial


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

        # print("Creating experiment with id {}".format(curr_id) if DEBUG else "")
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
def leave_curr_room(user_id):
    del USER_ROOMS[user_id]


#########################################################################################
# Socket Handler Helpers ################################################################
#########################################################################################
def _create_user(user_id):
    if user_id in USERS:
        return
    USERS[user_id] = Lock()
    if DEBUG: print(f'Created User: {user_id}')

def _delete_user(user_id):
    """
    Removes `user_id` from it's current game, if it exists. Rebroadcast updated game state to all
    other users in the relevant game.

    Leaving an active game force-ends the game for all other users, if they exist

    Leaving a waiting game causes the garbage collection of game memory, if no other users are in the
    game after `user_id` is removed
    """
    # Get pointer to current game if it exists
    experiment = get_curr_experiment(user_id)

    if not experiment:
        # Cannot leave a experiment if not currently in one
        return False

    # Acquire this experiment's lock to ensure all global state updates are atomic
    with experiment.lock:
        # Update socket state maintained by socketio
        leave_room(experiment.id)
        # Update user data maintained by this app
        leave_curr_room(user_id)

        # cleanup game if open
        game = experiment.game
        if game is not None:
            if user_id in game.players:
                game.remove_player(user_id)
            # else:
            #     game.remove_spectator(user_id)
            game.deactivate()
            experiment.close_game()



        # Whether the game was active before the user left
        was_active = experiment.id in ACTIVE_EXPERIMENTS


        # # Rebroadcast data and handle cleanup based on the transition caused by leaving
        # if was_active and open_game:
        #     # Active -> Empty
        #     experiment.game.deactivate()
        # elif open_game:
        #     # Waiting -> Empty
        #     cleanup_game(experiment.game)
        cleanup_experiment(experiment)

        # elif not was_active:
        #     # Waiting -> Waiting
        #     emit("waiting", {"in_game": True}, room=game.id)
        # elif was_active and game.is_ready():
        #     # Active -> Active
        #     pass
        # elif was_active and not game.is_empty():
        #     # Active -> Waiting
        #     game.deactivate()

    return was_active


def _leave_experiment(user_id):
    _delete_user(user_id)
    # raise NotImplementedError("Deprecated.")

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

@app.route("/server_full")
def server_full():
    # TODO: pass all of necessary config
    return render_template("server_full.html")


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

def print_server_info(header='',**kwargs):
    print(header + f'current server capacity:{len(EXPERIMENTS)}/{MAX_EXPERIMENTS}',**kwargs)
    print(header + f'free_ids={FREE_IDS.qsize()} ',**kwargs)
    print(header + f'free_map={len(FREE_MAP)}',**kwargs)
    print(header + f'active_experiments={len(ACTIVE_EXPERIMENTS)}',**kwargs)

@socketio.on("user_data")
def on_user_data(data):
    user_id = request.sid
    with USERS[user_id]:
        data = data['prolific_data']

        curr_experiment = get_curr_experiment(user_id)
        if curr_experiment is None:
            print("[BAD USER DATA REQUEST] No experiment found for user {}".format(user_id), file=sys.stderr)
        else:
            with curr_experiment.lock:
                curr_experiment.prolific_id = data.get("prolific_id", None)
                curr_experiment.study_id = data.get("study_id", None)
                curr_experiment.session_id = data.get("session_id", None)
    if DEBUG:
        print(f'\n User {user_id} provided data: {data}', file=sys.stderr,end=PRINTEND)
        print(f'\t| prolific_id: {curr_experiment.prolific_id}', file=sys.stderr,end=PRINTEND)
        print(f'\t| icond: {curr_experiment.icond}', file=sys.stderr)

@socketio.on("save")
def on_save(data):
    if DEBUG: print("on_save triggered", data)

    user_id = request.sid
    with USERS[user_id]:
        curr_experiment = get_curr_experiment(user_id)
        with curr_experiment.lock:
            curr_experiment.save_data()

@socketio.on("trigger_prolific_redirect")
def on_trigger_prolific_redirect(data):
    if DEBUG: print("on_save triggered", data)
    user_id = request.sid
    url = "https://app.prolific.com/submissions/complete?cc=C1NXVB3O"
    socketio.emit("redirect", {'url': url}, room=user_id)

@socketio.on("connect")
def on_connect():
    user_id = request.sid
    if user_id in USERS: return
    USERS[user_id] = Lock()

    """ Check if server is Full"""
    # if DEBUG:
    print(f'\nNew Connection: {user_id}', file=sys.stderr)
    print_server_info(header='\t| ', file=sys.stderr)


    if is_server_full():
        print(f'\t| Server is full, redirecting user {user_id} to server_full page.', file=sys.stderr)
        socketio.emit("redirect", {'url': '/server_full'}, room=user_id)
        return

    """ Try and create experiment """
    # user_id = request.sid
    _create_user(user_id)

    # create experiment
    with USERS[user_id]:
        if get_curr_experiment(user_id):
            raise IOError("User has in experiment")
            return  # TODO: Check if user is in a experiment that was closed out of

        # Create game if not previously in one
        params = {}
        # params = data.get("params", {})
        # if DEBUG: print("\t| params", params)

        # params = EXPERIMENT_CONFIG
        creation_params(params)
        _create_experiment(user_id, params)
        return

def is_server_full():
    return len(EXPERIMENTS) >= MAX_EXPERIMENTS

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

    # if DEBUG: print(f"create_game triggered: {data}" if DEBUG else "")

    user_id = request.sid

    with USERS[user_id]:
        curr_experiment = get_curr_experiment(user_id)

        if curr_experiment.game is None:
            print(f"Creating game for user {user_id} with data: {data}" if DEBUG else "")
            curr_experiment.open_game(**data)
        else:
            print(f"Found existing game for user {user_id}, Creating new game with data: {data}" if DEBUG else "")
            curr_experiment.close_game()  # Close any existing game before opening a new one
            curr_experiment.open_game(**data)

        curr_game = curr_experiment.game
        curr_game.client_ready = True

        if curr_game:
            if curr_game.is_ready():
                curr_game.activate()
                emit(
                    "start_game",
                    {"start_info": curr_game.to_json(), "is_priming": False},
                    room=curr_game.id,
                )
                socketio.start_background_task(play_game, curr_game, fps=6)

            elif curr_game.is_frozen: # just an image of priming stage
                # curr_game.activate()
                render_info = curr_game.render()
                emit(
                    "start_game",
                    {"start_info": render_info, "is_priming": True},
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
    """ Participants submits a survey response"""
    if DEBUG: print("survey_response triggered")
    user_id = request.sid
    with USERS[user_id]:
        curr_experiment = get_curr_experiment(user_id)

        # Double check no open games before logging survey data
        if curr_experiment.game is not None:
            curr_experiment.close_game()

        for key,responses in data.items():
            with curr_experiment.lock:
                curr_experiment.log_survey(key, responses)
                # curr_experiment.update_stage(key, True)

@socketio.on("update_stage")
def on_update_stage(data):
    if DEBUG: print("on_complete_stage triggered", data)

    # Joining Experiment
    for key,val in data.items():
        if key == 'join':
            # """ Participants clicks [Join] button on landing page"""
            # user_id = request.sid
            # _create_user(user_id)
            #
            # # create experiment
            # with USERS[user_id]:
            #     if get_curr_experiment(user_id):
            #         raise IOError("User has in experiment")
            #         return  # TODO: Check if user is in a experiment that was closed out of
            #
            #     # Create game if not previously in one
            #     params = data.get("params", {})
            #     if DEBUG: print("\t| params", params)
            #
            #     # params = EXPERIMENT_CONFIG
            #
            #     creation_params(params)
            #     _create_experiment(user_id, params)
            #     return
            return

        else:
            # All other updates
            user_id = request.sid
            with USERS[user_id]:
                curr_experiment = get_curr_experiment(user_id)
                with curr_experiment.lock:
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
    # if user_id not in USERS:
    #     return
    # with USERS[user_id]:
    #     _leave_experiment(user_id)
    #
    # del USERS[user_id]
    if user_id  in USERS:
        with USERS[user_id]:
            _leave_experiment(user_id)
        del USERS[user_id]

    print_server_info(header='\t| ', file=sys.stderr)

@socketio.on("leave")
def on_leave(data):
    raise NotImplementedError("Deprecated.")

@socketio.on("close_game")
def on_close_game(data):
    if DEBUG: print(f'Closing game...', end='')
    user_id = request.sid
    with USERS[user_id]:
        curr_experiment = get_curr_experiment(user_id)
        with curr_experiment.lock:
            # _close_hanging_games(curr_experiment)
            curr_experiment.close_game()
            curr_experiment.update_stage(curr_experiment.current_stage, True)
    # if DEBUG: print(f'game={curr_experiment.game}')
    # raise NotImplementedError("Deprecated.")

@socketio.on("complete_experiment")
def on_complete_experiment(data):
    user_id = request.sid
    with USERS[user_id]:
        curr_experiment = get_curr_experiment(user_id)
        if DEBUG: print(user_id, 'Closing experiment')

        with curr_experiment.lock:
            curr_experiment.close_experiment()


@socketio.on("request_stages")
def on_request_stages(data):
    user_id = request.sid
    # raise NotImplementedError("Deprecated.")
    STAGE_NAMES = []
    for key in EXPERIMENT_CONFIG['stages'].keys():
        if 'game_loop' in key:
            n = int(key[-1])
            _isurv = 0
            for i in range(n * len(LAYOUTS), (n + 1) *  len(LAYOUTS)):
                STAGE_NAMES.append(f'priming{i}')
                STAGE_NAMES.append(f'game{i}' )
                # STAGE_NAMES.append(f'trust_survey{i}' )
                if _isurv == 0 or _isurv == 3:
                    STAGE_NAMES.append(f'AC_trust_survey{i}')  # attention check
                else:
                    STAGE_NAMES.append(f'trust_survey{i}')
                _isurv += 1
        else:
            STAGE_NAMES.append(key)

    # STAGE_NAMES = [key for key in EXPERIMENT_CONFIG['stages'].keys()]
    socketio.emit('stage_data', {'stages':STAGE_NAMES,'debug':  EXPERIMENT_CONFIG['debug']}, room=user_id)

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
                 p_slip,
                 **kwargs
                 ):
        self.partner_color = kwargs.get("partner_color", 'green') # changes color
        self.layout = layout
        self.p_slip = p_slip
        kwargs['layouts'] = [layout]

        super(RiskyOvercookedGame, self).__init__(playerOne=partner_name,
                                                  max_game_time = EXPERIMENT_CONFIG.get('game_length',70),
                                                  gameTime = EXPERIMENT_CONFIG.get('game_length',70),
                                                  **kwargs)
        self.mdp_params = kwargs.get("mdp_params", {'neglect_boarders':True})
        self.id = kwargs.get("id", id(self))

        # Add human as player 0
        # self.ihuman = 0
        # human_id = 'human' + "_0"
        # self.add_player(human_id, idx=0, is_human=True)
        self.inpc = AGENTS['npc_player']
        self.ihuman = AGENTS['human_player']
        human_id = 'human' + f"_{self.ihuman}"
        self.add_player(human_id, idx=self.ihuman, is_human=True,buff_size=2) # ,buff_size=3

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

    def enqueue_action(self, player_id, action):
        """
        Add (player_id, action) pair to the pending action queue, without modifying underlying game state

        Note: This function IS thread safe
        """
        action = self.action_to_overcooked_action[action]

        if not self.is_active:
            # Could run into issues with is_active not being thread safe
            return
        if player_id not in self.players:
            # Only players actively in game are allowed to enqueue actions
            return
        try:
            player_idx = self.players.index(player_id)
            if self.pending_actions[player_idx].full():
                _ = self.pending_actions[player_idx].get(block=False)
            self.pending_actions[player_idx].put(action)
        except Full:
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
                    # while not self.pending_actions[i].empty():
                    #     joint_action[i] = self.pending_actions[i].get(block=False)
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
        can_slip = [self.mdp.check_can_slip(prev_state.players[i], self.state.players[i]) for i in range(2)]
        events = info['event_infos']
        # for key, val in info['event_infos'].items():
        #     print(f"{key}: {val}")
        did_slip = np.array(events['onion_slip']) + np.array(events['dish_slip']) + np.array(events['soup_slip'])
        did_slip = did_slip.tolist()

        # Send next state to all background consumers if needed
        if self.curr_tick % self.ticks_per_ai_action == 0:
            for npc_id in self.npc_policies:
                self.npc_state_queues[npc_id].put(self.state, block=False)

        # Update score based on soup deliveries that might have occured
        curr_reward = sum(info["sparse_reward_by_agent"]) if not self.is_tutorial else info["sparse_reward_by_agent"][self.ihuman]
        self.score += curr_reward

        for key, val in self.npc_policies.items():
            belief = str(self.npc_policies[key].belief) if isinstance(val, ToMAI) else str([1])


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
            "belief": belief,
            'can_slip': can_slip,
            'did_slip': did_slip,
        }

        self.trajectory.append(transition)


        # Return about the current transition
        return prev_state, joint_action, info

    def get_policy(self,npc_id, idx=0):
        if npc_id.lower() == "rs-tom":
            try:
                # agent = StayAI()
                agent = ToMAI(['averse', 'neutral', 'seeking'])
                # # agent.activate(self.mdp)
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
            raise ValueError(f"Inconsistent State {self.npc_players} {self.human_players}\\neq {self.players}")

        # self.curr_layout = self.layouts.pop()
        # self.mdp = OvercookedGridworld.from_layout_name(
        #     self.curr_layout, **self.mdp_params
        # )
        self.mdp_params["p_slip"] = self.p_slip
        self.mdp = OvercookedGridworld.from_layout_name(
            self.layout, **self.mdp_params
        )

        if self.debug:
            print(f'Activating OvercookedGame...\t',end='')
            print("\tLayout: {}".format(self.layout),end='')
            print("\tp_slip: {}".format(self.mdp.p_slip),end='')
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
                # self.npc_policies[key].observe(self.state,human_action,ai_action)
                # ai_action,human_action = self.view_action()
                self.npc_policies[key].observe(self.state, human_action, ai_action)

        return super(RiskyOvercookedGame, self).tick()

    def activate(self):
        # super(OvercookedGame, self).activate()
        self._is_active = True

        # Sanity check at start of each game
        if not self.npc_players.union(self.human_players) == set(self.players):
            raise ValueError("Inconsistent State")

        self.curr_layout = self.layouts.pop()
        if self.p_slip != 'default' and self.p_slip != 'def':
            self.mdp_params['p_slip'] = self.p_slip
        self.mdp = OvercookedGridworld.from_layout_name(
            self.curr_layout, **self.mdp_params
        )
        for key, val in self.npc_policies.items():
            if isinstance(val, ToMAI):
                # If the policy is a DQN vector feature policy, we need to activate it
                self.npc_policies[key].activate(self.mdp)

        if self.debug:
            print(f'\n [Activating OvercookedGame]', end='')
            print("\tLayout: {}".format(self.curr_layout), end='')
            print("\tp_slip: {}".format(self.mdp.p_slip), end='')
            print("\tWrite data: {}".format(self.write_data), end='')

        for key, val in self.npc_policies.items():
            if isinstance(val, ToMAI):
                self.npc_policies[key].activate(self.mdp)

                if self.debug:
                    print("\tCanidates...")
                    for can_fname in self.npc_policies[key].candidate_fnames:
                        print("\t\t{}".format(can_fname))

        # if self.show_potential:
        #     self.mp = MotionPlanner.from_pickle_or_compute(
        #         self.mdp, counter_goals=NO_COUNTERS_PARAMS
        #     )
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

    def to_json(self):
        obj_dict = {}
        obj_dict["terrain"] = self.mdp.terrain_mtx if self._is_active else None
        obj_dict["state"] = self.get_state() if self._is_active else None
        obj_dict["layout"] = self.layout if self._is_active else None
        obj_dict["p_slip"] = self.p_slip if self._is_active else None
        obj_dict['player_colors'] = {0:'blue', 1: self.partner_color}

        # print(obj_dict["terrain"])
        # print(obj_dict["state"])
        #
        # obj_dict["terrain"] = np.replace(obj_dict["terrain"], 2, self.partner_id, obj_dict["terrain"])

        return obj_dict


class Experiment:

    TRIALS = []
    for layout_str in LAYOUTS:
        p_slip = float('0.'+layout_str.split('pslip')[-1][1:])
        layout = layout_str.split('_')[:-1]
        layout = '_'.join(layout)
        TRIALS.append({'layout': layout, 'p_slip': p_slip})

    CONDITIONS = {
        0: ['rs-tom', 'rational'],  # infers risk-sensitivity vs assumes they are rational
        1: ['rational','rs-tom']  # infers risk-sensitivity vs assumes they are rational
    }


    #### INITIALIZATION #####################################################################

    def __init__(self, uid, **kwargs):
        self.id = uid # used for internal client id
        self.prolific_id = kwargs.get('prolific_id','NoProlificID' ) # the id used by prolific
        self.study_id = kwargs.get('study_id', 'NoStudyID') # the id used by the study
        self.session_id = kwargs.get('session_id', 'NoSessionID') # the id used by the session
        self.icond = None
        self.condition = None  # assigned when sampling condition

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # timestamp of experiment start


        colors = ['orange', 'purple']
        np.random.shuffle(colors)
        self.partner_colors = {'rs-tom':colors[0], 'rational':colors[1]}
        self.layouts, self.p_slips, self.partners = self.sample_condition()
        self.trial_params =list(zip(self.layouts, self.p_slips, self.partners))

        self.n_trials = len(self.layouts) # number of trials in the experiment

        self.stages = self.init_stage_data() # initialize data logging
        self.stage_names = list(self.stages.keys())
        self.stage_tstart = time()
        self.current_stage = self.stage_names[0]
        self.stage_idx = -1


        self.robot = None # loaded on trial begin
        self.game = None # loads when game is created
        self.lock = Lock()

        # TODO: Implement these attributes
        self.data_collection = True #kwargs.get("dataCollection", True)  # whether to collect data

        if DEBUG:
            print(f"\nInstantiating Experiment:")
            print(f"\t| ID: {uid}")
            for key, val in kwargs.items():
                print(f"\t| {key}: {val}")
            print('\t| Trials:')
            for trial in self.trial_params:
                print(f"\t| \t|{trial}")

    def sample_condition(self,all_games=True):
        """ Randomly sample a trial from the TRIALS list. """
        layouts = []
        p_slips = []
        partner_types = []

        if all_games:
            """ Each partner will play all layouts."""
            self.icond = np.random.randint(0,2) # 0 or 1
            self.condition = self.CONDITIONS[self.icond]

            for i, partner in enumerate(self.condition):
                trials = copy.deepcopy(self.TRIALS)
                np.random.shuffle(trials)
                for trial in trials:
                    layouts.append(trial['layout'])
                    p_slips.append(trial['p_slip'])
                    partner_types.append(partner)
        else: raise NotImplementedError("Only all_games=True is implemented.")

        return layouts,p_slips,partner_types

    def init_stage_data(self):
        STAGES = {}


        for stage_name, data_type in EXPERIMENT_CONFIG["stages"].items():

            if 'DummyData'.lower() in str(data_type).lower():
                STAGES[stage_name] = DummyData()

            elif 'SurveyData' in data_type:
                STAGES[stage_name] = SurveyData(stage_name)

            elif 'participant_information' in stage_name.lower():
                STAGES[stage_name] = DemographicData()

            elif 'risky_tutorial' in stage_name.lower():
                STAGES[stage_name] = eval(data_type)(stage_name, 0.0, 'Tutorial')

            elif 'game_loop' in stage_name.lower():
                n = int(stage_name[-1])
                _isurv = 0
                for i in range(n*int(self.n_trials / 2), (n+1)*int(self.n_trials / 2)):
                    STAGES[f'priming{i}'] = SurveyData(f'priming{i}')
                    STAGES[f'game{i}'] = InteractionData(self.layouts[i], self.p_slips[i], self.partners[i])
                    if _isurv == 0:
                        STAGES[f'AC_trust_survey{i}'] = SurveyData(f'trust_survey{i}') # attention check
                    else:
                        STAGES[f'trust_survey{i}'] = SurveyData(f'trust_survey{i}')  # attention check
                    _isurv +=1
            else:
                print(f"Unknown stage type {data_type} for stage {stage_name}")


        # print('Emitting stages...')
        # socketio.emit('stage_data', {'stages':[key for key in STAGES.keys()]}, room=self.id)
        if DEBUG:
            print(f"Experiment {self.id} initialized with stages:")
            for key, val in STAGES.items():
                print(f"\t| {key}: {val.__class__.__name__}")
        return STAGES


    #### INTERFACE ##########################################################################
    def log_trajectory(self, trajectory):
        assert self.game is not None, 'Experiment has no game to log trajectory for'
        assert isinstance(self.stages[self.current_stage],InteractionData), \
            f"Invalid Trajectory Logging Stage[{self.current_stage}] = {self.stages[self.current_stage]}"

        if self.data_collection:
            for transition in trajectory:
                t = transition['cur_gameloop']
                s = transition['state']
                Ja = eval(transition['joint_action'])
                aH = Ja[self.game.ihuman] #transition['joint_action'][self.game.ihuman]
                aR = Ja[self.game.inpc] #transition['joint_action'][self.game.inpc]
                info = {'belief': transition['belief'],
                        'reward': transition['reward'],
                        'score': transition['score'],
                        'can_slip': transition['can_slip'],
                        'did_slip': transition['did_slip']
                        }
                self.stages[self.current_stage].log_transition(t, s, aH, aR, info)
            self.stages[self.current_stage].complete = True

    def log_survey(self,name, response_dict):
        if self.data_collection:
            assert name in self.stages.keys() or "AC_"+name in self.stages.keys(), f"Survey Stage {name} not found in stages."
            if name not in self.stages.keys():
                name = "AC_"+name
            self.stages[name].set_responses(response_dict)
            self.update_stage(name, True)

            if DEBUG: print(f"Logging {name} SURVEY data: {self.stages[name]}")


    def update_stage(self,stage,completed):
        """ Update the current stage of the experiment. """
        assert stage in self.stages.keys(), f"Stage {stage} not found in stages."
        assert stage == self.current_stage, \
            f"Attemptingg to update stage {stage} when currently on {self.current_stage}"

        self.stages[stage].duration = self.stage_duration
        self.stages[stage].complete = completed
        self.stage_idx = self.stage_names.index(stage) + (1 if completed else -1)
        self.current_stage = self.stage_names[self.stage_idx]
        self.stage_tstart = time()
        print(f'Stage [{stage}] completed: {completed} dur {self.stages[stage].duration:.2f}s')
        if DEBUG:
            print(f"\n\nUPDATING STAGE: [{stage} -> {self.current_stage}]")

    @property
    def stage_duration(self):
        return time() - self.stage_tstart

    def verify(self):
        """ Verify that all stages are complete. """
        for key,val in self.stages.items():
            if not val:
                raise ValueError(f"Stage {key} is not complete.")

        self.data.verify()

    #### GAMESTATE MANAGEMENT ###############################################################
    def open_game(self,**kwargs):
        assert self.game is None, 'Experiment already exists'
        # if DEBUG: print("Opening game with params: ", kwargs)
        # try:
        name = kwargs.get('name', 'Was not defined...')

        if 'tutorial' in name:
            self.is_priming = False
            pc = 'green'
            self.game = RiskyOvercookedGame(name,'TutorialAI',p_slip='default',partner_color=pc,**kwargs)
            # self.game = self.load_tutorial()
        elif 'game' in name:
            self.is_priming = False
            trial_idx = int(kwargs['name'].split('game')[1])
            layout, p_slip, partner_type = self.trial_params[trial_idx]
            if DEBUG: print(f'Opening game {name} with layout={layout}, p_slip={p_slip}, partner_type={partner_type}')
            pc = self.partner_colors[partner_type]
            self.game = RiskyOvercookedGame(layout,partner_type,p_slip,partner_color=pc,**kwargs)

        elif 'priming' in name:
            self.is_priming =True
            trial_idx = int(kwargs['name'].split('priming')[1])
            layout, p_slip, partner_type = self.trial_params[trial_idx]
            pc = self.partner_colors[partner_type]
            self.game = RiskyOvercookedGame(layout, partner_type, p_slip,partner_color=pc, **kwargs)
            self.game.is_frozen = True


        self.game.id = self.id  # Set the game id to the experiment id
        return self.game

    def close_game(self):
        assert self.game is not None, 'Attempting to close a nonexistant game'

        with self.game.lock:
            if "priming" not in self.current_stage.lower():
                self.log_trajectory(self.game.trajectory)
            self.game.deactivate()
            cleanup_game(self.game.id)
            del self.game
            self.game = None


    def close_experiment(self):
        self.save_data()
        leave_curr_room(self.id)

        # Socketio tracking
        socketio.close_room(self.id)
        # Game tracking
        FREE_MAP[self.id] = True
        FREE_IDS.put(self.id)
        del EXPERIMENTS[self.id]

        if self.id in ACTIVE_EXPERIMENTS:
            ACTIVE_EXPERIMENTS.remove(self.id)

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
    # def load_model(self,layout,p_slip,partner_type):
    #     """ Load the model for the given layout, p_slip, and partner_type. """
    #     # This is a placeholder function. You need to implement the actual loading logic.
    #     pass

    def save_data(self):
        """ Save the experiment data to a file. """
        data_path = self.create_dirs()
        fname = f'{self.timestamp}__PID{self.prolific_id}__cond{self.icond}.pkl'
        fpath = os.path.join(data_path, fname)

        with open(fpath, "wb") as f:
            pickle.dump(self.stages, f)
        print(f'Saving data to {fpath}')
        # if DEBUG:
        #     # print(f"Saving data to {os.path.join(data_path, fname)}")
        #     print_link(fpath)
        # else:
        #

    # def create_dirs(self, DOCKER_VOLUME = "\\app\\data"):
    #     participant_folder = f'cond_{self.icond}'
    #     path = os.path.join(
    #         DOCKER_VOLUME,
    #         participant_folder
    #     )
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     return path
    def create_dirs(self, DOCKER_VOLUME="/app/data"):
        participant_folder = f'cond_{self.icond}'
        path = os.path.join(
            DOCKER_VOLUME,
            participant_folder
        )
        if not os.path.exists(path):
            os.makedirs(path)
        return path



    #### PROPERTIES AND SUCH ################################################################

    # @property
    # def current_stage(self):
    #     for key,val in self.stages.items():
    #         if not val.complete:
    #             return key
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

########################
    # fpss = deque(maxlen=10)
    status = Game.Status.ACTIVE
    if DEBUG: print('Starting Play')
    step = 0
    while status != Game.Status.DONE and status != Game.Status.INACTIVE:
        tstart = time()
        step += 1
        with game.lock:
            status = game.tick()
        socketio.emit(
            "state_pong", {"state": game.get_state()}, room=game.id
        )

        tdur = time() - tstart

        socketio.sleep(1 / fps - tdur)

        # fpss.append(1 / (time()-tstart))
        # print(f'Step {step} fps = {np.mean(fpss):.4f}')

    with game.lock:
        data = game.get_data(clear_trajectory=False)
        print(F'Traj Len = {len(game.trajectory)}/{step}')


        socketio.emit("end_game", {"status": status, "data": data}, room=game.id)

        # if status != Game.Status.INACTIVE:
        #     game.deactivate()
        game.deactivate()
        cleanup_game(game.id)
import inspect

def print_link(file=None, line=None):
    """ Print a link in PyCharm to a line in file.
        Defaults to line where this function was called. """
    if file is None:
        file = inspect.stack()[1].filename
    if line is None:
        line = inspect.stack()[1].lineno
    string = f'File "{file}", line {max(line, 1)}'.replace("\\", "/")
    print(string)
    return string

if __name__ == "__main__":
    # Dynamically parse host and port from environment variables (set by docker build)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 80))

    print("http://localhost?PROLIFIC_PID=123&STUDY_ID=456&SESSION_ID=789/")

    # print_link("http://localhost?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}/")


    # Attach exit handler to ensure graceful shutdown
    atexit.register(on_exit)

    # https://localhost:80 is external facing address regardless of build environment
    socketio.run(app, host=host, port=port, log_output=app.config["DEBUG"])
