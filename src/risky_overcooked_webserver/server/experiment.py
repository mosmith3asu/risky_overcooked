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
from threading import Lock

import queue
# from .data_logging import ExperimentData
from .data_logging import DummyData, DemographicData,SurveyData, InteractionData
from utils import ThreadSafeDict, ThreadSafeSet
from datetime import datetime

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
    for i in range(MAX_EXPERIMENTS): # Initialize our ID tracking data
        FREE_IDS.put(i)
        FREE_MAP[i] = True

    EXPERIMENTS = ThreadSafeDict()     # Mapping of game-id to game objects
    ACTIVE_EXPERIMENTS = ThreadSafeSet()     # Set of games IDs that are currently being played
    # WAITING_EXPERIMENTS = queue.Queue()  # Queue of games IDs that are waiting for additional players to join. Note that some of these IDs might be stale (i.e. if FREE_MAP[id] = True)
    USERS = ThreadSafeDict()  # Mapping of users to locks associated with the ID. Enforces user-level serialization
    # USER_ROOMS = ThreadSafeDict()     # Mapping of user id's to the current game (room) they are in

    # Mapping of string game names to corresponding classes
    # GAME_NAME_TO_CLS = {
    #     # "overcooked": CustomOvercookedGame,
    #     "overcooked": OvercookedGame,
    #     "tutorial": OvercookedTutorial,
    # }
    #
    # game._configure(MAX_GAME_LENGTH, AGENT_DIR)


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
def try_create_experiment(*args,**kwargs):
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
        experiment = Experiment(curr_id, **kwargs)
    except queue.Empty:
        err = RuntimeError("Server at max capacity")
        return None, err
    except Exception as e:
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

def get_experiment(game_id):
    return EXPERIMENTS.get(game_id, None)


def get_curr_experiment(user_id):
    return get_experiment(user_id)
    # return get_experiment(get_curr_room(user_id))



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
        emit("creation_failed", {"error": err.__repr__()})
        return

    # spectating = True
    # with experiment.lock:
        # if not experiment.is_full():
        #     spectating = False
        #     game.add_player(user_id)
        # else:
        #     spectating = True
        #     game.add_spectator(user_id)
        # join_room(game.id)
        # set_curr_room(user_id, game.id)

        # WAITING_GAMES.put(game.id)
        # emit("waiting", {"in_game": True}, room=game.id)

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
        active_games.append({"id": experiment_id, "state": experiment.to_json()})

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
    # # this params file should be a dictionary that can have these keys:
    # # playerZero: human/Rllib*agent
    # # playerOne: human/Rllib*agent
    # # layout: one of the layouts in the config file, I don't think this one is used
    # # gameTime: time in seconds
    # # oldDynamics: on/off
    # # dataCollection: on/off
    # # layouts: [layout in the config file], this one determines which layout to use, and if there is more than one layout, a series of game is run back to back
    # #
    #
    # use_old = False
    # if "oldDynamics" in params and params["oldDynamics"] == "on":
    #     params["mdp_params"] = {"old_dynamics": True}
    #     use_old = True
    #
    # if "dataCollection" in params and params["dataCollection"] == "on":
    #     # config the necessary setting to properly save data
    #     params["dataCollection"] = True
    #     mapping = {"human": "H"}
    #     # gameType is either HH, HA, AH, AA depending on the config
    #     gameType = "{}{}".format(
    #         mapping.get(params["playerZero"], "A"),
    #         mapping.get(params["playerOne"], "A"),
    #     )
    #     params["collection_config"] = {
    #         "time": datetime.today().strftime("%Y-%m-%d_%H-%M-%S"),
    #         "type": gameType,
    #     }
    #     if use_old:
    #         params["collection_config"]["old_dynamics"] = "Old"
    #     else:
    #         params["collection_config"]["old_dynamics"] = "New"
    #
    # else:
    #     params["dataCollection"] = False
    raise NotImplementedError("Deprecated.")

@socketio.on("join_experiment")
def on_join_experiment(data):
    """ Participants clicks [Join] button on landing page"""

    if DEBUG: print("join triggered")

    user_id = request.sid
    _create_user(user_id)

    # create experiment
    with USERS[user_id]:
        if get_curr_experiment(user_id):
            return  # TODO: Check if user is in a experiment that was closed out of

        # Create game if not previously in one
        params = data.get("params", {})
        if DEBUG: print("\t| params", params)

        creation_params(params)
        _create_experiment(user_id, params)
        return

@socketio.on("ready_game")
def on_ready_game(data):
    """ Participants clicks [Begin Game] button"""

    if DEBUG: print("ready_game triggered")

    user_id = request.sid

    with USERS[user_id]:
        curr_experiment = get_curr_experiment(user_id)
        curr_game = curr_experiment.game

        if curr_game is None:
            print('Game not initialized')
            return

        curr_game.client_ready = True

        if curr_game:
            if curr_game.is_ready():
                curr_game.activate()
                # ACTIVE_GAMES.add(curr_game.id)
                emit(
                    "start_game",
                    {"start_info": curr_game.to_json()},
                )
                socketio.start_background_task(curr_game.play_game, curr_game, fps=6)
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
        curr_experiment.log_survey(**data)


@socketio.on("on_complete_stage")
def on_complete_stage(data):
    if DEBUG: print("on_complete_stage triggered", data["stage"])
    user_id = request.sid
    with USERS[user_id]:
        curr_experiment = get_curr_experiment(user_id)
        curr_experiment.complete_stage(data["stage"])

#########################################################################################
#### EXPERIMENT MANAGER #################################################################
#########################################################################################
class Game:
    def __init__(self):
        pass


class Experiment:
    #TODO: Switch to predefined config

    TRIALS = [
        {'layout': 'risky_coordination_ring','p_slip': 0.4},
        {'layout': 'risky_multipath', 'p_slip': 0.15}
    ]

    CONDITIONS = {
        0: ['rstom', 'rational'],  # infers risk-sensitivity vs assumes they are rational
        1: ['rational','rstom']  # infers risk-sensitivity vs assumes they are rational
    }


    #### INITIALIZATION #####################################################################

    def __init__(self, uid,prolific_id):
        self.id = id # used for internal client id
        self.prolific_id = prolific_id # the id used by prolific
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # timestamp of experiment start

        self.layouts, self.p_slips, self.partners = self.sample_condition()
        self.n_trials = len(self.layouts) # number of trials in the experiment


        self.stages = self.init_stage_data() # initialize data logging
        self.condition = None # assigned when sampling condition
        self.robot = None # loaded on trial begin
        self.game = None # loads when game is created



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
        STAGES['demographic'] = DemographicData()
        STAGES['risk_propensity'] = SurveyData('risk_propensity')
        STAGES['instructions'] = DummyData()
        STAGES['tutorial0'] = InteractionData('tutorial0',tutorial_p_slip,partner)
        STAGES['tutorial1'] = InteractionData('tutorial1',tutorial_p_slip,partner)
        STAGES['tutorial2'] = InteractionData('tutorial2',tutorial_p_slip,partner)
        STAGES['tutorial3'] = InteractionData('tutorial3',tutorial_p_slip,partner)

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
        if name == 'demographic':
            self.data.pretrial.set_demographic(**response_dict)
        elif name == 'risk_propensity':
            self.data.pretrial.set_risk_propensity(response_dict)

    def complete_stage(self,stage):
        """ Update the current stage of the experiment. """
        if stage in self.stages.keys(): self.stages[stage].complete = True
        else:  raise ValueError(f"Stage {stage} not found in stages.")

    def verify(self):
        """ Verify that all stages are complete. """
        for key,val in self.stages.items():
            if not val:
                raise ValueError(f"Stage {key} is not complete.")

        self.data.verify()


    #### GAMESTATE MANAGEMENT ###############################################################
    def open_game(self,layout,p_slip,partner_type):
        # TODO: Make sure to try/except so server does not crash
        self.robot = self.load_model(layout,p_slip,partner_type)
        pass

    def close_game(self):
        self.robot = None
        pass

    def start_game(self,game_id):

        emit(
            "start_game",
            {"spectating": False, "start_info": self.game.to_json()},
            room=self.game.id,
        )

    def play_game(self,game: OvercookedGame, fps=6):
        """
        Asynchronously apply real-time game updates and broadcast state to all clients currently active
        in the game. Note that this loop must be initiated by a parallel thread for each active game

        game (Game object):     Stores relevant game state. Note that the game id is the same as to socketio
                                room id for all clients connected to this game
        fps (int):              Number of game ticks that should happen every second
        """
        status = Game.Status.ACTIVE
        while status != Game.Status.DONE and status != Game.Status.INACTIVE:
            with game.lock:
                status = game.tick()

            # if status == Game.Status.RESET:
            if status == Game.Status.RESET:
                with game.lock:
                    data = game.get_data()
                socketio.emit(
                    "reset_game",
                    {
                        "state": game.to_json(),
                        "timeout": game.reset_timeout,
                        "data": data,
                    },
                    room=game.id,
                )
                socketio.sleep(game.reset_timeout / 1000)
            else:
                socketio.emit(
                    "state_pong", {"state": game.get_state()}, room=game.id
                )
            socketio.sleep(1 / fps)

        with game.lock:
            data = game.get_data()
            socketio.emit(
                "end_game", {"status": status, "data": data}, room=game.id
            )

            if status != Game.Status.INACTIVE:
                game.deactivate()
            cleanup_game(game)

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
if __name__ == "__main__":
    main()
