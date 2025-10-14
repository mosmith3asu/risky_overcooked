import warnings
from dataclasses import dataclass, field, fields,is_dataclass
import os
import pickle

import matplotlib.pyplot as plt
import imageio
from risky_overcooked_py.mdp.overcooked_env import RiskyOvercooked, OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
import json
from risky_overcooked_rl.utils.visualization import TrajectoryVisualizer
######################################################################
############ BASE CLASSES ############################################
######################################################################

@dataclass
class DummyData:
    complete: bool = False

@dataclass
class Base:
    def verify(self,header_lvl=0,raise_error=True):
        invalid_fields = []
        for f in fields(self):
            name = f.name
            val = getattr(self, name)
            if val is None:
                invalid_fields.append(f'{name}:{val}')
            elif isinstance(val, str) and len(val) == 0:
                invalid_fields.append(f'{name}:{val}')
            elif isinstance(val, dict) and len(val.keys()) == 0:
                invalid_fields.append(f'{name}:{val}')
            elif isinstance(val, tuple) and len(val) == 0:
                invalid_fields.append(f'{name}:{val}')
            elif isinstance(val, list):
                if len(val) == 0:
                    invalid_fields.append(f'{name}:{val}')
                for i, v in enumerate(val):
                    if is_dataclass(v):
                        invalid_fields.append(v.verify(header_lvl=header_lvl + 1))
            elif is_dataclass(val):
                invalid_fields.append(val.verify(header_lvl=header_lvl+1))

        if len(invalid_fields) > 0:
            header = '\t'  # * header_lvl
            if header_lvl == 0:
                h = '\n\t' + f'{header}\t ' * header_lvl + "|"
                err_str = f'{h} Found unset fields in {self.__class__.__name__}:' + ''.join(
                    [f'{h}{s}' for s in invalid_fields])  # \ #+ f'\n|'.join(invalid_fields)
                if raise_error:  raise ValueError('\n' + err_str)
                else:  warnings.warn('\n' + err_str)
            else:
                h = '\n\t' + f'{header}\t ' * header_lvl + "|"
                name = f"{self.__class__.__name__}"
                name += f"({getattr(self, 'name')})"  if hasattr(self, 'name') else ''
                name += f"({getattr(self, 'trial_num')})" if hasattr(self, 'trial_num') else ''
                name += f"({getattr(self, 'layout')})" if hasattr(self, 'layout') else ''
                err_str = f'---> {name}:' + ''.join(
                    [f'{h} {s}' for s in invalid_fields])
                return err_str

    @property
    def summary(self):
        return self.__repr__()

@dataclass(frozen=True)
class FrozenBase:
    def verify(self,header_lvl=0,raise_error=True):
        invalid_fields = []
        for f in fields(self):
            name = f.name
            val = getattr(self, name)
            if val is None:
                invalid_fields.append(f'{name}:{val}')
            elif isinstance(val, str) and len(val) == 0:
                invalid_fields.append(f'{name}:{val}')
            elif isinstance(val, dict) and len(val.keys()) == 0:
                invalid_fields.append(f'{name}:{val}')
            elif isinstance(val, tuple) and len(val) == 0:
                invalid_fields.append(f'{name}:{val}')
            elif isinstance(val, list):
                if len(val) == 0:
                    invalid_fields.append(f'{name}:{val}')
                for i, v in enumerate(val):
                    if is_dataclass(v):
                        invalid_fields.append(v.verify(header_lvl=header_lvl + 1))
            elif is_dataclass(val):
                invalid_fields.append(val.verify(header_lvl=header_lvl + 1))

        if len(invalid_fields) > 0:
            header = '\t'  # * header_lvl
            if header_lvl == 0:
                h = '\n\t' + f'{header}\t ' * header_lvl + "|"
                err_str = f'{h} Found unset fields in {self.__class__.__name__}:' + ''.join(
                    [f'{h}{s}' for s in invalid_fields])  # \ #+ f'\n|'.join(invalid_fields)
                if raise_error:
                    raise ValueError('\n' + err_str)
                else:
                    warnings.warn('\n' + err_str)
            else:
                h = '\n\t' + f'{header}\t ' * header_lvl + "|"
                name = f"{self.__class__.__name__}"
                name += f"({getattr(self, 'name')})" if hasattr(self, 'name') else ''
                name += f"({getattr(self, 'trial_num')})" if hasattr(self, 'trial_num') else ''
                name += f"({getattr(self, 'layout')})" if hasattr(self, 'layout') else ''

                err_str = f'---> {name}:' + ''.join(
                    [f'{h} {s}' for s in invalid_fields])
                return err_str


@dataclass
class PseudoFrozenClass(Base):
    """A psudo-frozen class that freezes after set to not None"""
    def __setattr__(self, key, value):
        if key =='complete':
            super().__setattr__(key, value)
        elif hasattr(self, key) and getattr(self, key) is not None:  # and not self._initialized:
            raise AttributeError(f'Attribute [{key}] id is read-only')
        else:
            super().__setattr__(key, value)

######################################################################
############ ITEM CLASSES ############################################
######################################################################

@dataclass#(frozen=True)
class DemographicData(PseudoFrozenClass):
    player_id: str = None
    age: int = None
    sex: str = None
    complete = False
    def set(self,age,sex):
        self.age = age
        self.sex = sex
    @property
    def summary(self):
        return f'DemographicData(age={self.age}, sex={self.sex})'


@dataclass#(frozen=True)
class SurveyData(PseudoFrozenClass):
    """Class for keeping track of an item in inventory."""
    name: str
    player_id: str = None
    responses: dict = field(default_factory=dict)
    complete = False

    def set_responses(self,data_dict):
        for key, value in data_dict.items():
            self.responses[key] = value

    def __repr__(self):
        disp = ''
        disp += f'SurveyData(name={self.name}, player_id={self.player_id})\n'
        if self.responses is None:
            disp += '\t|No responses recorded\n'
        else:
            for key, value in self.responses.items():
                disp += f'\t|{key}: {value}\n'
        return disp

    @property
    def summary(self):
        return f'SurveyData(n_responses={len(self.responses)})'


@dataclass#(frozen=True)
class InteractionData(PseudoFrozenClass):
    """ Logs data from interaction/gameplay with the robot"""
    layout: str
    p_slip: float
    partner_type: str
    transition_history: list = field(default_factory=list)
    player_id: str = None
    complete = False

    def log_transition(self, t, s, aH, aR, info):
        """
        Logs a state transition and stores it in memory
        :param t: game timestep t
        :param s: state at time t
        :param aH: action of human
        :param aR: action of robot
        :return: None
        """
        transition = (t, s, aH, aR,  info)
        self.transition_history.append(transition)

    def __repr__(self):
        disp = ''
        disp += f'InteractionData(layout={self.layout}, p_slip={self.p_slip})\n'
        disp += f'\t | Layout: {self.layout}\n'
        disp += f'\t | p_slip: {self.p_slip}\n'
        disp += f'\t | Partner Type: {self.partner_type}\n'
        disp += f'\t | Len of interaction {len(self.transition_history)}\n'
        return disp

    @property
    def summary(self):
        return f'InteractionData(layout={self.layout}, p_slip={self.p_slip}, n_transitions={len(self.transition_history)})'

######################################################################
############ CUMULATIVE CLASSES ######################################
######################################################################
@dataclass(kw_only=True)
class TrialData(PseudoFrozenClass):
    player_id: str
    trial_num: int
    layout: str
    p_slip: float
    partner_type: str

    # Logging variables
    priming = None
    interaction: InteractionData = None
    trust_survey: SurveyData = None
    complete = False

    def __post_init__(self):
        self.interaction = InteractionData(self.player_id, self.layout, self.p_slip, self.partner_type)
        self.trust_survey = SurveyData(self.player_id, 'trust_survey')

    def set_priming(self, selection):
        """
        Logs the pretrial survey data that primes human into selecting strategy
        :param selection: string of the priming choice selection
        """
        self.priming = selection


    def set_trust_survey(self, survey_data):
        self.trust_survey.set_responses(survey_data)
    def log_transition(self, t, s, aH, aR, sp, info):
        self.interaction.log_transition(t, s, aH, aR, sp, info)

@dataclass
class Pretrial(PseudoFrozenClass):
    """Class for keeping track of an item in inventory."""
    player_id: str
    demographic: DemographicData = None
    risk_propensity: SurveyData = None
    tutorial_interactions: list = field(default_factory=list)
    n_tutorials = 4
    complete = False

    def __post_init__(self):
        self.risk_propensity = SurveyData(self.player_id, 'risk_propensity')
        for i in range(self.n_tutorials):
            interaction_obj = InteractionData(self.player_id, f'tutorial{i}', 0.0, 'tutorial')
            self.tutorial_interactions.append(interaction_obj)

    def set_demographic(self, age,sex):
        self.demographic = DemographicData(self.player_id,age,sex)

    def set_risk_propensity(self, response_dict):
        self.risk_propensity.set_responses(response_dict)

    def log_transition(self,tut_num, t, s, aH, aR, sp, info):
        self.tutorial_interactions[tut_num].log_transition(t, s, aH, aR, sp, info)




@dataclass
class ExperimentData(Base):
    player_id: str
    layouts: list
    p_slips: list
    partner_types: list

    # Logging variables
    pretrial: Pretrial = None
    trials: list = field(default_factory=list)
    posttrial_survey: SurveyData = None


    def __post_init__(self):
        self.pretrial = Pretrial(self.player_id)
        for layout,p_slip,partner_type in zip(self.layouts,self.p_slips,self.partner_types):
            trial_info = {
                'player_id': self.player_id,
                'trial_num': len(self.trials),
                'layout': layout,
                'p_slip': p_slip,
                'partner_type': partner_type
            }
            self.trials.append(TrialData(**trial_info))
        self.posttrial_survey = SurveyData(self.player_id, 'posttrial_survey')


class DataViewer:
    DOCKER_VOLUME = '\\app\\data'

    def __init__(self, fname, data_path=None):
        if data_path is  None: data_path = self.DOCKER_VOLUME
        file_path = os.path.join(data_path, fname)

        self.fname = fname
        self.path = file_path
        self.data = self.read_pickle_file(file_path)
        self.icond = fname.split('cond')[-1][0]  # Extract the condition from the file name
        self.prolific_id = fname.split('PID')[-1].split('__')[0]  # Extract the Prolific ID from the file name

        if 'participant_information' in self.data.keys():
            self.age = self.data['participant_information'].age
            self.sex = self.data['participant_information'].sex

    ################################################################################
    def summary(self):
        print(f'File: {self.fname}')
        for key, value in self.data.items():
            if not 'DummyData' in type(value).__name__:
                if 'priming' in key:
                    print(f'\t | ')
                if value.complete:
                    print(f'\t | {key}: {value.summary}')
                else:
                    print(f'\t | {key}: Incomplete Data')

    def view(self,stage_name):
        if 'tutorial' in stage_name or 'game' in stage_name:
            self.render_game(stage_name)
        elif (
                 'survey' in stage_name
              or 'priming' in stage_name
              or 'washout' in stage_name
              or 'participant_information' in stage_name
        ):
            self.print_survey(stage_name)
        else:
            raise ValueError(f'Unknown stage name: {stage_name}. Cannot render or print data.')

    def print_survey(self,stage_name):
        surveydata = self.data[stage_name]
        print(surveydata)
    def get_game_visualizer(self,stage_name):
        gamedata = self.data[stage_name]
        layout = gamedata.layout
        p_slip = gamedata.p_slip
        partner_type = gamedata.partner_type
        player_id = gamedata.player_id
        transition_history = gamedata.transition_history

        horizon = 360
        time_cost = 0
        MDP_PARAMS = {
            'p_slip': p_slip,
            'neglect_borders': True,
        }

        mdp = OvercookedGridworld.from_layout_name(layout, **MDP_PARAMS)
        base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, time_cost=time_cost)
        traj_vis = TrajectoryVisualizer(base_env)

        state_history = []
        for transition in transition_history:
            t, s, aH, aR, info = transition
            state_dict = json.loads(s)
            state_dict['players'][0]['idx'] = 0
            state_dict['players'][1]['idx'] = 1
            state = OvercookedState.from_dict(state_dict)
            state_history.append(state)
        traj_vis.que_trajectory(state_history)

        return traj_vis, state_history
    def render_game(self,stage_name):
        traj_vis,state_history = self.get_game_visualizer(stage_name)
        traj_vis.preview_trajectory(state_history)
        plt.show()

    def make_gif(self,stage_name,loop=0, fps=5):
        traj_vis, state_history = self.get_game_visualizer(stage_name)
        imgs = traj_vis.get_images(state_history)
        fname = stage_name
        # fname = self.fname.split('.')[0]  # Remove file extension
        # fname = self.fname.split('\\')[-1].split('/')[-1] # remove directory
        # fname += f'_{stage_name}'
        # print(fname)
        imageio.mimsave(f'./{fname}.gif', imgs,loop=loop, fps=fps)

    def read_pickle_file(self,file_path):
        """Read a pickle file and return the deserialized object."""
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            return data
        except:
            # Patch pickle modul
            import sys
            from risky_overcooked_webserver.server import data_logging
            sys.modules['data_logging'] = data_logging
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            return data


    def analyze(self):
        raise NotImplementedError



def main():
    # fname = "cond_0\\2025-08-02_20-48-36__{{%PID1234%}}__cond0.pkl"
    # fname = "cond_None\\2025-08-02_19-44-45__NoProlificID__condNone.pkl"
    # fname = "cond_1/2025-08-04_17-37-41__PID__cond1.pkl"
    # fname = "cond_1/2025-08-07_12-28-04__PID__cond1.pkl"
    fname = "cond_0/2025-09-02_16-12-04__PID123__cond0.pkl"
    viewer = DataViewer(fname)
    # viewer.print_survey('priming0')
    # viewer.render_game('risky_tutorial_3')

    # viewer.make_gif('risky_tutorial_0', fps = 10)
    # viewer.make_gif('risky_tutorial_1', fps = 10)
    # viewer.make_gif('risky_tutorial_2', fps = 10)
    # viewer.make_gif('risky_tutorial_3', fps = 10)
    # viewer.print_survey('trust_survey0')
    # viewer.make_gif('game0', fps=10)
    viewer.summary()

    viewer.print_survey('priming0')
    viewer.print_survey('trust_survey0')


if __name__ == "__main__":
    main()
