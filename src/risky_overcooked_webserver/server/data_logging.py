import warnings
from dataclasses import dataclass, field, fields,is_dataclass

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


@dataclass#(frozen=True)
class InteractionData(PseudoFrozenClass):
    """ Logs data from interaction/gameplay with the robot"""
    layout: str
    p_slip: float
    partner_type: str
    transition_history: list = field(default_factory=list)
    player_id: str = None
    complete = False

    def log_transition(self, t, s, aH, aR, sp, info):
        """
        Logs a state transition and stores it in memory
        :param t: game timestep t
        :param s: state at time t
        :param aH: action of human
        :param aR: action of robot
        :param sp: next state (after stochastic trnasition)
        :return: None
        """
        transition = (t, s, aH, aR, sp, info)
        self.transition_history.append(transition)

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



def main():
    experiment_config = {
        'player_id':'p1',
        'layouts': ['layout1','layout2'],
        'p_slips': [0.5,0.3],
        'partner_types': ['rational','irrational']
    }
    experiment = ExperimentData(**experiment_config)


    transition = {
        't': 0,
        's': 'state',
        'aH': 'human_action',
        'aR': 'robot_action',
        'sp': 'next_state',
        'info': {}
    }
    experiment.trials[0].log_transition(**transition)
    experiment.verify()


if __name__ == "__main__":
    main()
