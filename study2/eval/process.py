import sys
import copy
import pickle
import numpy as np
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import warnings
# from pingouin import cronbach_alpha

from study2.static import *
from study2.eval.eval_utils import get_unprocessed_fnames, get_processed_fnames

from risky_overcooked_rl.utils.evaluation_tools import CoordinationFluency
from risky_overcooked_rl.algorithms.DDQN.utils.agents import DQN_vector_feature
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld, Action, OvercookedState
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_rl.algorithms.DDQN.utils.game_thoery import QuantalResponse_torch
from risky_overcooked_rl.utils.visualization import TrajectoryVisualizer









































































INCLUDE_DICT_PIDs = {
'PID672135003f5e272c889620ea':1,
'PID66db9db4324609f1a7231f49':1,
'PID67e2ec1e3ccfcc02f94d6351':1,
'PID66b504cd131c63b36b682b8d':1,
'PID67ee6f72aa206db46d5e9d11':1,
'PID677317a5e11a1e2ab07f415e':0,
'PID666f76b68b89442817be678a':0,
'PID62ab399e15b98baaf3099d60':0,
'PID6726eb8bd8a35b7366b1790a':1,
'PID67435174b76a747138e8a47a':1,
'PID60c8467d9872c0d83f695499':1,
'PID66293ce3dba0764775195e58':0,
'PID66b01097e95b0626c2cd7b5c':0,
'PID67f192e08cf17c568074969d':0,
'PID616f727b03e888e0f8213eec':1,
'PID60c2cc653d0c6208fc8899fc':1,
'PID6079ff1600ff2b7455e1c3e0':1,
'PID5bce1e8453de590001846151':0,
'PID58a0c507890ea500014c4e9b':0,
'PID5d9d5debf346240014428500':1,
'PID6600a119385d8631c41cd795':1,
'PID64136bf30b27746cb96f7db8':1,
'PID66d13205a7d03cb99053fea4':0,
'PID5e0857276aab7c17f7e21662':1,
'PID66463e4efb99fba5a67a010a':0,
'PID5dce3ccc32ccbf0cd54263db':1,
'PID66c0f464e0ff62798027b486':1,
'PID5d4ac837d2844e0001ecf699':1,
'PID60a1e1f4a3707c983a98f185':1,
'PID66f305037af69346a9a55b42':1,
'PID68dd6ada1c06b7f58ff4020c':1,
'PID63474e67a5fd298c6103c409':1,
'PID67722f8e3a4f08a288a1f640':1,
'PID691525ea2e1385d5abf9772e':1,
'PID5f0f771eeab74413eb386872':1,
'PID66520f31e4b5df0b6b365ed3':0,
'PID60ac66a8e1bf5a1c51fa864c':1,
'PID5f3ac1732efa0a74f975b1a8':1,
'PID670d8b142f87173ffe3b4763':1,
'PID653703627539f3a8b2ed4af3':1,
'PID66d8e14e71c04a7d23ff43c9':0,
'PID5eb402bed161131f83db9ce4':0,
'PID65f366dfcb46b71238e9418d':1,
'PID60d50ceb7c563c73f91d9ec5':1,
'PID5dce29700ad506063969a4a5':1,
'PID663a6a13c0b6b4aff21552f8':1,
'PID5e29cc3230cf9d03580c34bf':1,
'PID5c8974ef34daa70015e92daf':1,
'PID666b59f95861737274e65238':1,
'PID68c08483061276d20b570579':1,
'PID66ba7e4126d266ff3cdf6d79':1,
'PID63d1c79ff86ec609af6f77ad':1,
}
EXCLUDE_PIDs = [pid for pid,flag in INCLUDE_DICT_PIDs.items() if flag == 0]

AGENT_CACHE = {}

class DataPoint:
    """
    Scores and Reverse Codes are framed as:
        - Risk-Seeking = higher score
        - More trust = higher score
        - Relative measures are framed as:
            - More trust in RS-ToM agent = higher score
            - RS-ToM agent more risk-seeking = higher score
    """
    @staticmethod
    def get_trust_questions(df_labels= False):
        labels = ['Dependable', 'Reliable', '-(Unresponsive)', 'Predictable',
            'Act consistently', 'Meet the needs of the task', 'Perform as expected'
        ]
        if df_labels:
            labels = [f'trust_q{i}_{label}' for i,label in enumerate(labels)]
        return labels
    @staticmethod
    def load_all_processed_data():
        fname_dict = get_processed_fnames(full_path=False)
        COND1_FNAMES = fname_dict['cond1']
        COND0_FNAMES = fname_dict['cond0']

        for ex_pid in EXCLUDE_PIDs:
            assert any(ex_pid in fname for fname in COND0_FNAMES + COND1_FNAMES), f"Excluded PID {ex_pid} not found in filenames"

        dps_cond1 = []
        dps_cond0 = []
        dps_all = []
        for fname in COND0_FNAMES:

            fpath = PROCESSED_COND0_DIR + "\\" + fname
            dp = DataPoint.load_processed(fpath)
            dp.included = all(ex_pid not in fname for ex_pid in EXCLUDE_PIDs)
            dps_cond0.append(dp)

        for fname in COND1_FNAMES:
            fpath = PROCESSED_COND1_DIR + "\\" + fname
            dp = DataPoint.load_processed(fpath)
            dp.included = all(ex_pid not in fname for ex_pid in EXCLUDE_PIDs)
            dps_cond1.append(dp)

        dps_all += dps_cond0 + dps_cond1
        data_dict = {'cond0': dps_cond0,
                     'cond1': dps_cond1,
                     'all': dps_all}
        return data_dict
    @classmethod
    def load_processed(cls, file_path):
        # raise NotImplementedError("Processed data loading not implemented yet")
        if ".pkl" not in file_path: file_path += ".pkl"
        # with open(file_path, 'rb') as file:
        #     data_point = pickle.load(file)
        # return data_point

        try:
            with open(file_path, 'rb') as file:
                data_point = pickle.load(file)
            return data_point
        except:
            # Patch pickle modul
            import sys
            from study2.eval.process import TorchPolicy
            sys.modules['TorchPolicy'] = TorchPolicy
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            return data

    def __init__(self, fname, min_survey_var = 0.0):
        data_path = RAW_COND0_DIR if 'cond0' in fname else RAW_COND1_DIR
        self._mean_idle_thresh = 0.6 # mean threshold for considering an agent "inactive" during a game
        self._max_idle_thresh = 0.9 # max threshold for considering an agent "inactive" during a game
        self._max_frozen_steps = 50 # threshold for considering an agent "frozen" during a game (0.1s per step, so 5s)

        # Load raw data
        file_path = os.path.join(data_path, fname)
        self.fname = fname
        self.path = file_path
        self._raw = self.load_raw(file_path)
        self.survey_range = (10,0)
        self._included = True


        # self.reverse_coded = []
        # self.reverse_coded += ['Unresponsive', 'Take too many risks', 'Played more safe']
        # self.reverse_coded += ['Safety first.', 'I do not take risks with my health.',
        #                        'I prefer to avoid risks.', 'I really dislike not knowing what is going to happen.']
        try:
            # Extract metadata/demographics
            self.age = self._raw['demographic'].responses['age']
            self.sex = self._raw['demographic'].responses['sex']
            self.icond = int(fname.split('cond')[-1][0])
            self.conds = [['rs-tom', 'rational'],['rational','rs-tom']][self.icond]


            # Begin parsing data
            self._def_dict = {0: [], 1: []}  # default timeseries format
            for c in self.conds: self._def_dict[c] = []

            self._stage_list = list(self._raw.keys())
            self._games = self.parse_stage_key('game')
            self._primings = self.parse_stage_key('priming')
            self._trust_surveys = self.parse_stage_key('trust_survey')
            self.layouts,self.p_slips = self.parse_ordered_layouts()
            self._models = self.preload_models()

            # Begin computations ########################

            # Surveys (single measure)
            self.RTP_responses, self.RTP_score, self.RTP_dur = self.compute_risk_taking_propensity()
            self.rel_trust_responses, self.rel_trust_score,self.rel_trust_dur = self.compute_relative_trust_score()
            self.rel_risk_perc_responses, self.rel_risk_perc_score,self.rel_risk_perc_dur = self.compute_relative_risk_perception()
            self.pass_attention_check, self.attention_check_responses = self.compute_attention()

            # Surveys (repeated measures)
            self.trust_responses, self.trust_scores, self.delta_trusts, self.trust_durs = self.compute_trust_scores()
            self.risk_perc_responses, self.risk_perc_scores, self.risk_perc_durs = self.compute_risk_perceptions()
            self.priming_labels,self.priming_scores, self.priming_durs = self.compute_priming_labels()
            self.trust_slopes = self.compute_trust_slopes()

            # Gameplay
            self.rewards = self.compute_rewards()
            self.nH_risks, self.nR_risks = self.compute_n_risks()
            self.predictability = self.compute_predictabilities()
            self.beliefs, self.inferences, self.accuracies = self.compute_belief_accuracies()
            self.C_ACTs, self.H_IDLEs, self.R_IDLEs = self.compute_coordination_fluency()
            self.H_frozen, self.R_frozen, self.any_frozen = self.compute_frozen()

            # Check data validity
            # if not self.pass_attention_check:
            #     print(f"Participant {self.fname} failed attention check {self.attention_check_responses}", file=sys.stderr)
            # else:
            #     print(f"Participant {self.fname} attention check passed: {self.attention_check_responses}")

            self.min_survey_var = min_survey_var
            self.surveys_valid,self.survey_approval_rate = self.check_survey_validity()
            self.was_active = self.check_activity()
            self.passed_review, self.manual_reviews = self.check_game_validity()

        except Exception as e:
            print(f"Error processing file {self.fname}: {e}", file=sys.stderr)
            raise e


    ###############################################
    # Metadata metrics ############################
    def _parse_survey_responses(self,responses,reverse_coded=()):
        """Survey helper: reverse code and compute average score"""
        # check that all reverse coded questions are in responses
        checks= [rcq in responses.keys() for rcq in reverse_coded]
        assert all(checks), f"Reverse coded questions {reverse_coded} not all in responses {responses.keys()}"

        # parse responses
        _responses = {}
        _score = 0
        for response_key, response_val in responses.items():
            key,val = self._check_RC(response_key,response_val, reverse_coded)
            _responses[key] = val
            _score += val
        _score /= len(responses)
        return _responses, _score

    def _check_RC(self, name, score, reverse_coded):
        """Survey helper: Checks if should reverse code a survey score"""
        if name in reverse_coded:
            new_score = self._RC(score)
            new_name = f'-({name})'
            return new_name, new_score
        return name, int(score)

    def _RC(self, score):
        """Survey helper: Reverse codes a survey score"""
        vmax, vmin = self.survey_range
        new_score = vmax + vmin - int(score)
        return new_score

    def _exclude_survey_items(self, responses, exclude_items):
        """Survey helper: Excludes certain survey items from responses"""
        responses = copy.deepcopy(responses)
        for item in exclude_items:
            assert item in responses.keys(), f"Item {item} not in responses {responses.keys()}"
            responses.pop(item, None)
        return responses

    def compute_risk_taking_propensity(self,N=7 ):
        reverse_coded = (
            'Safety first.',
            'I do not take risks with my health.',
            'I prefer to avoid risks.',
            'I really dislike not knowing what is going to happen.'
        )
        exclude_items = ['ID']

        #################################################
        duration = self._raw['risk_propensity'].duration
        responses = copy.deepcopy(self._raw['risk_propensity'].responses)
        responses = self._exclude_survey_items(responses, exclude_items=exclude_items)
        RTP_responses, RTP_score = self._parse_survey_responses(responses, reverse_coded=reverse_coded)
        assert len(RTP_responses) == N, f"Expected {N} RTP responses," \
                                       f" got {len(RTP_responses)} in filename {self.fname}"
        return RTP_responses, RTP_score, duration

    def compute_relative_trust_score(self, N=6):
        reverse_coded = ()
        exclude_items = ['ID', ' Took more risks', 'Played more safe']

        #################################################
        duration = self._raw['relative_trust_survey'].duration
        responses = copy.deepcopy(self._raw['relative_trust_survey'].responses)
        responses = self._exclude_survey_items(responses, exclude_items=exclude_items)
        rel_trust_responses, rel_trust_score = self._parse_survey_responses(responses,reverse_coded=reverse_coded)

        # If cond 1, reverse code the overall score
        if self.icond == 1:
            rel_trust_score = self._RC(rel_trust_score)
            for key,val in rel_trust_responses.items():
                rel_trust_responses[key] = self._RC(val)
        assert len(rel_trust_responses) == N, f"Expected {N} relative trust responses," \
                                              f" got {len(rel_trust_responses)} in filename {self.fname}"
        return rel_trust_responses, rel_trust_score, duration

    def compute_relative_risk_perception(self, N=2):
        reverse_coded = ('Played more safe',)
        exclude_items = ['ID',
                         'More dependable' ,
                         'More reliable',
                         'More predictable',
                         'Acted more consistently',
                         'Better met the needs of the task',
                        'Better performed as expected']

        #################################################
        duration = self._raw['relative_trust_survey'].duration
        responses = copy.deepcopy(self._raw['relative_trust_survey'].responses)
        responses = self._exclude_survey_items(responses, exclude_items=exclude_items)
        rel_risk_perception_responses, rel_risk_perception_score = self._parse_survey_responses(responses,reverse_coded=reverse_coded)

        # If cond 1, reverse code the overall score
        if self.icond == 1:
            rel_risk_perception_score = self._RC(rel_risk_perception_score)
            for key,val in rel_risk_perception_responses.items():
                rel_risk_perception_responses[key] = self._RC(val)

        assert len(rel_risk_perception_responses) == N, f"Expected {N} relative risk perception responses," \
                                                        f" got {len(rel_risk_perception_responses)} in filename {self.fname}"
        return rel_risk_perception_responses, rel_risk_perception_score, duration


    def compute_attention(self, N=2):
        # reverse_coded = ('Played more safe',)
        exclude_items = ['ID',
                         ' Took more risks',
                         'Played more safe'
                         'More dependable' ,
                         'More reliable',
                         'More predictable',
                         'Acted more consistently',
                         'Better met the needs of the task',
                        'Better performed as expected']

        #################################################
        attention_checks = []
        passed_checks = []

        for ic in range(len(self.conds)):
            cond_surveys = copy.deepcopy(self._trust_surveys[self.conds[ic]])

            # cond_games = self._games[self.conds[ic]]
            for _is, s in enumerate(cond_surveys):
                if "Play the guitar" in s.responses.keys():
                    key = "Play the guitar"

                elif "A car" in s.responses.keys():
                    key =  "A car"
                else:
                    continue

                resp = (key, int(s.responses[key]))
                attention_checks.append(resp)
                passed_checks.append(int(s.responses[key]) == 0)
        pass_attention = all(passed_checks)

        if not pass_attention:
            print(f"Warning: Participant {self.fname} failed attention check(s): {attention_checks}", file=sys.stderr)
        return pass_attention, attention_checks


    ###############################################
    # Trial metrics ###############################

    def compute_priming_labels(self):
        priming_labels = copy.deepcopy(self._def_dict)
        priming_scores = copy.deepcopy(self._def_dict)
        priming_durs = copy.deepcopy(self._def_dict)

        averse_responses = ['Take the longer detour that avoids all puddles',
                            'Pass objects to partner using counter tops to avoid all puddles' ]

        rational_responses = ['Take the middle route through one puddle',
                              "A mix of handing off items and avoiding puddles",
                              'Take the route through one puddle']

        seeking_responses = ['Take the most direct route by going through two puddles',
                             'Enter one puddle to handoff items to my partner',
                             'Take the most direct route by going through both puddles']

        for ic in range(len(self.conds)):
            cond_primings = self._primings[self.conds[ic]]
            priming_durs[self.conds[ic]] = [p.duration for p in cond_primings]
            priming_durs[ic] = [p.duration for p in cond_primings]

            labels, scores = [], []
            for p in cond_primings:
                if p.responses['priming'] in averse_responses:
                    label,score = 'averse', -1
                elif p.responses['priming'] in rational_responses:
                    label,score = 'rational', 0
                elif p.responses['priming'] in seeking_responses:
                    label,score = 'seeking',1
                else:
                    raise ValueError(f"Unknown priming response [{p.responses['priming']}] in filename {self.fname}")
                labels.append(label)
                scores.append(score)

            priming_labels[ic] = labels
            priming_labels[self.conds[ic]] = labels
            priming_scores[ic] = scores
            priming_scores[self.conds[ic]] = scores

        return priming_labels, priming_scores, priming_durs

    def compute_belief_accuracies(self):
        beliefs = copy.deepcopy(self._def_dict)
        inferences = copy.deepcopy(self._def_dict)
        accuracies = copy.deepcopy(self._def_dict)

        for ic in range(len(self.conds)):
            cond_games = self._games[self.conds[ic]]

            for ig, g in enumerate(cond_games):
                game_beliefs = np.empty([len(g.transition_history),3])  # time_steps,num_policies
                game_inferences = np.empty([len(g.transition_history)])  # time_steps
                for t, s, aH, aR, info in g.transition_history:
                    # b = np.array([info['belief'][p] for p in ['averse','neutral','seeking']])
                    b = json.loads(info['belief'].replace("'",'"'))
                    if len(b.keys()) == 1:  # rational
                        b = np.array([0,1,0])
                    else:
                        b = np.array([b[p] for p in ['averse', 'neutral', 'seeking']])

                    game_beliefs[t-1, :] = b
                    game_inferences[t-1] = np.argmax(b) - 1 # adjust to -1,0,1

                # true_policy = self.priming_scores[self.conds[ic]][ig]
                true_policy = self.priming_scores[ic][ig]

                game_accuracy = np.mean(game_inferences == true_policy)

                beliefs[self.conds[ic]].append(game_beliefs)
                beliefs[ic].append(game_beliefs)
                inferences[self.conds[ic]].append(game_inferences)
                inferences[ic].append(game_inferences)


                accuracies[self.conds[ic]].append(game_accuracy)
                accuracies[ic].append(game_accuracy)




        return beliefs,inferences,accuracies


        # raise NotImplementedError("Belief accuracy computation not implemented yet")

    def recompute_belief_accuracies(self):
        from risky_overcooked_webserver.server.game import BayesianBeliefUpdate
        # print("Recomputing belief accuracies with belief updater...")
        beliefs = copy.deepcopy(self._def_dict)
        inferences = copy.deepcopy(self._def_dict)
        accuracies = copy.deepcopy(self._def_dict)
        # try:
        for ic in range(len(self.conds)):
            cond_games = self._games[self.conds[ic]]
            # Instantiate Belief Updater

            agent_names = ['averse', 'neutral', 'seeking'] if self.conds[ic] == 'rs-tom' else ['neutral']



            for ig, g in enumerate(cond_games):
                game_beliefs = np.empty([len(g.transition_history), 3])  # time_steps,num_policies
                game_inferences = np.empty([len(g.transition_history)])  # time_steps
                mdp = OvercookedGridworld.from_layout_name(g.layout, p_slip=g.p_slip, neglect_boarders=True)

                policies = self._models[ic][ig]
                belief = BayesianBeliefUpdate(policies, policies,
                                              names=agent_names,
                                              iego=1,
                                              ipartner=0,
                                              capacity=10,
                                              alpha = 0.99
                                              )
                belief.reset_prior()


                # for t, state, aH, aR, info in g.transition_history:
                for t, state, aH, aR, info in g.transition_history:


                    if len(agent_names) == 1: # rational
                        b = np.array([0, 1, 0])
                    else:


                        state_dict = json.loads(state.replace("'", '"'))
                        state = OvercookedState.from_dict(state_dict)
                        # aH =  tuple(aR) if isinstance(aR, list) else aR
                        aH = tuple(aH) if isinstance(aH, list) else aH

                        obs = mdp.get_lossless_encoding_vector_astensor(state,  device='cpu').unsqueeze(0) # ,
                        human_iA = Action.ACTION_TO_INDEX[aH]

                        is_trivial = (aH == "Stay" or aH == (0, 0))
                        if not is_trivial:
                            belief.update_belief(obs, human_iA, is_only_partner_action=True)
                        b = belief.belief
                        # b = np.array([0, 1, 0])



                    game_beliefs[t - 1, :] = b
                    game_inferences[t - 1] = np.argmax(b) - 1  # adjust to -1,0,1

                # true_policy = self.priming_scores[self.conds[ic]][ig]
                true_policy = self.priming_scores[ic][ig]

                game_accuracy = np.mean(game_inferences == true_policy)

                beliefs[self.conds[ic]].append(game_beliefs)
                beliefs[ic].append(game_beliefs)
                inferences[self.conds[ic]].append(game_inferences)
                inferences[ic].append(game_inferences)

                accuracies[self.conds[ic]].append(game_accuracy)
                accuracies[ic].append(game_accuracy)
        # except Exception as e:
        #     print(f"Error recomputing belief accuracies for file {self.fname}: {e}", file=sys.stderr)
        #     return self.compute_belief_accuracies()

        return beliefs, inferences, accuracies

        # raise NotImplementedError("Belief accuracy computation not implemented yet")

    def compute_trust_scores(self,N=7  ):
        reverse_coded = ('Unresponsive',)
        exclude_items = ['ID','Take too many risks', 'Play too safe']

        trust_responses = copy.deepcopy(self._def_dict)
        trust_scores = copy.deepcopy(self._def_dict)
        delta_trusts = copy.deepcopy(self._def_dict)
        trust_durs = copy.deepcopy(self._def_dict)


        for ic in range(len(self.conds)):
            cond_surveys =  copy.deepcopy(self._trust_surveys[self.conds[ic]])
            trust_durs[self.conds[ic]] = [s.duration for s in cond_surveys]
            trust_durs[ic] = [s.duration for s in cond_surveys]

            # cond_games = self._games[self.conds[ic]]
            for _is, s in enumerate(cond_surveys):
                if "Play the guitar" in s.responses.keys(): extra_exclude =  ["Play the guitar"]
                elif "A car" in s.responses.keys(): extra_exclude =  ["A car"]
                else: extra_exclude = []

                responses = self._exclude_survey_items(s.responses, exclude_items=exclude_items + extra_exclude)
                coded_responses, score = self._parse_survey_responses(responses,reverse_coded=reverse_coded)
                assert len(coded_responses) == N, f"Expected {N} trust responses,got {len(coded_responses)} "

                trust_responses[self.conds[ic]].append(coded_responses)
                trust_responses[ic].append(coded_responses)

                trust_scores[self.conds[ic]].append(score)
                trust_scores[ic].append(score)

                if _is >= 1:
                    prev_score = trust_scores[self.conds[ic]][_is-1]
                    dtrust = score - prev_score
                    delta_trusts[ic].append(dtrust)
                    delta_trusts[self.conds[ic]].append(dtrust)

        return trust_responses, trust_scores, delta_trusts, trust_durs

    def compute_trust_slopes(self):
        trust_slopes = copy.deepcopy(self._def_dict)
        for ic in range(len(self.conds)):
            data = np.array(self.trust_scores[self.conds[ic]])
            x = np.arange(data.shape[0])
            coeffs = np.polyfit(x, data, 1)
            trust_slopes[self.conds[ic]] = coeffs[0]
            trust_slopes[ic] = coeffs[0]
        return trust_slopes

    def compute_risk_perceptions(self,N=2):
        reverse_coded = ('Play too safe',)
        exclude_items = ['ID', 'Dependable', 'Reliable', 'Unresponsive', 'Predictable',
                   'Act consistently', 'Meet the needs of the task', 'Perform as expected']

        rel_risk_responses = copy.deepcopy(self._def_dict)
        rel_risk_scores = copy.deepcopy(self._def_dict)
        rel_risk_durs = copy.deepcopy(self._def_dict)


        for ic in range(len(self.conds)):
            cond_surveys = copy.deepcopy(self._trust_surveys[self.conds[ic]])
            rel_risk_durs[self.conds[ic]] = [s.duration for s in cond_surveys]
            rel_risk_durs[ic] = [s.duration for s in cond_surveys]

            # cond_games = self._games[self.conds[ic]]
            for _is, s in enumerate(cond_surveys):
                if "Play the guitar" in s.responses.keys(): extra_exclude =  ["Play the guitar"]
                elif "A car" in s.responses.keys(): extra_exclude =  ["A car"]
                else: extra_exclude = []
                responses = self._exclude_survey_items(s.responses, exclude_items=exclude_items + extra_exclude)

                coded_responses, score = self._parse_survey_responses(responses,reverse_coded=reverse_coded)
                assert len(coded_responses) == N, f"Expected {N} risk perception responses,got {len(coded_responses)} "

                rel_risk_responses[self.conds[ic]].append(coded_responses)
                rel_risk_responses[ic].append(coded_responses)

                rel_risk_scores[self.conds[ic]].append(score)
                rel_risk_scores[ic].append(score)

        return rel_risk_responses, rel_risk_scores, rel_risk_durs

    ###############################################
    # Gameplay metrics ############################
    def compute_rewards(self):
        rewards = copy.deepcopy(self._def_dict)
        for ic in range(len(self.conds)):
            cond_games = self._games[self.conds[ic]]
            for g in cond_games:
                cum_rew = 0
                for t, s, aH,aR,info in g.transition_history:
                    cum_rew += np.mean(info['reward'])
                    # score += info['score']
                rewards[ic].append(cum_rew)
                rewards[self.conds[ic]].append(cum_rew)
        return rewards

    def compute_n_risks(self):
        nH_risks = copy.deepcopy(self._def_dict)
        nR_risks = copy.deepcopy(self._def_dict)

        for ic in range(len(self.conds)):
            cond_games = self._games[self.conds[ic]]
            for g in cond_games:
                nH,nR = 0,0
                for t, s, aH, aR, info in g.transition_history:
                    nH += info['can_slip'][0]
                    nR += info['can_slip'][1]
                nH_risks[ic].append(nH)
                nH_risks[self.conds[ic]].append(nH)
                nR_risks[ic].append(nR)
                nR_risks[self.conds[ic]].append(nR)

        return nH_risks,nR_risks

    def compute_coordination_fluency(self):
        C_ACT = copy.deepcopy(self._def_dict)
        H_IDLE = copy.deepcopy(self._def_dict)
        R_IDLE = copy.deepcopy(self._def_dict)

        for ic in range(len(self.conds)):
            cond_games = self._games[self.conds[ic]]
            for g in cond_games:
                state_history = []
                for t, s, aH, aR, info in g.transition_history:
                    state = OvercookedState.from_dict(json.loads(s))
                    state_history.append(state)
                cf = CoordinationFluency(state_history=state_history)
                cf_measures = cf.measures(iR=1,iH=0)

                C_ACT[self.conds[ic]].append(cf_measures['C_ACT'])
                C_ACT[ic].append(cf_measures['C_ACT'])

                H_IDLE[self.conds[ic]].append(cf_measures['H_IDLE'])
                H_IDLE[ic].append(cf_measures['H_IDLE'])

                R_IDLE[self.conds[ic]].append(cf_measures['R_IDLE'])
                R_IDLE[ic].append(cf_measures['R_IDLE'])
        return C_ACT,H_IDLE,R_IDLE

    def compute_predictabilities(self,iH=0,**kwargs):
        predictabilities = copy.deepcopy(self._def_dict)
        beliefs = copy.deepcopy(self._def_dict)

        mdp_params = kwargs.get("mdp_params", {'neglect_boarders': True})

        for ic in range(len(self.conds)):
            cond_games = self._games[self.conds[ic]]

            for ig, g in enumerate(cond_games):
                mdp_params["p_slip"] = g.p_slip
                mdp = OvercookedGridworld.from_layout_name(g.layout, **mdp_params)
                model = self._models[self.conds[ic]][ig]

                pred_history = []
                for t, s, aH, aR, info in g.transition_history:
                    ipolicy = np.argmax(info['belief'])
                    state = OvercookedState.from_dict(json.loads(s))
                    obs = mdp.get_lossless_encoding_vector_astensor(state, device='cpu').unsqueeze(0)
                    _, _, action_probs = model[ipolicy].choose_joint_action(obs)
                    aH_dist = action_probs[iH]
                    pred = np.argmax(aH_dist) == Action.ACTION_TO_INDEX[aH if type(aH) == str else tuple(aH)] # was the human action the most likely?
                    # pred = adist[np.argmax(adist)] # confidence of the prediction
                    pred_history.append(pred)

                predictabilities[self.conds[ic]].append(np.mean(pred_history))
                predictabilities[ic].append(np.mean(pred_history))

        return predictabilities
        # raise NotImplementedError("Predictability computation not implemented yet")

    ###############################################
    # Validity checks #############################
    def check_activity(self):
        """Validity check: did they actively participate in the task?"""
        # raise NotImplementedError("Activity check not implemented yet")
        KEYS = ['rs-tom', 'rational']
        H_IDLEs = [np.mean(self.H_IDLEs[key]) for key in KEYS]

        max_idle = max(H_IDLEs)
        mean_idle = np.mean(H_IDLEs)
        was_inactive = max_idle > self._max_idle_thresh or mean_idle > self._mean_idle_thresh

        if was_inactive:
            print(f"Warning: Participant {self.fname} was inactive (max_idle={max_idle}, mean_idle={mean_idle})",file=sys.stderr)
        return not was_inactive

    def _score_survey_validity(self, survey, min_var = 0.25, max_var = 2.5):
        """Checks agreement before and after reverse coding to detect straight-lining or random responses"""
        assert isinstance(survey, dict), "Survey must be a dictionary of responses"

        rc_responses = list(survey.values())
        raw_responses = []
        for key, val in survey.items():
            raw_val = self._RC(val) if "-" in key else val
            raw_responses.append(raw_val)


        raw_var = np.var(np.array(raw_responses))
        rc_var = np.var(np.array(rc_responses))

        # vmax, vmin = self.survey_range
        # raw_var = np.var(np.array(raw_responses)/vmax)
        # rc_var = np.var(np.array(rc_responses)/vmax)

        # raw_var = np.var(raw_responses)/len(survey)
        # rc_var  = np.var(rc_responses)/len(survey)
        # diff_var = rc_var - raw_var


        info = {}
        info['failed_rc'] = (rc_var > raw_var)      # Reverse Coded: The agreement should not decrease after reverse coding
        info['failed_uniform'] = raw_var < min_var  # Straight-lining: should not have extremely low variance in Raw
        info['failed_rand'] =  rc_var > max_var     # Random Responses: should not have extremely high variance in RC
        return raw_var, rc_var, info

    def check_survey_validity(self, approval_thresh=0.8,min_dur=15):
        """Computes variance of all survey responses to check for low-variance responses"""
        approvals = []

        single_survey = [
            [self.RTP_responses, self.RTP_dur],
            [self.rel_trust_responses, self.rel_trust_dur],
            [self.rel_risk_perc_responses, self.rel_risk_perc_dur],
        ]
        repeated_survey = [
            self.trust_responses,
            # self.risk_perc_responses
        ]

        failure_durs = []
        failure_item_durs = []

        # Test Cases
        # STRAIGHT_SURVEY1 = {'A': 1, 'B': 0, 'C': 0, 'D': 1, 'E': 0, 'F': 1 ,'-(G)': self._RC(0)}
        # RAND_SURVEY1     = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 4, '-(G)': self._RC(3)}
        # GOOD_SURVEY1     = {'A': 0, 'B': 1, 'C': 2, 'D': 0, 'E': 1, 'F': 2, '-(G)': 3}
        #
        # STRAIGHT_SURVEY2 = {'A': 0, '-(B)': self._RC(1)}
        # RAND_SURVEY2     = {'A': 1, '-(B)': self._RC(5)}
        # GOOD_SURVEY2     = {'A': 0, '-(B)': 3}

        # Single Surveys ###############
        for rps,dur in single_survey:
            raw_var, rc_var, fails = self._score_survey_validity(rps)

            if any(fails.values()) or (dur < min_dur if dur is not None else False):
                dur_str = f' (dur={dur:.1f}s)' if dur is not None else 'No Dur'
                approved = SurveyVisualizer(rps).review(title=f' ({dur_str}s) [{self.fname}]')
                if not approved:
                    dur = dur if dur is not None else -1
                    failure_durs.append(dur)
                    failure_item_durs.append(dur / len(rps))
            else:
                approved = True
            approvals.append(approved)

        # Repeated surveys ###############
        for survey in repeated_survey:
            for cond in self.conds:
                for i, rps in enumerate(survey[cond]):
                    rps = copy.deepcopy(rps)
                    rps.update(self.risk_perc_responses[cond][i])

                    dur = self.trust_durs[cond][i]

                    raw_var, rc_var, fails = self._score_survey_validity(rps)
                    # if True:
                    if any(fails.values()):
                    #     print(f"Warning: Participant {self.fname} failed survey validity check:"
                    #           f"\n raw_var={raw_var}, rc_var={rc_var}, fails={fails}", file=sys.stderr)
                        dur_str = f' (dur={dur:.1f}s)' if dur is not None else None
                        approved = SurveyVisualizer(rps).review(title=f' ({dur_str}s) [{self.fname}]')
                        if not approved:
                            dur = dur if dur is not None else -1
                            failure_durs.append(dur)
                            failure_item_durs.append(dur/len(rps))
                    else:
                        approved = True
                    approvals.append(approved)

        #
        approval_rate = np.array(approvals, dtype=int).mean()
        is_valid = approval_rate > approval_thresh


        # if not is_valid:
        #     print(f"\n\nWarning: Participant {self.fname} failed survey validity with approval rate {approval_rate}")
        #     print(f'Rejection Messsage:')
        #     print(f"Unfortunately, your responses to {(1-approval_rate)*100:0.1f}% of the provided surveys"
        #           f" failed one or more conservative validity checks, which indicate that they may not have been completed with sufficient attention or effort."
        #           f" This result was determined by "
        #             f"A) analysis of variance in reverse-coded questions (i.e., questions with opposite meaning) which show inconsistent responses and "
        #             f"B) unusually short completion times with an average of {np.mean(failure_durs):.1f} seconds per survey ({np.mean(failure_item_durs):.1f} seconds per item). "
        #           f"Both determinations indicate insufficient engagement with the content, typically due to straight-lined or random responses."
        #           " \n\nAs a result, we are unable to approve this submission. We appreciate your time, but to ensure data quality"
        #           " and fairness across participants, only valid and attentive responses can be accepted.")
        # else:
        #     print(f'Participant {self.fname} passed survey validity with approval rate {approval_rate}')
        return is_valid, approval_rate

    def check_game_validity(self):
        # if not self.surveys_valid:
        #     return False, []

        approvals = []
        for ic in range(len(self.conds)):


            cond_games = self._games[self.conds[ic]]
            for ig, g in enumerate(cond_games):
                was_frozen = self.H_frozen[self.conds[ic]][ig] or self.R_frozen[self.conds[ic]][ig]
                this_inactive = self.H_IDLEs[self.conds[ic]][ig] > self._max_idle_thresh

                if was_frozen or this_inactive or not self.was_active:
                    state_history = []
                    for t, s, aH, aR, info in g.transition_history:
                        state = OvercookedState.from_dict(json.loads(s))
                        state_history.append(state)
                    mdp = OvercookedGridworld.from_layout_name(g.layout, p_slip=g.p_slip,neglect_boarders=True)
                    env = OvercookedEnv.from_mdp(mdp, horizon=360)
                    visualizer = TrajectoryVisualizer(env)
                    title=f'{self.fname}\n {self.conds[ic]} - Game {ig+1} [Inactive: {was_frozen} Frozen: {was_frozen}]'
                    approved = visualizer.preview_approve_trajectory(state_history, title = title)
                    approvals.append(approved)

                    if not approved:
                        break
                else:
                    approvals.append(True)


        all_approved = np.all(approvals)
        if not all_approved:
            print(f"Warning: Participant {self.fname} failed manual review",file=sys.stderr)
        return all_approved,approvals

    def compute_frozen(self, start_step = 30, verbose=True):
        """Check if any agent was frozen for <thresh> steps"""
        human_frozen = copy.deepcopy(self._def_dict)
        robot_frozen = copy.deepcopy(self._def_dict)
        any_frozen = False

        for ic in range(len(self.conds)):
            cond_games = self._games[self.conds[ic]]
            for g in cond_games:
                maxH, maxR = 0, 0
                nH,nR = 0,0 # count of same state conseutively

                t, s, aH, aR, info = g.transition_history[start_step + 0]
                last_state = OvercookedState.from_dict(json.loads(s))
                for t, s, aH, aR, info in g.transition_history[start_step + 1:]:
                    state = OvercookedState.from_dict(json.loads(s))
                    sameH = state.players[0] == last_state.players[0]
                    sameR = state.players[1] == last_state.players[1]

                    if sameH:   nH += 1
                    else:       nH = 0

                    if sameR:   nR += 1
                    else:       nR = 0

                    maxH = max(maxH,nH)
                    maxR = max(maxR,nR)
                    last_state = state

                frozenH = (maxH >= self._max_frozen_steps)
                frozenR = (maxR >= self._max_frozen_steps)
                any_frozen = frozenH or frozenR or any_frozen

                if verbose:
                    if frozenH:
                        print(f"Warning: Participant {self.fname} had frozen HUMAN in layout {g.layout} (max frozen={maxH})",file=sys.stderr)
                    if frozenR:
                        print(f"Warning: Participant {self.fname} had frozen ROBOT in layout {g.layout} (max frozen={maxR})",file=sys.stderr)

                human_frozen[self.conds[ic]].append(frozenH)
                human_frozen[ic].append(frozenH)

                robot_frozen[self.conds[ic]].append(frozenR)
                robot_frozen[ic].append(frozenR)

        return human_frozen, robot_frozen, any_frozen

    def compute_saturated_trust(self,empty_val = ''):
        saturated_thresh= 0.1
        trust_score_sat = copy.deepcopy(self.trust_scores)
        dtrust_sat = copy.deepcopy(self.delta_trusts)
        for partner in ['rs-tom', 'rational']:
            max_val = 10
            values = trust_score_sat[partner].copy()
            values_convol = np.vstack([values[0:-1], values[1:]])
            is_saturated = np.all((values_convol > max_val * (1- saturated_thresh)) |
                                  (values_convol < max_val * (saturated_thresh)), axis=0)
            for i, is_sat in enumerate(is_saturated):
                if is_sat:
                    trust_score_sat[partner][i+1] = empty_val
                    trust_score_sat[self.conds.index(partner)][i+1] = empty_val
                    dtrust_sat[partner][i] = empty_val
                    dtrust_sat[self.conds.index(partner)][i] = empty_val
        return trust_score_sat, dtrust_sat



    def _recompute_measures(self):
        warnings.warn("Recomputing some measures from raw data. This should be done with raw data to save time")
        self.priming_labels, self.priming_scores, self.priming_durs = self.compute_priming_labels()
        self.beliefs, self.inferences, self.accuracies = self.compute_belief_accuracies()
        # self.beliefs, self.inferences, self.accuracies = self.recompute_belief_accuracies()
        self.trust_scores_sat, self.delta_trusts_sat = self.compute_saturated_trust()

    @property
    def include(self):
        return self._included
    @include.setter
    def include(self, val):
        self._included = int(val)

    @property
    def is_valid(self):
        # return self.was_active and self.surveys_valid and self.passed_review
        return self.surveys_valid and self.passed_review
    @property
    def PID(self):
        return self.fname.split('__')[1]
    ###############################################
    # Helper methods ##############################
    def parse_ordered_layouts(self):
        """Interates through raw data to get all games played in order"""
        layouts = copy.deepcopy(self._def_dict)
        p_slips = copy.deepcopy(self._def_dict)
        for key, cond_games in self._games.items():
            layouts[key] = [g.layout for g in cond_games]
            p_slips[key] = [g.p_slip for g in cond_games]
        return layouts,p_slips

    def parse_stage_key(self,stage_key):
        """Interates through raw data to get all games played in order"""
        all_stages = []
        stages = copy.deepcopy(self._def_dict)
        for i in range(20):
            key = f'{stage_key}{i}'
            if key in self._stage_list:
                all_stages.append(self._raw[key])
            elif "AC_" + key in self._stage_list:
                all_stages.append(self._raw["AC_" + key])
            else:
                break
        n_games = len(all_stages)

        stages[0] = all_stages[:n_games // 2]
        stages[self.conds[0]] = all_stages[:n_games // 2]

        stages[1] = all_stages[n_games // 2:]
        stages[self.conds[1]] =  all_stages[n_games // 2:]
        return stages

    def save(self, fname=None):
        fname = self.fname if fname is None else fname
        if ".pkl" not in fname: fname += ".pkl"
        if not self.is_valid:
            fname = 'REJECTED_' + fname
        if not self.is_valid:
            save_dir = PROCESSED_REJECT_DIR
        else:
            save_dir = PROCESSED_COND0_DIR if self.icond == 0 else PROCESSED_COND1_DIR

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_path = os.path.join(save_dir, fname)
        # save this DataPoint object to a pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print(f"Processed data saved to {file_path}")

    def load_raw(self, file_path):
        """Read a pickle file and return the deserialized object."""
        if ".pkl" not in file_path: file_path += ".pkl"
        if not os.path.exists(file_path):
            # check where file path is invalidated
            _dirs = str(file_path).split('\\')
            for i in range(len(_dirs)):
                _path = "\\".join(_dirs[:i+1])
                if not os.path.exists(_path):
                    raise ValueError(f"File path {file_path} is invalid at {_path}")

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

    def preload_models(self,**kwargs):
        models = copy.deepcopy(self._def_dict)
        mdp_params = kwargs.get("mdp_params", {'neglect_boarders': True})

        for ic, cond in enumerate(self.conds):
            cond_games = self._games[cond]
            for g in cond_games:


                if cond.lower() == 'rs-tom':
                    policies = self.load_policies(g.layout,g.p_slip, ['averse', 'neutral', 'seeking'])
                elif cond.lower() == 'rational':
                    policies = self.load_policies(g.layout, g.p_slip, ['neutral'])
                else:
                    raise ValueError(f"Unknown condition {cond} in filename {self.fname}")

                models[self.conds[ic]].append(policies)
                models[ic].append(policies)

        return models

    def load_policies(self, layout,p_slip, candidates):
        """
        Loads the policies for the AI. This is used to test the ToM agent and its ability to predict the actions of the AI.
        """
        candidate_fnames = [f'{layout}_pslip{str(p_slip).replace(".", "")}__{candidate}.pt' for candidate  in candidates]
        # print(f"All fnames {candidate_fnames}")
        policies = []
        for fname in candidate_fnames:
            PATH = AGENT_DIR + f'{fname}'
            if not os.path.exists(PATH):
                raise ValueError(f"Policy file {PATH} does not exist")
            try:
                policies.append(TorchPolicy(PATH,'cpu'))
            except Exception as e:
                raise print(f"Error loading policy from {PATH}\n{e}")
        return policies

    def to_pandas(self):
        """Convert the DataPoint metrics to a pandas DataFrame for easier analysis."""

        self._recompute_measures() # if necessary
        rs_thresh = -0.2565
        m_priming_score = np.mean(self.priming_scores[0] + self.priming_scores[1])
        empty_val = ''
        N = 10  # Number of data points (e.g., games)

        data = {
            # Metadata
            'include':[int(self.included)]*N,
            'ID': [self.PID] * N,
            'mdiff_reward': [np.mean(self.rewards['rs-tom']) - np.mean(self.rewards['rational'])]*N,
            'mdiff_trust': [np.mean(self.trust_scores['rs-tom']) - np.mean(self.trust_scores['rs-tom']) ]*N,
            'mdiff_dtrust': [np.mean(self.delta_trusts['rs-tom']) - np.mean(self.delta_trusts['rational'])]*N,

            'condition': [self.icond] * N,
            'age': [self.age] * N,
            'sex': [self.sex] * N,
            'risk_propensity': [self.RTP_score]*N,
            'm_priming_score': [m_priming_score]*N,
            'rs_label_zero': ['Seeking' if m_priming_score > 0 else 'Averse']*N,
            'rs_label_clustered': ['Seeking' if rs_thresh > 0 else 'Averse']*N,
            'm_human_risks': [np.mean(self.nH_risks[0] + self.nH_risks[1])] * N,
            'm_robot_risks': [np.mean(self.nR_risks[0] + self.nR_risks[1])] * N,


            # Game info
            'game_num': np.arange(10),
            'interaction_num': np.hstack([np.arange(5),np.arange(5)]),
            'partner_type': [self.conds[0]] * int(N / 2) + [self.conds[1]] * int(N / 2),
            'Reward': self.rewards[0] + self.rewards[1],
            'trust_score': self.trust_scores[0] + self.trust_scores[1],
            'dtrust': [empty_val] + self.delta_trusts[0] + [empty_val] + self.delta_trusts[1],
            'C-ACT': self.C_ACTs[0] + self.C_ACTs[1],

            'layout': self.layouts[0] + self.layouts[1],
            'p_slip': self.p_slips[0] + self.p_slips[1],
            'risk_preference': self.priming_labels[0] + self.priming_labels[1],
            'priming_score': self.priming_scores[0] + self.priming_scores[1],

            'num_human_risks': self.nH_risks[0] + self.nH_risks[1],
            'num_robot_risks': self.nR_risks[0] + self.nR_risks[1],
            'H-Pred': self.predictability[0] + self.predictability[1],
            'C-ACT': self.C_ACTs[0] + self.C_ACTs[1],
            'H-IDLE': self.H_IDLEs[0] + self.H_IDLEs[1],
            'R-IDLE': self.R_IDLEs[0] + self.R_IDLEs[1],
            'belief_accuracies': self.accuracies[0] + self.accuracies[1],

            # 'trust_slope': [self.trust_slopes[0]]*int(N/2) + [self.trust_slopes[1]]*int(N/2),
            # 'dtrust': [0] + self.delta_trusts[0] + [0] + self.delta_trusts[1],
            # 'trust_responses': self.trust_responses[0] + self.trust_responses[1],


        }

        response_items = self.get_trust_questions()
        for ipartner in range(2):
            for igame in range(5):
                for iq, qkey in enumerate(response_items):
                    key = f'trust_q{iq}_{qkey}'
                    if key not in data:
                        data[key] = []
                    data[key] += [0.1*self.trust_responses[ipartner][igame][qkey]]




        df = pd.DataFrame(data)
        return df

    def to_pandas_flat(self):
        """Convert the DataPoint metrics to a pandas DataFrame for easier analysis."""
        self._recompute_measures()  # if necessary
        m_priming_score =np.mean(self.priming_scores[0] + self.priming_scores[1])
        saturated_thresh = 0.1

        data_dict = {
            # Metadata
            'include':[int(self.included)],
            'ID': [self.PID],
            'condition': [self.icond],
            'age': [self.age],
            'sex': [self.sex],
            'risk_propensity': [self.RTP_score],
            'm_priming_score': [m_priming_score],
            # 'm_human_risks': [np.mean(self.nH_risks[0] + self.nH_risks[1])] * N,
            # 'm_robot_risks': [np.mean(self.nR_risks[0] + self.nR_risks[1])] * N,

        }

        TIMSERIES_DATA = {
            'reward': self.rewards,
            'dtrust': self.delta_trusts,
            'trust_score': self.trust_scores,
            'trust_score_sat': self.trust_scores_sat,
            'dtrust_sat': self.delta_trusts_sat,
            'CACT': self.C_ACTs,
        }
        MEAN_DIFFS = {}
        for key, all_data in TIMSERIES_DATA.items():
            means = []
            for partner in ['rs-tom', 'rational']:

                values = all_data[partner]
                for i in range(len(values)):
                    prefix = f'{partner}_G{i+1}_'
                    data_dict[prefix + key]  = [values[i]]
                # prefix = f'{partner}_mean_'
                # data_dict[prefix + key] = [np.mean(values)]
                means.append(np.mean([v for v in values if v != '']))


            MEAN_DIFFS[f'meandiff_{key}'] = means[0] - means[1]
            # prefix = f'{partner}_meandiff_'
            # data_dict[prefix + key] = [means[0] - means[1]]

        data_dict[f'RS_label'] = 'Seeking' if m_priming_score > 0 else 'Averse'

        # filter saturated trust scores
        # for partner in ['rs-tom', 'rational']:
        #     max_val = 10
        #     values = self.trust_scores[partner].copy()
        #     values_convol = np.vstack([values[0:-2],
        #                                values[1:-1]])
        #     is_saturated = np.all((values_convol > max_val-saturated_thresh) |
        #                           (values_convol < saturated_thresh), axis=0)
        #     # pad with leading 0
        #     is_saturated = np.insert(is_saturated, 0, False)
        #
        #     values[is_saturated] = 0


        # data_dict[f'rs-tom_trust_slope'] = [self.trust_slopes['rs-tom']]
        # data_dict[f'rational_trust_slope'] = [self.trust_slopes['rs-tom']]
        for key, val in MEAN_DIFFS.items():
            data_dict[key] = [val]


        df = pd.DataFrame(data_dict)
        return df

    def __repr__(self):
        KEYS = ['rs-tom', 'rational']
        meta_data = {
            'fname': self.fname,
            'RTP': self.RTP_score,
            'RelTrust': self.relative_trust_score,
            'RelRisk': self.relative_risk_perception,
            'Priming (-1A<=>1S)': np.mean([np.mean(self.priming_scores[key]) for key in KEYS]),
                     }
        print(f'Metadata:')
        for key,val in meta_data.items():
            print(f'  {key}: {val}')

        print(f'\nTrialdata:')
        data_by_index = {
            # 'Priming (-1A<=>1S)': {key: np.mean(self.priming_scores[key]) for key in KEYS},
            'Reward':           {key: np.mean(self.rewards[key]) for key in KEYS},
            '#Risk Human':      {key: np.mean(self.nH_risks[key]) for key in KEYS},
            '#Risk Robot':      {key: np.mean(self.nR_risks[key]) for key in KEYS},
            'Trust Score':      {key: np.mean(self.trust_scores[key]) for key in KEYS},
            'Risk Perception' : {key: np.mean(self.risk_perceptions[key]) for key in KEYS},
            'Delta Trust':      {key: np.mean(self.delta_trusts[key]) for key in KEYS},
            'Predictability':   {key: np.mean(self.predictability[key]) for key in KEYS},
            'C_ACT':            {key: np.mean(self.C_ACTs[key]) for key in KEYS},
            'H_IDLE':           {key: np.mean(self.H_IDLEs[key]) for key in KEYS},
            'R_IDLE':           {key: np.mean(self.R_IDLEs[key]) for key in KEYS},
        }
        df_index = pd.DataFrame.from_dict(data_by_index, orient='index')
        return str(df_index)

class TorchPolicy:
    """Handles Torch.NN interface and action selection.
    Is lightweight version of SelfPlay_QRE_OSA agent
    """

    def __init__(self, PATH, device, action_selection='softmax', ego_idx=1,
                 sophistication=8, belief_trick=True
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
        self.rationality = 1
        self.model = self.load_model(PATH)

        self.QRE = QuantalResponse_torch(rationality=self.rationality, belief_trick=belief_trick,
                                         sophistication=sophistication, joint_action_space=self.joint_action_space,
                                         device=self.device)

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
            _, _, joint_pA = self.choose_joint_action(obs)
            ego_pA = joint_pA[0, self.ego_idx]
            if self.action_selection == 'greedy':
                ego_action = Action.INDEX_TO_ACTION[np.argmax(ego_pA)]
            elif self.action_selection == 'softmax':
                ia = np.random.choice(np.arange(len(ego_pA)), p=ego_pA)
                ego_action = Action.INDEX_TO_ACTION[ia]
        return ego_action

    def choose_joint_action(self, obs, epsilon=0):
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

    def invert_obs(self, obs, N_PLAYER_FEAT=9):
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


class SurveyVisualizer:
    def __init__(self, responses, survey_range=(10,0)):
        """
        responses: dict[str, int] mapping question -> integer in [0, 9]
        """
        # Validate inputs lightly
        # vmax,vmin = survey_range
        #
        # # responses['Unresponsive'] = vmax + vmin - responses['Unresponsive']
        #
        # responses['1-(Unresponsive)'] = vmax + vmin - responses.pop('Unresponsive','ERROR: Reverse Code Name')
        #
        # _too_risky = responses.pop('Take too many risks', None)
        # # responses['Play too safe'] = vmax + vmin - responses['Play too safe']
        # responses['1-(Play too safe)'] = vmax + vmin - responses.pop('Play too safe', 'ERROR: Reverse Code Name')
        # responses['Take too many risks'] = _too_risky

        # for k, v in responses.items():
        #     if not isinstance(v, int) or not (vmin <= v <= vmax):
        #         raise ValueError(f"Response for '{k}' must be an int in [0, 10]. Got: {v}")
        self.responses = responses
        self._decision = None  # will be set True/False by buttons

    def plot_table(self, ax=None, title="Survey Responses"):
        """Draw the survey table on the given axes (or create one)."""
        questions = list(self.responses.keys())
        cols = list(range(11))
        n_questions = len(questions)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 0.2 * n_questions + 2))
        else:
            fig = ax.figure

        ax.set_axis_off()

        # Build cell text
        cell_text = []
        for q in questions:
            row = []
            for i in cols:
                if i == self.responses[q]:
                    row.append("X")
                elif "-" in q and i== 10-self.responses[q]: # reverse coded
                    row.append("-")
                else:
                    row.append("")

            # row = ["X" if i == self.responses[q] else "" for i in cols]
            cell_text.append(row)

        table = ax.table(
            cellText=cell_text,
            rowLabels=questions,
            colLabels=[str(c) for c in cols],
            loc='center',
            cellLoc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(0.7, 1)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Bold column headers
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')

        return fig, ax, table

    def review(self, title="Survey Responses", accept_label="Accept", reject_label="Reject",
               block=True, figsize=None):
        """
        Spawn an interactive review window with Accept/Reject buttons.
        Returns:
            True  -> Accept
            False -> Reject
            None  -> Window closed without selection
        """
        self._decision = None

        # Sizing
        n_rows = max(1, len(self.responses))
        if figsize is None:
            figsize = (10, 0.2 * n_rows + 2.8)  # extra height for buttons

        fig, ax = plt.subplots(figsize=figsize)

        # Draw table
        self.plot_table(ax=ax, title=title)

        #color last two rows
        table = ax.tables[0]
        n_cols = len(table.get_celld()[0,0].get_text().get_text())
        for i,q in enumerate(self.responses.keys()):
            if "-" in q:
                for col in range(11):
                    cell1 = table.get_celld()[(i+1, col)]
                    cell1.set_facecolor('#ffcccc')

        # for col in range(11):
        #     cell1 = table.get_celld()[(len(self.responses)-1, col)]
        #     cell2 = table.get_celld()[(len(self.responses), col)]
        #     cell1.set_facecolor('#ffcccc')
        #     cell2.set_facecolor('#ffcccc')

        # Button axes (in figure coordinates)
        btn_h = 0.07
        btn_w = 0.18
        y = 0.02
        ax_accept = fig.add_axes([0.25, y, btn_w, btn_h])
        ax_reject = fig.add_axes([0.57, y, btn_w, btn_h])

        btn_accept = Button(ax_accept, accept_label)
        btn_reject = Button(ax_reject, reject_label)

        # Callbacks
        def _on_accept(event):
            self._decision = True
            plt.close(fig)

        def _on_reject(event):
            self._decision = False
            plt.close(fig)

        def _on_key(event):
            # Keyboard shortcuts: 'a' accept, 'r' reject
            if event.key is None:
                return
            key = event.key.lower()
            if key == 'a':
                _on_accept(event)
            elif key == 'r':
                _on_reject(event)

        btn_accept.on_clicked(_on_accept)
        btn_reject.on_clicked(_on_reject)
        fig.canvas.mpl_connect('key_press_event', _on_key)

        # Leave some bottom margin for buttons
        plt.subplots_adjust(bottom=0.14)

        # Show and (optionally) block until user selects
        plt.show(block=block)

        # If non-blocking, return None now; decision can be read later
        return self._decision



def main():
    # fname = "2025-10-28_22-46-41__PID5dce29700ad506063969a4a5__cond1"
    # fname = "2025-10-28_23-12-28__PID58a0c507890ea500014c4e9b__cond0"
    # fname = "2025-10-28_23-57-46__PID5f3ac1732efa0a74f975b1a8__cond1"
    # fname = "2025-11-06_17-35-29__TEST_PID67f447d8bd15d28465f1ec51__cond0"
    # fname = "2025-11-06_18-23-16__PID64136bf30b27746cb96f7db8__cond1"
    # fname = "2025-12-01_22-27-04__PID63474e67a5fd298c6103c409__cond1"
    # fname = "2025-12-02_17-39-42__PID65f366dfcb46b71238e9418d__cond0"
    # fname = "2025-12-08_19-28-56__PID6788d82f8ec0422c248b737a__cond1"

    # fname = "2026-01-13_04-24-29__PID65c243c37e0d77ca70ee030e__cond0"
    #
    # # fname = "2025-12-16_04-17-35__PID66293ce3dba0764775195e58__cond0"
    # dp = DataPoint(fname)
    # dp.save()

    for fname in get_unprocessed_fnames():
        dp = DataPoint(fname)
        dp.save()


if __name__ == '__main__':
    main()