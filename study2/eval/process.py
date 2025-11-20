import sys
import copy
import pickle
import numpy as np
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
# from pingouin import cronbach_alpha

from study2.static import *
from study2.eval.eval_utils import get_unprocessed_fnames

from risky_overcooked_rl.utils.evaluation_tools import CoordinationFluency
from risky_overcooked_rl.algorithms.DDQN.utils.agents import DQN_vector_feature
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld, Action, OvercookedState
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_rl.algorithms.DDQN.utils.game_thoery import QuantalResponse_torch
from risky_overcooked_rl.utils.visualization import TrajectoryVisualizer

class DataPoint:
    """
    Scores and Reverse Codes are framed as:
        - Risk-Seeking = higher score
        - More trust = higher score
        - Relative measures are framed as:
            - More trust in RS-ToM agent = higher score
            - RS-ToM agent more risk-seeking = higher score
    """
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


        # self.reverse_coded = []
        # self.reverse_coded += ['Unresponsive', 'Take too many risks', 'Played more safe']
        # self.reverse_coded += ['Safety first.', 'I do not take risks with my health.',
        #                        'I prefer to avoid risks.', 'I really dislike not knowing what is going to happen.']

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

        # Surveys (repeated measures)
        self.trust_responses, self.trust_scores, self.delta_trusts, self.trust_durs = self.compute_trust_scores()
        self.risk_perc_responses, self.risk_perc_scores, self.risk_perc_durs = self.compute_risk_perceptions()
        self.priming_labels,self.priming_scores, self.priming_durs = self.compute_priming_labels()

        # Gameplay
        self.rewards = self.compute_rewards()
        self.nH_risks, self.nR_risks = self.compute_n_risks()
        self.predictability = self.compute_predictabilities()
        self.C_ACTs, self.H_IDLEs, self.R_IDLEs = self.compute_coordination_fluency()
        self.H_frozen, self.R_frozen, self.any_frozen = self.compute_frozen()

        # Check data validity

        self.min_survey_var = min_survey_var
        # self.surveys_valid,self.survey_approval_rate = self.check_survey_validity()
        self.was_active = self.check_activity()
        self.passed_review, self.manual_reviews = self.check_game_validity()



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
                    label,score = 'rational', -1
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
        raise NotImplementedError("Belief accuracy computation not implemented yet")

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
                responses = self._exclude_survey_items(s.responses, exclude_items=exclude_items)
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
                responses = self._exclude_survey_items(s.responses, exclude_items=exclude_items)
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

    def check_survey_validity(self, approval_thresh=0.9,min_dur=10):
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
            if any(fails.values()) or (dur < min_dur):
                print(f"\nWarning: Participant {self.fname} failed survey validity check:"
                      f"\n\t| Duration = {dur}"
                      f"\n\t| raw_var={raw_var}, rc_var={rc_var}, fails={fails}",file=sys.stderr)
                approved = SurveyVisualizer(rps).review(title=f' ({dur:.1f}s) [self.fname]')
            else: approved = True
            approvals.append(approved)

        # Repeated surveys ###############
        for survey in repeated_survey:
            for cond in self.conds:
                for i, rps in enumerate(survey[cond]):
                    rps = copy.deepcopy(rps)
                    rps.update(self.risk_perc_responses[cond][i])

                    raw_var, rc_var, fails = self._score_survey_validity(rps)
                    if True:
                    # if any(fails.values()):
                    #     print(f"Warning: Participant {self.fname} failed survey validity check:"
                    #           f"\n raw_var={raw_var}, rc_var={rc_var}, fails={fails}", file=sys.stderr)
                        approved = SurveyVisualizer(rps).review(title='')
                    else:
                        approved = True
                    approvals.append(approved)
        # for survey in repeated_survey:
        #     for cond in self.conds:
        #         for rps in survey[cond]:
        #             raw_var, rc_var, fails = self._score_survey_validity(rps)
        #             # if True:
        #             if any(fails.values()):
        #                 print(f"Warning: Participant {self.fname} failed survey validity check:"
        #                       f"\n raw_var={raw_var}, rc_var={rc_var}, fails={fails}", file=sys.stderr)
        #                 approved = SurveyVisualizer(rps).review(title='')
        #             else:
        #                 approved = True
        #             approvals.append(approved)





        # for cond in self.conds:
        #     cond_surveys = copy.deepcopy(repeated_surveys[0][cond])
        #     for survey in cond_surveys:
        #         raw_var, rc_var, fails = self._score_survey_validity(survey)
        #         if any(fails.values()):
        #             print(f"Warning: Participant {self.fname} failed survey validity check:"
        #                   f" raw_var={raw_var}, rc_var={rc_var}, fails={fails}",file=sys.stderr)
        #             approved = SurveyVisualizer(survey).review(title='')
        #         else:
        #             approved = True
        #         approvals.append(approved)
        #
        approval_rate = np.array(approvals, dtype=int).mean()
        is_valid = approval_rate > approval_thresh


        if not is_valid:
            print(f"\n\nWarning: Participant {self.fname} failed survey validity with approval rate {approval_rate}")
            print(f'Rejection Messsage:')
            print(f"Unfortunately, your responses to {(1-approval_rate)*100:0.1f}% of the provided surveys"
                  f" failed one or more conservative validity checks, which indicate that they may not have been completed with sufficient attention or effort."
                  f" This result was determined by analysis of variance in reverse-coded questions (i.e., questions with opposite meaning) which show"
                  f" inconsistent responses in favor of random or inattentive answering patterns rather than reflective engagement with the content ."
                  " \n\nAs a result, we are unable to approve this submission. We appreciate your time, but to ensure data quality"
                  " and fairness across participants, only valid and attentive responses can be accepted.")
        else:
            print(f'Participant {self.fname} passed survey validity with approval rate {approval_rate}')
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
                    title=f' {self.conds[ic]} - Game {ig+1} [Inactive: {was_frozen} Frozen: {was_frozen}]'
                    approved = visualizer.preview_approve_trajectory(state_history, title = title)
                    approvals.append(approved)
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



    @property
    def is_valid(self):
        return self.was_active and self.surveys_valid and self.passed_review

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

    # def load_processed(self, file_path):
    #     # raise NotImplementedError("Processed data loading not implemented yet")
    #     if ".pkl" not in file_path: file_path += ".pkl"
    #     with open(file_path, 'rb') as file:
    #         data_point = pickle.load(file)
    #     return data_point

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
        self.rationality = 20
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
    fname = "2025-10-28_22-46-41__PID5dce29700ad506063969a4a5__cond1"
    # fname = "2025-10-28_23-12-28__PID58a0c507890ea500014c4e9b__cond0"
    # fname = "2025-10-28_23-57-46__PID5f3ac1732efa0a74f975b1a8__cond1"
    # fname = "2025-11-06_17-35-29__TEST_PID67f447d8bd15d28465f1ec51__cond0"
    # fname = "2025-11-06_18-23-16__PID64136bf30b27746cb96f7db8__cond1"
    # fname = "2025-11-06_20-10-11__PID61501cb61a74bfb111a98657__cond0"
    dp = DataPoint(fname)
    dp.save()
    # for fname in get_unprocessed_fnames():
    #     dp = DataPoint(fname)
    #     dp.save()


if __name__ == '__main__':
    main()