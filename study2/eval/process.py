import sys
import copy
import pickle
import numpy as np
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from study2.static import *
from risky_overcooked_rl.utils.evaluation_tools import CoordinationFluency
from risky_overcooked_rl.algorithms.DDQN.utils.agents import DQN_vector_feature
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld, Action, OvercookedState
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_rl.algorithms.DDQN.utils.game_thoery import QuantalResponse_torch
from risky_overcooked_rl.utils.visualization import TrajectoryVisualizer

class DataPoint:
    def __init__(self, fname, data_path,min_survey_var = 0.0):

        # Load raw data
        file_path = os.path.join(data_path, fname)
        self.fname = fname
        self.path = file_path
        self._raw = self.load_raw(file_path)

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

        # Begin computations
        self.RTP_score = self.compute_risk_taking_propensity()
        self.relative_trust_score = self.compute_relative_trust_score()
        self.relative_risk_perception = self.compute_relative_risk_perception()
        self.priming_labels,self.priming_scores = self.compute_priming_labels()
        # self.beliefs = self.compute_belief_accuracies()

        self.rewards = self.compute_rewards()
        self.nH_risks, self.nR_risks = self.compute_n_risks()
        self.trust_scores,self.delta_trusts = self.compute_trust_scores()
        self.risk_perceptions = self.compute_risk_perceptions()
        self.predictability = self.compute_predictabilities()
        self.C_ACTs, self.H_IDLEs, self.R_IDLEs = self.compute_coordination_fluency()

        # Check data validity
        self.min_survey_var = min_survey_var
        self.survey_var = self.compute_survey_variance()
        self.surveys_valid = self.check_surveys()
        self.was_active = self.check_activity()
        self.passed_review,self.manual_reviews = self.check_manually()



    ###############################################
    # Metadata metrics ############################
    def compute_risk_taking_propensity(self):
        vmax, vmin = 9, 0
        responses = self._raw['risk_propensity'].responses
        ID = responses.pop('ID')

        # reverse_coded = np.zeros(len(responses))
        reverse_coded = [0,1,2,4]

        raw_vals = np.array(list(responses.values()),dtype=int)
        assert len(raw_vals) == 7, "Expected 7 responses for risk taking propensity survey"
        assert np.all(raw_vals >= vmin), f"Invalid (Min) response detected in risk taking propensity survey{raw_vals}"
        assert np.all(raw_vals <= vmax), f"Invalid (Max) response detected in risk taking propensity survey{raw_vals}"

        scores = np.array(list(responses.values()),dtype=int)
        scores[reverse_coded] = vmax + vmin - scores[reverse_coded] # reverse code
        return np.mean(scores)

    def compute_relative_trust_score(self):
        vmax, vmin = 9, 0
        all_responses = self._raw['relative_trust_survey'].responses

        raw_vals = np.array([
            all_responses['More dependable'],
            all_responses['More reliable'],
            all_responses['More predictable'],
            all_responses['Acted more consistently'],
            all_responses['Better met the needs of the task'],
            all_responses['Better performed as expected'],
        ],dtype=int)
        assert np.all(raw_vals >= vmin), f"Invalid (Min) response detected in risk taking propensity survey{raw_vals}"
        assert np.all(raw_vals <= vmax), f"Invalid (Max) response detected in risk taking propensity survey{raw_vals}"

        scores = raw_vals
        if self.icond == 1: scores = vmax + vmin - scores  # reverse code for cond1
        return np.mean(scores)

    def compute_relative_risk_perception(self):
        vmax, vmin = 9, 0
        all_responses = self._raw['relative_trust_survey'].responses

        raw_vals = np.array([
            all_responses[' Took more risks'],
            vmax + 1 - int(all_responses['Played more safe']),
        ], dtype=int)

        assert np.all(raw_vals >= vmin), f"Invalid (Min) response detected in risk taking propensity survey {raw_vals}"
        assert np.all(raw_vals <= vmax), f"Invalid (Max) response detected in risk taking propensity survey {raw_vals}"

        scores = raw_vals
        if self.icond == 1: scores = vmax + vmin - scores  # reverse code for cond1
        return np.mean(scores)

    ###############################################
    # Trial metrics ###############################

    def compute_priming_labels(self):
        priming_labels = copy.deepcopy(self._def_dict)
        priming_scores = copy.deepcopy(self._def_dict)
        averse_responses = ['Take the longer detour that avoids all puddles',
                            'Pass objects to partner using counter tops to avoid all puddles' ]
        rational_responses = ['Take the middle route through one puddle',]
        seeking_responses = ['Take the most direct route by going through two puddles']
        for ic in range(len(self.conds)):
            cond_primings = self._primings[self.conds[ic]]
            labels = []
            scores = []

            for p in cond_primings:
                if p.responses['priming'] in averse_responses:
                    label,score = 'averse', -1
                elif p.responses['priming'] in rational_responses:
                    label,score = 'rational', -1
                elif p.responses['priming'] in seeking_responses:
                    label,score = 'seeking',1
                else:
                    raise ValueError(f"Unknown priming response {p.responses['priming']} in filename {self.fname}")
                labels.append(label)
                scores.append(score)

            priming_labels[ic] = labels
            priming_labels[self.conds[ic]] = labels
            priming_scores[ic] = scores
            priming_scores[self.conds[ic]] = scores

        return priming_labels, priming_scores

    def compute_belief_accuracies(self):
        raise NotImplementedError("Belief accuracy computation not implemented yet")

    def compute_trust_scores(self):
        # raise NotImplementedError("Trust score computation not implemented yet")
        trust_scores = copy.deepcopy(self._def_dict)
        delta_trusts = copy.deepcopy(self._def_dict)

        for ic in range(len(self.conds)):
            vmax, vmin = 9, 0
            cond_surveys = self._trust_surveys[self.conds[ic]]
            # cond_games = self._games[self.conds[ic]]
            for _is, s in enumerate(cond_surveys):
                raw_vals = [
                    s.responses['Dependable'],
                    s.responses['Reliable'],
                    vmax + vmin - int(s.responses['Unresponsive']),
                    s.responses['Predictable'],
                    s.responses['Act consistently'],
                    s.responses['Meet the needs of the task'],
                    s.responses['Perform as expected'],
                    # s.responses['Take too many risks'],
                    # s.responses['Play too safe'],
                ]
                raw_vals = np.array(raw_vals, dtype=int)
                score = np.mean(raw_vals)
                trust_scores[self.conds[ic]].append(score)
                trust_scores[ic].append(score)

                if _is >= 1:
                    prev_score = trust_scores[self.conds[ic]][_is-1]
                    dtrust = score - prev_score
                    delta_trusts[ic].append(dtrust)
                    delta_trusts[self.conds[ic]].append(dtrust)

        return trust_scores, delta_trusts

    def compute_risk_perceptions(self):
        # raise NotImplementedError("Trust score computation not implemented yet")
        rel_risk_scores = copy.deepcopy(self._def_dict)

        for ic in range(len(self.conds)):
            vmax, vmin = 9, 0
            cond_surveys = self._trust_surveys[self.conds[ic]]
            # cond_games = self._games[self.conds[ic]]
            for _is, s in enumerate(cond_surveys):
                raw_vals = [
                    s.responses['Take too many risks'],
                    vmax + vmin - int(s.responses['Play too safe']),
                ]
                raw_vals = np.array(raw_vals, dtype=int)
                score = np.mean(raw_vals)
                rel_risk_scores[self.conds[ic]].append(score)
                rel_risk_scores[ic].append(score)



        return rel_risk_scores

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
    def check_activity(self,max_idle_thresh = 0.5, mean_idle_thresh = 0.3):
        """Validity check: did they actively participate in the task?"""
        # raise NotImplementedError("Activity check not implemented yet")
        KEYS = ['rs-tom', 'rational']
        H_IDLEs = [np.mean(self.H_IDLEs[key]) for key in KEYS]

        max_idle = max(H_IDLEs)
        mean_idle = np.mean(H_IDLEs)
        was_inactive = max_idle > max_idle_thresh or mean_idle > mean_idle_thresh

        if was_inactive:
            print(f"Warning: Participant {self.fname} was inactive (max_idle={max_idle}, mean_idle={mean_idle})",file=sys.stderr)
        return not was_inactive

    def compute_survey_variance(self):
        vmax, vmin = 9, 0
        within_vars = []
        for ic, cond in enumerate(self.conds):
            cond_surveys = self._trust_surveys[self.conds[ic]]

            for _is, s in enumerate(cond_surveys):
                raw_vals = [
                    s.responses['Dependable'],
                    s.responses['Reliable'],
                    vmax + vmin - int(s.responses['Unresponsive']),
                    s.responses['Predictable'],
                    s.responses['Act consistently'],
                    s.responses['Meet the needs of the task'],
                    s.responses['Perform as expected'],
                ]
                raw_vals = np.array(raw_vals, dtype=int)
                within_vars.append(np.var(raw_vals))

        return np.mean(within_vars)

    def check_surveys(self):
        """Validity check: did they complete the surveys in reasonable manner?"""
        is_valid = self.min_survey_var <= self.survey_var
        if not is_valid:
            print(f"Warning: Participant {self.fname} had low survey variance ({self.survey_var})",file=sys.stderr)
        return is_valid

    def check_manually(self):
        approvals = []
        for ic in range(len(self.conds)):
            # Check Surveys
            for s in self._trust_surveys[self.conds[ic]]:
                s.responses.pop('ID',None)  # remove ID field if it exists
                responses = {}
                for key in s.responses:
                    responses[key] = int(s.responses[key])
                approved = SurveyVisualizer(responses).review(title='')
                print(f'{approved}')
                approvals.append(approved)
                # plot survey responses in table

            # Check Gameplay
            cond_games = self._games[self.conds[ic]]
            for g in cond_games:
                state_history = []
                for t, s, aH, aR, info in g.transition_history:
                    state = OvercookedState.from_dict(json.loads(s))
                    state_history.append(state)
                mdp = OvercookedGridworld.from_layout_name(g.layout, p_slip=g.p_slip,neglect_boarders=True)
                env = OvercookedEnv.from_mdp(mdp, horizon=360)
                visualizer = TrajectoryVisualizer(env)
                approved = visualizer.preview_approve_trajectory(state_history)
                approvals.append(approved)


        all_approved = np.all(approvals)
        if not all_approved:
            print(f"Warning: Participant {self.fname} failed manual review",file=sys.stderr)
        return all_approved,approvals

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

    def load_processed(self, file_path):
        raise NotImplementedError("Processed data loading not implemented yet")

    def load_raw(self, file_path):
        """Read a pickle file and return the deserialized object."""
        if ".pkl" not in file_path: file_path += ".pkl"
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
    def __init__(self, responses):
        """
        responses: dict[str, int] mapping question -> integer in [0, 9]
        """
        # Validate inputs lightly
        for k, v in responses.items():
            if not isinstance(v, int) or not (0 <= v <= 9):
                raise ValueError(f"Response for '{k}' must be an int in [0, 9]. Got: {v}")
        self.responses = responses
        self._decision = None  # will be set True/False by buttons

    def plot_table(self, ax=None, title="Survey Responses"):
        """Draw the survey table on the given axes (or create one)."""
        questions = list(self.responses.keys())
        cols = list(range(10))
        n_questions = len(questions)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 0.2 * n_questions + 2))
        else:
            fig = ax.figure

        ax.set_axis_off()

        # Build cell text
        cell_text = []
        for q in questions:
            row = ["X" if i == self.responses[q] else "" for i in cols]
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
    # fname = "2025-09-15_20-08-54__PIDmax1__cond0.pkl"
    fname = "2025-10-15_10-35-32__PID123__cond1"
    # viewer = DataViewer(fname,data_path=RAW_COND0_DIR)
    # viewer.summary()


    dp = DataPoint(fname,data_path=RAW_COND1_DIR)
    dp.save()
    print(dp)

if __name__ == '__main__':
    main()