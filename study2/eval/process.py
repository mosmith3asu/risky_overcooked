import copy

import pickle
import numpy as np
import torch
import json

from study2.static import *
from risky_overcooked_rl.utils.evaluation_tools import CoordinationFluency
from risky_overcooked_rl.algorithms.DDQN.utils.agents import DQN_vector_feature
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld, Action, OvercookedState
from risky_overcooked_rl.algorithms.DDQN.utils.game_thoery import QuantalResponse_torch

class DataPoint:
    def __init__(self, fname, data_path):

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
        self.priming_labels = self.compute_priming_labels()
        # self.beliefs = self.compute_belief_accuracies()

        # self.rewards = self.compute_rewards()
        # self.nH_risks, self.nR_risks = self.compute_n_risks()
        self.trust_scores,self.delta_trusts = self.compute_trust_scores()
        ## self.KL_divergences = self.compute_KL_divergences()
        self.predictability = self.compute_predictabilities()
        self.C_ACT, self.H_IDLE, self.R_IDLE = self.compute_coordination_fluency()

        # Check data validity
        self.was_active = self.check_activity()
        self.surveys_valid = self.check_surveys()
        self.passed_review = self.check_manually()

    ###############################################
    # Metadata metrics ############################
    def compute_risk_taking_propensity(self):
        vmax, vmin = 9, 1
        responses = self._raw['risk_propensity'].responses
        ID = responses.pop('ID')

        # reverse_coded = np.zeros(len(responses))
        reverse_coded = [0,1,2,4]

        raw_vals = np.array(list(responses.values()),dtype=int)
        assert len(raw_vals) == 7, "Expected 7 responses for risk taking propensity survey"
        assert np.all(raw_vals >= vmin), f"Invalid (Min) response detected in risk taking propensity survey{raw_vals}"
        assert np.all(raw_vals <= vmax), f"Invalid (Max) response detected in risk taking propensity survey{raw_vals}"

        scores = np.array(list(responses.values()),dtype=int)
        scores[reverse_coded] = vmax - scores[reverse_coded] # reverse code
        return np.mean(scores)

    def compute_relative_trust_score(self):
        vmax, vmin = 9, 1
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
        if self.icond == 1: scores = vmax - scores  # reverse code for cond1
        return np.mean(scores)

    def compute_relative_risk_perception(self):
        vmax, vmin = 9, 1
        all_responses = self._raw['relative_trust_survey'].responses

        raw_vals = np.array([
            all_responses[' Took more risks'],
            all_responses['Played more safe'],
        ], dtype=int)

        assert np.all(
            raw_vals >= vmin), f"Invalid (Min) response detected in risk taking propensity survey{raw_vals}"
        assert np.all(
            raw_vals <= vmax), f"Invalid (Max) response detected in risk taking propensity survey{raw_vals}"

        scores = raw_vals
        if self.icond == 1: scores = vmax - scores  # reverse code for cond1
        return np.mean(scores)



        reverse_coded = [0, 1, 2, 4]

        raw_vals = np.array(list(responses.values()), dtype=int)
        assert len(raw_vals) == 7, "Expected 7 responses for risk taking propensity survey"
        assert np.all(raw_vals >= vmin), f"Invalid (Min) response detected in risk taking propensity survey{raw_vals}"
        assert np.all(raw_vals <= vmax), f"Invalid (Max) response detected in risk taking propensity survey{raw_vals}"

        scores = np.array(list(responses.values()), dtype=int)
        scores[reverse_coded] = vmax - scores[reverse_coded]  # reverse code
        return np.mean(scores)
        raise NotImplementedError("Relative trust score computation not implemented yet")

    ###############################################
    # Trial metrics ###############################

    def compute_priming_labels(self):
        priming_labels = copy.deepcopy(self._def_dict)
        for ic in range(len(self.conds)):
            cond_primings = self._primings[self.conds[ic]]
            labels = [p.responses['priming'] for p in cond_primings]
            priming_labels[ic] = labels
            priming_labels[self.conds[ic]] = labels
        return priming_labels

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
                    vmax - int(s.responses['Unresponsive']),
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
        raise NotImplementedError("Activity check not implemented yet")

    def check_surveys(self):
        """Validity check: did they complete the surveys in reasonable manner?"""
        raise NotImplementedError("Survey check not implemented yet")

    def check_manually(self):
        raise NotImplementedError("Manual check not implemented yet")

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

    def save_processed(self, file_path):
        raise NotImplementedError("Processed data saving not implemented yet")

    def load_processed(self, file_path):
        raise NotImplementedError("Processed data loading not implemented yet")

    def load_raw(self, file_path):
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

    def __repr__(self):
        s = ""
        s += f"[DataPoint: {self.fname}]"
        s += f"[age:{self.age} sex:{self.sex}]" #demographic




def main():
    fname = "2025-09-15_20-08-54__PIDmax1__cond0.pkl"
    # viewer = DataViewer(fname,data_path=RAW_COND0_DIR)
    # viewer.summary()

    DataPoint(fname,data_path=RAW_COND0_DIR)


if __name__ == '__main__':
    main()