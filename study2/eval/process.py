import copy
from study2.static import *
import pickle
import numpy as np

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

        # Begin computations
        self.RTP_score = self.compute_risk_taking_propensity()
        self.relative_trust_score = self.compute_relative_trust_score()
        self.relative_risk_perception = self.compute_relative_risk_perception()

        self.layouts = self.compute_layout_orders()
        self.priming_labels = self.compute_priming_labels()
        # self.beliefs = self.compute_belief_accuracies()

        # self.rewards = self.compute_rewards()
        # self.nH_risks, self.nR_risks = self.compute_n_risks()
        self.trust_scores,self.delta_trusts = self.compute_trust_scores()
        ## self.KL_divergences = self.compute_KL_divergences()
        # self.predictability = self.compute_predictabilities()
        self.coactivities = self.compute_coactivities()

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
    def compute_layout_orders(self):
        """Interates through raw data to get all games played in order"""
        layouts = copy.deepcopy(self._def_dict)
        for key,cond_games in self._games.items():
            layouts[key] = [g.layout for g in cond_games]
        return layouts

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
                    s.responses['Act consitently'],
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

    def compute_coactivities(self):
        raise NotImplementedError("Coactivity computation not implemented yet")

    def compute_KL_divergences(self):
        raise NotImplementedError("Predictability computation not implemented yet")

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



def main():
    fname = "2025-09-15_20-08-54__PIDmax1__cond0.pkl"
    # viewer = DataViewer(fname,data_path=RAW_COND0_DIR)
    # viewer.summary()

    DataPoint(fname,data_path=RAW_COND0_DIR)


if __name__ == '__main__':
    main()