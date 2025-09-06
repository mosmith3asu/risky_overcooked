import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from risky_overcooked_py.mdp.actions import Action

class BayesianBeliefUpdate():
    def __init__(self, partner_agents,response_agents,
                 capacity=10, alpha = 0.99,
                 names=None,title='',iego=0,ipartner=1):
        """
        Runs a Bayesian belief update on the partner's policy given a sequence of observations and actions
        :param partner_agents: possible policies for the (human) partner
        :param response_agents: possible paired (by index) policies for the (robot) ego agent
        :param capacity: number of transitions in memory to performe belief over
        :param alpha: decayed weighting on previous observatiosn
        :param names: names of policies (for plotting)
        :param title: title of plot
        """
        self.iego,self.ipartner = iego,ipartner
        self.candidate_partners = partner_agents
        self.candidate_responses = response_agents
        assert len(partner_agents) == len(response_agents), 'Must have same number of partner and response agents'
        self.n_candidates = len(partner_agents)
        self.candidate_likelihood_mem = [deque(maxlen=capacity) for _ in range(self.n_candidates)]

        self.belief = np.ones(len(partner_agents)) / len(partner_agents)
        self.belief_history = [self.belief]
        self.title = title
        self.names = names
        self.alpha = alpha




    def reset_prior(self):
        self.belief = np.ones(self.n_candidates) / self.n_candidates

    def update_belief(self, obs, action):
        """https://towardsdatascience.com/how-to-use-bayesian-inference-for-predictions-in-python-4de5d0bc84f3"""

        prior = self.belief

        likelihood = np.array([self.get_prob_partner_action(partner, obs, action)  for partner in self.candidate_partners])
        # likelihood += 1e-32
        likelihood = likelihood / np.sum(likelihood)
        for i, l in enumerate(likelihood):
            self.candidate_likelihood_mem[i].append(l)

        n_samples = len(self.candidate_likelihood_mem[0])
        decay_vec = np.array([self.alpha**(n_samples-i-1) for i in range(n_samples)])

        cum_likelihood = np.zeros(self.belief.size)
        for i, likelihoods in enumerate(self.candidate_likelihood_mem):
            cum_likelihood[i] = np.sum(np.array(likelihoods) * decay_vec)
        cum_likelihood = cum_likelihood / np.sum(cum_likelihood)
        assert not np.any(np.isnan(cum_likelihood)), 'NaN in cum_likelihood'


        pE = np.sum([cum_likelihood[i] * prior[i] for i in range(len(self.candidate_partners))])
        posterior = prior * cum_likelihood / pE
        posterior = posterior / np.sum(posterior)
        posterior = np.clip(posterior, 0.05, 1.0)  # Avoid NaN in belief
        posterior = posterior / np.sum(posterior)
        self.belief = posterior
        self.belief_history.append(self.belief)
        # pE = np.sum([likelihood[i] * prior[i] for i in range(len(self.candidate_partners))])
        # posterior = prior * likelihood/pE
        # posterior = posterior/np.sum(posterior)
        # self.belief = posterior
        # self.belief_history.append(self.belief)

    def update_belief(self, obs, action, is_only_partner_action=False):
        """https://towardsdatascience.com/how-to-use-bayesian-inference-for-predictions-in-python-4de5d0bc84f3"""

        prior = self.belief

        likelihood = np.array([self.get_prob_partner_action(partner, obs, action,is_only_partner_action=is_only_partner_action)
                               for partner in self.candidate_partners])
        # likelihood += 1e-32
        likelihood = likelihood / np.sum(likelihood)
        for i, l in enumerate(likelihood):
            self.candidate_likelihood_mem[i].append(l)

        n_samples = len(self.candidate_likelihood_mem[0])
        decay_vec = np.array([self.alpha**(n_samples-i-1) for i in range(n_samples)])

        cum_likelihood = np.zeros(self.belief.size)
        for i, likelihoods in enumerate(self.candidate_likelihood_mem):
            cum_likelihood[i] = np.sum(np.array(likelihoods) * decay_vec)
        cum_likelihood = cum_likelihood / np.sum(cum_likelihood)
        assert not np.any(np.isnan(cum_likelihood)), 'NaN in cum_likelihood'


        pE = np.sum([cum_likelihood[i] * prior[i] for i in range(len(self.candidate_partners))])
        posterior = prior * cum_likelihood / pE
        posterior = posterior / np.sum(posterior)
        posterior = np.clip(posterior, 0.05, 1.0)  # Avoid NaN in belief
        posterior = posterior / np.sum(posterior)
        self.belief = posterior
        self.belief_history.append(self.belief)
        # pE = np.sum([likelihood[i] * prior[i] for i in range(len(self.candidate_partners))])
        # posterior = prior * likelihood/pE
        # posterior = posterior/np.sum(posterior)
        # self.belief = posterior
        # self.belief_history.append(self.belief)
    def get_prob_partner_action(self,agent,obs,action,is_only_partner_action=False):
        """
        Gets probability that the partner took action given the observation
        - CPT partner assumes ego follows same policy
        """

        if not is_only_partner_action:
            # partner_action_index = joint_action_idx % 6
            action = Action.INDEX_TO_ACTION_INDEX_PAIRS[action][self.ipartner]

        _, _, action_probs = agent.choose_joint_action(obs, epsilon=0)
        return float(action_probs[0, self.ipartner, action])

    @property
    def most_likely_partner(self):
        return self.candidate_partners[np.argmax(self.belief)]

    @property
    def best_response(self):
        return self.candidate_responses[np.argmax(self.belief)]

    def plot_belief_history(self):
        fig,ax = plt.subplots()
        ax.plot(self.belief_history)
        ax.legend(self.names)
        ax.set_title(self.title)
        plt.show()

    def __repr__(self):
        print_dict = {}
        for i in range(self.n_candidates):
            agent_name = self.names[i] if self.names else f'Agent {i}'
            val =self.belief[i]
            print_dict[agent_name] = round(val,2)

        return str(print_dict)
class SimulatedAgent():
    dists = [
        np.array([[[0, 0, 0, 0, 0, 0], [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]]]),
        np.array([[[0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]]]),
        np.array([[[0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.5, 0.1, 0.1]]]),
        np.array([[[0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.5, 0.1, 0.1, 0.1]]]),
    ]
    def __init__(self,i):
        self.p_action = SimulatedAgent.dists[i]

    def sample_action(self):
        return np.random.choice(np.arange(6), p=self.p_action[0,1])
    def choose_joint_action(self,obs,epsilon=0):
        action_dists = self.p_action
        return None,None, action_dists

def main():
    all_beliefs = []
    SAMPLED_AGENT = 1
    partner_agents = [SimulatedAgent(i) for i in range(len(SimulatedAgent.dists ))]
    response_agents = [SimulatedAgent(i) for i in range(len(SimulatedAgent.dists ))]
    belief_updater = BayesianBeliefUpdate(partner_agents, response_agents,
                                          title=f'True: Agent {SAMPLED_AGENT}',
                                          names=[f'Agent {i}' for i in range(len(partner_agents))])
    print(belief_updater.belief)
    all_beliefs.append(belief_updater.belief)
    for i in range(100):
        action = partner_agents[SAMPLED_AGENT].sample_action()
        belief_updater.update_belief(None,action, agent=0)
        all_beliefs.append(belief_updater.belief)

        print(belief_updater.belief)

    belief_updater.plot_belief_history()
if __name__ =="__main__":
    main()

# DEPRICATED #################
# class BayesianBeliefUpdate():
#     def __init__(self, partner_agents,response_agents,names=None,title=''):
#         self.iego,self.ipartner = 0,1
#         self.candidate_partners = partner_agents
#         self.candidate_responses = response_agents
#         self.belief = np.ones(len(partner_agents)) / len(partner_agents)
#         self.belief_history = [self.belief]
#         self.title = title
#         self.names = names
#         # self.alpha = 0.001 # noise scale parameter
#         self.alpha = 0.0000  # noise scale parameter
#
#
#
#     def update_belief(self, obs, action):
#         """https://towardsdatascience.com/how-to-use-bayesian-inference-for-predictions-in-python-4de5d0bc84f3"""
#         obs = invert_obs(obs)
#         prior = self.belief
#
#         likelihood = np.array([self.get_prob_partner_action(partner, obs, action)  for partner in self.candidate_partners])
#         likelihood += self.alpha * np.random.rand(self.belief.size) # add noise
#         likelihood = likelihood / np.sum(likelihood)
#
#         pE = np.sum([likelihood[i] * prior[i] for i in range(len(self.candidate_partners))])
#         posterior = prior * likelihood/pE
#         posterior = posterior/np.sum(posterior)
#         self.belief = posterior
#         self.belief_history.append(self.belief)
#     def get_prob_partner_action(self,agent,obs,joint_action_idx):
#         """
#         Gets probability that the partner took action given the observation
#         - CPT partner assumes ego follows same policy
#         """
#         partner_action_index = joint_action_idx % 6
#         _, _, action_probs = agent.choose_joint_action(obs, epsilon=0)
#         return float(action_probs[0,self.ipartner,partner_action_index])
#
#     @property
#     def most_likely_partner(self):
#         return self.candidate_partners[np.argmax(self.belief)]
#
#     @property
#     def best_response(self):
#         return self.candidate_responses[np.argmax(self.belief)]
#
#     def plot_belief_history(self):
#         fig,ax = plt.subplots()
#         ax.plot(self.belief_history)
#         ax.legend(self.names)
#         ax.set_title(self.title)
#         plt.show()