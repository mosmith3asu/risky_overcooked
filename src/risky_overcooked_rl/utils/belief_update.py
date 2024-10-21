import numpy as np
import matplotlib.pyplot as plt
from src.risky_overcooked_rl.utils.state_utils import invert_obs
class BayesianBeliefUpdate():
    def __init__(self, partner_agents, response_agents):
        self.iego,self.ipartner = 0,1
        self.candidate_partners = partner_agents
        self.candidate_responses = response_agents
        self.belief = np.ones(len(partner_agents)) / len(partner_agents)
        self.belief_history = [self.belief]


    def update_belief(self, obs, action):
        """https://towardsdatascience.com/how-to-use-bayesian-inference-for-predictions-in-python-4de5d0bc84f3"""
        obs = invert_obs(obs)
        prior = self.belief

        likelihood = np.array([self.get_prob_partner_action(partner, obs, action)  for partner in self.candidate_partners])
        likelihood = likelihood / np.sum(likelihood)
        pE = np.sum([likelihood[i] * prior[i] for i in range(len(self.candidate_partners))])
        posterior = prior * likelihood/pE
        self.belief = posterior
        self.belief_history.append(self.belief)
    def get_prob_partner_action(self,agent,obs,joint_action_idx):
        """
        Gets probability that the partner took action given the observation
        - CPT partner assumes ego follows same policy
        """
        partner_action_index = joint_action_idx % 6
        _, _, action_probs = agent.choose_joint_action(obs, epsilon=0)
        return float(action_probs[0,self.ipartner,partner_action_index])

    @property
    def most_likely_partner(self):
        return self.candidate_partners[np.argmax(self.belief)]

    def plot_belief_history(self):
        plt.plot(self.belief_history)
        plt.show()

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
    belief_updater = BayesianBeliefUpdate(partner_agents, response_agents)
    print(belief_updater.belief)
    all_beliefs.append(belief_updater.belief)
    for i in range(100):
        action = partner_agents[SAMPLED_AGENT].sample_action()
        belief_updater.update_belief(None,action)
        all_beliefs.append(belief_updater.belief)

        print(belief_updater.belief)
    plt.plot(all_beliefs)
    plt.show()
if __name__ =="__main__":
    main()