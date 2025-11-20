import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from risky_overcooked_rl.algorithms.DDQN.utils.agents import SelfPlay_QRE_OSA_CPT
from risky_overcooked_rl.utils.belief_update import BayesianBeliefUpdate
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from itertools import count
from src.risky_overcooked_py.mdp.actions import Action
import study1 as Algorithm
from risky_overcooked_rl.utils.evaluation_tools import CoordinationFluency
import pickle
import pandas as pd


class Simulator():
    def __init__(self,layout,p_slip,n_trials=1000,rationality=10,horizon=400,time_cost=0.0):
        self.layout = layout
        self.p_slip = p_slip
        self.n_trials = n_trials
        self.horizon = horizon

        # Parse Config ---------------------------------------------------------
        config = Algorithm.get_default_config()
        config['agents']['model']["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
        config['env']['LAYOUT'] = layout
        config['env']['p_slip'] = p_slip
        config["ALGORITHM"] = 'Evaluate-' + config['ALGORITHM']
        config['env']['HORIZON'] = horizon
        config['env']['time_cost'] = time_cost

        # set up env ---------------------------------------------------------
        mdp = OvercookedGridworld.from_layout_name(config['env']['LAYOUT'])
        mdp.p_slip = config['env']['p_slip']
        obs_shape = mdp.get_lossless_encoding_vector_shape()
        n_actions = 36
        self.env = OvercookedEnv.from_mdp(mdp, horizon=config['env']['HORIZON'], time_cost=config['env']['time_cost'])

        # load policies ---------------------------------------------------------
        policy_fnames = {
            # 'Rational': f'{layout}_pslip0{int(p_slip*10)}__rational',
            # 'Averse':   f'{layout}_pslip0{int(p_slip * 10)}__b00_lam225_etap088_etan10_deltap061_deltan069',
            # 'Seeking':  f'{layout}_pslip0{int(p_slip * 10)}__b00_lam044_etap10_etan088_deltap061_deltan069'
            'Rational': f'{layout}_pslip{f"{p_slip}".replace(".", "")}__rational',
            'Averse':   f'{layout}_pslip{f"{p_slip}".replace(".", "")}__b-02_lam225_etap088_etan10_deltap061_deltan069',
            'Seeking':  f'{layout}_pslip{f"{p_slip}".replace(".", "")}__b-02_lam044_etap10_etan088_deltap061_deltan069'

        }
        self.policies = {}
        for p in policy_fnames:
            self.policies[p] = SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config['agents'],policy_fnames[p],
                                                          save_dir=Algorithm.get_absolute_save_dir())
            self.policies[p].rationality = rationality
            self.policies[p].model.eval()
        # self.policies = {
        #     'Rational': SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config, policy_fnames['Rational']),
        #     'Averse': SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config,  policy_fnames['Averse']),
        #     'Seeking': SelfPlay_QRE_OSA_CPT.from_file(obs_shape, n_actions, config,  policy_fnames['Seeking'])
        # }
        # for p in self.policies.keys():
        #     self.policies['Rational'].rationality = rationality
        #     self.policies[p].model.eval()



        # Testing Params ---------------------------------------------------------
        self.human_policies = ['Seeking', 'Averse']
        self.robot_conditions = ['Oracle', 'RS-ToM', 'Rational']
        # self.data = {
        #     'Reward': {'Seeking': {'Oracle': [], 'RS-ToM': [], 'Rational': []},
        #                'Averse': {'Oracle': [], 'RS-ToM': [], 'Rational': []}, },
        #     'Risks Taken': {'Seeking': {'Oracle': [], 'RS-ToM': [], 'Rational': []},
        #                     'Averse': {'Oracle': [], 'RS-ToM': [], 'Rational': []}, },
        #     'Robot Predictability': {'Seeking': {'Oracle': [], 'RS-ToM': [], 'Rational': []},
        #                              'Averse': {'Oracle': [], 'RS-ToM': [], 'Rational': []}, },
        #     'Human Predictability': {'Seeking': {'Oracle': [], 'RS-ToM': [], 'Rational': []},
        #                              'Averse': {'Oracle': [], 'RS-ToM': [], 'Rational': []}, },
        # }
        self.data = {
            'Reward': {'Seeking': {'Oracle': [], 'RS-ToM': [], 'Rational': []},
                       'Averse': {'Oracle': [], 'RS-ToM': [], 'Rational': []}, },
            'Risks Taken': {'Seeking': {'Oracle': [], 'RS-ToM': [], 'Rational': []},
                            'Averse': {'Oracle': [], 'RS-ToM': [], 'Rational': []}, },
            'Robot Predictability': {'Seeking': {'Oracle': [], 'RS-ToM': [], 'Rational': []},
                                     'Averse': {'Oracle': [], 'RS-ToM': [], 'Rational': []}, },
            'Human Predictability': {'Seeking': {'Oracle': [], 'RS-ToM': [], 'Rational': []},
                                     'Averse': {'Oracle': [], 'RS-ToM': [], 'Rational': []}, },
            'R-IDLE': {'Seeking': {'Oracle': [], 'RS-ToM': [], 'Rational': []},
                                     'Averse': {'Oracle': [], 'RS-ToM': [], 'Rational': []}, },
            'H-IDLE': {'Seeking': {'Oracle': [], 'RS-ToM': [], 'Rational': []},
                                     'Averse': {'Oracle': [], 'RS-ToM': [], 'Rational': []}, },
            'C-ACT': {'Seeking': {'Oracle': [], 'RS-ToM': [], 'Rational': []},
                                     'Averse': {'Oracle': [], 'RS-ToM': [], 'Rational': []}, },
        }


        # Plotting params
        self.colors = {'Oracle': tuple([50 / 255 for _ in range(3)]), 'RS-ToM': (255 / 255, 154 / 255, 0),
                       'Rational': (255 / 255, 90 / 255, 0),
                       'Seeking': (128/255, 0, 0), 'Averse': (255 / 255, 154 / 255, 0)}

    def run(self):
        torch.seed()
        random.seed()
        np.random.seed()

        stat_samples = {
            'risks_taken': [],
            'reward': [],
            'predictabilityH': [],
            'predictabilityR': [],

        }

        total_trials = len(self.robot_conditions) * len(self.human_policies) * self.n_trials

        for i,robot in enumerate(self.robot_conditions):
            for j,human in enumerate(self.human_policies):

                # Form robot condition ---------------------------------------------------------
                if robot == 'Oracle':
                    names = [human]
                    models = [self.policies[human]]
                elif robot == 'RS-ToM':
                    names = list(self.policies.keys())
                    models = list(self.policies.values())
                elif robot == 'Rational':
                    names = ['Rational']
                    models = [self.policies['Rational']]
                else: raise ValueError(f'Invalid robot condition: {robot}')

                # Form belief updater ---------------------------------------------------------
                belief_updater = BayesianBeliefUpdate(models, models, names=names,
                                                      title=f'Belief | {human} Partner')

                prev_trials = (i + j) * len(self.human_policies)
                for _ in range(self.n_trials):
                    belief_updater.reset_prior()
                    stats = self.simulate_trial(self.policies[human], belief_updater)
                    self.data['Reward'][human][robot].append(stats['reward'])
                    self.data['Risks Taken'][human][robot].append(stats['risks_taken'])
                    self.data['Robot Predictability'][human][robot].append(stats['predictabilityR'])
                    self.data['Human Predictability'][human][robot].append(stats['predictabilityH'])

                    self.data['R-IDLE'][human][robot].append(stats['R_IDLE'])
                    self.data['H-IDLE'][human][robot].append(stats['H_IDLE'])
                    self.data['C-ACT'][human][robot].append(stats['C_ACT'])

                    prog = round((prev_trials + _ + 1) / total_trials * 100, 2)
                    print(f'\r{prog}% complete | ',end='')
        print('\n')
        # self.plot_results()


    def interaction_plot(self):
        plt.ioff()
        x_names =  self.robot_conditions
        line_names = self.human_policies
        lstyle = ':'
        mstyle = 'o'
        msize = 4
        mfill = 'none'
        xoff = 0.05
        features = list(self.data.keys())
        fig, axs = plt.subplots(1, len(features), figsize=(23, 3),subplot_kw=dict(box_aspect=1),constrained_layout=True)
        fig.suptitle(f'{self.layout}  [pslip={self.p_slip} | tests={self.n_trials}]\n ')
        for i, feature in enumerate(features):
            d = self.data[feature]
            for l,line in enumerate(line_names):
                x = np.arange(len(x_names)) + (xoff if l == 0 else -xoff)
                y = [np.mean(d[line][x_name]) for x_name in x_names]
                std = [np.std(d[line][x_name]) for x_name in x_names]
                c =self.colors[line]
                axs[i].plot(x, y, label=line, linestyle=lstyle, marker=mstyle, markersize=msize,
                            fillstyle=mfill, color=c)
                axs[i].set_xticks(x)
                axs[i].set_xticklabels(x_names)
                axs[i].set_ylabel(f'{feature}')
                axs[i].set_xlim([min(x) - 0.5, max(x) + 0.5])
                axs[i].legend()
                axs[i].set_title('RPM' if self.layout=='risky_multipath' else 'RCR')

                # axs[i].set_aspect('equal', adjustable='box')
                # plt.axis('square')
        plt.show()

        # INVERTED #####################################################################################################
        # plt.ioff()
        # # x_names =  self.robot_conditions
        # # line_names = self.human_policies
        # x_names = self.human_policies
        # line_names = self.robot_conditions
        # lstyle = ':'
        # mstyle = 'o'
        # msize = 4
        # mfill = 'none'
        # xoff = 0.1
        # features = list(self.data.keys())
        # fig, axs = plt.subplots(1, len(features), figsize=(23, 3), subplot_kw=dict(box_aspect=1),
        #                         constrained_layout=True)
        # fig.suptitle(f'{self.layout}  [pslip={self.p_slip} | tests={self.n_trials}]\n ')
        # for i, feature in enumerate(features):
        #     d = self.data[feature]
        #     for l, line in enumerate(line_names):
        #         x = np.arange(len(x_names)) + (xoff * (l - 1))
        #         y = [np.mean(d[x_name][line]) for x_name in x_names]
        #         std = [np.std(d[x_name][line]) for x_name in x_names]
        #         c = self.colors[line]
        #         axs[i].plot(x, y, label=line, linestyle=lstyle, marker=mstyle, markersize=msize,
        #                     fillstyle=mfill, color=c)
        #         # y = [np.mean(d[line][x_name]) for x_name in x_names]
        #         # std = [np.std(d[line][x_name]) for x_name in x_names]
        #         # axs[i].plot(x, y, label=line, linestyle=lstyle, marker=mstyle, markersize=msize, fillstyle=mfill)
        #         axs[i].set_xticks(x)
        #         axs[i].set_xticklabels(x_names)
        #         axs[i].set_ylabel(f'{feature}')
        #         axs[i].set_xlim([min(x) - 0.5, max(x) + 0.5])
        #         axs[i].legend()
        #
        #         # axs[i].set_aspect('equal', adjustable='box')
        #         # plt.axis('square')
        # plt.show()



    def save_individual_plots(self,dir='results/'):
        plt.ioff()
        features = list(self.data.keys())
        plt.rcParams["font.size"] = "14"
        layout_acc = 'RMP' if self.layout=='risky_multipath' else 'RCR'
        add_legend = (self.layout=='risky_multipath')
        add_yinfo = (not self.layout=='risky_multipath')
        for i, feature in enumerate(features):
            fig, ax = plt.subplots(1, 1, figsize=(6, 3.1))
            # fig, ax = plt.subplots(1, 1, figsize=(3, 4))
            # fig.subplots_adjust(right=0.75)
            self.feature_barplot(feature, ax,
                                 legend=add_legend,
                                 add_yinfo=add_yinfo,
                                 # title=layout_acc
                                 )
            # fig.suptitle(f'{self.layout}  [pslip={self.p_slip} | tests={self.n_trials}]\n ')
            # fig.savefig(f'{feature}.png')
            # plt.close(fig)
            fig.tight_layout()
            fig.subplots_adjust(top=0.95,bottom=0.13,left = 0.144,right = 0.684)
            plt.savefig(f"{dir}Fig_{layout_acc}_{feature.replace(' ','').replace('-','')}.svg",bbox_inches='tight')
        plt.show()

    def feature_barplot(self,feature_name,ax,bar_width=0.35, legend=False,title=None,add_yinfo=True,xinfo=True,annotate=None):
        ylims = {
            # 'Reward': [-40 + reward_offset, 50 + reward_offset],
            'Reward': [0, 105 * (self.horizon / 200)],
            'Risks Taken': [0, 50 * (self.horizon / 200)],
            'Robot Predictability': [0, 1],
            'Human Predictability': [0, 1],
            'R-IDLE': [0, 1],
            'H-IDLE': [0, 1],
            'C-ACT': [0, 1],
        }

        agents = self.human_policies
        colors = self.colors
        x = np.arange(len(agents))
        d = self.data[feature_name]


        ax.bar(x - bar_width / 2, [np.mean(d[a]['Oracle']) for a in agents],
                   bar_width / 2, yerr=[np.std(d[a]['Oracle']) for a in agents],
                   label='Oracle', facecolor=colors['Oracle'], capsize=5)

        ax.bar(x, [np.mean(d[a]['RS-ToM']) for a in agents],
                   bar_width / 2, yerr=[np.std(d[a]['RS-ToM']) for a in agents],
                   label='RS-ToM', facecolor=colors['RS-ToM'], capsize=5)

        ax.bar(x + bar_width / 2, [np.mean(d[a]['Rational']) for a in agents],
                   bar_width / 2, yerr=[np.std(d[a]['Rational']) for a in agents],
                   label='Rational', facecolor=colors['Rational'], capsize=5)

        ax.set_xticks(x)
        # ax.set_xticklabels([f'Risk-{name}\n Partner' for name in agents])
        # ax.set_xticklabels([f'Risk-{name}' for name in agents])

        # ax.set_ylabel(f'{feature_name}' + (' (no time-cost)' if feature == 'Reward' else ''))
        ax.set_ylim(ylims[feature_name])

        if title is not None:
            ax.set_title(title)
        if legend:
            ax.legend(bbox_to_anchor=(1.04, 1.00), loc="upper left",borderaxespad=0)
        if xinfo:
            ax.set_xticklabels([f'{name}' for name in agents])
        else:
            ax.set_xticklabels([])

        if add_yinfo:
            label = feature_name
            label = label.replace('Human Predictability','H-PRED')\
                .replace('Robot Predictability','R-PRED')\
                .replace('Risks Taken','# RISKS')\
                # .replace('Reward','REWARD')\

            ax.set_ylabel(f'{label}')
            ax.set_ylim(ylims[feature_name])
        else:
            ax.set_yticklabels([])
        if annotate is not None:
            ax.annotate(annotate, (0.985, 0.97), xycoords='axes fraction', ha='right', va='top', color='grey')



    def plot_results(self):
        plt.ioff()
        fig, axs = plt.subplots(1, 7, figsize=(23, 3))
        features = list(self.data.keys())

        for i, feature in enumerate(features):
            self.feature_barplot(feature,axs[i])

            if i == len(features) - 1:
                # axs[i].legend(bbox_to_anchor=(1.05, 1, 1.06, 0), loc='upper left', borderaxespad=0.)

                fig.tight_layout()
                fig.subplots_adjust(top=0.75)
                left = axs[0].get_position().x0
                right = 0.942  # for 4 plats
                right = 0.693  # for 3 plats

                # right = axs[1].get_position().x1 - 0.046
                bottom = axs[-1].get_position().y1 * 1.05
                top = 1
                bbox = (left, bottom, right, top)
                plt.legend(*axs[-1].get_legend_handles_labels(), loc='lower center', ncols=4, bbox_to_anchor=bbox,
                           bbox_transform=plt.gcf().transFigure, mode='expand', borderaxespad=0.0, fontsize=12)

        fig.suptitle(f'{self.layout}  [pslip={self.p_slip} | tests={self.n_trials}]\n ')

        plt.show()

    def ANOVA(self):
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        # reformat data
        data = self.data

        features = list(data.keys())
        agents = self.human_policies
        conditions = self.robot_conditions

        # create dataframes
        df = pd.DataFrame(columns=['agent', 'condition']+[f.replace(' ','').replace('-','') for f in features])
        df_oracle = pd.DataFrame(columns=['agent', 'condition']+[f.replace(' ','').replace('-','') for f in features])
        for agent in agents:
            for condition in conditions:
                # if condition == 'Oracle':
                #     continue

                for i in range(self.n_trials):
                # for i in range(3):
                    row = {'agent': agent, 'condition': condition}

                    for feature in features:
                        row[feature.replace(' ','').replace('-','')] = data[feature][agent][condition][i]

                    # assert np.any(row.items() != np.nan), 'NAN VALUE FOUND'
                    if condition == 'Oracle':df_oracle = df_oracle._append(row, ignore_index=True)
                    else: df = df._append(row, ignore_index=True)
                    # df = pd.concat([df, pd.DataFrame(row)])


        # Perform ANOVA
        pairwise_groups = df['agent'].astype(str) +'_'+ df['condition'].replace('-')
        df = pd.concat((df, pairwise_groups.rename('PairwiseGroup')), axis=1)

        for feature in features:
            feature = feature.replace(' ','').replace('-','')
            print('\n\n\n################################################################################')
            print(f'{feature} ###################################################################')
            print('################################################################################')
            RS_ToM = df[feature][df["condition"] == "RS-ToM"]
            RS_ToM_averse = df[feature][(df["condition"] == "RS-ToM") & (df['agent'] == 'Averse')]
            RS_ToM_seeking = df[feature][(df["condition"] == "RS-ToM") & (df['agent'] == 'Seeking')]

            Rational = df[feature][df["condition"] == "Rational"]
            Rational_averse = df[feature][(df["condition"] == "Rational") & (df['agent'] == 'Averse')]
            Rational_seeking = df[feature][(df["condition"] == "Rational") & (df['agent'] == 'Seeking')]

            Oracle = df_oracle[feature][df_oracle["condition"] == "Oracle"]
            Oracle_averse = df_oracle[feature][(df_oracle["condition"] == "Oracle") & (df_oracle['agent'] == 'Averse')]
            Oracle_seeking = df_oracle[feature][(df_oracle["condition"] == "Oracle") & (df_oracle['agent'] == 'Seeking')]

            # summary_df = pd.DataFrame(columns=['condition','Seeking','Averse', 'Total'])
            # summary_df.loc['RS-ToM'] = ['RS-ToM', np.mean(RS_ToM_seeking), np.mean(RS_ToM_averse), np.mean(RS_ToM)]
            # summary_df.loc['RS-ToM'] = ['RS-ToM', np.mean(Rational_seeking), np.mean(Rational_averse), np.mean(Rational)]
            # summary_df.loc['RS-ToM'] = ['Total', np.mean(Seeking), np.mean(Averse), '']
            summary_df = pd.DataFrame(columns=['Seeking', 'Averse', 'Total'])
            summary_df.loc['Oracle'] = [np.mean(Oracle_seeking), np.mean(Oracle_averse), np.mean(Oracle)]
            summary_df.loc['RS-ToM'] = [np.mean(RS_ToM_seeking), np.mean(RS_ToM_averse), np.mean(RS_ToM)]
            summary_df.loc['Rational'] = [np.mean(Rational_seeking), np.mean(Rational_averse), np.mean(Rational)]
            # summary_df.loc['Total'] = [np.mean(Seeking), np.mean(Averse), '']
            summary_df.loc['Diff'] = [np.mean(RS_ToM_seeking)-np.mean(Rational_seeking),
                                      np.mean(RS_ToM_averse)-np.mean(Rational_averse),
                                      np.mean(RS_ToM)-np.mean(Rational)]
            summary_df.loc['% Change'] = [(np.mean(RS_ToM_seeking) - np.mean(Rational_seeking))/np.mean(Rational_seeking),
                                        (np.mean(RS_ToM_averse) - np.mean(Rational_averse))/np.mean(Rational_averse),
                                        (np.mean(RS_ToM) - np.mean(Rational))/np.mean(Rational)
                                        ]
            print(summary_df, end='\n\n')

            print(f'% increase from Rational : {100*(np.mean(RS_ToM) - np.mean(Rational))/ np.mean(Rational)}%')

            model = ols(f'{feature} ~ C(agent) + C(condition) + C(agent):C(condition)', data=df).fit()
            res = sm.stats.anova_lm(model, typ=2)
            print(res)



            # Post-Hoc Testing
            sig_interation = res['PR(>F)']['C(agent):C(condition)'] < 0.05

            if sig_interation:
                print('\nInteraction is significant...')



                tukey_results = pairwise_tukeyhsd(df[feature], groups = df['PairwiseGroup'])

                print(tukey_results)
                # print(pairwise_tukeyhsd(df[feature], df['agent'], df['condition']))
            else:
                print('Interaction is NOT significant')

            # Post-Hoc
            # Perform Tukey's HSD test
            # tukey_results = pairwise_tukeyhsd(np.concatenate([method_A_scores, method_B_scores, method_C_scores]),
            #                                   np.concatenate(
            #                                       [['A'] * len(method_A_scores), ['B'] * len(method_B_scores),
            #                                        ['C'] * len(method_C_scores)]))


    def predictability(self,pA_k,a_k,discrete=True):
        if discrete: return int(np.argmax(pA_k) == a_k)
        else: return pA_k[a_k]


    def simulate_trial(self,partner_policy, belief):


        iego, ipartner = 0, 1
        device = partner_policy.device

        state_history = []
        obs_history = []
        action_history = []
        predictabilityH = []
        predictabilityR = []
        actionprobH = []
        actionprobR = []
        cum_reward = 0

        rollout_info = {
            'onion_risked': np.zeros([1, 2]),
            'onion_pickup': np.zeros([1, 2]),
            'onion_drop': np.zeros([1, 2]),
            'dish_risked': np.zeros([1, 2]),
            'dish_pickup': np.zeros([1, 2]),
            'dish_drop': np.zeros([1, 2]),
            'soup_pickup': np.zeros([1, 2]),
            'soup_delivery': np.zeros([1, 2]),

            'soup_risked': np.zeros([1, 2]),
            'onion_slip': np.zeros([1, 2]),
            'dish_slip': np.zeros([1, 2]),
            'soup_slip': np.zeros([1, 2]),
            'onion_handoff': np.zeros([1, 2]),
            'dish_handoff': np.zeros([1, 2]),
            'soup_handoff': np.zeros([1, 2]),

        }

        self.env.reset()
        state_history.append(self.env.state.deepcopy())

        for t in count():
            obs = self.env.mdp.get_lossless_encoding_vector_astensor(self.env.state, device=device).unsqueeze(0)

            # CHOOSE ACTIONS ---------------------------------------------------------

            # Choose Partner (Human) Action
            _, partner_iJA, partner_pA = partner_policy.choose_joint_action(obs, epsilon=0)
            partner_iA = partner_iJA % 6

            # Choose Ego Action
            ego_policy = belief.best_response
            _, ego_iJA, ego_pA = ego_policy.choose_joint_action(obs, epsilon=0)
            ego_iA = ego_iJA // 6

            # Calc Predictability
            # ego_pA = ego_pA.detach().cpu().numpy()
            # partner_pA = partner_pA.detach().cpu().numpy()

            pa_R = ego_pA[0, iego, ego_iA]
            pa_hat_H = self.predictability(ego_pA[0, ipartner], partner_iA)

            pa_H = partner_pA[0, ipartner, partner_iA]
            pa_hat_R = self.predictability(partner_pA[0, iego], ego_iA)

            actionprobR.append(pa_R)
            actionprobH.append(pa_H)

            predictabilityH.append(pa_hat_H)
            predictabilityR.append(pa_hat_R)

            # Calc Joint Action
            action_idxs = (ego_iA, partner_iA)
            joint_action_idx = Action.INDEX_TO_ACTION_INDEX_PAIRS.index(action_idxs)
            joint_action = (Action.ALL_ACTIONS[ego_iA], Action.INDEX_TO_ACTION[partner_iA])

            # UPDATE BELIEF ---------------------------------------------------------
            belief.update_belief(obs, joint_action_idx)

            # STEP ---------------------------------------------------------
            next_state, reward, done, info = self.env.step(joint_action)
            state_history.append(next_state.deepcopy())

            # LOG ---------------------------------------------------------
            obs_history.append(obs)
            action_history.append(joint_action_idx)
            cum_reward += reward
            for key in rollout_info.keys():
                rollout_info[key] += np.array(info['mdp_info']['event_infos'][key])

            if done:  break

        stats = {
            'risks_taken': np.sum(rollout_info['onion_risked']) + np.sum(rollout_info['dish_risked']) + np.sum(
                rollout_info['soup_risked']),
            'reward': cum_reward,
            'predictabilityR': np.mean(predictabilityR),
            'predictabilityH': np.mean(predictabilityH),
        }

        EVAL = CoordinationFluency(state_history)
        for key,val in EVAL.measures().items():
            stats[key] = val

        return stats

    @property
    def fname(self):
        pslip = ''.join(f'{int(self.p_slip * 10)}'.split('.'))
        fname = f'simdata_{self.layout}_pslip0{pslip}_horizon{self.horizon}_ntrials{self.n_trials}'
        return fname
    def save_data(self):
        with open(self.fname, 'wb') as f:
            pickle.dump(self.data, f)
    def load_data(self):
        with open(self.fname, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.data = loaded_dict

if __name__ == "__main__":
    sim = Simulator('risky_coordination_ring',0.4)
    # sim = Simulator('risky_multipath', 0.15)
    # #
    sim.run()
    sim.save_data()
    # # sim.load_data()
    # # #
    # # # sim.save_individual_plots()
    # sim.ANOVA()
    # # # sim.interaction_plot()
    sim.plot_results()

    #####################################################################################################
    # FORMATTED PLOTS ###################################################################################
    #####################################################################################################
    # settings = {
    #     'Reward': {'xinfo':True, 'legend':True},
    #     'C-ACT': {'xinfo': True, 'legend': False},
    #     'Risks Taken': {'xinfo':True, 'legend':True},
    #     'Human Predictability': {'xinfo':True, 'legend':False},
    #     'Robot Predictability': {'xinfo': True, 'legend': False},
    #     'R-IDLE': {'xinfo':True, 'legend':True},
    #     'H-IDLE': {'xinfo':True, 'legend':True},
    #
    # }
    #
    # RCR_sim = Simulator('risky_coordination_ring', 0.4)
    # RMP_sim = Simulator('risky_multipath', 0.15)
    #
    # RCR_sim.load_data()
    # RMP_sim.load_data()
    # SIMS = [RCR_sim, RMP_sim]
    #
    # plt.ioff()
    # features = list(RCR_sim.data.keys())
    # plt.rcParams["font.size"] = "16"
    # for i, feature in enumerate(features):
    #     # fig, axs = plt.subplots(1, 2, figsize=(10, 3.1))
    #     fig, axs = plt.subplots(1, 2, figsize=(8, 3.1))
    #     for i,sim in enumerate(SIMS):
    #         layout_acc = 'RMP' if sim.layout == 'risky_multipath' else 'RCR'
    #         add_legend = False#(sim.layout == 'risky_multipath')
    #         add_yinfo = (not sim.layout == 'risky_multipath')
    #         # fig, ax = plt.subplots(1, 1, figsize=(6, 3.1))
    #         # fig, ax = plt.subplots(1, 1, figsize=(3, 4))
    #         # fig.subplots_adjust(right=0.75)
    #         sim.feature_barplot(feature, axs[i],
    #                              legend=add_legend,
    #                              add_yinfo=add_yinfo,
    #                              # title=layout_acc,
    #                              annotate=f'({layout_acc})',
    #                             xinfo=settings[feature]['xinfo']
    #                              )
    #         # fig.suptitle(f'{self.layout}  [pslip={self.p_slip} | tests={self.n_trials}]\n ')
    #         # fig.savefig(f'{feature}.png')
    #         # plt.close(fig)
    #         fig.tight_layout()
    #         # fig.subplots_adjust(top=0.95, bottom=0.13, left=0.144, right=0.684)
    #
    #
    #         # axs[i].legend(bbox_to_anchor=(1.05, 1, 1.06, 0), loc='upper left', borderaxespad=0.)
    #
    #
    #         key_loc = 'top'
    #
    #         # TOPKEY
    #         adj = 0.16
    #         if key_loc == 'top':
    #             fig.subplots_adjust(top=1-adj, wspace=0.1,left=0.123)
    #         else:
    #             fig.subplots_adjust(bottom=adj, wspace=0.1,left=0.123)
    #             bottom = 0.01
    #
    #         # fig.tight_layout()
    #
    #         #BOTKEY
    #         # fig.subplots_adjust(bottom=0.25, wspace=0.1)
    #         # bottom = 0.01
    #         # bot_adj = 0.25
    #
    #         # fig.subplots_adjust(bottom=bot_adj, wspace=0.1)
    #         left = axs[0].get_position().x0
    #         right = axs[-1].get_position().x1 - 0.123
    #
    #         # right = axs[1].get_position().x1 - 0.046
    #         # bottom = axs[-1].get_position().y1 * 1.05
    #         bottom = 0.01
    #         top = 0.99
    #         bbox = (left, bottom, right, top)
    #
    #         if settings[feature]['legend']:
    #
    #             plt.legend(*axs[-1].get_legend_handles_labels(),
    #                        loc='upper left' if key_loc == 'top' else 'lower left',
    #                        ncols=4, bbox_to_anchor=bbox,
    #                        bbox_transform=plt.gcf().transFigure, mode='expand', borderaxespad=0.0)
    #
    #     # plt.savefig(f"results/Fig_Both_{feature.replace(' ', '').replace('-', '')}_BKEY.svg", bbox_inches='tight')
    #     plt.savefig(f"results/Fig_Both_{feature.replace(' ', '').replace('-', '')}_{key_loc[0].upper()}KEY.svg", bbox_inches='tight')
    # plt.show()