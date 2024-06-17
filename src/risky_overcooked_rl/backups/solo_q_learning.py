import numpy as np
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.agents.benchmarking import AgentEvaluator,LayoutGenerator
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
# from risky_overcooked_py.mdp.overcooked_mdp import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
import pickle
import datetime
import os
from tempfile import TemporaryFile
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from itertools import product,count
from risky_overcooked_rl.utils.custom_tabular_agents import SoloQAgent
import matplotlib.pyplot as plt
from develocorder import (
    LinePlot,
    Heatmap,
    FilteredLinePlot,
    DownsampledLinePlot,
    set_recorder,
    record,
    set_update_period,
    set_num_columns,
)





if __name__ == "__main__":
    np.random.seed(0)
    # LAYOUT = "cramped_room_one_onion"
    # LAYOUT = "cramped_room"; HORIZON = 250; ITERATIONS = 10_000
    # LAYOUT = "sanity_check_3_onion"; HORIZON = 250; ITERATIONS = 10_000
    # LAYOUT = "sanity_check2"; HORIZON = 500; ITERATIONS = 10_000
    LAYOUT = "sanity_check3_single";HORIZON = 400;ITERATIONS = 10_000
    # LAYOUT = "sanity_check3"; HORIZON = 500; ITERATIONS = 10_000
    # LAYOUT = "cramped_room_single";HORIZON = 400;ITERATIONS = 10_000

    # Logger ----------------
    # axis labels
    # set_recorder(labeled=LinePlot(xlabel="Step", ylabel="Score"))
    # additional filtered values (window filter)
    set_recorder(filtered=FilteredLinePlot(filter_size=10, xlabel="Iteration",ylabel=f"Score ({LAYOUT})")) #max_length=50,
    set_update_period(1)  # [seconds]

    # Generate MDP and environment----------------
    # mdp_gen_params = {"layout_name": 'cramped_room_one_onion'}
    # mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
    # env = OvercookedEnv(mdp_fn, horizon=HORIZON)
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)


    # Generate agents
    q_agent = SoloQAgent(mdp,agent_index=0)
    stay_agent = StayAgent()
    # agents = [CustomQAgent(mdp, is_learning_agent=True, save_agent_file=True), StayAgent()]


    total_updates = 0

    iter_rewards = []
    for iter in range(ITERATIONS):
        # print(f"Iteration {iter}")
        env.reset()

        cum_reward = 0
        for t in count():
            state = env.state
            action1, _ = q_agent.action(state)
            action2, _ = stay_agent.action(state)
            joint_action = (action1, action2)
            next_state, reward, done, info = env.step(joint_action) # what is joint-action info?
            reward += info["shaped_r_by_agent"][0]
            # if  info["shaped_r_by_agent"][0] != 0:
            #     print(f"Shaped reward {info['shaped_r_by_agent'][0]}")
            q_agent.update(state, action1, reward, next_state, explore_decay_prog=total_updates/(HORIZON*ITERATIONS))
            total_updates += 1
            cum_reward += reward
            if done:
                break

            # trajectory.append((s_t, a_t, r_t, done, info))

        # record(filtered=cum_reward)
        iter_rewards.append(cum_reward)

        # if len(iter_rewards) > 10 == 0:

        if len(iter_rewards) % 50 == 0:

            # Test policy #########################
            N_tests = 1
            test_reward = 0
            exploration_proba_OLD = q_agent.exploration_proba
            q_agent.exploration_proba = 0
            for test in range(N_tests):
                env.reset()
                for t in count():
                    state = env.state
                    action1, _ = q_agent.action(state,enable_explore=False,rationality='max')
                    action2, _ = stay_agent.action(state)
                    joint_action = (action1, action2)
                    next_state, reward, done, _ = env.step(joint_action)  # what is joint-action info?
                    total_updates += 1
                    # print(f"P(explore) {q_agent.exploration_proba}")
                    test_reward += reward
                    pot_states = mdp.get_pot_states(state)
                    # if len(pot_states["ready"])>0:
                    #     soup = state.get_object(pot_states["ready"][0])
                    #     soup.ingredients
                    #     print("Soup is ready!")
                    #     q_agent.featurize_masks(env.state)

                    if done:
                        break
            record(filtered=test_reward / N_tests)
            q_agent.exploration_proba = exploration_proba_OLD
            # print(f"Iteration {iter} complete | 10-Mean {np.mean(iter_rewards[-10:])} | Qminmax {q_agent.Q_table.min()} {q_agent.Q_table.max()} | P(explore) {q_agent.exploration_proba} | test reward= {test_reward / N_tests}")
            print(f"Iteration {iter} complete | 10-Mean {np.mean(iter_rewards[-10:])} | P(explore) {q_agent.exploration_proba} | {N_tests} test reward= {test_reward / N_tests}")
        # if iter % 100 == 0:
        #     print(f"Iteration {iter} complete | 10-Mean {np.mean(iter_rewards[-10:])} | Qminmax {q_agent.Q_table.min()} {q_agent.Q_table.max()} | P(explore) {q_agent.exploration_proba} | test reward= {test_reward / N_tests}")



    fig,ax = plt.subplots()
    ax.plot(iter_rewards)
    plt.show()

    # Generate agents
    # ptfp = "/home/rise/PycharmProjects/overcooked_ai/test/single_agent/2024-03-14-14-36-55"  # post training file path
    # q_agent = CustomQAgent(agent_eval.env.mlam, path_file=ptfp, load_learned_agent=False)


    # agent_pair = AgentPair(CustomRandomAgent(), CustomRandomAgent())
    # This is for training agent
    # single_q_agent_pair= AgentPair(CustomQAgent(agent_eval.env.mlam, is_learning_agent=True, save_agent_file=True), StayAgent())
    # trajectories_single_greedy_agent = agent_eval.evaluate_agent_pair(single_q_agent_pair, num_games=1)
    # print("Random pair rewards", trajectories_single_greedy_agent["ep_returns"])

    #This is for testing agent
    #TODO:check if you can access the file directly from above


    # q_agent = CustomQAgent(agent_eval.env.mlam, path_file=ptfp, load_learned_agent=True)
    # single_q_agent_pair= AgentPair(q_agent, StayAgent())
    #
    #
    # # single_q_agent_pair= AgentPair(CustomQAgent(agent_eval.env.mlam, path_file=ptfp, load_learned_agent=True), StayAgent())
    # trajectories_single_greedy_agent = agent_eval.evaluate_agent_pair(single_q_agent_pair, num_games=1)
    # print("Random pair rewards", trajectories_single_greedy_agent["ep_returns"])
    # print([q_agent.Q_table.min(),q_agent.Q_table.max()])
    # #TODO:check if you are able to get reward as well as visualization

    #SAVE the environment available in Agents.py
    #Test agent in the evaluate_agent_pair