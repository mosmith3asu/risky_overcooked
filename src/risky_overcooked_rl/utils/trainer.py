# import numpy as np
# import matplotlib.pyplot as plt


class Trainer:
    def __init__(self,agent1,agent2,config):
        pass

    def train(self):
        pass

if __name__ == "__main__":
    # LAYOUT = "cramped_room_one_onion"
    LAYOUT = "sanity_check";
    HORIZON = 500;
    ITERATIONS = 10_000

    # Logger ----------------
    # axis labels
    # set_recorder(labeled=LinePlot(xlabel="Step", ylabel="Score"))
    # additional filtered values (window filter)
    set_recorder(
        filtered=FilteredLinePlot(filter_size=10, xlabel="Iteration", ylabel=f"Score ({LAYOUT})"))  # max_length=50,
    set_update_period(1)  # [seconds]

    # Generate MDP and environment----------------
    # mdp_gen_params = {"layout_name": 'cramped_room_one_onion'}
    # mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
    # env = OvercookedEnv(mdp_fn, horizon=HORIZON)
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)

    # Generate agents
    q_agent = CustomQAgent(mdp, agent_index=0, save_agent_file=True)
    stay_agent = StayAgent()
    # agents = [CustomQAgent(mdp, is_learning_agent=True, save_agent_file=True), StayAgent()]

    total_updates = 0

    iter_rewards = []
    for iter in range(ITERATIONS):
        env.reset()

        cum_reward = 0
        for t in count():
            state = env.state
            action1, _ = q_agent.action(state)
            action2, _ = stay_agent.action(state)
            joint_action = (action1, action2)
            next_state, reward, done, _ = env.step(joint_action)  # what is joint-action info?
            q_agent.update(state, action1, reward, next_state,
                           explore_decay_prog=total_updates / (HORIZON * ITERATIONS))
            total_updates += 1
            # print(f"P(explore) {q_agent.exploration_proba}")
            cum_reward += reward
            if done:
                break

            # trajectory.append((s_t, a_t, r_t, done, info))

        # record(filtered=cum_reward)
        iter_rewards.append(cum_reward)

        # if len(iter_rewards) > 10 == 0:

        if len(iter_rewards) % 10 == 0:
            # Test policy #########################
            N_tests = 3
            test_reward = 0
            exploration_proba_OLD = q_agent.exploration_proba
            q_agent.exploration_proba = 0
            for test in range(N_tests):
                env.reset()
                for t in count():
                    state = env.state
                    featurized_state = env.featurize_state_mdp(state, num_pots=1)
                    print(f"State {np.shape(featurized_state)}")
                    action1, _ = q_agent.action(state, rationality=9)
                    action2, _ = stay_agent.action(state)
                    joint_action = (action1, action2)
                    next_state, reward, done, _ = env.step(joint_action)  # what is joint-action info?
                    total_updates += 1
                    # print(f"P(explore) {q_agent.exploration_proba}")
                    test_reward += reward

                    if done:
                        break

            record(filtered=test_reward / N_tests)
            q_agent.exploration_proba = exploration_proba_OLD
            print(
                f"Iteration {iter} complete | 10-Mean {np.mean(iter_rewards[-10:])} | Qminmax {q_agent.Q_table.min()} {q_agent.Q_table.max()} | P(explore) {q_agent.exploration_proba} | test reward= {test_reward / N_tests}")


