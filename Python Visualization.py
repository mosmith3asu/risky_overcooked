from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair, FixedPlanAgent, GreedyHumanModel, RandomAgent, SampleAgent
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from IPython.display import display, Image
ae = AgentEvaluator.from_layout_name({"layout_name": "risky_cramped_room"}, {"horizon": 100})
trajs = ae.evaluate_human_model_pair()
# print(ae.env)
# print(ae.mdp.state_string)

action_probs = [ [RandomAgent(all_actions=True).action(state)[1]["action_probs"]]*2 for state in trajs["ep_states"][0]]
print(StateVisualizer().display_rendered_trajectory(trajs, ipython_display=False))