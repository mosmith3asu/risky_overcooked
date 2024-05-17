import numpy as np
from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.agents.benchmarking import AgentEvaluator,LayoutGenerator
from risky_overcooked_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
# from risky_overcooked_py.mdp.overcooked_mdp import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv,OvercookedEnvPettingZoo
import pickle
import datetime
import os
from tempfile import TemporaryFile
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from itertools import product,count
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
# import gym

mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
base_env = OvercookedEnv.from_mdp(mdp, horizon=500)

agent1 = GreedyHumanModel(mlam=base_env.mlam)
agent2 = GreedyHumanModel(mlam=base_env.mlam)
agent_pair= AgentPair(agent1, agent2)
# agent_pair = load_agent_pair("path/to/checkpoint", "ppo", "ppo")
env = OvercookedEnvPettingZoo(base_env, agent_pair)