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
import gym
mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
env = gym.make("Overcooked-v0",base_env = base_env, featurize_fn =base_env.featurize_state_mdp)