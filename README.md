![MDP python tests](https://github.com/HumanCompatibleAI/overcooked_ai/workflows/.github/workflows/pythontests.yml/badge.svg) ![overcooked-ai codecov](https://codecov.io/gh/HumanCompatibleAI/overcooked_ai/branch/master/graph/badge.svg) [![PyPI version](https://badge.fury.io/py/overcooked-ai.svg)](https://badge.fury.io/py/overcooked-ai) [!["Open Issues"](https://img.shields.io/github/issues-raw/HumanCompatibleAI/overcooked_ai.svg)](https://github.com/HumanCompatibleAI/minerl/overcooked_ai) [![GitHub issues by-label](https://img.shields.io/github/issues-raw/HumanCompatibleAI/overcooked_ai/bug.svg?color=red)](https://github.com/HumanCompatibleAI/overcooked_ai/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+label%3Abug) [![Downloads](https://pepy.tech/badge/overcooked-ai)](https://pepy.tech/project/overcooked-ai)
[![arXiv](https://img.shields.io/badge/arXiv-1910.05789-bbbbbb.svg)](https://arxiv.org/abs/1910.05789)

# Risky Overcooked

<p align="center">
  <img src=".\images\risky_coordination_ring_averse.gif" width="25%"> 
  <img src="./images/risky_coordination_ring_seeking.gif" width="25%">
  <i><br> Risk-sensitive stratagies. Risk-averse agents (left) coordinate to avoid puddles while risk-seeking agents 
          (right) remain more independent by traversing puddles.</i>
</p>

---

## Introduction

This repository contains the implementation of the _Risky Overcooked_ environment, a modification of the [
_overcooked_ai_](https://github.com/HumanCompatibleAI/overcooked_ai) environment from _Carrol 2019_.
Here, we provide a risky coordination task where agents must make and deliver onion soup as fast as possible.
This is done by placing 3 onions in the pot, waiting for the soup to cook, bringing a dish to the pot to pick up the
soup, and delivering the soup to the serving station.
Risk is explicitly incorporated into the environment by adding puddles in the way of the aforementioned subtasks.
When an agent enters a puddle with an object (e.g. dish, soup, or onion) it has a $p_{slip}$% chance of slipping and
losing the object.
Thus, agents must decide between:

- **Traverse the puddle:** resulting in either a shorter path when no slipping occurs or losing the object and having to
  retrieve another.
- **Detour around the puddle:** resulting in a longer path but ensuring the object is not lost.
- **Handoff object over a counter:** resulting in possible coordination costs to avoid puddles and long detours.

Provided in `src/risky_overcooked_rl/algorithms/DDQN` is the implementation of the multi-agent risk-sensitive
reinforcement learning (MARSRL) algorithm used to train risk-sensitive polices.
This algorithm implements a Double Deep Q-Network (DDQN) modified with risk-sensitive objective based on Cumulative
Prospect Theory (CPT).
We apply a level-k quantal response equilibrium (QRE) as a tractable alternative to Nash equilibrium policies.
<p align="center">
  <img src="./images/CPT_animation_all.gif" width="50%">
  <i><br> Fig. Prospect curves from CPT animation showing the risk-sensitive value of the action space for each agent in the
          _Risky Overcooked_ environment. In MARSRL, the all possible TD-targets, given stochastic state transitions,
          and their probabilities are passed through CPT to create a biased expectation of CPT-value over the TD-targets.</i>
</p>



The original version of this repository was released with the paper *Risk-Sensitive Theory of Mind: Coordinating with
Agents of Unknown Bias using Cumulative Prospect Theory* (ICML 2025).
It has since undergone minor modifications to improve the environment and algorithm implementations.
However, the original models and reproducible results can be found in `src/study_1/`.
---

## Research Papers:

[1] M. O. Smith and W. Zhang, “Risky Sensitive Theory of Mind: Coordinating with agents of Uknown Bias using Cumulative Prospect Theory," in Proceedings of the International Conference on Machine Learning (ICML), 2025. accepted


---

## Installation:
We currently offer only buidling from source.
1. Clone the repository:

```
git clone https://github.com/ASU-RISE-Lab/risky_overcooked.git
```

2. Install requirements 
```
pip install -r requirements.txt
```
3. Verify the installation

```bash
python ./testing/package_test.py
python ./testing/mdp_test.py
```


_PyPI installation coming soon..._
___

## Repository Overview:

`risky_overcooked_py/` contains:

- `overcooked_mdp.py`: logic for handling state transitions and gaim assets
- `overcooked_env.py`: environment logic for simulating episodes
- `data/layouts/`: directory containing the layout files for the environment. Current implementation lists several
  layouts but only _risky_coordination_ring_ and _risky_multipath_ tasks are well validated at the moment.

`risky_overcooked_rl/` contains:

- `algorithms/DDQN/`: implementation of the MARSRL algorithm based on DDQN with CPT (see readme for more info)
    - `models.py`: pretrained models for _risky_coordination_ring_ and _risky_multipath_ tasks.
    - `utils.py`: utility functions for curriculum training, game theory, and agents/networks
    - _More algorithms to come in the future..._


- `utils/`: high-level utilities
    - `risk_sensitivity.py`: implementation of the CPT compiled with `numba.py` for speed
    - `state_utils.py`: implementation custom state and observation handlers
    - `visualization.py`: resources for generating static figures and chronographs
    - `rl_logger_V2.py`: dynamic logger to track online training and interface

---

## Questions and Comments

Please direct your questions and comments to Mason O. Smith at mosmith3@asu.edu
