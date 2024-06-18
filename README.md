# Risky Overcooked
This repository contains the implementation of the _Risky Overcooked_ environment, a modification of the _overcooked_ai_ environment from _Carrol 2019_.

![img.png](img.png)
## Description
The standared _overcooked_ai_ environment implements a deterministic MDP.
The _Risky Overcooked_ environment adds a stochastic transition state that imposes a risk evaluation into agent decision-making.
The primary motivation for this modification is to investigate risk-sensitive reinforcement learning in a multi-agent setting.

---
## Installation
```bash
# tbd... ask mason for help if needed
```

## Usage
### Creating Custom Layouts
### Environment
``` python
# Define the environment
LAYOUT = 'cramped_room'
HORIZON = 400

# Initialize the environment
mdp = OvercookedGridworld.from_layout_name(LAYOUT)
env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)

# Define the agents
q_agent = SoloDeepQAgent(mdp,agent_index=0,policy_net=policy_net, config=config)
stay_agent = StayAgent()
agent_pair = AgentPair(q_agent, stay_agent)

#

```
### Visualization
```python

```
---
## Risk-Sensitive Reinforcement Learning (RSRL)
Several risk-sensitive reinforcement learning algorithms were implemented in the Risky Overcooked environment.

### Double Deep Q-Network (DDQN) 
The DDQN algorithm was implemented in the Risky Overcooked environment.


### Soft Actor-Critic (SAC)

---
## Modification Log
Several modifications were made to the MDP (`overcooked_mdp.py >> OvercookedGridworld()` object) 

### Stochastic Transitions (Risky Puddles):
To implement the stochastic transition state (risky puddles) the following modifications were made to the MDP:
- added compliance to puddle tile `W` in `layout` 
- added `resolved_enter_water(...)` and implemented in `get_state_transition()` to handle stochastic transition (slip/not slip) when entering risky puddle state. Currently, has a `p_slip=0.5` chance of slipping in puddle and losing held object. Exiting puddle into a non-puddle state is determinstic.   
- added `get_water_locations()` to return the locations of the puddle tiles
- **(NOT_IMPLEMENTED)** added `get_possible_transitions(joint_action)` to return the possible transitions to next state and probability of each state given a joint action. Used for computing risk-sensitve expectations. 

### Custom Lossless State Encoding
Also, a custom state encoding was used. This is a vector encoding that creates a lossless representation of the state space. 
This encoding is used by the agents to interact with the environment and can be found in `overcooked_mdp.py >> OvercookedGridworld()` object.
- `get_lossless_encoding_vector_shape()` returns the shape of the lossless encoding vector
- `get_lossless_encoding_vector()` returns the lossless encoding vector of the current state
- vector encoding is dynamically computer and will be different size depending on the number of agents layout of the environment
```
player_features (9 x n_players,):
    pi_position (2,): position of player i {x,y}
    pi_orientation (4,): 1-hot encoding of orientation of player i 
    pi_objs (3,): 1-hot encoding of object held by player i {onion,dish,soup} 
world_features:
    
```

### Other modifications to the MDP
- The ```OvercookedGridworld.old_dynamics=True``` method was enabled

### Agents
- Agents now use a losslsess vector encoding of the state space
  - base repository either uses a vector encoding (with losses) or a lossless mask encoding

---
## To Do
- Super inneficiently calling lossless encoding multiple times
  - SOL: calc once at end of each mdp.transition and store as self.lossless_state to retreive precalculated vector whenever called
- When slipping in puddle, agent will lose `slipped_moves_lossed=2` moves (i.e. not be able to move for 2 tics) to incur risk when not holding object