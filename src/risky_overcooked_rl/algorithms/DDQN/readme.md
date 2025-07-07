# Deep Double Q-Learning (DDQN) Algorithm
This is a  implementation of the Deep Double Q-Learning (DDQN) algorithm for reinforcement learning. 
The DDQN algorithm is an improvement over the standard DQN algorithm, which helps to reduce the overestimation 
bias in Q-learning by using two separate networks to estimate the Q-values.

## Usage

- Modify the configuration file `_config.yaml` to set the desired parameters for the training process such as layout, probability of slipping, and hyperparamters.
- To run a simple implementation of the DDQN algorithm, use the following command:
```bash
./run_ddqn_selfplay.py
```

### Batching Training
- Alternatively, you may batch training of many agents/layouts of the DDQN algorithm simultaneously:
  - This takes the default configuration file `_config.yaml` and runs the DDQN algorithm over several layouts and agent risk-sensitivities specified within `_batching.yaml` file.
  - Uses multiprocessing to run multiple instances of the DDQN algorithm in parallel, allowing for faster training and evaluation.
  - To do this, use the following command:
```bash
./run_ddqn_selfplay_batch.py
```
### Hyperparameter Optimization
- To optimize model hyperparamters, modify the search criteria within `main()` andrun the following command:
```bash
./eval/param_search.py
```
## Key Features

 ### Agents
- Self-play: each agent shares the same Q-network. During runtime, observations are inverted and both prospect transitions are pushed to the memory buffer (2 per timeste)
- level-k quantal response: `./utils/game_theory.py` contains a class that implements a level-k quantal response equalibrium, which can be used to evaluate the performance of the DDQN algorithm against a more sophisticated opponent.

### Curriculum Learning:
  - DDQN can have tractability issues is sparesley rewarded environments like Risky Overcooked
  - To overcome this, we apply a naive curriculum learning approach where agents learn subtasks:
     `deliver_soup >> pick_up_soup >> pick_up_dish >>  deliver_onion3 >> pick_up_onion3 >> 
    deliver_onion2 >> pick_up_onion2 >> deliver_onion1 >> full_task` (i.e. learn the task backwards)
  - Implementation can be found in ``src/risky_overcooked_rl/algorithms/DDQN/curriculum.py`` containing
    - `Curriculum` class: manages curriculum advancement, initialization, and subtask selection
    - `CurriculumTrainer` class: inherits `Trainer` class for minor modifications to handle `Curriculum` class in training loop
  - **Advancement Threshhold:**  The thresholds specifying how minimum proficiency in a task before advancing to 
        learning the next subtask (in number of soups completed) is found in `_config.yaml>>subtask_goals`
  - **Failed Learning Checks** Similarly, early stopping conditions (the maximum iterations allowed for each 
           subtask before considering the current model a failure) are found in `_config.yaml>>failure_checks`. 
            This is very helpful for parameter searches.

