# CSSE415 Group 11 - Reinforcement Learning without Neural Networks
## Usage
### Setup
1. Create a virtual environment with `python3 -m venv .env`
2. Activate the virtual environment with `source .env/bin/activate`
3. Install pip dependencies with `pip install -r requirements.txt`

### Files
#### CleanRL Training Scripts
These scripts are all based on [CleanRL's](https://github.com/vwxyzjn/cleanrl) single-file implementations of different reinforcement learning algorithms. 
* You can use the `--help` flag on all of these to see the available command-line arguments, or you can simply change any of the values in the `Args` dataclass at the top of each file.
* These scripts are also setup to work with both Tensorboard and Weights and Biases. If you have a W&B account, simply make sure the `track` argument is toggled on.
1. `cpole-ddpg.py`
   * DDPG Algorithm for Continuous Cart Pole
2. `cpole-ppo-continuous.py`
   * Continuous PPO for Continuous Cart Pole
3. `dqn-discrete.py`
   * DQN for Discrete Cart Pole (or any discrete action space environment)
4. `mtn-ppo-continuous.py`
   * Continuous PPO for Continuous Mountain Car
5. `ppo-discrete.py`
   * Discrete PPO for Discrete Cart Pole (or any discrete action space environment)
6. `ppo-continuous.py`
   * Continuous PPO for any continuous action space environment
7. `td3_continuous.py`
   * TD3 Algorithm for any continuous action space environment

#### Render Scripts
These scripts use pygame to render a GUI. If you are ssh'd into the Rose-Hulman servers, you will need to make sure you have X11 Forwarding enabled as well as a [Windows X Server](https://sourceforge.net/projects/xming/) running.
1. `playable-env.py`
   * This script allows you to play cartpole! Use left and right arrow keys or `a` and `d` to move left and right. Press `backspace` to exit.
2. `render-agent.py`
   * This script allows you to load a trained model and watch it play the environment. Press `backspace` to exit.
   * NOTE: This only works with models saved from `ppo-discrete.py` as it expects models to be saved with TorchScript, which is a real pain to get working.

#### Other
1. `grid-search.py`
   * A script that can do a hyperparameter gridsearch on any of the CleanRL Training Scripts.
2. `continuous_cartpole.py`
   * Our implementation of the Continuous Cart Pole Environment
3. `linear_models.py`
   * Our implementations of linear models in pytorch

