#!/usr/bin/env python
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from joblib import dump

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{args.exp_name}/{run_name}` folder"""
    model_type: str = "linear_regression"
    """the type of the model"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 300_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# TODO: I got this from ChatGPT, it looks right but I haven't verified it with any outside source
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
"""
SGDClassifier
    Logistic Regression
        loss='log'
        a probabilistic classifier
    Linear Support Vector Machine (SVM)
        loss='hinge'
    SVM with Squared Hinge Loss
        loss='squared_hinge'
        like hinge but is quadratically penalized
    Perceptron
        loss='perceptron'
        is the linear loss used by the perceptron algorithm
    Modified Huber Loss
        loss='modified_huber'
        is another smooth loss that brings tolerance to outliers as well as probability estimates.
SGDRegressor
    Linear Regression
        loss='squared_error' (no regularization)
    Ridge Regression
        loss='squared_error' and penalty='l2'
    Lasso Regression
        loss='squared_error' and penalty='l1'
    Elastic Net Regression
        loss='squared_error' and penalty='elasticnet'
    Huber Regression
        loss='huber' (more robust to outliers)
"""

model_to_args = {
    "logistic_regression": {'loss': 'log'},
    "svm": {'loss': 'hinge'},
    "svm_squared_hinge": {'loss': 'squared_hinge'},
    "perceptron": {'loss': 'perceptron'},
    "modified_huber": {'loss': 'modified_huber'},
    "linear_regression": {'loss': 'squared_error', 'penalty': None},
    "ridge_regression": {'loss': 'squared_error', 'penalty': 'l2'},
    "lasso_regression": {'loss': 'squared_error', 'penalty': 'l1'},
    "elastic_net_regression": {'loss': 'squared_error', 'penalty': 'elasticnet'},
    "huber_regression": {'loss': 'huber'},
}


# ALGO LOGIC: initialize agent here:
class QNetwork():
    def __init__(self, envs, model_type='linear_regression'):
        super().__init__()
        self.num_actions = envs.single_action_space.n
        self.is_fit = False
        self.scaler = StandardScaler()

        model = SGDRegressor(**model_to_args[model_type], alpha=0.0001, learning_rate='constant', eta0=args.learning_rate)

        if self.num_actions == 1:
            self.network = model
        else:
            self.network = MultiOutputRegressor(
                model,
                # n_jobs=-1
            )

    def partial_fit(self, X, y):
        if not self.is_fit:
            self.scaler.fit(X)
            self.is_fit = True
        X = self.scaler.transform(X)
        self.network.partial_fit(X, y)

    def predict(self, x):
        if not self.is_fit:
            return torch.randn(x.shape[0], self.num_actions) * 0.01
        pred = self.network.predict(self.scaler.transform(x))
        return torch.from_numpy(pred)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    curr_usr = os.getlogin()
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{curr_usr}/{args.exp_name}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, f"{curr_usr}/{args.exp_name}/{run_name}") for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs, args.model_type)
    target_network = deepcopy(q_network)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon or global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network.predict(obs)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_q_values = target_network.predict(data.next_observations)
                    target_max, _ = next_q_values.max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    td_target = td_target.type(torch.float64)
                current_action_values = q_network.predict(data.observations)
                current_action_values = current_action_values.type(torch.float64)
                old_val = current_action_values.gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                current_action_values[np.arange(len(data.actions)), data.actions] = td_target.type(torch.float64)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                observations = data.observations.reshape(-1, data.observations.shape[-1])
                q_network.partial_fit(data.observations, current_action_values)

            # update target network
            if global_step % args.target_network_frequency == 0:
                target_network = deepcopy(q_network)

    # Evaluate
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, True, f"{curr_usr}/{args.exp_name}/{run_name}-eval") for i in range(args.num_envs)]
    )
    eval_episodes = 10
    epsilon = 0.05
    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network.predict(obs)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    for idx, episodic_return in enumerate(episodic_returns):
        writer.add_scalar("eval/episodic_return", episodic_return, idx)

    if args.save_model:
        model_path = f"runs/{curr_usr}/{args.exp_name}/{run_name}/{args.model_type}.joblib"
        dump(q_network, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()