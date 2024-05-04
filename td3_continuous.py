"""Implementation of td3 for a continuous environment
docs and experiment results can be found at:
https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
"""
#pylint: disable=not-callable
import os
import random
import time
from dataclasses import dataclass
from typing import Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from linear_models import MultiTargetLinearRegressor

@dataclass
class Args:
    """Dataclass for run arguments"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: Union[str,None] = None
    """the wandb's project name"""
    wandb_entity: Union[str,None] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "ContinuousCartPole" # continuous cartpole? O_o
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


def make_env(env_id, seed, idx, capture_video, run_name):
    """Create the gymnasium environment"""
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

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    """Backend q-learning network"""
    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__()
        self.network = MultiTargetLinearRegressor(
            np.prod(env.observation_space.shape) + np.prod(env.action_space.shape),
            np.prod(env.action_space.shape))

    def forward(self, x, a):
        """Propagate the model forward"""
        return self.network(torch.cat([x,a],1))

class Actor(nn.Module):
    """frontend critic training network"""
    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__()
        self.network = MultiTargetLinearRegressor(np.prod(env.observation_space.shape),
                                                  np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0,
                                         dtype=torch.float32))
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0,
                                        dtype=torch.float32))

    def forward(self, x):
        """Propagate the model forward"""
        return self.network(x) * self.action_scale + self.action_bias

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    gym.register(id="ContinuousCartPole",
                 entry_point="continuous_cartpole:ContinuousCartPoleEnv",
                 max_episode_steps=500,
                 reward_threshold=475.0,)

    args = tyro.cli(Args)
    run = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join(
            [f"|{key}|{value}|"for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run)])
    ASSERT_ERROR = "only continuous action space is supported"
    assert isinstance(envs.single_action_space, gym.spaces.Box), ASSERT_ERROR

    ACTOR = Actor(envs).to(device)
    QF1 = QNetwork(envs).to(device)
    QF2 = QNetwork(envs).to(device)
    QF1_TARGET = QNetwork(envs).to(device)
    QF2_TARGET = QNetwork(envs).to(device)
    TARGET_ACTOR = Actor(envs).to(device)
    TARGET_ACTOR.load_state_dict(ACTOR.state_dict())
    QF1_TARGET.load_state_dict(QF1.state_dict())
    QF2_TARGET.load_state_dict(QF2.state_dict())
    q_optimizer = optim.Adam(list(QF1.parameters()) + list(QF2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(ACTOR.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = ACTOR(torch.Tensor(obs).to(device))
                actions += torch.normal(0, ACTOR.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(
                    envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for i, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[i] = infos["final_observation"][i]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (
                    torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * TARGET_ACTOR.action_scale

                next_state_actions = (TARGET_ACTOR(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = QF1_TARGET(data.next_observations, next_state_actions)
                qf2_next_target = QF2_TARGET(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = (data.rewards.flatten() + (1 - data.dones.flatten())
                                * args.gamma * (min_qf_next_target).view(-1))

            qf1_a_values = QF1(data.observations, data.actions).view(-1)
            qf2_a_values = QF2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -QF1(data.observations, ACTOR(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(ACTOR.parameters(), TARGET_ACTOR.parameters()):
                    target_param.data.copy_(args.tau*param.data + (1 - args.tau)*target_param.data)
                for param, target_param in zip(QF1.parameters(), QF1_TARGET.parameters()):
                    target_param.data.copy_(args.tau*param.data + (1 - args.tau)*target_param.data)
                for param, target_param in zip(QF2.parameters(), QF2_TARGET.parameters()):
                    target_param.data.copy_(args.tau*param.data + (1 - args.tau)*target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS",
                                  int(global_step / (time.time() - start_time)),
                                  global_step)

    if args.save_model:
        model_path = f"runs/{args.exp_name}/{run}/model.joblib"
        torch.save(ACTOR.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
