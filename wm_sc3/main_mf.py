import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Union

from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger

# CHANGE THIS IMPORT IF YOUR LOCAL FILENAME IS DIFFERENT
# For MB scenario 2 / static-model shift:
from env_3_mb3 import ReservoirEnv

from nets import HistoryEncoder, PolicyHead, CriticEncoder, CriticHead


def make_env(env_id: int):
    return ReservoirEnv(env_id)


# -----------------------------------------------------------------------------
# Env IDs
# -----------------------------------------------------------------------------
# These match the MB scenario-2 style:
# - training on digital twins / SIM_DECKs
# - evaluating on TRUE_DECK at env_id 151
train_env_ids = range(120,122) # Exa2ple environment IDs for training
test_env_ids = range(231,232)   # Example environment IDs for testing


train_envs = SubprocVectorEnv([
    lambda env_id=env_id: make_env(env_id) for env_id in train_env_ids
])
test_envs = SubprocVectorEnv([
    lambda env_id=env_id: make_env(env_id) for env_id in test_env_ids
])

env = make_env(0)
state_shape = {
    key: env.observation_space[key].shape
    for key in env.observation_space.spaces.keys()
}
action_shape = env.action_space.shape

primary_device = torch.device("cuda:0")
max_action = 1.0
min_action = -1.0
hidden_sizes = [256, 256]

# -----------------------------------------------------------------------------
# Actor / Critic wrappers for Tianshou SAC
# -----------------------------------------------------------------------------
class TianshouSACActor(nn.Module):
    """
    Actor for Tianshou SACPolicy: obs(dict) -> (mu, sigma), state.
    """
    def __init__(
        self,
        shared_encoder: HistoryEncoder,
        policy_head: PolicyHead,
        device: torch.device,
    ):
        super().__init__()
        self.encoder = shared_encoder
        self.policy_head = policy_head
        self.device = device
        self.max_action = 1
        self.min_action = -1
    def forward(
        self,
        obs: Dict[str, Union[torch.Tensor, np.ndarray]],
        state: Any = None,
        info: Dict | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], Any]:
        if info is None:
            info = {}

        hist = obs["history"]
        well = obs["well_observations"]

        if not isinstance(hist, torch.Tensor):
            hist = torch.as_tensor(hist, dtype=torch.float32)
        if not isinstance(well, torch.Tensor):
            well = torch.as_tensor(well, dtype=torch.float32)

        hist = hist.to(self.device)
        well = well.to(self.device)

        z = self.encoder(hist, well)
        mu, sigma = self.policy_head(z)
        return (mu, sigma), state


class TianshouSACCritic(nn.Module):
    """
    Critic for SAC: (obs(dict), action) -> Q-value.
    """
    def __init__(
        self,
        critic_encoder: CriticEncoder,
        critic_head: CriticHead,
        device: torch.device,
    ):
        super().__init__()
        self.encoder = critic_encoder
        self.head = critic_head
        self.device = device

    def forward(
        self,
        obs: Dict[str, Union[torch.Tensor, np.ndarray]],
        action: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        res_state = obs["res_state"]
        hist = obs["history"]
        well = obs["well_observations"]

        if not isinstance(res_state, torch.Tensor):
            res_state = torch.as_tensor(res_state, dtype=torch.float32)
        if not isinstance(hist, torch.Tensor):
            hist = torch.as_tensor(hist, dtype=torch.float32)
        if not isinstance(well, torch.Tensor):
            well = torch.as_tensor(well, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)

        res_state = res_state.to(self.device)
        hist = hist.to(self.device)
        well = well.to(self.device)
        action = action.to(self.device)

        z_teacher, history_features = self.encoder(res_state, hist, well)
        q_value = self.head(z_teacher, history_features, action)
        return q_value


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log_path_dir = "log/rrl"
os.makedirs(log_path_dir, exist_ok=True)

writer = SummaryWriter(log_path_dir)
logger = TensorboardLogger(writer, update_interval=100)


# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------
d_model = 128

shared_history_encoder = HistoryEncoder(d_model=d_model).to(primary_device)
actor_policy_head = PolicyHead(d_model=d_model, action_shape=action_shape).to(primary_device)
actor = TianshouSACActor(
    shared_encoder=shared_history_encoder,
    policy_head=actor_policy_head,
    device=primary_device,
)

actor_optim = torch.optim.Adam(
    list(shared_history_encoder.parameters()) + list(actor_policy_head.parameters()),
    lr=3e-4,
)

critic1_encoder = CriticEncoder(latent_dim=d_model).to(primary_device)
critic1_head = CriticHead(action_shape=action_shape, latent_dim=d_model).to(primary_device)
critic1 = TianshouSACCritic(critic1_encoder, critic1_head, device=primary_device)

critic2_encoder = CriticEncoder(latent_dim=d_model).to(primary_device)
critic2_head = CriticHead(action_shape=action_shape, latent_dim=d_model).to(primary_device)
critic2 = TianshouSACCritic(critic2_encoder, critic2_head, device=primary_device)

critic1_optim = torch.optim.Adam(
    [
        {"params": critic1_encoder.parameters()},
        {"params": critic1_head.parameters()},
    ],
    lr=1e-4,
)

critic2_optim = torch.optim.Adam(
    [
        {"params": critic2_encoder.parameters()},
        {"params": critic2_head.parameters()},
    ],
    lr=1e-4,
)


# -----------------------------------------------------------------------------
# Plain SAC policy (no mask needed for scenario 2)
# -----------------------------------------------------------------------------
target_entropy = -11
log_alpha = torch.zeros(1, requires_grad=True, device=primary_device)
alpha_optim = torch.optim.Adam([log_alpha], lr=1e-4)

policy = SACPolicy(
    actor=actor,
    actor_optim=actor_optim,
    critic=critic1,
    critic_optim=critic1_optim,
    critic2=critic2,
    critic2_optim=critic2_optim,
    tau=0.005,
    gamma=0.99,
    alpha=(target_entropy, log_alpha, alpha_optim),
    estimation_step=1,
    action_space=env.action_space,
    action_scaling=True,
    deterministic_eval=True,
)

policy._grad_norm = 1.0


# -----------------------------------------------------------------------------
# Replay / collectors
# -----------------------------------------------------------------------------
buffer_size = 800
buffer = VectorReplayBuffer(buffer_size, len(train_envs))

train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True)


# -----------------------------------------------------------------------------
# Warm-start from previous MF/SAC policy
# -----------------------------------------------------------------------------
ckpt_path = "policy.pth"   # change if needed

sd_old = torch.load(ckpt_path, map_location=primary_device)
sd_new = policy.state_dict()

translated = {}
for k, v in sd_old.items():
    # strip DataParallel prefix if present
    k_stripped = k.replace("dp_module.module.", "")
    translated[k_stripped] = v

overlap = {
    k: v
    for k, v in translated.items()
    if k in sd_new and sd_new[k].shape == v.shape
}

print(f"Warm-start: loading {len(overlap)} / {len(sd_new)} params from {ckpt_path}")
sd_new.update(overlap)
policy.load_state_dict(sd_new)

# -----------------------------------------------------------------------------
# Save best
# -----------------------------------------------------------------------------
def save_best_fn(policy: BasePolicy) -> None:
    torch.save(policy.state_dict(), os.path.join(log_path_dir, "policy_mf_sc2_upd.pth"))


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
print("Starting OffpolicyTrainer...")
result = OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=20,
    step_per_epoch=40,
    step_per_collect=20,
    episode_per_test=1,
    batch_size=64,
    save_best_fn=save_best_fn,
    logger=logger,
).run()

writer.close()
print("Training finished!", result)