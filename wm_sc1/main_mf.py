import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import math
from typing import Dict, Any, Union
import gymnasium as gym # For type hints if action_space is gymnasium.Space
import pickle
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, DataParallelNet
from tianshou.data import Batch          
from torch.utils.data import DataLoader, TensorDataset 
from custom_trainer_4 import DistillationSACPolicy
from env_3_mb1 import ReservoirEnv  # Replace with your actual environment import
from copy import deepcopy # <--- IMPORT THE DEEPCOPY FUNCTION
from tianshou.policy import SACPolicy
from tianshou.data import Batch
class Scenario1FailureMask:
    """
    Mask certain action dimensions to a fixed normalized value.
    """

    def __init__(self, dead_action_indices, dead_action_value: float = -1.0):
        # Normalize to a list[int]
        if isinstance(dead_action_indices, (int, np.integer)):
            idx_list = [int(dead_action_indices)]
        elif isinstance(dead_action_indices, str):
            # allow comma-separated, e.g. "8,10"
            idx_list = [int(tok) for tok in dead_action_indices.split(",") if tok.strip()]
        else:
            # assume iterable of ints
            idx_list = [int(i) for i in dead_action_indices]

        self.dead_action_indices = idx_list
        self.dead_action_value = float(dead_action_value)

    def mask(self, action: torch.Tensor) -> torch.Tensor:
        """
        action: tensor [..., A] in [-1, 1].
        Returns a new tensor with the dead dims overwritten.
        """
        if not self.dead_action_indices:
            return action

        a = action.clone()
        idx = torch.as_tensor(self.dead_action_indices, device=a.device, dtype=torch.long)
        a[..., idx] = self.dead_action_value
        return a

class MaskedSACPolicy(SACPolicy):
    """Standard SACPolicy, but clamp some action dims via Scenario1FailureMask."""
    def __init__(self, *args, failure_mask: Scenario1FailureMask | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_mask = failure_mask
        # device for masking
        self._device = next(self.actor.parameters()).device

    def forward(self, batch: Batch, state=None, **kwargs) -> Batch:
        # Let vanilla SAC do its thing
        out = super().forward(batch, state=state, **kwargs)

        # In collect / eval modes, out.act exists and is what Collector uses
        if self.failure_mask is not None and hasattr(out, "act") and out.act is not None:
            a = torch.as_tensor(out.act, device=self._device, dtype=torch.float32)
            a = self.failure_mask.mask(a)
            out.act = a.detach().cpu().numpy()

        return out
from tianshou.data.types import ObsBatchProtocol, ActStateBatchProtocol, RolloutBatchProtocol # Ensure these are imported
from tianshou.policy.base import TrainingStats # Ensure this is imported
import gymnasium as gym # For type hint
from nets import CURL, augment_observations, Res3DCNN, GRUGate, SkipConnection, RelativeMultiHeadAttention, GTrXLUnit, GTrXLNet, HistoryEncoder, PolicyHead, DistillationHead, CriticEncoder, CriticHead, StudentDistillationNetwork, Transposed3DCNN, StudentQHead, StudentCritic




def make_env(env_id):
    return ReservoirEnv(env_id)
train_env_ids = range(120,122) # Exa2ple environment IDs for training
test_env_ids = range(231,232)   # Example environment IDs for testing
# Correct creation:
train_envs = SubprocVectorEnv([
    lambda env_id=env_id: make_env(env_id) for env_id in train_env_ids
])
test_envs = SubprocVectorEnv([
    lambda env_id=env_id: make_env(env_id) for env_id in test_env_ids
])

# Create vectorized environments for training

# Create vectorized environments for testing
# Access a single environment to get the observation and action shapes
env = make_env(0)
state_shape = {
    key: env.observation_space[key].shape for key in env.observation_space.spaces.keys()
}


action_shape = env.action_space.shape

max_action = 1.0
min_action = -1.0
hidden_sizes = [256, 256]

device = torch.device('cuda')

class TianshouSACActor(nn.Module):
    """
    Actor for Tianshou SACPolicy: obs(dict) -> (mu, sigma), state.
    """
    def __init__(self, shared_encoder: HistoryEncoder, policy_head: PolicyHead, device: torch.device):
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
        info: Dict = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], Any]:
        if info is None:
            info = {}

        # Extract tensors from dict, move to device
        hist = obs["history"]
        well = obs["well_observations"]

        if not isinstance(hist, torch.Tensor):
            hist = torch.as_tensor(hist, dtype=torch.float32)
        if not isinstance(well, torch.Tensor):
            well = torch.as_tensor(well, dtype=torch.float32)

        hist = hist.to(self.device)
        well = well.to(self.device)

        # Encode, then get (mu, sigma) from PolicyHead
        z = self.encoder(hist, well)        # [B, d_model]
        mu, sigma = self.policy_head(z)     # sigma is std (as in your MF training)

        return (mu, sigma), state





class TianshouSACCritic(nn.Module):
    """
    Critic for SAC: (obs(dict), action) -> Q-value.
    """
    def __init__(self, critic_encoder: CriticEncoder, critic_head: CriticHead, device: torch.device):
        super().__init__()
        self.encoder = critic_encoder
        self.head = critic_head
        self.device = device

    def forward(
        self,
        obs: Dict[str, Union[torch.Tensor, np.ndarray]],
        action: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:

        # Extract obs components
        res_state = obs["res_state"]
        hist      = obs["history"]
        well      = obs["well_observations"]

        if not isinstance(res_state, torch.Tensor):
            res_state = torch.as_tensor(res_state, dtype=torch.float32)
        if not isinstance(hist, torch.Tensor):
            hist = torch.as_tensor(hist, dtype=torch.float32)
        if not isinstance(well, torch.Tensor):
            well = torch.as_tensor(well, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)

        res_state = res_state.to(self.device)
        hist      = hist.to(self.device)
        well      = well.to(self.device)
        action    = action.to(self.device)

        # Encode + head → Q
        z_teacher, history_features = self.encoder(res_state, hist, well)
        q_value = self.head(z_teacher, history_features, action)
        return q_value


log_path_dir = "log/rrl" # Changed log path slightly
os.makedirs(log_path_dir, exist_ok=True)
writer = SummaryWriter(log_path_dir)
logger = TensorboardLogger(writer,update_interval = 100)

primary_device = torch.device('cuda:0')

d_model = 128
shared_history_encoder = HistoryEncoder(d_model=d_model).to(primary_device)
actor_policy_head = PolicyHead(d_model=d_model, action_shape=action_shape).to(primary_device)
actor = TianshouSACActor(shared_history_encoder, actor_policy_head, device=primary_device)


actor_optim = torch.optim.Adam(
    list(shared_history_encoder.parameters()) +
    list(actor_policy_head.parameters()),
    lr=3e-4,
)
# --- Critic Setup ---

# 1. Instantiate the components for Critic 1
critic1_encoder = CriticEncoder(latent_dim=d_model).to(primary_device)
critic1_head = CriticHead(action_shape=action_shape, latent_dim=d_model).to(primary_device)
# Assemble the critic Tianshou expects
critic1 = TianshouSACCritic(critic1_encoder, critic1_head, device=primary_device)


# 2. Instantiate the components for Critic 2
critic2_encoder = CriticEncoder(latent_dim=d_model).to(primary_device)
critic2_head = CriticHead(action_shape=action_shape, latent_dim=d_model).to(primary_device)
critic2 = TianshouSACCritic(critic2_encoder, critic2_head, device=primary_device)

# Assemble the critic Tianshou expects


# 3. Create the optimizers
critic1_optim = torch.optim.Adam([
    {'params': critic1_encoder.parameters()},
    {'params': critic1_head.parameters()}
], lr=1e-4)

critic2_optim = torch.optim.Adam([
    {'params': critic2_encoder.parameters()},
    {'params': critic2_head.parameters()}
], lr=1e-4)

target_entropy = -11
# 2) Create a learnable log_alpha parameter and optimizer.
log_alpha = torch.zeros(1, requires_grad=True, device=primary_device)
alpha_optim = torch.optim.Adam([log_alpha], lr=1e-4)

failure_mask = Scenario1FailureMask(10, dead_action_value=-1.0)

policy = MaskedSACPolicy(
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
    failure_mask=failure_mask,
)


policy._grad_norm = 1.0  # if you still want gradient clipping as before
# =============================================================================
# YOUR ORIGINAL BUFFER, COLLECTOR, LOGGER, TRAINER SETUP - Mostly unchanged
# =============================================================================
buffer_size = 800
buffer = VectorReplayBuffer(buffer_size, len(train_envs))
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True) # Original was True
ckpt_path = "policy.pth"  # adjust path

ckpt_path = "policy.pth"
sd_old = torch.load(ckpt_path, map_location=primary_device)
sd_new = policy.state_dict()

translated = {}
for k, v in sd_old.items():
    # Strip the DataParallel prefix if present
    k_stripped = k.replace("dp_module.module.", "")
    translated[k_stripped] = v

# Only load overlapping, shape-compatible keys
overlap = {
    k: v
    for k, v in translated.items()
    if k in sd_new and sd_new[k].shape == v.shape
}

print(f"Warm-start: loading {len(overlap)} / {len(sd_new)} params from {ckpt_path}")
sd_new.update(overlap)
policy.load_state_dict(sd_new)

def save_best_fn(policy: BasePolicy) -> None:
    torch.save(policy.state_dict(), os.path.join(log_path_dir, "policy_mf_upd.pth"))

# Trainer
print("Starting OffpolicyTrainer...")
result = OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=20,  # Adjust training parameters as needed
    step_per_epoch=40,
    step_per_collect=20,
    episode_per_test=1,
    batch_size=64,
    save_best_fn = save_best_fn,
    logger=logger,
).run()

writer.close()
print("Training finished!", result)