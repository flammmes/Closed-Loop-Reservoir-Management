import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, DataParallelNet

from env_2_mf2 import ReservoirEnv  # Your environment file

def make_env(env_id):
    return ReservoirEnv(env_id)

# Create a list of environment IDs for training and testing
train_env_ids = range(1,76) # Example environment IDs for training
test_env_ids = range(76,77)   # Example environment IDs for testing

# Create vectorized environments for training
train_envs = SubprocVectorEnv([lambda env_id=env_id: make_env(env_id) for env_id in train_env_ids])

# Create vectorized environments for testing
test_envs = SubprocVectorEnv([lambda env_id=env_id: make_env(env_id) for env_id in test_env_ids])
# Access a single environment to get the observation and action shapes
env = make_env(0)
state_shape = {
    key: env.observation_space[key].shape for key in env.observation_space.spaces.keys()
}


action_shape = env.action_space.shape

max_action = 1.0
min_action = -1.0
hidden_sizes = [256, 256]


#Define the 1d conv layer here

##

device = torch.device('cuda')
###########################################################
# GTrXL Components (No true memory, just sequence processing)
###########################################################
class ActorWrapperDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.dp_module = nn.DataParallel(module)
        # Store the primary device determined during init
        self.primary_device = next(self.dp_module.parameters()).device

    def forward(self, obs: Dict[str, Union[torch.Tensor, np.ndarray]], state: Any = None, info: Dict = {}, **kwargs) -> Any:
        # Move dict values to primary device (cuda:0)
        obs_gpu = {}
        for k, v in obs.items():
            # Convert numpy first if necessary
            if not isinstance(v, torch.Tensor):
                 v = torch.as_tensor(v, dtype=torch.float32)
            obs_gpu[k] = v.to(self.primary_device)

        # Also move state if it's a tensor (common for RNN states)
        if isinstance(state, torch.Tensor):
            state = state.to(self.primary_device)
        # We generally don't move info dict contents

        # Pass GPU tensors as kwargs to nn.DataParallel
        return self.dp_module(
            history=obs_gpu["history"],
            well_observations=obs_gpu["well_observations"],
            state=state,
            info=info,
            **kwargs
        )

    def __getattr__(self, name):
         try: return super().__getattr__(name)
         except AttributeError:
             if hasattr(self, 'dp_module') and hasattr(self.dp_module, 'module'):
                 return getattr(self.dp_module.module, name)
             raise

# Wrapper for CRITIC (Previously named ScatterObsActDP)
class CriticWrapperDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.dp_module = nn.DataParallel(module)
        # Store the primary device determined during init
        self.primary_device = next(self.dp_module.parameters()).device

    # Matches call: critic(obs_dict_maybe_cpu, action_maybe_cpu_numpy)
    def forward(self, obs: Dict[str, Union[torch.Tensor, np.ndarray]], action: Union[torch.Tensor, np.ndarray], *args, **kwargs) -> Any:

        # Move obs dict values to primary device
        obs_gpu = {}
        for k, v in obs.items():
            if not isinstance(v, torch.Tensor):
                 v = torch.as_tensor(v, dtype=torch.float32)
            obs_gpu[k] = v.to(self.primary_device)

        # Move action to primary device
        if not isinstance(action, torch.Tensor):
             action = torch.as_tensor(action, dtype=torch.float32)
        action_gpu = action.to(self.primary_device)

        # Pass GPU tensors as kwargs to nn.DataParallel
        return self.dp_module(
            history=obs_gpu["history"],
            well_observations=obs_gpu["well_observations"],
            action=action_gpu,
            *args,
            **kwargs
        )

    def __getattr__(self, name):
         try: return super().__getattr__(name)
         except AttributeError:
             if hasattr(self, 'dp_module') and hasattr(self.dp_module, 'module'):
                 return getattr(self.dp_module.module, name)
             raise

class GRUGate(nn.Module):
    def __init__(self, d_model, init_gru_gate_bias=2.0):
        super().__init__()
        self.gru_cell = nn.GRUCell(d_model, d_model)
        nn.init.constant_(self.gru_cell.bias_ih, init_gru_gate_bias)

    def forward(self, x, y):
        # x,y: [B*T, d_model]
        return self.gru_cell(y, x)

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, hidden_dim, output_activation=None):
        super().__init__()
        self.w_1 = nn.Linear(d_model, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, d_model)
        self.output_activation = output_activation

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

class SkipConnection(nn.Module):
    def __init__(self, layer, fan_in_layer=None):
        super().__init__()
        self.layer = layer
        self.fan_in_layer = fan_in_layer

    def forward(self, x):
        y = self.layer(x)
        if self.fan_in_layer is not None:
            B, T, C = x.size()
            x_flat = x.reshape(B*T, C)
            y_flat = y.reshape(B*T, C)
            y = self.fan_in_layer(x_flat, y_flat).view(B, T, C)
            return y
        else:
            return x + y

class RelativeMultiHeadAttention(nn.Module):
    """Basic multihead self-attention (no actual relative bias)."""
    def __init__(self, d_model, num_heads, head_dim, input_layernorm=True, output_activation=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = d_model
        self.input_layernorm = input_layernorm

        if self.input_layernorm:
            self.layer_norm = nn.LayerNorm(d_model)
        # Project to queries, keys, values
        self.query = nn.Linear(d_model, num_heads * head_dim)
        self.key   = nn.Linear(d_model, num_heads * head_dim)
        self.value = nn.Linear(d_model, num_heads * head_dim)
        # Then project back
        self.output = nn.Linear(num_heads * head_dim, d_model)
        self.output_activation = output_activation

    def forward(self, x):
        # x: [B, T, d_model]
        if self.input_layernorm:
            x = self.layer_norm(x)

        B, T, _ = x.size()
        # shape => (B, T, num_heads, head_dim)
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # => (B, num_heads, T, head_dim)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        scores = torch.einsum('bhtd,bhsd->bhts', q, k) / (self.head_dim**0.5)  # shape (B, num_heads, T, T)
        att = F.softmax(scores, dim=-1)
        out = torch.einsum('bhts,bhsd->bhtd', att, v)  # shape (B, num_heads, T, head_dim)

        # reassemble
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.output(out)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out


class GTrXLUnit(nn.Module):
    """One block of MHA -> GRUGate -> FFN -> GRUGate."""
    def __init__(self, d_model, num_heads, head_dim, position_wise_mlp_dim, init_gru_gate_bias=2.0):
        super().__init__()
        self.mha_block = SkipConnection(
            RelativeMultiHeadAttention(
                d_model, num_heads, head_dim,
                input_layernorm=True,
                output_activation=nn.ReLU()
            ),
            fan_in_layer=GRUGate(d_model, init_gru_gate_bias)
        )
        self.ffn_block = SkipConnection(
            nn.Sequential(
                nn.LayerNorm(d_model),
                PositionwiseFeedforward(d_model, position_wise_mlp_dim, output_activation=nn.ReLU())
            ),
            fan_in_layer=GRUGate(d_model, init_gru_gate_bias)
        )

    def forward(self, x):
        x = self.mha_block(x)
        x = self.ffn_block(x)
        return x



class GTrXLNet(nn.Module):
    """GTrXL with N repeated layers + final LN."""
    def __init__(self, d_model, num_heads, head_dim, position_wise_mlp_dim, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            GTrXLUnit(d_model, num_heads, head_dim, position_wise_mlp_dim)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x

###########################################################
# Actor and Critic using GTrXL (No memory, just sequence)
###########################################################

class GTrXLActor(nn.Module):
    def __init__(
        self,
        state_shape: Dict[str, tuple],
        action_shape: tuple,
        max_action: float,
        min_action: float,
        hidden_sizes: list,
        device: torch.device,
        d_model=64,
        num_heads=4,
        head_dim=16,
        init_gru_gate_bias=2.0
    ):
        """
        state_shape["well_observations"] = (9, 30)
        state_shape["history"] = (history_length, 9, 30)

        We'll do:
          1) simple 1D conv on well_observations => shape [B, d_model]
          2) simple 1D conv on the entire history => shape [B, hist_len, d_model]
          3) pass the history through GTrXL
          4) combine current_feat + last_step => MLP => (mu, sigma)
        """
        super().__init__()
        self.device = device
        self.max_action = max_action
        self.min_action = min_action

        # Then out_channels = d_model if we want to directly get a 64-dim feature per time step
        self.obs_cnn = nn.Conv1d(in_channels=30, out_channels=d_model, kernel_size=3, padding=1)  ### CHANGED ###

        # GTrXL net
        self.gtrxl = GTrXLNet(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            position_wise_mlp_dim=256,
            num_layers=1
        )

        # MLP on top
        # We'll combine[current_feat, gtrxl_last_step], each is d_model => total 2*d_model
        combined_dim = 2 * d_model
        self.fc1 = nn.Linear(combined_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_mu = nn.Linear(hidden_sizes[1], action_shape[0])
        self.fc_std = nn.Linear(hidden_sizes[1], action_shape[0])

        # Optional init
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_std.weight)

    def forward(self, history=None, well_observations=None, state=None, info={}):
        B = well_observations.shape[0]
        
        # 1) Permute so shape is (B, 30, 9) for the conv
        well_obs_perm = well_observations.permute(0, 2, 1)  ### CHANGED ###
        # Now do the conv => shape[B, d_model, 9]
        current_feat_map = F.relu(self.obs_cnn(well_obs_perm))
        # Global average pool over the 9 "time" dimension
        current_feat = current_feat_map.mean(dim=2)  # => [B, d_model]

        # 2) History => shape [B, hist_len, 9, 30]
        hist_len = history.shape[1]
        # flatten =>[B*hist_len, 9, 30]
        hist_flat = history.view(B*hist_len, 9, 30)
        # permute =>[B*hist_len, 30, 9]
        hist_flat = hist_flat.permute(0, 2, 1)       ### CHANGED ###
        hist_feat_map = F.relu(self.obs_cnn(hist_flat))  # => [B*hist_len, d_model, 9]
        hist_feat_map = hist_feat_map.mean(dim=2)        # =>[B*hist_len, d_model]
        hist_feat_map = hist_feat_map.view(B, hist_len, -1)

        # 3) GTrXL => shape[B, hist_len, d_model]
        gtrxl_out = self.gtrxl(hist_feat_map)
        last_step = gtrxl_out[:, -1, :]

        # 4) Combine => (mu, sigma)
        combined = torch.cat([current_feat, last_step], dim=-1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_std = torch.clamp(self.fc_std(x), min=-5, max=2)
        std = log_std.exp()

        return (mu, std), state

# --------------------------------------
# GTrXL Critic
# --------------------------------------
class GTrXLCritic(nn.Module):
    def __init__(
        self,
        state_shape: Dict[str, tuple],
        action_shape: tuple,
        hidden_sizes: list,
        device: torch.device,
        d_model=64,
        num_heads=4,
        head_dim=16
    ):
        super().__init__()
        self.device = device

        # Same approach as actor: conv => GTrXL => ...
        self.obs_cnn = nn.Conv1d(in_channels=30, out_channels=d_model, kernel_size=3, padding=1)  ### CHANGED ###
        self.gtrxl = GTrXLNet(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            position_wise_mlp_dim=256,
            num_layers=1
        )

        # We combine [current_feat, last_step, action] => MLP => Q-value
        # => dimension = d_model + d_model + action_dim = 2*d_model + 11
        combined_dim = 2*d_model + action_shape[0]
        self.fc1 = nn.Linear(combined_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_out = nn.Linear(hidden_sizes[1], 1)

    def forward(self, history=None, well_observations=None, action=None):
        B = well_observations.shape[0]

        # 1) Permute => (B, 30, 9)
        well_obs_perm = well_observations.permute(0, 2, 1)    ### CHANGED ###
        current_feat_map = F.relu(self.obs_cnn(well_obs_perm))  # => (B, d_model, 9)
        current_feat = current_feat_map.mean(dim=2)             # => (B, d_model)

        # 2) History => shape [B, hist_len, 9, 30]
        hist_len = history.shape[1]
        hist_flat = history.view(B*hist_len, 9, 30)
        hist_flat = hist_flat.permute(0, 2, 1)                 ### CHANGED ###
        hist_feat_map = F.relu(self.obs_cnn(hist_flat))        # => (B*hist_len, d_model, 9)
        hist_feat_map = hist_feat_map.mean(dim=2)              # => (B*hist_len, d_model)
        hist_feat_map = hist_feat_map.view(B, hist_len, -1)

        gtrxl_out = self.gtrxl(hist_feat_map)
        last_step = gtrxl_out[:, -1, :]

        # Combine with action
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device)

        combined = torch.cat([current_feat, last_step, action], dim=-1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q_value = self.fc_out(x)
        return q_value

###########################################################
# Main Training Loop
###########################################################
primary_device = torch.device('cuda:0')


actor = GTrXLActor(
    state_shape=state_shape,
    action_shape=action_shape,
    max_action=max_action,
    min_action=min_action,
    hidden_sizes=hidden_sizes,
    device=device,
    d_model=64,     # or bigger if you prefer
    num_heads=4,
    head_dim=16
)

critic1 = GTrXLCritic(
    state_shape=state_shape,
    action_shape=action_shape,
    hidden_sizes=hidden_sizes,
    device=device,
    d_model=64,
    num_heads=4,
    head_dim=16
)

critic2 = GTrXLCritic(
    state_shape=state_shape,
    action_shape=action_shape,
    hidden_sizes=hidden_sizes,
    device=device,
    d_model=64,
    num_heads=4,
    head_dim=16
)
parallel_actor = ActorWrapperDP(actor).to(primary_device)
parallel_critic = CriticWrapperDP(critic1).to(primary_device)
parallel_critic2 = CriticWrapperDP(critic2).to(primary_device)

actor_optim = torch.optim.Adam(parallel_actor.parameters(), lr=3e-4)
critic1_optim = torch.optim.Adam(parallel_critic.parameters(), lr=1e-4)
critic2_optim = torch.optim.Adam(parallel_critic2.parameters(), lr=1e-4)
target_entropy = -11
# 2) Create a learnable log_alpha parameter and optimizer.
log_alpha = torch.zeros(1, requires_grad=True, device=primary_device)
alpha_optim = torch.optim.Adam([log_alpha], lr=1e-4)

policy = SACPolicy(
    actor=parallel_actor,
    actor_optim=actor_optim,
    critic=parallel_critic,
    critic_optim=critic1_optim,
    critic2=parallel_critic2,
    critic2_optim=critic2_optim,
    action_scaling = True,
    tau=0.005,  # Target network update rate
    gamma=0.98,  # Discount factor
    alpha=(target_entropy, log_alpha, alpha_optim),  
    estimation_step=1,  # Number of TD steps for Q-value estimation
    deterministic_eval=True,
    action_space=env.action_space,
    
     
)

buffer = VectorReplayBuffer(90000, len(train_envs))
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs,exploration_noise=True)


logdir = "log"
log_path = os.path.join(logdir, "sac")

# Define the save_best_fn
def save_best_fn(policy: BasePolicy) -> None:
    torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))


writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)

# 4. Train the agent
result = OffpolicyTrainer(
    policy=policy, 
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=30,  # Adjust training parameters as needed
    step_per_epoch=3000,
    step_per_collect=1500,
    #update_per_step = 0.1,
    episode_per_test=1,
    batch_size=2048,
    save_best_fn=save_best_fn,
    logger=logger,
).run()

writer.close()