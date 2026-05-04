import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger

from env import ReservoirEnv  # Your environment file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_sizes = [256, 256]

###########################################################
# GTrXL Components (No true memory, just sequence processing)
###########################################################

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
    def __init__(self, d_model, num_heads, head_dim, input_layernorm=True, output_activation=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.input_layernorm = input_layernorm
        if self.input_layernorm:
            self.layer_norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, num_heads * head_dim)
        self.key = nn.Linear(d_model, num_heads * head_dim)
        self.value = nn.Linear(d_model, num_heads * head_dim)
        self.output = nn.Linear(num_heads * head_dim, d_model)
        self.output_activation = output_activation

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.size()
        if self.input_layernorm:
            x = self.layer_norm(x)

        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = torch.einsum('bhid,bhjd->bhij', q, k) / (self.head_dim**0.5)
        att = F.softmax(att, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', att, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.output(out)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out

class GTrXLUnit(nn.Module):
    def __init__(self, d_model, num_heads, head_dim, position_wise_mlp_dim, init_gru_gate_bias=2.0):
        super().__init__()
        self.mha_block = SkipConnection(
            RelativeMultiHeadAttention(
                d_model, num_heads, head_dim, input_layernorm=True, output_activation=nn.ReLU()
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
    def __init__(self, d_model, num_heads, head_dim, position_wise_mlp_dim, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            GTrXLUnit(d_model, num_heads, head_dim, position_wise_mlp_dim) for _ in range(num_layers)
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
    def __init__(self, state_shape, action_shape, max_action, min_action, device):
        super().__init__()
        self.device = device
        self.max_action = max_action
        self.min_action = min_action

        self.obs_cnn = nn.Conv1d(
            in_channels=state_shape['well_observations'][1],
            out_channels=64,
            kernel_size=3,
            padding=1
        ).to(self.device)

        with torch.no_grad():
            sample = torch.randn(1, state_shape['well_observations'][1], state_shape['well_observations'][0]).to(self.device)
            feat = F.relu(self.obs_cnn(sample)).view(1,-1)
            d_model = feat.size(-1)

        self.gtrxl = GTrXLNet(d_model=d_model, num_heads=4, head_dim=16, position_wise_mlp_dim=256, num_layers=1).to(self.device)

        # Combine current frame features + last step GTrXL output
        combined_size = d_model*2
        self.fc1 = nn.Linear(combined_size, hidden_sizes[0]).to(self.device)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1]).to(self.device)
        self.fc_mu = nn.Linear(hidden_sizes[1], action_shape[0]).to(self.device)
        self.fc_std = nn.Linear(hidden_sizes[1], action_shape[0]).to(self.device)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_std.weight)

    def forward(self, obs, state=None, info={}):
        well_obs = torch.tensor(obs['well_observations'], dtype=torch.float32, device=self.device)
        history = torch.tensor(obs['history'], dtype=torch.float32, device=self.device)
        B, hist_len, c, w = history.shape

        # Current obs encoding
        current_feat = F.relu(self.obs_cnn(well_obs.permute(0,2,1))).view(B,-1)
        # History encoding
        hist = history.view(B*hist_len,c,w)
        hist_feat = F.relu(self.obs_cnn(hist)).view(B,hist_len,-1)

        # Pass through GTrXL
        gtrxl_out = self.gtrxl(hist_feat)
        last_step = gtrxl_out[:, -1, :]

        combined = torch.cat([current_feat, last_step], dim=-1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        sigma = torch.clamp(self.fc_std(x), min=-5, max=2).exp()

        return (mu, sigma), state

class GTrXLCritic(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.device = device
        self.obs_cnn = nn.Conv1d(
            in_channels=state_shape['well_observations'][1],
            out_channels=64,
            kernel_size=3,
            padding=1
        ).to(self.device)

        with torch.no_grad():
            sample = torch.randn(1, state_shape['well_observations'][1], state_shape['well_observations'][0]).to(self.device)
            feat = F.relu(self.obs_cnn(sample)).view(1,-1)
            d_model = feat.size(-1)

        self.gtrxl = GTrXLNet(d_model=d_model, num_heads=4, head_dim=16, position_wise_mlp_dim=256, num_layers=1).to(self.device)

        combined_size = d_model*2 + action_shape[0]
        self.fc1 = nn.Linear(combined_size, hidden_sizes[0]).to(self.device)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1]).to(self.device)
        self.fc_out = nn.Linear(hidden_sizes[1], 1).to(self.device)

    def forward(self, obs, action):
        well_obs = torch.tensor(obs['well_observations'], dtype=torch.float32, device=self.device)
        history = torch.tensor(obs['history'], dtype=torch.float32, device=self.device)
        B, hist_len, c, w = history.shape

        current_feat = F.relu(self.obs_cnn(well_obs.permute(0,2,1))).view(B,-1)
        hist = history.view(B*hist_len,c,w)
        hist_feat = F.relu(self.obs_cnn(hist)).view(B,hist_len,-1)

        gtrxl_out = self.gtrxl(hist_feat)
        last_step = gtrxl_out[:, -1, :]

        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        elif isinstance(action, torch.Tensor):
            action = action.to(self.device)

        combined = torch.cat([current_feat, last_step, action], dim=-1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q_value = self.fc_out(x)
        return q_value

###########################################################
# Main Training Loop
###########################################################

def make_env(env_id):
    return ReservoirEnv(env_id)

train_env_ids = range(1,65)
test_env_ids = range(65,75)

train_envs = SubprocVectorEnv([lambda env_id=env_id: make_env(env_id) for env_id in train_env_ids])
test_envs = SubprocVectorEnv([lambda env_id=env_id: make_env(env_id) for env_id in test_env_ids])

env = make_env(0)
state_shape = {key: env.observation_space[key].shape for key in env.observation_space.spaces.keys()}
action_shape = env.action_space.shape
max_action = env.action_space.high
min_action = env.action_space.low

actor = GTrXLActor(state_shape, action_shape, max_action, min_action, device=device)
critic1 = GTrXLCritic(state_shape, action_shape, device=device)
critic2 = GTrXLCritic(state_shape, action_shape, device=device)

actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic1_optim = torch.optim.Adam(critic1.parameters(), lr=3e-4)
critic2_optim = torch.optim.Adam(critic2.parameters(), lr=3e-4)

policy = SACPolicy(
    actor=actor,
    actor_optim=actor_optim,
    critic1=critic1,
    critic1_optim=critic1_optim,
    critic2=critic2,
    critic2_optim=critic2_optim,
    tau=0.005,
    gamma=0.99,
    alpha=0.5,
    estimation_step=1,
    deterministic_eval=False,
    action_space=env.action_space
)

buffer = VectorReplayBuffer(100000, len(train_envs))
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True)

logdir = "log"
log_path = os.path.join(logdir, "sac")
os.makedirs(log_path, exist_ok=True)

def save_best_fn(policy: BasePolicy) -> None:
    torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)

result = OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=15,
    step_per_epoch=3500,
    step_per_collect=10,
    episode_per_test=10,
    batch_size=256,
    save_best_fn=save_best_fn,
    logger=logger,
).run()

writer.close()
