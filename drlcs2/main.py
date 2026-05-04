import os
import argparse
from multiprocessing import Manager

# Third-party imports
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import matplotlib.pyplot as plt

from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import BasicLogger, TensorboardLogger, BaseLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from env import ReservoirEnv  # Ensure this module is correctly implemented


device = torch.device('cuda')




# Define a function to create a new environment with a unique env_id
def make_env(env_id):
    return ReservoirEnv(env_id)

# Create a list of environment IDs for training and testing
train_env_ids = range(1,65) # Example environment IDs for training
test_env_ids = range(65,75)   # Example environment IDs for testing

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

max_action = env.action_space.high
min_action = env.action_space.low
hidden_sizes = [256, 256]

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_length, hidden_dim]
        Returns:
            Tensor of shape [batch_size, seq_length, hidden_dim] with positional encodings added
        """
        batch_size, seq_length, hidden_dim = x.size()
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)  # [batch_size, seq_length]
        pos_emb = self.pos_embedding(positions)  # [batch_size, seq_length, hidden_dim]
        return x + pos_emb
class GRUGate(nn.Module):
    def __init__(self, hidden_dim, init_gru_gate_bias=2.0):
        super(GRUGate, self).__init__()
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        nn.init.constant_(self.gru_cell.bias_ih, init_gru_gate_bias)

    def forward(self, x, y):
        # x and y are of shape [batch_size, hidden_dim]
        h_new = self.gru_cell(y, x)
        return h_new

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, feedforward_dim, max_seq_length=20, init_gru_gate_bias=2.0):
        super(TransformerBlock, self).__init__()
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=max_seq_length)
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.gru_gate1 = GRUGate(hidden_dim, init_gru_gate_bias)

        self.positionwise_ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, hidden_dim)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.gru_gate2 = GRUGate(hidden_dim, init_gru_gate_bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_length, hidden_dim]
        Returns:
            Tensor of shape [batch_size, seq_length, hidden_dim]
        """
        # Apply positional encoding
        x = self.positional_encoding(x)  # Shape: [batch_size, seq_length, hidden_dim]

        # Detach past history except for the current observation
        past_history = x[:, :-1, :].detach()  # Shape: [batch_size, seq_length-1, hidden_dim]
        current_obs = x[:, -1:, :]            # Shape: [batch_size, 1, hidden_dim]

        # Combine detached past history with current observation
        combined_input = torch.cat([past_history, current_obs], dim=1)  # Shape: [batch_size, seq_length, hidden_dim]

        # Apply Multi-Head Attention
        attn_output, _ = self.attention(combined_input, combined_input, combined_input)  # Shape: [batch_size, seq_length, hidden_dim]

        # Residual connection and Layer Normalization
        x = self.layer_norm1(combined_input + attn_output)  # Shape: [batch_size, seq_length, hidden_dim]

        # Apply ReLU before GRU Gate as per Equation (15b)
        x_relu = torch.relu(x)  # Shape: [batch_size, seq_length, hidden_dim]

        # First GRU Gate
        # Reshape for GRUCell: [batch_size * seq_length, hidden_dim]
        x_reshaped = x.view(-1, x.shape[-1])                # Shape: [batch_size * seq_length, hidden_dim]
        x_relu_reshaped = x_relu.view(-1, x_relu.shape[-1])  # Shape: [batch_size * seq_length, hidden_dim]
        x = self.gru_gate1(x_reshaped, x_relu_reshaped).view(x.shape)  # Shape: [batch_size, seq_length, hidden_dim]

        # Position-wise Feedforward Network
        ffn_output = self.positionwise_ffn(x)  # Shape: [batch_size, seq_length, hidden_dim]

        # Residual connection and Layer Normalization
        x = self.layer_norm2(x + ffn_output)    # Shape: [batch_size, seq_length, hidden_dim]

        # Apply ReLU before second GRU Gate as per Equation (15d)
        x_relu_ffn = torch.relu(x)  # Shape: [batch_size, seq_length, hidden_dim]

        # Second GRU Gate
        x_reshaped_ffn = x_relu_ffn.view(-1, x_relu_ffn.shape[-1])            # Shape: [batch_size * seq_length, hidden_dim]
        ffn_output_reshaped = ffn_output.view(-1, ffn_output.shape[-1])        # Shape: [batch_size * seq_length, hidden_dim]
        x = self.gru_gate2(x_reshaped_ffn, ffn_output_reshaped).view(x.shape)  # Shape: [batch_size, seq_length, hidden_dim]

        return x

class Actor(nn.Module):
    def __init__(self, state_shape, action_shape, max_action, min_action, device=device):
        super().__init__()
        self.device = device

        # CNN for processing current well_observations
        self.well_conv = nn.Conv1d(
            in_channels=state_shape['well_observations'][1],
            out_channels=64,
            kernel_size=3,
            padding=1
        ).to(self.device) 

        # Calculate well_features size
        with torch.no_grad():
            sample_well_obs = torch.randn(1, state_shape['well_observations'][1], state_shape['well_observations'][0]).to(self.device)
            well_features = torch.relu(self.well_conv(sample_well_obs))
            well_features = well_features.view(well_features.size(0), -1)
            self.well_cnn_output_size = well_features.size(1)

        # TransformerBlock
        self.transformer_block = TransformerBlock(
            hidden_dim=self.well_cnn_output_size,
            num_heads=4,
            feedforward_dim=256
        ).to(self.device)

        # Fully connected layers
        self.history_output_size = self.well_cnn_output_size
        combined_size = self.well_cnn_output_size +self.history_output_size  # Since we process history separately
        self.fc1 = nn.Linear(combined_size, hidden_sizes[0]).to(self.device)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1]).to(self.device)
        self.fc_mu = nn.Linear(hidden_sizes[1], action_shape[0]).to(self.device)
        self.fc_std = nn.Linear(hidden_sizes[1], action_shape[0]).to(self.device)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_std.weight)
        self.max_action = max_action
        self.min_action = min_action

    def forward(self, obs, state=None, info={}):
        # Process current well_observations
        well_obs = torch.from_numpy(obs['well_observations']).to(self.device).permute(0, 2, 1).float()
        well_features = torch.relu(self.well_conv(well_obs))
        well_features = well_features.view(well_features.size(0), -1)  # Shape: [batch_size, well_cnn_output_size]

        # Get history
        history = torch.from_numpy(obs["history"]).to(self.device).float()  # Shape: [batch_size, history_length, well_cnn_output_size]
        batch_size, history_length, seq_length, features = history.shape
        history = history.view(-1, seq_length, features)  # Combine batch and history_length
        history = history.permute(0, 2, 1)  # Shape: [batch_size * history_length, features, seq_length]
        history_features = torch.relu(self.well_conv(history))
        history_features = history_features.view(batch_size, history_length, -1)


        # Pass history through TransformerBlock
        transformer_output = self.transformer_block(history_features)
        # Take the last output or mean (depending on your preference)
        history_output = transformer_output[:, -1, :]

        # Combine current well_features with history_output
        combined_features = torch.cat([well_features, history_output], dim=1)

        # Pass through fully connected layers
        x = torch.relu(self.fc1(combined_features))
        x = torch.relu(self.fc2(x))

        # Compute action mean and standard deviation
        mu = self.fc_mu(x)
        sigma = (torch.clamp(self.fc_std(x), min=-5, max=2)).exp()

        return (mu, sigma), state


        
class Critic(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_sizes, device=device):
        super().__init__()
        self.device = device

        # CNN for processing current well_observations
        self.well_conv = nn.Conv1d(
            in_channels=state_shape['well_observations'][1],
            out_channels=64,
            kernel_size=3,
            padding=1
        ).to(self.device) 

        # Calculate well_features size
        with torch.no_grad():
            sample_well_obs = torch.randn(1, state_shape['well_observations'][1], state_shape['well_observations'][0]).to(self.device)
            well_features = torch.relu(self.well_conv(sample_well_obs))
            well_features = well_features.view(well_features.size(0), -1)
            self.well_cnn_output_size = well_features.size(1)

        # TransformerBlock for processing history
        self.transformer_block = TransformerBlock(
            hidden_dim=self.well_cnn_output_size,
            num_heads=4,
            feedforward_dim=256
        ).to(self.device)

        # Fully connected layers
        # The input size includes the transformed history output and the action
        self.history_output_size = self.well_cnn_output_size
        combined_size = self.well_cnn_output_size +self.history_output_size + action_shape[0]
        self.fc1 = nn.Linear(combined_size, hidden_sizes[0]).to(self.device)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1]).to(self.device)
        self.fc_out = nn.Linear(hidden_sizes[1], 1).to(self.device)  # Output a single Q-value

    def forward(self, obs, action):
        # Process current well_observations
        well_obs = torch.from_numpy(obs['well_observations']).to(self.device).permute(0, 2, 1).float()
        well_features = torch.relu(self.well_conv(well_obs))
        well_features = well_features.view(well_features.size(0), -1)  # Shape: [batch_size, well_cnn_output_size]

        # Get history
        history = torch.from_numpy(obs['history']).to(self.device).float()
        batch_size, history_length, seq_length, features = history.shape
        history = history.view(-1, seq_length, features)  # Combine batch and history_length
        history = history.permute(0, 2, 1)  # Shape: [batch_size * history_length, features, seq_length]
        history_features = torch.relu(self.well_conv(history))
        history_features = history_features.view(batch_size, history_length, -1)
        # Pass history through TransformerBlock
        transformer_output = self.transformer_block(history_features)
        # Take the last output or mean (depending on your preference)
        history_output = transformer_output[:, -1, :]  # Shape: [batch_size, well_cnn_output_size]
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
        elif isinstance(action, torch.Tensor):
            action = action.to(self.device)
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

        # Combine history_output with action
        combined_features = torch.cat([well_features,history_output, action], dim=1)

        # Pass through fully connected layers
        x = torch.relu(self.fc1(combined_features))
        x = torch.relu(self.fc2(x))
        q_value = self.fc_out(x)

        return q_value
    

net_a = Actor(state_shape, action_shape, max_action,min_action, device=device)
actor_optim = torch.optim.Adam(net_a.parameters(), lr=3e-4)

net_c1 = Critic(state_shape, action_shape, hidden_sizes, device=device)
critic1_optim = torch.optim.Adam(net_c1.parameters(), lr=3e-4)

net_c2 = Critic(state_shape, action_shape, hidden_sizes, device=device)
critic2_optim = torch.optim.Adam(net_c2.parameters(), lr=3e-4)

# 3. Create the SAC policy
policy = SACPolicy(
    actor=net_a,
    actor_optim=actor_optim,
    critic1=net_c1,
    critic1_optim=critic1_optim,
    critic2=net_c2,
    critic2_optim=critic2_optim,
    tau=0.005,  # Target network update rate
    gamma=0.99,  # Discount factor
    alpha=0.5,   # Temperature parameter
    estimation_step=1,    
    deterministic_eval=False,
    action_space=env.action_space,
     
)

buffer = VectorReplayBuffer(100000, len(train_envs))
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
    max_epoch=15,  # Adjust training parameters as needed
    step_per_epoch=3500,
    step_per_collect=10,
    episode_per_test=10,
    batch_size=256,
    save_best_fn=save_best_fn,
    logger=logger,
).run()

writer.close()
