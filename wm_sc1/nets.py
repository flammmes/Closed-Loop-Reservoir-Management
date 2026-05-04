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
from env_3_mb1 import ReservoirEnv  # Replace with your actual environment import
from copy import deepcopy # <--- IMPORT THE DEEPCOPY FUNCTION

from tianshou.data.types import ObsBatchProtocol, ActStateBatchProtocol, RolloutBatchProtocol # Ensure these are imported
from tianshou.policy.base import TrainingStats # Ensure this is imported
import gymnasium as gym # For type hint


def augment_observations(
    obs_tensor: torch.Tensor,
    mask_prob: float = 0.15,
) -> torch.Tensor:
    """
    Applies random time-step masking to a batch of observations.
    Works for well_observations (B, 9, 30) or history (B, H, 9, 30).
    """
    shape = obs_tensor.shape
    device = obs_tensor.device
    
    # Mask the sequence dimension(s)
    mask_shape = shape[:-1]
    mask = torch.rand(*mask_shape, device=device) > mask_prob
    
    mask = mask.unsqueeze(-1) # Add feature dimension for broadcasting
    
    augmented_obs = obs_tensor * mask.float()
    return augmented_obs

# Now, modify the CURL class to handle two inputs
class CURL(nn.Module):
    def __init__(self, encoder: nn.Module, latent_dim: int, momentum: float = 0.999):
        super().__init__()
        self.momentum = momentum
        self.query_encoder = encoder
        self.key_encoder = deepcopy(encoder)
        for param in self.key_encoder.parameters():
            param.requires_grad = False

        projection_dim = 128
        self.projector = nn.Linear(latent_dim, projection_dim)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _update_momentum_encoder(self):
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    # --- THIS IS THE KEY CHANGE ---
    def forward(self, history: torch.Tensor, well_observations: torch.Tensor) -> torch.Tensor:
        """
        Calculates the InfoNCE contrastive loss for a history-aware encoder.
        """
        # Create two different augmented views of BOTH inputs
        history_q = augment_observations(history)
        well_obs_q = augment_observations(well_observations)
        
        history_k = augment_observations(history)
        well_obs_k = augment_observations(well_observations)
        
        # --- Compute Queries ---
        # Pass both arguments to the encoder
        query_latent = self.query_encoder(history_q, well_obs_q)
        query_proj = self.projector(query_latent)
        
        # --- Compute Keys ---
        with torch.no_grad():
            # Pass both arguments to the key encoder
            key_latent = self.key_encoder(history_k, well_obs_k)
            key_proj = self.projector(key_latent)
            
        logits = torch.matmul(query_proj, key_proj.T)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = self.cross_entropy_loss(logits, labels)
        
        return loss





class Res3DCNN(nn.Module):
    def __init__(self, in_channels=2, latent_dim=128):
        super().__init__()
        # 4-layer design
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



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


# =============================================================================
#  CORE ENCODER DEFINITIONS
# =============================================================================

class HistoryEncoder(nn.Module):
    """
    Encapsulates the logic for processing historical and current well observations
    using a shared Conv1D and a GTrXL network.
    """
    def __init__(self, d_model=128, num_heads=4, head_dim=32, num_gtrxl_layers=2):
        super().__init__()
        self.d_model = d_model
        
        # A single Conv1D layer to process each 9-step well observation sequence into a feature vector.
        # Input: (B, 30, 9), Output: (B, d_model, 9)
        self.obs_cnn = nn.Conv1d(in_channels=30, out_channels=d_model, kernel_size=3, padding=1)
        
        # The GTrXL network to process the sequence of feature vectors from the history.
        self.gtrxl = GTrXLNet(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            position_wise_mlp_dim=d_model * 2, # A common choice
            num_layers=num_gtrxl_layers
        )

    def forward(self, history: torch.Tensor, well_observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history (torch.Tensor): Shape (B, hist_len, 9, 30)
            well_observations (torch.Tensor): Shape (B, 9, 30)

        Returns:
            torch.Tensor: The final shared feature vector, shape (B, d_model)
        """
        B = well_observations.shape[0]
        hist_len = history.shape[1]

        # --- Process the history sequence ---
        # 1. Reshape for Conv1D: (B * hist_len, 9, 30) -> (B * hist_len, 30, 9)
        hist_flat = history.view(B * hist_len, 9, 30).permute(0, 2, 1)
        
        # 2. Apply Conv1D and pool: (B * hist_len, 30, 9) -> (B * hist_len, d_model, 9) -> (B * hist_len, d_model)
        hist_feat_map = F.relu(self.obs_cnn(hist_flat))
        hist_features = hist_feat_map.mean(dim=2) # Average pooling over the 9 steps
        
        # 3. Reshape for GTrXL: (B * hist_len, d_model) -> (B, hist_len, d_model)
        hist_sequence = hist_features.view(B, hist_len, self.d_model)
        
        # 4. Pass through GTrXL to get the history-aware representation
        gtrxl_output = self.gtrxl(hist_sequence) # Shape: (B, hist_len, d_model)
        history_embedding = gtrxl_output[:, -1, :] # Take the last output, shape: (B, d_model)

        # --- Process the current observation ---
        # 1. Reshape for Conv1D: (B, 9, 30) -> (B, 30, 9)
        current_obs_perm = well_observations.permute(0, 2, 1)
        
        # 2. Apply the SAME Conv1D and pool: (B, 30, 9) -> (B, d_model, 9) -> (B, d_model)
        current_feat_map = F.relu(self.obs_cnn(current_obs_perm))
        current_embedding = current_feat_map.mean(dim=2)

        # --- Combine and return ---
        # The final representation is a combination of the context from history
        # and the specifics of the current observation.
        # A simple addition or concatenation followed by a linear layer is common. Let's use addition.
        shared_feature_vector = history_embedding + current_embedding
        
        return shared_feature_vector


class PolicyHead(nn.Module):
    """The MLP head that computes the action distribution."""
    def __init__(self, d_model=128, action_shape=None, hidden_sizes=[256, 256]):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_sizes[1], action_shape[0])
        self.fc_std = nn.Linear(hidden_sizes[1], action_shape[0])

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(features)
        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_std(x)) + 1e-6
        return mu, sigma

class DistillationHead(nn.Module):
    """The MLP head that predicts the teacher's latent state."""
    def __init__(self, d_model=128, hidden_sizes=[256, 256]):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], d_model) # Output dim matches teacher's latent_dim
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
    


class CriticEncoder(nn.Module):
    """
    The new "Ultimate Critic" encoder. It processes BOTH the privileged 3D state
    with a 3D CNN AND the public history with a GTrXL.
    """
    def __init__(self, latent_dim=128, d_model=128, num_heads=4, head_dim=32):
        super().__init__()
        # --- Component 1: The Privileged 3D State Encoder (The Teacher) ---
        # This is the same 3D CNN as before. Its output is z_teacher for distillation.
        self.teacher_encoder = Res3DCNN(latent_dim=latent_dim)
        
        # --- Component 2: The History Encoder (The Critic's Own GTrXL) ---
        # We give the critic its own HistoryEncoder instance. It will learn to
        # process history from the perspective of estimating value.
        self.history_encoder = HistoryEncoder(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
    def forward(self, res_state: torch.Tensor, history: torch.Tensor, well_observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Processes all available information.

        Returns:
            A tuple containing:
            - z_teacher (torch.Tensor): The latent state from the 3D CNN.
            - history_features (torch.Tensor): The latent state from the GTrXL.
        """
        # Pass privileged 3D state through the 3D CNN
        z_teacher = self.teacher_encoder(res_state)
        
        # Pass public history through the critic's GTrXL
        history_features = self.history_encoder(history, well_observations)
        
        return z_teacher, history_features

class CriticHead(nn.Module):
    """The MLP head that computes the Q-value from the combined features."""
    def __init__(self, action_shape, latent_dim=128, d_model=128, hidden_sizes=[512, 512]):
        super().__init__()
        
        # The input dimension is now the sum of the two feature vectors plus the action
        combined_dim = latent_dim + d_model + action_shape[0]
        
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, z_teacher: torch.Tensor, history_features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Concatenate all features to form the input to the MLP
        combined_input = torch.cat([z_teacher, history_features, action], dim=1)
        q_value = self.net(combined_input)
        return q_value


class StudentDistillationNetwork(nn.Module):
    """
    This is the "shadow" network used only inside the policy's learn method.
    Its job is to provide the z_student prediction for the distillation loss.
    """
    def __init__(self, shared_encoder: HistoryEncoder, distillation_head: DistillationHead):
        super().__init__()
        self.encoder = shared_encoder
        self.distillation_head = distillation_head

    def forward(self, history: torch.Tensor, well_observations: torch.Tensor) -> torch.Tensor:
        shared_features = self.encoder(history, well_observations)
        z_student = self.distillation_head(shared_features)
        return z_student


class Transposed3DCNN(nn.Module):
    def __init__(self, latent_dim=128, out_channels=2):
        super().__init__()
        self.final_shape = (4, 163, 120) # Target D, H, W

        # The shape just before the pooling layer in YOUR Res3DCNN encoder
        self.initial_shape = (128, 1, 11, 8) # (C, D, H, W)
        
        self.fc = nn.Linear(latent_dim, np.prod(self.initial_shape))
        
        self.net = nn.Sequential(
            # Reversing conv4: Input (1, 11, 8) -> Target (1, 21, 15)
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Reversing conv3: Input (1, 21, 15) -> Target (2, 41, 30)
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 0, 1)),
            nn.ReLU(),

            # Reversing conv2: Input (2, 41, 30) -> Target (3, 82, 60)
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0, 1, 1)),
            nn.ReLU(),

            # Reversing conv1: Input (3, 82, 60) -> Target (4, 163, 120)
            nn.ConvTranspose3d(16, out_channels, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 0)),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, *self.initial_shape)
        x = self.net(x)
        
        # Final robust crop to guarantee the shape
        x = x[:, :, :self.final_shape[0], :self.final_shape[1], :self.final_shape[2]]
        
        return x
    

class StudentQHead(nn.Module):
    def __init__(self, d_model: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + action_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )
    def forward(self, z_hist: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z_hist, a], dim=-1))

class StudentCritic(nn.Module):
    """HistoryEncoder + StudentQHead"""
    def __init__(self, enc: HistoryEncoder, head: StudentQHead):
        super().__init__()
        self.enc, self.head = enc, head
    def forward(
        self,
        *,
        res_state: torch.Tensor | None = None,   # ignored
        history: torch.Tensor,
        well_observations: torch.Tensor,
        action: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        z = self.enc(history, well_observations)
        return self.head(z, action)
    

class LatentMapping(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        # Residual: good prior since pre/post spaces should be close
        return z + self.mlp(z)
