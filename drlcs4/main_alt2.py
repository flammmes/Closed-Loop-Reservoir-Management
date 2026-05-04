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

# Ensure you have your ReservoirEnv correctly implemented and accessible
from env import ReservoirEnv  # Replace with your actual environment import

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===============================
# 1. Model Definitions
# ===============================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        
        # Adjust div_term slicing for odd indices when d_model is odd
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])  # Odd indices
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, T, F]
        Returns:
            Tensor: Positionally encoded tensor of shape [batch_size, T, F]
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return x

class TransformerStatePredictor(nn.Module):
    def __init__(self, 
                 num_features, 
                 secondary_time_level, 
                 transformer_hidden_dim=256, 
                 transformer_num_heads=8, 
                 transformer_num_layers=4, 
                 transformer_dropout=0.1,
                 latent_dim=256, 
                 res_channels=2, 
                 res_depth=4, 
                 res_height=163, 
                 res_width=120):
        """
        Transformer-based State Predictor.

        Args:
            num_features (int): Number of features in well_obs.
            secondary_time_level (int): Secondary time levels in well_obs.
            transformer_hidden_dim (int): Hidden dimension in Transformer.
            transformer_num_heads (int): Number of attention heads.
            transformer_num_layers (int): Number of Transformer encoder layers.
            transformer_dropout (float): Dropout rate in Transformer.
            latent_dim (int): Dimension of the latent vector.
            res_channels (int): Number of channels in res_state.
            res_depth (int): Depth dimension of res_state.
            res_height (int): Height dimension of res_state.
            res_width (int): Width dimension of res_state.
        """
        super(TransformerStatePredictor, self).__init__()
        
        self.num_features = num_features
        self.secondary_time_level = secondary_time_level
        self.latent_dim = latent_dim
        self.res_channels = res_channels
        self.res_depth = res_depth
        self.res_height = res_height
        self.res_width = res_width
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=num_features, max_len=secondary_time_level)
        
        # Transformer Encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features, 
            nhead=transformer_num_heads, 
            dim_feedforward=transformer_hidden_dim, 
            dropout=transformer_dropout,
            batch_first=True  # Set batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=transformer_num_layers
        )
        
        # Linear layer to project Transformer's output to latent vector
        self.fc_latent = nn.Linear(num_features, latent_dim)
        
        # Decoder to map latent vector to res_state
        # Calculate the total size of res_state
        self.res_total_size = res_channels * res_depth * res_height * res_width
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.res_total_size)
        )
        
    def forward(self, well_obs):
        """
        Args:
            well_obs (Tensor): Input tensor of shape [batch_size, T, F]
        Returns:
            Tensor: res_state of shape [batch_size, C, D, H, W]
        """
        # Apply positional encoding
        x = self.positional_encoding(well_obs)  # [batch_size, T, F]
        
        # Pass through Transformer Encoder directly
        transformer_output = self.transformer_encoder(x)  # [batch_size, T, F]
        
        # Aggregate the Transformer output (mean pooling across T)
        transformer_output = transformer_output.mean(dim=1)  # [batch_size, F]
        
        # Project to latent vector
        latent = F.relu(self.fc_latent(transformer_output))  # [batch_size, latent_dim]
        
        # Decode to res_state
        res_state_flat = self.fc_decoder(latent)  # [batch_size, C*D*H*W]
        
        # Reshape to [batch_size, C, D, H, W]
        res_state = res_state_flat.view(
            -1, 
            self.res_channels, 
            self.res_depth, 
            self.res_height, 
            self.res_width
        )
        
        return res_state



class ThreeDCNN(nn.Module):
    def __init__(self, channels=2):
        """
        3D Convolutional Neural Network for processing res_state.

        Args:
            channels (int): Number of input channels.
        """
        super(ThreeDCNN, self).__init__()
        self.channels = channels 

        self.conv1 = nn.Conv3d(in_channels=self.channels, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.global_pool = nn.AdaptiveAvgPool3d(1) 

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, channels, D, H, W]
        Returns:
            Tensor: Flattened feature vector of shape [batch_size, 128]
        """
        x = F.relu(self.conv1(x))  # [batch_size, 16, D/2, H/2, W/2]
        x = F.relu(self.conv2(x))  # [batch_size, 32, D/4, H/4, W/4]
        x = F.relu(self.conv3(x))  # [batch_size, 64, D/8, H/8, W/8]
        x = F.relu(self.conv4(x))  # [batch_size, 128, D/16, H/16, W/16]
        x = self.global_pool(x)     # [batch_size, 128, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [batch_size, 128]
        return x


class ActorWithPredictor(nn.Module):
    def __init__(self, state_shape, action_shape, max_action, min_action, state_predictor, device=device):
        """
        Actor network integrating the TransformerStatePredictor.

        Args:
            state_shape (dict): Shape of the observations.
            action_shape (tuple): Shape of the action space.
            max_action (np.ndarray): Maximum action values.
            min_action (np.ndarray): Minimum action values.
            state_predictor (nn.Module): TransformerStatePredictor instance.
            device (torch.device): Device to run the model on.
        """
        super(ActorWithPredictor, self).__init__()
        self.device = device
        self.state_predictor = state_predictor

        self.cnn = ThreeDCNN().to(self.device)  # 3D CNN for res_state
        self.well_conv = nn.Conv1d(
            in_channels=state_shape['well_observations'][1],  # Number of features (15)
            out_channels=16,
            kernel_size=2,
            padding=1
        ).to(self.device)

        # Determine the size of well_features
        with torch.no_grad():
            sample_well_obs = torch.randn(1, *state_shape['well_observations']).to(self.device)
            well_features_size = self.well_conv(sample_well_obs.permute(0, 2, 1)).view(1, -1).size(1)

        # Determine the size of image_features from CNN
        with torch.no_grad():
            sample_res_state = torch.randn(1, *state_shape['res_state']).to(self.device)
            cnn_output_size = self.cnn(sample_res_state).view(1, -1).size(1)

        combined_size = cnn_output_size + well_features_size
        self.fc1 = nn.Linear(combined_size, 128).to(self.device)
        self.fc2 = nn.Linear(128, 128).to(self.device)
        self.fc_mu = nn.Linear(128, action_shape[0]).to(self.device)
        self.fc_std = nn.Linear(128, action_shape[0]).to(self.device)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_std.weight)
        self.max_action = max_action
        self.min_action = min_action

    def forward(self, obs, state=None, info={}):
        """
        Forward pass for the actor.

        Args:
            obs (dict): Observation dictionary containing 'well_observations' and 'res_state'.
            state (Any, optional): Recurrent state (unused).
            info (dict, optional): Additional information.

        Returns:
            tuple: (mu, sigma) for the action distribution, and state.
        """

        # Process well_observations
        well_obs = torch.from_numpy(obs['well_observations']).to(self.device).float()
        predicted_res_state = self.state_predictor(well_obs)  # [batch_size, C, D, H, W]
        image_features = self.cnn(predicted_res_state)        # [batch_size, 128]
        well_obs = well_obs.permute(0, 2, 1)  # [batch_size, F, T]
        well_features = torch.relu(self.well_conv(well_obs))
        # [batch_size, 16, T']
        well_features = well_features.view(well_features.size(0), -1)  # [batch_size, features]

        # Combine image_features and well_features
        combined_obs = torch.cat([image_features, well_features], dim=1)  # [batch_size, 128 + features]
        x = torch.relu(self.fc1(combined_obs))
        logits = torch.relu(self.fc2(x))

        mu = self.fc_mu(logits)
        sigma = self.fc_std(logits)
        sigma = torch.clamp(sigma, min=-5, max=2).exp()

        return (mu, sigma), state



class CriticWithPredictor(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_sizes, state_predictor, device=device):
        """
        Critic network integrating the TransformerStatePredictor.

        Args:
            state_shape (dict): Shape of the observations.
            action_shape (tuple): Shape of the action space.
            hidden_sizes (list): Hidden layer sizes.
            state_predictor (nn.Module): TransformerStatePredictor instance.
            device (torch.device): Device to run the model on.
        """
        super(CriticWithPredictor, self).__init__()
        self.device = device
        self.state_predictor = state_predictor

        self.cnn = ThreeDCNN().to(self.device)  # 3D CNN for res_state
        self.well_conv = nn.Conv1d(
            in_channels=state_shape['well_observations'][1],  # Number of features (15)
            out_channels=16,
            kernel_size=2,
            padding=1
        ).to(self.device)

        # Determine the size of well_features
        with torch.no_grad():
            sample_well_obs = torch.randn(1, *state_shape['well_observations']).to(self.device)
            well_features_size = self.well_conv(sample_well_obs.permute(0, 2, 1)).view(1, -1).size(1)

        # Determine the size of image_features from CNN
        with torch.no_grad():
            sample_res_state = torch.randn(1, *state_shape['res_state']).to(self.device)
            cnn_output_size = self.cnn(sample_res_state).view(1, -1).size(1)

        combined_size = cnn_output_size + well_features_size + action_shape[0]
        self.fc1 = nn.Linear(combined_size, hidden_sizes[0]).to(self.device)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1]).to(self.device)
        self.fc3 = nn.Linear(hidden_sizes[1], 1).to(self.device)  # Output a single value (Q-value)

    def forward(self, obs, action):
        """
        Forward pass for the critic.

        Args:
            obs (dict): Observation dictionary containing 'well_observations' and 'res_state'.
            action (np.ndarray): Action taken.

        Returns:
            Tensor: Q-value.
        """

        # Process well_observations
        well_obs = torch.from_numpy(obs['well_observations']).to(self.device).float()
        predicted_res_state = self.state_predictor(well_obs)  # [batch_size, C, D, H, W]
        image_features = self.cnn(predicted_res_state)        # [batch_size, 128]
        well_obs = well_obs.permute(0, 2, 1)  # [batch_size, F, T]
        well_features = torch.relu(self.well_conv(well_obs))
        # [batch_size, 16, T']
        well_features = well_features.view(well_features.size(0), -1)  # [batch_size, features]

        # Process action
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
        elif isinstance(action, torch.Tensor):
            action = action.to(self.device)
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

        # Combine image_features, well_features, and action
        combined_obs = torch.cat([image_features, well_features, action], dim=1)  # [batch_size, 128 + features + action_dim]
        x = torch.relu(self.fc1(combined_obs))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)  # [batch_size, 1]

        return q_value



# ===============================
# 2. Environment and Replay Buffer Setup
# ===============================

def make_env(env_id):

    return ReservoirEnv(env_id)


# Create a list of environment IDs for training and testing
train_env_ids = range(1, 65)  # Example environment IDs for training
test_env_ids = range(65, 75)   # Example environment IDs for testing

# Create vectorized environments for training
train_envs = SubprocVectorEnv([lambda env_id=env_id: make_env(env_id) for env_id in train_env_ids])

# Create vectorized environments for testing
test_envs = SubprocVectorEnv([lambda env_id=env_id: make_env(env_id) for env_id in test_env_ids])

# Access a single environment to get the observation and action shapes
env = make_env(0)
state_shape = {
    key: env.observation_space[key].shape for key in env.observation_space.spaces.keys()
}

print("Observation Shapes:", state_shape)  # Debug

action_shape = env.action_space.shape

max_action = env.action_space.high
min_action = env.action_space.low
hidden_sizes = [128, 128]


# ===============================
# 3. Instantiate Models and Optimizers
# ===============================

# Define the TransformerStatePredictor parameters based on state_shape['well_observations'] and state_shape['res_state']
num_features = state_shape['well_observations'][1]  # e.g., 15
secondary_time_level = state_shape['well_observations'][0]  # e.g., 9
res_channels = state_shape['res_state'][0]  # e.g., 2
res_depth = state_shape['res_state'][1]     # e.g., 4
res_height = state_shape['res_state'][2]    # e.g., 163
res_width = state_shape['res_state'][3]     # e.g., 120

# Instantiate the TransformerStatePredictor
state_predictor = TransformerStatePredictor(
    num_features=num_features,
    secondary_time_level=secondary_time_level,
    transformer_hidden_dim=256,
    transformer_num_heads=5,
    transformer_num_layers=4,
    transformer_dropout=0.1,
    latent_dim=1024,
    res_channels=res_channels,
    res_depth=res_depth,
    res_height=res_height,
    res_width=res_width
).to(device)

# Initialize the Actor and Critics with the Transformer-based StatePredictor
net_a = ActorWithPredictor(state_shape, action_shape, max_action, min_action, state_predictor, device=device)
actor_optim = torch.optim.Adam(net_a.parameters(), lr=1e-3)

net_c1 = CriticWithPredictor(state_shape, action_shape, hidden_sizes=[128, 128], state_predictor=state_predictor, device=device)
critic1_optim = torch.optim.Adam(net_c1.parameters(), lr=1e-3)

net_c2 = CriticWithPredictor(state_shape, action_shape, hidden_sizes=[128, 128], state_predictor=state_predictor, device=device)
critic2_optim = torch.optim.Adam(net_c2.parameters(), lr=1e-3)

# Define optimizer and loss function for the StatePredictor
state_predictor_optim = torch.optim.Adam(state_predictor.parameters(), lr=1e-3)
state_predictor_loss_fn = nn.MSELoss()


# ===============================
# 4. Create the SAC Policy
# ===============================

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
    estimation_step=1,  # Number of TD steps for Q-value estimation
    action_space=env.action_space,
    deterministic_eval=False,

).to(device)

# ===============================
# 5. Setup Replay Buffer and Collectors
# ===============================

buffer = VectorReplayBuffer(20000, len(train_envs))
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True)


# ===============================
# 6. Define Logging and Save Function
# ===============================

logdir = "log"
log_path = os.path.join(logdir, "sac")

# Ensure the log directory exists
os.makedirs(log_path, exist_ok=True)

# Define the save_best_fn
def save_best_fn(policy: BasePolicy) -> None:
    torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)


# ===============================
# 7. Define Training and Testing Functions
# ===============================

def train_fn(epoch, env_step):
    """
    Custom training function to train the StatePredictor.

    Args:
        epoch (int): Current epoch number.
        env_step (int): Current environment step.
    """
    # Ensure the policy is in training mode
    net_a.training_mode = True
    net_c1.training_mode = True
    net_c2.training_mode = True

    # Sample a batch from the replay buffer
    batch = buffer.sample(batch_size=64)

    # Extract well_observations and res_state from the batch
    well_obs = torch.from_numpy(batch.obs['well_observations']).float().to(device)  # Shape: [batch_size, T, F]
    res_state = torch.from_numpy(batch.obs['res_state']).float().to(device)        # Shape: [batch_size, C, D, H, W]

    # Obtain image_features from the policy's CNN (detach to prevent gradients flowing back)
    with torch.no_grad():
        image_features = net_a.cnn(res_state)  # Shape: [batch_size, cnn_output_size]
        image_features = image_features.view(image_features.size(0), -1)  # Flatten if necessary

    # Predict res_state using the TransformerStatePredictor
    predicted_res_state = state_predictor(well_obs)  # Shape: [batch_size, C, D, H, W]

    # Compute the loss (MSE between predicted and actual res_state)
    loss = state_predictor_loss_fn(predicted_res_state, res_state)

    # Backpropagation
    state_predictor_optim.zero_grad()
    loss.backward()
    state_predictor_optim.step()

    # Logging
    writer.add_scalar('StatePredictor/Loss', loss.item(), env_step)


def test_fn(epoch, env_step):
    """
    Custom testing function to switch the policy to testing mode.

    Args:
        epoch (int): Current epoch number.
        env_step (int): Current environment step.
    """
    # Switch the policy to testing mode
    net_a.training_mode = False
    net_c1.training_mode = False
    net_c2.training_mode = False


# ===============================
# 8. Train the Agent
# ===============================

result = OffpolicyTrainer(
    policy=policy, 
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=15,          # Adjust training parameters as needed
    step_per_epoch=3500,
    step_per_collect=10,
    episode_per_test=10,
    batch_size=64,
    save_best_fn=save_best_fn,
    train_fn=train_fn,
    test_fn=test_fn,
    logger=logger,
).run()

writer.close()
