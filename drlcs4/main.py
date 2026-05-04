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
hidden_sizes = [128, 128]


#Define the 1d conv layer here

##
class StatePredictor(nn.Module):
    def __init__(self, well_obs_shape, output_size, hidden_sizes=[256, 256]):
        super(StatePredictor, self).__init__()
        self.fc1 = nn.Linear(well_obs_shape[0] * well_obs_shape[1], hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
    
    def forward(self, well_obs):
        # Flatten the well_observations: [batch_size, num_wells, num_features] -> [batch_size, num_wells * num_features]
        x = well_obs.view(well_obs.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer without activation
        return x
    
class ThreeDCNN(nn.Module):
    def __init__(self, channels=2):  # Add channels as a parameter for flexibility
        super(ThreeDCNN, self).__init__()
        self.channels = channels 

        self.conv1 = nn.Conv3d(in_channels=self.channels, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.global_pool = nn.AdaptiveAvgPool3d(1) 
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        return x
    

class ActorWithPredictor(nn.Module):
    def __init__(self, state_shape, action_shape, max_action, min_action, state_predictor, device=device):
        super(ActorWithPredictor, self).__init__()
        self.device = device
        self.state_predictor = state_predictor
        self.training_mode = True  # Flag to switch between training and testing

        self.cnn = ThreeDCNN().to(self.device)  # Your existing 3D CNN
        self.well_conv = nn.Conv1d(
            in_channels=state_shape['well_observations'][1],  # Number of features (15)
            out_channels=16,
            kernel_size=2,
            padding=1
        ).to(self.device)

        # Determine the size of image_features
        with torch.no_grad():
            sample_input = torch.randn(1, *state_shape['res_state']).to(self.device)
            self.cnn_output_size = self.cnn(sample_input).view(1, -1).size(1)
            sample_well_obs = torch.randn(1, *state_shape['well_observations']).to(self.device)
            self.well_cnn_output_size = self.well_conv(sample_well_obs.permute(0, 2, 1)).view(1, -1).size(1)

        combined_size = self.cnn_output_size + self.well_cnn_output_size
        self.fc1 = nn.Linear(combined_size, 128).to(self.device)
        self.fc2 = nn.Linear(128, 128).to(self.device)
        self.fc_mu = nn.Linear(128, action_shape[0]).to(self.device)
        self.fc_std = nn.Linear(128, action_shape[0]).to(self.device)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_std.weight)
        self.max_action = max_action
        self.min_action = min_action

    def forward(self, obs, state=None, info={}):
        if self.training_mode:
            # Use the actual res_state and pass through CNN
            image = torch.from_numpy(obs['res_state']).to(self.device).float()
            image_features = self.cnn(image)
        else:
            # Use well_observations and pass through StatePredictor
            well_obs = torch.from_numpy(obs['well_observations']).to(self.device).float()
            with torch.no_grad():
                image_features = self.state_predictor(well_obs)
            # Ensure image_features shape matches self.cnn_output_size
            # If necessary, add a linear layer to match dimensions
            # Assuming StatePredictor output_size matches cnn_output_size

        # Process well_observations
        well_obs = torch.from_numpy(obs['well_observations']).to(self.device).float()
        well_obs = well_obs.permute(0, 2, 1)  # [batch_size, num_features, num_wells]
        well_features = torch.relu(self.well_conv(well_obs))
        well_features = well_features.view(well_features.size(0), -1)

        # Combine image_features and well_features
        combined_obs = torch.cat([image_features, well_features], dim=1)
        x = torch.relu(self.fc1(combined_obs))
        logits = torch.relu(self.fc2(x))

        mu = self.fc_mu(logits)
        sigma = self.fc_std(logits)
        sigma = torch.clamp(sigma, min=-5, max=2).exp()

        return (mu, sigma), state



        
class CriticWithPredictor(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_sizes, state_predictor, device=device):
        super(CriticWithPredictor, self).__init__()
        self.device = device
        self.state_predictor = state_predictor
        self.training_mode = True  # Flag to switch between training and testing

        self.cnn = ThreeDCNN().to(self.device)  # Your existing 3D CNN
        self.well_conv = nn.Conv1d(
            in_channels=state_shape['well_observations'][1],  # Number of features (15)
            out_channels=16,
            kernel_size=2,
            padding=1
        ).to(self.device)

        # Determine the size of image_features
        with torch.no_grad():
            sample_input = torch.randn(1, *state_shape['res_state']).to(self.device)
            cnn_output_size = self.cnn(sample_input).view(1, -1).size(1)
            sample_well_obs = torch.randn(1, *state_shape['well_observations']).to(self.device)
            well_cnn_output_size = self.well_conv(sample_well_obs.permute(0, 2, 1)).view(1, -1).size(1)

        combined_size = cnn_output_size + well_cnn_output_size + action_shape[0]
        self.fc1 = nn.Linear(combined_size, hidden_sizes[0]).to(self.device)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1]).to(self.device)
        self.fc3 = nn.Linear(hidden_sizes[1], 1).to(self.device)  # Output a single value (Q-value)

    def forward(self, obs, action):
        if self.training_mode:
            # Use the actual res_state and pass through CNN
            image = torch.from_numpy(obs['res_state']).to(self.device).float()
            image_features = self.cnn(image)
        else:
            # Use well_observations and pass through StatePredictor
            well_obs = torch.from_numpy(obs['well_observations']).to(self.device).float()
            with torch.no_grad():
                image_features = self.state_predictor(well_obs)
            # Ensure image_features shape matches self.cnn_output_size
            # If necessary, add a linear layer to match dimensions
            # Assuming StatePredictor output_size matches cnn_output_size

        # Process well_observations
        well_obs = torch.from_numpy(obs['well_observations']).to(self.device).float()
        well_obs = well_obs.permute(0, 2, 1)  # [batch_size, num_features, num_wells]
        well_features = torch.relu(self.well_conv(well_obs))
        well_features = well_features.view(well_features.size(0), -1)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
        elif isinstance(action, torch.Tensor):
            action = action.to(self.device)
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

        # Combine image_features, well_features, and action
        combined_obs = torch.cat([image_features, well_features, action], dim=1)
        x = torch.relu(self.fc1(combined_obs))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    

state_predictor = StatePredictor(
    well_obs_shape=state_shape['well_observations'],  # e.g., (9, 15)
    output_size=128,  # Example size; ensure it matches image_features size
    hidden_sizes=[256, 256]
).to(device)

# Initialize the Actor and Critics with the StatePredictor
net_a = ActorWithPredictor(state_shape, action_shape, max_action, min_action, state_predictor, device=device)
actor_optim = torch.optim.Adam(net_a.parameters(), lr=1e-3)

net_c1 = CriticWithPredictor(state_shape, action_shape, hidden_sizes=[128, 128], state_predictor=state_predictor, device=device)
critic1_optim = torch.optim.Adam(net_c1.parameters(), lr=1e-3)

net_c2 = CriticWithPredictor(state_shape, action_shape, hidden_sizes=[128, 128], state_predictor=state_predictor, device=device)
critic2_optim = torch.optim.Adam(net_c2.parameters(), lr=1e-3)



# Define optimizer and loss function for the StatePredictor
state_predictor_optim = torch.optim.Adam(state_predictor.parameters(), lr=1e-3)
state_predictor_loss_fn = nn.MSELoss()

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
    estimation_step=1,  # Number of TD steps for Q-value estimation
    action_space=env.action_space,
    deterministic_eval=False,

     
)

buffer = VectorReplayBuffer(20000, len(train_envs))
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs,exploration_noise=True)


logdir = "log"
log_path = os.path.join(logdir, "sac")

# Define the save_best_fn
def save_best_fn(policy: BasePolicy) -> None:
    torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

def train_fn(epoch, env_step):
    # Ensure the policy is in training mode
    net_a.training_mode = True
    net_c1.training_mode = True
    net_c2.training_mode = True

    # Sample a batch from the replay buffer
    batch = buffer.sample(batch_size=64)

    # Extract well_observations and res_state from the batch
    well_obs = torch.from_numpy(batch.obs['well_observations']).float().to(device)  # Shape: [batch_size, 9, 15]
    res_state = torch.from_numpy(batch.obs['res_state']).float().to(device)        # Shape: [batch_size, 2, 4, 163, 120]

    # Obtain image_features from the policy's CNN (detach to prevent gradients flowing back)
    with torch.no_grad():
        image_features = net_a.cnn(res_state)  # Shape: [batch_size, feature_size]
        image_features = image_features.view(image_features.size(0), -1)  # Flatten if necessary

    # Predict image_features using the StatePredictor
    predicted_features = state_predictor(well_obs)

    # Compute the loss (MSE between predicted and actual image_features)
    loss = nn.MSELoss()(predicted_features, image_features)

    # Backpropagation
    state_predictor_optim.zero_grad()
    loss.backward()
    state_predictor_optim.step()

    # Logging
    writer.add_scalar('StatePredictor/Loss', loss.item(), env_step)

def test_fn(epoch, env_step):
    # Switch the policy to testing mode
    net_a.training_mode = False
    net_c1.training_mode = False
    net_c2.training_mode = False

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
    batch_size=64,
    save_best_fn=save_best_fn,
    train_fn=train_fn,
    test_fn=test_fn,
    logger=logger,
).run()

writer.close()
