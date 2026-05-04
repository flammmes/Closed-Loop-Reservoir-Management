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

# Local imports
from env import ReservoirEnv  # Ensure this module is correctly implemented
device = torch.device('cuda')
from multiprocessing import Manager

# Initialize a Manager for shared state
manager = Manager()
train_masking_ratio = manager.Value('d', 0.0)  # Starts with no masking
test_masking_ratio = manager.Value('d', 1.0)  



# Define a function to create a new environment with a unique env_id
def make_train_env(env_id):
    return ReservoirEnv(env_id, train_masking_ratio)

def make_test_env(env_id):
    return ReservoirEnv(env_id, test_masking_ratio)
# Create a list of environment IDs for training and testing
train_env_ids = range(1,65) # Example environment IDs for training
test_env_ids = range(65,75)   # Example environment IDs for testing

# Create vectorized environments for training
train_envs = SubprocVectorEnv([lambda env_id=env_id: make_train_env(env_id) for env_id in train_env_ids])

# Create vectorized testing environments
test_envs = SubprocVectorEnv([lambda env_id=env_id: make_test_env(env_id) for env_id in test_env_ids])
# Access a single environment to get the observation and action shapes
env = make_train_env(0)
state_shape = {
    key: env.observation_space[key].shape for key in env.observation_space.spaces.keys()
}


action_shape = env.action_space.shape

max_action = env.action_space.high
min_action = env.action_space.low
hidden_sizes = [128, 128]

#Define the 1d conv layer here

##
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
    

class Actor(nn.Module):
    def __init__(self, state_shape, action_shape, max_action,min_action, device=device):
        super().__init__()
        self.device = device
        self.cnn = ThreeDCNN().to(self.device)  # Our 3D CNN
        self.well_conv = nn.Conv1d(
            in_channels=state_shape['well_observations'][1],  # Number of features (15)
            out_channels=16,  # You can adjust this
            kernel_size=2, 
            padding=1
        ).to(self.device)        
        # Calculate the size of the flattened 3D CNN output
        with torch.no_grad():
            sample_input = torch.randn(1, 2, 4, 163, 120).to(self.device)  # Example input shape
            cnn_output_size = self.cnn(sample_input).view(1, -1).size(1)
            sample_well_obs = torch.randn(1, *state_shape['well_observations']).to(self.device)
            well_cnn_output_size = self.well_conv(sample_well_obs.permute(0, 2, 1)).view(1, -1).size(1)

        # Combine the CNN output size with the sizes of other observations
        combined_size = cnn_output_size + well_cnn_output_size

        self.fc1 = nn.Linear(combined_size, 128).to(self.device) 
        self.fc2 = nn.Linear(128, 128).to(self.device)
        self.fc_mu = nn.Linear(128, action_shape[0]).to(self.device)
        self.fc_std = nn.Linear(128, action_shape[0]).to(self.device)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_std.weight)
        self.max_action = max_action
        self.min_action = min_action
        
        #self.sigma_param = nn.Parameter(torch.ones(action_shape[0], 1))*0.1

    def forward(self, obs, state=None, info={}):

        image = torch.from_numpy(obs['res_state']).to(self.device)
        image_features = self.cnn(image)
        well_obs = torch.from_numpy(obs['well_observations']).to(self.device)
        well_obs = well_obs.permute(0, 2, 1).float()  # Change the shape to [batch_size, num_features, num_wells]
        well_features = torch.relu(self.well_conv(well_obs))
        well_features = well_features.view(well_features.size(0), -1)
        # Flatten the CNN output and combine with other observations
        #This should change
        combined_obs  = torch.cat([image_features,well_features], dim=1)

        x = torch.relu(self.fc1(combined_obs))
        logits = torch.relu(self.fc2(x))

    # Compute the action mean (mu)
        mu = self.fc_mu(logits)
        sigma = self.fc_std(logits)
        #sigma = F.softplus(self.fc_std(logits)) + 1e-6



        sigma = (torch.clamp(sigma, min=-5, max=2)).exp() # Make sigma always positive
    # Compute the action standard deviation (sigma)


    # Return the action mean and standard deviation
        return (mu, sigma), state


        
class Critic(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_sizes, device=device):
        super().__init__()
        self.device = device
        self.cnn = ThreeDCNN().to(self.device)  # Our 3D CNN

        self.well_conv = nn.Conv1d(
            in_channels=state_shape['well_observations'][1],  # Number of features (15)
            out_channels=16,  # You can adjust this
            kernel_size=2, 
            padding=1
        ).to(self.device)        
        # Calculate the size of the flattened 3D CNN output
        with torch.no_grad():
            sample_input = torch.randn(1, 2, 4, 163, 120).to(self.device)  # Example input shape
            cnn_output_size = self.cnn(sample_input).view(1, -1).size(1)
            sample_well_obs = torch.randn(1, *state_shape['well_observations']).to(self.device)
            well_cnn_output_size = self.well_conv(sample_well_obs.permute(0, 2, 1)).view(1, -1).size(1)

        # Combine the CNN output size with the sizes of other observations and actions
        #This should change
        combined_size = cnn_output_size + well_cnn_output_size + action_shape[0]

        self.fc1 = nn.Linear(combined_size, hidden_sizes[0]).to(self.device)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1]).to(self.device)
        self.fc3 = nn.Linear(hidden_sizes[1], 1).to(self.device)  # Output a single value (Q-value)

    def forward(self, obs, action):
        image = torch.from_numpy(obs['res_state']).to(self.device)
        image_features = self.cnn(image)
        well_obs = torch.from_numpy(obs['well_observations']).to(self.device)
        well_obs = well_obs.permute(0, 2, 1).float()  # Change the shape to [batch_size, num_features, num_wells]
        well_features = torch.relu(self.well_conv(well_obs))
        well_features = well_features.view(well_features.size(0), -1)
        # Flatten the CNN output and combine with other observations and actions
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
        elif isinstance(action, torch.Tensor):
            action = action.to(self.device)
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")
        
        combined_obs  = torch.cat([image_features,well_features,action], dim=1)


        x = torch.relu(self.fc1(combined_obs))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

net_a = Actor(state_shape, action_shape, max_action,min_action, device=device)
actor_optim = torch.optim.Adam(net_a.parameters(), lr=1e-3)

net_c1 = Critic(state_shape, action_shape, hidden_sizes, device=device)
critic1_optim = torch.optim.Adam(net_c1.parameters(), lr=1e-3)

net_c2 = Critic(state_shape, action_shape, hidden_sizes, device=device)
critic2_optim = torch.optim.Adam(net_c2.parameters(), lr=1e-3)

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


writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)

def train_fn(epoch, env_step):
    """
    Callback function to adjust the train_masking_ratio based on the current epoch.
    """
    total_epochs = 10  # Total number of epochs
    first_fixed_epochs = 2  # First 3 epochs with new_ratio = 0
    last_fixed_epochs = 3  # Last 5 epochs with new_ratio = 1

    if epoch < first_fixed_epochs:
        new_ratio = 0  # Keep ratio 0 for the first few epochs
    elif epoch >= total_epochs - last_fixed_epochs:
        new_ratio = 1  # Keep ratio 1 for the last few epochs
    else:
        # Gradually increase new_ratio between first_fixed_epochs and (total_epochs - last_fixed_epochs)
        progress = (epoch - first_fixed_epochs) / (total_epochs - first_fixed_epochs - last_fixed_epochs)
        new_ratio = progress  # This will gradually increase the ratio
    # Update the masking ratio
    train_masking_ratio.value = min(new_ratio, 1.0)

    # Log the masking ratio
    writer.add_scalar("Masking/train_masking_ratio", train_masking_ratio.value, epoch)

def test_fn(epoch, env_step):
    """
    Callback function to ensure that test_masking_ratio is always set to 1.0 during testing.
    """
    test_masking_ratio.value = 1.0
    
    # Log the masking ratio
    writer.add_scalar("Masking/test_masking_ratio", test_masking_ratio.value, epoch)

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
    train_fn=train_fn,
    test_fn=test_fn,
    save_best_fn=save_best_fn,
    logger=logger,
).run()

writer.close()
