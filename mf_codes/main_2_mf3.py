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
from typing import Dict, Any, Union

from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.utils.net.common import Net, DataParallelNet

3
# Local imports
from env_2_mf3 import ReservoirEnv  # Ensure this module is correctly implemented
from multiprocessing import Manager

# Initialize a Manager for shared state
manager = Manager()
train_masking_ratio = manager.Value('d', 0.0)  # Starts with no masking
test_masking_ratio = manager.Value('d', 1.0)  

def make_env(env_id, masking_ratio):
    return ReservoirEnv(env_id, masking_ratio)
train_env_ids = range(1,76) # Example environment IDs for training
test_env_ids = range(76,77)   # Example environment IDs for testing
# Correct creation:
train_envs = SubprocVectorEnv([
    lambda env_id=env_id: make_env(env_id, train_masking_ratio) for env_id in train_env_ids
])
test_envs = SubprocVectorEnv([
    lambda env_id=env_id: make_env(env_id, test_masking_ratio) for env_id in test_env_ids
])

# Create a list of environment IDs for training and testing
train_env_ids = range(1,151) # Example environment IDs for training
test_env_ids = range(151,152)   # Example environment IDs for testing

# Create vectorized environments for training

# Create vectorized environments for testing
# Access a single environment to get the observation and action shapes
env = make_env(0,0)
state_shape = {
    key: env.observation_space[key].shape for key in env.observation_space.spaces.keys()
}


action_shape = env.action_space.shape

max_action = 1.0
min_action = -1.0
hidden_sizes =[256, 256]


#Define the 1d conv layer here

##

device = torch.device('cuda')

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
            res_state=obs_gpu["res_state"],
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
            res_state=obs_gpu["res_state"],
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




#Define the 1d conv layer here

##
class ThreeDCNN(nn.Module):
    def __init__(self, channels=2):
        super(ThreeDCNN, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=channels,out_channels=16,kernel_size=(2, 3, 3),stride=(1, 2, 2),padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv3d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.conv4 = nn.Conv3d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        #self.conv5 = nn.Conv3d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #x = F.relu(self.conv5(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

class Actor(nn.Module):
    def __init__(self, state_shape, action_shape, max_action,min_action,hidden_sizes):
        super().__init__()
        #self.cnn = ThreeDCNN()
        self.well_conv = nn.Conv1d(
            in_channels=state_shape['well_observations'][1],
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.device = torch.device('cuda')     
        # Calculate the size of the flattened 3D CNN output
        with torch.no_grad():
            #sample_input = torch.randn(1, 2, 4, 163, 120) # Example input shape
            #cnn_output_size = self.cnn(sample_input).view(1, -1).size(1)
            sample_well_obs = torch.randn(1, *state_shape['well_observations'])
            well_cnn_output_size = self.well_conv(sample_well_obs.permute(0, 2, 1)).view(1, -1).size(1)

        # Combine the CNN output size with the sizes of other observations
        combined_size =  well_cnn_output_size

        self.fc1 = nn.Linear(combined_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_mu = nn.Linear(hidden_sizes[1], action_shape[0])
        self.fc_std = nn.Linear(hidden_sizes[1], action_shape[0])
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_std.weight)
        self.max_action = max_action
        self.min_action = min_action
        
        #self.sigma_param = nn.Parameter(torch.ones(action_shape[0], 1))*0.1

    def forward(self, res_state,well_observations, state=None, info={}):
        #image = obs['res_state']
        #image = res_state
        #image_features = self.cnn(image)
        #well_obs = obs['well_observations']
        well_obs = well_observations
        well_obs = well_obs.permute(0, 2, 1).float()  # Change the shape to[batch_size, num_features, num_wells]
        well_features = torch.relu(self.well_conv(well_obs))
        well_features = well_features.view(well_features.size(0), -1)
        # Flatten the CNN output and combine with other observations
        #This should change
        combined_obs  = well_features

        x = torch.relu(self.fc1(combined_obs))
        logits = torch.relu(self.fc2(x))

    # Compute the action mean (mu)
        mu = self.fc_mu(logits)
        sigma = self.fc_std(logits)
        sigma = F.softplus(sigma) + 1e-6



        #sigma = (torch.clamp(sigma, min=-3, max=-1)).exp() # Make sigma always positive
    # Compute the action standard deviation (sigma)


    # Return the action mean and standard deviation
        return (mu, sigma), state


        
class Critic(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_sizes):
        super().__init__()
        self.cnn = ThreeDCNN()  # Our 3D CNN

        self.well_conv = nn.Conv1d(
            in_channels=state_shape['well_observations'][1],
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.device = torch.device('cuda')        
        # Calculate the size of the flattened 3D CNN output
        with torch.no_grad():
            sample_input = torch.randn(1, 2, 4, 163, 120)  # Example input shape
            cnn_output_size = self.cnn(sample_input).view(1, -1).size(1)
            sample_well_obs = torch.randn(1, *state_shape['well_observations'])
            well_cnn_output_size = self.well_conv(sample_well_obs.permute(0, 2, 1)).view(1, -1).size(1)

        # Combine the CNN output size with the sizes of other observations and actions
        #This should change
        combined_size = cnn_output_size + well_cnn_output_size + action_shape[0]

        self.fc1 = nn.Linear(combined_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1) # Output a single value (Q-value)

    def forward(self, res_state,well_observations, action):
        #image = obs['res_state']
        image = res_state
        image_features = self.cnn(image)
        #well_obs = obs['well_observations']
        well_obs = well_observations
        well_obs = well_obs.permute(0, 2, 1).float()  # Change the shape to [batch_size, num_features, num_wells]
        well_features = torch.relu(self.well_conv(well_obs))
        well_features = well_features.view(well_features.size(0), -1)
        # Flatten the CNN output and combine with other observations and actions
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        elif isinstance(action, torch.Tensor):
            action = action

        action = torch.as_tensor(action, device=image_features.device).float()

        combined_obs  = torch.cat([image_features,well_features,action], dim=1)


        x = torch.relu(self.fc1(combined_obs))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

primary_device = torch.device('cuda:0')

net_a = Actor(state_shape, action_shape, max_action,min_action, hidden_sizes)

parallel_actor = ActorWrapperDP(net_a).to(primary_device)

actor_optim = torch.optim.Adam(parallel_actor.parameters(), lr=3e-4)

net_c1 = Critic(state_shape, action_shape, hidden_sizes)
parallel_critic = CriticWrapperDP(net_c1).to(primary_device)
critic1_optim = torch.optim.Adam(parallel_critic.parameters(), lr=1e-4)

net_c2 = Critic(state_shape, action_shape, hidden_sizes)

parallel_critic2 = CriticWrapperDP(net_c2).to(primary_device)
critic2_optim = torch.optim.Adam(parallel_critic2.parameters(), lr=1e-4)
#target_entropy = -action_shape[0]  # e.g. -5.5
target_entropy = -11
# 2) Create a learnable log_alpha parameter and optimizer.
log_alpha = torch.zeros(1, requires_grad=True, device=primary_device)
alpha_optim = torch.optim.Adam([log_alpha], lr=1e-4)
# 3. Create the SAC policy
policy = SACPolicy(
    actor=parallel_actor,
    actor_optim=actor_optim,
    critic=parallel_critic,
    critic_optim=critic1_optim,
    critic2=parallel_critic2,
    critic2_optim=critic2_optim,
    action_scaling = True,
    tau=0.005,  # Target network update rate
    gamma=0.99,  # Discount factor
    alpha=(target_entropy, log_alpha, alpha_optim),  
    estimation_step=1,  # Number of TD steps for Q-value estimation
    deterministic_eval=True,
    action_space=env.action_space,
    
     
)

buffer = VectorReplayBuffer(90000, len(train_envs))
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs,exploration_noise=True)
p_schedule =[(0.0, 0), # Mask 0.0 for 10 epochs (0-9)
(0.25, 10), # Mask 0.25 for 4 epochs (10-13)
(0.50, 14), # Mask 0.50 for 4 epochs (14-17)
(0.75, 18), # Mask 0.75 for 4 epochs (18-21)
(1.00, 22)] # Mask 1.00 for 8 epochs (22-29)

ratio_by_epoch = {}
for p, start in p_schedule:
    end = p_schedule[p_schedule.index((p, start)) + 1][1] if p < 1.0 else 31
    for e in range(start, end):
        ratio_by_epoch[e] = p
_current_ratio = {"val": -1}

logdir = "log"
log_path = os.path.join(logdir, "sac")

# Define the save_best_fn
def save_best_fn(policy: BasePolicy) -> None:
    torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))


writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)

def train_fn(epoch, _):
    """Set mask ratio *and* reset replay buffer whenever it changes."""
    new_p = ratio_by_epoch[epoch]
    if new_p != _current_ratio["val"]:
        _current_ratio["val"] = new_p
        train_masking_ratio.value = new_p
        buffer.reset()                          # flush old-distribution data
        writer.add_text("curriculum",
                        f"mask ratio switched to {new_p:.2f} (buffer reset)",
                        epoch)
    # log every epoch
    writer.add_scalar("Masking/train_masking_ratio", new_p, epoch)
    writer.add_scalar("Masking/critic_ratio", new_p, epoch)
def test_fn(epoch, _):
    test_masking_ratio.value = 1.0
    writer.add_scalar("Masking/test_masking_ratio", 1.0, epoch)


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
    train_fn=train_fn,
    test_fn=test_fn,
    save_best_fn=save_best_fn,
    logger=logger,
).run()

writer.close()