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
from env_3_mf4 import ReservoirEnv  # Replace with your actual environment import
from copy import deepcopy # <--- IMPORT THE DEEPCOPY FUNCTION

from tianshou.data.types import ObsBatchProtocol, ActStateBatchProtocol, RolloutBatchProtocol # Ensure these are imported
from tianshou.policy.base import TrainingStats # Ensure this is imported
import gymnasium as gym # For type hint
from nets import CURL, augment_observations, Res3DCNN, GRUGate, SkipConnection, RelativeMultiHeadAttention, GTrXLUnit, GTrXLNet, HistoryEncoder, PolicyHead, DistillationHead, CriticEncoder, CriticHead, StudentDistillationNetwork, Transposed3DCNN, StudentQHead, StudentCritic


def make_env(env_id):
    return ReservoirEnv(env_id)
train_env_ids = range(1,76) # Exa2ple environment IDs for training
test_env_ids = range(76,77)   # Example environment IDs for testing
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
class ActorWrapperDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.dp_module = nn.DataParallel(module)
        # Store the primary device determined during init
        self.primary_device = next(self.dp_module.parameters()).device

    def forward(self, obs: Dict[str, Union[torch.Tensor, np.ndarray]], state: Any = None, info: Dict = {}, **kwargs) -> Any:
        # --- MODIFIED SECTION ---
        # Explicitly select and move ONLY the required observation key
        well_obs_tensor = obs["well_observations"]
        if not isinstance(well_obs_tensor, torch.Tensor):
             well_obs_tensor = torch.as_tensor(well_obs_tensor, dtype=torch.float32)
        well_obs_gpu = well_obs_tensor.to(self.primary_device)

        history_tensor = obs['history']
        if not isinstance(history_tensor, torch.Tensor):
             history_tensor = torch.as_tensor(history_tensor, dtype=torch.float32)
        # ------------------------
        history_tensor_gpu = history_tensor.to(self.primary_device)
        if isinstance(state, torch.Tensor):
            state = state.to(self.primary_device)

        return self.dp_module(
            well_observations=well_obs_gpu, # Pass the GPU tensor
            history = history_tensor_gpu,
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
            history = obs_gpu["history"],
            action=action_gpu,
            *args,
            **kwargs
        )

class TianshouSACActor(nn.Module):
    """
    This is the actor that will be passed to the Tianshou SACPolicy.
    It strictly adheres to the required output format: (mu, sigma), state.
    """
    def __init__(self, shared_encoder: HistoryEncoder, policy_head: PolicyHead):
        super().__init__()
        self.encoder = shared_encoder
        self.policy_head = policy_head
        self.max_action = max_action
        self.min_action = min_action
    def forward(self, history: torch.Tensor, well_observations: torch.Tensor, state: Any = None, info: Dict = {}) -> tuple[tuple[torch.Tensor, torch.Tensor], Any]:
        shared_features = self.encoder(history, well_observations)
        mu, sigma = self.policy_head(shared_features)
        return (mu, sigma), state




class TianshouSACCritic(nn.Module):
    """
    The updated Tianshou wrapper for the "Ultimate Critic".
    It correctly calls the new encoder and head.
    """
    def __init__(self, critic_encoder: CriticEncoder, critic_head: CriticHead):
        super().__init__()
        self.encoder = critic_encoder
        self.head = critic_head

    def forward(self, res_state: torch.Tensor, history: torch.Tensor, well_observations: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # 1. Get both feature vectors from the ultimate encoder
        z_teacher, history_features = self.encoder(res_state, history, well_observations)
        
        # 2. Pass them to the head to get the final Q-value
        q_value = self.head(z_teacher, history_features, action)
        
        return q_value

log_path_dir = "log/rrl" # Changed log path slightly
os.makedirs(log_path_dir, exist_ok=True)
writer = SummaryWriter(log_path_dir)
logger = TensorboardLogger(writer)

primary_device = torch.device('cuda:0')

d_model = 128
shared_history_encoder = HistoryEncoder(d_model=d_model).to(primary_device)
actor_policy_head = PolicyHead(d_model=d_model, action_shape=action_shape).to(primary_device)
actor_distill_head = DistillationHead(d_model=d_model).to(primary_device)
actor_for_tianshou = TianshouSACActor(shared_history_encoder, actor_policy_head)
student_net_for_distill = StudentDistillationNetwork(shared_history_encoder, actor_distill_head)
parallel_actor = ActorWrapperDP(actor_for_tianshou)


class ProjHead(nn.Module):
    def __init__(self, dim, hidden=256, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out)
        )
    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)

class ValueMatchHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
    def forward(self, z):
        return self.mlp(z).squeeze(-1)

teacher_proj = ProjHead(d_model).to(primary_device)
# for p in teacher_proj.parameters():
#     p.requires_grad = False  # frozen
teacher_proj.requires_grad_(False)  # do not optimize teacher side

student_proj = ProjHead(d_model).to(primary_device)
value_match_head = ValueMatchHead(d_model).to(primary_device)



curl_actor = CURL(encoder=shared_history_encoder, latent_dim=d_model).to(primary_device)

actor_optim = torch.optim.Adam([
    {'params': shared_history_encoder.parameters(), 'lr': 3e-4},
    {'params': actor_policy_head.parameters(), 'lr': 3e-4},
    {'params': actor_distill_head.parameters(), 'lr': 3e-4},
    {'params': curl_actor.projector.parameters(), 'lr': 3e-4},
    {'params': student_proj.parameters(), 'lr': 3e-4},
    {'params': value_match_head.parameters(), 'lr': 3e-4},
], lr=3e-4)
# --- Critic Setup ---

# 1. Instantiate the components for Critic 1
critic1_encoder = CriticEncoder(latent_dim=d_model).to(primary_device)
critic1_head = CriticHead(action_shape=action_shape, latent_dim=d_model).to(primary_device)
# Assemble the critic Tianshou expects
critic1_for_tianshou = TianshouSACCritic(critic1_encoder, critic1_head)
parallel_critic1 = CriticWrapperDP(critic1_for_tianshou)

# 2. Instantiate the components for Critic 2
critic2_encoder = CriticEncoder(latent_dim=d_model).to(primary_device)
critic2_head = CriticHead(action_shape=action_shape, latent_dim=d_model).to(primary_device)
# Assemble the critic Tianshou expects
critic2_for_tianshou = TianshouSACCritic(critic2_encoder, critic2_head)
parallel_critic2 = CriticWrapperDP(critic2_for_tianshou)

# 3. Create the optimizers
critic1_optim = torch.optim.Adam([
    {'params': critic1_encoder.parameters()},
    {'params': critic1_head.parameters()}
], lr=1e-4)

critic2_optim = torch.optim.Adam([
    {'params': critic2_encoder.parameters()},
    {'params': critic2_head.parameters()}
], lr=1e-4)


stud1_enc = HistoryEncoder(d_model=d_model).to(primary_device)
stud2_enc = HistoryEncoder(d_model=d_model).to(primary_device)
stud1_head = StudentQHead(d_model, action_shape[0]).to(primary_device)
stud2_head = StudentQHead(d_model, action_shape[0]).to(primary_device)

studQ1_for_tianshou = StudentCritic(stud1_enc, stud1_head)
studQ2_for_tianshou = StudentCritic(stud2_enc, stud2_head)

studQ1 = CriticWrapperDP(studQ1_for_tianshou).to(primary_device)
studQ2 = CriticWrapperDP(studQ2_for_tianshou).to(primary_device)

stud1_optim = torch.optim.Adam(list(stud1_enc.parameters()) + list(stud1_head.parameters()), lr=1e-4)
stud2_optim = torch.optim.Adam(list(stud2_enc.parameters()) + list(stud2_head.parameters()), lr=1e-4)

target_entropy = -11
# 2) Create a learnable log_alpha parameter and optimizer.
log_alpha = torch.zeros(1, requires_grad=True, device=primary_device)
alpha_optim = torch.optim.Adam([log_alpha], lr=1e-4)
# --- SAC Policy Setup ---
# The `action_space=env.action_space` (from the single 'env' instance) is crucial
# to avoid the ValueError with `action_scaling`.

DISTILL_START = 22
DISTILL_END = 30
CURL_START = 28
CURL_END = 30

# Define final loss weights
DISTILL_WEIGHT = 0.1
CURL_WEIGHT = 0.0

policy = DistillationSACPolicy(
    # New custom arguments
    student_critic=studQ1,
    student_critic2=studQ2,
    student_critic_optim=stud1_optim,
    student_critic2_optim=stud2_optim,

    student_distill_net=student_net_for_distill,
    critic1_encoder=critic1_encoder,
    critic2_encoder=critic2_encoder, # Pass the second critic's encoder too
    curl_module=curl_actor,
    teacher_proj=teacher_proj,
    student_proj=student_proj,
    value_match_head=value_match_head,
    distill_start_epoch=DISTILL_START,
    distill_end_epoch=DISTILL_END,
    curl_start_epoch=CURL_START,
    curl_end_epoch=CURL_END,
    distillation_loss_weight=DISTILL_WEIGHT,
    curl_loss_weight=CURL_WEIGHT,
    
    # Standard Tianshou arguments
    actor=parallel_actor,
    actor_optim=actor_optim,
    critic=parallel_critic1,
    critic_optim=critic1_optim,
    critic2=parallel_critic2,
    critic2_optim=critic2_optim,
    tau=0.005,
    gamma=0.98,
    alpha=(target_entropy, log_alpha, alpha_optim),  
    estimation_step=1,
    action_space=env.action_space,
    action_scaling=True
)
policy._grad_norm = 1.0 # Set grad norm for clipping

# =============================================================================
# YOUR ORIGINAL BUFFER, COLLECTOR, LOGGER, TRAINER SETUP - Mostly unchanged
# =============================================================================
buffer_size = 90000
buffer = VectorReplayBuffer(buffer_size, len(train_envs))
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True) # Original was True



def train_fn(epoch, env_step):
    # This is CRUCIAL for the annealing schedule in the policy
    policy.current_epoch = epoch
    
    # --- CORRECTED LOGGING ---
    # Log the current annealed weight for the distillation loss
    current_distill_w = policy._get_current_weight(
        policy.distill_start_epoch, 
        policy.distill_end_epoch, 
        policy.final_distill_weight
    )
    logger.writer.add_scalar("Annealing/distillation_weight", current_distill_w, epoch)
    
    # Log the current annealed weight for the CURL loss
    current_curl_w = policy._get_current_weight(
        policy.curl_start_epoch, 
        policy.curl_end_epoch, 
        policy.final_curl_weight
    )
    logger.writer.add_scalar("Annealing/curl_weight", current_curl_w, epoch)

def test_fn(epoch, env_step):
    pass # No changes needed here

def save_best_fn(policy: BasePolicy) -> None:
    torch.save(policy.state_dict(), os.path.join(log_path_dir, "policy.pth"))

# Trainer
print("Starting OffpolicyTrainer...")
result = OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=30,  # Adjust training parameters as needed
    step_per_epoch=3000,
    step_per_collect=1500,
    #update_per_step = 0.5,
    episode_per_test=1,
    batch_size=2048,
    save_best_fn = save_best_fn,
    train_fn=train_fn,
    test_fn=test_fn,
    logger=logger,
).run()

writer.close()
print("Training finished!", result)


print("\n--- Starting Post-RL Processing for World Model ---")

# --- Phase 1: Aggregate Experience ---
print("\n--- Phase 1: Aggregating Experience ---")
# We now only use the data collected during this RL run.
# The `buffer.sample(0)` command gets all data from the VectorReplayBuffer.
combined_buffer, _ = buffer.sample(0)
print(f"Experience aggregated successfully. Total size: {len(combined_buffer)}")


# --- Phase 2: Extract and Save All Relevant Encoders ---
print("\n--- Phase 2: Extracting Encoders from RL Run ---")

# 1. The Teacher Encoder (from the critic)
final_teacher_encoder = critic1_encoder.teacher_encoder
final_teacher_encoder.eval()
torch.save(final_teacher_encoder.state_dict(), "final_teacher_3d_encoder.pth")
print("Saved final Teacher to final_teacher_3d_encoder.pth")
for p in final_teacher_encoder.parameters():
    p.requires_grad = False
# 2. The Student Encoder BEFORE Fine-Tuning (The "Mixed Signal" Student)
# We need a deepcopy because the `student_to_finetune` object will have its weights changed.
from copy import deepcopy
pre_finetune_student_encoder = deepcopy(shared_history_encoder)
pre_finetune_student_encoder.eval()
torch.save(pre_finetune_student_encoder.state_dict(), "student_encoder_pre_finetuning.pth")
print("Saved Student BEFORE fine-tuning to student_encoder_pre_finetuning.pth")


# --- Phase 3: Post-Hoc Student Fine-Tuning ---
print("\n--- Phase 3: Fine-tuning Student Encoder on Final Teacher ---")
student_to_finetune = shared_history_encoder # This is the object we will modify

# Configuration for fine-tuning
FINETUNE_LR = 1e-4
FINETUNE_EPOCHS = 25 # Reduced slightly as it's often fast
FINETUNE_BATCH_SIZE = 4096

student_optimizer = torch.optim.Adam(student_to_finetune.parameters(), lr=FINETUNE_LR)
distill_loss_fn = nn.MSELoss()

# Create a dataset for fine-tuning
# NOTE: We now use `combined_buffer` which is the same as `rl_data`
res_states_tensor = torch.as_tensor(combined_buffer.obs.res_state, dtype=torch.float32)
history_tensor = torch.as_tensor(combined_buffer.obs.history, dtype=torch.float32)
well_obs_tensor = torch.as_tensor(combined_buffer.obs.well_observations, dtype=torch.float32)
distill_dataset = TensorDataset(res_states_tensor, history_tensor, well_obs_tensor)
distill_dataloader = DataLoader(distill_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
def charbonnier(x, y, eps=1e-6):
    return torch.sqrt((x - y).pow(2) + eps).mean()
print(f"Starting post-hoc distillation for {FINETUNE_EPOCHS} epochs...")
student_to_finetune.train() # Set student to train mode
for epoch in range(FINETUNE_EPOCHS):
    total_loss = 0
    for b_res, b_hist, b_well in distill_dataloader:
        b_res, b_hist, b_well = b_res.to(primary_device), b_hist.to(primary_device), b_well.to(primary_device)
        
        with torch.no_grad():
            z_teacher_target = final_teacher_encoder(b_res)
        z_student_pred = student_to_finetune(b_hist, b_well)
        loss = F.smooth_l1_loss(z_student_pred, z_teacher_target)  # or: charbonnier(z_student_pred, z_teacher_target)
        student_optimizer.zero_grad()
        loss.backward()
        student_optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(distill_dataloader)
    print(f"Epoch {epoch+1}/{FINETUNE_EPOCHS}, Fine-tune Distillation Loss: {avg_loss:.6f}")

# 3. The Student Encoder AFTER Fine-Tuning (The "Best" Student)
post_finetune_student_encoder = student_to_finetune
post_finetune_student_encoder.eval()
torch.save(post_finetune_student_encoder.state_dict(), "student_encoder_post_finetuning.pth")
print("Saved Student AFTER fine-tuning to student_encoder_post_finetuning.pth")


# --- Phase 4: Generate and Save ALL Latent Transition Datasets ---
print("\n--- Phase 4: Generating All Latent Transition Datasets ---")

# Prepare empty lists for each dataset
teacher_transitions = []
student_pre_ft_transitions =[]
student_post_ft_transitions =[]

processing_batch_size = 2048

# Extract raw data once
obs_t = combined_buffer.obs
obs_next_t = combined_buffer.obs_next
act_t = torch.as_tensor(combined_buffer.act, device=primary_device, dtype=torch.float32)
rew_t = torch.as_tensor(combined_buffer.rew, device=primary_device, dtype=torch.float32).unsqueeze(1)
done_t = torch.as_tensor(combined_buffer.done, device=primary_device, dtype=torch.float32).unsqueeze(1)
discount_t = (1.0 - done_t) * 0.98  # gamma = 0.98 above
print(f"Encoding and packaging {len(combined_buffer)} transitions...")
with torch.no_grad():
    for i in range(0, len(combined_buffer), processing_batch_size):
        # Batch raw observations
        b_obs = obs_t[i:i + processing_batch_size]
        b_obs_next = obs_next_t[i:i + processing_batch_size]
        b_act = act_t[i:i + processing_batch_size]
        b_rew = rew_t[i:i + processing_batch_size]

        # Move raw obs data to GPU for this batch
        b_res_state = torch.as_tensor(b_obs.res_state, device=primary_device, dtype=torch.float32)
        b_history = torch.as_tensor(b_obs.history, device=primary_device, dtype=torch.float32)
        b_well_obs = torch.as_tensor(b_obs.well_observations, device=primary_device, dtype=torch.float32)
        
        b_next_res_state = torch.as_tensor(b_obs_next.res_state, device=primary_device, dtype=torch.float32)
        b_next_history = torch.as_tensor(b_obs_next.history, device=primary_device, dtype=torch.float32)
        b_next_well_obs = torch.as_tensor(b_obs_next.well_observations, device=primary_device, dtype=torch.float32)

        # --- Encode with ALL THREE models ---
        z_teacher_t = final_teacher_encoder(b_res_state)
        z_teacher_next_t = final_teacher_encoder(b_next_res_state)

        z_student_pre_ft_t = pre_finetune_student_encoder(b_history, b_well_obs)
        z_student_pre_ft_next_t = pre_finetune_student_encoder(b_next_history, b_next_well_obs)

        z_student_post_ft_t = post_finetune_student_encoder(b_history, b_well_obs)
        z_student_post_ft_next_t = post_finetune_student_encoder(b_next_history, b_next_well_obs)
        
        # --- Package and save all three datasets ---
        # This loop is inefficient but clear. Can be optimized if slow.
        for j in range(len(b_res_state)):
            teacher_transitions.append(
                (z_teacher_t[j].cpu().clone(), z_teacher_next_t[j].cpu().clone(), b_act[j].cpu().clone(), b_rew[j].cpu().clone(),discount_t[i + j].cpu().clone())
            )
            student_pre_ft_transitions.append(
                (z_student_pre_ft_t[j].cpu().clone(), z_student_pre_ft_next_t[j].cpu().clone(), b_act[j].cpu().clone(), b_rew[j].cpu().clone(),discount_t[i + j].cpu().clone())
            )
            student_post_ft_transitions.append(
                (z_student_post_ft_t[j].cpu().clone(), z_student_post_ft_next_t[j].cpu().clone(), b_act[j].cpu().clone(), b_rew[j].cpu().clone(),discount_t[i + j].cpu().clone())
            )

# Save the final lists to disk
with open("teacher_wm_transitions.pkl", 'wb') as f:
    pickle.dump(teacher_transitions, f)
print(f"Saved {len(teacher_transitions)} TEACHER transitions.")

with open("student_pre_ft_wm_transitions.pkl", 'wb') as f:
    pickle.dump(student_pre_ft_transitions, f)
print(f"Saved {len(student_pre_ft_transitions)} STUDENT (PRE-FT) transitions.")

with open("student_post_ft_wm_transitions.pkl", 'wb') as f:
    pickle.dump(student_post_ft_transitions, f)
print(f"Saved {len(student_post_ft_transitions)} STUDENT (POST-FT) transitions.")

print("\nPost-RL processing complete.")




# --- Decoders ---
teacher_decoder = Transposed3DCNN(latent_dim=d_model, out_channels=2).to(primary_device)
teacher_decoder_optim = torch.optim.Adam(teacher_decoder.parameters(), lr=1e-4)

student_decoder = Transposed3DCNN(latent_dim=d_model, out_channels=2).to(primary_device)
# Warm-start student from teacher for faster/better convergence
student_decoder.load_state_dict(teacher_decoder.state_dict(), strict=False)
student_decoder_optim = torch.optim.Adam(student_decoder.parameters(), lr=1e-4)

# --- Regularizers / helpers ---
def tv3d(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    return dx.abs().mean() + dy.abs().mean() + dz.abs().mean()

LAMBDA_TV  = 1e-5   # total-variation weight
LAMBDA_PER = 1e-3   # latent-perceptual (via frozen teacher encoder)

# --- Decoder dataset (same tensors you already have in memory) ---
decoder_dataset = TensorDataset(
    torch.as_tensor(combined_buffer.obs.res_state, dtype=torch.float32),
    torch.as_tensor(combined_buffer.obs.history, dtype=torch.float32),
    torch.as_tensor(combined_buffer.obs.well_observations, dtype=torch.float32)
)
decoder_dataloader = DataLoader(
    decoder_dataset,
    batch_size=FINETUNE_BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# --- 3. The Decoder Training Loop ---
DECODER_EPOCHS = 50
print(f"Starting decoder training for {DECODER_EPOCHS} epochs...")
for epoch in range(DECODER_EPOCHS):
    total_teacher_loss = 0.0
    total_student_loss = 0.0

    for b_res, b_hist, b_well in decoder_dataloader:
        b_res  = b_res.to(primary_device)
        b_hist = b_hist.to(primary_device)
        b_well = b_well.to(primary_device)

        # --------- Teacher decoder (supervised) ---------
        teacher_decoder.train()
        with torch.no_grad():
            z_teacher = final_teacher_encoder(b_res)  # frozen teacher enc
        recon_teacher = teacher_decoder(z_teacher)
        loss_teacher = F.smooth_l1_loss(recon_teacher, b_res) + LAMBDA_TV * tv3d(recon_teacher)

        teacher_decoder_optim.zero_grad()
        loss_teacher.backward()
        teacher_decoder_optim.step()
        total_teacher_loss += loss_teacher.item()

        # --------- Student decoder (supervised + latent-perceptual) ---------
        student_decoder.train()
        with torch.no_grad():
            z_student = post_finetune_student_encoder(b_hist, b_well)

        recon_student = student_decoder(z_student)
        # base loss + TV
        loss_student = F.smooth_l1_loss(recon_student, b_res) + LAMBDA_TV * tv3d(recon_student)

        # latent-perceptual term: compare teacher-encoder activations
        # IMPORTANT: do NOT use no_grad here; teacher params are frozen so
        # gradients flow to recon_student but weights won't update.
        z_res_recon = final_teacher_encoder(recon_student)
        with torch.no_grad():
            z_res_target = final_teacher_encoder(b_res)

        loss_student = loss_student + LAMBDA_PER * F.smooth_l1_loss(z_res_recon, z_res_target)

        student_decoder_optim.zero_grad()
        loss_student.backward()
        student_decoder_optim.step()
        total_student_loss += loss_student.item()

    avg_teacher_loss = total_teacher_loss / len(decoder_dataloader)
    avg_student_loss = total_student_loss / len(decoder_dataloader)
    print(f"Epoch {epoch+1}/{DECODER_EPOCHS} -> "
          f"Teacher Recon+TV: {avg_teacher_loss:.6f} | "
          f"Student Recon+TV+Per: {avg_student_loss:.6f}")

# --- 4. Save the Final Trained Decoders ---
teacher_decoder.eval()
student_decoder.eval()
torch.save(teacher_decoder.state_dict(), "teacher_decoder.pth")
torch.save(student_decoder.state_dict(), "student_decoder.pth")
print("\nSaved final trained decoders to teacher_decoder.pth and student_decoder.pth")

print("\nFull pipeline complete. All artifacts generated.")