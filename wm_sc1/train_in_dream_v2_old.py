# train_in_dream_v2.py
import os
import argparse
import pickle
from typing import List, Optional

import numpy as np
import torch
from torch import nn

import gymnasium as gym
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import BasePolicy
from torch.utils.tensorboard import SummaryWriter

from world_models import (
    EnsembleWorldModel, ProbabilisticGRU,RSSMWorldModel, KoopmanWorldModel
)
from env_3 import ReservoirEnv
from nets import HistoryEncoder, PolicyHead

class Scenario1FailureMask:
    """
    Simple mask for scenario 1: one or more wells are known to be dead.
    We keep the action space 11D, but override those dimensions before:
      - sending actions to WM (imagination + wm_update)
      - sending actions to real env (optional; you can also enforce in env)
    """

    def __init__(self, dead_action_indices, dead_action_value=-1.0):
        """
        dead_action_indices: list of int indices in [0, 10] to treat as dead.
        dead_action_value: normalized action value in [-1,1] to enforce.
            - For example, -1.0 -> maps to action_low (minimum rate).
            - You can adjust to match your "dead" convention.
        """
        self.dead_action_indices = list(dead_action_indices)
        self.dead_action_value = float(dead_action_value)

    def mask(self, action: torch.Tensor) -> torch.Tensor:
        """
        action: tensor [..., 11] in [-1, 1].
        Returns a *new* tensor with dead dims overwritten.
        """
        if not self.dead_action_indices:
            return action

        a = action.clone()
        # assume a shape [B, A] or [*, A]
        idx = torch.as_tensor(self.dead_action_indices, device=a.device, dtype=torch.long)
        # broadcast dead value over the last dimension's selected indices
        a[..., idx] = self.dead_action_value
        return a
def log_collect_stats(writer, prefix: str, stat, step: int, base_env_id: int | None = None):
    """prefix: 'eval' or 'real'. Works with CollectStats or dict-like fallback."""
    # Number of steps / episodes
    n_steps = getattr(stat, "n_collected_steps", 0)
    n_eps   = getattr(stat, "n_collected_episodes", 0)
    writer.add_scalar(f"{prefix}/n_steps", float(n_steps), step)
    writer.add_scalar(f"{prefix}/n_episodes", float(n_eps), step)

    # Episode lengths
    if getattr(stat, "lens_stat", None) is not None:
        writer.add_scalar(f"{prefix}/len_mean", float(stat.lens_stat.mean), step)
        writer.add_scalar(f"{prefix}/len_min",  float(stat.lens_stat.min),  step)
        writer.add_scalar(f"{prefix}/len_max",  float(stat.lens_stat.max),  step)

    # Episode returns
    if getattr(stat, "returns_stat", None) is not None:
        rs = stat.returns_stat
        writer.add_scalar(f"{prefix}/returns_mean", float(rs.mean), step)
        writer.add_scalar(f"{prefix}/returns_std",  float(rs.std),  step)
        writer.add_scalar(f"{prefix}/returns_min",  float(rs.min),  step)
        writer.add_scalar(f"{prefix}/returns_max",  float(rs.max),  step)
        # histogram & per-env (useful for your fixed test IDs)
        if hasattr(stat, "returns"):
            writer.add_histogram(f"{prefix}/returns_hist", np.asarray(stat.returns), step)
            if base_env_id is not None:
                for i, R in enumerate(np.asarray(stat.returns).flatten()):
                    writer.add_scalar(f"{prefix}/env_{base_env_id + 10*i}_return", float(R), step)
    else:
        # Older dict fallback (kept for safety)
        r_mean = float(getattr(stat, "rew", 0.0) or getattr(stat, "rew_mean", 0.0))
        l_mean = float(getattr(stat, "len", 0.0) or getattr(stat, "len_mean", 0.0))
        writer.add_scalar(f"{prefix}/returns_mean", r_mean, step)
        writer.add_scalar(f"{prefix}/len_mean", l_mean, step)
# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def load_transitions_for_initial_states(pkl_path: str):
    with open(pkl_path, "rb") as f:
        transitions = pickle.load(f)
    first = transitions[0]
    if len(first) not in (4, 5):
        raise ValueError(f"Unexpected transition tuple length: {len(first)}")
    z_t = [t[0] for t in transitions]
    z_t = [torch.as_tensor(zt).view(-1).detach().clone() for zt in z_t]
    return z_t


def _read_collect_stats(stats, key_names, default=0.0):
    """Robustly extract a scalar from Tianshou CollectStats or dict.

    key_names: e.g. ["rew", "rew_mean"] or ["len", "len_mean"] or ["n/st","n_step"].
    Falls back to averaging 'rews' / 'lens' arrays if present.
    """
    import numpy as np

    # helper: try an attribute or dict key by several names
    def _try(obj, names):
        for k in names:
            # allow "n/st" style by mapping to attr "n_st"
            attr = k.replace("/", "_")
            if isinstance(obj, dict) and k in obj and np.isscalar(obj[k]):
                return float(obj[k])
            if hasattr(obj, attr):
                v = getattr(obj, attr)
                if np.isscalar(v):
                    return float(v)
            if hasattr(obj, k):
                v = getattr(obj, k)
                if np.isscalar(v):
                    return float(v)
        return None

    # 1) direct scalars
    val = _try(stats, key_names)
    if val is not None:
        return val

    # 2) nested .stat (some versions)
    if hasattr(stats, "stat") and isinstance(stats.stat, dict):
        val = _try(stats.stat, key_names)
        if val is not None:
            return val

    # 3) arrays fallback: 'rews' / 'lens'
    want_rew = any(k.startswith("rew") for k in key_names)
    want_len = any(k.startswith("len") for k in key_names)

    # object-like
    if not isinstance(stats, dict):
        if want_rew and hasattr(stats, "rews"):
            arr = getattr(stats, "rews")
            if isinstance(arr, (list, np.ndarray)) and len(arr):
                return float(np.mean(arr))
        if want_len and hasattr(stats, "lens"):
            arr = getattr(stats, "lens")
            if isinstance(arr, (list, np.ndarray)) and len(arr):
                return float(np.mean(arr))

    # dict-like
    if isinstance(stats, dict):
        if want_rew and "rews" in stats:
            arr = stats["rews"]
            if isinstance(arr, (list, np.ndarray)) and len(arr):
                return float(np.mean(arr))
        if want_len and "lens" in stats:
            arr = stats["lens"]
            if isinstance(arr, (list, np.ndarray)) and len(arr):
                return float(np.mean(arr))

    return float(default)


class LatentGaussianActor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
        )
        self.mu = nn.Linear(h2, action_dim)
        self.log_std = nn.Linear(h2, action_dim)
        nn.init.constant_(self.log_std.bias, -1.5)
    def forward(self, obs):
        if not torch.is_tensor(obs): obs = torch.as_tensor(obs, dtype=torch.float32)
        dev = self.mu.weight.device
        obs = obs.to(dev, non_blocking=True)
        x = self.net(self.norm(obs))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -5.0, 0.0)
        std = torch.exp(log_std)
        return mu, std  # pre-tanh
    
def warmstart_dreamer_actor_from_policy(
    actor: LatentGaussianActor,
    policy_ckpt: str,
    latent_dim: int,
    action_dim: int,
    device: torch.device,
):
    """
    Load the model-free PolicyHead weights from policy.pth and
    transplant them into the Dreamer LatentGaussianActor.

    Assumes policy.pth was saved from DistillationSACPolicy with
    actor = ActorWrapperDP(TianshouSACActor(shared HistoryEncoder, PolicyHead)).
    """
    if not os.path.exists(policy_ckpt):
        print(f"[v2] policy_ckpt '{policy_ckpt}' not found; skipping warm-start.")
        return

    print(f"[v2] loading model-free policy from: {policy_ckpt}")
    full_sd = torch.load(policy_ckpt, map_location=device)

    # Build a dummy PolicyHead with the correct shape
    temp_head = PolicyHead(d_model=latent_dim, action_shape=(action_dim,)).to(device)
    head_sd = temp_head.state_dict()

    # Extract just the actor's PolicyHead params from the big policy state_dict.
    # We expect keys like: "actor.dp_module.module.policy_head.net.0.weight", etc.
    mapped = {}
    for k, v in full_sd.items():
        if "actor" not in k or "policy_head" not in k:
            continue
        try:
            subkey = k.split("policy_head.", 1)[1]  # e.g. "net.0.weight"
        except IndexError:
            continue
        if subkey in head_sd and head_sd[subkey].shape == v.shape:
            mapped[subkey] = v

    if not mapped:
        print("[v2] no matching PolicyHead parameters found in policy_ckpt; warm-start skipped.")
        return

    print(f"[v2] loaded {len(mapped)}/{len(head_sd)} PolicyHead params from policy_ckpt.")
    head_sd.update(mapped)
    temp_head.load_state_dict(head_sd, strict=False)

    # Now copy temp_head → Dreamer actor.
    # PolicyHead.net = [LayerNorm, Linear, ReLU, Linear, ReLU]
    # Dreamer actor: norm + net = [Linear, ReLU, Linear, ReLU]

    with torch.no_grad():
        # LayerNorm
        actor.norm.weight.copy_(temp_head.net[0].weight)
        actor.norm.bias.copy_(temp_head.net[0].bias)

        # First Linear
        actor.net[0].weight.copy_(temp_head.net[1].weight)
        actor.net[0].bias.copy_(temp_head.net[1].bias)

        # Second Linear
        actor.net[2].weight.copy_(temp_head.net[3].weight)
        actor.net[2].bias.copy_(temp_head.net[3].bias)

        # Mean head
        actor.mu.weight.copy_(temp_head.fc_mu.weight)
        actor.mu.bias.copy_(temp_head.fc_mu.bias)

    print("[v2] Dreamer actor warm-started from model-free PolicyHead (mu path only).")

class LatentActorPolicy(BasePolicy):
    """Takes latent z directly (from RealToLatentEnv) and outputs actions."""
    def __init__(self, actor: LatentGaussianActor, device: torch.device,
                 action_dim: int, stochastic: bool = True,
                  failure_mask: Scenario1FailureMask | None = None):
        super().__init__(
            action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
            action_scaling=False,
            action_bound_method="clip",
        )
        self.actor = actor
        self.device = device
        self.stochastic = stochastic
        self.failure_mask = failure_mask

    @torch.no_grad()
    def forward(self, batch: Batch, state=None, **kwargs):
        z = torch.as_tensor(batch.obs, device=self.device, dtype=torch.float32)
        mu, std = self.actor(z)
        if self.stochastic:
            act = torch.tanh(mu + std * torch.randn_like(std))
        else:
            act = torch.tanh(mu)
        
        if self.failure_mask is not None:
            act = self.failure_mask.mask(act)

        return Batch(act=act.cpu().numpy(), state=state)

    # collection-only
    def learn(self, batch, *args, **kwargs):
        raise RuntimeError("LatentActorPolicy is collection-only; learn() should not be called.")

class ClippedAdam(torch.optim.Adam):
    def __init__(self, params, *args, clip_norm: float | None = None, **kwargs):
        super().__init__(params, *args, **kwargs)
        self._clip_norm = clip_norm
    def step(self, closure=None):
        if self._clip_norm is not None and self._clip_norm > 0:
            for group in self.param_groups:
                params = [p for p in group["params"] if p.grad is not None]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, self._clip_norm)
        return super().step(closure)


# --------------------------------------------------------------------------------------
# Networks: Actor & Value (latent)
# --------------------------------------------------------------------------------------



class LatentValue(nn.Module):
    def __init__(self, latent_dim: int, hidden=(256, 256)):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 1),
        )
    def forward(self, z):
        if not torch.is_tensor(z): z = torch.as_tensor(z, dtype=torch.float32)
        z = z.to(self.net[0].weight.device, non_blocking=True)
        return self.net(self.norm(z)).squeeze(-1)


# --------------------------------------------------------------------------------------
# Real -> Latent wrapper (subprocess)
# --------------------------------------------------------------------------------------
class RealToLatentEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, encoder: nn.Module, device: str | torch.device = "cpu"):
        super().__init__(env)
        self.encoder = encoder.eval()
        for p in self.encoder.parameters(): p.requires_grad_(False)
        self.device = torch.device(device)
        obs0, _ = env.reset()
        z0 = self._encode(obs0)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(z0.numel(),), dtype=np.float32
        )
    @torch.no_grad()
    def _encode(self, obs_dict):
        hist = torch.as_tensor(obs_dict["history"], device=self.device, dtype=torch.float32)
        well = torch.as_tensor(obs_dict["well_observations"], device=self.device, dtype=torch.float32)
        if hist.dim() == 3: hist = hist.unsqueeze(0)
        if well.dim() == 2: well = well.unsqueeze(0)
        z = self.encoder(hist, well).squeeze(0)
        return z
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        z = self._encode(obs)
        return z.cpu().numpy().astype(np.float32), info
    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        z = self._encode(obs)
        return z.cpu().numpy().astype(np.float32), rew, term, trunc, info


# --------------------------------------------------------------------------------------
# Build world model ensemble
# --------------------------------------------------------------------------------------
def build_world_model(model_type: str, latent_dim: int, action_dim: int, num_ensemble: int, device):
    models = []
    if model_type in ("gru", "geometric_gru"):
        for _ in range(num_ensemble):
            models.append(ProbabilisticGRU(latent_dim, action_dim).to(device))
    elif model_type == "rssm":
        for _ in range(num_ensemble):
            models.append(RSSMWorldModel(latent_dim, action_dim).to(device))
    elif model_type == "koopman":
        for _ in range(num_ensemble):
            models.append(KoopmanWorldModel(latent_dim, action_dim).to(device))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return EnsembleWorldModel(models)


# --------------------------------------------------------------------------------------
# Schedule + fear (same shaping you used in DreamEnv)
# --------------------------------------------------------------------------------------
def _apply_schedule_and_fear(raw_model: torch.Tensor, t: int) -> torch.Tensor:
    """raw_model: [B] tensor, differentiable. Returns [B] tensor with schedule+fear shaping."""
    device = raw_model.device
    if t <= 2:
        cap = torch.tensor(2.2, device=device)
        fear = raw_model > 2.2
    elif t <= 10:
        cap = torch.tensor(1.8, device=device)
        fear = raw_model >= 1.9
    elif t <= 13:
        cap = torch.tensor(1.3, device=device)
        fear = raw_model >= 1.4
    else:
        cap = torch.tensor(1.0, device=device)
        fear = raw_model >= 1.2

    # cap
    raw_sched = torch.minimum(raw_model, cap)

    # fear rule: zero out where fear is true
    raw_sched = torch.where(fear, torch.zeros_like(raw_sched), raw_sched)

    # early-steps floor
    if t <= 13:
        raw_sched = torch.where(raw_sched < 0.7, torch.zeros_like(raw_sched), raw_sched)

    return raw_sched


# --------------------------------------------------------------------------------------
# Imagination (Dreamer-lite; stop-grad through dynamics)
# --------------------------------------------------------------------------------------
def _one_step_wm(m, z, a, h, deterministic: bool,
                 diffusion_steps: int | None = None,
                 use_ema: bool = True):
    """
    Single differentiable WM step used inside Dreamer imagination.

    - GRU / geometric_GRU: use .forward → mean dynamics (no sampling).
    - RSSM: use .sample (deterministic prior mean, no @no_grad).
    - Koopman: call _predict_next + r_head (avoid .sample which is no_grad).
    - Diffusion: fall back to .sample; not used in scenario 0.
    """
    # Recurrent models (GRU, geometric_GRU, RSSM)
    if h is not None:
        # Probabilistic GRU (also used for geometric_gru)
        if isinstance(m, ProbabilisticGRU):
            mean, _std, r, h_next = m.forward(z, a, h)
            z_next = mean
            return z_next, r, h_next

        # RSSM: sample() is differentiable and already uses prior mean (no noise)
        if isinstance(m, RSSMWorldModel):
            z_next, r, h_next = m.sample(z, a, h)
            return z_next, r, h_next

        # Fallback: if some other recurrent model appears later
        z_next, r, h_next = m.sample(z, a, h)
        return z_next, r, h_next

    # Stateless models (Koopman, diffusion, etc.)
    # Koopman: use analytic prediction, avoid .sample (which is @torch.no_grad)
    if isinstance(m, KoopmanWorldModel):
        z_pred = m._predict_next(z, a)
        r_pred = m.r_head(torch.cat([z_pred, a], dim=-1))
        return z_pred, r_pred, None

    # Diffusion / other models with num_timesteps
    if hasattr(m, "num_timesteps"):
        steps = 20 if diffusion_steps is None else diffusion_steps
        z_next, r, _ = m.sample(z, a, h=None, steps=steps, use_ema=use_ema, ddim=True)
        return z_next, r, None

    # Generic stateless fallback
    z_next, r, _ = m.sample(z, a, h=None)
    return z_next, r, None


def imagine_ahead(actor, world_model, z0, H, device,
                  disagreement_coef: float = 0.0,
                  disagreement_on: str = "reward",
                  use_mean_dynamics: bool = True,
                  gamma: float = 0.99,
                  diffusion_steps: int | None = None,
                  use_ema: bool = True,
                  failure_mask: Scenario1FailureMask | None = None):
    """
    Build imagined rollouts with gradients flowing to the actor through WM.
    """
    B, D = z0.shape
    z = z0.to(device)

    # hidden?
    m0 = world_model.models[0]
    has_hidden = any([hasattr(m0, "hidden_dim"), hasattr(m0, "hidden"), hasattr(m0, "deter_dim")])
    h = None
    if has_hidden:
        Hdim = getattr(m0, "hidden_dim", getattr(m0, "hidden", getattr(m0, "deter_dim", None)))
        h = torch.zeros(B, Hdim, device=device)

    z_seq, a_seq, r_seq, disc_seq = [], [], [], []
    midx = torch.randint(low=0, high=len(world_model.models), size=(B,), device=device)

    for t in range(1, H + 1):
        # actor sample (reparameterized)
        mu, std = actor(z)
        eps = torch.randn_like(std)
        a = torch.tanh(mu + std * eps)
        if failure_mask is not None:
            a = failure_mask.mask(a)
        a_seq.append(a)

        # step each item with its chosen member for the main rollout
        z_in = z
        r_list, z_next_list, h_next_list = [], [], []
        for i in range(B):
            m = world_model.models[int(midx[i])]
            zi = z_in[i:i+1]
            ai = a[i:i+1]
            hi = None if h is None else h[i:i+1]
            zi1, ri, hi1 = _one_step_wm(m, zi, ai, hi, deterministic=use_mean_dynamics,
                                        diffusion_steps=diffusion_steps, use_ema=use_ema)
            z_next_list.append(zi1.squeeze(0))
            r_list.append(ri.squeeze(0))
            if h is not None:
                h_next_list.append(hi1.squeeze(0))

        z_next = torch.stack(z_next_list, 0)           # [B,D]
        raw_model = torch.stack(r_list, 0).squeeze(-1) # [B]

        # disagreement penalty across ensemble (deterministic forward for stability)
        dis_val = torch.zeros(B, device=device)
        if disagreement_coef > 0:
            pred_R, pred_Z = [], []
            for m in world_model.models:
                zi1_all, ri_all, _ = _one_step_wm(m, z_in, a, h, deterministic=True,
                                                  diffusion_steps=diffusion_steps, use_ema=use_ema)
                pred_R.append(ri_all.squeeze(-1))   # [B]
                pred_Z.append(zi1_all)              # [B,D]
            if disagreement_on == "latent":
                Z = torch.stack(pred_Z, 0)          # [E,B,D]
                dis_val = Z.std(dim=0).mean(dim=-1) # [B] mean over D
            else:  # 'reward'
                R = torch.stack(pred_R, 0)          # [E,B]
                dis_val = R.std(dim=0).abs()        # [B]

        shaped = _apply_schedule_and_fear(raw_model, t)
        r = shaped - disagreement_coef * dis_val

        z_seq.append(z_next)
        r_seq.append(r)
        disc_seq.append(torch.full((B,), gamma, device=device))

        # Do NOT detach — multi-step credit assignment to actor
        z = z_next
        if h is not None:
            h = torch.stack(h_next_list, 0)

    z_seq    = torch.stack(z_seq,    0)  # [T,B,D]
    a_seq    = torch.stack(a_seq,    0)  # [T,B,A]
    r_seq    = torch.stack(r_seq,    0)  # [T,B]
    disc_seq = torch.stack(disc_seq, 0)  # [T,B]
    return dict(z=z_seq, a=a_seq, r=r_seq, disc=disc_seq)



def lambda_returns(r, v, disc, bootstrap, lam=0.95):
    T, B = r.shape
    g = torch.zeros_like(r)
    next_g = bootstrap  # [B]
    for t in reversed(range(T)):
        td = r[t] + disc[t] * next_g - v[t]
        next_g = v[t] + lam * disc[t] * td
        g[t] = next_g
    return g


def dreamer_update(actor, value, actor_optim, value_optim,
                   world_model, start_z, H, gamma=0.99, lam=0.95,
                   disagreement_coef=0.0, device="cuda",
                   failure_mask: Scenario1FailureMask | None = None):
    # Freeze WM params so grads flow through it to the actor but WM weights aren't updated here.
    wm_flags = [p.requires_grad for p in world_model.parameters()]
    for p in world_model.parameters():
        p.requires_grad_(False)

    # Build imagination WITH gradients.
    traj = imagine_ahead(actor, world_model, start_z, H, device,
                         disagreement_coef=disagreement_coef,
                         use_mean_dynamics=True, gamma=gamma,failure_mask=failure_mask)

    # Restore WM flags.
    for p, f in zip(world_model.parameters(), wm_flags):
        p.requires_grad_(f)

    z_seq, r_seq, disc_seq = traj["z"], traj["r"], traj["disc"]  # z:[T,B,D], r:[T,B], disc:[T,B]
    T, B, D = z_seq.shape

    # ----- Critic update (NO grad to actor/WM) -----
    # Stop grad through imagination for the critic.
    z_flat_det = z_seq.detach().reshape(T * B, D)
    v_pred = value(z_flat_det).view(T, B)                 # critic depends only on its own params
    v_boot = value(z_seq[-1].detach())                    # bootstrap on detached last latent

    # λ-returns target for critic: stop-grad everywhere.
    with torch.no_grad():
        g_target = lambda_returns(r_seq.detach(), v_pred.detach(), disc_seq, v_boot.detach(), lam=lam)

    value_loss = torch.mean((v_pred - g_target) ** 2)
    value_optim.zero_grad(set_to_none=True)
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
    value_optim.step()

    # ----- Actor update (GRAD through r_seq/z_seq → actions → actor) -----
    # Baseline uses critic predictions but is stop-grad wrt actor.
    g_actor = lambda_returns(r_seq, v_pred.detach(), disc_seq, v_boot.detach(), lam=lam)
    actor_loss = -torch.mean(g_actor)

    actor_optim.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optim.step()

    return {
        "actor_loss": float(actor_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "r_mean": float(r_seq.mean().item()),
    }


# --------------------------------------------------------------------------------------
# Tianshou policy for real-env collection / eval
# --------------------------------------------------------------------------------------
class RealActorPolicy(BasePolicy):
    """Real env dict obs -> encoder -> actor -> action (collection-only)."""
    def __init__(self, encoder: nn.Module, actor: LatentGaussianActor,
                 device: torch.device, action_dim: int, stochastic: bool = True,
                 failure_mask: Scenario1FailureMask | None = None):
        super().__init__(
            action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
            action_scaling=False,           # env already expects [-1, 1]
            action_bound_method="clip",     # safety clip in [-1, 1]
        )
        self.encoder = encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.actor = actor
        self.device = device
        self.stochastic = stochastic
        self.failure_mask = failure_mask

    @torch.no_grad()
    def forward(self, batch: Batch, state=None, **kwargs):
        obs = batch.obs
        hist = torch.as_tensor(obs["history"], device=self.device, dtype=torch.float32)
        well = torch.as_tensor(obs["well_observations"], device=self.device, dtype=torch.float32)
        if hist.dim() == 3:
            hist = hist.unsqueeze(0)
        if well.dim() == 2:
            well = well.unsqueeze(0)
        z = self.encoder(hist, well)  # [B, D]
        mu, std = self.actor(z)
        if self.stochastic:
            a = torch.tanh(mu + std * torch.randn_like(std))
        else:
            a = torch.tanh(mu)

        if self.failure_mask is not None:
            a = self.failure_mask.mask(a)
        # IMPORTANT: return a single Batch with both act and state
        return Batch(act=a.cpu().numpy(), state=state)

    # We don't train this policy; satisfy abstract method.
    def learn(self, batch, *args, **kwargs):
        # If ever called by mistake, make it obvious.
        raise RuntimeError("RealActorPolicy is collection-only; learn() should not be called.")

# --------------------------------------------------------------------------------------
# WM update (real-only)
# --------------------------------------------------------------------------------------
def wm_update(world_model, wm_buffer, device, args, latent_dim, decoder_path=None,
              failure_mask: Scenario1FailureMask | None = None):
    if len(wm_buffer) < max(8, args.wm_batch_size // 4):
        return 0.0

    world_model.train()
    m0 = world_model.models[0]
    is_recurrent = any([hasattr(m0, "hidden_dim"), hasattr(m0, "hidden"), hasattr(m0, "deter_dim")])

    decoder = None
    use_geom = (args.model_type == "geometric_gru") and (decoder_path is not None)
    if use_geom:
        from nets import Transposed3DCNN
        decoder = Transposed3DCNN(latent_dim=latent_dim, out_channels=2).to(device).eval()
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    opt = torch.optim.Adam(world_model.parameters(), lr=args.wm_lr, weight_decay=args.wm_wd)

    total = 0.0
    for _ in range(args.wm_updates_per_epoch):
        batch, _ = wm_buffer.sample(min(args.wm_batch_size, len(wm_buffer)))
        z_t    = torch.as_tensor(batch.obs,      device=device, dtype=torch.float32)
        a_t    = torch.as_tensor(batch.act,      device=device, dtype=torch.float32)
        r_t    = torch.as_tensor(batch.rew,      device=device, dtype=torch.float32).unsqueeze(-1)
        z_next = torch.as_tensor(batch.obs_next, device=device, dtype=torch.float32)
        if failure_mask is not None:
            a_t = failure_mask.mask(a_t)
        h = None
        if is_recurrent:
            H = getattr(m0, "hidden_dim", getattr(m0, "hidden", getattr(m0, "deter_dim", None)))
            h = torch.zeros(z_t.size(0), H, device=device)

        if use_geom:
            loss = world_model.calculate_geometric_loss(z_next, z_t, a_t, r_t, h, decoder)
        else:
            loss = world_model.calculate_loss(z_next, z_t, a_t, r_t, h)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), args.wm_grad_clip)
        opt.step()
        total += float(loss.detach().item())

    world_model.eval()
    return total / max(1, args.wm_updates_per_epoch)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def sample_start_latents_from_wm_buffer(wm_buffer, n, device, initial_states):
    if len(wm_buffer) == 0:
        idx = np.random.randint(0, len(initial_states), size=n)
        z0 = torch.stack([initial_states[i] for i in idx], 0).to(device)
        return z0
    batch, _ = wm_buffer.sample(n)
    z0 = torch.as_tensor(batch.obs, device=device, dtype=torch.float32)
    return z0


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[v2] device: {device}")

    # ---- load initial latents for imagination seeding ----
    tag2pkl = {
        "student_post_ft": "student_post_ft_wm_transitions.pkl",
        "student_pre_ft": "student_pre_ft_wm_transitions.pkl",
        "teacher": "teacher_wm_transitions.pkl",
    }
    data_path = args.data_path or tag2pkl.get(args.latent_tag, "student_post_ft_wm_transitions.pkl")
    print(f"[v2] loading initial z_t from: {data_path}")
    initial_states = load_transitions_for_initial_states(data_path)
    latent_dim = initial_states[0].numel()
    action_dim = args.action_dim
    # ----------------------------------------------------------------------------------
    # Scenario 1: define which action indices are dead (well failure)
    # ----------------------------------------------------------------------------------
    # Example: suppose producer P3 is dead.
    # If your action ordering is: [P1..P8 (0..7), I1..I3 (8..10)],
    # then P3 corresponds to index 2.
    # TODO: update this list to match your actual failed wells.
    failure_mask = Scenario1FailureMask(dead_action_indices=[2], dead_action_value=-1.0)


    print(f"[v2] latent_dim={latent_dim}, action_dim={action_dim}, num_init={len(initial_states)}")

    # ---- world model ----
    world_model = build_world_model(args.model_type, latent_dim, action_dim, args.num_ensemble, device)

    if args.model_ckpt is not None and os.path.exists(args.model_ckpt):
        print(f"[v2] loading WM weights from: {args.model_ckpt}")
        sd = torch.load(args.model_ckpt, map_location=device)
        world_model.load_state_dict(sd, strict=False)
    else:
        print("[v2] no WM checkpoint found/provided; starting world model from random init.")

    world_model.eval()

    # ---- actor/value ----
    actor = LatentGaussianActor(latent_dim, action_dim, hidden=(256, 256)).to(device)
    value = LatentValue(latent_dim, hidden=(256, 256)).to(device)
    actor_optim = ClippedAdam(actor.parameters(), lr=args.lr, clip_norm=1.0)
    value_optim = ClippedAdam(value.parameters(), lr=args.lr, clip_norm=1.0)

    if args.policy_ckpt:
        try:
            warmstart_dreamer_actor_from_policy(
                actor=actor,
                policy_ckpt=args.policy_ckpt,
                latent_dim=latent_dim,
                action_dim=action_dim,
                device=device,
            )
        except Exception as e:
            print(f"[v2] WARNING: warm-start from '{args.policy_ckpt}' failed: {e}")
    else:
        print("[v2] no policy_ckpt provided; actor uses random init.")

    # ---- encoder for real collection/eval ----
    enc_path = args.encoder_ckpt or "student_encoder_post_finetuning.pth"
    print(f"[v2] loading student encoder from: {enc_path}")
    student_encoder = HistoryEncoder(d_model=latent_dim).to(device)
    student_encoder.load_state_dict(torch.load(enc_path, map_location=device))
    student_encoder.eval()

    # ---- real envs for WM data (TRAIN ids only; never TRUE decks) ----
    def make_real_latent(env_id):
        def _thunk():
            enc = HistoryEncoder(d_model=latent_dim).to("cpu")
            enc.load_state_dict(torch.load(enc_path, map_location="cpu"))
            enc.eval()
            base = ReservoirEnv(env_id=env_id)
            return RealToLatentEnv(base, enc, device="cpu")
        return _thunk

    train_ids = [args.train_env_base + i for i in range(args.num_real_train_envs)]
    real_envs = SubprocVectorEnv([make_real_latent(eid) for eid in train_ids])
    wm_buffer = VectorReplayBuffer(args.wm_buffer_size, len(real_envs))

    # stochastic policy for data collection
    data_policy = LatentActorPolicy(actor, device, action_dim=action_dim, stochastic=True,
                                   failure_mask=failure_mask)
    real_collector = Collector(data_policy, real_envs, wm_buffer, exploration_noise=False)

    # ---- real envs for evaluation (151,161,171,181,191,201 if num_test_envs=6) ----
    def make_real_eval(i):
        return ReservoirEnv(env_id=args.test_env_id + 10 * i)
    test_envs = SubprocVectorEnv([lambda i=i: make_real_eval(i) for i in range(args.num_test_envs)])
    eval_policy = RealActorPolicy(student_encoder, actor, device, action_dim=action_dim, stochastic=False,
                                 failure_mask=failure_mask)

    test_collector = Collector(eval_policy, test_envs)

    # ---- logging ----
    os.makedirs(args.logdir, exist_ok=True)
    save_dir = os.path.join(args.logdir, f"v2_{args.latent_tag}_{args.model_type}")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    # ---- map your original knobs to imagination cadence ----
    # Total "dream" budget you used to set via --step_per_epoch.
    # We turn it into number of imagination updates so that:
    #   updates * (batch=B) * (horizon=H) ≈ step_per_epoch
    B = max(1, int(args.num_train_envs))          # parallel imagination batch
    H = max(1, int(args.dream_horizon))           # rollout horizon
    imag_updates_per_epoch = max(1, int(args.step_per_epoch // max(1, (B * H))))
    print(f"[v2] imagination config: B={B} (from num_train_envs), H={H} (dream_horizon), "
          f"updates/epoch={imag_updates_per_epoch} (from step_per_epoch)")

    print(f"[v2] start training. logs/checkpoints -> {save_dir}")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        # 1) Collect real steps -> fill wm_buffer
        if args.real_steps_per_epoch > 0:
            real_res = real_collector.collect(n_step=args.real_steps_per_epoch, reset_before_collect=True)
            log_collect_stats(writer, "real", real_res, global_step)

        # 2) Update WM on real->latent (epoch boundary)
        if args.wm_updates_per_epoch > 0:
            avg_wm = wm_update(world_model, wm_buffer, device, args, latent_dim, decoder_path=args.decoder_path,
                              failure_mask=failure_mask)    
            writer.add_scalar("wm/avg_loss_epoch", float(avg_wm), global_step)

        # 3) Dreamer-style imagination updates (no replay; fresh trajectories each update)
        for _ in range(imag_updates_per_epoch):
            z0 = sample_start_latents_from_wm_buffer(wm_buffer, B, device, initial_states)
            stats = dreamer_update(
                actor, value, actor_optim, value_optim,
                world_model, z0, H=H, gamma=0.99, lam=args.lambda_,
                disagreement_coef=args.disagreement_coef,
                device=device,
                failure_mask=failure_mask,
            )
            writer.add_scalar("dreamer/actor_loss", stats["actor_loss"], global_step)
            writer.add_scalar("dreamer/value_loss", stats["value_loss"], global_step)
            writer.add_scalar("dreamer/r_mean",     stats["r_mean"],     global_step)
            global_step += 1

        # 4) Periodic real-env eval (deterministic)
        eval_policy.stochastic = False
        eres = test_collector.collect(n_episode=args.episode_per_test, reset_before_collect=True)
        log_collect_stats(writer, "eval", eres, global_step, base_env_id=args.test_env_id)


        # 5) Save checkpoints occasionally
        if (epoch % 5) == 0:
            torch.save(actor.state_dict(), os.path.join(save_dir, f"actor_epoch{epoch}.pth"))
            torch.save(value.state_dict(), os.path.join(save_dir, f"value_epoch{epoch}.pth"))
            torch.save(world_model.state_dict(), os.path.join(save_dir, f"wm_epoch{epoch}.pth"))

    print("\n--- v2 training finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dreamer-lite: real→WM update at epoch boundary, fresh imagination updates for actor/value. Arg names kept for compatibility.")
    # latent/source selection
    parser.add_argument("--latent_tag", type=str, default="student_post_ft",
                        choices=["student_post_ft", "student_pre_ft", "teacher"])
    parser.add_argument("--data_path", type=str, default=None)

    # world model
    parser.add_argument("--model_type", type=str, default="gru",
                        choices=["gru", "geometric_gru", "diffusion", "rssm", "sde", "koopman", "moe_gru"])
    parser.add_argument("--num_ensemble", type=int, default=5)
    parser.add_argument("--model_ckpt", type=str, default=None)

    # actions / dims
    parser.add_argument("--action_dim", type=int, default=11)

    # dream/imagination horizon (keeps your name)
    parser.add_argument("--dream_horizon", type=int, default=20)

    # disagreement shaping
    parser.add_argument("--disagreement_coef", type=float, default=0.02)
    parser.add_argument("--disagreement_on", type=str, default="reward", choices=["latent", "reward"])

    # encoder
    parser.add_argument("--encoder_ckpt", type=str, default=None)
    parser.add_argument("--policy_ckpt", type=str, default="policy.pth")
    # TRAINING LENGTHS — keep original flags
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--step_per_epoch", type=int, default=100_000)     # mapped to imagination budget
    parser.add_argument("--step_per_collect", type=int, default=1_000)     # accepted, unused in v2
    parser.add_argument("--episode_per_test", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2048)            # accepted, unused in v2
    parser.add_argument("--num_train_envs", type=int, default=64)          # mapped to imagination batch B
    parser.add_argument("--num_test_envs", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=0)             # accepted, unused in v2
    parser.add_argument("--logdir", type=str, default="log_model_based_v2")
    parser.add_argument("--cpu", action="store_true")

    # evaluation env id(s)
    parser.add_argument("--test_env_id", type=int, default=151)

    # real env mixing for WM updates (TRAIN digital twins only)
    parser.add_argument("--num_real_train_envs", type=int, default=1)
    parser.add_argument("--train_env_base", type=int, default=0)
    parser.add_argument("--real_steps_per_epoch", type=int, default=1000)

    # online WM optimizer (real-only)
    parser.add_argument("--wm_buffer_size", type=int, default=500_000)
    parser.add_argument("--wm_lr", type=float, default=1e-4)
    parser.add_argument("--wm_wd", type=float, default=0.0)
    parser.add_argument("--wm_updates_per_epoch", type=int, default=500)
    parser.add_argument("--wm_batch_size", type=int, default=512)
    parser.add_argument("--wm_grad_clip", type=float, default=1.0)
    parser.add_argument("--decoder_path", type=str, default=None)

    # model sampling knobs to mirror your old flags
    parser.add_argument("--gru_deterministic", action="store_true")
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--use_ema_at_eval", action="store_true", default=True)

    # dreamer-specific
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
