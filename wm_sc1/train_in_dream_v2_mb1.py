# train_in_dream_v2.py
import os
import argparse
import pickle
from typing import List, Optional

import numpy as np
import torch
from torch import nn
import math

import gymnasium as gym
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import BasePolicy
from torch.utils.tensorboard import SummaryWriter

from world_models import (
    EnsembleWorldModel, ProbabilisticGRU, RSSMWorldModel, KoopmanWorldModel
)
from env_3_mb1 import ReservoirEnv
from nets import HistoryEncoder, PolicyHead
def _safe_mean_over_valid(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    x: [B] or [B,A]
    valid: [B] float(0/1)
    returns scalar mean over valid batch items (and over action dim if present).
    """
    v = valid.detach()
    if x.dim() == 2:
        v = v[:, None]
    denom = v.sum().clamp_min(1.0)
    return (x.detach() * v).sum() / denom

def _get_all_from_buffer(buf):
    """Try to fetch *all* transitions as one Batch."""
    N = len(buf)
    try:
        idx = np.arange(N)
        batch = buf[idx]              # many tianshou buffers support this
    except Exception:
        batch, _ = buf.sample(N)      # fallback (order doesn’t matter for us)
    info = batch.info
    if isinstance(info, np.ndarray):
        info = info.tolist()
    return batch, info

def _build_seq_windows(info_list, done_list, seq_len: int):
    groups = {}
    for i, d in enumerate(info_list):
        if done_list[i]:
            continue  # <-- critical: exclude terminal transitions
        env_id = d.get("env_id", None)
        ep_id  = d.get("episode_id", None)
        t      = d.get("t_state", None)
        if env_id is None or ep_id is None or t is None:
            continue
        groups.setdefault((env_id, ep_id), []).append(i)

    windows = []
    for _, inds in groups.items():
        inds.sort(key=lambda k: info_list[k]["t_state"])
        ts = [info_list[k]["t_state"] for k in inds]
        for s in range(0, len(inds) - seq_len + 1):
            seg = inds[s:s+seq_len]
            seg_ts = ts[s:s+seq_len]
            if all(seg_ts[j+1] == seg_ts[j] + 1 for j in range(seq_len - 1)):
                windows.append(seg)
    return windows
class LatentGaussianActorFromPolicyHead(nn.Module):
    """
    Dreamer-style actor that *directly* uses a trained PolicyHead.
    This guarantees μ, σ are identical to the SAC actor for any given latent z.
    """
    def __init__(self, policy_head: nn.Module):
        super().__init__()
        self.policy_head = policy_head  # already trained

    def forward(self, z: torch.Tensor, state=None, info=None):
        mu, sigma = self.policy_head(z)
        return mu, sigma


def build_policy_head_from_policy_ckpt(
    policy_ckpt: str,
    latent_dim: int,
    action_dim: int,
    device: torch.device,
) -> Optional[PolicyHead]:
    """Load a PolicyHead state_dict that was saved directly from the SAC policy."""
    if not os.path.exists(policy_ckpt):
        print(f"[v2] policy_head ckpt '{policy_ckpt}' not found; skipping MF warm-start.")
        return None

    print(f"[v2] loading PolicyHead weights from: {policy_ckpt}")
    head = PolicyHead(d_model=latent_dim, action_shape=(action_dim,)).to(device)
    state = torch.load(policy_ckpt, map_location=device)
    head.load_state_dict(state, strict=True)
    head.eval()
    return head


def log_collect_stats(writer, prefix: str, stat, step: int, base_env_id: int | None = None):
    """prefix: 'eval' or 'real'. Works with CollectStats or dict-like fallback."""
    n_steps = getattr(stat, "n_collected_steps", 0)
    n_eps   = getattr(stat, "n_collected_episodes", 0)
    writer.add_scalar(f"{prefix}/n_steps", float(n_steps), step)
    writer.add_scalar(f"{prefix}/n_episodes", float(n_eps), step)

    if getattr(stat, "lens_stat", None) is not None:
        writer.add_scalar(f"{prefix}/len_mean", float(stat.lens_stat.mean), step)
        writer.add_scalar(f"{prefix}/len_min",  float(stat.lens_stat.min),  step)
        writer.add_scalar(f"{prefix}/len_max",  float(stat.lens_stat.max),  step)

    if getattr(stat, "returns_stat", None) is not None:
        rs = stat.returns_stat
        writer.add_scalar(f"{prefix}/returns_mean", float(rs.mean), step)
        writer.add_scalar(f"{prefix}/returns_std",  float(rs.std),  step)
        writer.add_scalar(f"{prefix}/returns_min",  float(rs.min),  step)
        writer.add_scalar(f"{prefix}/returns_max",  float(rs.max),  step)
        if hasattr(stat, "returns"):
            writer.add_histogram(f"{prefix}/returns_hist", np.asarray(stat.returns), step)
            if base_env_id is not None:
                for i, R in enumerate(np.asarray(stat.returns).flatten()):
                    writer.add_scalar(f"{prefix}/env_{base_env_id + 10*i}_return", float(R), step)
    else:
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


def default_encoder_ckpt_for_tag(latent_tag: str) -> str:
    if latent_tag == "student_pre_ft":
        return "student_encoder_pre_finetuning.pth"
    if latent_tag == "student_post_ft":
        return "student_encoder_post_finetuning.pth"
    if latent_tag == "teacher":
        return "teacher_encoder.pth"
    return "student_encoder_post_finetuning.pth"


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
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        dev = self.mu.weight.device
        obs = obs.to(dev, non_blocking=True)
        x = self.net(self.norm(obs))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -5.0, 0.0)
        std = torch.exp(log_std)
        return mu, std  # pre-tanh


class Scenario1FailureMask:
    """Mask certain action dimensions to a fixed normalized value."""
    def __init__(self, dead_action_indices, dead_action_value: float = -1.0):
        if isinstance(dead_action_indices, (int, np.integer)):
            idx_list = [int(dead_action_indices)]
        elif isinstance(dead_action_indices, str):
            idx_list = [int(tok) for tok in dead_action_indices.split(",") if tok.strip()]
        else:
            idx_list = [int(i) for i in dead_action_indices]

        self.dead_action_indices = idx_list
        self.dead_action_value = float(dead_action_value)

    def mask(self, action: torch.Tensor) -> torch.Tensor:
        """action: tensor [..., A] in [-1, 1]. Returns new tensor with dead dims overwritten."""
        if not self.dead_action_indices:
            return action
        a = action.clone()
        idx = torch.as_tensor(self.dead_action_indices, device=a.device, dtype=torch.long)
        a[..., idx] = self.dead_action_value
        return a


class LatentActorPolicy(BasePolicy):
    """Takes latent z directly (from RealToLatentEnv) and outputs actions."""
    def __init__(
        self,
        actor: nn.Module,
        device: torch.device,
        action_dim: int,
        stochastic: bool = True,
        failure_mask: Optional[Scenario1FailureMask] = None,
    ):
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
# Value network (latent)
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
        if not torch.is_tensor(z):
            z = torch.as_tensor(z, dtype=torch.float32)
        z = z.to(self.net[0].weight.device, non_blocking=True)
        return self.net(self.norm(z)).squeeze(-1)


# --------------------------------------------------------------------------------------
# Real -> Latent wrapper (subprocess)
# --------------------------------------------------------------------------------------
class RealToLatentEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, encoder: nn.Module, device: str | torch.device = "cpu", env_id: int | None = None):
        super().__init__(env)
        self.encoder = encoder.eval()
        for p in self.encoder.parameters(): p.requires_grad_(False)
        self.device = torch.device(device)

        self._env_id = int(env_id) if env_id is not None else int(getattr(env, "env_id", -1))
        self._episode_id = 0
        self._t_state = 0
        self._max_steps = int(getattr(env, "max_steps", 20))  # you said always 20

        obs0, info0 = env.reset()
        z0 = self._encode(obs0)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(z0.numel(),), dtype=np.float32)

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
        self._episode_id += 1
        self._t_state = 0
        info = dict(info)
        info.update({"env_id": self._env_id, "episode_id": self._episode_id, "t_state": self._t_state, "max_steps": self._max_steps})
        z = self._encode(obs)
        return z.cpu().numpy().astype(np.float32), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self._t_state += 1
        info = dict(info)
        info.update({"env_id": self._env_id, "episode_id": self._episode_id, "t_state": self._t_state, "max_steps": self._max_steps})
        z = self._encode(obs)
        return z.cpu().numpy().astype(np.float32), rew, term, trunc, info


# --------------------------------------------------------------------------------------
# World model ensemble
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


def lambda_returns_dreamer(r, disc, v_next, bootstrap, lam=0.95):
    # r, disc, v_next: [T,B]; bootstrap: [B]
    G = bootstrap
    out = torch.zeros_like(r)
    for t in reversed(range(r.size(0))):
        G = r[t] + disc[t] * ((1 - lam) * v_next[t] + lam * G)
        out[t] = G
    return out


# --------------------------------------------------------------------------------------
# Schedule + fear (tensor-safe, time-aware)
# --------------------------------------------------------------------------------------
def _apply_schedule_and_fear(raw_model: torch.Tensor, t_abs: torch.Tensor) -> torch.Tensor:
    # raw_model: [B], t_abs: [B] long
    t = t_abs.to(raw_model.device)

    cap = torch.where(
        t <= 2,  torch.tensor(2.2, device=raw_model.device),
        torch.where(
            t <= 10, torch.tensor(1.8, device=raw_model.device),
            torch.where(
                t <= 13, torch.tensor(1.3, device=raw_model.device),
                torch.tensor(1.0, device=raw_model.device)
            )
        )
    )

    fear = torch.where(
        t <= 2,  raw_model > 2.2,
        torch.where(
            t <= 10, raw_model >= 1.9,
            torch.where(
                t <= 13, raw_model >= 1.4,
                raw_model >= 1.2
            )
        )
    )

    raw_sched = torch.minimum(raw_model, cap)
    raw_sched = torch.where(fear, torch.zeros_like(raw_sched), raw_sched)

    # # early-steps floor (fixed: no Python `if` on tensor)
    # early = (t <= 13)
    # raw_sched = torch.where(early & (raw_sched < 0.7), torch.zeros_like(raw_sched), raw_sched)

    return raw_sched


# --------------------------------------------------------------------------------------
# Imagination (terminal-aware using t0/max_steps)
# --------------------------------------------------------------------------------------
def _one_step_wm(m, z, a, h, deterministic: bool,
                 diffusion_steps: int | None = None,
                 use_ema: bool = True):
    if h is not None:
        if isinstance(m, ProbabilisticGRU):
            mean, _std, r, h_next = m.forward(z, a, h)
            return mean, r, h_next

        if isinstance(m, RSSMWorldModel):
            z_next, r, h_next = m.sample(z, a, h)
            return z_next, r, h_next

        z_next, r, h_next = m.sample(z, a, h)
        return z_next, r, h_next

    if isinstance(m, KoopmanWorldModel):
        z_pred = m._predict_next(z, a)
        r_pred = m.r_head(torch.cat([z_pred, a], dim=-1))
        return z_pred, r_pred, None

    z_next, r, _ = m.sample(z, a, h=None)
    return z_next, r, None


def imagine_ahead(
    actor,
    world_model,
    z0: torch.Tensor,
    t0: torch.Tensor,
    max_steps: int,
    H: int,
    device,
    disagreement_coef: float = 0.0,
    disagreement_on: str = "reward",
    use_mean_dynamics: bool = True,
    gamma: float = 0.99,
    diffusion_steps: int | None = None,
    use_ema: bool = True,
    failure_mask: Optional[Scenario1FailureMask] = None,
        return_metrics: bool = True,   # <-- ADD

):
    B, D = z0.shape
    z = z0.to(device)
    t0 = t0.to(device).long()

    models = world_model.models
    E = len(models)
    m0 = models[0]

    has_hidden = any([hasattr(m0, "hidden_dim"), hasattr(m0, "hidden"), hasattr(m0, "deter_dim")])
    h_ens = None
    if has_hidden:
        Hdim = getattr(m0, "hidden_dim", getattr(m0, "hidden", getattr(m0, "deter_dim", None)))
        if Hdim is None:
            raise RuntimeError("Recurrent WM detected but could not infer hidden dimension.")
        h_ens = torch.zeros(E, B, Hdim, device=device)

    z_seq, a_seq, r_seq, disc_seq = [], [], [], []
    midx = torch.randint(low=0, high=E, size=(B,), device=device)
    bidx = torch.arange(B, device=device)
    ent_seq = []

    if return_metrics:
        met = {k: torch.zeros((), device=device) for k in [
            "mu_abs_mean",
            "mu_abs_alive_mean",
            "std_mean",
            "std_alive_mean",
            "pre_abs_mean",
            "act_abs_mean",
            "act_sat_frac",        # frac(|a|>0.97) over all dims
            "act_sat_alive_frac",  # same but ignoring dead dim(s)
            "ent_mean",
            "raw_r_mean",
            "shaped_r_mean",
            "dis_mean",
            "valid_frac",
        ]}
    for t in range(1, H + 1):
        t_abs = t0 + t
        valid = (t_abs <= max_steps).float()
        cont  = (t_abs <  max_steps).float()
        valid_mask = valid[:, None].bool()

        mu, std = actor(z)
        std = torch.clamp(std, 1e-6, 1e6)

        # sample first so pre/a exist for logging
        eps = torch.randn_like(std)
        pre = mu + std * eps
        a = torch.tanh(pre)
        if failure_mask is not None:
            a = failure_mask.mask(a)
        a_seq.append(a)

        # entropy (pre-tanh Gaussian, same as before)
        ent_per_dim = 0.5 * (1.0 + math.log(2.0 * math.pi)) + torch.log(std)  # [B,A]
        if failure_mask is not None and failure_mask.dead_action_indices:
            alive = torch.ones_like(ent_per_dim)
            idx = torch.as_tensor(failure_mask.dead_action_indices, device=std.device, dtype=torch.long)
            alive[:, idx] = 0.0
            ent = (ent_per_dim * alive).sum(dim=-1)  # [B]
        else:
            ent = ent_per_dim.sum(dim=-1)
        ent_seq.append(ent)

        # --- metrics AFTER pre/a exist ---
        if return_metrics:
            valid_count = valid.sum().clamp_min(1.0)

            mu_det  = mu.detach()
            std_det = std.detach()
            pre_det = pre.detach()
            a_det   = a.detach()

            if failure_mask is not None and failure_mask.dead_action_indices:
                alive = torch.ones_like(mu_det)
                idx = torch.as_tensor(failure_mask.dead_action_indices, device=device, dtype=torch.long)
                alive[:, idx] = 0.0
                alive_cnt = alive.sum(dim=-1).clamp_min(1.0)

                mu_abs_alive = (mu_det.abs() * alive).sum(dim=-1) / alive_cnt
                std_alive    = (std_det * alive).sum(dim=-1) / alive_cnt
                sat_alive    = ((a_det.abs() > 0.97).float() * alive).sum(dim=-1) / alive_cnt
            else:
                mu_abs_alive = mu_det.abs().mean(dim=-1)
                std_alive    = std_det.mean(dim=-1)
                sat_alive    = (a_det.abs() > 0.97).float().mean(dim=-1)

            mu_abs  = mu_det.abs().mean(dim=-1)
            std_m   = std_det.mean(dim=-1)
            pre_abs = pre_det.abs().mean(dim=-1)
            act_abs = a_det.abs().mean(dim=-1)
            sat_all = (a_det.abs() > 0.97).float().mean(dim=-1)

            met["mu_abs_mean"]        += (mu_abs * valid).sum() / valid_count
            met["mu_abs_alive_mean"]  += (mu_abs_alive * valid).sum() / valid_count
            met["std_mean"]           += (std_m * valid).sum() / valid_count
            met["std_alive_mean"]     += (std_alive * valid).sum() / valid_count
            met["pre_abs_mean"]       += (pre_abs * valid).sum() / valid_count
            met["act_abs_mean"]       += (act_abs * valid).sum() / valid_count
            met["act_sat_frac"]       += (sat_all * valid).sum() / valid_count
            met["act_sat_alive_frac"] += (sat_alive * valid).sum() / valid_count
            met["ent_mean"]           += (ent.detach() * valid).sum() / valid_count
            met["valid_frac"]         += valid.mean()
        std = torch.clamp(std, 1e-6, 1e6)

        # use one eps so "pre" matches the actual sampled action
        eps = torch.randn_like(std)
        pre = mu + std * eps
        a = torch.tanh(pre)
        if failure_mask is not None:
            a = failure_mask.mask(a)
        a_seq.append(a)

        z_in = z

        Z_list, R_list, H_list = [], [], []
        h_prev = h_ens

        for e, m in enumerate(models):
            he = None if h_ens is None else h_ens[e]
            ze1, re1, he1 = _one_step_wm(
                m, z_in, a, he, deterministic=use_mean_dynamics,
                diffusion_steps=diffusion_steps, use_ema=use_ema
            )
            Z_list.append(ze1)
            R_list.append(re1.view(B, -1))
            if h_ens is not None:
                H_list.append(he1)

        Z = torch.stack(Z_list, dim=0)              # [E,B,D]
        R = torch.stack(R_list, dim=0).squeeze(-1)  # [E,B]

        if h_ens is not None:
            Hnext = torch.stack(H_list, dim=0)      # [E,B,Hdim]
            h_ens = torch.where(valid_mask[None, ...], Hnext, h_prev)

        z_next = Z.mean(dim=0)      # [B,D]
        raw_model = R.mean(dim=0)   # [B]

        dis_val = torch.zeros(B, device=device)
        if disagreement_coef > 0:
            if disagreement_on == "latent":
                dis_val = Z.std(dim=0).mean(dim=-1)
            else:
                dis_val = R.std(dim=0).abs()

        shaped = _apply_schedule_and_fear(raw_model, t_abs)
        r = shaped - disagreement_coef * dis_val
        if return_metrics:
            valid_count = valid.sum().clamp_min(1.0)
            met["raw_r_mean"]    += (raw_model.detach() * valid).sum() / valid_count
            met["shaped_r_mean"] += (shaped.detach()    * valid).sum() / valid_count
            met["dis_mean"]      += (dis_val.detach()   * valid).sum() / valid_count
        r = r * valid
        disc = torch.full((B,), gamma, device=device) * cont

        z_next_eff = torch.where(valid_mask, z_next, z)

        z_seq.append(z_next_eff)
        r_seq.append(r)
        disc_seq.append(disc)

        z = z_next_eff

    out = dict(
        z=torch.stack(z_seq, 0),
        a=torch.stack(a_seq, 0),
        r=torch.stack(r_seq, 0),
        disc=torch.stack(disc_seq, 0),
        ent=torch.stack(ent_seq, 0),
    )
    if return_metrics:
        out["metrics"] = {k: (v / float(H)).item() for k, v in met.items()}
    return out

def dreamer_update(
    actor, value, actor_optim, value_optim,
    world_model, start_z, t0, max_steps, H,
    gamma=0.99, lam=0.95,
    disagreement_coef=0.0,
    disagreement_on: str = "reward",
    device="cuda",
    failure_mask: Optional[Scenario1FailureMask] = None,
):
    wm_flags = [p.requires_grad for p in world_model.parameters()]
    for p in world_model.parameters():
        p.requires_grad_(False)
    world_model.eval()

    traj = imagine_ahead(
        actor, world_model, start_z, t0, max_steps, H, device,
        disagreement_coef=disagreement_coef,
        disagreement_on=disagreement_on,
        use_mean_dynamics=True,
        gamma=gamma,
        failure_mask=failure_mask,
    )
    metrics = traj.get("metrics", {})

    ent_seq = traj["ent"]  # [T,B]

    for p, f in zip(world_model.parameters(), wm_flags):
        p.requires_grad_(f)

    z_seq, r_seq, disc_seq = traj["z"], traj["r"], traj["disc"]  # z:[T,B,D], r:[T,B], disc:[T,B]
    T, B, D = z_seq.shape

    z_t = torch.cat([start_z.unsqueeze(0), z_seq[:-1]], dim=0).detach()   # [H,B,D]
    z_tp1 = z_seq.detach()                                               # [H,B,D]

    v = value(z_t.reshape(T * B, D)).view(T, B)          # V(z_t)
    v_next = value(z_tp1.reshape(T * B, D)).view(T, B)   # V(z_{t+1})
    bootstrap = v_next[-1]                                # V(z_H)

    with torch.no_grad():
        g_target = lambda_returns_dreamer(r_seq.detach(), disc_seq, v_next, bootstrap, lam=lam)

    value_loss = ((v - g_target) ** 2).mean()
    value_optim.zero_grad(set_to_none=True)
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
    value_optim.step()

    # ---------------- Actor update ----------------
    g_actor = lambda_returns_dreamer(r_seq, disc_seq, v_next.detach(), bootstrap.detach(), lam=lam)
    ent_coef = getattr(args, "ent_coef", 1e-5)  # or pass explicitly
    ent_seq = ent_seq * (disc_seq > 0).float()   # or return `valid` from imagine_ahead and use that

    # incorporate entropy into reward, then compute returns on that
    r_aug = r_seq + ent_coef * ent_seq

    g_actor = lambda_returns_dreamer(r_aug, disc_seq, v_next.detach(), bootstrap.detach(), lam=lam)
    actor_loss = -g_actor.mean() 
    actor_optim.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optim.step()

    return {
        "actor_loss": float(actor_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "r_mean": float(r_seq.mean().item()),
        "metrics": metrics,
    }

# --------------------------------------------------------------------------------------
# Real-env policy (dict obs -> encoder -> actor)
# --------------------------------------------------------------------------------------
class RealActorPolicy(BasePolicy):
    """Real env dict obs -> encoder -> actor -> action (collection-only)."""
    def __init__(
        self,
        encoder: nn.Module,
        actor: nn.Module,
        device: torch.device,
        action_dim: int,
        stochastic: bool = True,
        failure_mask: Optional[Scenario1FailureMask] = None,
    ):
        super().__init__(
            action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
            action_scaling=False,
            action_bound_method="clip",
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

        return Batch(act=a.cpu().numpy(), state=state)

    def learn(self, batch, *args, **kwargs):
        raise RuntimeError("RealActorPolicy is collection-only; learn() should not be called.")


# --------------------------------------------------------------------------------------
# WM update (real-only)
# --------------------------------------------------------------------------------------
def wm_update(world_model, wm_buffer, device, args, latent_dim, wm_optim, decoder_path=None, seq_len: int = 10):
    if len(wm_buffer) < max(8, args.wm_batch_size // 4):
        return 0.0

    world_model.train()
    m0 = world_model.models[0]
    model_type = args.model_type

    is_recurrent = model_type in ("gru", "geometric_gru", "rssm")

    # ---- optional decoder for geometric_gru (same as you already do) ----
    decoder = None
    use_geom = (model_type == "geometric_gru") and (decoder_path is not None)
    if use_geom:
        from nets import Transposed3DCNN, LatentMapping
        decoder = Transposed3DCNN(latent_dim=latent_dim, out_channels=2).to(device)
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        decoder.eval()

        if args.latent_tag == "student_pre_ft":
            mapping = LatentMapping(dim=latent_dim, hidden=256, num_layers=3).to(device)
            mapping_ckpt = "latent_mappings/pre2post_mapping_best.pth"
            mapping.load_state_dict(torch.load(mapping_ckpt, map_location=device))
            mapping.eval()
            for p in mapping.parameters(): p.requires_grad_(False)

            class PreToPostDecoder(nn.Module):
                def __init__(self, mapping, decoder):
                    super().__init__()
                    self.mapping = mapping
                    self.decoder = decoder
                def forward(self, z_pre):
                    return self.decoder(self.mapping(z_pre))

            decoder = PreToPostDecoder(mapping, decoder).to(device)

    total = 0.0

    # ---- sequence path for recurrent models ----
    if is_recurrent and model_type != "koopman":
        # Build windows once per wm_update call
        batch_all, info_all = _get_all_from_buffer(wm_buffer)
        done_all = None
        if hasattr(batch_all, "done") and batch_all.done is not None:
            done_all = np.asarray(batch_all.done).astype(bool)
        else:
            term = np.asarray(getattr(batch_all, "terminated", np.zeros(len(wm_buffer)))).astype(bool)
            trunc = np.asarray(getattr(batch_all, "truncated", np.zeros(len(wm_buffer)))).astype(bool)
            done_all = term | trunc

        windows = _build_seq_windows(info_all,done_all, seq_len=seq_len)

        if len(windows) == 0:
            # fallback to one-step if something is wrong with info tagging
            # (but with the RealToLatentEnv patch above, you should have windows)
            model_type = "fallback_onestep"

        else:
            # pull arrays once
            z_all  = np.asarray(batch_all.obs)
            a_all  = np.asarray(batch_all.act)
            r_all  = np.asarray(batch_all.rew)
            zn_all = np.asarray(batch_all.obs_next)

            for _ in range(args.wm_updates_per_epoch):
                # sample seq batch
                B = args.wm_batch_size
                replace = len(windows) < B
                chosen = np.random.choice(len(windows), size=B, replace=replace)
                seg = np.asarray([windows[i] for i in chosen], dtype=np.int64)  # [B, T]

                z_seq  = torch.as_tensor(z_all[seg],  device=device, dtype=torch.float32)  # [B,T,D]
                a_seq  = torch.as_tensor(a_all[seg],  device=device, dtype=torch.float32)  # [B,T,A]
                r_seq  = torch.as_tensor(r_all[seg],  device=device, dtype=torch.float32)  # [B,T] or [B,T,1]
                zn_seq = torch.as_tensor(zn_all[seg], device=device, dtype=torch.float32)  # [B,T,D]

                if r_seq.dim() == 2:
                    r_seq = r_seq.unsqueeze(-1)  # [B,T,1]

                wm_optim.zero_grad(set_to_none=True)

                if use_geom:
                    # offline-style: flatten sequences into transitions for geometric term
                    B0, T0, D0 = z_seq.shape
                    z_flat  = z_seq.reshape(B0 * T0, D0)
                    a_flat  = a_seq.reshape(B0 * T0, -1)
                    r_flat  = r_seq.reshape(B0 * T0, -1)
                    zn_flat = zn_seq.reshape(B0 * T0, D0)
                    h0 = torch.zeros(z_flat.size(0), m0.hidden_dim, device=device)
                    loss = world_model.calculate_geometric_loss(zn_flat, z_flat, a_flat, r_flat, h0, decoder)
                else:
                    # true sequence loss (GRU / RSSM)
                    losses = []
                    for m in world_model.models:
                        losses.append(m.sequence_loss(z_seq, a_seq, r_seq, zn_seq))
                    loss = torch.stack(losses).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), args.wm_grad_clip)
                wm_optim.step()
                total += float(loss.detach().item())

            world_model.eval()
            return total / max(1, args.wm_updates_per_epoch)

    # ---- one-step path (koopman or fallback) ----
    for _ in range(args.wm_updates_per_epoch):
        batch, _ = wm_buffer.sample(min(args.wm_batch_size, len(wm_buffer)))
        if hasattr(batch, "done") and batch.done is not None:
            done = np.asarray(batch.done).astype(bool)
        else:
            term  = np.asarray(getattr(batch, "terminated", np.zeros(len(batch.obs)))).astype(bool)
            trunc = np.asarray(getattr(batch, "truncated",  np.zeros(len(batch.obs)))).astype(bool)
            done = term | trunc    
    
        mask = torch.as_tensor(~done, device=device)
        z_t    = torch.as_tensor(batch.obs,      device=device, dtype=torch.float32)[mask]
        a_t    = torch.as_tensor(batch.act,      device=device, dtype=torch.float32)[mask]
        r_t    = torch.as_tensor(batch.rew,      device=device, dtype=torch.float32).unsqueeze(-1)[mask]
        z_next = torch.as_tensor(batch.obs_next, device=device, dtype=torch.float32)[mask]

        if z_t.size(0) < 8:
            continue
        wm_optim.zero_grad(set_to_none=True)
        loss = world_model.calculate_loss(z_next, z_t, a_t, r_t, h=None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), args.wm_grad_clip)
        wm_optim.step()
        total += float(loss.detach().item())

    world_model.eval()
    return total / max(1, args.wm_updates_per_epoch)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def sample_start_latents_from_wm_buffer(
    wm_buffer,
    n,
    device,
    initial_states,
    default_max_steps: int = 20,
    t0_max: int = 10,
    oversample: int = 8,
):
    """
    Returns:
      z0: [B,D]
      t0: [B] long
      max_steps: int
    Bias: prefer samples with t_state <= t0_max (and not done).
    """
    if len(wm_buffer) == 0:
        idx = np.random.randint(0, len(initial_states), size=n)
        z0 = torch.stack([initial_states[i] for i in idx], 0).to(device)
        t0 = torch.zeros(n, device=device, dtype=torch.long)
        return z0, t0, int(default_max_steps)

    # oversample candidates, then filter
    K = min(len(wm_buffer), max(n, n * oversample))
    batch, _ = wm_buffer.sample(K)

    # done mask (robust across tianshou variants)
    if hasattr(batch, "done") and batch.done is not None:
        done = np.asarray(batch.done).astype(bool)
    else:
        term  = np.asarray(getattr(batch, "terminated", np.zeros(K))).astype(bool)
        trunc = np.asarray(getattr(batch, "truncated",  np.zeros(K))).astype(bool)
        done = term | trunc

    info = batch.info
    if isinstance(info, np.ndarray):
        info = info.tolist()

    t_arr = np.asarray([d.get("t_state", 0) if isinstance(d, dict) else 0 for d in info], dtype=np.int64)
    max_steps = int(info[0].get("max_steps", default_max_steps)) if (len(info) and isinstance(info[0], dict)) else int(default_max_steps)

    eligible = (~done) & (t_arr <= int(t0_max))
    idx_elig = np.where(eligible)[0]

    # fallback pool: any nonterminal
    idx_ok = np.where(~done)[0]
    if idx_ok.size == 0:
        # extreme fallback
        idx_ok = np.arange(K)

    if idx_elig.size >= n:
        chosen = np.random.choice(idx_elig, size=n, replace=False)
    else:
        # take all eligible + top up from nonterminal
        chosen = idx_elig.tolist()
        need = n - len(chosen)
        # avoid duplicates
        pool = np.setdiff1d(idx_ok, np.asarray(chosen, dtype=np.int64), assume_unique=False)
        if pool.size == 0:
            pool = idx_ok
        topup = np.random.choice(pool, size=need, replace=(pool.size < need))
        chosen = np.asarray(chosen + topup.tolist(), dtype=np.int64)

    z0 = torch.as_tensor(np.asarray(batch.obs)[chosen], device=device, dtype=torch.float32)
    t0 = torch.as_tensor(t_arr[chosen], device=device, dtype=torch.long)
    return z0, t0, max_steps


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[v2] device: {device}")

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
    print(f"[v2] latent_dim={latent_dim}, action_dim={action_dim}, num_init={len(initial_states)}")

    world_model = build_world_model(args.model_type, latent_dim, action_dim, args.num_ensemble, device)
    if args.model_ckpt is not None and os.path.exists(args.model_ckpt):
        print(f"[v2] loading WM weights from: {args.model_ckpt}")
        sd = torch.load(args.model_ckpt, map_location=device)
        world_model.load_state_dict(sd, strict=False)
    else:
        print("[v2] no WM checkpoint found/provided; starting world model from random init.")
    world_model.eval()

    policy_head = None
    if args.policy_ckpt:
        policy_head = build_policy_head_from_policy_ckpt(
            policy_ckpt=args.policy_ckpt,
            latent_dim=latent_dim,
            action_dim=action_dim,
            device=device,
        )

    if policy_head is not None:
        actor = LatentGaussianActorFromPolicyHead(policy_head).to(device)
        print("[v2] Actor = LatentGaussianActorFromPolicyHead (exact MF warm-start).")
    else:
        actor = LatentGaussianActor(latent_dim, action_dim, hidden=(256, 256)).to(device)
        print("[v2] WARNING: using randomly initialized LatentGaussianActor (no MF warm-start).")

    # scenario 1: kill certain action dims
    failure_mask = Scenario1FailureMask("10", dead_action_value=-1.0)

    value = LatentValue(latent_dim, hidden=(256, 256)).to(device)
    actor_optim = ClippedAdam(actor.parameters(), lr=args.lr, clip_norm=1.0)
    value_optim = ClippedAdam(value.parameters(), lr=args.lr, clip_norm=1.0)

    enc_path = args.encoder_ckpt or default_encoder_ckpt_for_tag(args.latent_tag)
    print(f"[v2] loading encoder from: {enc_path}")
    student_encoder = HistoryEncoder(d_model=latent_dim).to(device)
    student_encoder.load_state_dict(torch.load(enc_path, map_location=device))
    student_encoder.eval()

    def make_real_latent(env_id):
        def _thunk():
            enc = HistoryEncoder(d_model=latent_dim).to("cpu")
            enc.load_state_dict(torch.load(enc_path, map_location="cpu"))
            enc.eval()
            base = ReservoirEnv(env_id=env_id)
            return RealToLatentEnv(base, enc, device="cpu", env_id=env_id)
        return _thunk

    train_ids = [args.train_env_base + i for i in range(args.num_real_train_envs)]
    real_envs = SubprocVectorEnv([make_real_latent(eid) for eid in train_ids])
    wm_buffer = VectorReplayBuffer(args.wm_buffer_size, len(real_envs))

    data_policy = LatentActorPolicy(
        actor, device, action_dim=action_dim, stochastic=True, failure_mask=failure_mask
    )
    real_collector = Collector(data_policy, real_envs, wm_buffer, exploration_noise=False)

    def make_real_eval(i):
        return ReservoirEnv(env_id=args.test_env_id + 10 * i)
    test_envs = SubprocVectorEnv([lambda i=i: make_real_eval(i) for i in range(args.num_test_envs)])
    eval_policy = RealActorPolicy(
        student_encoder, actor, device, action_dim=action_dim, stochastic=False, failure_mask=failure_mask
    )
    test_collector = Collector(eval_policy, test_envs)

    os.makedirs(args.logdir, exist_ok=True)
    save_dir = os.path.join(args.logdir, f"v2_{args.latent_tag}_{args.model_type}")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    global_step = 0
    eval_policy.stochastic = False
    init_res = test_collector.collect(n_episode=args.episode_per_test, reset_before_collect=True)
    log_collect_stats(writer, "eval_initial", init_res, global_step, base_env_id=args.test_env_id)

    B = max(1, int(args.num_train_envs))
    H = max(1, int(args.dream_horizon))
    imag_updates_per_epoch = max(1, int(args.step_per_epoch // max(1, (B * H))))
    print(f"[v2] imagination config: B={B}, H={H}, updates/epoch={imag_updates_per_epoch}")

    print(f"[v2] start training. logs/checkpoints -> {save_dir}")
    wm_optim = torch.optim.Adam(
        world_model.parameters(),
        lr=args.wm_lr,
        weight_decay=args.wm_wd,
    )
    for epoch in range(1, args.epochs + 1):
        if args.real_steps_per_epoch > 0:
            real_res = real_collector.collect(n_step=args.real_steps_per_epoch, reset_before_collect=True)
            log_collect_stats(writer, "real", real_res, global_step)

            # --- real exploration diagnostics from replay buffer ---
            if len(wm_buffer) > 0:
                b, _ = wm_buffer.sample(min(512, len(wm_buffer)))
                a_real = torch.as_tensor(b.act, device=device, dtype=torch.float32)
                writer.add_scalar("real_action/abs_mean", a_real.abs().mean().item(), global_step)
                writer.add_scalar("real_action/sat_frac", (a_real.abs() > 0.97).float().mean().item(), global_step)

        if args.wm_updates_per_epoch > 0:
            avg_wm = wm_update(world_model, wm_buffer, device, args, latent_dim,wm_optim=wm_optim, decoder_path=args.decoder_path)
            writer.add_scalar("wm/avg_loss_epoch", float(avg_wm), global_step)

        for _ in range(imag_updates_per_epoch):
            z0, t0, max_steps = sample_start_latents_from_wm_buffer(
                wm_buffer, B, device, initial_states, default_max_steps=20
            )
            stats = dreamer_update(
                actor, value, actor_optim, value_optim,
                world_model, z0, t0, max_steps, H=H, gamma=0.99, lam=args.lambda_,
                disagreement_coef=args.disagreement_coef,
                disagreement_on=args.disagreement_on,
                device=device,
                failure_mask=failure_mask,
            )
            for k, v in stats.get("metrics", {}).items():
                writer.add_scalar(f"imag/{k}", v, global_step)
            writer.add_scalar("dreamer/actor_loss", stats["actor_loss"], global_step)
            writer.add_scalar("dreamer/value_loss", stats["value_loss"], global_step)
            writer.add_scalar("dreamer/r_mean",     stats["r_mean"],     global_step)
            global_step += 1

        eval_policy.stochastic = False
        eres = test_collector.collect(n_episode=args.episode_per_test, reset_before_collect=True)
        log_collect_stats(writer, "eval", eres, global_step, base_env_id=args.test_env_id)

        if (epoch % 5) == 0:
            torch.save(actor.state_dict(), os.path.join(save_dir, f"actor_epoch{epoch}.pth"))
            torch.save(value.state_dict(), os.path.join(save_dir, f"value_epoch{epoch}.pth"))
            torch.save(world_model.state_dict(), os.path.join(save_dir, f"wm_epoch{epoch}.pth"))

    print("\n--- v2 training finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dreamer-lite (scenario 1): real→WM update, fresh imagination updates, with action failure mask."
    )

    parser.add_argument("--latent_tag", type=str, default="student_post_ft",
                        choices=["student_post_ft", "student_pre_ft", "teacher"])
    parser.add_argument("--data_path", type=str, default=None)

    parser.add_argument("--model_type", type=str, default="gru",
                        choices=["gru", "geometric_gru", "rssm", "koopman"])
    parser.add_argument("--num_ensemble", type=int, default=10)
    parser.add_argument("--model_ckpt", type=str, default=None) 

    parser.add_argument("--action_dim", type=int, default=11)

    parser.add_argument("--dream_horizon", type=int, default=10)
    parser.add_argument("--disagreement_coef", type=float, default=0.15)
    parser.add_argument("--disagreement_on", type=str, default="reward", choices=["latent", "reward"])

    parser.add_argument("--encoder_ckpt", type=str, default=None)
    parser.add_argument("--policy_ckpt", type=str, default="policy_head_from_policy.pth")

    parser.add_argument("--epochs", type=int, default=20) 
    parser.add_argument("--step_per_epoch", type=int, default=3200)
    parser.add_argument("--episode_per_test", type=int, default=1)
    parser.add_argument("--num_train_envs", type=int, default=16)
    parser.add_argument("--num_test_envs", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log_mb_v2_prior_gru")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--test_env_id", type=int, default=151)

    parser.add_argument("--num_real_train_envs", type=int, default=2)
    parser.add_argument("--train_env_base", type=int, default=0)
    parser.add_argument("--real_steps_per_epoch", type=int, default=40)

    parser.add_argument("--wm_buffer_size", type=int, default=4000)
    parser.add_argument("--wm_lr", type=float, default=1e-4)
    parser.add_argument("--wm_wd", type=float, default=0.0)
    parser.add_argument("--wm_updates_per_epoch", type=int, default=64)
    parser.add_argument("--wm_batch_size", type=int, default=32)
    parser.add_argument("--wm_grad_clip", type=float, default=1.0)
    parser.add_argument("--decoder_path", type=str, default=None)

    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ent_coef", type=float, default=1e-5,
                    help="Entropy bonus coefficient for Dreamer actor loss.")
    args = parser.parse_args()
    main(args)
