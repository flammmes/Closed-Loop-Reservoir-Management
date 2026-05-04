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
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import BasePolicy
from torch.utils.tensorboard import SummaryWriter 

from world_models import (
    EnsembleWorldModel, ProbabilisticGRU,RSSMWorldModel, KoopmanWorldModel
)
from env_3_mb2 import ReservoirEnv
from nets import HistoryEncoder, PolicyHead
from abnormality_adapter import AbnormalityAdapter  # scenario 2/3
import math
from dataclasses import dataclass

def _done_from_batch(batch):
    """Return done mask as np.bool_ array of shape [B]."""
    if hasattr(batch, "done") and batch.done is not None:
        return np.asarray(batch.done).astype(bool)
    term  = np.asarray(getattr(batch, "terminated", np.zeros(len(batch.obs)))).astype(bool)
    trunc = np.asarray(getattr(batch, "truncated",  np.zeros(len(batch.obs)))).astype(bool)
    return term | trunc

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
@dataclass
class AdapterTrainCfg:
    steps_per_epoch: int = 50
    batch_size: int = 512
    lr: float = 1e-4
    grad_clip: float = 1.0
    z_coef: float = 1.0
    r_coef: float = 1.0
    reg_coef: float = 0.0
    apply_to: str = "both"          # latent|reward|both
    ensemble_loss: str = "avg"      # avg|sample
    drop_nan: bool = True
    drop_zero_z: bool = True
    seq_len: int = 10


def _wm_predict_ensemble(world_model, z, a, device, ensemble_loss: str = "avg"):
    """
    Returns (z_pred, r_pred) predicted by WM.
    z_pred: [B,D], r_pred: [B,1]
    ensemble_loss:
      - 'avg': mean across ensemble members
      - 'sample': choose random member per sample (B)
    """
    models = world_model.models
    E = len(models)
    B = z.size(0)

    m0 = models[0]
    is_recurrent = any([hasattr(m0, "hidden_dim"), hasattr(m0, "hidden"), hasattr(m0, "deter_dim")])

    # Collect per-ensemble predictions
    z_preds = []
    r_preds = []

    for m in models:
        if is_recurrent:
            # hidden size
            Hdim = getattr(m, "hidden_dim", getattr(m, "hidden", getattr(m, "deter_dim", None)))
            h0 = torch.zeros(B, Hdim, device=device)

            if isinstance(m, ProbabilisticGRU):
                mean, _std, r, _h1 = m(z, a, h0)
                z_next = mean
                r_pred = r
            elif isinstance(m, RSSMWorldModel):
                z_next, r_pred, _h1 = m.sample(z, a, h0)
            else:
                z_next, r_pred, _h1 = m.sample(z, a, h0)

        else:
            if isinstance(m, KoopmanWorldModel):
                z_next = m._predict_next(z, a)
                r_pred = m.r_head(torch.cat([z_next, a], dim=-1))
            else:
                z_next, r_pred, _ = m.sample(z, a, h=None)

        z_preds.append(z_next)           # [B,D]
        r_preds.append(r_pred)           # [B,1] (or [B] sometimes)

    Z = torch.stack(z_preds, dim=0)      # [E,B,D]
    R = torch.stack([rp.view(B, -1) for rp in r_preds], dim=0)  # [E,B,1]

    if ensemble_loss == "sample":
        idx = torch.randint(0, E, (B,), device=device)
        z_pred = Z[idx, torch.arange(B, device=device)]         # [B,D]
        r_pred = R[idx, torch.arange(B, device=device)]         # [B,1]
    else:
        z_pred = Z.mean(dim=0)                                  # [B,D]
        r_pred = R.mean(dim=0)                                  # [B,1]

    return z_pred, r_pred




def _init_h_ens(models, B: int, device: torch.device):
    """Initialize per-ensemble recurrent hidden states (GRU/RSSM)."""
    h_ens = []
    for m in models:
        if isinstance(m, ProbabilisticGRU):
            h_ens.append(torch.zeros(B, m.hidden_dim, device=device))
        elif isinstance(m, RSSMWorldModel):
            h_ens.append(torch.zeros(B, m.deter_dim, device=device))
        else:
            h_ens.append(None)
    return h_ens


@torch.no_grad()
def _wm_step_ens_recurrent(models, z_in, a_t, h_ens):
    """
    One recurrent step for each ensemble member.
    Returns:
      Z: [E,B,D]  (member-wise next latent predictions)
      R: [E,B,1]  (member-wise reward predictions)
      h_next: list length E of hidden states (or None for non-recurrent)
    """
    z_preds = []
    r_preds = []
    h_next = []

    B = z_in.size(0)

    for m, h in zip(models, h_ens):
        if isinstance(m, ProbabilisticGRU):
            mean, _std, r_pred, h1 = m(z_in, a_t, h)
            z_next = mean
        elif isinstance(m, RSSMWorldModel):
            z_next, r_pred, h1 = m.sample(z_in, a_t, h)
        elif isinstance(m, KoopmanWorldModel):
            z_next = m._predict_next(z_in, a_t)
            r_pred = m.r_head(torch.cat([z_next, a_t], dim=-1))
            h1 = None
        else:
            # fallback: treat as one-step stochastic model
            z_next, r_pred, _ = m.sample(z_in, a_t, h=None)
            h1 = None

        z_preds.append(z_next)                 # [B,D]
        r_preds.append(r_pred.view(B, -1))     # [B,1]
        h_next.append(h1)

    Z = torch.stack(z_preds, dim=0)            # [E,B,D]
    R = torch.stack(r_preds, dim=0)            # [E,B,1]
    return Z, R, h_next

def soft_reset_injector_outputs(actor, mu_row_scale=0.2, mu_bias=0.0, std_bias_add=1.0):
    """
    Keeps the actor/trunk intact, but de-saturates the last 3 action dims (injectors)
    by shrinking their mu mapping and increasing their exploration std.

    mu_row_scale: multiply injector rows of fc_mu.weight by this (smaller => less saturation)
    mu_bias: set injector biases to this (0 => tanh mean ~ 0)
    std_bias_add: add to injector fc_std.bias to increase sigma (softplus)
    """
    head = getattr(actor, "policy_head", None)
    if head is None:
        raise AttributeError("actor has no .policy_head")

    idx = [3,5,8,9,10]  # injector dims
    with torch.no_grad():
        # pull mu away from tanh saturation
        head.fc_mu.weight[idx, :] *= float(mu_row_scale)
        head.fc_mu.bias[idx] = float(mu_bias)

        # increase sigma for exploration on injectors
        head.fc_std.bias[idx] += float(std_bias_add)

def adapter_update_online(
    adapter: nn.Module,
    adapter_optim: torch.optim.Optimizer,
    world_model,
    wm_buffer,
    device,
    cfg: AdapterTrainCfg,
):
    """
    Online supervised adapter training using *sequence windows* from wm_buffer,
    stepping the recurrent WM hidden forward exactly like imagination does.

    Falls back to one-step training if no windows are found.
    """
    if adapter is None:
        return {"adapter_loss": 0.0, "adapter_n": 0}

    if len(wm_buffer) < max(64, cfg.batch_size):
        return {"adapter_loss": 0.0, "adapter_n": 0}

    # -----------------
    # Freeze WM
    # -----------------
    wm_req = [p.requires_grad for p in world_model.parameters()]
    for p in world_model.parameters():
        p.requires_grad_(False)
    world_model.eval()

    # Unfreeze adapter
    ad_req = [p.requires_grad for p in adapter.parameters()]
    for p in adapter.parameters():
        p.requires_grad_(True)
    adapter.train()

    # -----------------
    # Build sequence windows from buffer info
    # -----------------
    batch_all, info_all = _get_all_from_buffer(wm_buffer)
    done_all = _done_from_batch(batch_all)

    # info_all is already list-like from _get_all_from_buffer
    windows = _build_seq_windows(info_all, done_all, seq_len=cfg.seq_len)

    if len(windows) == 0:
        # --- fallback: keep your old behavior (one-step) ---
        # (This is basically your previous implementation with h=0)
        total_loss = 0.0
        total_n = 0
        updates_done = 0

        for _ in range(cfg.steps_per_epoch):
            batch, _ = wm_buffer.sample(min(cfg.batch_size, len(wm_buffer)))

            done = _done_from_batch(batch)
            keep = ~done
            if keep.sum() < 8:
                continue

            z  = torch.as_tensor(batch.obs[keep],      device=device, dtype=torch.float32)
            a  = torch.as_tensor(batch.act[keep],      device=device, dtype=torch.float32)
            r  = torch.as_tensor(batch.rew[keep],      device=device, dtype=torch.float32).view(-1, 1)
            zn = torch.as_tensor(batch.obs_next[keep], device=device, dtype=torch.float32)

            info = batch.info
            if isinstance(info, np.ndarray):
                info = info.tolist()
            keep_idx = np.where(keep)[0]
            t_state = np.asarray([
                (info[i].get("t_state", 0) if isinstance(info[i], dict) else 0)
                for i in keep_idx
            ], dtype=np.int64)
            t_before = np.clip(t_state - 1, 0, 20)
            t = torch.as_tensor(t_before, device=device, dtype=torch.long)

            # filter
            mask2 = torch.ones(z.size(0), device=device, dtype=torch.bool)
            if cfg.drop_nan:
                mask2 &= torch.isfinite(z).all(dim=-1)
                mask2 &= torch.isfinite(a).all(dim=-1)
                mask2 &= torch.isfinite(r).all(dim=-1)
                mask2 &= torch.isfinite(zn).all(dim=-1)
            if cfg.drop_zero_z:
                mask2 &= (z.abs().sum(dim=-1) > 0)
                mask2 &= (zn.abs().sum(dim=-1) > 0)

            z, a, r, zn, t = z[mask2], a[mask2], r[mask2], zn[mask2], t[mask2]
            if z.size(0) < 8:
                continue

            with torch.no_grad():
                z_pred, r_pred = _wm_predict_ensemble(world_model, z, a, device, cfg.ensemble_loss)

            z_corr, r_corr = adapter(z, a, z_pred, r_pred, t=t)

            loss = 0.0
            if cfg.apply_to in ("latent", "both"):
                loss = loss + cfg.z_coef * torch.mean((z_corr - zn) ** 2)
            if cfg.apply_to in ("reward", "both"):
                loss = loss + cfg.r_coef * torch.mean((r_corr - r) ** 2)
            if cfg.reg_coef > 0:
                loss = loss + cfg.reg_coef * (
                    torch.mean((z_corr - z_pred) ** 2) + torch.mean((r_corr - r_pred) ** 2)
                )

            adapter_optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.grad_clip)
            adapter_optim.step()

            total_loss += float(loss.detach().item())
            total_n += int(z.size(0))
            updates_done += 1

        # restore flags
        for p, f in zip(world_model.parameters(), wm_req):
            p.requires_grad_(f)
        for p, f in zip(adapter.parameters(), ad_req):
            p.requires_grad_(f)
        adapter.eval()

        denom = max(1, updates_done)
        return {"adapter_loss": total_loss / denom, "adapter_n": total_n}

    # -----------------
    # Sequence training path (recommended)
    # -----------------
    z_all  = np.asarray(batch_all.obs)
    a_all  = np.asarray(batch_all.act)
    r_all  = np.asarray(batch_all.rew)
    zn_all = np.asarray(batch_all.obs_next)

    # t_before array aligned to buffer indices
    t_state_all = np.asarray([
        (d.get("t_state", 0) if isinstance(d, dict) else 0) for d in info_all
    ], dtype=np.int64)
    t_before_all = np.clip(t_state_all - 1, 0, 20).astype(np.int64)

    total_loss = 0.0
    total_n = 0
    updates_done = 0

    # interpret cfg.batch_size as "roughly transitions"; convert to windows
    B_win = max(1, int(cfg.batch_size // max(1, cfg.seq_len)))
    
    for _ in range(cfg.steps_per_epoch):
        replace = len(windows) < B_win
        chosen = np.random.choice(len(windows), size=B_win, replace=replace)
        seg = np.asarray([windows[i] for i in chosen], dtype=np.int64)  # [B_win, T]

        z_seq  = torch.as_tensor(z_all[seg],  device=device, dtype=torch.float32)  # [B,T,D]
        a_seq  = torch.as_tensor(a_all[seg],  device=device, dtype=torch.float32)  # [B,T,A]
        r_seq  = torch.as_tensor(r_all[seg],  device=device, dtype=torch.float32)  # [B,T] or [B,T,1]
        zn_seq = torch.as_tensor(zn_all[seg], device=device, dtype=torch.float32)  # [B,T,D]
        t_seq  = torch.as_tensor(t_before_all[seg], device=device, dtype=torch.long)  # [B,T]

        if r_seq.dim() == 2:
            r_seq = r_seq.unsqueeze(-1)  # [B,T,1]
        elif r_seq.dim() == 3 and r_seq.size(-1) != 1:
            r_seq = r_seq[..., :1]

        # window-level filtering (drop whole windows if any step is bad)
        valid = torch.ones(z_seq.size(0), device=device, dtype=torch.bool)
        if cfg.drop_nan:
            valid &= torch.isfinite(z_seq).all(dim=-1).all(dim=-1)
            valid &= torch.isfinite(a_seq).all(dim=-1).all(dim=-1)
            valid &= torch.isfinite(r_seq).all(dim=-1).all(dim=-1)
            valid &= torch.isfinite(zn_seq).all(dim=-1).all(dim=-1)
        if cfg.drop_zero_z:
            valid &= (z_seq.abs().sum(dim=-1)  > 0).all(dim=-1)
            valid &= (zn_seq.abs().sum(dim=-1) > 0).all(dim=-1)

        z_seq, a_seq, r_seq, zn_seq, t_seq = z_seq[valid], a_seq[valid], r_seq[valid], zn_seq[valid], t_seq[valid]
        if z_seq.size(0) < 2:
            continue

        B0, T0, _ = z_seq.shape

        # init recurrent hidden once per window-batch
        h_ens = _init_h_ens(world_model.models, B0, device)

        # start from dataset z at first step
        z_in = z_seq[:, 0, :]

        loss = 0.0
        for t in range(T0):
            a_ti  = a_seq[:, t, :]
            r_ti  = r_seq[:, t, :]
            zn_ti = zn_seq[:, t, :]
            tt    = t_seq[:, t]

            Zm, Rm, h_ens = _wm_step_ens_recurrent(world_model.models, z_in, a_ti, h_ens)

            if cfg.ensemble_loss == "sample":
                e = int(torch.randint(0, Zm.size(0), (1,), device=device).item())
                z_pred = Zm[e]
                r_pred = Rm[e]
            else:
                z_pred = Zm.mean(dim=0)
                r_pred = Rm.mean(dim=0)

            z_corr, r_corr = adapter(z_in, a_ti, z_pred, r_pred, t=tt)

            step_loss = 0.0
            if cfg.apply_to in ("latent", "both"):
                step_loss = step_loss + cfg.z_coef * torch.mean((z_corr - zn_ti) ** 2)
            if cfg.apply_to in ("reward", "both"):
                step_loss = step_loss + cfg.r_coef * torch.mean((r_corr - r_ti) ** 2)
            if cfg.reg_coef > 0:
                step_loss = step_loss + cfg.reg_coef * (
                    torch.mean((z_corr - z_pred) ** 2) + torch.mean((r_corr - r_pred) ** 2)
                )

            loss = loss + step_loss

            # closed-loop *consistent with how you use adapter in imagination*
            if cfg.apply_to in ("latent", "both"):
                z_in = z_corr.detach()
            else:
                z_in = z_pred.detach()

        loss = loss / float(T0)

        adapter_optim.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.grad_clip)
        adapter_optim.step()

        total_loss += float(loss.detach().item())
        total_n += int(B0 * T0)
        updates_done += 1

    # restore flags
    for p, f in zip(world_model.parameters(), wm_req):
        p.requires_grad_(f)
    for p, f in zip(adapter.parameters(), ad_req):
        p.requires_grad_(f)

    adapter.eval()

    denom = max(1, updates_done)
    return {"adapter_loss": total_loss / denom, "adapter_n": total_n}
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

def default_encoder_ckpt_for_tag(latent_tag: str) -> str:
    if latent_tag == "student_pre_ft":
        return "student_encoder_pre_finetuning.pth"
    if latent_tag == "student_post_ft":
        return "student_encoder_post_finetuning.pth"
    if latent_tag == "teacher":
        return "teacher_encoder.pth"  # adjust if you use a different name
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
        if not torch.is_tensor(obs): obs = torch.as_tensor(obs, dtype=torch.float32)
        dev = self.mu.weight.device
        obs = obs.to(dev, non_blocking=True)
        x = self.net(self.norm(obs))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -5.0, 0.0)
        std = torch.exp(log_std)
        return mu, std  # pre-tanh
    

class LatentActorPolicy(BasePolicy):
    """Takes latent z directly (from RealToLatentEnv) and outputs actions."""
    def __init__(self, actor: nn.Module, device: torch.device,
                 action_dim: int, stochastic: bool = True):
        super().__init__(
            action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
            action_scaling=False,
            action_bound_method="clip",
        )
        self.actor = actor
        self.device = device
        self.stochastic = stochastic

    @torch.no_grad()
    def forward(self, batch: Batch, state=None, **kwargs):
        z = torch.as_tensor(batch.obs, device=self.device, dtype=torch.float32)
        mu, std = self.actor(z)
        if self.stochastic:
            act = torch.tanh(mu + std * torch.randn_like(std))
        else:
            act = torch.tanh(mu)
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

def lambda_returns_dreamer(r, disc, v_next, bootstrap, lam=0.95):
    # r, disc, v_next: [T,B]; bootstrap: [B]
    G = bootstrap
    out = torch.zeros_like(r)
    for t in reversed(range(r.size(0))):
        G = r[t] + disc[t] * ((1 - lam) * v_next[t] + lam * G)
        out[t] = G
    return out
# --------------------------------------------------------------------------------------
# Schedule + fear (same shaping you used in DreamEnv)
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

    # early-steps floor
    # if t <= 13:
    #     raw_sched = torch.where(raw_sched < 0.7, torch.zeros_like(raw_sched), raw_sched)

EARLY_INJ_FEAR = {
    "enabled": True,     # flip to False to disable
    "coef": 0.30,        # overall strength (reward units)
    "until": 13,         # last timestep to apply penalty (inclusive)
    "free_frac": 0.15,   # fraction of injector range allowed "free" (0..1 in [0,1] space)
    "anneal": "linear",  # "linear" or "flat"
}



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


    # Generic stateless fallback
    z_next, r, _ = m.sample(z, a, h=None)
    return z_next, r, None


def imagine_ahead(
    actor, world_model, z0, t0, max_steps, H, device,
    adapter: nn.Module | None = None,
    adapter_apply_to: str = "both",  # 'latent'|'reward'|'both'
    disagreement_after_adapter: bool = False,
    disagreement_coef: float = 0.0,
    disagreement_on: str = "reward",
    use_mean_dynamics: bool = True,
    gamma: float = 0.99,
    diffusion_steps: int | None = None,
    use_ema: bool = True,
):
    """
    Build imagined rollouts with gradients flowing to the actor through WM.
    FIX (Option B): keep per-ensemble hidden states for recurrent WMs when computing disagreement.
    """
    B, D = z0.shape
    z = z0.to(device)

    models = world_model.models
    E = len(models)
    m0 = models[0]

    # Detect recurrence and init per-ensemble hidden state: h_ens [E,B,Hdim]
    has_hidden = any([hasattr(m0, "hidden_dim"), hasattr(m0, "hidden"), hasattr(m0, "deter_dim")])
    h_ens = None
    if has_hidden:
        Hdim = getattr(m0, "hidden_dim", getattr(m0, "hidden", getattr(m0, "deter_dim", None)))
        if Hdim is None:
            raise RuntimeError("Recurrent WM detected but could not infer hidden dimension.")
        h_ens = torch.zeros(E, B, Hdim, device=device)

    z_seq, a_seq, r_seq, disc_seq = [], [], [], []

    # Fixed member per sample (same as your current behavior)
    midx = torch.randint(low=0, high=E, size=(B,), device=device)
    bidx = torch.arange(B, device=device)
    ent_seq = []
    valid_seq = []

    for t in range(1, H + 1):
        # absolute time for schedule/fear + termination masking
        t_abs = t0 + t                        # [B]
        valid = (t_abs <= max_steps).float()  # [B]
        cont  = (t_abs <  max_steps).float()  # [B]
        valid_mask = valid[:, None].bool()    # [B,1]
        valid_seq.append(valid)
        # actor sample (reparameterized)
        mu, std = actor(z)
        std = torch.clamp(std, 1e-6, 1e6)
        ent = (0.5 * (1.0 + math.log(2.0 * math.pi)) + torch.log(std)).sum(dim=-1)  # [B]
        ent_seq.append(ent)
        a = torch.tanh(mu + std * torch.randn_like(std))
        a_seq.append(a)

        z_in = z

        # ---- Run ALL ensemble members once with their OWN hidden state ----
        Z_list, R_list, H_list = [], [], []

        h_prev = h_ens  # keep for freezing invalid samples
        for e, m in enumerate(models):
            he = None if h_ens is None else h_ens[e]  # [B,Hdim] or None
            ze1, re1, he1 = _one_step_wm(
                m, z_in, a, he,
                deterministic=use_mean_dynamics,
                diffusion_steps=diffusion_steps,
                use_ema=use_ema,
            )
            Z_list.append(ze1)                    # [B,D]
            R_list.append(re1.view(B, -1))        # [B,1] (or [B,k] -> keep as [B,1]-ish)
            if h_ens is not None:
                H_list.append(he1)                # [B,Hdim]

        Z = torch.stack(Z_list, dim=0)            # [E,B,D]
        R = torch.stack(R_list, dim=0).squeeze(-1)  # [E,B]
        if h_ens is not None:
            Hnext = torch.stack(H_list, dim=0)    # [E,B,Hdim]
            # freeze hidden for samples that are already past max_steps
            h_ens = torch.where(valid_mask[None, ...], Hnext, h_prev)

        # ---- Select the "main" rollout member per sample ----
        z_next = Z.mean(dim=0)      # [B,D]
        raw_model = R.mean(dim=0)   # [B]
        t_before = (t_abs - 1).clamp_min(0)   # [B] long

        # ---- optional abnormality adapter (scenario 2/3) ----
        if adapter is not None:
            r_in = raw_model.unsqueeze(-1)   # [B,1]
            z_corr, r_corr = adapter(z_in, a, z_next, r_in, t=t_before)
            if adapter_apply_to in ("latent", "both"):
                z_next = z_corr
            if adapter_apply_to in ("reward", "both"):
                raw_model = r_corr.squeeze(-1)

        # ---- disagreement penalty across ensemble (NOW consistent for recurrent WMs) ----
        dis_val = torch.zeros(B, device=device)
        if disagreement_coef > 0:
            Z_dis = Z
            R_dis = R

            if adapter is not None and disagreement_after_adapter:
                # Apply adapter to ALL ensemble predictions (vectorized over E*B)
                z_rep = z_in.repeat(E, 1)
                a_rep = a.repeat(E, 1)
                z_pred_flat = Z.reshape(E * B, D)
                r_pred_flat = R.reshape(E * B, 1)
                t_flat = t_before.repeat(E)  # [E*B]
                zc_flat, rc_flat = adapter(z_rep, a_rep, z_pred_flat, r_pred_flat, t=t_flat)
                if adapter_apply_to in ("latent", "both"):
                    Z_dis = zc_flat.view(E, B, D)
                if adapter_apply_to in ("reward", "both"):
                    R_dis = rc_flat.view(E, B)

            if disagreement_on == "latent":
                dis_val = Z_dis.std(dim=0).mean(dim=-1)   # [B]
            else:  # 'reward'
                dis_val = R_dis.std(dim=0).abs()          # [B]
        
        shaped = _apply_schedule_and_fear(raw_model, t_abs)  # uses absolute time
        r = shaped - disagreement_coef * dis_val

        # ---- mask reward + discount & freeze state after terminal transition ----
        r = r * valid
        disc = torch.full((B,), gamma, device=device) * cont

        z_next_eff = torch.where(valid_mask, z_next, z)

        z_seq.append(z_next_eff)
        r_seq.append(r)
        disc_seq.append(disc)

        z = z_next_eff
    return dict(
        z=torch.stack(z_seq, 0),
        a=torch.stack(a_seq, 0),
        r=torch.stack(r_seq, 0),
        disc=torch.stack(disc_seq, 0),
        ent = torch.stack(ent_seq, 0),
        valid=torch.stack(valid_seq, 0)
    )

def dreamer_update(
    actor, value, actor_optim, value_optim,
    world_model, start_z,t0,max_steps, H, gamma=0.99, lam=0.95,
    adapter: nn.Module | None = None,
    adapter_apply_to: str = "both",
    disagreement_on: str = 'reward',
    disagreement_after_adapter: bool = False,
    disagreement_coef: float = 0.0,
    device: str = "cuda",
):
    # --- Freeze WM params (but keep graph wrt inputs!) ---
    wm_flags = [p.requires_grad for p in world_model.parameters()]
    for p in world_model.parameters():
        p.requires_grad_(False)
    world_model.eval()

    # --- Freeze adapter params too (if present) ---
    adapter_flags = None
    if adapter is not None:
        adapter_flags = [p.requires_grad for p in adapter.parameters()]
        for p in adapter.parameters():
            p.requires_grad_(False)
        adapter.eval()

    # --- Imagination WITH gradients to actor through (WM + adapter as functions of inputs) ---
    traj = imagine_ahead(
        actor, world_model, start_z,t0,max_steps, H, device,
        adapter=adapter,
        adapter_apply_to=adapter_apply_to,
        disagreement_on=disagreement_on,
        disagreement_after_adapter=disagreement_after_adapter,
        disagreement_coef=disagreement_coef,
        use_mean_dynamics=True,
        gamma=gamma,
    )
    ent_seq = traj["ent"]  # [T,B]

    # --- Restore flags ---
    for p, f in zip(world_model.parameters(), wm_flags):
        p.requires_grad_(f)
    if adapter is not None and adapter_flags is not None:
        for p, f in zip(adapter.parameters(), adapter_flags):
            p.requires_grad_(f)

    z_seq, r_seq, disc_seq = traj["z"], traj["r"], traj["disc"]  # z:[T,B,D], r:[T,B]
    T, B, D = z_seq.shape

    # ---------------- Critic update (no grad to actor/WM/adapter) ----------------
    z_flat_det = z_seq.detach().reshape(T * B, D)
    z_t = torch.cat([start_z.unsqueeze(0), z_seq[:-1]], dim=0).detach()   # [H,B,D]
    z_tp1 = z_seq.detach()                                               # [H,B,D]

    v = value(z_t.reshape(T * B, D)).view(T, B)          # V(z_t)
    v_next = value(z_tp1.reshape(T * B, D)).view(T, B)   # V(z_{t+1})
    bootstrap = v_next[-1]                                # V(z_H)

    with torch.no_grad():
        g_target = lambda_returns_dreamer(r_seq.detach(), disc_seq, v_next, bootstrap, lam=lam)
    valid = traj["valid"]
    den = valid.sum().clamp_min(1.0)
    value_loss = ((v - g_target.detach()) ** 2 * valid).sum() / den
    value_optim.zero_grad(set_to_none=True)
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
    value_optim.step()

    # ---------------- Actor update (grad flows through imagined rollout) ----------------
    g_actor = lambda_returns_dreamer(r_seq, disc_seq, v_next.detach(), bootstrap.detach(), lam=lam)
    ent_coef = getattr(args, "ent_coef", 1e-5)  # or pass explicitly
    ent_seq = ent_seq * (disc_seq > 0).float()   # or return `valid` from imagine_ahead and use that

    # incorporate entropy into reward, then compute returns on that
    r_aug = r_seq + ent_coef * ent_seq

    g_actor = lambda_returns_dreamer(r_aug, disc_seq, v_next.detach(), bootstrap.detach(), lam=lam)

    den = valid.sum().clamp_min(1.0)

    actor_loss = -(g_actor * valid).sum() / den   
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
    def __init__(self, encoder: nn.Module, actor: nn.Module,
                 device: torch.device, action_dim: int, stochastic: bool = True):
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
        # IMPORTANT: return a single Batch with both act and state
        return Batch(act=a.cpu().numpy(), state=state)

    # We don't train this policy; satisfy abstract method.
    def learn(self, batch, *args, **kwargs):
        # If ever called by mistake, make it obvious.
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

    # ---- optional abnormality adapter (scenario 2/3) ----
    adapter = None
    if getattr(args, "adapter_ckpt", None):
        if os.path.exists(args.adapter_ckpt):
            print(f"[v2] loading adapter from: {args.adapter_ckpt}")
            adapter = AbnormalityAdapter(latent_dim=latent_dim, action_dim=action_dim, use_time=True, max_steps=20, time_emb_dim=16).to(device)
            adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=device), strict=True)            
            adapter.eval()
            for p in adapter.parameters(): p.requires_grad_(False)
        else:
            print(f"[v2] adapter_ckpt provided but file not found: {args.adapter_ckpt}")
    adapter_optim = None
    adapter_cfg = None

    if adapter is not None:
        adapter_cfg = AdapterTrainCfg(
            steps_per_epoch=args.adapter_updates_per_epoch,
            batch_size=args.adapter_batch_size,
            lr=args.adapter_lr,
            grad_clip=args.adapter_grad_clip,
            z_coef=args.adapter_z_coef,
            r_coef=args.adapter_r_coef,
            reg_coef=args.adapter_reg_coef,
            apply_to=args.adapter_apply_to,
            ensemble_loss=args.adapter_ensemble_loss,
            drop_nan=args.adapter_drop_nan,
            drop_zero_z=args.adapter_drop_zero_z,
            seq_len=args.adapter_seq_len,
        )
        adapter_optim = torch.optim.Adam(adapter.parameters(), lr=adapter_cfg.lr)


    # ---- actor/value ----
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
    #soft_reset_injector_outputs(actor)
    value = LatentValue(latent_dim, hidden=(256, 256)).to(device)
    actor_optim = ClippedAdam(actor.parameters(), lr=args.lr, clip_norm=1.0)
    value_optim = ClippedAdam(value.parameters(), lr=args.lr, clip_norm=1.0)

    # ---- encoder for real collection/eval ----
    enc_path = args.encoder_ckpt or default_encoder_ckpt_for_tag(args.latent_tag)
    print(f"[v2] loading encoder from: {enc_path}")
    student_encoder = HistoryEncoder(d_model=latent_dim).to(device)
    student_encoder.load_state_dict(torch.load(enc_path, map_location=device))
    student_encoder.eval()
    print('done')
    # ---- real envs for WM data (TRAIN ids only; never TRUE decks) ----
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

    # stochastic policy for data collection
    data_policy = LatentActorPolicy(actor, device, action_dim=action_dim, stochastic=True)
    real_collector = Collector(data_policy, real_envs, wm_buffer, exploration_noise=False)
    # ---- real envs for evaluation (151,161,171,181,191,201 if num_test_envs=6) ----
    def make_real_eval(i):
        return ReservoirEnv(env_id=args.test_env_id + 10 * i)
    test_envs = SubprocVectorEnv([lambda i=i: make_real_eval(i) for i in range(args.num_test_envs)])
    eval_policy = RealActorPolicy(student_encoder, actor, device, action_dim=action_dim, stochastic=False)

    test_collector = Collector(eval_policy, test_envs)

    # ---- logging ----
    os.makedirs(args.logdir, exist_ok=True)
    save_dir = os.path.join(args.logdir, f"v2_{args.latent_tag}_{args.model_type}")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    # ---- initial eval BEFORE any Dreamer updates ----
    global_step = 0
    eval_policy.stochastic = False
    init_res = test_collector.collect(
        n_episode=args.episode_per_test,
        reset_before_collect=True
    )
    log_collect_stats(writer, "eval_initial", init_res, global_step, base_env_id=args.test_env_id)
    if getattr(init_res, "returns_stat", None) is not None:
        print(f"[v2] initial eval mean return: {float(init_res.returns_stat.mean):.4f}")
    else:
        print(f"[v2] initial eval done.")

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

    wm_optim = torch.optim.Adam(
    world_model.parameters(),
    lr=args.wm_lr,
    weight_decay=args.wm_wd,
)

    for epoch in range(1, args.epochs + 1):
        # 1) Collect real steps -> fill wm_buffer
        if args.real_steps_per_epoch > 0:
            real_res = real_collector.collect(n_step=args.real_steps_per_epoch, reset_before_collect=True)
            log_collect_stats(writer, "real", real_res, global_step)

        # 2) Update WM on real->latent (epoch boundary)
        if args.wm_updates_per_epoch > 0:
            avg_wm = wm_update(world_model, wm_buffer, device, args, latent_dim,wm_optim=wm_optim, decoder_path=args.decoder_path)
            writer.add_scalar("wm/avg_loss_epoch", float(avg_wm), global_step)

        if adapter is not None and adapter_optim is not None and args.adapter_updates_per_epoch > 0:
            ad_stats = adapter_update_online(
                adapter=adapter,
                adapter_optim=adapter_optim,
                world_model=world_model,
                wm_buffer=wm_buffer,
                device=device,
                cfg=adapter_cfg,
            )
            writer.add_scalar("adapter/loss_epoch", float(ad_stats["adapter_loss"]), global_step)
            writer.add_scalar("adapter/n_used", float(ad_stats["adapter_n"]), global_step)


        # 3) Dreamer-style imagination updates (no replay; fresh trajectories each update)
        for _ in range(imag_updates_per_epoch):
            z0, t0, max_steps = sample_start_latents_from_wm_buffer(
                wm_buffer, B, device, initial_states, default_max_steps=20
            )
            stats = dreamer_update(
                actor, value, actor_optim, value_optim,
                world_model, z0,t0,max_steps, H=H, gamma=0.99, lam=args.lambda_,
                adapter=adapter,
                adapter_apply_to=getattr(args, "adapter_apply_to", "both"),
                disagreement_on = getattr(args,'disagreement_on','reward'),
                disagreement_after_adapter=getattr(args, "disagreement_after_adapter", False),
                disagreement_coef=args.disagreement_coef,
                device=device
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
        if adapter is not None and (epoch % 5) == 0:
            torch.save(adapter.state_dict(), os.path.join(save_dir, f"adapter_epoch{epoch}.pth"))


    print("\n--- v2 training finished ---")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Dreamer-lite: real→WM update at epoch boundary, fresh imagination updates for actor/value."
    )

    # ---- latent/source selection ----
    parser.add_argument(
        "--latent_tag",
        type=str,
        default="student_post_ft",
        choices=["student_post_ft", "student_pre_ft", "teacher"],
        help="Which latent transitions to use for initial states / offline WM pretraining.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional explicit path to latent transition pkl. If None, inferred from latent_tag.",
    )

    # ---- world model ----
    parser.add_argument(
        "--model_type",
        type=str,
        default="gru",
        choices=["gru", "geometric_gru", "rssm", "koopman"],
        help="World-model architecture.",
    )
    parser.add_argument(
        "--num_ensemble",
        type=int,
        default=10,
        help="Number of ensemble members in the world model.",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="Path to pretrained world-model checkpoint (EnsembleWorldModel.state_dict).",
    )

    # ---- actions / dims ----
    parser.add_argument(
        "--action_dim",
        type=int,
        default=11,
        help="Action dimension in latent-space control.",
    )

    # ---- dream / imagination ----
    parser.add_argument(
        "--dream_horizon",
        type=int,
        default=20,
        help="Length of imagined rollouts H.",
    )
    parser.add_argument(
        "--disagreement_coef",
        type=float,
        default=0.15,
        help="Coefficient for ensemble disagreement penalty in imagined rewards.",
    )
    parser.add_argument(
        "--disagreement_on",
        type=str,
        default="reward",
        choices=["latent", "reward"],
        help="Whether to compute disagreement on reward or next latent.",
    )

    # ---- abnormality adapter (scenario 2/3) ----
    parser.add_argument(
        "--adapter_ckpt",
        type=str,
        default=None,
        help="Path to AbnormalityAdapter state_dict. If set, imagined rollouts use adapter-corrected dynamics/reward.",
    )
    parser.add_argument(
        "--adapter_apply_to",
        type=str,
        default="reward",
        choices=["latent", "reward", "both"],
        help="Whether adapter corrects latent transitions, rewards, or both during imagination.",
    )
    parser.add_argument(
        "--disagreement_after_adapter",
        action="store_true",
        help="If set, compute ensemble disagreement on adapter-corrected predictions (slower but more consistent).",
    )
    # ---- encoder / warm-start policy ----
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        default=None,
        help="Path to student HistoryEncoder ckpt. If None, falls back to 'student_encoder_post_finetuning.pth'.",
    )
    parser.add_argument(
        "--policy_ckpt",
        type=str,
        default="policy_head_from_policy.pth",
        help="Path to model-free SAC policy ckpt for actor warm-start.",
    )

    # ---- training lengths ----
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of Dreamer epochs.",
    )
    parser.add_argument(
        "--step_per_epoch",
        type=int,
        default=3200,
        help="Approximate number of imagined steps per epoch "
             "(used to set imagination updates via B * H * updates ≈ step_per_epoch).",
    )
    parser.add_argument(
        "--episode_per_test",
        type=int,
        default=1,
        help="Episodes per test env during evaluation.",
    )
    parser.add_argument(
        "--num_train_envs",
        type=int,
        default=8,
        help="Imagination batch size B (number of parallel latent trajectories).",
    )
    parser.add_argument(
        "--num_test_envs",
        type=int,
        default=1,
        help="Number of parallel eval environments.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="log_mb_v2_prior_gru",
        help="Base directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )

    # ---- evaluation env id(s) ----
    parser.add_argument(
        "--test_env_id",
        type=int,
        default=151,
        help="Base env_id for eval envs (151, 161, ... if num_test_envs > 1).",
    )

    # ---- real env mixing for WM updates (TRAIN digital twins only) ----
    parser.add_argument(
        "--num_real_train_envs",
        type=int,
        default=2,
        help="Number of parallel real environments used for WM updates.",
    )
    parser.add_argument(
        "--train_env_base",
        type=int,
        default=0,
        help="Base env_id for training envs (env_id = train_env_base + i).",
    )
    parser.add_argument(
        "--real_steps_per_epoch",
        type=int,
        default=40,
        help="Number of real environment steps collected per epoch.",
    )

    # ---- online WM optimizer (real-only) ----
    parser.add_argument(
        "--wm_buffer_size",
        type=int,
        default=4000,
        help="Replay buffer size for real→latent transitions used to update WM.",
    )
    parser.add_argument(
        "--wm_lr",
        type=float,
        default=1e-4,
        help="Learning rate for online WM optimizer.",
    )
    parser.add_argument(
        "--wm_wd",
        type=float,
        default=0.0,
        help="Weight decay for online WM optimizer.",
    )
    parser.add_argument(
        "--wm_updates_per_epoch",
        type=int,
        default=64,
        help="Number of WM gradient steps per epoch on real-latent buffer.",
    )
    parser.add_argument(
        "--wm_batch_size",
        type=int,
        default=32,
        help="Minibatch size for WM updates.",
    )
    parser.add_argument(
        "--wm_grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value for WM updates.",
    )
    parser.add_argument(
        "--decoder_path",
        type=str,
        default=None,
        help="Path to 3D decoder (for geometric_gru geometric loss).",
    )

    # ---- Dreamer-specific ----
    parser.add_argument(
        "--lambda_",
        type=float,
        default=0.95,
        help="Lambda for TD(λ) returns in actor/critic updates.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for actor and value networks.",
    )
    parser.add_argument("--adapter_updates_per_epoch", type=int, default=50)
    parser.add_argument("--adapter_lr", type=float, default=1e-4)
    parser.add_argument("--adapter_batch_size", type=int, default=128)
    parser.add_argument("--adapter_grad_clip", type=float, default=1.0)
    parser.add_argument("--adapter_z_coef", type=float, default=1.0)
    parser.add_argument("--adapter_r_coef", type=float, default=1.0)
    parser.add_argument("--adapter_reg_coef", type=float, default=0.0)
    parser.add_argument("--adapter_ensemble_loss", type=str, default="avg", choices=["avg", "sample"])
    parser.add_argument("--adapter_drop_nan", dest="adapter_drop_nan", action="store_true")
    parser.add_argument("--no_adapter_drop_nan", dest="adapter_drop_nan", action="store_false")
    parser.set_defaults(adapter_drop_nan=True)
    parser.add_argument("--adapter_seq_len", type=int, default=10)

    parser.add_argument("--adapter_drop_zero_z", dest="adapter_drop_zero_z", action="store_true")
    parser.add_argument("--no_adapter_drop_zero_z", dest="adapter_drop_zero_z", action="store_false")
    parser.set_defaults(adapter_drop_zero_z=True)
    parser.add_argument("--ent_coef", type=float, default=1e-5,
                    help="Entropy bonus coefficient for Dreamer actor loss.")
    args = parser.parse_args()
    main(args)
