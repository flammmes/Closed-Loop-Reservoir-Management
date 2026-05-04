# world_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import random
from torch.func import jvp  # <- use JVP instead of full Jacobian
import numpy as np
# ===================================================================
# MODEL 1: PROBABILISTIC GRU
# ===================================================================

class ProbabilisticGRU(nn.Module):
    """
    Probabilistic world model with a GRU core.
    Predicts a Normal(mean, std^2) over z_{t+1} and a deterministic reward.
    """
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.fc_state_output = nn.Linear(hidden_dim, latent_dim * 2)
        self.fc_reward_output = nn.Linear(hidden_dim, 1)
    def step(self, z_t, a_t, h):
        x = torch.relu(self.input_layer(torch.cat([z_t, a_t], dim=-1)))
        h_next = self.gru(x, h)
        reward = self.fc_reward_output(h_next)
        state_params = self.fc_state_output(h_next)
        mean, raw = torch.chunk(state_params, 2, dim=-1)
        std = F.softplus(raw) + 1e-3
        return mean, std, reward, h_next

    def sequence_loss(self, z_seq, a_seq, r_seq, z_next_seq, h0=None):
        """
        z_seq:      [B, T, D]
        a_seq:      [B, T, A]
        r_seq:      [B, T, 1]
        z_next_seq: [B, T, D]
        h0:         [B, H] or None -> zeros

        Returns scalar loss over all steps in the sequence(s).
        """
        B, T, D = z_seq.shape
        if h0 is None:
            h = torch.zeros(B, self.hidden_dim, device=z_seq.device)
        else:
            h = h0

        total_nll = 0.0
        total_r   = 0.0
        steps     = 0

        for t in range(T):
            z_t = z_seq[:, t, :]
            a_t = a_seq[:, t, :]
            r_t = r_seq[:, t, :]
            z_next = z_next_seq[:, t, :]

            mean, std, r_pred, h = self.step(z_t, a_t, h)

            var = std.pow(2).clamp_min(1e-6)
            nll_t = F.gaussian_nll_loss(mean, z_next, var, reduction="mean")
            r_loss_t = F.mse_loss(r_pred, r_t)

            total_nll = total_nll + nll_t
            total_r   = total_r   + r_loss_t
            steps += 1

        return (total_nll + total_r) / max(1, steps)
    def forward(self, z: torch.Tensor, a: torch.Tensor, h: torch.Tensor):
        """
        z: [B, D], a: [B, A], h: [B, H]
        returns: mean [B, D], std [B, D], reward [B, 1], h_next [B, H]
        """
        x = torch.relu(self.input_layer(torch.cat([z, a], dim=-1)))
        h_next = self.gru(x, h)
        reward = self.fc_reward_output(h_next)
        state_params = self.fc_state_output(h_next)
        mean, raw = torch.chunk(state_params, 2, dim=-1)
        # Keep variance well-conditioned
        std = F.softplus(raw) + 1e-3
        return mean, std, reward, h_next

    def calculate_loss(self, z_next_clean, z_t, a_t, r_t, h):
        """Gaussian NLL for state + MSE for reward."""
        mean, std, reward_pred, _ = self.forward(z_t, a_t, h)
        var = std.pow(2)
        nll = F.gaussian_nll_loss(mean, z_next_clean, var, reduction="mean")
        r_loss = F.mse_loss(reward_pred, r_t)
        return nll + r_loss

    def calculate_geometric_loss(self, z_next_clean, z_t, a_t, r_t, h, decoder: nn.Module):
        """
        Geometry-aware loss with a proper latent anchor:

          total = NLL_state
                + λ_geom * E[|| J_dec(z_eval) @ (mean - z*) ||^2]
                + MSE_reward

        where the expectation E[...] is approximated on a *random subset* of
        batch elements to keep memory usage reasonable.

        Args:
            z_next_clean: [N, D] target latents
            z_t:          [N, D] current latents
            a_t:          [N, A] actions
            r_t:          [N, 1] rewards
            h:            [N, H] GRU hidden (or zeros)
            decoder:      frozen 3D decoder used only to induce the metric
        """
        # ----- forward pass in latent -----
        mean, std, r_pred, _ = self.forward(z_t, a_t, h)  # [N, D], [N, D], [N, 1]
        e_full = (mean - z_next_clean)                    # [N, D]
        var = std.pow(2).clamp_min(1e-6)

        # (1) Latent NLL anchor over the FULL batch
        nll = F.gaussian_nll_loss(mean, z_next_clean, var, reduction="mean")

        # (2) Reward loss over the FULL batch
        r_loss = F.mse_loss(r_pred, r_t)

        # (3) Pullback (geometry) term via JVP on a random subset
        N = mean.size(0)
        device = mean.device

        # How many samples in this batch to use for the geometric term
        max_geo_samples = 32  # tune 16–64 depending on GPU
        if N > max_geo_samples:
            idx = torch.randperm(N, device=device)[:max_geo_samples]
            z_eval = mean[idx].detach()    # [N_geo, D], metric evaluated at prediction
            e      = e_full[idx]           # [N_geo, D], error direction
            # optional scaling to keep expectation roughly unbiased
            scale = float(N) / float(max_geo_samples)
        else:
            z_eval = mean.detach()
            e      = e_full
            scale  = 1.0

        def jvp_single(z_vec, v_vec):
            # function whose output we care about: decoded 3D field
            def dec_fn(zz):
                return decoder(zz.unsqueeze(0)).squeeze(0)
            # jvp returns (value, tangent); we only keep the tangent
            _, je = jvp(dec_fn, (z_vec,), (v_vec,))
            return je.reshape(-1)  # flatten spatial dims

        # process the subset in very small chunks to avoid OOM
        chunk_size = 4  # VERY small; tune if you have more memory
        geom_chunks = []
        N_geo = z_eval.size(0)

        for start in range(0, N_geo, chunk_size):
            end = min(start + chunk_size, N_geo)
            z_chunk = z_eval[start:end]  # [b, D]
            e_chunk = e[start:end]       # [b, D]

            Je_chunk = torch.vmap(jvp_single)(z_chunk, e_chunk)  # [b, M]
            M = Je_chunk.shape[1]
            # average squared pullback norm over this chunk
            geom_chunk = (Je_chunk.pow(2).sum(dim=-1) / float(M)).mean()
            geom_chunks.append(geom_chunk)

        if len(geom_chunks) > 0:
            geom_state = torch.stack(geom_chunks).mean() * scale
        else:
            geom_state = torch.tensor(0.0, device=device)

        # (4) Combine
        LAMB_GEOM = 0.05  # you can tune 0.01–0.1
        total = nll + LAMB_GEOM * geom_state + r_loss
        return total

    @torch.no_grad()
    def sample(self, z_t, a_t, h):
        """Sample z_{t+1} ~ Normal(mean, std^2) and return reward, next hidden."""
        mean, std, reward, h_next = self.forward(z_t, a_t, h)
        z_next = mean + torch.randn_like(mean) * std
        return z_next, reward, h_next


# =========================
# 1) Dreamer-style RSSM
# =========================
class RSSMWorldModel(nn.Module):
    """
    Stochastic (z_s) + deterministic (h) dynamics:
      h_{t+1} = GRU( phi([z_t, a_t]), h_t )
      p(z_s|h)  and  q(z_s|h, z_next)
      z_next_hat = g([h_{t+1}, z_s])   (predict the observed latent)
      r_hat      = r([h_{t+1}, z_s])
    Loss = MSE(z_next_hat, z_next) + beta*KL(q||p) + MSE(r_hat, r)
    """
    def __init__(self, latent_dim: int, action_dim: int,
                 deter_dim: int = 512, stoch_dim: int = 32, beta: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.beta = beta

        self.inp = nn.Sequential(
            nn.Linear(latent_dim + action_dim, deter_dim), nn.SiLU(),
        )
        self.gru = nn.GRUCell(deter_dim, deter_dim)

        # prior p(z_s | h): diagonal Normal
        self.prior = nn.Sequential(
            nn.Linear(deter_dim, 2 * stoch_dim)
        )
        # posterior q(z_s | h, z_next): diagonal Normal
        self.post = nn.Sequential(
            nn.Linear(deter_dim + latent_dim, 2 * stoch_dim)
        )

        # reconstruct observed latent and reward
        self.obs_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 512), nn.SiLU(),
            nn.Linear(512, latent_dim),
        )
        self.r_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 256), nn.SiLU(),
            nn.Linear(256, 1),
        )

    def _split(self, x):
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mean, log_std

    def _kl_diag(self, m_q, s_q, m_p, s_p):
        # KL( N(m_q, s_q^2) || N(m_p, s_p^2) ) diagonal
        var_q = torch.exp(2*s_q); var_p = torch.exp(2*s_p)
        kl = ( (var_q + (m_q - m_p)**2) / var_p + 2*(s_p - s_q) - 1 ).sum(dim=-1) * 0.5
        return kl.mean()

    def _step_dynamics(self, z_t, a_t, h):
        x = self.inp(torch.cat([z_t, a_t], dim=-1))
        h_next = self.gru(x, h)
        return h_next

    def sequence_loss(self, z_seq, a_seq, r_seq, z_next_seq, h0=None):
        """
        Multi-step RSSM loss over a sequence.

        z_seq:      [B, T, latent_dim]
        a_seq:      [B, T, action_dim]
        r_seq:      [B, T, 1]
        z_next_seq: [B, T, latent_dim]
        h0:         [B, deter_dim] or None -> zeros

        Returns a scalar loss averaged over time and batch:
            mean_t ( MSE(z_pred_t, z_next_t) + MSE(r_pred_t, r_t) + beta * KL_t )
        """
        B, T, D = z_seq.shape
        device = z_seq.device

        if h0 is None:
            h = torch.zeros(B, self.deter_dim, device=device)
        else:
            h = h0

        total_recon = 0.0
        total_r = 0.0
        total_kl = 0.0
        steps = 0

        for t in range(T):
            z_t = z_seq[:, t, :]          # [B, D]
            a_t = a_seq[:, t, :]          # [B, A]
            r_t = r_seq[:, t, :]          # [B, 1]
            z_next = z_next_seq[:, t, :]  # [B, D]

            # deterministic dynamics
            h = self._step_dynamics(z_t, a_t, h)   # h_{t+1}

            # prior p(z_s|h)
            prior_stats = self.prior(h)
            m_p, ls_p = self._split(prior_stats)

            # posterior q(z_s|h, z_next)
            post_input = torch.cat([h, z_next], dim=-1)
            post_stats = self.post(post_input)
            m_q, ls_q = self._split(post_stats)

            # reparameterize z_s ~ q
            z_s = m_q + torch.randn_like(m_q) * torch.exp(ls_q)

            # recon + reward
            feat = torch.cat([h, z_s], dim=-1)
            z_pred = self.obs_head(feat)
            r_pred = self.r_head(feat)

            recon = F.mse_loss(z_pred, z_next)      # averaged over batch
            r_loss = F.mse_loss(r_pred, r_t)        # averaged over batch
            kl = self._kl_diag(m_q, ls_q, m_p, ls_p)  # averaged over batch

            total_recon += recon
            total_r += r_loss
            total_kl += kl
            steps += 1

        return (total_recon + total_r + self.beta * total_kl) / max(1, steps)

    def calculate_loss(self, z_next_clean, z_t, a_t, r_t, h=None):
        B = z_t.size(0)
        if h is None:
            h = torch.zeros(B, self.deter_dim, device=z_t.device)

        h_next = self._step_dynamics(z_t, a_t, h)

        # prior p(z_s|h_next)
        prior_stats = self.prior(h_next)
        m_p, ls_p = self._split(prior_stats)

        # posterior q(z_s|h_next, z_next)
        post_input = torch.cat([h_next, z_next_clean], dim=-1)
        post_stats = self.post(post_input)
        m_q, ls_q = self._split(post_stats)

        # reparameterize
        z_s = m_q + torch.randn_like(m_q) * torch.exp(ls_q)

        # recon and reward
        feat = torch.cat([h_next, z_s], dim=-1)
        z_pred = self.obs_head(feat)
        r_pred = self.r_head(feat)

        # losses
        recon = F.mse_loss(z_pred, z_next_clean)
        r_loss = F.mse_loss(r_pred, r_t)
        kl = self._kl_diag(m_q, ls_q, m_p, ls_p)
        return recon + r_loss + self.beta * kl

    def sample(self, z_t, a_t, h=None):
        B = z_t.size(0)
        if h is None:
            h = torch.zeros(B, self.deter_dim, device=z_t.device)
        h_next = self._step_dynamics(z_t, a_t, h)
        m_p, ls_p = self._split(self.prior(h_next))
        z_s = m_p  # mean sample for determinism
        feat = torch.cat([h_next, z_s], dim=-1)
        z_pred = self.obs_head(feat)
        r_pred = self.r_head(feat)
        return z_pred, r_pred, h_next

# =========================
# 3) Koopman + residual
# =========================
class KoopmanWorldModel(nn.Module):
    """
    Linear controlled dynamics in latent with small residual:
      z_{t+1} ≈ A z_t + B a_t + res(z_t, a_t)
    Very stable; residual keeps flexibility.
    """
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256, res_scale: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        # A and B initialized near-zero for stability
        self.A = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        self.B = nn.Parameter(torch.zeros(latent_dim, action_dim))
        self.res = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.res_scale = res_scale
        self.r_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256), nn.SiLU(),
            nn.Linear(256, 1),
        )

    def _predict_next(self, z_t, a_t):
        lin = z_t @ self.A.t() + a_t @ self.B.t()
        res = self.res_scale * self.res(torch.cat([z_t, a_t], dim=-1))
        return lin + res

    def calculate_loss(self, z_next_clean, z_t, a_t, r_t, h=None):
        z_pred = self._predict_next(z_t, a_t)
        r_pred = self.r_head(torch.cat([z_pred, a_t], dim=-1))
        z_loss = F.mse_loss(z_pred, z_next_clean)
        r_loss = F.mse_loss(r_pred, r_t)
        # small spectral / Frobenius regularization to keep A,B tame
        reg = 1e-4 * (self.A.pow(2).sum() + self.B.pow(2).sum())
        return z_loss + r_loss + reg

    @torch.no_grad()
    def sample(self, z_t, a_t, h=None):
        z_pred = self._predict_next(z_t, a_t)
        r_pred = self.r_head(torch.cat([z_pred, a_t], dim=-1))
        return z_pred, r_pred, None

# ===================================================================
# ENSEMBLE WRAPPER
# ===================================================================

class EnsembleWorldModel(nn.Module):
    """
    Wrapper for an ensemble of world models (homogeneous type).
    """
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        assert len(models) > 0, "Ensemble must contain at least one model."
        self.models = nn.ModuleList(models)

    def calculate_loss(self, *args, **kwargs):
        """Average loss over models (scale independent of ensemble size)."""
        losses = []
        for m in self.models:
            losses.append(m.calculate_loss(*args, **kwargs))
        return torch.stack(losses).mean()

    def calculate_geometric_loss(self, *args, **kwargs):
        """Average geometric loss over models."""
        losses = []
        for m in self.models:
            if not hasattr(m, "calculate_geometric_loss"):
                raise NotImplementedError("Geometric loss requested but model lacks 'calculate_geometric_loss'.")
            losses.append(m.calculate_geometric_loss(*args, **kwargs))
        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        """Sample next state from one randomly chosen member."""
        idx = random.randint(0, len(self.models) - 1)
        return self.models[idx].sample(*args, **kwargs)
