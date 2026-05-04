# dream_env.py

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import random
from typing import List, Optional

from world_models import EnsembleWorldModel


class DreamEnv(gym.Env):
    """
    A fast latent-space environment driven by a trained world-model ensemble.
    Works with both GRU-based (stateful) and diffusion-based (stateless) models.

    Observations: latent z_t  (shape = [latent_dim], dtype=float32)
    Actions:      same Box(-1, 1, (11,)) as the real env (so policies are reusable)
    Rewards:      predicted by the world model
    Episode length: fixed 'dream_horizon' steps (no truncation by default)

    Notes
    -----
    - For GRU models, we maintain an internal hidden state 'h'.
    - For diffusion models, 'h' is None and ignored by the model.
    - The world model is expected to have been trained on (z_t, a_t) -> (distribution over z_{t+1}, reward).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Gymnasium API
    def __init__(
        self,
        world_model: EnsembleWorldModel,
        initial_states: List[torch.Tensor],
        action_dim: int = 11,
        dream_horizon: int = 20,
        device: str | torch.device = "cpu",
        gru_deterministic: bool = False,
        diffusion_steps: Optional[int] = None,
        use_ema_at_eval: bool = True,
        disagreement_coef: float = 0.0,
        disagreement_on: str = "latent",
    ):
        super().__init__()
        assert len(initial_states) > 0, "initial_states must be non-empty"

        self.world_model = world_model
        self.world_model.eval()

        self.initial_states = initial_states
        self.latent_dim = int(initial_states[0].numel())
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.dream_horizon = int(dream_horizon)

        self.gru_deterministic = bool(gru_deterministic)
        self.diffusion_steps = diffusion_steps
        self.use_ema_at_eval = use_ema_at_eval

        self.disagreement_coef = float(disagreement_coef)
        self.disagreement_on = str(disagreement_on)
        self._h_per_member = None
        # detect if model family is recurrent; support GRU, RSSM, MoE-GRU
        m0 = self.world_model.models[0]
        self._has_hidden = any([
            hasattr(m0, "hidden_dim"),  # ProbabilisticGRU
            hasattr(m0, "hidden"),      # MoEGRUWorldModel
            hasattr(m0, "deter_dim"),   # RSSMWorldModel
        ])

        # fixed ensemble member per episode to keep hidden state consistent
        self._member_idx: Optional[int] = None

        # gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.latent_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # runtime state
        self.current_z: Optional[torch.Tensor] = None
        self.current_h: Optional[torch.Tensor] = None
        self.current_step: int = 0

    @torch.no_grad()
    def _ensemble_disagreement(self, z_t: torch.Tensor, a_t: torch.Tensor):
        preds_latent, preds_reward = [], []
        for i, mi in enumerate(self.world_model.models):
            # recurrent member: use the tracked hidden
            if self._has_hidden and self._h_per_member is not None:
                h_in = self._h_per_member[i]
                if hasattr(mi, "forward"):
                    mean_i, _std_i, r_i, _ = mi.forward(z_t, a_t, h_in)
                    z_i = mean_i
                else:
                    z_i, r_i, _ = mi.sample(z_t, a_t, h_in)
            elif hasattr(mi, "num_timesteps"):
                z_i, r_i, _ = mi.sample(z_t, a_t, h=None, steps=10, use_ema=True, ddim=True)
            else:
                z_i, r_i, _ = mi.sample(z_t, a_t, h=None)

            preds_latent.append(z_i.squeeze(0))
            preds_reward.append(r_i.squeeze(0))

        if len(preds_latent) < 2:
            return torch.tensor(0.0, device=z_t.device)

        Z = torch.stack(preds_latent, dim=0)  # [E, D]
        R = torch.stack(preds_reward, dim=0)  # [E, 1]
        # use mean of per-dim std (scale-stable), not L2
        return (R.std(dim=0).abs().mean()
                if self.disagreement_on == "reward"
                else Z.std(dim=0).mean())



    @torch.no_grad()
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        self._member_idx = random.randrange(len(self.world_model.models))

        z0 = random.choice(self.initial_states).to(self.device)
        self.current_z = z0.detach().clone()
        self.current_step = 0

        if self._has_hidden:
            m = self.world_model.models[self._member_idx]
            H = getattr(m, "hidden_dim", getattr(m, "hidden", getattr(m, "deter_dim", None)))
            assert H is not None, "Cannot infer hidden size for recurrent model."
            self.current_h = torch.zeros(1, H, device=self.device)
            # keep per-member hidden states if you still want lockstep updates; otherwise remove block below
            self._h_per_member = []
            for mi in self.world_model.models:
                Hi = getattr(mi, "hidden_dim", getattr(mi, "hidden", getattr(mi, "deter_dim", None)))
                self._h_per_member.append(torch.zeros(1, Hi, device=self.device) if Hi is not None else None)
        else:
            self.current_h = None
            self._h_per_member = None

        obs = self.current_z.float().cpu().numpy().astype(np.float32)
        return obs, {}

    @torch.no_grad()
    def step(self, action: np.ndarray):
        a_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        z_t = self.current_z.unsqueeze(0)
        m = self.world_model.models[self._member_idx]

        # 1) World-model rollout (no shaping)
        if self._has_hidden:
            if self.gru_deterministic and hasattr(m, "forward"):
                mean, _std, r, h_next = m.forward(z_t, a_t, self.current_h)
                z_next = mean
            else:
                z_next, r, h_next = m.sample(z_t, a_t, self.current_h)
        else:
            steps = self.diffusion_steps if hasattr(m, "num_timesteps") else None
            if steps is None:
                z_next, r, _ = m.sample(z_t, a_t, h=None)
            else:
                z_next, r, _ = m.sample(z_t, a_t, h=None, steps=steps, use_ema=True, ddim=True)
            h_next = None

        # 2) Update latent/hidden
        self.current_z = z_next.squeeze(0).detach()
        self.current_h = h_next

        # (optional) keep other ensemble members' hidden in lockstep
        if self._has_hidden and self._h_per_member is not None:
            self._h_per_member[self._member_idx] = h_next if h_next is not None else self._h_per_member[self._member_idx]
            for i, mi in enumerate(self.world_model.models):
                if i == self._member_idx or self._h_per_member[i] is None:
                    continue
                h_i = self._h_per_member[i]
                if hasattr(mi, "forward"):
                    _mean_i, _std_i, _r_i, h_i_next = mi.forward(z_t, a_t, h_i)
                else:
                    _z_i, _r_i, h_i_next = mi.sample(z_t, a_t, h_i)
                self._h_per_member[i] = h_i_next

        self.current_step += 1

        # 3) Reward = raw model reward - disagreement penalty
        raw_model = float(r.squeeze().item())

        # # step index is 1-based here (current_step was incremented earlier)
        t = self.current_step

        # ---- step-dependent cap ----
        if t <= 2:          # steps 1..2
            cap = 2.2
        elif t <= 10:       # steps 3..10
            cap = 1.8
        elif t <= 13:       # steps 11..13
            cap = 1.3
        else:               # steps 14..horizon
            cap = 1.0

        # apply cap
        raw_sched = min(raw_model, cap)

        # ---- "fear" rule: if the model predicts above stricter thresholds, zero it ----
        if (t <= 2 and raw_model > 2.2) \
        or (3 <= t <= 10 and raw_model >= 1.9) \
        or (11 <= t <= 13 and raw_model >= 1.4) \
        or (t >= 14 and raw_model >= 1.2):
            raw_sched = 0.0

        # ---- early-steps floor ----
        if t <= 13 and raw_sched < 0.7:
            raw_sched = 0.0

        # final pre-penalty reward
        raw = raw_sched

        # disagreement penalty (unchanged)
        dis_val = 0.0
        if self.disagreement_coef > 0.0:
            dis = self._ensemble_disagreement(z_t, a_t)
            dis_val = float(dis.item())

        rew = raw - self.disagreement_coef * dis_val

        latent_norm = float(torch.linalg.norm(self.current_z).item())
        too_uncertain = (dis_val > 0.75)
        too_large_lat = (latent_norm > 30.)
        terminated = (self.current_step >= self.dream_horizon) or too_uncertain or too_large_lat
        truncated = False

        obs = self.current_z.float().cpu().numpy().astype(np.float32)
        info = {
            "raw_model": raw,
            "disagreement": dis_val,
        }
        return obs, rew, terminated, truncated, info

    def render(self):
        # No meaningful rendering in latent space; print a cheap diagnostic
        if self.current_z is not None:
            norm = torch.linalg.norm(self.current_z).item()
            print(f"[DreamEnv] step={self.current_step} ||z||={norm:.3f}")
