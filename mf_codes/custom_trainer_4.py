# custom_trainer_4.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import SACPolicy
from tianshou.policy.modelfree.sac import SACTrainingStats
from tianshou.utils.conversion import to_optional_float

# Your modules
from nets import StudentDistillationNetwork, CriticEncoder, CURL


@dataclass(kw_only=True)
class DistillationTrainingStats(SACTrainingStats):
    distillation_loss: float | None = None
    curl_loss: float | None = None
    proj_nce_loss: float | None = None
    value_align_loss: float | None = None
    student_q_loss: float | None = None   # monitor student critic regression to teacher targets


class DistillationSACPolicy(SACPolicy):
    """
    SAC with:
      - Teacher critics (privileged) updated as usual.
      - Student critics (public-only) trained to regress to teacher TD targets.
      - Actor gradient computed w.r.t. student critics (realizable under partial obs).
      - Optional representation distillation (teacher 3D latent -> student history latent).
      - Optional CURL and value-alignment auxiliary losses.
    """

    def __init__(
        self,
        # --- student-critic branch (public-only critics for actor gradient) ---
        student_critic: torch.nn.Module,
        student_critic2: torch.nn.Module,
        student_critic_optim: torch.optim.Optimizer,
        student_critic2_optim: torch.optim.Optimizer,

        # --- Distillation / representation pieces ---
        student_distill_net: StudentDistillationNetwork,  # student: HistoryEncoder (+ head)
        critic1_encoder: CriticEncoder,                   # teacher: has .teacher_encoder (3D)
        critic2_encoder: CriticEncoder,
        curl_module: Optional[CURL] = None,
        teacher_proj: Optional[torch.nn.Module] = None,   # should be FROZEN (requires_grad=False)
        student_proj: Optional[torch.nn.Module] = None,   # trainable
        value_match_head: Optional[torch.nn.Module] = None,

        # --- Schedules ---
        distill_start_epoch: int = 5,
        distill_end_epoch: int = 15,
        curl_start_epoch: int = 5,
        curl_end_epoch: int = 15,

        # --- Final weights for auxiliary losses ---
        distillation_loss_weight: float = 0.2,
        curl_loss_weight: float = 1.0,

        # --- Distill internals ---
        contrastive_tau: float = 0.07,
        value_loss_coef: float = 0.2,

        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        # Devices
        self.device = next(self.actor.parameters()).device

        # Student critics (public-only)
        self.student_critic = student_critic
        self.student_critic2 = student_critic2
        self.student_critic_optim = student_critic_optim
        self.student_critic2_optim = student_critic2_optim

        # Distillation / representation
        self.student_distill_net = student_distill_net
        self.critic1_encoder = critic1_encoder
        self.critic2_encoder = critic2_encoder
        self.curl_module = curl_module
        self.teacher_proj = teacher_proj
        self.student_proj = student_proj
        self.value_match_head = value_match_head

        # Schedules / weights
        self.distill_start_epoch = distill_start_epoch
        self.distill_end_epoch = distill_end_epoch
        self.curl_start_epoch = curl_start_epoch
        self.curl_end_epoch = curl_end_epoch
        self.final_distill_weight = distillation_loss_weight
        self.final_curl_weight = curl_loss_weight

        # Internals
        self.contrastive_tau = contrastive_tau
        self.value_loss_coef = value_loss_coef
        self.current_epoch = 0  # trainer sets this via train_fn

        # Loss helpers
        self._ce = torch.nn.CrossEntropyLoss()

    # ===== Utilities =====

    def _get_current_weight(self, start_epoch: int, end_epoch: int, final_weight: float) -> float:
        """Linear ramp from 0 to final_weight between [start_epoch, end_epoch]."""
        if self.current_epoch < start_epoch:
            return 0.0
        if self.current_epoch >= end_epoch:
            return final_weight
        progress = (self.current_epoch - start_epoch) / max(1, (end_epoch - start_epoch))
        return progress * final_weight

    def _contrastive_info_nce(self, s_proj: torch.Tensor, t_proj: torch.Tensor) -> torch.Tensor:
        """InfoNCE between L2-normalized student/teacher projections."""
        logits = (s_proj @ t_proj.T) / self.contrastive_tau
        labels = torch.arange(logits.size(0), device=logits.device)
        return self._ce(logits, labels)

    # ===== Core training step =====

    def learn(self, batch: Batch, **kwargs: Any) -> DistillationTrainingStats:
        """
        One training step:
          1) Update teacher critics via SAC's standard critic update (privileged).
          2) Build TD target using *target* teacher critics and actor(next_obs).
          3) Train student critics to regress to that target (public-only).
          4) Actor update: use student critics for policy gradient.
          5) Auxiliary: distillation, CURL, value-alignment.
          6) Alpha update (if auto).
          7) Soft-update targets.
        """
        # ------------------------------------------------------------
        # 1) TEACHER CRITIC UPDATES (privileged) via standard SAC
        # ------------------------------------------------------------
        td1, critic1_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        td2, critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)

        # ------------------------------------------------------------
        # 2) BUILD TEACHER TD TARGET with target critics
        # ------------------------------------------------------------
        with torch.no_grad():
            # forward needs a dummy info field
            next_out: Batch = self.forward(Batch(obs=batch.obs_next, info=Batch()))
            a_next = next_out.act
            logp_next = next_out.log_prob

            # target critics in Tianshou are critic_old / critic2_old
            q_next1 = self.critic_old(batch.obs_next, a_next).flatten()
            q_next2 = self.critic2_old(batch.obs_next, a_next).flatten()
            q_next_min = torch.min(q_next1, q_next2)

            # >>> convert rew/done to tensors (float32) on self.device
            rew  = torch.as_tensor(batch.rew,  device=self.device, dtype=torch.float32).flatten()
            done = torch.as_tensor(batch.done, device=self.device, dtype=torch.float32).flatten()

            target_q = rew + (1.0 - done) * self.gamma * (
                q_next_min - self.alpha * logp_next.flatten()
            )
        # ------------------------------------------------------------
        # 3) STUDENT CRITICS (public-only) REGRESS TO TEACHER TARGET
        # ------------------------------------------------------------
        q1s = self.student_critic(batch.obs, batch.act).flatten()
        q2s = self.student_critic2(batch.obs, batch.act).flatten()
        student_q_loss = F.mse_loss(q1s, target_q) + F.mse_loss(q2s, target_q)

        self.student_critic_optim.zero_grad()
        self.student_critic2_optim.zero_grad()
        student_q_loss.backward()
        self.student_critic_optim.step()
        self.student_critic2_optim.step()

        # ------------------------------------------------------------
        # 4) ACTOR UPDATE *USING STUDENT CRITICS*
        # ------------------------------------------------------------
        obs_out: Batch = self.forward(Batch(obs=batch.obs, info=Batch()))
        act = obs_out.act
        logp = obs_out.log_prob

        q1a_s = self.student_critic(batch.obs, act).flatten()
        q2a_s = self.student_critic2(batch.obs, act).flatten()
        min_q_s = torch.min(q1a_s, q2a_s)

        actor_rl_loss = (self.alpha * logp.flatten() - min_q_s).mean()

        # ------------------------------------------------------------
        # 5) AUXILIARY LOSSES: Distillation + CURL + Value Alignment
        # ------------------------------------------------------------
        current_distill_w = self._get_current_weight(
            self.distill_start_epoch, self.distill_end_epoch, self.final_distill_weight
        )
        current_curl_w = self._get_current_weight(
            self.curl_start_epoch, self.curl_end_epoch, self.final_curl_weight
        )

        # Defaults
        distill_loss = torch.zeros((), device=self.device)
        proj_nce_loss = torch.zeros((), device=self.device)
        value_align_loss = torch.zeros((), device=self.device)
        curl_loss = torch.zeros((), device=self.device)

        # --- DISTILLATION (teacher 3D latent -> student history latent) ---
        if current_distill_w > 0:
            # public inputs
            hist_gpu = torch.as_tensor(batch.obs.history, device=self.device, dtype=torch.float32)
            well_gpu = torch.as_tensor(batch.obs.well_observations, device=self.device, dtype=torch.float32)
            # privileged input
            res_gpu = torch.as_tensor(batch.obs.res_state, device=self.device, dtype=torch.float32)

            # student latent (trainable)
            z_student = self.student_distill_net(hist_gpu, well_gpu)  # [B, d_model]

            # teacher latent (NO grad)
            with torch.no_grad():
                z_teacher = self.critic1_encoder.teacher_encoder(res_gpu)  # [B, d_model]

            # projection-based InfoNCE if both projections provided
            if (self.student_proj is not None) and (self.teacher_proj is not None):
                s_proj = F.normalize(self.student_proj(z_student), dim=-1)
                with torch.no_grad():
                    t_proj = F.normalize(self.teacher_proj(z_teacher), dim=-1)
                proj_nce_loss = self._contrastive_info_nce(s_proj, t_proj)
                distill_loss = proj_nce_loss
            else:
                # cosine distance on normalized latents
                zs = F.normalize(z_student, dim=-1)
                with torch.no_grad():
                    zt = F.normalize(z_teacher, dim=-1)
                distill_loss = (1.0 - (zs * zt).sum(dim=-1)).mean()
                proj_nce_loss = distill_loss  # for logging consistency

            # Optional value alignment head on student latent
            if self.value_match_head is not None:
                v_pred = self.value_match_head(z_student).squeeze(-1)  # [B]
                v_targ = min_q_s.detach()
                value_align_loss = F.smooth_l1_loss(v_pred, v_targ)
                distill_loss = distill_loss + self.value_loss_coef * value_align_loss

        # --- CURL on public inputs only ---
        if current_curl_w > 0 and self.curl_module is not None:
            hist_gpu = torch.as_tensor(batch.obs.history, device=self.device, dtype=torch.float32)
            well_gpu = torch.as_tensor(batch.obs.well_observations, device=self.device, dtype=torch.float32)
            curl_loss = self.curl_module(history=hist_gpu, well_observations=well_gpu)
            self.curl_module._update_momentum_encoder()

        # --- Combine total actor loss ---
        combined_actor_loss = actor_rl_loss \
                            + current_distill_w * distill_loss \
                            + current_curl_w * curl_loss

        self.actor_optim.zero_grad()
        combined_actor_loss.backward()
        if getattr(self, "_grad_norm", 0) and self._grad_norm > 0:
            params =[]
            for g in self.actor_optim.param_groups:
                for p in g["params"]:
                    if p.grad is not None:
                        params.append(p)
            torch.nn.utils.clip_grad_norm_(params, max_norm=self._grad_norm)
        self.actor_optim.step()

        # ------------------------------------------------------------
        # 6) ALPHA UPDATE (entropy coef) if auto
        # ------------------------------------------------------------
        alpha_loss = None
        if self.is_auto_alpha:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()

        # ------------------------------------------------------------
        # 7) SOFT-UPDATE TARGET NETWORKS
        # ------------------------------------------------------------
        self.sync_weight()

        # ------------------------------------------------------------
        # Return stats
        # ------------------------------------------------------------
        return DistillationTrainingStats(
            actor_loss=actor_rl_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            alpha=to_optional_float(self.alpha),
            alpha_loss=to_optional_float(alpha_loss),
            distillation_loss=distill_loss.item(),
            curl_loss=curl_loss.item(),
            proj_nce_loss=proj_nce_loss.item(),
            value_align_loss=value_align_loss.item(),
            student_q_loss=student_q_loss.item(),
        )