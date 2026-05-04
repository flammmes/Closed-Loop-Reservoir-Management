import torch
from torch import nn

class AbnormalityAdapter(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden: tuple[int, int] = (256, 256),
        context_dim: int = 0,
        dz_scale: float = 1.0,
        dr_scale: float = 1.0,
        # ---- NEW ----
        use_time: bool = True,
        max_steps: int = 20,
        time_emb_dim: int = 16,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
        self.dz_scale = float(dz_scale)
        self.dr_scale = float(dr_scale)

        self.use_time = bool(use_time)
        self.max_steps = int(max_steps)
        self.time_emb_dim = int(time_emb_dim) if self.use_time else 0

        if self.use_time:
            # t in [0..max_steps-1] (we clamp)
            self.time_emb = nn.Embedding(self.max_steps + 1, self.time_emb_dim)

        in_dim = 2 * self.latent_dim + self.action_dim + 1 + self.context_dim + self.time_emb_dim
        h1, h2 = hidden
        self.norm = nn.LayerNorm(in_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.SiLU(),
            nn.Linear(h2, self.latent_dim + 1),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t, a_t, z_pred, r_pred, context=None, t=None):
        if z_t.dim() == 1: z_t = z_t.unsqueeze(0)
        if a_t.dim() == 1: a_t = a_t.unsqueeze(0)
        if z_pred.dim() == 1: z_pred = z_pred.unsqueeze(0)
        if r_pred.dim() == 1: r_pred = r_pred.unsqueeze(0)
        if r_pred.dim() == 0: r_pred = r_pred.view(1, 1)

        feats = [z_t, a_t, z_pred, r_pred]

        if context is not None:
            if context.dim() == 1: context = context.unsqueeze(0)
            feats.append(context)

        # ---- NEW: time embedding (no memory) ----
        if self.use_time:
            if t is None:
                t = torch.zeros(z_t.size(0), device=z_t.device, dtype=torch.long)
            if not torch.is_tensor(t):
                t = torch.as_tensor(t, device=z_t.device)
            t = t.to(z_t.device).long().view(-1)
            t = torch.clamp(t, 0, self.max_steps)  # safe
            feats.append(self.time_emb(t))         # [B, time_emb_dim]

        x = torch.cat(feats, dim=-1)
        x = self.norm(x)
        out = self.net(x)

        dz = out[..., : self.latent_dim] * self.dz_scale
        dr = out[..., self.latent_dim:] * self.dr_scale

        return z_pred + dz, r_pred + dr