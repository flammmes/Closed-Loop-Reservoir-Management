# train_abnormality_adapter_v2.py
#
# Train a residual AbnormalityAdapter on a small set of abnormal transitions (scenario 2/3).
# - Reads collector-style pickle dict: {"z","a","r","z_next","done", ...}
# - Loads a *frozen* scenario-0 world model ensemble checkpoint (state_dict of EnsembleWorldModel)
# - Learns dz/dr corrections so that (z_pred,r_pred) -> (z_next_abn,r_abn)
#
# Designed to match your Dreamer implementation where hidden state starts at zeros each rollout.

import argparse
import os
import pickle

import numpy as np
import torch
from torch import nn

from world_models import EnsembleWorldModel, ProbabilisticGRU, RSSMWorldModel, KoopmanWorldModel
from abnormality_adapter import AbnormalityAdapter

def build_windows_flat(done: np.ndarray, seq_len: int):
    """
    done: [N] bool. We EXCLUDE indices where done=True (terminal transition) from windows.
    Returns: list of start indices s, each window is s..s+seq_len-1 fully inside one episode.
    """
    N = done.shape[0]
    windows = []

    start = 0
    for i in range(N):
        if done[i]:
            end = i  # exclude terminal transition at i
            L = end - start
            if L >= seq_len:
                for s in range(start, end - seq_len + 1):
                    windows.append(s)
            start = i + 1

    # tail segment
    end = N
    L = end - start
    if L >= seq_len:
        for s in range(start, end - seq_len + 1):
            windows.append(s)

    return windows



def build_ensemble(
    model_type: str,
    latent_dim: int,
    action_dim: int,
    num_ensemble: int,
    gru_hidden: int,
    rssm_deter: int,
    rssm_stoch: int,
    rssm_beta: float,
    koop_hidden: int,
    koop_res_scale: float,
    device: torch.device,
) -> EnsembleWorldModel:
    models = []
    if model_type == "gru":
        for _ in range(num_ensemble):
            models.append(ProbabilisticGRU(latent_dim, action_dim, hidden_dim=gru_hidden).to(device))
    elif model_type == "rssm":
        for _ in range(num_ensemble):
            models.append(
                RSSMWorldModel(
                    latent_dim,
                    action_dim,
                    deter_dim=rssm_deter,
                    stoch_dim=rssm_stoch,
                    beta=rssm_beta,
                ).to(device)
            )
    elif model_type == "koopman":
        for _ in range(num_ensemble):
            models.append(
                KoopmanWorldModel(
                    latent_dim,
                    action_dim,
                    hidden=koop_hidden,
                    res_scale=koop_res_scale,
                ).to(device)
            )
    else:
        raise ValueError(f"Unknown model_type={model_type}")
    return EnsembleWorldModel(models).to(device)


@torch.no_grad()
def wm_step_ens_mean(models, z_in, a_t, model_type: str):
    """
    z_in: [B,D], a_t:[B,A]
    returns: z_pred_mean [B,D], r_pred_mean [B,1]
    Hidden is RESET each step here by design (matches your current adapter-training assumption).
    """
    B = z_in.size(0)
    Zs, Rs = [], []

    for m in models:
        if model_type == "gru":
            h0 = torch.zeros(B, m.hidden_dim, device=z_in.device)
            mean, _std, r_pred, _h1 = m.forward(z_in, a_t, h0)
            Zs.append(mean)
            Rs.append(r_pred.view(B, 1))

        elif model_type == "rssm":
            h0 = torch.zeros(B, m.deter_dim, device=z_in.device)
            z_pred, r_pred, _h1 = m.sample(z_in, a_t, h0)
            Zs.append(z_pred)
            Rs.append(r_pred.view(B, 1))

        elif model_type == "koopman":
            z_pred = m._predict_next(z_in, a_t)
            r_pred = m.r_head(torch.cat([z_pred, a_t], dim=-1)).view(B, 1)
            Zs.append(z_pred)
            Rs.append(r_pred)

        else:
            raise ValueError(model_type)

    Z = torch.stack(Zs, dim=0).mean(dim=0)  # [B,D]
    R = torch.stack(Rs, dim=0).mean(dim=0)  # [B,1]
    return Z, R


@torch.no_grad()
def init_h_ens(models, B: int, device, model_type: str):
    if model_type == "gru":
        H = models[0].hidden_dim
        return torch.zeros(len(models), B, H, device=device)
    if model_type == "rssm":
        H = models[0].deter_dim
        return torch.zeros(len(models), B, H, device=device)
    return None

@torch.no_grad()
def wm_step_ens_mean_recurrent(models, z_in, a_t, h_ens, model_type: str):
    B = z_in.size(0)
    Zs, Rs = [], []

    if model_type == "gru":
        for e, m in enumerate(models):
            mean, _std, r_pred, h_next = m.forward(z_in, a_t, h_ens[e])
            h_ens[e] = h_next
            Zs.append(mean); Rs.append(r_pred.view(B, 1))

    elif model_type == "rssm":
        for e, m in enumerate(models):
            z_pred, r_pred, h_next = m.sample(z_in, a_t, h_ens[e])
            h_ens[e] = h_next
            Zs.append(z_pred); Rs.append(r_pred.view(B, 1))

    elif model_type == "koopman":
        for m in models:
            z_pred = m._predict_next(z_in, a_t)
            r_pred = m.r_head(torch.cat([z_pred, a_t], dim=-1)).view(B, 1)
            Zs.append(z_pred); Rs.append(r_pred)

    else:
        raise ValueError(model_type)

    z_mean = torch.stack(Zs, 0).mean(0)
    r_mean = torch.stack(Rs, 0).mean(0)
    return z_mean, r_mean, h_ens

def load_abnormal_dict(pkl_path: str):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    z = np.asarray(obj["z"], dtype=np.float32)
    a = np.asarray(obj["a"], dtype=np.float32)
    r = np.asarray(obj["r"], dtype=np.float32).reshape(-1, 1)
    zn = np.asarray(obj["z_next"], dtype=np.float32)

    done = np.asarray(obj.get("done", np.zeros((len(z),), dtype=bool))).reshape(-1)
    return z, a, r, zn, done

def t_from_done(done: np.ndarray) -> np.ndarray:
    t = np.zeros_like(done, dtype=np.int64)
    cur = 0
    for i in range(len(done)):
        t[i] = cur
        cur += 1
        if bool(done[i]):
            cur = 0
    return t

def build_windows_flat(done: np.ndarray, seq_len: int) -> list[np.ndarray]:
    windows = []
    start = 0
    for i in range(len(done)):
        if bool(done[i]):
            end = i
            L = end - start + 1
            if L >= seq_len:
                for s in range(start, end - seq_len + 2):
                    windows.append(np.arange(s, s + seq_len, dtype=np.int64))
            start = i + 1
    # tail (if last episode not marked done)
    if start < len(done):
        end = len(done) - 1
        L = end - start + 1
        if L >= seq_len:
            for s in range(start, end - seq_len + 2):
                windows.append(np.arange(s, s + seq_len, dtype=np.int64))
    return windows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=20)  

    ap.add_argument("--abn_pkl", type=str, required=True, help="Abnormal transitions pickle (dict with z/a/r/z_next).")
    ap.add_argument("--wm_ckpt", type=str, required=True, help="Scenario-0 world model ensemble checkpoint (.pth).")
    ap.add_argument("--out_ckpt", type=str, required=True, help="Where to save the trained adapter (.pth).")

    ap.add_argument("--model_type", type=str, required=True, choices=["gru", "rssm", "koopman"])
    ap.add_argument("--num_ensemble", type=int, default=10)
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--action_dim", type=int, default=11)

    # Must match WM training config:
    ap.add_argument("--gru_hidden", type=int, default=512)
    ap.add_argument("--rssm_deter", type=int, default=512)
    ap.add_argument("--rssm_stoch", type=int, default=32)
    ap.add_argument("--rssm_beta", type=float, default=1.0)
    ap.add_argument("--koop_hidden", type=int, default=256)
    ap.add_argument("--koop_res_scale", type=float, default=0.1)

    # Adapter training:
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--z_coef", type=float, default=1.0)
    ap.add_argument("--r_coef", type=float, default=1.0)
    ap.add_argument("--reg_coef", type=float, default=1e-4)
    ap.add_argument("--apply_to", type=str, default="both", choices=["latent", "reward", "both"])

    ap.add_argument(
        "--ensemble_loss",
        type=str,
        default="avg",
        choices=["avg", "sample"],
        help="avg=average loss over all ensemble members (slower, best); sample=use 1 random member per batch (faster).",
    )

    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")

    # Cleaning (recommended if your env sometimes returns dummy obs after Flow failure)
    ap.add_argument("--drop_nan", action="store_true")
    ap.add_argument("--drop_zero_z", action="store_true")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")

    # ---- load abnormal transitions ----
    z, a, r, zn, done = load_abnormal_dict(args.abn_pkl)

    # ---- filtering FIRST (must keep done aligned) ----
    mask = np.ones((z.shape[0],), dtype=bool)
    if args.drop_nan:
        finite = (
            np.isfinite(z).all(axis=1)
            & np.isfinite(a).all(axis=1)
            & np.isfinite(r).all(axis=1)
            & np.isfinite(zn).all(axis=1)
        )
        mask &= finite
    if args.drop_zero_z:
        mask &= (np.abs(z).sum(axis=1) > 1e-8) & (np.abs(zn).sum(axis=1) > 1e-8)

    z, a, r, zn, done = z[mask], a[mask], r[mask], zn[mask], done[mask]

    # ---- NOW build time + windows on filtered done ----
    t_arr = t_from_done(done)
    wins = build_windows_flat(done, seq_len=args.seq_len)

    if len(wins) == 0:
        raise RuntimeError("No windows found from done. Check abnormality pickle ordering or done flags.")
    wins = np.asarray(wins, dtype=np.int64)

    W = wins.shape[0]
    perm_w = np.random.permutation(W)
    n_val_w = max(1, int(args.val_frac * W))

    val_w = torch.as_tensor(perm_w[:n_val_w], device=device, dtype=torch.long)
    tr_w  = torch.as_tensor(perm_w[n_val_w:], device=device, dtype=torch.long)

    if tr_w.numel() == 0:
        # extreme small-data fallback
        tr_w = val_w

    t_t = torch.from_numpy(t_arr.astype(np.int64)).to(device)
    N = z.shape[0]
    print(f"[adapter] loaded N={N} transitions from {args.abn_pkl} (after filtering)")

    z_t = torch.from_numpy(z).to(device)
    a_t = torch.from_numpy(a).to(device)
    r_t = torch.from_numpy(r).to(device)
    z_nt = torch.from_numpy(zn).to(device)

    # ---- frozen base WM ----
    wm = build_ensemble(
        model_type=args.model_type,
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        num_ensemble=args.num_ensemble,
        gru_hidden=args.gru_hidden,
        rssm_deter=args.rssm_deter,
        rssm_stoch=args.rssm_stoch,
        rssm_beta=args.rssm_beta,
        koop_hidden=args.koop_hidden,
        koop_res_scale=args.koop_res_scale,
        device=device,
    )
    wm.load_state_dict(torch.load(args.wm_ckpt, map_location=device), strict=True)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)

    # ---- adapter ----
    adapter = AbnormalityAdapter(
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        use_time=True,
        max_steps=20,
        time_emb_dim=16,
    ).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)

    z_loss_fn = nn.MSELoss()
    r_loss_fn = nn.SmoothL1Loss()

    # train/val split
    perm = np.random.permutation(N)
    n_val = max(1, int(args.val_frac * N))
    n_val_w = max(1, int(args.val_frac * W))
    val_w = torch.as_tensor(perm_w[:n_val_w], device=device)
    tr_w  = torch.as_tensor(perm_w[n_val_w:], device=device)
    val_idx = torch.as_tensor(perm[:n_val], device=device)
    tr_idx = torch.as_tensor(perm[n_val:], device=device)

    def run_epoch(train: bool) -> float:
        adapter.train(train)
        idx = tr_w if train else val_w
        if train:
            idx = idx[torch.randperm(idx.numel(), device=device)]

        total = 0.0
        count = 0

        for s in range(0, idx.numel(), args.batch_size):
            bw = idx[s : s + args.batch_size]
            seg = torch.as_tensor(wins[bw.detach().cpu().numpy()], device=device, dtype=torch.long)  # [B,T]
            B0, T0 = seg.shape

            zseq  = z_t[seg]     # [B,T,D]
            aseq  = a_t[seg]     # [B,T,A]
            rseq  = r_t[seg]     # [B,T,1]
            znseq = z_nt[seg]    # [B,T,D]
            tseq  = t_t[seg]     # [B,T]  (t_before for each transition)

            # init recurrent state ONCE per window-batch
            h_ens = init_h_ens(wm.models, B0, device, args.model_type)

            # start from dataset z at first step
            z_in = zseq[:, 0, :]

            loss = 0.0
            for t in range(T0):
                a_ti  = aseq[:, t, :]
                r_ti  = rseq[:, t, :]
                zn_ti = znseq[:, t, :]
                tt    = tseq[:, t]                 # [B] long

                zpred, rpred, h_ens = wm_step_ens_mean_recurrent(
                    wm.models, z_in, a_ti, h_ens, args.model_type
                )

                zcorr, rcorr = adapter(z_in, a_ti, zpred, rpred, t=tt)

                step_loss = 0.0
                if args.apply_to in ("latent", "both"):
                    step_loss = step_loss + args.z_coef * z_loss_fn(zcorr, zn_ti)
                if args.apply_to in ("reward", "both"):
                    step_loss = step_loss + args.r_coef * r_loss_fn(rcorr, r_ti)
                if args.reg_coef > 0:
                    step_loss = step_loss + args.reg_coef * (
                        (zcorr - zpred).pow(2).mean() + (rcorr - rpred).pow(2).mean()
                    )

                loss = loss + step_loss

                # closed-loop for next step, detach (no multi-step adapter credit assignment)
                z_in = zcorr.detach()

            loss = loss / float(T0)

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                opt.step()

            total += float(loss.detach().item()) * B0
            count += B0

        return total / max(1, count)

    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
    best = float("inf")

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(train=True)
        va = run_epoch(train=False)
        print(f"[adapter] epoch {ep:03d}  train={tr:.6f}  val={va:.6f}")
        if va < best:
            best = va
            torch.save(adapter.state_dict(), args.out_ckpt)

    print(f"[adapter] saved best adapter to {args.out_ckpt} (best_val={best:.6f})")


if __name__ == "__main__":
    main()
