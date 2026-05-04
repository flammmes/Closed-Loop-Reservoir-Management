# train_world_model.py

import os
import argparse
import pickle
from typing import List

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from world_models import ProbabilisticGRU, EnsembleWorldModel, RSSMWorldModel, KoopmanWorldModel
from nets import Transposed3DCNN  # only used for geometric loss

class GRUSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, z_t, a_t, r_t, z_next, discount, seq_len=10):
        self.z_t = z_t
        self.a_t = a_t
        self.r_t = r_t
        self.z_next = z_next
        self.discount = discount.view(-1)
        self.seq_len = seq_len

        self.seqs = []
        N = self.z_t.size(0)
        start = 0
        for i in range(N):
            if self.discount[i].item() == 0.0:  # terminal at i
                end = i
                length = end - start + 1
                # only use full-length windows of size seq_len
                if length >= self.seq_len:
                    for s in range(start, end - self.seq_len + 2):
                        e = s + self.seq_len - 1
                        self.seqs.append((s, e))
                start = i + 1
        # leftover tail as another episode
        if start < N:
            end = N - 1
            length = end - start + 1
            if length >= self.seq_len:
                for s in range(start, end - self.seq_len + 2):
                    e = s + self.seq_len - 1
                    self.seqs.append((s, e))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s, e = self.seqs[idx]
        z  = self.z_t[s:e+1].clone()   # [T, D] with T == seq_len
        a  = self.a_t[s:e+1].clone()
        r  = self.r_t[s:e+1].clone()
        zn = self.z_next[s:e+1].clone()
        return z, a, r, zn

def load_transitions(path: str, device: torch.device):
    """
    Expects tuples: (z_t, z_next, a_t, r_t, discount).
    Returns CPU tensors; they are moved to `device` inside the training loop.
    """
    import pickle
    import torch

    with open(path, "rb") as f:
        transitions = pickle.load(f)

    # Unzip respecting saved order
    z_t_list, z_next_list, a_list, r_list, disc_list = zip(*transitions)

    # IMPORTANT: keep these on CPU (no .to(device) here)
    z_t    = torch.stack([torch.as_tensor(x) for x in z_t_list])       # CPU
    z_next = torch.stack([torch.as_tensor(x) for x in z_next_list])    # CPU
    a_t    = torch.stack([torch.as_tensor(x) for x in a_list])         # CPU
    r_t    = torch.stack([torch.as_tensor(x) for x in r_list])         # CPU
    if r_t.dim() == 1:
        r_t = r_t.unsqueeze(-1)  # (B, 1)
    discount = torch.stack([torch.as_tensor(x) for x in disc_list])  # shape (N,1) or (N,)

    return z_t, a_t, r_t, z_next,discount


def build_ensemble(model_type: str, latent_dim: int, action_dim: int, num_models: int, args) -> EnsembleWorldModel:
    models: List[torch.nn.Module] = []
    if model_type in ["gru", "geometric_gru"]:
        for _ in range(num_models):
            models.append(ProbabilisticGRU(latent_dim, action_dim, hidden_dim=args.gru_hidden).to(args.device))
    elif model_type == "rssm":
        for _ in range(num_models):
            models.append(RSSMWorldModel(latent_dim, action_dim,
                                         deter_dim=getattr(args, "rssm_deter", 512),
                                         stoch_dim=getattr(args, "rssm_stoch", 32),
                                         beta=getattr(args, "rssm_beta", 1.0)).to(args.device))
    elif model_type == "koopman":
        for _ in range(num_models):
            models.append(KoopmanWorldModel(latent_dim, action_dim,
                                            hidden=getattr(args, "koop_hidden", 256),
                                            res_scale=getattr(args, "koop_res_scale", 0.1)).to(args.device))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return EnsembleWorldModel(models)



def main(args):
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.device = device
    print(f"Using device: {device}")

    # ----------------------------
    # pick dataset file by name
    # ----------------------------
    data_file_map = {
        "teacher": "teacher_wm_transitions.pkl",
        "student_pre_ft": "student_pre_ft_wm_transitions.pkl",
        "student_post_ft": "student_post_ft_wm_transitions.pkl",
    }
    data_path = args.data_path if args.data_path else data_file_map[args.data_type]
    print(f"Loading transitions: {data_path}")
    z_t, a_t, r_t, z_next, discount = load_transitions(data_path, device)

    latent_dim = z_t.shape[1]
    action_dim = a_t.shape[1]
    print(f"latent_dim={latent_dim}, action_dim={action_dim}, N={z_t.size(0)}")

    episodes = []
    start = 0
    N = discount.shape[0]
    disc_flat = discount.view(-1)
    for i in range(N):
        if disc_flat[i].item() == 0.0:  # terminal
            episodes.append((start, i))
            start = i + 1
    if start < N:
        episodes.append((start, N - 1))


    # ----------------------------
    # optional decoder for geometric loss
    # ----------------------------
    decoder = None
    if args.model_type == "geometric_gru":
        decoder_path = args.decoder_path
        if decoder_path is None:
            decoder_path = "teacher_decoder.pth" if args.data_type == "teacher" else "student_decoder.pth"
        print(f"[Geometric] Loading decoder from: {decoder_path}")
        decoder = Transposed3DCNN(latent_dim=latent_dim, out_channels=2).to(device)
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        decoder.eval()

    # ----------------------------
    # dataloader
    # ----------------------------
    if args.model_type in ["gru", "geometric_gru", "rssm"]:
        dataset = GRUSequenceDataset(z_t, a_t, r_t, z_next, discount, seq_len=10)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, pin_memory=(device.type=="cuda"),
                            num_workers=args.num_workers, drop_last=True)
    else:
        dataset = TensorDataset(z_t, a_t, r_t, z_next)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, pin_memory=(device.type=="cuda"),
                            num_workers=args.num_workers, drop_last=True)    # ----------------------------
    # ensemble + optim
    # ----------------------------
    ensemble = build_ensemble(args.model_type, latent_dim, action_dim, args.num_ensemble, args)
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best = float("inf")
    save_tag = f"{args.data_type}_{args.model_type}_ens{args.num_ensemble}"
    if args.model_type == "diffusion":
        save_tag += f"_T{args.timesteps}{'_res' if args.predict_residual else ''}{'_ema' if args.use_ema else ''}"

    print("Starting offline training...")
    # For GRU-based models we might need hidden_dim for geometric loss
    gru_hidden_dim = None
    if args.model_type in ["gru", "geometric_gru"]:
        gru_hidden_dim = ensemble.models[0].hidden_dim

    for epoch in range(1, args.epochs + 1):
        ensemble.train()
        total = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()

            if args.model_type in ["gru", "geometric_gru", "rssm"]:
                # batch: (z_seq, a_seq, r_seq, z_next_seq) with shapes [B, T, ...]
                b_z_seq, b_a_seq, b_r_seq, b_zn_seq = [
                    t.to(device, non_blocking=True) for t in batch
                ]

                if args.model_type == "geometric_gru":
                    # Need pretrained decoder for pullback metric
                    assert decoder is not None, "decoder_path must be provided for geometric_gru."

                    losses = []
                    for m in ensemble.models:
                        h = torch.zeros(b_z_seq.size(0), m.hidden_dim, device=device)  # [B,H]
                        loss_m = 0.0
                        for t in range(b_z_seq.size(1)):  # T
                            z_t  = b_z_seq[:, t]
                            a_t  = b_a_seq[:, t]
                            r_t  = b_r_seq[:, t]
                            z_tp1 = b_zn_seq[:, t]

                            # one GRU step to advance hidden (needed!)
                            _, _, _, h_next = m.forward(z_t, a_t, h)

                            # geometric loss evaluated using the correct h
                            loss_m = loss_m + m.calculate_geometric_loss(z_tp1, z_t, a_t, r_t, h, decoder)

                            h = h_next  # IMPORTANT: carry hidden
                        losses.append(loss_m / b_z_seq.size(1))
                    loss = torch.stack(losses).mean()

                elif args.model_type == "gru":
                    # Standard GRU: average over ensemble members’ sequence_loss
                    losses = []
                    for m in ensemble.models:
                        losses.append(m.sequence_loss(b_z_seq, b_a_seq, b_r_seq, b_zn_seq))
                    loss = torch.stack(losses).mean()

                else:  # args.model_type == "rssm"
                    # Multi-step ELBO-style sequence loss
                    losses = []
                    for m in ensemble.models:
                        losses.append(m.sequence_loss(b_z_seq, b_a_seq, b_r_seq, b_zn_seq))
                    loss = torch.stack(losses).mean()

            else:
                # Non-GRU models (e.g. Koopman): per-step training
                b_z, b_a, b_r, b_z_next = [
                    t.to(device, non_blocking=True) for t in batch
                ]
                h = None
                loss = ensemble.calculate_loss(
                    z_next_clean=b_z_next, z_t=b_z, a_t=b_a, r_t=b_r, h=h
                )

            # Standard optimizer step
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ensemble.parameters(), args.grad_clip)
            optimizer.step()

            total += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg = total / max(1, len(loader))
        scheduler.step(avg)
        print(f"Epoch {epoch} avg loss: {avg:.6f}")

        if avg < best:
            best = avg
            ckpt = os.path.join(args.outdir, f"{save_tag}_best.pth")
            torch.save(ensemble.state_dict(), ckpt)
            print(f"  ↳ saved new best to {ckpt} (loss {best:.6f})")

    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a world model ensemble offline.")
    # data
    p.add_argument("--data_type", type=str, default="student_post_ft",
                   choices=["teacher", "student_pre_ft", "student_post_ft"])
    p.add_argument("--data_path", type=str, default=None, help="Override path to transitions .pkl")

    # model
    p.add_argument("--model_type", type=str, default="gru",
                   choices=["gru", "geometric_gru", "diffusion", "rssm", "sde", "koopman", "moe_gru"])
    p.add_argument("--num_ensemble", type=int, default=5)

    # ── GRU / Geometric-GRU params ────────────────────────────────────────────────
    p.add_argument("--gru_hidden", type=int, default=512)

    # ── Diffusion params ──────────────────────────────────────────────────────────
    p.add_argument("--timesteps", type=int, default=200, help="Number of diffusion steps (training schedule).")
    p.add_argument("--diff_hidden", type=int, default=512)
    p.add_argument("--time_dim", type=int, default=64)
    p.add_argument("--predict_residual", action="store_true",
                   help="Have diffusion predict Δz instead of z_next directly.")
    p.add_argument("--use_ema", action="store_true", help="Track EMA weights during training.")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ddim_eta", type=float, default=0.0, help="Stochasticity for DDIM sampling (0 = deterministic).")

    # ── Geometric loss (for geometric_gru) ────────────────────────────────────────
    p.add_argument("--decoder_path", type=str, default=None,
                   help="Path to decoder weights to enable geometric pullback loss.")

    # ── RSSM params ───────────────────────────────────────────────────────────────
    p.add_argument("--rssm_deter", type=int, default=512, help="Deterministic hidden size.")
    p.add_argument("--rssm_stoch", type=int, default=32, help="Stochastic latent size.")
    p.add_argument("--rssm_beta", type=float, default=1.0, help="KL/ELBO weight (β).")

    # ── Latent SDE params ────────────────────────────────────────────────────────
    p.add_argument("--sde_hidden", type=int, default=512)

    # ── Koopman params ───────────────────────────────────────────────────────────
    p.add_argument("--koop_hidden", type=int, default=256)
    p.add_argument("--koop_res_scale", type=float, default=0.1,
                   help="Residual dynamics scale added to linear Koopman part.")

    # ── MoE-GRU params ───────────────────────────────────────────────────────────
    p.add_argument("--moe_hidden", type=int, default=512)
    p.add_argument("--moe_k", type=int, default=4, help="Number of experts.")

    # optim
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # misc
    p.add_argument("--outdir", type=str, default="wm_ckpts")
    p.add_argument("--cpu", action="store_true")

    args = p.parse_args()
    main(args)

