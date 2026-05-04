# train_latent_mapping.py
import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm


# ---------------------------------------------------------------------
#  Model: simple residual MLP for latent → latent mapping
# ---------------------------------------------------------------------
class LatentMapping(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        # Residual: good prior since pre/post spaces should be close
        return z + self.mlp(z)


# ---------------------------------------------------------------------
#  Data loading: build aligned (z_pre, z_post) pairs
# ---------------------------------------------------------------------
def load_latent_pairs(pre_path: str, post_path: str, use_next: bool = True):
    """Return tensors Z_pre, Z_post with shape [N, D]."""
    with open(pre_path, "rb") as f:
        pre_transitions = pickle.load(f)
    with open(post_path, "rb") as f:
        post_transitions = pickle.load(f)

    if len(pre_transitions) != len(post_transitions):
        raise ValueError(
            f"pre/post transition lengths differ: "
            f"{len(pre_transitions)} vs {len(post_transitions)}"
        )

    z_pre_list = []
    z_post_list = []

    for pre_t, post_t in zip(pre_transitions, post_transitions):
        # Each is (z_t, z_next, a_t, r_t, discount)
        z_pre_t, z_pre_next, *_ = pre_t
        z_post_t, z_post_next, *_ = post_t

        # current state
        z_pre_list.append(torch.as_tensor(z_pre_t).view(-1))
        z_post_list.append(torch.as_tensor(z_post_t).view(-1))

        # optionally also use next state to enlarge dataset
        if use_next:
            z_pre_list.append(torch.as_tensor(z_pre_next).view(-1))
            z_post_list.append(torch.as_tensor(z_post_next).view(-1))

    Z_pre = torch.stack(z_pre_list, dim=0).float()
    Z_post = torch.stack(z_post_list, dim=0).float()

    if Z_pre.shape != Z_post.shape:
        raise ValueError(f"Shape mismatch: {Z_pre.shape} vs {Z_post.shape}")

    return Z_pre, Z_post


# ---------------------------------------------------------------------
#  Training loop for a single mapping
# ---------------------------------------------------------------------
def train_mapping(
    src: torch.Tensor,
    tgt: torch.Tensor,
    device: torch.device,
    out_path: str,
    hidden: int = 256,
    num_layers: int = 3,
    batch_size: int = 2048,
    epochs: int = 100,
    lr: float = 1e-3,
    wd: float = 0.0,
    val_frac: float = 0.1,
    grad_clip: float = 1.0,
    label: str = "pre2post",
):
    """Train f: src → tgt and save the best model to out_path."""
    assert src.shape == tgt.shape
    N, D = src.shape

    dataset = TensorDataset(src, tgt)

    n_val = max(1, int(N * val_frac))
    n_train = N - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = LatentMapping(dim=D, hidden=hidden, num_layers=num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val = float("inf")
    best_epoch = -1

    print(f"\n=== Training {label} mapping ({D}-dim) on {N} samples ===")
    print(f"Train: {n_train}, Val: {n_val}")

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"[{label}] Epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            batch_size_actual = x.size(0)
            train_loss_sum += loss.item() * batch_size_actual
            train_count += batch_size_actual
            pbar.set_postfix(loss=loss.item())

        train_loss = train_loss_sum / max(1, train_count)

        # ---- validation ----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x)
                loss = F.mse_loss(pred, y)
                batch_size_actual = x.size(0)
                val_loss_sum += loss.item() * batch_size_actual
                val_count += batch_size_actual

        val_loss = val_loss_sum / max(1, val_count)

        print(
            f"[{label}] Epoch {epoch:03d} | "
            f"train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f}"
        )

        # ---- checkpoint ----
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), out_path)
            print(f"  ↳ new best {label} model saved to {out_path} (val MSE={best_val:.6f})")

    print(f"Finished {label}: best val MSE={best_val:.6f} at epoch {best_epoch}")


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    pre_path = args.pre_path
    post_path = args.post_path

    print(f"Loading pre-ft transitions from:  {pre_path}")
    print(f"Loading post-ft transitions from: {post_path}")
    Z_pre, Z_post = load_latent_pairs(pre_path, post_path, use_next=not args.ignore_next)
    print(f"Latent pairs loaded: shape {Z_pre.shape}")

    os.makedirs(args.outdir, exist_ok=True)

    # Train f_pre2post
    pre2post_path = os.path.join(args.outdir, "pre2post_mapping_best.pth")
    train_mapping(
        src=Z_pre,
        tgt=Z_post,
        device=device,
        out_path=pre2post_path,
        hidden=args.hidden,
        num_layers=args.layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.weight_decay,
        val_frac=args.val_frac,
        grad_clip=args.grad_clip,
        label="pre2post",
    )

    # Optionally train the inverse too
    if not args.only_pre2post:
        post2pre_path = os.path.join(args.outdir, "post2pre_mapping_best.pth")
        train_mapping(
            src=Z_post,
            tgt=Z_pre,
            device=device,
            out_path=post2pre_path,
            hidden=args.hidden,
            num_layers=args.layers,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.weight_decay,
            val_frac=args.val_frac,
            grad_clip=args.grad_clip,
            label="post2pre",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mappings between student_pre_ft and student_post_ft latent spaces.")

    parser.add_argument(
        "--pre_path",
        type=str,
        default="student_pre_ft_wm_transitions.pkl",
        help="Path to student_pre_ft_wm_transitions.pkl",
    )
    parser.add_argument(
        "--post_path",
        type=str,
        default="student_post_ft_wm_transitions.pkl",
        help="Path to student_post_ft_wm_transitions.pkl",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="latent_mappings",
        help="Directory to save mapping checkpoints.",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ignore_next", action="store_true", help="Use only z_t (not z_next) as training pairs.")
    parser.add_argument("--only_pre2post", action="store_true", help="Skip training the inverse post2pre mapping.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    args = parser.parse_args()
    main(args)
