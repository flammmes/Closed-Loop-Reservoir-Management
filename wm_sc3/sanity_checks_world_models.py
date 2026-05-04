# ===== sanity_checks_world_models.py =====
import os, math, pickle, argparse, numpy as np, torch
from typing import List, Tuple
from world_models import ProbabilisticGRU, LatentDiffusionModel, EnsembleWorldModel, RSSMWorldModel, LatentSDEWorldModel, KoopmanWorldModel, MoEGRUWorldModel

# ---------- IO ----------

def load_transitions_cpu(path: str):
    with open(path, "rb") as f:
        trans = pickle.load(f)
    z_t_list, z_next_list, a_list, r_list, _disc_list = zip(*trans)
    z_t    = torch.stack([torch.as_tensor(x) for x in z_t_list])      # CPU
    z_next = torch.stack([torch.as_tensor(x) for x in z_next_list])   # CPU
    a_t    = torch.stack([torch.as_tensor(x) for x in a_list])        # CPU
    r_t    = torch.stack([torch.as_tensor(x) for x in r_list])        # CPU
    if r_t.dim() == 1: r_t = r_t.unsqueeze(-1)
    return z_t, a_t, r_t, z_next

def build_ensemble_for_eval(model_type: str, latent_dim: int, action_dim: int, num_ens: int,
                            device: torch.device,
                            # GRU / Geom-GRU
                            gru_hidden: int = 512,
                            # Diffusion
                            timesteps: int = 100, diff_hidden: int = 512, time_dim: int = 64,
                            predict_residual: bool = False, use_ema: bool = False, ema_decay: float = 0.999,
                            ddim_eta: float = 0.0,
                            # RSSM
                            rssm_deter: int = 512, rssm_stoch: int = 32, rssm_beta: float = 1.0,
                            # SDE
                            sde_hidden: int = 512,
                            # Koopman
                            koop_hidden: int = 256, koop_res_scale: float = 0.1,
                            # MoE-GRU
                            moe_hidden: int = 512, moe_k: int = 4
                            ) -> EnsembleWorldModel:
    models: List[torch.nn.Module] = []
    if model_type in ["gru", "geometric_gru"]:
        for _ in range(num_ens):
            models.append(ProbabilisticGRU(latent_dim, action_dim, hidden_dim=gru_hidden).to(device))
    elif model_type == "diffusion":
        for _ in range(num_ens):
            models.append(LatentDiffusionModel(
                latent_dim=latent_dim, action_dim=action_dim,
                num_timesteps=timesteps, hidden_dim=diff_hidden, time_dim=time_dim,
                predict_residual=predict_residual, reward_detach=True,
                use_ema=use_ema, ema_decay=ema_decay, ddim_eta=ddim_eta
            ).to(device))
    elif model_type == "rssm":
        for _ in range(num_ens):
            models.append(RSSMWorldModel(latent_dim, action_dim,
                                         deter_dim=rssm_deter, stoch_dim=rssm_stoch, beta=rssm_beta).to(device))
    elif model_type == "sde":
        for _ in range(num_ens):
            models.append(LatentSDEWorldModel(latent_dim, action_dim, hidden=sde_hidden).to(device))
    elif model_type == "koopman":
        for _ in range(num_ens):
            models.append(KoopmanWorldModel(latent_dim, action_dim,
                                            hidden=koop_hidden, res_scale=koop_res_scale).to(device))
    elif model_type == "moe_gru":
        for _ in range(num_ens):
            models.append(MoEGRUWorldModel(latent_dim, action_dim,
                                           hidden=moe_hidden, num_experts=moe_k).to(device))
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    return EnsembleWorldModel(models)


# ---------- Checks ----------

@torch.no_grad()
def check_dataset(z_t, a_t, r_t, z_next):
    print("=== DATASET CHECK ===")
    print(f"z_t      : {tuple(z_t.shape)}, dtype={z_t.dtype}")
    print(f"a_t      : {tuple(a_t.shape)}, dtype={a_t.dtype}")
    print(f"r_t      : {tuple(r_t.shape)}, dtype={r_t.dtype}")
    print(f"z_next   : {tuple(z_next.shape)}, dtype={z_next.dtype}")
    for name, t in [("z_t", z_t), ("a_t", a_t), ("r_t", r_t), ("z_next", z_next)]:
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        mn, mx = t.min().item(), t.max().item()
        print(f"  {name}: NaN={has_nan} Inf={has_inf} range=[{mn:.4g}, {mx:.4g}]")
    z_norm = torch.linalg.norm(z_t, dim=-1).float().mean().item()
    step_norm = torch.linalg.norm((z_next - z_t), dim=-1).float().mean().item()
    print(f"  ||z_t|| mean ≈ {z_norm:.3f} | ||z_next - z_t|| mean ≈ {step_norm:.3f}")
    print()

@torch.no_grad()
def eval_one_step_gru(ens: EnsembleWorldModel,
                      z_t: torch.Tensor, a_t: torch.Tensor, r_t: torch.Tensor, z_next: torch.Tensor,
                      device: torch.device, batch_size: int = 8192, max_N: int = 50000):
    print("=== 1-STEP METRICS (GRU / Geometric-GRU) ===")
    N = min(z_t.size(0), max_N)
    z_t, a_t, r_t, z_next = z_t[:N], a_t[:N], r_t[:N], z_next[:N]
    H = ens.models[0].hidden_dim
    per_member_rmse = []
    per_member_r_mse = []
    # evaluate each member on mean prediction
    for idx, m in enumerate(ens.models):
        se_sum, rse_sum, count = 0.0, 0.0, 0
        for i in range(0, N, batch_size):
            b = slice(i, min(i+batch_size, N))
            zB = z_t[b].to(device); aB = a_t[b].to(device); rB = r_t[b].to(device); znB = z_next[b].to(device)
            h0 = torch.zeros(zB.size(0), H, device=device)
            mean, std, r_pred, _ = m.forward(zB, aB, h0)
            se_sum  += torch.mean((mean - znB)**2, dim=-1).sum().item()
            rse_sum += torch.mean((r_pred - rB)**2, dim=-1).sum().item()
            count   += zB.size(0)
        rmse = math.sqrt(se_sum / count)
        r_mse = rse_sum / count
        per_member_rmse.append(rmse)
        per_member_r_mse.append(r_mse)
        print(f"  member {idx}: latent RMSE={rmse:.4f}, reward MSE={r_mse:.5f}")
    print(f"  ensemble mean: latent RMSE={np.mean(per_member_rmse):.4f} ± {np.std(per_member_rmse):.4f}")
    print(f"  ensemble mean: reward MSE={np.mean(per_member_r_mse):.5f} ± {np.std(per_member_r_mse):.5f}")
    # disagreement vs error correlation
    print("\n  -> disagreement vs error (correlation)")
    # use ensemble means & variance
    errs, vars_ = [], []
    for i in range(0, N, batch_size):
        b = slice(i, min(i+batch_size, N))
        zB = z_t[b].to(device); aB = a_t[b].to(device); znB = z_next[b].to(device)
        h0 = torch.zeros(zB.size(0), H, device=device)
        preds = []
        for m in ens.models:
            mean, _, _, _ = m.forward(zB, aB, h0)
            preds.append(mean)
        P = torch.stack(preds, dim=0)           # [E, B, D]
        mu = P.mean(dim=0)                      # [B, D]
        var = P.var(dim=0, unbiased=False).mean(dim=-1)  # [B], average var across dims
        err = torch.mean((mu - znB)**2, dim=-1)          # [B]
        errs.append(err.cpu()); vars_.append(var.cpu())
    errs = torch.cat(errs); vars_ = torch.cat(vars_)
    # Pearson correlation
    ex = errs - errs.mean(); vx = vars_ - vars_.mean()
    corr = (ex * vx).sum() / (torch.sqrt((ex**2).sum()) * torch.sqrt((vx**2).sum()) + 1e-8)
    print(f"  Pearson corr(err, disagreement) = {corr.item():.3f}\n")

@torch.no_grad()
def eval_one_step_diffusion(ens: EnsembleWorldModel,
                            z_t: torch.Tensor, a_t: torch.Tensor, r_t: torch.Tensor, z_next: torch.Tensor,
                            device: torch.device, steps_list=(10, 20), batch_size: int = 4096, max_N: int = 20000):
    print("=== 1-STEP METRICS (Diffusion) ===")
    N = min(z_t.size(0), max_N)
    z_t, a_t, r_t, z_next = z_t[:N], a_t[:N], r_t[:N], z_next[:N]
    # deterministically pick member 0 (DDIM eta=0 default in your code)
    m = ens.models[0]
    for steps in steps_list:
        se_sum, rse_sum, count = 0.0, 0.0, 0
        for i in range(0, N, batch_size):
            b = slice(i, min(i+batch_size, N))
            zB = z_t[b].to(device); aB = a_t[b].to(device); rB = r_t[b].to(device); znB = z_next[b].to(device)
            z_pred, r_pred, _ = m.sample(zB, aB, h=None, steps=steps, use_ema=True, ddim=True)
            se_sum  += torch.mean((z_pred - znB)**2, dim=-1).sum().item()
            rse_sum += torch.mean((r_pred - rB)**2, dim=-1).sum().item()
            count   += zB.size(0)
        rmse = math.sqrt(se_sum / count)
        r_mse = rse_sum / count
        print(f"  steps={steps}: latent RMSE={rmse:.4f}, reward MSE={r_mse:.5f}")
    # step sensitivity
    print("\n  -> Step sensitivity on a small batch")
    B = min(2048, N)
    zB = z_t[:B].to(device); aB = a_t[:B].to(device)
    z10, _r10, _ = m.sample(zB, aB, steps=steps_list[0], use_ema=True, ddim=True)
    z20, _r20, _ = m.sample(zB, aB, steps=steps_list[1], use_ema=True, ddim=True)
    diff = torch.mean((z10 - z20)**2, dim=-1).sqrt().mean().item()
    print(f"  ||z(steps={steps_list[0]}) - z(steps={steps_list[1]})||_2 mean ≈ {diff:.4f}")
    # reproducibility
    torch.manual_seed(123)
    zA, _rA, _ = m.sample(zB, aB, steps=steps_list[1], use_ema=True, ddim=True)
    torch.manual_seed(123)
    zB2, _rB2, _ = m.sample(zB, aB, steps=steps_list[1], use_ema=True, ddim=True)
    rep = torch.mean((zA - zB2)**2).item()
    print(f"  reproducibility (same seed, same steps) MSE ≈ {rep:.6e}\n")

@torch.no_grad()
def eval_one_step_generic(ens: EnsembleWorldModel,
                          z_t: torch.Tensor, a_t: torch.Tensor, r_t: torch.Tensor, z_next: torch.Tensor,
                          device: torch.device, batch_size: int = 8192, max_N: int = 50000,
                          repeats: int = 1):
    """
    Memberwise 1-step RMSE/MSE using each model's .sample().
    For stochastic models (e.g., SDE) set repeats>1 to reduce MC noise.
    """
    print("=== 1-STEP METRICS (Generic) ===")
    N = min(z_t.size(0), max_N)
    z_t, a_t, r_t, z_next = z_t[:N], a_t[:N], r_t[:N], z_next[:N]

    per_member_rmse, per_member_r_mse = [], []

    def _h0_for(m, B):
        if hasattr(m, "hidden_dim"):   # ProbGRU / GeomGRU
            return torch.zeros(B, m.hidden_dim, device=device)
        if hasattr(m, "deter_dim"):    # RSSM
            return torch.zeros(B, m.deter_dim, device=device)
        if hasattr(m, "hidden"):       # MoE-GRU
            return torch.zeros(B, m.hidden, device=device)
        return None

    # memberwise errors
    for idx, m in enumerate(ens.models):
        se_sum, rse_sum, count = 0.0, 0.0, 0
        for i in range(0, N, batch_size):
            b = slice(i, min(i + batch_size, N))
            zB = z_t[b].to(device); aB = a_t[b].to(device); rB = r_t[b].to(device); znB = z_next[b].to(device)
            h0 = _h0_for(m, zB.size(0))
            if repeats == 1:
                z_pred, r_pred, _ = m.sample(zB, aB, h0)
            else:
                zs, rs = [], []
                for _ in range(repeats):
                    z_tmp, r_tmp, _ = m.sample(zB, aB, h0)
                    zs.append(z_tmp); rs.append(r_tmp)
                z_pred = torch.stack(zs, dim=0).mean(dim=0)
                r_pred = torch.stack(rs, dim=0).mean(dim=0)
            se_sum  += torch.mean((z_pred - znB)**2, dim=-1).sum().item()
            rse_sum += torch.mean((r_pred - rB)**2, dim=-1).sum().item()
            count   += zB.size(0)
        rmse = math.sqrt(se_sum / count)
        r_mse = rse_sum / count
        per_member_rmse.append(rmse); per_member_r_mse.append(r_mse)
        print(f"  member {idx}: latent RMSE={rmse:.4f}, reward MSE={r_mse:.5f}")

    print(f"  ensemble mean: latent RMSE={np.mean(per_member_rmse):.4f} ± {np.std(per_member_rmse):.4f}")
    print(f"  ensemble mean: reward MSE={np.mean(per_member_r_mse):.5f} ± {np.std(per_member_r_mse):.5f}")

    # disagreement vs error (correlation) using ensemble mean & variance
    print("\n  -> disagreement vs error (correlation)")
    errs, vars_ = [], []
    for i in range(0, N, batch_size):
        b = slice(i, min(i + batch_size, N))
        zB = z_t[b].to(device); aB = a_t[b].to(device); znB = z_next[b].to(device)
        preds = []
        for m in ens.models:
            h0 = _h0_for(m, zB.size(0))
            if repeats == 1:
                z_pred, _r_pred, _ = m.sample(zB, aB, h0)
            else:
                zs = []
                for _ in range(repeats):
                    z_tmp, _r_tmp, _ = m.sample(zB, aB, h0)
                    zs.append(z_tmp)
                z_pred = torch.stack(zs, dim=0).mean(dim=0)
            preds.append(z_pred)
        P = torch.stack(preds, dim=0)  # [E,B,D]
        mu = P.mean(dim=0)             # [B,D]
        var = P.var(dim=0, unbiased=False).mean(dim=-1)  # [B]
        err = torch.mean((mu - znB)**2, dim=-1)          # [B]
        errs.append(err.cpu()); vars_.append(var.cpu())
    errs = torch.cat(errs); vars_ = torch.cat(vars_)
    ex = errs - errs.mean(); vx = vars_ - vars_.mean()
    corr = (ex * vx).sum() / (torch.sqrt((ex**2).sum()) * torch.sqrt((vx**2).sum()) + 1e-8)
    print(f"  Pearson corr(err, disagreement) = {corr.item():.3f}\n")




@torch.no_grad()
def check_gru_sampling_vs_mean(ens: EnsembleWorldModel,
                               z_t: torch.Tensor, a_t: torch.Tensor,
                               device: torch.device, B: int = 2048):
    print("=== GRU: mean vs sampled sanity ===")
    m = ens.models[0]
    zB = z_t[:B].to(device); aB = a_t[:B].to(device)
    h0 = torch.zeros(B, m.hidden_dim, device=device)
    mean, std, _r, _ = m.forward(zB, aB, h0)
    # sample 3 times
    zs = []
    for _ in range(3):
        z_s, _r_s, _ = m.sample(zB, aB, h0)
        zs.append(z_s)
    zs = torch.stack(zs, dim=0)  # [3, B, D]
    # compare
    mean_samp = zs.mean(dim=0)
    rmse_mean_vs_sampmean = torch.mean((mean - mean_samp)**2).sqrt().item()
    avg_std = std.mean().item()
    spread = torch.mean(torch.std(zs, dim=0)).item()
    print(f"  avg(std from forward) ≈ {avg_std:.4f} | avg(std across samples) ≈ {spread:.4f}")
    print(f"  RMSE(mean, sample-mean) ≈ {rmse_mean_vs_sampmean:.4f}\n")

# ---------- Runner ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--model_type", type=str,
                    choices=["gru","geometric_gru","diffusion","rssm","sde","koopman","moe_gru"],
                    required=True)
    ap.add_argument("--num_ensemble", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda")
    # (only if you changed these at training time)
    ap.add_argument("--gru_hidden", type=int, default=512)
    ap.add_argument("--timesteps", type=int, default=100)
    ap.add_argument("--diff_hidden", type=int, default=512)
    ap.add_argument("--time_dim", type=int, default=64)
    ap.add_argument("--predict_residual", action="store_true")
    ap.add_argument("--use_ema", action="store_true")


    # RSSM
    ap.add_argument("--rssm_deter", type=int, default=512)
    ap.add_argument("--rssm_stoch", type=int, default=32)
    ap.add_argument("--rssm_beta", type=float, default=1.0)

    # SDE
    ap.add_argument("--sde_hidden", type=int, default=512)

    # Koopman
    ap.add_argument("--koop_hidden", type=int, default=256)
    ap.add_argument("--koop_res_scale", type=float, default=0.1)

    # MoE-GRU
    ap.add_argument("--moe_hidden", type=int, default=512)
    ap.add_argument("--moe_k", type=int, default=4)

    # (optional) repeats for stochastic eval (use >1 for SDE)
    ap.add_argument("--repeats", type=int, default=1)


    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device=="cuda" else "cpu")
    print(f"Device: {device}")

    z_t, a_t, r_t, z_next = load_transitions_cpu(args.data_path)
    check_dataset(z_t, a_t, r_t, z_next)

    latent_dim = z_t.shape[1]; action_dim = a_t.shape[1]
    ens = build_ensemble_for_eval(
        model_type=args.model_type, latent_dim=latent_dim, action_dim=action_dim, num_ens=args.num_ensemble,
        device=device,
        gru_hidden=args.gru_hidden,
        timesteps=args.timesteps, diff_hidden=args.diff_hidden, time_dim=args.time_dim,
        predict_residual=args.predict_residual, use_ema=args.use_ema,
        # new bits:
        rssm_deter=args.rssm_deter, rssm_stoch=args.rssm_stoch, rssm_beta=args.rssm_beta,
        sde_hidden=args.sde_hidden,
        koop_hidden=args.koop_hidden, koop_res_scale=args.koop_res_scale,
        moe_hidden=args.moe_hidden, moe_k=args.moe_k
    )
    ens.load_state_dict(torch.load(args.ckpt, map_location=device))
    ens.eval()

    if args.model_type in ["gru", "geometric_gru"]:
        eval_one_step_gru(ens, z_t, a_t, r_t, z_next, device=device)
        check_gru_sampling_vs_mean(ens, z_t, a_t, device=device)
    elif args.model_type == "diffusion":
        eval_one_step_diffusion(ens, z_t, a_t, r_t, z_next, device=device)
    else:
        # rssm / sde / koopman / moe_gru
        # tip: for SDE set --repeats 5 to reduce Monte Carlo noise
        eval_one_step_generic(ens, z_t, a_t, r_t, z_next, device=device, repeats=args.repeats)

if __name__ == "__main__":
    main()
