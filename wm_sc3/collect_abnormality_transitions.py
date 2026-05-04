
# collect_abnormality_transitions_random.py
#
# Collect a *small* dataset of abnormal transitions (scenario 2 leakage / scenario 3 faults)
# using a RANDOM policy (uniform in [-1, 1] for every action dim).
#
# Output format (pickle):
# {
#   "z":      (N, latent_dim) float32,
#   "a":      (N, action_dim) float32,
#   "r":      (N,)            float32,
#   "z_next": (N, latent_dim) float32,
#   "done":   (N,)            bool,
#   "info":   (optional)      whatever Tianshou stores (can be large / unpicklable),
# }
#
# Notes:
# - This script wraps ReservoirEnv with RealToLatentEnv using a frozen HistoryEncoder.
# - Policy is *pure random*, so it does not need PolicyHead / actor nets.
# - If you want to bias toward "inject less", add --bias_low (see flag below).

import argparse
import os
import pickle
import numpy as np
import torch

from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import BasePolicy

from env_3_mb3 import ReservoirEnv
from nets import HistoryEncoder
from train_in_dream_v2 import RealToLatentEnv
import gymnasium as gym
from tianshou.policy.base import RandomActionPolicy

class RandomContinuousPolicy(BasePolicy):

    def __init__(self, action_dim: int, seed: int | None = None, bias_low: float = 0.0):
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        super().__init__(action_space=action_space)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)
        self.bias_low = float(bias_low)

    def forward(self, batch: Batch, state=None, **kwargs):
        n = len(batch.obs)
        if self.bias_low <= 0.0:
            act = self.rng.uniform(-1.0, 1.0, size=(n, self.action_dim)).astype(np.float32)
        else:
            # bias_low in (0,1): mixes uniform with a Beta skewed to -1 (lower injections)
            u = self.rng.uniform(-1.0, 1.0, size=(n, self.action_dim)).astype(np.float32)
            b = self.rng.beta(0.5, 2.5, size=(n, self.action_dim)).astype(np.float32)  # in (0,1)
            b = 2.0 * b - 1.0  # -> (-1, 1), skewed toward -1
            act = (1.0 - self.bias_low) * u + self.bias_low * b
            act = np.clip(act, -1.0, 1.0).astype(np.float32)

        return Batch(act=act, state=None)

    def learn(self, batch: Batch, **kwargs):
        return {}


def make_latent_env_thunk(
    env_id: int,
    encoder_ckpt: str,
    latent_dim: int,
    device: str = "cpu",
    realizations_dir: str | None = None,
    scenario: int | None = None,
    force_sim_deck_idx: int | None = None,
):
    def _thunk():
        base = ReservoirEnv(env_id=env_id)

        # Optional knobs (won't break if env doesn't support them)
        if realizations_dir is not None and hasattr(base, "realizations_dir"):
            base.realizations_dir = realizations_dir
        if scenario is not None and hasattr(base, "scenario"):
            base.scenario = int(scenario)
        if force_sim_deck_idx is not None and hasattr(base, "force_sim_deck_idx"):
            base.force_sim_deck_idx = int(force_sim_deck_idx)

        enc = HistoryEncoder(d_model=latent_dim).to(device)
        enc.load_state_dict(torch.load(encoder_ckpt, map_location=device))
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)

        return RealToLatentEnv(base, enc, device=device)

    return _thunk


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--encoder_ckpt", type=str, required=True,
                    help="Student encoder checkpoint (HistoryEncoder state_dict).")
    ap.add_argument("--out_pkl", type=str, default="abnormality_transitions.pkl")
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--action_dim", type=int, default=11)

    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--base_env_id", type=int, default=0,
                    help="env_ids will be base_env_id..base_env_id+num_envs-1")

    ap.add_argument("--n_step", type=int, default=5000)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bias_low", type=float, default=0.0,
                    help="0.0 = uniform random. 0.3..0.7 biases actions toward -1 (lower injection).")

    ap.add_argument("--scenario", type=int, default=None,
                    help="2 (leakage) or 3 (faults), only used if your env supports it.")
    ap.add_argument("--realizations_dir", type=str, default=None,
                    help="Alternate realizations folder, only used if your env supports it.")

    ap.add_argument("--force_sim_deck_indices", type=str, default=None,
                    help="Comma-separated SIM deck indices (len == num_envs). Requires env.force_sim_deck_idx support.")

    ap.add_argument("--save_info", action="store_true",
                    help="Attempt to save buf.info too (can break pickling if info contains non-serializable objects).")

    args = ap.parse_args()

    # SubprocVectorEnv: keep everything CPU-friendly
    device = "cpu"

    forced = None
    if args.force_sim_deck_indices is not None:
        forced = [int(x) for x in args.force_sim_deck_indices.split(",")]
        if len(forced) != args.num_envs:
            raise ValueError("--force_sim_deck_indices must have exactly num_envs entries.")

    # ---- build vector envs ----
    env_fns = []
    for i in range(args.num_envs):
        env_id = args.base_env_id + i
        force_idx = forced[i] if forced is not None else None
        env_fns.append(
            make_latent_env_thunk(
                env_id=env_id,
                encoder_ckpt=args.encoder_ckpt,
                latent_dim=args.latent_dim,
                device=device,
                realizations_dir=args.realizations_dir,
                scenario=args.scenario,
                force_sim_deck_idx=force_idx,
            )
        )
    envs = SubprocVectorEnv(env_fns)

    # ---- random policy ----
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(args.action_dim,), dtype=np.float32)
    policy = RandomActionPolicy(action_space=action_space)

    # ---- collect ----
    buf = VectorReplayBuffer(args.n_step, args.num_envs)
    collector = Collector(policy, envs, buf, exploration_noise=False)
    collector.reset()

    print(f"[collect] collecting n_step={args.n_step} with num_envs={args.num_envs} (random policy) ...")
    stats = collector.collect(n_step=args.n_step)
    print(f"[collect] done. stats={stats}")

    # ---- export dataset ----
    N = len(buf)

    z      = np.asarray(buf.obs[:N], dtype=np.float32)
    a      = np.asarray(buf.act[:N], dtype=np.float32)
    r      = np.asarray(buf.rew[:N], dtype=np.float32)
    z_next = np.asarray(buf.obs_next[:N], dtype=np.float32)
    done   = np.asarray(buf.done[:N], dtype=bool)

    out = {"z": z, "a": a, "r": r, "z_next": z_next, "done": done}

    if args.save_info:
        try:
            out["info"] = buf.info[:N]
            print("[collect] saved buf.info too (if pickling fails, rerun without --save_info).")
        except Exception as e:
            print(f"[collect] could not save info: {e}")

    with open(args.out_pkl, "wb") as f:
        pickle.dump(out, f)

    print(f"[collect] wrote {N} transitions to: {args.out_pkl}")

    envs.close()


if __name__ == "__main__":
    main()
