import sys
import pickle
import random
import numpy as np
import torch

def tensor_stats(x: torch.Tensor, name="x"):
    x = x.detach().cpu()
    flat = x.flatten()
    # avoid printing huge stuff
    return {
        "name": name,
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "min": float(flat.min().item()) if flat.numel() else None,
        "max": float(flat.max().item()) if flat.numel() else None,
        "mean": float(flat.mean().item()) if flat.numel() else None,
        "std": float(flat.std().item()) if flat.numel() else None,
        "l2_norm": float(torch.norm(flat).item()) if flat.numel() else None,
    }

def to_tensor(x):
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x)

def main(path: str, sample_pairs: int = 5000, tol: float = 1e-6, save_starts: bool = False):
    print(f"Loading: {path}")
    with open(path, "rb") as f:
        transitions = pickle.load(f)

    N = len(transitions)
    print(f"Loaded transitions: N={N}")
    if N == 0:
        print("Empty file.")
        return

    first = transitions[0]
    print(f"Type(transitions[0]) = {type(first)}")
    if not isinstance(first, (tuple, list)):
        print("Unexpected: transition is not tuple/list.")
        return

    L = len(first)
    print(f"transition tuple length = {L} (expected 4 or 5)")
    print("---- First transition element types ----")
    for k, item in enumerate(first):
        print(f"  [{k}] type={type(item)}")

    # Interpret common formats
    # You described: (z_t, z_next, act, rew, discount)  OR maybe (z_t, z_next, act, rew)
    if L == 5:
        z_t0, z_n0, a0, r0, disc0 = first
    elif L == 4:
        z_t0, z_n0, a0, r0 = first
        disc0 = None
    else:
        print("Cannot interpret this format automatically.")
        return

    z_t0 = to_tensor(z_t0).view(-1)
    z_n0 = to_tensor(z_n0).view(-1)
    a0   = to_tensor(a0).view(-1)
    r0   = to_tensor(r0).view(-1)

    print("---- First transition tensor stats ----")
    print(tensor_stats(z_t0, "z_t"))
    print(tensor_stats(z_n0, "z_next"))
    print(tensor_stats(a0, "action"))
    print(tensor_stats(r0, "reward"))
    if disc0 is not None:
        disc0 = to_tensor(disc0).view(-1)
        print(tensor_stats(disc0, "discount"))
        # show unique discount values (common: {0, gamma})
        # careful: discount might be tensor shape [1]
        # We'll scan a small sample for unique values
        uniq = set()
        for _ in range(min(2000, N)):
            d = to_tensor(transitions[_][4]).detach().cpu().float().view(-1)
            uniq.add(float(d[0].item()))
        print(f"Sample unique discount values (first {min(2000, N)}): {sorted(list(uniq))}")

    # ------------------------------------------------------------
    # THE KEY TEST: are transitions stored sequentially?
    # If sequential within episodes, typically z_next[i] == z_t[i+1] (except at episode boundaries)
    # If from a replay buffer / shuffled, that relation will mostly NOT hold.
    # ------------------------------------------------------------
    if N < 2:
        print("Not enough transitions for adjacency test.")
        return

    M = min(sample_pairs, N - 1)
    idxs = [random.randint(0, N - 2) for _ in range(M)]

    diffs = []
    diffs_nonterminal = []
    diffs_terminal = []

    for i in idxs:
        zi_next = to_tensor(transitions[i][1]).detach().cpu().view(-1).float()
        zi1     = to_tensor(transitions[i + 1][0]).detach().cpu().view(-1).float()
        d = torch.norm(zi_next - zi1).item()
        diffs.append(d)

        if L == 5:
            di = float(to_tensor(transitions[i][4]).detach().cpu().view(-1)[0].item())
            if di == 0.0:
                diffs_terminal.append(d)
            else:
                diffs_nonterminal.append(d)

    diffs_np = np.asarray(diffs)
    print("\n==== Adjacency test: || z_next[i] - z_t[i+1] || ====")
    print(f"Sampled pairs: {M}")
    print(f"Mean diff: {diffs_np.mean():.6g}")
    print(f"Median diff: {np.median(diffs_np):.6g}")
    print(f"Min diff: {diffs_np.min():.6g}")
    print(f"Max diff: {diffs_np.max():.6g}")
    frac_match = float(np.mean(diffs_np < tol))
    print(f"Fraction < tol={tol}: {frac_match:.4%}")

    if L == 5 and (len(diffs_nonterminal) > 0 and len(diffs_terminal) > 0):
        dn = np.asarray(diffs_nonterminal)
        dt = np.asarray(diffs_terminal)
        print("\n---- Split by discount[i] ----")
        print(f"Nonterminal count: {len(dn)} | frac < tol: {float(np.mean(dn < tol)):.4%} | median={np.median(dn):.6g}")
        print(f"Terminal   count: {len(dt)} | frac < tol: {float(np.mean(dt < tol)):.4%} | median={np.median(dt):.6g}")

    # Heuristic conclusion
    # If sequential: nonterminal diffs should be ~0 (or tiny), terminal diffs should jump
    print("\n==== Interpretation ====")
    if frac_match > 0.8:
        print("Looks LIKE transitions are largely sequential (z_next[i] matches z[i+1] often).")
        if L == 5:
            # Extract episode starts: index 0, and i+1 after terminal transitions
            starts = [0]
            for i in range(N - 1):
                di = float(to_tensor(transitions[i][4]).detach().cpu().view(-1)[0].item())
                if di == 0.0:
                    starts.append(i + 1)
            starts = sorted(set(starts))
            print(f"Estimated number of episodes (from discount==0): {len(starts)}")
            print(f"First 20 start indices: {starts[:20]}")

            if save_starts:
                start_latents = [to_tensor(transitions[i][0]).detach().cpu() for i in starts]
                out_path = path.replace(".pkl", "_start_latents.pkl")
                with open(out_path, "wb") as f:
                    pickle.dump(start_latents, f)
                print(f"Saved {len(start_latents)} start latents to: {out_path}")
        else:
            print("But you have no discount flag, so we can't extract episode starts robustly.")
    else:
        print("Looks like transitions are NOT sequential (likely replay-buffer shuffled).")
        print("In that case, discount==0 does NOT tell you where step=0 states are in the file.")
        print("So using the whole 'z_t bag' as 'initial_states' is not actually 'starts'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_transitions.py <path_to_transitions.pkl>")
        sys.exit(1)
    path = sys.argv[1]
    # toggle save_starts=True if the file turns out to be sequential and you want a separate start-latents file
    main(path, sample_pairs=5000, tol=1e-6, save_starts=False)
