"""
Quick single-model visualization for the PRODUCTION (5-state) DQN.

Point it at ONE trained result — either the CHTC results tarball
(DQN_prod_<sc>_seed<seed>_results.tar.gz) or the extracted .pkl Q-table.
Produces a 4-panel figure highlighting the new production-level story.

Only needs: numpy, matplotlib, standard library (NO torch — the .pkl Q-table
is a plain dict of {state: {'keep':.., 'replace':..}} with 5-tuple keys
(parity, mac, mip, disease, production_level)).

Usage:
    python visualize_single.py DQN_prod_2025_seed1024_results.tar.gz
    python visualize_single.py collected/DQN_prod_2025_seed1024.pkl --out fig.png
"""
import argparse, os, re, pickle, tarfile, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

SCENARIO_NAMES = {
    "2025": "2025 Baseline", "OG": "Oversupply / Good market",
    "OB": "Oversupply / Bad market", "UG": "Undersupply / Good market",
    "UB": "Undersupply / Bad market",
}


def load_qtable(path):
    """Return (q_table, rewards, label). Accepts a .pkl or a *_results.tar.gz."""
    if path.endswith(".tar.gz") or path.endswith(".tgz"):
        tmp = tempfile.mkdtemp()
        with tarfile.open(path) as t:
            members = [m for m in t.getmembers() if m.name.endswith(".pkl")
                       and "_eval" not in m.name]
            if not members:
                raise SystemExit("No Q-table .pkl found inside the tarball.")
            t.extract(members[0], tmp)
            pkl_path = os.path.join(tmp, members[0].name)
    else:
        pkl_path = path
    with open(pkl_path, "rb") as f:
        q_table, rewards, epsilon = pickle.load(f)
    return q_table, rewards, os.path.basename(pkl_path)


def cow_value(q, s):
    return q[s]["keep"] - q[s]["replace"] if s in q else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to a *_results.tar.gz or a .pkl Q-table")
    ap.add_argument("--out", default=None, help="Output PNG (default: <input>_preview.png)")
    ap.add_argument("--ref-level", type=float, default=1.0,
                    help="Reference production level for the parity x MAC map (default 1.0)")
    args = ap.parse_args()

    q, rewards, label = load_qtable(args.input)
    levels = sorted({k[4] for k in q})
    parities = list(range(1, 13))
    macs = list(range(1, 21))

    # scenario/seed from filename
    m = re.search(r"DQN_prod_([A-Za-z0-9]+)_seed(\d+)", label)
    scen = m.group(1) if m else "?"
    seed = m.group(2) if m else "?"
    ann_return = (np.mean(rewards[-1000:]) / 15.0) if rewards else float("nan")
    scen_name = SCENARIO_NAMES.get(scen, scen)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"DRL replacement policy with production level — {scen_name} (seed {seed})\n"
                 f"steady-state return ≈ ${ann_return:,.0f}/cow-stall/yr",
                 fontsize=14, fontweight="bold")

    # ---- (a) Replacement boundary vs parity, at low/avg/high production ----
    axA = axes[0, 0]
    pick = [l for l in (0.7, 1.0, 1.3) if l in levels] or levels[::max(1, len(levels)//3)]
    colors = {0.7: "#d62728", 1.0: "#1f77b4", 1.3: "#2ca02c"}
    for L in pick:
        thr = []
        for p in parities:
            t = np.nan
            for mac in macs:
                if cow_value(q, (p, mac, 0, 0, L)) < 0:
                    t = mac; break
            thr.append(t)
        axA.plot(parities, thr, "o-", label=f"prod level {L:g}",
                 color=colors.get(L), linewidth=2)
    axA.set_xlabel("Parity"); axA.set_ylabel("MAC at which replacement becomes optimal")
    axA.set_title("(a) Replacement threshold by parity\nlow producers culled far earlier")
    axA.set_ylim(0, 21); axA.grid(alpha=0.3); axA.legend()

    # ---- (b) Cow-value heatmap parity x MAC (healthy, open) at reference level ----
    axB = axes[0, 1]
    L = args.ref_level if args.ref_level in levels else min(levels, key=lambda x: abs(x-args.ref_level))
    Z = np.array([[cow_value(q, (p, mac, 0, 0, L)) for mac in macs] for p in parities])
    vmax = np.nanmax(np.abs(Z))
    im = axB.imshow(Z, aspect="auto", origin="lower", cmap="RdYlGn",
                    norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
                    extent=[macs[0]-.5, macs[-1]+.5, parities[0]-.5, parities[-1]+.5])
    axB.set_xlabel("Month after calving (MAC)"); axB.set_ylabel("Parity")
    axB.set_title(f"(b) Cow value (keep − replace), healthy open cow\nat prod level {L:g}  (green=keep, red=replace)")
    fig.colorbar(im, ax=axB, label="$ cow value")

    # ---- (c) Cow value vs production level, young open cow across parities ----
    axC = axes[1, 0]
    for p in (1, 3, 5, 8):
        vals = [cow_value(q, (p, 3, 0, 0, L2)) for L2 in levels]
        axC.plot(levels, vals, "o-", label=f"parity {p}", linewidth=2)
    axC.axhline(0, color="k", lw=1, ls="--")
    axC.set_xlabel("Production level (multiplier on lactation curve)")
    axC.set_ylabel("$ cow value (keep − replace)")
    axC.set_title("(c) Cow value rises with production level\n(open cow, MAC 3); crosses 0 → cull below it")
    axC.grid(alpha=0.3); axC.legend()

    # ---- (d) Cow-value heatmap parity x production level (open, MAC 5) ----
    axD = axes[1, 1]
    fixed_mac = 5
    Zd = np.array([[cow_value(q, (p, fixed_mac, 0, 0, L2)) for L2 in levels] for p in parities])
    vmaxd = np.nanmax(np.abs(Zd))
    im2 = axD.imshow(Zd, aspect="auto", origin="lower", cmap="RdYlGn",
                     norm=TwoSlopeNorm(vcenter=0, vmin=-vmaxd, vmax=vmaxd),
                     extent=[-0.5, len(levels)-0.5, parities[0]-.5, parities[-1]+.5])
    axD.set_xticks(range(len(levels))); axD.set_xticklabels([f"{l:g}" for l in levels])
    axD.set_xlabel("Production level"); axD.set_ylabel("Parity")
    axD.set_title(f"(d) Cow value by parity × production level\n(open cow, MAC {fixed_mac})")
    fig.colorbar(im2, ax=axD, label="$ cow value")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.out or (re.sub(r"(_results\.tar\.gz|\.pkl)$", "", args.input) + "_preview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
