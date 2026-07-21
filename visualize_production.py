"""
visualize_production.py  —  production-level figures (Option A).

Queries the trained DQN network (<name>_model.pth) directly, so PRODUCTION
LEVEL is treated as CONTINUOUS (0.67-1.33) rather than the 5 coarse bins in
the compatibility .pkl. A couple of figures deliberately use a few DISCRETE
production levels (low / average / high) where a small-multiples comparison
reads more clearly than a continuous surface.

Every figure fixes a healthy, open cow (disease = 0, MIP = 0) unless noted,
and sweeps parity 1-12. All output PNGs go to --outdir (default: prod_outputs).

Figure menu (each is a separate function; --figs selects a subset):
  1 decision_map      parity x production level, keep/replace regions + M* line   [continuous]
  2 threshold_curve   critical production level M* vs parity, a few MACs           [continuous]  ** hero
  3 value_vs_prod     retention value ($) vs production level, line per parity     [continuous]
  4 value_surface     parity x production level heatmap of cow value ($)           [continuous]
  5 threshold_by_cat  replacement MAC threshold by parity, low/avg/high producers  [discrete]
  6 across_scenarios  M* vs parity, one line per available scenario                [continuous]

Usage:
  python visualize_production.py --collected_dir collected/ --outdir prod_outputs
  python visualize_production.py --collected_dir collected/ --seed 1024 --figs 1,2,3
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap, LinearSegmentedColormap

from net_utils import QFunction, find_qfunc, find_qfunc_ensemble

# ── configuration ────────────────────────────────────────────────────────────
PREFIX          = "DQN_prod"
PARITIES        = list(range(1, 13))
PROD_MIN, PROD_MAX = 0.67, 1.33          # truncation range of the production level
PROD_GRID       = np.linspace(PROD_MIN, PROD_MAX, 133)   # fine => "continuous"
PROD_MEAN       = 1.0
PROD_SD         = 0.11    # production-level distribution: truncated N(PROD_MEAN, PROD_SD)
SCENARIOS       = ["2025", "OG", "OB", "UG", "UB"]
SEEDS           = [42, 123, 456, 789, 1024]   # figures are averaged over these seeds (matches Fig 2/3)
SCEN_LABEL = {"2025": "2025 baseline", "OG": "oversupply / good price",
              "OB": "oversupply / bad price", "UG": "undersupply / good price",
              "UB": "undersupply / bad price"}
DEFAULT_MAC     = 5     # mid-lactation: a representative decision point with a clear boundary
KEEP_C, REPL_C  = "#2ca02c", "#d62728"

# Blue<->orange scale for the quantitative maps (avoids the green/red "good/bad" reading).
# Rule across all maps: blue = LOW value of the plotted quantity, orange = HIGH value.
BUOR = LinearSegmentedColormap.from_list(
    "BuOr", ["#2166ac", "#67a9cf", "#f7f7f7", "#fdb863", "#b35806"])


def value_curve(qf, parity, mac, prod_grid=PROD_GRID, mip=0, disease=0):
    """Retention value (Q_keep - Q_replace) across the production grid."""
    return np.array([qf.value((parity, mac, mip, disease, p)) for p in prod_grid])


def critical_prod(qf, parity, mac, mip=0, disease=0):
    """Production level M* where keep becomes optimal (value crosses 0, rising).
    Returns M* in [PROD_MIN, PROD_MAX], or PROD_MIN if kept even at the floor,
    or np.nan (shown as > PROD_MAX) if replaced even at the ceiling."""
    v = value_curve(qf, parity, mac, mip=mip, disease=disease)
    if v[0] >= 0:
        return PROD_MIN                      # kept even as the worst producer
    if v[-1] < 0:
        return np.nan                        # culled even as the best producer
    i = np.where((v[:-1] < 0) & (v[1:] >= 0))[0][0]
    x0, x1, y0, y1 = PROD_GRID[i], PROD_GRID[i + 1], v[i], v[i + 1]
    return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))    # linear interpolation


# ── Figure 1: decision map (continuous) ──────────────────────────────────────
def fig_decision_map(qf, tag, outdir, mac=DEFAULT_MAC):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    Z = np.array([value_curve(qf, p, mac) for p in PARITIES])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.contourf(PROD_GRID, PARITIES, np.where(Z >= 0, 1, 0),
                levels=[-0.5, 0.5, 1.5], colors=[REPL_C, KEEP_C], alpha=0.28)
    ax.contour(PROD_GRID, PARITIES, Z, levels=[0], colors="k", linewidths=2.2)
    ax.axvline(PROD_MEAN, color="gray", ls=":", lw=1)
    ax.set_xlabel("Production level  (multiplier on lactation curve)")
    ax.set_ylabel("Parity")
    ax.set_title(f"Optimal decision by parity x production level\n"
                 f"healthy open cow, MAC {mac} — {tag}")
    handles = [Patch(facecolor=KEEP_C, alpha=0.28, label="keep"),
               Patch(facecolor=REPL_C, alpha=0.28, label="replace"),
               Line2D([0], [0], color="k", lw=2.2, label="keep / replace boundary"),
               Line2D([0], [0], color="gray", ls=":", lw=1, label="average producer")]
    ax.set_xlim(PROD_MIN, PROD_MAX); ax.set_ylim(1, 12)
    ax.legend(handles=handles, loc="center left", fontsize=8, framealpha=0.9)
    _save(fig, outdir, f"1_decision_map_{tag}")


# ── Figure 2: critical threshold curve (continuous) — HERO ───────────────────
def fig_threshold_curve(qf, tag, outdir, macs=(1, 5, 9)):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for mac in macs:
        mstar = [critical_prod(qf, p, mac) for p in PARITIES]
        mstar = [PROD_MAX + 0.02 if (m is np.nan or (isinstance(m, float) and np.isnan(m)))
                 else m for m in mstar]
        ax.plot(PARITIES, mstar, "o-", lw=2, label=f"MAC {mac}")
    ax.axhline(PROD_MEAN, color="gray", ls=":", lw=1, label="average producer")
    ax.fill_between(PARITIES, PROD_MIN, PROD_MEAN, color="#67a9cf", alpha=0.06)
    ax.set_xlabel("Parity")
    ax.set_ylabel("Critical production level\n(keep if production ≥ this level)")
    ax.set_title(f"How productive must a cow be to be retained?\n{tag}")
    ax.set_xticks(PARITIES); ax.set_ylim(PROD_MIN, PROD_MAX + 0.03)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.text(0.02, 0.02, "higher line = agent demands more milk to keep the cow",
            transform=ax.transAxes, fontsize=8, style="italic", color="#555")
    _save(fig, outdir, f"2_threshold_curve_{tag}")


# ── Figure 3: retention value vs production level (continuous) ────────────────
def fig_value_vs_prod(qf, tag, outdir, mac=DEFAULT_MAC, parities=(1, 3, 5, 8, 11)):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for p in parities:
        ax.plot(PROD_GRID, value_curve(qf, p, mac), lw=2, label=f"parity {p}")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.axvline(PROD_MEAN, color="gray", ls=":", lw=1)
    ax.set_xlabel("Production level  (multiplier on lactation curve)")
    ax.set_ylabel("Retention value  =  Q(keep) − Q(replace)   [$]")
    ax.set_title(f"Retention value rises with production level\n"
                 f"healthy open cow, MAC {mac} — {tag}  (above 0 → keep)")
    ax.set_xlim(PROD_MIN, PROD_MAX); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _save(fig, outdir, f"3_value_vs_prod_{tag}")


# ── Figure 4: value surface heatmap (continuous) ─────────────────────────────
def fig_value_surface(qf, tag, outdir, mac=DEFAULT_MAC):
    Z = np.array([value_curve(qf, p, mac) for p in PARITIES])
    vmax = np.nanmax(np.abs(Z))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="RdYlGn",
                   norm=TwoSlopeNorm(0, -vmax, vmax),
                   extent=[PROD_MIN, PROD_MAX, PARITIES[0] - .5, PARITIES[-1] + .5])
    ax.contour(PROD_GRID, PARITIES, Z, levels=[0], colors="k", linewidths=1.5)
    ax.set_xlabel("Production level  (multiplier on lactation curve)")
    ax.set_ylabel("Parity")
    ax.set_title(f"Cow value ($) by parity x production level\n"
                 f"healthy open cow, MAC {mac} — {tag}")
    fig.colorbar(im, ax=ax, label="Q(keep) − Q(replace)  [$]")
    _save(fig, outdir, f"4_value_surface_{tag}")


# ── Figure 5: replacement MAC threshold, discrete producer categories ─────────
def fig_threshold_by_cat(qf, tag, outdir, cats=((0.8, "low (0.8)"),
                                                (1.0, "average (1.0)"),
                                                (1.2, "high (1.2)"))):
    macs = list(range(1, 21))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for prod, lbl in cats:
        thr = []
        for p in PARITIES:
            t = np.nan
            for mac in macs:
                if qf.value((p, mac, 0, 0, prod)) < 0:
                    t = mac; break
            thr.append(t)
        ax.plot(PARITIES, thr, "o-", lw=2, label=f"production {lbl}")
    ax.set_xlabel("Parity")
    ax.set_ylabel("MAC at which replacement becomes optimal\n(higher = kept longer)")
    ax.set_title(f"Replacement timing by producer category\n{tag}")
    ax.set_xticks(PARITIES); ax.set_ylim(0, 21); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _save(fig, outdir, f"5_threshold_by_category_{tag}")


# ── Figure 6: M* vs parity across scenarios (continuous) ─────────────────────
def fig_across_scenarios(qfuncs, outdir, mac=DEFAULT_MAC):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for scen, qf in qfuncs.items():
        mstar = [critical_prod(qf, p, mac) for p in PARITIES]
        mstar = [PROD_MAX + 0.02 if (isinstance(m, float) and np.isnan(m)) else m for m in mstar]
        ax.plot(PARITIES, mstar, "o-", lw=2, label=SCEN_LABEL.get(scen, scen))
    ax.axhline(PROD_MEAN, color="gray", ls=":", lw=1)
    ax.set_xlabel("Parity")
    ax.set_ylabel("Critical production level")
    ax.set_title(f"Retention threshold across market scenarios\n"
                 f"healthy open cow, MAC {mac} (mean of {len(SEEDS)} seeds)")
    ax.set_xticks(PARITIES); ax.set_ylim(PROD_MIN, PROD_MAX + 0.03)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _save(fig, outdir, "6_across_scenarios")


def critical_mac(qf, parity, prod, macs, mip=0, disease=0):
    """First MAC at which replacement becomes optimal (value < 0) for an open cow.
    Returns that MAC, or macs[-1]+1 if she is kept through the whole window."""
    for mac in macs:
        if qf.value((parity, mac, mip, disease, prod)) < 0:
            return mac
    return macs[-1] + 1


# ── Figure 7: critical production surface — M* over parity x MAC (health as 2 panels) ──
def fig_hurdle_map(qf, tag, outdir, macs=range(1, 16)):
    macs = list(macs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, disease, hlabel in zip(axes, (0, 1), ("healthy", "with mastitis")):
        Z = np.full((len(PARITIES), len(macs)), np.nan)
        for i, p in enumerate(PARITIES):
            for j, mac in enumerate(macs):
                Z[i, j] = critical_prod(qf, p, mac, disease=disease)   # PROD_MIN=easy, nan=always cull
        cull_all = np.isnan(Z)
        ax.set_facecolor("0.6")   # shows through where always-cull (nan)
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap=BUOR,
                       norm=TwoSlopeNorm(vcenter=PROD_MEAN, vmin=PROD_MIN, vmax=PROD_MAX),
                       extent=[macs[0] - .5, macs[-1] + .5, PARITIES[0] - .5, PARITIES[-1] + .5])
        cs = ax.contour(macs, PARITIES, np.where(cull_all, np.nan, Z),
                        levels=[0.8, 0.9, 1.0, 1.1, 1.2], colors="k", linewidths=0.8)
        ax.clabel(cs, fmt="%.1f", fontsize=7)
        ax.set_xlabel("Month after calving (MAC)")
        ax.set_title(f"{hlabel}")
    axes[0].set_ylabel("Parity")
    fig.colorbar(im, ax=axes,
                 label="Critical production level\n(min production to be KEPT; grey = cull at any level)",
                 fraction=0.046, pad=0.02)
    fig.suptitle(f"How productive must an open cow be to be retained?  —  {tag}",
                 fontsize=13, fontweight="bold")
    _save(fig, outdir, f"7_hurdle_map_{tag}", tight=False)


# ── Figure 8: retention-window surface — MAC* over parity x production (continuous) ──
def fig_stay_map(qf, tag, outdir, macs=range(1, 21), prod_grid=None, contour=False, suffix=""):
    """Two panels (healthy | with mastitis). Color = integer MAC at which replacement
    becomes optimal for an open cow; production level is a fine continuous axis."""
    macs = list(macs)
    prod_grid = np.linspace(PROD_MIN, PROD_MAX, 265) if prod_grid is None else prod_grid
    vmax = macs[-1] + 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, disease, hlabel in zip(axes, (0, 1), ("healthy", "with clinical mastitis")):
        Z = np.array([[critical_mac(qf, p, prod, macs, disease=disease) for prod in prod_grid]
                      for p in PARITIES], dtype=float)
        im = ax.pcolormesh(prod_grid, PARITIES, Z, cmap=BUOR, shading="auto", vmin=1, vmax=vmax)
        if contour:
            cs = ax.contour(prod_grid, PARITIES, Z, levels=[2, 4, 6, 8, 10],
                            colors="k", linewidths=0.9)
            ax.clabel(cs, fmt="MAC %d", fontsize=7)
        ax.axvline(PROD_MEAN, color="k", ls=":", lw=1)
        ax.set_xlabel("Production level  (multiplier on lactation curve)")
        ax.set_title(hlabel)
    axes[0].set_ylabel("Parity"); axes[0].set_yticks(PARITIES)
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02,
                        label="Month after calving when replacement becomes optimal\n"
                              "(high values = kept through lactation)")
    cbar.set_ticks(range(2, vmax + 1, 2))            # integer MAC ticks only
    cbar.ax.set_yticklabels([str(int(t)) for t in range(2, vmax + 1, 2)])
    fig.suptitle(f"How long can an open cow stay before culling is optimal?  —  {tag}",
                 fontsize=13, fontweight="bold")
    _save(fig, outdir, f"8_stay_map_{tag}{suffix}", tight=False)


def _save(fig, outdir, name, tight=True):
    os.makedirs(outdir, exist_ok=True)
    if tight:
        fig.tight_layout()
    path = os.path.join(outdir, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected_dir", default="collected/")
    ap.add_argument("--outdir", default="prod_outputs")
    ap.add_argument("--scenario", default="2025", help="scenario for per-scenario figs 1-5,7,8")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                    help="comma list of seeds to average over (default: all 5)")
    ap.add_argument("--mac", type=int, default=DEFAULT_MAC)
    ap.add_argument("--figs", default="1,2,3,4,5,6,7,8", help="comma list of figure numbers")
    ap.add_argument("--cmap", default=None,
                    help="override the heatmap colormap (e.g. coolwarm, viridis, PuOr, cividis, RdBu_r)")
    args = ap.parse_args()
    figs = set(x.strip() for x in args.figs.split(","))
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]

    if args.cmap:                       # let you test palettes without editing the file
        global BUOR
        BUOR = plt.get_cmap(args.cmap)
        print(f"Using colormap: {args.cmap}")

    qf = find_qfunc_ensemble(args.collected_dir, PREFIX, args.scenario, seeds)
    if qf is None:
        raise SystemExit(f"No _model.pth for {args.scenario} (seeds {seeds}) in {args.collected_dir}")
    n_loaded = len(getattr(qf, "members", [qf]))
    print(f"Rendering figures for {SCEN_LABEL.get(args.scenario, args.scenario)} "
          f"(mean of {n_loaded} seed(s)) -> {args.outdir}/")

    if "1" in figs: fig_decision_map(qf, args.scenario, args.outdir, mac=args.mac)
    if "2" in figs: fig_threshold_curve(qf, args.scenario, args.outdir)
    if "3" in figs: fig_value_vs_prod(qf, args.scenario, args.outdir, mac=args.mac)
    if "4" in figs: fig_value_surface(qf, args.scenario, args.outdir, mac=args.mac)
    if "5" in figs: fig_threshold_by_cat(qf, args.scenario, args.outdir)
    if "6" in figs:
        qfuncs = {}
        for scen in SCENARIOS:
            q = find_qfunc_ensemble(args.collected_dir, PREFIX, scen, seeds)
            if q is not None: qfuncs[scen] = q
        if qfuncs: fig_across_scenarios(qfuncs, args.outdir, mac=args.mac)
        else: print("  (fig 6 skipped: no models found)")
    if "7" in figs: fig_hurdle_map(qf, args.scenario, args.outdir)
    if "8" in figs:
        fig_stay_map(qf, args.scenario, args.outdir, contour=False, suffix="")
        fig_stay_map(qf, args.scenario, args.outdir, contour=True, suffix="_contour")
    print("Done.")


if __name__ == "__main__":
    main()