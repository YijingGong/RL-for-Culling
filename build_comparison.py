"""Build a markdown table comparing the exact DP solution to the DRL (DQN)
results reported in the manuscript. DRL numbers are transcribed from the paper
(Abstract, Table 4, Table 5). Run dp_solver.py first to create outputs/dp/."""
import json

dp = json.load(open('dp_results/dp_summary.json'))

# ---- DRL numbers as reported in the manuscript ----
DRL = {
    '2025': dict(ret=3082, cull=28.5, parity=2.71, life=3.5, p1=418, p2=313, p3=231),
    'OG':   dict(ret=None, cull=21.4, parity=3.40, life=4.7, p1=None, p2=None, p3=None),
    'OB':   dict(ret=1036, cull=21.1, parity=3.43, life=4.7, p1=None, p2=None, p3=None),
    'UG':   dict(ret=4545, cull=22.8, parity=3.27, life=4.4, p1=None, p2=None, p3=None),
    'UB':   dict(ret=None, cull=23.9, parity=3.08, life=4.2, p1=None, p2=None, p3=None),
}
NAME = {'2025': 'BL', 'OG': 'OG', 'OB': 'OB', 'UG': 'UG', 'UB': 'UB'}

def f(x, money=False):
    if x is None:
        return 'n/r'
    return (f"${x:,.0f}" if money else f"{x:.1f}")

lines = []
lines.append("# Exact DP benchmark vs. DRL (DQN) — comparison\n")
lines.append("Exact value-iteration solution of the same MDP (identical parameters and reward "
             "functions; gamma = 0.95) compared against the DRL results reported in the manuscript. "
             "\"n/r\" = not reported numerically in the current draft.\n")

lines.append("## Annual return per stall ($/yr)\n")
lines.append("| Scenario | DRL (paper) | DP (exact) | Difference |")
lines.append("|---|---|---|---|")
for k in ['2025', 'OG', 'OB', 'UG', 'UB']:
    d = dp[k]['annual_return_per_stall']; r = DRL[k]['ret']
    diff = f"{(d-r)/r*100:+.1f}%" if r else "—"
    lines.append(f"| {NAME[k]} | {f(r,1)} | ${d:,.0f} | {diff} |")

lines.append("\n## Annual culling rate (%) and herd structure\n")
lines.append("| Scenario | DRL cull | DP cull | DRL parity | DP parity | DRL life | DP life |")
lines.append("|---|---|---|---|---|---|---|")
for k in ['2025', 'OG', 'OB', 'UG', 'UB']:
    h = dp[k]
    lines.append(f"| {NAME[k]} | {f(DRL[k]['cull'])} | {h['annual_culling_rate_pct']:.1f} | "
                 f"{f(DRL[k]['parity'])} | {h['mean_parity_incl_springer']:.2f} | "
                 f"{f(DRL[k]['life'])} | {h['productive_life_yr']:.2f} |")

lines.append("\n## Value of a new pregnancy, baseline (avg MAC 3-10, $)\n")
lines.append("| Parity | DRL (paper) | DP (exact) |")
lines.append("|---|---|---|")
for p in (1, 2, 3):
    lines.append(f"| {p} | {f(DRL['2025'][f'p{p}'],1)} | ${dp['2025'][f'avg_preg_value_parity{p}']:,.0f} |")

lines.append("\n## Cost of clinical mastitis by lactation stage, baseline ($)\n")
DRL_CM = {'Early (MAC 1-3)': 387, 'Mid (MAC 4-6)': 339, 'Late (MAC 7-9)': 293, 'Extended (MAC>=10)': 147}
lines.append("| Stage | DRL (paper) | DP (exact) |")
lines.append("|---|---|---|")
for stage, val in DRL_CM.items():
    lines.append(f"| {stage} | ${val} | ${dp['2025']['cm_cost_by_stage'][stage]:,.0f} |")

open('dp_results/dp_vs_dqn_summary.md', 'w').write("\n".join(lines) + "\n")
print("\n".join(lines))
