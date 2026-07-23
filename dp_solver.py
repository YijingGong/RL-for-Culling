"""
Exact dynamic-programming (value-iteration) solution of the dairy-cow
replacement MDP defined in cow_environment2.py.

This module is the ground-truth benchmark for the DQN agent. It reuses the
*identical* economic/biological parameters (animal_constants_*.py) and the
*identical* reward functions (CowEnv.calculate_*), and reproduces the exact
state-transition logic of CowEnv.step() analytically (enumerating every
stochastic branch with its probability) rather than by sampling. The result is
the exact optimal value function V*, action-value function Q*, and optimal
policy for the infinite-horizon discounted problem (gamma = 0.95), the same
objective the DQN approximates.

Outputs (per scenario), written to outputs/dp/:
  - dp_<scn>_qtable.pkl     : {state: {'keep':Q, 'replace':Q}}  (same format as DQN q_table)
  - dp_<scn>_policy.csv      : state, V, Q_keep, Q_replace, cow_value, action
  - dp_<scn>_pregnancy.csv   : value of a new pregnancy by parity x conception MAC
  - dp_<scn>_cm_cost.csv     : cost of clinical mastitis by parity x lactation stage
  - dp_<scn>_herd.json       : annual return/stall, culling rates, mean parity, productive life
"""
import os, sys, json, pickle, argparse
import numpy as np
import cow_environment2 as ce
import utility

PARITY_RANGE = range(13)   # 0..12  (0 = springer)
MAC_RANGE    = range(21)   # 0..20
MIP_RANGE    = range(10)   # 0..9
DIS_RANGE    = range(2)    # 0/1
GAMMA        = 0.95
SPR          = (0, 0, 9, 0)   # springer state a replaced stall returns to
MAX_PARITY   = 12
MAX_MAC      = 20
SCENARIOS    = ['2025', 'OG', 'OB', 'UG', 'UB']


def all_states():
    S = []
    for p in PARITY_RANGE:
        for m in MAC_RANGE:
            for ip in MIP_RANGE:
                for d in DIS_RANGE:
                    s = (p, m, ip, d)
                    if utility.possible_state2(s, PARITY_RANGE, MAC_RANGE, MIP_RANGE,
                                               DIS_RANGE, dnb_min=3, dnb_max=10):
                        S.append(s)
    return S


def disease_dist(ac, parity, disease):
    """Distribution of next disease status given current. Returns [(prob, next_disease)]."""
    if disease == 1:
        rec = ac.MASTITIS_RECOVER_RATE
        return [(rec, 0), (1.0 - rec, 1)]
    risk = ac.MASTITIS_DISEASE_RISK[parity]
    return [(1.0 - risk, 0), (risk, 1)]


def breed_success_prob(ac, parity, mac, disease):
    health = max(0.0, ac.CONCEPTION_RATE[parity] - (mac - 3) * ac.CONCEPTION_RATE_DROP)
    p = health if disease == 0 else health * ac.MASTITIS_SICK_CONCEPTION_RATE_MULTIPLIER
    return min(max(p, 0.0), 1.0)


def transitions(env, ac, state, action):
    """Exact list of (prob, reward, next_state), mirroring CowEnv.step()."""
    parity, mac, mip, disease = state
    feed = env.calculate_feed_cost(parity, mac, mip, disease)

    if action == 'replace':
        slaughter = env.calculate_slaughter_income(parity, disease)
        return [(1.0, slaughter - ac.REPLACEMENT_COST - feed, SPR)]

    # ---- action == 'keep' ----
    out = []
    base_death = ac.DEATH_RATE[parity] / 100.0
    pd = base_death * (ac.MASTITIS_SICK_DEATH_RATE_MULTIPLIER if disease == 1 else 1.0)
    pd = min(max(pd, 0.0), 1.0)
    if pd > 0:
        out.append((pd, 0.0 - ac.REPLACEMENT_COST - feed, SPR))   # on-farm death
    p_surv = 1.0 - pd
    if p_surv <= 0:
        return out

    milk = env.calculate_milk_income(parity, mac, mip, disease)
    next_mac = mac + 1

    if mip == 9:  # calving
        calf = ac.CALF_PRICE
        next_parity = parity + 1
        if next_parity > MAX_PARITY:                      # dies after final calving (returns before disease block)
            out.append((p_surv, milk + calf - ac.REPLACEMENT_COST - feed, SPR))
            return out
        treatment = ac.MASTITIS_TREATMENT_COST_PER_MONTH if disease == 1 else 0.0
        r = milk + calf - treatment - feed
        for p_dis, nd in disease_dist(ac, parity, disease):
            out.append((p_surv * p_dis, r, (next_parity, 1, 0, nd)))
        return out

    if mip == 0:  # open -> possible breeding
        breed_attempt = (3 <= mac <= 10)
        breed_cost = ac.BREED_COST_PER_INSEM if breed_attempt else 0.0
        if next_mac > MAX_MAC:                            # reaches max lactation length while open
            if parity == MAX_PARITY:                      # forced exit, no slaughter (dies)
                r = milk - ac.REPLACEMENT_COST - feed
            else:                                         # forced exit, slaughter income (code drops breed/treatment cost here)
                r = milk + env.calculate_slaughter_income(parity, disease) - ac.REPLACEMENT_COST - feed
            out.append((p_surv, r, SPR))
            return out
        treatment = ac.MASTITIS_TREATMENT_COST_PER_MONTH if disease == 1 else 0.0
        r = milk - breed_cost - treatment - feed
        if breed_attempt:
            pb = breed_success_prob(ac, parity, mac, disease)
            mip_out = [(pb, 1), (1.0 - pb, 0)]
        else:
            mip_out = [(1.0, 0)]
        for p_mip, nmip in mip_out:
            for p_dis, nd in disease_dist(ac, parity, disease):
                out.append((p_surv * p_mip * p_dis, r, (parity, next_mac, nmip, nd)))
        return out

    # mip in 1..8 : continue pregnancy
    if next_mac > MAX_MAC:
        if parity == MAX_PARITY:
            r = milk - ac.REPLACEMENT_COST - feed
        else:
            r = milk + env.calculate_slaughter_income(parity, disease) - ac.REPLACEMENT_COST - feed
        out.append((p_surv, r, SPR))
        return out
    treatment = ac.MASTITIS_TREATMENT_COST_PER_MONTH if disease == 1 else 0.0
    r = milk - treatment - feed
    for p_dis, nd in disease_dist(ac, parity, disease):
        out.append((p_surv * p_dis, r, (parity, next_mac, mip + 1, nd)))
    return out


def solve(scn, tol=1e-7, max_iter=200000):
    ac = ce.set_scenario(scn)
    env = ce.CowEnv(PARITY_RANGE, MAC_RANGE, MIP_RANGE, DIS_RANGE)
    S = all_states()
    idx = {s: i for i, s in enumerate(S)}
    trans = {s: {'keep': transitions(env, ac, s, 'keep'),
                 'replace': transitions(env, ac, s, 'replace')} for s in S}
    for s in S:                                            # sanity: probabilities sum to 1
        for a in ('keep', 'replace'):
            tot = sum(p for p, _, _ in trans[s][a])
            assert abs(tot - 1.0) < 1e-9, f"prob sum {tot} for {s},{a}"

    V = np.zeros(len(S))
    for it in range(max_iter):
        delta = 0.0
        newV = np.empty(len(S))
        for i, s in enumerate(S):
            best = -1e18
            for a in ('keep', 'replace'):
                q = 0.0
                for p, r, s2 in trans[s][a]:
                    q += p * (r + GAMMA * V[idx[s2]])
                if q > best:
                    best = q
            newV[i] = best
            delta = max(delta, abs(best - V[i]))
        V = newV
        if delta < tol:
            break
    Q = {}
    for s in S:
        qk = sum(p * (r + GAMMA * V[idx[s2]]) for p, r, s2 in trans[s]['keep'])
        qr = sum(p * (r + GAMMA * V[idx[s2]]) for p, r, s2 in trans[s]['replace'])
        Q[s] = {'keep': qk, 'replace': qr}
    Vd = {s: V[idx[s]] for s in S}
    return S, idx, trans, Vd, Q, it + 1, delta


def stationary_metrics(S, idx, trans, Q):
    """Steady-state herd metrics under the optimal (greedy) policy."""
    n = len(S)
    pol = {s: ('keep' if Q[s]['keep'] >= Q[s]['replace'] else 'replace') for s in S}
    P = np.zeros((n, n))
    exp_rew = np.zeros(n)
    for i, s in enumerate(S):
        a = pol[s]
        for p, r, s2 in trans[s][a]:
            P[i, idx[s2]] += p
            exp_rew[i] += p * r
    pi = np.ones(n) / n
    for _ in range(200000):
        nxt = pi @ P
        if np.abs(nxt - pi).max() < 1e-13:
            pi = nxt
            break
        pi = nxt
    pi = pi / pi.sum()

    monthly_reward = float(pi @ exp_rew)
    annual_return = monthly_reward * 12.0

    vol = mort = invol = 0.0
    for i, s in enumerate(S):
        parity, mac, mip, dis = s
        a = pol[s]
        if a == 'replace':
            vol += pi[i]
            continue
        for p, r, s2 in trans[s]['keep']:
            if s2 == SPR:
                if parity == MAX_PARITY and (mip == 9 or mac == MAX_MAC):
                    invol += pi[i] * p
                elif mac == MAX_MAC and mip != 9:
                    invol += pi[i] * p
                else:
                    mort += pi[i] * p
    monthly_exit = vol + mort + invol
    annual_cull = 12.0 * monthly_exit
    mean_parity = float(sum(pi[i] * s[0] for i, s in enumerate(S)))
    prod_life = (1.0 / annual_cull) if annual_cull > 0 else float('nan')
    return {
        'annual_return_per_stall': annual_return,
        'monthly_reward': monthly_reward,
        'annual_culling_rate_pct': annual_cull * 100,
        'culling_voluntary_pct': 12.0 * vol * 100,
        'culling_mortality_pct': 12.0 * mort * 100,
        'culling_involuntary_pct': 12.0 * invol * 100,
        'mean_parity_incl_springer': mean_parity,
        'productive_life_yr': prod_life,
        'springer_time_fraction': float(sum(pi[i] for i, s in enumerate(S) if s == SPR)),
    }


def cow_value(Q, s):
    """Cow value = retention payoff = Q_keep - Q_replace (manuscript Eqn 4;
    De Vries, 2006)."""
    return Q[s]['keep'] - Q[s]['replace']


def pregnancy_values(Q):
    """Value of a new pregnancy at conception MAC m (healthy), computed as the
    difference in cow value (Q_keep - Q_replace; Eqn 4) between a pregnant and an
    open cow in the same month, consistent with De Vries (2006). No positive-RPO
    screen is applied, so pregnancy value and CM cost share one definition."""
    rows = []
    for p in range(1, 13):
        for m in range(3, 11):
            preg = (p, m + 1, 1, 0); open_ = (p, m + 1, 0, 0)
            if preg in Q and open_ in Q:
                rows.append({'parity': p, 'conception_mac': m,
                             'value_new_pregnancy': cow_value(Q, preg) - cow_value(Q, open_)})
    return rows


def cm_costs(Q):
    """Cost of CM for open cows, computed as the difference in cow value
    (Q_keep - Q_replace; Eqn 4) between a healthy and a CM cow in the same month
    (Bar et al., 2008; Cha et al., 2011)."""
    rows = []
    for p in range(1, 13):
        for mac in range(1, 21):
            h = (p, mac, 0, 0); s = (p, mac, 0, 1)
            if h in Q and s in Q:
                rows.append({'parity': p, 'mac': mac, 'cm_cost': cow_value(Q, h) - cow_value(Q, s)})
    return rows


def stage_avg(cm_rows):
    bins = {'Early (MAC 1-3)': range(1, 4), 'Mid (MAC 4-6)': range(4, 7),
            'Late (MAC 7-9)': range(7, 10), 'Extended (MAC>=10)': range(10, 21)}
    out = {}
    for name, rng in bins.items():
        vals = [r['cm_cost'] for r in cm_rows if r['mac'] in rng]
        out[name] = float(np.mean(vals)) if vals else float('nan')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='all')
    ap.add_argument('--outdir', default='outputs/dp')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    scns = SCENARIOS if args.scenario == 'all' else [args.scenario]
    summary = {}
    for scn in scns:
        print(f"\n===== Solving DP for scenario {scn} =====")
        S, idx, trans, Vd, Q, iters, delta = solve(scn)
        print(f"  states={len(S)}  iterations={iters}  final_delta={delta:.2e}")
        herd = stationary_metrics(S, idx, trans, Q)
        preg = pregnancy_values(Q)
        cm = cm_costs(Q)
        stages = stage_avg(cm)
        with open(f"{args.outdir}/dp_{scn}_qtable.pkl", 'wb') as f:
            pickle.dump(Q, f)
        with open(f"{args.outdir}/dp_{scn}_policy.csv", 'w') as f:
            f.write("parity,mac,mip,cm,V,Q_keep,Q_replace,cow_value,action\n")
            for s in S:
                qk, qr = Q[s]['keep'], Q[s]['replace']
                f.write(f"{s[0]},{s[1]},{s[2]},{s[3]},{Vd[s]:.2f},{qk:.2f},{qr:.2f},"
                        f"{qk-qr:.2f},{'keep' if qk>=qr else 'replace'}\n")
        with open(f"{args.outdir}/dp_{scn}_pregnancy.csv", 'w') as f:
            f.write("parity,conception_mac,value_new_pregnancy\n")
            for r in preg:
                f.write(f"{r['parity']},{r['conception_mac']},{r['value_new_pregnancy']:.2f}\n")
        with open(f"{args.outdir}/dp_{scn}_cm_cost.csv", 'w') as f:
            f.write("parity,mac,cm_cost\n")
            for r in cm:
                f.write(f"{r['parity']},{r['mac']},{r['cm_cost']:.2f}\n")
        herd['cm_cost_by_stage'] = stages
        for pp in (1, 2, 3):
            vals = [r['value_new_pregnancy'] for r in preg if r['parity'] == pp]
            herd[f'avg_preg_value_parity{pp}'] = float(np.mean(vals)) if vals else None
        with open(f"{args.outdir}/dp_{scn}_herd.json", 'w') as f:
            json.dump(herd, f, indent=2)
        summary[scn] = herd
        print(f"  annual_return/stall=${herd['annual_return_per_stall']:.0f}  "
              f"culling={herd['annual_culling_rate_pct']:.1f}%  "
              f"mean_parity={herd['mean_parity_incl_springer']:.2f}  "
              f"prod_life={herd['productive_life_yr']:.2f}yr")
        print("  CM cost by stage: " +
              ", ".join(f"{k.split('(')[0].strip()}=${v:.0f}" for k, v in stages.items()))
        print(f"  avg preg value P1/P2/P3 = "
              f"${herd['avg_preg_value_parity1']:.0f}/${herd['avg_preg_value_parity2']:.0f}/"
              f"${herd['avg_preg_value_parity3']:.0f}")
    with open(f"{args.outdir}/dp_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote DP outputs to {args.outdir}/")


if __name__ == '__main__':
    main()
