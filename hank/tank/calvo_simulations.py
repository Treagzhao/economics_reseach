"""
Simulation engine for "HANK After Cantillon" model.

Compares two price-adjustment rules:
  - compact (paper's):      P_t = min(P_t^*, P_{t-1}(1+g))
  - calvo   (correct):      P_t = (1-g)P_{t-1} + g P_t^*

Core parameters:
  Y0, P0  = baseline nominal income and price level (normalization)
  tau_R   = month when Ricardian agent R receives the injection
  Delta_tau = receipt gap (months), tau_H = tau_R + Delta_tau
  g       = monthly price adjustment rate (from Calvo theta_q = 0.75)
  mu_R, mu_H = injection sizes (% of own baseline income)
  beta    = monthly discount factor
"""

import math
from dataclasses import dataclass
from typing import Optional

# ============================================================
# Default parameters (paper Table 1)
# ============================================================
Y0 = 100.0
P0 = 1.0
TAU_R = 1
G = 0.091          # monthly Calvo: theta_q=0.75 -> theta_m=0.75^(1/3)=0.909 -> g=0.091
BETA = (1.05) ** (-1 / 12)  # 5% annual discount rate
I_STAR = BETA ** (-1) - 1   # steady-state monthly nominal rate


@dataclass
class Result:
    """Results from a single simulation run."""
    delta_star: float
    cantillon: float       # discounted sum of real-income deviations
    principal_gain: float  # discounted principal erosion per unit delta
    terminal_price: float
    peak_inflation: float  # max monthly inflation rate
    label: str = ""


# ============================================================
# Price path simulators
# ============================================================

def simulate_compact(mu_R: float, mu_H: float, delta_tau: int,
                     beta: float = BETA, phi_pi: float = 0.0,
                     sigma_ad: float = 0.0, floating_share: float = 0.0,
                     alpha: float = 1.0,
                     t_max_override: Optional[int] = None) -> Result:
    """
    Paper's compact form: P_t = min(P_t^*, P_{t-1}(1+g)).

    floating_share: fraction of H's debt that reprices at current i_t (0 = all fixed).
    alpha: baseline income ratio Y_R/Y_H (default 1 = equal incomes).
    """
    mu_bar = (alpha * mu_R + mu_H) / (1 + alpha)
    tau_h = TAU_R + delta_tau
    t2 = 1
    t_terminal = tau_h + t2
    t_max = t_max_override or max(tau_h + 300, 800)

    p_star_ii = P0 * (1 + alpha * mu_R / (1 + alpha))
    p_star_iii = P0 * (1 + mu_bar)

    P = [P0] * (t_max + 1)
    i_path = [I_STAR] * (t_max + 1)
    peak_infl = 0.0

    for t in range(t_max + 1):
        if t <= TAU_R:
            P[t] = P0
        else:
            infl = (P[t - 1] / P[t - 2] - 1) if t >= TAU_R + 2 else 0.0
            peak_infl = max(peak_infl, infl)
            i_path[t] = max(0.0, I_STAR + phi_pi * infl)

            p_star_base = p_star_ii if t < tau_h else p_star_iii
            rate_gap = i_path[t] - I_STAR
            p_star_eff = p_star_base * (1 - sigma_ad * rate_gap)
            p_star_eff = max(p_star_eff, P0 * 0.99)

            P[t] = min(p_star_eff, P[t - 1] * (1 + G))

    # Cantillon sum
    cant = 0.0
    for t in range(t_max + 1):
        y_h = Y0 if t < tau_h else Y0 * (1 + mu_H)
        cant += (beta ** t) * (y_h / P[t] - Y0 / P0)

    # Balance-sheet: principal erosion (applies to ALL debt)
    p_terminal = P[-1]
    pg = (Y0 / P0) * (1 - P0 / p_terminal)
    bs_principal = (beta ** t_terminal) * pg

    # TR penalty: extra interest on floating-rate share
    tr_penalty = 0.0
    if floating_share > 0:
        for t in range(t_max + 1):
            extra = max(0.0, i_path[t] - I_STAR)
            tr_penalty += (beta ** t) * extra / P[t]
        tr_penalty *= Y0 * floating_share

    bs_gain = bs_principal - tr_penalty
    ds = -cant / bs_gain if abs(bs_gain) > 1e-10 else float('inf')
    return Result(delta_star=ds, cantillon=cant, principal_gain=bs_gain,
                  terminal_price=p_terminal, peak_inflation=peak_infl)


def simulate_calvo(mu_R: float, mu_H: float, delta_tau: int,
                   beta: float = BETA, phi_pi: float = 0.0,
                   sigma_ad: float = 0.0, floating_share: float = 0.0,
                   alpha: float = 1.0,
                   t_max_override: Optional[int] = None) -> Result:
    """
    Correct Calvo formula: P_t = (1-g)P_{t-1} + g P_t^*.

    floating_share: fraction of H's debt that reprices at current i_t (0 = all fixed).
    alpha: baseline income ratio Y_R/Y_H (default 1 = equal incomes).
    """
    mu_bar = (alpha * mu_R + mu_H) / (1 + alpha)
    tau_h = TAU_R + delta_tau
    t2 = 1
    t_terminal = tau_h + t2
    t_max = t_max_override or max(tau_h + 500, 1500)

    p_star_ii = P0 * (1 + alpha * mu_R / (1 + alpha))
    p_star_iii = P0 * (1 + mu_bar)

    P = [P0] * (t_max + 1)
    i_path = [I_STAR] * (t_max + 1)
    peak_infl = 0.0

    for t in range(t_max + 1):
        if t <= TAU_R:
            P[t] = P0
        else:
            infl = (P[t - 1] / P[t - 2] - 1) if t >= TAU_R + 2 else 0.0
            peak_infl = max(peak_infl, infl)

            i_path[t] = max(0.0, I_STAR + phi_pi * infl)

            p_star_base = p_star_ii if t < tau_h else p_star_iii
            rate_gap = i_path[t] - I_STAR
            p_star_eff = p_star_base * (1 - sigma_ad * rate_gap)
            p_star_eff = max(p_star_eff, P0 * 0.995)

            P[t] = (1 - G) * P[t - 1] + G * p_star_eff

    # Cantillon sum
    cant = 0.0
    for t in range(t_max + 1):
        y_h = Y0 if t < tau_h else Y0 * (1 + mu_H)
        cant += (beta ** t) * (y_h / P[t] - Y0 / P0)

    # Balance-sheet: principal erosion (applies to ALL debt)
    p_terminal = P[-1]
    pg = (Y0 / P0) * (1 - P0 / p_terminal)
    bs_principal = (beta ** t_terminal) * pg

    # TR penalty: extra interest on floating-rate share
    tr_penalty = 0.0
    if floating_share > 0:
        for t in range(t_max + 1):
            extra = max(0.0, i_path[t] - I_STAR)
            tr_penalty += (beta ** t) * extra / P[t]
        tr_penalty *= Y0 * floating_share

    bs_gain = bs_principal - tr_penalty
    ds = -cant / bs_gain if abs(bs_gain) > 1e-10 else float('inf')
    return Result(delta_star=ds, cantillon=cant, principal_gain=bs_gain,
                  terminal_price=p_terminal, peak_inflation=peak_infl)


# ============================================================
# High-level comparison functions
# ============================================================

def compare_formulas(mu_R: float, mu_H: float, delta_tau: int = 18,
                     beta: float = BETA) -> dict:
    """Run both compact and calvo, return comparison dict."""
    r_c = simulate_compact(mu_R, mu_H, delta_tau, beta)
    r_k = simulate_calvo(mu_R, mu_H, delta_tau, beta)
    return {
        'mu_R': mu_R, 'mu_H': mu_H, 'delta_tau': delta_tau,
        'compact_ds': r_c.delta_star, 'calvo_ds': r_k.delta_star,
        'compact_cant': r_c.cantillon, 'calvo_cant': r_k.cantillon,
        'compact_tp': r_c.terminal_price, 'calvo_tp': r_k.terminal_price,
        'diff': r_k.delta_star - r_c.delta_star,
    }


def sweep_asymmetry(delta_tau: int = 18, beta: float = BETA,
                    use_calvo: bool = True):
    """Sweep asymmetry ratios from 1:1 to inf:1, holding mu_bar=0.10."""
    sim = simulate_calvo if use_calvo else simulate_compact
    ratios = [
        ("1:1", 0.100, 0.100),
        ("1.3:1", 0.113, 0.087),
        ("1.5:1", 0.120, 0.080),
        ("2:1", 0.133, 0.067),
        ("3:1", 0.150, 0.050),
        ("5:1", 0.167, 0.033),
        ("10:1", 0.182, 0.018),
        ("20:1", 0.191, 0.010),
        ("inf:1", 0.200, 0.000),
    ]
    results = []
    for name, mu_r, mu_h in ratios:
        r = sim(mu_r, mu_h, delta_tau, beta)
        r.label = name
        results.append(r)
    return results


def sweep_delta_tau(mu_R: float, mu_H: float,
                    dts: list = None, beta: float = BETA,
                    use_calvo: bool = True):
    """Sweep receipt gaps for given injection parameters."""
    if dts is None:
        dts = [0, 6, 12, 18, 24, 30, 36]
    sim = simulate_calvo if use_calvo else simulate_compact
    results = []
    for dt in dts:
        r = sim(mu_R, mu_H, dt, beta)
        r.label = f"Dt={dt}"
        results.append(r)
    return results


def sweep_income_inequality(mu_R: float, mu_H: float, delta_tau: int = 18,
                            beta: float = BETA, use_calvo: bool = True):
    """
    Sweep Y_R/Y_H ratios (alpha).

    When alpha > 1, R earns more than H in steady state.
    This increases the plateau price (R's expenditure has more weight)
    and changes the weighted-average terminal injection.
    """
    sim = simulate_calvo if use_calvo else simulate_compact
    alphas = [1, 1.5, 2, 3, 5, 10, 20, 50, 100]
    results = []
    for alpha in alphas:
        r = sim_unequal_income(alpha, mu_R, mu_H, delta_tau, beta, use_calvo)
        r.label = f"alpha={alpha}"
        results.append(r)
    return results


def sim_unequal_income(alpha: float, mu_R: float, mu_H: float,
                       delta_tau: int, beta: float, use_calvo: bool) -> Result:
    """
    Run simulation with Y_R = alpha * Y_H instead of Y_R = Y_H.

    Key changes:
    - Total income = (1+alpha)*Y_H, real output y = (1+alpha)*Y_H/P0
    - Notional P_star_II = P0 * [1 + alpha*mu_R/(1+alpha)]
    - Notional P_star_III = P0 * [1 + (alpha*mu_R + mu_H)/(1+alpha)]
    - Cantillon loss per period scales with alpha*mu_R/(1+alpha+alpha*mu_R)
    """
    y_h = Y0  # keep H's income as normalization baseline
    total_y = (1 + alpha) * y_h
    real_output = total_y / P0

    mu_bar_eff = (alpha * mu_R + mu_H) / (1 + alpha)

    tau_h = TAU_R + delta_tau
    t2 = 1
    t_terminal = tau_h + t2
    t_max = max(tau_h + 500, 1500) if use_calvo else max(tau_h + 300, 800)

    p_star_ii = P0 * (1 + alpha * mu_R / (1 + alpha))
    p_star_iii = P0 * (1 + mu_bar_eff)

    P = [P0] * (t_max + 1)

    for t in range(t_max + 1):
        if t <= TAU_R:
            P[t] = P0
        else:
            p_star = p_star_ii if t < tau_h else p_star_iii
            if use_calvo:
                P[t] = (1 - G) * P[t - 1] + G * p_star
            else:
                P[t] = min(p_star, P[t - 1] * (1 + G))

    cant = 0.0
    for t in range(t_max + 1):
        y_h_t = y_h if t < tau_h else y_h * (1 + mu_H)
        cant += (beta ** t) * (y_h_t / P[t] - y_h / P0)

    p_terminal = P[-1]
    pg = (y_h / P0) * (1 - P0 / p_terminal)
    bs_gain = (beta ** t_terminal) * pg

    ds = -cant / bs_gain if abs(bs_gain) > 1e-10 else float('inf')
    return Result(delta_star=ds, cantillon=cant, principal_gain=bs_gain,
                  terminal_price=p_terminal, peak_inflation=0.0)


def simulate_three_agent(alpha_a: float, alpha_b: float,
                         mu_a: float, mu_b: float, mu_c: float,
                         tau_a: int, tau_b: int, tau_c: int,
                         beta: float = BETA, use_calvo: bool = True) -> Result:
    """
    3-agent extension: A (early), B (intermediate), C (late, debtor).

    Incomes: Y_A = alpha_a * Y_C, Y_B = alpha_b * Y_C, Y_C = Y0.
    The 2-agent model is recovered when alpha_b = 0, tau_b = tau_a.
    """
    alpha_total = alpha_a + alpha_b + 1
    y_c = Y0
    total_y = (alpha_a + alpha_b + 1) * y_c
    real_output = total_y / P0

    # Income-weighted average injection
    mu_bar = (alpha_a * mu_a + alpha_b * mu_b + mu_c) / alpha_total

    t2 = 1
    t_terminal = tau_c + t2
    t_max = max(tau_c + 500, 1500) if use_calvo else max(tau_c + 300, 800)

    # Notional prices for each regime
    p_star_ii = P0 * (1 + alpha_a * mu_a / alpha_total)
    p_star_iii = P0 * (1 + (alpha_a * mu_a + alpha_b * mu_b) / alpha_total)
    p_star_iv = P0 * (1 + mu_bar)

    P = [P0] * (t_max + 1)

    for t in range(t_max + 1):
        if t <= tau_a:
            P[t] = P0
        else:
            if t < tau_b:
                p_star = p_star_ii
            elif t < tau_c:
                p_star = p_star_iii
            else:
                p_star = p_star_iv

            if use_calvo:
                P[t] = (1 - G) * P[t - 1] + G * p_star
            else:
                P[t] = min(p_star, P[t - 1] * (1 + G))

    # Cantillon sum for agent C
    cant = 0.0
    for t in range(t_max + 1):
        y_c_t = y_c if t < tau_c else y_c * (1 + mu_c)
        cant += (beta ** t) * (y_c_t / P[t] - y_c / P0)

    # Balance-sheet gain for agent C
    p_terminal = P[-1]
    pg = (y_c / P0) * (1 - P0 / p_terminal)
    bs_gain = (beta ** t_terminal) * pg

    ds = -cant / bs_gain if abs(bs_gain) > 1e-10 else float('inf')
    return Result(delta_star=ds, cantillon=cant, principal_gain=bs_gain,
                  terminal_price=p_terminal, peak_inflation=0.0)


def verify_n_agent_convergence():
    """
    Compare 3-agent δ* for terminal agent against 2-agent benchmark.
    Returns list of (description, δ*_2agent, δ*_3agent, pct_diff).
    """
    results = []
    beta = BETA
    dt_total = 18  # τ_C - τ_A

    # Baseline 2-agent: α=5, 5:1, Δτ=18
    alpha_2 = 5.0
    mu_r_2 = 0.1154
    mu_h_2 = 0.0231
    r2 = sim_unequal_income(alpha_2, mu_r_2, mu_h_2, dt_total, beta, use_calvo=True)

    # 3-agent: split R into A and B
    # α_A + α_B = 5 (same total early income weight)
    # α_A*μ_A + α_B*μ_B = α*μ_R = 5*0.1154 = 0.577 (same income-weighted early injection)
    # μ_C = μ_H = 0.0231
    # τ_C - τ_A = 18 (same total gap)
    configs = [
        # (desc, α_A, α_B, μ_A, μ_B, τ_A, τ_B, τ_C)
        ("A=80% early weight, midway", 4.0, 1.0, 0.1154*4/5 + 0.02, 0.1154, 1, 10, 19),
        ("A=50% early weight, midway", 2.5, 2.5, 0.1154, 0.1154, 1, 10, 19),
        ("A=80% early, B at 1/3", 4.0, 1.0, 0.13, 0.08, 1, 7, 19),
        ("A=50% early, B at 2/3", 2.5, 2.5, 0.10, 0.12, 1, 13, 19),
    ]

    for desc, aa, ab, ma, mb, ta, tb, tc in configs:
        # Adjust μ_A so income-weighted early injection matches 2-agent
        # α_A*μ_A + α_B*μ_B = 5*0.1154 = 0.577
        target_early = alpha_2 * mu_r_2
        ma_adj = (target_early - ab * mb) / aa
        r3 = simulate_three_agent(aa, ab, ma_adj, mb, mu_h_2, ta, tb, tc, beta, use_calvo=True)
        pct = (r3.delta_star - r2.delta_star) / r2.delta_star * 100
        results.append((desc, r2.delta_star, r3.delta_star, pct))

    return results


def sweep_discount_rates(mu_R: float, mu_H: float, delta_tau: int = 18,
                         use_calvo: bool = True):
    """Sweep annual discount rates from 0% to extreme values."""
    sim = simulate_calvo if use_calvo else simulate_compact
    annual_rates = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10,
                    0.20, 0.50, 1.0, 2.0, 5.0, 10.0]
    results = []
    for r_annual in annual_rates:
        if r_annual == 0.0:
            b = 1.0
        else:
            b = (1 + r_annual) ** (-1 / 12)
        res = sim(mu_R, mu_H, delta_tau, b)
        res.label = f"r={r_annual:.0%}"
        results.append(res)
    return results


def sweep_taylor(mu_R: float, mu_H: float, delta_tau: int = 18,
                 beta: float = BETA, use_calvo: bool = True,
                 alpha: float = 1.0):
    """Sweep Taylor rule parameters (phi_pi, sigma_ad, floating_share)."""
    sim = simulate_calvo if use_calvo else simulate_compact
    configs = [
        (0.0, 0.0, 0.0, "no TR, all fixed"),
        (0.5, 0.0, 0.3, "phi=0.5, float=30%"),
        (1.0, 0.0, 0.3, "phi=1.0, float=30%"),
        (1.5, 0.0, 0.3, "phi=1.5, float=30%"),
        (1.5, 0.0, 0.5, "phi=1.5, float=50%"),
        (1.5, 0.0, 0.8, "phi=1.5, float=80%"),
        (1.5, 0.1, 0.5, "phi=1.5, AD=0.1, float=50%"),
        (1.5, 0.3, 0.5, "phi=1.5, AD=0.3, float=50%"),
        (2.0, 0.0, 0.5, "phi=2.0, float=50%"),
        (3.0, 0.0, 0.5, "phi=3.0, float=50%"),
    ]
    results = []
    for phi_pi, sigma_ad, fs, label in configs:
        r = sim(mu_R, mu_H, delta_tau, beta, phi_pi, sigma_ad, fs, alpha)
        r.label = label
        results.append(r)
    return results


# ============================================================
# Print helpers
# ============================================================

def print_compare_table(cases: list):
    """Print comparison table for a list of (name, mu_R, mu_H) tuples."""
    print(f"{'Scenario':>20s}  {'mu_R':>7s} {'mu_H':>7s}  "
          f"{'Compact':>10s} {'Calvo':>10s} {'Diff':>10s} "
          f"{'Cant(c)':>12s} {'Cant(k)':>12s} {'Verdict':>25s}")
    print("-" * 125)
    for name, mu_r, mu_h in cases:
        d = compare_formulas(mu_r, mu_h)
        in_band_c = 10 <= d['compact_ds'] <= 16
        in_band_k = 10 <= d['calvo_ds'] <= 16

        if d['compact_ds'] > 16 and d['calvo_ds'] > 16:
            verdict = "CANTILLON (both)"
        elif d['compact_ds'] > 16:
            verdict = "FLIP: compact only"
        elif d['calvo_ds'] > 16:
            verdict = "FLIP: calvo only"
        elif d['compact_ds'] < 10 and d['calvo_ds'] < 10:
            verdict = "HANK survives (both)"
        else:
            verdict = "MARGINAL"

        print(f"{name:>20s}  {mu_r:>7.3f} {mu_h:>7.3f}  "
              f"{d['compact_ds']:>10.2f} {d['calvo_ds']:>10.2f} {d['diff']:>+10.2f} "
              f"{d['compact_cant']:>12.2f} {d['calvo_cant']:>12.2f} {verdict:>25s}")


def print_sweep_table(results: list, title: str = ""):
    """Print a table of Results from a sweep."""
    if title:
        print(f"\n{title}")
    print(f"{'Label':>25s}  {'delta*':>10s}  {'Cantillon':>12s}  "
          f"{'PrinGain':>10s}  {'Term P':>8s}  {'Infl pk':>8s}")
    print("-" * 85)
    for r in results:
        ds_str = f"{r.delta_star:.2f}" if abs(r.delta_star) < 9999 else "inf"
        infl_str = f"{r.peak_inflation*100:.2f}%"
        print(f"{r.label:>25s}  {ds_str:>10s}  {r.cantillon:>12.2f}  "
              f"{r.principal_gain:>10.4f}  {r.terminal_price:>8.4f}  {infl_str:>8s}")


# ============================================================
# Main: run all analyses when executed directly
# ============================================================

if __name__ == "__main__":
    print("=" * 100)
    print("HANK AFTER CANTILLON — SIMULATION ENGINE")
    print("=" * 100)

    # --- 1. Compact vs Calvo comparison ---
    print("\n" + "=" * 100)
    print("1. COMPACT FORM vs CORRECT CALVO: delta* across asymmetry ratios")
    print(f"   Delta_tau=18, r=5%, mu_bar=0.10")
    print("=" * 100)
    cases = [
        ("Symmetric 1:1", 0.100, 0.100),
        ("Mild asym 1.3:1", 0.113, 0.087),
        ("Moderate 2:1", 0.133, 0.067),
        ("Strong 3:1", 0.150, 0.050),
        ("QE-era 5:1", 0.167, 0.033),
        ("Extreme 10:1", 0.182, 0.018),
        ("Extreme 20:1", 0.191, 0.010),
        ("Limit (infty:1)", 0.200, 0.000),
    ]
    print_compare_table(cases)

    # --- 2. Delta_tau sweep ---
    print("\n" + "=" * 100)
    print("2. DELTA_TAU SWEEP: symmetric vs 5:1 (correct Calvo)")
    print("=" * 100)
    for label, mu_r, mu_h in [("Symmetric 1:1", 0.10, 0.10),
                               ("Asymmetric 5:1", 0.167, 0.033)]:
        results = sweep_delta_tau(mu_r, mu_h, use_calvo=True)
        print_sweep_table(results, f"\n{label}:")

    # --- 3. Asymmetry ratio sweep ---
    print("\n" + "=" * 100)
    print("3. ASYMMETRY RATIO SWEEP (correct Calvo)")
    print("=" * 100)
    results = sweep_asymmetry(use_calvo=True)
    print_sweep_table(results)

    # --- 4. Discount rate sweep ---
    print("\n" + "=" * 100)
    print("4. DISCOUNT RATE SWEEP (correct Calvo)")
    print("=" * 100)
    for label, mu_r, mu_h in [("Symmetric 1:1", 0.10, 0.10),
                               ("Asymmetric 5:1", 0.167, 0.033)]:
        results = sweep_discount_rates(mu_r, mu_h, use_calvo=True)
        print_sweep_table(results, f"\n{label}:")

    # --- 5. Income inequality sweep ---
    print("\n" + "=" * 100)
    print("5. INCOME INEQUALITY SWEEP (Y_R/Y_H = alpha)")
    print("=" * 100)
    for label, mu_r, mu_h in [("Symmetric 1:1", 0.10, 0.10),
                               ("Asymmetric 5:1", 0.167, 0.033)]:
        results = sweep_income_inequality(mu_r, mu_h, use_calvo=True)
        print_sweep_table(results, f"\n{label}:")

    # --- 6. Taylor rule sweep (alpha=1) ---
    print("\n" + "=" * 100)
    print("6. TAYLOR RULE SWEEP — alpha=1 (standard HANK equal-incomes premise)")
    print("=" * 100)
    for label, mu_r, mu_h in [("Symmetric 1:1", 0.10, 0.10),
                               ("Asymmetric 5:1", 0.167, 0.033)]:
        results = sweep_taylor(mu_r, mu_h, use_calvo=True, alpha=1.0)
        print_sweep_table(results, f"\n{label}:")

    # --- 7. Taylor rule sweep (alpha=5) ---
    print("\n" + "=" * 100)
    print("7. TAYLOR RULE SWEEP — alpha=5 (QE-era baseline income inequality)")
    print("=" * 100)
    for label, mu_r, mu_h in [("Symmetric 1:1", 0.10, 0.10),
                               ("Asymmetric 5:1", 0.167, 0.033)]:
        results = sweep_taylor(mu_r, mu_h, use_calvo=True, alpha=5.0)
        print_sweep_table(results, f"\n{label}:")

    print("\n" + "=" * 100)
    print("Done.")
    print("=" * 100)
