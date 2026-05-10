"""
MPC-extended TANK model: incorporates heterogeneous marginal propensities to consume.

Extends the baseline model from paper.md (Section 5.1) by allowing the
early recipient (R) to have MPC < 1, reflecting that wealthy households
receiving QE-driven capital gains spend only a fraction of the injection
in the short run.

Key equations (modified from the baseline):
  Regime II notional price:
    P_II = P0 * (1 + alpha * mpc_R * mu_R / (1 + alpha))
  Regime III notional price:
    P_III = P0 * (1 + (alpha * mpc_R * mu_R + mu_H) / (1 + alpha))
  Terminal effective injection:
    mu_bar_eff = (alpha * mpc_R * mu_R + mu_H) / (1 + alpha)

Both the Cantillon channel (transmission-interval price path) and the
balance-sheet channel (terminal inflation) are attenuated when mpc_R < 1.
The net effect on delta* is ambiguous and depends on parameter values.
"""

import math
from dataclasses import dataclass
from typing import Optional

Y0 = 100.0
P0 = 1.0
TAU_R = 1
G = 0.091          # monthly Calvo: theta_q=0.75 -> g=0.091
BETA = (1.05) ** (-1 / 12)  # 5% annual discount rate


@dataclass
class Result:
    delta_star: float
    cantillon: float
    principal_gain: float
    terminal_price: float
    peak_inflation: float
    mu_bar_eff: float
    label: str = ""


def simulate_calvo_mpc(
    mu_R: float, mu_H: float, delta_tau: int,
    alpha: float = 1.0, mpc_R: float = 1.0,
    beta: float = BETA,
    t_max_override: Optional[int] = None,
) -> Result:
    """
    Simulate with MPC heterogeneity.

    Parameters
    ----------
    mpc_R : float in [0, 1]
        Fraction of the income injection that R spends immediately.
        Baseline model uses mpc_R = 1.0.
    """
    tau_h = TAU_R + delta_tau
    t_terminal = tau_h + 1
    t_max = t_max_override or max(tau_h + 500, 1500)

    mu_bar_eff = (alpha * mpc_R * mu_R + mu_H) / (1 + alpha)

    p_star_ii = P0 * (1 + alpha * mpc_R * mu_R / (1 + alpha))
    p_star_iii = P0 * (1 + mu_bar_eff)

    P = [P0] * (t_max + 1)
    peak_infl = 0.0

    for t in range(t_max + 1):
        if t <= TAU_R:
            P[t] = P0
        else:
            infl = (P[t - 1] / P[t - 2] - 1) if t >= TAU_R + 2 else 0.0
            peak_infl = max(peak_infl, infl)

            p_star_base = p_star_ii if t < tau_h else p_star_iii
            P[t] = (1 - G) * P[t - 1] + G * p_star_base

    cant = 0.0
    for t in range(t_max + 1):
        y_h = Y0 if t < tau_h else Y0 * (1 + mu_H)
        cant += (beta ** t) * (y_h / P[t] - Y0 / P0)

    p_terminal = P[-1]
    pg = (Y0 / P0) * (1 - P0 / p_terminal)
    bs_gain = (beta ** t_terminal) * pg

    ds = -cant / bs_gain if abs(bs_gain) > 1e-10 else float('inf')
    return Result(
        delta_star=ds, cantillon=cant, principal_gain=bs_gain,
        terminal_price=p_terminal, peak_inflation=peak_infl,
        mu_bar_eff=mu_bar_eff,
    )


def sweep_mpc(
    mu_R: float, mu_H: float, delta_tau: int,
    alpha: float = 1.0,
    mpc_values: Optional[list] = None,
    beta: float = BETA,
) -> list:
    """Sweep over MPC_R values at fixed asymmetry and receipt gap."""
    if mpc_values is None:
        mpc_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []
    for mpc in mpc_values:
        r = simulate_calvo_mpc(mu_R, mu_H, delta_tau, alpha, mpc, beta)
        r.label = f"mpc_R={mpc:.1f}"
        results.append(r)
    return results


def sweep_delta_tau_mpc(
    mu_R: float, mu_H: float,
    mpc_values: Optional[list] = None,
    alpha: float = 1.0,
    dts: Optional[list] = None,
    beta: float = BETA,
) -> dict:
    """Sweep over receipt gaps for multiple MPC_R values.

    Returns dict mapping mpc_R -> list of Results (one per delta_tau).
    """
    if mpc_values is None:
        mpc_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    if dts is None:
        dts = [6, 12, 18, 24, 36]
    results = {}
    for mpc in mpc_values:
        mpc_results = []
        for dt in dts:
            r = simulate_calvo_mpc(mu_R, mu_H, dt, alpha, mpc, beta)
            r.label = f"Δτ={dt}, mpc={mpc:.1f}"
            mpc_results.append(r)
        results[mpc] = mpc_results
    return results


def sweep_asymmetry_mpc(
    delta_tau: int = 18, alpha: float = 1.0,
    mpc_values: Optional[list] = None,
    beta: float = BETA,
) -> dict:
    """Sweep over asymmetry ratios for multiple MPC_R values.

    Returns dict mapping mpc_R -> list of (ratio_label, Result).
    """
    if mpc_values is None:
        mpc_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    ratios = [
        ("1.3:1", 0.113, 0.087),
        ("1.5:1", 0.120, 0.080),
        ("2:1",   0.133, 0.067),
        ("3:1",   0.150, 0.050),
        ("5:1",   0.167, 0.033),
        ("inf:1", 0.200, 0.000),
    ]
    results = {}
    for mpc in mpc_values:
        mpc_results = []
        for label, mu_r, mu_h in ratios:
            r = simulate_calvo_mpc(mu_r, mu_h, delta_tau, alpha, mpc, beta)
            r.label = label
            mpc_results.append(r)
        results[mpc] = mpc_results
    return results


def sweep_alpha_mpc(
    mu_R: float, mu_H: float, delta_tau: int = 18,
    mpc_values: Optional[list] = None,
    alphas: Optional[list] = None,
    beta: float = BETA,
) -> dict:
    """Sweep over income inequality alpha for multiple MPC_R values."""
    if mpc_values is None:
        mpc_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    if alphas is None:
        alphas = [1, 2, 3, 5, 10, 20, 50, 100]
    results = {}
    for mpc in mpc_values:
        alpha_results = []
        for a in alphas:
            r = simulate_calvo_mpc(mu_R, mu_H, delta_tau, a, mpc, beta)
            r.label = f"α={a}"
            alpha_results.append(r)
        results[mpc] = alpha_results
    return results


def make_table_grid(
    dt_values: list,
    mpc_values: list,
    mu_R: float, mu_H: float,
    alpha: float = 1.0,
    beta: float = BETA,
) -> list:
    """Generate a 2D grid of delta* values: rows = delta_tau, cols = mpc_R."""
    rows = []
    for dt in dt_values:
        row = [dt]
        for mpc in mpc_values:
            r = simulate_calvo_mpc(mu_R, mu_H, dt, alpha, mpc, beta)
            row.append(r.delta_star)
        rows.append(row)
    return rows


def print_table_grid(grid: list, dt_values: list, mpc_values: list, title: str = ""):
    """Pretty-print the 2D grid."""
    if title:
        print(f"\n{title}")
    header = f"{'Δτ':>5s}" + "".join(f"  mpc={v:.1f}" for v in mpc_values)
    print(header)
    print("-" * len(header))
    for row in grid:
        line = f"{row[0]:>5d}"
        for val in row[1:]:
            if abs(val) < 9999:
                line += f"  {val:>8.1f}"
            else:
                line += f"  {'inf':>8s}"
        print(line)
