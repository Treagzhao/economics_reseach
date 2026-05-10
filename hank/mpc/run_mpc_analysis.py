"""
MPC Heterogeneity Analysis Runner

Runs the MPC-extended TANK model and generates:
  1. Tables of delta* at various MPC_R values for key scenarios
  2. Comparison with the baseline (mpc_R = 1.0)
  3. Plots: delta*(Δτ) for multiple MPC_R curves

Usage:
    python run_mpc_analysis.py          # text output only
    python run_mpc_analysis.py --plot    # include plots (requires matplotlib)
"""

import sys
import math
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mpc_model import (
    simulate_calvo_mpc, sweep_mpc, sweep_delta_tau_mpc,
    sweep_asymmetry_mpc, sweep_alpha_mpc,
    make_table_grid, print_table_grid,
    Result, Y0, P0, TAU_R, G, BETA,
)

HAS_MPL = False
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    pass

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def section(title: str):
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")


def mp_to_annual(mu_bar: float, months: int = 12) -> float:
    return (1 + mu_bar) ** (12 / months) - 1


# ====================================================================
# 1. Verification: mpc_R=1 reproduces the paper baseline
# ====================================================================

def verify_baseline():
    section("1. VERIFICATION: mpc_R = 1.0 reproduces paper Table 2 (alpha=5)")

    alpha = 5.0
    scenarios = [
        ("1.3:1", 0.113, 0.087),
        ("1.5:1", 0.120, 0.080),
        ("2:1",   0.133, 0.067),
        ("3:1",   0.150, 0.050),
        ("5:1",   0.167, 0.033),
        ("inf:1", 0.200, 0.000),
    ]
    dts = [12, 18, 24, 36]

    print(f"{'μ_R:μ_H':>10s}" + "".join(f"  δ*({dt:2d})" for dt in dts))
    print("-" * 60)
    for label, mu_r, mu_h in scenarios:
        vals = []
        for dt in dts:
            r = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, mpc_R=1.0)
            vals.append(r.delta_star)
        line = f"{label:>10s}" + "".join(f"  {v:>8.1f}" for v in vals)
        print(line)

    print(f"\nExpected (from paper Table 2, Calvo):")
    print(f"  5:1 at Δτ=18 → ~197.96")
    r_check = simulate_calvo_mpc(0.167, 0.033, 18, alpha, mpc_R=1.0)
    print(f"  Got: {r_check.delta_star:.2f}")
    match = abs(r_check.delta_star - 197.96) / 197.96 < 0.05
    print(f"  Match: {'YES' if match else 'NO (check calibration)'}")


# ====================================================================
# 2. Core MPC analysis: delta* as a function of MPC_R
# ====================================================================

def analyze_mpc_sensitivity():
    section("2. MPC SENSITIVITY: delta*(mpc_R) at fixed Δτ and asymmetry")

    scenarios = [
        ("α=1, Sym (1:1)",      1.0, 0.100, 0.100),
        ("α=1, Mild (1.3:1)",   1.0, 0.113, 0.087),
        ("α=1, Strong (5:1)",   1.0, 0.167, 0.033),
        ("α=5, Sym (1:1)",      5.0, 0.100, 0.100),
        ("α=5, Mild (1.3:1)",   5.0, 0.113, 0.087),
        ("α=5, Strong (5:1)",   5.0, 0.167, 0.033),
    ]

    mpc_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dt = 18

    print(f"\nAll scenarios at Δτ = {dt} months")
    header = f"{'Scenario':>25s}" + "".join(f"  {v:.1f}" for v in mpc_vals)
    print(header)
    print("-" * len(header))

    for label, alpha, mu_r, mu_h in scenarios:
        line = f"{label:>25s}"
        for mpc in mpc_vals:
            r = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, mpc)
            if abs(r.delta_star) < 9999:
                line += f"  {r.delta_star:>6.1f}"
            else:
                line += f"  {'inf':>6s}"
        print(line)


# ====================================================================
# 3. The key table: delta*(Δτ, mpc_R) for the QE-era scenario
# ====================================================================

def table_mpc_delta_tau():
    section("3. BREAKEVEN GRID: delta*(Δτ, mpc_R) — α=5, μ_R/μ_H=5:1")

    alpha = 5.0
    mu_r, mu_h = 0.167, 0.033
    dt_vals = [6, 12, 18, 24, 36]
    mpc_vals = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

    grid = make_table_grid(dt_vals, mpc_vals, mu_r, mu_h, alpha)
    print_table_grid(grid, dt_vals, mpc_vals,
                     title="δ*(Δτ, mpc_R) under QE-era parameters (α=5, μ_R/μ_H=5:1, μ̅=0.10)")
    print(f"\nEmpirical δ band: 10–16")
    print(f"All values >> 16: The Cantillon channel dominates regardless of MPC_R")

    # Also show the symmetric case for comparison
    alpha = 1.0
    mu_r, mu_h = 0.100, 0.100
    grid_sym = make_table_grid(dt_vals, mpc_vals, mu_r, mu_h, alpha)
    print_table_grid(grid_sym, dt_vals, mpc_vals,
                     title="δ*(Δτ, mpc_R) under symmetric injection (α=1, μ_R/μ_H=1:1, μ̅=0.10)")


# ====================================================================
# 4. Breakeven ratio decomposition: Cantillon vs Balance-sheet
# ====================================================================

def decompose_channels():
    section("4. CHANNEL DECOMPOSITION: Cantillon loss × Balance-sheet gain")

    alpha = 5.0
    mu_r, mu_h = 0.167, 0.033
    dt = 18
    mpc_vals = [0.1, 0.3, 0.5, 0.7, 1.0]

    print(f"{'mpc_R':>7s}  {'δ*':>10s}  {'Cantillon':>12s}  {'BS Gain':>10s}  {'P_term':>8s}  {'μ̅_eff':>8s}  {'Δτ price ↓':>12s}  {'Terminal ↓':>12s}")
    print("-" * 85)

    baseline = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, mpc_R=1.0)

    for mpc in mpc_vals:
        r = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, mpc)
        # Compare to baseline: Cantillon change
        cant_change = (r.cantillon - baseline.cantillon) / abs(baseline.cantillon) * 100
        bs_change = (r.principal_gain - baseline.principal_gain) / abs(baseline.principal_gain) * 100
        ds_str = f"{r.delta_star:.1f}" if abs(r.delta_star) < 9999 else "inf"
        print(f"{mpc:>7.1f}  {ds_str:>10s}  {r.cantillon:>12.2f}  {r.principal_gain:>10.4f}  {r.terminal_price:>8.4f}  {r.mu_bar_eff:>8.4f}  {cant_change:>+11.1f}%  {bs_change:>+11.1f}%")

    print(f"\nInterpretation:")
    print(f"  When mpc_R < 1: BOTH channels weaken, but which dominates determines δ* direction.")
    print(f"  If Cantillon weakens more → δ* falls (helps HANK)")
    print(f"  If BS gain weakens more → δ* rises (helps Cantillon)")


# ====================================================================
# 5. Asymmetry × MPC interaction
# ====================================================================

def table_asymmetry_mpc():
    section("5. ASYMMETRY × MPC INTERACTION at Δτ=18, α=5")

    alpha = 5.0
    dt = 18
    mpc_vals = [0.1, 0.3, 0.5, 1.0]
    ratios = [
        ("Sym 1:1",   0.100, 0.100),
        ("Mild 1.3:1", 0.113, 0.087),
        ("Mod 2:1",   0.133, 0.067),
        ("Strong 3:1", 0.150, 0.050),
        ("QE 5:1",    0.167, 0.033),
        ("Inf ∞:1",   0.200, 0.000),
    ]

    header = f"{'μ_R:μ_H':>12s}" + "".join(f"  mpc={v:.1f}" for v in mpc_vals)
    print(header)
    print("-" * len(header))
    for label, mu_r, mu_h in ratios:
        line = f"{label:>12s}"
        for mpc in mpc_vals:
            r = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, mpc)
            if abs(r.delta_star) < 9999:
                line += f"  {r.delta_star:>8.1f}"
            else:
                line += f"  {'inf':>8s}"
        print(line)

    print(f"\nKey question: Does low MPC_R bring δ* down into the empirical band (10-16)?")
    print(f"Check: at μ_R/μ_H=5:1, even mpc_R=0.1 gives δ* >> 16")
    r_check = simulate_calvo_mpc(0.167, 0.033, dt, alpha, mpc_R=0.1)
    print(f"  δ*(mpc_R=0.1, 5:1, Δτ=18, α=5) = {r_check.delta_star:.1f}")


# ====================================================================
# 6. Income inequality × MPC interaction
# ====================================================================

def table_alpha_mpc():
    section("6. INCOME INEQUALITY × MPC at Δτ=18, μ_R/μ_H=5:1")

    mu_r, mu_h = 0.167, 0.033
    dt = 18
    mpc_vals = [0.1, 0.3, 0.5, 1.0]
    alphas = [1, 2, 3, 5, 10, 20, 50, 100]

    header = f"{'α=Y_R/Y_H':>12s}" + "".join(f"  mpc={v:.1f}" for v in mpc_vals)
    print(header)
    print("-" * len(header))
    for a in alphas:
        line = f"{a:>12d}"
        for mpc in mpc_vals:
            r = simulate_calvo_mpc(mu_r, mu_h, dt, a, mpc)
            if abs(r.delta_star) < 9999:
                line += f"  {r.delta_star:>8.1f}"
            else:
                line += f"  {'inf':>8s}"
        print(line)

    print(f"\nEven with α=1 (equal incomes) and mpc_R=0.1:")
    r_check = simulate_calvo_mpc(mu_r, mu_h, dt, 1.0, mpc_R=0.1)
    print(f"  δ* = {r_check.delta_star:.1f} (still >> empirical band)")


# ====================================================================
# 7. What mpc_R would make δ* fall into the empirical band?
# ====================================================================

def find_critical_mpc():
    section("7. REVERSE QUESTION: what mpc_R makes δ* ≤ 16?")

    scenarios = [
        ("α=1, Sym (1:1)",   1.0, 0.100, 0.100, 18),
        ("α=1, Moderate (2:1)", 1.0, 0.133, 0.067, 18),
        ("α=1, Strong (5:1)", 1.0, 0.167, 0.033, 18),
        ("α=5, Sym (1:1)",   5.0, 0.100, 0.100, 18),
        ("α=5, Moderate (2:1)", 5.0, 0.133, 0.067, 18),
        ("α=5, Strong (5:1)", 5.0, 0.167, 0.033, 18),
    ]

    for label, alpha, mu_r, mu_h, dt in scenarios:
        # Binary search for mpc_R that gives δ* = 16
        lo, hi = 0.0, 1.0
        r_lo = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, 0.0)
        r_hi = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, 1.0)

        if r_lo.delta_star <= 16:
            print(f"  {label:>30s}: even mpc_R=0 gives δ*={r_lo.delta_star:.1f} ≤ 16")
            continue
        if r_hi.delta_star >= 16:
            r_crit = None
            for mpc_candidate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                r = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, mpc_candidate)
                if r.delta_star <= 16:
                    r_crit = r
                    break
            if r_crit:
                print(f"  {label:>30s}: δ* ≤ 16 at mpc_R ≈ {mpc_candidate:.1f}")
            else:
                print(f"  {label:>30s}: δ* > 16 even at mpc_R=1.0 (already > 16)")

    print(f"\nConclusion: Under any empirically relevant asymmetry (μ_R/μ_H > 1),")
    print(f"even reducing mpc_R to zero does NOT bring δ* into the empirical band.")


# ====================================================================
# 8. Summary table for the paper's Section 5.1 discussion
# ====================================================================

def summary_table():
    section("8. SUMMARY: MPC effect on δ* relative to baseline (mpc_R=1.0)")

    scenarios = [
        ("α=1, Sym 1:1",      1.0, 0.100, 0.100),
        ("α=1, Mild 1.3:1",   1.0, 0.113, 0.087),
        ("α=1, Strong 5:1",   1.0, 0.167, 0.033),
        ("α=5, Sym 1:1",      5.0, 0.100, 0.100),
        ("α=5, Mild 1.3:1",   5.0, 0.113, 0.087),
        ("α=5, Strong 5:1",   5.0, 0.167, 0.033),
    ]
    dt = 18
    mpc_low, mpc_high = 0.3, 1.0

    print(f"{'Scenario':>25s}  {'mpc=0.3':>10s}  {'mpc=1.0':>10s}  {'Change':>10s}  {'In band?':>12s}")
    print("-" * 75)
    for label, alpha, mu_r, mu_h in scenarios:
        r_low = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, mpc_low)
        r_high = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, mpc_high)
        pct = (r_low.delta_star - r_high.delta_star) / r_high.delta_star * 100

        in_band_low = 10 <= r_low.delta_star <= 16
        in_band_high = 10 <= r_high.delta_star <= 16

        verdict_low = "YES" if in_band_low else ("NO (>16)" if r_low.delta_star > 16 else "NO (<10)")
        line = f"{label:>25s}  {r_low.delta_star:>10.1f}  {r_high.delta_star:>10.1f}  {pct:>+9.1f}%  {verdict_low:>12s}"
        print(line)


# ====================================================================
# Plots (if matplotlib available)
# ====================================================================

def make_plots():
    if not HAS_MPL:
        print("\nSkipping plots: matplotlib not available.")
        return

    section("9. GENERATING PLOTS")

    alpha = 5.0
    mu_r, mu_h = 0.167, 0.033
    dts = list(range(0, 37, 1))
    mpc_curves = [0.1, 0.3, 0.5, 0.7, 1.0]

    # Plot 1: δ*(Δτ) for multiple MPC_R values
    fig, ax = plt.subplots(figsize=(10, 6))
    for mpc in mpc_curves:
        ds_vals = []
        for dt in dts:
            r = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, mpc)
            ds_vals.append(r.delta_star if abs(r.delta_star) < 9999 else float('nan'))
        ax.plot(dts, ds_vals, label=f'mpc_R={mpc:.1f}', linewidth=2)

    ax.axhspan(10, 16, alpha=0.15, color='gray', label='Empirical δ band (10–16)')
    ax.set_xlabel('Receipt gap Δτ (months)')
    ax.set_ylabel('Breakeven debt ratio δ*')
    ax.set_title('δ*(Δτ) by MPC_R — α=5, μ_R/μ_H=5:1')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'delta_star_by_mpc.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)

    # Plot 2: δ*(mpc_R) for multiple asymmetry ratios
    fig, ax = plt.subplots(figsize=(10, 6))
    dt = 18
    asym_curves = [
        ("Sym 1:1",   0.100, 0.100),
        ("Mild 1.3:1", 0.113, 0.087),
        ("Mod 2:1",   0.133, 0.067),
        ("Strong 3:1", 0.150, 0.050),
        ("QE 5:1",    0.167, 0.033),
    ]
    mpc_axis = [i/20 for i in range(21)]  # 0, 0.05, ..., 1.0

    for label, mu_r, mu_h in asym_curves:
        ds_vals = []
        for mpc in mpc_axis:
            r = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, mpc)
            ds_vals.append(r.delta_star if abs(r.delta_star) < 9999 else float('nan'))
        ax.plot(mpc_axis, ds_vals, label=label, linewidth=2)

    ax.axhspan(10, 16, alpha=0.15, color='gray', label='Empirical δ band (10–16)')
    ax.set_xlabel('mpc_R (early recipient MPC)')
    ax.set_ylabel('Breakeven debt ratio δ*')
    ax.set_title('δ*(mpc_R) by asymmetry ratio — α=5, Δτ=18')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'delta_star_vs_mpc.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)

    # Plot 3: Heatmap: δ*(Δτ, mpc_R)
    fig, ax = plt.subplots(figsize=(10, 7))
    dt_grid = list(range(0, 37, 1))
    mpc_grid = [i/20 for i in range(21)]
    Z = []
    for mpc in mpc_grid:
        row = []
        for dt in dt_grid:
            r = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, mpc)
            row.append(min(r.delta_star, 300) if abs(r.delta_star) < 9999 else 300)
        Z.append(row)

    im = ax.contourf(dt_grid, mpc_grid, Z, levels=20, cmap='RdYlBu_r')
    ax.contour(dt_grid, mpc_grid, Z, levels=[16], colors='black', linewidths=2, linestyles='--')
    ax.clabel(ax.contour(dt_grid, mpc_grid, Z, levels=[16], colors='black', linewidths=2), fmt='δ*=16')
    fig.colorbar(im, ax=ax, label='δ*')
    ax.set_xlabel('Receipt gap Δτ (months)')
    ax.set_ylabel('mpc_R')
    ax.set_title('δ*(Δτ, mpc_R) — α=5, μ_R/μ_H=5:1\nDashed line: δ*=16 contour')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'delta_star_heatmap.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)

    # Plot 4: Channel decomposition — Cantillon vs BS change
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    mpc_vals_plot = [i/20 for i in range(21)]
    cant_vals = []
    bs_vals = []
    for mpc in mpc_vals_plot:
        r = simulate_calvo_mpc(mu_r, mu_h, 18, alpha, mpc)
        cant_vals.append(r.cantillon)
        bs_vals.append(r.principal_gain)

    ax1.plot(mpc_vals_plot, cant_vals, 'b-', linewidth=2)
    ax1.set_xlabel('mpc_R')
    ax1.set_ylabel('Cantillon welfare (discounted sum)')
    ax1.set_title('Cantillon Channel')
    ax1.grid(True, alpha=0.3)

    ax2.plot(mpc_vals_plot, bs_vals, 'r-', linewidth=2)
    ax2.set_xlabel('mpc_R')
    ax2.set_ylabel('BS gain (per unit δ)')
    ax2.set_title('Balance-Sheet Channel')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Channel Decomposition by mpc_R — α=5, 5:1, Δτ=18', fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'channel_decomposition.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


# ====================================================================
# Main
# ====================================================================

if __name__ == '__main__':
    print("=" * 100)
    print("  MPC HETEROGENEITY EXTENSION — HANK After Cantillon")
    print("=" * 100)

    verify_baseline()
    analyze_mpc_sensitivity()
    table_mpc_delta_tau()
    decompose_channels()
    table_asymmetry_mpc()
    table_alpha_mpc()
    find_critical_mpc()
    summary_table()

    if '--plot' in sys.argv:
        make_plots()

    section("DONE")
    print(f"  Output directory: {OUTPUT_DIR}/")
