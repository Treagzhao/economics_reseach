"""
MPC对照表：不同mpc_R组合下的δ*
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mpc_model import simulate_calvo_mpc

# 参数设定
DT = 18                              # 默认接收时滞
MPC_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 场景设定
SCENARIOS = [
    # (标签,             α,   μ_R,   μ_H,   Δτ)
    ("α=1, Sym (1:1)",       1.0, 0.100, 0.100, 18),
    ("α=1, Mild (1.3:1)",    1.0, 0.113, 0.087, 18),
    ("α=1, Mod (2:1)",       1.0, 0.133, 0.067, 18),
    ("α=1, Strong (3:1)",    1.0, 0.150, 0.050, 18),
    ("α=1, QE (5:1)",        1.0, 0.167, 0.033, 18),
    ("α=1, Inf (∞:1)",       1.0, 0.200, 0.000, 18),

    ("α=5, Sym (1:1)",       5.0, 0.100, 0.100, 18),
    ("α=5, Mild (1.3:1)",    5.0, 0.113, 0.087, 18),
    ("α=5, Mod (2:1)",       5.0, 0.133, 0.067, 18),
    ("α=5, Strong (3:1)",    5.0, 0.150, 0.050, 18),
    ("α=5, QE (5:1)",        5.0, 0.167, 0.033, 18),
    ("α=5, Inf (∞:1)",       5.0, 0.200, 0.000, 18),
]

ALSO_SCENARIOS = [
    ("α=5, QE (5:1), Δτ=6",  5.0, 0.167, 0.033, 6),
    ("α=5, QE (5:1), Δτ=12", 5.0, 0.167, 0.033, 12),
    ("α=5, QE (5:1), Δτ=24", 5.0, 0.167, 0.033, 24),
    ("α=5, QE (5:1), Δτ=36", 5.0, 0.167, 0.033, 36),
]

print("=" * 135)
print("  MPC 对照表：δ*(mpc_R)  —  不同不对称度和收入不平等下的盈亏平衡债务比")
print("=" * 135)
print()
print(f"固定参数: Δτ = {DT} 个月, Calvo g = 0.091")
print(f"mpc_R 范围: {MPC_VALS[0]:.1f} ~ {MPC_VALS[-1]:.1f}")
print()

def print_block(title, scenarios):
    print(f"▌ {title}")
    print()

    # 表头
    header = f"{'场景':>26s}"
    for m in MPC_VALS:
        header += f"  {m:.1f}"
    print(header)
    print("-" * len(header))

    for label, alpha, mu_r, mu_h, dt in scenarios:
        row = f"{label:>26s}"
        for mpc in MPC_VALS:
            r = simulate_calvo_mpc(mu_r, mu_h, dt, alpha, mpc)
            if abs(r.delta_star) >= 9999:
                row += f"  {'inf':>6s}"
            elif r.delta_star < -999:
                row += f"  {'-∞':>6s}"
            else:
                row += f"  {r.delta_star:>6.1f}"
        print(row)
    print()

print_block("A. 不同非对称度 (α=1, 等收入份额)", SCENARIOS[:6])
print_block("B. 不同非对称度 (α=5, 高收入不平等)", SCENARIOS[6:])
print_block("C. 不同接收时滞 (α=5, μ_R/μ_H=5:1)", ALSO_SCENARIOS)

# ============================================================
# 补充：反向表格 — δ*能否进入实证区间[10,16]？
# ============================================================
print("=" * 135)
print("  补充：δ*能否进入实证区间[10, 16]？需要多大的mpc_R？")
print("=" * 135)
print()

scenarios_critical = [
    ("α=1, Sym (1:1)",   1.0, 0.100, 0.100),
    ("α=1, Mild (1.3:1)", 1.0, 0.113, 0.087),
    ("α=1, Mod (2:1)",   1.0, 0.133, 0.067),
    ("α=1, Strong (3:1)", 1.0, 0.150, 0.050),
    ("α=1, QE (5:1)",    1.0, 0.167, 0.033),
    ("α=5, Sym (1:1)",   5.0, 0.100, 0.100),
    ("α=5, Mild (1.3:1)", 5.0, 0.113, 0.087),
    ("α=5, Mod (2:1)",   5.0, 0.133, 0.067),
    ("α=5, Strong (3:1)", 5.0, 0.150, 0.050),
    ("α=5, QE (5:1)",    5.0, 0.167, 0.033),
]

header = f"{'场景':>26s}  {'δ*@mpc=0':>10s}  {'δ*@mpc=1':>10s}  {'δ*∈[10,16]?':>14s}  {'mpc_R区间':>20s}"
print(header)
print("-" * len(header))

for label, alpha, mu_r, mu_h in scenarios_critical:
    r0 = simulate_calvo_mpc(mu_r, mu_h, DT, alpha, 0.0)
    r1 = simulate_calvo_mpc(mu_r, mu_h, DT, alpha, 1.0)

    # 找δ*首次≥10的mpc_R
    mpc_low = None
    for mpc in [x / 100 for x in range(0, 101)]:
        r = simulate_calvo_mpc(mu_r, mu_h, DT, alpha, mpc)
        if r.delta_star >= 10:
            mpc_low = mpc
            break

    # 找δ*首次≥16的mpc_R
    mpc_high = None
    for mpc in [x / 100 for x in range(0, 101)]:
        r = simulate_calvo_mpc(mu_r, mu_h, DT, alpha, mpc)
        if r.delta_star >= 16:
            mpc_high = mpc
            break

    ds0_str = f"{r0.delta_star:.1f}" if abs(r0.delta_star) < 9999 else ("-∞" if r0.delta_star < 0 else "∞")
    ds1_str = f"{r1.delta_star:.1f}" if abs(r1.delta_star) < 9999 else ("-∞" if r1.delta_star < 0 else "∞")

    if mpc_low is not None and mpc_high is not None and mpc_low <= mpc_high:
        verdict = f"YES [{mpc_low:.2f}, {mpc_high:.2f}]"
        interval = f"[{mpc_low:.2f}, {mpc_high:.2f}]"
    elif r1.delta_star < 10:
        verdict = "NO (δ*<10 even at mpc=1)"
        interval = "—"
    elif r0.delta_star > 16:
        verdict = "NO (δ*>16 even at mpc=0)"
        interval = "—"
    else:
        # δ*跳跃穿过区间，但步长0.01没捕捉到
        verdict = "YES (crosses band)"
        interval = "~0.01宽"

    row = f"{label:>26s}  {ds0_str:>10s}  {ds1_str:>10s}  {verdict:>14s}  {interval:>20s}"
    print(row)

print()
print("-" * 135)
print("YES=存在mpc_R使δ*落在[10,16]内；NO=即使mpc_R从0变到1也无法进入该区间")
print()
