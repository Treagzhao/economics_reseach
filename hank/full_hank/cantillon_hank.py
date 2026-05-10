"""
Cantillon effect in full two-asset HANK: staggered and asymmetric receipt.

Two-sector extension: early recipients (high-e) work in sector A, late recipients
(low-e) work in sector B. Labor types are CES-aggregated with elasticity sigma.
Low sigma = limited substitutability = wage gains in A do NOT automatically
spill over to B = stronger Cantillon effect.

Key parameters:
- sigma: elasticity of substitution between labor types (1=CD, inf=perfect sub)
- kappaw_A: wage Phillips curve slope for early-recipient sector
- kappaw_B: wage Phillips curve slope for late-recipient sector

Output: breakeven debt ratio delta* for a grid of (DeltaTau, asymmetry) values.
"""

import numpy as np
from sequence_jacobian import simple, solved, combine, create_model, grids, hetblocks

hh = hetblocks.hh_twoasset.hh

# ====================================================================
# Standard two-asset HANK blocks
# ====================================================================

@simple
def pricing(pi, mc, r, Y, kappap, mup):
    nkpc = kappap * (mc - 1 / mup) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) \
           / (1 + r(+1)) - (1 + pi).apply(np.log)
    return nkpc

@simple
def arbitrage(div, p, r):
    equity = div(+1) + p(+1) - p * (1 + r(+1))
    return equity

@simple
def labor(Y, w, K, Z, alpha):
    N = (Y / Z / K(-1) ** alpha) ** (1 / (1 - alpha))
    mc = w * N / (1 - alpha) / Y
    return N, mc

@simple
def investment(Q, K, r, N, mc, Z, delta, epsI, alpha):
    inv = (K / K(-1) - 1) / (delta * epsI) + 1 - Q
    val = alpha * Z(+1) * (N(+1) / K) ** (1 - alpha) * mc(+1) - \
        (K(+1) / K - (1 - delta) + (K(+1) / K - 1) ** 2 / (2 * delta * epsI)) + \
        K(+1) / K * Q(+1) - (1 + r(+1)) * Q
    return inv, val

@simple
def dividend(Y, w, N, K, pi, mup, kappap, delta, epsI):
    psip = mup / (mup - 1) / 2 / kappap * (1 + pi).apply(np.log) ** 2 * Y
    k_adjust = K(-1) * (K / K(-1) - 1) ** 2 / (2 * delta * epsI)
    I = K - (1 - delta) * K(-1) + k_adjust
    div = Y - w * N - I - psip
    return psip, I, div

@simple
def taylor(rstar, pi, phi):
    i = rstar + phi * pi
    return i

@simple
def fiscal(r, w, N, G, Bg):
    tax = (r * Bg + G) / w / N
    return tax

@simple
def finance(i, p, pi, r, div, omega, pshare):
    rb = r - omega
    ra = pshare(-1) * (div + p) / p(-1) + (1 - pshare(-1)) * (1 + r) - 1
    fisher = 1 + i(-1) - (1 + r) * (1 + pi)
    return rb, ra, fisher

@simple
def wage(pi, w):
    piw = (1 + pi) * w / w(-1) - 1
    return piw

@simple
def union(piw, N, tax, w, UCE, kappaw, muw, vphi, frisch, beta):
    wnkpc = kappaw * (vphi * N ** (1 + 1 / frisch) - (1 - tax) * w * N * UCE / muw) + beta * \
            (1 + piw(+1)).apply(np.log) - (1 + piw).apply(np.log)
    return wnkpc

@simple
def mkt_clearing(p, A, B, Bg, C, I, G, CHI, psip, omega, Y):
    wealth = A + B
    asset_mkt = p + Bg - wealth
    goods_mkt = C + I + G + CHI + psip + omega * B - Y
    return asset_mkt, wealth, goods_mkt

@simple
def share_value(p, tot_wealth, Bh):
    pshare = p / (tot_wealth - Bh)
    return pshare

@solved(unknowns={'pi': (-0.1, 0.1)}, targets=['nkpc'], solver="brentq")
def pricing_solved(pi, mc, r, Y, kappap, mup):
    nkpc = kappap * (mc - 1 / mup) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / \
           (1 + r(+1)) - (1 + pi).apply(np.log)
    return nkpc

@solved(unknowns={'p': (5, 15)}, targets=['equity'], solver="brentq")
def arbitrage_solved(div, p, r):
    equity = div(+1) + p(+1) - p * (1 + r(+1))
    return equity

@simple
def partial_ss(Y, N, K, r, tot_wealth, Bg, delta):
    p = tot_wealth - Bg
    mc = 1 - r * (p - K) / Y
    mup = 1 / mc
    alpha = (r + delta) * K / Y / mc
    Z = Y * K ** (-alpha) * N ** (alpha - 1)
    w = mc * (1 - alpha) * Y / N
    return p, mc, mup, alpha, Z, w

@simple
def union_ss(tax, w, UCE, N, muw, frisch):
    vphi = (1 - tax) * w * UCE / muw / N ** (1 + 1 / frisch)
    wnkpc = vphi * N ** (1 + 1 / frisch) - (1 - tax) * w * UCE / muw
    return vphi, wnkpc

# ====================================================================
# Two-sector extension: separate labor markets for early/late recipient groups
# ====================================================================

@simple
def labor_two_sector(Y, w_A, w_B, K, Z, alpha, sigma):
    """CES aggregation of two labor types. sigma = elasticity of substitution."""
    N = (Y / Z / K(-1) ** alpha) ** (1 / (1 - alpha))

    if abs(sigma - 1.0) < 1e-6:
        N_A = 0.5 * N
        N_B = 0.5 * N
    else:
        wage_index = (w_A**(1 - sigma) + w_B**(1 - sigma))**(1 / (1 - sigma))
        N_A = N * (w_A / wage_index)**(-sigma)
        N_B = N * (w_B / wage_index)**(-sigma)

    mc = (w_A * N_A + w_B * N_B) / (1 - alpha) / Y
    return N_A, N_B, N, mc


@simple
def wage_A(pi, w_A):
    piw_A = (1 + pi) * w_A / w_A(-1) - 1
    return piw_A

@simple
def wage_B(pi, w_B):
    piw_B = (1 + pi) * w_B / w_B(-1) - 1
    return piw_B


@simple
def union_A(piw_A, N_A, tax, w_A, UCE, kappaw_A, muw, vphi_A, frisch, beta):
    wnkpc_A = kappaw_A * (vphi_A * N_A ** (1 + 1 / frisch) - (1 - tax) * w_A * N_A * UCE / muw) \
              + beta * (1 + piw_A(+1)).apply(np.log) - (1 + piw_A).apply(np.log)
    return wnkpc_A


@simple
def union_B(piw_B, N_B, tax, w_B, UCE, kappaw_B, muw, vphi_B, frisch, beta):
    wnkpc_B = kappaw_B * (vphi_B * N_B ** (1 + 1 / frisch) - (1 - tax) * w_B * N_B * UCE / muw) \
              + beta * (1 + piw_B(+1)).apply(np.log) - (1 + piw_B).apply(np.log)
    return wnkpc_B


@simple
def partial_ss_two_sector(Y, N_A, N_B, K, r, tot_wealth, Bg, delta, sigma):
    # CES aggregate N
    if abs(sigma - 1.0) < 1e-6:
        N = (N_A * N_B) ** 0.5
    else:
        rho = (sigma - 1) / sigma
        N = (N_A ** rho + N_B ** rho) ** (1 / rho)

    p = tot_wealth - Bg
    mc = 1 - r * (p - K) / Y
    mup = 1 / mc
    alpha = (r + delta) * K / Y / mc
    Z = Y * K ** (-alpha) * N ** (alpha - 1)
    w = mc * (1 - alpha) * Y / N
    w_A, w_B = w, w
    return p, mc, mup, alpha, Z, w, w_A, w_B, N


@simple
def dividend_two_sector(Y, w_A, N_A, w_B, N_B, K, pi, mup, kappap, delta, epsI):
    psip = mup / (mup - 1) / 2 / kappap * (1 + pi).apply(np.log) ** 2 * Y
    k_adjust = K(-1) * (K / K(-1) - 1) ** 2 / (2 * delta * epsI)
    I = K - (1 - delta) * K(-1) + k_adjust
    div = Y - (w_A * N_A + w_B * N_B) - I - psip
    return psip, I, div


@simple
def fiscal_two_sector(r, w_A, N_A, w_B, N_B, G, Bg):
    tax = (r * Bg + G) / (w_A * N_A + w_B * N_B)
    return tax


@simple
def union_ss_A(tax, w_A, UCE, N_A, muw, frisch):
    vphi_A = (1 - tax) * w_A * UCE / muw / N_A ** (1 + 1 / frisch)
    wnkpc_A = vphi_A * N_A ** (1 + 1 / frisch) - (1 - tax) * w_A * UCE / muw
    return vphi_A, wnkpc_A


@simple
def union_ss_B(tax, w_B, UCE, N_B, muw, frisch):
    vphi_B = (1 - tax) * w_B * UCE / muw / N_B ** (1 + 1 / frisch)
    wnkpc_B = vphi_B * N_B ** (1 + 1 / frisch) - (1 - tax) * w_B * UCE / muw
    return vphi_B, wnkpc_B


# ====================================================================
# Cantillon-specific: income with staggered group-specific transfers
# ====================================================================

def make_grids(bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z):
    b_grid = grids.agrid(amax=bmax, n=nB)
    a_grid = grids.agrid(amax=amax, n=nA)
    k_grid = grids.agrid(amax=kmax, n=nK)[::-1].copy()
    e_grid, _, Pi = grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)
    return b_grid, a_grid, k_grid, e_grid, Pi


def income_cantillon(e_grid, tax, w, N, transfer_early, transfer_late):
    """Income with Cantillon transfers.

    Early transfer: only top-productivity agent(s) receive it.
    Late transfer: only bottom-productivity agent(s) receive it.

    Shares are normalized so the aggregate mean = 1.
    For nZ=3: top agent gets 3x aggregate, bottom agent gets 3x aggregate.
    """
    nZ = len(e_grid)
    z_labor = (1 - tax) * w * N * e_grid

    order = np.argsort(e_grid)
    n_high = max(1, nZ // 3)
    n_low = max(1, nZ // 3)

    early_share = np.zeros(nZ)
    early_share[order[-n_high:]] = 1.0
    early_share *= nZ / n_high

    late_share = np.zeros(nZ)
    late_share[order[:n_low]] = 1.0
    late_share *= nZ / n_low

    z_grid = z_labor + transfer_early * early_share + transfer_late * late_share
    return z_grid


def income_two_sector(e_grid, tax, w_A, N_A, w_B, N_B, transfer_early, transfer_late):
    """Income with sector-specific wages and Cantillon transfers.

    Each agent's effective wage is a sector-weighted average of w_A and w_B.
    High-e agents → 100% w_A; low-e agents → 100% w_B; middle → 50/50.
    Income shares are proportional to e * effective_wage, normalized so
    aggregate labor income = (1-tax)*(w_A*N_A + w_B*N_B).
    """
    nZ = len(e_grid)
    order = np.argsort(e_grid)
    n_high = max(1, nZ // 3)
    n_low = max(1, nZ // 3)

    high_idx = set(order[-n_high:])
    low_idx = set(order[:n_low])

    w_eff = np.zeros(nZ)
    for i in range(nZ):
        if i in high_idx:
            w_eff[i] = w_A
        elif i in low_idx:
            w_eff[i] = w_B
        else:
            w_eff[i] = 0.5 * w_A + 0.5 * w_B

    # Income share proportional to e * effective wage
    raw = e_grid * w_eff
    share = raw / raw.sum()

    total_labor = (1 - tax) * (w_A * N_A + w_B * N_B)
    z_labor = nZ * total_labor * share

    # Transfers: same distribution as one-sector version
    early_share = np.zeros(nZ)
    early_share[order[-n_high:]] = 1.0
    early_share *= nZ / n_high

    late_share = np.zeros(nZ)
    late_share[order[:n_low]] = 1.0
    late_share *= nZ / n_low

    z_grid = z_labor + transfer_early * early_share + transfer_late * late_share
    return z_grid


# ====================================================================
# Model construction
# ====================================================================

def build_model(kappaw=0.1):
    """Build two-asset HANK with Cantillon transfer capability."""
    calibration = {
        'Y': 1., 'N': 1.0, 'K': 10., 'r': 0.0125, 'rstar': 0.0125,
        'tot_wealth': 14, 'delta': 0.02, 'pi': 0.,
        'kappap': 0.1, 'muw': 1.1, 'Bh': 1.04, 'Bg': 2.8, 'G': 0.2,
        'eis': 0.5, 'frisch': 1, 'chi0': 0.25, 'chi2': 2, 'epsI': 4,
        'omega': 0.005, 'kappaw': kappaw, 'phi': 1.5,
        'nZ': 3, 'nB': 10, 'nA': 16, 'nK': 4,
        'bmax': 50, 'amax': 4000, 'kmax': 1, 'rho_z': 0.966, 'sigma_z': 0.92,
        'transfer_early': 0.0, 'transfer_late': 0.0,
    }

    household = hh.add_hetinputs([income_cantillon, make_grids])

    production = combine([labor, investment])
    production_solved = production.solved(
        unknowns={'Q': 1., 'K': 10.}, targets=['inv', 'val'], solver='broyden_custom'
    )

    blocks = [
        household, pricing_solved, arbitrage_solved, production_solved,
        dividend, taylor, fiscal, share_value, finance, wage, union, mkt_clearing
    ]
    model = create_model(blocks, name='Two-Asset HANK with Cantillon')

    blocks_ss = [
        household, partial_ss, dividend, taylor, fiscal, share_value,
        finance, union_ss, mkt_clearing
    ]
    model_ss = create_model(blocks_ss, name='Two-Asset HANK SS')

    unknowns_ss = {'beta': 0.976, 'chi1': 6.5}
    targets_ss = {'asset_mkt': 0., 'B': 'Bh'}
    cali = model_ss.solve_steady_state(calibration, unknowns_ss, targets_ss,
                                       solver='broyden_custom')
    ss = model.steady_state(cali)

    unknowns = ['r', 'w', 'Y']
    targets = ['asset_mkt', 'fisher', 'wnkpc']
    exogenous = ['rstar', 'Z', 'G', 'transfer_early', 'transfer_late']

    return ss, model, unknowns, targets, exogenous


def build_model_two_sector(kappaw=0.1, sigma=5.0, kappaw_A=None, kappaw_B=None):
    """Build two-asset HANK with TWO labor sectors.

    sigma: elasticity of substitution between sector A and B labor.
      Lower sigma = less substitutability = less GE wage spillover.
    kappaw_A: wage stickiness for early-recipient sector (default = kappaw).
    kappaw_B: wage stickiness for late-recipient sector (default = kappaw).
    """
    if kappaw_A is None:
        kappaw_A = kappaw
    if kappaw_B is None:
        kappaw_B = kappaw

    # CES-consistent SS values for N_A, N_B (w_A=w_B → equal shares)
    if abs(sigma - 1.0) < 1e-6:
        nAB = 0.5
    else:
        nAB = 1.0 * 2 ** (-sigma / (sigma - 1))

    calibration = {
        'Y': 1., 'N': 1.0, 'K': 10., 'r': 0.0125, 'rstar': 0.0125,
        'tot_wealth': 14, 'delta': 0.02, 'pi': 0.,
        'kappap': 0.1, 'muw': 1.1, 'Bh': 1.04, 'Bg': 2.8, 'G': 0.2,
        'eis': 0.5, 'frisch': 1, 'chi0': 0.25, 'chi2': 2, 'epsI': 4,
        'omega': 0.005, 'kappaw_A': kappaw_A, 'kappaw_B': kappaw_B,
        'sigma': sigma, 'phi': 1.5,
        'nZ': 3, 'nB': 10, 'nA': 16, 'nK': 4,
        'bmax': 50, 'amax': 4000, 'kmax': 1, 'rho_z': 0.966, 'sigma_z': 0.92,
        'transfer_early': 0.0, 'transfer_late': 0.0,
        'N_A': nAB, 'N_B': nAB, 'w_A': 0.66, 'w_B': 0.66,
    }

    household = hh.add_hetinputs([income_two_sector, make_grids])

    production = combine([labor_two_sector, investment])
    production_solved = production.solved(
        unknowns={'Q': 1., 'K': 10.}, targets=['inv', 'val'], solver='broyden_custom'
    )

    blocks = [
        household, pricing_solved, arbitrage_solved, production_solved,
        dividend_two_sector, taylor, fiscal_two_sector, share_value, finance,
        wage_A, wage_B, union_A, union_B, mkt_clearing
    ]
    model = create_model(blocks, name='Two-Asset HANK with Cantillon (2-sector)')

    blocks_ss = [
        household, partial_ss_two_sector, dividend_two_sector, taylor, fiscal_two_sector,
        share_value, finance, union_ss_A, union_ss_B, mkt_clearing
    ]
    model_ss = create_model(blocks_ss, name='Two-Asset HANK SS (2-sector)')

    unknowns_ss = {'beta': 0.976, 'chi1': 6.5}
    targets_ss = {'asset_mkt': 0., 'B': 'Bh'}
    cali = model_ss.solve_steady_state(calibration, unknowns_ss, targets_ss,
                                       solver='broyden_custom')
    ss = model.steady_state(cali)

    unknowns = ['r', 'w_A', 'w_B', 'Y']
    targets = ['asset_mkt', 'fisher', 'wnkpc_A', 'wnkpc_B']
    exogenous = ['rstar', 'Z', 'G', 'transfer_early', 'transfer_late']

    return ss, model, unknowns, targets, exogenous


# ====================================================================
# Simulation and welfare
# ====================================================================

def build_shocks(T, delta_tau, transfer_early_val, transfer_late_val, tau_R=1):
    """Build shock paths for given transfer magnitudes."""
    te = np.zeros(T)
    if tau_R < T:
        te[tau_R:] = transfer_early_val

    tl = np.zeros(T)
    tau_H = tau_R + delta_tau
    if tau_H < T:
        tl[tau_H:] = transfer_late_val

    return te, tl


def compute_irfs_from_G(G, model_outputs, unknowns, T, te_real, tl_real):
    """Compute IRFs from pre-computed Jacobian and real transfer paths."""
    irfs = {}
    for var in model_outputs:
        irf_val = np.zeros(T)
        for sn, path in [('transfer_early', te_real), ('transfer_late', tl_real)]:
            if var in G and sn in G[var]:
                irf_val += G[var][sn] @ path
        irfs[var] = irf_val
    for var in unknowns:
        if var in G:
            irf_val = np.zeros(T)
            for sn, path in [('transfer_early', te_real), ('transfer_late', tl_real)]:
                if sn in G[var]:
                    irf_val += G[var][sn] @ path
            irfs[var] = irf_val
    irfs['transfer_early'] = te_real
    irfs['transfer_late'] = tl_real
    return irfs


def compute_irfs_nominal(G, model_outputs, unknowns, T, te_nominal, tl_nominal,
                          max_iter=50, tol=1e-10, damp=0.5):
    """Fixed-point iteration for nominal transfers.

    te_nominal, tl_nominal: nominal transfer paths (flat after start date).
    The real value erodes with the price level: real_t = nominal_t / P_t.
    P_t evolves endogenously with cumulative inflation.

    Returns (irfs, iterations, converged).
    """
    # Initial guess: zero inflation → P=1 everywhere
    P_path = np.ones(T)
    P_prev = P_path.copy()

    for it in range(max_iter):
        # Deflate nominal transfers by current P
        te_real = te_nominal / P_path
        tl_real = tl_nominal / P_path

        irfs = compute_irfs_from_G(G, model_outputs, unknowns, T, te_real, tl_real)

        # New price level from inflation path
        pi_irf = irfs.get('pi', np.zeros(T))
        P_new = np.cumprod(1.0 + pi_irf)

        # Dampened update
        P_path = damp * P_new + (1 - damp) * P_prev

        max_diff = np.max(np.abs(P_path - P_prev))
        if max_diff < tol:
            return irfs, it + 1, True

        P_prev = P_path.copy()

    return irfs, max_iter, False


def compute_welfare(ss, irfs, mu_R, mu_H, delta_debt, nZ=3, n_low=1,
                   beta_annual=0.95, nominal_transfers=True, spillover=1.0):
    """Compute welfare for late group.

    spillover: fraction of aggregate wage AND employment changes that reach late group.
    1.0 = perfect spillover (uniform labor market); 0.0 = zero spillover (full separation).
    Only used in one-sector mode.
    """
    T = len(irfs.get('pi', irfs.get('Y', np.zeros(300))))
    beta = beta_annual ** (1 / 12)

    tax_ss = ss['tax'] if 'tax' in ss else 0.36

    # Detect model type and extract relevant SS/IRF variables
    two_sector = ('w_B' in irfs and 'N_B' in irfs)

    if two_sector:
        w_ss = ss['w_B'] if 'w_B' in ss else 0.66
        N_ss = ss['N_B'] if 'N_B' in ss else 0.5
        dw = irfs.get('w_B', np.zeros(T))
        dN = irfs.get('N_B', np.zeros(T))
    else:
        w_ss = ss['w'] if 'w' in ss else 0.66
        N_ss = ss['N'] if 'N' in ss else 1.0
        dw = irfs.get('w', np.zeros(T))
        dN = irfs.get('N', np.zeros(T))

    te = irfs.get('transfer_early', np.zeros(T))
    tl = irfs.get('transfer_late', np.zeros(T))
    pi_irf = irfs.get('pi', np.zeros(T))

    w_path = w_ss + dw
    N_path = N_ss + dN
    P_path = np.cumprod(1.0 + pi_irf)

    # Per-agent labor income in SS
    e_low = 0.5
    if two_sector:
        # Two-sector: computes via income_two_sector formula
        w_A_ss = ss['w_A'] if 'w_A' in ss else 0.66
        w_B_ss = ss['w_B'] if 'w_B' in ss else 0.66
        N_A_ss = ss['N_A'] if 'N_A' in ss else 0.5
        N_B_ss = ss['N_B'] if 'N_B' in ss else 0.5
        e_high, e_mid = 1.5, 1.0
        nZ = 3

        raw_ss = e_high * w_A_ss + e_mid * 0.5 * (w_A_ss + w_B_ss) + e_low * w_B_ss
        share_low_ss = e_low * w_B_ss / raw_ss
        total_labor_ss = (1 - tax_ss) * (w_A_ss * N_A_ss + w_B_ss * N_B_ss)
        y_late_labor_ss = nZ * total_labor_ss * share_low_ss

        w_A_path = w_A_ss + irfs.get('w_A', np.zeros(T))
        w_B_path = w_B_ss + irfs.get('w_B', np.zeros(T))
        N_A_path = N_A_ss + irfs.get('N_A', np.zeros(T))
        N_B_path = N_B_ss + irfs.get('N_B', np.zeros(T))

        raw_path = e_high * w_A_path + e_mid * 0.5 * (w_A_path + w_B_path) + e_low * w_B_path
        share_low_path = e_low * w_B_path / raw_path
        total_labor_path = (1 - tax_ss) * (w_A_path * N_A_path + w_B_path * N_B_path)
        y_late_labor_path = nZ * total_labor_path * share_low_path
    else:
        # One-sector with spillover: late group gets spillover fraction of GE gains
        w_eff_path = w_ss + spillover * dw
        N_eff_path = N_ss + spillover * dN
        y_late_labor_ss = (1 - tax_ss) * w_ss * N_ss * e_low
        y_late_labor_path = (1 - tax_ss) * w_eff_path * N_eff_path * e_low

    late_share = nZ / n_low

    if nominal_transfers:
        y_late = y_late_labor_path + tl * late_share / P_path
    else:
        y_late = y_late_labor_path + tl * late_share

    y_late_ss = y_late_labor_ss

    # Cantillon welfare
    beta_vec = beta ** np.arange(T)
    cantillon_pv = np.sum(beta_vec * (y_late - y_late_ss))
    norm = y_late_ss if abs(y_late_ss) > 1e-10 else 1.0
    cantillon_welfare = cantillon_pv / norm

    # Balance-sheet
    P_terminal = P_path[-1]
    if delta_debt is not None and P_terminal > 1e-10:
        bs_welfare = delta_debt * (1.0 - 1.0 / P_terminal)
    else:
        bs_welfare = 0.0

    return {
        'cantillon_welfare': cantillon_welfare,
        'balance_sheet_welfare': bs_welfare,
        'total_welfare': cantillon_welfare + bs_welfare,
        'P_terminal': P_terminal,
        'y_late_ss': y_late_ss,
    }


def find_delta_star(ss, irfs, mu_R, mu_H, nZ=3, n_low=1, beta_annual=0.95,
                    nominal_transfers=True, spillover=1.0):
    """Find delta* where Cantillon + balance-sheet welfare = 0."""
    res0 = compute_welfare(ss, irfs, mu_R, mu_H, delta_debt=0.0, nZ=nZ, n_low=n_low,
                           beta_annual=beta_annual, nominal_transfers=nominal_transfers,
                           spillover=spillover)
    res1 = compute_welfare(ss, irfs, mu_R, mu_H, delta_debt=1.0, nZ=nZ, n_low=n_low,
                           beta_annual=beta_annual, nominal_transfers=nominal_transfers,
                           spillover=spillover)

    bs_per_delta = res1['balance_sheet_welfare']
    if abs(bs_per_delta) < 1e-12:
        return np.inf if res0['cantillon_welfare'] < 0 else -np.inf

    return -res0['cantillon_welfare'] / bs_per_delta




# ====================================================================
# Main: sweep over parameters
# ====================================================================

def run_sweep_one_sector(T=300, bar_mu=0.10, kappaw_values=None):
    """One-sector sweep (original model)."""
    import time

    if kappaw_values is None:
        kappaw_values = [0.1]

    mu_ratios = [1.5, 2.0, 3.0, 5.0]
    delta_taus = [6, 12, 18, 24, 36]
    nZ, n_high, n_low = 3, 1, 1
    e_high, e_low = 1.5, 0.5
    all_results = {}

    for kw in kappaw_values:
        t0 = time.time()
        print(f"\n{'='*80}")
        print(f"ONE-SECTOR: kappaw = {kw:.3f}")
        print(f"{'='*80}")
        ss, model, unknowns, targets, exogenous = build_model(kappaw=kw)

        inc_scale = (1 - ss['tax']) * ss['w'] * ss['N']
        print(f"  inc_scale={inc_scale:.4f}, w={ss['w']:.4f}, N={ss['N']:.4f}, tax={ss['tax']:.4f}")

        print("  Computing Jacobian...")
        G = model.solve_jacobian(
            ss, unknowns, targets, exogenous, T=T,
            shock_list=['transfer_early', 'transfer_late', 'rstar', 'Z', 'G']
        )
        print(f"  done in {time.time() - t0:.1f}s")

        print(f"\n{'μ_R/μ_H':<10}", end="")
        for dt in delta_taus:
            print(f"  δ*(Δτ={dt:2d})", end="")
        print(f"  {'μ_R':<8} {'μ_H':<8}")
        print("-" * 70)

        results = {}
        model_outputs = list(model.outputs)
        for mr in mu_ratios:
            mu_H = 2 * bar_mu / (2.0 + 1)  # fixed at mr=2 benchmark, not shrinking with mr
            mu_R = mr * mu_H
            t_early_mag = mu_R * inc_scale * e_high * n_high / nZ
            t_late_mag = mu_H * inc_scale * e_low * n_low / nZ
            print(f"{mr:<10.1f}", end="")
            for dt in delta_taus:
                te_nom, tl_nom = build_shocks(T, dt, t_early_mag, t_late_mag)
                irfs, nit, conv = compute_irfs_nominal(
                    G, model_outputs, unknowns, T, te_nom, tl_nom)
                ds = find_delta_star(ss, irfs, mu_R, mu_H, nZ=nZ, n_low=n_low,
                                     nominal_transfers=False)
                print(f"{ds:>13.1f}", end="")
                results[(mr, dt)] = ds
            print(f"  {mu_R:<8.4f} {mu_H:<8.4f}")
        print(f"  Elapsed: {time.time() - t0:.1f}s")
        all_results[kw] = results

    return all_results


def run_sweep_two_sector(T=300, bar_mu=0.10, sigma_values=None,
                          kappaw=0.1, kappaw_A=None, kappaw_B=None):
    """Two-sector sweep: separate labor markets for early/late groups."""
    import time

    if sigma_values is None:
        sigma_values = [3, 5, 10]

    mu_ratios = [1.5, 2.0, 3.0, 5.0]
    delta_taus = [6, 12, 18, 24, 36]
    nZ, n_high, n_low = 3, 1, 1
    e_high, e_low = 1.5, 0.5
    all_results = {}

    for sigma in sigma_values:
        t0 = time.time()
        print(f"\n{'='*80}")
        print(f"TWO-SECTOR: sigma={sigma} (kappaw_A={kappaw_A or kappaw}, kappaw_B={kappaw_B or kappaw})")
        print(f"{'='*80}")

        ss, model, unknowns, targets, exogenous = build_model_two_sector(
            kappaw=kappaw, sigma=sigma, kappaw_A=kappaw_A, kappaw_B=kappaw_B)

        inc_scale = (1 - ss['tax']) * (ss['w_A'] * ss['N_A'] + ss['w_B'] * ss['N_B'])
        print(f"  w_A={ss['w_A']:.4f}, w_B={ss['w_B']:.4f}, N_A={ss['N_A']:.4f}, N_B={ss['N_B']:.4f}")
        print(f"  inc_scale={inc_scale:.4f}, tax={ss['tax']:.4f}")

        print("  Computing Jacobian...")
        G = model.solve_jacobian(
            ss, unknowns, targets, exogenous, T=T,
            shock_list=['transfer_early', 'transfer_late', 'rstar', 'Z', 'G']
        )
        print(f"  done in {time.time() - t0:.1f}s")

        print(f"\n{'μ_R/μ_H':<10}", end="")
        for dt in delta_taus:
            print(f"  δ*(Δτ={dt:2d})", end="")
        print(f"  {'μ_R':<8} {'μ_H':<8}")
        print("-" * 70)

        results = {}
        model_outputs = list(model.outputs)
        for mr in mu_ratios:
            mu_H = 2 * bar_mu / (2.0 + 1)  # fixed at mr=2 benchmark, not shrinking with mr
            mu_R = mr * mu_H

            # Transfer calibrated to sector-specific income scales
            t_early_mag = mu_R * inc_scale * e_high * n_high / nZ
            t_late_mag = mu_H * inc_scale * e_low * n_low / nZ

            print(f"{mr:<10.1f}", end="")
            for dt in delta_taus:
                te_nom, tl_nom = build_shocks(T, dt, t_early_mag, t_late_mag)
                irfs, nit, conv = compute_irfs_nominal(
                    G, model_outputs, unknowns, T, te_nom, tl_nom)
                ds = find_delta_star(ss, irfs, mu_R, mu_H, nZ=nZ, n_low=n_low,
                                     nominal_transfers=False)
                print(f"{ds:>13.1f}", end="")
                results[(mr, dt)] = ds
            print(f"  {mu_R:<8.4f} {mu_H:<8.4f}")
        print(f"  Elapsed: {time.time() - t0:.1f}s")
        all_results[sigma] = results

    # Cross-section for mu_R/mu_H=2.0
    print(f"\n{'='*80}")
    print("CROSS-SECTION: δ* for μ_R/μ_H=2.0 across sigma values")
    print(f"{'='*80}")
    print(f"{'sigma':<10}", end="")
    for dt in delta_taus:
        print(f"  δ*(Δτ={dt:2d})", end="")
    print()
    print("-" * 70)
    for sigma in sigma_values:
        print(f"{sigma:<10.0f}", end="")
        for dt in delta_taus:
            ds = all_results[sigma][(2.0, dt)]
            print(f"{ds:>13.1f}", end="")
        print()

    print(f"\nEmpirical δ band: 10–16")
    return all_results


def run_sweep_spillover(T=300, bar_mu=0.10, kappaw=0.1, spillover_values=None):
    """Sweep over wage spillover: fraction of dw that reaches late group."""
    import time

    if spillover_values is None:
        spillover_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    mu_ratios = [1.5, 2.0, 3.0, 5.0]
    delta_taus = [6, 12, 18, 24, 36]
    nZ, n_high, n_low = 3, 1, 1
    e_high, e_low = 1.5, 0.5

    print("Building one-sector model...")
    t0 = time.time()
    ss, model, unknowns, targets, exogenous = build_model(kappaw=kappaw)
    inc_scale = (1 - ss['tax']) * ss['w'] * ss['N']
    print(f"  inc_scale={inc_scale:.4f}, w={ss['w']:.4f}, N={ss['N']:.4f}, tax={ss['tax']:.4f}")

    print("Computing Jacobian...")
    G = model.solve_jacobian(
        ss, unknowns, targets, exogenous, T=T,
        shock_list=['transfer_early', 'transfer_late', 'rstar', 'Z', 'G']
    )
    print(f"  done in {time.time() - t0:.1f}s")

    model_outputs = list(model.outputs)
    all_results = {}

    for sp in spillover_values:
        print(f"\n{'='*80}")
        print(f"SPILLOVER = {sp:.2f} (late group captures {sp*100:.0f}% of dw)")
        print(f"{'='*80}")

        print(f"{'μ_R/μ_H':<10}", end="")
        for dt in delta_taus:
            print(f"  δ*(Δτ={dt:2d})", end="")
        print(f"  {'μ_R':<8} {'μ_H':<8}")
        print("-" * 70)

        results = {}
        for mr in mu_ratios:
            mu_H = 2 * bar_mu / (2.0 + 1)  # fixed at mr=2 benchmark, not shrinking with mr
            mu_R = mr * mu_H
            t_early_mag = mu_R * inc_scale * e_high * n_high / nZ
            t_late_mag = mu_H * inc_scale * e_low * n_low / nZ
            print(f"{mr:<10.1f}", end="")
            for dt in delta_taus:
                te_nom, tl_nom = build_shocks(T, dt, t_early_mag, t_late_mag)
                irfs, nit, conv = compute_irfs_nominal(
                    G, model_outputs, unknowns, T, te_nom, tl_nom)
                ds = find_delta_star(ss, irfs, mu_R, mu_H, nZ=nZ, n_low=n_low,
                                     nominal_transfers=False, spillover=sp)
                print(f"{ds:>13.1f}", end="")
                results[(mr, dt)] = ds
            print(f"  {mu_R:<8.4f} {mu_H:<8.4f}")
        all_results[sp] = results

    # Cross-section
    print(f"\n{'='*80}")
    print("CROSS-SECTION: δ* for μ_R/μ_H=2.0 across spillover values")
    print(f"{'='*80}")
    print(f"{'spillover':<12}", end="")
    for dt in delta_taus:
        print(f"  δ*(Δτ={dt:2d})", end="")
    print()
    print("-" * 80)
    for sp in spillover_values:
        print(f"{sp:<12.2f}", end="")
        for dt in delta_taus:
            ds = all_results[sp][(2.0, dt)]
            print(f"{ds:>13.1f}", end="")
        print()

    # Also show μ_R/μ_H=3.0
    print(f"\nCROSS-SECTION: δ* for μ_R/μ_H=3.0 across spillover values")
    print(f"{'spillover':<12}", end="")
    for dt in delta_taus:
        print(f"  δ*(Δτ={dt:2d})", end="")
    print()
    print("-" * 80)
    for sp in spillover_values:
        print(f"{sp:<12.2f}", end="")
        for dt in delta_taus:
            ds = all_results[sp][(3.0, dt)]
            print(f"{ds:>13.1f}", end="")
        print()

    print(f"\nEmpirical δ band: 10–16")
    return all_results


if __name__ == '__main__':
    run_sweep_spillover(spillover_values=[0.0, 0.25, 0.5, 0.75, 1.0], kappaw=0.1)
