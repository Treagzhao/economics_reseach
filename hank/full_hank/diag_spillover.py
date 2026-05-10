"""Diagnose: check magnitudes of dw, dN, pi in IRFs."""
import numpy as np
import sys
sys.path.insert(0, 'sequence-jacobian/src')
from sequence_jacobian import simple, solved, combine, create_model, grids, hetblocks

# Import the model functions from cantillon_hank.py
from cantillon_hank import (build_model, build_shocks,
                            compute_irfs_nominal,
                            find_delta_star, compute_welfare)

ss, model, unknowns, targets, exogenous = build_model(kappaw=0.1)
inc_scale = (1 - ss['tax']) * ss['w'] * ss['N']
T = 300

G = model.solve_jacobian(ss, unknowns, targets, exogenous, T=T,
    shock_list=['transfer_early', 'transfer_late', 'rstar', 'Z', 'G'])

# mu_R/mu_H = 2.0, dt=12
bar_mu = 0.10
mr = 2.0
mu_H = 2 * bar_mu / (mr + 1)
mu_R = mr * mu_H
e_high, e_low = 1.5, 0.5
nZ, n_high, n_low = 3, 1, 1
t_early_mag = mu_R * inc_scale * e_high * n_high / nZ
t_late_mag = mu_H * inc_scale * e_low * n_low / nZ

te_nom = np.zeros(T)
tl_nom = np.zeros(T)
tau_R = 1
dt = 12
te_nom[tau_R:] = t_early_mag
tau_H = tau_R + dt
tl_nom[tau_H:] = t_late_mag

irfs, nit, conv = compute_irfs_nominal(
    G, list(model.outputs), unknowns, T, te_nom, tl_nom)

print(f'SS w={ss["w"]:.6f}, N={ss["N"]:.6f}, tax={ss["tax"]:.6f}')
print(f'Convergence: {conv}, iterations: {nit}')

for var in ['w', 'N', 'pi', 'Y', 'C', 'r']:
    if var in irfs:
        path = irfs[var]
        print(f'{var}: max(|d{var}|)={np.max(np.abs(path)):.8f}, '
              f'at t=12: {path[12]:.8f}, '
              f'at t=24: {path[24]:.8f}')

# Price level
pi_irf = irfs.get('pi', np.zeros(T))
P_path = np.cumprod(1.0 + pi_irf)
print(f'P_terminal = {P_path[-1]:.8f}')

# Check welfare components
res0 = compute_welfare(ss, irfs, mu_R, mu_H, delta_debt=0.0, nZ=nZ, n_low=1,
                       beta_annual=0.95, nominal_transfers=False)
res1 = compute_welfare(ss, irfs, mu_R, mu_H, delta_debt=1.0, nZ=nZ, n_low=1,
                       beta_annual=0.95, nominal_transfers=False)
print(f'\nCantillon welfare: {res0["cantillon_welfare"]:.10f}')
print(f'BS welfare per delta: {res1["balance_sheet_welfare"]:.10f}')
print(f'delta* = {-res0["cantillon_welfare"] / res1["balance_sheet_welfare"]:.2f}')
print(f'y_late_ss = {res0["y_late_ss"]:.6f}')
print(f'P_terminal = {res0["P_terminal"]:.8f}')

# Manual check with spillover=0
tax_ss = ss['tax']
w_ss = ss['w']
N_ss = ss['N']
e_low = 0.5
dw = irfs.get('w', np.zeros(T))
dN = irfs.get('N', np.zeros(T))
tl = irfs.get('transfer_late', np.zeros(T))
late_share = nZ / 1

for sp in [0.0, 0.5, 1.0]:
    w_eff = w_ss + sp * dw
    N_eff = N_ss + sp * dN
    y_late = (1 - tax_ss) * w_eff * N_eff * e_low + tl * late_share
    y_late_ss = (1 - tax_ss) * w_ss * N_ss * e_low
    beta = 0.95 ** (1/12)
    beta_vec = beta ** np.arange(T)
    cantillon_pv = np.sum(beta_vec * (y_late - y_late_ss))
    norm = y_late_ss
    cw = cantillon_pv / norm
    print(f'\nspillover={sp:.1f}: cantillon_welfare={cw:.10f}, delta* = {-cw / res1["balance_sheet_welfare"]:.2f}')
