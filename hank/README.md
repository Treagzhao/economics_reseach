# Replication Code for "HANK After Cantillon: Two Hidden Premises and a Breakeven Condition"

This repository contains the simulation code for Zhao (2026).

## Structure

- `tank/` — TANK baseline simulation (Section 4, Tables 2–4). Self-contained, no external dependencies beyond NumPy.
- `full_hank/` — Two-asset HANK simulation with staggered receipt and spillover (Section 5.5, Table 6). Requires `sequence-jacobian`.
- `mpc/` — MPC heterogeneity extension (Section 5.1). Builds on the TANK baseline.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### TANK Baseline (Tables 2–4)

```bash
cd tank
python calvo_simulations.py
```

Generates breakeven debt ratios $\delta^*$ for all asymmetry ratios, receipt gaps, and income ratios reported in the paper.

To verify the key claim independently:

```bash
python independent_verification.py
```

### Full HANK Spillover (Table 6)

```bash
cd full_hank
python diag_spillover.py
```

Parameters: $\kappa_w = 0.1$ (monthly Calvo wage adjustment, standard HANK calibration), $\bar{\mu} = 0.10$ (aggregate injection), $\mu_R/\mu_H = 2.0$ (asymmetry ratio). The spillover parameter is swept from 0 to 1. Calvo wage rigidity follows the same quarterly-to-monthly conversion as the price Calvo parameter ($\theta_q = 0.75 \Rightarrow \theta_m \approx 0.909$, $\kappa_w = 0.1$).

The concavity of $\delta^*$ in $\mu_R/\mu_H$ — with the peak at $\mu_R/\mu_H = 2.0$ — is identified by grid search over $[1.3, 5.0]$. The full grid is reported in the accompanying research note (`hank/full_hank/findings.md` in the main manuscript repository).

### MPC Extension (Section 5.1)

```bash
cd mpc
python run_mpc_analysis.py
```

Sweeps $\text{mpc}_R$ from 0 to 1 for all asymmetry scenarios. Key result: $\delta^*$ requires $\text{mpc}_R \approx 0.21$ to enter the empirical debt band $[10, 16]$.

## Parameter Sources

| Parameter | Value | Source |
|-----------|-------|--------|
| Monthly Calvo price adjustment $g$ | 0.091 | $\theta_q = 0.75$ (Kaplan, Moll, Violante 2018) → $\theta_m = 0.75^{1/3}$ |
| Monthly Calvo wage adjustment $\kappa_w$ | 0.1 | Standard HANK calibration (Auclert, Rognlie, Straub 2024) |
| Aggregate injection $\bar{\mu}$ | 0.10 | Cumulative nominal income effect of large-scale QE |
| Asymmetry ratio $\mu_R/\mu_H$ | 2.0 (HANK peak), 5.0 (QE-era) | Saez (2016); peak verified by grid search |
| Income ratio $\alpha$ | 5.0 | Saez (2016), top 1% vs. bottom 99% |
| Transmission lag $\Delta\tau$ | 12–36 months | Lenza and Slacalek (2024); Peneva (2013) |
| Household debt ratio $\delta$ | 10–16 (monthly) | Fed Z.1, NYFed CCP, SCF |
