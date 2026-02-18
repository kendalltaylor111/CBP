"""
CBP Simulation - Enhanced Version
==================================
Tests scaling of discrete vs. continuous tracking for 1/f^α noise.
Generates figures and prints tables for convergence, power ratios, and exponents.
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import time
import json
import os

# Set random seed for reproducibility, but we'll vary it for multiple runs
base_seed = 42

def generate_colored_noise(alpha, N, dt, seed):
    """Generate 1/f^α noise with given seed."""
    rng = np.random.RandomState(seed)
    freqs = np.fft.rfftfreq(N, d=dt)
    psd = np.zeros_like(freqs)
    psd[1:] = freqs[1:] ** (-alpha)
    # Avoid zero at DC
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    spectrum = np.sqrt(psd) * np.exp(1j * phases)
    spectrum[0] = 0  # zero DC component
    x = np.fft.irfft(spectrum, n=N)
    return x / np.std(x)  # normalize to unit variance

def discrete_track(signal, dt, fs):
    """
    Discrete tracking: sample at fs, reconstruct with linear interpolation.
    Returns RMSE and theoretical minimum power (rate-distortion bound).
    """
    N = len(signal)
    t = np.arange(N) * dt
    # sampling interval
    interval = max(1, int(round(1.0 / (fs * dt))))
    if interval >= N:
        # too low fs: can't reconstruct
        return 1.0, 0.0
    idx = np.arange(0, N, interval)
    reconstructed = np.interp(t, t[idx], signal[idx])
    rmse = np.sqrt(np.mean((signal - reconstructed) ** 2))
    # Minimum power from rate-distortion for Gaussian source:
    # Rate = 0.5 * log2(1/ε)  (for MSE distortion)
    # Power = rate * fs   (in arbitrary units, assuming energy per sample = 1)
    # We cap rate at a minimum to avoid negative/zero
    rate = 0.5 * np.log2(max(1.0 / max(rmse, 1e-15), 1.0))
    power = fs * rate
    return rmse, power

def continuous_track(signal, dt, f_c, Q=20):
    """
    Continuous tracking: optimal linear filter (Wiener) given signal's PSD.
    Power modeled as damping loss in an oscillator: ~ (1/Q) * integral f * |H|^2 * S(f) df.
    Returns RMSE, damping power, and bandwidth (f_c) as a simple proxy.
    """
    N = len(signal)
    sig_fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=dt)
    # Signal power spectral density (estimated from this realization)
    psd = np.abs(sig_fft) ** 2 / N

    # Ideal filter: Wiener = S_signal / (S_signal + S_noise). Here we assume no noise.
    # But to avoid perfect reconstruction, we use a lowpass with cutoff f_c.
    # For continuous tracking, a physical oscillator has a resonant peak.
    # We'll use a second-order lowpass (Butterworth) with cutoff f_c.
    # Transfer function magnitude squared: |H|^2 = 1 / (1 + (f/f_c)^4)  (2nd order)
    H2 = 1.0 / (1.0 + (freqs / f_c) ** 4)
    filtered_spectrum = sig_fft * np.sqrt(H2)
    filtered = np.fft.irfft(filtered_spectrum, n=N)
    rmse = np.sqrt(np.mean((signal - filtered) ** 2))

    # Damping power: (1/Q) * integral f * |H|^2 * psd df   (as in original)
    # This represents energy dissipated in an oscillator maintaining resonance.
    df = freqs[1] - freqs[0]
    power_damping = (1.0 / Q) * np.sum(freqs * psd * H2) * df

    # Also compute bandwidth proxy (just f_c)
    power_bw = f_c

    return rmse, power_damping, power_bw

def fit_power_law(x, y, x_range=None):
    """Fit power law y = C * x^beta in log-log space."""
    mask = np.isfinite(np.log10(x)) & np.isfinite(np.log10(y)) & (x > 0) & (y > 0)
    if x_range is not None:
        mask &= (x >= x_range[0]) & (x <= x_range[1])
    if np.sum(mask) < 3:
        return np.nan, np.nan, 0
    lx, ly = np.log10(x[mask]), np.log10(y[mask])
    coeffs = np.polyfit(lx, ly, 1)
    beta = coeffs[0]
    C = 10 ** coeffs[1]
    residuals = ly - np.polyval(coeffs, lx)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return beta, C, r2

def run_sweep(alpha, N, dt, Q=20, n_points=60, n_seeds=5):
    """
    Run sweep for one (alpha, N) over multiple seeds, returning averaged results.
    """
    # Frequency range: ensure we have enough cycles
    f_min = 5.0 / (N * dt)   # at least 5 cycles
    f_max = 0.9 * 0.5 / dt    # Nyquist limit * 0.9
    f_cutoffs = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

    # Storage for all seeds
    all_disc_rmse = []
    all_disc_power = []
    all_cont_rmse = []
    all_cont_power_damp = []
    all_cont_power_bw = []

    for seed in range(base_seed, base_seed + n_seeds):
        signal = generate_colored_noise(alpha, N, dt, seed)

        disc_rmse = []
        disc_power = []
        cont_rmse = []
        cont_power_damp = []
        cont_power_bw = []

        for fc in f_cutoffs:
            fs = 2.0 * fc   # Nyquist sampling

            # Discrete
            rmse_d, p_d = discrete_track(signal, dt, fs)
            disc_rmse.append(rmse_d)
            disc_power.append(p_d)

            # Continuous
            rmse_c, p_damp, p_bw = continuous_track(signal, dt, fc, Q)
            cont_rmse.append(rmse_c)
            cont_power_damp.append(p_damp)
            cont_power_bw.append(p_bw)

        all_disc_rmse.append(disc_rmse)
        all_disc_power.append(disc_power)
        all_cont_rmse.append(cont_rmse)
        all_cont_power_damp.append(cont_power_damp)
        all_cont_power_bw.append(cont_power_bw)

    # Average over seeds
    disc_rmse_avg = np.mean(all_disc_rmse, axis=0)
    disc_power_avg = np.mean(all_disc_power, axis=0)
    cont_rmse_avg = np.mean(all_cont_rmse, axis=0)
    cont_power_damp_avg = np.mean(all_cont_power_damp, axis=0)
    cont_power_bw_avg = np.mean(all_cont_power_bw, axis=0)

    return {
        'f_cutoffs': f_cutoffs,
        'disc_rmse': disc_rmse_avg,
        'disc_power': disc_power_avg,
        'cont_rmse': cont_rmse_avg,
        'cont_power_damp': cont_power_damp_avg,
        'cont_power_bw': cont_power_bw_avg,
    }

# ====================================================================
# Main execution
# ====================================================================
if __name__ == '__main__':

    print("=" * 70)
    print("CBP ENHANCED SIMULATION - ASYMPTOTIC REGIME PROBE")
    print("=" * 70)

    alphas = [1.1, 1.2, 1.3, 1.5, 1.7, 1.9]
    Q = 20
    n_seeds = 10  # number of random realizations for error bars

    configs = [
        (2**16, 1e-4, "N=64K, 6.5s"),
        (2**18, 1e-4, "N=256K, 26s"),
        (2**20, 1e-4, "N=1M, 105s"),
    ]

    all_results = {}

    # Run sweeps
    for N, dt, label in configs:
        print(f"\n{'='*70}")
        print(f"Configuration: {label}")
        print(f"  Samples: {N:,}")
        print(f"  Duration: {N*dt:.1f}s")
        print(f"  Freq range: {1/(N*dt):.4f} Hz to {1/(2*dt):.0f} Hz")
        print(f"  Dynamic range: {np.log10(1/(2*dt)) - np.log10(1/(N*dt)):.1f} decades")
        print(f"  Using {n_seeds} random seeds for averaging")
        print(f"{'='*70}")

        print(f"\n{'α':>5} | {'β_disc (meas)':>14} | {'β_disc (theo)':>14} | "
              f"{'β_cont (meas)':>14} | {'β_cont (theo)':>14} | {'R²_d':>6} | {'R²_c':>6}")
        print("-" * 90)

        results_this_N = {}

        for alpha in alphas:
            t0 = time.time()
            result = run_sweep(alpha, N, dt, Q, n_points=60, n_seeds=n_seeds)
            elapsed = time.time() - t0

            # Determine achievable RMSE range for fitting
            # Use range that is actually reached by both methods
            rmse_all = np.concatenate([result['disc_rmse'], result['cont_rmse']])
            rmse_min = max(np.min(rmse_all), 1e-4)
            rmse_max = min(np.max(rmse_all), 0.5)
            fit_lo = rmse_min
            fit_hi = rmse_max

            beta_d, _, r2_d = fit_power_law(
                result['disc_rmse'], result['disc_power'], (fit_lo, fit_hi))
            beta_c, _, r2_c = fit_power_law(
                result['cont_rmse'], result['cont_power_damp'], (fit_lo, fit_hi))

            theory_d = -2.0 / (alpha - 1) if alpha > 1 else np.nan
            # For continuous, the theoretical scaling from rate-distortion for an optimal filter
            # with no noise is actually -2 (since you can always filter more).
            # But with our damping model, we expect shallower due to losses.
            # We'll keep the original expression for reference, but note it's not used.
            theory_c = -2.0 * (2 - alpha) / (alpha - 1) if alpha > 1 else np.nan

            print(f"{alpha:5.1f} | {beta_d:14.2f} | {theory_d:14.2f} | "
                  f"{beta_c:14.2f} | {theory_c:14.2f} | {r2_d:6.3f} | {r2_c:6.3f}"
                  f"  [{elapsed:.1f}s]")

            results_this_N[alpha] = {
                'result': result,
                'beta_d': beta_d, 'beta_c': beta_c,
                'theory_d': theory_d, 'theory_c': theory_c,
                'r2_d': r2_d, 'r2_c': r2_c,
                'fit_range': (fit_lo, fit_hi),
            }

        all_results[(N, dt)] = results_this_N

    # ====================================================================
    # Convergence table
    # ====================================================================
    print("\n\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS: β_disc vs N")
    print("=" * 70)

    print(f"\n{'α':>5} | ", end="")
    for N, dt, label in configs:
        print(f"{'N=' + str(N//1024) + 'K':>10} | ", end="")
    print(f"{'Theory':>10}")
    print("-" * 60)

    for alpha in alphas:
        print(f"{alpha:5.1f} | ", end="")
        for N, dt, label in configs:
            beta = all_results[(N, dt)][alpha]['beta_d']
            print(f"{beta:10.2f} | ", end="")
        theory = -2.0 / (alpha - 1)
        print(f"{theory:10.2f}")

    print("\nβ_cont (damping model):")
    for alpha in alphas:
        print(f"{alpha:5.1f} | ", end="")
        for N, dt, label in configs:
            beta = all_results[(N, dt)][alpha]['beta_c']
            print(f"{beta:10.2f} | ", end="")
        theory = -2.0 * (2 - alpha) / (alpha - 1) if alpha > 1 else np.nan
        print(f"{theory:10.2f}")

    # ====================================================================
    # Power ratios at biological fidelity (ε ~ 0.001)
    # ====================================================================
    print("\n\n" + "=" * 70)
    print("POWER RATIOS AT TARGET FIDELITY (ε = 0.001, using largest N)")
    print("=" * 70)

    N_best, dt_best = 2**20, 1e-4
    target_eps = 0.001

    for alpha in alphas:
        r = all_results[(N_best, dt_best)][alpha]['result']
        # Find indices where rmse is closest to target
        idx_d = np.argmin(np.abs(r['disc_rmse'] - target_eps))
        idx_c = np.argmin(np.abs(r['cont_rmse'] - target_eps))

        rmse_d_actual = r['disc_rmse'][idx_d]
        rmse_c_actual = r['cont_rmse'][idx_c]
        if rmse_d_actual < 0.01 and rmse_c_actual < 0.01:
            ratio = r['disc_power'][idx_d] / r['cont_power_damp'][idx_c]
            print(f"α = {alpha:.1f}:  P_disc/P_cont = {ratio:.2f}x  "
                  f"(disc ε={rmse_d_actual:.4f}, cont ε={rmse_c_actual:.4f})")
        else:
            print(f"α = {alpha:.1f}:  target ε not reached (best disc ε={rmse_d_actual:.4f}, "
                  f"cont ε={rmse_c_actual:.4f})")

    # ====================================================================
    # Generate figures
    # ====================================================================

    # Figure 1: Convergence plot (scaling for each α and N)
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors_N = {2**16: '#1f77b4', 2**18: '#ff7f0e', 2**20: '#2ca02c'}

    for i, alpha in enumerate(alphas):
        ax = axes[i // 3, i % 3]

        for N, dt, label in configs:
            r = all_results[(N, dt)][alpha]['result']
            mask_d = r['disc_rmse'] > 0
            ax.loglog(r['disc_rmse'][mask_d], r['disc_power'][mask_d],
                     'o-', color=colors_N[N], markersize=3, alpha=0.7,
                     label=f'Disc {label}')
            mask_c = r['cont_rmse'] > 0
            ax.loglog(r['cont_rmse'][mask_c], r['cont_power_damp'][mask_c],
                     's--', color=colors_N[N], markersize=3, alpha=0.4,
                     label=f'Cont {label}')

        # Add theoretical slope line for discrete
        theory_d = -2.0 / (alpha - 1)
        x_ref = np.array([0.01, 0.3])
        y_ref_d = 1e3 * (x_ref / 0.1) ** theory_d
        ax.loglog(x_ref, y_ref_d, 'k:', linewidth=2, alpha=0.5,
                 label=f'Theory β={theory_d:.1f}')

        ax.set_title(f'α = {alpha}', fontsize=13, fontweight='bold')
        ax.set_xlabel('RMSE (ε)')
        ax.set_ylabel('Power')
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')

    fig1.suptitle('Scaling Convergence Across Signal Length N (averaged over 10 seeds)',
                  fontsize=15, fontweight='bold')
    fig1.tight_layout()
    fig1.savefig('fig_convergence.png', dpi=200, bbox_inches='tight')
    print("\nSaved fig_convergence.png")

    # Figure 2: Power ratio vs RMSE for largest N
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    colors_alpha = {1.1: '#d62728', 1.2: '#e377c2', 1.3: '#ff7f0e',
                    1.5: '#2ca02c', 1.7: '#1f77b4', 1.9: '#9467bd'}

    for alpha in alphas:
        r = all_results[(N_best, dt_best)][alpha]['result']
        # Compute ratio at matching RMSE via interpolation
        # Use log-log interpolation for smoother curve
        log_disc_rmse = np.log10(r['disc_rmse'])
        log_disc_power = np.log10(r['disc_power'])
        log_cont_rmse = np.log10(r['cont_rmse'])
        log_cont_power = np.log10(r['cont_power_damp'])

        # Interpolate continuous power at discrete RMSE points
        log_cont_power_interp = np.interp(log_disc_rmse, log_cont_rmse, log_cont_power,
                                           left=np.nan, right=np.nan)
        mask = np.isfinite(log_cont_power_interp) & (r['disc_rmse'] > 0)
        if np.sum(mask) > 2:
            ratio = 10**(log_disc_power[mask] - log_cont_power_interp[mask])
            ax2.semilogy(log_disc_rmse[mask], ratio, 'o-', color=colors_alpha[alpha],
                         markersize=4, linewidth=1.5, label=f'α={alpha}')

    ax2.set_xlabel('log₁₀(RMSE)', fontsize=13)
    ax2.set_ylabel('Power Ratio: Discrete / Continuous', fontsize=13)
    ax2.set_title(f'Metabolic Penalty Ratio vs Fidelity (N={N_best}, averaged over {n_seeds} seeds)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=7, color='gray', linestyle='--', alpha=0.7)
    ax2.annotate('7× (paper claim for α=1.5)', xy=(-1.5, 7), fontsize=10, color='gray')
    ax2.axhline(y=1, color='black', linestyle='-', alpha=0.3)
    fig2.tight_layout()
    fig2.savefig('fig_ratio.png', dpi=200, bbox_inches='tight')
    print("Saved fig_ratio.png")

    # ====================================================================
    # Save results to JSON (generic path)
    # ====================================================================
    # Create a serializable summary
    summary = {}
    for (N, dt), res in all_results.items():
        key = f"N={N}_dt={dt}"
        summary[key] = {}
        for alpha, data in res.items():
            summary[key][f"alpha={alpha}"] = {
                'beta_disc_meas': data['beta_d'],
                'beta_disc_theory': data['theory_d'],
                'beta_cont_meas': data['beta_c'],
                'beta_cont_theory': data['theory_c'],
                'r2_disc': data['r2_d'],
                'r2_cont': data['r2_c'],
                'fit_range_lo': data['fit_range'][0],
                'fit_range_hi': data['fit_range'][1],
            }

    # Add power ratios
    summary['power_ratios_at_1e-3'] = {}
    for alpha in alphas:
        r = all_results[(N_best, dt_best)][alpha]['result']
        idx_d = np.argmin(np.abs(r['disc_rmse'] - target_eps))
        idx_c = np.argmin(np.abs(r['cont_rmse'] - target_eps))
        rmse_d_actual = r['disc_rmse'][idx_d]
        rmse_c_actual = r['cont_rmse'][idx_c]
        if rmse_d_actual < 0.01 and rmse_c_actual < 0.01:
            ratio = r['disc_power'][idx_d] / r['cont_power_damp'][idx_c]
            summary['power_ratios_at_1e-3'][f"alpha={alpha}"] = {
                'ratio': ratio,
                'disc_rmse_achieved': rmse_d_actual,
                'cont_rmse_achieved': rmse_c_actual,
            }
        else:
            summary['power_ratios_at_1e-3'][f"alpha={alpha}"] = {
                'ratio': None,
                'disc_rmse_achieved': rmse_d_actual,
                'cont_rmse_achieved': rmse_c_actual,
            }

    # Save to a file in the current directory
    with open('simulation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved simulation_results.json")

    print("\n\nDone. Check the generated PNG and JSON files.")