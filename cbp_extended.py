"""
CBP Simulation - Extended Range
================================
Tests whether theoretical scaling exponents emerge at larger N
(more frequency resolution, wider dynamic range).

Key question: Does P_discrete ~ ε^(-2/(α-1)) hold in the asymptotic regime?
"""

import numpy as np
from scipy import signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

np.random.seed(42)


def generate_colored_noise(alpha, N, dt, seed=42):
    rng = np.random.RandomState(seed)
    freqs = np.fft.rfftfreq(N, d=dt)
    psd = np.zeros_like(freqs)
    psd[1:] = freqs[1:] ** (-alpha)
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    spectrum = np.sqrt(psd) * np.exp(1j * phases)
    spectrum[0] = 0
    x = np.fft.irfft(spectrum, n=N)
    return x / np.std(x)


def discrete_track(signal, dt, fs):
    N = len(signal)
    t = np.arange(N) * dt
    interval = max(1, int(round(1.0 / (fs * dt))))
    idx = np.arange(0, N, interval)
    if len(idx) < 2:
        return 1.0
    reconstructed = np.interp(t, t[idx], signal[idx])
    return np.sqrt(np.mean((signal - reconstructed) ** 2))


def continuous_track(signal, dt, f_c, Q=20):
    N = len(signal)
    sig_fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=dt)
    H = np.zeros(len(freqs))
    mask = freqs > 0
    H[mask] = 1.0 / np.sqrt(1.0 + (freqs[mask] / f_c) ** 8)
    filtered = np.fft.irfft(sig_fft * H, n=N)
    rmse = np.sqrt(np.mean((signal - filtered) ** 2))
    
    # Power models
    psd = np.abs(sig_fft) ** 2 / N
    power_damping = np.sum(freqs * psd * (H ** 2) / Q) * (freqs[1] - freqs[0])
    power_bw = f_c
    
    return rmse, power_damping, power_bw


def fit_power_law(x, y, x_range=None):
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


def run_sweep(alpha, N, dt, Q=20, n_points=60):
    """Run power-vs-error sweep with more points and wider range."""
    signal = generate_colored_noise(alpha, N, dt)
    f_max = 1.0 / (2 * dt)
    f_min = 10.0 / (N * dt)  # At least 10 cycles
    
    f_cutoffs = np.logspace(np.log10(f_min), np.log10(f_max * 0.9), n_points)
    
    disc_rmse = []
    disc_power = []
    cont_rmse = []
    cont_power_damp = []
    cont_power_bw = []
    
    for fc in f_cutoffs:
        fs = 2.0 * fc
        rmse_d = discrete_track(signal, dt, fs)
        disc_rmse.append(rmse_d)
        bits = max(1, np.log2(1.0 / max(rmse_d, 1e-15)))
        disc_power.append(fs * bits)
        
        rmse_c, p_damp, p_bw = continuous_track(signal, dt, fc, Q)
        cont_rmse.append(rmse_c)
        cont_power_damp.append(p_damp)
        cont_power_bw.append(p_bw)
    
    return {
        'f_cutoffs': np.array(f_cutoffs),
        'disc_rmse': np.array(disc_rmse),
        'disc_power': np.array(disc_power),
        'cont_rmse': np.array(cont_rmse),
        'cont_power_damp': np.array(cont_power_damp),
        'cont_power_bw': np.array(cont_power_bw),
    }


if __name__ == '__main__':
    
    print("=" * 70)
    print("CBP EXTENDED SIMULATION - ASYMPTOTIC REGIME PROBE")
    print("=" * 70)
    
    alphas = [1.1, 1.2, 1.3, 1.5, 1.7, 1.9]
    Q = 20
    
    # Test multiple N values to see convergence
    configs = [
        (2**16, 1e-4, "N=64K, 6.5s"),
        (2**18, 1e-4, "N=256K, 26s"),
        (2**20, 1e-4, "N=1M, 105s"),
    ]
    
    all_results = {}
    
    for N, dt, label in configs:
        print(f"\n{'='*70}")
        print(f"Configuration: {label}")
        print(f"  Samples: {N:,}")
        print(f"  Duration: {N*dt:.1f}s")
        print(f"  Freq range: {1/(N*dt):.4f} Hz to {1/(2*dt):.0f} Hz")
        print(f"  Dynamic range: {np.log10(1/(2*dt)) - np.log10(1/(N*dt)):.1f} decades")
        print(f"{'='*70}")
        
        print(f"\n{'α':>5} | {'β_disc (meas)':>14} | {'β_disc (theo)':>14} | "
              f"{'β_cont (meas)':>14} | {'β_cont (theo)':>14} | {'R²_d':>6} | {'R²_c':>6}")
        print("-" * 90)
        
        results_this_N = {}
        
        for alpha in alphas:
            t0 = time.time()
            result = run_sweep(alpha, N, dt, Q, n_points=60)
            elapsed = time.time() - t0
            
            # Fit in middle range to avoid edge effects
            # Use a range that should be well within the asymptotic regime
            fit_lo = 0.003
            fit_hi = 0.3
            
            beta_d, _, r2_d = fit_power_law(
                result['disc_rmse'], result['disc_power'], (fit_lo, fit_hi))
            beta_c, _, r2_c = fit_power_law(
                result['cont_rmse'], result['cont_power_damp'], (fit_lo, fit_hi))
            
            theory_d = -2.0 / (alpha - 1)
            theory_c = -2.0 * (2 - alpha) / (alpha - 1)
            
            print(f"{alpha:5.1f} | {beta_d:14.2f} | {theory_d:14.2f} | "
                  f"{beta_c:14.2f} | {theory_c:14.2f} | {r2_d:6.3f} | {r2_c:6.3f}"
                  f"  [{elapsed:.1f}s]")
            
            results_this_N[alpha] = {
                'result': result,
                'beta_d': beta_d, 'beta_c': beta_c,
                'theory_d': theory_d, 'theory_c': theory_c,
                'r2_d': r2_d, 'r2_c': r2_c,
            }
        
        all_results[(N, dt)] = results_this_N
    
    # ================================================================
    # ANALYSIS: Does increasing N recover theoretical exponents?
    # ================================================================
    
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
    
    print(f"\n{'α':>5} | ", end="")
    for N, dt, label in configs:
        print(f"{'N=' + str(N//1024) + 'K':>10} | ", end="")
    print(f"{'Theory':>10}")
    print("-" * 60)
    
    print("\nβ_cont (damping model):")
    for alpha in alphas:
        print(f"{alpha:5.1f} | ", end="")
        for N, dt, label in configs:
            beta = all_results[(N, dt)][alpha]['beta_c']
            print(f"{beta:10.2f} | ", end="")
        theory = -2.0 * (2 - alpha) / (alpha - 1)
        print(f"{theory:10.2f}")
    
    # ================================================================
    # FIGURE: Convergence plot
    # ================================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors_N = {2**16: '#1f77b4', 2**18: '#ff7f0e', 2**20: '#2ca02c'}
    
    for i, alpha in enumerate(alphas):
        ax = axes[i // 3, i % 3]
        
        for N, dt, label in configs:
            r = all_results[(N, dt)][alpha]['result']
            mask = (r['disc_rmse'] > 0.001) & (r['disc_rmse'] < 0.5)
            
            ax.loglog(r['disc_rmse'][mask], r['disc_power'][mask],
                     'o-', color=colors_N[N], markersize=3, alpha=0.7,
                     label=f'Disc {label}')
            
            mask_c = (r['cont_rmse'] > 0.001) & (r['cont_rmse'] < 0.5)
            ax.loglog(r['cont_rmse'][mask_c], r['cont_power_damp'][mask_c],
                     's--', color=colors_N[N], markersize=3, alpha=0.4,
                     label=f'Cont {label}')
        
        # Add theoretical slope lines
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
    
    fig.suptitle('Scaling Convergence Across Signal Length N',
                fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig('/home/claude/fig_convergence.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("\nSaved fig_convergence.png")
    
    # ================================================================
    # FIGURE: Power ratio (discrete/continuous) vs ε for each α
    # ================================================================
    
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    colors_alpha = {1.1: '#d62728', 1.2: '#e377c2', 1.3: '#ff7f0e', 
                    1.5: '#2ca02c', 1.7: '#1f77b4', 1.9: '#9467bd'}
    
    # Use largest N for best resolution
    N_best = 2**20
    dt_best = 1e-4
    
    for alpha in alphas:
        r = all_results[(N_best, dt_best)][alpha]['result']
        
        # Compute ratio at matching RMSE values
        # Interpolate continuous power at discrete RMSE values
        mask = (r['disc_rmse'] > 0.003) & (r['disc_rmse'] < 0.4)
        
        # For each discrete RMSE, find closest continuous RMSE
        ratios = []
        rmses = []
        for j in range(len(r['disc_rmse'])):
            if not mask[j]:
                continue
            rmse_d = r['disc_rmse'][j]
            # Find continuous power at same RMSE
            idx_c = np.argmin(np.abs(r['cont_rmse'] - rmse_d))
            if abs(r['cont_rmse'][idx_c] - rmse_d) / rmse_d < 0.3:
                ratio = r['disc_power'][j] / r['cont_power_damp'][idx_c]
                if ratio > 0 and np.isfinite(ratio):
                    ratios.append(ratio)
                    rmses.append(rmse_d)
        
        if len(rmses) > 2:
            ax2.semilogy(np.log10(rmses), ratios, 'o-', color=colors_alpha[alpha],
                        markersize=5, linewidth=2, label=f'α={alpha}')
    
    ax2.set_xlabel('log₁₀(RMSE)', fontsize=13)
    ax2.set_ylabel('Power Ratio: Discrete / Continuous', fontsize=13)
    ax2.set_title('Metabolic Penalty Ratio vs Fidelity (N=1M)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=7, color='gray', linestyle='--', alpha=0.7)
    ax2.annotate('7× (paper claim for α=1.5)', xy=(-1.5, 7),
                fontsize=10, color='gray')
    ax2.axhline(y=1, color='black', linestyle='-', alpha=0.3)
    fig2.tight_layout()
    fig2.savefig('/home/claude/fig_ratio.png', dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print("Saved fig_ratio.png")
    
    # ================================================================
    # CRITICAL: What does the simulation ACTUALLY show at ε = 10^-3?
    # ================================================================
    
    print("\n\n" + "=" * 70)
    print("POWER RATIOS AT BIOLOGICAL FIDELITY (N=1M)")
    print("=" * 70)
    
    for alpha in alphas:
        r = all_results[(N_best, dt_best)][alpha]['result']
        
        # Find closest to ε = 0.001
        idx_d = np.argmin(np.abs(r['disc_rmse'] - 0.001))
        idx_c = np.argmin(np.abs(r['cont_rmse'] - 0.001))
        
        actual_d = r['disc_rmse'][idx_d]
        actual_c = r['cont_rmse'][idx_c]
        
        if actual_d > 0 and actual_d < 0.01:
            p_d = r['disc_power'][idx_d]
            p_c_damp = r['cont_power_damp'][idx_c]
            p_c_bw = r['cont_power_bw'][idx_c]
            
            ratio_damp = p_d / p_c_damp if p_c_damp > 0 else np.inf
            ratio_bw = p_d / p_c_bw if p_c_bw > 0 else np.inf
            
            print(f"  α={alpha}: actual_ε_d={actual_d:.4f}, actual_ε_c={actual_c:.4f}")
            print(f"         P_d/P_c(damp)={ratio_damp:.1f}x, P_d/P_c(bw)={ratio_bw:.1f}x")
        else:
            print(f"  α={alpha}: ε=10⁻³ not reached (closest: {actual_d:.4f})")
    
    # ================================================================
    # BANDWIDTH ANALYSIS: required bandwidth to achieve ε
    # ================================================================
    
    print("\n\n" + "=" * 70)
    print("BANDWIDTH ANALYSIS: f_c required to achieve target RMSE")
    print("=" * 70)
    
    targets = [0.1, 0.01, 0.001]
    
    for alpha in alphas:
        r = all_results[(N_best, dt_best)][alpha]['result']
        print(f"\n  α = {alpha}:")
        for target in targets:
            # Discrete: find fs needed
            idx_d = np.argmin(np.abs(r['disc_rmse'] - target))
            # Continuous: find fc needed
            idx_c = np.argmin(np.abs(r['cont_rmse'] - target))
            
            if r['disc_rmse'][idx_d] < target * 2:
                fc_d = r['f_cutoffs'][idx_d]
                fc_c = r['f_cutoffs'][idx_c]
                ratio = fc_d / fc_c if fc_c > 0 else np.inf
                print(f"    ε={target}: fc_disc={fc_d:.0f}Hz, fc_cont={fc_c:.0f}Hz, "
                      f"ratio={ratio:.1f}x")
            else:
                print(f"    ε={target}: not achieved")
    
    print("\n\nDone.")
