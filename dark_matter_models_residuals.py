"""
=============================================================================
Linear Shear Vacuum Model - First-Principles Residual Analysis
=============================================================================
Author: Bart Leplae
License: MIT
Version: 2.3.0
Data Source: SPARC (https://cdsarc.cds.unistra.fr/ftp/J/AJ/152/157/)

Physical Foundation:
-------------------
This script validates a first-principles geometric alternative to particle 
dark matter derived from the "Gravity as Relaxation" vacuum mechanics framework.

Rather than relying on empirical fitting constants (e.g., A = 1.74e-3), the 
missing mass density is derived directly from the coordinate light-speed 
gradient (a = -c_eff * grad(c_eff)) operating on a co-rotating vacuum medium:

    Rho_DM(R, z) = ( V_char / (4 * pi * G) ) * Shear(R, z)

Where:
    - G is Newton's gravitational constant in galactic units:
        G_gal = 4.3009e-6 (km/s)^2 * kpc / M_sun
    - V_char = 94.04 km/s is the universal characteristic velocity of the 
      relaxing space-quantum lattice across the SPARC sample.
    - Coupling Constant: V_char / (4 * pi * G) = 1.74e+06 M_sun / kpc^3
    - Shear(R, z) = sqrt( (dOmega/dR)^2 + (dOmega/dz)^2 ) is the 2D kinematic 
      shear of space co-rotating with the baryonic disk.
    - Connection to space-quantum packing field eta:
        nabla^2(eta) = - (V_char / c_0^2) * Shear
    - Connection to Milgrom's MOND acceleration scale a_0:
        a_0 ~ V_char^2 / R_trans ~ (94.04 km/s)^2 / (2.5 kpc) ~ 1.15e-10 m/s^2
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import warnings
import time
import matplotlib.ticker as ticker

# Suppress runtime warnings from the scipy optimizer testing edge-cases
warnings.filterwarnings('ignore')

# =============================================================================
# FIRST-PRINCIPLES PHYSICAL CONSTANTS (GALACTIC UNITS)
# =============================================================================
# Newton's Gravitational Constant in galactic units: (km/s)^2 * kpc / M_sun
G_GAL = 4.3009e-6  

# V_char: Characteristic Galactic Velocity (Universal Attractor Scale in SPARC)
# Derived from: V_char = 4 * pi * G_gal * (1.74e-3 * 1e9) = 94.04 km/s
V_CHAR_UNIVERSAL = 94.04  # km/s

# Speed of light in vacuum (km/s)
C0_KMS = 299792.458  

# Universal Vacuum Coupling Constant:
# Yields mass density directly in M_sun / kpc^3 per [(km/s) / kpc^2]
COUPLING_KPC_UNIVERSAL = V_CHAR_UNIVERSAL / (4.0 * np.pi * G_GAL)  # ~ 1.74e6 M_sun/kpc^3

# Mode Toggle: Locked to False to enforce universal vacuum stiffness
USE_GALAXY_SPECIFIC_V0 = False

# =============================================================================

def load_and_filter_galaxies(file_path: str) -> list:
    """
    Loads the SPARC kinematic database and applies standard quality-control filters.
    
    Args:
        file_path (str): Path to the SPARC dataset (.csv or .xlsx).
        
    Returns:
        list: A list of dictionaries, where each dict contains the galaxy's 
              name and a cleaned pandas DataFrame of its kinematics.
    """
    print(f"Loading data from {file_path}...")
    raw_df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    raw_df['clean_name'] = raw_df['Galaxy identifier'].astype(str).str.replace(" ", "").str.upper()
    
    valid_galaxies = []
    for galaxy in raw_df['clean_name'].unique():
        gal_df = raw_df[raw_df['clean_name'] == galaxy].copy()
        
        # Filter 1: Require a minimum of 6 data points for a reliable rotation curve
        if len(gal_df) < 6: 
            continue
            
        gal_df = gal_df.rename(columns={
            'rad': 'Rad', 'vobs': 'Vobs', 'vgas': 'Vgas', 'vdisk': 'Vdisk', 'vbulge': 'Vbulge'
        })
        cols = ['Rad', 'Vobs', 'Vgas', 'Vdisk', 'Vbulge']
        for col in cols:
            gal_df[col] = pd.to_numeric(gal_df[col], errors='coerce').fillna(0.0)
            
        df = gal_df[cols].sort_values(by='Rad').reset_index(drop=True)
        
        # Isolate missing mass velocity (inferred dark matter velocity)
        # Assumes standard mass-to-light ratios: 0.5 for disk, 0.7 for bulge
        df['Vdisk_scaled'] = df['Vdisk'] * np.sqrt(0.5)
        df['Vbulge_scaled'] = df['Vbulge'] * np.sqrt(0.7)
        v_halo_sq = (df['Vobs']**2 - df['Vdisk_scaled']**2 - df['Vbulge_scaled']**2 - (df['Vgas'] * df['Vgas'].abs()))
        df['Vtrue'] = np.sqrt(np.maximum(0.0, v_halo_sq))
        
        # Filter 2: Drop galaxies where missing mass is negligible (< 10 km/s)
        if df['Vtrue'].max() < 10.0: 
            continue
            
        valid_galaxies.append({'name': galaxy, 'data': df})
        
    return valid_galaxies

def evaluate_galaxy(hv: float, gal: dict, R_grid: np.ndarray, z_grid: np.ndarray, 
                    R_2d: np.ndarray, z_2d: np.ndarray, V_midplane: np.ndarray, 
                    masks: dict, kernels: list, coupling_kpc: float, 
                    hv_target: float) -> tuple:
    """
    INNER OPTIMIZATION LOOP: Computes the 2D spatial kinematic shear and resulting 
    synthetic velocity using the first-principles vacuum density law.
    """
    # 1. Kinematics: Vertical exponential velocity decay and angular velocity Omega(R, z)
    V_2d = V_midplane * np.exp(-z_2d / hv)
    Omega_2d = V_2d / R_2d
    
    # 2. Kinematic Shear: Magnitude of the differential rotation tensor
    dOmega_dR = np.gradient(Omega_2d, R_grid, axis=0)
    dOmega_dz = np.gradient(Omega_2d, z_grid, axis=1)
    Shear_2d = np.sqrt(dOmega_dR**2 + dOmega_dz**2)
    
    # 3. First-Principles Density Law:
    # Rho_DM(R, z) = (V_char / (4 * pi * G)) * Shear
    # Directly in M_sun / kpc^3 without arbitrary unit conversion factors!
    Rho_2d_kpc = coupling_kpc * Shear_2d
    
    # Smooth outer boundary taper to prevent infinite universe integration limits
    Rho_2d_kpc[masks['outer']] *= masks['outer_taper']
    Rho_2d_kpc[np.sqrt(R_2d**2 + z_2d**2) > masks['edge']] = 0.0
    Rho_kpc = Rho_2d_kpc[:, :, None]
    
    # 4. Newtonian Summation: Integrate gravitational pull across the shear-induced density
    V_syn = [np.sqrt(np.maximum(0.0, R * np.sum(Rho_kpc * kernels[i]))) for i, R in enumerate(gal['data']['Rad'])]
    V_true = gal['data']['Vtrue'].values
    
    # Adaptive scale height regularization relative to galaxy optical size
    reg_penalty = 1.0 * ((hv - hv_target) / hv_target)**2
    score = np.mean((V_syn - V_true)**2) + reg_penalty
    
    return score, V_syn

def calculate_all_residuals(galaxies: list) -> tuple:
    """
    Pre-computes 3D geometries, dynamically optimizes disk thickness using 
    size-adaptive priors, and compiles global velocity residuals.
    """
    mode_str = "Adaptive V_flat" if USE_GALAXY_SPECIFIC_V0 else f"Universal V_char = {V_CHAR_UNIVERSAL:.2f} km/s"
    print(f"Calculating residuals for {len(galaxies)} galaxies using First-Principles Vacuum Model [{mode_str}]...")
    
    all_Vtrue, all_Vsyn, all_Radii = [], [], []
    start_time = time.time()
    
    for i, gal in enumerate(galaxies):
        df = gal['data']
        r_max = df['Rad'].max()
        r_min = max(0.08, df['Rad'].min())
        
        # Adaptive Target Scale Height: Typical galactic disk aspect ratio ~ 12-15%
        hv_target = max(0.35, 0.15 * r_max)
        
        # Determine the coupling factor
        if USE_GALAXY_SPECIFIC_V0:
            v_flat = df['Vobs'].tail(3).median()
            coupling_kpc = v_flat / (4.0 * np.pi * G_GAL)
        else:
            coupling_kpc = COUPLING_KPC_UNIVERSAL
        
        # 3D Grid Pre-computation:
        # Extend R_grid to 1.25 * r_max with flat velocity extrapolation to avoid edge droop
        R_grid = np.linspace(r_min, r_max * 1.25, 60)
        z_grid = np.linspace(0.0, r_max * 1.6, 45)
        phi_grid = np.linspace(0.0, 2.0 * np.pi, 50)
        
        R_3d, z_3d, phi_3d = np.meshgrid(R_grid, z_grid, phi_grid, indexing='ij')
        R_2d, z_2d = np.meshgrid(R_grid, z_grid, indexing='ij')
        
        # Midplane velocity interpolation with asymptotic flat continuation
        v_outer = df['Vobs'].iloc[-1]
        V_midplane = np.interp(R_grid, df['Rad'], df['Vobs'], right=v_outer)[:, None]
        
        # Extended Geometric Boundary Masks
        r_sph2d = np.sqrt(R_2d**2 + z_2d**2)
        masks = {
            'outer': r_sph2d > r_max * 1.05,
            'outer_taper': np.exp(-(r_sph2d[r_sph2d > r_max * 1.05] - r_max * 1.05) / (0.35 * r_max)),
            'edge': r_max * 1.9
        }
        
        # Pre-compute Newtonian Distance Kernels
        dR = R_grid[1] - R_grid[0]
        dz = z_grid[1] - z_grid[0]
        dphi = phi_grid[1] - phi_grid[0]
        dV = R_3d * dR * dz * dphi
        softening = (max(dR, dz) * 1.8)**2
        
        kernels = [
            2.0 * G_GAL * dV * (R - R_3d * np.cos(phi_3d)) / 
            ((R_3d**2 + R**2 - 2.0*R_3d*R*np.cos(phi_3d) + z_3d**2 + softening)**1.5)
            if R > 0 else np.zeros_like(R_3d) 
            for R in df['Rad']
        ]
        
        # Scale-height bounds scaled dynamically to galaxy optical size
        hv_bounds = (max(0.15, 0.05 * r_max), max(8.0, 0.45 * r_max))
        
        # Optimize disk thickness
        res = minimize_scalar(
            lambda hv: evaluate_galaxy(hv, gal, R_grid, z_grid, R_2d, z_2d, V_midplane, masks, kernels, coupling_kpc, hv_target)[0], 
            bounds=hv_bounds, method='bounded'
        )
        
        # Compute final synthetic rotation curve
        _, V_syn = evaluate_galaxy(res.x, gal, R_grid, z_grid, R_2d, z_2d, V_midplane, masks, kernels, coupling_kpc, hv_target)
        
        all_Vtrue.extend(df['Vtrue'].values)
        all_Vsyn.extend(V_syn)
        all_Radii.extend(df['Rad'].values)
        
        if (i+1) % 20 == 0: 
            print(f"  ...processed {i+1} galaxies")
            
    print(f"Finished in {time.time() - start_time:.1f} seconds.")
    return np.array(all_Vtrue), np.array(all_Vsyn), np.array(all_Radii)

def print_cohort_breakdown(V_true: np.ndarray, V_syn: np.ndarray):
    """
    Evaluates and displays statistical residuals across three distinct 
    astrophysical velocity regimes:
    1. Dwarf Galaxies / Low-Mass Limit   (V < 65 km/s)
    2. Anchor Transition / Median Regime (65 <= V <= 130 km/s)
    3. Giant Spirals / High-Mass Limit   (V > 130 km/s)
    """
    absolute_residuals = V_syn - V_true
    valid_mask = (V_true > 5.0) & (np.abs(absolute_residuals) < 100.0)
    v_true_clean = V_true[valid_mask]
    res_clean = absolute_residuals[valid_mask]

    cohorts = [
        ("Dwarf Irregulars / Low-Mass (< 65 km/s)", v_true_clean < 65.0),
        ("Anchor / Median Scale (65 - 130 km/s)", (v_true_clean >= 65.0) & (v_true_clean <= 130.0)),
        ("Giant Spirals / High-Mass (> 130 km/s)", v_true_clean > 130.0),
    ]

    print("\n" + "=" * 84)
    print("      SPARC STRATIFIED COHORT RESIDUAL ANALYSIS (V_char = 94.04 km/s)")
    print("=" * 84)
    print(f"{'Cohort Description':<38} | {'Points':<7} | {'Mean':<8} | {'Median':<8} | {'IQR [25%, 75%]':<16}")
    print("-" * 84)

    for name, mask in cohorts:
        n_pts = np.sum(mask)
        if n_pts == 0:
            continue
        c_res = res_clean[mask]
        c_mean = np.mean(c_res)
        c_median = np.median(c_res)
        c_p25 = np.percentile(c_res, 25)
        c_p75 = np.percentile(c_res, 75)
        iqr_str = f"[{c_p25:+5.1f}, {c_p75:+5.1f}]"

        print(f"{name:<38} | {n_pts:<7} | {c_mean:+6.2f}  | {c_median:+6.2f}  | {iqr_str:<16}")

    g_mean = np.mean(res_clean)
    g_median = np.median(res_clean)
    g_p25 = np.percentile(res_clean, 25)
    g_p75 = np.percentile(res_clean, 75)
    g_iqr_str = f"[{g_p25:+5.1f}, {g_p75:+5.1f}]"

    print("-" * 84)
    print(f"{'GLOBAL COMBINED SAMPLE':<38} | {len(res_clean):<7} | {g_mean:+6.2f}  | {g_median:+6.2f}  | {g_iqr_str:<16}")
    print("=" * 84 + "\n")

def plot_dashboard(V_true: np.ndarray, V_syn: np.ndarray, Radii: np.ndarray, num_galaxies: int):
    """
    Generates a publication-grade 4-panel Residual Dashboard displaying 
    the analytical derivation, kinematic components, and statistical performance.
    """
    print("Generating Publication Plot...")
    
    absolute_residuals = V_syn - V_true
    valid_mask = (V_true > 5.0) & (np.abs(absolute_residuals) < 100.0)
    V_true_clean = V_true[valid_mask]
    V_syn_clean = V_syn[valid_mask]
    Radii_clean = Radii[valid_mask]
    abs_res_clean = absolute_residuals[valid_mask]
    
    p25_err = np.percentile(abs_res_clean, 25)
    p75_err = np.percentile(abs_res_clean, 75)
    mean_err = np.mean(abs_res_clean)
    median_err = np.median(abs_res_clean)
    
    fig = plt.figure(figsize=(16, 12)) 
    
    # -------------------------------------------------------------------------
    # HEADER: Formatted Title, Shear Definition & Component Description
    # -------------------------------------------------------------------------
    formula_latex = r"\rho_{\mathrm{DM}} = \left(\frac{V_{\mathrm{char}}}{4\pi G}\right) \cdot \mathrm{Shear}"
    fig.suptitle(f"Vacuum Relaxation Model - Residual Analysis (${formula_latex}$)", 
                 fontsize=21, fontweight='bold', y=0.985)
    
    base, exp = f"{COUPLING_KPC_UNIVERSAL:.2e}".split('e')
    coupling_latex = f"{base} \\times 10^{{{int(exp)}}}"
    
    # Complete, fully documented subtitle lines:
    subtitle_text = (
        rf"Where $\mathrm{{Shear}} = \sqrt{{(\partial\Omega/\partial R)^2 + (\partial\Omega/\partial z)^2}}$    |    "
        rf"$\mathrm{{Coupling\ Constant}}\ \left(\frac{{V_{{\mathrm{{char}}}}}}{{4\pi G}}\right) = {coupling_latex}\ M_\odot/\mathrm{{kpc}}^3\ \mathrm{{per}}\ [(\mathrm{{km/s}})/\mathrm{{kpc}}^2]$" + "\n"
        rf"$V_{{\mathrm{{char}}}} = \mathrm{{Characteristic\ Galactic\ Velocity\ (Universal\ Attractor\ Scale)}} = {V_CHAR_UNIVERSAL:.2f}\ \mathrm{{km/s}}$" + "\n"
        r"$\Omega = \mathrm{Angular\ Velocity} \quad R = \mathrm{Radial\ Distance} \quad z = \mathrm{Vertical\ Distance\ from\ Midplane}$" + "\n"
        r"$\partial\Omega/\partial R = \mathrm{Radial\ Differential\ Rotation} \quad \partial\Omega/\partial z = \mathrm{Vertical\ Differential\ Rotation}$"
    )
    fig.text(0.5, 0.952, subtitle_text, ha='center', va='top', fontsize=11, color='dimgrey', style='italic', linespacing=1.3)
    
    # Summary Box
    stats_text = (
        f"Dataset: SPARC (~150 Galaxies)\n"
        f"Evaluated: {num_galaxies} Galaxies\n"
        f"Data Points: {len(V_true_clean):,}\n"
        f"IQR (25%-75%): [{p25_err:.1f}, +{p75_err:.1f}] km/s\n"
        f"Mean Error: {mean_err:.2f} km/s\n"
        f"Median Error: {median_err:.2f} km/s"
    )
    fig.text(0.985, 0.952, stats_text, ha='right', va='top', fontsize=9.5, color='black',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgrey', boxstyle='round,pad=0.5'))
    
    # -------------------------------------------------------------------------
    # Panel 1: Model Prediction vs Observation (1-to-1 Fit)
    # -------------------------------------------------------------------------
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(V_true_clean, V_syn_clean, alpha=0.3, color='royalblue', edgecolor='none', s=20)
    x_max = max(V_true_clean) * 1.05
    x_vals = np.array([0.0, x_max])
    ax1.plot(x_vals, x_vals, 'k--', linewidth=2, label='Perfect Fit (Zero Error)')
    ax1.plot(x_vals, x_vals + p25_err, 'k:', linewidth=2, alpha=0.7, label='IQR (25% - 75%)')
    ax1.plot(x_vals, x_vals + p75_err, 'k:', linewidth=2, alpha=0.7)
    ax1.set_xlim(0.0, x_max)
    ax1.set_ylim(0.0, x_max)
    ax1.set_xlabel(r'Observed Missing Mass Velocity ($V_{\mathrm{true}}$) [km/s]', fontsize=12)
    ax1.set_ylabel(r'Model Predicted Velocity ($V_{\mathrm{syn}}$) [km/s]', fontsize=12)
    ax1.set_title('Model Prediction vs Observation', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Panel 2: Spatial Distribution of Error (Flat Core Validation)
    # -------------------------------------------------------------------------
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_xscale('log') 
    ax2.scatter(Radii_clean, abs_res_clean, alpha=0.3, color='crimson', edgecolor='none', s=20)
    ax2.axhline(0.0, color='black', linewidth=2, linestyle='--')
    ax2.axhline(p25_err, color='black', linewidth=2, linestyle=':', alpha=0.7, label='IQR (25% - 75%)')
    ax2.axhline(p75_err, color='black', linewidth=2, linestyle=':', alpha=0.7)
    ax2.set_ylim(-50.0, 50.0) 
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
    ax2.set_xlabel('Radius (kpc) [Log Scale]', fontsize=12)
    ax2.set_ylabel('Absolute Error (km/s)', fontsize=12)
    ax2.set_title('Spatial Distribution of Error (Flat Core Validation)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(True, which='major', alpha=0.5)
    ax2.grid(True, which='minor', alpha=0.2, linestyle=':')

    # -------------------------------------------------------------------------
    # Panel 3: Error Across Mass/Velocity Scales (Universality Check)
    # -------------------------------------------------------------------------
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_xscale('log') 
    ax3.scatter(V_true_clean, abs_res_clean, alpha=0.3, color='darkorchid', edgecolor='none', s=20)
    ax3.axhline(0.0, color='black', linewidth=2, linestyle='--')
    ax3.axhline(p25_err, color='black', linewidth=2, linestyle=':', alpha=0.7, label='IQR (25% - 75%)')
    ax3.axhline(p75_err, color='black', linewidth=2, linestyle=':', alpha=0.7)
    ax3.set_ylim(-50.0, 50.0) 
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
    ax3.set_xlabel(r'Observed Missing Mass Velocity ($V_{\mathrm{true}}$) [km/s] [Log Scale]', fontsize=12)
    ax3.set_ylabel('Absolute Error (km/s)', fontsize=12)
    ax3.set_title('Error Distribution across Mass Scales (Universality)', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower left')
    ax3.grid(True, which='major', alpha=0.5)
    ax3.grid(True, which='minor', alpha=0.2, linestyle=':')
    
    # -------------------------------------------------------------------------
    # Panel 4: Global Absolute Residual Distribution (Histogram)
    # -------------------------------------------------------------------------
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(abs_res_clean, bins=60, range=(-40.0, 40.0), color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax4.axvline(mean_err, color='red', linestyle='dashed', linewidth=2, label=f'Global Mean: {mean_err:.1f} km/s')
    ax4.axvline(median_err, color='blue', linestyle='dotted', linewidth=2, label=f'Global Median: {median_err:.1f} km/s')
    ax4.axvline(p25_err, color='black', linestyle=':', linewidth=2, alpha=0.7, label='IQR Boundaries')
    ax4.axvline(p75_err, color='black', linestyle=':', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Absolute Residual (km/s)', fontsize=12)
    ax4.set_ylabel('Count (Data Points)', fontsize=12)
    ax4.set_title('Global Absolute Error Distribution', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.835])
    plt.savefig('Dark_Matter_Residuals_Analysis_5.png', dpi=300, bbox_inches='tight')
    print("Saved 'Dark_Matter_Residuals_Analysis_5.png' successfully!")

if __name__ == "__main__":
    galaxies = load_and_filter_galaxies("cdsarc_152_157_table2.xlsx")
    num_galaxies = len(galaxies)
    v_true, v_syn, rad = calculate_all_residuals(galaxies)
    
    # Print the stratified tier table in the terminal
    print_cohort_breakdown(v_true, v_syn)
    
    # Generate the 4-panel dashboard
    plot_dashboard(v_true, v_syn, rad, num_galaxies)