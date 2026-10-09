"""
=============================================================================
Linear Shear Vacuum Model - Virtual 3D Rotation Curve Generator (V2.3)
=============================================================================
Author: Bart Leplae
License: MIT
Data Source: Synthetic / Virtual Vacuum SPARC Baseline

Physical Foundation:
-------------------
Derives virtual galaxy rotation curves from the first-principles 
vacuum mechanics framework ("Gravity as Relaxation"):

    Rho_DM(R, z) = ( V_char / (4 * pi * G) ) * Shear(R, z)

Where:
    - G is Newton's gravitational constant in galactic units:
        G_gal = 4.3009e-6 (km/s)^2 * kpc / M_sun
    - V_char = 94.04 km/s is the universal characteristic velocity of the 
      relaxing space-quantum lattice across the SPARC sample.
    - Coupling Constant: V_char / (4 * pi * G) = 1.74e+06 M_sun / kpc^3
    - Shear(R, z) = sqrt( (dOmega/dR)^2 + (dOmega/dz)^2 )
=============================================================================
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re

warnings.filterwarnings('ignore')

# =============================================================================
# FIRST-PRINCIPLES PHYSICAL CONSTANTS (GALACTIC UNITS)
# =============================================================================
# Newton's Gravitational Constant in galactic units: (km/s)^2 * kpc / M_sun
G_GAL = 4.3009e-6  

# Universal Characteristic Velocity (Median galactic attractor scale in SPARC)
V_CHAR_UNIVERSAL = 94.04  # km/s

# Speed of light in vacuum (km/s)
C0_KMS = 299792.458  

# Universal Vacuum Coupling Constant:
# Yields mass density directly in M_sun / kpc^3 per [(km/s) / kpc^2]
COUPLING_KPC_UNIVERSAL = V_CHAR_UNIVERSAL / (4.0 * np.pi * G_GAL)  # ~ 1.74e6 M_sun/kpc^3

# Baseline Mass-to-Light ratios
UPSILON_DISK = 0.5   
UPSILON_BULGE = 0.7  

def load_config(config_path: str):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Could not find configuration file at '{config_path}'.")
        return None

def evaluate_galaxy(hv: float, R_grid: np.ndarray, z_grid: np.ndarray, 
                    R_2d: np.ndarray, z_2d: np.ndarray, V_midplane: np.ndarray, 
                    masks: dict, kernels: list, Rad_obs: np.ndarray, V_true: np.ndarray) -> tuple:
    """
    Computes 2D kinematic shear and resulting synthetic velocity using 
    the first-principles vacuum density law.
    """
    # 1. Kinematics: Vertical exponential velocity decay and angular velocity map
    V_2d = V_midplane * np.exp(-z_2d / hv)
    Omega_2d = V_2d / R_2d
    
    # 2. Kinematic Shear: Magnitude of the differential rotation tensor
    dOmega_dR = np.gradient(Omega_2d, R_grid, axis=0)
    dOmega_dz = np.gradient(Omega_2d, z_grid, axis=1)
    Shear_2d = np.sqrt(dOmega_dR**2 + dOmega_dz**2)
    
    # 3. First-Principles Density Law: Directly in M_sun / kpc^3
    Rho_2d_kpc = COUPLING_KPC_UNIVERSAL * Shear_2d
    Rho_2d_kpc[masks['outer']] *= masks['outer_taper']
    Rho_2d_kpc[np.sqrt(R_2d**2 + z_2d**2) > masks['edge']] = 0.0
    Rho_kpc = Rho_2d_kpc[:, :, None]
    
    # 4. Newtonian Summation across distance kernels
    V_syn = np.array([np.sqrt(np.maximum(0.0, R * np.sum(Rho_kpc * kernels[i]))) for i, R in enumerate(Rad_obs)])
    mse = np.mean((V_syn - V_true)**2)
    return mse, V_syn

def process_galaxy_3d(df_target: pd.DataFrame, target_config: dict, output_folder: str):
    galaxy_name = target_config['name']
    
    # Extract dynamic parameters (Assuming K, c, and hv_fixed are configured)
    u_disk = target_config.get('upsilon_disk', UPSILON_DISK)
    u_bulge = target_config.get('upsilon_bulge', UPSILON_BULGE)
    description = target_config.get('description', 'Standard Virtual Galaxy')
    
    K_val = target_config['K']
    c_val = target_config['c']
    best_hv = float(target_config['hv_fixed'])
    
    df_target = df_target[df_target['Rad'] > 0.01].copy()
    df_target = df_target.sort_values(by='Rad').reset_index(drop=True)
    
    if len(df_target) < 3:
        print(f"  -> Skipping {galaxy_name}: Not enough data points.")
        return

    R_obs = df_target['Rad'].values
    V_obs = df_target['Vobs'].values
    
    V_disk_sq = u_disk * (pd.to_numeric(df_target['Vdisk']).values ** 2)
    V_bulge_sq = u_bulge * (pd.to_numeric(df_target['Vbulge']).values ** 2)
    V_gas_sq = pd.to_numeric(df_target['Vgas']).values * np.abs(pd.to_numeric(df_target['Vgas']).values)
    
    V_bar_sq = V_disk_sq + V_bulge_sq + V_gas_sq
    V_bar_sq[V_bar_sq < 0] = 0
    V_bar = np.sqrt(V_bar_sq)
    
    V_true_sq = (V_obs ** 2) - V_bar_sq
    V_true_sq[V_true_sq < 0] = 0
    V_true = np.sqrt(V_true_sq)
    
    # 3D Grid Pre-computation:
    # Extended to 1.25 * R_max with flat velocity continuation to prevent boundary artifacts
    R_max = R_obs.max()
    r_min = max(0.08, R_obs.min())
    
    R_grid = np.linspace(r_min, R_max * 1.25, 60)
    z_grid = np.linspace(0.0, R_max * 1.6, 45)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, 50)
    
    R_3d, z_3d, phi_3d = np.meshgrid(R_grid, z_grid, phi_grid, indexing='ij')
    R_2d, z_2d = np.meshgrid(R_grid, z_grid, indexing='ij')
    
    V_midplane = np.interp(R_grid, R_obs, V_obs, right=V_obs[-1])[:, None]
    
    r_sph2d = np.sqrt(R_2d**2 + z_2d**2)
    masks = {
        'outer': r_sph2d > R_max * 1.05,
        'outer_taper': np.exp(-(r_sph2d[r_sph2d > R_max * 1.05] - R_max * 1.05) / (0.35 * R_max)),
        'edge': R_max * 1.9
    }
    
    dR = R_grid[1] - R_grid[0]
    dz = z_grid[1] - z_grid[0]
    dphi = phi_grid[1] - phi_grid[0]
    dV = R_3d * dR * dz * dphi
    softening = (max(dR, dz) * 1.8)**2
    
    kernels = []
    for R in R_obs:
        if R > 0:
            denom = (R_3d**2 + R**2 - 2.0*R_3d*R*np.cos(phi_3d) + z_3d**2 + softening)**1.5
            k = 2.0 * G_GAL * dV * (R - R_3d * np.cos(phi_3d)) / denom
            kernels.append(k)
        else:
            kernels.append(np.zeros_like(R_3d))
            
    # =========================================================
    # PHYSICS INTEGRATION (Using Forced hv)
    # =========================================================
    print(f"  -> Using fixed hv = {best_hv:.2f} kpc for {galaxy_name}")
    _, V_syn = evaluate_galaxy(best_hv, R_grid, z_grid, R_2d, z_2d, V_midplane, masks, kernels, R_obs, V_true)
    V_tot_predicted = np.sqrt(V_bar_sq + V_syn**2)

    # =========================================================
    # CALCULATE COSMOLOGICAL MASS & ENERGY
    # =========================================================
    # 1. Total Vacuum Energy (E = K^2 * c / G)
    E_total = ((K_val**2) * c_val) / G_GAL
    
    # 2. Apparent Missing Mass at R_max (M = V^2 * R / G)
    M_missing_max_billions = ((V_true[-1]**2) * R_max / G_GAL) / 1e9

    # =========================================================
    # PLOTTING
    # =========================================================
    plt.figure(figsize=(12, 7.5))
    
    # Foundational curves
    plt.plot(R_obs, V_obs, 'ko-', linewidth=2, label=r'Target Total Velocity ($V_{\mathrm{obs}}$)', markersize=6)
    plt.plot(R_obs, V_bar, 'b--', linewidth=2, label=r'Visible Matter Contribution ($V_{\mathrm{bar}}$)')
    
    # Model predictions
    plt.plot(R_obs, V_syn, 'm-.', linewidth=2.5, label=r'Model Predicted Vacuum Mass ($V_{\mathrm{syn}}$)')
    plt.plot(R_obs, V_tot_predicted, 'g-', linewidth=3, alpha=0.7, label=r'Total Model Prediction ($V_{\mathrm{syn}} + V_{\mathrm{bar}}$)')

    max_y = max(V_obs) * 1.50 
    
    plt.title(f'{galaxy_name} 3D Rotation Curve Breakdown\nTesting the Linear Shear Vacuum Model', 
              fontsize=16, fontweight='bold', pad=22)
    plt.xlabel('Radial Distance (kpc)', fontsize=13)
    plt.ylabel('Orbital Velocity (km/s)', fontsize=13)

    # --- DYNAMIC PARAMETER BOX BUILDER ---
    base, exp = f"{COUPLING_KPC_UNIVERSAL:.2e}".split('e')
    coupling_latex = f"{base} \\times 10^{{{int(exp)}}}"

    param_lines = [f"{description}"]
    
    # Line 2: Universal Physics (Coupling Constant and V_char)
    physics_str = (
        rf"Coupling Constant $\left(\frac{{V_{{\mathrm{{char}}}}}}{{4\pi G}}\right) = {coupling_latex}\ M_\odot/\mathrm{{kpc}}^3$   |   "
        rf"$V_{{\mathrm{{char}}}}$ (Universal Attractor Velocity) = {V_CHAR_UNIVERSAL:.2f} km/s"
    )
    param_lines.append(physics_str)
    
    # Line 3: Virtual Generation Variables & Geometry
    gen_str = (
        f"Terminal Velocity ($K$) = {K_val} km/s   |   "
        f"Energy Scale ($c$) = {c_val} kpc$\\cdot$(km/s)$^2$   |   "
        f"Disk Thickness ($hv$): {best_hv:.2f} kpc"
    )
    param_lines.append(gen_str)
    
    # Line 4: The Physical Consequences
    cosmo_str = (
        f"Total Vacuum Energy ($E = \\frac{{K^2 \\cdot c}}{{G}}$) = {E_total:.2e} $M_\\odot\\cdot$(km/s)$^2$   |   "
        f"Apparent Missing Mass (at {R_max:.1f} kpc) = {M_missing_max_billions:.1f} Billion $M_\\odot$"
    )
    param_lines.append(cosmo_str)
    
    context_text = "\n".join(param_lines)
    
    plt.text(0.5, 0.98, context_text, transform=plt.gca().transAxes, fontsize=10.5,
             ha='center', va='top', bbox=dict(facecolor='white', alpha=0.95, edgecolor='lightgrey', boxstyle='round,pad=0.55'))    
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.xlim(0.0, max(R_obs) * 1.05)
    plt.ylim(0.0, max_y)
    plt.tight_layout()
    
    safe_name = re.sub(r'\W+', '_', galaxy_name)
    output_path = os.path.join(output_folder, f'{safe_name}_rotation_curves.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"  -> Saved 3D plot for {galaxy_name}: {output_path}")

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'virtual_galaxies_to_generate.yml'
    
    config = load_config(config_file)
    if not config: 
        exit()
        
    data_file = config.get('data_file', 'cdsarc_152_157_table2.xlsx')
    output_folder = config.get('output_folder', 'Shear_Analysis_Plots')
    target_galaxies = config.get('target_galaxies', [])
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"Loading Dataset from {data_file}...")
    try:
        df_sparc = pd.read_excel(data_file) if data_file.endswith('.xlsx') else pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.")
        exit()
        
    col_name = 'Name' if 'Name' in df_sparc.columns else 'Galaxy' if 'Galaxy' in df_sparc.columns else 'Galaxy identifier' if 'Galaxy identifier' in df_sparc.columns else None
    
    if col_name:
        df_sparc['Clean_Galaxy'] = df_sparc[col_name].astype(str).str.replace(r'\W+', '', regex=True).str.upper()
    else:
        print("Error: Could not identify Galaxy Name column.")
        exit()
    
    df_sparc = df_sparc.rename(columns={
        'rad': 'Rad', 'vobs': 'Vobs', 'errv': 'errV', 
        'vgas': 'Vgas', 'vdisk': 'Vdisk', 'vbulge': 'Vbulge'
    })
    
    print(f"Processing {len(target_galaxies)} virtual galaxies via 3D Integration...")
    for target in target_galaxies:
        target_config = {'name': target} if isinstance(target, str) else target
        target_name = target_config['name']
        
        clean_target = re.sub(r'\W+', '', str(target_name)).upper()
        df_target = df_sparc[df_sparc['Clean_Galaxy'] == clean_target].copy()
        
        if df_target.empty:
            print(f"  -> Warning: Galaxy '{target_name}' not found. Skipping.")
        else:
            process_galaxy_3d(df_target, target_config, output_folder)
            
    print("\nBatch processing complete!")