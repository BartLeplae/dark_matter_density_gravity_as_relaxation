"""
=============================================================================
Linear Shear Vacuum Model - 3D Batch Rotation Curve Generator (V2.3)
=============================================================================
Author: Bart Leplae
License: MIT
Data Source: SPARC (https://cdsarc.cds.unistra.fr/ftp/J/AJ/152/157/)

Physical Foundation:
-------------------
Derives individual galactic rotation curves from the first-principles 
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
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
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
                    masks: dict, kernels: list, Rad_obs: np.ndarray, 
                    V_true: np.ndarray, hv_target: float) -> tuple:
    """
    Computes 2D kinematic shear and resulting synthetic velocity using 
    the first-principles vacuum density law and size-adaptive regularization.
    """
    # 1. Kinematics: Vertical exponential velocity decay and angular velocity map
    V_2d = V_midplane * np.exp(-z_2d / hv)
    Omega_2d = V_2d / R_2d
    
    # 2. Kinematic Shear: Magnitude of differential rotation tensor
    dOmega_dR = np.gradient(Omega_2d, R_grid, axis=0)
    dOmega_dz = np.gradient(Omega_2d, z_grid, axis=1)
    Shear_2d = np.sqrt(dOmega_dR**2 + dOmega_dz**2)
    
    # 3. First-Principles Density Law: Yields M_sun / kpc^3 directly
    Rho_2d_kpc = COUPLING_KPC_UNIVERSAL * Shear_2d
    Rho_2d_kpc[masks['outer']] *= masks['outer_taper']
    Rho_2d_kpc[np.sqrt(R_2d**2 + z_2d**2) > masks['edge']] = 0.0
    Rho_kpc = Rho_2d_kpc[:, :, None]
    
    # 4. Newtonian Summation across distance kernels
    V_syn = np.array([np.sqrt(np.maximum(0.0, R * np.sum(Rho_kpc * kernels[i]))) for i, R in enumerate(Rad_obs)])
    
    # Size-adaptive regularization penalty
    reg_penalty = 1.0 * ((hv - hv_target) / hv_target)**2
    mse = np.mean((V_syn - V_true)**2) + reg_penalty
    return mse, V_syn

def process_galaxy_3d(df_target: pd.DataFrame, target_config: dict, output_folder: str):
    galaxy_name = target_config['name']
    
    # Extract dynamic parameters
    u_disk = target_config.get('upsilon_disk', UPSILON_DISK)
    u_bulge = target_config.get('upsilon_bulge', UPSILON_BULGE)
    description = target_config.get('description', 'Standard SPARC Galaxy')
    
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
    # Extended to 1.25 * R_max with flat velocity continuation to prevent outer edge droop
    R_max = R_obs.max()
    r_min = max(0.08, R_obs.min())
    hv_target = max(0.35, 0.15 * R_max)
    
    R_grid = np.linspace(r_min, R_max * 1.25, 60)
    z_grid = np.linspace(0.0, R_max * 1.6, 45)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, 50)
    
    R_3d, z_3d, phi_3d = np.meshgrid(R_grid, z_grid, phi_grid, indexing='ij')
    R_2d, z_2d = np.meshgrid(R_grid, z_grid, indexing='ij')
    
    # Asymptotic flat continuation for outer extrapolation
    V_midplane = np.interp(R_grid, R_obs, V_obs, right=V_obs[-1])[:, None]
    
    # Extended Geometric Boundary Masks
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
    # OPTIMIZATION / OVERRIDE
    # =========================================================
    if 'hv_fixed' in target_config:
        best_hv = float(target_config['hv_fixed'])
        print(f"  -> Manual Override: using hv = {best_hv:.2f} kpc for {galaxy_name}")
    else:
        hv_bounds = (max(0.15, 0.05 * R_max), max(8.0, 0.45 * R_max))
        res = minimize_scalar(
            lambda hv: evaluate_galaxy(hv, R_grid, z_grid, R_2d, z_2d, V_midplane, masks, kernels, R_obs, V_true, hv_target)[0],
            bounds=hv_bounds, method='bounded'
        )
        best_hv = res.x
    
    _, V_syn = evaluate_galaxy(best_hv, R_grid, z_grid, R_2d, z_2d, V_midplane, masks, kernels, R_obs, V_true, hv_target)
    V_tot_predicted = np.sqrt(V_bar_sq + V_syn**2)

    # =========================================================
    # CALCULATE MASS / ENERGY EQUIVALENTS (at R_max)
    # =========================================================
    # 1. Kinematic Required Mass (Classical Spherical Approximation: V^2 * R / G)
    M_missing_kinematic = ((V_true[-1]**2) * R_max / G_GAL) / 1e9

    # 2. Volume-Integrated Vacuum Energy (Shear Field Integration)
    V_2d = V_midplane * np.exp(-z_2d / best_hv)
    Omega_2d = V_2d / R_2d
    Shear_2d = np.sqrt(np.gradient(Omega_2d, R_grid, axis=0)**2 + np.gradient(Omega_2d, z_grid, axis=1)**2)
    
    Rho_2d_kpc = COUPLING_KPC_UNIVERSAL * Shear_2d
    Rho_2d_kpc[masks['outer']] *= masks['outer_taper']
    Rho_2d_kpc[np.sqrt(R_2d**2 + z_2d**2) > masks['edge']] = 0.0
    
    dV_layer = 2.0 * np.pi * R_2d * dR * dz * 2.0  # Top and bottom hemispheres
    M_missing_integrated = np.sum(Rho_2d_kpc * dV_layer) / 1e9 

    # 3. Pure Rotation Energy (Gravitomagnetic equivalent mass)
    Omega_2d_sq = (V_2d / R_2d)**2
    Rho_rot_2d = Omega_2d_sq / (4.0 * np.pi * G_GAL) 
    Rho_rot_2d[np.sqrt(R_2d**2 + z_2d**2) > masks['edge']] = 0.0
    M_pure_rotation = np.sum(Rho_rot_2d * dV_layer) / 1e9 

    # =========================================================
    # PLOTTING
    # =========================================================
    plt.figure(figsize=(12, 7.5))
    plt.plot(R_obs, V_obs, 'ko-', linewidth=2, label=r'Observed Total Velocity ($V_{\mathrm{obs}}$)', markersize=6)
    plt.plot(R_obs, V_bar, 'b--', linewidth=2, label=r'Visible Matter Contribution ($V_{\mathrm{bar}}$)')
    plt.fill_between(R_obs, V_bar, V_obs, color='crimson', alpha=0.15, label='The "Missing Mass" Anomaly')
    
    plt.plot(R_obs, V_syn, 'm-.', linewidth=2.5, label=r'Model Predicted Vacuum Mass ($V_{\mathrm{syn}}$)')
    plt.plot(R_obs, V_tot_predicted, 'g-', linewidth=3, alpha=0.7, label=r'Total Model Prediction ($V_{\mathrm{syn}} + V_{\mathrm{bar}}$)')

    max_y = max(V_obs) * 1.55 
    
    plt.title(f'{galaxy_name} 3D Rotation Curve Breakdown\nTesting the Linear Shear Vacuum Model', 
              fontsize=16, fontweight='bold', pad=22)
    plt.xlabel('Radial Distance (kpc)', fontsize=13)
    plt.ylabel('Orbital Velocity (km/s)', fontsize=13)
    
    base, exp = f"{COUPLING_KPC_UNIVERSAL:.2e}".split('e')
    coupling_str = f"{base}e+{int(exp):02d}"
    
    context_text = (
        f"{description}\n"
        f"Optimized Disk Thickness ($hv$): {best_hv:.2f} kpc   |   "
        f"Coupling Constant: {coupling_str} $M_\\odot/\\mathrm{{kpc}}^3$   |   "
        f"$V_{{\\mathrm{{char}}}}$: {V_CHAR_UNIVERSAL:.2f} km/s\n"
        f"Mass-to-Light Ratios: $\\Upsilon_{{\\mathrm{{disk}}}}$ = {u_disk}, $\\Upsilon_{{\\mathrm{{bulge}}}}$ = {u_bulge}\n"
        f"Req. Kinematic Mass (Sphere): {M_missing_kinematic:.2f} Billion $M_\\odot$   |   "
        f"Integrated Space Energy (Shear): {M_missing_integrated:.2f} Billion $M_\\odot$\n"
        f"Integrated Pure Rotation (3D): {M_pure_rotation:.2f} Billion $M_\\odot$"
    )
    
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
    
    print(f"  -> Saved 3D plot for {galaxy_name} (hv = {best_hv:.2f} kpc): {output_path}")

if __name__ == "__main__":
    config = load_config('galaxies_to_generate.yml')
    if not config: 
        exit()
        
    data_file = config.get('data_file', 'cdsarc_152_157_table2.xlsx')
    output_folder = config.get('output_folder', 'galaxy_rotation_curves')
    target_galaxies = config.get('target_galaxies', [])
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"Loading SPARC Database from {data_file}...")
    try:
        df_sparc = pd.read_excel(data_file) if data_file.endswith('.xlsx') else pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.")
        exit()
        
    col_name = 'Name' if 'Name' in df_sparc.columns else 'Galaxy' if 'Galaxy' in df_sparc.columns else 'Galaxy identifier' if 'Galaxy identifier' in df_sparc.columns else None
    
    df_sparc['Clean_Galaxy'] = df_sparc[col_name].astype(str).str.replace(r'\W+', '', regex=True).str.upper()

    df_sparc = df_sparc.rename(columns={
        'rad': 'Rad', 'vobs': 'Vobs', 'errv': 'errV', 
        'vgas': 'Vgas', 'vdisk': 'Vdisk', 'vbulge': 'Vbulge'
    })

    print(f"Processing {len(target_galaxies)} galaxies via 3D Integration...")
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