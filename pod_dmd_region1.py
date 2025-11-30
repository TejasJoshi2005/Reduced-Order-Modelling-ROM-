import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import re


data_folder = "../dataset/2D_DND_3900_region1"
output_folder = "../results/"
os.makedirs(output_folder, exist_ok=True)


all_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]


files = sorted(all_files, key=lambda f: int(re.findall(r'\d+', f)[-1]))


n_snapshots = len(files)
print(f"Total snapshots found: {n_snapshots}")
print("First 15 sorted files:")
print(files[:15])


print("Loading sample snapshot to get coordinates...")
sample_path = os.path.join(data_folder, files[0])
sample_data = np.loadtxt(sample_path)

n_points, n_vars = sample_data.shape  # e.g., (1794816, 6)
print(f"Data shape: {n_points} points x {n_vars} variables")


# Assumes x=col 0, y=col 1
x_coords = sample_data[:, 0]
y_coords = sample_data[:, 1]
print("Coordinates loaded.")

# Define a plotting grid (what we interpolate ONTO) 
Nx_plot = 300  # Resolution for plotting in x
Ny_plot = 200  # Resolution for plotting in y
xi = np.linspace(x_coords.min(), x_coords.max(), Nx_plot)
yi = np.linspace(y_coords.min(), y_coords.max(), Ny_plot)
grid_x, grid_y = np.meshgrid(xi, yi)


# 3 = u (x-velocity)
# 4 = v (y-velocity)
VAR_INDEX = 3 
VAR_NAME = "u-velocity"
print(f"Analyzing variable: Column {VAR_INDEX} ({VAR_NAME})")


# X will be (n_points, n_snapshots)
X = np.zeros((n_points, n_snapshots))

print("Loading all snapshots into data matrix X...")
for i, f in enumerate(files):
    if (i+1) % 50 == 0:
        print(f"  Loading snapshot {i+1}/{n_snapshots}")
    snapshot_path = os.path.join(data_folder, f)
    # Load the full (N_points, 6) data
    data = np.loadtxt(snapshot_path)
    # Store ONLY the variable we want (e.g., u-velocity)
    X[:, i] = data[:, VAR_INDEX]

print(f"Data matrix X shape: {X.shape}") # Should be (1794816, n_snapshots)


def mode_to_2d(vec):
    """
    Interpolate a 1D mode vector (from unstructured points)
    onto the 2D structured plotting grid.
    """
    # griddata(points, values, xi)
    print("  Interpolating mode for visualization...")
    grid_mode = griddata((x_coords, y_coords), vec, (grid_x, grid_y), method='cubic')
    return np.nan_to_num(grid_mode) # Handle NaNs from interpolation

# Visualization of first few snapshots (Now uses griddata) 
num_visualize = 5
print(f"Visualizing first {num_visualize} snapshots...")
for i in range(num_visualize):
    # Get the snapshot vector from our matrix X
    snapshot_vec = X[:, i]
    
    # Interpolate for plotting
    snapshot_2d = mode_to_2d(snapshot_vec)

    plt.figure(figsize=(8, 5))
    plt.contourf(grid_x, grid_y, snapshot_2d, levels=100, cmap='jet')
    plt.colorbar(label=VAR_NAME)
    plt.title(f"Flow Snapshot {i+1}", fontsize=12)
    plt.xlabel("X Coordinate") # <-- Fixed label
    plt.ylabel("Y Coordinate") # <-- Fixed label
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"flow_snapshot_{i+1}.png"), dpi = 150)
    plt.close()
print("Initial snapshot visualization complete.")

# --- Start POD/DMD ---
print("Subtracting mean...")
X_mean = np.mean(X, axis=1, keepdims=True)
X_fluc = X - X_mean
print("Mean-subtracted data matrix shape:", X_fluc.shape)

print("Performing SVD...")
U, S, VT = np.linalg.svd(X_fluc, full_matrices=False)
print("SVD complete.")

print("Shapes:")
print("U (POD modes):", U.shape)
print("S (singular values):", S.shape)
print("VT (temporal coefficients):", VT.shape)

energy = S**2 / np.sum(S**2)
print("Energy captured by first 5 modes:", energy[:5])

cum_energy = np.cumsum(energy)
num_dominant_modes = np.searchsorted(cum_energy, 0.95) + 1
print(f"Number of modes capturing 95% energy: {num_dominant_modes}")


dt = 1.0
num_plot_modes = 6
dmd_rank = num_dominant_modes if 'num_dominant_modes' in globals() else min(20, VT.shape[0])
dpi = 150
n_pixels, num_snapshots = X_fluc.shape # n_pixels is n_points

# --- POD PLOTS ---
print("Generating POD plots...")

# 1) Singular values (linear scale)
plt.figure()
plt.plot(np.arange(1, len(S) + 1), S, marker='o', color='b')
plt.xlabel('POD Mode Number', fontsize=11)
plt.ylabel('Singular Value', fontsize=11)
plt.title('POD Singular Values (Linear Scale)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'singular_values_linear.png'), dpi=dpi)
plt.close()

# 2) Singular values (log scale)
plt.figure()
plt.semilogy(np.arange(1, len(S) + 1), S, marker='o', color='r')
plt.xlabel('POD Mode Number', fontsize=11)
plt.ylabel('Singular Value (log scale)', fontsize=11)
plt.title('POD Singular Values (Log Scale)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'singular_values_log.png'), dpi=dpi)
plt.close()

# 3) Modal energy and cumulative energy
# (These plots are fine, no changes needed)
energy = S**2 / np.sum(S**2)
cum_energy = np.cumsum(energy)

plt.figure()
plt.bar(np.arange(1, len(S) + 1), energy, color='skyblue', edgecolor='k')
plt.xlabel('POD Mode Number', fontsize=11)
plt.ylabel('Energy Fraction', fontsize=11)
plt.title('Energy Captured by Each POD Mode', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'modal_energy_bar.png'), dpi=dpi)
plt.close()

plt.figure()
plt.plot(np.arange(1, len(S) + 1), cum_energy, 'o-', color='g')
plt.axhline(0.95, color='k', linestyle='--', label='95% Energy Threshold')
plt.xlabel('Number of POD Modes', fontsize=11)
plt.ylabel('Cumulative Energy Fraction', fontsize=11)
plt.title('Cumulative Energy Distribution', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'cumulative_energy.png'), dpi=dpi)
plt.close()


# 4) POD Spatial Modes (Contours)
for k in range(min(num_plot_modes, U.shape[1])):
    print(f"Plotting POD Mode {k+1}")
    mode_vec = U[:, k]
    # --- KEY CHANGE: Use interpolation function ---
    arr = mode_to_2d(mode_vec)
    
    plt.figure(figsize=(8, 5))
    # --- KEY CHANGE: Plot using grid_x, grid_y ---
    plt.contourf(grid_x, grid_y, arr, levels=50, cmap='jet')
    plt.xlabel('X Coordinate', fontsize=10) # <-- Fixed label
    plt.ylabel('Y Coordinate', fontsize=10) # <-- Fixed label
    plt.title(f'POD Spatial Mode {k+1}', fontsize=12)
    plt.colorbar(label='Mode Amplitude')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'POD_mode_{k+1:02d}.png'), dpi=dpi)
    plt.close()

# 5) POD Temporal Coefficients
# (This plot is fine, no changes needed)
modal_time = (np.diag(S) @ VT)  # More stable way to write S*VT
for k in range(min(num_plot_modes, modal_time.shape[0])):
    plt.figure()
    plt.plot(np.arange(num_snapshots) * dt, np.real(modal_time[k, :]), color='b')
    plt.xlabel('Time (t)', fontsize=11)
    plt.ylabel(f'Modal Amplitude a{k+1}(t)', fontsize=11)
    plt.title(f'POD Temporal Coefficient for Mode {k+1}', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'POD_temporal_mode_{k+1:02d}.png'), dpi=dpi)
    plt.close()

# 6) Reconstruction Quality (RMSE)
print("Plotting POD Reconstruction...")
r_recon = num_dominant_modes if 'num_dominant_modes' in globals() else min(10, U.shape[1])
Ur = U[:, :r_recon]
Sr = S[:r_recon]
VTr = VT[:r_recon, :]

X_recon = (Ur @ np.diag(Sr) @ VTr) + X_mean  # reconstructed field
X_orig = X + X_mean # Get back original (non-fluctuating) data

snap_idx = num_snapshots // 2  # pick mid snapshot

# --- KEY CHANGE: Must interpolate vectors for plotting ---
orig_snap_vec = X_orig[:, snap_idx]
recon_snap_vec = X_recon[:, snap_idx]

orig_snap_2d = mode_to_2d(orig_snap_vec)
recon_snap_2d = mode_to_2d(recon_snap_vec)
diff_snap_2d = orig_snap_2d - recon_snap_2d

rmse_time = np.sqrt(np.mean((X_orig - X_recon)**2, axis=0))

plt.figure(figsize=(15, 5)) # Wider figure
plt.subplot(1, 3, 1)
# --- KEY CHANGE: Plot using grid_x, grid_y ---
plt.contourf(grid_x, grid_y, orig_snap_2d, levels=50, cmap='jet')
plt.title('Original Snapshot', fontsize=11)
plt.xlabel('X Coordinate'); plt.ylabel('Y Coordinate') # <-- Fixed label
plt.colorbar(label=VAR_NAME)
plt.axis('equal')

plt.subplot(1, 3, 2)
# --- KEY CHANGE: Plot using grid_x, grid_y ---
plt.contourf(grid_x, grid_y, recon_snap_2d, levels=50, cmap='jet')
plt.title(f'Reconstructed (r={r_recon})', fontsize=11)
plt.xlabel('X Coordinate'); plt.ylabel('Y Coordinate') # <-- Fixed label
plt.colorbar(label=VAR_NAME)
plt.axis('equal')

plt.subplot(1, 3, 3)
# --- KEY CHANGE: Plot using grid_x, grid_y ---
plt.contourf(grid_x, grid_y, diff_snap_2d, levels=50, cmap='seismic')
plt.title('Difference (Original - Reconstructed)', fontsize=11)
plt.xlabel('X Coordinate'); plt.ylabel('Y Coordinate') # <-- Fixed label
plt.colorbar(label='Error Magnitude')
plt.axis('equal')

plt.suptitle(f'POD Reconstruction Snapshot #{snap_idx}', fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(output_folder, f'reconstruction_snapshot_{snap_idx:03d}.png'), dpi=dpi)
plt.close()

# (RMSE plot is fine, no changes needed)
plt.figure()
plt.plot(np.arange(num_snapshots) * dt, rmse_time, color='r')
plt.xlabel('Time (t)', fontsize=11)
plt.ylabel('RMSE across field', fontsize=11)
plt.title('POD Reconstruction Error Over Time', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'reconstruction_rmse_time.png'), dpi=dpi)
plt.close()

# 7) Cross-Correlation (This plot is fine, no changes needed)
if modal_time.shape[0] >= 2:
    a1, a2 = np.real(modal_time[0, :]), np.real(modal_time[1, :])
    corr = np.correlate(a1 - np.mean(a1), a2 - np.mean(a2), mode='full')
    lags = np.arange(-len(a1) + 1, len(a1))
    plt.figure()
    plt.plot(lags * dt, corr, color='m')
    plt.xlabel('Time Lag (t)', fontsize=11)
    plt.ylabel('Cross-Correlation', fontsize=11)
    plt.title('Cross-Correlation: Temporal Coefficients (Mode 1 vs Mode 2)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'crosscor_mode1_mode2.png'), dpi=dpi)
    plt.close()

# =============== DMD ANALYSIS ===============
print("Starting DMD Analysis...")

# Your DMD function is correct and works on the X matrix
def compute_dmd(X, r=None, dt=1.0):
    """Exact DMD computation"""
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    Ux, Sx, Vtx = np.linalg.svd(X1, full_matrices=False)
    Vx = Vtx.T
    if r is None:
        r = Ux.shape[1]
    r = min(r, Ux.shape[1])
    Uxr, Sxr, Vxr = Ux[:, :r], Sx[:r], Vx[:, :r]

    # Low-rank operator
    A_tilde = Uxr.T @ X2 @ Vxr @ np.diag(1 / Sxr)

    # Eigen-decomposition
    eigvals, W = np.linalg.eig(A_tilde)

    # DMD Modes
    Phi = X2 @ Vxr @ np.diag(1 / Sxr) @ W

    # Amplitudes
    x0 = X[:, 0]
    b, *_ = np.linalg.lstsq(Phi, x0, rcond=None)

    # Time evolution
    tvec = np.arange(0, X.shape[1]) * dt
    omega = np.log(eigvals) / dt
    eigs_matrix = np.array([eigvals**t for t in tvec]).T
    coef = b[:, None] * eigs_matrix

    # Reconstruction
    X_dmd = Phi @ coef

    return {'eigvals': eigvals, 'modes': Phi, 'amplitudes': b,
            'time_evolution': coef, 'reconstruction': X_dmd,
            'omega': omega, 'tvec': tvec}

dmd_res = compute_dmd(X_fluc, r=dmd_rank, dt=dt)
print("DMD computation complete.")

# 8) DMD Eigenvalues on Complex Plane
# (This plot is fine, no changes needed)
print("Plotting DMD results...")
eigvals = dmd_res['eigvals']
plt.figure()
plt.scatter(np.real(eigvals), np.imag(eigvals), c='r', marker='o', label='DMD eigenvalues')
theta = np.linspace(0, 2 * np.pi, 300)
plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=0.8, label='Unit Circle')
plt.xlabel('Real(λ)', fontsize=11)
plt.ylabel('Imag(λ)', fontsize=11)
plt.title('DMD Eigenvalues in Complex Plane', fontsize=12)
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'DMD_eigenvalues_complex_plane.png'), dpi=dpi)
plt.close()

# 9) DMD Spatial Modes
Phi = dmd_res['modes']
for k in range(min(num_plot_modes, Phi.shape[1])):
    print(f"Plotting DMD Mode {k+1}")
    mode_vec = np.real(Phi[:, k])
    # --- KEY CHANGE: Use interpolation function ---
    arr = mode_to_2d(mode_vec)
    
    plt.figure(figsize=(8, 5))
    # --- KEY CHANGE: Plot using grid_x, grid_y ---
    plt.contourf(grid_x, grid_y, arr, levels=50, cmap='jet')
    plt.xlabel('X Coordinate', fontsize=10) # <-- Fixed label
    plt.ylabel('Y Coordinate', fontsize=10) # <-- Fixed label
    plt.title(f'DMD Spatial Mode {k+1} (Real Part)', fontsize=12)
    plt.colorbar(label='Mode Amplitude')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'DMD_mode_{k+1:02d}.png'), dpi=dpi)
    plt.close()

# 10) DMD Temporal Amplitudes
# (This plot is fine, no changes needed)
time = dmd_res['tvec']
for k in range(min(num_plot_modes, dmd_res['time_evolution'].shape[0])):
    plt.figure()
    plt.plot(time, np.real(dmd_res['time_evolution'][k, :]), color='g')
    plt.xlabel('Time (t)', fontsize=11)
    plt.ylabel(f'DMD Amplitude b{k+1}(t)', fontsize=11)
    plt.title(f'DMD Temporal Coefficient for Mode {k+1}', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'DMD_temporal_mode_{k+1:02d}.png'), dpi=dpi)
    plt.close()

# 11) DMD Reconstruction vs Original (mid snapshot)
print("Plotting DMD Reconstruction...")
X_dmd_full = np.real(dmd_res['reconstruction'] + X_mean)

# --- KEY CHANGE: Must interpolate vectors for plotting ---
orig_snap_vec = X_orig[:, snap_idx]
dmd_snap_vec = X_dmd_full[:, snap_idx]

orig_snap_2d = mode_to_2d(orig_snap_vec)
dmd_snap_2d = mode_to_2d(dmd_snap_vec)
diff_dmd_snap_2d = orig_snap_2d - dmd_snap_2d

plt.figure(figsize=(15, 5)) # Wider figure
plt.subplot(1, 3, 1)
# --- KEY CHANGE: Plot using grid_x, grid_y ---
plt.contourf(grid_x, grid_y, orig_snap_2d, levels=50, cmap='jet')
plt.title('Original Snapshot', fontsize=11)
plt.xlabel('X Coordinate'); plt.ylabel('Y Coordinate') # <-- Fixed label
plt.colorbar(label=VAR_NAME)
plt.axis('equal')

plt.subplot(1, 3, 2)
# --- KEY CHANGE: Plot using grid_x, grid_y ---
plt.contourf(grid_x, grid_y, dmd_snap_2d, levels=50, cmap='jet')
plt.title('DMD Reconstructed Snapshot', fontsize=11)
plt.xlabel('X Coordinate'); plt.ylabel('Y Coordinate') # <-- Fixed label
plt.colorbar(label=VAR_NAME)
plt.axis('equal')

plt.subplot(1, 3, 3)
# --- KEY CHANGE: Plot using grid_x, grid_y ---
plt.contourf(grid_x, grid_y, diff_dmd_snap_2d, levels=50, cmap='seismic')
plt.title('Difference (Original - DMD)', fontsize=11)
plt.xlabel('X Coordinate'); plt.ylabel('Y Coordinate') # <-- Fixed label
plt.colorbar(label='Error Magnitude')
plt.axis('equal')

plt.suptitle(f'DMD Reconstruction Snapshot #{snap_idx}', fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(output_folder, f'DMD_reconstruction_snapshot_{snap_idx:03d}.png'), dpi=dpi)
plt.close()

# 12) Compare POD vs DMD Reconstruction Error
# (This plot is fine, no changes needed)
rmse_dmd = np.sqrt(np.mean((X_orig - X_dmd_full)**2, axis=0))
plt.figure()
plt.plot(np.arange(num_snapshots) * dt, rmse_time, label='POD RMSE', color='b')
plt.plot(np.arange(num_snapshots) * dt, rmse_dmd, label='DMD RMSE', color='r')
plt.xlabel('Time (t)', fontsize=11)
plt.ylabel('RMSE across field', fontsize=11)
plt.title('POD vs DMD Reconstruction Error', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'POD_vs_DMD_RMSE.png'), dpi=dpi)
plt.close()

print("✅ All labeled plots saved in:", output_folder)
