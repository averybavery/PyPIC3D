"""Weakly ionized two-stream instability: full PyPIC3D runs with and without
electron-neutral Monte Carlo collisions (elastic + excitation) in a cold argon gas.
Each curve is the mean over SEEDS runs with a shaded +/- 1 standard deviation band; --plot redraws from saved runs."""
import os
import shutil
import sys

import jax
import matplotlib.pyplot as plt
import numpy as np

from PyPIC3D.__main__ import run_PyPIC3D
import toml

jax.config.update("jax_enable_x64", True)

# ----------------------------- input parameters -----------------------------
NX = 100                        # grid cells
N_BEAM = 1500                   # macro-electrons per beam
V_BEAM = 0.5e8                  # beam drift speeds +/- V_BEAM [m/s]
VTH_E = 1.5e7                   # electron thermal speed [m/s]
T_END = 5.413e-8                # simulated time [s]
GAS_DENSITY = 1.5e20            # argon density [m^-3]
GAS_TEMPERATURE = 300.0         # [K]
SIGMA_ELASTIC = 1.0e-19         # elastic cross section [m^2] (nu ~ n sigma v ~ 7.5e8 1/s)
SIGMA_EXCITATION = 2.0e-20      # excitation cross section above threshold [m^2]
EXCITATION_ENERGY = 11.5        # argon excitation threshold [eV]
SEEDS = range(5)                # independent random seeds (same seeds with and without collisions)
OUTPUT_DIR = "outputs/two_stream_mcc"

M_AR = 6.6335209e-26
BEAM = """
[particle{i}]
name = "{key}"
N_particles = {n}
number_density = 1.0e15
charge = -1.602e-19
mass = 9.1093837e-31
vth = {vth}
initial_vx = {v}
initial_x = "{x_path}"
"""
COLLISIONS = """
[collisions.{key}_argon]
type = "background_mcc"
species = "{key}"
background_density = {n_gas}
background_temperature = {t_gas}
background_mass = {m_gas}
[collisions.{key}_argon.elastic]
cross_section = {sigma}
[collisions.{key}_argon.excitation]
energy = {e_exc}
cross_section = [[0.0, 0.0], [{e_exc}, 0.0], [{e_exc_plus}, {sigma_exc}], [1.0e4, {sigma_exc}]]
"""


def write_config(name, collisions, seed):
    out = os.path.join(OUTPUT_DIR, name, f"seed{seed}")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    text = f"""[simulation_parameters]
name = "{name}"
seed = {seed}
t_wind = {T_END}
solver = "electrodynamic_yee"
Nx = {NX}
Ny = 1
Nz = 1
x_wind = 1
y_wind = 1
z_wind = 1
output_dir = "{out}"
cfl = 0.99
shape_factor = 2
current_calculation = "j_from_rhov"
relativistic = true

[plotting]
plotting_interval = 4
plot_openpmd_particles = false
plot_openpmd_fields = false
"""
    for i, (key, v) in enumerate((("electron1", V_BEAM), ("electron2", -V_BEAM)), start=1):
        x_path = os.path.join(out, f"{key}_x.npy")
        np.save(x_path, np.random.default_rng([seed, i]).uniform(-0.5, 0.5, N_BEAM))
        text += BEAM.format(i=i, key=key, n=N_BEAM, vth=VTH_E, v=v, x_path=x_path)
        if collisions:
            text += COLLISIONS.format(key=key, n_gas=GAS_DENSITY, t_gas=GAS_TEMPERATURE, m_gas=M_AR, sigma=SIGMA_ELASTIC,
                                      e_exc=EXCITATION_ENERGY, e_exc_plus=EXCITATION_ENERGY * 1.001, sigma_exc=SIGMA_EXCITATION)
    text += f"""
[particle3]
name = "ions"
N_particles = {2 * N_BEAM}
charge = 1.602e-19
mass = 1.67e-27
vth = 3.5e4
initial_x = "{os.path.join(out, 'ions_x.npy')}"
"""
    np.save(os.path.join(out, "ions_x.npy"), np.random.default_rng([seed, 3]).uniform(-0.5, 0.5, 2 * N_BEAM))
    path = os.path.join(out, f"{name}.toml")
    with open(path, "w") as handle:
        handle.write(text)
    return path, out


# ----------------------------- run the simulations -----------------------------
runs = {}
for name, collisions in (("collisionless", False), ("mcc", True)):
    seeds = []
    for seed in SEEDS:
        out = os.path.join(OUTPUT_DIR, name, f"seed{seed}")
        if "--plot" not in sys.argv:
            print(f"\n=== Two-stream run: {name}, seed {seed} ===", flush=True)
            path, out = write_config(name, collisions, seed)
            np.random.seed(seed)                                      # PyPIC3D samples thermal velocities with NumPy
            run_PyPIC3D(toml.load(path))
        load = lambda f: np.loadtxt(os.path.join(out, "data", f), delimiter=",", ndmin=2)
        field, kinetic, momentum = load("electric_field_energy.txt"), load("kinetic_energy.txt"), load("total_momentum.txt")
        seeds.append((field[:, 1], kinetic[:, 1] / kinetic[0, 1], momentum[:, 1] / momentum[0, 1]))
    runs[name] = field[:, 0], np.array(seeds)                         # time, (seed, quantity, sample)

# ----------------------------- plot, save, and print -----------------------------
fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
colors = {"collisionless": ("tab:red", "tab:blue", "tab:green"), "mcc": ("tab:orange", "tab:cyan", "tab:olive")}
labels = {"collisionless": "collisionless", "mcc": "with electron-neutral MCC"}
for name, (t, data) in runs.items():
    for k in range(3):
        mean, std = data[:, k].mean(0), data[:, k].std(0)
        ax[k].fill_between(t, mean - std, mean + std, color=colors[name][k], alpha=0.25, lw=0)
        ax[k].plot(t, mean, color=colors[name][k], lw=2, label=labels[name])
    peak, final = data[:, 0].max(1), data[:, 1, -1]
    print(f"{name:14s} peak E-field energy = {peak.mean():.3e} +/- {peak.std():.3e} J, "
          f"final kinetic / initial = {final.mean():.4f} +/- {final.std():.4f}")
ax[0].set(ylabel="Electric field energy [J]", title=rf"Weakly ionized two-stream comparison (mean $\pm 1\sigma$, {len(SEEDS)} seeds)")
ax[1].set(ylabel="Kinetic energy / initial")
ax[2].set(ylabel=r"$\sum m|\mathbf{v}|$ / initial", xlabel="Time [s]")
for a in ax:
    a.legend()
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "two_stream_mcc.png"), dpi=300)
print(f"Saved {OUTPUT_DIR}/two_stream_mcc.png")
