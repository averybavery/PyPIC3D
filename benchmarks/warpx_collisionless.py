"""Collisionless physics benchmarks of PyPIC3D against WarpX and exact theory (1D electromagnetic PIC,
identical inputs in both codes):
  1. two-stream instability growth rate          (cold-beam theory, gamma = omega_b / 2)
  2. Langmuir-wave dispersion omega(k)            (exact kinetic root; Bohm-Gross for reference)
  3. Landau damping at k lambda_D = 0.5           (exact kinetic rate gamma = -0.1534 omega_p)
  4. total-energy conservation of the same runs
Run on Linux/WSL with WarpX on PATH:  python benchmarks/warpx_collisionless.py"""
import os
import shutil
import subprocess
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton
from scipy.signal import find_peaks
from scipy.special import wofz

from PyPIC3D.__main__ import run_PyPIC3D

jax.config.update("jax_enable_x64", True)

# ----------------------------- input parameters -----------------------------
TWO_STREAM = dict(n_beam=1.0e15, v0=1.5e7, cells=64, ppc=200, t_end=25.0)  # t_end in 1 / omega_b
K_LAMBDA_D = [0.1, 0.2, 0.3, 0.4, 0.5]  # Langmuir-wave scan, one wavelength per box
DENSITY, VTH_OVER_C = 1.0e16, 0.05      # warm plasma: electron density [m^-3], thermal speed / c
CELLS, PPC = 32, 2000                   # Langmuir runs: cells per wavelength, electrons per cell
AMPLITUDE = 0.05                        # density perturbation dn/n, seeded as a velocity wave u = U sin(kx)
T_END_WP = 30.0                         # Langmuir runs: simulated time in 1 / omega_p
OUTPUT_DIR = "outputs/warpx_collisionless"
WARPX = shutil.which("warpx.1d")

E, ME, MP, EPS, C = 1.602176634e-19, 9.1093837e-31, 1.67262192e-27, 8.8541878128e-12, 2.99792458e8
VTH = VTH_OVER_C * C
OMEGA_P = np.sqrt(DENSITY * E**2 / (EPS * ME))
LAMBDA_D = VTH / OMEGA_P


def kinetic_root(k):
    """Exact Maxwellian Langmuir root omega / omega_p for k lambda_D = k."""
    f = lambda w: 1 + (1 + w / (np.sqrt(2) * k) * 1j * np.sqrt(np.pi) * wofz(w / (np.sqrt(2) * k))) / k**2
    return newton(f, np.sqrt(1 + 3 * k * k) + 0j)


def run_pypic3d(name, config):
    """Run PyPIC3D; return (time, E-field energy, total energy)."""
    out = os.path.join(OUTPUT_DIR, name, "pypic3d")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    config["simulation_parameters"]["output_dir"] = out
    for species in (v for k, v in config.items() if k.startswith("particle")):
        for key in ("initial_x", "initial_vx"):
            if isinstance(species.get(key), np.ndarray):
                np.save(os.path.join(out, f"{species['name']}_{key}.npy"), species[key])
                species[key] = os.path.join(out, f"{species['name']}_{key}.npy")
    run_PyPIC3D(config)
    load = lambda f: np.loadtxt(os.path.join(out, "data", f), delimiter=",", ndmin=2)
    field, total = load("electric_field_energy.txt"), load("total_energy.txt")
    return field[:, 0], field[:, 1], total[:, 1]


def run_warpx(name, inputs):
    """Run WarpX 1D; return (time, E-field energy, total energy)."""
    out = os.path.abspath(os.path.join(OUTPUT_DIR, name, "warpx"))
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    with open(os.path.join(out, "inputs"), "w") as handle:
        handle.write(inputs + "\nwarpx.reduced_diags_names = FE PE\nFE.type = FieldEnergy\nFE.intervals = 5\n"
                     "PE.type = ParticleEnergy\nPE.intervals = 5\n")
    log = subprocess.run([WARPX, "inputs"], cwd=out, capture_output=True, text=True)
    if log.returncode:
        raise RuntimeError(log.stdout[-3000:] + log.stderr[-3000:])
    fe = np.loadtxt(os.path.join(out, "diags/reducedfiles/FE.txt"))
    pe = np.loadtxt(os.path.join(out, "diags/reducedfiles/PE.txt"))
    return fe[:, 1], fe[:, 3], fe[:, 2] + pe[:, 2]


def warpx_deck(length, cells, dt, steps, species):
    return f"""max_step = {steps}
amr.n_cell = {cells}
amr.max_level = 0
geometry.dims = 1
geometry.prob_lo = 0
geometry.prob_hi = {length}
boundary.field_lo = periodic
boundary.field_hi = periodic
boundary.particle_lo = periodic
boundary.particle_hi = periodic
warpx.const_dt = {dt}
warpx.use_filter = 0
algo.current_deposition = direct
algo.particle_shape = 1
particles.species_names = {" ".join(s[0] for s in species)}
""" + "".join(f"""{name}.charge = {q}
{name}.mass = {m}
{name}.injection_style = NRandomPerCell
{name}.num_particles_per_cell = {ppc}
{name}.profile = constant
{name}.density = {density}
{name}.momentum_distribution_type = maxwellian
{name}.maxwellian_u_mean_distribution_type = parser
{name}.ux_mean_function(x,y,z) = "0"
{name}.uy_mean_function(x,y,z) = "0"
{name}.uz_mean_function(x,y,z) = "{drift}"
{name}.maxwellian_u_std_distribution_type = constant
{name}.ux_std = {vth / C}
{name}.uy_std = {vth / C}
{name}.uz_std = {vth / C}
{extra}""" for name, q, m, ppc, density, vth, drift, extra in species)


def oscillation(t, energy, omega_guess):
    """Frequency and damping rate from successive maxima of the E-field energy (oscillates at 2 omega)."""
    peaks, _ = find_peaks(energy, distance=max(int(0.7 * np.pi / omega_guess / (t[1] - t[0])), 1))
    peaks = [i for i in peaks if energy[i] > 3e-3 * energy[peaks[0]]][:12]
    omega = np.pi / np.mean(np.diff(t[peaks]))
    gamma = np.polyfit(t[peaks], np.log(energy[peaks]), 1)[0] / 2
    return omega, gamma


# ----------------------------- run the benchmarks -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = {}

ts = TWO_STREAM
omega_b = np.sqrt(ts["n_beam"] * E**2 / (EPS * ME))
length = 2 * np.pi * ts["v0"] / (np.sqrt(3) / 2 * omega_b)
dt = 0.99 * (length / ts["cells"]) / C
print(f"[1/2] Two-stream instability: omega_b = {omega_b:.3e} 1/s", flush=True)
beam = lambda name, v: {"name": name, "N_particles": ts["cells"] * ts["ppc"], "charge": -E, "mass": ME,
                        "vth": 0.01 * ts["v0"], "initial_vx": v, "number_density": ts["n_beam"]}
start = time.perf_counter()
results["two_stream"] = {"PyPIC3D": run_pypic3d("two_stream", {
    "simulation_parameters": {"name": "two_stream", "solver": "electrodynamic_yee", "t_wind": ts["t_end"] / omega_b,
                              "Nx": ts["cells"], "Ny": 1, "Nz": 1, "x_wind": length, "y_wind": 1.0, "z_wind": 1.0,
                              "cfl": 0.99, "shape_factor": 1, "current_calculation": "j_from_rhov", "filter_j": "none"},
    "plotting": {"plotting_interval": 5},
    "particle1": beam("beam_plus", ts["v0"]), "particle2": beam("beam_minus", -ts["v0"]),
    "particle3": {"name": "ions", "N_particles": ts["cells"] * ts["ppc"], "charge": E, "mass": MP, "vth": 1.0,
                  "number_density": 2 * ts["n_beam"]}})}
print(f"      PyPIC3D {time.perf_counter() - start:.0f} s; WarpX ...", flush=True)
results["two_stream"]["WarpX"] = run_warpx("two_stream", warpx_deck(length, ts["cells"], dt, int(ts["t_end"] / omega_b / dt), [
    ("beamp", -E, ME, ts["ppc"], ts["n_beam"], 0.01 * ts["v0"], ts["v0"] / C, ""),
    ("beamm", -E, ME, ts["ppc"], ts["n_beam"], 0.01 * ts["v0"], -ts["v0"] / C, ""),
    ("ions", E, MP, ts["ppc"], 2 * ts["n_beam"], 0.0, 0.0, "")]))

print(f"[2/2] Langmuir waves: omega_p = {OMEGA_P:.3e} 1/s, lambda_D = {LAMBDA_D:.3e} m, v_th = {VTH_OVER_C} c", flush=True)
rng = np.random.default_rng(0)
for kl in K_LAMBDA_D:
    k = kl / LAMBDA_D
    length = 2 * np.pi / k
    dt = 0.99 * (length / CELLS) / C
    n = CELLS * PPC
    x = rng.uniform(-length / 2, length / 2, n)
    u_wave = AMPLITUDE * OMEGA_P / k                      # continuity: dn/n = k U / omega
    vx = VTH * rng.standard_normal(n) + u_wave * np.sin(k * x)
    start = time.perf_counter()
    case = f"langmuir_k{kl:g}"
    print(f"  k lambda_D = {kl:g}: PyPIC3D ...", flush=True)
    results[case] = {"PyPIC3D": run_pypic3d(case, {
        "simulation_parameters": {"name": case, "solver": "electrodynamic_yee", "t_wind": T_END_WP / OMEGA_P,
                                  "Nx": CELLS, "Ny": 1, "Nz": 1, "x_wind": length, "y_wind": 1.0, "z_wind": 1.0,
                                  "cfl": 0.99, "shape_factor": 1, "current_calculation": "j_from_rhov", "filter_j": "none",
                                  "relativistic": True},
        "plotting": {"plotting_interval": 5},
        "particle1": {"name": "electrons", "N_particles": n, "charge": -E, "mass": ME, "vth": VTH, "number_density": DENSITY,
                      "initial_x": x, "initial_vx": vx},
        "particle2": {"name": "ions", "N_particles": n, "charge": E, "mass": MP, "vth": 0.0, "number_density": DENSITY,
                      "update_x": False, "update_y": False, "update_z": False}})}
    print(f"  k lambda_D = {kl:g}: PyPIC3D {time.perf_counter() - start:.0f} s; WarpX ...", flush=True)
    results[case]["WarpX"] = run_warpx(case, warpx_deck(length, CELLS, dt, int(T_END_WP / OMEGA_P / dt), [
        ("electrons", -E, ME, PPC, DENSITY, VTH, f"{u_wave / C}*sin({k}*z)", ""),
        ("ions", E, MP, PPC, DENSITY, 0.0, 0.0, "ions.do_not_push = 1\n")]))

# ----------------------------- plot, save, and print -----------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))
style = {"PyPIC3D": dict(color="C0", lw=2), "WarpX": dict(color="C1", lw=0, marker="o", ms=4)}
summary = []

for code, (t, field, _) in results["two_stream"].items():
    peak = np.argmax(field)
    window = (field > 1e-3 * field[peak]) & (field < 0.3 * field[peak]) & (t < t[peak])
    rate = np.polyfit(t[window], np.log(field[window]), 1)[0] / 2 / omega_b
    summary.append(f"two-stream       {code:8s} gamma = {rate:.3f} omega_b (theory 0.500), error {100 * abs(rate / 0.5 - 1):.1f}%")
    ax[0, 0].semilogy(t * omega_b, field / field.max(), label=code, **{**style[code], "markevery": 4})
    if code == "PyPIC3D":
        t_fit = t[window] * omega_b
        ax[0, 0].semilogy(t_fit, field[window][0] / field.max() * np.exp(t_fit - t_fit[0]), "k--", lw=1.5,
                          label=r"theory $e^{2\gamma t}$, $\gamma=\omega_b/2$")
ax[0, 0].set(xlabel=r"$\omega_b t$", ylabel="E-field energy / max", title="(a) Two-stream instability", ylim=(1e-6, 2))

k_theory = np.linspace(0.05, 0.55, 100)
roots = np.array([kinetic_root(k) for k in k_theory])
ax[0, 1].plot(k_theory, roots.real, "k-", lw=1.5, label="kinetic theory")
ax[0, 1].plot(k_theory, np.sqrt(1 + 3 * k_theory**2), "k:", lw=1.2, label="Bohm-Gross")
for code in ("PyPIC3D", "WarpX"):
    measured = [oscillation(*results[f"langmuir_k{kl:g}"][code][:2], OMEGA_P * np.sqrt(1 + 3 * kl**2)) for kl in K_LAMBDA_D]
    w = np.array([m[0] for m in measured]) / OMEGA_P
    ax[0, 1].plot(K_LAMBDA_D, w, "o" if code == "PyPIC3D" else "s", ms=9 if code == "PyPIC3D" else 6,
                  mfc="none" if code == "PyPIC3D" else style[code]["color"], color=style[code]["color"], mew=2, label=code)
    for kl, wk in zip(K_LAMBDA_D, w):
        summary.append(f"Langmuir k={kl:g}   {code:8s} omega = {wk:.4f} omega_p (kinetic {kinetic_root(kl).real:.4f}), "
                       f"error {100 * abs(wk / kinetic_root(kl).real - 1):.2f}%")
ax[0, 1].set(xlabel=r"$k\lambda_D$", ylabel=r"$\omega/\omega_p$", title="(b) Langmuir-wave dispersion")

root = kinetic_root(0.5)
for code, (t, field, total) in ((c, results["langmuir_k0.5"][c]) for c in ("PyPIC3D", "WarpX")):
    w, g = oscillation(t, field, OMEGA_P * np.sqrt(1.75))
    summary.append(f"Landau k=0.5     {code:8s} gamma = {g / OMEGA_P:.4f} omega_p (kinetic {root.imag:.4f}), "
                   f"error {100 * abs(g / OMEGA_P / root.imag - 1):.1f}%")
    ax[1, 0].semilogy(t * OMEGA_P, field / field.max(), label=code, **{**style[code], "markevery": 3})
    drift = (total - total[0]) / total[0]
    ax[1, 1].plot(t * OMEGA_P, 100 * drift, label=code, **{**style[code], "markevery": 10})
    summary.append(f"energy k=0.5     {code:8s} max |dE_total / E_total| = {100 * np.max(np.abs(drift)):.3f}%")
t_theory = np.linspace(0, T_END_WP, 200)
ax[1, 0].semilogy(t_theory, np.exp(2 * root.imag * t_theory), "k--", lw=1.5, label=rf"theory $e^{{2\gamma t}}$, $\gamma={root.imag:.4f}\,\omega_p$")
ax[1, 0].set(xlabel=r"$\omega_p t$", ylabel="E-field energy / max", title=r"(c) Landau damping, $k\lambda_D = 0.5$", ylim=(1e-4, 2))
ax[1, 1].axhline(0, color="k", lw=1)
ax[1, 1].set(xlabel=r"$\omega_p t$", ylabel="total energy change [%]", title=r"(d) Energy conservation, $k\lambda_D = 0.5$")
for a in ax.flat:
    a.legend(fontsize=9)
    a.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "warpx_collisionless.png"), dpi=300)
np.savez(os.path.join(OUTPUT_DIR, "warpx_collisionless.npz"),
         **{f"{case}_{code}": np.column_stack(v) for case, r in results.items() for code, v in r.items()})
print("\n".join(summary))
print(f"Saved {OUTPUT_DIR}/warpx_collisionless.png and .npz")
