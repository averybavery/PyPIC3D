"""Candidate collisionless benchmarks of PyPIC3D against WarpX and exact theory (identical inputs):
  1. electromagnetic wave in a cold plasma: omega^2 = omega_p^2 + c^2 k^2      (1D, seeded transverse velocity)
  2. single-electron gyration with E x B drift: exact trochoid                 (3D, external E and B)
  3. Weibel instability of a bi-Maxwellian: exact kinetic growth rate          (1D, k along the cold direction)
Run on Linux/WSL with WarpX on PATH:  python benchmarks/warpx_candidates.py   (--plot redraws from the .npz)"""
import os
import shutil
import subprocess
import sys
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
import openpmd_api as io
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.special import wofz

from PyPIC3D.__main__ import run_PyPIC3D
from PyPIC3D.initialization import initialize_simulation

jax.config.update("jax_enable_x64", True)

# ----------------------------- input parameters -----------------------------
EM_WAVE = dict(density=1.0e16, ck_over_wp=[0.5, 1.0, 2.0, 3.0], u_over_c=1e-3, cells=64, ppc=50, t_end=40.0)
GYRATION = dict(b0=1.0e-3, e_over_cb=0.005, v_over_c=0.02, periods=3.0, steps_per_period=60)  # E/(cB), speed / c
WEIBEL = dict(density=1.0e16, vth_cold=0.02, vth_hot=0.2, cells=64, ppc=500, growth_times=14.0)  # vth / c
OUTPUT_DIR = "outputs/warpx_candidates"
WARPX = {d: shutil.which(f"warpx.{d}d") for d in (1, 3)}

E, ME, MP, EPS, C = 1.602176634e-19, 9.1093837e-31, 1.67262192e-27, 8.8541878128e-12, 2.99792458e8
Z = lambda x: 1j * np.sqrt(np.pi) * wofz(x)  # plasma dispersion function


def weibel_growth(k, wp, vx, vy):
    """Purely growing Weibel root gamma for k along x (thermal speed vx), hot along y (vy); 0 if stable."""
    f = lambda g: (-(g**2) - (C * k) ** 2 - wp**2 + wp**2 * (vy / vx) ** 2
                   * (1 + (1j * g / (np.sqrt(2) * k * vx)) * Z(1j * g / (np.sqrt(2) * k * vx)))).real
    return brentq(f, 1e-6 * wp, 5 * wp) if f(1e-6 * wp) > 0 else 0.0


def pypic3d_energies(name, config):
    """Run PyPIC3D (array initial_* entries are saved as .npy); return time, E-field and B-field energy."""
    out = os.path.join(OUTPUT_DIR, name, "pypic3d")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    config["simulation_parameters"].update(name=name, output_dir=out, Ny=1, Nz=1, y_wind=1.0, z_wind=1.0)
    for species in (v for k, v in config.items() if k.startswith("particle")):
        for key in [k for k, v in species.items() if isinstance(v, np.ndarray)]:
            np.save(os.path.join(out, f"{species['name']}_{key}.npy"), species[key])
            species[key] = os.path.join(out, f"{species['name']}_{key}.npy")
    start = time.perf_counter()
    run_PyPIC3D(config)
    print(f"      PyPIC3D {name}: {time.perf_counter() - start:.0f} s", flush=True)
    load = lambda f: np.loadtxt(os.path.join(out, "data", f), delimiter=",", ndmin=2)
    e, b = load("electric_field_energy.txt"), load("magnetic_field_energy.txt")
    return e[:, 0], e[:, 1], b[:, 1]


def warpx_run(name, dims, inputs):
    out = os.path.abspath(os.path.join(OUTPUT_DIR, name, "warpx"))
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    with open(os.path.join(out, "inputs"), "w") as handle:
        handle.write(inputs)
    log = subprocess.run([WARPX[dims], "inputs"], cwd=out, capture_output=True, text=True)
    if log.returncode:
        raise RuntimeError(log.stdout[-3000:] + log.stderr[-3000:])
    return out


def warpx_energies(name, length, cells, dt, steps, species, interval=5):
    """1D WarpX run (WarpX z = PyPIC3D x, WarpX x = PyPIC3D y); species = (name, q, m, ppc, n, (vx, vy, vz), ux_mean)."""
    deck = f"""max_step = {steps}
amr.n_cell = {cells}
amr.max_level = 0
geometry.dims = 1
geometry.prob_lo = {-length / 2}
geometry.prob_hi = {length / 2}
boundary.field_lo = periodic
boundary.field_hi = periodic
boundary.particle_lo = periodic
boundary.particle_hi = periodic
warpx.const_dt = {dt}
warpx.use_filter = 0
algo.current_deposition = direct
algo.particle_shape = 1
particles.species_names = {" ".join(s[0] for s in species)}
warpx.reduced_diags_names = FE
FE.type = FieldEnergy
FE.intervals = {interval}
""" + "".join(f"""{n}.charge = {q}
{n}.mass = {m}
{n}.injection_style = NRandomPerCell
{n}.num_particles_per_cell = {ppc}
{n}.profile = constant
{n}.density = {dens}
{n}.momentum_distribution_type = maxwellian
{n}.maxwellian_u_mean_distribution_type = parser
{n}.ux_mean_function(x,y,z) = "{ux}"
{n}.uy_mean_function(x,y,z) = "0"
{n}.uz_mean_function(x,y,z) = "0"
{n}.maxwellian_u_std_distribution_type = constant
{n}.ux_std = {v[1] / C}
{n}.uy_std = {v[2] / C}
{n}.uz_std = {v[0] / C}
{extra}""" for n, q, m, ppc, dens, v, ux, extra in species)
    fe = np.loadtxt(os.path.join(warpx_run(name, 1, deck), "diags/reducedfiles/FE.txt"))
    return fe[:, 1], fe[:, 3], fe[:, 4]


def frequency(t, energy, omega_guess):
    """omega from successive maxima of a field energy that oscillates at 2 omega."""
    peaks, _ = find_peaks(energy, distance=max(int(0.6 * np.pi / omega_guess / (t[1] - t[0])), 1))
    return np.pi / np.mean(np.diff(t[peaks]))


def growth(t, energy):
    """Exponential growth rate (of the amplitude) over the linear phase, 1e-3 to 0.3 of the peak."""
    peak = np.argmax(energy)
    window = (energy > 1e-3 * energy[peak]) & (energy < 0.3 * energy[peak]) & (t < t[peak])
    return np.polyfit(t[window], np.log(energy[window]), 1)[0] / 2


# ----------------------------- run the benchmarks -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(0)
em, gy, wb = EM_WAVE, GYRATION, WEIBEL
wp_em = np.sqrt(em["density"] * E**2 / (EPS * ME))
wp_wb = np.sqrt(wb["density"] * E**2 / (EPS * ME))
omega_c = E * gy["b0"] / ME
results = {}

if "--plot" in sys.argv:
    saved = np.load(os.path.join(OUTPUT_DIR, "warpx_candidates.npz"))
    results = {key: saved[key] for key in saved.files}
else:
    print(f"[1/3] EM waves in a cold plasma (omega_p = {wp_em:.3e} 1/s)", flush=True)
    for ck in em["ck_over_wp"]:
        k = ck * wp_em / C
        length = 2 * np.pi / k
        dt = 0.99 * length / em["cells"] / C
        n = em["cells"] * em["ppc"]
        x = rng.uniform(-length / 2, length / 2, n)
        u = em["u_over_c"] * C
        vth = 1e-4 * C
        case = f"em_ck{ck:g}"
        results[f"{case}_PyPIC3D"] = np.column_stack(pypic3d_energies(case, {
            "simulation_parameters": {"solver": "electrodynamic_yee", "t_wind": em["t_end"] / wp_em, "Nx": em["cells"],
                                      "x_wind": length, "cfl": 0.99, "shape_factor": 1, "current_calculation": "j_from_rhov",
                                      "filter_j": "none"},
            "plotting": {"plotting_interval": 2},
            "particle1": {"name": "electrons", "N_particles": n, "charge": -E, "mass": ME, "vth": vth,
                          "number_density": em["density"], "initial_x": x,
                          "initial_vy": vth * rng.standard_normal(n) + u * np.sin(k * x)},
            "particle2": {"name": "ions", "N_particles": n, "charge": E, "mass": MP, "vth": 0.0,
                          "number_density": em["density"], "update_x": False, "update_y": False, "update_z": False}}))
        results[f"{case}_WarpX"] = np.column_stack(warpx_energies(case, length, em["cells"], dt, int(em["t_end"] / wp_em / dt), [
            ("electrons", -E, ME, em["ppc"], em["density"], (vth,) * 3, f"{u / C}*sin({k}*z)", ""),
            ("ions", E, MP, em["ppc"], em["density"], (0.0,) * 3, "0", "ions.do_not_push = 1\n")], interval=2))

    print("[2/3] Single-electron gyration with E x B drift", flush=True)
    dt = 2 * np.pi / omega_c / gy["steps_per_period"]
    steps = int(gy["periods"] * gy["steps_per_period"])
    e0, v0 = gy["e_over_cb"] * C * gy["b0"], gy["v_over_c"] * C
    length, cells = 400 * v0 / omega_c, 8
    out = os.path.join(OUTPUT_DIR, "gyration", "pypic3d")
    os.makedirs(out, exist_ok=True)
    for name, value in (("ey", e0), ("bz", gy["b0"])):
        np.save(os.path.join(out, f"{name}.npy"), np.full((cells,) * 3, value))
    for name, value in (("x", 0.0), ("y", 0.0), ("z", 0.0), ("vx", v0), ("vy", 0.0), ("vz", 0.0)):
        np.save(os.path.join(out, f"{name}.npy"), np.array([value]))
    loop, particles, fields, static, dynamic, *_, species_config = initialize_simulation({
        "simulation_parameters": {"name": "gyration", "output_dir": out, "solver": "electrodynamic_yee", "Nt": steps, "dt": dt,
                                  "Nx": cells, "Ny": cells, "Nz": cells, "x_wind": length, "y_wind": length, "z_wind": length,
                                  "shape_factor": 1, "current_calculation": "j_from_rhov", "filter_j": "none"},
        "plotting": {"plotting_interval": 10**9},
        "particle1": {"name": "electron", "N_particles": 1, "charge": -E, "mass": ME, "vth": 0.0, "weight": 1.0,
                      **{f"initial_{k}": os.path.join(out, f"{k}.npy") for k in ("x", "y", "z", "vx", "vy", "vz")}},
        "field1": {"name": "Ey", "type": 1, "path": os.path.join(out, "ey.npy"), "evolve": False},
        "field2": {"name": "Bz", "type": 5, "path": os.path.join(out, "bz.npy"), "evolve": False}})
    step = jax.jit(lambda p, f: loop(p, species_config, f, static, dynamic))
    track = []
    for i in range(steps + 1):
        slot = np.argmax(np.asarray(particles.active).reshape(-1))
        track.append(np.asarray(particles.x).reshape(-1, 3)[slot])
        if i < steps:
            particles, fields = step(particles, fields)
    results["gyration_PyPIC3D"] = np.column_stack((np.arange(steps + 1) * dt, np.array(track)))
    out = warpx_run("gyration", 3, f"""max_step = {steps}
amr.n_cell = {cells} {cells} {cells}
amr.max_level = 0
geometry.dims = 3
geometry.prob_lo = {-length / 2} {-length / 2} {-length / 2}
geometry.prob_hi = {length / 2} {length / 2} {length / 2}
boundary.field_lo = periodic periodic periodic
boundary.field_hi = periodic periodic periodic
boundary.particle_lo = periodic periodic periodic
boundary.particle_hi = periodic periodic periodic
warpx.const_dt = {dt}
algo.particle_pusher = boris
algo.particle_shape = 1
particles.species_names = electron
particles.E_ext_particle_init_style = constant
particles.E_external_particle = 0. {e0} 0.
particles.B_ext_particle_init_style = constant
particles.B_external_particle = 0. 0. {gy['b0']}
electron.species_type = electron
electron.injection_style = SingleParticle
electron.single_particle_pos = 0. 0. 0.
electron.single_particle_u = {v0 / C} 0. 0.
electron.single_particle_weight = 1.0
diagnostics.diags_names = diag1
diag1.intervals = 1
diag1.diag_type = Full
diag1.format = openpmd
diag1.openpmd_backend = h5
diag1.fields_to_plot = none
""")
    series = io.Series(os.path.join(out, "diags/diag1/openpmd_%T.h5"), io.Access.read_only)
    rows = []
    for index, it in series.iterations.items():
        p = it.particles["electron"]
        chunks = [p["position"][d].load_chunk() for d in "xyz"] + [p["positionOffset"][d].load_chunk() for d in "xyz"]
        series.flush()
        rows.append([it.time * it.time_unit_SI] + [float(chunks[d][0] + chunks[d + 3][0]) for d in range(3)])
    results["gyration_WarpX"] = np.array(sorted(rows))

    vx, vy = wb["vth_cold"] * C, wb["vth_hot"] * C
    ks = np.linspace(0.05, 3.0, 120) * wp_wb / C
    k_max = ks[np.argmax([weibel_growth(k, wp_wb, vx, vy) for k in ks])]
    gamma_max = weibel_growth(k_max, wp_wb, vx, vy)
    length = 2 * np.pi / k_max
    dt = 0.99 * length / wb["cells"] / C
    t_end = wb["growth_times"] / gamma_max
    n = wb["cells"] * wb["ppc"]
    print(f"[3/3] Weibel instability: gamma_max = {gamma_max / wp_wb:.4f} omega_p at ck/omega_p = {k_max * C / wp_wb:.3f}", flush=True)
    results["weibel_PyPIC3D"] = np.column_stack(pypic3d_energies("weibel", {
        "simulation_parameters": {"solver": "electrodynamic_yee", "t_wind": t_end, "Nx": wb["cells"], "x_wind": length,
                                  "cfl": 0.99, "shape_factor": 1, "current_calculation": "j_from_rhov", "filter_j": "none"},
        "plotting": {"plotting_interval": 5},
        "particle1": {"name": "electrons", "N_particles": n, "charge": -E, "mass": ME, "vth": vx, "number_density": wb["density"],
                      "initial_x": rng.uniform(-length / 2, length / 2, n), "initial_vx": vx * rng.standard_normal(n),
                      "initial_vy": vy * rng.standard_normal(n)},
        "particle2": {"name": "ions", "N_particles": n, "charge": E, "mass": MP, "vth": 0.0, "number_density": wb["density"],
                      "update_x": False, "update_y": False, "update_z": False}}))
    results["weibel_WarpX"] = np.column_stack(warpx_energies("weibel", length, wb["cells"], dt, int(t_end / dt), [
        ("electrons", -E, ME, wb["ppc"], wb["density"], (vx, vy, vx), "0", ""),
        ("ions", E, MP, wb["ppc"], wb["density"], (0.0,) * 3, "0", "ions.do_not_push = 1\n")]))
    results["weibel_theory"] = np.array([[k_max, gamma_max, wp_wb]])
    np.savez(os.path.join(OUTPUT_DIR, "warpx_candidates.npz"), **results)

# ----------------------------- plot, save, and print -----------------------------
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
style = {"PyPIC3D": dict(color="C0", ms=9, mfc="none", mew=2, marker="o", lw=0),
         "WarpX": dict(color="C1", ms=5, marker="s", lw=0)}
summary = []

ck = np.linspace(0, 3.3, 100)
ax[0].plot(ck, np.sqrt(1 + ck**2), "k-", lw=1.5, label=r"theory $\omega^2=\omega_p^2+c^2k^2$")
ax[0].plot(ck, ck, "k:", lw=1, label="light line")
for code in ("PyPIC3D", "WarpX"):
    w = [frequency(*results[f"em_ck{c:g}_{code}"][:, :2].T, wp_em * np.sqrt(1 + c**2)) / wp_em for c in em["ck_over_wp"]]
    ax[0].plot(em["ck_over_wp"], w, label=code, **style[code])
    for c, wc in zip(em["ck_over_wp"], w):
        summary.append(f"EM wave  ck/wp={c:g}  {code:8s} omega = {wc:.4f} omega_p (theory {np.sqrt(1 + c**2):.4f}), "
                       f"error {100 * abs(wc / np.sqrt(1 + c**2) - 1):.2f}%")
ax[0].set(xlabel=r"$ck/\omega_p$", ylabel=r"$\omega/\omega_p$", title="Electromagnetic waves in a plasma")

t = results["gyration_PyPIC3D"][:, 0]
e0, v0 = gy["e_over_cb"] * C * gy["b0"], gy["v_over_c"] * C
vd = e0 / gy["b0"]                                               # E x B drift along +x
x_ex = vd * t + (v0 - vd) / omega_c * np.sin(omega_c * t)        # electron gyrates counter-clockwise about +z
y_ex = (v0 - vd) / omega_c * (1 - np.cos(omega_c * t))
r_l = abs(v0 - vd) / omega_c
ax[1].plot(x_ex / r_l, y_ex / r_l, "k-", lw=1.5, label="exact")
for code in ("PyPIC3D", "WarpX"):
    tr = results[f"gyration_{code}"]
    xe = vd * tr[:, 0] + (v0 - vd) / omega_c * np.sin(omega_c * tr[:, 0])
    ye = (v0 - vd) / omega_c * (1 - np.cos(omega_c * tr[:, 0]))
    error = np.max(np.hypot(tr[:, 1] - xe, tr[:, 2] - ye)) / r_l
    summary.append(f"gyration {code:8s} max position error = {100 * error:.2f}% of the Larmor radius over {gy['periods']:g} periods")
    ax[1].plot(tr[::2, 1] / r_l, tr[::2, 2] / r_l, label=code, **{**style[code], "ms": style[code]["ms"] - 2})
ax[1].set(xlabel=r"$x / r_L$", ylabel=r"$y / r_L$", title=r"Gyration and $\mathbf{E}\times\mathbf{B}$ drift", aspect="equal",
          ylim=(-1, 3.5))

k_max, gamma_max, wp = results["weibel_theory"][0]
for code in ("PyPIC3D", "WarpX"):
    t, _, b = results[f"weibel_{code}"].T
    g = growth(t, b)
    summary.append(f"Weibel   {code:8s} gamma = {g / wp:.4f} omega_p (kinetic theory {gamma_max / wp:.4f}), "
                   f"error {100 * abs(g / gamma_max - 1):.1f}%")
    ax[2].semilogy(t * gamma_max, b / b.max(), color=style[code]["color"], lw=2 if code == "PyPIC3D" else 0,
                   marker=None if code == "PyPIC3D" else "s", ms=3, markevery=3, label=code)
    if code == "PyPIC3D":
        peak = np.argmax(b)
        window = (b > 1e-3 * b[peak]) & (b < 0.3 * b[peak]) & (t < t[peak])
        tf = t[window] * gamma_max
        ax[2].semilogy(tf, b[window][0] / b.max() * np.exp(2 * (tf - tf[0])), "k--", lw=1.5,
                       label=rf"theory $e^{{2\gamma t}}$, $\gamma = {gamma_max / wp:.3f}\,\omega_p$")
ax[2].set(xlabel=r"$\gamma_{\max} t$", ylabel="B-field energy / max", title="Weibel instability", ylim=(1e-6, 2))
for a in ax:
    a.legend(fontsize=9, loc="upper left")
    a.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "warpx_candidates.png"), dpi=300)
print("\n".join(summary))
print(f"Saved {OUTPUT_DIR}/warpx_candidates.png and .npz")
