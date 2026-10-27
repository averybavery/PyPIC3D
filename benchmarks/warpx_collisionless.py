"""Collisionless physics benchmarks of PyPIC3D against WarpX and exact theory (1D PIC, identical inputs):
  (a) two-stream instability growth rate     cold-beam theory, gamma = omega_b / 2
  (b) WarpX Langmuir test                    Examples/Tests/langmuir/inputs_test_1d_langmuir_multi (run longer);
                                             analytic E = eps m c^2 k / e sin(kz) sin(omega_p t), WarpX tolerance 5%
  (c) Landau damping at k lambda_D = 0.5     exact kinetic rate gamma = -0.1534 omega_p
  (d) electromagnetic waves in a cold plasma   omega^2 = omega_p^2 + c^2 k^2 (seeded transverse velocity)
Run on Linux/WSL with WarpX on PATH:  python benchmarks/warpx_collisionless.py   (--plot redraws from the saved .npz)"""
import os
import shutil
import sys
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
TWO_STREAM = dict(n_beam=1.0e15, v0=1.5e7, cells=64, ppc=200, t_end=25.0)       # t_end in 1 / omega_b
LANGMUIR = dict(n0=2.0e24, eps=0.01, length=40e-6, cells=128, ppc=2, steps=400)  # WarpX test (80 steps there)
LANDAU = dict(density=1.0e16, vth_over_c=0.05, k_lambda_d=0.5, cells=32, ppc=2000, amplitude=0.05, t_end=30.0)
EM_WAVE = dict(density=1.0e16, ck_over_wp=[0.5, 1.0, 2.0, 3.0], u_over_c=1e-3, cells=64, ppc=50, t_end=40.0)
SEEDS = range(5)                         # independent random seeds; figures show mean +/- 1 std
OUTPUT_DIR = "outputs/warpx_collisionless"
WARPX = shutil.which("warpx.1d")

E, ME, MP, EPS, C = 1.602176634e-19, 9.1093837e-31, 1.67262192e-27, 8.8541878128e-12, 2.99792458e8


def kinetic_root(k):
    """Exact Maxwellian Langmuir root omega / omega_p for k lambda_D = k."""
    f = lambda w: 1 + (1 + w / (np.sqrt(2) * k) * 1j * np.sqrt(np.pi) * wofz(w / (np.sqrt(2) * k))) / k**2
    return newton(f, np.sqrt(1 + 3 * k * k) + 0j)


def run_pypic3d(name, config):
    """Run PyPIC3D (arrays in initial_x / initial_vx are saved as .npy); return (time, E-field energy, total energy)."""
    out = os.path.join(OUTPUT_DIR, name, "pypic3d")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    config["simulation_parameters"].update(name=name, output_dir=out, Ny=1, Nz=1, y_wind=1.0, z_wind=1.0)
    for species in (v for k, v in config.items() if k.startswith("particle")):
        for key in ("initial_x", "initial_vx"):
            if isinstance(species.get(key), np.ndarray):
                np.save(os.path.join(out, f"{species['name']}_{key}.npy"), species[key])
                species[key] = os.path.join(out, f"{species['name']}_{key}.npy")
    start = time.perf_counter()
    run_PyPIC3D(config)
    print(f"      PyPIC3D {name}: {time.perf_counter() - start:.0f} s", flush=True)
    load = lambda f: np.loadtxt(os.path.join(out, "data", f), delimiter=",", ndmin=2)
    field, total = load("electric_field_energy.txt"), load("total_energy.txt")
    return field[:, 0], field[:, 1], total[:, 1]


def run_warpx(name, inputs, interval=5):
    """Run a WarpX 1D deck; return (time, E-field energy, total energy)."""
    out = os.path.abspath(os.path.join(OUTPUT_DIR, name, "warpx"))
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    with open(os.path.join(out, "inputs"), "w") as handle:
        handle.write(inputs + f"\nwarpx.reduced_diags_names = FE PE\nFE.type = FieldEnergy\nFE.intervals = {interval}\n"
                              f"PE.type = ParticleEnergy\nPE.intervals = {interval}\n")
    log = subprocess.run([WARPX, "inputs"], cwd=out, capture_output=True, text=True)
    if log.returncode:
        raise RuntimeError(log.stdout[-3000:] + log.stderr[-3000:])
    fe = np.loadtxt(os.path.join(out, "diags/reducedfiles/FE.txt"))
    pe = np.loadtxt(os.path.join(out, "diags/reducedfiles/PE.txt"))
    return fe[:, 1], fe[:, 3], fe[:, 2] + pe[:, 2]


def warpx_deck(length, cells, steps, species, extra=""):
    """1D periodic WarpX deck; species = (name, charge, mass, ppc, density, vth, mean_uz_expression)."""
    return f"""max_step = {steps}
amr.n_cell = {cells}
amr.max_level = 0
geometry.dims = 1
geometry.prob_lo = {-length / 2}
geometry.prob_hi = {length / 2}
boundary.field_lo = periodic
boundary.field_hi = periodic
boundary.particle_lo = periodic
boundary.particle_hi = periodic
warpx.use_filter = 0
{extra}
particles.species_names = {" ".join(s[0] for s in species)}
""" + "".join(f"""{name}.charge = {q}
{name}.mass = {m}
{name}.injection_style = NRandomPerCell
{name}.num_particles_per_cell = {ppc}
{name}.profile = constant
{name}.density = {density}
{name}.momentum_distribution_type = maxwellian
{name}.maxwellian_u_mean_distribution_type = parser
{name}.ux_mean_function(x,y,z) = "{drift[0] if isinstance(drift, tuple) else 0}"
{name}.uy_mean_function(x,y,z) = "0"
{name}.uz_mean_function(x,y,z) = "{drift[1] if isinstance(drift, tuple) else drift}"
{name}.maxwellian_u_std_distribution_type = constant
{name}.ux_std = {vth / C}
{name}.uy_std = {vth / C}
{name}.uz_std = {vth / C}
""" for name, q, m, ppc, density, vth, drift in species)


# ----------------------------- run the benchmarks -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
ts, lg, ld, em = TWO_STREAM, LANGMUIR, LANDAU, EM_WAVE
omega_b = np.sqrt(ts["n_beam"] * E**2 / (EPS * ME))
omega_l, k_l = np.sqrt(2 * lg["n0"] * E**2 / (EPS * ME)), 2 * np.pi / 20e-6
omega_p = np.sqrt(ld["density"] * E**2 / (EPS * ME))
wp_em = np.sqrt(em["density"] * E**2 / (EPS * ME))
warpx_langmuir = f"""max_step = {lg['steps']}
amr.n_cell = {lg['cells']}
amr.max_grid_size = 64
amr.max_level = 0
geometry.dims = 1
geometry.prob_lo = -20.e-6
geometry.prob_hi = 20.e-6
boundary.field_lo = periodic
boundary.field_hi = periodic
algo.field_gathering = energy-conserving
algo.current_deposition = esirkepov
warpx.use_filter = 0
algo.particle_shape = 1
warpx.cfl = 0.8
my_constants.epsilon = {lg['eps']}
my_constants.n0 = {lg['n0']}
my_constants.wp = sqrt(2.*n0*q_e**2/(epsilon0*m_e))
my_constants.kp = wp/clight
my_constants.k = 2.*pi/20.e-6
particles.species_names = electrons positrons
""" + "".join(f"""{name}.charge = {q}
{name}.mass = m_e
{name}.injection_style = "NUniformPerCell"
{name}.num_particles_per_cell_each_dim = {lg['ppc']}
{name}.zmin = -20.e-6
{name}.zmax = 20.e-6
{name}.profile = constant
{name}.density = n0
{name}.momentum_distribution_type = parse_momentum_function
{name}.momentum_function_ux(x,y,z) = "{sign}epsilon * k/kp * sin(k*x) * cos(k*y) * cos(k*z)"
{name}.momentum_function_uy(x,y,z) = "{sign}epsilon * k/kp * cos(k*x) * sin(k*y) * cos(k*z)"
{name}.momentum_function_uz(x,y,z) = "{sign}epsilon * k/kp * cos(k*x) * cos(k*y) * sin(k*z)"
""" for name, q, sign in (("electrons", "-q_e", ""), ("positrons", "q_e", "-")))


def run_seed(seed):
    """All four benchmarks for one random seed; returns {case: {code: (time, E-field energy, total energy)}}."""
    rng, results = np.random.default_rng(seed), {}
    pypic = lambda name, config: (np.random.seed(seed), run_pypic3d(f"{name}/seed{seed}", config))[1]  # NumPy draws thermal velocities
    warp = lambda name, deck, interval=5: run_warpx(f"{name}/seed{seed}", deck + f"\nwarpx.random_seed = {seed + 1}\n", interval)

    length = 2 * np.pi * ts["v0"] / (np.sqrt(3) / 2 * omega_b)
    dt = 0.99 * (length / ts["cells"]) / C
    n = ts["cells"] * ts["ppc"]
    print(f"[seed {seed}] Two-stream instability", flush=True)
    beam = lambda name, v: {"name": name, "N_particles": n, "charge": -E, "mass": ME, "vth": 0.01 * ts["v0"], "initial_vx": v,
                            "number_density": ts["n_beam"], "initial_x": rng.uniform(-length / 2, length / 2, n)}
    results["two_stream"] = {
        "PyPIC3D": pypic("two_stream", {
            "simulation_parameters": {"solver": "electrodynamic_yee", "t_wind": ts["t_end"] / omega_b, "Nx": ts["cells"],
                                      "x_wind": length, "cfl": 0.99, "shape_factor": 1, "current_calculation": "j_from_rhov",
                                      "filter_j": "none"},
            "plotting": {"plotting_interval": 5},
            "particle1": beam("beam_plus", ts["v0"]), "particle2": beam("beam_minus", -ts["v0"]),
            "particle3": {"name": "ions", "N_particles": n, "charge": E, "mass": MP, "vth": 1.0, "number_density": 2 * ts["n_beam"],
                          "initial_x": rng.uniform(-length / 2, length / 2, n)}}),
        "WarpX": warp("two_stream", warpx_deck(length, ts["cells"], int(ts["t_end"] / omega_b / dt), [
            ("beamp", -E, ME, ts["ppc"], ts["n_beam"], 0.01 * ts["v0"], ts["v0"] / C),
            ("beamm", -E, ME, ts["ppc"], ts["n_beam"], 0.01 * ts["v0"], -ts["v0"] / C),
            ("ions", E, MP, ts["ppc"], 2 * ts["n_beam"], 0.0, 0.0)], f"warpx.const_dt = {dt}\nalgo.current_deposition = direct\n"
                                                                     "algo.particle_shape = 1"))}

    print(f"[seed {seed}] WarpX Langmuir test (quiet start: only the tiny thermal noise changes with the seed)", flush=True)
    n = lg["cells"] * lg["ppc"]
    x = (np.arange(n) + 0.5) * lg["length"] / n - lg["length"] / 2      # NUniformPerCell quiet start
    u = lg["eps"] * k_l / (omega_l / C) * np.sin(k_l * x)                 # u = gamma v / c
    v = u * C / np.sqrt(1 + u**2)
    species = lambda name, q, vx: {"name": name, "N_particles": n, "charge": q, "mass": ME, "vth": 1e-3,
                                   "number_density": lg["n0"], "initial_x": x, "initial_vx": vx}
    results["langmuir"] = {
        "PyPIC3D": pypic("langmuir", {
            "simulation_parameters": {"solver": "electrodynamic_yee", "Nt": lg["steps"], "Nx": lg["cells"], "x_wind": lg["length"],
                                      "cfl": 0.8, "shape_factor": 1, "current_calculation": "esirkepov", "filter_j": "none"},
            "plotting": {"plotting_interval": 2},
            "particle1": species("electrons", -E, v), "particle2": species("positrons", E, -v)}),
        "WarpX": warp("langmuir", warpx_langmuir, interval=2)}

    vth = ld["vth_over_c"] * C
    k = ld["k_lambda_d"] * omega_p / vth
    length = 2 * np.pi / k
    dt = 0.99 * (length / ld["cells"]) / C
    print(f"[seed {seed}] Landau damping", flush=True)
    n = ld["cells"] * ld["ppc"]
    x = rng.uniform(-length / 2, length / 2, n)
    u_wave = ld["amplitude"] * omega_p / k                                # continuity: dn/n = k U / omega
    results["landau"] = {
        "PyPIC3D": pypic("landau", {
            "simulation_parameters": {"solver": "electrodynamic_yee", "t_wind": ld["t_end"] / omega_p, "Nx": ld["cells"],
                                      "x_wind": length, "cfl": 0.99, "shape_factor": 1, "current_calculation": "j_from_rhov",
                                      "filter_j": "none"},
            "plotting": {"plotting_interval": 5},
            "particle1": {"name": "electrons", "N_particles": n, "charge": -E, "mass": ME, "vth": vth, "number_density": ld["density"],
                          "initial_x": x, "initial_vx": vth * rng.standard_normal(n) + u_wave * np.sin(k * x)},
            "particle2": {"name": "ions", "N_particles": n, "charge": E, "mass": MP, "vth": 0.0, "number_density": ld["density"],
                          "update_x": False, "update_y": False, "update_z": False}}),
        "WarpX": warp("landau", warpx_deck(length, ld["cells"], int(ld["t_end"] / omega_p / dt), [
            ("electrons", -E, ME, ld["ppc"], ld["density"], vth, f"{u_wave / C}*sin({k}*z)"),
            ("ions", E, MP, ld["ppc"], ld["density"], 0.0, 0.0)],
            f"warpx.const_dt = {dt}\nalgo.current_deposition = direct\nalgo.particle_shape = 1\nions.do_not_push = 1"))}

    print(f"[seed {seed}] Electromagnetic waves in a cold plasma", flush=True)
    for ck in em["ck_over_wp"]:
        k = ck * wp_em / C
        length = 2 * np.pi / k
        dt = 0.99 * length / em["cells"] / C
        n, u, vth = em["cells"] * em["ppc"], em["u_over_c"] * C, 1e-4 * C
        x = rng.uniform(-length / 2, length / 2, n)
        case = f"em{ck:g}"
        results[case] = {
            "PyPIC3D": pypic(case, {
                "simulation_parameters": {"solver": "electrodynamic_yee", "t_wind": em["t_end"] / wp_em, "Nx": em["cells"],
                                          "x_wind": length, "cfl": 0.99, "shape_factor": 1, "current_calculation": "j_from_rhov",
                                          "filter_j": "none"},
                "plotting": {"plotting_interval": 2},
                "particle1": {"name": "electrons", "N_particles": n, "charge": -E, "mass": ME, "vth": vth,
                              "number_density": em["density"], "initial_x": x,
                              "initial_vy": vth * rng.standard_normal(n) + u * np.sin(k * x)},
                "particle2": {"name": "ions", "N_particles": n, "charge": E, "mass": MP, "vth": 0.0,
                              "number_density": em["density"], "update_x": False, "update_y": False, "update_z": False}}),
            # transverse seed: PyPIC3D v_y (1D along x) corresponds to WarpX u_x (1D along z)
            "WarpX": warp(case, warpx_deck(length, em["cells"], int(em["t_end"] / wp_em / dt), [
                ("electrons", -E, ME, em["ppc"], em["density"], vth, (f"{u / C}*sin({k}*z)", "0")),
                ("ions", E, MP, em["ppc"], em["density"], 0.0, 0.0)],
                f"warpx.const_dt = {dt}\nalgo.current_deposition = direct\nalgo.particle_shape = 1\nions.do_not_push = 1"),
                interval=2)}
    return results


path = os.path.join(OUTPUT_DIR, "warpx_collisionless_seeds.npz")
if "--plot" in sys.argv:
    saved = np.load(path)
    runs = [{} for _ in SEEDS]
    for key in saved.files:
        case, code, seed = key.rsplit("|", 2)
        runs[int(seed)].setdefault(case, {})[code] = tuple(saved[key].T)
else:
    runs = [run_seed(seed) for seed in SEEDS]
    np.savez(path, **{f"{case}|{code}|{s}": np.column_stack(v) for s, r in enumerate(runs) for case, cr in r.items() for code, v in cr.items()})

# ----------------------------- plot, save, and print -----------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))
colors = {"PyPIC3D": "C0", "WarpX": "C1"}
summary = []
stats = lambda values: f"{np.mean(values):.4f} +/- {np.std(values):.4f}"


def band(a, x, curves, code):
    """Mean over seeds (line for PyPIC3D, markers for WarpX) with a shaded +/- 1 std band."""
    m, s = np.mean(curves, 0), np.std(curves, 0)
    a.fill_between(x, m - s, m + s, color=colors[code], alpha=0.25, lw=0)
    style = dict(lw=2) if code == "PyPIC3D" else dict(lw=0, marker="o", ms=3, markevery=4)
    a.plot(x, m, color=colors[code], label=rf"{code} (mean $\pm 1\sigma$)", **style)


for code in ("PyPIC3D", "WarpX"):
    t = runs[0]["two_stream"][code][0]
    curves, rates = [], []
    for r in runs:
        field = r["two_stream"][code][1]
        peak = np.argmax(field)
        window = (field > 3e-2 * field[peak]) & (field < 0.3 * field[peak]) & (t < t[peak])  # last decade before saturation, above the noise
        rates.append(np.polyfit(t[window], np.log(field[window]), 1)[0] / 2 / omega_b)
        curves.append(np.log10(field / field.max()))
    summary.append(f"(a) two-stream  {code:8s} gamma = {stats(rates)} omega_b (theory 0.5000), "
                   f"error {100 * (np.mean(rates) / 0.5 - 1):+.1f}% +/- {100 * np.std(rates) / 0.5:.1f}%")
    band(ax[0, 0], t * omega_b, curves, code)
t_fit = np.linspace(2, 7, 50)
ax[0, 0].plot(t_fit, (t_fit - 7) / np.log(10) - 0.5, "k--", lw=1.5, label=r"theory slope $e^{2\gamma t}$, $\gamma=\omega_b/2$")
ax[0, 0].set(xlabel=r"$\omega_b t$", ylabel=r"$\log_{10}$(E-field energy / max)", title="Two-stream instability", ylim=(-6, 0.3))

amplitude = lg["eps"] * ME * C**2 * k_l / E
for code in ("PyPIC3D", "WarpX"):
    t = runs[0]["langmuir"][code][0]
    theory = EPS / 2 * amplitude**2 * np.sin(omega_l * t) ** 2 * lg["length"] / 2
    curves = [r["langmuir"][code][1] / theory.max() for r in runs]
    errors = [np.max(np.abs(c - theory / theory.max())) for c in curves]
    summary.append(f"(b) Langmuir    {code:8s} max |W - W_theory| / W_max = {100 * np.mean(errors):.2f}% +/- {100 * np.std(errors):.2f}%"
                   " (WarpX test tolerance 5%)")
    band(ax[0, 1], t * omega_l, curves, code)
ax[0, 1].plot(t * omega_l, np.sin(omega_l * t) ** 2, "k--", lw=1.2, label=r"theory $\sin^2\omega_p t$")
ax[0, 1].set(xlabel=r"$\omega_p t$", ylabel="E-field energy / theory max", title="Langmuir oscillation")

root = kinetic_root(ld["k_lambda_d"])
for code in ("PyPIC3D", "WarpX"):
    t = runs[0]["landau"][code][0]
    curves, gammas, omegas = [], [], []
    for r in runs:
        field = r["landau"][code][1]
        peaks, _ = find_peaks(field, distance=max(int(0.7 * np.pi / (root.real * omega_p) / (t[1] - t[0])), 1))
        floor = np.median(field[len(field) // 2:])
        peaks = [i for i in peaks if field[i] > 5 * floor][:4]          # first peaks, above the particle-noise floor
        gammas.append(np.polyfit(t[peaks], np.log(field[peaks]), 1)[0] / 2 / omega_p)
        omegas.append(np.pi / np.mean(np.diff(t[peaks])) / omega_p)
        curves.append(np.log10(field / field.max()))
    summary.append(f"(c) Landau      {code:8s} gamma = {stats(gammas)} omega_p (kinetic {root.imag:.4f}), "
                   f"omega = {stats(omegas)} omega_p (kinetic {root.real:.4f})")
    band(ax[1, 0], t * omega_p, curves, code)
t_theory = np.linspace(0, ld["t_end"], 200)
ax[1, 0].plot(t_theory, 2 * root.imag * t_theory / np.log(10), "k--", lw=1.5,
              label=rf"theory $e^{{2\gamma t}}$, $\gamma={root.imag:.3f}\,\omega_p$")
ax[1, 0].set(xlabel=r"$\omega_p t$", ylabel=r"$\log_{10}$(E-field energy / max)",
             title=rf"Landau damping, $k\lambda_D = {ld['k_lambda_d']}$", ylim=(-4, 0.3))

ck = np.linspace(0, 3.3, 100)
ax[1, 1].plot(ck, np.sqrt(1 + ck**2), "k--", lw=1.5, label=r"theory $\omega^2=\omega_p^2+c^2k^2$")
ax[1, 1].plot(ck, ck, "k:", lw=1, label="light line")
for code in ("PyPIC3D", "WarpX"):
    measured = []
    for c in em["ck_over_wp"]:
        w = []
        for r in runs:
            t, field, _ = r[f"em{c:g}"][code]
            peaks, _ = find_peaks(field, distance=max(int(0.6 * np.pi / (wp_em * np.sqrt(1 + c**2)) / (t[1] - t[0])), 1))
            w.append(np.pi / np.mean(np.diff(t[peaks])) / wp_em)
        measured.append(w)
        summary.append(f"(d) EM wave     {code:8s} ck/wp = {c:g}: omega = {stats(w)} omega_p (theory {np.sqrt(1 + c**2):.4f}), "
                       f"error {100 * (np.mean(w) / np.sqrt(1 + c**2) - 1):+.2f}%")
    ax[1, 1].errorbar(em["ck_over_wp"], np.mean(measured, 1), np.std(measured, 1), lw=0, elinewidth=2, capsize=4, label=code,
                      **(dict(color="C0", marker="o", ms=10, mfc="none", mew=2) if code == "PyPIC3D" else dict(color="C1", marker="o", ms=4)))
ax[1, 1].set(xlabel=r"$ck/\omega_p$", ylabel=r"$\omega/\omega_p$", title="Electromagnetic waves in a plasma")
for a in ax.flat:
    a.legend(fontsize=9)
    a.grid(alpha=0.3, which="both")
fig.suptitle(rf"PyPIC3D vs WarpX vs theory: mean $\pm 1\sigma$ over {len(SEEDS)} seeds")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "warpx_collisionless.png"), dpi=300)
print("\n".join(summary))
print(f"Saved {OUTPUT_DIR}/warpx_collisionless.png and {path}")
