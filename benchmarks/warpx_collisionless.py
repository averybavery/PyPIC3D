"""Collisionless physics benchmarks of PyPIC3D against WarpX and exact theory (1D PIC, identical inputs):
  (a) two-stream instability growth rate     cold-beam theory, gamma = omega_b / 2
  (b) WarpX Langmuir test                    Examples/Tests/langmuir/inputs_test_1d_langmuir_multi (run longer);
                                             analytic E = eps m c^2 k / e sin(kz) sin(omega_p t), WarpX tolerance 5%
  (c) Landau damping at k lambda_D = 0.5     exact kinetic rate gamma = -0.1534 omega_p
  (d) WarpX energy-conservation test         Examples/Tests/energy_conserving_thermal_plasma (1D);
                                             WarpX tolerance |dE / E| < 0.3%
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
ENERGY = dict(n0=1.0e30, te_ev=100.0, cells=8, ppc=4, steps=500, tolerance=0.003)  # WarpX test values
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
{name}.ux_mean_function(x,y,z) = "0"
{name}.uy_mean_function(x,y,z) = "0"
{name}.uz_mean_function(x,y,z) = "{drift}"
{name}.maxwellian_u_std_distribution_type = constant
{name}.ux_std = {vth / C}
{name}.uy_std = {vth / C}
{name}.uz_std = {vth / C}
""" for name, q, m, ppc, density, vth, drift in species)


# ----------------------------- run the benchmarks -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = {}
rng = np.random.default_rng(0)
ts, lg, ld, en = TWO_STREAM, LANGMUIR, LANDAU, ENERGY
omega_b = np.sqrt(ts["n_beam"] * E**2 / (EPS * ME))
omega_l, k_l = np.sqrt(2 * lg["n0"] * E**2 / (EPS * ME)), 2 * np.pi / 20e-6
omega_p = np.sqrt(ld["density"] * E**2 / (EPS * ME))
wpe = E * np.sqrt(en["n0"] / (ME * EPS))
if "--plot" in sys.argv:
    saved = np.load(os.path.join(OUTPUT_DIR, "warpx_collisionless.npz"))
    for key in saved.files:
        case, code = key.split("_", 1) if not key.startswith("two_stream") else ("two_stream", key[11:])
        results.setdefault(case, {})[code] = tuple(saved[key].T)
else:

    ts = TWO_STREAM
    omega_b = np.sqrt(ts["n_beam"] * E**2 / (EPS * ME))
    length = 2 * np.pi * ts["v0"] / (np.sqrt(3) / 2 * omega_b)
    dt = 0.99 * (length / ts["cells"]) / C
    print(f"[1/4] Two-stream instability (omega_b = {omega_b:.3e} 1/s)", flush=True)
    beam = lambda name, v: {"name": name, "N_particles": ts["cells"] * ts["ppc"], "charge": -E, "mass": ME,
                            "vth": 0.01 * ts["v0"], "initial_vx": v, "number_density": ts["n_beam"]}
    results["two_stream"] = {
        "PyPIC3D": run_pypic3d("two_stream", {
            "simulation_parameters": {"solver": "electrodynamic_yee", "t_wind": ts["t_end"] / omega_b, "Nx": ts["cells"],
                                      "x_wind": length, "cfl": 0.99, "shape_factor": 1, "current_calculation": "j_from_rhov",
                                      "filter_j": "none"},
            "plotting": {"plotting_interval": 5},
            "particle1": beam("beam_plus", ts["v0"]), "particle2": beam("beam_minus", -ts["v0"]),
            "particle3": {"name": "ions", "N_particles": ts["cells"] * ts["ppc"], "charge": E, "mass": MP, "vth": 1.0,
                          "number_density": 2 * ts["n_beam"]}}),
        "WarpX": run_warpx("two_stream", warpx_deck(length, ts["cells"], int(ts["t_end"] / omega_b / dt), [
            ("beamp", -E, ME, ts["ppc"], ts["n_beam"], 0.01 * ts["v0"], ts["v0"] / C),
            ("beamm", -E, ME, ts["ppc"], ts["n_beam"], 0.01 * ts["v0"], -ts["v0"] / C),
            ("ions", E, MP, ts["ppc"], 2 * ts["n_beam"], 0.0, 0.0)], f"warpx.const_dt = {dt}\nalgo.current_deposition = direct\n"
                                                                     "algo.particle_shape = 1"))}

    lg = LANGMUIR
    omega_l = np.sqrt(2 * lg["n0"] * E**2 / (EPS * ME))
    k_l = 2 * np.pi / 20e-6
    print(f"[2/4] WarpX Langmuir test, {lg['steps']} steps (omega_p = {omega_l:.3e} 1/s)", flush=True)
    n = lg["cells"] * lg["ppc"]
    x = (np.arange(n) + 0.5) * lg["length"] / n - lg["length"] / 2      # NUniformPerCell quiet start
    u = lg["eps"] * k_l / (omega_l / C) * np.sin(k_l * x)                 # u = gamma v / c
    v = u * C / np.sqrt(1 + u**2)
    species = lambda name, q, vx: {"name": name, "N_particles": n, "charge": q, "mass": ME, "vth": 1e-3,
                                   "number_density": lg["n0"], "initial_x": x, "initial_vx": vx}
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
    results["langmuir"] = {
        "PyPIC3D": run_pypic3d("langmuir", {
            "simulation_parameters": {"solver": "electrodynamic_yee", "Nt": lg["steps"], "Nx": lg["cells"], "x_wind": lg["length"],
                                      "cfl": 0.8, "shape_factor": 1, "current_calculation": "esirkepov", "filter_j": "none"},
            "plotting": {"plotting_interval": 2},
            "particle1": species("electrons", -E, v), "particle2": species("positrons", E, -v)}),
        "WarpX": run_warpx("langmuir", warpx_langmuir, interval=2)}

    ld = LANDAU
    omega_p = np.sqrt(ld["density"] * E**2 / (EPS * ME))
    vth = ld["vth_over_c"] * C
    k = ld["k_lambda_d"] * omega_p / vth
    length = 2 * np.pi / k
    dt = 0.99 * (length / ld["cells"]) / C
    print(f"[3/4] Landau damping, k lambda_D = {ld['k_lambda_d']} (omega_p = {omega_p:.3e} 1/s)", flush=True)
    n = ld["cells"] * ld["ppc"]
    x = rng.uniform(-length / 2, length / 2, n)
    u_wave = ld["amplitude"] * omega_p / k                                # continuity: dn/n = k U / omega
    results["landau"] = {
        "PyPIC3D": run_pypic3d("landau", {
            "simulation_parameters": {"solver": "electrodynamic_yee", "t_wind": ld["t_end"] / omega_p, "Nx": ld["cells"],
                                      "x_wind": length, "cfl": 0.99, "shape_factor": 1, "current_calculation": "j_from_rhov",
                                      "filter_j": "none"},
            "plotting": {"plotting_interval": 5},
            "particle1": {"name": "electrons", "N_particles": n, "charge": -E, "mass": ME, "vth": vth, "number_density": ld["density"],
                          "initial_x": x, "initial_vx": vth * rng.standard_normal(n) + u_wave * np.sin(k * x)},
            "particle2": {"name": "ions", "N_particles": n, "charge": E, "mass": MP, "vth": 0.0, "number_density": ld["density"],
                          "update_x": False, "update_y": False, "update_z": False}}),
        "WarpX": run_warpx("landau", warpx_deck(length, ld["cells"], int(ld["t_end"] / omega_p / dt), [
            ("electrons", -E, ME, ld["ppc"], ld["density"], vth, f"{u_wave / C}*sin({k}*z)"),
            ("ions", E, MP, ld["ppc"], ld["density"], 0.0, 0.0)],
            f"warpx.const_dt = {dt}\nalgo.current_deposition = direct\nalgo.particle_shape = 1\nions.do_not_push = 1"))}

    en = ENERGY
    wpe = E * np.sqrt(en["n0"] / (ME * EPS))
    length, n = 10 * C / wpe, en["cells"] * en["ppc"]
    print(f"[4/4] WarpX energy-conservation test, {en['steps']} steps", flush=True)
    x = (np.arange(n) + 0.5) * length / n - length / 2
    thermal = lambda m: np.sqrt(en["te_ev"] * E / m)
    energy_case = lambda solver: run_pypic3d(f"energy_{solver}", {
        "simulation_parameters": {"solver": solver, "Nt": en["steps"], "dt": 0.2 / wpe, "Nx": en["cells"], "x_wind": length,
                                  "shape_factor": 2, "current_calculation": "esirkepov", "filter_j": "none"},
        "plotting": {"plotting_interval": 10},
        "particle1": {"name": "electrons", "N_particles": n, "charge": -E, "mass": ME, "vth": thermal(ME), "number_density": en["n0"],
                      "initial_x": x},
        "particle2": {"name": "protons", "N_particles": n, "charge": E, "mass": MP, "vth": thermal(MP), "number_density": en["n0"],
                      "initial_x": x}})
    warpx_energy = f"""max_step = {en['steps']}
    amr.n_cell = {en['cells']}
    amr.max_level = 0
    geometry.dims = 1
    warpx.do_electrostatic = labframe
    algo.field_gathering = energy-conserving
    algo.particle_shape = 2
    warpx.use_filter = 0
    my_constants.n0 = {en['n0']}
    my_constants.T = {en['te_ev']}
    my_constants.wpe = q_e*sqrt(n0/(m_e*epsilon0))
    my_constants.de0 = clight/wpe
    warpx.const_dt = 0.2/wpe
    geometry.prob_lo = 0.0
    geometry.prob_hi = 10.*de0
    boundary.field_lo = periodic
    boundary.field_hi = periodic
    boundary.particle_lo = periodic
    boundary.particle_hi = periodic
    particles.species_names = electrons protons
    """ + "".join(f"""{name}.species_type = {kind}
    {name}.injection_style = "NUniformPerCell"
    {name}.num_particles_per_cell_each_dim = {en['ppc']}
    {name}.profile = constant
    {name}.density = n0
    {name}.momentum_distribution_type = gaussian
    {name}.ux_th = sqrt(T*q_e/{mass})/clight
    {name}.uy_th = sqrt(T*q_e/{mass})/clight
    {name}.uz_th = sqrt(T*q_e/{mass})/clight
    """ for name, kind, mass in (("electrons", "electron", "m_e"), ("protons", "proton", "m_p")))
    results["energy"] = {"PyPIC3D (electrostatic)": energy_case("electrostatic"),
                         "PyPIC3D (electromagnetic)": energy_case("electrodynamic_yee"),
                         "WarpX (electrostatic)": run_warpx("energy", warpx_energy, interval=10)}

# ----------------------------- plot, save, and print -----------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))
style = {"PyPIC3D": dict(color="C0", lw=2), "WarpX": dict(color="C1", lw=0, marker="o", ms=4)}
summary = []

for code, (t, field, _) in results["two_stream"].items():
    peak = np.argmax(field)
    window = (field > 1e-3 * field[peak]) & (field < 0.3 * field[peak]) & (t < t[peak])
    rate = np.polyfit(t[window], np.log(field[window]), 1)[0] / 2 / omega_b
    summary.append(f"(a) two-stream  {code:26s} gamma = {rate:.3f} omega_b (theory 0.500), error {100 * abs(rate / 0.5 - 1):.1f}%")
    ax[0, 0].semilogy(t * omega_b, field / field.max(), label=code, **{**style[code], "markevery": 4})
    if code == "PyPIC3D":
        t_fit = t[window] * omega_b
        ax[0, 0].semilogy(t_fit, field[window][0] / field.max() * np.exp(t_fit - t_fit[0]), "k--", lw=1.5,
                          label=r"theory $e^{2\gamma t}$, $\gamma=\omega_b/2$")
ax[0, 0].set(xlabel=r"$\omega_b t$", ylabel="E-field energy / max", title="Two-stream instability", ylim=(1e-6, 2))

amplitude = lg["eps"] * ME * C**2 * k_l / E
for code, (t, field, _) in results["langmuir"].items():
    theory = EPS / 2 * amplitude**2 * np.sin(omega_l * t) ** 2 * lg["length"] / 2
    error = np.max(np.abs(field - theory)) / theory.max()
    summary.append(f"(b) Langmuir    {code:26s} max |W - W_theory| / W_max = {100 * error:.2f}% (WarpX test tolerance 5%)")
    ax[0, 1].plot(t * omega_l, field / theory.max(), label=code, **{**style[code], "markevery": 3})
t = results["langmuir"]["PyPIC3D"][0]
ax[0, 1].plot(t * omega_l, np.sin(omega_l * t) ** 2, "k--", lw=1.2, label=r"theory $\sin^2\omega_p t$")
ax[0, 1].set(xlabel=r"$\omega_p t$", ylabel="E-field energy / theory max", title="Langmuir oscillation")

root = kinetic_root(ld["k_lambda_d"])
for code, (t, field, _) in results["landau"].items():
    peaks, _ = find_peaks(field, distance=max(int(0.7 * np.pi / (root.real * omega_p) / (t[1] - t[0])), 1))
    floor = np.median(field[len(field) // 2:])
    peaks = [i for i in peaks if field[i] > 5 * floor]  # stop before the particle-noise floor
    gamma = np.polyfit(t[peaks], np.log(field[peaks]), 1)[0] / 2 / omega_p
    omega = np.pi / np.mean(np.diff(t[peaks])) / omega_p
    summary.append(f"(c) Landau      {code:26s} gamma = {gamma:.4f} omega_p (kinetic {root.imag:.4f}), "
                   f"omega = {omega:.4f} omega_p (kinetic {root.real:.4f})")
    ax[1, 0].semilogy(t * omega_p, field / field.max(), label=code, **{**style[code], "markevery": 3})
t_theory = np.linspace(0, ld["t_end"], 200)
ax[1, 0].semilogy(t_theory, np.exp(2 * root.imag * t_theory), "k--", lw=1.5,
                  label=rf"theory $e^{{2\gamma t}}$, $\gamma={root.imag:.3f}\,\omega_p$")
ax[1, 0].set(xlabel=r"$\omega_p t$", ylabel="E-field energy / max", title=rf"Landau damping, $k\lambda_D = {ld['k_lambda_d']}$",
             ylim=(1e-4, 2))

for (code, (t, _, total)), color in zip(results["energy"].items(), ("C0", "C2", "C1")):
    drift = (total - total[0]) / total[0]
    summary.append(f"(d) energy      {code:26s} max |dE / E| = {100 * np.max(np.abs(drift)):.3f}% "
                   f"({'pass' if np.max(np.abs(drift)) < en['tolerance'] else 'FAIL'} at WarpX's 0.3%)")
    ax[1, 1].plot(t * wpe, 100 * drift, color=color, lw=2 if "PyPIC3D" in code else 0, marker=None if "PyPIC3D" in code else "o",
                  ms=4, label=code)
ax[1, 1].axhspan(-100 * en["tolerance"], 100 * en["tolerance"], color="0.85", label="WarpX test tolerance")
ax[1, 1].set(xlabel=r"$\omega_{pe} t$", ylabel="total energy change [%]", title="Energy conservation")
for a in ax.flat:
    a.legend(fontsize=9)
    a.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "warpx_collisionless.png"), dpi=300)
np.savez(os.path.join(OUTPUT_DIR, "warpx_collisionless.npz"),
         **{f"{case}_{code}": np.column_stack(v) for case, r in results.items() for code, v in r.items()})
print("\n".join(summary))
print(f"Saved {OUTPUT_DIR}/warpx_collisionless.png and .npz")
