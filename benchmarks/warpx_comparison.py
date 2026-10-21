"""Physics benchmarks of PyPIC3D against WarpX and theory, on identical inputs:
  1. two-stream instability growth rate (1D electromagnetic PIC)
  2. collisional isotropization (WarpX/Smilei test, exact theory)
  3. electron-positron thermalization and electron-ion equilibration (Spitzer)
  4. MCC relaxation of argon ions in a 300 K argon bath (exact theory)
Run on Linux/WSL with WarpX on PATH:  python benchmarks/warpx_comparison.py"""
import os
import shutil
import subprocess
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import openpmd_api as io

from PyPIC3D.__main__ import run_PyPIC3D
from PyPIC3D.collisions import apply_collisions
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles

jax.config.update("jax_enable_x64", True)

# ----------------------------- input parameters -----------------------------
TWO_STREAM = dict(n_beam=1.0e15, v0=1.5e7, cells=64, ppc=200, t_end_wb=25.0)  # beam density [m^-3], drift [m/s]
CELLS, PPC = 8, 80              # collision tests: 8^3 cells, macro-particles per cell per species
OUTPUT_DIR = "outputs/warpx_comparison"
WARPX = {d: shutil.which(f"warpx.{d}d") for d in (1, 3)}

E, ME, MP, KB, EPS, C, M_AR = (1.602176634e-19, 9.1093837e-31, 1.67262192e-27, 1.380649e-23, 8.8541878128e-12,
                               2.99792458e8, 6.6335209e-26)
SPITZER_TIME = lambda m, t, n, clog: 1 / (2 * E**4 * n * clog / (3 * (2 * np.pi) ** 1.5 * EPS**2 * m[0] * m[1]
                                                                     * (t[0] * E / m[0] + t[1] * E / m[1]) ** 1.5))
COLLISION_CASES = {  # name: (masses, charges, per-direction temperatures [eV], density, coulomb log, dt, steps)
    "isotropization": ((ME,), (-E,), ([5.62, 5.1, 5.1],), 1.116e28, 2.0, 1.4e-17, 100),
    "thermalization": ((ME, ME), (-E, E), ([120.0] * 3, [40.0] * 3), 1.0e26, 10.0,
                       SPITZER_TIME((ME, ME), (120.0, 40.0), 1.0e26, 10.0) / 410, 410),
    "equilibration": ((ME, 25 * ME), (-E, E), ([100.0] * 3, [20.0] * 3), 1.0e26, 10.0,
                      SPITZER_TIME((ME, 25 * ME), (100.0, 20.0), 1.0e26, 10.0) / 2900, 2900),
    "mcc": ((M_AR,), (E,), ([3000 * KB / E] * 3,), 1.0e20, 0.0, 2.5e-8, 400),
}
N_GAS, T_GAS, NU_MCC = 1.0e20, 300.0, 1.0e6
MCC_ENERGY = np.logspace(-6, 3, 400)
MCC_SIGMA = NU_MCC / (N_GAS * np.sqrt(2 * MCC_ENERGY * E / (M_AR / 2)))  # sigma ~ 1/g: constant collision rate


def warpx(name, dims, inputs):
    """Run a WarpX input deck in its own directory; return the directory."""
    out = os.path.abspath(os.path.join(OUTPUT_DIR, name, "warpx"))
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    with open(os.path.join(out, "inputs"), "w") as handle:
        handle.write(inputs)
    np.savetxt(os.path.join(out, "elastic.dat"), np.column_stack((MCC_ENERGY, MCC_SIGMA)))
    log = subprocess.run([WARPX[dims], "inputs"], cwd=out, capture_output=True, text=True)
    if log.returncode:
        raise RuntimeError(log.stdout[-3000:] + log.stderr[-3000:])
    return out


def warpx_species(name, mass, charge, density, temps, ppc, extra="", drift=0.0):
    th = [np.sqrt(t * E / mass) / C for t in temps]
    return f"""{name}.charge = {charge}
{name}.uz_m = {drift / C}
{name}.mass = {mass}
{name}.injection_style = NRandomPerCell
{name}.num_particles_per_cell = {ppc}
{name}.profile = constant
{name}.density = {density}
{name}.momentum_distribution_type = gaussian
{name}.ux_th = {th[0]}
{name}.uy_th = {th[1]}
{name}.uz_th = {th[2]}
{extra}"""


def warpx_temperatures(out, species, masses):
    """Per-direction temperatures [eV] from WarpX openPMD particle output."""
    series = io.Series(os.path.join(out, "diags/diag1/openpmd_%T.h5"), io.Access.read_only)
    times, temps = [], []
    for index, iteration in series.iterations.items():
        times.append(iteration.time * iteration.time_unit_SI)
        row = []
        for name, m in zip(species, masses):
            chunks = [iteration.particles[name]["momentum"][d].load_chunk() for d in "xyz"]
            series.flush()
            # load_chunk buffers are only filled by flush()
            v = np.stack(chunks, -1) / m
            row.append(m * np.var(v, axis=0) / E)
        temps.append(row)
    order = np.argsort(times)
    return np.array(times)[order], np.array(temps)[order]


def pypic3d_collisions(masses, charges, temps, density, clog, dt, steps, specs, every):
    """0-D PyPIC3D collision run on the same 8^3-cell layout as WarpX; per-direction temperatures [eV]."""
    n_s, n = len(masses), PPC * CELLS**3
    keys = jax.random.split(jax.random.PRNGKey(7), n_s + 2)
    u = jnp.stack([jnp.sqrt(jnp.asarray(t) * E / m) * jax.random.normal(k, (n, 3)) for k, t, m in zip(keys, temps, masses)])
    particles = TiledParticles(x=jax.random.uniform(keys[-1], (1, 1, 1, n_s, n, 3), maxval=CELLS), u=u[None, None, None],
                               active=jnp.ones((1, 1, 1, n_s, n), bool), key=keys[-2])
    config = SpeciesConfig(charge=jnp.asarray(charges), mass=jnp.asarray(masses), weight=jnp.full(n_s, density / PPC),
                           update_x=jnp.zeros((n_s, 3), bool))
    world = SimpleNamespace(dx=1.0, dy=1.0, dz=1.0, dt=dt, eps=EPS, kb=KB)
    step = jax.jit(lambda p: apply_collisions(p, config, SimpleNamespace(collisions=specs), world)[0])
    times, out = [], []
    for i in range(steps + 1):
        if i % every == 0:
            v = np.asarray(particles.u[0, 0, 0])
            times.append(i * dt)
            out.append(np.asarray(masses)[:, None] * np.var(v, axis=1) / E)
        if i < steps:
            particles = step(particles)
    return np.array(times), np.array(out)


def spitzer_theory(masses, t0, density, clog, times):
    t, out, last = np.array(t0, float) * E, [], 0.0
    for time_point in times:
        h = (time_point - last) / 200
        for _ in range(200):
            rate = 2 * E**4 * density * clog / (3 * (2 * np.pi) ** 1.5 * EPS**2 * masses[0] * masses[1]
                                                * (t[0] / masses[0] + t[1] / masses[1]) ** 1.5)
            t = t + h * rate * np.array([t[1] - t[0], t[0] - t[1]])
        out.append(t / E)
        last = time_point
    return np.array(out)


def isotropization_theory(times, dt, density, clog):
    tpar, tper = 5.62 * E, 5.1 * E
    a = 1 - tper / tpar
    mu = (E**4 * density * clog / (8 * np.pi**1.5 * EPS**2 * ME**0.5 * tpar**1.5) * a**-2
          * (-3 + (3 - a) * np.arctanh(a**0.5) / a**0.5))
    tx, ty, out = tpar, tper, []
    for i in range(int(round(times[-1] / dt)) + 1):
        if np.any(np.isclose(i * dt, times, rtol=1e-9, atol=0.0)):
            out.append([tx / E, ty / E])
        tx, ty = tx + dt * mu * (ty - tx) * 2, ty + dt * mu * (tx - ty)
    return np.array(out)


# ----------------------------- run the benchmarks -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = {}

# 1. Two-stream instability: cold symmetric beams, box sized to the fastest-growing mode (gamma = omega_b / 2).
ts = TWO_STREAM
omega_b = np.sqrt(ts["n_beam"] * E**2 / (EPS * ME))
length = 2 * np.pi * ts["v0"] / (np.sqrt(3) / 2 * omega_b)
t_end = ts["t_end_wb"] / omega_b
n_part = ts["cells"] * ts["ppc"]
print(f"[1/4] Two-stream: omega_b = {omega_b:.3e} 1/s, theory field-energy growth 2*gamma = {omega_b:.3e} 1/s", flush=True)
start = time.perf_counter()
out = os.path.join(OUTPUT_DIR, "two_stream", "pypic3d")
shutil.rmtree(out, ignore_errors=True)
beam = lambda v, n: {"name": f"beam{'+' if v > 0 else '-'}", "N_particles": n_part, "charge": -E, "mass": ME,
                     "vth": 0.01 * ts["v0"], "initial_vx": v, "number_density": n}
run_PyPIC3D({
    "simulation_parameters": {"name": "two_stream", "output_dir": out, "solver": "electrodynamic_yee", "t_wind": t_end,
                              "Nx": ts["cells"], "Ny": 1, "Nz": 1, "x_wind": length, "y_wind": 1.0, "z_wind": 1.0, "cfl": 0.99,
                              "shape_factor": 1, "current_calculation": "j_from_rhov", "filter_j": "none"},
    "plotting": {"plotting_interval": 20},
    "particle1": beam(ts["v0"], ts["n_beam"]), "particle2": beam(-ts["v0"], ts["n_beam"]),
    "particle3": {"name": "ions", "N_particles": n_part, "charge": E, "mass": MP, "vth": 1.0, "number_density": 2 * ts["n_beam"]},
})
pyp = np.loadtxt(os.path.join(out, "data", "electric_field_energy.txt"), delimiter=",", ndmin=2)
print(f"      PyPIC3D done in {time.perf_counter() - start:.0f} s; running WarpX ...", flush=True)
dt = 0.99 * (length / ts["cells"]) / C
beam_wx = lambda name, v: warpx_species(name, ME, -E, ts["n_beam"], [ME * (0.01 * ts["v0"]) ** 2 / E] * 3, ts["ppc"], drift=v)
out_wx = warpx("two_stream", 1, f"""max_step = {int(t_end / dt)}
amr.n_cell = {ts['cells']}
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
particles.species_names = beamp beamm ions
{beam_wx('beamp', ts['v0'])}{beam_wx('beamm', -ts['v0'])}{warpx_species('ions', MP, E, 2 * ts['n_beam'], [0.0] * 3, ts['ppc'])}
warpx.reduced_diags_names = FE
FE.type = FieldEnergy
FE.intervals = 20
""")
wx = np.loadtxt(os.path.join(out_wx, "diags/reducedfiles/FE.txt"))
results["two_stream"] = {"PyPIC3D": pyp[:, :2], "WarpX": wx[:, [1, 3]]}  # time, E-field energy

# 2-4. Collision benchmarks.
for k, (case, (masses, charges, temps, density, clog, dt, steps)) in enumerate(COLLISION_CASES.items(), start=2):
    print(f"[{min(k, 4)}/4] {case}: {steps} steps of {dt:.2e} s, {PPC * CELLS**3:,} particles per species", flush=True)
    n_s = len(masses)
    if case == "mcc":
        specs = (("mcc", 0, N_GAS, T_GAS, M_AR, (("elastic", 0.0, tuple(MCC_ENERGY), tuple(MCC_SIGMA), -1, -1),)),)
        collisions = (f"collisions.collision_names = mcc\nmcc.type = background_mcc\nmcc.species = s0\n"
                      f"mcc.background_density = {N_GAS}\nmcc.background_temperature = {T_GAS}\nmcc.background_mass = {M_AR}\n"
                      f"mcc.scattering_processes = elastic\nmcc.elastic_cross_section = elastic.dat\n")
    else:
        specs = tuple(("binary", a, b, clog) for a in range(n_s) for b in range(a, n_s))
        names = " ".join(f"c{a}{b}" for _, a, b, _ in specs)
        collisions = f"collisions.collision_names = {names}\n" + "".join(
            f"c{a}{b}.species = s{a} s{b}\nc{a}{b}.CoulombLog = {clog}\n" for _, a, b, _ in specs)
    start = time.perf_counter()
    t_p, temp_p = pypic3d_collisions(masses, charges, temps, density, clog, dt, steps, specs, max(steps // 40, 1))
    print(f"      PyPIC3D done in {time.perf_counter() - start:.0f} s; running WarpX ...", flush=True)
    length = 1.0e-6 * CELLS
    species = "".join(warpx_species(f"s{s}", masses[s], charges[s], density, temps[s], PPC, f"s{s}.do_not_deposit = 1\n")
                      for s in range(n_s))
    out_wx = warpx(case, 3, f"""max_step = {steps}
amr.n_cell = {CELLS} {CELLS} {CELLS}
amr.max_grid_size = {CELLS}
amr.blocking_factor = {CELLS}
amr.max_level = 0
geometry.dims = 3
geometry.prob_lo = 0 0 0
geometry.prob_hi = {length} {length} {length}
boundary.field_lo = periodic periodic periodic
boundary.field_hi = periodic periodic periodic
boundary.particle_lo = periodic periodic periodic
boundary.particle_hi = periodic periodic periodic
warpx.const_dt = {dt}
algo.maxwell_solver = none
algo.particle_shape = 1
particles.species_names = {" ".join(f"s{s}" for s in range(n_s))}
{species}{collisions}diagnostics.diags_names = diag1
diag1.intervals = {max(steps // 40, 1)}
diag1.diag_type = Full
diag1.format = openpmd
diag1.openpmd_backend = h5
diag1.fields_to_plot = none
""")
    t_w, temp_w = warpx_temperatures(out_wx, [f"s{s}" for s in range(n_s)], masses)
    results[case] = {"PyPIC3D": (t_p, temp_p), "WarpX": (t_w, temp_w)}

# ----------------------------- plot, save, and print -----------------------------
fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
a = ax[0, 0]
for code, data in results["two_stream"].items():
    a.semilogy(data[:, 0] * omega_b, data[:, 1] / data[:, 1].max(), lw=2, label=code)
    window = (data[:, 1] > 1e-3 * data[:, 1].max()) & (data[:, 1] < 0.3 * data[:, 1].max()) & (data[:, 0] < data[np.argmax(data[:, 1]), 0])
    rate = np.polyfit(data[window, 0], np.log(data[window, 1]), 1)[0]
    print(f"two-stream  {code:8s} field-energy growth = {rate / omega_b:.3f} omega_b (theory 1.000), error {100 * abs(rate / omega_b - 1):.1f}%")
t_ref = np.linspace(4, 14, 10)
a.semilogy(t_ref, 1e-5 * np.exp(t_ref - 14), "k--", label=r"theory $e^{2\gamma t}$, $\gamma = \omega_b/2$")
a.set(xlabel=r"$\omega_b t$", ylabel="E-field energy / max", title="Two-stream instability", ylim=(1e-9, 2))

t_p, temp_p = results["isotropization"]["PyPIC3D"]
t_w, temp_w = results["isotropization"]["WarpX"]
theory = isotropization_theory(t_p, COLLISION_CASES["isotropization"][5], 1.116e28, 2.0)
a = ax[0, 1]
for d, label in ((0, r"$T_\parallel$"), (1, r"$T_\perp$")):
    a.plot(t_p * 1e15, temp_p[:, 0, d], lw=2, color=f"C{d}", label=f"PyPIC3D {label}")
    a.plot(t_w * 1e15, temp_w[:, 0, d], "o", ms=4, color=f"C{d}", label=f"WarpX {label}")
    a.plot(t_p * 1e15, theory[:, d], "k--", lw=1)
a.set(xlabel="time [fs]", ylabel="temperature [eV]", title="Isotropization (- - theory)")
print(f"isotropization  final T_par: PyPIC3D {temp_p[-1, 0, 0]:.4f}  WarpX {temp_w[-1, 0, 0]:.4f}  theory {theory[-1, 0]:.4f} eV")

for a, case in ((ax[0, 2], "thermalization"), (ax[1, 0], "equilibration")):
    masses, _, temps, density, clog, *_ = COLLISION_CASES[case]
    t_p, temp_p = results[case]["PyPIC3D"]
    t_w, temp_w = results[case]["WarpX"]
    theory = spitzer_theory(masses, [temps[0][0], temps[1][0]], density, clog, t_p)
    for s in range(2):
        a.plot(t_p * 1e12, temp_p[:, s].mean(-1), lw=2, color=f"C{s}", label=f"PyPIC3D species {s}")
        a.plot(t_w * 1e12, temp_w[:, s].mean(-1), "o", ms=4, color=f"C{s}", label=f"WarpX species {s}")
        a.plot(t_p * 1e12, theory[:, s], "k--", lw=1, label="Spitzer (Maxwellian)" if s == 0 else None)
    a.set(xlabel="time [ps]", ylabel="temperature [eV]", title=case.capitalize())
    dt0 = temps[0][0] - temps[1][0]
    gap = lambda tt: tt[-1, 0].mean() - tt[-1, 1].mean()
    print(f"{case:15s} final dT: PyPIC3D {gap(temp_p):.2f}  WarpX {gap(temp_w):.2f}  Spitzer {theory[-1, 0] - theory[-1, 1]:.2f} eV "
          f"(initial {dt0:.0f} eV)")

t_p, temp_p = results["mcc"]["PyPIC3D"]
t_w, temp_w = results["mcc"]["WarpX"]
a = ax[1, 1]
a.plot(t_p * 1e6, temp_p[:, 0].mean(-1) * E / KB, lw=2, label="PyPIC3D")
a.plot(t_w * 1e6, temp_w[:, 0].mean(-1) * E / KB, "o", ms=4, label="WarpX")
a.plot(t_p * 1e6, T_GAS + (temp_p[0, 0].mean() * E / KB - T_GAS) * np.exp(-0.5 * NU_MCC * t_p), "k--", label="theory")
a.set(xlabel=r"time [$\mu$s]", ylabel="Ar$^+$ temperature [K]", title="MCC relaxation in 300 K argon")
for code, (t, temp) in results["mcc"].items():
    theory = T_GAS + (temp[0, 0].mean() * E / KB - T_GAS) * np.exp(-0.5 * NU_MCC * t)
    print(f"mcc             {code:8s} max |T - theory| = {np.max(np.abs(temp[:, 0].mean(-1) * E / KB - theory)):.1f} K")
ax[1, 2].axis("off")
for a in ax.flat[:5]:
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "warpx_comparison.png"), dpi=200)
np.savez(os.path.join(OUTPUT_DIR, "warpx_comparison.npz"),
         **{f"{case}_{code}_{i}": arr for case, r in results.items() for code, v in r.items()
            for i, arr in enumerate(v if isinstance(v, tuple) else (v,))})
print(f"Saved {OUTPUT_DIR}/warpx_comparison.png and .npz")
