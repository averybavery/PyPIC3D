"""Parallel scaling of PyPIC3D (CPU cores, JAX devices/tiles, GPU) and WarpX (OpenMP
threads) on the same 3D uniform thermal plasma. Run on Linux/WSL (or a GPU node):
    python benchmarks/scaling.py
Each configuration runs in a fresh subprocess so core affinity and device counts apply.

GPUs: run benchmarks/perlmutter.sh on NERSC (or any CUDA node with jax[cuda12]). Results are saved per
machine as outputs/scaling/scaling_<machine>.json; copy them together and run
    python benchmarks/scaling.py --plot
to redraw both figures (PyPIC3D alone, PyPIC3D vs WarpX) from every machine."""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------- input parameters -----------------------------
GRID = (32, 32, 32)             # cells
PPC = 8                         # macro-particles per cell per species (electrons + ions)
STEPS = 10                      # timed steps (after compilation / warm-up)
CORES = [2**i for i in range(8) if 2**i <= os.cpu_count()]  # CPU cores to pin (strong scaling)
DEVICES = [1, 2, 4, 8]          # JAX CPU devices = tiles along x (domain decomposition, all cores)
GPUS = [g for g in (1, 2, 4) if g <= len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.split("GPU ")) - 1] if shutil.which("nvidia-smi") else []
WARPX_BIN = shutil.which("warpx.3d")  # set to None to skip WarpX
DENSITY, TEMPERATURE_EV = 1.0e18, 10.0
OUTPUT_DIR = "outputs/scaling"
MACHINE = os.environ.get("NERSC_HOST", platform.node())

E, ME, MP, EPS, C = 1.602176634e-19, 9.1093837e-31, 1.67262192e-27, 8.8541878128e-12, 2.99792458e8
DX = np.sqrt(EPS * TEMPERATURE_EV * E / (DENSITY * E**2))      # one Debye length per cell
VTH_E, VTH_I = np.sqrt(TEMPERATURE_EV * E / ME), np.sqrt(TEMPERATURE_EV * E / MP)
N_PARTICLES = PPC * int(np.prod(GRID))


def pypic3d_worker(cores, devices, backend):
    """Time PyPIC3D steps in this process (called through the subprocess entry point below)."""
    if cores:
        os.sched_setaffinity(0, range(cores))
    os.environ.update(JAX_PLATFORMS=backend, JAX_NUM_CPU_DEVICES=str(devices), JAX_ENABLE_X64="1",
                      CUDA_VISIBLE_DEVICES=",".join(map(str, range(devices))))
    import jax
    from PyPIC3D.initialization import initialize_simulation

    out = os.path.join(OUTPUT_DIR, f"pypic3d_{backend}_{cores}c_{devices}d")
    os.makedirs(out, exist_ok=True)
    species = lambda name, q, m, vth: {"name": name, "N_particles": N_PARTICLES, "charge": q, "mass": m, "vth": float(vth),
                                       "number_density": DENSITY}
    config = {
        "simulation_parameters": {"name": "scaling", "output_dir": out, "solver": "electrodynamic_yee", "Nt": STEPS,
                                  "Nx": GRID[0], "Ny": GRID[1], "Nz": GRID[2], "x_wind": GRID[0] * DX, "y_wind": GRID[1] * DX,
                                  "z_wind": GRID[2] * DX, "cfl": 0.99, "shape_factor": 1, "current_calculation": "j_from_rhov",
                                  "filter_j": "none", "relativistic": False, "particle_tile_nx": GRID[0] // devices,
                                  "particle_tile_capacity_factor": 1.5},
        "plotting": {"plotting_interval": 10**9},
        "particle1": species("electrons", -E, ME, VTH_E),
        "particle2": species("ions", E, MP, VTH_I),
    }
    loop, particles, fields, static, dynamic, *_, species_config = initialize_simulation(config)
    step = jax.jit(lambda p, f: loop(p, species_config, f, static, dynamic))
    start = time.perf_counter()
    for _ in range(2):
        particles, fields = jax.block_until_ready(step(particles, fields))
    compile_s = time.perf_counter() - start
    # two warm-up calls: the first step's output shardings differ from the initial state, forcing one re-compile
    start = time.perf_counter()
    for _ in range(STEPS):
        particles, fields = step(particles, fields)
    jax.block_until_ready(fields)
    per_step = (time.perf_counter() - start) / STEPS
    return {"code": "PyPIC3D", "machine": MACHINE, "platform": backend, "device_kind": jax.devices()[0].device_kind, "cores": cores,
            "devices": devices, "s_per_step": per_step, "compile_s": compile_s, "overflow": bool(fields[-1])}


def run_pypic3d(cores, devices, backend="cpu"):
    args = json.dumps([cores, devices, backend])
    result = subprocess.run([sys.executable, __file__, "--worker", args], capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    if not lines:
        print(result.stdout[-2000:], result.stderr[-3000:])
        return None
    return json.loads(lines[-1][7:])


def run_warpx(threads):
    out = os.path.abspath(os.path.join(OUTPUT_DIR, f"warpx_{threads}t"))
    os.makedirs(out, exist_ok=True)
    species = lambda name, q, m, vth: f"""{name}.charge = {q}
{name}.mass = {m}
{name}.injection_style = NRandomPerCell
{name}.num_particles_per_cell = {PPC}
{name}.profile = constant
{name}.density = {DENSITY}
{name}.momentum_distribution_type = gaussian
{name}.ux_th = {vth / C}
{name}.uy_th = {vth / C}
{name}.uz_th = {vth / C}
"""
    with open(os.path.join(out, "inputs"), "w") as handle:
        handle.write(f"""max_step = {STEPS + 1}
amr.n_cell = {GRID[0]} {GRID[1]} {GRID[2]}
amr.max_grid_size = {max(GRID)}
amr.max_level = 0
geometry.dims = 3
geometry.prob_lo = 0 0 0
geometry.prob_hi = {GRID[0] * DX} {GRID[1] * DX} {GRID[2] * DX}
boundary.field_lo = periodic periodic periodic
boundary.field_hi = periodic periodic periodic
boundary.particle_lo = periodic periodic periodic
boundary.particle_hi = periodic periodic periodic
warpx.cfl = 0.99
warpx.do_dive_cleaning = 0
algo.maxwell_solver = yee
algo.current_deposition = direct
warpx.use_filter = 0
algo.particle_pusher = boris
algo.particle_shape = 1
particles.species_names = electrons ions
{species("electrons", -E, ME, VTH_E)}{species("ions", E, MP, VTH_I)}""")
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    log = subprocess.run(["taskset", "-c", f"0-{threads - 1}", WARPX_BIN, "inputs"], cwd=out, env=env,
                         capture_output=True, text=True).stdout
    calls, seconds = re.findall(r"WarpX::Evolve::step\s+(\d+)\s+([\d.eE+-]+)", log)[-1]
    # the profiler prints exclusive then inclusive tables; the last entry is the inclusive step time
    return {"code": "WarpX", "machine": MACHINE, "platform": "cpu", "cores": threads, "devices": 1, "s_per_step": float(seconds) / int(calls)}


# ----------------------------- run the benchmarks -----------------------------
if __name__ == "__main__" and "--worker" in sys.argv:
    print("RESULT " + json.dumps(pypic3d_worker(*json.loads(sys.argv[-1]))))
    sys.exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
if "--plot" not in sys.argv:
    print(f"{MACHINE}: grid {GRID}, {2 * N_PARTICLES:,} macro-particles, {STEPS} timed steps, GPUs {GPUS}")
    results = []
    for cores in CORES:
        print(f"PyPIC3D  {cores} core(s), 1 device ...", flush=True)
        results.append(run_pypic3d(cores, 1))
    for devices in DEVICES[1:]:
        print(f"PyPIC3D  all cores, {devices} CPU devices/tiles ...", flush=True)
        results.append(run_pypic3d(0, devices))
    for gpus in GPUS:
        print(f"PyPIC3D  {gpus} GPU(s), one tile per GPU ...", flush=True)
        results.append(run_pypic3d(0, gpus, "cuda"))
    for threads in CORES if WARPX_BIN else []:
        print(f"WarpX    {threads} OpenMP thread(s) ...", flush=True)
        results.append(run_warpx(threads))
    results = [r for r in results if r]
    with open(os.path.join(OUTPUT_DIR, f"scaling_{MACHINE}.json"), "w") as handle:
        json.dump(results, handle, indent=1)

# ----------------------------- plot, save, and print -----------------------------
results = [r for f in sorted(os.listdir(OUTPUT_DIR)) if f.startswith("scaling_") and f.endswith(".json")
           for r in json.load(open(os.path.join(OUTPUT_DIR, f)))]
for r in results:
    r.setdefault("machine", "laptop")
    print(f"  {r['machine']:10s} {r['code']:8s} {r['platform']:5s} cores={r['cores'] or 'all':>3} devices={r['devices']}  "
          f"{r['s_per_step'] * 1e3:9.2f} ms/step  {r['s_per_step'] * 1e9 / (2 * N_PARTICLES):8.1f} ns/particle/step")


def series(code, machine, kind):
    """(resources, seconds per step) for CPU cores, CPU devices, or GPUs on one machine."""
    pick = {"cores": lambda r: r["platform"] == "cpu" and r["devices"] == 1 and r["cores"],
            "devices": lambda r: r["platform"] == "cpu" and (r["devices"] > 1 or (r["cores"] == 1 and r["devices"] == 1)),
            "gpus": lambda r: r["platform"] == "cuda"}[kind]
    rows = sorted((r["cores"] if kind == "cores" else r["devices"], r["s_per_step"]) for r in results
                  if r["code"] == code and r["machine"] == machine and pick(r))
    return np.array(rows).T if rows else (np.array([]), np.array([]))


machines = sorted({r["machine"] for r in results})
title = f"{2 * N_PARTICLES:,} particles, {int(np.prod(GRID)):,} cells"
styles = {"cores": ("o", "-", "CPU cores"), "devices": ("^", "--", "CPU devices (tiles)"), "gpus": ("*", "-", "GPUs")}
for name, codes in (("scaling_pypic3d", ("PyPIC3D",)), ("scaling_vs_warpx", ("PyPIC3D", "WarpX"))):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    top, line = 1, 0
    for code in codes:
        for machine in machines:
            for kind, (marker, ls, label) in styles.items():
                n, t = series(code, machine, kind)
                if not len(n):
                    continue
                color, line = f"C{line}", line + 1
                tag = f"{code} {label}" + (f" [{machine}]" if len(machines) > 1 else "")
                ax[0].loglog(n, t * 1e3, marker=marker, ls=ls, lw=2, ms=8, color=color, label=tag)
                ax[1].plot(n, t[0] / t, marker=marker, ls=ls, lw=2, ms=8, color=color, label=tag)
                top = max(top, n.max())
    ideal = np.array([1, top])
    ax[1].plot(ideal, ideal, "k:", label="ideal")
    ax[0].set(xlabel="CPU cores / devices / GPUs", ylabel="wall time per step [ms]", title=title)
    ax[1].set(xlabel="CPU cores / devices / GPUs", ylabel="speed-up vs. 1 unit", title="Strong scaling")
    ax[0].set_xscale("log", base=2)
    ax[0].xaxis.set_major_formatter(plt.ScalarFormatter())
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=300)
    print(f"Saved {OUTPUT_DIR}/{name}.png")
