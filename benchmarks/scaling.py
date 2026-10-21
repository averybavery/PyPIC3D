"""Parallel scaling of PyPIC3D (CPU cores, JAX devices/tiles, GPU) and WarpX (OpenMP
threads) on the same 3D uniform thermal plasma. Run on Linux/WSL (or a GPU node):
    python benchmarks/scaling.py
Each configuration runs in a fresh subprocess so core affinity and device counts apply.

GPU (NVIDIA): on any Linux machine with a CUDA GPU (Google Colab, UW CHTC GPU Lab, a lab node):
    git clone -b avery-workshop https://github.com/<owner>/PyPIC3D && cd PyPIC3D
    pip install -U "jax[cuda12]" -e .
    python benchmarks/scaling.py        # adds a "PyPIC3D GPU" row/line; send back outputs/scaling/"""
import json
import os
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
CORES = [c for c in (1, 2, 4, 8) if c <= os.cpu_count()]  # CPU cores to pin (strong scaling)
DEVICES = [1, 2, 4, 8]          # JAX CPU devices = tiles along x (domain decomposition, all cores)
RUN_GPU = shutil.which("nvidia-smi") is not None  # time the GPU too (needs jax[cuda12])
WARPX_BIN = shutil.which("warpx.3d")  # set to None to skip WarpX
DENSITY, TEMPERATURE_EV = 1.0e18, 10.0
OUTPUT_DIR = "outputs/scaling"

E, ME, MP, EPS, C = 1.602176634e-19, 9.1093837e-31, 1.67262192e-27, 8.8541878128e-12, 2.99792458e8
DX = np.sqrt(EPS * TEMPERATURE_EV * E / (DENSITY * E**2))      # one Debye length per cell
VTH_E, VTH_I = np.sqrt(TEMPERATURE_EV * E / ME), np.sqrt(TEMPERATURE_EV * E / MP)
N_PARTICLES = PPC * int(np.prod(GRID))


def pypic3d_worker(cores, devices, platform):
    """Time PyPIC3D steps in this process (called through the subprocess entry point below)."""
    if cores:
        os.sched_setaffinity(0, range(cores))
    os.environ.update(JAX_PLATFORMS=platform, JAX_NUM_CPU_DEVICES=str(devices), JAX_ENABLE_X64="1")
    import jax
    from PyPIC3D.initialization import initialize_simulation

    out = os.path.join(OUTPUT_DIR, f"pypic3d_{platform}_{cores}c_{devices}d")
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
    return {"code": "PyPIC3D", "platform": platform, "device_kind": jax.devices()[0].device_kind, "cores": cores,
            "devices": devices, "s_per_step": per_step, "compile_s": compile_s, "overflow": bool(fields[-1])}


def run_pypic3d(cores, devices, platform="cpu"):
    args = json.dumps([cores, devices, platform])
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
    return {"code": "WarpX", "platform": "cpu", "cores": threads, "devices": 1, "s_per_step": float(seconds) / int(calls)}


# ----------------------------- run the benchmarks -----------------------------
if __name__ == "__main__" and "--worker" in sys.argv:
    print("RESULT " + json.dumps(pypic3d_worker(*json.loads(sys.argv[-1]))))
    sys.exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Uniform plasma: grid {GRID}, {2 * N_PARTICLES:,} macro-particles, {STEPS} timed steps")
results = []
for cores in CORES:
    print(f"PyPIC3D  {cores} core(s), 1 device ...", flush=True)
    results.append(run_pypic3d(cores, 1))
for devices in DEVICES[1:]:
    print(f"PyPIC3D  all cores, {devices} CPU devices/tiles ...", flush=True)
    results.append(run_pypic3d(0, devices))
if RUN_GPU:
    print("PyPIC3D  GPU (if visible) ...", flush=True)
    results.append(run_pypic3d(0, 1, "cuda"))
if WARPX_BIN:
    for threads in CORES:
        print(f"WarpX    {threads} OpenMP thread(s) ...", flush=True)
        results.append(run_warpx(threads))
results = [r for r in results if r]
for r in results:
    print(f"  {r['code']:8s} {r['platform']:5s} cores={r['cores'] or 'all':>3} devices={r['devices']}  "
          f"{r['s_per_step'] * 1e3:9.2f} ms/step  {r['s_per_step'] * 1e9 / (2 * N_PARTICLES):7.1f} ns/particle/step"
          + (f"  (compile {r['compile_s']:.1f} s)" if "compile_s" in r else ""))

# ----------------------------- plot, save, and print -----------------------------
with open(os.path.join(OUTPUT_DIR, "scaling.json"), "w") as handle:
    json.dump(results, handle, indent=1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for code, marker in (("PyPIC3D", "o"), ("WarpX", "s")):
    rows = [r for r in results if r["code"] == code and r["platform"] == "cpu" and r["devices"] == 1 and r["cores"]]
    if rows:
        cores = np.array([r["cores"] for r in rows])
        t = np.array([r["s_per_step"] for r in rows])
        ax[0].loglog(cores, t * 1e3, marker=marker, lw=2, label=code)
        ax[0].loglog(cores, t[0] * 1e3 / cores * cores[0], "k:", lw=1)
        ax[1].plot(cores, t[0] / t, marker=marker, lw=2, label=f"{code} (cores)")
rows = [r for r in results if r["code"] == "PyPIC3D" and r["platform"] == "cpu" and (r["devices"] > 1 or not r["cores"])]
base = next((r["s_per_step"] for r in results if r["code"] == "PyPIC3D" and r["cores"] == 1), None)
if rows and base:
    ax[1].plot([r["devices"] for r in rows], [base / r["s_per_step"] for r in rows], "^-", lw=2, label="PyPIC3D (devices/tiles)")
for r in results:
    if r["platform"] == "cuda":
        ax[0].axhline(r["s_per_step"] * 1e3, color="tab:green", ls="--", label=f"PyPIC3D GPU ({r['device_kind']})")
ax[1].plot(CORES, CORES, "k:", label="ideal")
ax[0].set(xlabel="CPU cores", ylabel="wall time per step [ms]", title=f"{2 * N_PARTICLES:,} particles, {np.prod(GRID):,} cells")
ax[1].set(xlabel="CPU cores / devices", ylabel="speed-up vs 1 core", title="Strong scaling")
for a in ax:
    a.legend()
    a.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "scaling.png"), dpi=200)
print(f"Saved {OUTPUT_DIR}/scaling.png and scaling.json")
