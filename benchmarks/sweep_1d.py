"""1D problem-size sweep of PyPIC3D on CPU and GPU, on 1 and 2 devices (domain split into tiles):
  - grid sweep:     NX_SWEEP cells at PPC_FIXED particles per cell
  - particle sweep: PPC_SWEEP particles per cell at NX_FIXED cells
Each point is the mean +/- standard deviation over REPEATS fresh processes. Run on a GPU node:
    python benchmarks/sweep_1d.py            (writes outputs/sweep_1d/sweep_<machine>.json and the figure)
    python benchmarks/sweep_1d.py --plot     (redraws from the saved JSON)"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------- input parameters -----------------------------
NX_SWEEP, PPC_FIXED = [5_000, 10_000, 20_000], 1_000
PPC_SWEEP, NX_FIXED = [100, 250, 500, 1_000, 2_000, 4_000], 10_000
STEPS, REPEATS = 5, 3            # timed steps per run (after 2 warm-up steps), fresh processes per point
GPUS = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.split("GPU ")) - 1 if shutil.which("nvidia-smi") else 0
HARDWARE = [("cpu", 1), ("cpu", 2)] + [("cuda", n) for n in (1, 2) if n <= GPUS]  # (backend, devices = tiles)
DENSITY, TEMPERATURE_EV = 1.0e18, 10.0
OUTPUT_DIR = "outputs/sweep_1d"
MACHINE = os.environ.get("NERSC_HOST", platform.node())

E, ME, MP, EPS = 1.602176634e-19, 9.1093837e-31, 1.67262192e-27, 8.8541878128e-12
DX = np.sqrt(EPS * TEMPERATURE_EV * E / (DENSITY * E**2))           # one Debye length per cell


def worker(backend, devices, nx, ppc):
    """Time PyPIC3D steps in this process (called through the subprocess entry point below)."""
    os.environ.update(JAX_PLATFORMS=backend, JAX_NUM_CPU_DEVICES=str(devices), JAX_ENABLE_X64="1",
                      CUDA_VISIBLE_DEVICES=",".join(map(str, range(devices))), XLA_PYTHON_CLIENT_PREALLOCATE="false")
    import jax
    from PyPIC3D.initialization import initialize_simulation

    species = lambda name, q, m: {"name": name, "N_particles": nx * ppc, "charge": q, "mass": m,
                                  "vth": float(np.sqrt(TEMPERATURE_EV * E / m)), "number_density": DENSITY}
    loop, particles, fields, static, dynamic, *_, species_config = initialize_simulation({
        "simulation_parameters": {"name": "sweep", "output_dir": os.path.join(OUTPUT_DIR, "run"), "solver": "electrodynamic_yee",
                                  "Nt": STEPS, "Nx": nx, "Ny": 1, "Nz": 1, "x_wind": nx * DX, "y_wind": 1.0, "z_wind": 1.0,
                                  "cfl": 0.99, "shape_factor": 1, "current_calculation": "j_from_rhov", "filter_j": "none",
                                  "particle_tile_nx": nx // devices, "particle_tile_capacity_factor": 1.5},
        "plotting": {"plotting_interval": 10**9},
        "particle1": species("electrons", -E, ME), "particle2": species("ions", E, MP)})
    step = jax.jit(lambda p, f: loop(p, species_config, f, static, dynamic))
    for _ in range(2):                                                # compile (twice: first outputs change sharding)
        particles, fields = jax.block_until_ready(step(particles, fields))
    start = time.perf_counter()
    for _ in range(STEPS):
        particles, fields = step(particles, fields)
    jax.block_until_ready(fields)
    return (time.perf_counter() - start) / STEPS


def measure(backend, devices, nx, ppc):
    times = []
    for _ in range(REPEATS):
        out = subprocess.run([sys.executable, __file__, "--worker", json.dumps([backend, devices, nx, ppc])],
                             capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(out.stderr[-1500:])
            return None
        times.append(float(line[-1][7:]))
    return {"backend": backend, "devices": devices, "nx": nx, "ppc": ppc, "particles": 2 * nx * ppc,
            "mean": float(np.mean(times)), "std": float(np.std(times)), "times": times}


# ----------------------------- run the sweep -----------------------------
if "--worker" in sys.argv:
    print("RESULT", worker(*json.loads(sys.argv[-1])))
    sys.exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
path = os.path.join(OUTPUT_DIR, f"sweep_{MACHINE}.json")
if "--plot" not in sys.argv:
    points = sorted({(nx, PPC_FIXED) for nx in NX_SWEEP} | {(NX_FIXED, ppc) for ppc in PPC_SWEEP}, key=lambda p: p[0] * p[1])
    print(f"{MACHINE}: {len(points)} problem sizes x {len(HARDWARE)} hardware setups x {REPEATS} repeats")
    results = []
    for backend, devices in HARDWARE:
        for nx, ppc in points:
            start = time.perf_counter()
            r = measure(backend, devices, nx, ppc)
            if r:
                results.append(r)
                print(f"  {backend} x{devices}  nx={nx:6d} ppc={ppc:5d}  {r['mean'] * 1e3:10.1f} +/- {r['std'] * 1e3:7.1f} ms/step  "
                      f"{r['mean'] * 1e9 / r['particles']:7.2f} ns/particle  ({time.perf_counter() - start:.0f} s)", flush=True)
            with open(path, "w") as handle:
                json.dump(results, handle, indent=1)

# ----------------------------- plot, save, and print -----------------------------
results = json.load(open(path))
label = lambda b, d: f"{'GPU' if b == 'cuda' else 'CPU node'}{'s' if d > 1 and b == 'cuda' else ''} x{d}"
pick = lambda b, d, key, value: sorted((r for r in results if r["backend"] == b and r["devices"] == d and r[key] == value),
                                       key=lambda r: r["particles"])
hardware = sorted({(r["backend"], r["devices"]) for r in results})
shades = dict(zip(hardware, plt.cm.Blues(np.linspace(0.45, 0.95, len(hardware)))))
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for b, d in hardware:
    rows = sorted((r for r in results if r["backend"] == b and r["devices"] == d), key=lambda r: r["particles"])
    n = np.array([r["particles"] for r in rows])
    ax[0].errorbar(n, [r["mean"] * 1e9 / r["particles"] for r in rows], [r["std"] * 1e9 / r["particles"] for r in rows],
                   marker="o", lw=2, capsize=3, color=shades[(b, d)], label=label(b, d))
    rows = pick(b, d, "nx", NX_FIXED)
    ax[1].errorbar([r["ppc"] for r in rows], [r["mean"] * 1e3 for r in rows], [r["std"] * 1e3 for r in rows],
                   marker="o", lw=2, capsize=3, color=shades[(b, d)], label=label(b, d))
for b in ("cpu", "cuda"):
    one = {r["particles"]: r for r in results if r["backend"] == b and r["devices"] == 1}
    two = sorted((r for r in results if r["backend"] == b and r["devices"] == 2 and r["particles"] in one), key=lambda r: r["particles"])
    if two:
        eff = [one[r["particles"]]["mean"] / (2 * r["mean"]) for r in two]
        err = [e * np.hypot(one[r["particles"]]["std"] / one[r["particles"]]["mean"], r["std"] / r["mean"]) for e, r in zip(eff, two)]
        ax[2].errorbar([r["particles"] for r in two], eff, err, marker="o", lw=2, capsize=3,
                       color=shades.get((b, 2), "C0"), label=f"{'GPU' if b == 'cuda' else 'CPU'}: 2 devices vs 1")
ax[2].axhline(1.0, color="k", ls=":", lw=1, label="perfect scaling")
ax[0].set(xscale="log", yscale="log", xlabel="total particles", ylabel="wall time per particle-step [ns]", title="Cost per particle")
ax[1].set(xscale="log", yscale="log", xlabel=f"particles per cell (nx = {NX_FIXED:,})", ylabel="wall time per step [ms]",
          title="Particle-count sweep")
ax[2].set(xscale="log", xlabel="total particles", ylabel="parallel efficiency  t(1) / (2 t(2))", title="Two-device efficiency")
for a in ax:
    a.legend(fontsize=9)
    a.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"sweep_1d_{MACHINE}.png"), dpi=300)
print(f"Saved {OUTPUT_DIR}/sweep_1d_{MACHINE}.png")
