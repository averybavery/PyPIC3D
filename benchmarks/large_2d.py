"""Large 2D strong-scaling run: 800 x 800 cells, 500M macro-particles (electrons + ions), on 1, 2, 4 GPUs,
sweeping the user-set particle_batch_size (no code changes). Run on a GPU node:
    python benchmarks/large_2d.py            (writes outputs/large_2d/large_2d_<machine>.json and the figure)
    python benchmarks/large_2d.py --plot     (redraws from the saved JSON)
LARGE2D_NX / LARGE2D_PARTICLES shrink the problem for a quick test; with no GPU, CPU devices stand in for GPUs."""
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
NX = int(os.environ.get("LARGE2D_NX", 800))                       # cells along x and y
PARTICLES = int(os.environ.get("LARGE2D_PARTICLES", 500_000_000))  # macro-particles, half electrons and half ions
STEPS, TIMEOUT = 3, 1800         # timed steps per run (after 2 warm-up steps); seconds before a run is abandoned
BATCHES = [None, 2**20, 2**22, 2**24, 10**12]  # particle_batch_size: None = automatic default, 10**12 = whole tile
GPUS = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.split("GPU ")) - 1 if shutil.which("nvidia-smi") else 0
ACCEL, COUNTS = ("cuda", [n for n in (1, 2, 4) if n <= GPUS]) if GPUS else ("cpu", [1, 2])
DENSITY, TEMPERATURE_EV = 1.0e18, 10.0
OUTPUT_DIR = "outputs/large_2d"
MACHINE = os.environ.get("NERSC_HOST", platform.node())

E, ME, MP, EPS = 1.602176634e-19, 9.1093837e-31, 1.67262192e-27, 8.8541878128e-12
DX = np.sqrt(EPS * TEMPERATURE_EV * E / (DENSITY * E**2))           # one Debye length per cell


def worker(backend, devices, batch):
    """Time PyPIC3D steps in this process (called through the subprocess entry point below)."""
    os.environ.update(JAX_PLATFORMS=backend, JAX_NUM_CPU_DEVICES=str(devices), JAX_ENABLE_X64="1",
                      CUDA_VISIBLE_DEVICES=",".join(map(str, range(devices))), XLA_PYTHON_CLIENT_MEM_FRACTION="0.95")
    import jax
    import PyPIC3D
    from PyPIC3D.initialization import initialize_simulation

    species = lambda name, q, m: {"name": name, "N_particles": PARTICLES // 2, "charge": q, "mass": m,
                                  "vth": float(np.sqrt(TEMPERATURE_EV * E / m)), "number_density": DENSITY}
    start = time.perf_counter()
    loop, particles, fields, static, dynamic, *_, species_config = initialize_simulation({
        "simulation_parameters": {"name": "large_2d", "output_dir": os.path.join(OUTPUT_DIR, "run"), "solver": "electrodynamic_yee",
                                  "Nt": STEPS, "Nx": NX, "Ny": NX, "Nz": 1, "x_wind": NX * DX, "y_wind": NX * DX, "z_wind": 1.0,
                                  "cfl": 0.99, "shape_factor": 1, "current_calculation": "j_from_rhov", "filter_j": "none",
                                  "particle_tile_nx": NX // devices, "particle_tile_capacity_factor": 1.2,
                                  **({} if batch is None else {"particle_batch_size": batch})},
        "plotting": {"plotting_interval": 10**9},
        "particle1": species("electrons", -E, ME), "particle2": species("ions", E, MP)})
    init_s = time.perf_counter() - start
    step = jax.jit(lambda p, f: loop(p, species_config, f, static, dynamic))
    for _ in range(2):                                                # compile (twice: first outputs change sharding)
        particles, fields = jax.block_until_ready(step(particles, fields))
    start = time.perf_counter()
    for _ in range(STEPS):
        particles, fields = step(particles, fields)
    jax.block_until_ready(fields)
    return {"s_per_step": (time.perf_counter() - start) / STEPS, "init_s": init_s, "batch_size": int(static.particle_batch_size),
            "cores": len(os.sched_getaffinity(0)), "pypic3d": os.path.dirname(PyPIC3D.__file__)}


def measure(backend, devices, batch):
    try:
        out = subprocess.run([sys.executable, __file__, "--worker", json.dumps([backend, devices, batch])],
                             capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"backend": backend, "devices": devices, "batch": batch, "error": f"over {TIMEOUT} s"}
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        return {"backend": backend, "devices": devices, "batch": batch, "error": out.stderr[-600:]}
    return {"backend": backend, "devices": devices, "batch": batch, "particles": PARTICLES, "nx": NX, **json.loads(line[-1][7:])}


# ----------------------------- run the benchmark -----------------------------
if "--worker" in sys.argv:
    print("RESULT", json.dumps(worker(*json.loads(sys.argv[-1]))))
    sys.exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
path = os.path.join(OUTPUT_DIR, f"large_2d_{MACHINE}.json")
if "--plot" not in sys.argv:
    runs = [(ACCEL, n, b) for n in COUNTS for b in (BATCHES if n > 1 else [None])]
    print(f"{MACHINE}: {NX} x {NX} cells, {PARTICLES:,} particles, {len(runs)} runs, GPUs {GPUS}")
    results = []
    for backend, devices, batch in runs:
        start = time.perf_counter()
        r = measure(backend, devices, batch)
        results.append(r)
        print(f"  {backend} x{devices}  batch {batch or 'auto'}: " + (r["error"] if "error" in r else
              f"batch size {r['batch_size']:,}  {r['s_per_step']:.3f} s/step  (init {r['init_s']:.0f} s)  {r['pypic3d']}") +
              f"  [{time.perf_counter() - start:.0f} s]", flush=True)
        with open(path, "w") as handle:
            json.dump(results, handle, indent=1)

# ----------------------------- plot, save, and print -----------------------------
results = [r for r in json.load(open(path)) if "error" not in r]
hardware = sorted({(r["backend"], r["devices"]) for r in results})
runs = {k: [r for r in results if (r["backend"], r["devices"]) == k] for k in hardware}
best = {k: min(rows, key=lambda r: r["s_per_step"]) for k, rows in runs.items()}
default = {k: next((r for r in rows if r["batch"] is None), None) for k, rows in runs.items()}
one = best.get(hardware[0]) if hardware and hardware[0][1] == 1 else None   # 1-device reference
shades = dict(zip(hardware, plt.cm.Blues(np.linspace(0.45, 0.95, len(hardware)))))
name = lambda b, d: f"{d} {'GPU' if b == 'cuda' else 'CPU device'}{'s' if d > 1 else ''}"
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for k in hardware:
    rows = sorted((r for r in runs[k] if r["batch"] is not None), key=lambda r: r["batch_size"])
    if rows:
        ax[0].plot([r["batch_size"] for r in rows], [r["s_per_step"] for r in rows], "o-", lw=2, color=shades[k], label=name(*k))
    if default[k]:
        ax[0].plot(default[k]["batch_size"], default[k]["s_per_step"], "s", ms=9, mfc="none", mew=2, color=shades[k],
                   label=f"{name(*k)}, default batch size")
if one:
    x, width = np.arange(len(hardware)), 0.38
    for shift, pick, label, style in ((-0.5, default, "default batch size", dict(hatch="//", fill=False)),
                                      (0.5, best, "best batch size", {})):
        values = [one["s_per_step"] / pick[k]["s_per_step"] if pick[k] else np.nan for k in hardware]
        bars = ax[1].bar(x + shift * width, values, width, label=label, edgecolor=[shades[k] for k in hardware],
                         color=[shades[k] for k in hardware], lw=1.5, **style)
        ax[1].bar_label(bars, labels=[f"{v:.2g}x" if np.isfinite(v) else "" for v in values], fontsize=9, padding=2)
    ax[1].plot(x, [d for _, d in hardware], "k:", lw=1, label="ideal")
    ax[1].set_xticks(x, [name(*k) for k in hardware])
    ax[1].legend(fontsize=9)
ax[0].set(xscale="log", yscale="log", xlabel="particle_batch_size", ylabel="wall time per step [s]", title="Batch-size sweep")
ax[1].set(yscale="log", ylabel="speed-up vs. 1 GPU", title="Multi-GPU speed-up")
ax[0].legend(fontsize=9)
for a in ax:
    a.grid(alpha=0.3, which="both")
fig.suptitle(f"PyPIC3D 2D, {NX} x {NX} cells, {PARTICLES / 1e6:g}M particles")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"large_2d_{MACHINE}.png"), dpi=300)
for k in hardware:
    print(f"  {name(*k)}: best batch size {best[k]['batch_size']:,}, {best[k]['s_per_step']:.3f} s/step"
          + (f", {one['s_per_step'] / best[k]['s_per_step']:.2f}x 1 device" if one else ""))
print(f"Saved {OUTPUT_DIR}/large_2d_{MACHINE}.png")
