"""Large 2D strong-scaling run: 800 x 800 cells, 500M macro-particles (electrons + ions), on 1, 2, 4 GPUs,
sweeping the user-set particle_batch_size (no code changes). Run on a GPU node:
    python benchmarks/large_2d.py            (writes outputs/large_2d/large_2d_<machine>_<particles>M.json and the figure)
    python benchmarks/large_2d.py --plot     (redraws from every saved particle count)
LARGE2D_PARTICLES sets the particle count (e.g. 100000000 for a problem that fits on one GPU), LARGE2D_GPUS the
device counts (e.g. 1; results for other counts already saved are kept); LARGE2D_NX and
LARGE2D_PARTICLES also shrink it for a quick test. With no GPU, CPU devices stand in for GPUs."""
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
WANTED = [int(n) for n in os.environ.get("LARGE2D_GPUS", "1,2,4").split(",")]  # device counts to run (e.g. LARGE2D_GPUS=1)
ACCEL, COUNTS = ("cuda", [n for n in WANTED if n <= GPUS]) if GPUS else ("cpu", [n for n in WANTED if n <= 2])
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
    import PyPIC3D.initialization
    from PyPIC3D.initialization import initialize_simulation

    # skip the one-time initial histograms: they gather every particle onto GPU 0 (out of memory at 500M);
    # the timed step is unchanged
    PyPIC3D.initialization.particles_for_output = lambda *args, **kwargs: []

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
        return {"backend": backend, "devices": devices, "batch": batch, "error": out.stderr[-4000:]}
    return {"backend": backend, "devices": devices, "batch": batch, "particles": PARTICLES, "nx": NX, **json.loads(line[-1][7:])}


# ----------------------------- run the benchmark -----------------------------
if "--worker" in sys.argv:
    print("RESULT", json.dumps(worker(*json.loads(sys.argv[-1]))))
    sys.exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
path = os.path.join(OUTPUT_DIR, f"large_2d_{MACHINE}_{PARTICLES // 10**6}M.json")
if "--plot" not in sys.argv:
    runs = [(ACCEL, n, b) for n in COUNTS for b in BATCHES]
    print(f"{MACHINE}: {NX} x {NX} cells, {PARTICLES:,} particles, {len(runs)} runs, GPUs {GPUS}")
    # keep earlier results for this particle count that this call does not rerun
    results = [r for r in (json.load(open(path)) if os.path.exists(path) else []) if (r["backend"], r["devices"], r["batch"]) not in runs]
    for backend, devices, batch in runs:
        start = time.perf_counter()
        r = measure(backend, devices, batch)
        results.append(r)
        print(f"  {backend} x{devices}  batch {batch or 'auto'}: " + (r["error"].strip().splitlines()[-1] if "error" in r else
              f"batch size {r['batch_size']:,}  {r['s_per_step']:.3f} s/step  (init {r['init_s']:.0f} s)  {r['pypic3d']}") +
              f"  [{time.perf_counter() - start:.0f} s]", flush=True)
        with open(path, "w") as handle:
            json.dump(results, handle, indent=1)

# ----------------------------- plot, save, and print -----------------------------
# left: time per step vs batch size at every particle count; right: speed-up vs 1 GPU (best batch size)
runs = {int(f.split("_")[-1][:-6]): json.load(open(os.path.join(OUTPUT_DIR, f))) for f in os.listdir(OUTPUT_DIR)
        if f.startswith(f"large_2d_{MACHINE}_") and f.endswith("M.json")}
name = lambda b, d: f"{d} {'GPU' if b == 'cuda' else 'CPU device'}{'s' if d > 1 else ''}"
keys = sorted({(m, r["backend"], r["devices"]) for m, rows in runs.items() for r in rows})
shades = dict(zip(keys, plt.cm.Blues(np.linspace(0.4, 0.95, len(keys)))))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
oom = []
for m, b, d in keys:
    rows = [r for r in runs[m] if (r["backend"], r["devices"]) == (b, d)]
    ok = sorted((r for r in rows if "error" not in r and r["batch"] is not None), key=lambda r: r["batch_size"])
    color, label = shades[(m, b, d)], f"{name(b, d)}, {m}M particles"
    if ok:
        ax[0].plot([r["batch_size"] for r in ok], [r["s_per_step"] for r in ok], "o-", lw=2, ms=7, color=color, label=label)
    for r in rows:
        if r["batch"] is None and "error" not in r:
            ax[0].plot(r["batch_size"], r["s_per_step"], "s", ms=9, mfc="none", mew=2, color=color)
        elif r["batch"] is None and r["error"].startswith("over") and ok:   # timed out: 2 warm-up + STEPS steps took longer
            floor = (TIMEOUT - np.mean([o["init_s"] for o in ok])) / (STEPS + 2)
            ax[0].annotate("", (1024, 2 * floor), (1024, floor), arrowprops=dict(arrowstyle="->", color=color, lw=2))
            ax[0].plot([1024, 1024], [floor, 2.5 * floor], "s", ms=9, mfc="none", mew=2, color=color, markevery=[0])
            ax[0].text(1300, floor, f"default batch size:\n> {floor / 60:.0f} min/step (timed out)", color=color, fontsize=8, va="center")
    status = lambda r: (f"{r['s_per_step']:.3f} s" if "error" not in r else "out of memory" if "RESOURCE_EXHAUSTED" in r["error"]
                        else r["error"].strip().splitlines()[-1][:60])
    print(f"  {label}: " + ", ".join(f"batch {r['batch'] or 'auto'} {status(r)}" for r in rows))
    if all("RESOURCE_EXHAUSTED" in r.get("error", "") for r in rows if r["batch"] is not None or len(rows) == 1):
        oom.append(f"{name(b, d)}, {m}M: out of memory")
for m, rows in sorted(runs.items()):
    hardware = sorted({(r["backend"], r["devices"]) for r in rows if "error" not in r})
    best = {k: min(r["s_per_step"] for r in rows if (r["backend"], r["devices"]) == k and "error" not in r) for k in hardware}
    if hardware and hardware[0][1] == 1 and len(hardware) > 1:
        n = [d for _, d in hardware]
        ax[1].plot(n, [best[hardware[0]] / best[k] for k in hardware], "o-", lw=2, ms=7,
                   color=shades[(m, *hardware[-1])], label=f"{m}M particles (best batch size)")
        print(f"  {m}M speed-up vs 1 device: " + ", ".join(f"{d}: {best[hardware[0]] / best[k]:.2f}x" for k, (_, d) in zip(hardware, hardware)))
ax[0].text(0.03, 0.04, "\n".join(oom), transform=ax[0].transAxes, ha="left", va="bottom", fontsize=8, color="0.35")
ax[1].plot([1, 4], [1, 4], "k:", lw=1, label="ideal")
ax[0].set(xscale="log", yscale="log", xlabel="particle_batch_size", ylabel="wall time per step [s]", title="Batch-size sweep")
ax[1].set(xscale="log", yscale="log", xlabel="GPUs", ylabel="speed-up vs. 1 GPU", title="Strong scaling")
ax[1].set_xticks([1, 2, 4], ["1", "2", "4"])
ax[1].set_yticks([1, 2, 4], ["1", "2", "4"])
ax[1].minorticks_off()
for a in ax:
    a.legend(fontsize=9)
    a.grid(alpha=0.3, which="both")
fig.suptitle(f"PyPIC3D 2D, {NX} x {NX} cells, NERSC Perlmutter A100 80 GB" if MACHINE == "perlmutter" else f"PyPIC3D 2D, {NX} x {NX} cells")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"large_2d_{MACHINE}.png"), dpi=300)
print(f"Saved {OUTPUT_DIR}/large_2d_{MACHINE}.png")
