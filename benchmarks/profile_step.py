"""Where does a PyPIC3D step spend its time on 1 vs 2 devices? Times each stage of the electrodynamic Yee step
(push, position update, tile refresh, current deposition, field update) and lists the most expensive XLA ops of the
full step from a JAX profiler trace. Same 1D setup as benchmarks/sweep_1d.py. Run on a GPU node:
    python benchmarks/profile_step.py        (writes outputs/profile_step/profile_<machine>.json)"""
import collections
import glob
import gzip
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

# ----------------------------- input parameters -----------------------------
NX, PPC = 10_000, [250, 1_000]   # grid cells, particles per cell per species
REPEATS, TOP_OPS = 5, 15         # timed calls per stage (median), XLA ops listed from the trace
GPUS = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.split("GPU ")) - 1 if shutil.which("nvidia-smi") else 0
HARDWARE = [("cuda", n) for n in (1, 2) if n <= GPUS] or [("cpu", 1), ("cpu", 2)]  # (backend, devices = tiles)
DENSITY, TEMPERATURE_EV = 1.0e18, 10.0
OUTPUT_DIR = "outputs/profile_step"
MACHINE = os.environ.get("NERSC_HOST", platform.node())

E, ME, MP, EPS = 1.602176634e-19, 9.1093837e-31, 1.67262192e-27, 8.8541878128e-12
DX = np.sqrt(EPS * TEMPERATURE_EV * E / (DENSITY * E**2))           # one Debye length per cell


def worker(backend, devices, nx, ppc):
    """Time each stage of one step in this process (called through the subprocess entry point below)."""
    os.environ.update(JAX_PLATFORMS=backend, JAX_NUM_CPU_DEVICES=str(devices), JAX_ENABLE_X64="1",
                      CUDA_VISIBLE_DEVICES=",".join(map(str, range(devices))), XLA_PYTHON_CLIENT_PREALLOCATE="false")
    import jax
    from PyPIC3D.deposition.J_from_rhov import J_from_rhov
    from PyPIC3D.initialization import initialize_simulation
    from PyPIC3D.particles.particle_tile_communication import refresh_tiled_particle_tiles, update_tiled_particle_positions
    from PyPIC3D.pusher.particle_push import particle_push
    from PyPIC3D.solvers.yee.first_order_yee import update_B, update_E
    from PyPIC3D.utilities.field_helpers import yee_push_fields

    species = lambda name, q, m: {"name": name, "N_particles": nx * ppc, "charge": q, "mass": m,
                                  "vth": float(np.sqrt(TEMPERATURE_EV * E / m)), "number_density": DENSITY}
    loop, p, f, static, dyn, *_, sc = initialize_simulation({
        "simulation_parameters": {"name": "profile", "output_dir": os.path.join(OUTPUT_DIR, "run"), "solver": "electrodynamic_yee",
                                  "Nt": 5, "Nx": nx, "Ny": 1, "Nz": 1, "x_wind": nx * DX, "y_wind": 1.0, "z_wind": 1.0,
                                  "cfl": 0.99, "shape_factor": 1, "current_calculation": "j_from_rhov", "filter_j": "none",
                                  "particle_tile_nx": nx // devices, "particle_tile_capacity_factor": 1.5},
        "plotting": {"plotting_interval": 10**9},
        "particle1": species("electrons", -E, ME), "particle2": species("ions", E, MP)})
    E_field, B_field, J, pml = f[0], f[1], f[2], f[6]
    push_E, push_B = jax.jit(lambda: yee_push_fields(E_field, B_field, f[5], static, dyn))()
    B_half = lambda e, b: update_B(e, b, static, dyn, pml)[0]
    stages = {
        "full step": (lambda p, f: loop(p, sc, f, static, dyn), (p, f)),
        "particle push": (lambda p, e, b: particle_push(p, sc, e, b, static, dyn), (p, push_E, push_B)),
        "position update (1 of 2)": (lambda p: update_tiled_particle_positions(p, sc, dyn.dt / 2), (p,)),
        "tile refresh (1 of 2)": (lambda p: refresh_tiled_particle_tiles(p, static, dyn), (p,)),
        "current deposition": (lambda p, j: J_from_rhov(p, sc, j, static, dyn), (p, J)),
        "field update B, E, B": (lambda e, b, j: B_half(e, update_E(e, B_half(e, b), j, static, dyn, pml)[0]), (E_field, B_field, J)),
    }
    times = {}
    for name, (fn, args) in stages.items():
        step = jax.jit(fn)
        out = jax.block_until_ready(step(*args))
        if name == "full step":                                       # compile twice: first outputs change sharding
            args = jax.block_until_ready(step(*out))
        samples = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            jax.block_until_ready(step(*args))
            samples.append(time.perf_counter() - start)
        times[name] = float(np.median(samples))
        if name == "full step":
            trace_dir = os.path.join(OUTPUT_DIR, "trace", f"{backend}{devices}_ppc{ppc}")
            shutil.rmtree(trace_dir, ignore_errors=True)
            with jax.profiler.trace(trace_dir, create_perfetto_trace=True):
                jax.block_until_ready(step(*args))
            events = json.load(gzip.open(glob.glob(f"{trace_dir}/**/perfetto_trace.json.gz", recursive=True)[0]))
            ops = collections.Counter()
            for e in events["traceEvents"] if isinstance(events, dict) else events:
                if e.get("ph") == "X" and not e["name"].startswith(("$", "ThunkExecutor", "PjitFunction", "Thunk")):
                    ops[e["name"]] += e.get("dur", 0) * 1e-6                # summed over devices and threads [s]
    return {"backend": backend, "devices": devices, "nx": nx, "ppc": ppc, "stages": times, "top_ops": ops.most_common(TOP_OPS)}


# ----------------------------- run the profile -----------------------------
if "--worker" in sys.argv:
    print("RESULT", json.dumps(worker(*json.loads(sys.argv[-1]))))
    sys.exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"{MACHINE}: profiling {len(PPC)} problem sizes on {HARDWARE}", flush=True)
results = []
for ppc in PPC:
    for backend, devices in HARDWARE:
        print(f"  {backend} x{devices}, nx = {NX}, ppc = {ppc} ...", flush=True)
        out = subprocess.run([sys.executable, __file__, "--worker", json.dumps([backend, devices, NX, ppc])], capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(out.stderr[-1500:])
            continue
        results.append(json.loads(line[-1][7:]))
        with open(os.path.join(OUTPUT_DIR, f"profile_{MACHINE}.json"), "w") as handle:
            json.dump(results, handle, indent=1)

# ----------------------------- print and save -----------------------------
for ppc in PPC:
    rows = [r for r in results if r["ppc"] == ppc]
    print(f"\nnx = {NX}, ppc = {ppc} ({2 * NX * ppc:,} particles): stage time [ms]")
    print(f"  {'stage':26s}" + "".join(f"{r['backend'] + ' x' + str(r['devices']):<14s}" for r in rows) + ("ratio 2/1" if len(rows) == 2 else ""))
    for stage in rows[0]["stages"] if rows else []:
        values = [1e3 * r["stages"][stage] for r in rows]
        print(f"  {stage:26s}" + "".join(f"{v:<14.1f}" for v in values) + (f"{values[1] / values[0]:.2f}" if len(rows) == 2 else ""))
    for r in rows:
        print(f"  most expensive XLA ops, {r['backend']} x{r['devices']} full step [ms, summed over devices]:")
        for name, seconds in r["top_ops"]:
            print(f"    {1e3 * seconds:9.1f}  {name[:100]}")
print(f"\nSaved {OUTPUT_DIR}/profile_{MACHINE}.json")
