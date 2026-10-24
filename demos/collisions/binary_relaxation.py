"""Binary Coulomb collision demos (0-D, all particles in one cell):
  1. temperature equilibration of two equal-mass species, against the Spitzer rate and WarpX (identical inputs)
  2. electron-positron thermalization, with the exponential relaxation rate of |T_e - T_p| checked
     against the Spitzer rate nu = 2 nu_ep evaluated at t = 0"""
import os
import shutil
import subprocess
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from PyPIC3D.collisions import apply_collisions
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles

jax.config.update("jax_enable_x64", True)

# ----------------------------- input parameters -----------------------------
N = 20_480                          # macro-particles per species
CELLS = 8                           # particles fill CELLS^3 cells (WarpX's minimum block), paired cell by cell
WARPX = shutil.which("warpx.3d")    # set to None to skip the WarpX comparison
COULOMB_LOG = 10.0
EQUIL = dict(t_a=12_000.0, t_b=1_000.0, density=1.0e20, steps=100, relax_steps=25.0)  # [K], [m^-3]; ~e-fold in relax_steps
THERM = dict(t_e=2_800.0, t_p=30.0, rate=0.008, dt=0.1, t_end=20.0)  # [K]; target nu / omega_pe; dt, t_end in 1 / omega_pe
OUTPUT_DIR = "outputs/binary_relaxation"
SEED = 0

E, ME, KB, EPS = 1.602176634e-19, 9.1093837e-31, 1.380649e-23, 8.8541878128e-12


def spitzer(ta, tb, n):
    """Equal-mass energy-exchange rate nu_ab (dT_a/dt = nu_ab (T_b - T_a)); temperatures in K."""
    return 2 * E**4 * n * COULOMB_LOG / (3 * (2 * np.pi) ** 1.5 * EPS**2 * ME**2 * (KB * (ta + tb) / ME) ** 1.5)


def run(temps, density, dt, steps, label):
    """Collide an electron and a positron population in one cell; return T_a(t), T_b(t) [K]."""
    keys = jax.random.split(jax.random.PRNGKey(SEED), 4)
    u = jnp.stack([jnp.sqrt(KB * t / ME) * jax.random.normal(k, (N, 3)) for k, t in zip(keys, temps)])
    particles = TiledParticles(x=jax.random.uniform(keys[2], (1, 1, 1, 2, N, 3), maxval=CELLS), u=u[None, None, None],
                               active=jnp.ones((1, 1, 1, 2, N), bool), key=keys[3])
    config = SpeciesConfig(charge=jnp.array([-E, E]), mass=jnp.array([ME, ME]), weight=jnp.full(2, density * CELLS**3 / N),
                           update_x=jnp.zeros((2, 3), bool))
    specs = (("binary", 0, 0, COULOMB_LOG), ("binary", 1, 1, COULOMB_LOG), ("binary", 0, 1, COULOMB_LOG))
    world = SimpleNamespace(dx=1.0, dy=1.0, dz=1.0, dt=dt, eps=EPS, kb=KB)
    step = jax.jit(lambda p: apply_collisions(p, config, SimpleNamespace(collisions=specs), world)[0])
    history = []
    for i in range(steps + 1):
        v = np.asarray(particles.u[0, 0, 0])
        history.append(ME * np.mean(np.sum((v - v.mean(1, keepdims=True)) ** 2, -1), -1) / (3 * KB))
        if i % max(steps // 5, 1) == 0:
            print(f"  {label}: step {i:4d}/{steps}  T = {history[-1][0]:8.1f} K, {history[-1][1]:8.1f} K", flush=True)
        if i < steps:
            particles = step(particles)
    return np.array(history)


def run_warpx(temps, density, dt, steps):
    """Same electron-positron box in WarpX (Perez binary collisions); return T_a(t), T_b(t) [K]."""
    out = os.path.abspath(os.path.join(OUTPUT_DIR, "warpx_equilibration"))
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    length = 1.0e-6 * CELLS
    species = "".join(f"""{name}.charge = {q}
{name}.mass = m_e
{name}.injection_style = NRandomPerCell
{name}.num_particles_per_cell = {N // CELLS**3}
{name}.profile = constant
{name}.density = {density}
{name}.do_not_deposit = 1
{name}.momentum_distribution_type = gaussian
{name}.ux_th = {np.sqrt(KB * t / ME) / 2.99792458e8}
{name}.uy_th = {np.sqrt(KB * t / ME) / 2.99792458e8}
{name}.uz_th = {np.sqrt(KB * t / ME) / 2.99792458e8}
""" for name, q, t in (("a", "-q_e", temps[0]), ("b", "q_e", temps[1])))
    with open(os.path.join(out, "inputs"), "w") as handle:
        handle.write(f"""max_step = {steps}
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
particles.species_names = a b
{species}collisions.collision_names = caa cbb cab
caa.species = a a
cbb.species = b b
cab.species = a b
caa.CoulombLog = {COULOMB_LOG}
cbb.CoulombLog = {COULOMB_LOG}
cab.CoulombLog = {COULOMB_LOG}
warpx.reduced_diags_names = PE
PE.type = ParticleEnergy
PE.intervals = 1
""")
    subprocess.run([WARPX, "inputs"], cwd=out, capture_output=True, check=True)
    pe = np.loadtxt(os.path.join(out, "diags/reducedfiles/PE.txt"))
    return 2 / 3 * pe[:, 6:8] / KB                                    # mean kinetic energy per particle -> T [K]


def spitzer_curve(temps, density, dt, steps):
    """Integrate dT_a/dt = nu (T_b - T_a), dT_b/dt = nu (T_a - T_b) with the temperature-dependent Spitzer rate."""
    t, out = np.array(temps, float), [np.array(temps, float)]
    for _ in range(steps):
        for _ in range(50):
            t = t + dt / 50 * spitzer(*t, density) * np.array([t[1] - t[0], t[0] - t[1]])
        out.append(t.copy())
    return np.array(out)


# ----------------------------- run the simulations -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("[1/2] Temperature equilibration")
eq = EQUIL
nu_eq = 2 * spitzer(eq["t_a"], eq["t_b"], eq["density"])
dt_eq = 1 / (nu_eq * eq["relax_steps"])
eq_history = run((eq["t_a"], eq["t_b"]), eq["density"], dt_eq, eq["steps"], "equilibration")
eq_spitzer = spitzer_curve((eq["t_a"], eq["t_b"]), eq["density"], dt_eq, eq["steps"])
eq_warpx = run_warpx((eq["t_a"], eq["t_b"]), eq["density"], dt_eq, eq["steps"]) if WARPX else None

print("[2/2] Electron-positron thermalization")
th = THERM
omega_1 = np.sqrt(E**2 / (EPS * ME))                                 # omega_pe at n = 1 m^-3
density = (th["rate"] / (2 * spitzer(th["t_e"], th["t_p"], 1.0) / omega_1)) ** 2  # nu / omega_pe scales as sqrt(n)
omega_pe = omega_1 * np.sqrt(density)
nu = 2 * spitzer(th["t_e"], th["t_p"], density)                      # rate of d|T_e - T_p| / dt at t = 0
steps = int(round(th["t_end"] / th["dt"]))
print(f"  n = {density:.3e} m^-3, omega_pe = {omega_pe:.3e} 1/s, Spitzer nu = {nu / omega_pe:.4f} omega_pe")
th_history = run((th["t_e"], th["t_p"]), density, th["dt"] / omega_pe, steps, "thermalization")

# ----------------------------- plot, save, and print -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))
steps_axis = np.arange(eq["steps"] + 1)
ax.plot(steps_axis, eq_history[:, 0], color="tab:blue", lw=2, label="species A (PyPIC3D)")
ax.plot(steps_axis, eq_history[:, 1], color="tab:orange", lw=2, label="species B (PyPIC3D)")
if eq_warpx is not None:
    ax.plot(steps_axis[::3], eq_warpx[::3, 0], "o", color="tab:blue", ms=4, mfc="none", label="species A (WarpX)")
    ax.plot(steps_axis[::3], eq_warpx[::3, 1], "o", color="tab:orange", ms=4, mfc="none", label="species B (WarpX)")
ax.plot(steps_axis, eq_spitzer[:, 0], color="0.4", ls="-.", lw=1.3, label="Spitzer (Maxwellian) theory")
ax.plot(steps_axis, eq_spitzer[:, 1], color="0.4", ls="-.", lw=1.3)
ax.axhline(eq_history[0].mean(), color="k", ls="--", lw=1.5, label="isolated-system equilibrium")
ax.set(xlabel="Collision step", ylabel="Temperature [K]", title="Binary-collision temperature equilibration")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "binary_equilibration.png"), dpi=300)

t = np.arange(steps + 1) * th["dt"]                                  # time in 1 / omega_pe
dT = np.abs(th_history[:, 0] - th_history[:, 1])
slope, intercept = np.polyfit(t, np.log(dT), 1)
nu_fit = -slope
equilibrium = th_history[0].mean()
analytic = 0.5 * dT[0] * np.exp(-nu / omega_pe * t)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].plot(t, th_history[:, 0], color="#9bbcff", lw=2, label="Electrons (data)")
ax[0].plot(t, th_history[:, 1], color="#ff9aaa", lw=2, label="Positrons (data)")
ax[0].plot(t, equilibrium + analytic, "k--", lw=1.8, label=f"Analytic Spitzer rate (nu={nu / omega_pe:.4f})")
ax[0].plot(t, equilibrium - analytic, "k--", lw=1.8)
ax[0].axhline(equilibrium, color="0.55", ls=":", label="Theoretical equilibrium")
ax[0].set(title="Coupled Temperature Relaxation", xlabel=r"Time ($\omega_{pe}^{-1}$)", ylabel="Temperature (K)")
ax[1].semilogy(t, dT, color="#c45bd5", lw=2, label=r"$|T_e - T_p|$ data")
ax[1].semilogy(t, np.exp(intercept - nu_fit * t), "k--", lw=1.8, label=f"Exponential fit (nu={nu_fit:.4f})")
ax[1].set(title="Relaxation Rate Verification", xlabel=r"Time ($\omega_{pe}^{-1}$)", ylabel=r"$|\Delta T|$ (log scale)")
for a in ax:
    a.grid(True, linestyle=":", alpha=0.55)
    a.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "binary_thermalization.png"), dpi=300)

print(f"Equilibration: equilibrium {eq_history[0].mean():.0f} K; final dT: PyPIC3D {eq_history[-1, 0] - eq_history[-1, 1]:.0f} K, "
      + (f"WarpX {eq_warpx[-1, 0] - eq_warpx[-1, 1]:.0f} K, " if eq_warpx is not None else "")
      + f"Spitzer {eq_spitzer[-1, 0] - eq_spitzer[-1, 1]:.0f} K (initial {eq['t_a'] - eq['t_b']:.0f} K)")
print(f"Thermalization: Spitzer nu = {nu / omega_pe:.5f} omega_pe, fitted nu = {nu_fit:.5f} omega_pe, "
      f"ratio {nu_fit * omega_pe / nu:.3f}")
print(f"Saved {OUTPUT_DIR}/binary_equilibration.png and binary_thermalization.png")
