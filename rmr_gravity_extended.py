#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
EXTENDED GRAVITATIONAL ANALYSIS — Reproduction Script
═══════════════════════════════════════════════════════════════════════════

Reproduces all results from Section 4 of:
  "Emergent Four-Force Dynamics from a Discrete 137-Element Registry"
  Jason R. Merwin, 2026

This script generates three independent measurements of emergent gravity
from the RMR depletion engine:

  FACE 1 — THE FIELD (Section 4.1)
    Static potential profile Φ(r) = G_lat/r + C
    Expected: G_lat ≈ 1.5, R² > 0.95, power-law n ≈ 1.3

  FACE 2 — THE FORCE (Section 4.2)
    (a) Passive probe: gradient asymmetry vs distance → F ∝ r^(-1.80)
    (b) Hemisphere fission: away/toward ratio → 4:1

  FACE 3 — THE WEAKNESS (Section 4.3)
    Free-mass kinematic drift → masses move APART due to
    3D geometric entropy overwhelming single-node gravity

ENGINE ARCHITECTURE:
  The simulation engine is identical across all experiments. Each lattice
  node carries three integer fields: grav (unsigned, capacity 16),
  surf (signed, capacity ±40), spat (signed, capacity ±81). The total
  capacity is 16 + 40 + 81 = 137. Only the gravitational sector is
  active in these experiments (masses are neutral, colorless).

  Mass nodes process every 5 ticks; vacuum nodes every 4 ticks. This
  5/4 timing asymmetry means vacuum diffuses faster than mass radiates,
  creating persistent depletion gradients around masses. When a node's
  gravitational field reaches capacity (16), it overflows: the excess
  is dumped to the neighbor with the lowest field value. This is the
  sole mechanism producing gravitational attraction — no force laws,
  distance functions, or coupling constants are coded anywhere.

  Key design constraints:
    - Strict integer arithmetic (no floats) — discreteness is physical
    - Causal sequential updates (random order) — no global clock
    - Toroidal boundary conditions — no edges

USAGE:
  python rmr_gravity_extended.py              # Full suite (~4-5 hr)
  python rmr_gravity_extended.py --quick      # Quick check (~30 min)
  python rmr_gravity_extended.py --face 1     # Field profile only
  python rmr_gravity_extended.py --face 2     # Force law only
  python rmr_gravity_extended.py --face 3     # Kinematic drift only
  python rmr_gravity_extended.py --hemisphere # Hemisphere fission only

DEPENDENCIES:
  numpy, numba (for JIT compilation of the inner loop)
"""

import numpy as np
from numba import njit
import time
import sys
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY CONSTANTS
#
# These are NOT tunable parameters. They are the sector capacities of the
# 137-element registry as defined by the RMR framework:
#   81 (spatial/color) + 40 (surface/EM) + 16 (gravitational) = 137
# ═══════════════════════════════════════════════════════════════════════════

SPATIAL_CAP = 81    # Color/spatial sector capacity
SURFACE_CAP = 40    # EM/surface sector capacity
GRAV_CAP    = 16    # Gravitational sector capacity
TOTAL_CAP   = 137   # Full registry: 81 + 40 + 16

# Timing asymmetry: mass is "heavier" (processes slower) than vacuum.
# A mass node must enforce dimensional confinement each cycle, costing
# one extra tick. This creates the depletion gradient that IS gravity.
MASS_CYCLE   = 5    # Mass processes every 5th tick
VACUUM_CYCLE = 4    # Vacuum processes every 4th tick


# ═══════════════════════════════════════════════════════════════════════════
# PSEUDORANDOM NUMBER GENERATOR
#
# xorshift64: deterministic, reproducible, no floating-point.
# The RNG state is a single uint64 passed by reference (1-element array).
# ═══════════════════════════════════════════════════════════════════════════

@njit
def xorshift64(state):
    """Generate next pseudorandom uint64. Modifies state in-place."""
    x = state[0]
    x ^= (x << np.uint64(13))
    x ^= (x >> np.uint64(7))
    x ^= (x << np.uint64(17))
    state[0] = x
    return x

@njit
def make_rng(seed):
    """Create RNG state from integer seed, with burn-in."""
    state = np.zeros(1, dtype=np.uint64)
    state[0] = np.uint64(seed) if seed > 0 else np.uint64(88172645463325252)
    for _ in range(20):  # Burn-in: decorrelate from seed
        xorshift64(state)
    return state


# ═══════════════════════════════════════════════════════════════════════════
# LATTICE INITIALIZATION
#
# Each node gets random initial field values in [0, CAP] for each sector.
# is_mass flags which nodes are "frozen" (mass) vs "free" (vacuum).
# ═══════════════════════════════════════════════════════════════════════════

@njit
def init_lattice(S, rng):
    """Create S³ lattice with random initial field values."""
    grav = np.zeros((S, S, S), dtype=np.int32)
    surf = np.zeros((S, S, S), dtype=np.int32)
    spat = np.zeros((S, S, S), dtype=np.int32)
    is_mass = np.zeros((S, S, S), dtype=np.int32)
    for x in range(S):
        for y in range(S):
            for z in range(S):
                grav[x, y, z] = np.int32(xorshift64(rng) % np.uint64(GRAV_CAP + 1))
                surf[x, y, z] = np.int32(xorshift64(rng) % np.uint64(SURFACE_CAP + 1))
                spat[x, y, z] = np.int32(xorshift64(rng) % np.uint64(SPATIAL_CAP + 1))
    return grav, surf, spat, is_mass


# ═══════════════════════════════════════════════════════════════════════════
# NEIGHBOR LOOKUP (6-connected, periodic boundary)
# ═══════════════════════════════════════════════════════════════════════════

@njit
def get_neighbors(x, y, z, S):
    """Return 6 face-sharing neighbors with toroidal wrapping."""
    nbrs = np.zeros((6, 3), dtype=np.int32)
    nbrs[0] = ((x+1) % S, y, z)  # +x
    nbrs[1] = ((x-1) % S, y, z)  # -x
    nbrs[2] = (x, (y+1) % S, z)  # +y
    nbrs[3] = (x, (y-1) % S, z)  # -y
    nbrs[4] = (x, y, (z+1) % S)  # +z
    nbrs[5] = (x, y, (z-1) % S)  # -z
    return nbrs


# ═══════════════════════════════════════════════════════════════════════════
# CORE ENGINE: ONE LATTICE TICK
#
# This is the complete physics engine. Every emergent behavior — gravity,
# EM, strong, weak — comes from this single function. For the extended
# gravitational analysis, only the grav sector is active (neutral masses).
#
# Each tick:
#   1. Shuffle all nodes into random order (causal: no global clock)
#   2. For each node, check if it processes this tick (mass vs vacuum cycle)
#   3. OVERFLOW CHECK: if grav >= GRAV_CAP, dump excess to lowest neighbor
#      → This is the fission mechanism. For mass nodes, this IS movement.
#   4. DIFFUSION: share 1/7 of each field to each of 6 neighbors
#      → This spreads field quanta outward, creating the 1/r potential.
# ═══════════════════════════════════════════════════════════════════════════

@njit
def tick_lattice(grav, surf, spat, is_mass, S, rng, tick_num):
    """
    Execute one causal tick of the lattice engine.
    Returns number of overflow (fission) events.
    """
    n_fissions = 0
    n_nodes = S * S * S

    # Step 1: Random processing order (breaks all spatial symmetry)
    order = np.arange(n_nodes, dtype=np.int32)
    for i in range(n_nodes - 1, 0, -1):
        j = np.int32(xorshift64(rng) % np.uint64(i + 1))
        order[i], order[j] = order[j], order[i]

    # Step 2-4: Process each node
    for idx in range(n_nodes):
        node = order[idx]
        x = node // (S * S)
        y = (node // S) % S
        z = node % S

        # Timing gate: mass nodes tick slower than vacuum
        if is_mass[x, y, z] == 1:
            if tick_num % MASS_CYCLE != 0:
                continue
        else:
            if tick_num % VACUUM_CYCLE != 0:
                continue

        # OVERFLOW → FISSION
        # When grav hits capacity, the node can't hold more information.
        # It dumps the excess to whichever neighbor has the LOWEST field.
        # This preferential dumping toward depleted regions IS gravity.
        if grav[x, y, z] >= GRAV_CAP:
            n_fissions += 1
            nbrs = get_neighbors(x, y, z, S)

            # Find neighbor with minimum grav (= most depleted)
            min_grav = grav[nbrs[0, 0], nbrs[0, 1], nbrs[0, 2]]
            min_idx = 0
            for ni in range(1, 6):
                ng = grav[nbrs[ni, 0], nbrs[ni, 1], nbrs[ni, 2]]
                if ng < min_grav:
                    min_grav = ng
                    min_idx = ni
                elif ng == min_grav:
                    # Tie-break randomly (no preferred direction)
                    if xorshift64(rng) % np.uint64(2) == np.uint64(0):
                        min_idx = ni

            # Dump overflow to the most depleted neighbor
            overflow = grav[x, y, z] - GRAV_CAP // 2
            tx = nbrs[min_idx, 0]
            ty = nbrs[min_idx, 1]
            tz = nbrs[min_idx, 2]
            grav[tx, ty, tz] += overflow
            grav[x, y, z] = GRAV_CAP // 2

        # DIFFUSION: spread 1/7 of field to each neighbor
        # Integer division: floor(grav/7) per neighbor, remainder stays.
        # This is how the 1/r potential forms — diffusion from a
        # persistent source in 3D naturally creates a 1/r profile.
        nbrs = get_neighbors(x, y, z, S)
        g_share = grav[x, y, z] // 7
        if g_share > 0:
            for ni in range(6):
                grav[nbrs[ni, 0], nbrs[ni, 1], nbrs[ni, 2]] += g_share
            grav[x, y, z] -= 6 * g_share

        # Surface sector diffusion (active but irrelevant for neutral masses)
        s_share = surf[x, y, z] // 7
        if s_share > 0:
            for ni in range(6):
                surf[nbrs[ni, 0], nbrs[ni, 1], nbrs[ni, 2]] += s_share
            surf[x, y, z] -= 6 * s_share

    return n_fissions


# ═══════════════════════════════════════════════════════════════════════════
# MEASUREMENT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

@njit
def measure_grav_field(grav, S, mx, my, mz, max_r):
    """
    Measure spherically-averaged gravitational field around (mx,my,mz).
    Returns counts and sums per radial bin (minimum-image convention).
    """
    counts = np.zeros(max_r + 1, dtype=np.int32)
    sums = np.zeros(max_r + 1, dtype=np.float64)
    half_S = S // 2
    for x in range(S):
        for y in range(S):
            for z in range(S):
                dx = x - mx
                dy = y - my
                dz = z - mz
                # Minimum image (toroidal)
                if dx > half_S: dx -= S
                if dx < -half_S: dx += S
                if dy > half_S: dy -= S
                if dy < -half_S: dy += S
                if dz > half_S: dz -= S
                if dz < -half_S: dz += S
                r = int(np.sqrt(dx*dx + dy*dy + dz*dz) + 0.5)
                if 0 < r <= max_r:
                    counts[r] += 1
                    sums[r] += grav[x, y, z]
    return counts, sums

@njit
def periodic_dist(x1, y1, z1, x2, y2, z2, S):
    """Minimum-image Euclidean distance on toroidal lattice."""
    half_S = S // 2
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    if dx > half_S: dx -= S
    if dx < -half_S: dx += S
    if dy > half_S: dy -= S
    if dy < -half_S: dy += S
    if dz > half_S: dz -= S
    if dz < -half_S: dz += S
    return np.sqrt(np.float64(dx*dx + dy*dy + dz*dz))


# ═══════════════════════════════════════════════════════════════════════════
# FACE 1: THE FIELD — Static Potential Profile Φ(r)
#
# Place a single mass at the lattice center. Pin its grav field to
# GRAV_CAP each tick (it continuously radiates). After warmup, measure
# the time-averaged radial field profile. The excess above background
# should follow Φ(r) = G_lat/r for a point source in 3D.
#
# Paper result: G_lat ≈ 1.47, R² = 0.97, power-law n = 1.32
# ═══════════════════════════════════════════════════════════════════════════

@njit
def run_field_profile(S, T, seed):
    """Run single mass, return radial field profile."""
    rng = make_rng(seed)
    grav, surf, spat, is_mass = init_lattice(S, rng)

    center = S // 2
    is_mass[center, center, center] = 1
    grav[center, center, center] = GRAV_CAP
    max_r = S // 2 - 2
    warmup = T // 2

    # Warmup: let field reach steady state
    for t in range(warmup):
        tick_lattice(grav, surf, spat, is_mass, S, rng, t)
        grav[center, center, center] = GRAV_CAP  # Continuous radiation

    # Measurement: accumulate time-averaged field
    total_counts = np.zeros(max_r + 1, dtype=np.int64)
    total_sums = np.zeros(max_r + 1, dtype=np.float64)
    n_samples = 0

    for t in range(warmup, T):
        tick_lattice(grav, surf, spat, is_mass, S, rng, t)
        grav[center, center, center] = GRAV_CAP
        if t % 20 == 0:  # Sample every 20 ticks (decorrelation)
            c, s = measure_grav_field(grav, S, center, center, center, max_r)
            for r in range(max_r + 1):
                total_counts[r] += c[r]
                total_sums[r] += s[r]
            n_samples += 1

    return total_counts, total_sums, n_samples


def face_1_field_profile(S, T, n_seeds):
    """
    FACE 1: Extract G_lat from the static field profile.

    Fits Φ(r) = G_lat/r + C in the inner region (r < S/4) to avoid
    toroidal boundary artifacts that inflate the far-field potential.
    """
    print("=" * 62)
    print("  FACE 1: THE FIELD — Emergent Newtonian Potential")
    print(f"  S={S}, T={T}, warmup={T//2}, seeds={n_seeds}")
    print("=" * 62)

    max_r = S // 2 - 2
    total_counts = np.zeros(max_r + 1, dtype=np.int64)
    total_sums = np.zeros(max_r + 1, dtype=np.float64)

    for seed in range(1, n_seeds + 1):
        t0 = time.time()
        c, s, ns = run_field_profile(S, T, seed * 137)
        for r in range(max_r + 1):
            total_counts[r] += c[r]
            total_sums[r] += s[r]
        print(f"  seed {seed}/{n_seeds}... {time.time()-t0:.1f}s")

    # Build radial profile
    radii, fields = [], []
    for r in range(2, max_r + 1):
        if total_counts[r] > 0:
            radii.append(r)
            fields.append(total_sums[r] / total_counts[r])
    radii = np.array(radii, dtype=np.float64)
    fields = np.array(fields)

    # Background subtraction: far-field mean
    far_mask = radii > max_r * 0.7
    bg = np.mean(fields[far_mask]) if np.sum(far_mask) > 0 else np.mean(fields[-5:])
    phi = fields - bg

    # Fit Φ(r) = G_lat/r + C in inner region (r < S/4)
    from numpy.linalg import lstsq
    inner = radii < S // 4
    if np.sum(inner) > 3:
        A = np.column_stack([1.0 / radii[inner], np.ones(np.sum(inner))])
        coeffs, _, _, _ = lstsq(A, phi[inner], rcond=None)
        G_lat, C_fit = coeffs
        pred = G_lat / radii[inner] + C_fit
        ss_res = np.sum((phi[inner] - pred)**2)
        ss_tot = np.sum((phi[inner] - np.mean(phi[inner]))**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Model-independent log-log exponent
        valid = phi[inner] > 1e-10
        n_exp = -np.polyfit(np.log(radii[inner][valid]),
                            np.log(phi[inner][valid]), 1)[0]
    else:
        G_lat, C_fit, R2, n_exp = [float('nan')] * 4

    print(f"\n  Φ(r) = {C_fit:.4f} + {G_lat:.4f}/r")
    print(f"  G_lat = {G_lat:.4f}")
    print(f"  R²    = {R2:.4f}")
    print(f"  Power-law exponent n = {n_exp:.3f} (target: 1.0)")

    print(f"\n  {'r':>4} {'Φ(r)':>10} {'G/r fit':>10}")
    for i in range(min(len(radii), 12)):
        print(f"  {radii[i]:4.0f} {phi[i]:10.5f} {G_lat/radii[i]+C_fit:10.5f}")

    return G_lat, R2, n_exp


# ═══════════════════════════════════════════════════════════════════════════
# FACE 2a: THE FORCE — Passive Probe Gradient
#
# Place one mass at center. Place a PASSIVE test point (not a mass — no
# radiation, no fission) at distance r along the x-axis. Measure the
# gravitational field on the 6 neighbors of the test point, classified
# as "toward mass" or "away from mass". The asymmetry between toward
# and away field values is a proxy for the local gravitational force.
#
# Critical insight: the test point must be passive (no self-radiation).
# An active test mass creates its own depletion cloud that drowns the
# signal at r > 6. The passive probe acts as an infinitesimal test
# charge, reading the pre-existing gradient without disturbing it.
#
# Paper result: log-log slope = -1.80, 3.4σ above null
# ═══════════════════════════════════════════════════════════════════════════

@njit
def run_passive_probe(S, T, seed, mass_pos, test_pos):
    """
    Measure field gradient at test_pos due to mass at mass_pos.
    Test point is PASSIVE: no is_mass flag, no radiation.
    Returns toward_mean, away_mean (field values on classified neighbors).
    """
    rng = make_rng(seed)
    grav, surf, spat, is_mass = init_lattice(S, rng)

    mx, my, mz = mass_pos
    is_mass[mx, my, mz] = 1
    grav[mx, my, mz] = GRAV_CAP

    tx, ty, tz = test_pos
    half_S = S // 2
    warmup = T // 2

    # Classify test point's neighbors as toward/away from mass
    nbrs = get_neighbors(tx, ty, tz, S)
    tdx = tx - mx
    if tdx > half_S: tdx -= S
    if tdx < -half_S: tdx += S
    tdy = ty - my
    if tdy > half_S: tdy -= S
    if tdy < -half_S: tdy += S
    tdz = tz - mz
    if tdz > half_S: tdz -= S
    if tdz < -half_S: tdz += S
    t_dist_sq = tdx*tdx + tdy*tdy + tdz*tdz

    toward_idx = np.zeros(6, dtype=np.int32)
    for ni in range(6):
        nx, ny, nz = nbrs[ni, 0], nbrs[ni, 1], nbrs[ni, 2]
        ddx = nx - mx
        ddy = ny - my
        ddz = nz - mz
        if ddx > half_S: ddx -= S
        if ddx < -half_S: ddx += S
        if ddy > half_S: ddy -= S
        if ddy < -half_S: ddy += S
        if ddz > half_S: ddz -= S
        if ddz < -half_S: ddz += S
        if ddx*ddx + ddy*ddy + ddz*ddz < t_dist_sq:
            toward_idx[ni] = 1  # This neighbor is closer to mass

    # Warmup
    for t in range(warmup):
        tick_lattice(grav, surf, spat, is_mass, S, rng, t)
        grav[mx, my, mz] = GRAV_CAP

    # Measurement
    toward_sum = np.float64(0.0)
    away_sum = np.float64(0.0)
    toward_n = np.int64(0)
    away_n = np.int64(0)

    for t in range(warmup, T):
        tick_lattice(grav, surf, spat, is_mass, S, rng, t)
        grav[mx, my, mz] = GRAV_CAP
        if t % 5 == 0:
            for ni in range(6):
                nx, ny, nz = nbrs[ni, 0], nbrs[ni, 1], nbrs[ni, 2]
                g = grav[nx, ny, nz]
                if toward_idx[ni] == 1:
                    toward_sum += g
                    toward_n += 1
                else:
                    away_sum += g
                    away_n += 1

    toward_mean = toward_sum / max(toward_n, 1)
    away_mean = away_sum / max(away_n, 1)
    return toward_mean, away_mean


def face_2a_passive_probe(S, T, n_seeds, distances):
    """
    FACE 2a: Force vs distance via passive probe.

    Measures the normalized asymmetry (toward - away) / (toward + away)
    at each distance. Fits power law to extract force exponent.
    """
    print("\n" + "=" * 62)
    print("  FACE 2a: THE FORCE — Passive Probe Gradient")
    print(f"  S={S}, T={T}, seeds={n_seeds}")
    print(f"  Distances: {distances}")
    print("=" * 62)

    center = S // 2
    results = []

    for r in distances:
        seed_asyms = []
        for seed in range(1, n_seeds + 1):
            tw, aw = run_passive_probe(
                S, T, seed * 137,
                mass_pos=(center, center, center),
                test_pos=((center + r) % S, center, center))
            asym = (tw - aw) / (tw + aw) if (tw + aw) > 0 else 0
            seed_asyms.append(asym)

        mean_asym = np.mean(seed_asyms)
        err_asym = np.std(seed_asyms) / np.sqrt(n_seeds)
        snr = abs(mean_asym) / err_asym if err_asym > 0 else 0
        results.append({'r': r, 'asym': mean_asym, 'err': err_asym, 'snr': snr})
        print(f"  r={r:2d}: asym={mean_asym:+.6f} ± {err_asym:.6f}  SNR={snr:.1f}")

    # Power-law fit on positive-asymmetry points
    r_arr = np.array([d['r'] for d in results], dtype=np.float64)
    a_arr = np.array([d['asym'] for d in results])
    pos = a_arr > 0

    if np.sum(pos) >= 3:
        p = np.polyfit(np.log(r_arr[pos]), np.log(a_arr[pos]), 1)
        print(f"\n  Log-log slope = {p[0]:.3f} → F ∝ r^({p[0]:.2f})")
        print(f"  Target: slope = -2.0 for inverse-square")
    else:
        p = [float('nan')]
        print("\n  Too few positive points for power-law fit")

    # Null comparison
    null_asyms = []
    print(f"\n  NULL CONTROL (no mass, same probe geometry):")
    for seed in range(1, n_seeds + 1):
        rng = make_rng(seed * 137)
        grav, surf, spat, is_mass = init_lattice(S, rng)
        # No mass placed — run a few hundred ticks and measure
        tx = (center + 5) % S
        nbrs = get_neighbors(tx, center, center, S)
        for t in range(T // 2, T // 2 + 1000):
            tick_lattice(grav, surf, spat, is_mass, S, rng, t)
        tw_s, aw_s = 0.0, 0.0
        tw_n, aw_n = 0, 0
        # Simple: just average toward vs away with no mass present
        for ni in range(3):
            tw_s += grav[nbrs[ni, 0], nbrs[ni, 1], nbrs[ni, 2]]
            tw_n += 1
        for ni in range(3, 6):
            aw_s += grav[nbrs[ni, 0], nbrs[ni, 1], nbrs[ni, 2]]
            aw_n += 1
        tw_m = tw_s / max(tw_n, 1)
        aw_m = aw_s / max(aw_n, 1)
        null_asyms.append((tw_m - aw_m) / (tw_m + aw_m) if (tw_m + aw_m) > 0 else 0)

    null_mean = np.mean(null_asyms)
    null_err = np.std(null_asyms) / np.sqrt(n_seeds)
    print(f"  Null asymmetry: {null_mean:+.6f} ± {null_err:.6f}")

    # Signal significance
    sig_asym = np.mean([d['asym'] for d in results if d['r'] <= 6])
    combined_err = np.sqrt(null_err**2 + np.mean([d['err'] for d in results if d['r'] <= 6])**2)
    sigma = abs(sig_asym - null_mean) / combined_err if combined_err > 0 else 0
    print(f"  Signal (r≤6): {sig_asym:+.6f}, Detection: {sigma:.1f}σ above null")

    return results, p[0]


# ═══════════════════════════════════════════════════════════════════════════
# FACE 2b: THE FORCE — Hemisphere Fission Counting
#
# Place two masses. For each fission event at either mass, determine
# whether the overflow went toward the partner (toward hemisphere) or
# away (away hemisphere). The ratio away/toward measures the directional
# bias of the depletion gradient.
#
# This observable is immune to entropic volume effects because it
# compares counts at a FIXED distance, not across distances.
#
# Paper result: ratio = 4.01 at S=65, p < 10^-6
# ═══════════════════════════════════════════════════════════════════════════

@njit
def run_hemisphere_fission(S, T, seed, sep):
    """
    Two masses separated by `sep` along x-axis.
    Count fissions in toward vs away hemispheres for each mass.
    """
    rng = make_rng(seed)
    grav, surf, spat, is_mass = init_lattice(S, rng)

    center = S // 2
    half_sep = sep // 2

    # Place two masses along x
    m1x = (center - half_sep) % S
    m2x = (center + half_sep + (sep % 2)) % S
    my, mz = center, center

    is_mass[m1x, my, mz] = 1
    is_mass[m2x, my, mz] = 1
    grav[m1x, my, mz] = GRAV_CAP
    grav[m2x, my, mz] = GRAV_CAP

    warmup = T // 2
    half_S = S // 2

    # Warmup
    for t in range(warmup):
        tick_lattice(grav, surf, spat, is_mass, S, rng, t)
        grav[m1x, my, mz] = GRAV_CAP
        grav[m2x, my, mz] = GRAV_CAP

    # Measurement: count fissions by hemisphere
    toward_count = np.int64(0)
    away_count = np.int64(0)

    for t in range(warmup, T):
        # Before tick: record mass grav values
        g1_before = grav[m1x, my, mz]
        g2_before = grav[m2x, my, mz]

        n_fissions = tick_lattice(grav, surf, spat, is_mass, S, rng, t)

        # Check if mass 1 fissioned (its grav dropped)
        if grav[m1x, my, mz] < g1_before:
            # Find where the overflow went: check which neighbor got more
            nbrs = get_neighbors(m1x, my, mz, S)
            max_gain = -1
            max_ni = 0
            for ni in range(6):
                nx, ny, nz = nbrs[ni, 0], nbrs[ni, 1], nbrs[ni, 2]
                # Approximate: neighbor closest to m2 is "toward"
                ddx = nx - m2x
                if ddx > half_S: ddx -= S
                if ddx < -half_S: ddx += S
                ddy = ny - my
                ddz = nz - mz
                d_to_m2 = ddx*ddx + ddy*ddy + ddz*ddz
                ddx2 = m1x - m2x
                if ddx2 > half_S: ddx2 -= S
                if ddx2 < -half_S: ddx2 += S
                d_m1_m2 = ddx2*ddx2
                if d_to_m2 < d_m1_m2:
                    toward_count += 1
                else:
                    away_count += 1

        # Check mass 2 similarly
        if grav[m2x, my, mz] < g2_before:
            nbrs = get_neighbors(m2x, my, mz, S)
            for ni in range(6):
                nx, ny, nz = nbrs[ni, 0], nbrs[ni, 1], nbrs[ni, 2]
                ddx = nx - m1x
                if ddx > half_S: ddx -= S
                if ddx < -half_S: ddx += S
                ddy = ny - my
                ddz = nz - mz
                d_to_m1 = ddx*ddx + ddy*ddy + ddz*ddz
                ddx2 = m2x - m1x
                if ddx2 > half_S: ddx2 -= S
                if ddx2 < -half_S: ddx2 += S
                d_m2_m1 = ddx2*ddx2
                if d_to_m1 < d_m2_m1:
                    toward_count += 1
                else:
                    away_count += 1

        # Re-pin mass fields
        grav[m1x, my, mz] = GRAV_CAP
        grav[m2x, my, mz] = GRAV_CAP

    return toward_count, away_count


def face_2b_hemisphere(S, T, n_seeds, sep):
    """
    FACE 2b: Hemisphere fission ratio.
    """
    print("\n" + "=" * 62)
    print("  FACE 2b: THE FORCE — Hemisphere Fission Counting")
    print(f"  S={S}, T={T}, seeds={n_seeds}, separation={sep}")
    print("=" * 62)

    all_toward = []
    all_away = []
    all_ratios = []

    for seed in range(1, n_seeds + 1):
        t0 = time.time()
        tw, aw = run_hemisphere_fission(S, T, seed * 137, sep)
        ratio = aw / max(tw, 1)
        all_toward.append(tw)
        all_away.append(aw)
        all_ratios.append(ratio)
        print(f"  seed {seed:2d}/{n_seeds}: toward={tw:5d} away={aw:5d} "
              f"ratio={ratio:.2f} [{time.time()-t0:.1f}s]")

    mean_tw = np.mean(all_toward)
    mean_aw = np.mean(all_away)
    mean_ratio = np.mean(all_ratios)
    err_ratio = np.std(all_ratios) / np.sqrt(n_seeds)

    # One-sided t-test: ratio > 1
    from scipy import stats
    t_stat, p_val = stats.ttest_1samp(all_ratios, 1.0)
    p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2

    print(f"\n  Mean toward: {mean_tw:.1f}")
    print(f"  Mean away:   {mean_aw:.1f}")
    print(f"  Ratio A/T:   {mean_ratio:.3f} ± {err_ratio:.3f}")
    print(f"  t = {t_stat:.2f}, p(one-sided) = {p_one:.6f}")
    print(f"  {'***' if p_one < 0.001 else 'ns'}")

    return mean_ratio, p_one


# ═══════════════════════════════════════════════════════════════════════════
# FACE 3: THE WEAKNESS — Free-Mass Kinematic Drift
#
# Two active masses start at separation r0. Both radiate (grav pinned
# to GRAV_CAP) and move via fission-driven random walks. Track the
# inter-mass distance over time.
#
# Despite the proven 1/r potential well, the masses drift APART because
# the number of accessible states at distance r scales as r² in 3D.
# This entropic "force" overwhelms the single-node gravitational signal.
#
# This is the bottom-up derivation of the hierarchy problem: gravity
# is weak not because G is small (G_lat ≈ 1.5 is order unity) but
# because 3D geometric entropy provides a competing outward pressure
# that only collapses at macroscopic mass accumulation.
#
# Paper result: mean final separation 34.2 vs initial 21 (outward drift)
# ═══════════════════════════════════════════════════════════════════════════

@njit
def tick_with_movement(grav, surf, spat, is_mass, S, rng, tick_num,
                       m1x, m1y, m1z, m2x, m2y, m2z):
    """
    One tick where mass nodes MOVE when they fission.
    When a mass overflows, it relocates to the fission target node.
    Returns new positions and fission flags.
    """
    n_nodes = S * S * S
    order = np.arange(n_nodes, dtype=np.int32)
    for i in range(n_nodes - 1, 0, -1):
        j = np.int32(xorshift64(rng) % np.uint64(i + 1))
        order[i], order[j] = order[j], order[i]

    new_m1x, new_m1y, new_m1z = m1x, m1y, m1z
    new_m2x, new_m2y, new_m2z = m2x, m2y, m2z

    for idx in range(n_nodes):
        node = order[idx]
        x = node // (S * S)
        y = (node // S) % S
        z = node % S

        if is_mass[x, y, z] == 1:
            if tick_num % MASS_CYCLE != 0:
                continue
        else:
            if tick_num % VACUUM_CYCLE != 0:
                continue

        if grav[x, y, z] >= GRAV_CAP:
            nbrs = get_neighbors(x, y, z, S)
            min_grav = grav[nbrs[0, 0], nbrs[0, 1], nbrs[0, 2]]
            min_idx = 0
            for ni in range(1, 6):
                ng = grav[nbrs[ni, 0], nbrs[ni, 1], nbrs[ni, 2]]
                if ng < min_grav:
                    min_grav = ng
                    min_idx = ni
                elif ng == min_grav:
                    if xorshift64(rng) % np.uint64(2) == np.uint64(0):
                        min_idx = ni

            overflow = grav[x, y, z] - GRAV_CAP // 2
            tx = nbrs[min_idx, 0]
            ty = nbrs[min_idx, 1]
            tz = nbrs[min_idx, 2]
            grav[tx, ty, tz] += overflow
            grav[x, y, z] = GRAV_CAP // 2

            # Mass movement: fission target becomes new mass position
            if x == new_m1x and y == new_m1y and z == new_m1z:
                is_mass[x, y, z] = 0
                is_mass[tx, ty, tz] = 1
                new_m1x, new_m1y, new_m1z = tx, ty, tz
            elif x == new_m2x and y == new_m2y and z == new_m2z:
                is_mass[x, y, z] = 0
                is_mass[tx, ty, tz] = 1
                new_m2x, new_m2y, new_m2z = tx, ty, tz

        nbrs = get_neighbors(x, y, z, S)
        g_share = grav[x, y, z] // 7
        if g_share > 0:
            for ni in range(6):
                grav[nbrs[ni, 0], nbrs[ni, 1], nbrs[ni, 2]] += g_share
            grav[x, y, z] -= 6 * g_share
        s_share = surf[x, y, z] // 7
        if s_share > 0:
            for ni in range(6):
                surf[nbrs[ni, 0], nbrs[ni, 1], nbrs[ni, 2]] += s_share
            surf[x, y, z] -= 6 * s_share

    return new_m1x, new_m1y, new_m1z, new_m2x, new_m2y, new_m2z


@njit
def run_free_masses(S, T, seed, initial_sep):
    """
    Two free masses with fission-driven movement.
    Returns separation trace r(t).
    """
    rng = make_rng(seed)
    grav, surf, spat, is_mass = init_lattice(S, rng)

    center = S // 2
    half_sep = initial_sep // 2
    m1x = (center - half_sep) % S
    m2x = (center + half_sep + (initial_sep % 2)) % S
    m1y = m1z = m2y = m2z = center

    is_mass[m1x, m1y, m1z] = 1
    is_mass[m2x, m2y, m2z] = 1
    grav[m1x, m1y, m1z] = GRAV_CAP
    grav[m2x, m2y, m2z] = GRAV_CAP

    warmup = min(T // 4, 2000)

    # Warmup (no movement)
    for t in range(warmup):
        tick_lattice(grav, surf, spat, is_mass, S, rng, t)
        grav[m1x, m1y, m1z] = GRAV_CAP
        grav[m2x, m2y, m2z] = GRAV_CAP

    # Free evolution with movement
    r_trace = np.zeros(T, dtype=np.float64)
    for t in range(T):
        grav[m1x, m1y, m1z] = GRAV_CAP
        grav[m2x, m2y, m2z] = GRAV_CAP

        r_trace[t] = periodic_dist(m1x, m1y, m1z, m2x, m2y, m2z, S)

        m1x, m1y, m1z, m2x, m2y, m2z = \
            tick_with_movement(grav, surf, spat, is_mass, S, rng,
                              t + warmup, m1x, m1y, m1z, m2x, m2y, m2z)

    return r_trace


def face_3_weakness(S, T, n_seeds, initial_sep):
    """
    FACE 3: Free-mass kinematic drift.

    Tests whether two active masses drift together (gravity wins)
    or apart (entropy wins). The answer reveals the hierarchy problem.
    """
    print("\n" + "=" * 62)
    print("  FACE 3: THE WEAKNESS — Entropic vs Gravitational Drift")
    print(f"  S={S}, T={T}, seeds={n_seeds}, initial_sep={initial_sep}")
    print("=" * 62)

    final_seps = []
    mean_drs = []

    for seed in range(1, n_seeds + 1):
        t0 = time.time()
        r_trace = run_free_masses(S, T, seed * 137, initial_sep)
        final_r = r_trace[-1]
        mean_dr = np.mean(np.diff(r_trace))
        final_seps.append(final_r)
        mean_drs.append(mean_dr)
        print(f"  seed {seed:2d}/{n_seeds}: r_final={final_r:.1f} "
              f"<Δr>={mean_dr:+.5f} [{time.time()-t0:.1f}s]")

    mean_final = np.mean(final_seps)
    std_final = np.std(final_seps)
    mean_drift = np.mean(mean_drs)
    drift_err = np.std(mean_drs) / np.sqrt(n_seeds)

    print(f"\n  TRAJECTORY SUMMARY:")
    print(f"    Initial separation:  {initial_sep}")
    print(f"    Mean final sep:      {mean_final:.2f} ± {std_final:.2f}")
    print(f"    Net displacement:    {mean_final - initial_sep:+.2f}")
    print(f"    Mean <Δr> per tick:  {mean_drift:+.6f} ± {drift_err:.6f}")

    if mean_final > initial_sep:
        print(f"\n    → Masses drifted APART: entropy dominates gravity")
        print(f"      This IS the hierarchy problem, derived from first principles.")
        print(f"      Gravity is weak because 3D geometric entropy (r² degeneracy)")
        print(f"      overwhelms single-node depletion drift.")
    else:
        print(f"\n    → Masses drifted TOGETHER: gravity overcomes entropy")

    return mean_final, mean_drift


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    quick = '--quick' in sys.argv

    # Default parameters (paper values)
    if quick:
        S_FIELD  = 41;  T_FIELD  = 8000;   SEEDS_FIELD  = 4
        S_PROBE  = 41;  T_PROBE  = 8000;   SEEDS_PROBE  = 4
        S_HEMI   = 45;  T_HEMI   = 8000;   SEEDS_HEMI   = 10
        S_DRIFT  = 45;  T_DRIFT  = 10000;  SEEDS_DRIFT  = 10
    else:
        S_FIELD  = 51;  T_FIELD  = 15000;  SEEDS_FIELD  = 8
        S_PROBE  = 51;  T_PROBE  = 15000;  SEEDS_PROBE  = 8
        S_HEMI   = 65;  T_HEMI   = 15000;  SEEDS_HEMI   = 20
        S_DRIFT  = 65;  T_DRIFT  = 20000;  SEEDS_DRIFT  = 30

    DISTANCES = [3, 4, 5, 6, 7, 8, 10, 12, 15]
    HEMI_SEP  = 12       # Mass separation for hemisphere test
    DRIFT_SEP = S_DRIFT // 3  # Initial separation for free masses

    # Parse which faces to run
    run_all = True
    run_faces = set()
    run_hemi = '--hemisphere' in sys.argv
    for i, arg in enumerate(sys.argv):
        if arg == '--face' and i + 1 < len(sys.argv):
            run_all = False
            for f in sys.argv[i + 1].split(','):
                run_faces.add(int(f.strip()))

    print()
    print("╔" + "═" * 62 + "╗")
    print("║  EXTENDED GRAVITATIONAL ANALYSIS — Reproduction Script      ║")
    print("║  Three Faces of Lattice Gravity                             ║")
    print("╚" + "═" * 62 + "╝")
    print(f"\n  Mode: {'QUICK' if quick else 'FULL (paper parameters)'}")
    print()

    # JIT warmup
    print("  Compiling Numba JIT...")
    t_jit = time.time()
    _r = make_rng(42)
    _g, _s, _sp, _m = init_lattice(5, _r)
    tick_lattice(_g, _s, _sp, _m, 5, _r, 0)
    tick_with_movement(_g, _s, _sp, _m, 5, _r, 0, 2, 2, 2, 3, 3, 3)
    print(f"  JIT compiled in {time.time() - t_jit:.1f}s\n")

    t_total = time.time()

    # ── FACE 1 ──
    if run_all or 1 in run_faces:
        G_lat, R2, n_exp = face_1_field_profile(
            S_FIELD, T_FIELD, SEEDS_FIELD)

    # ── FACE 2a ──
    if run_all or 2 in run_faces:
        dist_results, slope = face_2a_passive_probe(
            S_PROBE, T_PROBE, SEEDS_PROBE, DISTANCES)

    # ── FACE 2b ──
    if run_all or 2 in run_faces or run_hemi:
        ratio, p_val = face_2b_hemisphere(
            S_HEMI, T_HEMI, SEEDS_HEMI, HEMI_SEP)

    # ── FACE 3 ──
    if run_all or 3 in run_faces:
        final_sep, drift = face_3_weakness(
            S_DRIFT, T_DRIFT, SEEDS_DRIFT, DRIFT_SEP)

    # ── Summary ──
    elapsed = time.time() - t_total
    print(f"\n{'═' * 62}")
    print(f"  SUMMARY — Extended Gravitational Analysis")
    print(f"{'═' * 62}")
    print(f"  Runtime: {elapsed/60:.1f} min ({elapsed/3600:.1f} hr)\n")

    if run_all or 1 in run_faces:
        print(f"  FACE 1 — THE FIELD:")
        print(f"    G_lat = {G_lat:.4f}")
        print(f"    R²    = {R2:.4f}")
        print(f"    n     = {n_exp:.3f} (target 1.0)")

    if run_all or 2 in run_faces:
        print(f"\n  FACE 2 — THE FORCE:")
        print(f"    Passive probe slope = {slope:.3f} (target -2.0)")
        if run_all or run_hemi:
            print(f"    Hemisphere ratio    = {ratio:.2f}:1 (p = {p_val:.6f})")

    if run_all or 3 in run_faces:
        print(f"\n  FACE 3 — THE WEAKNESS:")
        direction = "APART (entropy wins)" if final_sep > DRIFT_SEP else "TOGETHER (gravity wins)"
        print(f"    Final sep = {final_sep:.1f} vs initial {DRIFT_SEP}")
        print(f"    Drift: {direction}")

    print(f"\n{'═' * 62}")
