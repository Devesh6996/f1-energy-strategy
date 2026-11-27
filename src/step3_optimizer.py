"""
F1 Energy Strategy - Step 3: Energy deployment optimizer
=========================================================
Builds a lap simulator from a REAL lap and finds the best way to spend and
harvest electrical energy within the 2026 rules.

Simulator ("quasi-steady-state", the standard approach in motorsport):
  - Corners & braking zones keep the real driver's speeds (grip-limited: energy can't change them).
  - Straights are simulated with physics: engine + electric power, drag (from Step 2c's drag range),
    the rules' speed-based MGU-K cap, traction limits, and braking from the real braking performance.
  - The battery is tracked metre by metre: 4 MJ usable, a per-lap harvest limit, braking regen,
    lift-and-coast and super-clipping harvest.

Optimizer ("energy market", greedy marginal value):
  - Repeatedly spend the next packet of energy where it saves the most time, and harvest where it
    costs the least time (braking regen is free; lift-and-coast and super-clipping cost time).
  - Stop when no trade saves time any more.

Questions answered:
  1. Best deployment map for this lap vs. a simple "deploy until empty" strategy.
  2. Which harvesting style is faster here: lift-and-coast only, super-clipping only, or both?
  3. (race mode, --defend) Lap-time cost of keeping enough energy to defend each long straight.

Run:
    python src/step3_optimizer.py --event Monza --driver LEC
    python src/step3_optimizer.py --event Belgium --driver VER --mode race --defend
    python src/step3_optimizer.py --event Monza --driver LEC --robustness    # repeat at both drag bounds
"""

import argparse
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import step2_power_model as s2
from step1_energy_fingerprint import OUT_DIR, label_energy_modes, load_session

OPT_DIR = OUT_DIR / "optimizer"

# --- Rules & car (adjustable) ------------------------------------------------
SOC_MAX_MJ = 4.0              # usable battery
HARVEST_CAP_MJ = {"quali": 7.0, "race": 8.5}   # max harvest per lap (rules vary by circuit/session)
RACE_SOC0_MJ = 2.0            # race mode: start the lap half full and finish with at least as much
RACE_DRAW_MJ = 0.0            # race mode: how much of the reserve the lap may use (0 = energy-neutral)
DEPLOY_LIMIT_MJ = float("inf")  # max electrical energy deployable per lap (set by the joint calibration)
EFF = s2.DRIVETRAIN_EFF       # battery -> wheels (deploy) and wheels -> battery (harvest)
LIFT_HARVEST_KW = 350         # MGU-K harvest while lifting and coasting
QUALI_MASS = 778
RACE_MASS = 768 + 35          # mid-race fuel

# --- Simulator / optimizer settings ------------------------------------------
DS = 8.0                      # metres per simulation step
CHUNK_M = 200                 # straights are split into chunks of this length; each gets a power level
STEP_KW = 70                  # optimizer changes a chunk's level by this much at a time
LIFT_STEP_M = 30              # ... or extends a lift-and-coast by this much
LIFT_MAX_M = 300
MAX_BRAKE_DECEL = 59.0        # m/s^2 (~6 g): no F1 car brakes harder than this
CORNER_DRAG = 0.04            # tyre scrub in corners: extra resistance = this x lateral force
                              #   (~ the tyre's slip angle in radians; 0.03-0.05 is typical)
GEO_SMOOTH_M = 120            # smooth position data over this distance before taking curvature
# Mid-2026 rule: full 350 kW only in key acceleration zones (corner exit to braking point);
# elsewhere - here: flat-out CORNERS - deployment is limited to 250 kW.
NON_ACCEL_ZONE_KW = 250
CORNER_RADIUS_M = 800         # flat-out sections tighter than this radius count as corners, not straights
SC_MAX_KW = 350               # deepest super-clip (harvest at full throttle)


# ============================================================================= track model
def lap_geometry(lap, lap_df):
    """X, Y, Z (metres) at every telemetry sample, from FastF1 position data. None if unavailable."""
    try:
        pos = lap.get_pos_data()
        t_pos = pos["Time"].dt.total_seconds().to_numpy()
        t_car = lap_df["Time"].dt.total_seconds().to_numpy()
        geo = pd.DataFrame({"Distance": lap_df["Distance"].to_numpy()})
        for c in ("X", "Y", "Z"):
            geo[c] = np.interp(t_car, t_pos, pos[c].to_numpy(dtype=float) / 10.0)   # 1/10 m -> m
        if geo[["X", "Y"]].std().min() < 10:        # broken / empty position data
            return None
        return geo
    except Exception:
        return None


class Track:
    """A real lap turned into a simulation grid."""

    def __init__(self, df: pd.DataFrame, geo: pd.DataFrame = None):
        self.df = df
        s = np.arange(0, df["Distance"].max(), DS)
        self.s = s
        self.n = len(s)
        self.v_obs = np.interp(s, df["Distance"], df["Speed"]) / 3.6
        idx = np.clip(np.searchsorted(df["Distance"].to_numpy(), s), 0, len(df) - 1)
        mode = df["mode"].to_numpy()[idx]
        # Power-limited ("free") = where the driver was flat out, clipping or coasting before a brake.
        free = np.isin(mode, ["full_deploy", "clipping", "lift_coast"])
        free = pd.Series(free).rolling(3, center=True, min_periods=1).median().astype(bool).to_numpy()
        self.free = free
        self._geometry(geo)
        # Grip limit: real speeds in corners/braking; in flat-out sections only the cornering-grip limit
        free_limit = np.maximum(self.v_grip, self.v_obs)       # the real car did go this fast there
        self.limit = np.where(free, free_limit, self.v_obs)

        # Chunks: each free run split into ~CHUNK_M pieces
        self.chunk = np.full(self.n, -1)
        self.runs = []                                         # (start, end) index of each free run
        c, i = 0, 0
        while i < self.n:
            if free[i]:
                j = i
                while j < self.n and free[j]:
                    j += 1
                self.runs.append((i, j))
                k = max(1, int(round((j - i) * DS / CHUNK_M)))
                for part in np.array_split(np.arange(i, j), k):
                    self.chunk[part] = c
                    c += 1
                i = j
            else:
                i += 1
        self.n_chunks = c
        # A braking zone starts where a free run ends -> lift-and-coast happens just before it
        self.brake_starts = [end for _, end in self.runs if end < self.n]
        self.longest_runs = sorted(self.runs, key=lambda r: r[1] - r[0], reverse=True)

        self._fit_limits(df)
        self._backward_pass()
        # plain-Python copies for the simulation loop (indexing numpy arrays one by one is slow)
        self.vb_l, self.free_l, self.chunk_l = self.vb.tolist(), self.free.tolist(), self.chunk.tolist()
        self.kappa_l, self.grade_l = self.kappa.tolist(), self.grade.tolist()
        # Deployment limit per point: 350 kW on straights, 250 kW in flat-out corners (needs geometry)
        corner = self.free & (self.kappa > 1.0 / CORNER_RADIUS_M) if self.has_geo else np.zeros(self.n, bool)
        self.flat_corner = corner
        self.zone_cap_l = np.where(corner, NON_ACCEL_ZONE_KW, s2.MGU_K_MAX_KW).astype(float).tolist()
        self.trac_v_l, self.trac_a_l = self.trac_v.tolist(), self.trac_a.tolist()

    def _geometry(self, geo):
        """Curvature (1/m) and gradient (rise/run) along the lap, plus a cornering-grip limit
        fitted from how hard the real car cornered."""
        n = self.n
        self.kappa, self.grade, self.v_grip = np.zeros(n), np.zeros(n), np.full(n, np.inf)
        self.has_geo = geo is not None
        self.grip_coef = (np.nan, np.nan)
        if not self.has_geo:
            return
        from scipy.signal import savgol_filter
        win = max(5, int(GEO_SMOOTH_M / DS) | 1)
        xyz = [savgol_filter(np.interp(self.s, geo["Distance"], geo[c]), win, 3) for c in ("X", "Y", "Z")]
        x, y, z = xyz
        dx, dy = np.gradient(x, DS), np.gradient(y, DS)
        ddx, ddy = np.gradient(dx, DS), np.gradient(dy, DS)
        self.kappa = np.abs(dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-6)
        self.kappa = np.clip(self.kappa, 0, 0.05)              # tighter than a 20 m radius = noise
        path_ds = np.hypot(dx, dy)                             # metres of XY path per metre of s
        self.grade = np.clip(np.gradient(z, DS) / np.maximum(path_ds, 0.5), -0.2, 0.2)
        # Lateral grip grows with downforce: a_lat_max = c0 + c1 v^2. Fit the upper envelope
        a_lat = self.v_obs ** 2 * self.kappa
        ok = (self.kappa > 1 / 3000) & (self.v_obs > 60 / 3.6)
        if ok.sum() > 20:
            d = pd.DataFrame({"v": self.v_obs[ok], "a": a_lat[ok]})
            q = d.groupby((d["v"] * 3.6 // 20)).agg(v=("v", "median"), a=("a", lambda a: a.quantile(0.95)))
            A = np.column_stack([np.ones(len(q)), q["v"] ** 2])
            c0, c1 = np.clip(np.linalg.lstsq(A, q["a"], rcond=None)[0], [5, 0], [40, 0.01])
            self.grip_coef = (c0, c1)
            excess = self.kappa - c1
            self.v_grip = np.where(excess > 1e-6, np.sqrt(c0 / np.maximum(excess, 1e-9)), np.inf)

    def _fit_limits(self, df):
        v = df["Speed"] / 3.6
        # Braking: decel grows with downforce ~ a + b v^2. Fit on the upper part of the real data.
        br = df[df["Brake"] & (df["Speed"] > 100) & (df["accel"] < 0)]
        if len(br) > 10:
            bins = (br["Speed"] // 20)
            q = br.groupby(bins).agg(v=("Speed", "median"), d=("accel", lambda a: -a.quantile(0.25)))
            A = np.column_stack([np.ones(len(q)), (q["v"] / 3.6) ** 2])
            self.brake_coef = np.clip(np.linalg.lstsq(A, q["d"], rcond=None)[0], [10, 0], [45, 0.006])
        else:
            self.brake_coef = np.array([35.0, 0.003])
        # Traction: at low speed the tyres, not the power, limit acceleration.
        acc = df[df["mode"].eq("full_deploy") & (df["accel"] > 0)]
        bins = np.arange(0, 400, 20)
        env = acc.groupby(pd.cut(acc["Speed"], bins), observed=False)["accel"].quantile(0.95)
        self.trac_v = np.array([b.mid for b in env.index]) / 3.6
        self.trac_a = (env.to_numpy() * 1.15)
        ok = ~np.isnan(self.trac_a)
        self.trac_v, self.trac_a = self.trac_v[ok], self.trac_a[ok]

    def drag_bounds(self, mass):
        """Re-measure the drag range on THIS lap with the SAME physics as the simulator:
        gradient, corner scrub and the zone-dependent deployment limit. Same method as Step 2c
        (lower bound: engine power only; upper bound: max deployment the rules allow), but
        consistent with the simulator, so the calibration check is like-for-like."""
        df = self.df
        full = df["mode"].eq("full_deploy").astype(int)
        stable = full.rolling(2 * s2.EDGE_TRIM + 1, center=True, min_periods=1).min().eq(1).to_numpy()
        d = df["Distance"].to_numpy()
        grade = np.interp(d, self.s, self.grade)
        kappa = np.interp(d, self.s, self.kappa)
        zone = np.interp(d, self.s, np.array(self.zone_cap_l))
        v = df["Speed"].to_numpy() / 3.6
        y = (mass * df["accel"].to_numpy() * v + s2.CRR * mass * s2.G * v
             + mass * s2.G * grade * v + CORNER_DRAG * mass * v * v * kappa * v)
        speed = df["Speed"].to_numpy()
        fit_min = max(250.0, float(np.quantile(speed[stable], 0.85))) if stable.any() else 250.0
        sel = stable & (speed >= fit_min)
        if sel.sum() < 8:
            return np.nan, np.nan
        x = v[sel] ** 3
        allowed = np.array([(s2.ICE_NOMINAL_KW + min(cap_kw(sp), z)) * 1000 * EFF
                            for sp, z in zip(speed[sel], zone[sel])])
        ice = s2.ICE_NOMINAL_KW * 1000 * EFF
        k_hi = float(np.median((allowed - y[sel]) / x))
        k_lo = float(np.median((ice - y[sel]) / x))
        return (k_lo if k_lo > 0 else np.nan), (k_hi if k_hi > 0 else np.nan)

    def measured_deploy_mj(self, k, mass):
        """Electrical energy the REAL car deployed on this lap, inferred from its telemetry at drag k:
        wheel power from the equation of motion (incl. gradient + corner scrub), minus the engine's share,
        integrated over the full-throttle samples."""
        df = self.df
        d = df["Distance"].to_numpy()
        grade = np.interp(d, self.s, self.grade)
        kappa = np.interp(d, self.s, self.kappa)
        v = df["Speed"].to_numpy() / 3.6
        p = v * (mass * df["accel"].to_numpy() + k * v * v + s2.CRR * mass * s2.G
                 + mass * s2.G * grade + CORNER_DRAG * mass * v * v * kappa)
        p = pd.Series(p).rolling(5, center=True, min_periods=1).median().to_numpy()
        full = df["Throttle"].to_numpy() >= 98
        elec = np.where(full, np.clip(p - s2.ICE_NOMINAL_KW * 1000 * EFF, 0, None), 0) / EFF
        dt = np.diff(df["t"].to_numpy(), prepend=df["t"].iloc[0])
        return float((elec * dt).sum() / 1e6)

    def brake_decel(self, v):
        return min(self.brake_coef[0] + self.brake_coef[1] * v * v, MAX_BRAKE_DECEL)

    def traction_limit(self, v):
        if v > 50 or not self.trac_v_l:              # above ~180 km/h acceleration is power-limited
            return 1e9
        xs, ys = self.trac_v_l, self.trac_a_l
        if v <= xs[0]:
            return ys[0]
        for j in range(1, len(xs)):
            if v <= xs[j]:
                return ys[j - 1] + (ys[j] - ys[j - 1]) * (v - xs[j - 1]) / (xs[j] - xs[j - 1])
        return ys[-1]

    def _backward_pass(self):
        """Fastest speed at each point that still lets the car brake for what comes next."""
        vb = self.limit.copy()
        vb[-1] = min(vb[-1], self.v_obs[-1])
        for i in range(self.n - 2, -1, -1):
            nxt = vb[i + 1]
            vb[i] = min(vb[i], math.sqrt(nxt * nxt + 2 * self.brake_decel(nxt) * DS))
        self.vb = vb


# ============================================================================= simulator
def cap_kw(v_kmh):
    """2026 MGU-K deployment cap (plain-Python version of s2.deployment_cap_kw: much faster per call)."""
    if v_kmh >= 345:
        return 0.0
    c = 6900 - 20 * v_kmh if v_kmh >= 340 else 1800 - 5 * v_kmh
    return max(0.0, min(float(s2.MGU_K_MAX_KW), c))


class Car:
    def __init__(self, k, mass):
        self.k, self.m = k, mass
        self.ice_w = s2.ICE_NOMINAL_KW * 1000 * EFF
        self.roll = s2.CRR * mass * s2.G


def simulate(track, car, levels, lift, soc0, harvest_cap, record=False, deploy_floor=0.0):
    """One lap. levels[c] = battery power (kW) in chunk c (+ deploy, - super-clip harvest).
    lift[z] = metres of lift-and-coast before braking zone z. Returns a dict."""
    n, vb, free, chunk = track.n, track.vb_l, track.free_l, track.chunk_l
    coast = [False] * n
    for z, start in enumerate(track.brake_starts):
        d = int(lift[z] / DS)
        for j in range(max(0, start - d), start):
            coast[j] = True
    levels = list(levels)
    kappa, grade, zone_cap = track.kappa_l, track.grade_l, track.zone_cap_l
    mg = car.m * s2.G

    v = [0.0] * n
    v[0] = min(track.v_obs[0], vb[0])
    soc, harvested, deployed, t = soc0 * 1e6, 0.0, 0.0, 0.0
    deploy_cap_j = DEPLOY_LIMIT_MJ * 1e6
    cap_j, soc_max = harvest_cap * 1e6, SOC_MAX_MJ * 1e6
    m, k, roll, ice = car.m, car.k, car.roll, car.ice_w
    rec_p = [0.0] * n if record else None
    rec_soc = [0.0] * n if record else None

    for i in range(n - 1):
        vi = v[i]
        batt_kw = 0.0                                # + out of battery, - into battery
        if coast[i]:
            p_wheel = -LIFT_HARVEST_KW * 1000
            batt_kw = -LIFT_HARVEST_KW * EFF
        elif free[i]:
            lvl = levels[chunk[i]]
            if lvl > 0:
                lvl = min(lvl, cap_kw(vi * 3.6), zone_cap[i])
                if soc <= deploy_floor * 1e6 or deployed >= deploy_cap_j:
                    lvl = 0.0
                p_wheel = ice + lvl * 1000 * EFF
                batt_kw = lvl
            elif lvl < 0:
                p_wheel = ice + lvl * 1000            # harvesting at full throttle costs wheel power
                batt_kw = lvl * EFF
            else:
                p_wheel = ice
        else:
            p_wheel = ice

        extra = mg * grade[i] + CORNER_DRAG * m * vi * vi * kappa[i]   # climbing + tyre scrub in corners
        a = (p_wheel / max(vi, 5.0) - k * vi * vi - roll - extra) / m
        a = min(a, track.traction_limit(vi))
        v_next = math.sqrt(max(vi * vi + 2 * a * DS, 1.0))
        if v_next > vb[i + 1]:
            v_next = vb[i + 1]
            # Regen only from a REAL braking force: the deceleration beyond what drag and rolling
            # resistance already provide. Holding a corner speed (v_next ~ vi) harvests nothing.
            a_real = (v_next * v_next - vi * vi) / (2 * DS)
            v_avg = 0.5 * (vi + v_next)
            f_brake = -m * a_real - k * v_avg * v_avg - roll - extra
            if f_brake > 0:
                regen_kw = min(s2.MGU_K_MAX_KW, f_brake * v_avg / 1000)
                batt_kw = -regen_kw * EFF
            else:
                batt_kw = 0.0
        v[i + 1] = v_next
        dt = 2 * DS / (vi + v_next)
        t += dt

        e = batt_kw * 1000 * dt
        if e > 0:
            e = min(e, max(soc, 0.0), max(deploy_cap_j - deployed, 0.0))
            soc -= e
            deployed += e
        elif e < 0:
            room = min(soc_max - soc, cap_j - harvested)
            stored = min(-e, max(room, 0.0))
            soc += stored
            harvested += stored
        if record:
            rec_p[i], rec_soc[i] = batt_kw, soc / 1e6

    out = {"time": t, "soc_end": soc / 1e6, "deployed_mj": deployed / 1e6, "harvested_mj": harvested / 1e6}
    if record:
        rec_p[-1], rec_soc[-1] = rec_p[-2], rec_soc[-2]
        out.update(v=np.array(v), batt_kw=np.array(rec_p), soc=np.array(rec_soc), coast=np.array(coast))
    return out


# ============================================================================= optimizer
LEVEL_OPTIONS = np.arange(-SC_MAX_KW, s2.MGU_K_MAX_KW + 1, STEP_KW).astype(float)   # -350 ... +350 kW
LIFT_OPTIONS = np.arange(0, LIFT_MAX_M + 1, LIFT_STEP_M).astype(float)              # 0 ... 300 m
PRICES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]                                     # s per MJ


def feasible(r, mode, soc0):
    return mode == "quali" or r["soc_end"] >= soc0 - RACE_DRAW_MJ - 0.02


def optimize(track, car, mode, allow_lift=True, allow_sc=True, fixed=None, init=None, prices=None):
    """ECMS-style optimizer (Equivalent Consumption Minimization Strategy, as used for hybrid cars).

    Energy gets a PRICE lam (seconds per MJ). For a given price, every straight chunk and every
    braking zone picks its best option to minimise  lap_time + lam * net_energy_used, one at a
    time, repeated until nothing changes (coordinate descent). We try a range of prices and keep
    the fastest lap that respects the energy rules. The winning price = the value of energy.
    Returns (levels, lift, result, price)."""
    soc0 = SOC_MAX_MJ if mode == "quali" else RACE_SOC0_MJ
    cap = HARVEST_CAP_MJ[mode]
    fixed = fixed or {}
    lv_opts = LEVEL_OPTIONS if allow_sc else LEVEL_OPTIONS[LEVEL_OPTIONS >= 0]
    lf_opts = LIFT_OPTIONS if allow_lift else np.array([0.0])

    if init is not None:
        levels, lift = init[0].copy(), init[1].copy()
        if not allow_sc:
            levels = np.maximum(levels, 0)
        if not allow_lift:
            lift[:] = 0
    else:
        levels, lift = np.zeros(track.n_chunks), np.zeros(len(track.brake_starts))
    for c, lvl in fixed.items():
        levels[c] = lvl
    free_chunks = [c for c in range(track.n_chunks) if c not in fixed]

    def sim(lv, lf):
        return simulate(track, car, lv, lf, soc0, cap)

    def descend(lam, levels, lift):
        cost = lambda r: r["time"] + lam * (r["deployed_mj"] - r["harvested_mj"])
        for sweep in range(3):
            changed = False
            for c in free_chunks:
                cur = levels[c]
                vals = []
                for o in lv_opts:
                    levels[c] = o
                    vals.append(cost(sim(levels, lift)))
                levels[c] = lv_opts[int(np.argmin(vals))]
                changed |= levels[c] != cur
            for z in range(len(lift)):
                cur = lift[z]
                vals = []
                for o in lf_opts:
                    lift[z] = o
                    vals.append(cost(sim(levels, lift)))
                lift[z] = lf_opts[int(np.argmin(vals))]
                changed |= lift[z] != cur
            if not changed:
                break
        return levels, lift, sim(levels, lift)

    best, last_infeasible = None, None
    r0 = sim(levels, lift)
    if init is not None and feasible(r0, mode, soc0):     # never return worse than the starting point
        best = (levels.copy(), lift.copy(), r0, init[2] if len(init) > 2 else np.nan)
    for lam in (prices or PRICES):
        levels, lift, r = descend(lam, levels, lift)
        if feasible(r, mode, soc0):
            if best is None or r["time"] < best[2]["time"]:
                best = (levels.copy(), lift.copy(), r, lam)
        elif best is None:
            last_infeasible = (levels.copy(), lift.copy(), lam)

    # Bisection on the price between the last too-cheap and the first feasible price (ECMS tuning)
    if best is not None and last_infeasible is not None and mode == "race":
        lo, hi = last_infeasible[2], best[3]
        lv_w, lf_w = best[0].copy(), best[1].copy()
        for _ in range(5):
            mid = (lo + hi) / 2
            lv_w, lf_w, r = descend(mid, lv_w, lf_w)
            if feasible(r, mode, soc0):
                hi = mid
                if r["time"] < best[2]["time"]:
                    best = (lv_w.copy(), lf_w.copy(), r, mid)
            else:
                lo = mid

    # Polish: spend any leftover energy, one step at a time, wherever it saves the most time
    if best is not None:
        levels, lift, r, lam = best[0].copy(), best[1].copy(), best[2], best[3]
        for _ in range(300):
            trials = []
            for c in free_chunks:
                if levels[c] < s2.MGU_K_MAX_KW and (levels[c] >= 0 or True):
                    lv = levels.copy()
                    lv[c] = min(levels[c] + STEP_KW, s2.MGU_K_MAX_KW)
                    if allow_sc or lv[c] >= 0:
                        trials.append((lv, lift))
            for z in range(len(lift)):
                if lift[z] > 0:
                    lf = lift.copy()
                    lf[z] = max(lift[z] - LIFT_STEP_M, 0.0)
                    trials.append((levels, lf))
            best_trial = None
            for lv, lf in trials:
                rt = sim(lv, lf)
                if feasible(rt, mode, soc0) and rt["time"] < r["time"] - 1e-4 and \
                        (best_trial is None or rt["time"] < best_trial[2]["time"]):
                    best_trial = (lv.copy(), lf.copy(), rt)
            if best_trial is None:
                break
            levels, lift, r = best_trial
        best = (levels, lift, r, lam)

    if best is None:                                   # nothing met the energy rules
        r = sim(levels, lift)
        r["infeasible"] = True
        best = (levels.copy(), lift.copy(), r, np.nan)
    return best


def naive_levels(track):
    """Baseline: deploy the maximum allowed everywhere at full throttle until the battery is empty."""
    return np.full(track.n_chunks, float(s2.MGU_K_MAX_KW)), np.zeros(len(track.brake_starts))


# ============================================================================= helpers
def drag_for(session, driver, lap_df, which):
    """Drag factor k from Step 2c's season file if available, else a topspeed fit on this lap."""
    try:
        rnd = int(session.event["RoundNumber"])
    except Exception:
        rnd = -1
    f = OUT_DIR / "season" / f"{rnd:02d}_drag.csv"
    if rnd > 0 and f.exists():
        d = pd.read_csv(f)
        row = d[d["driver"] == driver]
        if len(row) and pd.notna(row["k_hi"].iloc[0]):
            lo, hi = row["k_lo"].iloc[0], row["k_hi"].iloc[0]
            lo = hi if pd.isna(lo) else lo
            return {"lo": lo, "hi": hi, "mid": (lo + hi) / 2}[which], (lo, hi)
    k = s2.fit_car_model(lap_df, "topspeed")["k"]
    print("  (no Step 2c drag range for this event - using a topspeed fit; run step2c_season.py first for ranges)")
    return k, (k, k)


def plot_strategy(track, lap_df, result, title, path):
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1.4, 1]})
    a1.plot(lap_df["Distance"], lap_df["Speed"], color="grey", lw=1, label="real lap")
    a1.plot(track.s, result["v"] * 3.6, color="tab:blue", lw=1.2, label="simulated, optimal strategy")
    a1.set_ylabel("Speed (km/h)")
    a1.set_title(title)
    a1.legend(loc="lower right")
    a1.grid(alpha=0.3)

    p = result["batt_kw"]
    a2.fill_between(track.s, 0, np.where(p > 0, p, 0), color="tab:green", step="post", label="deploy")
    sc = np.where((p < 0) & track.free & ~result["coast"], p, 0)
    lc = np.where(result["coast"], p, 0)
    rg = np.where((p < 0) & ~track.free & ~result["coast"], p, 0)
    a2.fill_between(track.s, 0, sc, color="tab:red", step="post", label="super-clip harvest")
    a2.fill_between(track.s, 0, lc, color="tab:orange", step="post", label="lift-and-coast harvest")
    a2.fill_between(track.s, 0, rg, color="tab:cyan", step="post", label="braking regen")
    a2.set_ylabel("Battery power (kW)\n+ out / - in")
    a2.legend(loc="lower right", ncol=4, fontsize=8)
    a2.grid(alpha=0.3)

    a3.plot(track.s, result["soc"], color="black")
    a3.set_ylabel("Battery (MJ)")
    a3.set_ylim(0, SOC_MAX_MJ * 1.05)
    a3.set_xlabel("Distance (m)")
    a3.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================================= reference lap
TOW_GAP_M = 120          # a car closer than this ahead on a fast straight = the lap was towed
CLEAN_LAP_WINDOW = 1.005 # a clean-air lap must be within 0.5% of the driver's best to be representative


def clean_air_lap(session, driver, allow_tow=False):
    """The driver's fastest lap WITHOUT a car close ahead on the fast straights.
    A towed lap makes the real car look quicker than it is in clean air, which would bias
    the drag calibration low. Returns (lap, telemetry, note)."""
    laps = [lap for _, lap in session.laps.pick_drivers(driver).iterlaps() if pd.notna(lap["LapTime"])]
    if not laps:
        raise ValueError(f"No timed laps for {driver}")
    laps.sort(key=lambda l: l["LapTime"])
    best = laps[0]["LapTime"].total_seconds()
    if allow_tow:
        return laps[0], laps[0].get_car_data().add_distance(), "fastest lap (tow check skipped)"
    for lap in laps:
        if lap["LapTime"].total_seconds() > best * CLEAN_LAP_WINDOW:
            break
        try:
            gap_tel = lap.get_car_data().add_distance().add_driver_ahead()
            fast = gap_tel[(gap_tel["Speed"] >= 280) & (gap_tel["Throttle"] >= 98)]
            min_gap = float(fast["DistanceToDriverAhead"].min()) if len(fast) else np.nan
        except Exception:
            return laps[0], laps[0].get_car_data().add_distance(), \
                "fastest lap (could not check for a tow - gap data unavailable)"
        if pd.isna(min_gap) or min_gap >= TOW_GAP_M:
            label = "fastest lap" if lap is laps[0] else \
                f"fastest CLEAN-AIR lap (+{lap['LapTime'].total_seconds() - best:.3f} s vs best, which was towed)"
            gap_txt = "no car within range" if pd.isna(min_gap) else f"closest car ahead {min_gap:.0f} m"
            return lap, lap.get_car_data().add_distance(), f"{label}; {gap_txt}"
    return laps[0], laps[0].get_car_data().add_distance(), \
        (f"fastest lap - WARNING: every lap within 0.5% of the best had a car < {TOW_GAP_M} m ahead; "
         "calibrated drag may be biased LOW")


# ============================================================================= recharge-lap test
RECHARGE_START_MJ = 0.5                        # start the lap almost empty
RECHARGE_GAINS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]   # 3.5 MJ = refill to a full 4 MJ battery


def recharge_test(track, car, tag, title_prefix):
    """How much lap time does it cost to finish a race lap with MORE energy than it started with?
    Start almost empty, require a gain of G MJ, optimise, compare with the energy-neutral lap."""
    global RACE_SOC0_MJ, RACE_DRAW_MJ
    saved = (RACE_SOC0_MJ, RACE_DRAW_MJ)
    RACE_SOC0_MJ = RECHARGE_START_MJ
    prices = PRICES + [6.0, 10.0, 16.0]          # recharging can make energy very "expensive"
    rows, init = [], None
    print(f"\n  Recharge-lap test: start with {RECHARGE_START_MJ} MJ, finish with at least that + G")
    # Hardest target first: each easier target starts from the previous solution (which already
    # satisfies it), so the cost can only fall as the target gets easier -> a consistent curve.
    for g in sorted(RECHARGE_GAINS, reverse=True):
        RACE_DRAW_MJ = -g                         # negative draw = the lap must GAIN g MJ
        lv, lf, r, lam = optimize(track, car, "race", init=init, prices=prices)
        ok = not r.get("infeasible")
        if ok:
            init = (lv, lf, lam)
        rows.append({"gain_MJ": g, "lap_time_s": r["time"] if ok else np.nan, "feasible": ok,
                     "harvested_MJ": r["harvested_mj"], "deployed_MJ": r["deployed_mj"],
                     "soc_end_MJ": r["soc_end"], "lift_m": float(lf.sum()), "superclip_chunks": int((lv < 0).sum()),
                     "energy_price_s_per_MJ": lam})
    RACE_SOC0_MJ, RACE_DRAW_MJ = saved

    d = pd.DataFrame(rows).sort_values("gain_MJ").reset_index(drop=True)
    base = d.loc[0, "lap_time_s"]
    d["cost_vs_neutral_s"] = d["lap_time_s"] - base
    d["cost_per_MJ_s"] = np.where(d["gain_MJ"] > 0, d["cost_vs_neutral_s"] / d["gain_MJ"], np.nan)
    print(d[["gain_MJ", "feasible", "lap_time_s", "cost_vs_neutral_s", "cost_per_MJ_s", "harvested_MJ",
             "deployed_MJ", "lift_m", "superclip_chunks"]].round(3).to_string(index=False))
    full = d[(d["gain_MJ"] == RECHARGE_GAINS[-1])]
    if len(full) and full["feasible"].iloc[0]:
        print(f"  -> A FULL recharge lap ({RECHARGE_START_MJ} -> {RECHARGE_START_MJ + RECHARGE_GAINS[-1]:.1f} MJ) "
              f"costs {full['cost_vs_neutral_s'].iloc[0]:.2f} s vs an energy-neutral lap.")
    else:
        last = d[d["feasible"]]["gain_MJ"].max()
        print(f"  -> A full recharge is NOT possible in one lap here (harvest limit / layout); "
              f"max gain {last} MJ.")
    d.to_csv(OPT_DIR / f"{tag}_recharge_test.csv", index=False)

    ok = d[d["feasible"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ok["gain_MJ"], ok["cost_vs_neutral_s"], marker="o")
    ax.set_xlabel("Energy gained over the lap (MJ)")
    ax.set_ylabel("Lap time lost vs energy-neutral lap (s)")
    ax.set_title(f"{title_prefix} - cost of recharging the battery in one lap")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OPT_DIR / f"{tag}_recharge_test.png", dpi=150)
    plt.close(fig)
    return d


# ============================================================================= calibration
CAL_PRICES = [0.1, 0.25, 0.5, 1.0]      # faster price grid for calibration runs


def calibrate_drag(track, real_t, k_lo, k_hi):
    """Find the drag k at which the OPTIMISED simulated qualifying lap matches the real lap time.
    The real driver's lap is (close to) optimal for the real car, so this is the fair target.
    The independent test: does the calibrated drag fall inside Step 2c's measured drag range?"""
    def opt_time(k):
        _, _, r, _ = optimize(track, Car(k, QUALI_MASS), "quali", prices=CAL_PRICES)
        return r["time"]

    lo, hi = k_lo * 0.6, k_hi * 2.0
    t_lo, t_hi = opt_time(lo), opt_time(hi)
    if real_t <= t_lo:
        return lo, "simulator too SLOW even with very low drag"
    if real_t >= t_hi:
        return hi, "simulator too FAST even with very high drag -> physics missing (e.g. cornering losses)"
    for _ in range(7):
        mid = 0.5 * (lo + hi)
        if opt_time(mid) < real_t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), "ok"


def calibrate_joint(track, real_t, k_lo, k_hi):
    """Find drag k INSIDE the measured range at which the simulator - limited to the energy the
    telemetry says the real car deployed at that drag - reproduces the real lap time.
    Returns (k, energy_mj, status)."""
    global DEPLOY_LIMIT_MJ

    def gap(k):
        global DEPLOY_LIMIT_MJ
        e = track.measured_deploy_mj(k, QUALI_MASS)
        DEPLOY_LIMIT_MJ = e
        _, _, r, _ = optimize(track, Car(k, QUALI_MASS), "quali", prices=CAL_PRICES)
        return r["time"] - real_t, e

    g_lo, e_lo = gap(k_lo)
    g_hi, e_hi = gap(k_hi)
    print(f"  low drag  (CdA {2 * k_lo / s2.RHO:.2f}): real car deployed {e_lo:.1f} MJ -> simulated lap {g_lo:+.2f} s vs real")
    print(f"  high drag (CdA {2 * k_hi / s2.RHO:.2f}): real car deployed {e_hi:.1f} MJ -> simulated lap {g_hi:+.2f} s vs real")
    if g_lo * g_hi > 0:
        DEPLOY_LIMIT_MJ = float("inf")
        side = "FAST" if g_hi < 0 else "SLOW"
        return np.nan, np.nan, f"no consistent solution - simulator too {side} across the whole drag range"
    lo, hi = k_lo, k_hi
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        g, e = gap(mid)
        if (g < 0) == (g_lo < 0):
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    e = track.measured_deploy_mj(k, QUALI_MASS)
    DEPLOY_LIMIT_MJ = e
    return k, e, "consistent"


# ============================================================================= main
def run_case(track, car, lap_df, mode, tag, title_prefix, defend):
    global SOC_MAX_MJ, RACE_SOC0_MJ, RACE_DRAW_MJ, DEPLOY_LIMIT_MJ
    rows = []
    soc0 = SOC_MAX_MJ if mode == "quali" else RACE_SOC0_MJ

    # Validation: simulator with the naive strategy vs the real lap
    nl, nf = naive_levels(track)
    naive = simulate(track, car, nl, nf, soc0, HARVEST_CAP_MJ[mode], record=True,
                     deploy_floor=0.0 if mode == "quali" else soc0)   # race: never dip below the reserve
    real_t = float(lap_df.attrs.get("lap_time", lap_df["t"].iloc[-1] - lap_df["t"].iloc[0]))
    v_real = np.interp(track.s, lap_df["Distance"], lap_df["Speed"])
    rmse = float(np.sqrt(np.mean((naive["v"] * 3.6 - v_real) ** 2)))
    print(f"\n  Validation - real lap {real_t:.3f} s | simulated 'deploy until empty' {naive['time']:.3f} s "
          f"({(naive['time'] / real_t - 1) * 100:+.1f}%) | speed-trace RMSE {rmse:.1f} km/h")
    # Where does the gap come from? Time per metre, real vs simulated, split by section type
    dt_sim = DS / np.maximum(naive["v"], 1.0)
    dt_real = DS / np.maximum(v_real / 3.6, 1.0)
    gap_free = float((dt_sim - dt_real)[track.free].sum())
    gap_grip = float((dt_sim - dt_real)[~track.free].sum())
    print(f"  Gap by section: straights/flat-out {gap_free:+.2f} s | corners & braking {gap_grip:+.2f} s "
          f"| braking fit {track.brake_coef[0]:.0f} + {track.brake_coef[1]:.4f} v^2 m/s^2 (capped {MAX_BRAKE_DECEL:.0f})")
    print(f"  Energy (naive sim): deployed {naive['deployed_mj']:.2f} MJ, harvested {naive['harvested_mj']:.2f} MJ")
    if mode == "race":
        print("  (race mode: race fuel load + kept battery reserve vs a QUALIFYING lap - this gap is expected; "
              "judge the simulator by the calibration check above)")
    elif abs(naive["time"] / real_t - 1) > 0.03:
        print("  WARNING: simulator is more than 3% off the real lap - treat the strategy results with caution.")
    fig, (b1, b2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    b1.plot(track.s, v_real, color="grey", lw=1, label="real lap")
    b1.plot(track.s, naive["v"] * 3.6, color="tab:purple", lw=1, label="simulator, 'deploy until empty'")
    b1.fill_between(track.s, 0, 360, where=~track.free, color="grey", alpha=0.08, step="mid",
                    label="corners & braking (real speeds kept)")
    b1.set_ylabel("Speed (km/h)")
    b1.set_title(f"{title_prefix} - simulator validation ({mode})")
    b1.legend(loc="lower right", fontsize=8)
    b1.grid(alpha=0.3)
    b2.plot(track.s, np.cumsum(dt_sim - dt_real), color="black")
    b2.axhline(0, color="grey", lw=0.6)
    b2.set_ylabel("Cumulative time gap (s)\n(+ = simulator slower)")
    b2.set_xlabel("Distance (m)")
    b2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OPT_DIR / f"{tag}_{mode}_validation.png", dpi=130)
    plt.close(fig)
    naive_name = "naive: deploy max until empty" if mode == "quali" else "naive: deploy max until reserve hit"
    rows.append({"strategy": naive_name, **{k: naive[k] for k in
                 ("time", "deployed_mj", "harvested_mj", "soc_end")}})

    # Each style gets several starting points, so the comparison isn't decided by optimizer noise:
    # from scratch, from naive full deployment, and (2nd round) from the other strategies' best laps
    # with the disallowed harvesting switched off.
    styles = {"lift-and-coast only (Red Bull / McLaren style)": (True, False),
              "super-clip only (Ferrari style)": (False, True)}
    better = lambda a, b: a if (a[2].get("infeasible", False), a[2]["time"]) <= \
        (b[2].get("infeasible", False), b[2]["time"]) else b
    t0 = time.time()
    results = {}
    for name, (al, asc) in styles.items():
        res = optimize(track, car, mode, allow_lift=al, allow_sc=asc)
        res = better(res, optimize(track, car, mode, allow_lift=al, allow_sc=asc, init=naive_levels(track)))
        results[name] = res
    cands = [optimize(track, car, mode), optimize(track, car, mode, init=naive_levels(track))] + \
            [optimize(track, car, mode, init=(lv, lf, lam)) for lv, lf, _, lam in results.values()]
    opt = min(cands, key=lambda x: (x[2].get("infeasible", False), x[2]["time"]))
    # 2nd round: restart each style from the free optimum (projected onto what that style allows)
    for name, (al, asc) in styles.items():
        results[name] = better(results[name], optimize(track, car, mode, allow_lift=al, allow_sc=asc,
                                                       init=(opt[0], opt[1], opt[3])))
        opt = better(opt, optimize(track, car, mode, init=(results[name][0], results[name][1], results[name][3])))
    for name, res in results.items():
        print(f"  {name:50s} {res[2]['time']:.3f} s", flush=True)
    print(f"  {'optimal (lift-and-coast + super-clip allowed)':50s} {opt[2]['time']:.3f} s   "
          f"(all styles: {time.time() - t0:.0f} s)", flush=True)
    for name, (lv, lf, r, lam) in [("optimal (both allowed)", opt)] + list(results.items()):
        rows.append({"strategy": name, **{k: r[k] for k in ("time", "deployed_mj", "harvested_mj", "soc_end")},
                     "lift_m_total": float(lf.sum()), "superclip_chunks": int((lv < 0).sum()),
                     "energy_price_s_per_MJ": lam})
    best_result = (opt[0], opt[1])

    lv, lf = best_result
    rec = simulate(track, car, lv, lf, soc0, HARVEST_CAP_MJ[mode], record=True)
    plot_strategy(track, lap_df, rec, f"{title_prefix} - optimal energy strategy ({mode})",
                  OPT_DIR / f"{tag}_{mode}_strategy.png")

    # Value of energy: how much would 1 MJ MORE usable energy be worth on this lap?
    saved = (SOC_MAX_MJ, RACE_DRAW_MJ, DEPLOY_LIMIT_MJ)
    DEPLOY_LIMIT_MJ += 1.0                # the per-lap deployment limit rises with the extra MJ
    if mode == "quali":
        SOC_MAX_MJ += 1.0                 # start the lap with 1 MJ more
    else:
        RACE_DRAW_MJ = 1.0                # allowed to use 1 MJ of the battery reserve this lap
    _, _, r_plus, _ = optimize(track, car, mode, init=best_result)
    SOC_MAX_MJ, RACE_DRAW_MJ, DEPLOY_LIMIT_MJ = saved
    value = rec["time"] - r_plus["time"]
    what = "starting the lap with 1 MJ more" if mode == "quali" else "using 1 MJ of the battery reserve"
    print(f"\n  Value of energy: {what} would save {value:.3f} s per lap "
          f"(optimizer's energy price: {opt[3]} s/MJ)")

    summary = pd.DataFrame(rows)
    summary["gap_to_optimal_s"] = summary["time"] - summary.loc[1, "time"]
    summary["value_of_1MJ_s"] = value
    print("\n" + summary.round(3).to_string(index=False))
    summary.to_csv(OPT_DIR / f"{tag}_{mode}_strategies.csv", index=False)

    if defend and mode == "race":
        print("\n  Defence: full deployment on a long straight, energy-neutral lap.")
        print("  If the whole straight is impossible, the longest defendable part is found.")
        base_t = summary.loc[1, "time"]
        drows = []
        for start, end in track.longest_runs[:3]:
            length = (end - start) * DS
            row = {"straight_from_m": round(track.s[start]), "to_m": round(track.s[end - 1]),
                   "length_m": round(length), "defendable_m": 0, "lap_time_cost_s": None}
            L = length
            while L >= CHUNK_M * 0.99:
                stop = start + int(L / DS)
                chunks = sorted(set(track.chunk[start:stop]) - {-1})
                fixed = {c: float(s2.MGU_K_MAX_KW) for c in chunks}
                _, _, r, _ = optimize(track, car, mode, fixed=fixed, init=best_result)
                if not r.get("infeasible"):
                    row.update(defendable_m=round(L), lap_time_cost_s=round(max(r["time"] - base_t, 0.0), 3))
                    if r["time"] < base_t - 0.01:
                        print(f"    (note: this constrained run beat the optimum by {base_t - r['time']:.3f} s "
                              f"-> the optimizer missed a better solution; cost shown as 0)")
                    break
                L -= CHUNK_M
            drows.append(row)
            msg = (f"{row['defendable_m']} of {row['length_m']} m, costs {row['lap_time_cost_s']} s/lap"
                   if row["defendable_m"] else "not possible within the battery limits")
            print(f"    straight {row['straight_from_m']}-{row['to_m']} m: {msg}", flush=True)
        d = pd.DataFrame(drows)
        d.to_csv(OPT_DIR / f"{tag}_defence.csv", index=False)

    return summary


def main():
    global DEPLOY_LIMIT_MJ
    parser = argparse.ArgumentParser(description="Optimise energy deployment on a real lap.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--event", default="Monza")
    parser.add_argument("--session", default="Q")
    parser.add_argument("--driver", default="LEC")
    parser.add_argument("--mode", choices=["quali", "race"], default="quali")
    parser.add_argument("--drag", choices=["lo", "mid", "hi"], default="mid")
    parser.add_argument("--robustness", action="store_true", help="repeat at both drag bounds")
    parser.add_argument("--defend", action="store_true", help="race mode: cost of defending each straight")
    parser.add_argument("--allow-tow", action="store_true",
                        help="use the fastest lap even if it was towed (default: fastest clean-air lap)")
    parser.add_argument("--recharge-test", action="store_true",
                        help="race mode: lap-time cost of finishing the lap with more energy (up to a full recharge)")
    parser.add_argument("--joint", action="store_true",
                        help="try joint drag + energy calibration first (see README: often degenerate)")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="skip calibrating drag to the real lap (use the Step 2c drag range directly)")
    args = parser.parse_args()

    OPT_DIR.mkdir(parents=True, exist_ok=True)
    session = load_session(args.year, args.event, args.session)
    lap, tel, note = clean_air_lap(session, args.driver, args.allow_tow)
    lap_no = lap.get("LapNumber") if hasattr(lap, "get") else None
    lap_no = int(lap_no) if lap_no is not None and pd.notna(lap_no) else "?"
    print(f"Reference lap {lap_no} ({lap['LapTime'].total_seconds():.3f} s): {note}")
    lap_df = label_energy_modes(tel)
    lap_df.attrs["lap_time"] = lap["LapTime"].total_seconds()
    geo = lap_geometry(lap, lap_df)
    track = Track(lap_df, geo)
    if track.has_geo:
        climb = float(np.sum(np.clip(np.diff(np.interp(track.s, geo["Distance"], geo["Z"])), 0, None)))
        print(f"Track geometry from position data: total climb {climb:.0f} m, "
              f"cornering grip {track.grip_coef[0]:.0f} + {track.grip_coef[1]:.4f} v^2 m/s^2, "
              f"{int((track.free & (track.v_grip < 400 / 3.6)).sum() * DS)} m of flat-out sections are grip-limited, "
              f"{int(track.flat_corner.sum() * DS)} m are corners (250 kW deployment limit)")
    else:
        print("WARNING: no position data - simulating without gradient and cornering effects")
    name = session.event["EventName"]
    print(f"{name} {args.year} - {args.driver} - {track.n_chunks} straight chunks, "
          f"{len(track.brake_starts)} braking zones")

    mass = QUALI_MASS if args.mode == "quali" else RACE_MASS
    real_t = lap["LapTime"].total_seconds()            # official lap time
    if args.robustness or args.no_calibrate:
        drags = ["lo", "hi"] if args.robustness else [args.drag]
    else:
        drags = ["calibrated"]
    for which in drags:
        if which == "calibrated":
            _, (lo2c, hi2c) = drag_for(session, args.driver, lap_df, "mid")
            lo, hi = lo2c, hi2c
            if track.has_geo:
                glo, ghi = track.drag_bounds(QUALI_MASS)
                if pd.notna(ghi):
                    lo, hi = (glo if pd.notna(glo) else ghi), ghi
                    print(f"\nDrag range on this lap with the simulator's physics (gradient, corner scrub, "
                          f"250 kW zones): CdA {2 * lo / s2.RHO:.2f}-{2 * hi / s2.RHO:.2f} "
                          f"(Step 2c flat-track range: {2 * lo2c / s2.RHO:.2f}-{2 * hi2c / s2.RHO:.2f})")
            jstatus = "skipped"
            if args.joint:
                print(f"\nJoint calibration: is there a drag INSIDE the measured range where the simulator, limited "
                      f"to the energy the real car deployed, matches the real {real_t:.3f} s? (a few minutes)", flush=True)
                t0 = time.time()
                kj, ej, jstatus = calibrate_joint(track, real_t, lo, hi)
                if jstatus == "consistent":
                    # Degeneracy check: a qualifying lap that leaves battery energy unused is not credible
                    _, _, rq, _ = optimize(track, Car(kj, QUALI_MASS), "quali", prices=CAL_PRICES)
                    if rq["soc_end"] > 1.0:
                        print(f"  WARNING: degenerate solution - the simulated qualifying lap ends with "
                              f"{rq['soc_end']:.1f} MJ unused in the battery. Real cars don't leave energy on a quali "
                              f"lap, so the per-lap deployment limit is binding unrealistically. Drag and deployment "
                              f"cannot be separated from speed telemetry alone (identifiability). "
                              f"Falling back to drag-only calibration.")
                        jstatus = "degenerate"
            if jstatus == "consistent":
                print(f"  YES -> CdA {2 * kj / s2.RHO:.2f} m^2 with {ej:.1f} MJ deployed per lap "
                      f"({time.time() - t0:.0f} s). Drag AND energy are consistent with the telemetry: simulator validated.")
                print(f"  All strategies below are limited to {ej:.1f} MJ of deployment per lap.")
                k = kj
                car = Car(k, mass)
                print(f"\n=== Drag: jointly calibrated (CdA {2 * k / s2.RHO:.2f} m^2; range "
                      f"{2 * lo / s2.RHO:.2f}-{2 * hi / s2.RHO:.2f}; {ej:.1f} MJ/lap) ===")
                tag = f"{args.year}_{args.event}_{args.driver}_calibrated".replace(" ", "")
                run_case(track, car, lap_df, args.mode, tag, f"{name} {args.driver}", args.defend)
                if args.recharge_test and args.mode == "race":
                    recharge_test(track, car, tag, f"{name} {args.driver}")
                continue
            if jstatus not in ("skipped", "degenerate"):
                print(f"  NO: {jstatus}. Falling back to drag-only calibration (no energy limit).")
            DEPLOY_LIMIT_MJ = float("inf")
            print(f"\nCalibrating drag so the optimised simulated lap matches the real {real_t:.3f} s "
                  f"(takes a minute or two)...", flush=True)
            t0 = time.time()
            k, status = calibrate_drag(track, real_t, lo, hi)
            inside = lo * 0.95 <= k <= hi * 1.05
            off = (k / hi - 1) * 100 if k > hi else (k / lo - 1) * 100 if k < lo else 0.0
            print(f"  Calibrated CdA {2 * k / s2.RHO:.2f} m^2 ({status}; {time.time() - t0:.0f} s)")
            print(f"  Independent check vs measured drag range {2 * lo / s2.RHO:.2f}-{2 * hi / s2.RHO:.2f}: "
                  + ("INSIDE the range -> simulator and drag measurement agree."
                     if inside else f"OUTSIDE by {off:+.0f}% -> something is still missing from the model."))
        else:
            k, (lo, hi) = drag_for(session, args.driver, lap_df, which)
        car = Car(k, mass)
        print(f"\n=== Drag: {which} (CdA {2 * k / s2.RHO:.2f} m^2; range {2 * lo / s2.RHO:.2f}-{2 * hi / s2.RHO:.2f}) ===")
        tag = f"{args.year}_{args.event}_{args.driver}_{which}".replace(" ", "")
        run_case(track, car, lap_df, args.mode, tag, f"{name} {args.driver}", args.defend)
        if args.recharge_test:
            if args.mode == "race":
                recharge_test(track, car, tag, f"{name} {args.driver}")
            else:
                print("  (--recharge-test needs --mode race)")
    print(f"\nSaved plots + tables to: {OPT_DIR}")


if __name__ == "__main__":
    main()
