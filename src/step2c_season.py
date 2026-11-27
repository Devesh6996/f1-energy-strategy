"""
F1 Energy Strategy - Step 2c: The whole 2026 season + tyre degradation
=======================================================================
For every completed 2026 race weekend:
  1. QUALIFYING -> fit each car's drag as a RANGE (bounds), not one assumed value:
        lower bound: car is using engine power only (it always has at least that at full throttle)
        upper bound: car is deploying the maximum the rules allow
     Setups are locked after qualifying (parc fermé), so the same drag applies in the race.
  2. RACE -> analyse every green-flag race lap with that drag range, the race fuel load
     (70 kg at the start, burning down), and the tyre's age and compound.
Then it answers two questions across the season:
  A. Which energy habits are stable team traits, and which depend on the track?
  B. How does tyre age affect clipping / super-clipping / lift-and-coast, once fuel burn is
     separated out? (Fuel and tyre age both rise lap by lap; they can be separated because
     tyre age RESETS at a pit stop while fuel keeps falling.)

Run (resumable - events already done are skipped):
    python src/step2c_season.py --cache-dir C:\\f1cache          # all completed 2026 events
    python src/step2c_season.py --rounds 14 16 --cache-dir C:\\f1cache   # just some rounds
    python src/step2c_season.py --analyse-only                   # re-run the analysis on saved data

Outputs (outputs/season/):
    <round>_drag.csv, <round>_Q_laps.csv, <round>_R_laps.csv   per event
    season_team_summary.csv, season_tyre_effects.csv, tyre_effect_<metric>.png
"""

import argparse
import os
import warnings
from pathlib import Path

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import step2_power_model as s2
from step1_energy_fingerprint import CACHE_DIR, OUT_DIR, find_clip_zones, label_energy_modes

SEASON_DIR = OUT_DIR / "season"

BASE_MASS_KG = 768        # 2026 minimum mass incl. driver, without fuel
QUALI_FUEL_KG = 10        # rough qualifying fuel load
FUEL_START_KG = 70        # 2026 race fuel allowance
FUEL_END_KG = 2           # what's left at the flag (rough)
FIT_SPEED_FLOOR = 250     # km/h: drag is fitted on the fastest full-throttle samples, never below this
FIT_SPEED_QUANTILE = 0.85 #   ... specifically the top 15% of full-throttle speeds at that track
QUICK = 1.07              # keep laps within 107% of the session's fastest
R_DRY_AIR = 287.05
SUPER_CLIP = "super clipping (harvesting at full throttle)"
METRICS = ["clipping_%", "lift_coast_%", "super_clip_robust", "electric_mj_mid"]


# ----------------------------------------------------------------------------- loading
def setup_cache(cache_dir):
    path = Path(cache_dir or os.environ.get("F1_CACHE_DIR") or CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    try:
        fastf1.Cache.configure(cache_dir=str(path))
    except (AttributeError, TypeError):
        fastf1.Cache.enable_cache(str(path))
    if "onedrive" in str(path).lower():
        print("TIP: your cache is inside OneDrive. A full season is several GB - use "
              "--cache-dir C:\\f1cache to keep it out of OneDrive.")
    return path


def completed_rounds(year):
    sched = fastf1.get_event_schedule(year, include_testing=False)
    sched = sched[sched["RoundNumber"] > 0]
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    col = "Session5DateUtc" if "Session5DateUtc" in sched else "EventDate"
    dates = pd.to_datetime(sched[col], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    done = sched[dates < now - pd.Timedelta(hours=6)]
    return [(int(r.RoundNumber), str(r.EventName)) for r in done.itertuples()]


def load(year, rnd, kind):
    session = fastf1.get_session(year, rnd, kind)
    session.load(laps=True, telemetry=True, weather=True, messages=False)
    return session


def air_density(session):
    try:
        w = session.weather_data
        rho = float(w["Pressure"].median()) * 100 / (R_DRY_AIR * (float(w["AirTemp"].median()) + 273.15))
        if 0.8 < rho < 1.35:
            return rho
    except Exception:
        pass
    return s2.RHO


def lap_temperatures(session):
    """Track and air temperature at the start of every lap (interpolated from the weather feed)."""
    w = session.weather_data
    wt = w["Time"].dt.total_seconds().to_numpy()
    laps = session.laps
    start = laps["LapStartTime"].dt.total_seconds().to_numpy()
    return pd.DataFrame({
        "driver": laps["Driver"].to_numpy(),
        "lap_number": laps["LapNumber"].to_numpy(),
        "track_temp": np.round(np.interp(start, wt, w["TrackTemp"].to_numpy(dtype=float)), 1),
        "air_temp": np.round(np.interp(start, wt, w["AirTemp"].to_numpy(dtype=float)), 1),
    }).dropna(subset=["lap_number"]).astype({"lap_number": int})


def add_weather(year, rounds=None):
    """Add track/air temperature to race-lap files already saved - loads only laps + weather (fast)."""
    for f in sorted(SEASON_DIR.glob("[0-9][0-9]_R_laps.csv")):
        rnd = int(f.name[:2])
        if rounds and rnd not in rounds:
            continue
        d = pd.read_csv(f)
        if "track_temp" in d and d["track_temp"].notna().mean() > 0.5:
            print(f"Round {rnd:2d}: temperatures already added")
            continue
        try:
            session = fastf1.get_session(year, rnd, "R")
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            temps = lap_temperatures(session)
        except Exception as err:
            print(f"Round {rnd:2d}: could not add temperatures ({err})")
            continue
        d = d.drop(columns=[c for c in ("track_temp", "air_temp") if c in d]).merge(
            temps, on=["driver", "lap_number"], how="left")
        d.to_csv(f, index=False)
        print(f"Round {rnd:2d}: track temp {d['track_temp'].min():.0f}-{d['track_temp'].max():.0f} C added")


# ----------------------------------------------------------------------------- physics
def drag_bounds(dfs, mass):
    """Lower/upper bound on the drag factor k from a driver's qualifying laps."""
    parts = []
    for df in dfs:
        v = df["Speed"] / 3.6
        full = df["mode"].eq("full_deploy").astype(int)
        stable = full.rolling(2 * s2.EDGE_TRIM + 1, center=True, min_periods=1).min().eq(1)
        parts.append(pd.DataFrame({
            "speed": df.loc[stable, "Speed"],
            "x": v[stable] ** 3,
            "y": mass * df.loc[stable, "accel"] * v[stable] + s2.CRR * mass * s2.G * v[stable],
        }))
    s = pd.concat(parts) if parts else pd.DataFrame()
    if len(s) < 10:
        return np.nan, np.nan, np.nan
    fit_min = max(FIT_SPEED_FLOOR, float(s["speed"].quantile(FIT_SPEED_QUANTILE)))
    top = s[s["speed"] >= fit_min]
    if len(top) < 8:
        return np.nan, np.nan, fit_min
    ice_w = s2.ICE_NOMINAL_KW * 1000 * s2.DRIVETRAIN_EFF
    k_hi = float(np.median((s2.allowed_wheel_power_w(top["speed"]) - top["y"]) / top["x"]))
    k_lo = float(np.median((ice_w - top["y"]) / top["x"]))
    return (k_lo if k_lo > 0 else np.nan), (k_hi if k_hi > 0 else np.nan), fit_min


def lap_metrics(df, k_lo, k_hi, mass):
    """Energy metrics for one lap, evaluated at both drag bounds."""
    s2.MASS_KG = mass
    seg = df["Distance"].diff().fillna(0)
    share = seg.groupby(df["mode"]).sum() / seg.sum() * 100
    out = {"clipping_%": round(share.get("clipping", 0.0), 2),
           "lift_coast_%": round(share.get("lift_coast", 0.0), 2),
           "full_deploy_%": round(share.get("full_deploy", 0.0), 2)}
    p_max = (s2.ICE_NOMINAL_KW + s2.MGU_K_MAX_KW) * 1000 * s2.DRIVETRAIN_EFF
    for tag, k in (("hi", k_hi), ("lo", k_lo)):
        if pd.isna(k):
            out[f"electric_mj_{tag}"], out[f"super_clip_{tag}"] = np.nan, np.nan
            continue
        model = {"p_max_w": p_max, "k": k, "cda_m2": np.nan, "mode": "topspeed"}
        d = s2.add_power_columns(df, model)
        zones = s2.classify_zones(d, find_clip_zones(d))
        out[f"electric_mj_{tag}"] = round(s2.electric_energy_mj(d, model), 2)
        out[f"super_clip_{tag}"] = int(zones["reason"].eq(SUPER_CLIP).sum()) if not zones.empty else 0
    # Upper-bound drag gives the HIGHEST power estimate, so a zone still below the
    # engine-only level there is super clipping under any drag in the range -> "robust".
    out["super_clip_robust"] = out["super_clip_hi"]
    lo, hi = out["electric_mj_lo"], out["electric_mj_hi"]
    out["electric_mj_mid"] = round(np.nanmean([lo, hi]), 2) if not (pd.isna(lo) and pd.isna(hi)) else np.nan
    return out


# ----------------------------------------------------------------------------- per event
def process_event(year, rnd, name, refresh):
    f_drag, f_q, f_r = (SEASON_DIR / f"{rnd:02d}_{t}.csv" for t in ("drag", "Q_laps", "R_laps"))
    if not refresh and f_drag.exists() and f_q.exists() and f_r.exists():
        print(f"Round {rnd:2d} {name}: already done (use --refresh to redo)")
        return

    print(f"\nRound {rnd:2d} {name}: qualifying...", flush=True)
    q = load(year, rnd, "Q")
    rho = air_density(q)
    q_mass = BASE_MASS_KG + QUALI_FUEL_KG
    quick = q.laps.pick_quicklaps(QUICK)

    drag_rows, q_rows, labelled_q = [], [], {}
    for driver in sorted(quick["Driver"].unique()):
        dfs = []
        for _, lap in quick.pick_drivers(driver).iterlaps():
            try:
                dfs.append((lap, label_energy_modes(lap.get_car_data().add_distance())))
            except Exception:
                continue
        k_lo, k_hi, fit_min = drag_bounds([d for _, d in dfs], q_mass)
        team = dfs[0][0].get("Team", "") if dfs else ""
        drag_rows.append({"round": rnd, "event": name, "driver": driver, "team": team,
                          "rho": round(rho, 3), "fit_min_kmh": round(fit_min) if pd.notna(fit_min) else None,
                          "k_lo": k_lo, "k_hi": k_hi,
                          "cda_lo": round(2 * k_lo / rho, 2) if pd.notna(k_lo) else None,
                          "cda_hi": round(2 * k_hi / rho, 2) if pd.notna(k_hi) else None})
        labelled_q[driver] = (k_lo, k_hi)
        for lap, df in dfs:
            q_rows.append({"round": rnd, "event": name, "driver": driver, "team": team,
                           "lap_number": int(lap["LapNumber"]),
                           "lap_time_s": lap["LapTime"].total_seconds(),
                           **lap_metrics(df, k_lo, k_hi, q_mass)})
    pd.DataFrame(drag_rows).to_csv(f_drag, index=False)
    pd.DataFrame(q_rows).to_csv(f_q, index=False)

    print(f"Round {rnd:2d} {name}: race...", flush=True)
    r = load(year, rnd, "R")
    total = int(r.laps["LapNumber"].max())
    laps = r.laps.pick_wo_box().pick_track_status("1").pick_quicklaps(QUICK)
    r_rows = []
    for _, lap in laps.iterlaps():
        if lap["LapNumber"] <= 1 or pd.isna(lap["TyreLife"]):   # skip the start lap / unknown tyre age
            continue
        driver = lap["Driver"]
        k_lo, k_hi = labelled_q.get(driver, (np.nan, np.nan))
        if pd.isna(k_hi):
            continue
        fuel = FUEL_START_KG - (FUEL_START_KG - FUEL_END_KG) * (lap["LapNumber"] - 1) / max(total - 1, 1)
        try:
            df = label_energy_modes(lap.get_car_data().add_distance())
        except Exception:
            continue
        r_rows.append({"round": rnd, "event": name, "driver": driver, "team": lap.get("Team", ""),
                       "lap_number": int(lap["LapNumber"]), "lap_time_s": lap["LapTime"].total_seconds(),
                       "stint": lap.get("Stint"), "compound": str(lap.get("Compound", "")).upper(),
                       "tyre_life": float(lap["TyreLife"]), "fuel_kg": round(fuel, 1),
                       **lap_metrics(df, k_lo, k_hi, BASE_MASS_KG + fuel)})
    r_df = pd.DataFrame(r_rows)
    try:
        r_df = r_df.merge(lap_temperatures(r), on=["driver", "lap_number"], how="left")
    except Exception:
        pass
    r_df.to_csv(f_r, index=False)
    print(f"Round {rnd:2d} {name}: {len(q_rows)} quali laps, {len(r_rows)} race laps saved.")


# ----------------------------------------------------------------------------- analysis A
def analyse_teams(drag, q):
    print("\n" + "=" * 90 + "\nA. TEAM ENERGY HABITS ACROSS THE SEASON (qualifying)\n" + "=" * 90)
    team = (q.groupby(["round", "event", "team"])[["clipping_%", "lift_coast_%", "super_clip_robust",
                                                   "electric_mj_mid"]].mean()
            .join(drag.groupby(["round", "event", "team"])[["cda_lo", "cda_hi"]].mean()))
    team = team.round(2).reset_index()
    team.to_csv(SEASON_DIR / "season_team_summary.csv", index=False)

    print("\nPer track (team averages):")
    track = (team.groupby(["round", "event"])[["clipping_%", "super_clip_robust", "lift_coast_%",
                                                "cda_lo", "cda_hi"]].mean().round(2))
    print(track.to_string())

    print("\nHow stable is each habit across tracks? (mean Spearman rank agreement of teams, track vs track)")
    for m in ["lift_coast_%", "super_clip_robust", "clipping_%", "cda_hi"]:
        wide = team.pivot_table(index="team", columns="round", values=m)
        corr = wide.corr(method="spearman").to_numpy()
        vals = corr[np.triu_indices_from(corr, k=1)]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            print(f"  {m:20s} {np.mean(vals):+.2f}  ({len(vals)} track pairs)")
    print("  ~+1 = same team order everywhere (a team trait); ~0 = order changes with the track.")

    print("\nSeason ranking (average rank across tracks, 1 = most):")
    ranks = team.copy()
    for m in ["lift_coast_%", "super_clip_robust"]:
        ranks[m + "_rank"] = ranks.groupby("round")[m].rank(ascending=False)
    print(ranks.groupby("team")[["lift_coast_%_rank", "super_clip_robust_rank"]].mean()
          .round(1).sort_values("lift_coast_%_rank").to_string())


# ----------------------------------------------------------------------------- analysis B
def within(df, cols, group):
    return df[cols] - df.groupby(group)[cols].transform("mean")


def ols_cluster(y, X, clusters):
    """OLS with cluster-robust standard errors (laps of the same driver-race are not independent)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    groups = pd.Series(clusters).astype(str).to_numpy()
    for g in np.unique(groups):
        idx = groups == g
        s = X[idx].T @ u[idx]
        meat += np.outer(s, s)
    n, k, G = len(y), X.shape[1], len(np.unique(groups))
    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    se = np.sqrt(np.diag(corr * xtx_inv @ meat @ xtx_inv))
    return beta, se


def analyse_tyres(r):
    print("\n" + "=" * 90 + "\nB. HOW TYRE AGE AFFECTS ENERGY USE (race laps, fuel burn separated out)\n" + "=" * 90)
    r = r[r["compound"].isin(["SOFT", "MEDIUM", "HARD"])].dropna(subset=["tyre_life", "fuel_kg"]).copy()
    if len(r) < 200:
        print("Not enough race laps yet.")
        return
    r["group"] = r["round"].astype(str) + "_" + r["driver"]
    compounds = [c for c in ["SOFT", "MEDIUM", "HARD"] if (r["compound"] == c).sum() > 50]
    for c in compounds:
        r[f"age_{c}"] = np.where(r["compound"] == c, r["tyre_life"], 0.0)
        r[f"is_{c}"] = (r["compound"] == c).astype(float)
    regressors = [f"age_{c}" for c in compounds] + ["fuel_kg"] + [f"is_{c}" for c in compounds[1:]]
    has_temp = "track_temp" in r and r["track_temp"].notna().mean() > 0.8
    if has_temp:
        r = r.dropna(subset=["track_temp"]).copy()
        regressors.append("track_temp")          # the track cools/heats during a race -> control for it

    rows = []
    for metric in METRICS + ["lap_time_s"]:
        d = r.dropna(subset=[metric])
        Xw = within(d, regressors, "group")
        yw = within(d, [metric], "group")[metric]
        beta, se = ols_cluster(yw, Xw, d["group"])
        for name, b, e in zip(regressors, beta, se):
            if name.startswith("age_") or name == "fuel_kg":
                rows.append({"metric": metric, "effect_of": name.replace("age_", "tyre age (") + (")" if name.startswith("age_") else ""),
                             "per_unit": "per lap of tyre age" if name.startswith("age_") else "per kg of fuel",
                             "coef": b, "se": e, "t": b / e if e > 0 else np.nan, "laps": len(d)})
    eff = pd.DataFrame(rows)
    eff.to_csv(SEASON_DIR / "season_tyre_effects.csv", index=False)

    pd.set_option("display.width", 200)
    show = eff.copy()
    show["significant"] = np.where(show["t"].abs() > 2, "yes", "no")
    show["over 20 laps"] = np.where(show["per_unit"].str.startswith("per lap"), show["coef"] * 20, np.nan)
    print(show[["metric", "effect_of", "coef", "se", "t", "significant", "over 20 laps", "laps"]]
          .round(3).to_string(index=False))
    print("\ncoef = change in the metric per lap of tyre age (or per kg of fuel), within the same driver")
    print("and race. |t| > 2 is roughly 'statistically clear'. 'over 20 laps' = effect of a 20-lap-old tyre.")

    if has_temp:
        analyse_heat(r, compounds)

    # Per-track tyre effect on clipping (is it consistent?)
    print("\nTyre-age effect on clipping %, per track (all compounds pooled):")
    per = []
    for (rnd, ev), d in r.groupby(["round", "event"]):
        if len(d) < 150 or d["stint"].nunique() < 2:
            continue
        Xw = within(d, ["tyre_life", "fuel_kg"], "group")
        yw = within(d, ["clipping_%"], "group")["clipping_%"]
        b, e = ols_cluster(yw, Xw, d["group"])
        per.append({"round": rnd, "event": ev, "clip_%_per_tyre_lap": b[0], "se": e[0], "laps": len(d)})
    if per:
        print(pd.DataFrame(per).round(3).to_string(index=False))

    # Plots: metric vs tyre age, after removing driver-race averages and the fuel effect
    for metric in ["clipping_%", "super_clip_robust", "lift_coast_%"]:
        d = r.dropna(subset=[metric]).copy()
        fuel_coef = eff[(eff.metric == metric) & (eff.effect_of == "fuel_kg")]["coef"]
        fuel_coef = float(fuel_coef.iloc[0]) if len(fuel_coef) else 0.0
        d["adj"] = within(d, [metric], "group")[metric] - fuel_coef * within(d, ["fuel_kg"], "group")["fuel_kg"]
        fig, ax = plt.subplots(figsize=(9, 5))
        colours = {"SOFT": "tab:red", "MEDIUM": "goldenrod", "HARD": "grey"}
        for c in compounds:
            dc = d[d["compound"] == c]
            b = dc.groupby(dc["tyre_life"].clip(upper=40).round())["adj"].agg(["mean", "size"])
            b = b[b["size"] >= min(20, max(3, len(dc) // 60))]   # enough laps per point
            ax.plot(b.index, b["mean"], marker="o", ms=3, color=colours[c], label=c.title())
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("Tyre age (laps)")
        ax.set_ylabel(f"{metric} vs driver's race average\n(fuel effect removed)")
        ax.set_title(f"Tyre age vs {metric} - all 2026 races analysed")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(SEASON_DIR / f"tyre_effect_{metric.replace('%', 'pct')}.png", dpi=150)
        plt.close(fig)
    print(f"\nPlots saved: {SEASON_DIR / 'tyre_effect_*.png'}")


def analyse_heat(r, compounds):
    """Do tyres degrade faster - and does clipping respond differently - on hot tracks?
    1) split races into cooler / hotter halves by race-average track temperature and compare the
       tyre-age effects; 2) a formal test: tyre age x track temperature interaction."""
    print("\n" + "-" * 90 + "\nTRACK TEMPERATURE: do hot tracks change the tyre effects?\n" + "-" * 90)
    race_t = r.groupby("round")["track_temp"].mean()
    cut = float(race_t.median())
    r = r.copy()
    r["heat"] = np.where(r["round"].map(race_t) > cut, "hot", "cool")
    print(f"Race-average track temperatures: {race_t.min():.0f}-{race_t.max():.0f} C; split at {cut:.0f} C "
          f"({(race_t <= cut).sum()} cooler races, {(race_t > cut).sum()} hotter races)")

    rows = []
    for metric in ["lap_time_s", "clipping_%", "super_clip_robust", "lift_coast_%"]:
        for heat in ("cool", "hot"):
            d = r[(r["heat"] == heat)].dropna(subset=[metric])
            comps = [c for c in compounds if (d["compound"] == c).sum() > 50]
            regs = [f"age_{c}" for c in comps] + ["fuel_kg", "track_temp"] + [f"is_{c}" for c in comps[1:]]
            beta, se = ols_cluster(within(d, [metric], "group")[metric], within(d, regs, "group"), d["group"])
            for name, b, e in zip(regs, beta, se):
                if name.startswith("age_"):
                    rows.append({"metric": metric, "compound": name[4:], "races": heat, "per_tyre_lap": b, "se": e})
        # formal interaction: does the per-lap tyre effect grow with track temperature?
        d = r.dropna(subset=[metric]).copy()
        centre = float(d["track_temp"].mean())
        d["age_x_temp"] = d["tyre_life"] * (d["track_temp"] - centre)
        regs = [f"age_{c}" for c in compounds] + ["age_x_temp", "fuel_kg", "track_temp"] + \
               [f"is_{c}" for c in compounds[1:]]
        beta, se = ols_cluster(within(d, [metric], "group")[metric], within(d, regs, "group"), d["group"])
        i = regs.index("age_x_temp")
        rows.append({"metric": metric, "compound": "ALL", "races": "interaction per +10 C",
                     "per_tyre_lap": beta[i] * 10, "se": se[i] * 10})
    t = pd.DataFrame(rows)
    t["t"] = t["per_tyre_lap"] / t["se"]
    t.to_csv(SEASON_DIR / "season_tyre_heat_effects.csv", index=False)

    pd.set_option("display.width", 200)
    wide = t[t["races"].isin(["cool", "hot"])].pivot_table(index=["metric", "compound"], columns="races",
                                                            values="per_tyre_lap")
    inter = t[t["compound"] == "ALL"].set_index("metric")
    wide = wide.join(inter[["per_tyre_lap", "t"]].rename(
        columns={"per_tyre_lap": "extra_per_lap_per_+10C", "t": "t_interaction"}), on="metric")
    print(wide.round(4).to_string())
    print("\ncool/hot = change in the metric per lap of tyre age, in cooler vs hotter races.")
    print("extra_per_lap_per_+10C = how much the per-lap tyre effect grows for every 10 C hotter track")
    print("(all compounds pooled); |t_interaction| > 2 = clearly real.")


# ----------------------------------------------------------------------------- main
def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Season-wide energy analysis + tyre degradation.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--rounds", nargs="+", type=int, default=None, help="default: all completed rounds")
    parser.add_argument("--cache-dir", default=None, help="where FastF1 stores downloads (keep out of OneDrive)")
    parser.add_argument("--refresh", action="store_true", help="re-process events already done")
    parser.add_argument("--analyse-only", action="store_true", help="skip downloading, just analyse saved data")
    parser.add_argument("--exclude-rounds", nargs="+", type=int, default=None,
                        help="leave these rounds out of the analysis (e.g. to test whether one track drives a result)")
    parser.add_argument("--add-weather", action="store_true",
                        help="add track/air temperature to already-saved race laps (fast), then analyse")
    args = parser.parse_args()

    SEASON_DIR.mkdir(parents=True, exist_ok=True)
    if not args.analyse_only:
        setup_cache(args.cache_dir)
        events = completed_rounds(args.year)
        if args.rounds:
            done = {e[0] for e in events}
            for rnd in args.rounds:
                if rnd not in done:
                    print(f"Round {rnd}: not finished yet (or finished less than 6 hours ago) - nothing to download.")
            events = [e for e in events if e[0] in args.rounds]
        print(f"{len(events)} completed events to process.")
        for rnd, name in events:
            try:
                process_event(args.year, rnd, name, args.refresh)
            except Exception as err:
                print(f"Round {rnd} {name}: FAILED ({type(err).__name__}: {err}) - skipping, will retry next run")
        if args.add_weather:      # after downloading, so new rounds get their temperatures too
            add_weather(args.year, args.rounds)

    files = lambda t: sorted(SEASON_DIR.glob(f"[0-9][0-9]_{t}.csv"))
    read = lambda t: pd.concat([pd.read_csv(f) for f in files(t)], ignore_index=True) if files(t) else pd.DataFrame()
    drag, q, r = read("drag"), read("Q_laps"), read("R_laps")
    if q.empty:
        raise SystemExit("No data saved yet.")
    if args.exclude_rounds:
        drop = set(args.exclude_rounds)
        names = sorted(set(q.loc[q["round"].isin(drop), "event"]))
        drag, q = drag[~drag["round"].isin(drop)], q[~q["round"].isin(drop)]
        r = r[~r["round"].isin(drop)] if not r.empty else r
        print(f"\nEXCLUDED from the analysis: rounds {sorted(drop)} ({', '.join(names)})")
    print(f"\nData: {q['round'].nunique()} events, {len(q)} quali laps, {len(r)} race laps.")
    analyse_teams(drag, q)
    if not r.empty:
        analyse_tyres(r)


if __name__ == "__main__":
    main()
