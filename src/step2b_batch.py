"""
F1 Energy Strategy - Step 2b: Batch analysis (all flying laps, all drivers)
============================================================================
Step 2 looked at ONE lap per driver - a hint, not proof. This script runs the
power model over every flying lap of every driver, for one or more events, and
checks the "possible slipstream" label against the REAL gap to the car ahead.

How drag is fitted: per lap (topspeed fit), then the driver's median across
all their laps is used for every lap -> much more stable than a single lap.
Air density is computed from each session's weather (temperature + pressure),
so tracks at different altitudes / temperatures compare fairly.

Run:
    python src/step2b_batch.py                                  # Monza qualifying, all drivers
    python src/step2b_batch.py --events Monza Spa Baku          # several events
    python src/step2b_batch.py --drivers PIA NOR LEC HAM        # only some drivers
    python src/step2b_batch.py --no-tow                         # faster: skip the gap-to-car-ahead check

Outputs (in outputs/):
    batch_<year>_<session>_laps.csv     one row per lap
    batch_<year>_<session>_zones.csv    one row per clipping zone (with gap to car ahead)
    batch_<year>_<session>_drivers.csv  one row per driver per event
"""

import argparse
import warnings

import fastf1
import numpy as np
import pandas as pd

import step2_power_model as s2
from step1_energy_fingerprint import OUT_DIR, find_clip_zones, label_energy_modes, setup_cache

TOW_GAP_M = 120          # car ahead closer than this on a straight = in the tow (~1.4 s at 300 km/h)
TOW_MIN_SPEED = 280      # km/h -> only check the gap on fast straights
QUICK_LAP_THRESHOLD = 1.07   # keep laps within 107% of the session's fastest lap

# 2026 power-unit suppliers (matched on team name, case-insensitive)
PU_SUPPLIERS = {
    "Mercedes": ["mercedes", "mclaren", "williams", "alpine"],
    "Ferrari": ["ferrari", "haas", "cadillac"],
    "Red Bull Ford": ["red bull", "racing bulls", "visa"],
    "Honda": ["aston"],
    "Audi": ["audi", "sauber", "kick"],
}


def power_unit(team: str) -> str:
    t = str(team).lower()
    for pu, keys in PU_SUPPLIERS.items():
        if any(k in t for k in keys):
            return pu
    return "unknown"
R_DRY_AIR = 287.05       # J/(kg K)


def load_event(year, event, session_name):
    setup_cache()
    session = fastf1.get_session(year, event, session_name)
    session.load(laps=True, telemetry=True, weather=True, messages=False)
    return session


def air_density(session) -> float:
    """Air density from the session's median air temperature and pressure."""
    try:
        w = session.weather_data
        temp_k = float(w["AirTemp"].median()) + 273.15
        pressure_pa = float(w["Pressure"].median()) * 100   # mbar -> Pa
        rho = pressure_pa / (R_DRY_AIR * temp_k)
        if 0.8 < rho < 1.35:
            return rho
    except Exception:
        pass
    return s2.RHO


def tow_info(lap):
    """Gap to the car ahead along the lap (or None if FastF1 can't compute it)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tel = lap.get_car_data().add_distance().add_driver_ahead()
        return tel[["Distance", "Speed", "Throttle", "DistanceToDriverAhead"]]
    except Exception:
        return None


def lap_metrics(event, driver, lap, df, zones, model, gap):
    share = df["Distance"].diff().fillna(0).groupby(df["mode"]).sum()
    share = share / share.sum() * 100
    reasons = zones["reason"].value_counts() if not zones.empty else pd.Series(dtype=int)

    # Power above what the rules allow at high speed (non-clipping) = less drag than modelled.
    # On a lap spent in someone's tow this should be clearly positive.
    hs = (df["Speed"] >= s2.TOP_FIT_MIN) & (df["Throttle"] >= 98) & df["mode"].ne("clipping")
    excess = float((df.loc[hs, "power_kw"] - df.loc[hs, "allowed_kw"]).median()) if hs.any() else np.nan

    towed, min_gap = np.nan, np.nan
    if gap is not None:
        fast = gap[(gap["Speed"] >= TOW_MIN_SPEED) & (gap["Throttle"] >= 98)]
        if len(fast):
            min_gap = float(fast["DistanceToDriverAhead"].min())
            towed = bool(min_gap < TOW_GAP_M)

    return {
        "event": event,
        "driver": driver,
        "team": lap.get("Team", ""),
        "power_unit": power_unit(lap.get("Team", "")),
        "lap_number": int(lap["LapNumber"]) if pd.notna(lap.get("LapNumber")) else None,
        "lap_time_s": round(lap["LapTime"].total_seconds(), 3),
        "compound": lap.get("Compound", ""),
        "clipping_%": round(share.get("clipping", 0.0), 1),
        "lift_coast_%": round(share.get("lift_coast", 0.0), 1),
        "full_deploy_%": round(share.get("full_deploy", 0.0), 1),
        "electric_mj": round(s2.electric_energy_mj(df, model), 2),
        "zones_super_clip": int(reasons.get("super clipping (harvesting at full throttle)", 0)),
        "zones_reduced_deploy": int(reasons.get("reduced deployment / battery low", 0)),
        "zones_reg_taper": int(reasons.get("regulation taper", 0)),
        "zones_above_limit": int(reasons.get("above modelled limit (tow or model error)", 0)),
        "excess_power_kw": None if pd.isna(excess) else round(excess),
        "min_gap_ahead_m": None if pd.isna(min_gap) else round(min_gap),
        "towed": towed,
    }


def analyse_driver(event, driver, laps, rho, do_tow):
    """Two passes: fit drag on every lap, then re-analyse every lap with the driver's median drag."""
    labelled, k_values = [], []
    for _, lap in laps.iterlaps():
        try:
            df = label_energy_modes(lap.get_car_data().add_distance())
            k = s2.fit_car_model(df, "topspeed")["k"]
        except Exception:
            continue
        labelled.append((lap, df))
        if pd.notna(k):
            k_values.append(k)

    if len(k_values) < 1:
        return [], [], None

    k = float(np.median(k_values))
    model = {"p_max_w": (s2.ICE_NOMINAL_KW + s2.MGU_K_MAX_KW) * 1000 * s2.DRIVETRAIN_EFF,
             "k": k, "cda_m2": 2 * k / rho, "mode": "topspeed"}

    lap_rows, zone_rows = [], []
    for lap, df in labelled:
        df = s2.add_power_columns(df, model)
        zones = s2.classify_zones(df, find_clip_zones(df))
        gap = tow_info(lap) if do_tow else None
        lap_rows.append(lap_metrics(event, driver, lap, df, zones, model, gap))

        for _, z in zones.iterrows():
            zone_gap = np.nan
            if gap is not None:
                seg = gap[gap["Distance"].between(z["start_m"] - 50, z["end_m"])]
                if len(seg):
                    zone_gap = float(seg["DistanceToDriverAhead"].median())
            zone_rows.append({"event": event, "driver": driver,
                              "lap_number": lap_rows[-1]["lap_number"], **z.to_dict(),
                              "gap_ahead_m": None if pd.isna(zone_gap) else round(zone_gap)})

    driver_row = {"event": event, "driver": driver, "laps": len(lap_rows),
                  "cda_m2": round(model["cda_m2"], 2),
                  "cda_spread": round(2 * np.std(k_values) / rho, 2) if len(k_values) > 1 else 0.0}
    return lap_rows, zone_rows, driver_row


def print_reports(laps_df, zones_df, drivers_df):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n=== Per driver (averages over all flying laps) ===")
    cols = ["clipping_%", "lift_coast_%", "electric_mj", "zones_super_clip", "zones_above_limit"]
    per_driver = (laps_df.groupby(["event", "team", "driver"])
                  .agg(laps=("lap_time_s", "size"), best_lap=("lap_time_s", "min"),
                       **{c: (c, "mean") for c in cols})
                  .round(2).reset_index()
                  .merge(drivers_df[["event", "driver", "cda_m2", "cda_spread"]], on=["event", "driver"]))
    print(per_driver.sort_values(["event", "best_lap"]).to_string(index=False))

    print("\n=== By power-unit supplier (same engine -> similar energy strategy?) ===")
    by_pu = (laps_df.groupby("power_unit")
             .agg(teams=("team", "nunique"), drivers=("driver", "nunique"), laps=("lap_time_s", "size"),
                  **{c: (c, "mean") for c in cols[:4]})
             .round(2).sort_values("zones_super_clip", ascending=False))
    print(by_pu.to_string())
    print("If energy strategy is set by the power unit, teams sharing an engine should look alike.")

    print("\n=== Teammate comparison (same car, so differences = driver style) ===")
    for (event, team), grp in per_driver.groupby(["event", "team"]):
        if len(grp) != 2:
            continue
        a, b = grp.sort_values("best_lap").to_dict("records")
        print(f"{event} | {team}: {a['driver']} vs {b['driver']}  "
              f"clipping {a['clipping_%']:.1f}% vs {b['clipping_%']:.1f}%  |  "
              f"lift-coast {a['lift_coast_%']:.1f}% vs {b['lift_coast_%']:.1f}%  |  "
              f"super-clip zones/lap {a['zones_super_clip']:.1f} vs {b['zones_super_clip']:.1f}  |  "
              f"electric {a['electric_mj']:.1f} vs {b['electric_mj']:.1f} MJ  "
              f"({a['laps']} vs {b['laps']} laps)")

    if laps_df["towed"].notna().any():
        print(f"\n=== Validation 1: do towed laps look like 'extra power'? ===")
        t = laps_df.dropna(subset=["towed", "excess_power_kw"]).copy()
        t["towed"] = t["towed"].map({True: f"car < {TOW_GAP_M} m ahead on a straight", False: "clean air"})
        print(t.groupby("towed")["excess_power_kw"].agg(["median", "mean", "size"]).round(0).to_string())
        print("If the physics is right, towed laps should show clearly MORE excess power (less drag).")

    if "gap_ahead_m" in zones_df and zones_df["gap_ahead_m"].notna().any() \
            and zones_df["reason"].eq("above modelled limit (tow or model error)").any():
        print("\n=== Validation 2: are 'above modelled limit' zones explained by a REAL car ahead? ===")
        z = zones_df.dropna(subset=["gap_ahead_m"]).copy()
        z["car_ahead_close"] = z["gap_ahead_m"] < TOW_GAP_M
        z["labelled_slipstream"] = z["reason"].eq("above modelled limit (tow or model error)")
        table = z.groupby("labelled_slipstream")["car_ahead_close"].agg(["mean", "size"])
        table.index = table.index.map({True: "above modelled limit", False: "all other zones"})
        table.columns = [f"share with car < {TOW_GAP_M} m ahead", "zones"]
        print(table.round(2).to_string())
        print("If tow explained them, the first share would be clearly HIGHER. If not -> model error.")


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty-slice means on zones with no data
    parser = argparse.ArgumentParser(description="Run the power model over all flying laps.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--events", nargs="+", default=["Monza"])
    parser.add_argument("--session", default="Q")
    parser.add_argument("--drivers", nargs="+", default=None, help="default: all drivers")
    parser.add_argument("--no-tow", action="store_true", help="skip the gap-to-car-ahead check")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_laps, all_zones, all_drivers = [], [], []

    for event in args.events:
        try:
            session = load_event(args.year, event, args.session)
        except Exception as err:
            print(f"Skipping {event}: could not load ({err})")
            continue
        rho = air_density(session)
        name = session.event["EventName"]
        print(f"\n{name}: air density {rho:.3f} kg/m^3")

        quick = session.laps.pick_quicklaps(QUICK_LAP_THRESHOLD)
        drivers = args.drivers or sorted(quick["Driver"].unique())
        for i, driver in enumerate(drivers, 1):
            laps = quick.pick_drivers(driver)
            if laps.empty:
                continue
            print(f"  [{i}/{len(drivers)}] {driver}: {len(laps)} flying laps...", flush=True)
            lap_rows, zone_rows, driver_row = analyse_driver(name, driver, laps, rho, not args.no_tow)
            if driver_row is None:
                print(f"      skipped (no laps fast enough for the topspeed fit)")
                continue
            all_laps += lap_rows
            all_zones += zone_rows
            all_drivers.append(driver_row)

    if not all_laps:
        raise SystemExit("Nothing analysed - check the event names and driver codes.")

    laps_df, zones_df, drivers_df = pd.DataFrame(all_laps), pd.DataFrame(all_zones), pd.DataFrame(all_drivers)
    tag = f"batch_{args.year}_{args.session}"
    laps_df.to_csv(OUT_DIR / f"{tag}_laps.csv", index=False)
    zones_df.to_csv(OUT_DIR / f"{tag}_zones.csv", index=False)
    drivers_df.to_csv(OUT_DIR / f"{tag}_drivers.csv", index=False)

    print_reports(laps_df, zones_df, drivers_df)
    print(f"\nAnalysed {len(laps_df)} laps. Saved CSVs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
