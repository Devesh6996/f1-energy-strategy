"""
F1 Energy Strategy - Step 2e: Track conditions (rain, wind, humidity) for every lap
==================================================================================
Step 2c saved the laps but not the conditions they were driven in. This step adds them, using
only the lap timing and the weather feed (no telemetry, so it is fast and mostly reads the cache).

For every saved qualifying and race lap it records:
    - rain during the lap (FastF1 'Rainfall' flag) and the tyre compound (INTERMEDIATE / WET = wet lap)
    - wind speed and direction, humidity, air/track temperature, air pressure at that moment
    - when in the session the lap started (needed later for track evolution)
    - whether the lap time was deleted (track limits)

Then it checks:
    A. Which sessions were wet (those laps are dropped from pace comparisons in Step 2d)
    B. Whether the odd effective-drag numbers (e.g. Sao Paulo, Monaco) line up with wind, rain or
       humidity at those tracks.

About wind: FastF1 gives wind speed and the direction it blows FROM, but the car position
coordinates (X/Y) are NOT aligned with north, so we can't tell which straight had a headwind.
What we CAN test is whether windier sessions give stranger drag numbers. Wind is the same for every
team at a track, so the team comparisons in Step 2d (which compare teams at the same track) are
mostly protected; absolute drag values per track are not.

Run (after Step 2c):
    python src/step2e_conditions.py --cache-dir C:\\f1cache
    python src/step2e_conditions.py --analyse-only      # re-run the checks without loading anything
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
SEASON_DIR = OUT_DIR / "season"
CACHE_DIR = ROOT / "data" / "cache"

WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
WET_SESSION_SHARE = 0.3      # a qualifying session counts as wet if over 30% of its laps are wet
R_DRY, R_VAPOUR = 287.05, 461.5


# ----------------------------------------------------------------------------- loading
def setup_cache(cache_dir):
    import fastf1
    path = Path(cache_dir or os.environ.get("F1_CACHE_DIR") or CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    try:
        fastf1.Cache.configure(cache_dir=str(path))
    except (AttributeError, TypeError):
        fastf1.Cache.enable_cache(str(path))


def humid_air_density(pressure_hpa, temp_c, humidity_pct):
    """Air density including water vapour (humid air is slightly LESS dense than dry air)."""
    t = temp_c + 273.15
    p_sat = 6.1078 * 10 ** (7.5 * temp_c / (temp_c + 237.3))           # hPa (Tetens)
    p_v = humidity_pct / 100 * p_sat
    p_d = pressure_hpa - p_v
    return (p_d * 100) / (R_DRY * t) + (p_v * 100) / (R_VAPOUR * t)


def lap_conditions(session):
    """One row per lap: conditions while the lap was driven."""
    laps = session.laps
    w = session.weather_data.sort_values("Time")
    wt = w["Time"].dt.total_seconds().to_numpy(float)
    start = laps["LapStartTime"].dt.total_seconds().to_numpy(float)
    dur = laps["LapTime"].dt.total_seconds().to_numpy(float)
    end = start + np.where(np.isfinite(dur), dur, 0.0)
    mid = (start + end) / 2

    def at(col):
        return np.round(np.interp(mid, wt, w[col].to_numpy(float)), 2) if col in w else np.full(len(laps), np.nan)

    # rain: any 'Rainfall' reading during the lap, or the reading nearest to it
    rain_flag = w["Rainfall"].astype(bool).to_numpy() if "Rainfall" in w else np.zeros(len(w), bool)
    rain = np.zeros(len(laps), bool)
    for i, (a, b) in enumerate(zip(start, end)):
        if not np.isfinite(a):
            continue
        inside = (wt >= a) & (wt <= b)
        rain[i] = rain_flag[inside].any() if inside.any() else rain_flag[np.argmin(np.abs(wt - a))]

    # wind direction is an angle: average it as a vector, not as a number (350 and 10 degrees -> 0)
    if "WindDirection" in w:
        ang = np.deg2rad(w["WindDirection"].to_numpy(float))
        wind_dir = np.rad2deg(np.arctan2(np.interp(mid, wt, np.sin(ang)), np.interp(mid, wt, np.cos(ang)))) % 360
    else:
        wind_dir = np.full(len(laps), np.nan)

    compound = laps["Compound"].astype(str).str.upper().to_numpy() if "Compound" in laps else np.full(len(laps), "")
    out = pd.DataFrame({
        "driver": laps["Driver"].to_numpy(),
        "lap_number": laps["LapNumber"].to_numpy(),
        "session_time_s": np.round(start, 1),
        "compound": compound,
        "rain": rain,
        "wet": rain | np.isin(compound, list(WET_COMPOUNDS)),
        "wind_speed_ms": at("WindSpeed"),
        "wind_dir_deg": np.round(wind_dir, 0),
        "humidity_%": at("Humidity"),
        "air_temp": at("AirTemp"),
        "track_temp": at("TrackTemp"),
        "pressure_hpa": at("Pressure"),
        "deleted": laps["Deleted"].fillna(False).astype(bool).to_numpy() if "Deleted" in laps else False,
        # track status during the lap: 1 green, 2 yellow, 4 safety car, 5 red flag, 6 VSC, 7 VSC ending
        "track_status": laps["TrackStatus"].astype(str).to_numpy() if "TrackStatus" in laps else "",
    }).dropna(subset=["lap_number"]).astype({"lap_number": int})
    out["rho_humid"] = np.round(humid_air_density(out["pressure_hpa"], out["air_temp"], out["humidity_%"]), 4)
    return out


def collect(year, rounds=None, refresh=False):
    import fastf1
    try:
        fastf1.set_log_level("ERROR")      # hide the cache / rate-limit chatter
    except Exception:
        pass
    for f in sorted(SEASON_DIR.glob("[0-9][0-9]_Q_laps.csv")):
        rnd = int(f.name[:2])
        if rounds and rnd not in rounds:
            continue
        for kind in ("Q", "R"):
            out = SEASON_DIR / f"{rnd:02d}_{kind}_conditions.csv"
            if out.exists() and not refresh:
                continue
            try:
                s = fastf1.get_session(year, rnd, kind)
                # messages=True: FastF1 needs the race-control messages to know which laps were deleted
                s.load(laps=True, telemetry=False, weather=True, messages=True)
                lap_conditions(s).to_csv(out, index=False)
                print(f"Round {rnd:2d} {kind}: conditions saved")
            except Exception as err:
                print(f"Round {rnd:2d} {kind}: could not load conditions ({type(err).__name__}: {err})")


# ----------------------------------------------------------------------------- analysis
def load_conditions(kind):
    files = sorted(SEASON_DIR.glob(f"[0-9][0-9]_{kind}_conditions.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f).assign(round=int(f.name[:2])) for f in files], ignore_index=True)


def analyse():
    qc = load_conditions("Q")
    if qc.empty:
        raise SystemExit("No conditions saved yet - run without --analyse-only first.")
    q = pd.concat([pd.read_csv(f) for f in sorted(SEASON_DIR.glob("[0-9][0-9]_Q_laps.csv"))], ignore_index=True)
    drag = pd.concat([pd.read_csv(f) for f in sorted(SEASON_DIR.glob("[0-9][0-9]_drag.csv"))], ignore_index=True)
    rc = load_conditions("R")

    # conditions on the quali laps Step 2c actually used
    used = q[["round", "event", "driver", "lap_number"]].merge(qc, on=["round", "driver", "lap_number"], how="left")
    missing = used["wet"].isna().mean()
    if missing > 0.05:
        print(f"Note: {missing:.0%} of saved quali laps have no matching conditions row.")

    by = used.groupby(["round", "event"]).agg(
        wet_laps_share=("wet", "mean"), wind_ms=("wind_speed_ms", "mean"), wind_max_ms=("wind_speed_ms", "max"),
        wind_dir_change_deg=("wind_dir_deg", lambda a: _circ_range(a)), humidity=("humidity_%", "mean"),
        air_temp=("air_temp", "mean"), rho_humid=("rho_humid", "mean"), deleted_share=("deleted", "mean"))
    by = by.join(drag.groupby("round")[[c for c in ("rho", "cda_lo", "cda_hi") if c in drag]].mean(), on="round")
    by["rho_change_%"] = (by["rho_humid"] / by["rho"] - 1) * 100 if "rho" in by else np.nan
    if not rc.empty:
        by = by.join(rc.groupby("round")["rain"].mean().rename("race_rain_share"), on="round")
        if "track_status" in rc:
            nl = neutral_laps(rc)
            by = by.join(pd.Series({r: len(v) for r, v in nl.items()}, name="race_sc_vsc_red_laps"), on="round")
    by = by.reset_index()
    by["wet_quali"] = by["wet_laps_share"] > WET_SESSION_SHARE
    by.to_csv(SEASON_DIR / "conditions_by_round.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 100 + "\nA. CONDITIONS IN QUALIFYING (laps used by Step 2c)\n" + "=" * 100)
    cols = ["round", "event", "wet_laps_share", "wind_ms", "wind_max_ms", "wind_dir_change_deg", "humidity",
            "rho_change_%", "deleted_share", "cda_hi"] + [c for c in ("race_rain_share", "race_sc_vsc_red_laps") if c in by]
    print(by[cols].round(2).to_string(index=False))
    wet = by.loc[by["wet_quali"], "event"].tolist()
    print(f"\nWet qualifying sessions (> {WET_SESSION_SHARE:.0%} wet laps): {', '.join(wet) if wet else 'none'}")
    print("rho_change_% = how much humidity changes the air density used for drag (humid air is lighter).")
    print("Step 2d now drops wet laps automatically, and ignores drag from wet qualifying sessions.")
    if by["deleted_share"].fillna(0).eq(0).all():
        print("NOTE: no deleted laps found anywhere - the race-control messages were probably not loaded."
              " Re-run with --refresh.")

    print("\n" + "=" * 100 + "\nB. DO THE ODD DRAG NUMBERS LINE UP WITH THE CONDITIONS?\n" + "=" * 100)
    d = by.dropna(subset=["cda_hi"]).copy()
    d["drag_vs_median"] = d["cda_hi"] - d["cda_hi"].median()
    for col, what in [("wind_ms", "average wind"), ("wind_max_ms", "strongest wind"),
                      ("humidity", "humidity"), ("wet_laps_share", "share of wet laps")]:
        x = d[col].astype(float)
        if x.notna().sum() >= 5 and x.std() > 0:
            r = np.corrcoef(x[x.notna()], d.loc[x.notna(), "drag_vs_median"])[0, 1]
            rs = pd.Series(x).rank().corr(d["drag_vs_median"].rank())
            print(f"  effective drag vs {what:18s}: Pearson r = {r:+.2f}, rank r = {rs:+.2f}  ({x.notna().sum()} tracks)")
    print("  |r| above ~0.45 with 21 tracks is roughly 'clear'. Track layout still dominates drag numbers,")
    print("  so a weak correlation does NOT mean wind doesn't matter - only that it isn't the main story.")
    dry = d[~d["wet_quali"]]
    if len(dry) >= 5:
        r_dry = np.corrcoef(dry["humidity"], dry["drag_vs_median"])[0, 1]
        print(f"  without wet qualifying sessions: drag vs humidity r = {r_dry:+.2f}")
        print(f"  humidity changes air density by only {d['rho_change_%'].min():+.1f}% to {d['rho_change_%'].max():+.1f}%,"
              " far too little to move drag by the amounts seen - a correlation means humid TRACKS differ"
              " (e.g. slow street circuits), not that humidity itself adds drag.")
    top = d.reindex(d["drag_vs_median"].abs().sort_values(ascending=False).index).head(4)
    print("\nThe four most unusual drag numbers and their conditions:")
    print(top[["round", "event", "cda_hi", "drag_vs_median", "wet_laps_share", "wind_ms", "humidity"]].round(2)
          .to_string(index=False))

    # within a session: does a driver's drag move with the wind during the laps they set?
    dw = used.groupby(["round", "driver"])["wind_speed_ms"].mean().rename("driver_wind")
    dd = drag.join(dw, on=["round", "driver"]).dropna(subset=["cda_hi", "driver_wind"])
    dd = dd[~dd["round"].isin(by.loc[by["wet_quali"], "round"])]
    if len(dd) > 30:
        # within each round: deviation from the round's average (track and conditions cancel)
        dd["x"] = dd["driver_wind"] - dd.groupby("round")["driver_wind"].transform("mean")
        dd["y"] = dd["cda_hi"] - dd.groupby(["round", "team"])["cda_hi"].transform("mean")
        per = []
        for rnd, g in dd.groupby("round"):
            if g["x"].std() > 0.05 and len(g) >= 8:
                per.append({"round": rnd, "slope_cda_per_ms": np.polyfit(g["x"], g["y"], 1)[0],
                            "wind_spread_ms": g["x"].std()})
        if per:
            per = pd.DataFrame(per)
            per["abs"] = per["slope_cda_per_ms"].abs()
            print(f"\nWithin a session, wind differs by up to {per['wind_spread_ms'].max():.1f} m/s between drivers'"
                  f" laps. Change in effective drag per +1 m/s of wind (teammates compared), median over"
                  f" {len(per)} tracks: {per['slope_cda_per_ms'].median():+.3f}. Largest:")
            print(per.sort_values("abs", ascending=False).head(5).drop(columns="abs").round(3).to_string(index=False))
            print("  The sign can differ by track (headwind vs tailwind on the main straight); a few large slopes"
                  " at windy tracks would point to wind affecting the drag numbers there.")


NEUTRAL = ("4", "5", "6", "7")    # safety car, red flag, VSC, VSC ending


def neutral_laps(rc):
    """Race lap numbers run (partly) under SC / VSC / red flag, per round (from the leaders' laps)."""
    out = {}
    for rnd, g in rc.groupby("round"):
        st = g["track_status"].astype(str)
        hit = st.apply(lambda x: any(c in x for c in NEUTRAL))
        # a lap counts as neutralised if at least a quarter of the cars on that lap saw SC/VSC/red
        share = hit.groupby(g["lap_number"]).mean()
        out[int(rnd)] = sorted(int(l) for l in share[share >= 0.25].index)
    return out


def _circ_range(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return np.nan
    ang = np.deg2rad(a)
    mean = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    dev = np.rad2deg(np.angle(np.exp(1j * (ang - mean))))
    return float(dev.max() - dev.min())


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Add rain, wind and humidity to every saved lap.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--rounds", nargs="+", type=int, default=None)
    parser.add_argument("--refresh", action="store_true", help="reload conditions even if already saved")
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    if not args.analyse_only:
        setup_cache(args.cache_dir)
        collect(args.year, args.rounds, args.refresh)
    analyse()


if __name__ == "__main__":
    main()
