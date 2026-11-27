"""
F1 Energy Strategy - Step 1: Energy fingerprint detector
=========================================================
Public telemetry has no battery / ERS channel, so we INFER what the car is
doing with its energy from speed, throttle and brake:

    braking      -> brakes on (regen under braking)
    lift_coast   -> off throttle, no brake, still fast (lift-and-coast harvesting)
    clipping     -> full throttle but LOSING speed (battery out / super clipping)
    full_deploy  -> full throttle and still accelerating (or holding speed)
    partial      -> everything else (partial throttle, corners)

Run (from the f1-energy-strategy folder, with the venv active):
    python src/step1_energy_fingerprint.py --year 2026 --event Monza --session Q --driver VER

Outputs (in outputs/):
    <tag>_labelled.csv      every telemetry sample with its inferred mode
    <tag>_clip_zones.csv    each clipping zone: where it starts/ends, speed lost
    <tag>_speed_trace.png   speed vs distance, coloured by mode
"""

import argparse
from pathlib import Path

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
OUT_DIR = ROOT / "outputs"

# --- Thresholds: tune them by looking at the plots! -----------------------
FULL_THROTTLE = 98      # % throttle counted as "flat out"
OFF_THROTTLE = 5        # % throttle counted as "lifted"
CLIP_DECEL = -0.5       # m/s^2 -> slowing down this fast at full throttle = clipping
CLIP_MIN_SPEED = 250    # km/h -> v2: only count clipping on fast sections; slower
                        #   full-throttle slowdowns are usually cornering load / tyre scrub
COAST_MIN_SPEED = 120   # km/h -> below this, a lift is just a corner, not a coast
MIN_CLIP_ZONE_M = 30    # metres -> ignore clipping "zones" shorter than this (noise)
SMOOTH_WINDOW = 7       # samples -> v2: was 5; more smoothing = fewer false clipping blips

MODE_COLOURS = {
    "full_deploy": "tab:green",
    "clipping": "tab:red",
    "lift_coast": "tab:orange",
    "braking": "tab:blue",
    "partial": "lightgrey",
}


def setup_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fastf1.Cache.configure(cache_dir=str(CACHE_DIR))   # FastF1 >= 3.9
    except (AttributeError, TypeError):
        fastf1.Cache.enable_cache(str(CACHE_DIR))           # older FastF1


def load_session(year: int, event: str, session_name: str):
    """Load a session once (cached locally after the first download)."""
    setup_cache()
    session = fastf1.get_session(year, event, session_name)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return session


def fastest_lap_telemetry(session, driver: str):
    """Return (lap, telemetry-with-distance) for a driver's fastest lap."""
    lap = session.laps.pick_drivers(driver).pick_fastest()
    if lap is None or pd.isna(lap["LapTime"]):
        raise ValueError(f"No valid timed lap found for {driver}")
    return lap, lap.get_car_data().add_distance()


def load_fastest_lap(year: int, event: str, session_name: str, driver: str):
    session = load_session(year, event, session_name)
    lap, tel = fastest_lap_telemetry(session, driver)
    return session, lap, tel


def label_energy_modes(tel: pd.DataFrame) -> pd.DataFrame:
    """Add an 'accel' column and an inferred 'mode' to every telemetry sample."""
    df = tel[["Time", "Distance", "Speed", "Throttle", "Brake", "nGear", "RPM"]].copy()

    df["t"] = df["Time"].dt.total_seconds()
    df = df.drop_duplicates("t").reset_index(drop=True)

    # Throttle value 104 means "no data" in the F1 feed -> treat as missing
    df.loc[df["Throttle"] > 100, "Throttle"] = np.nan
    df["Throttle"] = df["Throttle"].ffill().bfill()
    df["Brake"] = df["Brake"].astype(float) > 0

    # Longitudinal acceleration from a lightly smoothed speed trace (data is ~4 Hz and noisy)
    v = df["Speed"] / 3.6  # km/h -> m/s
    v_smooth = v.rolling(SMOOTH_WINDOW, center=True, min_periods=1).median()
    df["accel"] = np.gradient(v_smooth.to_numpy(), df["t"].to_numpy())

    full = df["Throttle"] >= FULL_THROTTLE
    off = df["Throttle"] <= OFF_THROTTLE
    brake = df["Brake"]
    fast = df["Speed"] >= CLIP_MIN_SPEED

    df["mode"] = np.select(
        [
            brake,
            full & fast & (df["accel"] < CLIP_DECEL),
            full,
            off & ~brake & (df["Speed"] > COAST_MIN_SPEED),
        ],
        ["braking", "clipping", "full_deploy", "lift_coast"],
        default="partial",
    )
    return df


def find_clip_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive 'clipping' samples into zones and measure the speed lost in each."""
    is_clip = df["mode"].eq("clipping")
    zone_id = (is_clip != is_clip.shift()).cumsum()

    zones = []
    for _, block in df[is_clip].groupby(zone_id[is_clip]):
        length = block["Distance"].iloc[-1] - block["Distance"].iloc[0]
        speed_lost = block["Speed"].iloc[0] - block["Speed"].iloc[-1]
        if length < MIN_CLIP_ZONE_M or speed_lost < 1:   # v2: no real speed lost = noise
            continue
        zones.append({
            "start_m": round(block["Distance"].iloc[0]),
            "end_m": round(block["Distance"].iloc[-1]),
            "length_m": round(length),
            "speed_in_kmh": round(block["Speed"].iloc[0]),
            "speed_out_kmh": round(block["Speed"].iloc[-1]),
            "speed_lost_kmh": round(block["Speed"].iloc[0] - block["Speed"].iloc[-1]),
        })
    return pd.DataFrame(zones)


def mode_summary(df: pd.DataFrame) -> pd.Series:
    """Share of the lap distance spent in each mode."""
    seg = df["Distance"].diff().fillna(0)
    share = seg.groupby(df["mode"]).sum() / seg.sum() * 100
    return share.sort_values(ascending=False).round(1)


def plot_speed_trace(df: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["Distance"], df["Speed"], color="black", linewidth=0.6, alpha=0.4)
    for mode, colour in MODE_COLOURS.items():
        part = df[df["mode"] == mode]
        ax.scatter(part["Distance"], part["Speed"], s=8, color=colour, label=mode)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(title)
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Infer energy deployment modes from one F1 lap.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--event", default="Monza", help="Event name or location, e.g. 'Monza', 'Spa'")
    parser.add_argument("--session", default="Q", help="FP1, FP2, FP3, Q, S, R")
    parser.add_argument("--driver", default="VER", help="Three-letter driver code")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session, lap, tel = load_fastest_lap(args.year, args.event, args.session, args.driver)
    df = label_energy_modes(tel)
    zones = find_clip_zones(df)

    tag = f"{args.year}_{args.event}_{args.session}_{args.driver}".replace(" ", "")
    title = f"{session.event['EventName']} {args.year} {args.session} - {args.driver} - lap {lap['LapTime']}"

    df.to_csv(OUT_DIR / f"{tag}_labelled.csv", index=False)
    zones.to_csv(OUT_DIR / f"{tag}_clip_zones.csv", index=False)
    plot_speed_trace(df, title, OUT_DIR / f"{tag}_speed_trace.png")

    print(f"\n{title}")
    print("\nShare of lap distance per mode (%):")
    print(mode_summary(df).to_string())
    print(f"\nClipping zones found: {len(zones)}")
    if not zones.empty:
        print(zones.to_string(index=False))
    print(f"\nSaved results to: {OUT_DIR}")


if __name__ == "__main__":
    main()
