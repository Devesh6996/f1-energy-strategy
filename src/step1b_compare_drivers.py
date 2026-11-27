"""
F1 Energy Strategy - Step 1b: Compare energy fingerprints across drivers
=========================================================================
Loads the session ONCE, labels each driver's fastest lap, and produces:
  - one chart: speed traces on top + an "energy map" strip per driver below
  - a summary table (lap time, % of lap in each mode, total speed lost to clipping)
  - the clipping zones that several drivers share (real effect vs. noise)

Tip: compare TEAMMATES (same car + power unit) to isolate driver style,
     e.g. --drivers LEC HAM   or   --drivers NOR PIA   or   --drivers RUS ANT
     Comparing different teams mixes driver style with car/power-unit differences.

Run:
    python src/step1b_compare_drivers.py --drivers VER LEC NOR
    python src/step1b_compare_drivers.py --event Monza --session Q --drivers LEC HAM
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from step1_energy_fingerprint import (
    MODE_COLOURS,
    OUT_DIR,
    fastest_lap_telemetry,
    find_clip_zones,
    label_energy_modes,
    load_session,
    mode_summary,
)

BIN_M = 50  # distance bin size (m) used to match clipping zones between drivers


def summarise(driver: str, lap, df: pd.DataFrame, zones: pd.DataFrame) -> dict:
    share = mode_summary(df)
    lap_time = lap["LapTime"].total_seconds()
    row = {"driver": driver, "lap_time_s": round(lap_time, 3)}
    for mode in MODE_COLOURS:
        row[f"{mode}_%"] = share.get(mode, 0.0)
    row["clip_zones"] = len(zones)
    row["clip_speed_lost_kmh"] = int(zones["speed_lost_kmh"].sum()) if not zones.empty else 0
    return row


def shared_clip_zones(labelled: dict, min_drivers: int = 2) -> pd.DataFrame:
    """Find stretches of track where at least `min_drivers` drivers are clipping."""
    hits = {}
    for driver, df in labelled.items():
        clip_bins = set((df.loc[df["mode"] == "clipping", "Distance"] // BIN_M).astype(int))
        for b in clip_bins:
            hits.setdefault(b, set()).add(driver)

    shared = sorted(b for b, drivers in hits.items() if len(drivers) >= min_drivers)
    if not shared:
        return pd.DataFrame(columns=["start_m", "end_m", "drivers"])

    # Merge neighbouring bins into ranges
    ranges, start, prev = [], shared[0], shared[0]
    for b in shared[1:] + [None]:
        if b is not None and b == prev + 1:
            prev = b
            continue
        drivers = set().union(*(hits[x] for x in range(start, prev + 1)))
        ranges.append({"start_m": start * BIN_M, "end_m": (prev + 1) * BIN_M,
                       "drivers": " ".join(sorted(drivers))})
        if b is not None:
            start = prev = b
    return pd.DataFrame(ranges)


def plot_comparison(labelled: dict, laps: dict, title: str, path) -> None:
    n = len(labelled)
    fig, axes = plt.subplots(
        1 + n, 1, figsize=(15, 5 + 0.7 * n), sharex=True,
        gridspec_kw={"height_ratios": [6] + [1] * n},
    )
    ax_speed = axes[0]
    for driver, df in labelled.items():
        ax_speed.plot(df["Distance"], df["Speed"], linewidth=1.1,
                      label=f"{driver} ({laps[driver]['LapTime'].total_seconds():.3f}s)")
    ax_speed.set_ylabel("Speed (km/h)")
    ax_speed.set_title(title)
    ax_speed.legend(loc="lower right", fontsize=9)
    ax_speed.grid(alpha=0.3)

    # Energy-map strips: one row per driver, coloured by inferred mode
    for ax, (driver, df) in zip(axes[1:], labelled.items()):
        for mode, colour in MODE_COLOURS.items():
            part = df[df["mode"] == mode]
            ax.scatter(part["Distance"], [0] * len(part), marker="|", s=300,
                       color=colour, label=mode)
        ax.set_yticks([])
        ax.set_ylabel(driver, rotation=0, labelpad=20, va="center")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.9), ncol=5, fontsize=8)
    axes[-1].set_xlabel("Distance (m)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare inferred energy modes across drivers.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--event", default="Monza")
    parser.add_argument("--session", default="Q")
    parser.add_argument("--drivers", nargs="+", default=["VER", "LEC", "NOR"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = load_session(args.year, args.event, args.session)

    labelled, laps, rows = {}, {}, []
    for driver in args.drivers:
        try:
            lap, tel = fastest_lap_telemetry(session, driver)
        except ValueError as err:
            print(f"Skipping {driver}: {err}")
            continue
        df = label_energy_modes(tel)
        zones = find_clip_zones(df)
        labelled[driver], laps[driver] = df, lap
        rows.append(summarise(driver, lap, df, zones))

    if not labelled:
        raise SystemExit("No drivers with valid laps - check the driver codes.")

    summary = pd.DataFrame(rows).sort_values("lap_time_s")
    shared = shared_clip_zones(labelled)

    tag = f"{args.year}_{args.event}_{args.session}_{'-'.join(labelled)}".replace(" ", "")
    title = f"{session.event['EventName']} {args.year} {args.session} - energy fingerprints"
    plot_comparison(labelled, laps, title, OUT_DIR / f"{tag}_compare.png")
    summary.to_csv(OUT_DIR / f"{tag}_summary.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"\n{title}\n")
    print(summary.to_string(index=False))
    print(f"\nClipping zones shared by 2+ drivers (likely real, not noise):")
    print(shared.to_string(index=False) if not shared.empty else "  none")
    print(f"\nSaved chart + summary to: {OUT_DIR}")


if __name__ == "__main__":
    main()
