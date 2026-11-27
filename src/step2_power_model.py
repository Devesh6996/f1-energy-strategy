"""
F1 Energy Strategy - Step 2: Physics power model
=================================================
Turns telemetry into ESTIMATED POWER (kW) at every point of the lap, then
separates two reasons a car stops accelerating at full throttle:

    energy-limited      -> the car delivers LESS power than it is allowed to
                           (battery empty, or super clipping = harvesting at full throttle)
    regulation taper    -> the car delivers what the rules allow, but the rules
                           cut MGU-K deployment at high speed

Physics (straight-line equation of motion):
    P_wheel = v * ( m*a  +  k*v^2  +  Crr*m*g )
              power   inertia  drag    rolling resistance
    with  k = 0.5 * rho * CdA

Two unknowns per car: P_max (total power) and k (drag). We fit both from
the telemetry itself, using full-throttle samples at speeds where the rules
still allow the full 350 kW of electric power:
    m*a*v + Crr*m*g*v  =  P_max  -  k * v^3        <- a straight line in v^3!

Run:
    python src/step2_power_model.py --drivers PIA NOR
    python src/step2_power_model.py --drivers LEC HAM --event Monza --session Q

Outputs (in outputs/):
    <tag>_<DRV>_power.png      speed + estimated power vs. what the rules allow
    <tag>_<DRV>_fit.png        the drag/power fit (check this looks like a clean line!)
    <tag>_<DRV>_zones.csv      every clipping zone, classified
    <tag>_power_summary.csv    one row per driver
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step1_energy_fingerprint import (
    OUT_DIR,
    fastest_lap_telemetry,
    find_clip_zones,
    label_energy_modes,
    load_session,
)

# --- Car & physics assumptions (all adjustable) ------------------------------
MASS_KG = 780          # 2026 minimum is 768 kg incl. driver, WITHOUT fuel -> + ~10 kg quali fuel
RHO = 1.18             # air density kg/m^3 (warm day, near sea level)
CRR = 0.015            # rolling resistance coefficient (rough estimate for F1 tyres)
G = 9.81

# --- 2026 MGU-K rules ----------------------------------------------------------
MGU_K_MAX_KW = 350
ICE_NOMINAL_KW = 400   # published figure, only used as a sanity check

# --- Fit settings --------------------------------------------------------------
FIT_SPEED_MIN = 150    # km/h -> below this, cars can be traction-limited (wheelspin)
FIT_SPEED_MAX = 285    # km/h -> above ~290 the rules start cutting deployment
FIT_MODE = "topspeed"  # "topspeed" (default), "envelope" or "free" - see fit_car_model()
TOP_FIT_MIN = 290      # km/h -> topspeed mode fits drag on the straights, where the
TOP_FIT_MAX = 340      #   rules' speed-based deployment cap tells us the power
MAX_LAP_ELECTRIC_MJ = 12   # 4 MJ usable battery + ~7-9 MJ max recharge per lap
DRIVETRAIN_EFF = 0.95  # share of PU power that reaches the wheels (gearbox/tyre losses)
ENVELOPE_PCT = 0.80    # per speed bin, use this quantile = moments the car was at full power
FIT_BIN_KMH = 10       # bin samples by speed
EDGE_TRIM = 3          # samples -> skip this many at each end of a full-throttle run
OUTLIER_MADS = 2.5     # drop fit points further than this from the line, then refit
ENERGY_LIMIT_KW = 60   # deficit above this = car is genuinely short of power
SUPERCLIP_MARGIN_KW = 50   # power this far below the ICE-only level = MGU-K is HARVESTING
TOW_MARGIN_KW = -40        # power this far ABOVE what the rules allow = the model is missing something:
                           #   slipstream (less drag) OR model error (cap formula / ICE power at ~330 km/h).
                           #   Step 2b showed most of these zones had NO car close ahead -> mostly model error.

REASON_COLOURS = {
    "super clipping (harvesting at full throttle)": "tab:red",
    "reduced deployment / battery low": "tab:purple",
    "regulation taper": "tab:orange",
    "above modelled limit (tow or model error)": "tab:cyan",
    "unclear": "grey",
}


def deployment_cap_kw(speed_kmh):
    """Max MGU-K deployment allowed at a given speed (overtake mode OFF).

    2026 rule: 1800 - 5*v below 340 km/h (capped at 350 kW),
               6900 - 20*v between 340-345 km/h, zero above 345 km/h.
    Below ~290 km/h this is simply the full 350 kW.
    """
    v = np.asarray(speed_kmh, dtype=float)
    cap = np.minimum(MGU_K_MAX_KW, 1800 - 5 * v)
    cap = np.where(v >= 340, 6900 - 20 * v, cap)
    cap = np.where(v >= 345, 0, cap)
    return np.clip(cap, 0, MGU_K_MAX_KW)


def allowed_wheel_power_w(speed_kmh):
    """Max power at the wheels the rules allow: (ICE + MGU-K cap) x drivetrain efficiency."""
    return (ICE_NOMINAL_KW + deployment_cap_kw(speed_kmh)) * 1000 * DRIVETRAIN_EFF


def fit_car_model(df: pd.DataFrame, mode: str = FIT_MODE) -> dict:
    """Fit the car's drag factor k (and, in 'free' mode, total power P_max).

    Equation along a straight:  m*a*v + Crr*m*g*v  =  P(v)  -  k * v^3

    'topspeed' -> (default) use full-throttle samples at 290-340 km/h on the
                  straights. There the rules cap MGU-K power by speed, so P(v)
                  is KNOWN, and the car is in low-drag (straight-line) aero mode.
    'envelope' -> fix peak power from the rules and fit k from the top edge of the
                  150-285 km/h data. On real 2026 data this overestimates drag
                  (corner exits: traction limits, high-drag aero mode).
    'free'     -> fit P_max and k together. Fails on 2026 data because cars do
                  not deploy full power everywhere.
    """
    v = df["Speed"] / 3.6
    # Only samples well INSIDE a full-throttle run: near the start/end of a run the
    # smoothing window still "sees" the slower phase before/after it and biases accel.
    is_full = df["mode"].eq("full_deploy").astype(int)
    stable = is_full.rolling(2 * EDGE_TRIM + 1, center=True, min_periods=1).min().eq(1)
    lo, hi = (TOP_FIT_MIN, TOP_FIT_MAX) if mode == "topspeed" else (FIT_SPEED_MIN, FIT_SPEED_MAX)
    mask = stable & df["Speed"].between(lo, hi)
    sample = pd.DataFrame({
        "speed_kmh": df.loc[mask, "Speed"],
        "x": v[mask] ** 3,
        "y": MASS_KG * df.loc[mask, "accel"] * v[mask] + CRR * MASS_KG * G * v[mask],
    })
    sample["bin"] = (sample["speed_kmh"] // FIT_BIN_KMH).astype(int)
    if len(sample) < 8 or sample["bin"].nunique() < 2:
        raise ValueError(f"Too few full-throttle samples at {lo}-{hi} km/h for '{mode}' fit "
                         f"(try --fit envelope on slower tracks).")

    p_max = (ICE_NOMINAL_KW + MGU_K_MAX_KW) * 1000 * DRIVETRAIN_EFF
    if mode == "free":
        binned = sample.groupby("bin")[["speed_kmh", "x", "y"]].median()
        A = np.column_stack([np.ones(len(binned)), -binned["x"]])
        p_max, k = np.linalg.lstsq(A, binned["y"], rcond=None)[0]
    elif mode == "envelope":
        binned = sample.groupby("bin").agg(speed_kmh=("speed_kmh", "median"),
                                           x=("x", "median"),
                                           y=("y", lambda s: s.quantile(ENVELOPE_PCT)))
        k_bin = (p_max - binned["y"]) / binned["x"]
        k = float(np.median(k_bin[k_bin > 0])) if (k_bin > 0).any() else np.nan
    else:  # topspeed
        k_i = (allowed_wheel_power_w(sample["speed_kmh"]) - sample["y"]) / sample["x"]
        k = float(np.median(k_i[k_i > 0])) if (k_i > 0).any() else np.nan
        binned = sample.groupby("bin")[["speed_kmh", "x", "y"]].median()

    return {"p_max_w": p_max, "k": k, "cda_m2": 2 * k / RHO, "mode": mode,
            "binned": binned, "sample": sample}


def add_power_columns(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Estimated wheel power, power the rules allow, and the gap between them."""
    out = df.copy()
    v = out["Speed"] / 3.6
    p_w = v * (MASS_KG * out["accel"] + model["k"] * v ** 2 + CRR * MASS_KG * G)
    out["power_kw"] = (p_w / 1000).rolling(5, center=True, min_periods=1).median()

    out["cap_kw"] = deployment_cap_kw(out["Speed"])
    if model["mode"] == "free":
        ice_kw = model["p_max_w"] / 1000 - MGU_K_MAX_KW * DRIVETRAIN_EFF
        out["allowed_kw"] = ice_kw + out["cap_kw"] * DRIVETRAIN_EFF
    else:
        out["allowed_kw"] = allowed_wheel_power_w(out["Speed"]) / 1000
    out["deficit_kw"] = out["allowed_kw"] - out["power_kw"]
    out.loc[out["Throttle"] < 98, "deficit_kw"] = np.nan   # only meaningful at full throttle
    return out


def safe_round(x):
    return None if pd.isna(x) else int(round(x))


def classify_zones(df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    if zones.empty:
        return zones
    rows = []
    for _, z in zones.iterrows():
        # +/-10 m margin: zone bounds are rounded, very short zones could end up empty
        seg = df[df["Distance"].between(z["start_m"] - 10, z["end_m"] + 10)]
        deficit = seg["deficit_kw"].mean()
        cap = seg["cap_kw"].mean()
        power = seg["power_kw"].mean()
        ice_wheel_kw = ICE_NOMINAL_KW * DRIVETRAIN_EFF
        if seg.empty or pd.isna(deficit):
            reason = "unclear"
        elif deficit > ENERGY_LIMIT_KW:
            # Below the ICE-only level the MGU-K must be taking energy OUT (harvesting).
            # An empty battery alone would still leave ~the ICE's power.
            if power < ice_wheel_kw - SUPERCLIP_MARGIN_KW:
                reason = "super clipping (harvesting at full throttle)"
            else:
                reason = "reduced deployment / battery low"
        elif deficit < TOW_MARGIN_KW:
            reason = "above modelled limit (tow or model error)"
        elif cap < MGU_K_MAX_KW - 20:
            reason = "regulation taper"
        else:
            reason = "unclear"
        rows.append({**z.to_dict(),
                     "power_kw": safe_round(seg["power_kw"].mean()),
                     "allowed_kw": safe_round(seg["allowed_kw"].mean()),
                     "deficit_kw": safe_round(deficit),
                     "rule_cap_kw": safe_round(cap),
                     "reason": reason})
    return pd.DataFrame(rows)


def electric_energy_mj(df: pd.DataFrame, model: dict) -> float:
    """Rough electrical energy deployed at full throttle: power above the ICE's share."""
    ice_kw = ICE_NOMINAL_KW * DRIVETRAIN_EFF if model["mode"] != "free" \
        else model["p_max_w"] / 1000 - MGU_K_MAX_KW * DRIVETRAIN_EFF
    full = df["Throttle"] >= 98
    electric_kw = (df["power_kw"] - ice_kw).clip(lower=0).where(full, 0) / DRIVETRAIN_EFF
    dt = df["t"].diff().fillna(0)
    return float((electric_kw * dt).sum() / 1000)   # kJ -> MJ


def plot_power(df, zones, title, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.4]})
    ax1.plot(df["Distance"], df["Speed"], color="black", linewidth=1)
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)

    full = df["Throttle"] >= 98
    ax2.plot(df["Distance"], df["allowed_kw"], color="tab:green", linewidth=1.2,
             label="Max power the rules allow (ICE + MGU-K cap)")
    ax2.scatter(df.loc[full, "Distance"], df.loc[full, "power_kw"], s=6,
                color="tab:blue", label="Estimated power at full throttle")
    ax2.set_ylabel("Power (kW)")
    ax2.set_ylim(0, df["allowed_kw"].max() * 1.25)
    ax2.grid(alpha=0.3)

    colours = REASON_COLOURS
    for _, z in zones.iterrows():
        for ax in (ax1, ax2):
            ax.axvspan(z["start_m"], z["end_m"], color=colours[z["reason"]], alpha=0.25)
    ax2.axhline(ICE_NOMINAL_KW * DRIVETRAIN_EFF, color="black", linestyle=":", linewidth=1,
                label="ICE-only power (below this = harvesting)")
    for reason in (zones["reason"].unique() if not zones.empty else []):
        ax2.axvspan(0, 0, color=colours[reason], alpha=0.25, label=f"Zone: {reason}")
    ax2.legend(loc="lower left", fontsize=8, ncol=2)
    ax2.set_xlabel("Distance (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fit(model, title, path):
    s, b = model["sample"], model["binned"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(s["speed_kmh"], s["y"] / 1000, s=6, alpha=0.3, color="grey", label="samples")
    label = f"speed-bin top edge ({ENVELOPE_PCT:.0%} quantile)" if model["mode"] == "envelope" else "speed-bin medians"
    ax.scatter(b["speed_kmh"], b["y"] / 1000, s=30, color="tab:blue", label=label)
    lo, hi = s["speed_kmh"].min(), s["speed_kmh"].max()
    v = np.linspace(lo, hi, 100) / 3.6
    p = model["p_max_w"] if model["mode"] == "free" else allowed_wheel_power_w(v * 3.6)
    ax.plot(v * 3.6, (p - model["k"] * v ** 3) / 1000, color="tab:red",
            label=f"{model['mode']} fit: P_max={model['p_max_w']/1000:.0f} kW, CdA={model['cda_m2']:.2f} m²")
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Power going into acceleration + rolling (kW)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sanity_warnings(driver, model, df=None, energy_mj=None):
    p_kw, cda = model["p_max_w"] / 1000, model["cda_m2"]
    warn = []
    if energy_mj is not None and energy_mj > MAX_LAP_ELECTRIC_MJ:
        warn.append(f"{energy_mj:.1f} MJ electric energy is more than a 2026 car can deploy "
                    f"(~{MAX_LAP_ELECTRIC_MJ} MJ max) -> drag probably overestimated")
    if df is not None:
        full = df["Throttle"] >= 98
        over = (df.loc[full, "power_kw"] > 1.15 * df.loc[full, "allowed_kw"]).mean()
        if over > 0.15:
            warn.append(f"{over:.0%} of full-throttle samples exceed the max power the rules "
                        f"allow -> drag probably overestimated")
    if model["mode"] == "free" and not 550 <= p_kw <= 900:
        warn.append(f"P_max {p_kw:.0f} kW looks off (expect roughly {ICE_NOMINAL_KW}+{MGU_K_MAX_KW} kW, minus losses)")
    if not 0.4 <= cda <= 1.6:
        warn.append(f"CdA {cda:.2f} m² looks off (expect roughly 0.6-1.2 m²)")
    if pd.isna(cda):
        warn.append("could not fit drag - no bin reached full power")
    if model["mode"] == "free" and model["r2"] < 0.7:
        warn.append(f"fit R² {model['r2']:.2f} is weak - check the fit plot")
    for w in warn:
        print(f"  WARNING [{driver}]: {w}")


def main():
    parser = argparse.ArgumentParser(description="Estimate power and classify clipping zones.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--event", default="Monza")
    parser.add_argument("--session", default="Q")
    parser.add_argument("--drivers", nargs="+", default=["PIA", "NOR"])
    parser.add_argument("--fit", choices=["topspeed", "envelope", "free"], default=FIT_MODE,
                        help="topspeed (default) = fit drag on the straights where the rules fix the power")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = load_session(args.year, args.event, args.session)
    tag = f"{args.year}_{args.event}_{args.session}".replace(" ", "")

    rows = []
    for driver in args.drivers:
        try:
            lap, tel = fastest_lap_telemetry(session, driver)
            df = label_energy_modes(tel)
            model = fit_car_model(df, args.fit)
        except ValueError as err:
            print(f"Skipping {driver}: {err}")
            continue

        df = add_power_columns(df, model)
        zones = classify_zones(df, find_clip_zones(df))
        energy = electric_energy_mj(df, model)

        title = f"{session.event['EventName']} {args.year} {args.session} - {driver} - {lap['LapTime'].total_seconds():.3f}s"
        plot_power(df, zones, title, OUT_DIR / f"{tag}_{driver}_power.png")
        plot_fit(model, f"{driver}: drag / power fit", OUT_DIR / f"{tag}_{driver}_fit.png")
        zones.to_csv(OUT_DIR / f"{tag}_{driver}_zones.csv", index=False)

        reasons = zones["reason"].value_counts() if not zones.empty else pd.Series(dtype=int)
        rows.append({
            "driver": driver,
            "lap_time_s": round(lap["LapTime"].total_seconds(), 3),
            "p_max_kw": round(model["p_max_w"] / 1000),
            "cda_m2": round(model["cda_m2"], 2),
            "fit": model["mode"],
            "electric_mj_full_throttle": round(energy, 2),
            "zones_super_clip": int(reasons.get("super clipping (harvesting at full throttle)", 0)),
            "zones_reduced_deploy": int(reasons.get("reduced deployment / battery low", 0)),
            "zones_reg_taper": int(reasons.get("regulation taper", 0)),
            "zones_above_limit": int(reasons.get("above modelled limit (tow or model error)", 0)),
        })

        print(f"\n=== {driver} ===")
        sanity_warnings(driver, model, df, energy)
        if not zones.empty:
            print(zones[["start_m", "end_m", "speed_lost_kmh", "power_kw",
                         "allowed_kw", "deficit_kw", "reason"]].to_string(index=False))

    if not rows:
        raise SystemExit("No drivers could be modelled.")
    summary = pd.DataFrame(rows).sort_values("lap_time_s")
    summary.to_csv(OUT_DIR / f"{tag}_power_summary.csv", index=False)
    pd.set_option("display.width", 180)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved plots + tables to: {OUT_DIR}")
    print("First check the *_fit.png plots: the blue dots should follow the red line.")


if __name__ == "__main__":
    main()
