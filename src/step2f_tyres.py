"""
F1 Energy Strategy - Step 2f: Tyre degradation by the ACTUAL Pirelli compound (C1-C5)
=====================================================================================
"SOFT" is not one tyre: Pirelli brings three of its five compounds (C1 hardest ... C5 softest) to each
race and calls them hard / medium / soft there. The soft at Suzuka (C3) is the HARD at Monaco. Step 2c
grouped laps by the name, which mixes different rubber. This step uses the real compound.

The compound table is in reference/pirelli_compounds_2026.csv (from Pirelli's announcements).

Method (race laps from Step 2c, cleaned with Step 2e: no rain, no restart laps, fuel rebuilt after SC):
  1. Fuel correction: lap time minus fuel effect x fuel load (fuel effect fitted per race in Step 2d).
     Within a stint fuel and tyre age change together lap by lap, so they can't be separated there -
     the fuel effect has to come from outside the stint.
  2. Clean air only: laps less than 1.5 s behind another car are left out (dirty air changes lap time).
  3. Degradation of every stint = slope of the corrected lap time vs tyre age (% of a lap per lap).
  4. Compare compounds WITHIN the same race (event effects), because hard compounds go to hard tracks
     (Suzuka, Silverstone) and soft ones to gentle tracks (Monaco) - comparing raw averages would mix
     the compound with the track.

Run (after Step 2c + 2e, and Step 2d once so the fuel fits exist):
    python src/step2f_tyres.py
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import step2d_development as s2d

ROOT = Path(__file__).resolve().parents[1]
SEASON_DIR = ROOT / "outputs" / "season"
DEV_DIR = ROOT / "outputs" / "development"
REF = ROOT / "reference" / "pirelli_compounds_2026.csv"

DRY = ["HARD", "MEDIUM", "SOFT"]
CLEAN_AIR_GAP_S = 1.5
MIN_STINT_LAPS = 8
MIN_STINT_SPAN = 6           # tyre age must change by at least this many laps within the stint
FUEL_DEFAULT_S_PER_KG = 0.035
ENERGY = ["clipping_%", "lift_coast_%", "electric_mj_mid"]


def compound_table():
    ref = pd.read_csv(REF)
    return ref


def actual_compound(event, name, ref):
    ev = str(event).lower()
    for _, row in ref.iterrows():
        if row["event_key"] in ev:
            return row[str(name).lower()] if str(name).lower() in ("hard", "medium", "soft") else None
    return None


def load_laps(verbose=True):
    ref = compound_table()
    rcond = s2d.conditions("R")
    fuel_fits = DEV_DIR / "race_pace_fuel_fits.csv"
    fuel = pd.read_csv(fuel_fits).set_index("round")["fuel_s_per_kg"] if fuel_fits.exists() else pd.Series(dtype=float)
    if fuel.empty and verbose:
        print(f"No Step 2d fuel fits found - using {FUEL_DEFAULT_S_PER_KG} s/kg everywhere (run Step 2d once).")
    parts, missing = [], set()
    for f in sorted(SEASON_DIR.glob("[0-9][0-9]_R_laps.csv")):
        r = pd.read_csv(f)
        if r.empty:
            continue
        r = s2d.clean_race_laps(r, rcond, verbose=False)
        r = r[r["compound"].astype(str).str.upper().isin(DRY)].dropna(subset=["lap_time_s", "tyre_life", "fuel_kg"])
        if r.empty:
            continue
        rnd = int(r["round"].iloc[0])
        event = r["event"].iloc[0]
        r["C"] = [actual_compound(event, c, ref) for c in r["compound"].str.upper()]
        if r["C"].isna().all():
            missing.add(event)
            continue
        med = r["lap_time_s"].median()
        fe = float(fuel.get(rnd, FUEL_DEFAULT_S_PER_KG * med / 90))
        r["lap_corr_s"] = r["lap_time_s"] - fe * r["fuel_kg"]
        r["lap_pct"] = r["lap_corr_s"] / med * 100
        r["median_lap_s"] = med
        if "gap_ahead_start_s" in r:
            gap = r[["gap_ahead_start_s", "gap_ahead_end_s"]].mean(axis=1)
            r = r[gap.isna() | (gap >= CLEAN_AIR_GAP_S)]
        # mistakes / traffic not caught by the gap: more than 2% slower than the stint's median
        r = r[r["lap_corr_s"] <= r.groupby(["driver", "stint"])["lap_corr_s"].transform("median") * 1.02]
        parts.append(r)
    if missing and verbose:
        print(f"No compound info for: {', '.join(sorted(missing))} - add them to {REF.name}")
    return pd.concat(parts, ignore_index=True)


def stint_slopes(laps, max_age=None, min_laps=MIN_STINT_LAPS, min_span=MIN_STINT_SPAN):
    rows = []
    if max_age:
        laps = laps[laps["tyre_life"] <= max_age]
    for (rnd, drv, st), g in laps.groupby(["round", "driver", "stint"]):
        if len(g) < min_laps or g["tyre_life"].max() - g["tyre_life"].min() < min_span:
            continue
        x = g["tyre_life"].to_numpy(float)
        row = {"round": rnd, "event": g["event"].iloc[0], "driver": drv, "team": g["team"].iloc[0], "stint": st,
               "name": g["compound"].iloc[0], "C": g["C"].iloc[0], "laps": len(g),
               "track_temp": g["track_temp"].mean() if "track_temp" in g else np.nan,
               "median_lap_s": g["median_lap_s"].iloc[0]}
        row["deg_pct_per_lap"] = np.polyfit(x, g["lap_pct"].to_numpy(float), 1)[0]
        row["deg_s_per_lap"] = row["deg_pct_per_lap"] / 100 * row["median_lap_s"]
        for m in ENERGY:
            if m in g and g[m].notna().sum() >= MIN_STINT_LAPS:
                row[f"{m}_per_lap"] = np.polyfit(x[g[m].notna()], g[m].dropna().to_numpy(float), 1)[0]
        rows.append(row)
    return pd.DataFrame(rows)


def within_event(st, metric, ref_c="C3", by="C"):
    """Weighted regression: stint slope = event effect + compound effect. Compound effects are relative
    to ref_c, compared only within the same race. SEs clustered by event.
    by="name" compares the race's own labels (HARD/MEDIUM/SOFT) instead of C1-C5."""
    d = st.dropna(subset=[metric]).copy()
    d["C"] = d[by]
    comps = sorted(d["C"].unique())
    if ref_c not in comps:
        ref_c = comps[len(comps) // 2]
    cdum = pd.get_dummies(d["C"], dtype=float).drop(columns=ref_c)
    edum = pd.get_dummies(d["round"], dtype=float)
    X = np.column_stack([cdum.to_numpy(), edum.to_numpy()])
    y = d[metric].to_numpy(float)
    w = np.sqrt(d["laps"].to_numpy(float))
    Xw, yw = X * w[:, None], y * w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    u = yw - Xw @ beta
    xtx = np.linalg.pinv(Xw.T @ Xw)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in d["round"].unique():
        idx = (d["round"] == g).to_numpy()
        sc = Xw[idx].T @ u[idx]
        meat += np.outer(sc, sc)
    G = d["round"].nunique()
    V = xtx @ meat @ xtx * G / max(G - 1, 1)
    out = []
    for i, c in enumerate(cdum.columns):
        se = float(np.sqrt(max(V[i, i], 0)))
        n_ev = d.loc[d["C"] == c, "round"].nunique()
        out.append({"compound": c, "vs": ref_c, "extra_deg": beta[i], "se": se,
                    "t": beta[i] / se if se > 0 else np.nan, "races": n_ev, "stints": int((d["C"] == c).sum())})
    return pd.DataFrame(out)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Tyre degradation by actual Pirelli compound.")
    ap.add_argument("--max-age", type=int, default=None,
                    help="only use tyre ages up to N laps, so every compound is compared over the same window")
    ap.add_argument("--min-laps", type=int, default=MIN_STINT_LAPS)
    args = ap.parse_args()
    laps = load_laps()
    st = stint_slopes(laps, args.max_age, args.min_laps, min(MIN_STINT_SPAN, args.min_laps - 1))
    if args.max_age:
        print(f"Robustness run: tyre ages 1-{args.max_age} only, stints of {args.min_laps}+ laps")
    st.to_csv(SEASON_DIR / "stint_degradation.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(st)} clean-air stints of {args.min_laps}+ laps from {st['round'].nunique()} races")

    print("\n" + "=" * 90 + "\nA. WHAT 'SOFT' / 'MEDIUM' / 'HARD' ACTUALLY WERE (stints per name x compound)\n" + "=" * 90)
    print(pd.crosstab(st["name"], st["C"]).reindex(DRY).to_string())
    print("Same name, different rubber: e.g. a SOFT can be anything from C3 to C5 depending on the race.")

    # selection check: short stints are dropped - if soft tyres that wear fast get pitted early, only
    # the soft stints that held up survive, and softs look better than they are
    allst = laps.groupby(["round", "driver", "stint"]).agg(C=("C", "first"), age=("tyre_life", "max"))
    kept = st.set_index(["round", "driver", "stint"]).index
    allst["kept"] = allst.index.isin(kept)
    sel = allst.groupby("C").agg(stints=("kept", "size"), kept_share=("kept", "mean"),
                                 median_stint_laps=("age", "median"))
    print("\nSelection check - share of each compound's stints long enough to be used:")
    print(sel.round(2).to_string())
    print("If softer compounds have a much lower kept share, their degradation is underestimated (survivors).")

    print("\n" + "=" * 90 + "\nB. DEGRADATION PER LAP BY ACTUAL COMPOUND (fuel-corrected, clean air)\n" + "=" * 90)
    raw = st.groupby("C").agg(races=("round", "nunique"), stints=("laps", "size"),
                              median_s_per_lap=("deg_s_per_lap", "median"),
                              median_pct_per_lap=("deg_pct_per_lap", "median"))
    print("Raw averages (mixes compound with track - hard compounds go to hard tracks):")
    print(raw.round(4).to_string())
    for metric, unit in [("deg_pct_per_lap", "% of a lap per lap"), ("deg_s_per_lap", "s per lap")]:
        we = within_event(st, metric)
        print(f"\nWithin the same race, extra degradation vs {we['vs'].iloc[0]} ({unit}):")
        print(we.round(4).to_string(index=False))
    print("Positive = degrades faster than the reference compound at the same race. |t| > 2 = clear.")
    wr = within_event(st, "deg_s_per_lap", ref_c="HARD", by="name")
    print("\nSame question by the race's own labels - does that race's SOFT wear faster than its HARD? (s per lap):")
    print(wr.round(4).to_string(index=False))
    # C3 plays a different role depending on the race: the hard (C3-C5 races), medium (C2-C4) or soft (C1-C3)
    c3 = st[st["C"] == "C3"].groupby("name")["deg_s_per_lap"].agg(["size", "median"]).reindex(DRY)
    print("\nC3 by the role it played at that race (median s per lap):")
    print(c3.round(4).to_string())

    if "track_temp" in st and st["track_temp"].notna().sum() > 30:
        print("\n" + "=" * 90 + "\nC. HEAT: does a hotter track speed up degradation, compound by compound?\n" + "=" * 90)
        d = st.dropna(subset=["track_temp"])
        rows = []
        for c, g in d.groupby("C"):
            if len(g) >= 15 and g["track_temp"].std() > 2:
                b = np.polyfit(g["track_temp"], g["deg_pct_per_lap"], 1)[0]
                rows.append({"compound": c, "stints": len(g), "temp_range_C": f"{g['track_temp'].min():.0f}-{g['track_temp'].max():.0f}",
                             "extra_deg_pct_per_lap_per_10C": b * 10})
        if rows:
            print(pd.DataFrame(rows).round(4).to_string(index=False))
            print("Across races, so track layout is mixed in - read as a pattern, not a precise number.")

    cols = [f"{m}_per_lap" for m in ENERGY if f"{m}_per_lap" in st]
    if cols:
        print("\n" + "=" * 90 + "\nD. ENERGY USE AS THE TYRE AGES, BY COMPOUND (median change per lap of tyre age)\n" + "=" * 90)
        print(st.groupby("C")[cols].median().round(4).to_string())
        for c in cols:
            we = within_event(st, c)
            sig = we[we["t"].abs() > 2]
            if len(sig):
                print(f"  {c}: clear within-race differences vs {we['vs'].iloc[0]}: "
                      + ", ".join(f"{r.compound} {r.extra_deg:+.4f} (t {r.t:.1f})" for r in sig.itertuples()))

    fig, ax = plt.subplots(figsize=(9, 5))
    order = sorted(st["C"].unique())
    data = [st.loc[st["C"] == c, "deg_s_per_lap"] for c in order]
    try:
        ax.boxplot(data, tick_labels=order, showfliers=False)       # matplotlib >= 3.9
    except TypeError:
        ax.boxplot(data, labels=order, showfliers=False)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("degradation (s per lap of tyre age, fuel-corrected)")
    ax.set_title("2026 tyre degradation by Pirelli compound (clean-air race stints)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SEASON_DIR / "degradation_by_compound.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {SEASON_DIR / 'stint_degradation.csv'} and degradation_by_compound.png")


if __name__ == "__main__":
    main()
