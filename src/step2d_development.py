"""
F1 Energy Strategy - Step 2d: Season development + the engine-rule natural experiment
======================================================================================
Uses the season data saved by Step 2c (no new telemetry downloads).

1. DEVELOPMENT: how each team's qualifying pace (vs the median team), race pace, effective drag and energy habits
   changed from the first half of the season to the second, plus a trend over the rounds.

2. NATURAL EXPERIMENT (difference-in-differences): from 1 June 2026 the FIA also measured engine
   compression ratio HOT, closing the loophole Mercedes was believed to use. Mercedes supplies
   Mercedes, McLaren, Williams and Alpine. Did those four teams lose ground after 1 June,
   relative to teams with other engines, at the same tracks?
     - each team is compared with the other teams AT THE SAME TRACK (track effects cancel)
     - team effects cancel too (two-way fixed effects)
     - checks: pre-trend (were the groups already drifting apart?) and placebo tests
       (fake cut-off dates, other engine groups) - does the real effect stand out?

Note: "effective drag" assumes the same 400 kW engine for everyone, so a WEAKER engine shows up as
MORE effective drag. If Mercedes engines lost power on 1 June, their effective drag should RISE.

Run:
    python src/step2d_development.py --cache-dir C:\\f1cache          # dates from the FastF1 schedule
    python src/step2d_development.py --cutoff-round 6                 # or: first round under the new rule
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / "data" / "cache"      # FastF1 is only needed to read the schedule (dates)

SEASON_DIR = OUT_DIR / "season"
DEV_DIR = OUT_DIR / "development"
RULE_DATE = pd.Timestamp("2026-06-01")
TREATED_PU = "Mercedes"

PU_SUPPLIERS = {
    "Mercedes": ["mercedes", "mclaren", "williams", "alpine"],
    "Ferrari": ["ferrari", "haas", "cadillac"],
    "Red Bull Ford": ["red bull", "racing bulls", "visa"],
    "Honda": ["aston"],
    "Audi": ["audi", "sauber", "kick"],
}
METRICS = {
    "pace_gap_%": "qualifying pace vs median team (%) - higher = slower",
    "race_pace_%": "race pace vs median team (%), fuel + tyre corrected - higher = slower",
    "cda_hi": "effective drag (CdA, upper bound) - higher = more drag OR weaker engine",
    "electric_mj_mid": "electrical energy deployed per quali lap (MJ)",
    "super_clip_robust": "super-clipping zones per quali lap",
    "lift_coast_%": "lift-and-coast (% of lap)",
}


def power_unit(team):
    t = str(team).lower()
    for pu, keys in PU_SUPPLIERS.items():
        if any(k in t for k in keys):
            return pu
    return "unknown"


# ----------------------------------------------------------------------------- data
def load_panel():
    files = lambda t: sorted(SEASON_DIR.glob(f"[0-9][0-9]_{t}.csv"))
    if not files("Q_laps"):
        raise SystemExit("No Step 2c data found in outputs/season - run step2c_season.py first.")
    q = pd.concat([pd.read_csv(f) for f in files("Q_laps")], ignore_index=True)
    drag = pd.concat([pd.read_csv(f) for f in files("drag")], ignore_index=True)

    # Step 2e conditions (if saved): drop wet laps, and drag from wet qualifying sessions
    cond = conditions("Q")
    if not cond.empty:
        keep = ["round", "driver", "lap_number", "wet"] + (["deleted"] if "deleted" in cond else [])
        q = q.merge(cond[keep], on=["round", "driver", "lap_number"], how="left")
        share = q.groupby("round")["wet"].mean()
        wet_rounds = sorted(share[share > 0.3].index)
        n_wet = int(q["wet"].eq(True).sum())
        n_del = int(q["deleted"].eq(True).sum()) if "deleted" in q else 0
        # deleted laps (track limits) don't count in qualifying, so they can't be anyone's best lap
        q = q[q["wet"].ne(True) & (q["deleted"].ne(True) if "deleted" in q else True)]
        q = q.drop(columns=[c for c in ("wet", "deleted") if c in q])
        drag = drag[~drag["round"].isin(wet_rounds)]
        print(f"Conditions (Step 2e): dropped {n_wet} wet and {n_del} deleted quali laps; "
              f"drag ignored for wet qualifying in rounds {wet_rounds if wet_rounds else 'none'}")

    # Pace vs the MEDIAN team at each round (not pole: if the fastest team slows down, every other
    # team's gap to pole shrinks and they would all look like they improved)
    best = q.groupby(["round", "event", "team"])["lap_time_s"].min().reset_index()
    med = best.groupby("round")["lap_time_s"].median().rename("median_team")
    best = best.join(med, on="round")
    best["pace_gap_%"] = (best["lap_time_s"] / best["median_team"] - 1) * 100
    habits = q.groupby(["round", "team"])[["electric_mj_mid", "super_clip_robust", "lift_coast_%"]].mean()
    dr = drag.groupby(["round", "team"])[["cda_hi", "cda_lo"]].mean()
    panel = best.set_index(["round", "team"]).join(habits).join(dr)
    rp = race_pace(verbose=True)
    if rp.empty:
        print("No race laps found (NN_R_laps.csv) - race pace is skipped.")
        panel["race_pace_%"] = np.nan
    else:
        # outer join: a wet qualifying (no quali pace) can still have a dry race, and vice versa
        panel = panel.join(rp.set_index(["round", "team"]), how="outer")
    panel = panel.reset_index()
    panel["event"] = panel.groupby("round")["event"].transform(lambda e: e.ffill().bfill())
    panel["power_unit"] = panel["team"].map(power_unit)
    return panel


DRY = {"SOFT", "MEDIUM", "HARD"}
MIN_TEAM_LAPS = 15                       # a team needs this many clean laps in a race to get a race pace
FUEL_S_PER_KG_RANGE = (0.015, 0.06)      # plausible fuel effect per kg for a ~90 s lap (scaled by lap length)
FUEL_S_PER_KG_DEFAULT = 0.03


def conditions(kind):
    """Per-lap conditions saved by Step 2e (empty if Step 2e hasn't been run)."""
    files = sorted(SEASON_DIR.glob(f"[0-9][0-9]_{kind}_conditions.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f).assign(round=int(f.name[:2])) for f in files], ignore_index=True)


GAP_BANDS = [0.0, 1.0, 2.0, 1e9]   # seconds behind the car ahead; the last band (> 2 s) is "clean air"
GAP_NAMES = ["within_1s", "within_1_2s"]
NEUTRAL = ("4", "5", "6", "7")    # track status codes: safety car, red flag, VSC, VSC ending
RESTART_LAPS = 2                  # green laps right after a neutralisation that are dropped
SC_FUEL_SHARE = 0.5               # fuel burned on a neutralised lap, relative to a green lap (rough)


def neutral_lap_numbers(rc_round):
    st = rc_round["track_status"].astype(str)
    hit = st.apply(lambda x: any(c in x for c in NEUTRAL))
    share = hit.groupby(rc_round["lap_number"]).mean()
    return set(int(l) for l in share[share >= 0.25].index)


def race_pace(verbose=False):
    """Race pace per team and race, corrected for fuel load and tyre age.

    For each race, on clean green-flag dry laps (Step 2c already dropped pit laps, SC/VSC laps and
    laps slower than 107%):
        lap_time = race constant + fuel effect * fuel_kg
                   + compound offset + compound-specific degradation * tyre_life
                   + dirty air (within 1 s / 1-2 s of the car ahead, if Step 2e has the gaps)
                   + team effect
    Fuel and tyre age can be separated because teams pit at different laps. If the fuel effect comes
    out implausible, a standard value is used instead. The team effect is converted to % of the
    race's median lap and measured against the median team (same idea as qualifying pace).
    """
    files = sorted(SEASON_DIR.glob("[0-9][0-9]_R_laps.csv"))
    rows, fuel_log = [], []
    rcond = conditions("R")
    for f in files:
        r = pd.read_csv(f)
        if r.empty or "tyre_life" not in r:
            continue
        if not rcond.empty and "track_status" in rcond:
            # Step 2c already dropped the neutralised laps themselves. Two effects remain:
            #  1) restart laps: cold tyres, cars bunched up, and a battery recharged to full under SC
            #  2) fuel: cars burn much less under SC/VSC, so later laps carry more fuel than a straight
            #     line from 70 kg would say -> rebuild the fuel load lap by lap
            g = rcond[rcond["round"] == r["round"].iloc[0]]
            neutral = neutral_lap_numbers(g)
            if neutral:
                total = int(g["lap_number"].max())
                w = np.array([SC_FUEL_SHARE if l in neutral else 1.0 for l in range(1, total + 1)])
                burned = np.concatenate([[0.0], np.cumsum(w)])          # fuel units burned before lap n
                fuel = 70 - 68 * burned[np.clip(r["lap_number"].to_numpy(int) - 1, 0, total)] / w.sum()
                r["fuel_kg"] = np.round(fuel, 1)
                last = np.full(len(r), -10**6)
                for l in sorted(neutral):
                    last = np.where(r["lap_number"].to_numpy() > l, l, last)
                restart = (r["lap_number"].to_numpy() - last) <= RESTART_LAPS
                if verbose:
                    print(f"  round {int(r['round'].iloc[0])}: {len(neutral)} SC/VSC/red laps -> fuel rebuilt, "
                          f"{int(restart.sum())} restart laps dropped")
                r = r[~restart]
        if not rcond.empty:      # drop laps driven in rain (dry tyres on a damp track are not race pace)
            r = r.merge(rcond[["round", "driver", "lap_number", "rain"]], on=["round", "driver", "lap_number"], how="left")
            n_rain = int(r["rain"].eq(True).sum())
            r = r[r["rain"].ne(True)].drop(columns="rain")
            if verbose and n_rain:
                print(f"  round {int(r['round'].iloc[0]) if len(r) else '?'}: dropped {n_rain} race laps driven in rain")
            if "gap_ahead_start_s" in rcond:
                r = r.merge(rcond[["round", "driver", "lap_number", "gap_ahead_start_s", "gap_ahead_end_s"]],
                            on=["round", "driver", "lap_number"], how="left")
        r = r[r["compound"].astype(str).str.upper().isin(DRY)].dropna(subset=["lap_time_s", "tyre_life", "fuel_kg"])
        counts = r.groupby("team")["lap_time_s"].transform("size")
        r = r[counts >= MIN_TEAM_LAPS]
        if r["team"].nunique() < 4:
            continue
        rnd = int(r["round"].iloc[0])
        med_lap = r["lap_time_s"].median()
        # drop the slowest laps of each driver's stint (traffic, mistakes) - 2% above the stint's median
        stint_med = r.groupby(["driver", "stint"])["lap_time_s"].transform("median")
        r = r[r["lap_time_s"] <= stint_med * 1.02]

        comp = r["compound"].str.upper()
        teams = sorted(r["team"].unique())
        cols = {"const": np.ones(len(r)), "fuel": r["fuel_kg"].to_numpy(float)}
        comps = sorted(comp.unique())
        for c in comps:
            on = (comp == c).to_numpy(float)
            if c != comps[0]:
                cols[f"is_{c}"] = on
            cols[f"deg_{c}"] = on * r["tyre_life"].to_numpy(float)
        has_gap = "gap_ahead_start_s" in r and r["gap_ahead_start_s"].notna().mean() > 0.5
        if has_gap:
            # dirty air: average gap to the car ahead over the lap (start and end of the lap)
            gap = r[["gap_ahead_start_s", "gap_ahead_end_s"]].mean(axis=1).fillna(GAP_BANDS[-1]).to_numpy()
            for (a, b), name in zip(zip(GAP_BANDS[:-1], GAP_BANDS[1:]), GAP_NAMES):
                cols[name] = ((gap >= a) & (gap < b)).astype(float)
        for t in teams[1:]:
            cols[f"team_{t}"] = (r["team"] == t).to_numpy(float)
        X = np.column_stack(list(cols.values()))
        names = list(cols)
        y = r["lap_time_s"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        fuel = beta[names.index("fuel")]
        lo, hi = (v * med_lap / 90 for v in FUEL_S_PER_KG_RANGE)
        fixed = not (lo <= fuel <= hi)
        if fixed:      # implausible -> fix the fuel effect and refit the rest
            fuel = FUEL_S_PER_KG_DEFAULT * med_lap / 90
            keep = [i for i, n in enumerate(names) if n != "fuel"]
            beta2, *_ = np.linalg.lstsq(X[:, keep], y - fuel * cols["fuel"], rcond=None)
            beta = np.insert(beta2, names.index("fuel"), fuel)
        entry = {"round": rnd, "fuel_s_per_kg": round(float(fuel), 4), "fixed": fixed,
                 "laps": len(r), "median_lap_s": round(med_lap, 2)}
        if has_gap:
            for name in GAP_NAMES:
                entry[f"{name}_s"] = round(float(beta[names.index(name)]), 3)
                entry[f"{name}_laps"] = int(cols[name].sum())
        fuel_log.append(entry)
        eff = {teams[0]: 0.0}
        eff.update({t: beta[names.index(f"team_{t}")] for t in teams[1:]})
        eff = pd.Series(eff)
        pct = (eff - eff.median()) / med_lap * 100
        for t, v in pct.items():
            rows.append({"round": rnd, "team": t, "race_pace_%": float(v)})
    if verbose and fuel_log:
        fl = pd.DataFrame(fuel_log)
        print(f"Race pace: {len(fl)} races; fuel effect {fl['fuel_s_per_kg'].median():.3f} s/kg (median), "
              f"standard value used in {int(fl['fixed'].sum())} races")
        DEV_DIR.mkdir(parents=True, exist_ok=True)
        fl.to_csv(DEV_DIR / "race_pace_fuel_fits.csv", index=False)
        if f"{GAP_NAMES[0]}_s" in fl:
            print("Dirty air - lap time lost vs clean air (more than 2 s behind), per race:")
            show = fl[["round"] + [c for n in GAP_NAMES for c in (f"{n}_s", f"{n}_laps")]]
            print(show.to_string(index=False))
            for n in GAP_NAMES:
                v = fl[f"{n}_s"].dropna()
                print(f"  {n}: median {v.median():+.3f} s/lap over {len(v)} races "
                      f"({(v > 0).mean():.0%} of races slower than clean air)")
            print("  Note: in 2026 a car within 1 s also gets overtake mode (extra electrical energy), so this is"
                  " the NET effect: aero loss in corners minus the extra boost on the straights.")
    return pd.DataFrame(rows, columns=["round", "team", "race_pace_%"])


def round_dates(year, cache_dir):
    import fastf1
    path = Path(cache_dir or os.environ.get("F1_CACHE_DIR") or CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    try:
        fastf1.Cache.configure(cache_dir=str(path))
    except (AttributeError, TypeError):
        fastf1.Cache.enable_cache(str(path))
    sched = fastf1.get_event_schedule(year, include_testing=False)
    col = "Session5DateUtc" if "Session5DateUtc" in sched else "EventDate"
    d = pd.to_datetime(sched[col], errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return dict(zip(sched["RoundNumber"].astype(int), d))


# ----------------------------------------------------------------------------- 1. development
def development(panel):
    print("\n" + "=" * 92 + "\n1. CAR DEVELOPMENT: first half vs second half of the season (qualifying)\n" + "=" * 92)
    mid = panel["round"].median()
    panel = panel.assign(half=np.where(panel["round"] <= mid, "1st half", "2nd half"))
    rows = []
    for team, d in panel.groupby("team"):
        row = {"team": team, "power_unit": d["power_unit"].iloc[0]}
        for m in ["pace_gap_%", "race_pace_%", "cda_hi", "electric_mj_mid", "super_clip_robust", "lift_coast_%"]:
            h = d.groupby("half")[m].mean()
            row[f"{m} 1st"] = h.get("1st half", np.nan)
            row[f"{m} 2nd"] = h.get("2nd half", np.nan)
        # trend in pace over the season: % of a lap per 10 rounds (negative = getting closer to pole)
        x, y = d["round"].to_numpy(float), d["pace_gap_%"].to_numpy(float)
        if len(d) >= 5:
            b = np.polyfit(x, y, 1)[0]
            resid = y - np.polyval(np.polyfit(x, y, 1), x)
            se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
            row["pace_trend_per_10_rounds"], row["trend_t"] = b * 10, b / se if se > 0 else np.nan
        rows.append(row)
    t = pd.DataFrame(rows)
    t["pace change"] = t["pace_gap_% 2nd"] - t["pace_gap_% 1st"]
    t["race pace change"] = t["race_pace_% 2nd"] - t["race_pace_% 1st"]
    t = t.sort_values("pace change")
    t.to_csv(DEV_DIR / "team_development.csv", index=False)

    pd.set_option("display.width", 220)
    show = t[["team", "power_unit", "pace_gap_% 1st", "pace_gap_% 2nd", "pace change",
              "pace_trend_per_10_rounds", "trend_t", "race_pace_% 1st", "race_pace_% 2nd", "race pace change",
              "cda_hi 1st", "cda_hi 2nd",
              "lift_coast_% 1st", "lift_coast_% 2nd"]]
    print(f"(split after round {mid:.0f}; pace = best qualifying lap vs the median team, in % of a lap;"
          f" negative = faster than the median; negative change = improved)")
    print(show.round(3).to_string(index=False))
    print("trend_t: |t| > 2 = a clear trend over the season (not just one or two good weekends).")

    fig, ax = plt.subplots(figsize=(13, 6))
    for team, d in panel.sort_values("round").groupby("team"):
        sm = d["pace_gap_%"].rolling(3, min_periods=1, center=True).mean()
        ax.plot(d["round"], sm, marker="o", ms=3, label=team)
    ax.set_xlabel("Round")
    ax.set_ylabel("Qualifying pace vs median team (%)\n3-round rolling average (up = faster)")
    ax.set_title("2026 qualifying pace over the season")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(DEV_DIR / "pace_over_season.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- 2. DiD
def did_estimate(panel, metric, treated_mask, post_mask, trends=False):
    """Two-way fixed effects: metric = team effect + round effect + beta * (treated x post).
    trends=True also gives every team its own linear development trend over the season, so a rival
    steadily improving (or getting worse) can't be mistaken for the rule's effect."""
    d = panel.dropna(subset=[metric]).copy()
    tr = treated_mask.loc[d.index].astype(float).to_numpy()
    po = post_mask.loc[d.index].astype(float).to_numpy()
    teams = pd.get_dummies(d["team"], drop_first=True, dtype=float)
    rounds = pd.get_dummies(d["round"], drop_first=True, dtype=float)
    cols = [tr * po, np.ones(len(d)), teams.to_numpy(), rounds.to_numpy()]
    if trends:
        rnd = d["round"].to_numpy(float)
        rnd = (rnd - rnd.mean()) / rnd.std()
        cols.append(pd.get_dummies(d["team"], dtype=float).to_numpy()[:, 1:] * rnd[:, None])
    X = np.column_stack(cols)
    y = d[metric].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    # cluster-robust SE by team (laps of a team across rounds are not independent)
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in d["team"].unique():
        idx = (d["team"] == g).to_numpy()
        s = X[idx].T @ u[idx]
        meat += np.outer(s, s)
    G, n, k = d["team"].nunique(), len(y), X.shape[1]
    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    se = float(np.sqrt(max((corr * xtx_inv @ meat @ xtx_inv)[0, 0], 0)))
    return float(beta[0]), se


def natural_experiment(panel, cutoff_round, treated_teams=None, label="1 June compression rule"):
    from itertools import combinations
    panel = panel.copy()
    if treated_teams:
        keys = [t.lower() for t in treated_teams]
        treated = panel["team"].str.lower().apply(lambda t: any(k in t for k in keys))
        title = f"{label}: {', '.join(treated_teams)} vs the other teams"
    else:
        treated = panel["power_unit"].eq(TREATED_PU)
        title = "Mercedes-engined teams before vs after the 1 June compression rule"
    print("\n" + "=" * 92 + f"\n2. NATURAL EXPERIMENT: {title}\n" + "=" * 92)
    post = panel["round"] >= cutoff_round
    tteams = sorted(panel.loc[treated, "team"].unique())
    if not tteams:
        print("No teams matched --treated.")
        return
    print(f"Treated: {', '.join(tteams)}")
    print(f"Before the change: rounds {int(panel['round'].min())}-{cutoff_round - 1}; "
          f"after: rounds {cutoff_round}-{int(panel['round'].max())}")
    if (panel["round"] < cutoff_round).sum() == 0 or panel.loc[panel["round"] < cutoff_round, "round"].nunique() < 2 \
            or post.sum() == 0:
        print("Not enough rounds on both sides of the cut-off.")
        return

    rows = []
    for m, desc in METRICS.items():
        b, se = did_estimate(panel, m, treated, post)
        bt, set_ = did_estimate(panel, m, treated, post, trends=True)
        # Placebo 1: the same group with FAKE cut-off dates (only rounds before the real one,
        # so the real rule change can't leak in)
        fakes, fakes_t = [], []
        first = int(panel["round"].min())
        for c in range(first + 2, cutoff_round - 1):
            pre = panel[panel["round"] < cutoff_round]
            fb, _ = did_estimate(pre, m, treated.loc[pre.index], pre["round"] >= c)
            fakes.append(fb)
            fbt, _ = did_estimate(pre, m, treated.loc[pre.index], pre["round"] >= c, trends=True)
            fakes_t.append(fbt)
        # Placebo 2: OTHER groups at the REAL date - other engine groups, or (for a team list)
        # every same-size group of OTHER teams
        others, others_t = [], []
        if treated_teams:
            controls = sorted(set(panel["team"]) - set(tteams))
            for combo in list(combinations(controls, len(tteams)))[:60]:
                ob, _ = did_estimate(panel, m, panel["team"].isin(combo), post)
                others.append(ob)
                obt, _ = did_estimate(panel, m, panel["team"].isin(combo), post, trends=True)
                others_t.append(obt)
        else:
            for pu in panel["power_unit"].unique():
                if pu in (TREATED_PU, "unknown") or panel["power_unit"].eq(pu).sum() < 4:
                    continue
                ob, _ = did_estimate(panel, m, panel["power_unit"].eq(pu), post)
                others.append(ob)
                obt, _ = did_estimate(panel, m, panel["power_unit"].eq(pu), post, trends=True)
                others_t.append(obt)
        placebo = np.array(fakes + others)
        rank = float(np.mean(np.abs(placebo) >= abs(b))) if len(placebo) and se > 1e-9 else np.nan
        # the team-trends number needs its OWN placebo test: fitting a trend from few rounds can make
        # large numbers appear for any team, so compare it with the same model run on placebos
        placebo_t = np.array([v for v in fakes_t + others_t if np.isfinite(v)])
        rank_t = float(np.mean(np.abs(placebo_t) >= abs(bt))) if len(placebo_t) and set_ > 1e-9 else np.nan
        rows.append({"metric": m, "effect_after_rule": b, "se": se, "t": b / se if se > 1e-9 else np.nan,
                     "with_team_trends": bt, "t_trends": bt / set_ if set_ > 1e-9 else np.nan,
                     "placebo_runs": len(placebo), "share_placebos_as_big": rank,
                     "share_placebos_as_big_trends": rank_t, "meaning": desc})
    t = pd.DataFrame(rows)
    tag = "compression_rule" if not treated_teams else "_".join(k.replace(" ", "") for k in treated_teams) + f"_from_round{cutoff_round}"
    t.to_csv(DEV_DIR / f"{tag}_did.csv", index=False)
    print(t.round(4).to_string(index=False))
    print(f"\neffect_after_rule = change for the treated teams from round {cutoff_round}, relative to other teams at the same")
    print("tracks. share_placebos_as_big = how often a FAKE date or OTHER group of teams gives an effect at least")
    print("as big. with_team_trends = same, but every team also gets its own development trend over the season.")
    print("share_placebos_as_big_trends = the same placebo test for the team-trends number.")
    print("Believe an effect only if |t| > 2 AND few placebos are as big (< 0.2) - in at least one version,")
    print("and the other version points the same way. A big team-trends number with a high placebo share")
    print("is the trend extrapolation talking, not the upgrade.")
    print("Only 11 teams -> standard errors are rough; the placebo comparison is the more honest test.")

    # Event-study style plot: treated minus control, relative to each track's average
    if treated_teams:
        panel["rel_drag"] = panel["cda_hi"] - panel.groupby("round")["cda_hi"].transform("mean")
        gd = panel.groupby(["round", treated.rename("treated")])["rel_drag"].mean().unstack()
        dd = gd[True] - gd[False]
        print("\nEffective drag (CdA), treated minus others, by round (relative to the pre-change average):")
        print((dd - dd[dd.index < cutoff_round].mean()).round(3).to_string())
    panel["rel"] = panel["pace_gap_%"] - panel.groupby("round")["pace_gap_%"].transform("mean")
    g = panel.groupby(["round", treated.rename("treated")])["rel"].mean().unstack()
    diff = g[True] - g[False]
    pre_mean = diff[diff.index < cutoff_round].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(diff.index, diff - pre_mean, marker="o")
    ax.axvline(cutoff_round - 0.5, color="red", linestyle="--", label=label)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Round")
    who = "Mercedes-engined" if not treated_teams else " + ".join(tteams)
    ax.set_ylabel(f"{who} minus other teams\n(qualifying pace vs median, %, vs pre-change average)")
    ax.set_title(f"{who}: pace relative to the field around '{label}' (up = slower)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DEV_DIR / f"{tag}_event_study.png", dpi=150)
    plt.close(fig)

    if panel["race_pace_%"].notna().any():
        panel["rel_race"] = panel["race_pace_%"] - panel.groupby("round")["race_pace_%"].transform("mean")
        gr = panel.groupby(["round", treated.rename("treated")])["rel_race"].mean().unstack()
        if True in gr and False in gr:
            race_diff = (gr[True] - gr[False]).dropna()
            print("\nRace pace (%), treated minus others, by round (relative to the pre-change average; - = faster):")
            print((race_diff - race_diff[race_diff.index < cutoff_round].mean()).round(3).to_string())
        else:
            race_diff = pd.Series(dtype=float)
    else:
        race_diff = pd.Series(dtype=float)

    print()
    for name, series in [("qualifying pace", diff), ("race pace", race_diff)]:
        pre = series[series.index < cutoff_round].dropna()
        if len(pre) >= 3:
            slope = np.polyfit(pre.index.to_numpy(float), pre.to_numpy(float), 1)[0]
            print(f"Pre-trend check ({name}): before the change the gap between the groups drifted by "
                  f"{slope * 10:+.3f}% per 10 rounds. Near 0 = moving in parallel (needed for a fair comparison).")


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Season development + compression-rule natural experiment.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cutoff-round", type=int, default=None,
                        help="first round under the new rule (default: from the FastF1 schedule, 1 June)")
    parser.add_argument("--treated", nargs="+", default=None,
                        help="test a list of teams instead of the Mercedes engine group, e.g. --treated Ferrari \"Red Bull Racing\"")
    parser.add_argument("--label", default=None, help="name of the change, for titles (e.g. 'Macarena wing')")
    parser.add_argument("--window", nargs=2, type=int, default=None, metavar=("FIRST", "LAST"),
                        help="only use rounds FIRST..LAST (e.g. between two upgrade packages)")
    args = parser.parse_args()

    DEV_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    print(f"Season data: {panel['round'].nunique()} rounds, {panel['team'].nunique()} teams")
    if args.window:
        a, b = args.window
        panel = panel[panel["round"].between(a, b)].copy()
        print(f"Window: only rounds {a}-{b} are used ({panel['round'].nunique()} rounds)")

    cutoff = args.cutoff_round
    if cutoff is None and args.treated:
        raise SystemExit("With --treated, also give --cutoff-round (e.g. 4 for Miami).")
    if cutoff is None:
        try:
            dates = round_dates(args.year, args.cache_dir)
            after = [r for r, d in dates.items() if pd.notna(d) and d >= RULE_DATE and r in set(panel["round"])]
            cutoff = min(after)
            print(f"Compression rule from {RULE_DATE.date()}: first race under it = round {cutoff}")
        except Exception as err:
            raise SystemExit(f"Could not read the schedule ({err}); pass --cutoff-round N instead.")

    if not args.treated:
        development(panel)
    natural_experiment(panel, cutoff, args.treated, args.label or ("change" if args.treated else "1 June compression rule"))
    print(f"\nSaved tables + plots to: {DEV_DIR}")


if __name__ == "__main__":
    main()
