"""
F1 Energy Strategy - Step 2g: Power-unit element age (new engines, batteries, ...)
==================================================================================
Each driver may only use a limited number of each power-unit element per season (more = grid penalty),
so teams introduce new ones a few times a year. The FIA publishes, at every event, how many of each
element every driver has used so far ("PU Elements used per Driver up to now"):
    ICE  combustion engine      TC  turbocharger      EXH  exhaust      MGU-K  motor-generator
    ES   energy store (battery) PUCE control electronics              PUANC ancillary components

This step downloads those documents, works out at which event each driver started using a new element
(count at the next event minus count at this event), and tests whether a NEW engine or battery shows up
in the data:
    - effective drag (CdA, upper bound) - a fresher/stronger engine looks like LESS drag in our fit,
      because the fit can't separate engine power from drag
    - qualifying lap time
    - electrical energy deployed per lap (a fresh battery might store/deliver more)

Comparison: TEAMMATES at the same event. Same car, same track, same weather - one driver on a new
element, the other not. Pairs where both got the new element at the same event tell us nothing and
are skipped automatically (their difference is zero).

Limits of the data: teams keep a POOL of units and rotate them, so "new at this event" means first used
here; an older unit may be used again later (e.g. in practice). "Age" = events since the last new one,
which is approximate.

Needs: pip install pdfplumber requests   (and Step 2c data; Step 2e for wet sessions)
Run:
    python src/step2g_pu_elements.py --cache-dir C:\\f1cache
    python src/step2g_pu_elements.py --analyse-only
"""

import argparse
import os
import re
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEASON_DIR = ROOT / "outputs" / "season"
PDF_DIR = ROOT / "data" / "fia_pu"
OVERRIDES = ROOT / "reference" / "fia_pu_urls.csv"   # optional: round,url for documents with odd names
URL = "https://www.fia.com/system/files/decision-document/{year}_{slug}_-_pu_elements_used_per_driver_up_to_now.pdf"
ELEMENTS = ["ICE", "TC", "EXH", "MGU-K", "ES", "PUCE", "PUANC"]
TESTED = ["ICE", "ES", "MGU-K"]
MIN_PAIRS, MIN_TEAMS = 30, 6        # below this a test is skipped (standard errors unreliable)
ALLOWED_2026 = {"ICE": 4, "TC": 4, "EXH": 4, "MGU-K": 3, "ES": 3, "PUCE": 3, "PUANC": 6}  # before penalties


# ----------------------------------------------------------------------------- documents
def slugify(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def season_events(year, cache_dir=None):
    """round -> event name for the rounds saved by Step 2c, plus the NEXT event (its document tells us
    what was new at the last saved round)."""
    ev = {}
    for f in sorted(SEASON_DIR.glob("[0-9][0-9]_Q_laps.csv")):
        d = pd.read_csv(f, usecols=["round", "event"], nrows=1)
        ev[int(d["round"].iloc[0])] = d["event"].iloc[0]
    try:
        import fastf1
        if cache_dir:
            fastf1.Cache.enable_cache(str(cache_dir))
        sched = fastf1.get_event_schedule(year, include_testing=False)
        nxt = max(ev) + 1
        row = sched[sched["RoundNumber"] == nxt]
        if len(row):
            ev[nxt] = row["EventName"].iloc[0]
    except Exception as err:
        print(f"(could not read the schedule for the next event: {err})")
    return ev


# FIA document names that differ from FastF1's event name
SLUG_ALIASES = {
    "barcelona": ["barcelona-catalunya_grand_prix"],
    "bahrain": ["bahrain_grand_prix_in_malaysia", "bahrain_grand_prix_malaysia", "malaysian_grand_prix"],
}


def candidate_urls(year, name):
    base = slugify(name)
    slugs = [base] + [a for key, al in SLUG_ALIASES.items() if key in base for a in al]
    return [URL.format(year=year, slug=sl) for sl in slugs]


def download(year, events):
    import requests
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    over = pd.read_csv(OVERRIDES).set_index("round")["url"].to_dict() if OVERRIDES.exists() else {}
    for rnd, name in sorted(events.items()):
        path = PDF_DIR / f"{rnd:02d}_pu_elements.pdf"
        if path.exists() and path.stat().st_size > 1000:
            continue
        urls = [over[rnd]] if rnd in over else candidate_urls(year, name)
        got, last = False, ""
        for url in urls:
            for attempt in range(2):                     # one retry: fia.com sometimes times out
                try:
                    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (research project)"})
                    if r.status_code == 200 and r.content[:4] == b"%PDF":
                        path.write_bytes(r.content)
                        got = True
                    else:
                        last = f"{r.status_code} at {url}"
                    break
                except Exception as err:
                    last = f"{type(err).__name__} at {url}"
                    time.sleep(3)
            time.sleep(1.0)                              # be polite to the FIA server
            if got:
                break
        if got:
            print(f"Round {rnd:2d} {name}: downloaded")
        elif rnd == 1:
            print(f"Round  1 {name}: no document ({last}) - nothing had been used before round 1, so counts = 0")
        else:
            print(f"Round {rnd:2d} {name}: not found ({last})\n"
                  f"   -> find the 'PU Elements used per Driver up to now' document on fia.com and add "
                  f"'{rnd},<url>' to {OVERRIDES.relative_to(ROOT)} (first line of the file: round,url)")


def parse_pdf(path):
    """Rows like '03 Red Bull Racing RB Ford Max Verstappen 5 5 5 3 3 3 6' -> number + counts."""
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    cols = ELEMENTS
    for line in text.splitlines():                      # header tells us the element order
        toks = line.replace("MGU K", "MGU-K").split()
        found = [t for t in toks if t.upper().replace("-", "") in {e.replace("-", "") for e in ELEMENTS}]
        if len(found) >= 5 and "ICE" in [f.upper() for f in found]:
            norm = {e.replace("-", ""): e for e in ELEMENTS}
            cols = [norm[f.upper().replace("-", "")] for f in found]
            break
    n = len(cols)
    pat = re.compile(r"^\s*(\d{1,2})\s+(.+?)\s+((?:\d+\s+){%d}\d+)\s*$" % (n - 1))
    rows = []
    for line in text.splitlines():
        m = pat.match(line)
        if m:
            counts = [int(x) for x in m.group(3).split()]
            rows.append({"number": int(m.group(1)), "name": m.group(2), **dict(zip(cols, counts))})
    return pd.DataFrame(rows)


def driver_numbers(year, rounds, cache_dir):
    """FastF1 abbreviation (as used in our lap files) for every car number, per round."""
    import fastf1
    fastf1.set_log_level("ERROR")
    if cache_dir:
        fastf1.Cache.enable_cache(str(cache_dir))
    out = []
    for rnd in rounds:
        try:
            s = fastf1.get_session(year, rnd, "Q")
            s.load(laps=False, telemetry=False, weather=False, messages=False)
            res = s.results[["DriverNumber", "Abbreviation", "TeamName"]].copy()
            res["round"] = rnd
            out.append(res)
        except Exception as err:
            print(f"Round {rnd}: could not read driver numbers ({err})")
    d = pd.concat(out, ignore_index=True)
    d["number"] = pd.to_numeric(d["DriverNumber"], errors="coerce").astype("Int64")
    return d.rename(columns={"Abbreviation": "driver", "TeamName": "team_ff1"})[["round", "number", "driver", "team_ff1"]]


def build_usage(year, events, cache_dir):
    parts = []
    for rnd in sorted(events):
        path = PDF_DIR / f"{rnd:02d}_pu_elements.pdf"
        if path.exists():
            d = parse_pdf(path)
            if len(d) < 10:
                print(f"Round {rnd}: only {len(d)} rows parsed - check {path.name}")
            parts.append(d.assign(round=rnd))
    if not parts:
        raise SystemExit("No FIA documents available - run without --analyse-only first.")
    u = pd.concat(parts, ignore_index=True)
    first = min(events)
    if first not in set(u["round"]):
        # before the first round nobody has used anything: counts are zero
        base = u[u["round"] == u["round"].min()].copy()
        base[[c for c in ELEMENTS if c in base]] = 0
        u = pd.concat([base.assign(round=first), u], ignore_index=True)
    nums = driver_numbers(year, sorted(u["round"].unique()), cache_dir)
    u = u.merge(nums, on=["round", "number"], how="left")
    # rounds with no session yet (e.g. the next event): take each car number's latest known driver
    latest = nums.sort_values("round").groupby("number")[["driver", "team_ff1"]].last()
    for col in ("driver", "team_ff1"):
        u[col] = u[col].fillna(u["number"].map(latest[col]))
    u = u.sort_values(["number", "round"])
    cols = [c for c in ELEMENTS if c in u]
    for c in cols:
        # The 'up to now' document is published before running starts, so it does NOT yet count elements
        # fitted for that event (checked: Las Vegas 2026, Piastri '4' up to now + new ICE that weekend).
        # New at event k = count in the next event's document - count in this one.
        g = u.groupby("number")
        nxt, nxt_round = g[c].shift(-1), g["round"].shift(-1)
        u[f"new_{c}"] = np.where(nxt_round == u["round"] + 1, (nxt - u[c]).clip(lower=0), np.nan)
        ages, age = [], {}
        for num, new in zip(u["number"], u[f"new_{c}"]):
            if pd.isna(new):
                age[num] = None
                ages.append(np.nan)
                continue
            age[num] = 0 if new > 0 else (age[num] + 1 if age.get(num) is not None else np.nan)
            ages.append(age[num])
        u[f"age_{c}"] = ages
    u.to_csv(SEASON_DIR / "pu_usage.csv", index=False)
    return u


# ----------------------------------------------------------------------------- analysis
def driver_outcomes():
    q = pd.concat([pd.read_csv(f) for f in sorted(SEASON_DIR.glob("[0-9][0-9]_Q_laps.csv"))], ignore_index=True)
    drag = pd.concat([pd.read_csv(f) for f in sorted(SEASON_DIR.glob("[0-9][0-9]_drag.csv"))], ignore_index=True)
    cbr = SEASON_DIR / "conditions_by_round.csv"
    wet = set(pd.read_csv(cbr).query("wet_quali")["round"]) if cbr.exists() else set()
    q = q[~q["round"].isin(wet)]
    best = q.loc[q.groupby(["round", "driver"])["lap_time_s"].idxmin()]
    keep = ["round", "driver", "team", "lap_time_s"] + [c for c in ("electric_mj_mid",) if c in best]
    out = best[keep].merge(drag[["round", "driver", "cda_hi"]], on=["round", "driver"], how="left")
    out.loc[out["round"].isin(wet), "cda_hi"] = np.nan
    out["lap_pct"] = out["lap_time_s"] / out.groupby("round")["lap_time_s"].transform("median") * 100
    return out


def teammate_test(df, x, y):
    """Teammate differences at the same event; driver-pair effects remove the usual gap between the two."""
    rows = []
    for (rnd, team), g in df.dropna(subset=[x, y]).groupby(["round", "team"]):
        if len(g) != 2:
            continue
        a, b = g.sort_values("driver").iloc[0], g.sort_values("driver").iloc[1]
        rows.append({"team": team, "pair": f"{a['driver']}-{b['driver']}", "dx": a[x] - b[x], "dy": a[y] - b[y]})
    d = pd.DataFrame(rows)
    # too few teammate pairs with different ages, or too few teams, gives meaningless standard errors
    if d.empty or (d["dx"] != 0).sum() < MIN_PAIRS or d.loc[d["dx"] != 0, "team"].nunique() < MIN_TEAMS:
        return None
    d["dx"] -= d.groupby("pair")["dx"].transform("mean")
    d["dy"] -= d.groupby("pair")["dy"].transform("mean")
    sxx = (d["dx"] ** 2).sum()
    if sxx <= 0:
        return None
    b = (d["dx"] * d["dy"]).sum() / sxx
    u = d["dy"] - b * d["dx"]
    meat = sum(((g["dx"] * u[g.index]).sum()) ** 2 for _, g in d.groupby("team"))
    G = d["team"].nunique()
    se = np.sqrt(meat / sxx ** 2 * G / max(G - 1, 1))
    return {"x": x, "y": y, "effect": b, "se": se, "t": b / se if se > 0 else np.nan,
            "informative_pairs": int((d["dx"].abs() > 1e-9).sum()), "pairs": len(d)}


def analyse(u):
    pd.set_option("display.width", 200)
    cols = [c for c in ELEMENTS if c in u]
    print("\n" + "=" * 90 + "\nA. ELEMENTS USED BY THE LAST DOCUMENT (per driver)\n" + "=" * 90)
    last = u[u["round"] == u["round"].max()]
    print(last[["number", "driver", "name"] + cols].sort_values(cols[0], ascending=False).to_string(index=False))
    newcols = [f"new_{c}" for c in cols]
    # consistency: per car number, the new elements summed over the season must equal the final count
    # (a count that goes DOWN between documents means a misread row or a corrected document)
    fin = u.sort_values("round").groupby("number")[cols].last()
    tot = u.groupby("number")[newcols].sum().rename(columns=lambda c: c.replace("new_", ""))
    bad = (tot - fin).abs().sum(axis=1)
    drops = sum(int((u.groupby("number")[c].diff() < 0).sum()) for c in cols)
    print(f"\nConsistency: {drops} element counts went DOWN between documents; "
          f"{int((bad > 0).sum())} car numbers where season total of new elements != final count"
          + (" - check: " + ", ".join(str(n) for n in bad[bad > 0].index) if (bad > 0).any() else " - OK"))
    per_team = u.groupby("team_ff1")[newcols].sum().astype(int)
    print("(per team = both cars, attributed to the team the car number raced for at each event)")
    print("\nNew elements introduced per team over the season (both cars):")
    print(per_team.rename(columns=lambda c: c.replace("new_", "")).to_string())

    out = driver_outcomes().merge(u[["round", "driver"] + newcols + [f"age_{c}" for c in cols]],
                                  on=["round", "driver"], how="left")
    print("\n" + "=" * 90 + "\nB. DOES A NEW ELEMENT SHOW UP? (teammates at the same event)\n" + "=" * 90)
    res = []
    for c in TESTED:
        if f"new_{c}" not in out:
            continue
        out[f"is_new_{c}"] = (out[f"new_{c}"] > 0).astype(float).where(out[f"new_{c}"].notna())
        for y, what in [("cda_hi", "effective drag"), ("lap_pct", "quali lap time (% of median)"),
                        ("electric_mj_mid", "electrical energy per lap (MJ)")]:
            if y not in out:
                continue
            for x, xl in [(f"is_new_{c}", f"new {c} this event"), (f"age_{c}", f"{c} age (events)")]:
                r = teammate_test(out, x, y)
                if r:
                    r.update({"element": c, "x": xl, "outcome": what})
                    res.append(r)
    if not res:
        print(f"Not enough teammate pairs with different element ages (need {MIN_PAIRS}+ pairs from {MIN_TEAMS}+ teams).")
        return
    res = pd.DataFrame(res)[["element", "x", "outcome", "effect", "se", "t", "informative_pairs", "pairs"]]
    print(res.round(4).to_string(index=False))
    print(f"\n{len(res)} tests: with this many, 1-2 can reach |t| > 2 by chance. Expect a REAL engine effect to show"
          " as less effective drag AND a faster lap together, for 'new ICE'.")
    out.to_csv(SEASON_DIR / "pu_driver_panel.csv", index=False)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Power-unit element age vs drag / pace / energy.")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--cache-dir", default=os.environ.get("F1_CACHE_DIR"))
    ap.add_argument("--analyse-only", action="store_true", help="use already downloaded documents")
    args = ap.parse_args()
    events = season_events(args.year, args.cache_dir)
    if not args.analyse_only:
        download(args.year, events)
    u = build_usage(args.year, events, args.cache_dir)
    analyse(u)


if __name__ == "__main__":
    main()
