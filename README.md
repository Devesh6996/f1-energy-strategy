# F1 Energy Deployment Strategy (2026 regulations)

Infer how F1 cars deploy and harvest electrical energy from public telemetry,
then optimise deployment per driver style - for the fastest lap and for defending.

## Roadmap
- [x] **Step 1 - Energy fingerprint detector**: label each part of a lap as
      full deploy / clipping / lift-and-coast / braking (`src/step1_energy_fingerprint.py`)
- [x] Step 1b - Compare drivers (teammates!) side by side (`src/step1b_compare_drivers.py`)
- [x] **Step 2 - Physics power model**: estimate power (kW) from telemetry, fit each car's drag + total power,
      and classify clipping as energy-limited vs. regulation taper (`src/step2_power_model.py`)
      *Lesson from real data: a free two-parameter fit fails on 2026 laps (gave 279 kW, R^2 0.12) because cars do NOT
      deploy full power everywhere. Fix: 'envelope' fit - peak power fixed from the rules, drag fitted from the top edge
      of the data. That then OVERESTIMATED drag on real data (CdA ~2, 18-27 MJ/lap: physically impossible),
      likely because corner exits are traction-limited / in high-drag aero mode. Final fix: 'topspeed' fit -
      drag measured at 290-340 km/h on the straights, where the rules' speed-based MGU-K cap makes power known.
      Physics checks added: power must not exceed what the rules allow; energy must stay under ~12 MJ/lap.
      On a simulated car with active aero (CdA 1.25 in corners, 0.90 on straights): envelope 1.18-1.35,
      topspeed 0.90-0.91.*
- [x] **Step 2b - Batch analysis**: every flying lap of every driver, several events; drag fitted per driver
      (median over laps), air density from session weather, and slipstream checked against the REAL gap
      to the car ahead (`src/step2b_batch.py`)
- [x] **Step 2c - Whole 2026 season + tyre degradation** (`src/step2c_season.py`): drag fitted in qualifying
      as a RANGE (engine-only vs full-deployment bound), applied to green-flag race laps with a 70 kg fuel model;
      team habits compared across all tracks; tyre-age effect on clipping / super-clipping / lift-coast /
      energy estimated with driver-race fixed effects, separating tyre age from fuel burn (tyre age resets at
      pit stops, fuel does not), cluster-robust standard errors.
      *Validated on a simulated season: recovered built-in lap-time effects exactly (0.050 s/tyre lap,
      0.030 s/kg fuel) and the built-in rise of clipping with tyre age, consistently on all 3 tracks.*
- [ ] **Step 3 - Deployment optimizer** (`src/step3_optimizer.py`): quasi-steady-state lap simulator built from a
      real lap (corners/braking keep real speeds; straights simulated with engine + MGU-K, drag range from
      Step 2c, the rules' speed cap, traction and braking limits fitted from telemetry, 4 MJ battery, harvest
      cap) + an **ECMS optimizer** (energy gets a price in s/MJ; coordinate descent per straight chunk and
      braking zone; price bisection for energy-neutral race laps; leftover-energy polish).
      Compares lift-and-coast-only vs super-clip-only vs both; value of 1 MJ; race defence cost per straight.
      *Development notes: a greedy optimizer got stuck in local optima (a constrained 'defence' run beat its
      'optimum' by 0.9 s) -> replaced by ECMS. Simulator sped up 10x (numpy-in-loop -> plain Python).
      On a simulated lap the optimum independently reproduces the real-world pattern: deploy at corner exits,
      super-clip at the ends of straights, full braking regen. Energy price (0.66 s/MJ) matches the directly
      measured value of 1 MJ (0.58 s/MJ), as theory predicts.*
      *First real-data runs: regen was being counted in corners with no braking (fixed: regen only from real braking
      force) and braking was uncapped (now <= 6 g). Error then -2.5% (Monza) / -3.4% (Spa), almost all on the
      straights. Now drag is CALIBRATED so the optimised lap matches the real lap; the independent check is whether
      the calibrated drag falls inside Step 2c's measured range. Calibration uses the driver's fastest CLEAN-AIR lap
      (no car < 120 m ahead on fast straights, within 0.5% of their best): a towed lap would bias drag low.*
      *Calibration then landed OUTSIDE the measured drag range at both tracks (Monza CdA 1.29 vs 0.75-1.15; Spa 1.35
      vs 0.56-1.00), with the miss much bigger at Spa -> missing physics that costs time when flat out. Spa's big
      elevation changes and many flat-out corners pointed to the cause. Added from FastF1 X/Y/Z position data:
      gradient force (m*g*slope), a cornering-grip limit fitted from the real lap's own cornering, and tyre scrub
      in corners (0.04 x lateral force). Calibration target is now the official lap time.*
      *Result: calibrated drag moved much closer (Monza 1.29 -> 1.21, now 5% above the range; Spa 1.35 -> 1.22,
      22% above). Next addition: the mid-2026 rule limiting deployment to 250 kW outside key acceleration zones,
      applied to flat-out corners (radius < 800 m).*
      *With it: **Monza INSIDE the range (CdA 1.19 vs 0.75-1.15) -> simulator validated at Monza.** Spa still +21%.
      Found a flaw in the check itself: Step 2c's drag range was measured assuming a FLAT track, while the simulator
      now has hills - not like-for-like at Spa (137 m of climb). The check now re-measures the drag range on the same
      lap with the simulator's own physics (gradient, corner scrub, 250 kW zones).*
      *Result: Spa got WORSE (like-for-like range 0.44-0.86 -> calibration +42% outside) -> the flat-track explanation
      is REJECTED. Better lead - energy: at Monza the simulated optimum uses ~9.9 MJ vs ~9.7 MJ measured for the real
      car (Step 2b) and validates; at Spa it uses ~8.4 MJ vs ~5.5 MJ measured and fails. Spa is the longest lap, so a
      tighter real per-lap energy budget than the modelled rules would hurt it most. Calibration then absorbs the
      missing energy as 'extra drag'.*
      *Now: JOINT calibration. For each drag in the measured range, the telemetry also gives the energy the real car
      deployed. Test: is there a drag INSIDE the range where the simulator, limited to that energy, matches the real
      lap time? YES -> drag and energy both consistent with the data (validated); NO -> something else is missing.
      Caveat: calibration assumes the real lap's energy use is near-optimal; any remaining gap also includes how far
      the real strategy is from optimal (seen on a simulated test driver that deploys in a fixed pattern).*
      *Result: both tracks returned "YES" (Monza CdA 0.74 with 2.8 MJ; Spa 0.85 with 4.2 MJ) - but the solutions are
      DEGENERATE: the per-lap deployment limit binds, all three styles tie exactly, and at Monza the quali lap ends
      with a FULL battery (no team leaves energy on a quali lap). Real cars super-clip 4-5x/lap at Spa, i.e. they are
      harvest-limited, which contradicts a deployment-limited solution. Rejected.*
      *KEY LESSON - identifiability: the energy inferred from telemetry depends strongly on the assumed drag (Monza
      2.5 MJ at CdA 0.70 vs 7.8 MJ at 1.15). From speed data alone, 'more drag' and 'less deployment' look identical,
      so they cannot be separated with public data. Default is back to drag-only calibration (validated at Monza,
      harvest-limited like real cars); joint calibration is optional (--joint) with an automatic degeneracy check.*
      **Status: validated at Monza; Spa has a documented gap that public data cannot resolve.**
      **Correction after the setup (drag) robustness test at Monza:** at both ends of the drag range, super-clip-only and
      lift-and-coast-only came out essentially TIED (lo CdA 0.75: +0.101 vs +0.095 s; hi CdA 1.15: +0.571 vs +0.546 s
      off optimal), and the ranking flipped vs the calibrated run (CdA ~1.19) - a change too small to explain it.
      -> the restricted-style results carried optimizer noise of a few tenths of a second, as large as the claimed
      gap. 'Super-clip beats lift-and-coast at Monza' is NOT robust. Robust: combining both is fastest (by 0.1-0.55 s)
      in every run. Spa's gap was larger (lift-only +1.1-1.5 s vs super-clip-only +0.01-0.17 s) - not yet robustness-
      tested. The value of energy depends strongly on drag (0.15 s/MJ at CdA 0.75 vs 0.42 s/MJ at 1.15): only the
      calibrated value (~0.5 s/MJ) is meaningful. Fix: each style now gets several starting points (scratch, naive full
      deployment, and the free optimum with disallowed harvesting switched off) so comparisons aren't decided by noise.**
      **Re-test with the lower-noise optimizer, both tracks, both drag extremes (gap to the full optimum, s):**

      | Track / drag | lift-and-coast only | super-clip only | value of 1 MJ |
      |---|---|---|---|
      | Monza, CdA 0.75 | +0.095 | **+0.026** | 0.15 s |
      | Monza, CdA 1.15 | +0.455 | **+0.073** | 0.37 s |
      | Spa, CdA 0.56   | +0.053 | **+0.009** | 0.14 s |
      | Spa, CdA 1.00   | +0.324 | **+0.074** | 0.24 s |

      **Super-clipping beats lift-and-coast in all 4 cases**, and on its own gets within 0.01-0.07 s of the full optimum.
      The margin depends on the setup: small (0.04-0.07 s) for low-drag cars, large (0.25-0.38 s) for high-drag cars,
      which are more energy-hungry - the value of energy also rises with drag. So the earlier 'tie' was optimizer
      noise; the conclusion holds, with a setup-dependent margin.
      *Robust so far: super-clip-only beat lift-and-coast-only at both tracks on every model version
      (latest: Monza +0.07 vs +0.74 s off optimal; Spa +0.17 vs +1.12 s).*
      *Early (uncalibrated) result: super-clip-only beats lift-and-coast-only at Monza (+0.14 vs +0.38 s off optimal)
      and Spa (+0.07 vs +0.54 s); combining both is fastest.*
- [ ] **Step 2d - Season development + engine-rule natural experiment** (`src/step2d_development.py`, uses saved
      Step 2c data only). (1) Each team's qualifying pace vs the median team, effective drag and energy habits,
      1st vs 2nd half + trend. (2) From 1 June 2026 the FIA also measured engine compression ratio HOT, closing the
      loophole Mercedes was believed to use (Mercedes says its engine was legal). Difference-in-differences:
      Mercedes-engined teams (Mercedes, McLaren, Williams, Alpine) vs others, before vs after, two-way fixed effects
      (team + track), team-clustered SEs, plus team-specific development trends, a pre-trend check and placebo
      tests (fake dates, other engine groups). A weaker engine shows up as MORE effective drag.
      *Design lessons from the synthetic test: pace vs POLE creates phantom improvements when the fastest team slows
      -> measure vs the median team. A rival developing during the season biases a plain DiD (0.32% found vs 0.25%
      built in) -> with team trends: 0.266%. Effective-drag jump recovered (0.047 vs 0.05 built in).*
      **Real 2026 results (21 rounds, rule from round 6):**
      - Development: the two fastest teams lost part of their advantage GRADUALLY over the season - Mercedes
        -1.55% -> -1.17% vs the median team (trend t = 2.5), Ferrari -1.20% -> -0.77% (t = 2.2) -> the field
        catching up, not a step change. Aston Martin improved most (2.85% -> 1.91% behind the median) but noisily
        (t = -1.7). Red Bull ~flat between halves (-0.98% -> -0.94%) - but see the Miami test below: the half split
        hid a clear step at round 4.
      - Half-vs-half effective drag and lift-and-coast are confounded by the track mix (high-drag Singapore /
        Sao Paulo in the 2nd half). Lift-and-coast roughly halved for EVERY team in the 2nd half -> rule/track
        effect rather than team choice; worth a dedicated check.
      - **Compression-rule experiment: no clear effect.** Pace +0.19% slower after 1 June (t = 1.4; with team
        trends +0.12%, t = 0.3; 17% of placebos as big); effective drag moved the wrong way (-0.02, 83% of placebos
        as big). Pre-trend check fails (-0.37% per 10 rounds before the rule). -> No clear evidence from qualifying
        telemetry that the rule cost Mercedes-engined teams pace or power. Limits: only 5 pre-rule rounds and 11
        teams (only large effects detectable); engineers may have recovered performance elsewhere. A clean null.
      - **What the test can and cannot detect:** 95% range for the pace effect -0.08% to +0.47% (~ -0.08 to +0.42 s
        on a 90 s lap); effective drag -0.06 to +0.02. Published estimates of the loophole are mostly ~10-15 hp
        (~10 kW, reported as ~0.2-0.4 s/lap) -> INSIDE these ranges, so it cannot be ruled out. A very large
        loss (e.g. ~80 hp, which would mean ~1.5 s+ and ~+0.1 effective CdA by rough scaling) is ruled out.
        Note this measures what changed ON 1 June, not the size of the advantage (the rule may have only partly
        closed it, or the trick may not have been used at full strength).
      - **Generalised to any group of teams** (`--treated ... --cutoff-round N --label ...`), with a placebo built
        from every same-size group of OTHER teams. First use: the 'Macarena' rotating rear wing raced by Ferrari and
        Red Bull from Miami (round 4); reported straight-line gain 5-10 km/h for the package. Rough upper bound from
        this project's Monza drag sensitivity: ~0.4-0.8 s/lap at Monza (less at twisty tracks). Caveat: Miami also
        brought big upgrade packages and the energy-rule tweaks -> the test measures the whole Miami package.
        *Synthetic test: built-in -0.06 CdA recovered as -0.072 (with team trends), 0 of 36 placebo pairs as big;
        other changes in the comparison group (e.g. a rival's mid-season step) bias the estimate.*
        **Real results (from Miami, round 4; placebo = each other single team):**
        - Ferrari + Red Bull together: pace -0.43% (t = -1.9; 14% of pairs as big) - suggestive; effective drag no change.
        - **Red Bull alone: pace -0.65% (~0.55 s/lap, t = -6.5, 1 of 10 teams as big) and lift-and-coast -1.5 pts
          (t = -4.2, survives team trends)** -> a clear step at Miami, matching reports of its gap to the front roughly
          halving; attributed to the whole upgrade package, with a changed energy approach.
        - **Ferrari alone (the original Macarena wing): pace -0.12% (7 of 10 teams as big) - no clear change**; effective
          drag -0.04 (expected direction, but 7 of 10 teams as big). Its -0.36 MJ energy drop is likely the same weak drag
          hint seen again: energy is inferred USING the fitted drag (identifiability).
        - Answer: the wing alone shows little measurable qualifying-pace advantage; the Miami gain was Red Bull's package.
          Can't separate 'small effect' from 'spent on downforce'. Limits: 3 pre-Miami races, energy rules changed at Miami.
        - **Correction:** the 1st-vs-2nd-half development table showed Red Bull 'flat' - wrong conclusion. The step came at
          round 4, so the 1st half mixed 3 slow and 8 fast races. Lesson: test at the actual date of a change, not
          with arbitrary half-season splits.
- [ ] Step 4 - Driver-style profiles (cluster drivers, feed profiles to the optimiser)
- [ ] Step 5 - Defence with slipstream: estimate tow effect (drag reduction vs. gap) from 2026 race telemetry,
      then a two-car simulation (attacker with tow + overtake mode vs. defender's deployment choices)

## Data
FastF1 (official F1 timing + telemetry). There is **no battery / ERS channel**
in public data - energy use is inferred from speed, throttle and brake.

## Run
```
python src/step1_energy_fingerprint.py --year 2026 --event Monza --session Q --driver VER
python src/step1b_compare_drivers.py --drivers LEC HAM
python src/step2_power_model.py --drivers PIA NOR
python src/step2b_batch.py --events Monza
python src/step2c_season.py --cache-dir C:\f1cache
python src/step3_optimizer.py --event Monza --driver LEC
python src/step3_optimizer.py --event Belgium --driver VER --mode race --defend
python src/step3_optimizer.py --event Monza --driver LEC --mode race --recharge-test
python src/step2d_development.py --cache-dir C:\f1cache
python src/step2d_development.py --treated Ferrari "Red Bull Racing" --cutoff-round 4 --label "Macarena wing"
```

## Findings so far (Monza 2026 Q, fastest laps)
- Clipping **locations** are shared by all cars -> set by the circuit and the rules.
- Clipping **amount** differs between teammates in the same car: PIA clipped 11.3% of the lap vs NOR 5.5%,
  while NOR lifted-and-coasted more (2.7% vs 1.0%). PIA was 0.1 s faster. Ferrari's LEC/HAM were near-identical.
- **Step 2 result (topspeed fit, no warnings):** Ferrari teammates fit identical drag (CdA 1.28), McLarens within 7%;
  energy 8.6-11.3 MJ/lap (physically plausible). Absolute CdA depends on mass/efficiency assumptions;
  relative comparisons are the robust part.
- **PIA vs NOR:** PIA's clipping is *super clipping* - power fell to ~119 kW, far below the ICE-only ~380 kW,
  so the MGU-K was harvesting at full throttle. NOR harvested more by lift-and-coast and deployed more
  (11.3 vs 8.6 MJ). Two strategies, same car; PIA 0.1 s faster on this lap.
- **Step 2b (all flying laps, 5-6 per driver) - the one-lap PIA vs NOR story did NOT replicate:**
  clipping 9.6% vs 8.9%, lift-coast 1.7% vs 2.0%, super-clip zones/lap 0.6 vs 0.8 -> near-identical.
- **What does hold:** a team-level difference, consistent across BOTH teammates - Ferrari ~2.3-2.4 super-clip
  zones/lap and ~0% lift-coast vs McLaren ~0.6-0.8 and ~2% lift-coast -> car/power-unit strategy, not driver.
- Zones with power ABOVE the rules' maximum were first labelled 'slipstream'. Checked against the real gap to
  the car ahead: only 7% had a car within 120 m (vs 3% of other zones) -> mostly model error near ~330 km/h,
  not tow. Label renamed 'above modelled limit (tow or model error)'.

### Full grid, Monza 2026 Q (97 flying laps, 22 drivers, 11 teams)
- **Engine hypothesis rejected.** Ferrari super-clips ~2.3-2.4 zones/lap, but Haas (0.0-0.2) and Cadillac (0.0)
  use the same Ferrari engine. Mercedes-powered teams also range from 0.0-0.2 (Williams) to 1.4-1.5 (Mercedes).
  -> Super-clipping is a **team / car-specific strategy**, not set by the power unit.
- **Teammates agree** on super-clipping (Spearman 0.88 across 11 teams) and on effective drag (0.81)
  -> the model measures real car properties, not noise.
- **Super-clipping does not predict pace** (Spearman -0.18 vs best lap): harvesting at full throttle and
  lift-and-coast both work.
- **Effective drag strongly tracks pace** (Spearman 0.85). Caution: it is drag RELATIVE to the assumed
  400 kW engine and 780 kg mass - a weaker engine or heavier car also shows up as "more drag".
- **Known limitation:** 7 drivers (mostly slower cars) exceed ~12 MJ/lap, which is not physically possible
  -> absolute energy is biased high where effective drag is high. Use energy for comparisons between
  similar cars only.

### Second track: Spa 2026 Q (184 laps total, Monza + Spa)
- **The track dominates.** Everyone super-clips far more at Spa (2.3-5.5 zones/lap vs 0-3 at Monza;
  clipping 15-21% of the lap vs 4-12%). Haas, Cadillac and Williams barely super-clipped at Monza but do
  at Spa -> the Monza team differences were partly track-specific.
- **What is stable across tracks:** lift-and-coast habits (team rank agreement Monza vs Spa: 0.83).
  Ferrari lifts and coasts the least at both tracks; McLaren, Red Bull, Cadillac and Aston the most.
  Super-clipping rank agreement is weaker (0.50).
- **At Spa, more super-clipping goes with faster laps** (Spearman -0.54; Monza -0.18). Correlation, not
  cause: faster cars may simply have better energy systems.
- **Model flaw found:** effective drag comes out HIGHER at Monza (CdA 1.40-1.84) than at Spa (1.05-1.36),
  but Monza is the lowest-downforce track of the year - physically backwards. Monza energy (~2 MJ/km)
  is also implausible vs Spa (~0.8 MJ/km). Likely cause: the topspeed fit assumes cars deploy at the rules'
  cap at 290-340 km/h when not clipping; at Monza they spend long stretches there under energy management,
  which inflates drag. Spa results look physically plausible. Next fix: bracket drag between an
  "engine only" and a "full deployment" bound instead of assuming one.

### Full 2026 season (21 events, 2,104 qualifying laps, 16,181 green-flag race laps)
**A. Team energy habits**
- **Lift-and-coast is the most stable team trait** (mean rank agreement across 210 track pairs: +0.54).
  Red Bull (avg rank 1.7) and McLaren (2.1) lift-and-coast the most; **Ferrari the least (9.3 of 11)**.
- **Ferrari instead super-clips the most** (avg rank 3.6) -> two opposite ways to harvest energy:
  Red Bull/McLaren lift before braking, Ferrari harvests while flat out.
- Super-clipping order is mostly track-dependent (+0.15). Track matters hugely: clipping ~1-2% of the lap at
  Monaco, Singapore, Hungary, Sao Paulo vs 16-18% at Silverstone, Spa, Las Vegas.
- Drag bounds are unreliable at low-speed tracks (Monaco 0.90-2.14, Sao Paulo 1.22-2.31, Singapore
  0.78-1.67): too few high-speed samples. Monza (0.79-1.25) and Spa (0.61-1.10) now overlap -> the earlier
  "Monza has more drag" contradiction disappears once drag is a range.

**B. Tyre age vs energy use (driver-race fixed effects, fuel separated, cluster-robust SEs)**
- **Sanity check passed on real data:** lap time +0.029-0.032 s per lap of tyre age (~0.6 s on a 20-lap-old
  tyre) and +0.040 s per kg of fuel - in line with the usual rule of thumb (~0.03 s/kg).
- **Tyre wear has only a small effect on clipping:** +0.03 percentage points per lap on mediums (clear),
  not clear on softs/hards. **Fuel burn matters ~4x more:** -0.031 pts per kg -> a car that burns 70 kg
  clips ~2 pts more of the lap by the end (lighter car = faster on the straights = runs out of energy sooner).
- **Lift-and-coast falls as tyres age** (softs -0.013, hards -0.009 pts/lap, both clear).
- **Electric energy deployed falls on worn softs/mediums** (~-0.2 MJ per lap on a 20-lap-old tyre).
- **Track-specific:** tyre age raises clipping clearly at **Las Vegas (+0.134 pts/lap)** and **Baku (+0.064)** -
  street circuits with long straights - and barely anywhere else.
- Caveats: race laps include battles and overtake mode; fuel model is linear from a 70 kg maximum;
  21 per-track tests -> ~1 'clear' result expected by chance.
- **Track temperature (added):** hot tracks wear tyres faster, and the first analysis pooled hot and cool races.
  `step2c_season.py --add-weather` adds each lap's track/air temperature from FastF1's weather feed (no
  re-processing of telemetry) and re-runs the analysis with (1) track temperature as a control, (2) cooler vs
  hotter races compared, (3) a tyre-age x track-temperature interaction test.
  *Validated on a simulated season with a built-in heat effect: recovered it exactly (0.0833 s/lap in the hot race,
  +0.0167 s/lap per +10 C).*
  **Real 2026 result (21 races, track temps 14-54 C race average, split at 38 C):**
  - **Softs and mediums degrade ~2x faster on hot tracks** (lap time per tyre lap: soft 0.018 -> 0.039 s,
    medium 0.019 -> 0.038 s; hard ~unchanged 0.035 -> 0.031 s). Interaction: +0.0075 s/lap per +10 C (t = 4.7).
    Most trustworthy heat result - matches tyre physics.
  - Clipping rises LESS with tyre age on hot tracks (medium +0.046 vs +0.019 pts/lap; interaction t = -5.9) -
    treat with caution: temperature varies mostly BETWEEN circuits (heat is confounded with layout), and Las Vegas
    (coldest, 0-14 C) has by far the strongest tyre-clipping effect (+0.134/lap) and may drive it alone.
    Test: `--analyse-only --exclude-rounds 21`.
    **Result without Las Vegas: the effect survives** (interaction -0.0168 pts/lap per +10 C, t = -5.3; medium
    +0.044 cool vs +0.019 hot) -> not a single-track artefact. Still confounded with circuit layout (temperature
    varies mostly between circuits); separating them needs the same tracks in different conditions (more seasons).
    The heat effect on tyre degradation also survives (+0.0065 s/lap per +10 C, t = 3.8).


### Testing one upgrade between two others (`--window`)
Red Bull brought several big packages (Miami round 4, Austria round 8, Baku round 15, which was raced
in Malaysia in round 16). A plain before/after test of Baku would count the Miami and Austria gains as
"before", so the comparison is limited to the rounds between packages:

    python src\step2d_development.py --treated "Red Bull Racing" --cutoff-round 15 --window 8 21 --label "Baku package"
    python src\step2d_development.py --treated "Red Bull Racing" --cutoff-round 8 --window 4 14 --label "Austria package"

Check on synthetic data: a built-in -0.30% step at round 15 was recovered as -0.30% (team-trends version,
t = -24); the version without trends gave -0.58% because Red Bull was already improving before Baku.
So for upgrade tests, read the team-trends column.


### Race pace (`race_pace_%`)
Qualifying pace only sees one fast lap. Race pace is estimated per race from clean green-flag dry laps
(Step 2c's `NN_R_laps.csv`, no new downloads):

    lap_time = race constant + fuel effect x fuel_kg + compound offset + compound degradation x tyre age + team effect

- Fuel and tyre age can be told apart because teams pit on different laps. If the fitted fuel effect is
  outside 0.015-0.06 s/kg (per 90 s of lap) a standard 0.03 s/kg is used; the fits are saved to
  `outputs/development/race_pace_fuel_fits.csv`.
- Laps more than 2% slower than the driver's stint median (traffic, mistakes) are dropped; a team needs
  at least 15 clean laps in a race.
- The team effect is converted to % of the race's median lap and measured against the median team.
- Check on synthetic races with a built-in -0.40% Red Bull step at round 15 (3 random seeds): recovered
  -0.33 to -0.39% with team trends, fuel effect recovered as 0.029 s/kg (true 0.030). Slightly smaller
  than the truth because the % is taken of the median race lap, which includes fuel and tyre time.
- Limits: dirty air is not modelled (a car stuck in traffic looks slower), and team strategy choices
  (managing tyres, saving fuel) count as "pace".


### Findings: Red Bull's upgrade packages (qualifying vs race pace)
| Package (first round) | Quali pace | Race pace | Verdict |
|---|---|---|---|
| Miami (4) | -0.65% (t -6.5) | **-0.95% (t -8.4), no placebo as big** | Real step, bigger on Sundays than in qualifying |
| Austria (8) | +0.22% (t 2.1, placebo 0.45) | -0.01% (t -0.04) | No step detected |
| Baku (15), raced in Malaysia (16) | +0.05% (t 0.2) | +0.07% (t 0.5) | No step detected |

- Fuel effect fitted from the race laps: 0.042 s/kg (median of 21 races; standard value used in 3).
- Red Bull's race-pace advantage after Miami (~1% vs the start of the season) shrank in rounds 19-21
  (late season, data up to Las Vegas). Qatar and Abu Dhabi were still to come when this was run:
  rerun after them to see whether the fade continues. Malaysia (round 16) sits at the same level as rounds 4-11: the win there fits the
  Miami-level car, not a new step from the Baku package.
- Lesson: the team-trends column gave large numbers for Austria and Baku race pace (-0.63%, -0.83%)
  because a linear trend fitted to a few noisy rounds was extrapolated. Step 2d now runs the placebo test
  on the team-trends version too (`share_placebos_as_big_trends`).
- Limits: only 2-3 rounds before Miami; dirty air and tyre/fuel saving count as pace.


## Step 2e - Track conditions (rain, wind, humidity)
`src/step2e_conditions.py` adds the conditions to every saved lap from the lap timing + weather feed
(no telemetry): rain during the lap, wet tyres, wind speed/direction, humidity, temperatures, pressure,
when in the session the lap started, and deleted laps (track limits). Saved as `NN_Q_conditions.csv`
and `NN_R_conditions.csv`.

    python src\step2e_conditions.py --cache-dir C:\f1cache
    python src\step2e_conditions.py --analyse-only

- A qualifying session is "wet" if over 30% of its laps were wet (rain or intermediate/wet tyres).
  Step 2d now drops wet quali laps, ignores drag from wet sessions, and drops race laps driven in rain.
- Wind limitation: FastF1 X/Y coordinates are not aligned with north, so headwind vs tailwind on a
  given straight can't be computed. The script tests instead whether windier sessions give unusual
  drag numbers. Wind is the same for every team at a track, so team comparisons are mostly protected.
- Humidity: humid air is slightly lighter than dry air; `rho_change_%` shows how much that would
  change the air density used in the drag fit.
- Tested on synthetic sessions (one wet qualifying, one race with a rain spell): the wet session and
  the rain laps were found and dropped.

Findings (21 rounds):
- Sao Paulo qualifying was fully wet (100% wet laps, 90% humidity): its drag outlier (CdA 2.29 vs
  median ~1.1) is rain, not the car. Step 2d now leaves it out. Dutch GP: 16% wet quali laps, 14% rain
  laps in the race - those laps are dropped individually.
- Humidity correlates with drag across tracks (r = +0.72), but humid air only lowers density by
  0.2-1.2%, far too little to explain the differences. Humid tracks are Sao Paulo (wet) and slow street
  circuits like Singapore - so it is the track type, not the humidity.
- The other outliers (Monaco 2.14, Singapore 1.67) are dry. Monaco turned out to be a modelling error:
  the FIA's special Monaco MGU-K map (see below) - with it, Monaco's drag is 1.20, close to the median.
- Wind: within a session it differs by at most 0.5 m/s between drivers' laps, worth ~0.01-0.04 CdA -
  negligible for team comparisons.

### Safety cars, VSCs and red flags
- Already handled since Step 2c: laps run under SC/VSC/red flag and pit in/out laps are never used
  (`pick_track_status("1")`, `pick_wo_box()`).
- Added now (Step 2e stores the track status of every lap; Step 2d uses it in race pace):
  - the 2 green laps after each neutralisation are dropped: cold tyres, cars bunched up (dirty air),
    and in 2026 a battery recharged to full under the safety car;
  - the fuel load is rebuilt lap by lap: a neutralised lap is assumed to burn half a normal lap's fuel,
    so laps after a safety car carry more fuel than a straight line from 70 kg would say.
- Free pit stops under SC change tyre ages, but tyre age is already in the race-pace model.

Results after the conditions + safety-car fixes (21 rounds):
- Deleted (track-limits) laps: 0-7% of quali laps per round (most at COTA); now excluded from quali pace.
- Restart laps dropped: 8-112 per race; Dutch GP: 113 race laps in rain dropped.
- Miami package result is unchanged: quali -0.67% (t -6.7), race -0.98% (t -9.1), no placebo as big.
- Round 6 (Monaco) stays a race-pace outlier for Red Bull (+0.29% while neighbours are ~ -1%). Monaco
  races are mostly trains behind slower cars - a dirty-air problem, handled in the next step.
- The panel now keeps a race when qualifying was wet (Sao Paulo race pace is used, quali pace isn't).

### Dirty air in race pace
- Step 2e now also saves the gap to the car physically ahead (any car, lapped ones included) when each
  lap starts and ends, from the line-crossing times. In-laps (car in the pit lane) don't count.
- Race pace now includes two dirty-air terms: within 1 s and 1-2 s of the car ahead (clean air = more than
  2 s). So a team stuck in a train (e.g. Monaco) is no longer counted as slow.
- The per-race dirty-air cost is printed and saved in `race_pace_fuel_fits.csv`. In 2026 a car within 1 s
  also gets overtake mode (extra electrical energy), so the number is the NET effect.
- Check on synthetic races with a built-in cost of 0.6 s (within 1 s) and 0.2 s (1-2 s): estimated
  0.46 s and 0.22 s; the Red Bull step was recovered as -0.41% (true -0.40%).
- Caveat: who ends up close behind is not random (a faster car catches up), so the dirty-air cost is an
  association, partly controlled by the team effects.

Dirty-air findings (21 races):
- Running within 1 s of the car ahead costs a median +0.33 s/lap vs clean air; 1-2 s behind costs +0.11 s.
- Biggest costs on slow, twisty tracks: Monaco +0.76, Singapore +0.73, Madrid +0.68, Austin +0.62,
  Hungary +0.58 s/lap.
- No cost, or even a gain, on long-straight tracks: Monza -0.06, Las Vegas -0.01 (-0.28 at 1-2 s),
  Baku 0.00, Montreal -0.02. There, the tow plus 2026 overtake mode cancels the aero loss in corners.
- Miami package, race pace: -0.88% (t -7.6, no placebo as big), down from -0.98%. Part of Red Bull's
  apparent race-pace gain was clean air from running at the front - the upgrade effect is still clear.
- Monaco stays an outlier for Red Bull (+0.42%): not explained by dirty air. One race; Monaco race pace
  is shaped by deliberate pace management and is the least reliable race of the season.

### Event-specific MGU-K map: Monaco "Rev 1"
At Monaco the FIA mandated a different MGU-K map: full 350 kW only up to 200 km/h, then a linear taper
to zero at 300 km/h (standard map: full power to ~290 km/h). Straight-line aero mode was also not
allowed. The drag fit assumed the standard map, so at Monaco it credited the cars with ~300 kW more
power at top speed than they had - and read the missing speed as drag (CdA 2.14, the biggest dry outlier).
`step2_power_model.py` now has `EVENT_MAPS` / `set_deployment_map()`, and Step 2c picks the map from the
event name. Re-run Monaco: `python src\step2c_season.py --rounds 6 --refresh --cache-dir C:\f1cache`.
Source for the map: the FIA "Power Unit Information" document published for every event (it also lists
the maximum recharge per lap and the full-power sectors, which vary by track).

Monaco re-run with the Rev 1 map: effective drag 2.14 -> 1.20 (season median ~1.1). Remaining dry
outliers: Singapore 1.67, Silverstone 1.56, Zandvoort 1.46 - to check against their FIA PU documents.

### Track evolution in qualifying
The track gets faster through qualifying, so a team knocked out in Q1 set its best lap on a slower
track than a team in Q3. Step 2d now fits, per round, lap_time = driver effect + slope x session time on
each driver's push laps (within 2% of their best) and corrects every lap to the track at the end of the
session. `pace_gap_%` is the corrected quali pace; `pace_gap_raw_%` is kept for comparison. Fits are in
`outputs/development/track_evolution.csv`.
- Check on synthetic sessions with -0.30 s per 10 min built in: recovered -0.30 s per 10 min in every
  round; the corrected pace had ~9x smaller standard errors than the raw pace.
- The slope also contains drivers pushing harder later in the session - also not car pace.

Track-evolution findings (20 dry qualifying sessions):
- The track gains a median 0.21 s per 10 minutes of qualifying, about 1.2 s over a whole session.
- Miami package, qualifying pace: -0.53% corrected (t -6.6, no placebo as big) vs -0.68% uncorrected.
  About a fifth of Red Bull's apparent qualifying gain came from reaching later sessions on a faster
  track; the rest is the car. Race pace (no track-evolution issue) stays -0.88%.
