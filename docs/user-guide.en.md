# SeaArrayCalib User Guide

[中文版](user-guide.zh-CN.md) · [Project home and setup](../README.md) · [Numerical workflow](model.md) · [CSV formats](data-format.md)

SeaArrayCalib is a local web application for GNSS–acoustic (GNSS-A) survey planning and seafloor beacon calibration. It combines array geometry, complete routes, acoustic scheduling, sound-field assumptions and joint estimation to compare how to conduct a survey, how long it takes and what performance the declared model predicts. This guide describes the features available in the current interface.

## Contents

1. [Capabilities](#capabilities)
2. [Installation and startup](#startup)
3. [Your first simulation](#first-run)
4. [Array, route and operational settings](#settings)
5. [C1, C2, C3 and risk radius](#risk)
6. [Fixed-round and target-time-gap refinement](#refinement)
7. [Background plus historical profile simulation](#profiles)
8. [Joint calibration of field data](#field)
9. [Interpreting results](#results)
10. [Exporting and saving](#export)
11. [Troubleshooting](#troubleshooting)
12. [Suggested workflow and model scope](#scope)

<a id="capabilities"></a>
## 1. Capabilities

The app has two workspaces: **Survey design** and **Field data**. Switch between 中文 and EN without losing inputs, uploaded files or results within the current page session.

**Survey design supports:**

- Editable 3D coordinates for 1–64 beacons, with configurable layout and depth.
- Six complete route families: circle, figure eight, clover, square, four-petal rose and circle plus cross, with scale and orientation searches.
- Candidate generation accounting for speed, full path length, serial acoustic interrogation, usable slant range and observation coverage, followed by selection at ten time budgets.
- A file-free Munk sound-field demonstration or user-supplied background and dated historical sound-speed profiles.
- Nominal precision, average sound-field risk and worst-direction risk within a declared perturbation set, with an independent design risk radius.
- Fixed-round or target-time-gap scale refinement, with geometry, scores, joint estimates, missing observations and failure reasons retained.
- Profile-based route selection using training dates, followed by validation on one or every held-out date.

**Field data supports:**

- Three CSV inputs: GNSS, acoustic travel times and sound-speed profile, with beacon-ID detection and data previews.
- Joint calibration of selected beacons by default, optional same-model single-beacon fitting and other solvers listed in the interface.
- Attitude rotation of the GNSS-antenna-to-transducer lever arm, with coordinate estimates, internal precision, residuals and fitted parameters.
- Geographic and local-coordinate track displays, raw points or smoothed display curves, and selectable result exports.

Survey design compares generated prospective operations. Field data processes the uploaded set of track observations; it does not automatically convert that track into a collection of candidate routes for risk ranking.

<a id="startup"></a>
## 2. Installation and startup

For first-time Windows setup, install Python 3.12 and Node.js 22.12 or a newer 22.x release. Open a terminal in the project directory and run:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
npm ci
.\start_all.bat
```

After setup, double-click `start_all.bat`. The launcher checks ports 8000 and 5173, stops identified previous SeaArrayCalib services, and restarts both services. Open [SeaArrayCalib locally](http://127.0.0.1:5173/). The website uses port 5173 and the calculation service uses port 8000. Logs are written to `.run/`. Unrelated programs occupying either port are reported and left running. The launcher does not install dependencies automatically.

To stop both services, double-click `stop_all.bat`, or run this from the project directory:

```powershell
powershell -ExecutionPolicy Bypass -File stop_web.ps1
```

See the [README](../README.md) for manual macOS/Linux startup. Results are held in the current calculation service's memory: export before restarting it. Refreshing the page is not a way to save a run.

<a id="first-run"></a>
## 3. Your first simulation

1. Open **Survey design** and select **Ideal Munk model**. No profile files are needed.
2. Check the beacon coordinates. The initial four-beacon array spans 6 km × 6 km, with E/N = ±3000 m and depths 3150, 3180, 3220 and 3250 m. Up is positive, so seafloor U coordinates are negative.
3. Review **Survey settings**. Current UI defaults include 3 kn, 5–25 h budgets, eight seed scales, four orientations, one refinement round, C3, ρ = 1 and a 10 km usable slant range.
4. For a smaller introductory run, select one family, one 180° division, three seed scales and zero refinement rounds. Choose appropriate bounds and search density before a substantive comparison.
5. Set range, sampling and noise to match your equipment. Demonstration defaults do not infer equipment performance from sonar frequency.
6. Select **Run simulation** and inspect progress or errors. The first calculation may include numerical-function compilation time.
7. After completion, switch budgets and inspect Plans A/B/C, candidate status, coordinate estimates and fit outcomes, then export.

The **Reset** button in the array editor restores the initial four-beacon coordinates, not every setting in the app. When inputs change, existing results still belong to their saved run configuration; rerun to evaluate the new settings.

<a id="settings"></a>
## 4. Array, route and operational settings

### Array and lever arm

Coordinates use east, north, up (ENU), in metres. Routes are centred on the array's horizontal centroid. The search varies shape, scale and orientation, not centre offset. The supported beacon-count limit does not imply that every layout is identifiable.

The lever arm is the transducer displacement relative to the GNSS antenna in vessel right, forward, up (RFU) coordinates. A downward component is negative. Simulation rotates it with route heading; field processing uses supplied attitude. The two workspaces have different example lever arms: enter the values for your installation.

### Six complete route families

Circle, figure eight, clover, square and four-petal rose are closed. Circle plus cross executes one circle and two perpendicular diameters with a straight transfer, ending at the final endpoint without forcing a return. Transfer time always counts. **Sample straight transfer** is off by default.

Scale means maximum radial extent from the route centre. A square's side is √2 times that extent. Bounds are entered as multiples of mean beacon depth, defaulting to 0.5–10. At 3200 m mean depth, this is 1.6–32 km. Large geometries can be generated, but over-budget or unsupported candidates cannot become eligible plans.

**180° divisions** controls orientation: 1 gives 0°; 2 gives 0°/90°; 4 gives 0°/45°/90°/135°. For circles, the angle changes starting phase; for other shapes, it primarily rotates the geometry. Sailing order is fixed, and scale refinement does not optimize a separate visiting order.

### Speed and time budgets

Default speed is 3 kn, approximately 1.5433 m/s. Switching units preserves physical speed. Enter minimum and maximum budgets in hours, from 0.1 to 48, with minimum strictly below maximum. The app generates ten logarithmically spaced caps including both endpoints.

A budget is an upper bound on the full operation, so a chosen route may be shorter. Routes are not truncated, repeated or padded with fictitious waiting to fill the cap. Budget switching reuses candidates and observations from the same run instead of redrawing noise at every cap.

### Sampling, coverage and noise

- Sampling can be requested in Hz or seconds per ping. Interrogation is serial, addressing one beacon per slot. Actual slots also respect conservative two-way propagation occupancy plus a one-second reply allowance; a higher requested rate may therefore add no observations.
- Usable slant range is the straight 3D transducer-to-addressed-beacon distance limit. Exceeding it excludes that observation; observations of other beacons are assessed separately.
- Ray support, dropout and per-beacon coverage also affect feasibility. At least eight valid observations per beacon are currently required, together with the other geometry and estimation checks.
- Simulation acoustic timing noise is entered in **seconds**, initially 0.0001 s. GNSS noise is in metres, initially 0.05 m. Field-data timing noise is entered in **milliseconds**.
- Munk mode exposes horizontal sound-speed gradients, temporal variability and correlation time. Profile mode uses uploaded data, so the analytic perturbation controls are inactive there.
- The random seed supports reproduction of a given configuration. Changing the candidate pool can change the time nodes of the common random field; equal seeds alone do not imply identical samples across different searches.

<a id="risk"></a>
## 5. C1, C2, C3 and risk radius

**C1: nominal coordinate variance.** Joint-coordinate uncertainty under the specified model, noise and acoustic nuisance-parameter priors.

**C2: average coordinate risk.** C1 plus an average environmental contribution propagated from the declared sound-field perturbation second moment.

**C3: worst-direction coordinate risk.** C1 plus the largest coordinate contribution over directions in the declared ellipsoidal perturbation set.

Within a criterion, lower is better; scores are in m². They are design predictions, not guarantees of realized error. A lower C3 does not guarantee smaller errors on every validation date.

### Design perturbation multiplier / risk radius ρ

The independent risk radius defaults to 1 and accepts 0–100. It primarily supports comparisons of Munk design-risk assumptions and also works with uploaded profiles. For profiles, keep ρ = 1 to use the training-derived perturbation scale.

- ρ = 0: C2 and C3 reduce to C1.
- ρ = 1: use the baseline design perturbation scale.
- ρ = 2: the **environmental terms** in C2/C3 become four times as large; C1 itself is unchanged.

Radius changes design scores and final selections, not validation fields, noise, dropout or the estimator. To retain a common pool across radius changes, local refinement always uses C1/C2/C3 at ρ = 1. Your chosen radius scores the resulting pool.

Profile mode has a separate **Profile anomaly multiplier**, with factors 0.5, 1, 2 and 5. It changes both training-risk amplitude and validation sound-speed anomalies. Keep both controls at 1 to preserve the supplied amplitude. To examine design-risk sensitivity alone, keep the profile multiplier at 1 and vary ρ. Radius is neither a confidence probability nor a permitted position-error ceiling.

<a id="refinement"></a>
## 6. Fixed-round and target-time-gap refinement

The search starts with logarithmic seed scales within the bounds, supplemented by scales at full-route budget boundaries. Refinement then adds candidates. **Every earlier candidate remains available and is compared alongside new ones.**

### Fixed rounds: 0, 1, 2 or 3

Zero uses seeds and budget boundaries without local refinement; the default is one round. In each round, the search selects winners for each family, orientation, budget and unit-radius criterion, finds their immediate smaller and larger neighbouring scales, and adds geometric-mean midpoints on both sides. Duplicate proposals are merged.

For example, if scale 2 wins with neighbours 1 and 4, the first round adds √2 ≈ 1.414 and √8 ≈ 2.828. If 2 still wins, the second round adds approximately 1.682 and 2.378 between it and its new neighbours. If the winner moves, the next round follows the updated scores. A two-round setting stops after two rounds; an unchanged winner does not trigger unlimited refinement.

### Refine to target time gap

Select **Refine to target time gap**, after the three-round option. **Minimum time unit (minutes)** appears, defaulting to **5 minutes**, with a supported range of 0.1–2880 minutes.

The same midpoint rule is applied, but only to current winner-neighbour intervals whose **complete sailing-duration difference exceeds the target**. Winners are reselected after each round until all currently eligible refinement intervals meet the target. Duration includes transfers and depends on vessel speed.

Five minutes is not calculation runtime, sampling interval or unused budget. It controls local duration spacing near winning scales, not spacing across the entire grid. A smaller target generally costs more computation and does not establish objective convergence or continuous global optimality.

Results report completed rounds, target and largest remaining neighbour gap. Having no eligible neighbouring interval is reported separately and is not labelled target attainment. Exceeding 2000 candidates, or failing to meet the target within 30 rounds, produces an explicit error. Reduce families, orientations or seed count, narrow the scale range, or increase the target gap before rerunning.

<a id="profiles"></a>
## 7. Background plus historical profile simulation

### Start with the South China Sea example

1. In **Survey design → Sound-field model**, select **Background + historical profiles**.
2. Choose **Load South China Sea example**. This replaces the two profile inputs and date split without changing array, range or route settings. **Download example files** only downloads a ZIP; it does not change inputs.
3. Check the counts: 182 training days, a 14-day gap and 170 validation days in 2020. Training end and validation start are inclusive; dates between them are unused.
4. Initially keep the profile multiplier and ρ at 1, select one held-out date, and set the detailed solution date.
5. Select **Check profiles**, review depth coverage and dates, then check survey settings and run.
6. Once familiar with the outputs, choose **Every held-out date** if needed. Work grows with distinct selected routes and validation dates.

The historical example contains background-aligned reanalysis daily profiles, with documented processing and deep assumptions, not daily CTD measurements. The split is retrospective. See the [example documentation](../examples/south_china_sea_profiles/README.md) for sources and processing.

### Supply your own two CSVs

Background fields are `depth_m,sound_speed_mps`. Historical fields are `date,depth_m,sound_speed_mps`, using `YYYY-MM-DD` dates. Depth is positive downward in metres; speed is in m/s. Prepare compatible regional profiles and reference conventions: historical minus background defines the anomalies.

Both inputs must start at depth zero and have sufficient common coverage to at least 100 m below the deepest beacon. Every input depth knot is retained in a common piecewise-linear grid. There is no automatic smoothing, decimation or deep extrapolation. Duplicate date/depth pairs, missing values and other invalid inputs are rejected. Each file is limited to 16 MiB; see the [model document](model.md) for further data-size limits.

A manual upload replaces only its corresponding input. Check file names and source labels. Statistics update automatically when files or dates change and count profiles actually present, rather than just calendar days.

### How validation works

Only training profiles enter risk modelling, scoring and selection; validation errors do not select replacement routes. Each test day's profile is static throughout the complete survey, and compared routes use the same daily profile. Identical selected route/date jobs are fitted once and reused.

All candidates retain design scores, but only selected routes undergo held-out fitting; A/B/C alternatives are also fitted on the detail date. Other candidates are labelled **Design score only**. A failed selected route retains its selection and reason instead of being replaced by a successful alternative.

<a id="field"></a>
## 8. Joint calibration of field data

### Prepare three files

- GNSS: `Time_UTC,Latitude_deg,Longitude_deg,Altitude_m`, optionally with `Pitch_deg,Roll_deg,Heading_deg` in degrees. Altitude refers to the GNSS antenna.
- Travel times: `Time_UTC,Beacon_ID,Delay_s`. Each row addresses one beacon; delay is in seconds. Use the app's one-way-equivalent convention. Do not halve data already on that convention.
- Sound-speed profile: for example `Depth_m,Velocity_mps`; the example aliases `Depth (m),SV (m/s)` are also accepted. Depth is positive downward and speed is in m/s.

Use UTF-8 CSV with headers and preserve UTC and gaps. GNSS must cover the acoustic observation times. Joint and same-model single-beacon modes require valid time alignment, without index substitution or navigation extrapolation. Attitude rotates the lever arm; attitude angles alone do not constitute full inertial-navigation data.

### Run the bundled field example

1. Open **Field data** and manually upload `gnss.csv`, `delay.csv` and `ssp.csv` from `examples/large_circle_cross/`.
2. Review previews and detected IDs, then select `2,5,6`. IDs may be comma- or space-separated; leaving the field blank uses all detected IDs.
3. Keep the default **Joint array calibration**. For this example, use RFU lever arm [1.585, 6.716, −29.05] m.
4. Check timing and GNSS noise. Current field defaults are 0.5 ms and 0.05 m.
5. Select **Process field data**, inspect coordinates, internal precision, residuals, fitted parameters and statuses, then export.

The example contains 4380 acoustic observations of three beacons. Circle and cross portions come from separate periods, not one continuous voyage. The [example README](../examples/large_circle_cross/README.md) records provenance, timing and settings.

### Joint versus single-beacon fitting

The default joint model estimates free 3D coordinates for N beacons, shared sound-speed offset, common effective delay, linear delay drift and N−1 relative delays: 4N+2 parameters. B5 is the relative-delay reference when selected; otherwise the lowest selected ID is used.

**Single-beacon fitting** uses the same propagation and estimation model independently for each selected beacon. Other displayed choices include TOA and ESKF approaches with different parameterizations and output meanings. Use them for comparisons appropriate to their models rather than treating every parameter as equivalent to a joint-mode output.

Uploaded sound-speed values are used at full resolution without smoothing or decimation. Below the deepest supplied point, the current field ray tracer applies a 0.016 (m/s)/m linear extension. This is a model assumption: provide adequate measured depth where possible, and distinguish it from profile simulation, which does not extrapolate automatically. Joint and same-model single-beacon fits do not automatically remove observations based on fitted residuals; prepare quality-controlled inputs.

<a id="results"></a>
## 9. Interpreting results

### Two different sets of labels

C1/C2/C3 define the score. A/B/C define the score–duration choice within the selected criterion. With C3 selected, for example:

- A chooses the eligible route with the lowest design score within the current budget.
- B chooses the shortest route among candidates with scores at or below the eligible median.
- C equally weights score and duration after separate min–max normalization over eligible candidates.

Plan C is therefore not criterion C3. Check the selected criterion, budget and saved radius before interpreting a plan.

### Feasibility, fitting and errors

Design eligibility and validation fit success are recorded separately. Munk recommendations are fixed before fitting design-feasible candidates. A failed recommended fit retains its recommendation and reason. Budgets without eligible routes remain visible.

Simulation has coordinate truth, allowing coordinate-error reporting. Profile-comparison MSE averages squared errors over 3N coordinate components in m²; the corresponding per-beacon 3D RMS is √(3 × MSE). Means over successful dates are conditional when failures occur. Paired C1–C3 results additionally require both routes to succeed on the same dates. One realization or date does not establish long-term statistical performance.

Field observations usually lack independent coordinate truth. A timing residual is observed time minus the complete fitted prediction, displayed in ms, not a coordinate error. Coordinate standard deviations express internal precision conditional on model and noise assumptions, not independently established absolute accuracy. Small residuals alone do not establish strong geometry.

### Track displays

Field tracks switch between raw points and smoothed curves. Smoothing changes display only, not observations or positions used in fitting. Faint dashed links indicate navigation gaps and do not imply observations within those gaps.

<a id="export"></a>
## 10. Exporting and saving

After calculation, use **Export results** to select all, top 10 or custom results. Include track coordinates, observations and residuals as needed; field and applicable profile results can include original inputs. ZIPs always contain summary CSV, settings and JSON, with a folder per selected result. Simulation also offers run JSON and current-plan observation CSV downloads.

Simulation top 10 uses the saved criterion, selected budget and GDOP condition. Validation failures do not replace design recommendations. **All** retains failed candidates and reasons. In profile mode, daily trials and detail-date observations are stored separately; unvalidated candidates are not presented as completed fits.

Field top 10 sorts successful beacon results by residual RMS within the uploaded track. It is not a trajectory-C3 or absolute-position-accuracy ranking. Original input files, when included, remain complete, including rows for unselected beacons.

Exports use saved results without rerunning calculations. Settings retain units, seed and applicable input-source information. CSV/JSON field names are unchanged by language switching. Keep the full ZIP rather than screenshots alone, and download before restarting the calculation service.

<a id="troubleshooting"></a>
## 11. Troubleshooting

**The page opens but calculations fail:** check that the service on port 8000 is running and inspect `.run/` logs. Double-click `start_all.bat` to detect and restart existing SeaArrayCalib services. If another application occupies a port, identify that application before freeing the port. Export any results you need before restarting.

**No eligible route at a budget:** inspect candidate reasons, then check complete duration, scale, usable range, ray support, per-beacon coverage and GDOP. Change physical limits only when consistent with the actual equipment and task.

**Refinement is slow or exceeds the candidate limit:** reduce families, orientations or seeds, narrow the bounds, or use fewer fixed rounds/a larger target gap. Zero rounds still includes budget-boundary candidates, so total count may exceed the displayed seed count.

**The best scale does not change after the first of two rounds:** the second round still runs. Old points remain. Target-gap mode stops according to neighbour spacing, not whether the winner's ID changes.

**Profile validation fails:** check both files, headers, dates and split, duplicate depths, sea-level start and coverage to 100 m below the deepest beacon. Both training and validation need usable profiles; inspect counts rather than calendar ranges alone.

**Field times do not align:** check UTC/time zones, duplicate or unsorted epochs and GNSS coverage. Do not renumber time or invent navigation points to conceal real gaps.

**Fits fail or observations are missing:** retain and inspect the reported reasons. Range, ray support, coverage, geometry and numerical acceptance are distinct. A numerical acceptance failure alone does not determine the magnitude of coordinate error.

**Changing ρ does not change validation anomalies:** this is expected. Radius changes design risk. To test different physical amplitudes, change the profile anomaly multiplier or the relevant Munk field parameters.

**An old task cannot be exported after restart:** results were in service memory. Use the ZIP downloaded before restart; rerunning creates a new task.

<a id="scope"></a>
## 12. Suggested workflow and model scope

Start with a small Munk search, then configure equipment and regional conditions. With background and historical data, check the split and run one date at multiplier 1 and ρ = 1 before evaluating the required date range. For field data, check units, UTC, lever arm and profile depth before using the default joint fit.

SeaArrayCalib searches a finite candidate pool and evaluates local-model risk under declared perturbation assumptions. Profile simulation uses a static 1D profile for each complete survey; Munk mode uses prescribed analytic perturbations. Outputs do not automatically establish a real 3D ocean, equipment performance or independent field accuracy. For scientific comparisons, fix conditions, retain failures and full costs, and distinguish design scores, simulation truth errors and field internal metrics.

The software is MIT-licensed; bundled reanalysis information retains [its source terms and attribution](../examples/south_china_sea_profiles/README.md#source-and-attribution). See the [numerical workflow](model.md) for detailed definitions and limits.
