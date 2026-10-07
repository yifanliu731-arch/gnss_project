# SeaArray · Survey Design

## Quick start / 一键启动

**Windows:** after the first-time setup below, double-click **`start_all.bat`**. It starts the API and website at <http://127.0.0.1:5173/> and writes logs to `.run/`. The launcher keeps the console free; occupied ports are reported. It never installs dependencies automatically.

**首次使用：**安装 Python 3.12 和 Node.js 22（22.12 或更新），在项目目录执行：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
npm ci
.\start_all.bat
```

The app is self-contained. The launcher prefers this project's `.venv`; an available local development environment or Python on PATH can also be used. To stop, run `powershell -ExecutionPolicy Bypass -File stop_web.ps1`.

### macOS / Linux / manual development

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm ci
# Terminal 1
.venv/bin/python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
# Terminal 2
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

## What it does

A Vue / FastAPI application for configurable multi-beacon GNSS-A survey design. Switch **中文 / EN** in the header without losing inputs, uploaded files or results.

- General-N joint simulation, curved-ray propagation, serial acoustic scheduling and per-beacon range rejection.
- Six complete track families: circle, figure eight, clover, square, four-petal rose and circle plus cross. The first five are closed curves. Circle plus cross runs one circle and two diameters joined by a straight transfer; it ends at the second diameter without a forced return. Transfer time is counted, while transfer sampling is off by default. No centre-return legs or extra laps are added.
- User-defined scale limits, speed, array, risk assumptions and minimum/maximum time budgets. Ten logarithmically spaced caps include both endpoints; complete tracks are selected within each cap.
- Rotation uses **180° divisions**: 1 → 0°; 2 → 0°/90°; 3 → 0°/60°/120°; 4 → 0°/45°/90°/135°.
- C1 / C2 / C3 selection: nominal variance, average coordinate risk and worst-case coordinate risk under a declared perturbation set. Candidate refinement uses all three criteria on one shared pool.
- Optional two-file sound-field simulation: measured background plus dated same-region historical profiles; training/gap/validation split; 0.5×, 1×, 2× and 5× anomaly amplitudes; exact held-out profile ray tracing and comparison of the frozen selections. The default Munk demonstration remains available.
- Measured GNSS, travel-time and SSP upload with automatic beacon-ID discovery. Joint array calibration is the default; single-beacon fitting uses the same curved-ray/MAP model independently for each selected beacon. Legacy TOA/ABIC/ESKF solvers remain available.
- Field tracks switch between smoothed curves and unconnected points. Curve smoothing is display-only within each segment; faint, thin, widely spaced dashed links mark navigation gaps. Joint/single-model residual plots identify beacons and UTC, report RMS, and use observed minus fitted travel time in milliseconds.

See [numerical workflow](docs/model.md) for assumptions, parameter definitions and limits.

## Try profile-driven simulation / 声场扰动与留出验证

In **Survey design → Sound-field model**, select **Background + historical profiles** and click **Load South China Sea example**. The [two documented CSVs](examples/south_china_sea_profiles/README.md) contain the derived fixed background and all 366 dated profiles from 2020. Use **Download example files** for manual uploads. Input formats are `depth_m,sound_speed_mps` and `date,depth_m,sound_speed_mps` (ISO dates).

The sample separates 182 training days, a 14-day gap and 170 validation days. Start with one held-out date, or select all held-out dates for a longer run. Choose C1, C2 or C3 for the displayed optimal plan; the result also compares the three frozen selections at each time cap. Only training data guide scores and selection. Identical selected routes are fitted once per date and reused. Failures and missing arrivals remain visible; MSE on successful dates is explicitly conditional when any fit fails. This software example is a retrospective holdout, not a never-inspected confirmatory dataset.

File statistics update automatically after an upload or a date-setting change. Counts reflect dated profiles actually present, not the calendar span. Depth knots and maximum depth come from the supplied profiles. Loading the example replaces both files and the split settings; a manual upload replaces only its corresponding file. The page labels each input's source. Downloading the example leaves the current inputs unchanged.

中文：先选择“背景＋历史剖面”，加载内置南海资料或上传自己的两份 CSV；设定建模、验证日期和扰动倍数，再用右侧现有航迹、设备量程和预算设置运行。样例只加载资料，不更改这些设置。C2/C3 均考虑声场扰动：C2 关注平均风险，C3 关注规定椭球内最坏方向，二者不保证在所有测试中给出相同或最优结果。每日剖面在一次航次内保持静态；资料来源、背景对齐及深层处理见样例说明。

## Try measured data / 实测样例

The [large-circle-plus-cross example](examples/large_circle_cross/README.md) includes exactly the three upload CSVs: [GNSS](examples/large_circle_cross/gnss.csv), [delay](examples/large_circle_cross/delay.csv), [SSP](examples/large_circle_cross/ssp.csv). Use beacon IDs **2,5,6**. The example preserves original UTC across separate observation periods; it is not a single continuous voyage. See its README for lever arm, settings and provenance.

## Repository layout

```text
src/                     Vue interface and translations
backend/                 API, numerical workflow and regression tests
vendor/seaarray_core/     Bundled numerical primitives + source hashes
examples/                Small documented measured-data example
docs/                    Model and data-format documentation
scripts/                 Numerical-module verification
.github/workflows/       Build and regression checks
start_all.bat            Windows one-click entry
start_web.ps1            Local service launcher
stop_web.ps1             Stop services started by that launcher
```

## Development checks

```sh
python -m pip install -r requirements-dev.txt
python scripts/sync_core.py
python -m unittest discover -s backend -p "test_*.py"
npm run build
```

The bundled numerical modules are tracked by a source-hash manifest. Run `scripts/sync_core.py` to verify their integrity. Numerical changes require regression checks and updated validation.

Alternate service addresses use frontend `VITE_API_URL` and backend `CORS_ORIGINS`. This is a local scientific tool; the launcher binds to loopback. CI and builds do not establish experimental validity. See [contribution guide](CONTRIBUTING.md).

## License

[MIT](LICENSE) for the software and the measured track example. The derived Copernicus profile example retains its [upstream data terms and attribution](examples/south_china_sea_profiles/README.md#source-and-attribution). No remote repository is required for local use.

## Result export and speed units

Survey design defaults to 3 kn (approximately 1.5433 m/s). Select kn or m/s to enter speed; switching units preserves the underlying physical speed.

After a calculation, **Export results** offers all results, the top 10, or a custom selection, with optional track coordinates, observations and residuals. A ZIP contains summary CSV, saved settings, JSON and one folder per selected result. Simulation top 10 uses the saved C1/C2/C3 criterion, the currently selected budget and the saved GDOP limit. All includes failed candidates. In profile mode, all candidates have training-stage design status, but only selected plans have held-out fits. Detailed observations refer to the chosen detail date; validation trials and comparisons are separately exported, with the input CSV text and hashes in the saved settings. Field processing operates on one uploaded track: its top 10 is per-beacon residual RMS, not trajectory C3 or independent position accuracy. Original uploaded files can be included separately. Exports use saved data, without rerunning fits. Download before restarting the API; results are held in memory for the current server session.
