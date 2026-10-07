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
- C1 / C3 comparisons, candidate refinement, plots and CSV/JSON exports.
- Measured GNSS, travel-time and SSP upload with automatic beacon-ID discovery. Joint array calibration is the default; single-beacon fitting uses the same curved-ray/MAP model independently for each selected beacon. Legacy TOA/ABIC/ESKF solvers remain available.
- Field tracks switch between smoothed curves and unconnected points. Curve smoothing is display-only within each segment; faint, thin, widely spaced dashed links mark navigation gaps. Joint/single-model residual plots identify beacons and UTC, report RMS, and use observed minus fitted travel time in milliseconds.

See [numerical workflow](docs/model.md) for assumptions, parameter definitions and limits.

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

[MIT](LICENSE), including the bundled example. No remote repository is required for local use.

## Result export and speed units

Survey design defaults to 3 kn (approximately 1.5433 m/s). Select kn or m/s to enter speed; switching units preserves the underlying physical speed.

After a calculation, **Export results** offers all results, the top 10, or a custom selection, with optional track coordinates, observations and residuals. A ZIP contains summary CSV, saved settings, JSON and one folder per selected result. Simulation top 10 uses the saved C1/C3 criterion, the currently selected budget and the saved GDOP limit. All includes failed candidates. Field processing operates on one uploaded track: its top 10 is per-beacon residual RMS, not trajectory C3 or independent position accuracy. Original uploaded field files are optional and remain complete even with a custom beacon selection. Exports use saved data, without rerunning fits. Download before restarting the API; results are held in memory for the current server session.
