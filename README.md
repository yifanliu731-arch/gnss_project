# SeaArrayCalib

A local web application for **GNSS-A survey planning and joint seafloor beacon calibration**. Compare complete survey trajectories within a sailing-time budget, evaluate prescribed sound-field risk, and estimate beacon coordinates from simulated or measured observations.

**GNSS-A 航迹规划与海底信标联合标定工具。** 支持中英文界面；运行和数据处理都在本机完成。

- [中文使用说明](docs/user-guide.zh-CN.md)
- [English user guide](docs/user-guide.en.md)
- [Model and physical conventions](docs/model.md)
- [Input CSV formats](docs/data-format.md)
- [Paper simulation reproduction / 论文仿真复现](docs/reproducibility.md)

## Capabilities

- Plan six complete trajectory families: circle, figure eight, clover, square, four-petal rose, and circle plus cross. Include transfer distances in sailing time.
- Configure beacon layouts, lever arms, vessel speed, acoustic range, sampling, noise, and ten logarithmically spaced time budgets.
- Compare C1 nominal variance, C2 average coordinate risk, and C3 worst-case coordinate risk under the specified perturbation model.
- Simulate a Munk profile or upload a background profile with dated historical profiles; separate risk training from held-out validation.
- Process measured GNSS, acoustic travel-time, and sound-speed CSV files. Joint and single-beacon processing are available; the user guides describe each solver.
- Inspect tracks, coverage, diagnostics and residuals; export saved settings, summaries and selected results. Failed candidates and fits retain their reasons.

Recommendations are fixed before validation. Switching language preserves inputs and results. Export results before restarting: jobs and uploaded data are held in API memory for the current session.

## Installation

Use **Python 3.12** and **Node.js 22.12 or later in the Node 22 series**. Download or clone this repository, then open a terminal in its root directory.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
npm ci
```

Double-click **start_all.bat**, then open **http://127.0.0.1:5173/**. The launcher checks ports 8000 and 5173, restarts identified previous SeaArrayCalib services, and waits for both services. It leaves unrelated port owners running. Logs are in `.run/`.

Double-click **stop_all.bat** to stop the services. The launcher does not install dependencies automatically.

中文：首次使用先执行上述安装步骤，以后双击 `start_all.bat` 启动、`stop_all.bat` 关闭；重启前请导出需要保留的结果。

### macOS / Linux

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm ci
```

Start the backend and frontend in two terminals:

```sh
# Terminal 1
.venv/bin/python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
# Terminal 2
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Stop each service with Ctrl+C. The backend address can be configured with frontend `VITE_API_URL`; allowed browser origins use backend `CORS_ORIGINS`.

## First run and example data

For simulation, enter the array and operating conditions, choose a sound-field mode and trajectory families, and run the calculation. See the user guide for units, risk radius, refinement settings and interpretation.

For measured data, use the [large-circle-plus-cross example](examples/large_circle_cross/README.md): upload `gnss.csv`, `delay.csv` and `ssp.csv`, and select beacon IDs **2, 5, 6**. This example preserves original UTC across separate observation periods; it is not one continuous voyage.

The [South China Sea profile example](examples/south_china_sea_profiles/README.md) provides `background.csv` and `history.csv`. It contains processed profiles, not two raw CTD casts. The interface can load or download these files. Their provenance, date split and upstream attribution are documented alongside them.

There are **seven data CSVs** in the repository: three measured-upload files, two processed profile files, and the two numerical-library templates. The template SSP duplicates the measured SSP. No other field-observation archive is included.

## Reproduce the main paper simulation

After installing the Python dependencies:

```sh
python -X utf8 scripts/reproduce_paper.py smoke
python -X utf8 scripts/reproduce_paper.py all
```

The smoke check replays three first-day cases. The full command recomputes the 702 fixed geometry scores, 30 budget selections, and 510 main-comparison route/day attempts, including retained failures. It uses the existing profile CSVs and writes reports under the ignored `.cache/` directory. See [scope, configuration and numerical checks](docs/reproducibility.md).

The public measured example supports a field-processing demonstration. The full twelve-case field archive is available upon reasonable request to the corresponding author as described in the manuscript.

## Project structure

```text
src/                       Vue interface, plots and translations
public/                    Browser assets
backend/                   FastAPI, solvers, propagation and regression tests
vendor/seaarray_core/       Bundled numerical primitives and integrity manifest
examples/                  Approved CSV examples, provenance and replay protocol
docs/                      User guides, model, data formats and reproduction
scripts/                   Core verification, replay and service helpers
.github/workflows/         Automated build and numerical checks
start_all.bat              Windows start entry
stop_all.bat               Windows stop entry
start_web.ps1, stop_web.ps1 Service launch/stop implementation
```

`node_modules/`, `dist/`, `.venv/`, `.run/` and `.cache/` are local dependency, build or runtime directories and are excluded from Git. The application includes its numerical dependencies and does not require a separate research checkout.

## Development and verification

```sh
python -m pip install -r requirements-dev.txt
python scripts/sync_core.py
python -X utf8 -m unittest discover -s backend -p "test_*.py"
node --experimental-strip-types scripts/test_track_segments.mjs
npm run build
```

The core manifest verifies exact file bytes and executable AST fingerprints. `scripts/sync_core.py --source PATH` additionally verifies the recorded upstream revision. English comments and explicitly retained configuration exports are documented adaptations; numerical changes require review and new validation. See [CONTRIBUTING.md](CONTRIBUTING.md).

A successful fit or build does not establish independent field-position accuracy. The model guide explains assumptions, supported propagation branches, uncertainty and validation limits.

## License and citation

The software and measured track example use the [MIT license](LICENSE). Derived Copernicus profiles retain their [upstream data terms and attribution](examples/south_china_sea_profiles/README.md#source-and-attribution).

When using the software, cite **SeaArrayCalib** and record the exact version or commit. Machine-readable metadata is in [CITATION.cff](CITATION.cff).
