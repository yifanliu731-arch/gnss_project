# CSV inputs

SeaArrayCalib: [中文使用说明](user-guide.zh-CN.md) · [English user guide](user-guide.en.md).

Use UTF-8 CSV with headers. Example files are under `examples/large_circle_cross/`.

- GNSS: `Time_UTC,Latitude_deg,Longitude_deg,Altitude_m`; optional measured `Pitch_deg,Roll_deg,Heading_deg` (degrees). Use explicit UTC (`Z`), sorted unique epochs and the antenna altitude. RFU lever arm is in metres. Attitude alone rotates the lever arm; raw gyro and accelerometer columns are required for INS fusion.
- Delay: `Time_UTC,Beacon_ID,Delay_s`. One row is one beacon observation; a missing out-of-range beacon does not invalidate the other beacons at that time. Use the supplied one-way travel-time convention; do not divide it again. Select the IDs to solve in the UI.
- SSP: `Depth (m),SV (m/s)` (the example uses accepted aliases). Depth is positive down; speed is metres per second. Supply a profile covering the required depth.

Keep original time gaps and clock conventions. Do not concatenate separate campaigns by renumbering their time. The sample's GNSS brackets every accepted acoustic epoch. Field-data processing defaults to **joint array calibration** of all selected beacons, with shared acoustic parameters. **Single-beacon fitting** is an optional mode that fits each selected beacon independently using the same propagation and estimation model. Blank beacon IDs use all IDs detected in the travel-time file. Simulation exports use ENU coordinates with negative U below sea level. CSV/JSON keys remain unchanged when switching interface language.

实测数据默认采用**多信标联合标定**，同时估计所选信标坐标及共享声学参数；**单信标拟合**是可选模式，使用相同传播与估计模型逐个独立求解。信标编号留空时使用走时文件中检测到的全部编号。保留原始 UTC 和观测缺口，不通过重新编号时间拼接不同航次。

The app uses the uploaded SSP values without smoothing or decimation. The example SSP is the existing background profile, copied byte-for-byte; it is not processed again by the web app.

## Profile-driven simulation / 剖面驱动仿真

These are two different inputs from the three-file field workflow above:

- Background: `depth_m,sound_speed_mps`.
- Historical profiles: `date,depth_m,sound_speed_mps`, with dates in `YYYY-MM-DD` format.

Use UTF-8 CSV. Profiles must start at depth zero and share sufficient coverage to at least 100 m below the deepest beacon. Duplicate date/depth pairs and missing or nonphysical values are rejected. All supplied knots are retained in the common piecewise-linear grid; no automatic deep extrapolation is applied in this simulation mode. Each file is limited to 16 MiB. See the [profile example](../examples/south_china_sea_profiles/README.md) and [numerical workflow](model.md) for further limits and assumptions.

剖面仿真需要“背景”和“带日期的历史剖面”两份文件，与实测标定的三份文件不同。深度向下为正，单位米；声速单位 m/s。所有剖面须从海面覆盖到最深信标以下至少100 m，仿真不会自动向深处外推。先使用页面中的“检查剖面”核对资料及日期划分。
