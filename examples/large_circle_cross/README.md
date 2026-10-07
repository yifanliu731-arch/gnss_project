# Large circle + cross / 大圆加十字

Upload `gnss.csv`, `delay.csv` and `ssp.csv` in **Field data / 实测数据**. Select beacon IDs **2,5,6**, the default **joint array calibration** solver, and the RFU lever arm **[1.585, 6.716, -29.05] m**. GNSS attitude columns describe measured angles; no raw gyro/accelerometer stream is present, so the app applies attitude to the lever arm without running an inertial filter.

本例为大圆加十字组合的 4,380 条声学观测，包含三个信标。三个文件分别上传到 GNSS、时延、声速剖面入口；信标填 2,5,6，使用默认多信标联合求解，杆臂右/前/上为 1.585、6.716、−29.05 m。UTC 跨越 2021-05-01 至 2021-05-03，十字和圆形来自不同时段，不能理解为一次连续走航。

`delay.csv` contains accepted measured observations from the large-circle-plus-cross combination. `gnss.csv` contains original navigation rows around those observations, with measured latitude/longitude/altitude and attitude. `ssp.csv` is the measured background profile. Timestamps remain UTC; delays remain on the original one-way scale (do not halve them). No missing segment has been synthesized. Provenance and hashes are in `provenance.json`.

The default field solver fits the three beacon coordinates and shared acoustic parameters jointly (14 parameters); optional single-beacon mode fits each beacon separately with the same propagation/MAP model. Use this example to check file uploads, full-resolution profile previews, lever-arm settings, joint results and export. Default noise settings are 0.5 ms timing and 0.05 m GNSS; internal precision is conditional on these settings, not independently verified coordinate accuracy. Separate observation periods and time-correlated residuals should be considered when interpreting the fit.

This example is distributed under the repository's MIT license.

The app uses the uploaded SSP values without smoothing or decimation. The example SSP is the existing background profile, copied byte-for-byte; it is not processed again by the web app.
