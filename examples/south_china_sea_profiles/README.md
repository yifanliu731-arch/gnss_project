# South China Sea profile example / 南海声速剖面测试资料

In **Survey design → Sound-field model → Background + historical profiles**, choose **Load South China Sea example**. This loads the two CSVs and the 182 / 14 / 170 calendar split without changing your array, equipment range or route settings. **Download example files** provides the same inputs for manual uploads.

网页“航迹设计 → 声场模型 → 背景＋历史剖面”中可一键加载，或下载后分别上传两个 CSV。样例不会更改阵列、设备量程和选轨设置。默认只检验一个留出日期，适合先确认功能；选择“全部留出日期”会对每档预算冻结的 C1/C3 方案逐日检验，耗时更长。0.5、1、2、5 为扰动幅度倍数。

## Files and interpretation

- `background.csv`: `depth_m,sound_speed_mps`. A fixed mean background constructed from two measured CTD casts and 185 background-aligned regional profiles (2021-04-15 through 2021-05-21, five longitudes). It is a derived background, not an unmodified single CTD cast.
- `history.csv`: `date,depth_m,sound_speed_mps`. All 366 daily profiles in 2020, at the central grid point near 18.75°N, 116.500015°E. These are **background-aligned equivalent profiles**, not raw absolute GLORYS sound speed.
- `provenance.json`: source, processing, split and file SHA-256 hashes.

The historical file is constructed as:

`aligned_history(date,z) = background(z) + taper(z) × [GLORYS_2020(date,z) − mean(GLORYS_2021_37days(center,z))]`.

Subtracting the uploaded background therefore recovers the intended year-to-year anomaly. The taper is one through 3000 m, decreases with a cosine to zero at 3200 m and stays zero below. The fixed background below the measured 3242 m support uses an explicitly assumed 0.016 (m/s)/m extension. The supplied calculation grid has 376 points from 0 to 3750 m in 10 m steps; this is a prepared numerical grid, not the original observation sampling.

Training: **2020-01-01–06-30, 182 days**. Gap: **07-01–07-14, 14 days**. Validation: **07-15–12-31, 170 days**. The archive was explored before this split: this example is a **retrospective time holdout**, with validation values excluded from fitting and route selection, not a never-inspected confirmatory dataset. Dates are temporally related. Each daily profile remains static during its simulated survey and does not specify subdaily dynamics or a 3D ocean.

The risk basis uses the uncentered second moment of training anomalies, preserving their mean offset. Its unit ellipsoid is a declared design scale, **not** an empirical maximum or a 95% confidence region. The web applies your selected route search, array, range, budget and random seed; changing those settings changes the results. A quick software test does not reproduce an entire controlled experiment.

## Source and attribution

Generated using E.U. Copernicus Marine Service Information; [Global Ocean Physics Reanalysis, DOI 10.48670/moi-00021](https://doi.org/10.48670/moi-00021).

[GLORYS12 product GLOBAL_MULTIYEAR_PHY_001_030](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description), dataset `cmems_mod_glo_phy_my_0.083deg_P1D-m`, version `202311`. Potential temperature and practical salinity were converted with TEOS-10 (`CT_from_pt`, `SA_from_SP`, `sound_speed`) at the profile pressure/location. Daily reanalysis is model-based information, not direct daily CTD measurements.

The derived Copernicus information retains the [Copernicus Marine licence and acknowledgement conditions](https://marine.copernicus.eu/user-corner/service-commitments-and-licence). The software's MIT licence does not replace upstream data terms. See [official citation guidance](https://help.marine.copernicus.eu/en/articles/4444611-citing-copernicus-marine-products-and-services).
