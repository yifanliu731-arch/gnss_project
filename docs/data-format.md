# CSV inputs

Use UTF-8 CSV with headers. Example files are under `examples/large_circle_cross/`.

- GNSS: `Time_UTC,Latitude_deg,Longitude_deg,Altitude_m`; optional measured `Pitch_deg,Roll_deg,Heading_deg` (degrees). Use explicit UTC (`Z`), sorted unique epochs and the antenna altitude. RFU lever arm is in metres. Attitude alone rotates the lever arm; raw gyro and accelerometer columns are required for INS fusion.
- Delay: `Time_UTC,Beacon_ID,Delay_s`. One row is one beacon observation; a missing out-of-range beacon does not invalidate the other beacons at that time. Use the supplied one-way travel-time convention; do not divide it again. Select the IDs to solve in the UI.
- SSP: `Depth (m),SV (m/s)` (the example uses accepted aliases). Depth is positive down; speed is metres per second. Supply a profile covering the required depth.

Keep original time gaps and clock conventions. Do not concatenate separate campaigns by renumbering their time. The sample's GNSS brackets every accepted acoustic epoch. Upload fitting is per beacon. Simulation exports use ENU coordinates with negative U below sea level. CSV/JSON keys remain unchanged when switching interface language.

The app uses the uploaded SSP values without smoothing or decimation. The example SSP is the existing background profile, copied byte-for-byte; it is not processed again by the web app.
