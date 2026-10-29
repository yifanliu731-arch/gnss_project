# Reproducing the paper simulation

The repository includes a command-line protocol for the paper's main synthetic
comparison, in addition to the interactive application. It runs from this
repository alone, using the two existing public profile CSV files. It does not
need another research checkout, raw CTD files, or an external reanalysis download.

Install the Python dependencies from the repository root, then run:

```sh
python -X utf8 scripts/reproduce_paper.py smoke
python -X utf8 scripts/reproduce_paper.py all
```

`smoke` checks the first held-out day for the nominal-selected rose, risk-selected
rose, and four complete station-centered circles. `all` recomputes all 702 frozen
geometry scores, checks the 30 selections at ten sailing-time caps, and checks
510 route/day attempts for the main native-amplitude comparison. `scores` and
`daily` run these two parts separately. The full calculation takes substantially
longer than the smoke check. `--workers 1` reduces concurrent memory use.

The protocol fixes the 6 km × 6 km four-beacon array, 3 kn vessel speed, 0.5 ms
one-way acoustic noise, 0.05 m navigation noise, and original route/day random
streams. The profile split contains 182 training days, a 14-day gap, and 170
evaluation days. The uncentered training second moment has rank 32. Training
profiles determine design scores; evaluation profiles do not select routes.

The JSON file `examples/paper_protocol.json` contains synthetic geometry
configurations and numerical regression references, not additional raw field
observations. The original 702-candidate pool is fixed because later interactive
search refinements can generate a different pool. Re-running today's interface
with similar settings is therefore not an exact replay of that historical pool.

Daily validation retraces each full survey through its prescribed profile,
preserves unsupported arrivals and failed fits, and draws noise on the complete
planned observation grid before masking arrivals. Transfers in the four-circle
baseline count toward sailing time and have no acquisition. All 18 joint
parameters, including the 12 coordinate components, remain free.

Checks compare eligibility, selections, and coordinate errors with frozen
references. The coordinate tolerance is 0.00001 m; score tolerances are recorded
in the protocol. Outputs include a source/input hash manifest and JSON reports
under `.cache/paper-reproduction/`, which is ignored by Git. A changed source or
input requires a fresh `--output` directory so an old cache cannot certify a new
implementation. RMSE comparisons use shared successful dates and explicitly
state when they are conditional on successful fits.

The public measured example covers the original large-circle-plus-cross case.
The complete twelve-case field archive and all supplementary simulation
experiments are outside this command's scope. Other field observations remain
available by reasonable request to Fan Gao (gaofan@sdu.edu.cn), as described in
the manuscript. Do not interpret a passing build or optimizer convergence as
independent field-position accuracy.

# 论文仿真复现

在项目根目录安装依赖后，执行上述 `smoke` 可核对三条主要对照航迹的首个验证日；
执行 `all` 会重算冻结的702条候选、十档预算的30项选择，以及三条主要对照航迹的
170日结果，共510次尝试。输入直接使用仓库已有的背景和历史剖面CSV，不需要私有研究目录。

固定候选池、随机种子、完整航时、漏测、拟合失败和共同有效日期均保留。输出在被Git忽略的
`.cache/`中。配置或计算源码改变后须指定新的输出目录，不能沿用旧结果冒充通过。
公开的大圆加十字样例可用于实测流程验证；其余完整十二组实测资料仍按论文声明联系高凡获取。
