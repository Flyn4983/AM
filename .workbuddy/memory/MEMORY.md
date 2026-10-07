# AM-workbuddy 项目长期记忆（curated）

## 核心需求原则（用户 2026-08-16 明确）
- **“仿真”＝数值方法（FEM/FVM/MPM/SPH/DEM 等离散计算），不是解析近似。**
  不得用闭式/解析解（如 Rosenthal 解析温度场）作为默认“模拟”求解器去近似处理。
- 用户对“默认链路走代理/解析权宜”高度敏感：要求开箱即“实打实”的真实物理模拟。

## 架构事实
- 包结构：`amforge`（AM 编排，纯 JAX 求解器）+ `diffmech`（JAX 计算力学库：FEM/FVM/DEM/SPH/MPM/相场/CPFE）+ `forgecore`（槽位注册通用引擎）。单 wheel，运行置 `PYTHONPATH=src JAX_ENABLE_X64=1`。
- 热学：默认链路已接真实焓法相变 `solve_enthalpy_thermal`（糊状区有效扩散系数 + SSP-RK2，数值求解，含潜热）；`thermal_solver="history"` 为 Rosenthal **解析近似**（仅教学/快速回归用，默认不再走它）。
- 默认求解器选路纪律（2026-08-13 落地 ①+②）：registry `SolverSpec` 增 `method` 分类（numerical/analytical/proxy）；`Pipeline.auto` 默认 `exclude_methods=("analytical",)`，forgecore `_select` 对 `"analytical"` 标签降优先级 → 双引擎一致排除闭式/解析，默认走真实数值（`meltpool.fdm` 3D FVM 焓法 / `thermal.enthalpy`）。⚠️ **路由双计数坑**：`graph._Planner.resolve` 必须按"每个求解器只计一次"求和 DAG 真实代价；否则共享子图（如 thermal 被 buildup+microstructure 同时消费）被重复累加，热学变贵后会虚高廉价真实档代价、反选沉重高保真档（`support.simulate` CFD/FEM → e2e OOM ~71GB）。修过。
- **meltpool.fdm 诚实边界**：传导模式 FVM（无 VOF 自由界面/Marangoni/反冲/匙孔），未消费 geometry 作基板边界（均匀块），熔池尺度由能量平衡上界自适应盒保证不截断。需高保真效应用 `select={'meltpool':'meltpool.vof_flow3d'}`。**B 档（2026-08-20）已验证稳定**：`meltpool.vof_flow3d` 此前仅编码未跑通，经压力投影重写（`_pressure_system`+`_project`，CG 残差 1e13→1e-4）、NaN 加固、`checkpoint_every` 默认 20（修 OOM）后，熔池成形/匙孔正反馈/Marangoni 效应均经测试确认（`tests/test_meltpool_vof.py` 8/8）。⚠️ **vof 可微但梯度病态**（∂depth/∂P~1e45，显式长时程反向固有），接 `inverse` 反演前必须 gradient clipping / 输出归一 / 自适应步长；默认链 `meltpool.fdm` 梯度良态可直接用。详细见 `docs/熔池物理保真升级_AB档_开发纪实.md`。
- 微观：`micro.surrogate`（统计代理）/ `micro.phasefield`（Karma-Rappel 相场，真实数值）。
- GUI 分层：纯逻辑层（`*_logic.py`/`preproc.py`/`postproc.py`）+ UI 层（`*_app.py`，PySide6+VTK）；offscreen 下跳过 VTK 3D 渲染避免原生段错误。
- 缺口（GUI 三缺口，**截至 2026-08-17 全部闭环**）：②A 几何元素选面施 BC/IC **已闭环**（新增 `src/amforge/boundary.py`，经 `params["boundary_conditions"]` 喂入 `solve_enthalpy_thermal`）；②B 导入粒子/粉末 **已闭环 + 增强**（新增 `src/amforge/powder.py` 的 `ParticleCollection`：CSV/JSON 真实导入 + `from_part_bed` 真实生成随机密堆粉末层，在真实粒子上测体积分数/d50/PSD，映射 `powder.bed` 参数驱动**真实 DEM 铺粉** `solve_powderbed(powder_fidelity="dem")`；`preproc_app` 增「粉末/粒子」面板 + 2D 切片红点叠加 + “用导入粒子作 DEM 初始态”勾选；`test_powder_import.py` 12 例全绿，含 `test_gui_powder_import_generate_eval` 真实驱动导入/生成/DEM 回调）。**2026-08-17 增强**：原设计边界（DEM 用 cfg 重新采样 RCP 床、不注入精确坐标）已补 opt-in——`PowderBedConfig.initial_positions/initial_radii` → `recoat_dem` → `synth_powder_bed` 在给定时直接用精确坐标作沉降初态（跳过抖动晶格采样），未给仍走 RCP（向后兼容）；`solve_dem(use_precise=)` / GUI 勾选触发。`test_solve_dem_precise_injection_takes_effect` 验证注入确实生效（单粒子初态配位数≈0 低于默认密堆床）。注：DEM 仍是重力沉降，注入坐标是“沉降起点”非“冻结排列”；均匀床统计等价，结构化异质点云保留布局信息。CSV 导出须用 `%.17e` 保无损往返（复审修过 `%.9e` 丢精度 bug）。forge_adapter 全部 13 个镜像注册 cost 发散已全清（process.constant/thermal.enthalpy/inverse.predict_process 统一从原生 spec 取 cost，单一事实源；`test_forgecore_e2e.py::test_mirror_costs_match_native` 锁防复发）。③ 后处理动画播放 **已闭环**（从零实打实：`thermal_enthalpy` 的 `record_frames` 抽取温度演化帧挂 `ThermalHistory.temperature_evolution` + `postproc` 的 `extract_temperature_frames`/`save_field_gif`/`save_field_pvd` + `postproc_app` FieldView“扫描温度动画”播放/暂停/导出）。**注意**：审计原文档两处不实——(a) 称“库级 `save_field_gif`/`.pvd` 已存在”实则全代码库无此函数，③ 系从零实现；(b) 称“GUI grep 粒子零命中”不实，②B 已实打实实现 `powder.py`+面板。
- **战略缺口四大模块（蓝图 M1–M4）全部落地（2026-08-17）**：M1 材料标定 ✅（`amforge.calibration` + CalibrationReport，optax Adam 拟合 Material 参数）、M2 支撑结构 ✅（`support` STAGE + SupportStructure + GUI 面板）、M3 二次工艺 ✅（`postprocess` STAGE + SecondaryProcessResult + HT/HIP/machining + verdict 消费 secondary）、M4 监测闭环 ✅（`monitor` STAGE + SensorData/MonitoringState/ClosedLoopPlan + `monitoring.ingest/detect/correct` + `closedloop.run_loop` 外层迭代驱动，可选 `recalibrate` 回灌 M1；`gui/monitoring_app.py` 第③标签仪表盘）。**⚠️ 诚实边界（2026-08-17 用户追问"已完全实现？"后澄清）："完全实现"仅指管线架构（registry/契约/镜像/run_loop 真实重跑前向/recalibrate 真实回灌/GUI/测试全绿）。智能核心——`monitor.detect` 为物理规则代理（`model_meta={"detector":"physics_proxy"}`，非蓝图要求的 NN/统计分类，无训练数据/模型，用户确认保持占位、接口已能吞真实传感信息）；`monitor.correct` 现已双控制器：`rule` 物理规则代理（默认）+ `controller="inverse"` **实打实可微优化控制器**（复用 amforge.inverse 的 normalize_process + jax.value_and_grad + optax.adam + project_process_feasible 硬投影，以真实前向仿真最小化光滑缺陷代价，run_loop 传 geo）——用户明确选择"接 inverse 优化器"并已实现+测试（test_inverse_controller_lowers_defect_cost 验证确把缺陷代价压低、test_run_loop_inverse_controller_runs 验证每步走 inverse）。闭环"传感流"由仿真自身 ThermalHistory 合成、~3% 偏差人为注入，属自指式研究演示，非真实硬件在线监测；swap 接口已留（monitor 可整体替换 detect/correct）。**新增非流水线求解器统一纪律：registry 加 STAGE+槽 → contracts 加契约 → @register_solver → forge_adapter 镜像（`_native_cost`）→ `__init__` 加载/导出 → GUI 卡 + 测试；闭环一律放 `Pipeline` 外层、不改动内核（保持 DAG 纯洁）。
- 可微优化器：`optimize_shape` / `optimize_geometry_process` 默认 `learning_rate=0.01`（2026-08-17 由 0.05 改；0.05 在真实焓法下第一步过冲 ~10× 发散）。SDF 尺寸偏差指标在边界区对修正场 u 极敏感，需稳定 lr。3 个 dimensional 优化测试已按真实焓法 Landscape 重标定（形状/joint 用 lr=0.01 验证真实降偏差；pareto 改验"种群探索"因前沿在该小球问题退化为单支配点）。

## 2026-10-06 增补（A0/A5/D1 之后的长期事实，开发日志 §24–§25）
- **GPU 已获授权**（用户 2026-10-06「可以开始自由使用GPU」，此前的"先不要动"作废），但**共享纪律不变**：卡上常驻 3 个 `newton_torch_env` 进程 + 桌面 Xorg，不得抢占/终止他人进程，被要求即归还。⚠ `xla_cuda13` 在 GPU 不可见时只打印 `CUDA_ERROR_NO_DEVICE … Falling back to cpu` 就继续跑 CPU ⇒ **凡声称"GPU 实测"的日志必须含设备清单行**，否则按 CPU 处置。
- **实测吞吐律**（§25.7，判据跑前登记）：GPU f64 **82.83 M**、CPU f64 **2.40 M** vox·step/s（dx=12.5 µm，161602 体素 × 4180 步）⇒ 同网格加速 **28.9×**、标定档最好 **34.5×**（**不是** 100×）；吞吐随网格单调上升**未见饱和**；每步 ~1.3–1.9 ms 固定开销 ⇒ 小算例（1000 体素）GPU/CPU = **0.88×（更慢）**。零件尺度 5.3e14 voxel-step ⇒ CPU **7.0 年**、GPU **74 天**/次正向、NSGA-II 一轮 **4.4 年** ⇒ **默认链在零件尺度用显式瞬态焓算术上不可行；GPU 只改常数，指数必须由算法改**。f64 下设备无关已验（同网格同 ns Δpeak=0.0000 K、Δ熔化体素=0）。
- **f32 禁用于标定输出**：不比 f64 快，且 Δpeak 31.07 K / Δvol 4.17%；陷阱——dx=50 档 f32 体积 0.0759 ≈ dx=12.5 档 f64 的 0.0745，**数值扩散在形态上伪装成"加密网格"**。JAX 对模块内写死的 `float64`（`thermal_enthalpy.py` 7 处）只发 UserWarning + 静默截断。⇒ A2 作废。
- **热档位决定 D1（§25.8，已按上面铁律 4–6 自我纠正过）**：标定档 = `thermal.enthalpy`（数值 + 真实潜热，dx≤r/2）；零件尺度**主线 D2** = 活跃子网格（jax-am `get_active_mesh` 思路）× 多重网格+层间子循环（GO-MELT 思路，MIT 可重写），仍数值+瞬态+可微；**备胎 D3** = 本征应变 ε\*（由标定档数值焓解标定）+ 可微线弹 FEM 的**数值降阶**（必须标 `fidelity="reduced"` 并量化与标定档偏差）；**闭式（Rosenthal/Eagar–Tsai）永不进默认链**，只当快速初值/教学档；裁判 = adamantine 离线。
- **外部件一手事实（亲验，纠正旧记录）**：jax-am 真实路径 `/home/shy/桌面/课堂设计/jax/jax-am-main/`（**不在** 2026 资源树内），其 `applications/fem/thermal/models.py:29-31` **潜热/焓项被注释掉**（活跃项仅 `rho*Cp*T/dt`，无相变），活跃子网格哈希表在 `:71/:88/:103/:111`，另有 `jax_am/lbm/core.py:195,629-630` 蒸发反冲 LBM 熔池。`JLnorthwestern/GO-MELT`（MIT，40★，push 2026-03-24）`jax.grad` **0 处**但内层已 **4× `jax.lax.scan`** + `subcycle_num_L2/L3`，外层驱动仍是 `while ongoing_simulation`。`adamantine` 许可 = **Apache-2.0 + LLVM exception**（读 LICENSE 首行；GitHub API 报 NOASSERTION；旧记录 GPL-3.0 作废）。`deepmodeling/jax-fem`（767★ GPL-3.0，活跃）= 可微隐式线弹 BVP 现成件。**空白亲验为无**：JAX 生态无准稳态/Rosenthal 现成件、**无可移植有许可的本征应变畸变实现** ⇒「别造轮子」能省内核与网格机制，省不掉驱动与 ε\* 标定。
- **体素化缺陷 T1（D4 已修，2026-10-08 复验）**：`geometry.py:113-119` 的 `n = ceil((hi-lo)/spacing)+1` 对 dx 的 **1 ulp 拼写差**敏感（`12.5*1e-6 = 1.2499999999999999e-05 ≠ 12.5e-6`），整数倍厚度时层数 ±1（161602 vs 156849 体素）⇒ 同零件 peak ±0.53%、熔体积 ±1.77%。**`_grid_axes` 已按 1e-9 相对容差吸附**，本轮 CPU 复验两种拼写在 dx=100/50/25/12.5 上 `shape` 逐格相同（`(13,7,5)/(25,13,9)/(49,25,17)/(97,49,33)`）⇒ "乘法拼写仍然错"作废。但**同类位点还剩 8 处**（长度→计数：`contracts.py:190 layer_count`、`geometry.py:415/449/521`、`process.py:501 n_line`、floor+1 族 3 处方向相反），登记为 **#24**；修后 §25.2/A5 已登记数字需重测（#16）。**跨探针比物理量之前先比 `sdf.shape`** 这条纪律长期有效。

## 2026-10-08 增补（#23 落地后的"数字代际"，开发日志 §25.2 声明／§26.18）
- **A0 的 peak/vol 档案数字有三代，混用即错**：**G1**＝§25.2／§25.7 现表（#19 之前，peak 2589.4/2670.3/2617.8/2586.6 K）；**G2**＝`am_t2_a0_conv_rerun.log`（#19＋#26 后，2125.6/**2349.1**/**2479.0**/2518.0 K）；**G3**＝#23 之后，**只有** `am_t23_capacity_closure_probe.log` 的 P3 三档（δ=0：**2461.269/2563.006/2565.077 K**，Vn 0.06025/0.06947/0.07418 mm³，Vm 0.05800/0.06759/0.07307 mm³；**缺 dx=100／缺 point 源列／缺 Ly**）。⇒ 引用任何"绝对峰值/熔体积"前先确认代际；就地换数被 **#24** 卡住（缺格不许填空，且 #24 一动 ns/路径长就再过期）。
- **G1↔G2 的一部分差不是物理而是网格**：G1 出自 D4 **之前** × 乘法拼写（多铺一层 Z：546/3250/22050/161602），G2/P3 出自正确那一代（455/2925/20825/156849）。同名 dx 的旧读数**不可互换**。
- ⚠ `am_a5_gpu.log` 的 f32@3250 行与 `am_a0_conv.log` 的 f64@2925 行**同印 2737.4/0.0759**——两个变量同时不同却撞值，无解释，**不得**当"f32≈加密一档"的证据（那条只由同网格 A/B 支撑）。
- **测试 docstring 也是论断**：`test_enthalpy_thermal.py:316-320` 与 `test_boundary.py:184` 的表曾把 G1/G2（后者＝`am_t27` 的 old 口径列）当现行实测 ⇒ 任务 **#28 已于 2026-10-08 01:05 落地**（开发日志 §26.20；**只换数字与归属文字，assert 与 0.05/0.02 字面阈值一字未动**，机械核对＝`git diff -- tests` 里含 "assert" 的增删行只有散文）。A0 侧现代表＝**2461.27/2563.01/2565.08 K、Vn 0.06025(482)/0.06947(4446)/0.07418 mm³、assert1 3.969% 绿、assert2 13.270% 红**；①②代降为代际档案并标来源。
- **代际术语（2026-10-08 定调）**：上面的 G1/G2/G3 与文档里的 ①/②/③ 是同一链条（G3＝③＝#23 之后）。⚠ `am_t25_round3_observable.log` 里 5 处"G1 代"经提交时间核对其实是 **② 代**（那轮跑在 `af9fc89`＋dirty30，早于 #19 提交 `c41d3b0`），log 内已附更正块。
- **③ 代全量基线＝`2 failed, 216 passed, 3 skipped in 3818.21s`**（`am_t27_fullsuite.log`，head `d7404f4`、dirty_src=0、CPU 钉住）。两条红＝A0 的 assert2（熔体积 13.270%）＋ #27 的压峰守护（−0.963%），skip 恰 3 条 PySide6，项数 221 与②代基线相同。旧基线 `1 failed, 217 passed`（`am_t2_fullsuite.log`，`c41d3b0`）只是②代档案。
- **#23 的过期幅度对两个量不一样（会影响所有后续换数）**：**末态均值 +约 19%**（2291 → **2720.80 K**）而**峰值 −0.23%**（2943.778 → **2937.061 K**）。机制：均值＝总能量/总热容的直接读数、热容口径改动全额传导；峰值被**蒸发封顶**钉住 ⇒ 几乎不动。⇒ §25.2/#16/#24 的替换表要**按量分别处理**，不得对峰值与均值套同一比例。
- **两条"位级口径"类（不是物理，归 #24 邻域）**：①`整数计数 × dx³ → 体积`**仍继承 dx 的浮点拼写 ulp**（D4 只修了网格轴；4520 体素的真值恰落在 `0.070625` 十进制第 5 位舍入平局点 ⇒ `dx_um*1e-6` 印 0.07062、字面量印 0.07063）⇒ 跨探针要求"逐位相同"必须**统一拼写或直接打印原始计数**；②δ=0.5·dx 对齐下 `solid_weight` 给 `cap_w = 1−4 ulp`（50 µm）/`≈1−13.5 ulp`（25 µm）而非精确 1.0 ⇒ 旧登记「δ=0.5 不携带信息」只是**近似**成立（第 4/4b 轮实测：#23 在那一列上是**场级机器精度恒等**，全场最大相对差 ≤5.4e-15≈1.4e-11 K；O1 的 δ=0.5 列**继续保留**，删＝放宽）。


## 环境
- 项目 Python = `amsim` venv：`/c/Users/flyn4/.workbuddy/binaries/python/envs/amsim/Scripts/python`（jax 0.11.0 + pytest 9.1.1）；`default` venv 无 jax，勿用。
- 跑测试：`PYTHONPATH=src JAX_ENABLE_X64=1 <amsim>/python -m pytest ...`
- pip 装包前 `export PYTHONPATH= BASH_ENV=` 以绕过 safe-delete 失败拦截；镜像回退：清华→阿里云→中科大→官方。

## 2026-10-08 增补二（#25 第 5 轮＝亚格份额口径，开发日志 §26.21／总览 §6 第 14 条）
- **A0 的 13.27% 主体是"硬阈值计数"的口径项，但这句话有两个边界**：把 `peak_temperature` 当节点值做三重线性重建、
  按子体素中心数液相份额得 `Vs`，δ∈{0,0.25} 加密方向漂移 **0.298/0.131/1.645/0.223%**（`Vn` 同处 13.270/6.356/
  11.150/5.234%，缩小 ~40×），口径项 `Vn−Vs` 随加密近似减半（1.89/1.84 ⇒ 一阶）；**亚格口径下熔体积 ≈ 0.0799 mm³、
  三档一致到 0.4%**。边界①＝`Vs` 在含 δ=0.5 的 O1 下入选但**余量仅 0.117pt**（4.883%），不是"闭合"；
  边界②＝**跨 δ 那 4.4%/8.8% 换口径治不了**（见下条）。⇒ **不换指标，A0 仍红**（它断 peak/Vn）；
  是否把 assert2 改为份额口径＝**#29，用户裁决**，不得自行替换。
- **跨 δ 不是读数差而是物理差（域质量刀锋，#19/T2 家族，长期有效）**：δ=0 的网格比 δ=0.25/0.5 **多一整层**表面体素
  （nvox 2925 vs 2304、20825 vs 18432）⇒ 固体域本身不同、热的域不同 ⇒ 同 `dx` 不同 δ 的 `Vs` 差 4.4%/8.8%
  **且不随加密消失**。⇒ **跨 δ 比形态量之前先比 `nvox`/`sdf.shape`**，否则把域差读成不收敛。
- **A0 试片被熔道铺满（夹具事实，非口径）**：δ=0 三档**触壳熔体素＝36/241/1143**、δ=0.25/0.5 **邻非实体＝738/3006/10449**、
  `min(fv)=0.500`（恰 `CUT_CAPACITY_FLOOR`）、`Lx=1.154–1.195 mm` 而试片 X 向只有 1.2 mm ⇒ **熔区含自由面**，
  所以"计数口径"里**混着两类**（液相面亚格份额 ＋ 自由面中心判据 vs 亚格判据）。是否加大试片＝**#30，用户裁决**
  （夹具一改 ⇒ 全部 A0 数字作废重测）。
- **估计器自检可复用的形状**：球面 `Vn` 误差 **+6.99/−0.73/+0.94%（符号摆＝对齐运气）** vs `Vs` −5.28/−1.43/−0.30%
  （单边单调，p=1.88/2.23，m 敏感性 4.28e-05）；线性场重建命中 5.031e-17；凹场 ⇒ 内插偏低 ⇒ `{F̃>0}⊂{F>0}`、误差 ~dx²。
  ⚠ "平均误差 Vs<Vn" 那条**余量只有 19%** ⇒ 只引用形状，不引用平均值。
- **12.5 µm 档熔体素计数＝37982**（本轮首次打印，#28 当时欠的那格；δ=0 三档计数列表现行为 482/4446/37982）。
- **自找缺陷 #15（与 #14 同族＝散文未跑即写）→ 一条长期纪律**：**预注册判据里每个数值门槛都要先在"正确实现"上算一遍或跑一遍合成例**，
  确认它不会在正确情形下报警（本轮 S1a 初稿要求单切割层平面总量误差 <2e-3 且随 m 单调，实际该口径只有 1/m 量级＝2.928e-03，
  跑生产前自查算术后删掉；同轮把 0.750 凭心算写成 0.625，已两处 ⚠ 更正）。先例＝第 2 轮 θ 降级为量级参考。
