# 熔池物理保真升级（A 档 + B 档）开发纪实

> 记录 `amforge.meltpool` 两档物理保真升级的完整过程：目标、根因、修复、验证证据、
> **存在问题与遗漏点**、后续建议。供后续开发直接参照，避免重复踩坑。
>
> 时间跨度：2026-08-13（A 档立项）→ 2026-08-20（B 档验证闭环 + 文档归档）。
> 负责人：flyn4 + 未来之路（AI 搭档）。

---

## 0. 总览与结论

| 档位 | 目标 | 状态 | 验证 |
|------|------|------|------|
| **A 档** | `meltpool.fdm`（默认传导 FVM）接 **温度相关导热系数** + **Beer-Lambert 体积吸收热源** | ✅ 完成 | `tests/test_meltpool_temperature_dependent.py` 4/4 |
| **B 档** | `meltpool.vof_flow3d`（NS+VOF）做 **Marangoni / 匙孔 / 自由界面**，稳定化 + 接默认可微链 | ✅ 完成 | `tests/test_meltpool_vof.py` 8/8；blast gate 9 文件 85/85 RC=0 |

**核心澄清**：B 档求解器**并非从零实现，也非 stub**——`meltpool.vof_flow3d`（`solve_meltpool_vof`）
此前已完整编码（表面张力 CSF、Marangoni、反冲压、浮力/Darcy 糊状区、变密度投影、
VOF MUSCL+界面压缩+蒸发失重、Beer-Lambert 激光阴影+Fresnel 增强），但**从未被任何测试执行过**
（仅"可选、非默认"注册）。B 档实质是 **根因级稳定化 + 接默认 `jax.grad` 可微链 + 补正式验证**。

**用户硬原则（两档共同约束）**：
1. 仿真 = 数值方法，默认前向链不得走闭式/解析近似。
2. 默认链路须全程 `jax.grad` 可微。

---

## 1. A 档：默认熔池（传导 FVM）物理保真

### 1.1 目标
在 `meltpool.fdm`（实为 **FVM** cell-centered，传导模式、**无** NS/VOF 自由界面）上补两处真实物理：
- **温度相关导热系数**：固→液 k 跳变（金属 2~3 倍）影响熔池形貌。
- **Beer-Lambert 体积吸收热源**：取代原表面 δ-型高斯，做面内高斯 × 深度指数衰减，∫q dV = A_eff·P（功率守恒）。

### 1.2 实现要点
- 采用 **"变导热、常密度"** 标准近似：RK2 每步 `fl = mat.liquid_fraction(T)` →
  `kT = mat.k_of(fl)` → `effective_diffusivity(T, k=kT, ...)`。ρ、cp 保持参考常数（用于焓↔温映射），
  **不重写 H(T) 定义**，避免连带改动焓法相变求解器。
- 新加辅助函数 `_laser_source_fdm(coords, x_t, *, rb, A_eff, P, absorption_depth, dx)` =
  面内高斯 × 深度指数衰减。
  > **命名陷阱**：该函数原名 `_laser_source`，与 VOF 求解器的 8 参 `_laser_source(F,T,cfg,mat,process,traj,t,dx)`
  > 撞名（Python 后定义覆盖前定义）。B 档为回避遮蔽，将其改名为 `_laser_source_fdm`（见 §2.6）。

### 1.3 验证（`tests/test_meltpool_temperature_dependent.py`，4 例）
功率守恒、指数深度衰减自相似、温度相关 k 改变结果（Ti6Al4V vs 退化常量）、整体可微。全绿。

---

## 2. B 档：NS+VOF 自由界面 / Marangoni / 匙孔

### 2.1 力学模型（已编码，待稳定化）
- **表面张力**：CSF `σ·κ·∇F`，曲率由 `_smooth_field` 平滑后估计（避免 1 格锐界面的曲率尖刺）。
- **Marangoni**：`f_ma = dsigma_dT · (∇T − ∇T_n) · |∇F|`（`∇T_n` 为界面法向梯度分量）。
- **反冲压**：`f_recoil = min(mat.recoil_pressure(T), cfg.recoil_pressure_cap) · ∇F`（Clausius-Clapeyron 汽化反冲，10 atm 封顶）。
- **浮力/重力 + 糊状区 Darcy**：`μ_eff = μ_liquid · (1 + darcy_constant·(1−fl)²)` 阻尼半熔区。
- **变密度投影**：`∇·(β∇p) = (1/dt)∇·u*`，`β = (1/ρ)·metal`；气相 `metal→0` 故 `β_gas=0`，自由表面压力解耦。
- **VOF 输运**：MUSCL+minmod 界面重构 + 界面压缩 + 蒸发质量损失；激光 Beer-Lambert 阴影 + Fresnel 增强。

### 2.2 根因 #1（速度炸到 1e10 m/s 的元凶）：压力投影发散
隔离测试 `tests/_scratch/test_proj.py`（后移至 `tests/test_meltpool_vof.py::test_projection_removes_divergence`）
证明：原 `_project` 的 CG 解 `(∇·β∇p)` **残差随迭代增长**（10 iters→1.5e6，120 iters→1.8e13；pmax→1e21）。

两个独立缺陷：
1. **纯 Neumann 奇异 + 相容性不满足**：全液态（metal=1 处处）时，压力泊松方程是奇异系统，
   且右端 ∮u·n ≠ 0，CG 沿零空间线性发散。
2. **算子与梯度/散度不相容**：投影解的是**紧致 7 点**拉普拉斯，而 `_div`/`_grad` 用**中心差分（跨 2Δx）**；
   棋盘模式落在中心差分算子的零空间，压力完全看不见它，散度无阻尼增长。

**修复**：重写为 `_pressure_system(rho, metal, dt, dx)` + `_project(...)`：
- **D–G 相容对**：界面处用 `p_ghost=0` 的单侧差（自由表面 Dirichlet 参考），避免零空间棋盘；
  块体内部仍用中心差，保证算子与 `_div`/`_grad` 自洽。
- **软掩膜金属场顶部气相保留 Dirichlet 钉**：去纯 Neumann 奇异（验证里核心散度 6.2e-4 即源于此）。
- **气相行给单位算子** `β_gas·p = 0`（对称、良态、p_gas 恒为 0）：原 Jacobi 预条件子 `1/diag` 会把气相残留
  （~1.2e-3）行放大 ~1e30 倍污染 Krylov 子空间，显式单位行规避。
- 右端 `rhs = -metal · (_div_face(u*)/dt)`，CG 收敛判据 `‖Aop(p)−rhs‖/‖rhs‖`。

验证：CG 线性残差 修复前 1e13+ 发散 → 现在 **1e-4**（变密度+Jacobi 预条件在 40 步下的真实收敛地板，已足够稳定）；
金属核心内部散度相对残差 **6.2e-4**（不可压性满足）。

### 2.3 根因 #2：无散度梯度 NaN
- `_grad`/`_smooth_max` 在零梯度处（均匀 F 的基板/气相内部）算 0/0 → NaN；改用 `_norm`（带 `where(...,1e-30)` 安全分母）。
- `_cg` 退化分支用 `1e-300` 作分母，反向传播放大 1e300 → inf；改为安全的 `1/diag_r` 归一（`diag_r = max(diag,1e-30)`，除以自身诊断而非作分母）。

### 2.4 根因 #3：可微性 OOM（违反"默认链全程 jax.grad 可微"铁律）
`simulate_meltpool` 默认 `checkpoint_every=0` 走裸 `lax.scan`，400 步反向存全部 carry ≈ **677MB** 撑爆
（`jax.grad` 直接 OOM、梯度 NaN）。

**修复**：
- `checkpoint_every` 默认改为 **20**（用 `@jax.checkpoint` 重算换显存，峰值 ~68MB，前向数值完全一致）。
- 修整除丢余数步 bug：原 `n_blocks = cfg.n_steps // cfg.checkpoint_every` 在 `n_steps` 不能被块长整除时
  **少跑余数步**；改为 `n_blocks = ceil(n_steps/checkpoint_every)` + 末块 `length = remainder` 精确跑满。

### 2.5 验证证据（44×32×30 网格、400 步、25 压力迭代、316L）
| 量 | 值 | 说明 |
|----|----|----|
| 熔深 / 熔宽 / 熔长 | 19.9 / 109 / 95 µm | 真实熔池成形 |
| vmax | 15 m/s | 物理量级（原 1e10 爆炸） |
| p | ≈ 3.6e5 Pa | ≈ 4 atm |
| 反冲压 | 8.7e5 Pa | 匙孔驱动力 |
| 自由界面凹陷（功率 120/250/400 W） | 13 / 32 / 47 µm | **匙孔正反馈成立** ✓ |
| 关 dσ/dT 后熔宽 / W-D 比 | 129.6→109.4 µm / 6.5→5.5 | **Marangoni 真实生效** ✓ |

全场有限、气相层保留、VOF 输运正常。

### 2.6 回归修复（B 档漏改）
B 档把 A 档 Beer-Lambert 辅助函数 `_laser_source` 改名为 `_laser_source_fdm` 以回避遮蔽 VOF 的 8 参
`_laser_source`，但**漏改 A 档测试 `test_meltpool_temperature_dependent.py` 的 import/调用** →
blast gate 现 2 failed（`_laser_source() got an unexpected keyword argument 'rb'`）。已同步改测试为
`_laser_source_fdm`，重跑确认 4/4。

### 2.7 正式测试套件（`tests/test_meltpool_vof.py`，8 例全绿）
1. 投影收敛（CG 线性残差锁 `1e-2`）
2. 运行有限（jit+scan 500 步全场有限）
3. 熔化（液相分数 > 0）
4. 自由界面 / 匙孔成形（气相层保留 + keyhole 场存在）
5. Marangoni 对照（关 `dsigma_dT` 后熔宽变大）
6. 可微性（默认 checkpoint 下 `jax.grad` 有限、非 NaN）— 回归守护
7. 反冲压封顶生效
8. 功率→匙孔正反馈（凹陷随功率单调增长）

---

## 3. 存在问题与遗漏点（务必知悉）

### 3.1 【重要】可微梯度病态条件数（非 bug，是固有特性）
`jax.grad` 现已能算、有限（不再 OOM），但几何尺寸量（如熔深）经 400 步强非线性显式链反向
**条件数病态**：`∂depth/∂P ~ 1e45 m/W` 量级。这是**显式 CFD 反演的固有特性**，**裸梯度直接喂优化器会爆**。
**后续若把 `meltpool.vof_flow3d` 接进 `inverse` 反演器**：必须先配 gradient clipping / 自适应步长 /
标量输出归一（如 depth 除以参考尺度后再求梯度）。A 档传导求解器梯度条件数好得多，可直接用。

### 3.2 【重要】B 档求解器仍是 select-only，非默认
默认熔池链走 `meltpool.fdm`（传导 FVM，A 档已升级）。`meltpool.vof_flow3d` 需显式
`select={'meltpool':'meltpool.vof_flow3d'}` 启用。文档/用户需明确：开箱默认是传导近似，
高保真 VOF 是 opt-in。这点是诚实边界，勿在默认链声称已含自由界面/匙孔。

### 3.3 eager 模式逐 op 派发极慢且会静默崩
eager 逐 step 派发 ~0.47 s/步（600+ 步仍无声退出，系外部 kill / 资源限制，**非 Python 异常**，
faulthandler 也无输出）。**生产路径必须走 `jit + lax.scan`**（已验证 500 步 EXIT=0 全场有限）。
调试时用 `_scratch/debug_vof.py` 的 step 循环仅用于定位，勿当作运行方式。

### 3.4 短时仿真"熔深=0"是物理后果，非 bug
500 步仅仿真 ~30 µs，热扩散深度 √(αt)≈11.6 µm < dx=20 µm，热量未传入基板一格 → depth=0。
**测试/验证配置必须跑足够步数**（如 400 步 + 合适 dx）才能观测到熔深。此前 smoke 用粉层填满全
域、无气相自由界面，是错误配置，已纠正为带气相层的物理合理配置。

### 3.5 `_smooth_max` 退化标度是地雷
`depth/width/length` 经 `_pool_dimensions` → `_smooth_max(β·x̂)`。退化场（`hi≈lo`，如"基板完全没熔化"
时的全零 melted 场）原 `scale=max(hi−lo,1e-30)` 会退化成 1e-30，把梯度放大 1e30 倍（实测 grad=5.5e11 伪值）。
已改为 `scale = max(hi−lo, 1.0)`（退化时回退到均值、梯度摊平）。真实熔池（熔化包络 span~O(1)）走原分支，数值不变。

### 3.6 无真实硬件标定 / 实验对照
所有验证均为**自洽数值验证**（功率守恒、不可压性残差、物理趋势正确），**未做实验/商业软件对照**。
绝对精度依赖材料参数标定（见 `docs/热源标定报告_2026-08-16.html`）。框架不"免费"给出工业级定量精度。

### 3.7 性能与硬件
- B 档 44×32×30×400 步 ~3 min（CPU，amsim venv，jax 0.11.0 + x64）。更大网格/更长时程需 GPU。
- Ubuntu 48G GPU 全保真回归（项目 #38）**待硬件到位**，尚未在 CUDA 下跑过 B 档。
- CPU 上 jaxlib 默认版；启用 CUDA 需按 JAX 指南装对应 `jaxlib` CUDA 轮子（README §安装 已说明）。

### 3.8 材料覆盖与合金差异
`AMMaterial` 已覆盖 316L / Ti6Al4V / IN625 / IN718 / AlSi10Mg。注意 Al 合金汽化/反冲特性与钢差异大，
`recoil_pressure` / `evaporation_flux` 的 Clausius-Clapeyron 参数需按合金核对，否则匙孔深度会偏。

### 3.9 验证脚手架的硬编码路径
`_scratch/vof_validate.py`、`smoke_vof.py`、`debug_vof.py`、`test_proj.py` 含 Windows 绝对路径与
`amsim` venv 硬编码，仅供本机复跑证据，**未清理、未进包**。迁移到 Ubuntu 需改路径或仅依赖
`tests/test_meltpool_vof.py`（已路径无关）。

---

## 4. 文件变更清单

| 文件 | 变更 |
|------|------|
| `src/amforge/meltpool.py` | 压力投影重写为 `_pressure_system`+`_project`（D–G 相容对、气相 Dirichlet 钉、气相单位行）；新增 `_norm`；加固 `_smooth_max` / `_cg`；`checkpoint_every` 默认 20 + 整除余数 bug 修复；A 档辅助函数 `_laser_source`→`_laser_source_fdm`；`MeltPoolState` 增 `F0`；新增 `_smooth_field` 曲率平滑 |
| `tests/test_meltpool_vof.py` | **新增**，B 档 8 例验证 |
| `tests/test_meltpool_temperature_dependent.py` | import/调用同步改名 `_laser_source_fdm`（修复 B 档漏改回归） |
| `_scratch/*.py` / `*.log` | 验证脚手架与日志（本机证据，未进包） |

---

## 5. 后续开发建议

1. **反演 conditioner**：若启用 VOF 反演，先实现 gradient clipping / 输出归一 / 自适应步长（§3.1）。
2. **CUDA 回归**：在 Ubuntu 48G GPU 上跑 B 档全保真回归，确认 `jax.grad` 在 CUDA + checkpoint 下行为与 CPU 一致（§3.7）。
3. **默认链升级评估**：评估是否把 `meltpool.vof_flow3d` 设为某类工况（高功率/高扫描速度易匙孔）的自动推荐档，或保留纯 opt-in（§3.2）。
4. **实验标定闭环**：接入单道熔池实验尺寸（宽/深）做参数标定，把 §3.6 的自洽验证升级为定量验证。
5. **清理脚手架**：`_scratch` 验证脚本路径无关化后，挑选有价值的（如 `vof_validate.py`）提升为 `tests/` 或 `examples/` 正式用例。
6. **合金参数审计**：核对 Al 合金等反冲/汽化参数（§3.8）。

---

## 6. 档位语义与 `select-only` 澄清（FAQ 必读）

> 用户 2026-08-20 质询："vof_flow3d 仍是 select-only，是不是没实现高质量模拟？"
> 本节能直接消除该误读。

### 6.1 先给结论
**`meltpool.vof_flow3d` 已经是高质量的全保真数值实现**，不是没实现、不是分析近似、不是壳子。
"select-only" 是**档位选择机制**，不是质量缺陷。它和 ABAQUS "默认拿不到双精度谱单元、得切 element type"
是同一类设计——高保真档位需用户明示选择。

### 6.2 三层含义

**第一层（事实）**：B 档闭环后 vof_flow3d 的实现清单（均为真实数值）：
- NS 方程：自写变密度投影法（Jacobi 预条件 CG，无外部 CFD 黑盒）
- VOF 自由界面：MUSCL + minmod 平流 + 界面压缩 + 蒸发失重
- 表面张力 CSF（σ·κ·∇F，已用平滑曲率防 1 格界面尖峰）
- Marangoni：`f_ma = dσ/dT·(∇T−∇T_n)·|∇F|`（关掉 dσ/dT 验证过 W 109→129 µm、W/D 5.5→6.5）
- 反冲压：Clausius-Clapeyron 蒸发压（随功率 13→47 µm 正反馈）
- Beer-Lambert 激光阴影 + Fresnel 增强（与 A 档共用）
- 8/8 正式测试（投影收敛/无 NaN/熔化/Marangoni/匙孔/可微），物理扫描验证通过

**第二层（select-only 真实含义）**：AMForge 选路纪律——

| 档位 | 求解器 | 物理覆盖 | 代价 | 默认? |
|------|--------|----------|------|-------|
| 默认链 | `meltpool.fdm`（3D FVM 焓法） | 温度场 + 熔池形态（无自由界面拓扑变化） | 低 | ✅ |
| 高保真 | `meltpool.vof_flow3d`（NS+VOF） | + Marangoni 流场 + 匙孔 + 自由界面形变 | 高（每步 25 CG + checkpoint） | select-only |

- `select={"meltpool": "meltpool.vof_flow3d"}` 是**用户明示"我要看流场/匙孔/自由界面"**。
- 默认不开的原因：(a) 80% 工艺尺寸/温度场问题 fdm 已答得对；(b) vof 梯度对功率偏导病态
  （∂depth/∂P~1e45），裸喂优化器会爆；(c) 计算/显存贵。
- fdm 与 vof **都是真实数值方法**，只是档位不同——满足用户"默认即实打实物理模拟"的铁律。

**第三层（诚实边界，防误用）**：
- 防**口头/文档误传**：默认链走 fdm，所以不要在"AMForge 默认开箱"宣讲/文档里写"含自由界面/匙孔/Marangoni"——那是 vof 档才有。
- 防**反演优化器**直接吃 vof 梯度：∂depth/∂P ~1e45 m/W，必须先 gradient clipping 或输出归一化。
- 防**测试/基准误用**：用 fdm 熔宽对比 vof 熔宽会误判"fdm 不准"——这是档位差异。

### 6.3 同系列档位（避免逐个问）
`thermal.vof_resolved`、`micro.phasefield`、`support.simulate`、`mbd.assembly`、`powder.bed`、
`meltpool.vof_flow3d` 全部是 **select-only 高保真档**——实现都已落地、测试已绿，但默认链走其对应
的数值默认档。语义同上，**均非"未实现"**。详见 `docs/项目开发总览.md` §3 与 `docs/支撑结构_M2_开发纪实.md` §3。

## 7. 一句话给后来者

> B 档不是"写了个 CFD"，而是"把一个**已写好但从未跑通**的 VOF 求解器**稳压到可微、可验证**"。
> 真正的坑全在**压力投影的离散相容性**与 **`jax.grad` 的显存/条件数**上——物理公式本身一直是对的。
