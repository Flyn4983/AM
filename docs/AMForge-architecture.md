# AMForge 通用增材制造仿真软件 — 架构框架

> 目标：开发一套**商业级、通用**的增材制造（AM）全流程仿真软件，对标 Abaqus/ANSYS 的交互体验与可扩展 API，
> 同时内建**可微分模拟**能力用于工艺优化。涵盖三大真实物理仿真域 + 一个优化域。

---

## 0. 设计第一性原理（最重要）

**定义一个唯一的、JAX 可追踪的 canonical 模型对象（SimulationModel）。GUI 与 Python API 只是它的两个编辑器。**

由此导出四个性质，构成"通用性 + 好用 + 可微"的统一：

1. **单一真源**：GUI 操作与脚本操作都落到一个模型对象，导出同一份 `.amf` 工程文件；二者可互相无损转换。
2. **积木式可组合**：模型 = 一组有类型的"积木块"（Part / Material / Step / BC / IC / Mesh / Solver / Output）。
   新增物理或数值方法 = 新增积木块 + 注册一个求解器，**核心永不硬编码某种方法**。
3. **无缝链接**：编排层按块的"数据端口类型"自动推断数据流、拓扑排序、装配成计算图（DAG）——即 amforge 既有 registry+DAG 的泛化。
4. **可微是免费的**：因为模型图是 JAX pytree，既能正向 `run()`，也能 `value_and_grad` 整体求梯度，
   直接支撑工艺优化 / 逆设计，无需为优化另写一套。

---

## 1. 分层架构（对应架构图）

| 层 | 职责 | 关键产物 |
|---|---|---|
| **表现层 Presentation** | GUI 前处理 + Python 积木式 API + 统一模型文件；同一模型两视图 | GUI 窗口、Python DSL、`*.amf` |
| **模型层 Model / Session** | 唯一 canonical 模型：JAX-pytree 积木图，可序列化/版本化/可微 | `SimulationModel` 对象 |
| **编排层 Orchestration** | `SolverRegistry`（插件注册）+ `SimulationGraph`（自动选路装配 DAG） | 计算图、求解顺序 |
| **求解层 Solver** | 5 数值方法 × 3 物理域的具体实现 | 各 `solver_*` 模块 |
| **可微优化层 Differentiable** | `loss = f(仿真输出, 目标)`；`jax.value_and_grad` 穿透全图；optax 反传 | 优化器回路 |
| **后处理层 Post** | 场浏览器（温度/应力/微观/变形）、VTK/Paraview 导出、数字样机动画 | 可视化、报告 |

---

## 2. 积木块目录（"像搭积木一样"的 API 与 GUI 同源）

所有块均为版本化 pytree，带 typed 输入/输出端口，供自动装配识别。

| 块类型 | 字段要点 | 端口 |
|---|---|---|
| `Part` | 几何表达：SDF / 网格(meshio) / 粒子集；维度、包围盒 | out: geometry |
| `Material` | ρ, Cp, k(_s/_l), **T_solidus/T_liquidus/T_boil**, **潜热 h_f**, E, ν, α, σ_y, 硬化, Hall-Petch, CRSS, CP 滑移系 | out: material |
| `Step` | 物理类型：宏观热/热-力、微观相场/CPFE、数字样机；方法(FVM/FEM/DEM/SPH/MPM)；模式(SLM/LSF) | in: 上游；out: 结果 |
| `BoundaryCondition` | 类型：固定/对流/辐射/热源(激光)；区域选择(面/体/扫描路径) | in: 场 |
| `InitialCondition` | 温度场、应力场、微观初始组织 | in: 场 |
| `Mesh` | 体/面/粒子离散；hex8/tet/粒子数；逐层激活策略 | in: geometry |
| `Solver` | 方法 + 容差 + 并行策略 + 求解器特定参数 | in: 端口；out: 结果 |
| `Output` | 场请求（温度/应力/相/变形）、采样、导出格式 | in: 结果 |

GUI 的"模型树"与 API 的 `model.add_part(...)` 等一一对应，二者读写同一对象。

---

## 3. 三大仿真域 × 五方法（求解层）

### 域① 宏观复杂构型成型（SLM + LSF，多物理 + 固液/液固相变）
- **FVM**：热传导 + **潜热/相变**（enthalpy / 表观热容法，把固-液-固相变接进热链）——*当前缺口，需补*。
- **FEM**：逐层热-力耦合（已实测跑通 `solve_thermomechanical`），需补**塑性/相变本构**。
- **DEM / SPH / MPM**：粉末尺度（最接近 Flow3D 粉床），`particle_am` 含 solidus/liquidus/solidification——待端到端验证。

### 域② 微观模拟（→ 宏观本构映射）
- **相场法**：凝固/枝晶——需 AM 专用化（各向异性 γ、热梯度取向、凝固速率）。
- **CPFE**：滑移系晶体塑性（模块齐备），经 voigt/reuss 上采样 → 宏观本构。
- 链路：工艺参数 → 微观组织 → 本构场（即"工艺→本构"映射，用户核心诉求②）。

### 域③ 数字样机（FEM 强度 + 多体）
- FEM 强度校核（热-力 + `load_cases` 载荷工况）——已可用。
- **多体动力学（关节/约束/装配）**——*当前缺失，最大缺口*。需新建或确认用 FEM 子模型+载荷工况近似。

### 可微优化（横切四层）
`loss = w1·几何偏差 + w2·残余应力 + w3·应变 + w4·缺陷 + w5·安全罚项`；
`jax.value_and_grad` 穿透整图；optax 反传。已在代理链验证，需迁移到高保真链。

---

## 4. 与现有代码的连续性（不要重写，要重构封装）

| 现有资产 | 在框架中的位置 | 动作 |
|---|---|---|
| `amforge` registry + `graph.Pipeline.auto/run/function` | → **编排层** `SolverRegistry` + `SimulationGraph`（泛化，不再 AM 专属） | 重构升级 |
| `amforge` 9 个 frozen dataclass 契约 | → **模型层** typed 端口（扩展为通用块） | 泛化 |
| `diffmech.methods.{fvm,fem,dem,sph,mpm,phase_field,cpfe,am}` | → **求解层** 各 `solver_*` 实现 | 包装注册 + 补缺口 |
| `amforge/inverse.py` 可微闭环 | → **可微优化层** | 迁移到高保真链 |
| `meshio 5.3.5`（VTK/VTU/msh/inp/e/med/ans） | → `Part` 几何导入 | 接入通用几何导入器 |
| `geometry.from_sdf_fn / from_stl / from_mesh` | → `Part` 解析/网格表达 | 保留 |

---

## 5. 真实缺口（决定"真实模拟"能否兑现）

1. **连续介质热解缺相变** —— 补 enthalpy/表观热容法（硬指标）。
2. **多体数字样机缺失** —— 新建多体模块，或与你确认用 FEM 子模型近似。
3. **相场需 AM 专用化** —— 接枝晶各向异性、热梯度取向。
4. **可微优化从代理迁高保真** —— 端到端反传。

---

## 6. 分阶段路线（建议）

- **P0 架构落地**：本文档定稿；把 registry/DAG 泛化为 `ForgeCore`（编排层），定义 `SimulationModel` + 积木块 schema + 序列化 + 校验器。**已完成**：`src/forgecore/`（registry/graph/model/serialization）+ `src/amforge/forge_adapter.py`（9 求解器接入）+ `tests/test_forgecore_e2e.py`（5 项通过，含整链可微无 NaN）。
- **P1 模型层 + 积木式 API**：先写 API（GUI 依赖它）；`model.add_*` DSL；`.amf` 读写；`ModelValidator`。
- **P2 求解器适配**：把 `diffmech` 求解器包成 registry 条目；补 4 个缺口（热相变 / 塑性 / 多体 / AM 相场）。
- **P3 GUI 前处理**（技术栈已定）：**PySide6 + VTK（Python 原生桌面）**——模型树 + VTK 3D 视口（导入模型/网格/粒子）+ 属性编辑器（材料/BC/IC/工艺/求解器）+ 作业运行；与积木式 API 读写同一 `SimulationModel`，双向同步。
- **P4 后处理 + 数字样机 + 可微优化**：场浏览器（温度/应力/微观/变形）、VTK/Paraview 导出、数字样机动画；可微优化 UI/API（loss=几何偏差+残余应力+应变+缺陷+安全罚项，jax.value_and_grad+optax）。
- **P5 工程化**：打包、示例工程、文档、回归测试（商业级可靠性）。

---

## 7. 已拍板的关键选择（2026-08-12 更新）

- **GUI 技术栈 = PySide6 + VTK（Python 原生桌面）**。
  理由：用户要求"Python 兼容、本地高性能、可拓展"，且类 Abaqus 体验最贴近桌面 IDE。
  前处理：PySide6 窗口（模型树 / 属性编辑器 / 作业日志）；VTK 视口负责导入与可视化
  模型、网格、粒子；后端直接调用 JAX 求解器，无网络开销。后期可再套 Tauri/Electron 壳
  兼顾跨平台分发，但内核不变。
- **多体 = 真多体动力学，且与 FEM 协同（混合仿真）**。
  关键约束：增材制造件本体用 **FEM** 计算强度/残余应力（保证关键部件精度）；
  其余机构/周边部件用 **多体动力学（MBD）** 计算运动/接触（保证整体计算速度）。
  二者通过协同仿真接口（FEM 子模型边界位移 ↔ MBD 关节载荷）耦合。这是数字样机
  域③的核心架构，也是仓库当前最大缺口。
- **首里程碑优先级 = 模型层 + 积木式 API（地基）**：已交付（`ForgeCore` + 适配层 + 测试）。

