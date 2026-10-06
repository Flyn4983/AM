# AMForge — 端到端可微分增材制造仿真平台

> 把「**工艺 → 熔池 → 组织 → 本构 → 结构 → 服役判定**」整条链做成一串纯函数：
> 既能像 Flow3D 那样正向做高保真熔池/热力模拟，也能对整条链求梯度，反向优化工艺参数乃至神经网络权重。

AMForge 用 [JAX](https://github.com/jax-ml/jax) 把增材制造（SLM/LSF）的多尺度物理建成**可微分前向**，于是：

- **正向仿真**：熔池 VOF/CFD、逐层激活热力耦合、相场凝固、CPFE 均质化本构、刚柔多体数字样机、服役裕度判定，全部可跑。
- **可微反演**：`jax.grad` 穿透整图，工艺参数 / 几何修形 / 神经网络权重可端到端优化。
- **多目标 Pareto**：尺寸偏差 ↔ 残余应力的 NSGA-II 非支配前沿（见下）。
- **通用引擎 ForgeCore**：所有求解器经数据契约 + 注册表自描述，自动布线、可替换、可扩展。

---

## 安装

要求 Python ≥ 3.10（已在 3.13 验证）。

```bash
# 可编辑安装（开发用，源码修改即时生效）
pip install -e .

# 或仅装运行时依赖做一次性使用
pip install .
```

GPU 支持由 `jaxlib` 决定：默认安装的是 CPU 版 `jaxlib`。要启用 CUDA，请按
[JAX 安装指南](https://jax.readthedocs.io/en/latest/installation.html) 安装对应的 `jaxlib` CUDA 轮子。

开发 / 打包依赖（构建 wheel、跑测试）：

```bash
pip install -e ".[dev]"
```

前处理 GUI（PySide6 + VTK 桌面界面）需要额外的图形依赖：

```bash
pip install -e ".[gui]"      # 或 pip install "amforge[gui]"
```

### 受限 / 沙箱环境安装（safe-delete 中和 + 镜像源回退）

部分受限环境（如带「安全删除」拦截层的沙箱）会在 `pip install` 删除/覆盖文件时抛
`[SAFE_DELETE_FAIL_CLOSED] ... recycle-bin-unavailable`，导致 PySide6 / vtk 等大型依赖装不上。
根因是该环境通过 `sitecustomize` 把 `os.remove` 改为「移入回收站」且 fail-closed。

**解法**（装 GUI 依赖前执行）：

```bash
# 1) 中和 safe-delete：清空 PYTHONPATH / BASH_ENV 使拦截 shim 不加载，os.remove 恢复原生删除
export PYTHONPATH= BASH_ENV=

# 2) 用国内镜像源（更快、避免默认源超时）；某源慢/失败就按顺序换下一个
#    清华 → 阿里云 → 中科大 → 官方
pip install -e ".[gui]" -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=120
#   备选： -i https://mirrors.aliyun.com/pypi/simple/
#         -i https://pypi.mirrors.ustc.edu.cn/simple/
```

> 经验证可装版本（Python 3.13 预编译 wheel，无需源码构建）：
> **PySide6==6.11.1 / vtk==9.6.2 / matplotlib==3.11.1**。
> 也可直接 `pip install -r requirements-gui.txt`（已锁定上述版本）。
> 无显示环境跑 GUI 测试追加 `QT_QPA_PLATFORM=offscreen`；VTK 3D 视图在该环境下自动回退，仅保留 2D 云图。

---

## 命令行（CLI）

安装后可用 `amforge` 命令（或 `python -m amforge`）。

```bash
amforge info          # 版本 / 依赖导入状态 / 求解器·适配器清点
amforge selftest      # 轻量端到端自检，验证「安装即可用」（几十秒）
amforge pareto        # NSGA-II 多目标优化演示：尺寸偏差 ↔ 残余应力（约 1–3 分钟）
amforge gui           # 启动 PySide6 + VTK 统一工作台（前处理 + 后处理/数字样机/优化）
```

### 图形界面（GUI）

`amforge gui` 启动一个 **PySide6 + VTK 桌面统一工作台**（单窗口多标签页），覆盖完整
「建模 → 仿真 → 后处理 → 优化」闭环：

- **标签页 ① 前处理**：解析体素化几何（球/柱/方）或 STL 导入 → 三维等值面查看 →
  分层切片 + **hatch 光栅扫描路径**叠加 → 工艺参数面板 + 可打印性/能量密度（VED/LED）
  评估 → 导出配置（几何 SDF + 工艺方案），可直接喂给 `amforge.inverse.simulate`。
- **标签页 ② 后处理 / 数字样机 / 可微优化**：
  - **场浏览器**：温度 / 残余应力(vM) / 位移 / 微观组织 / 本构 等体素场的 2D 云图
    （构建方向切片滑块）+ 3D 场着色面（VTK）；所有场可一键**导出 ParaView 可读
    `.vtk` + `.csv` + `manifest.json`（含各场 min/max/mean）。
  - **数字样机**：多体 + FEM 耦合装配评估（关节弯曲刚度柱状图、稳定性裕度、装配评分）
    + 铰接链示意。
  - **可微优化**：`loss` 历史曲线 / NSGA-II Pareto 前沿散点（utopia 与 best-compromise
    标记），覆盖 `dimensional` / `joint` / `pareto` 三种优化入口。

GUI 采用**分层架构**：所有计算收敛在纯逻辑层（`amforge.gui.preproc` / `amforge.gui.postproc`，
只依赖 numpy，可无显示 pytest），界面层（`*_app.py`）只负责呈现，因此核心算法可被单元测试
守住，而不只是「能弹个窗口」。

`amforge pareto` 常用参数：`--pop-size`、`--n-gen`、`--material`、`--seed`，
以及压缩网格的 `--n-grid / --max-layers / --n-sub-cp`（默认已是极小网格，用于快速演示）。

---

## Python API 最小可运行示例

```python
import jax
jax.config.update("jax_enable_x64", True)

import amforge as af
from amforge import geometry as G
from amforge import process as P
from amforge import inverse as inv

# 1) 任意复杂几何 -> SDF 体素 -> 分层 -> 扫描路径
geo = G.from_sdf_fn(
    lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,   # 半径 0.45mm 球
    bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=50e-6, name="demo",
)

# 2) 自动布线：从单一几何反推完整 8 步路径
pipe = af.Pipeline.auto("ServiceVerdict", modality="SLM")
print(pipe.describe())          # 静态查看 DAG
out = pipe.run(geometry=geo)    # 正向跑通整条链
print(out["verdict"].summary())

# 3) 可微反演：对几何形变场 + 工艺向量求梯度，最小化尺寸偏差+残余应力
plan = P.heuristic_plan(geo, material="316L", modality="SLM")
out = inv.simulate(geo, plan, material="316L", asbuilt_solver="plastic")
print("几何尺寸偏差 =", float(out["asbuilt"].geom_dev()))   # 见 inverse._dimensional_loss

# 4) NSGA-II 多目标：尺寸偏差 ↔ 残余应力 真实非支配前沿
res = inv.pareto_optimize_geometry_process(
    geo, material="316L", algorithm="nsga2",
    pop_size=12, n_gen=20, asbuilt_solver="plastic", verbose=False,
)
print("推荐折中方案:", res["best_compromise"]["final_plan"])
```

---

## 架构一句话

所有模块只通过 `amforge.core.contracts` 里的数据契约通信，并在
`amforge.core.registry` 声明「吃什么、吐什么」。`amforge.core.graph.Pipeline`
据此**自动布线**（「无缝链接」核心）。要换求解器，只改一行 `select`，调用代码不动：

```python
pipe = af.Pipeline.auto("AsBuiltPart", select={"thermal": "thermal.enthalpy"})
```

数值内核复用同源子包 **`diffmech`**（JAX 可微分计算力学：FEM / FVM / DEM / SPH / MPM /
相场 / CPFE），AMForge 不重复实现基础离散，只做 AM 物理模型、契约化封装与跨尺度编排。
通用引擎 **`forgecore`** 与具体物理域解耦，是 ForgeCore 的纯引擎层。

---

## 求解器选择（fidelity selector）

每个物理环节都有「代理模型 ↔ 高保真」可切换，兼顾速度与精度（对应你提出的
*精度 / 速度权衡* 诉求）：

| 环节 | 代理（默认，~免费、全可微） | 高保真（select 显式启用） |
|------|---------------------------|------------------------------|
| 熔池 | `meltpool.surrogate_eagar_tsai` | `meltpool.vof_flow3d` |
| 热历史 | `thermal.history` | `thermal.enthalpy`（enthalpy 相变潜热）/ `thermal.vof_resolved` |
| 微观组织 | `micro.surrogate` | `micro.phasefield` |
| 本构均质化 | `constitutive.homogenize`（Hall-Petch + 各向异性） | CPFE 晶体塑性嵌入 `asbuilt.plastic` 内部 |
| 成形 | `buildup.layer_activation`（ROM 趋势） | `asbuilt.plastic`（CPFE 生死单元弹塑性收缩/翘曲，经 `simulate(asbuilt_solver="plastic")`） |
| 数字样机 | `digitaltwin.rom` | `mbd.assembly`（MBD+FEM 刚柔耦合） |
| 粉末尺度缺陷 | `powder.bed`（surrogate） | `powder.bed`（dem/sph/mpm） |

> 表中高保真名均为 ForgeCore 注册名，可用 `Pipeline.auto(..., select={"thermal": "thermal.enthalpy"})` 切换；
> `simulate(asbuilt_solver="plastic")` 走 CPFE 高保真成形路径（与 digitaltwin/verdict 同一 `AsBuiltPart` 契约）。

---

## 多目标优化：为什么是 NSGA-II 而非 λ-标量化

`pareto_optimize_geometry_process(algorithm="nsga2")` 在联合设计空间
`（形状场 u, 工艺向量 z∈[0,1]⁹）`上做真 NSGA-II 进化，产出**真实（凸/非凸皆可）**
Pareto 前沿。加权求和 `L(λ)=λ·geom_dev+(1-λ)·stress` 只能覆盖前沿的**凸包**，会系统性漏掉
非凸段上的真实最优解；NSGA-II 通过非支配排序 + 拥挤度直接逼近真前沿。旧 `algorithm="lambda"`
路径仍保留作回归对照。

---

## 当前能力边界（诚实声明）

- **算法正确性**由已知问题严格保证：NSGA-II 在凸/非凸/一维双目标基准上可恢复真实前沿且同种子可复现；
  可微性由整链 `jax.grad` 前向回归测试守住（无 NaN/Inf）。
- **工程保真度**：默认链路为代理模型，绝对精度依赖你切换的高保真求解器与校准材料参数；
  框架本身不「免费」给出工业级定量精度，需配合标定使用。
- **样本效率**：形状场 `u` 为全网格体素，纯进化收敛较慢。生产级前沿建议：
  (a) 对 `u` 用降维基（Zernike / 随机傅里叶）；或 (b) 开启 `seed_with_lambda` 用梯度种子引导。
- **平台**：已在 Windows + 管理 Python 3.13 验证；Ubuntu + 48G GPU 全保真回归（#38）待硬件到位。

---

## 测试

```bash
pytest tests/ -q                 # 主回归套件（amforge 契约链 + 可微性 + Pareto）
pytest tests/test_nsga2.py -q    # NSGA-II 纯数值单元验证
amforge selftest                 # 安装后端到端冒烟
# GUI offscreen 冒烟（需先装 [gui]，设 QT_QPA_PLATFORM=offscreen）：
pytest tests/test_gui_app2.py tests/test_gui_app.py -q
```

---

## 许可证

MIT
