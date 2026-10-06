"""NSGA-II 多目标进化算法核心（Deb & Agrawal, 2002）。

与 AMForge 物理前向**完全解耦**：仅消费目标向量 ``f ∈ ℝ^m``，产出
**非支配前沿（non-dominated front）+ 拥挤度（crowding distance）**。设计变量
``x`` 为任意实数向量（在 AMForge 的几何+工艺协同优化中即 ``concat(u_flat, z)``）。

为什么需要它（对应任务 #37）：原 ``pareto_optimize_geometry_process`` 用
``λ``-标量化 ``L(λ)=λ·geom_dev+(1-λ)·stress`` 扫描生成"前沿"。但加权求和**只能
覆盖 Pareto 前沿的凸包（convex hull）**，一旦真实前沿出现非凸段（尺寸精度与残余
应力此消彼长且存在权衡拐点），``λ``-扫描会**漏掉**凹段上的真实最优解。NSGA-II 用
**快速非支配排序 + 拥挤度多样性**直接逼近真实（凸/非凸皆可）Pareto 前沿，是
多目标优化而非"多起点单目标"。

实现要点（工程严谨，非轻量妥协）：

* ``fast_non_dominated_sort``：经典 O(m·N²) 非支配分层，返回各层个体索引；
* ``crowding_distance``：每层内沿各目标排序后的相邻间距和，边界解给 ``+∞``
  保证极端解必被保留，中间解按密度稀疏化；
* ``sbx_crossover``：模拟二进制交叉（SBX），分布指数 ``eta_c`` 控制子代贴近父代程度；
* ``polynomial_mutation``：多项式变异，分布指数 ``eta_m``；
* ``crowded_binary_tournament``：二元锦标赛，先比非支配秩、再比拥挤度；
* ``nsga2_minimize``：主循环 = 变异生成子代 → 父子合并(2N) → 非支配排序 →
  按 (秩, 拥挤度) 截断回 N。

确定性：``NSGA2Config.seed`` 经 ``numpy.random.default_rng`` 控制，相同 seed 复现
相同前沿（回归测试据此断言可复现性）。目标函数 ``objective`` 在 Python 循环中逐个
求值（不 jit），以保证与 tracer/numpy 混合的物理前向（``_joint_forward``）兼容。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Sequence

__all__ = [
    "NSGA2Config", "NSGA2Result",
    "fast_non_dominated_sort", "crowding_distance",
    "sbx_crossover", "polynomial_mutation", "crowded_binary_tournament",
    "nsga2_minimize",
]


# ---------------------------------------------------------------------------
# 配置与结果容器
# ---------------------------------------------------------------------------
@dataclass
class NSGA2Config:
    """NSGA-II 超参。``pop_size`` 种群规模；``n_gen`` 进化代数。

    ``eta_c``（SBX 分布指数，越大子代越贴父代）、``eta_m``（多项式变异分布指数）、
    ``p_crossover``（每对父代交叉概率）、``p_mut_gene``（每基因变异概率，
    设为 ``0`` 时自动取经典值 ``1/n_var``）。
    """
    pop_size: int = 24
    n_gen: int = 30
    eta_c: float = 15.0
    eta_m: float = 20.0
    p_crossover: float = 0.9
    p_mut_gene: float = 0.0       # 0 ⇒ 经典 1/n_var
    seed: int = 0
    u_clip: float = 3.0           # 设计变量软边界（仅用于 SBX 截断，非硬约束）


@dataclass
class NSGA2Result:
    """NSGA-II 结果。

    ``pareto_x`` / ``pareto_f``：最终非支配前沿的设计向量与目标向量（按目标数 m 排列）；
    ``population_x`` / ``population_f``：末代完整种群（含被支配解，用于多样性分析）；
    ``pareto_indices``：末代种群中非支配个体的索引；``config``：回传配置。
    """
    pareto_x: np.ndarray
    pareto_f: np.ndarray
    population_x: np.ndarray
    population_f: np.ndarray
    pareto_indices: Sequence[int]
    config: NSGA2Config
    # 整个进化过程见到的全局理想点 (min f1, min f2, ...) —— 供 utopia 估计
    ideal_point: tuple = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# 1. 支配关系与非支配排序
# ---------------------------------------------------------------------------
def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """``a`` 支配 ``b``（最小化）：``a`` 各目标 ``≤ b`` 且至少一项严格 ``<``。"""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> list[list[int]]:
    """快速非支配排序（Deb 2002）。

    :param F: 目标矩阵 ``(N, m)``，第 i 行是第 i 个个体的 m 个目标（均最小化）。
    :return: 分层列表，``fronts[0]`` 为第一非支配层（最优），依次递减。
    """
    F = np.asarray(F, dtype=np.float64)
    N = F.shape[0]
    S = [[] for _ in range(N)]          # S[p]：p 支配的个体
    n = np.zeros(N, dtype=int)          # n[p]：支配 p 的个体数
    fronts: list[list[int]] = [[]]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if _dominates(F[p], F[q]):
                S[p].append(q)
            elif _dominates(F[q], F[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    if fronts and not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(F: np.ndarray, front: Sequence[int]) -> np.ndarray:
    """计算某非支配层内各个体的拥挤度（沿各目标归一化相邻间距之和）。

    边界（该目标最大/最小）个体拥挤度置 ``+∞``，保证极端解必留存；层内 ≤2 个
    个体时全部给 ``+∞``。返回长度 ``len(front)`` 的数组，与 ``front`` 顺序对齐。
    """
    F = np.asarray(F, dtype=np.float64)
    k = len(front)
    cd = np.zeros(k, dtype=np.float64)
    if k <= 2:
        cd[:] = np.inf
        return cd
    Ff = F[list(front)]                 # (k, m)
    m = Ff.shape[1]
    for obj in range(m):
        order = np.argsort(Ff[:, obj], kind="stable")
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        fmin = Ff[order[0], obj]
        fmax = Ff[order[-1], obj]
        rng = fmax - fmin
        if rng <= 0.0:
            continue
        for idx in range(1, k - 1):
            cd[order[idx]] += (Ff[order[idx + 1], obj] - Ff[order[idx - 1], obj]) / rng
    return cd


# ---------------------------------------------------------------------------
# 2. 变异算子（SBX 交叉 + 多项式变异）
# ---------------------------------------------------------------------------
def sbx_crossover(x1: np.ndarray, x2: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                  eta_c: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """模拟二进制交叉（Simulated Binary Crossover, Deb & Agrawal 1995）。

    逐基因以 0.5 概率交叉；子代分布由 ``beta_q`` 控制，``eta_c`` 越大子代越贴近
    父代。全程向量化（对 738 维设计向量也高效）。返回两个子代。
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    n = x1.shape[0]
    rand = rng.random(n)               # 是否对该基因交叉
    do = rand <= 0.5
    xmin = np.minimum(x1, x2)
    xmax = np.maximum(x1, x2)
    dx = np.maximum(np.abs(x1 - x2), 1e-14)     # 防止除零（dx→0 时 beta_q*dx→0）
    beta = 1.0 + 2.0 * (xmin - lb) / dx
    alpha = 2.0 - np.power(beta, -(eta_c + 1.0))
    u = rng.random(n)
    beta_q = np.where(
        u <= 1.0 / alpha,
        np.power(u * alpha, 1.0 / (eta_c + 1.0)),
        np.power(1.0 / (2.0 - u * alpha), 1.0 / (eta_c + 1.0)),
    )
    c1 = 0.5 * ((x1 + x2) - beta_q * dx)
    c2 = 0.5 * ((x1 + x2) + beta_q * dx)
    c1 = np.where(do, c1, x1)
    c2 = np.where(do, c2, x2)
    return np.clip(c1, lb, ub), np.clip(c2, lb, ub)


def polynomial_mutation(x: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                         eta_m: float, p_mut: float,
                         rng: np.random.Generator) -> np.ndarray:
    """多项式变异（Deb & Agrawal 1999）。

    每基因以概率 ``p_mut`` 变异，扰动量由 ``delta`` 给出，``eta_m`` 越大越贴近原值。
    """
    x = np.asarray(x, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    n = x.shape[0]
    mask = rng.random(n) < p_mut
    u = rng.random(n)
    delta = np.where(
        u < 0.5,
        np.power(2.0 * u, 1.0 / (eta_m + 1.0)) - 1.0,
        1.0 - np.power(2.0 * (1.0 - u), 1.0 / (eta_m + 1.0)),
    )
    xnew = x + delta * (ub - lb)
    xnew = np.where(mask, xnew, x)
    return np.clip(xnew, lb, ub)


# ---------------------------------------------------------------------------
# 3. 选择算子
# ---------------------------------------------------------------------------
def crowded_binary_tournament(rank: np.ndarray, cd: np.ndarray,
                              rng: np.random.Generator) -> int:
    """拥挤度二元锦标赛：随机抽两人，先比非支配秩（小优），同秩比拥挤度（大优）。"""
    a, b = rng.integers(0, len(rank), 2)
    if rank[a] < rank[b]:
        return int(a)
    if rank[b] < rank[a]:
        return int(b)
    return int(a) if cd[a] >= cd[b] else int(b)


# ---------------------------------------------------------------------------
# 4. 主循环
# ---------------------------------------------------------------------------
def nsga2_minimize(objective: Callable[[np.ndarray], np.ndarray],
                   lb: np.ndarray, ub: np.ndarray, n_obj: int,
                   config: NSGA2Config, *, init_pop: np.ndarray | None = None,
                   verbose: bool = False) -> NSGA2Result:
    """运行 NSGA-II，返回非支配前沿。

    :param objective: ``x (1d array) -> f (1d, 长度 n_obj)``，最小化。在 Python
        循环中逐个求值（兼容 tracer/numpy 混合的物理前向）。
    :param lb, ub: 设计变量下/上界（1d，长度 = 变量数）。
    :param n_obj: 目标维数 ``m``。
    :param config: 超参（见 :class:`NSGA2Config`）。
    :param init_pop: 可选初始种群 ``(pop_size, n_var)``；不提供则均匀随机初始化。
    :return: :class:`NSGA2Result`。
    """
    cfg = config
    rng = np.random.default_rng(cfg.seed)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    n_var = lb.shape[0]
    N = int(cfg.pop_size)
    p_mut = cfg.p_mut_gene if cfg.p_mut_gene > 0 else 1.0 / n_var

    def _eval(pop_2d: np.ndarray) -> np.ndarray:
        rows = []
        for ind in pop_2d:
            f = np.asarray(objective(ind), dtype=np.float64).ravel()
            if f.shape[0] != n_obj:
                raise ValueError(
                    f"objective 返回维度 {f.shape[0]} 与 n_obj={n_obj} 不符")
            rows.append(f)
        return np.stack(rows, axis=0)

    # 初始种群
    if init_pop is not None:
        pop = np.asarray(init_pop, dtype=np.float64)
        if pop.shape[0] != N:
            raise ValueError(f"init_pop 行数 {pop.shape[0]} 与 pop_size={N} 不符")
        pop = np.clip(pop, lb, ub)
    else:
        pop = rng.uniform(lb, ub, size=(N, n_var))
    F = _eval(pop)
    ideal = tuple(float(v) for v in np.min(F, axis=0))

    for gen in range(int(cfg.n_gen)):
        # ---- 当前种群的非支配秩 + 拥挤度 ----
        fronts = fast_non_dominated_sort(F)
        cd = np.zeros(N, dtype=np.float64)
        for fr in fronts:
            cd[list(fr)] = crowding_distance(F, fr)
        rank = np.zeros(N, dtype=int)
        for r, fr in enumerate(fronts):
            for i in fr:
                rank[i] = r

        # ---- 变异生成子代（size = N）----
        off: list[np.ndarray] = []
        while len(off) < N:
            p1 = crowded_binary_tournament(rank, cd, rng)
            p2 = crowded_binary_tournament(rank, cd, rng)
            c1, c2 = sbx_crossover(pop[p1], pop[p2], lb, ub, cfg.eta_c, rng)
            c1 = polynomial_mutation(c1, lb, ub, cfg.eta_m, p_mut, rng)
            c2 = polynomial_mutation(c2, lb, ub, cfg.eta_m, p_mut, rng)
            off.append(c1)
            if len(off) < N:
                off.append(c2)
        off = np.stack(off, axis=0)
        Foff = _eval(off)

        # ---- 父子合并(2N) → 非支配排序 → 截断回 N ----
        comb = np.concatenate([pop, off], axis=0)
        Fcomb = np.concatenate([F, Foff], axis=0)
        M = comb.shape[0]
        cfronts = fast_non_dominated_sort(Fcomb)
        ccd = np.zeros(M, dtype=np.float64)
        for fr in cfronts:
            ccd[list(fr)] = crowding_distance(Fcomb, fr)
        cr = np.zeros(M, dtype=int)
        for r, fr in enumerate(cfronts):
            for i in fr:
                cr[i] = r

        new_pop: list[np.ndarray] = []
        new_F: list[np.ndarray] = []
        for fr in cfronts:
            if len(new_pop) + len(fr) <= N:
                new_pop.extend(comb[list(fr)])
                new_F.extend(Fcomb[list(fr)])
            else:
                order = sorted(fr, key=lambda i: -ccd[i])   # 拥挤度大者优先
                need = N - len(new_pop)
                for i in order[:need]:
                    new_pop.append(comb[i])
                    new_F.append(Fcomb[i])
                break
        pop = np.stack(new_pop, axis=0)
        F = np.stack(new_F, axis=0)
        gmin = np.min(F, axis=0)
        ideal = tuple(min(ideal[j], float(gmin[j])) for j in range(n_obj))

        if verbose:
            n_rank0 = len(cfronts[0]) if cfronts else 0
            print(f"  nsga2 gen {gen:3d}  fronts={len(cfronts)}  "
                  f"rank0={n_rank0}  ideal=({', '.join(f'{v:.3e}' for v in ideal)})")

    # ---- 末代非支配前沿 ----
    final_fronts = fast_non_dominated_sort(F)
    pareto_idx = list(final_fronts[0])
    return NSGA2Result(
        pareto_x=pop[pareto_idx],
        pareto_f=F[pareto_idx],
        population_x=pop,
        population_f=F,
        pareto_indices=pareto_idx,
        config=cfg,
        ideal_point=ideal,
    )
