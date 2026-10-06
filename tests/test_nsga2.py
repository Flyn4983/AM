"""NSGA-II 核心算法单元测试（纯数值，不依赖物理前向，快速可复现）。

验证 #37 的算法正确性：非支配排序、拥挤度、SBX/多项式变异、以及 NSGA-II 主循环
在已知双目标问题上恢复真实（凸/非凸）Pareto 前沿且确定性可复现。
"""
import numpy as np
import jax.numpy as jnp

from amforge.nsga2 import (
    NSGA2Config, fast_non_dominated_sort, crowding_distance,
    sbx_crossover, polynomial_mutation, nsga2_minimize,
)


def _assert_non_dominated(F):
    """断言矩阵 F 的每一行彼此互不支配（即整张表即为第一非支配层）。"""
    fronts = fast_non_dominated_sort(F)
    assert fronts[0] == list(range(F.shape[0])), \
        f"前沿并非全部非支配: fronts[0]={fronts[0]}"


def test_fast_non_dominated_sort_convex_front():
    """凸权衡的 4 点互不支配；加入一个被支配点应落入第 2 层。"""
    # A=(1,4) B=(2,3) C=(3,2) D=(4,1) 凸前沿，互不可支配
    # E=(3,3) 被 B=(2,3) 与 C=(3,2) 同时支配
    F = np.array([[1., 4.], [2., 3.], [3., 2.], [4., 1.], [3., 3.]])
    fronts = fast_non_dominated_sort(F)
    assert fronts[0] == [0, 1, 2, 3], f"front0 应为凸前沿四点, 得到 {fronts[0]}"
    assert fronts[1] == [4], f"第2层应为被支配点 E, 得到 {fronts[1]}"


def test_fast_non_dominated_sort_nonconvex():
    """非凸前沿：加权求和会漏掉凹段，非支配排序应完整保留。

    A=(1,1) 支配一切；B=(2,3) 与 C=(3,2) 凹段互不可支配；D=(4,4) 被 B、C 支配。
    """
    F = np.array([[1., 1.], [2., 3.], [3., 2.], [4., 4.]])
    fronts = fast_non_dominated_sort(F)
    assert fronts[0] == [0], f"front0 应仅 A, 得到 {fronts[0]}"
    # B 与 C 凹段同属第 2 层（互不支配），D 被二者支配落入第 3 层
    assert set(fronts[1]) == {1, 2}, f"front1 应为凹段 B,C, 得到 {fronts[1]}"
    assert fronts[2] == [3], f"front2 应为 D, 得到 {fronts[2]}"


def test_crowding_distance_boundary_is_inf():
    """单层内沿某目标排序后，最小/最大个体的拥挤度应为 +∞。"""
    F = np.array([[0.0, 5.0], [1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    cd = crowding_distance(F, [0, 1, 2, 3, 4])
    assert np.isinf(cd[0]) and np.isinf(cd[4]), "边界个体拥挤度应为 inf"
    assert np.all(np.isfinite(cd[1:4])), "中间个体拥挤度应有限"


def test_crowding_distance_le_two_is_inf():
    """层内 ≤2 个个体时全部给 +∞（保证极端解必留存）。"""
    F = np.array([[0.0, 1.0], [1.0, 0.0]])
    cd = crowding_distance(F, [0, 1])
    assert np.all(np.isinf(cd))


def test_sbx_identical_parents_yield_parents():
    """父代完全相同 → SBX 子代应等于父代（退化情形稳定）。"""
    rng = np.random.default_rng(0)
    x = np.array([0.3, 0.7, 0.1, 0.9])
    lb = np.zeros(4); ub = np.ones(4)
    c1, c2 = sbx_crossover(x, x.copy(), lb, ub, eta_c=15.0, rng=rng)
    assert np.allclose(c1, x) and np.allclose(c2, x), "相同父代 SBX 应返回父代"


def test_sbx_stays_in_bounds():
    """SBX 子代必须落在 [lb, ub] 内。"""
    rng = np.random.default_rng(7)
    lb = np.array([-2.0, 0.0]); ub = np.array([2.0, 5.0])
    for _ in range(200):
        x1 = rng.uniform(lb, ub); x2 = rng.uniform(lb, ub)
        c1, c2 = sbx_crossover(x1, x2, lb, ub, eta_c=15.0, rng=rng)
        assert np.all(c1 >= lb - 1e-9) and np.all(c1 <= ub + 1e-9)
        assert np.all(c2 >= lb - 1e-9) and np.all(c2 <= ub + 1e-9)


def test_polynomial_mutation_stays_in_bounds():
    """多项式变异结果必须落在 [lb, ub] 内。"""
    rng = np.random.default_rng(3)
    lb = np.array([-1.0]); ub = np.array([1.0])
    for _ in range(200):
        x = rng.uniform(lb, ub)
        xm = polynomial_mutation(x, lb, ub, eta_m=20.0, p_mut=1.0, rng=rng)
        assert lb - 1e-9 <= xm <= ub + 1e-9


def _obj_quad(x):
    """经典一维双目标：f1=x², f2=(x-2)²，真实 Pareto 前沿为 x∈[0,2]。"""
    x = float(np.asarray(x, dtype=np.float64).ravel()[0])
    return np.array([x * x, (x - 2.0) ** 2])


def test_nsga2_recovers_true_front():
    """NSGA-II 在一维双目标问题上恢复真实前沿（凸），且全部非支配。"""
    cfg = NSGA2Config(pop_size=40, n_gen=50, seed=0)
    res = nsga2_minimize(_obj_quad, np.array([-2.0]), np.array([2.0]), 2, cfg)
    F = res.pareto_f
    assert F.shape[0] >= 5, f"前沿过薄: {F.shape[0]}"
    _assert_non_dominated(F)
    # 前沿应覆盖真实区：f1∈[0,4]，最小 f1≈0（x≈0），最大 f1≈4（x≈2）
    assert F[:, 0].min() < 0.5, f"前沿未触及 x≈0 (min f1={F[:,0].min():.3f})"
    assert F[:, 0].max() > 3.0, f"前沿未触及 x≈2 (max f1={F[:,0].max():.3f})"
    # 对应设计变量应落在真实 Pareto 区 [0,2]
    assert np.all(res.pareto_x[:, 0] >= -0.05) and np.all(res.pareto_x[:, 0] <= 2.05)


def test_nsga2_deterministic_with_seed():
    """相同 seed 复现相同前沿（回归可复现性）。"""
    cfg_a = NSGA2Config(pop_size=30, n_gen=30, seed=42)
    cfg_b = NSGA2Config(pop_size=30, n_gen=30, seed=42)
    ra = nsga2_minimize(_obj_quad, np.array([-2.0]), np.array([2.0]), 2, cfg_a)
    rb = nsga2_minimize(_obj_quad, np.array([-2.0]), np.array([2.0]), 2, cfg_b)
    assert np.allclose(ra.pareto_f, rb.pareto_f), "相同 seed 应复现相同前沿"
    assert np.allclose(ra.pareto_x, rb.pareto_x)


def test_nsga2_init_pop_respected():
    """传入 init_pop 时种群规模应等于 pop_size（初始化路径正确）。"""
    rng = np.random.default_rng(0)
    init = rng.uniform(-2, 2, size=(20, 1))
    cfg = NSGA2Config(pop_size=20, n_gen=5, seed=1)
    res = nsga2_minimize(_obj_quad, np.array([-2.0]), np.array([2.0]), 2,
                         cfg, init_pop=init)
    assert res.population_x.shape == (20, 1)
