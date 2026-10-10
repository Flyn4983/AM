#!/bin/bash
# A3 面的**启动记录**：表头 CMD 行的可执行副本（面表头里的 timeout/参数由此变成可复查读数，而不是散文）。
# 已跑（2026-10-10，#47＝#10/A3 腿①「只预检不解」，两条都是 §26.29 归档表头 CMD 的逐字转录）：
#   见下面 case preflight / controls。
# 模板（**尚未跑**，只是把 §26.29 (j) 的排期落成命令，跑之前不许引用它们的任何数字）：
#   smoke / pilot / full。
# 红线：全程 CUDA_VISIBLE_DEVICES="" ＋ JAX_PLATFORMS=cpu ⇒ 不碰 GPU；因此本脚本产出的墙钟只作
#   **可行性代价登记**，不得写成吞吐/加速/预算性能结论（性能数字一律等 RTX A6000 窗口）。
set -euo pipefail
cd "$(dirname "$0")/../.."   # 仓库根
E=docs/evidence/2026-10-10
PY=/tmp/amvenv/bin/python
TIMEOUT_S=590               # 单次运行上限（表头 CMD 里的那个 590 的出处）

stage="${1:-}"
case "$stage" in
  preflight)   # 已跑：主面，零次求解
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=src \
      timeout $TIMEOUT_S $PY $E/am_a3_driver.py --stage preflight --tracks 1,2,3 --arms norm \
      > /tmp/a3_preflight_raw.out 2> /tmp/a3_preflight_raw.err
    cp -p /tmp/a3_preflight_raw.out $E/_raw_a3_preflight.out
    cp -p /tmp/a3_preflight_raw.err $E/_raw_a3_preflight.err
    ;;
  controls)    # 已跑：控件面（判据敏感性，同样零次求解）
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=src:$E \
      timeout $TIMEOUT_S $PY $E/am_a3_preflight_controls.py \
      > /tmp/a3_ctrl_raw.out 2> /tmp/a3_ctrl_raw.err
    cp -p /tmp/a3_ctrl_raw.out $E/_raw_a3_preflight_controls.out
    cp -p /tmp/a3_ctrl_raw.err $E/_raw_a3_preflight_controls.err
    ;;
  smoke)       # 模板：dx=r_b（J6 档），只验接线 ⇒ **不产出形态数字**
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=src \
      timeout $TIMEOUT_S $PY $E/am_a3_driver.py --stage smoke --tracks 1,2,3 --arms norm \
      > /tmp/a3_smoke_raw.out 2> /tmp/a3_smoke_raw.err
    ;;
  pilot)       # 模板：dx=r_b/2，N=1..3，两臂，取每道均摊实测
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=src \
      timeout $TIMEOUT_S $PY $E/am_a3_driver.py --stage pilot --tracks 1,2,3 --arms both \
      > /tmp/a3_pilot_raw.out 2> /tmp/a3_pilot_raw.err
    ;;
  *)
    echo "用法：am_a3_run.sh {preflight|controls|smoke|pilot}（full 需 --measured-per-track，见 §26.29 (j)）"
    echo "预算闸：BUDGET_S ＝ 4 * 3600.0 s（出处＝am_a3_driver.py 的模块级常量，值由审计件现算）"
    exit 2
    ;;
esac
echo "rc=$? stage=$stage"
