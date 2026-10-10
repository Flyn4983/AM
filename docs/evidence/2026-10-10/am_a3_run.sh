#!/bin/bash
# A3 面的**启动记录**：表头 CMD 行的可执行副本（面表头里的 timeout/参数由此变成可复查读数，而不是散文）。
# 已跑（2026-10-10）：
#   #47＝#10/A3 腿①「只预检不解」：case preflight / controls（§26.29 归档表头 CMD 的逐字转录）。
#   #50＝A3 腿②smoke（dx=r_b，只验接线、J6/G6 按构造不过⇒只作 [SOFT] 报告）：本轮实跑的是
#     --tracks 1 的三条腿（缺省卡／--amb-k 293.15／--ic-probe），读数见 $E/_raw_a3_smoke_*.out 与
#     §26.29。下面 case smoke-amb／smoke-probe 是这两条腿的**逐字转录**，入库件已在 ⇒ 直接再跑会被
#     keep 拒绝（要重跑就换名）；case smoke（--tracks 1,2,3 的梯跑）**尚未跑**，排在 pilot 探针
#     四腿之后，避免与确定性控件 C 并发。
#   pilot **尚未整跑**（case pilot 仍是模板）；pilot 网格上的 #51 敏感度探针四腿在
#     am_a3_ic_clamp_probe.sh（另一支脚本），不在本脚本里。
#   full **尚未跑**，跑之前不许引用它的任何数字。
# 红线：全程 CUDA_VISIBLE_DEVICES="" ＋ JAX_PLATFORMS=cpu ⇒ 不碰 GPU；因此本脚本产出的墙钟只作
#   **可行性代价登记**，不得写成吞吐/加速/预算性能结论（性能数字一律等 RTX A6000 窗口）。
set -euo pipefail
# 仓库根＝本脚本所在目录往上三层；先把 dirname 绝对化，否则用相对路径调用本脚本时
#   「docs/evidence/2026-10-10/../..」只剥两层 ⇒ 落在 docs/（2026-10-10 14:3x 实测：四条腿全部
#   rc=2＝python 的「can't open file .../docs/docs/evidence/...」）。下面再显式验一次驱动在不在。
E_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$E_DIR/../../.."
E=docs/evidence/2026-10-10
[ -f "$E/am_a3_driver.py" ] || { echo "ABORT = 仓库根解错：$PWD 下没有 $E/am_a3_driver.py"; exit 4; }
echo "ROOT = $PWD"
PY=/tmp/amvenv/bin/python
TIMEOUT_S=590               # 单次运行上限（表头 CMD 里的那个 590 的出处）

# keep <入库前缀> </tmp 里的 stage 基名>：out/err 两份都 cp -p 入库；**目标已存在就拒绝**
#   ——上一轮有过「重跑覆写掉红件」的事故，所以每次尝试必须换名（见 §26.29 口径）。
keep() {
  local tag="$1" base="$2" ext
  for ext in out err; do
    if [ -e "$E/$tag.$ext" ]; then
      echo "REFUSE = $E/$tag.$ext 已存在 ⇒ 换名再跑（别覆写上一件，红件也是证据）"; exit 3
    fi
    cp -p "$base.$ext" "$E/$tag.$ext"
  done
  ls -l "$E/$tag.out" "$E/$tag.err"
}

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
  smoke)       # 已跑（本轮）：dx=r_b 接线档，J6/G6 走 [SOFT] ⇒ 不产出形态数字
    B=/tmp/a3_smoke_raw
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=src \
      timeout $TIMEOUT_S $PY $E/am_a3_driver.py --stage smoke --tracks 1,2,3 --arms norm \
      > $B.out 2> $B.err
    keep _raw_a3_smoke_tracks123_norm $B
    ;;
  smoke-amb)   # 已跑（本轮）：夹具保真腿＝--amb-k 293.15 让声明的 T0=20°C 可表示 ⇒ G8a/G8b 全绿
    B=/tmp/a3_smoke_amb_raw
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=src \
      timeout $TIMEOUT_S $PY $E/am_a3_driver.py --stage smoke --tracks 1 --arms norm \
        --amb-k 293.15 > $B.out 2> $B.err
    keep _raw_a3_smoke_g8fix_ambk29315 $B
    ;;
  smoke-probe) # 已跑（本轮）：--ic-probe 只验「G8 保留为 FAIL 时读数跑不跑完」的接线（smoke 档仍不印数字）
    B=/tmp/a3_smoke_probe_raw
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=src \
      timeout $TIMEOUT_S $PY $E/am_a3_driver.py --stage smoke --tracks 1 --arms norm \
        --ic-probe > $B.out 2> $B.err
    keep _raw_a3_smoke_icprobe $B
    ;;
  pilot)       # 模板：dx=r_b/2，N=1..3，两臂，取每道均摊实测（**缺省材料卡 ⇒ G8a 必红 ⇒ 只出墙钟不出形态**）
    B=/tmp/a3_pilot_raw
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=src \
      timeout $TIMEOUT_S $PY $E/am_a3_driver.py --stage pilot --tracks 1,2,3 --arms both \
      > $B.out 2> $B.err
    keep _raw_a3_pilot_tracks123_both $B
    ;;
  pilot-amb)   # 可评分的 pilot 面：--amb-k 293.15 让夹具声明的 T0=20°C 在焓模型里可表示
    #   （#51 的探针已实测：钳位对 solidus 宽/积的影响＝0.00µm／0µm²，见 _raw_a3ic_{A,B,C,D,E,F}）
    B=/tmp/a3_pilot_amb_raw
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=src \
      timeout $TIMEOUT_S $PY $E/am_a3_driver.py --stage pilot --tracks 1,2,3 --arms both \
        --amb-k 293.15 > $B.out 2> $B.err
    keep _raw_a3_pilot_amb_tracks123_both $B
    ;;
  *)
    echo "用法：am_a3_run.sh {preflight|controls|smoke|smoke-amb|smoke-probe|pilot|pilot-amb}（full 需 --measured-per-track，见 §26.29 (j)）"
    echo "预算闸：BUDGET_S ＝ 4 * 3600.0 s（出处＝am_a3_driver.py 的模块级常量，值由审计件现算）"
    exit 2
    ;;
esac
echo "rc=$? stage=$stage"
