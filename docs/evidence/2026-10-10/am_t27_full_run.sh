#!/bin/bash
# #27 代码腿全量回归面（CPU 钉住；不碰 GPU）。
# 起跑前后各取一次 src+tests 指纹 ⇒ 证明回归期间没有动过被测代码。
# 期望记分＝基线 (2 failed, 281 passed, 3 skipped) ＋ 5 条新测试（默认档未切换
#   ⇒ 两条红灯必须逐位不变：A0 test_meltpool_converges_across_beam_resolving_grids
#   与 #27 test_convection_removes_energy，后者打印 2908.771450618092 vs 2937.060997302282）。
REPO="/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
LOG=/tmp/am_t27_full.log
finger() { cd "$REPO" && find src tests -name '*.py' -not -path '*__pycache__*' -print0 \
           | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1; }
cd "$REPO"
{ echo "CMD: CUDA_VISIBLE_DEVICES= JAX_ENABLE_X64=1 PYTHONPATH=src /tmp/amvenv/bin/python -m pytest -q"
  echo "TREE: head=$(git -c core.quotePath=false rev-parse --short HEAD) dirty=$(git -c core.quotePath=false status --porcelain -- src tests | wc -l)"
  echo "DATE_START: $(date '+%F %T %z')"
  echo "SRCFINGER_START: $(finger)"
} > "$LOG"
CUDA_VISIBLE_DEVICES= JAX_ENABLE_X64=1 PYTHONPATH=src /tmp/amvenv/bin/python -m pytest -q >> "$LOG" 2>&1
RC=$?
{ echo "SRCFINGER_END: $(finger)"
  echo "DATE_END: $(date '+%F %T %z')"
  echo "rc=$RC"
} >> "$LOG"
