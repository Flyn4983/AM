#!/bin/bash
# #35 全量回归面（CPU 钉住）。起跑前后各取一次 src+tests 指纹 ⇒ 证明回归期间没有动过被测代码。
REPO="/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
LOG=/tmp/am_t35_full.log
finger() { cd "$REPO" && find src tests -name '*.py' -not -path '*__pycache__*' -print0 \
           | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1; }
cd "$REPO"
{ echo "CMD: CUDA_VISIBLE_DEVICES= JAX_ENABLE_X64=1 PYTHONPATH=src /tmp/amvenv/bin/python -m pytest -q"
  echo "TREE: head=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- src tests | wc -l)"
  echo "DATE_START: $(date '+%F %T %z')"
  echo "SRCFINGER_START: $(finger)"
} > "$LOG"
CUDA_VISIBLE_DEVICES= JAX_ENABLE_X64=1 PYTHONPATH=src /tmp/amvenv/bin/python -m pytest -q >> "$LOG" 2>&1
RC=$?
{ echo "SRCFINGER_END: $(finger)"
  echo "DATE_END: $(date '+%F %T %z')"
  echo "rc=$RC"
} >> "$LOG"
