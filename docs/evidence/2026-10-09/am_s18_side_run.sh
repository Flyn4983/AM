#!/bin/bash
# #18 侧翼两面（CPU 钉住）：新网件的收集数 ＋ 「新网件＋那条登记红灯」同跑的定向面。
# 两面各自印 CMD/ENV/DATE ⇒ 归档件自带命令口径（首版定向面的 stdout 没带命令行，无法证明跑的是哪两个文件）。
REPO="/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
LOG=/tmp/am_s18_side.log
cd "$REPO" || exit 9
PIN=(env CUDA_VISIBLE_DEVICES= JAX_ENABLE_X64=1 PYTHONPATH=src /tmp/amvenv/bin/python -m pytest)
{ echo "===== COLLECT 段 ====="
  echo "CMD: CUDA_VISIBLE_DEVICES= JAX_ENABLE_X64=1 PYTHONPATH=src /tmp/amvenv/bin/python -m pytest -q --collect-only tests/test_scan_program.py"
  echo "DATE: $(date '+%F %T %z')"
  "${PIN[@]}" -q --collect-only tests/test_scan_program.py
  echo "rc_col=$?"
  echo "===== TARGETED 段 ====="
  echo "CMD: CUDA_VISIBLE_DEVICES= JAX_ENABLE_X64=1 PYTHONPATH=src /tmp/amvenv/bin/python -m pytest -q tests/test_scan_program.py tests/test_enthalpy_thermal.py"
  echo "DATE: $(date '+%F %T %z')"
  "${PIN[@]}" -q tests/test_scan_program.py tests/test_enthalpy_thermal.py
  echo "rc_tgt=$?"
} > "$LOG" 2>&1
