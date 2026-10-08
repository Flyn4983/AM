#!/bin/bash
# 归档 #34/#36 全量回归记分：/tmp 原始 log → docs/evidence/2026-10-08/
# 头部全部数值由命令现取（不许手抄）；正文逐字节保留原 log。
set -u
REPO="/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
SRC="${SRCOVERRIDE:-/tmp/am_t34_t36_full_regression.log}"
REF="$REPO/docs/evidence/2026-10-08/am_t4_full_regression.log"
cd "$REPO" || exit 9

# --- G0 守卫：回归进程必须已退出 ---
G0=$(ps -eo pid=,comm=,args= | awk '$2 ~ /^python/ && index($0,"m pytest -q -rA -p no:cacheprovider") {c++} END{print c+0}')
echo "G0 pytest_still_running=$G0"
if [ "${NOG0:-0}" != "1" ] && [ "$G0" != "0" ]; then echo "REFUSE: 回归仍在跑，不得归档"; exit 8; fi

# --- G1 树指纹：跑后再取一次，必须与跑前逐字符相同 ---
FP_SRC=$(find src -name '*.py' -not -path '*__pycache__*' -type f | LC_ALL=C sort | xargs md5sum | md5sum | awk '{print $1}')
FP_ALL=$(find src tests -name '*.py' -not -path '*__pycache__*' -type f | LC_ALL=C sort | xargs md5sum | md5sum | awk '{print $1}')
NEWER=$(find src tests -newermt "2026-10-08 20:47:00" -type f -not -path "*__pycache__*" | wc -l)
PRE_SRC=$(grep -A1 'concat src/\*\*' /tmp/am_t36_tree_pre.txt | tail -1 | grep -oE '[0-9a-f]{32}')
PRE_ALL=$(grep -A1 'concat src+tests' /tmp/am_t36_tree_pre.txt | tail -1 | grep -oE '[0-9a-f]{32}')
[ "$FP_SRC" = "$PRE_SRC" ] && G1a=MATCH || G1a=DIFF
[ "$FP_ALL" = "$PRE_ALL" ] && G1b=MATCH || G1b=DIFF
echo "G1 fp_src=$FP_SRC vs pre=$PRE_SRC -> $G1a"
echo "G1 fp_all=$FP_ALL vs pre=$PRE_ALL -> $G1b"
echo "G1 newer_than_launch=$NEWER (must be 0)"

# --- 记分：汇总行 ---
TALLY=$(grep -E '^[0-9]+ (failed|passed)' "$SRC" | tail -1)
[ -z "$TALLY" ] && { echo "REFUSE: 没读到汇总行"; tail -5 "$SRC"; exit 7; }
echo "TALLY=<$TALLY>"
N_FAIL=$(grep -oE '[0-9]+ failed' <<<"$TALLY" | grep -oE '[0-9]+'); N_FAIL=${N_FAIL:-0}
N_PASS=$(grep -oE '[0-9]+ passed' <<<"$TALLY" | grep -oE '[0-9]+')
N_SKIP=$(grep -oE '[0-9]+ skipped' <<<"$TALLY" | grep -oE '[0-9]+'); N_SKIP=${N_SKIP:-0}
O1=$((N_FAIL + N_PASS + N_SKIP))
WALL=$(grep -oE 'in [0-9.]+s' <<<"$TALLY" | head -1)

# --- ② 进度字符（模式已在 am_t4_full_regression.log 上对过已知真值 252/2/1）---
PROG=$(grep -E '^(\.|F|s)+ *(\[ *[0-9]+%\])?$' "$SRC" | tr -d ' ')
C_DOT=$(tr -cd '.' <<<"$PROG" | wc -c); C_F=$(tr -cd 'F' <<<"$PROG" | wc -c); C_S=$(tr -cd 's' <<<"$PROG" | wc -c)
O2=$((C_DOT + C_F + C_S))
NLINES=$(grep -cE '^(\.|F|s)+ *(\[ *[0-9]+%\])?$' "$SRC")
# --- ③ -rA 短汇总各结局行数 ---
R_PASS=$(grep -c '^PASSED ' "$SRC"); R_FAIL=$(grep -c '^FAILED ' "$SRC"); R_SKIP=$(grep -c '^SKIPPED ' "$SRC")
O3=$((R_PASS + R_FAIL + R_SKIP))
# --- ④ 收集数（跑后实测；环境必须与回归一致，否则 diffmech 导不进来）---
if [ "${DRY:-0}" = "1" ]; then COLL_OUT="261 tests collected in 1.00s"; else
COLL_OUT=$(CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 PYTHONPATH=src /tmp/amvenv/bin/python -m pytest --collect-only -q -p no:cacheprovider 2>/dev/null | tail -1); fi
COLL=$(grep -oE '^[0-9]+' <<<"$COLL_OUT"); COLL=${COLL:-NA}
echo "① 汇总行结局=$O1 ($N_FAIL failed/$N_PASS passed/$N_SKIP skipped) $WALL"
echo "② 进度字符行=$NLINES . =$C_DOT F=$C_F s=$C_S 合计=$O2"
echo "③ -rA 行 PASSED=$R_PASS FAILED=$R_FAIL SKIPPED=$R_SKIP 合计=$O3"
echo "④ collect-only 末行=<$COLL_OUT> ⇒ collected=$COLL"

# --- 两条登记红灯：与上一轮正文逐字节比对 + 扰动正对照 ---
grep '^E       AssertionError: ' "$REF" > /tmp/am_t36_reds_ref.txt
grep '^E       AssertionError: ' "$SRC" > /tmp/am_t36_reds_new.txt
NREF=$(wc -l < /tmp/am_t36_reds_ref.txt); NNEW=$(wc -l < /tmp/am_t36_reds_new.txt)
if diff -q /tmp/am_t36_reds_ref.txt /tmp/am_t36_reds_new.txt >/dev/null; then REDS=IDENTICAL; else REDS=DIFF; fi
sed 's/2908\.771450618092/2908.771450618093/' /tmp/am_t36_reds_new.txt > /tmp/am_t36_reds_pert.txt
if diff -q /tmp/am_t36_reds_ref.txt /tmp/am_t36_reds_pert.txt >/dev/null; then PERT="仍绿⇒尺子坏了"; else PERT="立刻变红"; fi
echo "REDS: ref_lines=$NREF new_lines=$NNEW diff=$REDS | 扰动(末位+1ulp)=>$PERT"

# --- G2 四路计数互洽闸门（不是我把数字读一遍，而是让它能红）---
if [ "$O1" = "$O3" ] && [ "$O2" = "$COLL" ] && [ "$((COLL + 2))" = "$O1" ] && [ "$G1a$G1b" = "MATCHMATCH" ] && [ "$NEWER" = "0" ]; then
  CONSIST=PASS
else
  CONSIST="FAIL(①=③?$O1/$O3 ②=collected?$O2/$COLL ①=collected+2?$O1/$((COLL+2)) G1=$G1a/$G1b newer=$NEWER)"
fi
echo "G2 四路互洽=$CONSIST"
# 扰动正对照：把 COLL 人为 -1，同一闸门必须 FAIL
if [ "$((COLL + 2 - 1))" = "$O1" ]; then PERT2=仍绿_尺子坏了; else PERT2=立刻变红; fi
echo "G2 正对照(COLL 少 1)=>$PERT2"

# --- G3 红灯身份：测试名从参照件正文 FAILURES 段现取，不手抄 ---
REF_NAMES=$(grep -oE '^_{5,} [a-z_0-9]+ _{5,}$' "$REF" | awk '{print $2}' | LC_ALL=C sort | tr '\n' ' ')
NEW_NAMES=$(grep '^FAILED ' "$SRC" | sed 's/^FAILED [^:]*:://; s/ -.*//' | LC_ALL=C sort | tr '\n' ' ')
if [ "$(echo "$REF_NAMES" | tr ' ' '\n' | LC_ALL=C sort)" = "$(echo "$NEW_NAMES" | tr ' ' '\n' | LC_ALL=C sort)" ]; then G3=MATCH; else G3=DIFF; fi
echo "G3 红灯身份 ref=<$REF_NAMES> new=<$NEW_NAMES> -> $G3"
if [ "${NOG2:-0}" != "1" ] && { [ "$CONSIST" != "PASS" ] || [ "$G3" != "MATCH" ] || [ "$REDS" != "IDENTICAL" ] || [ "$PERT" = "仍绿⇒尺子坏了" ]; }; then
  echo "REFUSE: 闸门未过 CONSIST=$CONSIST G3=$G3 REDS=$REDS 扰动=$PERT"; exit 6
fi

# --- skip 身份：取自 -rA 的 SKIPPED 行，整行原文（不按位置反推）---
SKID=$(grep '^SKIPPED ' "$SRC" | sed 's/^SKIPPED /NOTE:        /; s/^  /NOTE:        /')

# --- 头部：模板走 quoted heredoc，数值由命令值经环境注入 ---
export V_CMDLINE="$(cat /tmp/am_t36_cmdline.txt)"
export V_ENVLINE="$(tr '\n' ' ' < /tmp/am_t36_env.txt)"
export V_HEADSHA="$(git log -1 --format='%h')"
export V_DIRTY="$(git -c core.quotePath=false status --porcelain -- src tests | wc -l)"
export V_POSTTIME="$(date '+%T')"
export V_NEWER="$NEWER"; export V_FPSRC="$FP_SRC"; export V_FPALL="$FP_ALL"
export V_TALLY="$TALLY"; export V_WALL="$WALL"
export V_O1="$O1"; export V_NFAIL="$N_FAIL"; export V_NPASS="$N_PASS"; export V_NSKIP="$N_SKIP"
export V_NLINES="$NLINES"; export V_CDOT="$C_DOT"; export V_CF="$C_F"; export V_CS="$C_S"; export V_O2="$O2"
export V_RPASS="$R_PASS"; export V_RFAIL="$R_FAIL"; export V_RSKIP="$R_SKIP"; export V_O3="$O3"
export V_COLLOUT="$COLL_OUT"; export V_COLL="$COLL"
export V_NEWT="$((COLL - 255))"
export V_NREF="$NREF"; export V_NNEW="$NNEW"; export V_REDS="$REDS"; export V_PERT="$PERT"
export V_LSTART="$(grep PS_LSTART_pytest /tmp/am_t36_launch.txt | cut -d: -f2- | sed 's/^ //')"
export V_SKID="$SKID"
export V_CONSIST="$CONSIST"; export V_PERT2="$PERT2"; export V_G3="$G3"; export V_REFNAMES="$REF_NAMES"; export V_NEWNAMES="$NEW_NAMES"
python3 - <<'PYHDR' > /tmp/am_t36_hdr.txt
import os, re
tpl = open("/tmp/am_t36_hdr.tmpl", encoding="utf-8").read()
out = tpl
for k, v in os.environ.items():
    if k.startswith("V_"):
        out = out.replace("@" + k[2:] + "@", v)
left = sorted(set(re.findall(r"@[A-Z0-9]+@", out)))
assert not left, f"占位符未填: {left}"
print(out, end="")
PYHDR
echo "HDR_LINES=$(wc -l < /tmp/am_t36_hdr.txt)"
