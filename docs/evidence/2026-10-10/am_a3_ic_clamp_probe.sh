#!/bin/bash
# #51 敏感度探针的**启动记录**：四条 pilot 腿的可执行副本（口径写在注释里，数字只从归档件现取）。
#
# 要问的问题：逆向映射把低于材料卡 T_ambient 的初温静默顶成 T_ambient（G8a 实测缺口 +6.85K，
#   占升到 T_sol 所需显焓的 0.5424%），**这到底改不改熔池形态**？决定 #51 是「修生产的二分下界」
#   还是「夹具侧 --amb-k 覆写就够了」。
# 关键机制（已实测，别当成「等于跑 300K 初温」）：``initial_enthalpy_field`` 存的是
#   H0=enthalpy_of_temperature(293.15)=−3.278057e7 J/m³（**负的**），而 T(H0) 读回 300.0000K
#   ⇒ 被顶替的那条腿能量比真 300K 态少 6.85K 的显焓，却报出 300K。所以 A 腿必须用 --ic-probe
#   让 G8 保留为 FAIL、只把读数跑完，不能用 --t0-c 造一个「representable 的 300K」来替代它。
#
# 四腿（全部 pilot 档＝dx=r_b/2＝21.250µm，tracks=1，臂=norm，CPU 钉住）：
#   A clamped   ：缺省材料卡（T_amb=300K）＋夹具 IC 293.15K ⇒ 今天真实跑的状态（G8a FAIL，读数留）
#   B faithful  ：--amb-k 293.15 ⇒ 夹具声明值得以表示（G8a/G8b 全绿）
#   C repeat-A  ：与 A 逐字相同的一条 ⇒ 确定性控件：C 与 A 的形态行必须**逐字节相同**，
#                否则 A−B 的差里混着运行间噪声，任何敏感度结论作废
#   D extreme   ：--ic-probe --amb-k 1500 ⇒ 可失败控件：IC 被顶到 1500K，读数**必须**大幅移动；
#                若 D≈A 则本探针是瞎的，「钳位不影响形态」这句话就不许说。
#                注意 D 同时动了环境温度（散热边也变了）⇒ 它是「探针不是瞎的」的证据，
#                **不能**拿来线性外推 6.85K 的影响（14:4x 实测：D 的 solidus 宽 +14.02µm／
#                A 与 B 却逐字节相同 ⇒ 二者比值远超 1206.85/6.85＝176 倍，非线性＋混杂）。
#   E slope-IC  ：--amb-k 293.15 ＋ --t0-c 26.85 ⇒ **只**把初温抬 +6.85K、环境温度钉在 293.15K，
#                两腿都可表示⇒G8a/G8b 全绿（不用 --ic-probe）。Δ(E−B) 就是纯 IC 斜率×6.85K。
#   F slope-IC2 ：同上但 --t0-c 106.85（+86.85K）⇒ 给 E 的「零」配一个**能红**的对照：
#                若 F 与 B 相同而 D 能移动，说明 293K 附近确实平；若 F 也纹丝不动而 E 不动，
#                则本探针在该尺度上无分辨力，任何「钳位无害」的结论作废。
# 红线：全程 CUDA_VISIBLE_DEVICES="" ＋ JAX_PLATFORMS=cpu ⇒ 不碰 GPU；墙钟只作可行性代价登记；
#   A/C/D 三腿的形态数字带 [PROBE] 标记且 all_pass=False ⇒ **不得**进 A3 的 18 道趋势评分。
# 每次尝试用**不同的档案名**：目标件已存在就直接拒绝（上一轮有过把红件覆写没了的事故）。
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
TIMEOUT_S=590

arm="${1:-}"
case "$arm" in
  A) ARGS=(--stage pilot --tracks 1 --arms norm --ic-probe) ;;
  B) ARGS=(--stage pilot --tracks 1 --arms norm --amb-k 293.15) ;;
  C) ARGS=(--stage pilot --tracks 1 --arms norm --ic-probe) ;;
  D) ARGS=(--stage pilot --tracks 1 --arms norm --ic-probe --amb-k 1500) ;;
  E) ARGS=(--stage pilot --tracks 1 --arms norm --amb-k 293.15 --t0-c 26.85) ;;
  F) ARGS=(--stage pilot --tracks 1 --arms norm --amb-k 293.15 --t0-c 106.85) ;;
  *) echo "用法：$0 {A|B|C|D|E|F}（A clamped／B faithful／C repeat-A 确定性控件／D extreme 可失败控件／E＋F 固定环境温度下的 d(形态)/d(IC) 斜率腿）"; exit 2 ;;
esac
TAG="_raw_a3ic_${arm}"
if [ -e "$E/$TAG.out" ] || [ -e "$E/$TAG.err" ]; then
  echo "REFUSE = $E/$TAG.{out,err} 已存在 ⇒ 换名（每次尝试留自己的档案，别覆写上一件）"; exit 3
fi
echo "CMD = CUDA_VISIBLE_DEVICES=\"\" JAX_PLATFORMS=cpu PYTHONPATH=src timeout $TIMEOUT_S $PY $E/am_a3_driver.py ${ARGS[*]}"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=src \
  timeout $TIMEOUT_S $PY $E/am_a3_driver.py "${ARGS[@]}" \
  > /tmp/a3ic_${arm}.out 2> /tmp/a3ic_${arm}.err
echo "rc=$? arm=$arm"
cp -p /tmp/a3ic_${arm}.out $E/$TAG.out
cp -p /tmp/a3ic_${arm}.err $E/$TAG.err
ls -l $E/$TAG.out $E/$TAG.err
