#!/usr/bin/env python3
"""面表头的**数值溯源审计**（`am_s26_hdr.py` 生成器保护的补齐版）。

一层面（表头＋原始 stdout＋落盘回执）里，表头中的每一个数字都必须落到一个可复查的出处；出处只有五类：

  C 命令读数   ＝ md5（整串或前 12 位）、字节数、行数、日期／mtime／时区、git 计数、nvidia-smi 的 PID 与显卡名
  D 器件常量   ＝ 被测 `.py` 源码里出现的数字（判据阈值、扰动参数、预算秒数都属于这一类）
  B 本面正文   ＝ 该数字串出现在**本面**原始 stdout 的某条十进制数里（含 0.000596 这种整串被小数点切开）
  X 跨面正文   ＝ 数字在**另一面**正文里 ⇒ 只有引用句**自带出处词**（控件面／主面／K1…C4…G4a…X1…）才允许，
                 否则判红。这条是 §26.29 (k)② 那处误归因（把控件面 K2b 的数挂在主面 G4a 名下）的仪器化
  S 结构式     ＝ `cmp -i A:B` 的偏移、表头＋正文＝回执的字节配对；配对里的**加法本身要重算**

为什么要本件：§26.29 (i) 如实登记本轮两张 A3 面表头没走生成器（生成器的片号闸门绑定 HEAD，而本片当时尚未
提交），于是"字段只从命令现取"这层保护对它们缺席。本件不重写已入库的字节，而是把同一层保护改成**审计**：
逐个数现查，查不到出处就拒绝（rc=3）。

自证三条（缺一把尺子就是半个尺子，正对照面＝`am_a3_hdr_audit_controls.py`）：
 ① 未扰动基线必须 rc=0 且印出全过的判读行；
 ② 每条扰动必须让本件**变红**，且输出里出现**指定的闸门句**（只看 rc 会把环境错当成闸门红＝§26.28 实测）；
 ③ 扰动改的是**输入面**（表头／正文的 zctl_ 副本，回执按同一路子重建），副本与运行都在**本件自己的目录**
    ⇒ 被验的是这一件真跑的判据，不是另写的平行实现；副本名一律 `zctl_` 前缀，并被下面的池口径**排除**
    （否则副本自己的 md5 会进池，把表头里真正无出处的读数洗白＝仪器给自己开后门）。
    本件原先有个 `HDR_INJECT` 环境变量：它只把判读极性从"须 0 红"翻成"须 >0 红"，**不注入任何东西**
    ⇒ 一条"能把自己说成绿"的对照不是对照，已删；红路由 ② 的真扰动证。
"""
import datetime
import glob
import os
import re
import subprocess
import sys

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(D)))
DAY = os.path.basename(D)
def not_control(path):
    """正对照面的临时副本（`zctl_*`）不得进入任何出处池：它们的 md5／字节数会把表头里
    真正无出处的读数洗白——池里出现被测对象自己的脚手架＝仪器给自己开后门。"""
    return not os.path.basename(path).startswith("zctl_")


def git(*a):
    return subprocess.run(["git", "-c", "core.quotePath=false", *a], cwd=REPO,
                          capture_output=True, text=True).stdout


def md5(path):
    return subprocess.run(["md5sum", path], capture_output=True,
                          text=True).stdout.split()[0]


def hm(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%H:%M:%S")


def grep_lines(text, pat):
    """`grep -c` 的语义＝**命中行数**，不是出现次数（同一行两处只数 1）。"""
    rx = re.compile(pat)
    return sum(1 for l in text.splitlines() if rx.search(l))


ATTR = re.compile(r"控件面|主面|正文|K\d|C\d|G\d|J\d|X\d|N\d|§|r_b|dx|BUDGET")


class Corpus:
    def __init__(self, label, hdr, raw, log, other):
        self.label, self.hdr_p, self.raw_p, self.log_p, self.other = label, hdr, raw, log, other
        self.hdr = open(hdr, encoding="utf-8").read()
        self.body = open(raw, encoding="utf-8").read()
        self.log = open(log, encoding="utf-8").read()

    # ---------- 出处池 ----------
    def cmd_pool(self):
        p, m = set(), set()

        def num(x):
            p.add(str(x))

        sizes = {self.hdr_p: os.path.getsize(self.hdr_p), self.raw_p: os.path.getsize(self.raw_p),
                 self.log_p: os.path.getsize(self.log_p)}
        for f, s in sizes.items():
            num(s)
        arc = sorted(glob.glob(os.path.join(REPO, "docs/evidence/*/am_a3*"))
                     + [f for f in glob.glob(os.path.join(D, "*")) if not_control(f)])
        for f in arc:
            if os.path.isfile(f):
                m.add(md5(f)); m.add(md5(f)[:12])
        for t in (git("log", "-1", "--format=%cd", "--date=iso-strict"),
                  git("log", "-1", "--format=%cd", "--date=short"),
                  git("rev-parse", "--short", "HEAD"), git("rev-parse", "HEAD"),
                  hm(self.raw_p), hm(self.hdr_p), hm(self.log_p),
                  subprocess.run(["date", "+%F %T %z"], capture_output=True,
                                 text=True).stdout,
                  subprocess.run(["date", "+%z"], capture_output=True, text=True).stdout,
                  "2026-10-10"):
            for tok in re.findall(r"\d+", t or ""):
                p.add(tok)
                p.add(tok.lstrip("0") or "0")
        nv = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.split()
        for x in nv:
            for tok in re.findall(r"\d+", x):
                p.add(tok)
        gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout
        for tok in re.findall(r"\d+", gpu_name):
            p.add(tok); p.add(tok.lstrip("0") or "0")
        self.gpu_name = gpu_name.strip()
        self.nv_pids = [x for x in nv if x.strip()]
        return p, m

    def dev_pool(self):
        # D 池只收**被测器件**（驱动器＋控件面）与运行脚本；本审计件自己必须排除，
        # 否则它源码里的阈值/正则数字会把表头里真正无出处的数洗白（＝仪器给自己开后门）。
        p = set()
        cands = sorted(glob.glob(os.path.join(D, "am_a3_*.py")) + [os.path.join(D, "am_a3_run.sh")])

        def keep(c, py_only=False):
            b = os.path.basename(c)
            return (os.path.isfile(c) and "hdr_audit" not in b and not b.startswith("zctl_")
                    and not (py_only and not b.endswith(".py")))

        self.dev_files = [os.path.basename(c) for c in cands if keep(c)]
        for f in [c for c in cands if keep(c)]:
            for tok in re.findall(r"\d+", open(f, encoding="utf-8").read()):
                p.add(tok)
                v = tok.lstrip("0")
                if v:
                    p.add(v)
        # 器件里的**模块级数值常量**要按定义式求值（`BUDGET_S = 4 * 3600.0` 的读数是 14400，
        # 而源码里根本没有 "14400" 这串字符）：只允许纯数字与 + - * / ( ) . 的表达式，别的一律不 eval。
        for f in [c for c in cands if keep(c, py_only=True)]:
            for name, expr in re.findall(r"^([A-Z][A-Z0-9_]*) = ([0-9.+\-*/() ]+)$",
                                         open(f, encoding="utf-8").read(), re.M):
                if not re.fullmatch(r"[0-9.+\-*/() ]+", expr):
                    continue
                try:
                    v = eval(expr, {"__builtins__": {}}, {})
                except Exception:
                    continue
                for tok in re.findall(r"\d+", repr(v)):
                    p.add(tok.lstrip("0") or "0"); p.add(tok)
                if isinstance(v, float) and v == int(v):
                    p.add(str(int(v)))
                p.add(str(v))
        return p

    @staticmethod
    def digit_runs(text):
        """正文里每条十进制数拆成的数字串：既收整段（17017），也收被小数点/科学记数切开后的片段。"""
        runs = set()
        for num in re.findall(r"\d+(?:\.\d+)?(?:e-?\d+)?", text):
            for tok in re.findall(r"\d+", num):
                runs.add(tok)
                v = tok.lstrip("0")
                if v:
                    runs.add(v)
        return runs

    def classify(self, tok, C, Dv, B, X, M):
        # 一律**精确**命中：md5 的十六进制串里可能碰巧含某段数字，允许子串命中就等于给仪器开后门
        if tok in B:
            return "B"
        if tok in C or tok in M:
            return "C"
        if tok in Dv:
            return "D"
        if tok in X:
            return "X"
        return None

    def hex_pool(self):
        """表头里的十六进制读数（md5 前 12／整串、git 短哈希）单独成池：
        它们的数字段**不是**测量值，混进十进制普查会造出假"无出处"。"""
        h = set()
        for f in sorted(glob.glob(os.path.join(REPO, "docs/evidence/*/am_a3*"))
                        + [f for f in glob.glob(os.path.join(D, "*")) if not_control(f)]):
            if os.path.isfile(f):
                h.add(md5(f)); h.add(md5(f)[:12])
        for sha in git("log", "--format=%H %h", "-30").split():
            h.add(sha)
        return h

    def logical_lines(self):
        """表头的**续行是缩进的**：把缩进行并回上一条逻辑行，再按句读切。
        若直接按 `\\n` 切，"控件面 K3：…（换行）…闭合残差 1.11e-16" 会被拆成两句 ⇒ 出处词丢了，
        跨面引用被判无归属＝假红（本轮实测到，故把口径写进仪器）。"""
        out = []
        for raw in self.hdr.split("\n"):
            if raw.startswith((" ", "\t")) and out:
                out[-1] += " " + raw.strip()
            else:
                out.append(raw)
        return out

    def sentences(self):
        sents = []
        for ll in self.logical_lines():
            sents += [s for s in re.split(r"[。；;]", ll) if s.strip()]
        return sents

    def score_arith(self, sent):
        """记分分解式 `37 ＝ 4（控件 C1–C4）＋ 3 个道次 × 11 条闸门（G1…J11）＝ 4 + 33` 的**重算**：
        道次数从正文 `tracks=[1, 2, 3]` 现取，闸门数从同一句的括号枚举**现数**，乘积与和重算。
        返回 (可判为 S 的数字串集合, 红项列表)。"""
        reds, toks = [], set()
        m = re.search(r"(\d+) 个道次 × (\d+) 条闸门", sent)
        if not m:
            return toks, reds
        tracks_hdr, gates_hdr = int(m.group(1)), int(m.group(2))
        bt = re.search(r"tracks=\[([^\]]*)\]", self.body)
        tracks = len([x for x in bt.group(1).split(",") if x.strip()]) if bt else None
        enum = re.search(r"[（(]((?:[^（）]*?(?:G|J)\w+){2,}[^（）]*?)[）)]", sent)
        labels = sorted(set(re.findall(r"\b(?:G|J)\d+[a-c]?\b", enum.group(1)))) if enum else []
        total = re.search(r"(\d+)\s*＝\s*(\d+)[^0-9]*＋", sent)
        toks.add(str(tracks_hdr)); toks.add(str(gates_hdr))
        for l in labels:
            toks.update(re.findall(r"\d+", l))
        if tracks is None:
            reds.append("记分分解式说 %d 个道次，但正文里取不到 tracks=[…]" % tracks_hdr)
        elif tracks != tracks_hdr:
            reds.append("道次数不符：表头 %d／正文 tracks=[…] 现取 %d" % (tracks_hdr, tracks))
        if len(labels) != gates_hdr:
            reds.append("闸门枚举现数 %d 条（%s）≠ 表头声称 %d 条" % (
                len(labels), ",".join(labels), gates_hdr))
        prod = tracks_hdr * gates_hdr
        toks.add(str(prod))
        if str(prod) not in sent:
            reds.append("表头那句没有 %d×%d＝%s 的乘积读数，分解式无法闭合" % (
                tracks_hdr, gates_hdr, prod))
        if total:
            head, part = int(total.group(1)), int(total.group(2))
            toks.add(str(part))
            if part + prod != head:
                reds.append("记分总分 %d ≠ %d＋%d" % (head, part, prod))
        return toks, reds

    # ---------- 逐句溯源 ----------
    def audit(self):
        C, M = self.cmd_pool()
        Dv, H = self.dev_pool(), self.hex_pool()
        B = self.digit_runs(self.body)
        X = self.digit_runs(open(self.other, encoding="utf-8").read())
        hits, unproven, unattributed, badhex = {}, [], [], []
        arith_reds = []
        for si, sent in enumerate(self.sentences()):
            stoks, sreds = self.score_arith(sent)
            arith_reds += sreds
            for hx in re.findall(r"\b[0-9a-f]{7,32}\b", sent):
                if hx not in H:
                    badhex.append((si, hx, sent.strip()[:60]))
            clean = re.sub(r"\b[0-9a-f]{7,32}\b", " ", sent)
            toks = re.findall(r"\d+", clean)
            if not toks:
                continue
            attr = bool(ATTR.search(sent))
            for tok in toks:
                k = "S" if tok in stoks else self.classify(tok, C, Dv, B, X, M)
                if k is None:
                    unproven.append((si, tok, sent.strip()[:60]))
                elif k == "X" and not attr:
                    unattributed.append((si, tok, sent.strip()[:60]))
                else:
                    hits[k] = hits.get(k, 0) + 1
        return (hits, unproven, unattributed, badhex, arith_reds,
                len(C), len(Dv), len(B), len(X))

    # ---------- 表头声称的读数 vs 实测 ----------
    def claims(self):
        reds, rows = [], []
        pair = [
            ("行首 [PASS] 行数", r"^  \[PASS\] ", r"PASS\s*=\s*\*{0,2}(\d+)"),
            ("行首 [FAIL] 行数", r"^  \[FAIL\] ", r"FAIL\s*=\s*\*{0,2}(\d+)"),
            ("不锚标签位的 [FAIL] 行数", r"\[FAIL\]", r"数出 \*\*(\d+)\*\*"),
        ]
        for name, gpat, cpat in pair:
            got = grep_lines(self.body, gpat)
            m = re.search(cpat, self.hdr)
            if m:
                rows.append("  %-26s 实测=%-3d 表头=%-3s %s" % (
                    name, got, m.group(1), "OK" if int(m.group(1)) == got else "⇒ 红"))
                if int(m.group(1)) != got:
                    reds.append("%s 表头声称 %s 而实测 %s" % (name, m.group(1), got))
            else:
                rows.append("  %-26s 实测=%-3d 表头未声称" % (name, got))
        for key in ("SOLVE_VERDICT", "READOUT", "width_um", "靶"):
            got = grep_lines(self.body, re.escape(key))
            m = re.search(re.escape(key) + r"[^0-9]{0,10}?(\d+) ?行", self.hdr)
            if m:
                rows.append("  零声称 %-18s 实测=%d 表头=%s %s" % (
                    key, got, m.group(1), "OK" if int(m.group(1)) == got == 0 else "⇒ 红"))
                if int(m.group(1)) != got or got != 0:
                    reds.append("%s 零声称破了（表头 %s／实测 %s）" % (key, m.group(1), got))
        # md5 前 12 与字节数：正文原始件
        m = re.search(r"md5 前 12＝([0-9a-f]{12})／(\d+) 字节", self.hdr)
        if m:
            ok = m.group(1) == md5(self.raw_p)[:12] and int(m.group(2)) == os.path.getsize(self.raw_p)
            rows.append("  原始件 md5/字节 表头=%s/%s 实测=%s/%s %s" % (
                m.group(1), m.group(2), md5(self.raw_p)[:12], os.path.getsize(self.raw_p),
                "OK" if ok else "⇒ 红"))
            if not ok:
                reds.append("原始件 md5 或字节数与实测不符")
        # 结构：偏移＋正文＝回执（并把加法重算）
        off = re.search(r"表头偏移 (\d+)", self.hdr)
        hsz, bsz, lsz = (os.path.getsize(self.hdr_p), os.path.getsize(self.raw_p),
                         os.path.getsize(self.log_p))
        if off:
            ok = int(off.group(1)) == hsz
            rows.append("  表头偏移 表头=%s 实测表头字节=%d %s" % (off.group(1), hsz, "OK" if ok else "⇒ 红"))
            if not ok:
                reds.append("cmp 偏移 %s ≠ 表头实际字节 %d" % (off.group(1), hsz))
        for a, b, c in re.findall(r"(\d+)[＋+](\d+)=(\d+)", self.hdr):
            good = int(a) + int(b) == int(c)
            rows.append("  加法 %s＋%s=%s 重算=%d %s" % (a, b, c, int(a) + int(b), "OK" if good else "⇒ 红"))
            if not good:
                reds.append("表头那句 %s＋%s=%s 算不出来" % (a, b, c))
        # 记分分解式（37＝4＋3×11）的重算在 audit() 的 score_arith() 里做，这里不重复算
        # cat 复现式
        if (self.hdr + self.body).encode() != open(self.log_p, "rb").read():
            reds.append("cat 表头 原始件 ≠ 落盘正文 ⇒ 复现式破了")
        else:
            rows.append("  cat 表头＋原始件 == 落盘正文（逐字节）OK")
        # mtime 顺序声称
        m = re.search(r"mtime (\d\d:\d\d:\d\d)", self.hdr)
        if m:
            ok = m.group(1) == hm(self.raw_p)
            rows.append("  正文 mtime 表头=%s 实测=%s %s" % (m.group(1), hm(self.raw_p), "OK" if ok else "⇒ 红"))
            if not ok:
                reds.append("表头声称 mtime %s 实测 %s" % (m.group(1), hm(self.raw_p)))
        # head= 声称（状态相关：现在仍等于 HEAD 才 OK，否则只印两面）
        m = re.search(r"head=([0-9a-f]{7,40})", self.hdr)
        if m:
            cur = git("rev-parse", "HEAD").strip()
            same = m.group(1) == cur or cur.startswith(m.group(1))
            rows.append("  HEAD 表头=%s 审计时=%s %s" % (
                m.group(1), cur[:7], "OK" if same else "⇒ 已推进（表头是生成时刻的状态）"))
        return rows, reds

    def gitface(self):
        claimed = sorted({int(m.group(1)) for m in re.finditer(r"dirty[^\d]{0,4}(\d+)", self.hdr)})
        por = [l for l in git("status", "--porcelain").splitlines() if l.strip()]
        return ("dirty 类：表头声称 %s｜审计时现算 total=%d evidence=%d src+tests=%d｜我的 amvenv 进程（按 comm）=%d"
                % (claimed, len(por),
                   len([l for l in por if "docs/evidence/" in l]),
                   len([l for l in por if re.search(r"(^| )(src|tests)/", l)]),
                   self.mine()))

    @staticmethod
    def mine():
        n = 0
        for pid in [x for x in os.listdir("/proc") if x.isdigit()]:
            if int(pid) == os.getpid():
                continue
            try:
                comm = open("/proc/%s/comm" % pid, "rb").read().decode(errors="replace").strip()
                cl = open("/proc/%s/cmdline" % pid, "rb").read().decode(errors="replace")
            except OSError:
                continue
            if comm.startswith("python") and "/tmp/amvenv" in cl:
                n += 1
        return n


def faces():
    return [
        Corpus("主面", os.path.join(D, "am_a3_preflight_hdr.txt"),
               os.path.join(D, "_raw_a3_preflight.out"), os.path.join(D, "am_a3_preflight.log"),
               os.path.join(D, "_raw_a3_preflight_controls.out")),
        Corpus("控件面", os.path.join(D, "am_a3_preflight_controls_hdr.txt"),
               os.path.join(D, "_raw_a3_preflight_controls.out"),
               os.path.join(D, "am_a3_preflight_controls.log"),
               os.path.join(D, "_raw_a3_preflight.out")),
    ]


def gpu_name_line():
    return subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def main():
    if "HDR_INJECT" in os.environ:
        # 旧命令不能静默通过：极性翻转 env 已删，红路只能靠真扰动（正对照面）造
        raise SystemExit("HDR_INJECT 已删除（翻转判读极性不是扰动）⇒ 拒绝运行；"
                         "要证红请跑 am_a3_hdr_audit_controls.py")
    print("面表头数值溯源审计｜仓库＝%s｜HEAD＝%s｜审计时刻＝%s" % (
        REPO, git("rev-parse", "--short", "HEAD").strip(),
        datetime.datetime.now().strftime("%F %T %z")))
    print("钉住：CUDA_VISIBLE_DEVICES=%r（空串＝CPU）｜本件不 import jax｜显卡（C 池里的 6000 一类数的出处）＝%r"
          % (os.environ.get("CUDA_VISIBLE_DEVICES"), gpu_name_line()))
    print("出处口径：C 命令读数／D 器件常量／B 本面正文／X 跨面正文（引用句须自带出处词）／S 结构式；"
          "grep 计数一律按 **命中行数**（＝`grep -c` 语义），不按出现次数")
    print("")
    all_ok = True
    for c in faces():
        hits, unproven, unattr, badhex, arith, nc, nd, nb, nx = c.audit()
        tot = sum(hits.values()) + len(unproven) + len(unattr)
        print("── %s（%s，回执 %d 字节；池 C=%d D=%d B=%d X=%d HEX=%d｜D 池来源=%s）──" % (
            c.label, os.path.basename(c.hdr_p), os.path.getsize(c.log_p), nc, nd, nb, nx,
            len(c.hex_pool()), "、".join(c.dev_files)))
        print("  溯源普查：表头数字 %d 个＝%s／无出处 %d／跨面无归属 %d／无出处十六进制读数 %d" % (
            tot, "／".join("%s=%d" % (k, v) for k, v in sorted(hits.items())),
            len(unproven), len(unattr), len(badhex)))
        for si, tok, frag in unproven[:14]:
            print("     无出处: 句#%s 数字 %s ｜ %s" % (si, tok, frag))
        for si, tok, frag in unattr[:14]:
            print("     跨面无归属: 句#%s 数字 %s（该数只出现在另一面正文，引用句却没有出处词）｜ %s"
                  % (si, tok, frag))
        for si, hx, frag in badhex[:8]:
            print("     无出处 hex: 句#%s %s ｜ %s" % (si, hx, frag))
        rows, reds = c.claims()
        for r in rows:
            print(r)
        for r in reds + arith:
            print("     红: [%s] %s" % (c.label, r))
        print("  %s" % c.gitface())
        print("  正文运行 mtime=%s｜表头 mtime=%s｜回执 mtime=%s" % (
            hm(c.raw_p), hm(c.hdr_p), hm(c.log_p)))
        print("")
        bad = len(unproven) + len(unattr) + len(reds) + len(badhex) + len(arith)
        ok = bad == 0
        all_ok = all_ok and ok
        print("  本面判读＝%s（红项合计 %d）" % ("PASS" if ok else "PROBLEM", bad))
        print("")
    print("AUDIT_VERDICT =", "ALL HEADER READINGS PROVENANCE-PASS" if all_ok else
          "PROBLEM（见上面「无出处」/「跨面无归属」/「红:」行）")
    sys.exit(0 if all_ok else 3)


if __name__ == "__main__":
    main()
