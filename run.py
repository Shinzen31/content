#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel scheduler for single-env dFBA runs (chunk-wise)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
示例：
    python run.py --start_env 0 --stop_env 3 --chunk 4 --timeout 1200 --plot

2025-07-31
-----------
* 回退到 2025-07-15 的调度 / 合并逻辑，解决子进程 stdout 堵塞
* trajectory 快照 + 2×2 子图绘制
* **新增 --plot 开关**；默认不绘图，可按需开启
* 绘图改进：
    - 过滤掉始终为 0 的“伪伙伴”曲线（解决 bs=2 却出现 4 条线的问题）
    - 图例放到图外顶部，动态计算 ncol，避免被截断
"""
import argparse
import gc
import itertools
import os
import shutil
import subprocess
import sys
import traceback
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ─────────── 常量 ───────────
RESULTS_DIR = "results"
TRAJ_DIR    = "trajectories"
PLOT_DIR    = "plots"
TARGET_MODEL = "GCF_000010425.1_ASM1042v1_protein_gapfilled_noO2_appliedMedium.xml"

# ─────────── 合并单 env-bs 结果 ───────────
def merge_one(env: int, bs: int):
    """
    把 dfba_final_biomass_env{env}_bs{bs}.csv 合并进
    results_bs{bs}/dfba_final_biomass.csv（覆盖同 env 行）
    """
    src = Path(RESULTS_DIR) / f"dfba_final_biomass_env{env}_bs{bs}.csv"
    if not src.exists():
        print(f"⚠️  Missing {src}")
        return

    dst_dir = Path(f"results_bs{bs}")
    dst_dir.mkdir(exist_ok=True)
    dst = dst_dir / "dfba_final_biomass.csv"

    df_new = pd.read_csv(src)
    if dst.exists():
        df = pd.read_csv(dst)
        df = df[df["env_id"] != env]            # 覆盖同 env 行
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df = df.sort_values(["env_id", "group_id", "model_name"])
    df.to_csv(dst, index=False)

# ─────────── 占位文件（超时 / 崩溃） ───────────
def create_placeholder(env: int, bs: int):
    out = Path(RESULTS_DIR) / f"dfba_final_biomass_env{env}_bs{bs}.csv"
    if out.exists():
        return
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    header = ("env_id,group_id,model_name,final_biomass,factor,baseline_biomass,"
              "best_target_biomass,best_target_factor,best_partner_names\n")
    out.write_text(header + f"{env},0,{TARGET_MODEL},0,0,0,0,0,\n")

# ─────────── trajectory 快照（用于绘图） ───────────
def snapshot_trajectories(env: int, bs: int):
    Path(TRAJ_DIR).mkdir(exist_ok=True)
    for prefix in ("biomass", "metabolites", "metabolites_delta"):
        src = Path(TRAJ_DIR) / f"{prefix}_env{env}.csv"
        dst = Path(TRAJ_DIR) / f"{prefix}_env{env}_bs{bs}.csv"
        if src.exists():
            shutil.copy(src, dst)

# ─────────── 汇总伙伴出现频率 ───────────
def collect_frequency():
    partner, all_models = {}, set()
    for bs in range(2, 6):
        fp = Path(f"results_bs{bs}") / "dfba_final_biomass.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        all_models |= {m for m in df["model_name"].unique() if m != TARGET_MODEL}
        partner[bs] = {}
        for env, rows in df[df.model_name == TARGET_MODEL].groupby("env_id"):
            # factor 优先，其次 final_biomass
            if "factor" in rows.columns and rows["factor"].notna().any():
                gid = int(rows.loc[rows["factor"].idxmax(), "group_id"])
            else:
                gid = int(rows.loc[rows["final_biomass"].idxmax(), "group_id"])
            plist = df[(df.env_id == env) &
                       (df.group_id == gid) &
                       (df.model_name != TARGET_MODEL)]["model_name"].tolist()
            partner[bs][env] = plist

    if not partner:
        print("❌ No data to summarize; skipped partner_frequency.csv")
        return

    env_ids = sorted({e for d in partner.values() for e in d})
    cols    = sorted(all_models)
    freq = pd.DataFrame(0.0, index=env_ids, columns=cols)
    for env_map in partner.values():
        for env, plist in env_map.items():
            for m in plist:
                freq.at[env, m] += 1
    freq /= len(partner)   # 归一化到 0-1
    freq.to_csv("partner_frequency.csv")
    print("✅ partner_frequency.csv generated")

# ─────────── 绘制 2×2 子图 ───────────
def plot_env_biomass_subplots(env_ids):
    Path(PLOT_DIR).mkdir(exist_ok=True)
    pos = {2: (0, 0), 3: (0, 1), 4: (1, 0), 5: (1, 1)}

    for eid in env_ids:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        drawn = False

        for bs in (2, 3, 4, 5):
            fn = Path(TRAJ_DIR) / f"biomass_env{eid}_bs{bs}.csv"
            if not fn.exists():
                continue

            df = pd.read_csv(fn)
            if "time" not in df.columns or TARGET_MODEL not in df.columns:
                continue      # 非法 / 空文件

            t  = df["time"]
            ax = axes[pos[bs]]

            base_col = f"{TARGET_MODEL}_baseline"
            if base_col in df.columns and df[base_col].max() > 1e-10:
                ax.plot(t, df[base_col], lw=2, ls="--", label="baseline")

            # target
            ax.plot(t, df[TARGET_MODEL], lw=1.8, label="target")

            # partners（过滤掉全零列）
            partners = [c for c in df.columns
                        if c not in ("time", base_col, TARGET_MODEL)
                        and df[c].max() > 1e-10]

            style_cycle = itertools.cycle(["-", "-.", ":", (0, (3, 1, 1, 1))])
            for p in partners:
                ax.plot(t, df[p], lw=1, ls=next(style_cycle), alpha=0.9, label=p)

            ax.set_title(f"bs={bs}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            drawn = True

        if not drawn:                 # 没有任何数据
            plt.close(fig)
            continue

        # —— 合并图例到顶部 ——
        uniq = {}
        for ax in axes.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                # 保留第一个出现的 label
                uniq.setdefault(l, h)

        handles, labels = list(uniq.values()), list(uniq.keys())
        ncol = ceil(len(labels)/3) if len(labels) > 6 else len(labels)
        fig.legend(handles, labels,
                   loc="upper center",
                   bbox_to_anchor=(0.5, 1.04),
                   ncol=ncol,
                   fontsize=6,
                   frameon=False)

        fig.suptitle(f"Environment {eid}", fontsize=12, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.94])   # 预留顶部给图例
        outp = Path(PLOT_DIR) / f"biomass_env{eid}.png"
        plt.savefig(outp, dpi=300)
        plt.close(fig)
        print(f"✅ {outp.name}")

# ─────────── 主流程 ───────────
def main():
    ap = argparse.ArgumentParser("Chunk-wise parallel driver for dFBA")
    ap.add_argument("--start_env", type=int, default=0)
    ap.add_argument("--stop_env",  type=int, required=True)
    ap.add_argument("--chunk",     type=int, default=15,
                    help="并行批大小（按 env 划分）")
    ap.add_argument("--timeout",   type=int, default=600,
                    help="子进程超时（秒）")
    ap.add_argument("--plot", action="store_true",
                    help="运行结束后绘制 2×2 子图")
    args = ap.parse_args()

    env_ids = list(range(args.start_env, args.stop_env + 1))

    # —— 清理旧快照 ——
    if Path(TRAJ_DIR).exists():
        for old in Path(TRAJ_DIR).glob("*_bs*.csv"):
            old.unlink()

    # —— 清理旧汇总 ——
    for bs in range(2, 6):
        fp = Path(f"results_bs{bs}") / "dfba_final_biomass.csv"
        if fp.exists():
            fp.unlink()

    # —— 分段并行调度 ——
    for i in range(0, len(env_ids), args.chunk):
        batch_envs = env_ids[i : i + args.chunk]
        print(f"\n>>> Processing environments {batch_envs[0]}–{batch_envs[-1]}")

        # 启动子进程
        procs = {env: [] for env in batch_envs}
        for env in batch_envs:
            for bs in range(2, 6):
                p = subprocess.Popen(
                    [sys.executable, "dFBA.py", "--env", str(env), "--bs", str(bs)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                procs[env].append((bs, p))

        # 收集结果
        for env in batch_envs:
            for bs, p in procs[env]:
                try:
                    out, err = p.communicate(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    print(f"⏰ Timeout: env {env}, bs {bs} — killed")
                    p.kill()
                    out, err = p.communicate()
                    create_placeholder(env, bs)
                except Exception:
                    print(f"‼️ Error: env {env}, bs {bs}\n{traceback.format_exc()}")
                    create_placeholder(env, bs)
                else:
                    if p.returncode != 0:
                        print(f"⚠️  Subprocess exited non-zero (env {env}, bs {bs})")
                        create_placeholder(env, bs)
                finally:
                    # 打印子进程输出
                    if out:
                        sys.stdout.write(out.decode(errors="ignore"))
                    if err:
                        sys.stderr.write(err.decode(errors="ignore"))

                    # 合并结果 & 快照
                    merge_one(env, bs)
                    snapshot_trajectories(env, bs)
                    gc.collect()

    # —— 汇总出现频率 ——
    collect_frequency()

    # —— 绘图（可选） ——
    if args.plot:
        plot_env_biomass_subplots(env_ids)

if __name__ == "__main__":
    main()
