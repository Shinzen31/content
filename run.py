#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel scheduler for single-env dFBA runs  (chunk-wise)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python run.py --start_env 0 --stop_env 99 --chunk 3 --timeout 600

修改记录
--------
2025-07-15
* merge_one()：写入前删除旧 env 行 → 覆盖而非追加
* 保证结果始终按 env_id、group_id、model_name 排序
"""

import argparse
import os
import subprocess
import sys
import gc
import traceback
import pandas as pd

RESULTS_DIR   = "results"
TARGET_MODEL  = "GCF_000010425.1_ASM1042v1_protein_gapfilled_noO2_appliedMedium.xml"

# ─────────── 文件合并 ───────────
def merge_one(env: int, bs: int):
    """
    把单 env-bs 文件合并到 results_bs{bs}/dfba_final_biomass.csv

    改进：
    - 若目标文件已存在，先删除同 env_id 的旧行，再追加新结果
    - 最终对整张表按 env_id → group_id → model_name 排序
    """
    src = os.path.join(RESULTS_DIR, f"dfba_final_biomass_env{env}_bs{bs}.csv")
    if not os.path.exists(src):
        print(f"⚠️  Missing {src}")
        return

    dst_dir = f"results_bs{bs}"
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "dfba_final_biomass.csv")

    df_new = pd.read_csv(src)

    if os.path.exists(dst):
        df = pd.read_csv(dst)
        # 删除旧 env 结果，实现“覆盖”
        df = df[df["env_id"] != env]
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    # 始终保持固定顺序
    df = df.sort_values(["env_id", "group_id", "model_name"])
    df.to_csv(dst, index=False)

# ─────────── 占位文件 ───────────
def create_placeholder(env: int, bs: int):
    """子进程超时 / 崩溃时写一个空结果（占位行可被后续 merge_one 覆盖）"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"dfba_final_biomass_env{env}_bs{bs}.csv")
    if os.path.exists(out):
        return
    header = ("env_id,group_id,model_name,final_biomass,factor,baseline_biomass,"
              "best_target_biomass,best_target_factor,best_partner_names\n")
    with open(out, "w") as f:
        f.write(header)
        f.write(f"{env},0,{TARGET_MODEL},0,0,0,0,0,\n")

# ─────────── 统计频率 ───────────
def collect_frequency():
    """读取各 bs 汇总 CSV，生成 partner_frequency.csv（按 factor 选最佳组）"""
    partner, all_models = {}, set()

    for bs in range(2, 6):
        fp = f"results_bs{bs}/dfba_final_biomass.csv"
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        all_models |= {m for m in df["model_name"].unique() if m != TARGET_MODEL}
        partner[bs] = {}
        for env, rows in df[df.model_name == TARGET_MODEL].groupby("env_id"):
            # 选 factor 最大的 group；若列缺失则退回 final_biomass
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
    freq /= len(partner)
    freq.to_csv("partner_frequency.csv")
    print("✅ partner_frequency.csv generated")

# ─────────── 主流程 ───────────
def main():
    ap = argparse.ArgumentParser("Chunk-wise parallel driver for dFBA")
    ap.add_argument("--start_env", type=int, default=0)
    ap.add_argument("--stop_env",  type=int, required=True)
    ap.add_argument("--chunk",     type=int, default=15)
    ap.add_argument("--timeout",   type=int, default=600)
    args = ap.parse_args()

    env_ids = list(range(args.start_env, args.stop_env + 1))

    # 清理旧汇总
    for bs in range(2, 6):
        path = f"results_bs{bs}/dfba_final_biomass.csv"
        if os.path.exists(path):
            os.remove(path)

    # 分段并行
    for i in range(0, len(env_ids), args.chunk):
        batch_envs = env_ids[i : i + args.chunk]
        print(f"\n>>> Processing environments {batch_envs[0]}–{batch_envs[-1]}")
        procs = {}
        for env in batch_envs:
            procs[env] = []
            for bs in range(2, 6):
                p = subprocess.Popen(
                    [sys.executable, "dFBA.py", "--env", str(env), "--bs", str(bs)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                procs[env].append((bs, p))

        # 等待并合并
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
                    print(f"‼️  Error: env {env}, bs {bs}\n{traceback.format_exc()}")
                    create_placeholder(env, bs)
                else:
                    if p.returncode != 0:
                        print(f"⚠️  Subprocess exited non-zero (env {env}, bs {bs})")
                        create_placeholder(env, bs)

                if out:
                    sys.stdout.write(out.decode(errors="ignore"))
                if err:
                    sys.stderr.write(err.decode(errors="ignore"))

                # ←← 覆盖式合并
                merge_one(env, bs)
            gc.collect()

    collect_frequency()

if __name__ == "__main__":
    main()
