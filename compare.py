#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare.py
~~~~~~~~~~
1. 选取 factor 前 100 的最佳组（跨 0–9999 个环境、bs=2-5）
2. 汇总其 Top-10 Δ 代谢物出现频率
3. 为每条再随机生成一个不同组、重新模拟，写入 random/
4. 汇总随机组 Top-10 Δ 代谢物频率
"""

from __future__ import annotations
import random, gc, json
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import cobra   # 依赖 dFBA.simulate

import dFBA as dfba   # 直接复用函数 & 全局常量

# ─────────── 配置 ───────────
BSS           = (2, 3, 4, 5)
RESULTS_BSDIR = "results_bs{}"
RANDOM_DIR    = Path("random")
RANDOM_DIR.mkdir(exist_ok=True)

random.seed(2025)

# ─────────── Step-0：加载所有结果表 ───────────
dfs_final, dfs_topd = [], []
for bs in BSS:
    dir_bs = Path(RESULTS_BSDIR.format(bs))
    fp_final   = dir_bs / "dfba_final_biomass.csv"
    fp_topd    = dir_bs / "topdelta.csv"
    if fp_final.exists():
        df = pd.read_csv(fp_final)
        df["bs"] = bs
        dfs_final.append(df)
    if fp_topd.exists():
        df = pd.read_csv(fp_topd)
        df["bs"] = bs
        dfs_topd.append(df)

df_final_all = pd.concat(dfs_final, ignore_index=True)
df_topd_all  = pd.concat(dfs_topd,  ignore_index=True)

# ─────────── Step-1：挑选 factor 前 100 ───────────
df_target = df_final_all[df_final_all.model_name == dfba.TARGET_FILE].copy()
df_top100 = df_target.sort_values("factor", ascending=False).head(100)

# 保存挑选列表（可选）
df_top100.to_csv("best100_factor.csv", index=False)

# ─────────── Step-2：汇总所需完整信息 & 统计频率 ───────────
selected_groups = []        # 记录全部信息
met_counter     = Counter() # 统计代谢物频率

for _, row in df_top100.iterrows():
    env_id  = int(row.env_id)
    bs      = int(row.bs)
    grp_id  = int(row.group_id)

    # 抓组内全部行
    df_bs   = df_final_all[(df_final_all.bs == bs)]
    grp_rows = df_bs[(df_bs.env_id == env_id) & (df_bs.group_id == grp_id)]

    # 伙伴列表
    partners = grp_rows[grp_rows.model_name != dfba.TARGET_FILE]["model_name"].tolist()

    # Top-10 Δ
    grp_tdelta = df_topd_all[(df_topd_all.bs == bs) &
                             (df_topd_all.env_id == env_id) &
                             (df_topd_all.group_id == grp_id)]

    # 更新计数器
    met_counter.update(grp_tdelta["metabolite_id"].tolist())

    selected_groups.append(dict(
        env_id   = env_id,
        bs       = bs,
        group_id = grp_id,
        factor   = row.factor,
        partners = partners,
        partner_rows = grp_rows.to_dict("records"),
        topdelta = grp_tdelta.to_dict("records"),
    ))

# 频率表
df_freq_best = pd.DataFrame.from_dict(
    dict(met_counter), orient="index", columns=["count"]
).sort_values("count", ascending=False)
df_freq_best.to_csv("best100_metabolite_frequency.csv")

print("✔️  完成最优 100 组频率统计")

# ─────────── Step-3：为每条随机生成一个不同组并重新模拟 ───────────
# 预备：环境载入、模型清单
envs_list = list(dfba.load_environment_ball(dfba.ENV_FILE))
all_models = [p.name for p in Path(dfba.MODEL_DIR).glob("*.xml")
              if p.name != dfba.TARGET_FILE]

def simulate_group(env_idx: int, partner_names: list[str]) -> list[dict[str, object]]:
    """给定 env_idx & partner_names，返回该组 Top-10 Δ 列表"""
    env_dict  = envs_list[env_idx]
    env_names = list(env_dict)
    models = [cobra.io.read_sbml_model(Path(dfba.MODEL_DIR) / dfba.TARGET_FILE)]
    for name in partner_names:
        models.append(cobra.io.read_sbml_model(Path(dfba.MODEL_DIR)/name))

    sol = dfba.simulate(models, env_names, env_dict)
    n_mod = len(models)
    mets_block = sol.y[n_mod:, :]
    net_delta  = mets_block[:, -1] - mets_block[:, 0]
    top_idx    = np.argsort(np.abs(net_delta))[::-1][:10]

    return [dict(env_id=env_idx,
                 metabolite_id=env_names[i],
                 delta=float(net_delta[i])) for i in top_idx]

random_records = []
rand_counter   = Counter()

for idx, info in enumerate(selected_groups, 1):
    env_id = info["env_id"]
    n_part = len(info["partners"])
    bs     = info["bs"]

    # 随机挑选不同伙伴
    original_set = set(info["partners"])
    candidate_pool = [m for m in all_models if m not in original_set]
    partner_rand = random.sample(candidate_pool, n_part)
    # 防止意外相同
    while set(partner_rand) == original_set:
        partner_rand = random.sample(candidate_pool, n_part)

    top10_rand = simulate_group(env_id, partner_rand)
    rand_counter.update([d["metabolite_id"] for d in top10_rand])

    random_records.append(dict(
        env_id       = env_id,
        bs           = bs,
        partners     = partner_rand,
        topdelta     = top10_rand
    ))

    # 写单文件
    df_rand_single = pd.DataFrame(top10_rand)
    df_rand_single.to_csv(
        RANDOM_DIR / f"topdelta_env{env_id}_rand.csv", index=False
    )
    print(f"  [{idx:3}/100] env {env_id} random group done")
    gc.collect()

# 汇总随机频率
df_freq_rand = pd.DataFrame.from_dict(
    dict(rand_counter), orient="index", columns=["count"]
).sort_values("count", ascending=False)
df_freq_rand.to_csv(RANDOM_DIR / "random100_metabolite_frequency.csv")

# 保存随机组概要
Path(RANDOM_DIR / "random_groups.json").write_text(
    json.dumps(random_records, indent=2)
)

print("🎉  随机 100 组模拟完成并统计频率")
