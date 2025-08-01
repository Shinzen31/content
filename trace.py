#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_biomass.py – standalone utility to draw 2×2 biomass subplots for dFBA runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Usage examples
--------------
# 仅绘制指定环境
python plot_biomass.py --env_ids 0 1 2 3

# 覆盖默认常量
python plot_biomass.py --env_ids 5 6 --traj_dir other_traj --plot_dir figs

说明
----
• 依赖的数据文件：
  - 轨迹：   {TRAJ_DIR}/biomass_env<eid>_bs<bs>.csv
  - 结果汇总：results_bs<bs>/dfba_final_biomass.csv
• 仅包含绘图逻辑，不会触发任何计算或子进程。
"""
import argparse
import itertools
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ─────────── 默认常量（可通过 CLI 覆盖） ───────────
TARGET_MODEL = "GCF_000010425.1_ASM1042v1_protein_gapfilled_noO2_appliedMedium.xml"
TRAJ_DIR     = "trajectories"
PLOT_DIR     = "plots"

# ─────────── 主绘图函数 ───────────

def plot_env_biomass_subplots(env_ids, traj_dir=TRAJ_DIR, plot_dir=PLOT_DIR, target_model=TARGET_MODEL):
    """Draws a 2×2 grid (bs = 2–5) of biomass curves for each environment ID.

    Parameters
    ----------
    env_ids : list[int]
        Environment IDs to plot.
    traj_dir : str | Path, optional
        Directory containing biomass trajectory CSVs.
    plot_dir : str | Path, optional
        Output directory for PNG files.
    target_model : str, optional
        Name of the target model whose biomass will be highlighted.
    """
    traj_dir = Path(traj_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True)

    # 预加载最佳伙伴 → {(env_id, bs): {partner names}}
    best_map = {}
    for bs in range(2, 6):
        fp = Path(f"results_bs{bs}") / "dfba_final_biomass.csv"
        if not fp.exists():
            continue
        df_r = pd.read_csv(fp)
        for eid in env_ids:
            rows = df_r[(df_r.env_id == eid) & (df_r.model_name == target_model)]
            if rows.empty:
                continue
            row = rows.loc[
                rows["factor"].idxmax() if "factor" in rows.columns and rows["factor"].notna().any() else rows["final_biomass"].idxmax()
            ]
            best_map[(eid, bs)] = set(str(row.get("best_partner_names", "")).split(";"))

    # bs→子图位置
    pos = {2: (0, 0), 3: (0, 1), 4: (1, 0), 5: (1, 1)}

    for eid in env_ids:
        # 增大画布，预留左右空间给图例，上方空间给标题
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.05, right=0.90, top=0.85, hspace=0.7, wspace=0.6)
        drawn = False

        for bs in (2, 3, 4, 5):
            fn = traj_dir / f"biomass_env{eid}_bs{bs}.csv"
            if not fn.exists():
                continue
            df = pd.read_csv(fn)
            if "time" not in df.columns or target_model not in df.columns:
                continue

            allowed = best_map.get((eid, bs), set())
            ax = axes[pos[bs]]
            t = df["time"]

            base_col = f"{target_model}_baseline"
            if base_col in df.columns and df[base_col].max() > 1e-10:
                ax.plot(t, df[base_col], lw=2, ls="--", label="baseline")

            ax.plot(t, df[target_model], lw=1.8, label="target")

            partners = [c for c in df.columns
                        if c not in ("time", base_col, target_model) and c in allowed]

            style_cycle = itertools.cycle(["-", "-.", ":", (0, (3, 1, 1, 1))])
            for p in partners:
                ax.plot(t, df[p], lw=1, ls=next(style_cycle), alpha=0.9, label=p)

            ax.set_title(f"bs={bs}", fontsize=9)
            ax.tick_params(labelsize=7)

            # —— legend 垂直排列 ——
            hdl, lbl = ax.get_legend_handles_labels()
            if hdl:
                # 将图例放置于子图外侧，正上方，并与子图标题保持距离
                ax.legend(
                    hdl, lbl,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.10),
                    ncol=1,
                    fontsize=6,
                    frameon=False,
                )
            drawn = True

        if not drawn:
            plt.close(fig)
            continue

        # 标题放在外部，避免遮挡子图
        fig.suptitle(f"Environment {eid}", fontsize=14, y=0.98)
        outp = plot_dir / f"biomass_env{eid}.png"
        plt.savefig(outp, dpi=300)
        plt.close(fig)
        print(f"✅ {outp}")

# ─────────── CLI ───────────

def _parse_cli():
    p = argparse.ArgumentParser("Standalone biomass plotting tool")
    p.add_argument("--env_ids", type=int, nargs='+', required=True,
                   help="Environment ID list, e.g. --env_ids 0 1 2")
    p.add_argument("--traj_dir", default=TRAJ_DIR, help="Trajectory directory")
    p.add_argument("--plot_dir", default=PLOT_DIR, help="Output directory for figures")
    p.add_argument("--target_model", default=TARGET_MODEL, help="Target model name")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_cli()
    plot_env_biomass_subplots(
        args.env_ids,
        traj_dir=args.traj_dir,
        plot_dir=args.plot_dir,
        target_model=args.target_model,
    )
