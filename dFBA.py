#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group-dynamic FBA runner (target-in-every-group) — single-environment mode
=========================================================================
• 与旧版等价的计算 & 输出格式  
• Beam Search 组队 + 轨迹落盘  
• GLPK 求解器，任何 infeasible → μ = 0（避免子进程崩溃）
"""

from __future__ import annotations
import argparse, gc, sys, traceback
from pathlib import Path
from math import ceil        # ← 原版保留
import cobra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ───────────────── Parameters ──────────────────
MODEL_DIR      = "models_gapfilled"
TARGET_FILE    = "GCF_000010425.1_ASM1042v1_protein_gapfilled_noO2_appliedMedium.xml"
ENV_FILE       = "environment_ball.tsv"
TRAJ_DIR       = "trajectories"
TIME_SPAN      = (0, 48)          # h
N_POINTS       = 40
INIT_BIOMASS   = 0.1
MAX_UPTAKE     = 10.0
RESULT_DIR     = "results"
EPS_BASE       = 1e-9
EXTINCT_EPS    = 1e-8             # ←★ 新增：灭绝阈值，低于该值不再复活
MONOD_CONSTANT = 10
YIELD_FACTOR   = 0.1              # 仅对分泌 (v>0) 缩放
WATER_CONSTANT = 10               # EX_cpd00001_e0 下限
BEAM_WIDTH     = 5

cobra.Configuration().solver = "glpk"

# ─────────── 基础工具函数 ───────────
def apply_env(model: cobra.Model, env: dict[str, float],
              *, upper_bound: float = MAX_UPTAKE):
    """给模型设置摄取上限并优化"""
    for rid, conc in env.items():
        if model.reactions.has_id(rid):
            rxn = model.reactions.get_by_id(rid)
            rxn.lower_bound = -abs(conc)
            rxn.upper_bound = upper_bound
    model.reactions.EX_cpd00001_e0.lower_bound = WATER_CONSTANT
    return model.optimize()

def avoid_zero_met(states: np.ndarray, changes: np.ndarray) -> np.ndarray:
    """阻止环境代谢物变为负值"""
    return np.where(states + changes < 0, -states, changes)

# ─────────── 老师原版逻辑：get_growth ───────────
def get_growth(model: cobra.Model,
               env_names: list[str],
               env_state: dict[str, float]) -> tuple[float, dict[str, float]]:
    """
    返回 (μ, flux_dict)
    • μ = 最优生长率；求解 infeasible → μ=0, flux=0
    • 对分泌通量 (v>0) 乘 YIELD_FACTOR，摄取或 0 通量保持原值
    """
    uptake_bounds = {rid: env_state.get(rid, 0.0) /
                     (env_state.get(rid, 0.0) + MONOD_CONSTANT)
                     for rid in env_names}
    try:
        solution = apply_env(model, uptake_bounds)
    except Exception:
        return 0.0, {rid: 0.0 for rid in env_names}

    mu = solution.objective_value or 0.0
    fluxes: dict[str, float] = {}
    for rid in env_names:
        if model.reactions.has_id(rid):
            v = solution.fluxes.get(rid, 0.0)
            fluxes[rid] = v * YIELD_FACTOR if v > 0 else v
        else:
            fluxes[rid] = 0.0
    return mu, fluxes

# ─────────── 老师原版逻辑：simulate + 灭绝保护 ───────────
def simulate(models_orig: list[cobra.Model],
             env_names: list[str],
             env_initial: dict[str, float],
             *,
             t_span=TIME_SPAN, n_points=N_POINTS, show_plots=False):
    """积分求解 dFBA 动力学，返回 OdeResult"""
    n_models = len(models_orig)

    def ode(t, y):
        biomasses = np.maximum(y[:n_models], 0.0)
        mets      = np.maximum(y[n_models:], 0.0)
        env_state = dict(zip(env_names, mets))

        d_bio  = np.zeros(n_models)
        d_mets = np.zeros_like(mets)

        for i in range(n_models):
            mu, fluxes = get_growth(models_orig[i], env_names, env_state)
            d_bio[i]  = mu * biomasses[i]
            d_mets   += biomasses[i] * np.array([fluxes[r] for r in env_names])

        # ←★ 新增 3 行：灭绝后不再复活 ★→
        extinct = biomasses < EXTINCT_EPS
        d_bio[extinct] = 0.0
        d_bio = np.maximum(d_bio, -biomasses)   # 数值安全防负

        d_mets = avoid_zero_met(mets, d_mets)
        return np.concatenate((d_bio, d_mets))

    y0 = np.concatenate(([INIT_BIOMASS]*n_models,
                         [env_initial.get(r, 0.0) for r in env_names]))
    sol = solve_ivp(ode, t_span, y0, t_eval=np.linspace(*t_span, n_points))

    if show_plots:
        plt.figure(figsize=(6,4))
        for i in range(n_models):
            plt.plot(sol.t, sol.y[i], label=f"Pop {i+1}")
        plt.xlabel("Time (h)"); plt.ylabel("Biomass")
        plt.legend(); plt.tight_layout(); plt.show()

        plt.figure(figsize=(6,4))
        for j,rid in enumerate(env_names):
            plt.plot(sol.t, sol.y[n_models+j], label=rid)
        plt.xlabel("Time (h)"); plt.ylabel("Concentration")
        plt.legend(ncol=2, fontsize=7); plt.tight_layout(); plt.show()

    return sol

# ─────────── 其它 I/O 工具 ───────────
def load_environment_ball(path: str | Path):
    tbl = pd.read_csv(path, sep="\t")
    tbl["concentration"] = pd.to_numeric(tbl["concentration"], errors="coerce").fillna(0.0)
    if "env_id" in tbl.columns:
        for _, g in tbl.groupby("env_id"):
            yield dict(zip(g["metabolite_id"], g["concentration"]))
    else:
        n_met = tbl["metabolite_id"].nunique()
        for i in range(len(tbl)//n_met):
            chunk = tbl.iloc[i*n_met:(i+1)*n_met]
            yield dict(zip(chunk["metabolite_id"], chunk["concentration"]))

def iter_model_paths(model_dir: str | Path):
    return sorted(Path(model_dir).glob("*.xml"))

# ─────────── Beam Search ───────────
def beam_search_groups(partner_paths: list[Path],
                       target_model: cobra.Model,
                       env_names: list[str],
                       env_state: dict[str, float],
                       *, chunk: int, beam_width: int) -> list[list[Path]]:
    mu_base, _ = get_growth(target_model, env_names, env_state)
    mu_base = mu_base if mu_base > 0 else EPS_BASE

    def score(_: list[Path]) -> float:
        mu_now, _ = get_growth(target_model, env_names, env_state)
        return (mu_now or 0.0) / mu_base

    beam: list[tuple[list[Path], set[Path]]] = [([], set())]
    for _ in range(chunk):
        pool: list[tuple[list[Path], set[Path], float]] = []
        for sel, used in beam:
            for p in partner_paths:
                if p in used: continue
                pool.append((sel+[p], used|{p}, score(sel+[p])))
        pool.sort(key=lambda x: x[2], reverse=True)
        beam = [(l, u) for l, u, _ in pool[:beam_width]]
    return [l for l, _ in beam]

# ─────────── 主运行函数 ───────────
def run_single_env(env_id: int, batch_size: int, *, show_plots=False):
    envs = list(load_environment_ball(ENV_FILE))
    if not (0 <= env_id < len(envs)):
        raise ValueError(f"env_id {env_id} out of range (0–{len(envs)-1})")

    target_path, partner_paths = None, []
    for p in iter_model_paths(MODEL_DIR):
        (target_path := p) if p.name == TARGET_FILE else partner_paths.append(p)
    if target_path is None:
        raise FileNotFoundError(f"Target model {TARGET_FILE} not found")
    if batch_size < 2:
        raise ValueError("batch_size must be ≥2")

    env        = envs[env_id]
    env_names  = list(env)
    chunk_size = batch_size - 1
    if len(partner_paths) < chunk_size:
        raise RuntimeError("Not enough partner models to form a full group.")

    Path(TRAJ_DIR).mkdir(exist_ok=True)
    traj = {}

    # baseline
    solo_model = cobra.io.read_sbml_model(target_path)
    base_sol   = simulate([solo_model], env_names, env)
    base_bio   = base_sol.y[0, -1] or EPS_BASE
    print(f"[Baseline] target alone biomass = {base_bio:.6f}")

    traj["time"] = base_sol.t
    traj[f"{TARGET_FILE}_baseline"] = base_sol.y[0]

    # Beam Search
    groups = beam_search_groups(partner_paths, solo_model,
                                env_names, env,
                                chunk=chunk_size, beam_width=BEAM_WIDTH)
    print(f"Beam Search generated {len(groups)} candidate groups")

    records, best_bio, best_factor = [], -np.inf, -np.inf
    best_partners: list[str] = []

    for g_idx, group_other in enumerate(groups):
        batch_paths = [target_path] + group_other
        print(f"-- Group {g_idx}: {[p.name for p in batch_paths]}")

        models = [cobra.io.read_sbml_model(p) for p in batch_paths]
        sol = simulate(models, env_names, env, show_plots=show_plots)

        for i, p in enumerate(batch_paths):
            traj[p.name] = sol.y[i]

        for i, p in enumerate(batch_paths):
            final_bio = sol.y[i, -1]
            is_target = p.name == TARGET_FILE
            factor    = final_bio / base_bio if is_target else np.nan

            print(f"   {p.name:<42} final biomass = {final_bio:>8.4f}"
                  f"{' | factor = %.4f' % factor if is_target else ''}")

            records.append(dict(
                env_id=env_id, group_id=g_idx, model_name=p.name,
                final_biomass=final_bio, factor=factor,
                baseline_biomass=base_bio if is_target else np.nan
            ))

            if is_target and factor > best_factor:
                best_factor, best_bio = factor, final_bio
                best_partners = [q.name for q in batch_paths if q.name != TARGET_FILE]

        del models
        gc.collect()

    print(f"** Env {env_id}: best target biomass = {best_bio:.4f} | "
          f"best factor = {best_factor:.4f} partners = {best_partners}")

    df = pd.DataFrame(records)
    df["best_target_biomass"] = best_bio
    df["best_target_factor"]  = best_factor
    df["best_partner_names"]  = ";".join(best_partners)

    cols = ["env_id","group_id","model_name","final_biomass","factor",
            "baseline_biomass","best_target_biomass","best_target_factor",
            "best_partner_names"]
    Path(RESULT_DIR).mkdir(exist_ok=True)
    df[cols].to_csv(Path(RESULT_DIR) /
        f"dfba_final_biomass_env{env_id}_bs{batch_size}.csv", index=False)

    pd.DataFrame(traj).to_csv(Path(TRAJ_DIR) /
        f"biomass_env{env_id}.csv", index=False)

# ─────────── CLI ───────────
def main():
    ap = argparse.ArgumentParser("Group-dynamic dFBA runner (single-env)")
    ap.add_argument("--env", type=int)
    ap.add_argument("--env_start", type=int)
    ap.add_argument("--env_stop",  type=int)
    ap.add_argument("--batch_size","--bs", type=int, dest="batch_size", default=3)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    try:
        if args.env is None and (args.env_start is None or args.env_stop is None):
            ap.error("Either --env or both --env_start/--env_stop must be specified")
        if args.env is not None:
            run_single_env(args.env, args.batch_size, show_plots=args.plot)
        else:
            for eid in range(args.env_start, args.env_stop+1):
                run_single_env(eid, args.batch_size, show_plots=args.plot)
    except Exception:
        traceback.print_exc(); sys.exit(1)
    finally:
        gc.collect()

if __name__ == "__main__":
    main()
