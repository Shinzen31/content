#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group-dynamic FBA runner (target-in-every-group) — single-environment mode
=========================================================================

• 保留原有全部计算 & 绘图逻辑  
• 通过 --env 与 --bs 参数，可由外部调度脚本并行运行  
• 结果写入 results/dfba_final_biomass_env{env}_bs{bs}.csv
"""

from __future__ import annotations
import os, sys, gc, argparse
from math import ceil
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ───────────────── Parameters ──────────────────
MODEL_DIR   = "models_gapfilled"
TARGET_FILE = "GCF_000010425.1_ASM1042v1_protein_gapfilled_noO2_appliedMedium.xml"
ENV_FILE    = "environment_ball.tsv"
TIME_SPAN   = (0, 48)
N_POINTS    = 40
INIT_BIOMASS = 0.1
MAX_UPTAKE   = 10.0
RESULT_DIR   = "results"
EPS_BASE     = 1e-9
MONOD_CONSTANT = 10
YIELD_FACTOR = 0.1
WATER_CONSTANT = 10

cobra.Configuration().solver = "glpk"
# cobra.Configuration().solver = "gurobi"

# ─────────── Basic functions ───────────
def apply_env(model: cobra.Model, env: dict[str, float],
              upper_bound: float = MAX_UPTAKE):
    for rid, conc in env.items():
        if model.reactions.has_id(rid):
            rxn = model.reactions.get_by_id(rid)
            rxn.lower_bound = -abs(conc)
            rxn.upper_bound = upper_bound
    model.reactions.EX_cpd00001_e0.lower_bound = WATER_CONSTANT
    return model.optimize()

def avoid_zero_met(states, changes):
    return np.where(states + changes < 0, -states, changes)

def get_growth(model, env_names, env_state):
    uptake_bounds = {rid: env_state.get(rid, 0.0) /
                     (env_state.get(rid, 0.0) + MONOD_CONSTANT)
                     for rid in env_names}
    solution = apply_env(model, uptake_bounds)
    mu = solution.objective_value #* 0.1
    fluxes = {}
    for rid in env_names:
        if model.reactions.has_id(rid):
            v = solution.fluxes.get(rid, 0.0)
            fluxes[rid] = v * YIELD_FACTOR if v > 0 else v
        else:
            fluxes[rid] = 0.0              # ← FIX: 确保键存在
    return mu, fluxes

def simulate(models_orig, env_names, env_initial,
             *, t_span=TIME_SPAN, n_points=N_POINTS, show_plots=False):
    n_models = len(models_orig)

    def ode(t, y):
        biomasses = np.maximum(y[:n_models], 0.0)
        mets      = np.maximum(y[n_models:], 0.0)
        env_state = dict(zip(env_names, mets))

        d_bio  = np.zeros(n_models)
        d_mets = np.zeros_like(mets)

        for i in range(n_models):
            mu_norm, fluxes = get_growth(models_orig[i], env_names, env_state)
            d_bio[i]  = mu_norm * biomasses[i]
            d_mets   += biomasses[i] * np.array([fluxes[r] for r in env_names])

        d_mets = avoid_zero_met(mets, d_mets)
        return np.concatenate((d_bio, d_mets))

    y0 = np.concatenate(([INIT_BIOMASS] * n_models,
                         [env_initial.get(r, 0.0) for r in env_names]))
    sol = solve_ivp(ode, t_span, y0, t_eval=np.linspace(*t_span, n_points))

    if show_plots:
        plt.figure(figsize=(6, 4))
        for i in range(n_models):
            plt.plot(sol.t, sol.y[i], label=f"Pop {i+1}")
        plt.xlabel("Time (h)"); plt.ylabel("Biomass")
        plt.title("Group Population Growth")
        plt.legend(); plt.tight_layout(); plt.show()

        plt.figure(figsize=(6, 4))
        for j, rid in enumerate(env_names):
            plt.plot(sol.t, sol.y[n_models + j], label=rid)
        plt.xlabel("Time (h)"); plt.ylabel("Concentration")
        plt.title("Environment Metabolite Dynamics")
        plt.legend(ncol=2, fontsize=7); plt.tight_layout(); plt.show()

    return sol

# ─────────── Utility functions ───────────
def load_environment_ball(path: str | Path):
    tbl = pd.read_csv(path, sep="\t")
    tbl["concentration"] = pd.to_numeric(tbl["concentration"],
                                         errors="coerce").fillna(0.0)
    if "env_id" in tbl.columns:
        for _, g in tbl.groupby("env_id"):
            yield dict(zip(g["metabolite_id"], g["concentration"]))
    else:
        n_met = tbl["metabolite_id"].nunique()
        for i in range(len(tbl) // n_met):
            chunk = tbl.iloc[i * n_met:(i + 1) * n_met]
            yield dict(zip(chunk["metabolite_id"], chunk["concentration"]))

def iter_model_paths(model_dir: str | Path):
    return sorted(Path(model_dir).glob("*.xml"))

# ─────────── Core runner ───────────
def run_single_env(env_id: int, batch_size: int, show_plots=False):
    envs = list(load_environment_ball(ENV_FILE))
    if not (0 <= env_id < len(envs)):
        raise ValueError(f"env_id {env_id} out of range (0-{len(envs)-1})")

    target_path, other_paths = None, []
    for p in iter_model_paths(MODEL_DIR):
        (target_path := p) if p.name == TARGET_FILE else other_paths.append(p)
    if target_path is None:
        raise FileNotFoundError(f"Target model {TARGET_FILE} not found")
    if batch_size < 2:
        raise ValueError("batch_size must be ≥2")

    env = envs[env_id]
    env_names = list(env)
    chunk = batch_size - 1
    num_groups = ceil(len(other_paths) / chunk) if other_paths else 1
    records  = []

    # ─── baseline ───────────────────────────────────
    solo_model  = cobra.io.read_sbml_model(target_path)
    base_sol    = simulate([solo_model], env_names, env)
    base_bio    = base_sol.y[0, -1] or EPS_BASE
    print(f"[Baseline] target alone biomass = {base_bio:.6f}")

    print(f"\n=== Environment {env_id}  (batch_size={batch_size}) ===")
    best_bio = -np.inf
    best_factor = -np.inf
    best_partners = []

    for g_idx in range(num_groups):
        group_other = other_paths[g_idx * chunk : g_idx * chunk + chunk]
        batch_paths = [target_path] + group_other
        print(f"-- Group {g_idx}: {[p.name for p in batch_paths]}")

        models = [cobra.io.read_sbml_model(p) for p in batch_paths]
        sol = simulate(models, env_names, env, show_plots=show_plots)

        for i, p in enumerate(batch_paths):
            final_bio = sol.y[i, -1]
            is_target = p.name == TARGET_FILE
            factor    = final_bio / base_bio if is_target else np.nan

            print(f"   {p.name:<40} final biomass = {final_bio:>8.4f}"
                  f"{' | factor = %.4f' % factor if is_target else ''}")

            records.append(dict(
                env_id=env_id,
                group_id=g_idx,
                model_name=p.name,
                final_biomass=final_bio,
                factor=factor,
                baseline_biomass=base_bio if is_target else np.nan
            ))

            if is_target:
                if final_bio > best_bio:
                    best_bio = final_bio
                if factor > best_factor:
                    best_factor   = factor
                    best_partners = [q.name for q in batch_paths if q.name != TARGET_FILE]

        del models
        gc.collect()

    print(f"** Env {env_id}: best target biomass = {best_bio:.4f} "
          f"| best factor = {best_factor:.4f} partners = {best_partners}")

    # ─── 写结果 ───────────────────────────────────────
    df = pd.DataFrame(records)
    df["best_target_biomass"] = best_bio
    df["best_target_factor"]  = best_factor
    df["best_partner_names"]  = ";".join(best_partners)

    ordered = ["env_id","group_id","model_name",
               "final_biomass","factor","baseline_biomass",
               "best_target_biomass","best_target_factor","best_partner_names"]
    df = df[ordered]

    Path(RESULT_DIR).mkdir(exist_ok=True)
    out_file = Path(RESULT_DIR) / f"dfba_final_biomass_env{env_id}_bs{batch_size}.csv"
    df.to_csv(out_file, index=False)

# ─────────── CLI ───────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group-dynamic dFBA runner (single-env mode friendly)")
    parser.add_argument("--env", type=int, help="single environment id to simulate")
    parser.add_argument("--env_start", type=int, help="first env_id (inclusive)")
    parser.add_argument("--env_stop",  type=int, help="last  env_id (inclusive)")
    parser.add_argument("--batch_size", "--bs", type=int, dest="batch_size", default=3,
                        help="group size including target (default 3)")
    parser.add_argument("--plot", action="store_true", help="show biomass & metabolite plots")
    args = parser.parse_args()

    if args.env is None and (args.env_start is None or args.env_stop is None):
        parser.error("Either --env or both --env_start/--env_stop must be specified")

    if args.env is not None:
        run_single_env(args.env, args.batch_size, args.plot)
    else:
        for eid in range(args.env_start, args.env_stop + 1):
            run_single_env(eid, args.batch_size, args.plot)

    gc.collect()
    os._exit(0)
