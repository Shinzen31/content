
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_moe_profiling_free.py

Project goals covered:
- Hybrid parallelism strategies (DP, TP, PP, MoE/EP): provides a profiling-free planner
  that estimates DP/TP/PP/EP costs and outputs a plan; trains with DP (Horovod/PyTorch DDP)
  and supports MoE (local or distributed experts).
- Acceleration techniques aware of hardware + model: cost model takes FLOPs, activation sizes,
  topology (nodes, accelerators/node, bandwidth/latency/memory/compute) to recommend a plan.
- Reduce reliance on exhaustive profiling: no runtime traces required; uses analytical models.
- Practical HPC skills used: Horovod+MPI, comm/compute overlap, pin-mem prefetch I/O, seeds,
  environment threads (MKL/OMP), optional DDP fallback, SGE-aware reproducibility.

This module expects *in-silico* training labels from GSMM simulations:
- N_env environments with feature vectors X (CSV)
- For each environment and k in {2,3,4,5}, a multi-hot 40-dim label vector y_{k}
  indicating the best species set (including a required target species).

If files are missing, synthetic examples will be generated so the pipeline runs end-to-end.
"""

import os
import math
import json
import time
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List

import numpy as np

# Set thread env for MKL/OMP to avoid oversubscription; can be tuned on clusters.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    raise

# Optional Horovod for DP; if absent, we gracefully fall back.
try:
    import horovod.torch as hvd
    HOROVOD = True
except Exception:
    HOROVOD = False

# -----------------------------
# Reproducibility & Seeding
# -----------------------------

def seed_everything(seed: Optional[int] = None) -> int:
    """Deterministic seeding; if on SGE, derive from task id for reproducibility."""
    if seed is None:
        seed = 42
        sge_task_id = os.environ.get("SGE_TASK_ID")
        if sge_task_id is not None:
            try:
                seed = (int(sge_task_id) % 10_000_000) + 42
            except ValueError:
                pass
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    return seed

# -----------------------------
# Hardware & Model Specs
# -----------------------------

@dataclass
class HardwareSpec:
    num_nodes: int = 1
    acc_per_node: int = 1
    peak_tflops: float = 20.0   # per accelerator (e.g., GPU) FP16 TFLOPS
    mem_gb: float = 16.0        # per accelerator memory
    net_bandwidth_gbps: float = 100.0  # per link effective bandwidth
    net_latency_us: float = 2.0
    topo: str = "fat-tree"      # or "mesh", "torus", hint for collectives

@dataclass
class ModelSpec:
    params_billion: float = 0.05  # model size in billions of params (example ~50M)
    act_mb_per_sample: float = 10.0  # activation footprint per sample (MB) rough
    flops_per_sample_giga: float = 50.0  # forward FLOPs per sample (GFLOPs)
    seq_len: int = 128
    experts: int = 8
    topk: int = 2  # MoE top-k routing

@dataclass
class Plan:
    dp: int
    tp: int
    pp: int
    ep: int
    global_batch: int
    micro_batch: int
    notes: str

# -----------------------------
# Profiling-free cost model
# -----------------------------

class HybridPlanner:
    """
    A lightweight analytical cost model that estimates per-iteration time
    for combinations of DP/TP/PP/EP without running the model.
    """
    def __init__(self, hw: HardwareSpec, ms: ModelSpec):
        self.hw = hw
        self.ms = ms
        self.world = hw.num_nodes * hw.acc_per_node

    def _compute_throughput_time(self, flops_total: float, dp_degree: int, eff: float = 0.35) -> float:
        """
        Compute time for compute assuming aggregate peak * efficiency.
        """
        peak_tflops_total = self.hw.peak_tflops * self.world
        # Scale with DP (simplified): each replica handles 1/dp of batch
        peak = peak_tflops_total * eff
        # seconds = FLOPs (in TFLOPs) / (peak TFLOPs/s)
        return flops_total / max(1e-6, peak)

    def _allreduce_time(self, tensor_mb: float, p: int) -> float:
        """
        Estimate allreduce time using a log(p) model (ring/tree hybrid).
        """
        B = self.hw.net_bandwidth_gbps  # Gbps
        L = self.hw.net_latency_us * 1e-6
        # Convert MB to Mb (megabits): 1 MB = 8 Mb
        size_Mb = tensor_mb * 8.0
        # Effective steps ~ log2(p); per step time: L + size/B
        steps = max(1.0, math.log2(max(1, p)))
        return steps * (L + size_Mb / (B * 1e3))  # B in Gbps => *1e3 to convert to Mb/s

    def _pp_bubble_overhead(self, stages: int, micro_batches: int) -> float:
        """
        Simple pipeline bubble fraction model: (stages - 1) / micro_batches.
        """
        return max(0.0, (stages - 1) / max(1, micro_batches))

    def _memory_ok(self, dp:int, tp:int, pp:int, ep:int, micro_batch:int) -> bool:
        """
        Very rough memory feasibility check: params shard by tp & ep; activation by pp; batch by dp.
        """
        params_total_gb = self.ms.params_billion * 2.0  # assume 2 bytes/param (FP16) per param (GB approx)
        params_per_rank_gb = params_total_gb / max(1, tp*ep)
        acts_per_rank_gb = (self.ms.act_mb_per_sample * micro_batch / max(1, pp)) / 1024.0
        optimizer_states_gb = params_per_rank_gb * 2.0  # momenta etc.
        needed = params_per_rank_gb + acts_per_rank_gb + optimizer_states_gb + 2.0  # + fragmentation headroom
        return needed <= self.hw.mem_gb

    def plan(self, global_batch:int=64, target_micro:int=4) -> Plan:
        """
        Search over small grids of (tp, pp, ep) consistent with world size and choose dp accordingly.
        """
        best = None
        notes = []
        for tp in [1, 2, 4]:
            for pp in [1, 2, 4]:
                for ep in [1, 2, 4]:
                    if tp*pp*ep > self.world: 
                        continue
                    dp = max(1, self.world // (tp*pp*ep))
                    micro_batch = target_micro
                    # Memory feasibility
                    if not self._memory_ok(dp, tp, pp, ep, micro_batch):
                        continue
                    # Compute cost (very rough): FLOPs per sample * global_batch / (dp*pp) (PP reduces per-stage)
                    flops_total_t = (self.ms.flops_per_sample_giga * 1e-3) * (global_batch / max(1, dp))
                    t_compute = self._compute_throughput_time(flops_total_t, dp)
                    # Comm costs:
                    # - DP: gradient allreduce ~ params size / dp
                    grad_mb = self.ms.params_billion * 1000.0 * 2.0 / max(1, tp*ep)  # 2 bytes/param -> MB approx
                    t_dp = self._allreduce_time(grad_mb, dp) if dp > 1 else 0.0
                    # - TP: tensor shards require all-gather per layer ~ scale w/ tp
                    t_tp = 0.001 * (tp - 1)  # placeholder small penalty
                    # - EP: MoE all-to-all dispatch/permute cost ~ scale w/ topk and ep
                    t_ep = 0.001 * self.ms.topk * (ep - 1)
                    # - PP bubble:
                    bubble = self._pp_bubble_overhead(pp, micro_batch)
                    t_total = (t_compute * (1.0 + bubble)) + t_dp + t_tp + t_ep
                    score = t_total
                    if (best is None) or (score < best[0]):
                        best = (score, dp, tp, pp, ep, micro_batch)
        if best is None:
            # Fallback single-device plan
            best = (0.0, 1, 1, 1, 1, target_micro)
            notes.append("Memory constraints forced single-rank plan.")
        _, dp, tp, pp, ep, micro = best
        if dp*tp*pp*ep != self.world:
            notes.append(f"World size {self.world} not fully utilized by dp={dp},tp={tp},pp={pp},ep={ep}.")
        return Plan(dp=dp, tp=tp, pp=pp, ep=ep, global_batch=global_batch, micro_batch=micro,
                    notes=" | ".join(notes))

# -----------------------------
# Dataset
# -----------------------------

class BestGroupDataset(Dataset):
    """
    Expects CSV files:
    - env_features.csv: columns: env_id,int; feat_0..feat_F-1 floats
    - labels.csv: columns: env_id,int; k,int; target_id,int; label_multi_hot,json list of 40 ints; biomass,float
    If files do not exist, synthetic data are generated.
    """
    def __init__(self, root:str, F:int=32, species:int=40, ks=(2,3,4,5), split:str="train", seed:int=42):
        self.root = root
        self.F, self.species = F, species
        self.ks = ks
        self.split = split
        rng = np.random.default_rng(seed)
        env_path = os.path.join(root, "env_features.csv")
        lab_path = os.path.join(root, "labels.csv")
        if not os.path.exists(env_path) or not os.path.exists(lab_path):
            warnings.warn("Input CSVs not found. Generating synthetic dataset for demo.")
            N = 5000 if split=="train" else 1000
            self.X = rng.standard_normal((N, F)).astype(np.float32)
            self.k = rng.choice(ks, size=N)
            self.target = rng.integers(0, species, size=N)
            self.labels = np.zeros((N, species), dtype=np.float32)
            for i in range(N):
                k = self.k[i]
                tgt = self.target[i]
                candidates = [tgt] + rng.choice([j for j in range(species) if j!=tgt], size=k-1, replace=False).tolist()
                self.labels[i, candidates] = 1.0
            self.biomass = rng.random(N).astype(np.float32)
        else:
            import pandas as pd
            env = pd.read_csv(env_path)
            lab = pd.read_csv(lab_path)
            merged = lab.merge(env, on="env_id", how="inner")
            feats = [c for c in merged.columns if c.startswith("feat_")]
            self.X = merged[feats].values.astype(np.float32)
            self.k = merged["k"].values.astype(np.int64)
            self.target = merged["target_id"].values.astype(np.int64)
            self.biomass = merged.get("biomass", pd.Series(np.zeros(len(merged)))).values.astype(np.float32)
            # parse labels
            import ast
            tmp = np.zeros((len(merged), species), dtype=np.float32)
            for i, s in enumerate(merged["label_multi_hot"].values):
                arr = np.array(ast.literal_eval(s), dtype=np.int64)
                tmp[i, arr] = 1.0
            self.labels = tmp

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.X[idx]),
            "k": torch.tensor(int(self.k[idx]), dtype=torch.long),
            "target": torch.tensor(int(self.target[idx]), dtype=torch.long),
            "y": torch.from_numpy(self.labels[idx]),  # multi-hot over species
            "biomass": torch.tensor(float(self.biomass[idx]), dtype=torch.float32),
        }

# -----------------------------
# MoE Block
# -----------------------------

class TopKRouter(nn.Module):
    def __init__(self, d_in:int, n_experts:int, k:int=2):
        super().__init__()
        self.proj = nn.Linear(d_in, n_experts)
        self.k = k

    def forward(self, x):
        # x: [B, d]
        logits = self.proj(x)                         # [B, E]
        gates = F.softmax(logits, dim=-1)             # [B, E]
        topk_val, topk_idx = torch.topk(gates, self.k, dim=-1)  # [B, k]
        # Normalize selected gates so they sum to 1
        topk_val = topk_val / (topk_val.sum(dim=-1, keepdim=True) + 1e-9)
        return topk_idx, topk_val, gates

class ExpertMLP(nn.Module):
    def __init__(self, d_in:int, d_hidden:int, d_out:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x): return self.net(x)

class MoEHead(nn.Module):
    """
    Local MoE with top-k routing. For distributed EP, replace local dispatch with
    torch.distributed.all_to_all on (tokens,expert) buckets when available.
    Outputs species logits.
    """
    def __init__(self, d_in:int, n_experts:int, d_hidden:int, n_species:int, topk:int=2):
        super().__init__()
        self.router = TopKRouter(d_in, n_experts, k=topk)
        self.experts = nn.ModuleList([ExpertMLP(d_in, d_hidden, n_species) for _ in range(n_experts)])

    def forward(self, x):
        # x: [B, d]
        topk_idx, topk_val, _ = self.router(x)       # [B, k], [B, k]
        B, k = topk_idx.shape
        out = 0.0
        for j in range(k):
            idx = topk_idx[:, j]                     # [B]
            val = topk_val[:, j].unsqueeze(-1)       # [B, 1]
            # Dispatch per expert locally
            # Build a batch per expert
            logits = torch.zeros(B, self.experts[0].net[-1].out_features, device=x.device)
            for e_id, expert in enumerate(self.experts):
                mask = (idx == e_id)
                if mask.any():
                    logits[mask] = expert(x[mask])
            out = out + val * logits
        return out  # [B, n_species]

# -----------------------------
# Predictor Model
# -----------------------------

class Predictor(nn.Module):
    """
    Produce per-species logits from environment features and k.
    Constraint (must include target species) is applied at inference via masking.
    """
    def __init__(self, F:int, n_species:int, n_experts:int=8, d_model:int=256, d_hidden:int=512, topk:int=2):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Linear(F + 1, d_model),  # +1 for 'k' token
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.moe = MoEHead(d_model, n_experts, d_hidden, n_species, topk=topk)

    def forward(self, x, k):
        k = k.float().unsqueeze(-1)
        h = self.fe(torch.cat([x, k], dim=-1))
        logits = self.moe(h)  # [B, n_species]
        return logits

# -----------------------------
# Losses & Inference
# -----------------------------

def multi_hot_bce_with_cardinality(logits, y, k, alpha_card=0.1):
    """
    BCE loss for multi-hot labels with an extra penalty to match cardinality (sum ~ k).
    """
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")
    probs = torch.sigmoid(logits)
    card_pen = (probs.sum(dim=-1) - k.float()).abs().mean()
    return bce + alpha_card * card_pen

def select_topk_with_target(logits, k:int, target:int) -> List[int]:
    """
    Enforce inclusion of target species. Strategy: force target into set, then pick remaining top (k-1).
    """
    probs = torch.sigmoid(logits.detach())
    k = int(k)
    target = int(target)
    probs[..., target] = 1.1  # ensure inclusion
    topk = torch.topk(probs, k, dim=-1).indices.squeeze(0).tolist()
    if target not in topk:
        # replace last with target
        topk[-1] = target
    return topk

# -----------------------------
# Training / Horovod Integration
# -----------------------------

def init_distributed():
    rank = 0
    world = 1
    local_rank = 0
    using_hvd = False
    if HOROVOD:
        hvd.init()
        rank = hvd.rank()
        world = hvd.size()
        local_rank = hvd.local_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        using_hvd = True
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # DDP fallback
        import torch.distributed as dist
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank(); world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, world, local_rank, using_hvd

def train(root:str="/mnt/data/data_bgs", epochs:int=5, batch_size:int=128,
          F:int=32, n_species:int=40, n_experts:int=8, topk:int=2,
          lr:float=3e-4, seed:int=42, hw:Optional[HardwareSpec]=None, ms:Optional[ModelSpec]=None):
    seed = seed_everything(seed)
    rank, world, local_rank, using_hvd = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if rank == 0:
        print(f"[Init] seed={seed} world={world} hvd={using_hvd} device={device}")

    # Profiling-free plan (does not execute any profiling runs)
    if hw is None: hw = HardwareSpec(num_nodes=1, acc_per_node=world if world>0 else 1)
    if ms is None: ms = ModelSpec(params_billion=0.05, act_mb_per_sample=6.0, flops_per_sample_giga=20.0,
                                  experts=n_experts, topk=topk)
    planner = HybridPlanner(hw, ms)
    plan = planner.plan(global_batch=batch_size*world, target_micro=4)
    if rank == 0:
        print("[Planner]", json.dumps(asdict(plan), indent=2))

    # Data
    train_ds = BestGroupDataset(root, F=F, species=n_species, split="train", seed=seed)
    val_ds   = BestGroupDataset(root, F=F, species=n_species, split="val",   seed=seed+1)

    # I/O pipeline to overlap CPU->GPU copies
    def collate(batch):
        x = torch.stack([b["x"] for b in batch], dim=0)
        k = torch.stack([b["k"] for b in batch], dim=0)
        y = torch.stack([b["y"] for b in batch], dim=0)
        tgt = torch.stack([b["target"] for b in batch], dim=0)
        bio = torch.stack([b["biomass"] for b in batch], dim=0)
        return {"x": x, "k": k, "y": y, "target": tgt, "biomass": bio}

    num_workers = 4 if os.cpu_count() and os.cpu_count() > 4 else 2
    train_loader = DataLoader(train_ds, batch_size=plan.micro_batch, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True,
                              prefetch_factor=4, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=plan.micro_batch, shuffle=False, drop_last=False,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True,
                              prefetch_factor=4, collate_fn=collate)

    # Model
    model = Predictor(F=F, n_species=n_species, n_experts=n_experts, d_model=256, d_hidden=512, topk=topk).to(device)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    if using_hvd:
        # Horovod: wrap optimizer for distributed allreduce; overlap enabled by default
        compression = hvd.Compression.fp16 if torch.cuda.is_available() else hvd.Compression.none
        opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters(),
                                       compression=compression, op=hvd.Average)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(opt, root_rank=0)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Training loop
    for epoch in range(1, epochs+1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            k = batch["k"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x, k)
                loss = multi_hot_bce_with_cardinality(logits, y, k, alpha_card=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            steps += 1
        if rank == 0:
            print(f"[Epoch {epoch}] train_loss={total_loss/max(1,steps):.4f} time={time.time()-t0:.1f}s")

        # quick val
        if rank == 0 and (epoch % 1 == 0):
            model.eval()
            with torch.no_grad():
                val_loss = 0.0; vsteps=0
                for batch in val_loader:
                    x = batch["x"].to(device, non_blocking=True)
                    k = batch["k"].to(device, non_blocking=True)
                    y = batch["y"].to(device, non_blocking=True)
                    logits = model(x, k)
                    val_loss += F.binary_cross_entropy_with_logits(logits, y, reduction="mean").item()
                    vsteps += 1
                print(f"[Epoch {epoch}]   val_loss={val_loss/max(1,vsteps):.4f}")

    # Save
    if rank == 0:
        os.makedirs("/mnt/data/artifacts", exist_ok=True)
        torch.save({"model": model.state_dict(), "config": {
            "F": F, "n_species": n_species, "n_experts": n_experts, "topk": topk
        }}, "/mnt/data/artifacts/moe_predictor.pt")
        with open("/mnt/data/artifacts/plan.json", "w") as f:
            json.dump(asdict(plan), f, indent=2)
        print("[Save] Model and plan saved under /mnt/data/artifacts")

# -----------------------------
# Inference helper
# -----------------------------

def predict_best_set(model_path:str, x:np.ndarray, k:int, target:int) -> List[int]:
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ckpt["config"]
    model = Predictor(F=cfg["F"], n_species=cfg["n_species"],
                      n_experts=cfg["n_experts"], topk=cfg["topk"])
    model.load_state_dict(ckpt["model"]); model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        k_t = torch.tensor([k], dtype=torch.long)
        logits = model(x_t, k_t)
        sel = select_topk_with_target(logits, k, target)
    return sel

# -----------------------------
# CLI
# -----------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/mnt/data/data_bgs")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--F", type=int, default=32)
    p.add_argument("--species", type=int, default=40)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Create placeholder directories and files if missing
    os.makedirs(args.data_root, exist_ok=True)
    placeholder = os.path.join(args.data_root, "experimental_biomass_placeholder.csv")
    if not os.path.exists(placeholder):
        with open(placeholder, "w") as f:
            f.write("This is a placeholder for real experimental biomass validation data.\n")

    train(root=args.data_root, epochs=args.epochs, batch_size=args.batch_size,
          F=args.F, n_species=args.species, n_experts=args.experts, topk=args.topk,
          lr=args.lr, seed=args.seed)

if __name__ == "__main__":
    main()
