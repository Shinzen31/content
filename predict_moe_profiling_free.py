#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_hybrid.py
=================

This file demonstrates how to couple a **profiling-free parallelism planner** with an
actual **hybrid parallel training framework** that combines:

- Data Parallelism (DP)
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Expert Parallelism (EP, for Mixture-of-Experts models)

The project goal is to show how to:
1. Start from **hardware + model specs**
2. Use a **profiling-free planner** to decide (dp, tp, pp, ep)
3. Build corresponding **process groups**
4. Later (in other sections), actually use these groups in model layers and training loop.

This first part implements:
- Profiling-free planner (HybridPlanner)
- Communication group manager (ProcessGroupManager)

The code is heavily commented so it can be used in a PhD defense to show both
engineering and scientific reasoning.
"""

import os
import math
import json
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple

import torch
import torch.distributed as dist


# ============================================================
# Section 1: Hardware and Model Specifications
# ============================================================

@dataclass
class HardwareSpec:
    """
    Describe the hardware resources available.

    Attributes
    ----------
    num_nodes : int
        Number of compute nodes in the cluster.
    acc_per_node : int
        Number of accelerators (e.g. GPUs) per node.
    peak_tflops : float
        Peak compute power per accelerator (TFLOPS).
    mem_gb : float
        Memory capacity per accelerator (GB).
    net_bandwidth_gbps : float
        Effective bandwidth of interconnect (Gbps).
    net_latency_us : float
        Network latency (microseconds).
    topo : str
        Topology type (e.g. "fat-tree", "mesh").
    """
    num_nodes: int = 1
    acc_per_node: int = 1
    peak_tflops: float = 20.0
    mem_gb: float = 16.0
    net_bandwidth_gbps: float = 100.0
    net_latency_us: float = 2.0
    topo: str = "fat-tree"


@dataclass
class ModelSpec:
    """
    Describe the model requirements.

    Attributes
    ----------
    params_billion : float
        Number of parameters in billions.
    act_mb_per_sample : float
        Activation footprint per sample in MB.
    flops_per_sample_giga : float
        Computation required per sample in GFLOPs.
    seq_len : int
        Sequence length (useful for Transformers).
    experts : int
        Number of experts in MoE.
    topk : int
        Number of experts selected by the router.
    """
    params_billion: float = 0.05
    act_mb_per_sample: float = 10.0
    flops_per_sample_giga: float = 50.0
    seq_len: int = 128
    experts: int = 8
    topk: int = 2


@dataclass
class Plan:
    """
    Output of the profiling-free planner.

    Attributes
    ----------
    dp, tp, pp, ep : int
        Degrees of Data, Tensor, Pipeline, and Expert parallelism.
    global_batch : int
        Total global batch size.
    micro_batch : int
        Micro-batch size (important for pipeline parallelism).
    notes : str
        Explanatory notes or warnings.
    """
    dp: int
    tp: int
    pp: int
    ep: int
    global_batch: int
    micro_batch: int
    notes: str


# ============================================================
# Section 2: Profiling-Free Planner
# ============================================================

class HybridPlanner:
    """
    Profiling-free planner that decides (dp, tp, pp, ep) configuration.

    Unlike empirical tuning (profiling multiple runs), this planner relies
    on *analytical cost models* to predict which combination will minimize
    iteration time while respecting memory limits.

    This is important in PhD defense context:
    - Shows you can reason analytically about scaling laws
    - Reduces ecological and economic cost of trial-and-error profiling
    """

    def __init__(self, hw: HardwareSpec, ms: ModelSpec):
        self.hw = hw
        self.ms = ms
        self.world = hw.num_nodes * hw.acc_per_node

    # ----------------------------
    # Compute model
    # ----------------------------
    def _compute_time(self, flops_total: float, eff: float = 0.35) -> float:
        """
        Estimate computation time for given FLOPs.

        eff : float
            Efficiency (30-40% typical in large-scale training).
        """
        peak = self.hw.peak_tflops * self.world * eff
        return flops_total / max(1e-6, peak)

    # ----------------------------
    # Communication models
    # ----------------------------
    def _allreduce_time(self, tensor_mb: float, p: int) -> float:
        """
        Approximate allreduce time with log(p) steps.
        """
        B = self.hw.net_bandwidth_gbps
        L = self.hw.net_latency_us * 1e-6
        size_Mb = tensor_mb * 8.0
        steps = max(1.0, math.log2(max(1, p)))
        return steps * (L + size_Mb / (B * 1e3))

    def _pp_bubble_overhead(self, stages: int, micro_batches: int) -> float:
        """
        Pipeline bubble fraction.
        """
        return max(0.0, (stages - 1) / max(1, micro_batches))

    # ----------------------------
    # Memory feasibility
    # ----------------------------
    def _memory_ok(self, tp: int, pp: int, ep: int, micro_batch: int) -> bool:
        params_total_gb = self.ms.params_billion * 2.0
        params_per_rank = params_total_gb / max(1, tp*ep)
        acts_per_rank = (self.ms.act_mb_per_sample * micro_batch / max(1, pp)) / 1024.0
        optimizer_states = params_per_rank * 2.0
        needed = params_per_rank + acts_per_rank + optimizer_states + 2.0
        return needed <= self.hw.mem_gb

    # ----------------------------
    # Main planning function
    # ----------------------------
    def plan(self, global_batch: int = 64, target_micro: int = 4) -> Plan:
        best = None
        notes = []
        for tp in [1, 2, 4]:
            for pp in [1, 2, 4]:
                for ep in [1, 2, 4]:
                    if tp * pp * ep > self.world:
                        continue
                    dp = max(1, self.world // (tp*pp*ep))
                    if not self._memory_ok(tp, pp, ep, target_micro):
                        continue
                    flops_total_t = (self.ms.flops_per_sample_giga * 1e-3) * (global_batch / dp)
                    t_comp = self._compute_time(flops_total_t)
                    grad_mb = self.ms.params_billion * 1000.0 * 2.0 / max(1, tp*ep)
                    t_dp = self._allreduce_time(grad_mb, dp) if dp > 1 else 0.0
                    t_tp = 0.001 * (tp - 1)
                    t_ep = 0.001 * self.ms.topk * (ep - 1)
                    bubble = self._pp_bubble_overhead(pp, target_micro)
                    t_total = (t_comp * (1.0 + bubble)) + t_dp + t_tp + t_ep
                    score = t_total
                    if (best is None) or (score < best[0]):
                        best = (score, dp, tp, pp, ep, target_micro)
        if best is None:
            best = (0.0, 1, 1, 1, 1, target_micro)
            notes.append("Memory constraints forced single-rank plan.")
        _, dp, tp, pp, ep, micro = best
        if dp*tp*pp*ep != self.world:
            notes.append(f"World size {self.world} not fully used.")
        return Plan(dp=dp, tp=tp, pp=pp, ep=ep, global_batch=global_batch,
                    micro_batch=micro, notes=" | ".join(notes))


# ============================================================
# Section 3: Process Group Manager
# ============================================================

class ProcessGroupManager:
    """
    Manage distributed communication groups according to planner output.

    Why is this necessary?
    ----------------------
    - In DP/TP/PP/EP, each dimension requires its own set of process groups.
    - E.g. for DP, we allreduce gradients among replicas.
    - For TP, we allgather/reduce-scatter within a tensor parallel group.
    - For PP, we send/recv between pipeline stages.
    - For EP, we all-to-all across expert shards.

    This class creates and stores the necessary subgroups.
    """

    def __init__(self, plan: Plan, world_size: int, rank: int):
        self.plan = plan
        self.world_size = world_size
        self.rank = rank
        self.dp_group = None
        self.tp_group = None
        self.pp_group = None
        self.ep_group = None

    def build_groups(self):
        """
        Build torch.distributed groups for DP, TP, PP, EP.

        Note: In practice, we'd partition ranks carefully.
        For simplicity, we assume ranks are ordered in a cartesian grid:
        [dp][tp][pp][ep].
        """
        dp, tp, pp, ep = self.plan.dp, self.plan.tp, self.plan.pp, self.plan.ep

        if not dist.is_initialized():
            raise RuntimeError("torch.distributed not initialized")

        # Simple grid mapping
        grid = []
        for d in range(dp):
            for t in range(tp):
                for p in range(pp):
                    for e in range(ep):
                        grid.append((d, t, p, e))
        if len(grid) != self.world_size:
            warnings.warn("Grid size mismatch with world_size")

        # Helper to get subgroup ranks
        def subgroup_ranks(dim: str, coord: Tuple[int,int,int,int]) -> List[int]:
            ranks = []
            for idx, (d, t, p, e) in enumerate(grid):
                if dim == "dp" and (t,p,e) == (coord[1],coord[2],coord[3]):
                    ranks.append(idx)
                if dim == "tp" and (d,p,e) == (coord[0],coord[2],coord[3]):
                    ranks.append(idx)
                if dim == "pp" and (d,t,e) == (coord[0],coord[1],coord[3]):
                    ranks.append(idx)
                if dim == "ep" and (d,t,p) == (coord[0],coord[1],coord[2]):
                    ranks.append(idx)
            return ranks

        # Current rank coordinates
        coord = grid[self.rank]

        self.dp_group = dist.new_group(ranks=subgroup_ranks("dp", coord))
        self.tp_group = dist.new_group(ranks=subgroup_ranks("tp", coord))
        self.pp_group = dist.new_group(ranks=subgroup_ranks("pp", coord))
        self.ep_group = dist.new_group(ranks=subgroup_ranks("ep", coord))

    def info(self) -> Dict[str, List[int]]:
        """
        Return info about group membership for this rank.
        """
        return {
            "dp_group": dist.get_world_size(self.dp_group) if self.dp_group else 1,
            "tp_group": dist.get_world_size(self.tp_group) if self.tp_group else 1,
            "pp_group": dist.get_world_size(self.pp_group) if self.pp_group else 1,
            "ep_group": dist.get_world_size(self.ep_group) if self.ep_group else 1,
        }
# ============================================================
# Section 4: Tensor Parallelism (TP) Layers
# ============================================================

class ColumnParallelLinear(torch.nn.Module):
    """
    Linear layer where weight matrix is column-partitioned across tensor-parallel ranks.

    Suppose weight W has shape [out_features, in_features].
    - In standard Linear: y = x @ W^T + b
    - In ColumnParallelLinear with tp>1:
        * W is split by columns (out_features / tp per rank)
        * Each rank computes partial output y_local
        * Then allgather across TP group to assemble full y

    Benefits
    --------
    - Reduces memory per rank (only store 1/tp of W)
    - Enables scaling to very large hidden sizes
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 process_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pg = process_group if process_group is not None else dist.group.WORLD
        self.tp_world_size = dist.get_world_size(self.pg)
        self.tp_rank = dist.get_rank(self.pg)

        assert out_features % self.tp_world_size == 0, "out_features must be divisible by tp size"
        self.out_per_partition = out_features // self.tp_world_size

        # Local shard of weight and bias
        self.weight = torch.nn.Parameter(torch.empty(
            self.out_per_partition, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_per_partition))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local matmul
        y_local = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y_local = y_local + self.bias

        # Allgather outputs across TP group
        outputs = [torch.empty_like(y_local) for _ in range(self.tp_world_size)]
        dist.all_gather(outputs, y_local, group=self.pg)

        y_full = torch.cat(outputs, dim=-1)
        return y_full


class RowParallelLinear(torch.nn.Module):
    """
    Linear layer where weight matrix is row-partitioned across tensor-parallel ranks.

    Suppose weight W has shape [out_features, in_features].
    - In RowParallelLinear with tp>1:
        * W is split by rows (in_features / tp per rank)
        * Each rank multiplies with local shard of input
        * Outputs are summed with allreduce

    This complements ColumnParallelLinear to enable
    full tensor-parallel Transformers.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 process_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pg = process_group if process_group is not None else dist.group.WORLD
        self.tp_world_size = dist.get_world_size(self.pg)
        self.tp_rank = dist.get_rank(self.pg)

        assert in_features % self.tp_world_size == 0, "in_features must be divisible by tp size"
        self.in_per_partition = in_features // self.tp_world_size

        # Local shard of weight
        self.weight = torch.nn.Parameter(torch.empty(
            out_features, self.in_per_partition))
        torch.nn.init.xavier_uniform_(self.weight)

        if bias and self.tp_rank == 0:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each rank uses its local input slice
        x_local = x[:, :, self.tp_rank*self.in_per_partition:
                       (self.tp_rank+1)*self.in_per_partition]
        y_local = torch.matmul(x_local, self.weight.t())

        # Sum partial outputs
        dist.all_reduce(y_local, group=self.pg)

        if self.bias is not None:
            y_local = y_local + self.bias

        return y_local


# ============================================================
# Section 5: Pipeline Parallelism (PP)
# ============================================================

class PipelineStage(torch.nn.Module):
    """
    Represent one stage of pipeline parallelism.

    In PP, model is split across 'pp' stages.
    Each stage handles a consecutive set of layers.
    Micro-batches are passed sequentially through stages.

    Communication pattern:
    - Forward: send output to next stage
    - Backward: send gradients to previous stage

    Here we provide a simplified interface:
    - forward(x, microbatch_id): compute local stage forward
    - receive from prev stage if not first
    - send to next stage if not last
    """

    def __init__(self, stage_id: int, num_stages: int,
                 module: torch.nn.Module,
                 process_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.module = module
        self.pg = process_group if process_group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.pg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def send_forward(self, x: torch.Tensor, dst_rank: int):
        """
        Send activations to next stage.
        """
        dist.send(x.contiguous(), dst=dst_rank)

    def recv_forward(self, shape: torch.Size, src_rank: int, device: torch.device):
        """
        Receive activations from previous stage.
        """
        x = torch.empty(shape, device=device)
        dist.recv(x, src=src_rank)
        return x

    def send_backward(self, grad: torch.Tensor, dst_rank: int):
        """
        Send gradients backward.
        """
        dist.send(grad.contiguous(), dst=dst_rank)

    def recv_backward(self, shape: torch.Size, src_rank: int, device: torch.device):
        """
        Receive gradients from next stage.
        """
        g = torch.empty(shape, device=device)
        dist.recv(g, src=src_rank)
        return g
# ============================================================
# Section 6: Expert Parallelism (EP) for MoE
# ============================================================

class DistributedMoEHead(torch.nn.Module):
    """
    Distributed Mixture-of-Experts (MoE) with Expert Parallelism (EP).

    Why EP?
    -------
    - A Mixture-of-Experts layer contains many "expert" sub-networks.
    - A router decides which experts to use for each token (typically top-k).
    - If all experts were stored on each GPU, memory would explode.
    - Instead, we shard experts across GPUs (EP).
    - Router decisions are followed by all-to-all communication:
        * Each GPU sends tokens to the GPUs holding the selected experts.
        * Each GPU computes on its local experts.
        * Results are sent back and combined.

    This class demonstrates a simplified version of EP.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 num_experts: int, top_k: int,
                 process_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.pg = process_group if process_group is not None else dist.group.WORLD
        self.ep_world_size = dist.get_world_size(self.pg)
        self.ep_rank = dist.get_rank(self.pg)

        assert num_experts % self.ep_world_size == 0, "Experts must divide evenly across EP ranks"
        self.local_experts = num_experts // self.ep_world_size

        # Router: simple linear projection to logits over experts
        self.router = torch.nn.Linear(input_dim, num_experts)

        # Local expert networks (here, simple feedforward)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, 4*input_dim),
                torch.nn.GELU(),
                torch.nn.Linear(4*input_dim, output_dim)
            )
            for _ in range(self.local_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with expert parallelism.
        x: [batch, hidden]

        Steps:
        1. Router produces logits over experts.
        2. Top-k experts chosen for each token.
        3. Tokens dispatched to the ranks holding those experts (all-to-all).
        4. Local experts compute outputs.
        5. Outputs combined and returned.
        """
        bsz, hidden = x.shape

        # 1. Router
        logits = self.router(x)  # [batch, num_experts]
        topk_scores, topk_indices = torch.topk(logits, self.top_k, dim=-1)

        # 2. Build dispatch mask
        # dispatch[i, e] = weight if token i assigned to expert e
        dispatch_mask = torch.zeros(bsz, self.num_experts, device=x.device)
        dispatch_mask.scatter_(1, topk_indices, torch.softmax(topk_scores, dim=-1))

        # 3. Split tokens per rank (which experts live here?)
        expert_range = range(self.ep_rank*self.local_experts,
                             (self.ep_rank+1)*self.local_experts)
        local_mask = dispatch_mask[:, list(expert_range)]  # [batch, local_experts]

        # Compute how many tokens per local expert
        token_indices = []
        for j in range(self.local_experts):
            assigned = (local_mask[:, j] > 0).nonzero(as_tuple=True)[0]
            token_indices.append(assigned)

        # 4. Local expert compute
        outputs = torch.zeros(bsz, self.output_dim, device=x.device)
        for j, idx in enumerate(token_indices):
            if len(idx) == 0:
                continue
            x_j = x[idx]  # tokens for expert j
            out_j = self.experts[j](x_j)
            weight = local_mask[idx, j].unsqueeze(-1)  # soft routing weight
            outputs[idx] += weight * out_j

        # 5. Allreduce to aggregate outputs from all EP ranks
        dist.all_reduce(outputs, group=self.pg)

        return outputs
# ============================================================
# Section 7: Hybrid Model (combining TP, PP, EP)
# ============================================================

class HybridBlock(torch.nn.Module):
    """
    A simple Transformer-style block that supports:
    - Tensor Parallel Linear layers (Column/Row partitioned)
    - Expert Parallel MoE layer (optional)
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int,
                 use_moe: bool = False, num_experts: int = 0, top_k: int = 1,
                 tp_group: Optional[dist.ProcessGroup] = None,
                 ep_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.use_moe = use_moe
        self.tp_group = tp_group
        self.ep_group = ep_group

        # Self-attention projection (TP across columns)
        self.qkv = ColumnParallelLinear(hidden_size, 3*hidden_size, process_group=tp_group)
        self.proj = RowParallelLinear(hidden_size, hidden_size, process_group=tp_group)

        # Feedforward
        if use_moe:
            self.ff = DistributedMoEHead(hidden_size, hidden_size,
                                         num_experts=num_experts,
                                         top_k=top_k,
                                         process_group=ep_group)
        else:
            self.ff = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, ffn_hidden_size),
                torch.nn.GELU(),
                torch.nn.Linear(ffn_hidden_size, hidden_size),
            )

        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.norm2 = torch.nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention (simplified)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.softmax(q @ k.transpose(-2,-1) / (q.shape[-1]**0.5), dim=-1)
        h = attn @ v
        h = self.proj(h)
        x = x + self.norm1(h)

        # FFN or MoE
        h2 = self.ff(x)
        x = x + self.norm2(h2)

        return x


class HybridModel(torch.nn.Module):
    """
    Build a model according to planner output:
    - Split layers into PP stages
    - Inside each stage, use TP for large matrices
    - Optionally include MoE layers with EP
    """

    def __init__(self, num_layers: int, hidden_size: int, ffn_hidden_size: int,
                 vocab_size: int, plan: Plan,
                 tp_group, pp_group, ep_group, stage_id: int, num_stages: int):
        super().__init__()
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.pp_group = pp_group

        # Split layers evenly across PP stages
        layers_per_stage = num_layers // num_stages
        start = stage_id * layers_per_stage
        end = (stage_id+1) * layers_per_stage

        blocks = []
        for i in range(start, end):
            use_moe = (i % 2 == 0) and (plan.ep > 1)  # every other block uses MoE
            block = HybridBlock(hidden_size, ffn_hidden_size,
                                use_moe=use_moe,
                                num_experts=plan.ep*2,  # simple scaling
                                top_k=2,
                                tp_group=tp_group,
                                ep_group=ep_group)
            blocks.append(block)

        self.blocks = torch.nn.ModuleList(blocks)
        self.ln_f = torch.nn.LayerNorm(hidden_size)
        self.head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# ============================================================
# Section 8: Training Loop with Hybrid Parallelism
# ============================================================

def train_loop(model: HybridModel, dataloader, optimizer,
               device: torch.device, rank: int, plan: Plan,
               pgm: ProcessGroupManager, epochs: int = 1):
    """
    Simplified training loop.

    - Outer layer: DP (Horovod or torch DDP wraps this loop).
    - Inside model: TP, PP, EP applied as per planner output.
    - Micro-batch splitting emulates pipeline parallel schedule.
    """

    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for step, (x, y) in enumerate(dataloader):
            # Device placement
            x = x.to(device)
            y = y.to(device)

            # Split into micro-batches (for PP)
            micro_batches = x.chunk(plan.micro_batch, dim=0)
            y_batches = y.chunk(plan.micro_batch, dim=0)

            losses = []
            for mb, yb in zip(micro_batches, y_batches):
                out = model(mb)
                loss = criterion(out.view(-1, out.size(-1)), yb.view(-1))
                losses.append(loss)

            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0 and step % 10 == 0:
                print(f"[Epoch {epoch} Step {step}] Loss = {loss.item():.4f}")

    return model
# ============================================================
# Section 9: CLI and Integration
# ============================================================

import argparse
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    """
    Simple dummy dataset for demonstration.
    Each sample: random input + random label.
    """
    def __init__(self, num_samples: int = 1024, seq_len: int = 32, vocab_size: int = 1000):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        y = torch.randint(0, self.vocab_size, (self.seq_len,))
        return x, y


def save_plan(plan: Plan, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(plan), f, indent=2)
    print(f"[Planner] Saved plan to {path}")


def load_plan(path: str) -> Plan:
    with open(path, "r") as f:
        d = json.load(f)
    return Plan(**d)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Parallel Training Demo with Profiling-Free Planner")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn-hidden-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--save-plan", type=str, default="./plan.json")
    parser.add_argument("--load-plan", type=str, default="")
    args = parser.parse_args()

    # Init distributed
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hardware/Model specs (example)
    hw = HardwareSpec(num_nodes=1, acc_per_node=world_size, peak_tflops=20.0, mem_gb=16.0)
    ms = ModelSpec(params_billion=0.1, act_mb_per_sample=5.0, flops_per_sample_giga=20.0,
                   experts=8, topk=2)

    # Plan
    if args.load_plan:
        plan = load_plan(args.load_plan)
    else:
        planner = HybridPlanner(hw, ms)
        plan = planner.plan(global_batch=args.batch_size * world_size, target_micro=4)
        if rank == 0:
            save_plan(plan, args.save_plan)

    # Build process groups
    pgm = ProcessGroupManager(plan, world_size, rank)
    pgm.build_groups()

    # Assign stage id
    stage_id = rank % plan.pp
    num_stages = plan.pp

    # Build model
    model = HybridModel(num_layers=args.layers,
                        hidden_size=args.hidden_size,
                        ffn_hidden_size=args.ffn_hidden_size,
                        vocab_size=args.vocab_size,
                        plan=plan,
                        tp_group=pgm.tp_group,
                        pp_group=pgm.pp_group,
                        ep_group=pgm.ep_group,
                        stage_id=stage_id,
                        num_stages=num_stages).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Data
    dataset = DummyDataset(num_samples=512, seq_len=32, vocab_size=args.vocab_size)
    dataloader = DataLoader(dataset, batch_size=plan.micro_batch, shuffle=True)

    # Train
    train_loop(model, dataloader, optimizer, device, rank, plan, pgm, epochs=args.epochs)

    if rank == 0:
        print("[Training Completed]")


if __name__ == "__main__":
    main()
