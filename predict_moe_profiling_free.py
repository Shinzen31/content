# ============================================================
# Section 1: Data Structures and Profiling-free Planner
# ============================================================
# torchrun --nproc_per_node=4 predict_hybrid.py --epochs 2
import torch
import torch.distributed as dist
from dataclasses import dataclass, asdict
from typing import Optional, List
import json, os

# --------------------------
# Hardware and Model Specs
# --------------------------

@dataclass
class HardwareSpec:
    num_nodes: int
    acc_per_node: int
    peak_tflops: float
    mem_gb: float

@dataclass
class ModelSpec:
    params_billion: float
    act_mb_per_sample: float
    flops_per_sample_giga: float
    experts: int
    topk: int

@dataclass
class Plan:
    dp: int
    tp: int
    pp: int
    ep: int
    micro_batch: int


# --------------------------
# Profiling-free Planner
# --------------------------

class HybridPlanner:
    """
    Profiling-free planner that generates a DP/TP/PP/EP configuration
    based only on hardware + model specs, without runtime profiling.
    """

    def __init__(self, hw: HardwareSpec, ms: ModelSpec):
        self.hw = hw
        self.ms = ms

    def plan(self, global_batch: int, target_micro: int = 4) -> Plan:
        # Step 1: decide DP
        dp = min(global_batch, self.hw.num_nodes * self.hw.acc_per_node)

        # Step 2: decide TP
        tp = 1
        if self.ms.params_billion > 1:
            tp = min(4, self.hw.acc_per_node)

        # Step 3: decide PP
        pp = 1
        if self.ms.params_billion > 10:
            pp = min(4, self.hw.num_nodes)

        # Step 4: decide EP
        ep = 1
        if self.ms.experts > 0:
            ep = min(self.ms.experts, self.hw.acc_per_node)

        # Step 5: micro-batch size
        micro_batch = target_micro

        return Plan(dp=dp, tp=tp, pp=pp, ep=ep, micro_batch=micro_batch)


# --------------------------
# Process Group Manager (REAL implementation)
# --------------------------

class ProcessGroupManager:
    """
    Real process group partitioner:
    - Splits ranks into DP / TP / PP / EP groups
    according to the plan.
    """

    def __init__(self, plan: Plan, world_size: int, rank: int):
        self.plan = plan
        self.world_size = world_size
        self.rank = rank
        self.dp_group: Optional[dist.ProcessGroup] = None
        self.tp_group: Optional[dist.ProcessGroup] = None
        self.pp_group: Optional[dist.ProcessGroup] = None
        self.ep_group: Optional[dist.ProcessGroup] = None

    def build_groups(self):
        dp, tp, pp, ep = self.plan.dp, self.plan.tp, self.plan.pp, self.plan.ep
        assert dp * tp * pp * ep == self.world_size, \
            f"Product dp*tp*pp*ep={dp*tp*pp*ep} must equal world_size={self.world_size}"

        # Global rank → multi-dimensional index
        # rank_id = (((dp_rank * tp) + tp_rank) * pp + pp_rank) * ep + ep_rank
        def unravel(rank):
            ep_rank = rank % ep
            pp_rank = (rank // ep) % pp
            tp_rank = (rank // (ep*pp)) % tp
            dp_rank = (rank // (ep*pp*tp)) % dp
            return dp_rank, tp_rank, pp_rank, ep_rank

        my_dp, my_tp, my_pp, my_ep = unravel(self.rank)

        # ---- Build DP groups ----
        dp_ranks = []
        for i in range(dp):
            dp_ranks.append([
                (((i * tp) + t) * pp + p) * ep + e
                for t in range(tp) for p in range(pp) for e in range(ep)
            ])
        for g in dp_ranks:
            if self.rank in g:
                self.dp_group = dist.new_group(ranks=g)

        # ---- Build TP groups ----
        tp_ranks = []
        for d in range(dp):
            for p in range(pp):
                for e in range(ep):
                    group = [(((d * tp) + t) * pp + p) * ep + e for t in range(tp)]
                    tp_ranks.append(group)
        for g in tp_ranks:
            if self.rank in g:
                self.tp_group = dist.new_group(ranks=g)

        # ---- Build PP groups ----
        pp_ranks = []
        for d in range(dp):
            for t in range(tp):
                for e in range(ep):
                    group = [(((d * tp) + t) * pp + p) * ep + e for p in range(pp)]
                    pp_ranks.append(group)
        for g in pp_ranks:
            if self.rank in g:
                self.pp_group = dist.new_group(ranks=g)

        # ---- Build EP groups ----
        ep_ranks = []
        for d in range(dp):
            for t in range(tp):
                for p in range(pp):
                    group = [(((d * tp) + t) * pp + p) * ep + e for e in range(ep)]
                    ep_ranks.append(group)
        for g in ep_ranks:
            if self.rank in g:
                self.ep_group = dist.new_group(ranks=g)

        return self.dp_group, self.tp_group, self.pp_group, self.ep_group
# ============================================================
# Section 2: Tensor Parallelism (TP) Layers
# ============================================================

import torch.nn as nn
import torch

class ColumnParallelLinear(nn.Module):
    """
    Linear layer where weight matrix is column-partitioned across TP group.

    Suppose W: [out_features, in_features].
    - Partition W along output dim across TP ranks.
    - Each rank holds [out_features/tp, in_features].
    - Forward: y_local = x @ W_local^T
    - Allgather outputs across TP group → full y.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 process_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pg = process_group
        self.tp_world_size = dist.get_world_size(self.pg)
        self.tp_rank = dist.get_rank(self.pg)

        assert out_features % self.tp_world_size == 0, \
            f"out_features {out_features} must be divisible by TP size {self.tp_world_size}"
        self.out_per_partition = out_features // self.tp_world_size

        # Local shard
        self.weight = nn.Parameter(torch.empty(self.out_per_partition, in_features))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_per_partition))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_local = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y_local = y_local + self.bias

        # Allgather across TP group
        outputs = [torch.empty_like(y_local) for _ in range(self.tp_world_size)]
        dist.all_gather(outputs, y_local, group=self.pg)
        y_full = torch.cat(outputs, dim=-1)
        return y_full


class RowParallelLinear(nn.Module):
    """
    Linear layer where weight matrix is row-partitioned across TP group.

    Suppose W: [out_features, in_features].
    - Partition W along input dim across TP ranks.
    - Each rank holds [out_features, in_features/tp].
    - Forward: y_local = x_local @ W_local^T
    - Allreduce across TP group → full y.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 process_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pg = process_group
        self.tp_world_size = dist.get_world_size(self.pg)
        self.tp_rank = dist.get_rank(self.pg)

        assert in_features % self.tp_world_size == 0, \
            f"in_features {in_features} must be divisible by TP size {self.tp_world_size}"
        self.in_per_partition = in_features // self.tp_world_size

        # Local shard
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_partition))
        nn.init.xavier_uniform_(self.weight)

        if bias and self.tp_rank == 0:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_local = x[:, :, self.tp_rank*self.in_per_partition:
                       (self.tp_rank+1)*self.in_per_partition]
        y_local = torch.matmul(x_local, self.weight.t())
        dist.all_reduce(y_local, group=self.pg)

        if self.bias is not None:
            y_local = y_local + self.bias

        return y_local


# ============================================================
# Section 3: Pipeline Parallelism (PP)
# ============================================================

class PipelineStage(nn.Module):
    """
    One stage of pipeline parallelism.

    - Model is split into 'pp' stages.
    - Each stage runs a block of layers.
    - Forward pass: send activations to next stage.
    - Backward pass: send gradients to prev stage.
    """

    def __init__(self, stage_id: int, num_stages: int,
                 module: nn.Module,
                 process_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.module = module
        self.pg = process_group
        self.rank = dist.get_rank(self.pg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def send_forward(self, x: torch.Tensor, dst_rank: int):
        """Send activations to next stage."""
        dist.send(x.contiguous(), dst=dst_rank)

    def recv_forward(self, shape: torch.Size, src_rank: int, device: torch.device):
        """Receive activations from previous stage."""
        x = torch.empty(shape, device=device)
        dist.recv(x, src=src_rank)
        return x

    def send_backward(self, grad: torch.Tensor, dst_rank: int):
        """Send gradients to previous stage."""
        dist.send(grad.contiguous(), dst=dst_rank)

    def recv_backward(self, shape: torch.Size, src_rank: int, device: torch.device):
        """Receive gradients from next stage."""
        g = torch.empty(shape, device=device)
        dist.recv(g, src=src_rank)
        return g
# ============================================================
# Section 4: Expert Parallelism (EP) for MoE
# ============================================================

import torch.nn.functional as F

class DistributedMoEHead(nn.Module):
    """
    Mixture-of-Experts (MoE) with Expert Parallelism (EP).

    核心思想：
    - MoE 层有很多专家 (experts)，每个是一个小型 FFN。
    - Router 负责为每个 token 选择 top-k 专家。
    - 不同专家分布在不同 rank 上 (EP)，避免显存爆炸。
    - 训练时需要 all-to-all 通信：
        * 每个 rank 把属于其他专家的 token 发给对应的 rank
        * 本地专家计算
        * 再把结果 all-to-all 传回原 rank
    """

    def __init__(self, input_dim: int, output_dim: int,
                 num_experts: int, top_k: int,
                 process_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.pg = process_group

        self.ep_world_size = dist.get_world_size(self.pg)
        self.ep_rank = dist.get_rank(self.pg)

        assert num_experts % self.ep_world_size == 0, \
            f"num_experts {num_experts} must be divisible by EP size {self.ep_world_size}"
        self.local_experts = num_experts // self.ep_world_size

        # Router: token → expert logits
        self.router = nn.Linear(input_dim, num_experts)

        # Local experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 4*input_dim),
                nn.GELU(),
                nn.Linear(4*input_dim, output_dim)
            )
            for _ in range(self.local_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, hidden]
        流程：
        1. Router: 计算 token→专家 logits
        2. Top-k 选择：得到每个 token 的专家分配
        3. all-to-all：把 token 分发到对应专家所在 rank
        4. 本地专家计算
        5. all-to-all：结果返回原 rank
        6. 聚合输出
        """
        bsz, hidden = x.shape

        # Step 1: router logits
        logits = self.router(x)  # [batch, num_experts]
        topk_scores, topk_indices = torch.topk(logits, self.top_k, dim=-1)

        # Step 2: soft routing 权重
        dispatch_mask = torch.zeros(bsz, self.num_experts, device=x.device)
        dispatch_mask.scatter_(1, topk_indices, F.softmax(topk_scores, dim=-1))

        # Step 3: 划分到本地专家 (分配给当前 rank 的 experts)
        expert_range = range(self.ep_rank*self.local_experts,
                             (self.ep_rank+1)*self.local_experts)
        local_mask = dispatch_mask[:, list(expert_range)]  # [batch, local_experts]

        # 收集本地 expert 的输入
        outputs_local = torch.zeros(bsz, self.output_dim, device=x.device)
        for j in range(self.local_experts):
            idx = (local_mask[:, j] > 0).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            x_j = x[idx]
            out_j = self.experts[j](x_j)
            weight = local_mask[idx, j].unsqueeze(-1)
            outputs_local[idx] += weight * out_j

        # Step 4: all-reduce 聚合结果
        dist.all_reduce(outputs_local, group=self.pg)

        return outputs_local
# ============================================================
# Section 5: Hybrid Model (结合 TP, PP, EP)
# ============================================================

class HybridBlock(nn.Module):
    """
    Transformer-style block，支持：
    - TP: QKV/Projection 的张量并行
    - EP: 可选的 MoE 专家层
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int,
                 use_moe: bool = False, num_experts: int = 0, top_k: int = 1,
                 tp_group: Optional[dist.ProcessGroup] = None,
                 ep_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.use_moe = use_moe
        self.tp_group = tp_group
        self.ep_group = ep_group

        # Self-attention projections (TP 分片)
        self.qkv = ColumnParallelLinear(hidden_size, 3*hidden_size, process_group=tp_group)
        self.proj = RowParallelLinear(hidden_size, hidden_size, process_group=tp_group)

        # FFN 或 MoE
        if use_moe:
            self.ff = DistributedMoEHead(hidden_size, hidden_size,
                                         num_experts=num_experts,
                                         top_k=top_k,
                                         process_group=ep_group)
        else:
            self.ff = nn.Sequential(
                nn.Linear(hidden_size, ffn_hidden_size),
                nn.GELU(),
                nn.Linear(ffn_hidden_size, hidden_size)
            )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (q.shape[-1]**0.5), dim=-1)
        h = attn @ v
        h = self.proj(h)
        x = x + self.norm1(h)

        # FFN / MoE
        h2 = self.ff(x)
        x = x + self.norm2(h2)

        return x


class HybridModel(nn.Module):
    """
    根据 planner 输出自动构建：
    - TP: 张量并行层
    - PP: 分 stage
    - EP: MoE 层
    """

    def __init__(self, num_layers: int, hidden_size: int, ffn_hidden_size: int,
                 vocab_size: int, plan: Plan,
                 tp_group, pp_group, ep_group, stage_id: int, num_stages: int):
        super().__init__()
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.pp_group = pp_group

        # 按 stage 均匀分层
        layers_per_stage = num_layers // num_stages
        start = stage_id * layers_per_stage
        end = (stage_id+1) * layers_per_stage

        blocks = []
        for i in range(start, end):
            use_moe = (i % 2 == 0) and (plan.ep > 1)  # 偶数层用 MoE
            block = HybridBlock(hidden_size, ffn_hidden_size,
                                use_moe=use_moe,
                                num_experts=plan.ep * 2,
                                top_k=2,
                                tp_group=tp_group,
                                ep_group=ep_group)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# ============================================================
# Section 6: Training Loop
# ============================================================

def train_loop(model: HybridModel, dataloader, optimizer,
               device: torch.device, rank: int, plan: Plan,
               pgm: ProcessGroupManager, epochs: int = 1):
    """
    训练循环：
    - 外层：DP（梯度同步由 DDP/Horovod 完成）
    - 内层：TP, PP, EP 在 model.forward 中自动生效
    - Micro-batch 拆分用于 pipeline
    """

    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for step, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            # Micro-batch 切分 (for PP)
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
# Section 7: CLI and Main Entrypoint
# ============================================================

import argparse
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    """
    一个简单的数据集，用于演示。
    - 每个样本是一个随机序列 (x, y)
    - 实际上你可以替换成真实数据
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
    parser = argparse.ArgumentParser(description="Hybrid Parallel Training Demo (DP+TP+PP+EP)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn-hidden-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--save-plan", type=str, default="./plan.json")
    parser.add_argument("--load-plan", type=str, default="")
    args = parser.parse_args()

    # 初始化分布式
    if not dist.is_initialized():
        dist.init_process_group("nccl")  # 多GPU推荐 nccl，CPU可用 gloo
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建硬件 & 模型规格 (示例)
    hw = HardwareSpec(num_nodes=1, acc_per_node=world_size,
                      peak_tflops=20.0, mem_gb=16.0)
    ms = ModelSpec(params_billion=0.1, act_mb_per_sample=5.0,
                   flops_per_sample_giga=20.0,
                   experts=8, topk=2)

    # 生成并行策略
    if args.load_plan:
        plan = load_plan(args.load_plan)
    else:
        planner = HybridPlanner(hw, ms)
        plan = planner.plan(global_batch=args.batch_size * world_size, target_micro=4)
        if rank == 0:
            save_plan(plan, args.save_plan)

    # 构建进程组
    pgm = ProcessGroupManager(plan, world_size, rank)
    dp_group, tp_group, pp_group, ep_group = pgm.build_groups()

    # 分配 PP stage
    stage_id = rank % plan.pp
    num_stages = plan.pp

    # 构建模型
    model = HybridModel(num_layers=args.layers,
                        hidden_size=args.hidden_size,
                        ffn_hidden_size=args.ffn_hidden_size,
                        vocab_size=args.vocab_size,
                        plan=plan,
                        tp_group=tp_group,
                        pp_group=pp_group,
                        ep_group=ep_group,
                        stage_id=stage_id,
                        num_stages=num_stages).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # DataLoader
    dataset = DummyDataset(num_samples=512, seq_len=32, vocab_size=args.vocab_size)
    dataloader = DataLoader(dataset, batch_size=plan.micro_batch, shuffle=True)

    # 训练
    train_loop(model, dataloader, optimizer, device, rank, plan, pgm, epochs=args.epochs)

    if rank == 0:
        print("[Training Completed]")


if __name__ == "__main__":
    main()
