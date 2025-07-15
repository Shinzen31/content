#!/bin/bash
#$ -S /bin/bash
#$ -N dfba_arr             # 作业名
#$ -q long.q               # 使用 long.q
#$ -cwd
#$ -t 1-100                # 100 个 array task (ID: 1…100)
#$ -tc 4                   # 同时最多 4 个 task 运行/预留
#$ -pe thread 32           # 每 task 请求 32 线程
#$ -l h_vmem=4G            # 每线程 4 GB，总 128 GB
#$ -l h_rt=168:00:00       # 单 task 最长 168 h（7 天）
#$ -o logs/dfba.$TASK_ID.out
#$ -e logs/dfba.$TASK_ID.err

module load miniconda3
source activate dfba

# ── 计算本 task 负责的 env 范围 ───────────────────────────
ENV_PER_TASK=100
IDX=$(( SGE_TASK_ID - 1 ))         # 0-based
START=$(( IDX * ENV_PER_TASK ))
STOP=$(( START + ENV_PER_TASK - 1 ))
[ $STOP -gt 9999 ] && STOP=9999    # 不超过 9999

echo "Task $SGE_TASK_ID handles env $START – $STOP (32 threads)"

# ── 调 run.py：chunk × 4(bs)=32，对应申请的 32 threads ──
python run.py \
  --start_env $START \
  --stop_env  $STOP \
  --chunk     8 \
  --timeout   5400   # 1.5 小时，可按需再调
