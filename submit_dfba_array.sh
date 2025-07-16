#!/bin/bash
#$ -S /bin/bash
#$ -N dfba_arr_fast        # 作业名
#$ -q long.q,highmem.q,maiage.q   # 多队列，提高可调度性
#$ -cwd
#$ -t 1-10                 # 先跑 10 份；跑完再提交下一批也可以
#$ -tc 3                   # 同时最多 3 个 task 运行/预留 → 更易立即调度
#$ -pe thread 32           # 每 task 32 slot
#$ -l h_vmem=4G            # 每 slot 4G → 128G/任务，所有目标节点都绰绰有余
#$ -l h_rt=168:00:00       # 7 天上限（按需调）
#$ -o logs/dfba.$TASK_ID.out
#$ -e logs/dfba.$TASK_ID.err

module load miniconda3
source activate dfba

# ---- 把 0-9999 环境均分为 10 段，每段 1000 ----
ENV_PER_TASK=1000
IDX=$(( SGE_TASK_ID - 1 ))
START=$(( IDX * ENV_PER_TASK ))
STOP=$(( START + ENV_PER_TASK - 1 ))
[ $STOP -gt 9999 ] && STOP=9999

echo "Task $SGE_TASK_ID handles env $START – $STOP (32 threads, multi-queue)"

# chunk × 4(bs=2..5)=32，与 -pe thread 32 匹配
python run.py \
  --start_env $START \
  --stop_env  $STOP \
  --chunk     8 \
  --timeout   5400
