#!/bin/bash
#$ -S /bin/bash
#$ -N dfba_arr               # 作业名
#$ -q long.q                 # 选择支持大并发的队列
#$ -cwd                      # 在当前目录运行
#$ -t 1-100                  # 100 个 array task；ID = 1…100
#$ -pe thread 128            # 每 task 占满整机 128 线程
#$ -l h_vmem=4G              # 每 slot 4G，总共 512G，节点够用
#$ -l h_rt=168:00:00         # 单 task 最长 168 小时（可按需调整）
#$ -o logs/dfba.$TASK_ID.out
#$ -e logs/dfba.$TASK_ID.err

module load miniconda3
source activate dfba

# 每个 task 处理的 env 数
ENV_PER_TASK=100

# 由于 SGE_TASK_ID 从 1 开始，减 1 得到 0-based idx
IDX=$(( SGE_TASK_ID - 1 ))
START=$(( IDX * ENV_PER_TASK ))
STOP=$(( START + ENV_PER_TASK - 1 ))
# 最后一块不要超过 9999
if [ $STOP -gt 9999 ]; then
  STOP=9999
fi

echo "Task $SGE_TASK_ID (idx=$IDX) handles env $START – $STOP"

# 调用 run.py
python run.py \
  --start_env $START \
  --stop_env  $STOP  \
  --chunk     32      \
  --timeout   3600
