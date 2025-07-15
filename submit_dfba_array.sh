#!/bin/bash
#$ -S /bin/bash
#$ -N dfba_arr               # 作业名
#$ -q long.q                 # 挑选支持128线程的 long.q
#$ -cwd                      # 在当前目录运行
#$ -t 0-99                   # 100 个 array task
#$ -pe thread 128            # 每个 Task 占满整台节点的128个线程
#$ -l h_vmem=4G              # 每 slot 4 GB，总共 128×4 GB = 512 GB
#$ -l h_rt=168:00:00         # 如果需要可把单 task 最长时间调成队列上限（比如 7 天）
#$ -o logs/dfba.$TASK_ID.out # 标准输出
#$ -e logs/dfba.$TASK_ID.err # 标准错误

module load miniconda3
source activate dfba

# 每个 Task 负责 100 个环境
ENV_PER_TASK=100
START=$(( SGE_TASK_ID * ENV_PER_TASK ))
STOP=$(( START + ENV_PER_TASK - 1 ))
[ $STOP -gt 9999 ] && STOP=9999

echo "Task $SGE_TASK_ID handles env $START – $STOP"

# 让 run.py 在每批并行 32 个 env、共 32×4=128 进程
python run.py \
  --start_env $START \
  --stop_env  $STOP  \
  --chunk     32      \
  --timeout   3600    # 可根据单批耗时再调大
