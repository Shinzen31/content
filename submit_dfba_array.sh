#!/bin/bash
#$ -S /bin/bash
#$ -N dfba_arr_long12       # 作业名
#$ -q long.q                # 只用有权限的 long.q
#$ -cwd                     # 在当前目录运行
#$ -t 1-10                  # 共 10 个数组任务
#$ -tc 2                    # 同时最多 4 个 task → 分在 n123,n124,n126,n127
#$ -pe thread 12            # 每 task 12 slot
#$ -l h_vmem=4G             # 每 slot 4G → 48G/任务
#$ -l h_rt=168:00:00        # 最长 7 天

# 确保日志目录（处理可能的 Windows 回车目录名）
mkdir -p logs

module load miniconda3
source activate dfba

# 计算要处理的环境区间
ENV_PER_TASK=1000
IDX=$((SGE_TASK_ID - 1))
START=$((IDX * ENV_PER_TASK))
STOP=$((START + ENV_PER_TASK - 1))
[ $STOP -gt 9999 ] && STOP=9999

LOG_OUT="logs/dfba12_tc4.${JOB_ID}.${SGE_TASK_ID}.out"
LOG_ERR="logs/dfba12_tc4.${JOB_ID}.${SGE_TASK_ID}.err"

echo "[$(date)] Task $SGE_TASK_ID env $START–$STOP (12 slots × tc4)" >"$LOG_OUT"

# chunk=3 → 3 env/batch ×4(bs=2..5)=12 并行子进程
python run.py \
  --start_env "$START" \
  --stop_env  "$STOP" \
  --chunk     6 \
  --timeout   5400 \
  >>"$LOG_OUT" 2>>"$LOG_ERR"
