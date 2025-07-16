#!/bin/bash
#$ -S /bin/bash
#$ -N dfba_arr_c4_4task   # 作业名
#$ -q long.q
#$ -cwd
#$ -t 1-4                 # 四个 array task
#$ -tc 4                  # 最多同时运行 4 个
#$ -pe thread 16          # 并行 16 核 → chunk=4 的 16 个子进程
#$ -l h_vmem=2G           # 每 slot 2 GB → 总 32 GB
#$ -l h_rt=168:00:00      # 最长 7 天

mkdir -p logs

# —— 指定你 unpack 后的 Conda 环境前缀 —— 
ENVBASE=/home/chlu/work/conda_envs/base

# （可选）把环境的 bin 放到 PATH，方便后续调用
export PATH=$ENVBASE/bin:$PATH
export LD_LIBRARY_PATH=$ENVBASE/lib:$LD_LIBRARY_PATH

# 验证一下 Python 和 cobra
$ENVBASE/bin/python - <<'EOF' || { echo "Conda env check failed" >&2; exit 1; }
import cobra
print("cobra", cobra.__version__, "— OK")
EOF

# 计算环境段
ENV_PER_TASK=2500
IDX=$((SGE_TASK_ID - 1))
START=$((IDX * ENV_PER_TASK))
STOP=$((START + ENV_PER_TASK - 1))
[ $STOP -gt 9999 ] && STOP=9999

LOG_OUT="logs/dfba_c4_4task.${JOB_ID}.${SGE_TASK_ID}.out"
LOG_ERR="logs/dfba_c4_4task.${JOB_ID}.${SGE_TASK_ID}.err"
echo "[$(date)] Task $SGE_TASK_ID env $START–$STOP (chunk=4, 16 slots)" >"$LOG_OUT"

# 最终执行：用指向解包环境的 python
/usr/bin/time -v $ENVBASE/bin/python ~/work/content/run.py \
  --start_env "$START" \
  --stop_env  "$STOP" \
  --chunk     4 \
  --timeout   5400 \
  >>"$LOG_OUT" 2>>"$LOG_ERR"
