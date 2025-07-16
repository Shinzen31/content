#!/bin/bash
#
# run_parallel8.sh
# 启动 8 个并行 dFBA 区间任务；每任务独立目录，避免输出冲突；
# nohup 后台运行，可断开 SSH；日志分别记录。

### ==== 用户可配置区 ==== ###
ENVBASE=/home/chlu/work/conda_envs/base         # 解包后的conda环境
WORKDIR=/home/chlu/work/content                 # 有 run.py / dFBA.py / 数据的目录
TOTAL_ENV=10000                                 # env_id 范围：0..TOTAL_ENV-1
TASKS=8                                         # 并发任务数
TIMEOUT=5400                                    # run.py 的 --timeout
### ================================= ###

PARLOG=$WORKDIR/logs_parallel8                  # 主日志目录
OUT_BASE=$WORKDIR/task_outputs8                 # 每任务工作区根目录
mkdir -p "$PARLOG" "$OUT_BASE"

# 使用解包环境
export PATH=$ENVBASE/bin:$PATH
export LD_LIBRARY_PATH=$ENVBASE/lib:$LD_LIBRARY_PATH

# 快测环境
python - <<'EOF' || { echo "[FATAL] Conda env import失败" >&2; exit 1; }
import cobra
print("cobra OK:", cobra.__version__)
EOF

# 按 TASKS 分段（最后一段吃余数）
BASE_SIZE=$(( TOTAL_ENV / TASKS ))      # 整除部分
REM=$(( TOTAL_ENV % TASKS ))            # 余数

for (( i=0; i<TASKS; i++ )); do
  EXTRA=0
  # 把余数平均分配到前 REM 个任务（典型的负载均衡分段）
  if (( i < REM )); then
    EXTRA=1
  fi

  SIZE=$(( BASE_SIZE + EXTRA ))
  # 起止 env
  if (( i == 0 )); then
    START=0
  else
    # 重新累计：也可利用上一次 STOP+1
    # 为清晰起见，先计算累计起点：
    START=0
    for (( j=0; j<i; j++ )); do
      EXTRAJ=$(( j < REM ? 1 : 0 ))
      START=$(( START + BASE_SIZE + EXTRAJ ))
    done
  fi
  STOP=$(( START + SIZE - 1 ))
  if (( STOP >= TOTAL_ENV )); then
    STOP=$(( TOTAL_ENV - 1 ))
  fi

  TASKDIR=$OUT_BASE/task${i}
  mkdir -p "$TASKDIR"

  LOG_OUT=$PARLOG/task${i}.launcher.out
  LOG_ERR=$PARLOG/task${i}.launcher.err

  echo "[$(date)] Launch task $i env ${START}-${STOP} -> $TASKDIR" >"$LOG_OUT"

  # 后台启动：cd 到独立目录，运行 run.py；stdout/stderr 各记录
  nohup bash -c "
    cd $TASKDIR
    /usr/bin/time -v $ENVBASE/bin/python $WORKDIR/run.py \
      --start_env $START \
      --stop_env  $STOP \
      --chunk     4 \
      --timeout   $TIMEOUT \
      > stdout.log 2> stderr.log
  " >>"$LOG_OUT" 2>>"$LOG_ERR" &

  PID=$!
  echo "  PID $PID" >>"$LOG_OUT"
  # 解除与当前shell关联
  disown %$PID
done

echo "[$(date)] All $TASKS tasks launched. Logs: $PARLOG  Outputs: $OUT_BASE"
