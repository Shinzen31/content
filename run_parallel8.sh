#!/bin/bash
#
# run_parallel8.sh
# 8 并发 dFBA 区间任务；每任务独立子目录；自动 nohup 守护；
# 为子任务目录建立脚本/数据符号链接，避免相对路径问题。

###############################################################################
# 自守护：如非 nohup 环境，重启自身到后台并退出前台
###############################################################################
if [[ -z "$NOHUP_STARTED" ]]; then
  export NOHUP_STARTED=1
  mkdir -p logs_parallel8
  echo "[INFO] Relaunching under nohup..." > logs_parallel8/master.out
  nohup bash "$0" "$@" >> logs_parallel8/master.out 2>&1 &
  echo "[INFO] Daemonized. Master log: logs_parallel8/master.out"
  exit 0
fi

###############################################################################
# 基础路径
###############################################################################
WORKROOT=/home/chlu/work
ENVBASE=$WORKROOT/conda_envs/base
WORKDIR=$WORKROOT/content
PARLOG=$WORKDIR/logs_parallel8
OUT_BASE=$WORKDIR/task_outputs8

mkdir -p "$PARLOG" "$OUT_BASE"

###############################################################################
# 任务划分参数
###############################################################################
TOTAL_ENV=10000
TASKS=8
TIMEOUT=5400
CHUNK=4

###############################################################################
# 环境
###############################################################################
export PATH="$ENVBASE/bin:$PATH"
export LD_LIBRARY_PATH="$ENVBASE/lib:$LD_LIBRARY_PATH"

python - <<'EOF' || { echo "[FATAL] Conda env import失败" >&2; exit 1; }
import cobra
print("cobra OK:", cobra.__version__)
EOF

###############################################################################
# 分段
###############################################################################
BASE_SIZE=$(( TOTAL_ENV / TASKS ))
REM=$(( TOTAL_ENV % TASKS ))

START=0
for (( i=0; i<TASKS; i++ )); do
  EXTRA=$(( i < REM ? 1 : 0 ))
  SIZE=$(( BASE_SIZE + EXTRA ))
  STOP=$(( START + SIZE - 1 ))
  (( STOP >= TOTAL_ENV )) && STOP=$(( TOTAL_ENV - 1 ))

  TASKDIR=$OUT_BASE/task${i}
  mkdir -p "$TASKDIR"

  LOG_OUT=$PARLOG/task${i}.launcher.out
  LOG_ERR=$PARLOG/task${i}.launcher.err

  {
    echo "[$(date)] Launch task $i env ${START}-${STOP}"
    echo "  WORKDIR: $TASKDIR"
    echo "  Logs   : stdout.log, stderr.log"
  } >"$LOG_OUT"

  # 注意：我们在任务目录里创建脚本/数据链接，解决 run.py 子进程找不到 dFBA.py 问题
  nohup bash -c "
    cd $TASKDIR
    ln -sf $WORKDIR/dFBA.py dFBA.py
    ln -sf $WORKDIR/run.py run.py
    ln -sf $WORKDIR/environment_ball.tsv environment_ball.tsv
    ln -sf $WORKDIR/models_gapfilled models_gapfilled

    /usr/bin/time -v $ENVBASE/bin/python $WORKDIR/run.py \
      --start_env $START \
      --stop_env  $STOP \
      --chunk     $CHUNK \
      --timeout   $TIMEOUT \
      > stdout.log 2> stderr.log
  " >>"$LOG_OUT" 2>>"$LOG_ERR" &

  PID=$!
  echo "  PID $PID" >>"$LOG_OUT"

  START=$(( STOP + 1 ))
done

{
  echo "[$(date)] All $TASKS tasks launched."
  echo "Master log: $PARLOG/master.out"
  echo "Outputs: $OUT_BASE/task0 ... task$((TASKS-1))"
} >> "$PARLOG/master.out"

echo "All tasks launched. Logs in $PARLOG, outputs in $OUT_BASE"
