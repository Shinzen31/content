#!/bin/bash
#
# run_parallel8_daemon.sh
# 8 并发 dFBA 区间任务；每任务独立子目录；nohup+disown 自我守护；
# 断开 SSH 后继续运行；路径全部基于 /home/chlu/work/* 。

###############################################################################
# 自我守护：若首次调用（尚未在 nohup 下），重新以 nohup+disown 后台启动自身并退出前台进程
###############################################################################
if [[ -z "$NOHUP_STARTED" ]]; then
  export NOHUP_STARTED=1
  mkdir -p logs_parallel8   # 临时目录（前台阶段也可记录）
  echo "[INFO] Relaunching under nohup..." > logs_parallel8/master.out
  nohup bash "$0" "$@" >> logs_parallel8/master.out 2>&1 &
  disown
  echo "[INFO] Daemonized. Master log: logs_parallel8/master.out"
  exit 0
fi

###############################################################################
# 基础路径（全部从 /home/chlu/work 开始）
###############################################################################
WORKROOT=/home/chlu/work
ENVBASE=$WORKROOT/conda_envs/base          # 解包后的 Conda 环境前缀
WORKDIR=$WORKROOT/content                  # run.py / dFBA.py / 数据
PARLOG=$WORKDIR/logs_parallel8             # 各并发任务启动日志
OUT_BASE=$WORKDIR/task_outputs8            # 各并发任务工作区根

mkdir -p "$PARLOG" "$OUT_BASE"

###############################################################################
# 任务划分参数
###############################################################################
TOTAL_ENV=10000        # env_id 范围：0..9999
TASKS=8                # 并发块数
TIMEOUT=5400           # run.py --timeout
CHUNK=4                # run.py --chunk

###############################################################################
# 环境：使用解包 Conda（无需 module load / conda activate）
###############################################################################
export PATH="$ENVBASE/bin:$PATH"
export LD_LIBRARY_PATH="$ENVBASE/lib:$LD_LIBRARY_PATH"

# 快速验证 cobra
python - <<'EOF' || { echo "[FATAL] Conda env import 失败" >&2; exit 1; }
import cobra
print("cobra OK:", cobra.__version__)
EOF

###############################################################################
# 计算每块起止 env（负载均衡：前 REM 块 +1）
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
    echo "  Workdir: $TASKDIR"
    echo "  Logs   : $TASKDIR/stdout.log , $TASKDIR/stderr.log"
  } >"$LOG_OUT"

  # 后台启动：切换到该任务独立目录；run.py 写出的 results/* 都隔离在此
  nohup bash -c "
    cd $TASKDIR
    /usr/bin/time -v $ENVBASE/bin/python $WORKDIR/run.py \
      --start_env $START \
      --stop_env  $STOP \
      --chunk     $CHUNK \
      --timeout   $TIMEOUT \
      > stdout.log 2> stderr.log
  " >>"$LOG_OUT" 2>>"$LOG_ERR" &

  PID=$!
  echo "  PID $PID" >>"$LOG_OUT"
  disown %$PID

  START=$(( STOP + 1 ))
done

echo "[$(date)] All $TASKS tasks launched. Master logs: $PARLOG ; outputs: $OUT_BASE" >> "$PARLOG/master.out"
echo "All tasks launched."
