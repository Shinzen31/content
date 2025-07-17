#!/bin/bash
#
# run_parallel8.sh
# 8 并发 dFBA 区间任务；每任务独立子目录；自动 nohup 守护；
# 为子任务目录建立脚本/数据符号链接；强制无缓冲输出便于 tail -f 监控。
#
# 使用步骤：
#   cd /home/chlu/work/content
#   chmod +x run_parallel8.sh
#   ./run_parallel8.sh          # 前台立刻返回；后台运行
#

###############################################################################
# 自守护：若非 nohup 环境则重启自身到后台并退出前台
###############################################################################
if [[ -z "${NOHUP_STARTED:-}" ]]; then
  export NOHUP_STARTED=1
  mkdir -p logs_parallel8
  echo "[INFO] Relaunching under nohup..." > logs_parallel8/master.out
  # 将所有参数透传
  nohup bash "$0" "$@" >> logs_parallel8/master.out 2>&1 &
  echo "[INFO] Daemonized. Master log: logs_parallel8/master.out"
  exit 0
fi

set -euo pipefail

###############################################################################
# 基础路径（按需修改）
###############################################################################
WORKROOT=/home/chlu/work
ENVBASE=$WORKROOT/conda_envs/base
WORKDIR=$WORKROOT/content
PARLOG=$WORKDIR/logs_parallel8
OUT_BASE=$WORKDIR/task_outputs8

mkdir -p "$PARLOG" "$OUT_BASE"

###############################################################################
# Conda 环境检查（一次性；失败立即退出）
###############################################################################
echo "[INFO] Checking cobra import from $ENVBASE..." | tee -a "$PARLOG/master.out"
if ! "$ENVBASE/bin/python" - <<'EOF' 2>>"$PARLOG/master.out" >>"$PARLOG/master.out"; then
import cobra
print("cobra OK:", cobra.__version__)
EOF
then
  echo "[FATAL] Conda env import失败" | tee -a "$PARLOG/master.out"
  exit 1
fi

###############################################################################
# 自动检测 environment_ball.tsv 的 env 数目
# 兼容含 env_id 列或按块分割（与 dFBA.py 相同逻辑）
###############################################################################
ENV_TSV="$WORKDIR/environment_ball.tsv"
if [[ ! -f "$ENV_TSV" ]]; then
  echo "[FATAL] Not found: $ENV_TSV" | tee -a "$PARLOG/master.out"
  exit 1
fi

TOTAL_ENV=$("$ENVBASE/bin/python" - <<EOF)
import pandas as pd, sys
df = pd.read_csv("$ENV_TSV", sep="\t")
if "env_id" in df.columns:
    print(df["env_id"].nunique())
else:
    n_met = df["metabolite_id"].nunique()
    print(len(df) // n_met)
EOF
)
echo "[INFO] Detected $TOTAL_ENV environments (env_id 0..$((TOTAL_ENV-1)))." | tee -a "$PARLOG/master.out"

###############################################################################
# 任务参数（可根据需要覆写环境变量）
###############################################################################
TASKS=${TASKS:-8}           # 并发数量
TIMEOUT=${TIMEOUT:-5400}    # run.py 子任务 --timeout
CHUNK=${CHUNK:-4}           # run.py 子任务 --chunk

# 可通过环境变量 START_ENV / STOP_ENV 覆写范围；默认全范围
START_ENV=${START_ENV:-0}
STOP_ENV=${STOP_ENV:-$((TOTAL_ENV-1))}

echo "[INFO] Master will distribute env_id $START_ENV..$STOP_ENV across $TASKS tasks." \
  | tee -a "$PARLOG/master.out"
echo "[INFO] Per-run.py chunk=$CHUNK timeout=$TIMEOUT sec." \
  | tee -a "$PARLOG/master.out"

###############################################################################
# 计算每个任务的分配区间（尽量均匀）
###############################################################################
TOTAL_RANGE=$(( STOP_ENV - START_ENV + 1 ))
BASE_SIZE=$(( TOTAL_RANGE / TASKS ))
REM=$(( TOTAL_RANGE % TASKS ))

###############################################################################
# 启动 TASKS 个后台任务
###############################################################################
pids=()

for (( i=0, start=$START_ENV; i<TASKS; i++ )); do
  extra=$(( i < REM ? 1 : 0 ))
  size=$(( BASE_SIZE + extra ))
  stop=$(( start + size - 1 ))
  (( stop > STOP_ENV )) && stop=$STOP_ENV

  TASKDIR=$OUT_BASE/task${i}
  mkdir -p "$TASKDIR"

  LOG_OUT=$PARLOG/task${i}.launcher.out
  LOG_ERR=$PARLOG/task${i}.launcher.err

  {
    echo "[$(date)] Launch task $i env ${start}-${stop}"
    echo "  WORKDIR: $TASKDIR"
    echo "  Logs   : stdout.log, stderr.log"
  } >"$LOG_OUT"

  # 在子 shell 中运行任务（不再二次 nohup；主进程本身已 nohup）
  (
    set -euo pipefail
    cd "$TASKDIR"

    # 建立脚本 / 数据符号链接（覆盖旧链接）
    ln -sfn "$WORKDIR/dFBA.py"              dFBA.py
    ln -sfn "$WORKDIR/run.py"               run.py
    ln -sfn "$WORKDIR/environment_ball.tsv" environment_ball.tsv
    ln -sfn "$WORKDIR/models_gapfilled"     models_gapfilled

    # 无缓冲输出：python -u；stdbuf用于 /usr/bin/time 产生输出行缓冲
    stdbuf -oL -eL /usr/bin/time -v "$ENVBASE/bin/python" -u ./run.py \
      --start_env "$start" \
      --stop_env  "$stop" \
      --chunk     "$CHUNK" \
      --timeout   "$TIMEOUT" \
      > stdout.log 2> stderr.log
  ) >>"$LOG_OUT" 2>>"$LOG_ERR" &

  pid=$!
  echo "  PID $pid" >>"$LOG_OUT"
  pids+=("$pid")

  start=$(( stop + 1 ))
  (( start > STOP_ENV )) && break
done

###############################################################################
# 打印 PID 列表到 master log 并退出（任务后台持续）
###############################################################################
{
  echo "[$(date)] All $TASKS tasks launched (some may have empty ranges if STOP_ENV small)."
  echo "Master log: $PARLOG/master.out"
  echo "PIDs: ${pids[*]}"
  echo "Tail individual task launchers in $PARLOG/taskN.launcher.out and .err."
  echo "Tail runtime logs in $OUT_BASE/taskN/stdout.log (and stderr.log)."
} >> "$PARLOG/master.out"

echo "All tasks launched. Logs in $PARLOG, outputs in $OUT_BASE"
