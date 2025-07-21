#!/bin/bash
#$ -S /bin/bash
#$ -N dfba100          # 作业名
#$ -cwd                # 在提交目录执行
#$ -V                  # 继承环境
#$ -q short.q          # ←★ 如需改成 long.q/migale.q 请替换
#$ -pe thread 16       # 4 env × 4 bs = 16 进程
#$ -l h_vmem=4G        # 每 slot 4 GB → 总 64 GB
#$ -l h_rt=12:00:00    # 最长 12 小时
#$ -o logs/$JOB_NAME.$TASK_ID.out
#$ -e logs/$JOB_NAME.$TASK_ID.err
#$ -t 1-100            # 0–9999 共 100 个批次（每 100 env 一批）

## ---------- 用户自定义区 ----------
WORKROOT=/home/chlu/work
ENVBASE=$WORKROOT/conda_envs/base
TOTAL_ENVS=10000          # 0–9999
CHUNK_ENV=100             # 每任务跑 100 env
PARA_ENV=4                # run.py --chunk
TIMEOUT=1200              # run.py --timeout
## ----------------------------------

START_ENV=$(( (SGE_TASK_ID-1)*CHUNK_ENV ))
STOP_ENV=$(( START_ENV+CHUNK_ENV-1 ))
if [ $STOP_ENV -ge $((TOTAL_ENVS-1)) ]; then
    STOP_ENV=$((TOTAL_ENVS-1))
fi

echo "[`date`] TASK $SGE_TASK_ID → env $START_ENV‒$STOP_ENV"

# 1) 激活 conda
source "$ENVBASE/../../etc/profile.d/conda.sh"
conda activate "$ENVBASE"

# 2) 让 glibc 主动归还未用内存，缓解碎片
export MALLOC_TRIM_THRESHOLD_=1   # :contentReference[oaicite:2]{index=2}

# 3) 进入项目目录并运行
cd "$WORKROOT/content"
python run.py \
   --start_env "$START_ENV" \
   --stop_env  "$STOP_ENV"  \
   --chunk     "$PARA_ENV"  \
   --timeout   "$TIMEOUT"

EXIT=$?
conda deactivate
echo "[`date`] TASK $SGE_TASK_ID finished (exit $EXIT)"
exit $EXIT
