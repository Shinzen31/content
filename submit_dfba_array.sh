#!/bin/bash
#$ -S /bin/bash
#$ -N dfba_test
#$ -q long.q
#$ -cwd
#$ -pe thread 1          # 只要 1 slot
#$ -l h_vmem=4G
#$ -l h_rt=24:00:00
#$ -o logs/dfba_test.out
#$ -e logs/dfba_test.err

module load miniconda3
source activate dfba

python dFBA.py --env 0 --bs 3
