#!/bin/bash
partition=$1
job_name=$2
train_gpu=$3
num_node=$4
options=$5
total_process=$((train_gpu*num_node))

mkdir -p log

port=$(( $RANDOM % 300 + 23450 ))

GLOG_vmodule=MemcachedClient=-1 \
srun --partition=$partition \
--mpi=pmi2 -n$total_process \
--gres=gpu:$train_gpu \
--ntasks-per-node=$train_gpu \
--job-name=$job_name \
--kill-on-bad-exit=1 \
--cpus-per-task=7 \
-x BJ-IDC1-10-10-16-[56,83,87,88] \
python -u tools/agent_run.py $options
