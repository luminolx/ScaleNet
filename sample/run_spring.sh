#!/bin/bash
job_name=$1
train_gpu=$2
num_node=$3
options=$4
total_process=$((train_gpu*num_node))

mkdir -p log

port=$(( $RANDOM % 300 + 23450 ))

while true
do
    GLOG_vmodule=MemcachedClient=-1 \
    srun -p spring_scheduler \
    --comment=spring-submit \
    --mpi=pmi2 -n$total_process \
    --gres=gpu:$train_gpu \
    --ntasks-per-node=$train_gpu \
    --job-name=$job_name \
    --kill-on-bad-exit=1 \
    --cpus-per-task=7 \
    -x BJ-IDC1-10-10-16-[30,42,80] \
    python -u tools/agent_run.py $options  2>&1 |tee -a log/$job_name.log
    
    if grep "[END] Finish sampling." log/$job_name.log
    then
        echo "done"
        break
    fi
done

# GLOG_vmodule=MemcachedClient=-1 \
# srun -p spring_scheduler \
# --comment=spring-submit \
# --mpi=pmi2 -n$total_process \
# --gres=gpu:$train_gpu \
# --ntasks-per-node=$train_gpu \
# --job-name=$job_name \
# --kill-on-bad-exit=1 \
# --cpus-per-task=7 \
# python -u tools/agent_run.py $options
