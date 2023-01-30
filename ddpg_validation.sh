#!/usr/bin/bash


env_name=(
    'cartpole'
    'cartpole'
    'hopper'
    'hopper'
)

task_name=(
    'balance'
    'swingup'
    'hop'
    'stand'
)

network_size=(
    64
    128
    256
)


for ((i=0;i<3;i+=1))
do
    for j in "${!env_name[@]}"
    do
        for num_param in ${network_size[*]}
        do  python main_drl_simple.py \
            --env_name=${env_name[j]} \
            --task_name=${task_name[j]} \
            --hidden1=$num_param \
            --hidden2=$num_param \
            --seed=$(expr $i + 100)
            
        done
    done
done