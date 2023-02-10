#!/usr/bin/bash

envs=(
    "LunarLanderContinuous-v2"
    "BipedalWalker-v3"
    "BipedalWalkerHardcore-v3"
)

algs=(
    "indepedent_learning"
    "fedavg"
    "vanila_swarm_learning"
)

for ((i=0;i<1;i+=1))
do
    for env in ${envs[*]}
    do
        for alg in ${algs[*]}
        do  python main_drl_stable.py --env_name=${env} --alg=${alg} --ratio_update 0.5 --comm_round 50 --num_iterations 1000
        
        done
    done
done
