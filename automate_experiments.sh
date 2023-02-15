#!/usr/bin/bash

envs=(
    "LunarLanderContinuous-v2"
    "BipedalWalker-v3"
    "BipedalWalkerHardcore-v3"
)

algs=(
    "independent_learning"
    #"fedavg"
    #"vanila_swarm_learning"
)

comm_rounds=(
    1
    10
    25

)

num_iterations=(
    100000
    10000
    4000
)

drl_algos=(
    "DDPG"
    "SAC"
    "TD3"
)

for ((i=0;i<1;i+=1))
do
    for env in ${envs[*]}
    do
        for drl_algo in ${drl_algos[*]}
        do  
            for j in "${!comm_rounds[@]}"
            do
                python main_drl_stable.py --env_name=${env} --alg="independent_learning" --drl_algo=${drl_algo} --ratio_update 0.5 --comm_round=${comm_rounds[j]} --num_iterations=${num_iterations[j]}
            done
        done
    done
done
