#!/usr/bin/bash

envs=(
    "HalfCheetah-v3"
    "Ant-v3"
    #"LunarLanderContinuous-v2"
    #"BipedalWalker-v3"
    #"BipedalWalkerHardcore-v3"

)

algs=(
    "independent_learning"
    "fedavg"
    "vanila_swarm_learning"
)

comm_rounds=(
    100
    250
    500

)

num_iterations=(
    5000
    2000
    1000
)

drl_algos=(
    #"DDPG"
    #"SAC"
    "TD3"
)

for ((i=0;i<1;i+=1))
do
    for env in ${envs[*]}
    do
        for alg in ${algs[*]}
        do  
            for j in "${!comm_rounds[@]}"
            do
                python main_drl_stable.py --env_name=${env} --alg=${alg} --drl_algo="TD3" --ratio_update 0.5 --comm_round=${comm_rounds[j]} --num_iterations=${num_iterations[j]}
            done
        done
    done
done
