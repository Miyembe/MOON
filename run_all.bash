#!/usr/bin/bash

python3 main.py --dataset=cifar10     --model=simple-cnn     --alg=moon     --lr=0.01     --mu=5     --epochs=10     --comm_round=10     --n_parties=10     --partition=noniid     --beta=0.5     --logdir='./logs/'
python3 main.py --dataset=cifar10     --model=simple-cnn     --alg=fedavg     --lr=0.01     --mu=5     --epochs=10     --comm_round=10     --n_parties=10     --partition=noniid     --beta=0.5     --logdir='./logs/'
python3 main.py --dataset=cifar10     --model=simple-cnn     --alg=fedprox     --lr=0.01     --mu=5     --epochs=10     --comm_round=10     --n_parties=10     --partition=noniid     --beta=0.5     --logdir='./logs/'
python3 main.py --dataset=cifar10     --model=simple-cnn     --alg=local_training     --lr=0.01     --mu=5     --epochs=30     --comm_round=100     --n_parties=10     --partition=noniid     --beta=0.5     --logdir='./logs/'

