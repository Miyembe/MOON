import numpy as np
import json
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import matplotlib
import matplotlib.pyplot as plt
import time
import dmc2gym
import gym
import cv2
from PIL import Image
import imageio
import csv


from copy import deepcopy

from stable_baselines3 import DDPG
#from ddpg.ddpg import DDPG
from ddpg.evaluator import Evaluator
from collections import OrderedDict

from dm_control import suite


# from model import *
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="resnet50", help="neural network used in training"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar100", help="dataset used for training"
    )
    parser.add_argument("--net_config", type=lambda x: list(map(int, x.split(", "))))
    parser.add_argument(
        "--partition", type=str, default="homo", help="the data partitioning strategy"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="learning rate (default: 0.1)"
    )
    parser.add_argument("--epochs", type=int, default=5, help="number of local epochs")
    parser.add_argument(
        "--n_parties",
        type=int,
        default=6,
        help="number of workers in a distributed cluster",
    )
    parser.add_argument(
        "--alg",
        type=str,
        default="fedavg",
        help="communication strategy: fedavg/fedprox",
    )
    parser.add_argument(
        "--ratio_update",
        type=float,
        default=0.5,
        help="the ratio of updating the model parameters",
    )
    parser.add_argument(
        "--comm_round", type=int, default=50, help="number of maximum communication roun"
    )
    parser.add_argument("--seed", type=int, default=142, help="Random seed")
    parser.add_argument(
        "--dropout_p",
        type=float,
        required=False,
        default=0.0,
        help="Dropout probability. Default=0.0",
    )
    parser.add_argument(
        "--datadir", type=str, required=False, default="./data/", help="Data directory"
    )
    parser.add_argument(
        "--reg", type=float, default=1e-5, help="L2 regularization strength"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=False,
        default="./logs/",
        help="Log directory path",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=False,
        default="./output/",
        help="Log directory path",
    )
    parser.add_argument(
        "--modeldir",
        type=str,
        required=False,
        default="./models/",
        help="Model directory path",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="The parameter for the dirichlet distribution for data partitioning",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="The device to run the program"
    )
    parser.add_argument(
        "--log_file_name", type=str, default=None, help="The log file name"
    )
    parser.add_argument("--optimizer", type=str, default="sgd", help="the optimizer")
    parser.add_argument(
        "--mu", type=float, default=1, help="the mu parameter for fedprox or moon"
    )
    parser.add_argument(
        "--out_dim",
        type=int,
        default=256,
        help="the output dimension for the projection layer",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="the temperature parameter for contrastive loss",
    )
    parser.add_argument(
        "--local_max_epoch",
        type=int,
        default=100,
        help="the number of epoch for local optimal training",
    )
    parser.add_argument(
        "--model_buffer_size",
        type=int,
        default=1,
        help="store how many previous models for contrastive loss",
    )
    parser.add_argument("--pool_option", type=str, default="FIFO", help="FIFO or BOX")
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="how many clients are sampled in each round",
    )
    parser.add_argument(
        "--load_model_file",
        type=str,
        default=None,
        help="the model to load as global model",
    )
    parser.add_argument(
        "--load_pool_file",
        type=str,
        default=None,
        help="the old model pool path to load",
    )
    parser.add_argument(
        "--load_model_round",
        type=int,
        default=None,
        help="how many rounds have executed for the loaded model",
    )
    parser.add_argument(
        "--load_first_net",
        type=int,
        default=1,
        help="whether load the first net as old net or not",
    )
    parser.add_argument(
        "--normal_model",
        type=int,
        default=0,
        help="use normal model or aggregate model",
    )
    parser.add_argument("--loss", type=str, default="contrastive")
    parser.add_argument("--save_model", type=int, default=0)
    parser.add_argument("--use_project_head", type=int, default=1)
    parser.add_argument(
        "--server_momentum", type=float, default=0, help="the server momentum (FedAvgM)"
    )

    # DDPG Related arguments
    parser.add_argument(
        "--mode", default="train", type=str, help="support option: train/test"
    )
    # parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument(
        "--hidden1",
        default=128,
        type=int,
        help="hidden num of first fully connect layer",
    )
    parser.add_argument(
        "--hidden2",
        default=128,
        type=int,
        help="hidden num of second fully connect layer",
    )
    parser.add_argument("--rate", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "--prate",
        default=0.0001,
        type=float,
        help="policy net learning rate (only for DDPG)",
    )
    parser.add_argument(
        "--warmup",
        default=100,
        type=int,
        help="time without training but only filling the replay memory",
    )
    parser.add_argument("--discount", default=0.99, type=float, help="")
    parser.add_argument("--bsize", default=64, type=int, help="minibatch size")
    parser.add_argument("--rmsize", default=6000000, type=int, help="memory size")
    parser.add_argument("--window_length", default=1, type=int, help="")
    parser.add_argument(
        "--tau", default=0.001, type=float, help="moving average for target network"
    )
    parser.add_argument("--ou_theta", default=0.15, type=float, help="noise theta")
    parser.add_argument("--ou_sigma", default=0.2, type=float, help="noise sigma")
    parser.add_argument("--ou_mu", default=0.0, type=float, help="noise mu")
    parser.add_argument(
        "--validate_episodes",
        default=20,
        type=int,
        help="how many episode to perform during validate experiment",
    )
    parser.add_argument("--num_iterations", default=5000, type=int, help="")
    parser.add_argument("--max_episode_length", default=20, type=int, help="")
    parser.add_argument(
        "--validate_steps",
        default=2000,
        type=int,
        help="how many steps to perform a validate experiment",
    )
    parser.add_argument("--output", default="output", type=str, help="")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--init_w", default=0.003, type=float, help="")
    parser.add_argument(
        "--train_iter", default=200000, type=int, help="train iters each timestep"
    )
    parser.add_argument(
        "--epsilon", default=50000, type=int, help="linear decay of exploration policy"
    )
    # parser.add_argument('--seed', default=142, type=int, help='')
    parser.add_argument(
        "--resume", default="default", type=str, help="Resuming model path for testing"
    )
    parser.add_argument(
        "--env_name", default="Pendulum-v1", type=str, help="domain name for dm2gym"
    )
    parser.add_argument(
        "--task_name", default="swingup", type=str, help="task name for the domain"
    )  # If gym is used, use "None" for task_name

    args = parser.parse_args()
    return args


def train_net(
    agent_id,
    agent,
    env,
    num_iterations,
    max_episode_length,
    evaluate,
    validate_steps,
    args,
    round,
    writer,
    logging_path
):
    _, logs = agent.learn(total_timesteps=num_iterations)

    # write logs from agent.learn into global writer for tensorboard.
    for log in logs:
        real_timestep = log['time/total_timesteps'] + round*args.num_iterations
        for key, value in log.items():
            writer.add_scalar(f"agent_{agent_id}/{key}", value, real_timestep)
    agent_parameters = agent.policy.state_dict()
    return agent_parameters


def local_train_net(
    agents,
    envs,
    args,
    writer,
    logging_path,
    global_model=None,
    prev_model_pool=None,
    server_c=None,
    clients_c=None,
    round=None,
    device="cuda:0",
) -> list: # list of parameters [n]: n is the number of agents
    # avg_acc = 0.00
    # acc_list = []
    # if global_model:
    #     global_model.cuda(device)
    # if server_c:
    #     server_c.cuda(device)
    #     server_c_collector = list(server_c.cuda(device).parameters())
    #     new_server_c_collector = copy.deepcopy(server_c_collector)
    # random_state = np.random.RandomState(42)
    list_parameters = []
    for agent_id, agent in enumerate(agents):
        # dataidxs = net_dataidx_map[net_id]

        # logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

        print(f"local_train: agent_id: {agent_id}")
        """
        run the env with n_epochs - another for loop. 
        """
        evaluate = Evaluator(
            args.validate_episodes,
            args.validate_steps,
            args.output,
            max_episode_length=args.max_episode_length,
        )
        # 20230106 Define DDPG agent
        agent_parameters = train_net(
            agent_id,
            agent,
            envs[agent_id],
            args.num_iterations,
            args.max_episode_length,
            evaluate,
            args.validate_steps,
            args,
            round,
            writer,
            logging_path
        )
        list_parameters.append(agent_parameters)

        # logger.info("net %d final test acc %f" % (net_id, testacc))
        # avg_acc += testacc
        # acc_list.append(testacc)
    # avg_acc /= args.n_parties
    # if global_model:
    #     global_model.to('cuda:0')
    # if server_c:
    #     for param_index, param in enumerate(server_c.parameters()):
    #         server_c_collector[param_index] = new_server_c_collector[param_index]
    #     server_c.to('cuda:0')

    return list_parameters


def display_frame(frame):
    height, width, _ = frame.shape
    dpi = 70
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frame)
    plt.show()
    time.sleep(0.1)
    plt.close()


def init_envs(n_envs, env_name, task_name=None):
    envs = []
    for i in range(n_envs):
        if task_name == None:
            print(f"env_name: {env_name}, seed: {args.seed}")
            env = gym.make(id=env_name)
            # frame = env.physics.render()
            # print(f"frame: {frame}")
            # display_frame(frame)
            envs.append(env)
        else:
            env = dmc2gym.make(
                domain_name=env_name, task_name=task_name, seed=args.seed
            )
            envs.append(env)
    return envs


def init_agents(n_agents, envs, logging_path, algo_name="DDPG"):
    agents = []
    for i in range(n_agents):
        n_states = envs[i].observation_space.shape[0]
        n_actions = envs[i].action_space.shape[0]
        if algo_name == "DDPG":
            agent = DDPG("MlpPolicy", envs[i], verbose=1)
        # elif algo_name == "DQN":
        #    agent = DQN(n_states, n_actions, args)
        agents.append(agent)

    return agents


def write_video_PIL(frames, file_name, fps=30):

    imageio.mimwrite(uri=file_name, ims=frames, fps=fps, format=".gif")


def write_frame_number(frame, text):
    frame = cv2.UMat(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    font_color = (0, 0, 255)  # BGR format for red color

    # Define the text to be added and its position
    text_position = (2, 10)

    # Add the text to the image
    cv2.putText(
        frame, text, text_position, font, font_scale, font_color, font_thickness
    )

    new_frame = frame.get()
    return new_frame


def evaluate_agents(
    agents, envs, args, file_path, max_episode_step=200, num_episodes=5, device="cuda:0"
):

    # 0. Initisalise lists to contain
    frames = [[] for _ in range(len(agents))]
    avg_rewards = [[] for _ in range(len(agents))]
    avg_steps = [[] for _ in range(len(agents))]
    std_rewards = [[] for _ in range(len(agents))]
    std_steps = [[] for _ in range(len(agents))]

    # 1. Run agents to collect samples and rewards
    for i, agent in enumerate(agents):
        rewards = []
        steps = []
        for n in range(num_episodes):
            observation = envs[i].reset()
            agent.reset(observation)
            done = False
            step = 0
            episode_reward = 0
            while done == False:
                # observation....
                action = agent.select_action(observation)

                # env response with next_observation, reward, terminate_info
                observation2, reward, done, info = envs[i].step(action)
                observation2 = deepcopy(observation2)
                if step >= max_episode_step - 1:
                    done = True

                frame = envs[i].render()
                frame = write_frame_number(frame, f"Episode: {n}")
                frames[i].append(frame)

                step += 1
                episode_reward += reward
                observation = deepcopy(observation2)

                if done:  # end of episode
                    if args.debug:
                        prGreen(
                            "#{}: episode_reward:{} steps:{}".format(
                                episode, episode_reward, step
                            )
                        )
                    rewards.append(episode_reward)
                    steps.append(step)

        avg_reward = np.average(rewards)
        std_reward = np.std(rewards)
        avg_rewards[i].append(avg_reward)
        std_rewards[i].append(std_reward)

        avg_step = np.average(steps)
        std_step = np.std(steps)
        avg_steps[i].append(avg_step)
        std_steps[i].append(std_step)

    print(f"Evaluation: Frames and samples are collected.")

    # 2. Process collected frames into videos.
    # Check the frame.

    # fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    for i, agents in enumerate(agents):
        write_video_PIL(frames[i], f"{file_path}/{i}.gif", fps=30)
    #     writer = cv2.VideoWriter(f"{file_path}/{i}.avi", fourcc, 10.0, (500, 500))
    #     for i, frame in enumerate(frames[i]):
    #         writer.write(frame)
    #         if i == 0:
    #             cv2.imwrite(f"{file_path}/{i}.jpg", frame)
    #     writer.release()

    print(f"Evaluation: Videos are saved.")

    # 3. Save csv file with average & std of reward and steps
    with open(f"{file_path}/{'avg_std_rewards_steps'}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["avg_rewards", "std_rewards", "avg_steps", "std_steps"])
        for i in range(len(avg_rewards)):
            writer.writerow(
                [avg_rewards[i], std_rewards[i], avg_steps[i], std_steps[i]]
            )

    print(f"Evaluation: CSV file is saved.")

def convert_list_tensor_to_numpy(x: list) -> list: # I will move this kind of codes into utils
    list_np_array = []
    for tensor in x:
        np_array = tensor.cpu().numpy()
        list_np_array.append(np_array)
    return list_np_array

def average_parameters(list_parameters: list) -> OrderedDict:
    dict_parameters_post = OrderedDict()
    num_agents = len(list_parameters)
    for key, value in list_parameters[0].items():
        list_layers = []
        for i in range(num_agents):
            parameters = list_parameters[i]
            values = convert_list_tensor_to_numpy(parameters[key])
            list_layers.append(values)
        averaged_layer = np.mean(list_layers, axis=0) # Need to check if it is correctly averaged.
        averaged_layer = torch.from_numpy(averaged_layer)
        # before the values are appended into list_layers, it needs to 
        dict_parameters_post.update([(key, averaged_layer)]) # Need to check if it is correctly updated.
    return dict_parameters_post

def compare_averaged_value(list_parameters, dict_parameters_post):
    list_first_values = []
    num_agents = len(list_parameters)
    for parameters in list_parameters:
        list_values = list(parameters.values())
        first_value = list_values[0][0][0].cpu()
        list_first_values.append(np.array(first_value))

    averaged_first_value = list(dict_parameters_post.values())[0][0][0]
    averaged_first_value_from_list = np.mean(list_first_values)
    if averaged_first_value == averaged_first_value_from_list:
        print(f"The values are successfully averaged!")
    else:
        print(f"The averaged value is different to the true average.")    

def update_parameters(list_parameters, ratio_update) -> OrderedDict:
    # Only two elements are given in the list.
    # First element is the original parameters.
    # Second element is the averaged parameters.
    dict_parameters_post = OrderedDict()
    num_agents = len(list_parameters)
    for key, value in list_parameters[0].items():
        list_layers = []
        for i in range(num_agents):
            parameters = list_parameters[i]
            values = convert_list_tensor_to_numpy(parameters[key])
            if i == 0:
                values = ratio_update * np.array(values)
            elif i == 1:
                values = (1-ratio_update) * np.array(values)
            else:
                raise ValueError("More than two set of parameters are given. Only two parameters can be updated")
            list_layers.append(values)
        updated_layer = np.sum(list_layers, axis=0) # Need to check if it is correctly averaged.
        updated_layer = torch.from_numpy(updated_layer)
        # before the values are appended into list_layers, it needs to 
        dict_parameters_post.update([(key, updated_layer)]) # Need to check if it is correctly updated.
    return dict_parameters_post

def federated_learning(list_parameters, ratio_update) -> list:
    # 1. average all the parameters
    num_agents = len(list_parameters)
    list_updated_parameters = []
    dict_averaged_parameters = average_parameters(list_parameters)

    # 2. update individual parameters with averaged parameters
    for i in range(num_agents):
        list_parameters_to_update = [list_parameters[i], dict_averaged_parameters]
        updated_parameters = update_parameters(list_parameters_to_update, ratio_update)
        list_updated_parameters.append(updated_parameters)
    
    return list_updated_parameters

def swarm_learning(list_parameters, ratio_update) -> list:
    # 1. average all the parameters
    num_agents = len(list_parameters)
    list_dict_averaged_parameters = []
    for i in range(num_agents):
        if i == 0:
            list_parameters_to_average = [list_parameters[-1], list_parameters[0], list_parameters[1]]
        elif i == num_agents-1:
            list_parameters_to_average = [list_parameters[0], list_parameters[i], list_parameters[i-1]]
        else: 
            list_parameters_to_average = [list_parameters[i+1], list_parameters[i], list_parameters[i-1]]
        dict_averaged_parameters = average_parameters(list_parameters_to_average)
        list_dict_averaged_parameters.append(dict_averaged_parameters)

    list_updated_parameters = []
    # 2. update individual parameters with averaged parameters
    for i in range(num_agents):
        list_parameters_to_update = [list_parameters[i], list_dict_averaged_parameters[i]]
        updated_parameters = update_parameters(list_parameters_to_update, ratio_update)
        list_updated_parameters.append(updated_parameters)
    return list_updated_parameters

        
if __name__ == "__main__":

    # let's make another function to initialise the environments and just call the instances.
    # Just as init_nets. Let's initailise the environment and links with the net number with id.
    # So that one net uses one env exclusively.
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    mkdirs(args.outdir)

    # print args
    print(
        f"env_name: {args.env_name}, task_name: {args.task_name}, network_layer: {args.hidden1}, seed: {args.seed}"
    )

    if args.log_file_name is None:
        argument_path = (
            "experiment_arguments-%s.json"
            % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
        )
    else:
        argument_path = args.log_file_name + ".json"
    with open(os.path.join(args.logdir, argument_path), "w") as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = "experiment_log-%s" % (
            datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
        )
    log_path = args.log_file_name + ".log"
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        level=logging.DEBUG,
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.seed
    logger.info("#" * 100)

    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-")
    exp_name = time_str + args.alg
    tensorboard_dir = os.path.join("tensorboard/")
    tensorboard_path = os.path.join(args.logdir, tensorboard_dir)
    if not os.path.exists("tensorboard_path"):
        mkdirs(tensorboard_path)
    ten_file_path = os.path.join(tensorboard_path, exp_name)

    writer = SummaryWriter(ten_file_path)

    # Evaluation file saving path
    eval_file_path = os.path.join(args.outdir, exp_name)
    mkdirs(eval_file_path)

    # Seed generation
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    # logger.info("Partitioning data")
    # X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
    #     args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    # n_party_per_round = int(args.n_parties * args.sample_fraction)
    # party_list = [i for i in range(args.n_parties)]
    # party_list_rounds = []
    # if n_party_per_round != args.n_parties:
    #     for i in range(args.comm_round):
    #         party_list_rounds.append(random.sample(party_list, n_party_per_round))
    # else:
    #     for i in range(args.comm_round):
    #         party_list_rounds.append(party_list)

    # n_classes = len(np.unique(y_train))

    # train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
    #                                                                            args.datadir,
    #                                                                            args.batch_size,
    #                                                                            32)

    # print("len train_dl_global:", len(train_ds_global))
    # train_dl=None
    # data_size = len(test_ds_global)

    # logger.info("Initializing nets")
    # nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cuda:0')

    ### dm_control env added
    env_name = args.env_name
    task_name = args.task_name
    envs = init_envs(args.n_parties, env_name)

    # Check Obs and Act space for envs - leave it for future use
    print(f"Observation Space: {envs[0].observation_space.shape[0]}")
    print(f"Action Space: {envs[0].action_space.shape[0]}")

    ### DDPG agents are added
    num_agents = args.n_parties
    agents = init_agents(num_agents, envs, ten_file_path, algo_name="DDPG")

    # global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cuda:0')
    # global_model = global_models[0]
    n_comm_rounds = args.comm_round
    # if args.load_model_file and args.alg != 'plot_visual':
    #     global_model.load_state_dict(torch.load(args.load_model_file))
    #     n_comm_rounds -= args.load_model_round

    # if args.server_momentum:
    #     moment_v = copy.deepcopy(global_model.state_dict())
    #     for key in moment_v:
    #         moment_v[key] = 0
    # Let's put another argument.

    if args.alg == "independent_learning":
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))


            list_parameters = local_train_net(
                agents, envs, args, round=round, writer=writer, logging_path=ten_file_path, device=device
            )
            
            ########################                   
            ## Parameter Loading  ##
            ########################
            # Load the processed parameters into the agent for next run
            #for i in range(num_agents):
            #    agents[i].policy.load_state_dict(list_parameters[i])

    if args.alg == "fedavg":
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))


            list_parameters = local_train_net(
                agents, envs, args, round=round, writer=writer, logging_path=ten_file_path, device=device
            )
            #print(f"Round: {round}, List_keys: {list(list_parameters[0].keys())}")

            #dict_parameters_post = average_parameters(list_parameters)
            #compare_averaged_value(list_parameters, dict_parameters_post)
            list_updated_parameters = federated_learning(list_parameters, args.ratio_update)
            ########################                   
            ### FedAvg algorithm ###
            ########################

            # total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            # fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]


            # for net_id, net in enumerate(nets_this_round.values()):
            #     net_para = net.state_dict()
            #     if net_id == 0:
            #         for key in net_para:
            #             global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            #     else:
            #         for key in net_para:
            #             global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            # if args.server_momentum:
            #     delta_w = copy.deepcopy(global_w)
            #     for key in delta_w:
            #         delta_w[key] = old_w[key] - global_w[key]
            #         moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
            #         global_w[key] = old_w[key] - moment_v[key]

            # global_model.load_state_dict(global_w)

            # #logger.info('global n_training: %d' % len(train_dl_global))
            # logger.info('global n_test: %d' % len(test_dl))
            # global_model.cuda(device)
            # train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            # test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            # logger.info('>> Global Model Test accuracy: %f' % test_acc)
            # logger.info('>> Global Model Train loss: %f' % train_loss)
            # writer.add_scalar(f'Accuracy/train/global', train_acc, round)
            # writer.add_scalar(f'Accuracy/test/global', test_acc, round)
            # writer.add_scalar(f'Loss/global', train_loss, round)
            # mkdirs(args.modeldir+'fedavg/')
            # global_model.to('cuda:0')

            # torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            # torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
        
            ########################                   
            ## Parameter Loading  ##
            ########################
            # Load the processed parameters into the agent for next run
            for i in range(num_agents):
                agents[i].policy.load_state_dict(list_updated_parameters[i])

        # evaluate_agents(
        #     agents, envs, args, eval_file_path, max_episode_step=100, num_episodes=5
        # )
    if args.alg == "vanila_swarm_learning":
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))


            list_parameters = local_train_net(
                agents, envs, args, round=round, writer=writer, logging_path=ten_file_path, device=device
            )
    
            list_updated_parameters = swarm_learning(list_parameters, args.ratio_update)
              
            ########################                   
            ## Parameter Loading  ##
            ########################
            # Load the processed parameters into the agent for next run
            for i in range(num_agents):
                agents[i].policy.load_state_dict(list_updated_parameters[i])

        # evaluate_agents(
        #     agents, envs, args, eval_file_path, max_episode_step=100, num_episodes=5
        # )
