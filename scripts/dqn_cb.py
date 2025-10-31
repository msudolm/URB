import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import ast
import json
import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from collections     import deque
from routerl         import TrafficEnvironment
from tqdm            import tqdm

from baseline_models import BaseLearningModel
from utils           import clear_SUMO_files
from utils           import print_agent_counts

from routerl import Keychain as kc
import dqn_cb_utils


#
#
# IQL Network implementation was here
#
#


# Main script to run the IQL experiment
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    args = parser.parse_args()
    ALGORITHM = "dqn_cb"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed
    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Device is: ", device)
        
    # Parameter setting
    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"], env_params, task_params

    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    
    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

    # Read origin-destinations
    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

    
    # Copy agents.csv from custom_network_folder to records_folder
    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps
            
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["torch_seed"] = torch_seed
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["script"] = os.path.abspath(__file__)
    dump_config["algorithm"] = ALGORITHM
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    
    # Initialize the environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = True,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters" : {
                "model" : human_model
            },
            "machine_parameters" : {
                "behavior" : av_behavior,
                "observation_type" : "previous_agents_plus_start_time"
            }
        },
        environment_parameters = {
            "save_every" : save_every,
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo"
        }, 
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : smooth_by,
            "plot_choices" : plot_choices,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : number_of_paths,
            "beta" : path_gen_beta,
            "num_samples" : num_samples,
            "visualize_paths" : False
        } 
    )

    env.start()
    env.reset()
    print_agent_counts(env)


    ### Human learning phase ###
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()


    # Mutation
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)
    print_agent_counts(env)
    
    # Set policies for machine agents
    n_observation_table_features = 7+1 ## note: pass from config later; 7 features + is_acting; TODO
    q_net = dqn_cb_utils.DQN(
            state_size = n_observation_table_features * len(env.machine_agents), ##
            action_space_size = env.environment_params[kc.ACTION_SPACE_SIZE], ##
            device=device,
            eps_init=eps_init,
            eps_decay=eps_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            lr=lr, 
            num_epochs=num_epochs,
            num_hidden=num_hidden,
            widths=widths)
    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}
    
    
    ### Learning phase ###
    pbar.set_description("AV learning")

    # ##########    Agents info print   ######################################
    # print(f"env.all_agents: {env.all_agents}\n")
    # print(f"env.machine_agents: {env.machine_agents}\n")
    # print(f"env.human_agents: {env.human_agents}\n")
    # print(f"env.possible_agents: {env.possible_agents}\n")

    # print(f"env.all_agents details:")
    # print(f"ID | start_time | origin | destination | kind")
    # for agent in env.all_agents:
    #     print(f"{agent.id}, {agent.start_time}, {agent.origin}, {agent.destination}, {agent.kind}")
    # ########################################################################



    os.makedirs(plots_folder, exist_ok=True)

    global_observation = dqn_cb_utils.GlobalObservation(env.machine_agents)
    train_every_counter = 0 # Counter for training Q-Net every k agents 

    for episode in range(training_eps):
        env.reset()
        global_observation.reset()
        temp_memory = {agent.id: dict() for agent in env.machine_agents} # Keep buffer data (observation, action) before reward (-travel_time) is known
        # print([(key, type(key)) for key in temp_memory])

        # Simulate trafic day by day, collect data and keep training the network
        for agent_id in env.agent_iter():
            # print(f"\nCurrent agent: {agent_id}")
            #print(f"Travel times list: {env.travel_times_list}\n")
            
            _, reward, termination, truncation, info = env.last() # observation, reward, termination, truncation, info

            assert isinstance(agent_id, str) and agent_id.isnumeric()
            agent_id_int = int(agent_id)
            agent_obj = agent_lookup[agent_id]



            if termination or truncation: # Episode finished: add (s,a,r) tuples to replay buffer
                train_every_counter += 1

                # print(f"train_every_counter: {train_every_counter}")
                # print(f"update_every_k_agents: {update_every_k_agents}")
                # print(f"train_every_counter % update_every_k_agents == {train_every_counter % update_every_k_agents}\n")

                # Add agent (s,a,r) to the replay buffer
                #reward, = [-info[kc.TRAVEL_TIME] for info in env.travel_times_list if str(info[kc.AGENT_ID]) == agent_id] # can be made more effective if the env contained info keyed by agent id
                #reward = - global_observation.get_agent_feature(agent_id=agent_id_int, feature='travel_time')
                assert isinstance(reward, (np.floating, float)) and reward<0, f"Reward: {reward} ({type(reward)}); agent: {agent_id}"
                state, action = temp_memory[agent_id_int]['observation'], temp_memory[agent_id_int]['action']
                q_net.push(state, action, reward)
                action = None

                # Learn (optionally may be changed to after each k episodes)
                if train_every_counter % update_every_k_agents == 0:
                    train_every_counter = 0 # reset counter
                    q_net.learn()

            else:

                # Update global observation (check env state and add machines that have finished since last step)
                global_observation.update_recently_finished_machines(env)
                global_observation.add_agent(agent_obj)

                # Get agent observation from global observation
                agent_observation = global_observation.generate_agent_observation(agent_id_int)

                # Select agent action
                action = q_net.act(agent_observation)
                temp_memory[agent_id_int].update({'observation': agent_observation, 'action': action}) # save to add to the buffer when reward is known
                global_observation.add_agent_action(agent_id_int, action)



                # print(f"Observation table:\n{global_observation.observation_table}")
                # print(f"Agent {agent_id} observation:\n{agent_observation}")
                

            env.step(action)
            """Note: travel times for agents finished after last agant departure will not be included in global observation - 
            because: env.step(action) first add them to env.travel_times_list, then resets this list to [] in one call,
            so it is impossible to get these last part of travel times in this arrangement.
            But, the other fact is that we do not necessarly need them in global observation - for the last agent departing,
            he does know these times at his start timepoint anyway. So they can be left as unknown in global obs when the episode ends."""

            
        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()
    
    
    ### Testing phase ###
    ####################################
    # for agent in env.machine_agents:
    #     agent.model.epsilon = 0.0
    #     agent.model.q_network.eval()
    ####################################

    q_net.epsilon = 0.0
    q_net.q_network.eval()

        
    pbar.set_description("Testing")
    global_observation = dqn_cb_utils.GlobalObservation(env.machine_agents)

    for episode in range(test_eps):
        env.reset()
        global_observation.reset()

        for agent_id in env.agent_iter():
            agent_id_int, agent_obj = int(agent_id), agent_lookup[agent_id]

            _, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None

            else:
                # Update global observation
                global_observation.update_recently_finished_machines(env)
                global_observation.add_agent(agent_obj)

                # Get agent observation from global observation
                agent_observation = global_observation.generate_agent_observation(agent_id_int)

                action = q_net.act(agent_observation)
                global_observation.add_agent_action(agent_id_int, action)

            env.step(action)
        pbar.update()

    ########################################################################        
    #         observation, reward, termination, truncation, info = env.last()
    #         if termination or truncation:
    #             action = None
    #         else:
    #             action = agent_lookup[agent_id].model.act(observation)
    #         env.step(action)
    #     pbar.update()
    ##########################################################################
    
    # Finalize the experiment
    pbar.close()
    env.plot_results()
    losses_df = pd.DataFrame({"losses": q_net.loss}) #no agent.model.loss in centralized dqn
    losses_df.to_csv(os.path.join(records_folder, "losses.csv"))#, index=False)
    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)