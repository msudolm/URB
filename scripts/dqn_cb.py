from __future__ import annotations

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





####################################   OBSERVATION AND Q-NETWORK IMPLEMENTATION   #####################################################

###############################
# Global observation
################################
class GlobalObservation():
    """
    A class for storing and managing global observation for machine agents routing decisions.
    """
    def __init__(self, agents: list[BaseAgent])->None:

        # Columns in observation table & default values for representing empty data
        self.features = {
            'start_time': -1,
            'origin': -1,
            'destination': -1,
            'route': -1,
            'travel_time': -1.0,
            'has_finished': 0,
            'is_known': 0,
            #'is_acting': 0, #<- kept in self.acting_agent_id and appended only when generating agent's observation
        }

        self._initialize_observation_table(agents)
        self.finished = False
        self.acting_agent_id = None


    def _assert_agents_time_sorted(self, agents: list[BaseAgent])->None:
        """
        Check that the list of agents is sorted by start time in non-descending order.
        """

        if len(agents) <2:
            return
        for i in range(1,len(agents)):
            assert agents[i-1].start_time <= agents[i].start_time, f"i={i}; Agent_{i-1} ({agents[i-1].kind} {agents[i-1].id}) start_time: {agents[i-1].start_time}, Agent_{i}  ({agents[i].kind} {agents[i].id})start_time: {agents[i].start_time}"
        return

    def _initialize_observation_table(self, agents: list[BaseAgent] )->None:
        """
        Initialize observation table.
        Set row indices to agent IDs (integers) sorted by start times. Set column names to self.features.
        Fill columns with default values for each feature.

        Args:
            agents (list[BaseAgent]): list of agents to be included in the global observation table.

        Returns:
            None
        """

        idx_time_sorted = [agent.id for agent in sorted(agents, key=lambda x: x.start_time)] # note: some permutations on agents with equal start times - here or somwhere else?
        #self._assert_agents_time_sorted(agents)

        empty_data = {
            key: [value] * len(idx_time_sorted)
            for key, value in self.features.items()
        }
        self.observation_table = pd.DataFrame(empty_data, index=idx_time_sorted)
        return

    def reset(self)->None:
        """
        Reset the global observation:
            - Fill observation table (pd.DataFrame of shape=(n_machine_agents,n_features)) with default values for each feature column.
            - Set self.agents_finished flag to False.
            - Set currently acting agent to None.

        Args:
            None
        Returns:
            None
        """
        # Fill columns with default empty values for columns
        self.observation_table[:] = pd.DataFrame(self.features, index=self.observation_table.index) # note: add permuting agents with the same start times here or somwhere else (e.g. compare _initialize_observation_table)
        self.acting_agent_id = None
        self.agents_finished = False 
        return

    def _get_unfinished_agents_ids(self):
        """
        Get IDs of the agents that started but not yet finished their drives according to the current status of the observation table.
        """
        df = self.observation_table
        return df.index[df['is_known'] & ~df['has_finished']]

    def update_recently_finished_machines(self, env: TrafficEnvironment)->None:
        """
        Update observation table with current state of the environment  => update agents finished from the last update -> their 'is_finished' indicators and travel times.
        """

        # Get travel times from agents that finished after the last snapshot
        unfinished_agents = self._get_unfinished_agents_ids() # <- Agents that was active in the last snapshot (integer IDs)
        finished_agents_times = { # <- Agents that finished after last snapshot
            info[kc.AGENT_ID] : info[kc.TRAVEL_TIME]
            for info in env.travel_times_list
            if info[kc.AGENT_ID] in unfinished_agents and
            kc.TRAVEL_TIME in info and 
            info[kc.TRAVEL_TIME] != self.features['travel_time'] # check if assigned travel time is not 'empty value'
        } 

        # Update observation table with finished agents info
        self.observation_table['travel_time'].update(pd.Series(finished_agents_times))
        self.observation_table['has_finished'].update(pd.Series({agent: 1 for agent in finished_agents_times}))

        if self.observation_table['has_finished'].all():
            self.finished = True
        return 



    ####### Agent-related methods #######

    def add_agent(self, agent: BaseAgent)->None:
        """
        Add agent information to the observation table to a row indexed with agent ID.
        Move acting agent indicator to the added agent.

        Args:
            agent (BaseAgent): An agent whose info is to be added.
        Returns:
            None
        """
        assert self.observation_table.at[agent.id, 'is_known'] == 0, "Trying to overwrite information for known agent"
        self.observation_table.loc[agent.id, ['start_time', 'origin', 'destination']] = [agent.start_time, agent.origin, agent.destination]
        self.observation_table.at[agent.id, 'is_known'] = 1
        self.acting_agent_id = agent.id
        return

    def add_agent_action(self, agent_id: int, action: int)->None:
        """
        Add action value (route identifier) to 'route' column in agent's row.
        """
        assert agent_id == self.acting_agent_id
        self.observation_table.at[agent_id, 'route'] = action
        return

    def get_agent_feature(self, agent_id: int, feature: str)->Any:
        return self.observation_table.at[agent_id, feature]

    def generate_agent_observation(self, agent_id: int)->np.ndarray:
        """
        Return the agent's view of the global observation table.

        Constructs a view of the global observation table for the specified agent.
        Adds a one-hot column indicating agent as currently acting.

        Args:
            agent_id (int): The unique identifier of the agent for whom the observation is generated (agent.id).
        Returns:
            np.ndarray: A NumPy array representing the agent’s view of the global observation, preserving original column order
                with an additional one-hot column containing a 1 in the row corresponding to the acting agent added as a last column.
        """

        # Ensure that agent ID in observation table index (for int vs str(int) matching)
        assert agent_id in self.observation_table.index, f"agent_id: {agent_id} (type: {type(agent_id)})\nobservation_table_index: {self.observation_table.index}"
        
        # Check that agents in the global observation table are ordered by start time
        assert self._is_column_prefix_sorted(self.observation_table, colname='start_time', empty_val=-1), f"'start_time' column is not sorted in non-descending order! Possibly contains empty values for earlier agents.\nTable: {self.observation_table}"
        # Optionally: ensure that there no data in rows 'below' agent in the table (assumed for current implementation, future agents may be also considered in next implementations)
        
        # Get observation table view for the agent
        obs = self.observation_table.copy()
        obs['is_acting'] = 0
        obs.at[agent_id, 'is_acting'] = 1
        return obs.to_numpy().flatten()




    ####### Helper methods #######

    def _is_column_prefix_sorted(self, df: pd.DataFrame, colname: str, empty_val:Any)->bool:
        """
        Check if all non-empty values of data frame column are contained in column prefix (not mixed with empty values)
        and sorted in a non-descending order.

        Args:
            df (pd.DataFrame): DataFrame to be checked.
            colname (str): Name of the DataFrame column to be checked.
            empty_value (Any): Value representing empty data.
        Returns:
            bool: True if the column is in the form: [nonempty_values_nondesc] + [empty_values], False otherwise.
        """

        # Get column values and empty vals indices
        col = df[colname]
        values = col.to_numpy()
        empty_ilocs, = np.where(values == empty_val)


        # Get column prefix containing all nonempty values (early escape with False if empty values mixed with nonempty values)
        if len(empty_ilocs) > 0:
            first_empty_iloc = empty_ilocs[0]

            # Check if no empty value between nonempty values (if so, return False)
            if not np.all(values[first_empty_iloc:]==empty_val): 
                return False
            column_prefix = values[:first_empty_iloc] 
        else:
            column_prefix = values

        
        if len(column_prefix) <2:
            return True
        
        return np.all(column_prefix[:-1] <= column_prefix[1:]) # compare elements with their right neigbours



###############################
# DQN Network
################################

### Simplified single-DQN implementation for single-step decision-making
class DQN(BaseLearningModel):
    def __init__(self, state_size, action_space_size,
                 device="cpu", eps_init=0.99, eps_decay=0.998,
                 buffer_size=256, batch_size=16, lr=0.003, 
                 num_epochs=1, num_hidden=2, widths=[32, 64, 32]):
        ##raise NotImplementedError
        super().__init__()
        self.device = device
        self.action_space_size = action_space_size
        self.epsilon = eps_init
        self.eps_decay = eps_decay
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.q_network = Network(state_size, action_space_size, num_hidden, widths).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.loss = list()

    def act(self, state: np.ndarray)->int:
        """
        Act epsilon-greedy.
        """
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        return action
    
    def push(self, state, action, reward):
        """
        Add (s,a,r) tuple to the buffer.
        """
        # All interactions are single-step, so we only store the last state, action, and reward
        self.memory.append((state, action, reward))
        return

    def learn(self):
        if len(self.memory) < self.batch_size: return
        step_loss = list()
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards = zip(*batch)
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device) ## TODO: check how actions are encoded in buffer (int or one-hot)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

            # Predict Q-values (travel times) for actions, compare with recorded travel times
            current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)
            target_q_values = rewards_tensor

            loss = self.loss_fn(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            step_loss.append(loss.item())
        self.loss.append(sum(step_loss)/len(step_loss))
        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon *= self.eps_decay


class Network(nn.Module):
    def __init__(self, in_size, out_size, num_hidden, widths):
        super(Network, self).__init__()
        assert len(widths) == (num_hidden + 1), "DQN widths and number of layers mismatch!"
        
        self.input_layer = nn.Linear(in_size, widths[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(widths[x], widths[x+1]) for x in range(num_hidden)])
        self.out_layer = nn.Linear(widths[-1], out_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.out_layer(x)
        return x

##################################################################################################################################






# Main script to run the DQN experiment
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
    q_net = DQN(
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
    os.makedirs(plots_folder, exist_ok=True)

    global_observation = GlobalObservation(env.machine_agents)
    train_every_counter = 0 # Counter for training Q-Net every k agents 

    for episode in range(training_eps):
        env.reset()
        global_observation.reset()
        temp_memory = {agent.id: dict() for agent in env.machine_agents} # Keep buffer data (observation, action) before reward is known (termination/truncation iteration)

        # Simulate trafic day by day, collect data and keep training the network
        for agent_id in env.agent_iter():
            
            _, reward, termination, truncation, info = env.last() # observation, reward, termination, truncation, info

            assert isinstance(agent_id, str) and agent_id.isnumeric()
            agent_id_int = int(agent_id)
            agent_obj = agent_lookup[agent_id]



            if termination or truncation: # Episode finished: add (s,a,r) tuples to replay buffer
                train_every_counter += 1

                # Add agent (s,a,r) to the replay buffer
                assert isinstance(reward, (np.floating, float)) and reward<0, f"Reward: {reward} ({type(reward)}); agent: {agent_id}"
                state, action = temp_memory[agent_id_int]['observation'], temp_memory[agent_id_int]['action']
                q_net.push(state, action, reward)

                action = None

                # Learn (optionally may be changed from each-k-agents to each-k-episodes)
                if train_every_counter % update_every_k_agents == 0:
                    train_every_counter = 0 # reset counter
                    q_net.learn()

            else:

                # Update global observation - with env state change
                global_observation.update_recently_finished_machines(env)
                global_observation.add_agent(agent_obj)

                # Get agent observation from global observation
                agent_observation = global_observation.generate_agent_observation(agent_id_int)

                # Select agent action
                action = q_net.act(agent_observation)
                temp_memory[agent_id_int].update({'observation': agent_observation, 'action': action}) # save to add to the buffer when reward is known
                global_observation.add_agent_action(agent_id_int, action)


                

            env.step(action)
            """Note: travel times for agents finished after last agant departure will not be included in global observation - 
            because: env.step(action) first adds them to env.travel_times_list, then resets this list to [] in one call,
            so it is impossible to get these last part of travel times in this arrangement.
            But, the other fact is that we do not necessarly need them in global observation - for the last agent departing,
            he does know these times at his start timepoint anyway. So they can be left as unknown in global obs when the episode ends."""

            
        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()
    
    
    ### Testing phase ###
    q_net.epsilon = 0.0
    q_net.q_network.eval()

    pbar.set_description("Testing")
    global_observation = GlobalObservation(env.machine_agents)

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

    
    # Finalize the experiment
    pbar.close()
    env.plot_results()
    losses_df = pd.DataFrame({"losses": q_net.loss})
    losses_df.to_csv(os.path.join(records_folder, "losses.csv"))
    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)