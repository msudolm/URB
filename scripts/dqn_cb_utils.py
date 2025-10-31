from __future__ import annotations

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections     import deque
from routerl         import TrafficEnvironment
from tqdm            import tqdm

from routerl import Keychain as kc

from baseline_models import BaseLearningModel


################################
# Global observation
################################
class GlobalObservation():
    """ A class for storing and managing global observation for machine agents routing decisions.
    """
    def __init__(self, agents: list[BaseAgent])->None:

        self.features = { # Column names in observation table and default values for representing  unknown data
            'start_time': -1,
            'origin': -1,
            'destination': -1,
            'route': -1,
            'travel_time': -1.0,
            'has_finished': 0,
            #'is_acting': 0, #<- kept in self.acting_agent_id and appended only when generating agent's observation
            'is_known': 0,
        }

        self._initialize_observation_table(agents)
        self.finished = False
        self.acting_agent_id = None


    def _assert_agents_time_sorted(self,agents: list[BaseAgent])->None:
        if len(agents) <2:
            return
        for i in range(1,len(agents)):
            assert agents[i-1].start_time <= agents[i].start_time, f"i={i}; Agent_{i-1} ({agents[i-1].kind} {agents[i-1].id}) start_time: {agents[i-1].start_time}, Agent_{i}  ({agents[i].kind} {agents[i].id})start_time: {agents[i].start_time}"
        return

    def _initialize_observation_table(self, agents: list[BaseAgent] )->None:
        """ Initialize observation table.
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
        """ Reset global observation:
                - fill observation table (pd.DataFrame of shape=(n_machine_agents,n_features)) with default values for each feature column.
                - set self.agents_finished flag to False.
                - set currently acting agent to None.
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
        """ Get IDs of the agents that started but not yet finished their drives according to the current status of the observation table.
        """
        df = self.observation_table
        return df.index[df['is_known'] & ~df['has_finished']]

    def update_recently_finished_machines(self, env: TrafficEnvironment)->None:
        """ Update observation table with current state of the environment  - update agents finished from the last update - 'is_finished' indicators and travel times.
        """

        # Get travel times from agents that finished after the last snapshot
        unfinished_agents = self._get_unfinished_agents_ids() # <- Agents that was active in the last snapshot (integer IDs)
        finished_agents_times = { # <- Agents that finished after last snapshot
            info[kc.AGENT_ID] : info[kc.TRAVEL_TIME] # agent_id: travel_time
            for info in env.travel_times_list
            if info[kc.AGENT_ID] in unfinished_agents and
            kc.TRAVEL_TIME in info and 
            info[kc.TRAVEL_TIME] != self.features['travel_time'] # ensure that nonempty value assigned (real travel time) 
        } 

        # Update observation table with finished agents info
        self.observation_table['travel_time'].update(pd.Series(finished_agents_times))
        self.observation_table['has_finished'].update(pd.Series({agent: 1 for agent in finished_agents_times}))

        if self.observation_table['has_finished'].all():
            self.finished = True
        return 



    #####################################
    ####### Agent-related methods #######
    #####################################

    def add_agent(self, agent: BaseAgent)->None:
        """ Add agent information to the observation table to a row indexed with agent ID.
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
        assert agent_id == self.acting_agent_id
        self.observation_table.at[agent_id, 'route'] = action
        return


    def generate_agent_observation(self, agent_id: int)->np.ndarray:
        """Return the agent's view of the global observation table.

        Constructs a view of the global observation table for the specified agent
        and adds a one-hot column indicating agent as currently acting.

        Args:
            agent_id (int): The unique identifier of the agent for whom the observation is generated (agent.id).
        Returns:
            np.ndarray: A NumPy array representing the agent’s view of the global observation, preserving original column order
                with an additional one-hot column containing a 1 in the row corresponding to the acting agent added as a last column.
        """

        # Agent ID assertion
        assert agent_id in self.observation_table.index, f"agent_id: {agent_id} (type: {type(agent_id)})\nobservation_table_index: {self.observation_table.index}"
        
        # Ensure that the global observation table is sorted by agent start time
        assert self._is_column_prefix_sorted(self.observation_table, colname='start_time', empty_val=-1), f"'start_time' column is not sorted in non-descending order! Possibly contains empty values for earlier agents.\nTable: {self.observation_table}"
        # Optional: ensure that there no data in rows 'below' agent in the table (assumed for current implementation, future agents may be also considered in next implementations)
        
        # Get observation table view for the agent
        obs = self.observation_table.copy()
        obs['is_acting'] = 0
        obs.at[agent_id, 'is_acting'] = 1
        return obs.to_numpy().flatten()


    def get_agent_feature(self, agent_id: int, feature: str)->Any:
        return self.observation_table.at[agent_id, feature]



    ####### Helper functions #######

    def _is_column_prefix_sorted(self, df: pd.DataFrame, colname: str, empty_val:Any)->bool:
        """ Check if all non-empty values of data frame column are contained in column prefix (not mixed with empty values)
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

            if not np.all(values[first_empty_iloc:]==empty_val): # Check if no empty value between nonempty values (if so, return False)
                return False

            column_prefix = values[:first_empty_iloc] 
        else:
            column_prefix = values

        
        if len(column_prefix) <2:
            return True
        
        return np.all(column_prefix[:-1] <= column_prefix[1:]) # compare elements with their right neigbours







################################
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
        #raise NotImplementedError ## TODO: fully inspect for correctnes for centralized DQN version
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        self.last_state = state ## TODO: safely remove this
        self.last_action = action ## TODO: safely remove this
        return action
    
    def push(self, state, action, reward):
        # Add (s,a,r)
        # All interactions are single-step, so we only store the last state, action, and reward
        self.memory.append((state, action, reward))
        return

    def learn(self):
        # raise NotImplementedError ## TODO: fully inspect for correctnes for centralized DQN version
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

