import torch
import random
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_


def collect_random(env, U, dataset, num_samples=100):
    state = env.reset()
    for _ in range(num_samples):
        action = random.sample(range(0, (5*(env.C+1))**2), 1)
        next_state, reward, done = env.step(state,action[0])
        
        dataset.add(state, action, reward, next_state, done)
            
        state = next_state
        if env.done:
            state = env.reset()

            
            
def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-Online", help="Run name, default: CQL-Online")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes, default: 200")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs, default: 150")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--min_eps", type=float, default=0.01, help="Minimal Epsilon, default: 4")
    parser.add_argument("--eps_frames", type=int, default=1e4, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e5")
    parser.add_argument("--Batch_online", type=int, default=32, help="Batch size for online RL, default: 32")
    parser.add_argument("--Batch_offline", type=int, default=128, help="Batch size for Offline RL, default: 128")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    parser.add_argument("--M", type=int, default=40, help="Number of devices, default: 40")
    parser.add_argument("--C", type=int, default=10, help="Number of clusters, default: 10")
    parser.add_argument("--Num_Cells", type=int, default=10, help="Number of cells, default: 10")
    parser.add_argument("--U", type=int, default=2, help="Number of agents, default: 2")
    parser.add_argument("--DELTA", type=int, default=150, help="Power weight in the reward function, default: 500")
    
    return parser.parse_args(args=[])


def eval_runs(env, agent, eval_runs=100):
    """
    Makes an evaluation run with the current epsilon
    """
    
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()
        rewards = 0
        action = 0
        while True:
#             for u in range(env.U):
            action = agent.get_action(state, 0)
#                 action[u] = action_agent[0]
                
            next_state, reward, done = env.step(state,action[0])
            
            rewards += env.Total_reward
            state = next_state
            if env.done:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)


def prep_dataloader(state_dim,dataset,num_UAVs,batch_size=256):
    
    state_start = str(0)
    state_end = str(state_dim - 1)
    action_start = str(state_dim)
    action_end = str(state_dim + num_UAVs - 1) # 1 --> (num_UAVs - 1)
    reward_col = str(state_dim + num_UAVs -1+1) # 2 --> (num_UAVs -1+1)
    next_state_start = str(state_dim + num_UAVs -1+2) # 3 --> (num_UAVs -1+2)
    next_state_end = str(state_dim + num_UAVs -1+2 + state_dim - 1)
    done_col = str(state_dim + num_UAVs -1+2 + state_dim - 1 + 1)    
    
    
    states_df = dataset.loc[:, state_start : state_end]
    actions_df = dataset.loc[:, action_start:action_end]
    rewards_df = dataset.loc[:,reward_col]
    next_states_df = dataset.loc[:, next_state_start : next_state_end]
    done_df = dataset.loc[:,done_col]

    tensors = {}
    tensors["observations"] = torch.tensor(states_df.values,dtype=torch.float)
    tensors["actions"] = torch.tensor(actions_df.values,dtype=torch.long)
    tensors["rewards"] = torch.tensor(rewards_df.values,dtype=torch.float).unsqueeze(1)
    tensors["next_observations"] = torch.tensor(next_states_df.values,dtype=torch.float)
    tensors["terminals"] = torch.tensor(done_df.values,dtype=torch.float).unsqueeze(1)
    


    tensordata = TensorDataset(tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"],
                               tensors["next_observations"],
                               tensors["terminals"])
    
    
    dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True)

    return dataloader

