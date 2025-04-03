import torch
import torch.nn as nn

class DDQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size):
        super(DDQN, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.head_1 = nn.Linear(self.input_shape[0], layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)

    def forward(self, input):
        """
        
        """
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        
        return out

class CQR_DQN(nn.Module):
    def __init__(self, state_size, action_size,layer_size, seed, N, layer_type="ff"):
        super(CQR_DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.N = N

        self.head_1 = nn.Linear(self.input_shape[0], layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size*N)
        #weight_init([self.head_1, self.ff_1])


    
    def forward(self, input):
        """
        
        """
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        
        return out.view(input.shape[0], self.N, self.action_size)
    
    def get_action(self,input):
        x = self.forward(input)
        return x.mean(dim=1)