import numpy as np
import random
import itertools
import math
from k_means_constrained import KMeansConstrained

class environment():
    A_max = 100 # max. AOI
    Bo = 10**(-30/10) # channel gain (-30 dB)
    h = 100 # UAV height in meters
    xc = 100 # horizontal distance between cells centers
    yc = 100 # vertical distance between cells centers
    B = 1e6 # bandwidth
    S = 5e6 # packet size
    sigma = (10**(-100/10)) * (10e-3) # noise (-100 dBm)
    rate = 5e6 # a chosen rate
    veloc = 25 # UAV velocity
    
    max_num = rate * xc / (S * veloc) # calculate the maximum number of devices per cluster

    steps_mov = 1 # Number of cells covered by the UAV
    
    v_u = np.matrix([[0, 0], # movement action (N,S,E,W,I)
                   [steps_mov,0],
                   [-steps_mov,0],
                   [0,steps_mov],
                   [0,-steps_mov]])
    length_episode = 100 # episode length
    
    ###################################################################################################
    # init function is called when the class is defined
    def __init__(self, Dev_Coord,config):
        self.Dev_Coord = Dev_Coord # coordinates of the devices
        self.M = config.M # number of devices
        self.U = config.U # number of UAvs
        self.C = config.C # number of clusters
        self.Num_Cells = config.Num_Cells # number of cells
        self.DELTA = config.DELTA # delta value as a power factor
        
        self.A_m_max = 100 # max AoI of the system
        self.directions = 5 # number of possible directions
        
        a1 = list(range(0,self.C+1)) # select one of the devices or select none for UAV 1.
        a2 = list(range(0,self.directions)) # move up, down, right, left or don't move for UAV 1.
        a3 = list(range(0,self.C+1)) # select one of the devices or select none for UAV 2.
        a4 = list(range(0,self.directions)) # move up, down, right, left or don't move for UAV 2.
        a5 = list([a1,a2,a3,a4]) # concatenate all actions.
        self.all_actions = list(itertools.product(*a5)) # generate a lookup table.
        self.action_space = np.arange(((self.C+1)*self.directions)**2) # size of the action space
        self.observation_space = np.append(np.zeros(self.U*2),np.ones(self.C)) # size of the state space
        
        # Perform clustering
        clf = KMeansConstrained(n_clusters=self.C,size_min=1,size_max=self.max_num,random_state=0)
        clf.fit_predict(self.Dev_Coord)
        self.Labels = clf.labels_ # cluster labels
    
    ###################################################################################################
    # reset function to reset and initialize all parameters and return initial state
    def reset(self):
        # Define UAVs initial locations
        self.UAV_init_coord = np.array([])
        for i in range(self.U):
            self.init = np.random.randint(self.Num_Cells, size=2)
            self.UAV_init_coord = np.append(self.UAV_init_coord,self.init)
        
        self.A_m = []
        self.cntt = 1
        self.done = 0
        
        self.A_m = np.ones(self.C) # append number of initial age for all clusters
            
        return np.concatenate((np.asarray(self.UAV_init_coord).reshape(-1), self.A_m), axis=None)
    
    ###################################################################################################
    # Update the AoI
    def AoI_Calc(self,clust,A_m):
        if(clust<self.C): # if the selected action is to serve cluster (the opposite means not to serve any)
            A_m[clust] = 1
        return A_m

    ###################################################################################################
    # Update the UAV trajectory
    def Update_Trajec(self,l_U,V_n_Rnd):
        check_0 = 0 # check for exceeding grid coordinates
        l_U = l_U+V_n_Rnd
        for i in range(2):
            l_U[0,i] = max(0,l_U[0,i])
            l_U[0,i] = min(self.Num_Cells-1,l_U[0,i])
        
        l_U = np.asarray(l_U).reshape(-1)
        return l_U
        
    ###################################################################################################
    # Estimate the power
    def Min_Power(self,clust,L_U):
        MIN_PWR = 0
        if(clust<self.C): # if the selected action is to serve cluster (the opposite means not to serve any)
            x = np.where(self.Labels == clust) # detect the devices in that cluster
            for x_cntt in range(len(x)): # loop over those devices
                MIN_Rd = self.xc*math.dist(L_U, self.Dev_Coord[x[0][x_cntt]])
                MIN_PWR = MIN_PWR + ((MIN_Rd**2+self.h**2)*(2**(self.S/self.B)-1)*self.sigma)/self.Bo
        return MIN_PWR
    
    ###################################################################################################
    # Calculate the reward
    def Reward_Calc(self,A_m,Pwr):
        reward = 0
        reward = reward-Pwr*self.DELTA-(np.sum(A_m)/self.C)
        return reward
    
    ###################################################################################################
    # Take a step in the environment
    def step(self,state,action_chosen):
        
        self.L_U = state[0:self.U*2] # UAV positions
        self.A_m = state[self.U*2:self.U*2+self.C] # AoI
        
        self.A_m = np.minimum(self.A_m+1,self.A_m_max) # age increment
        
        self.Pwr = 0 # initialize power consumed
        L_U_new = np.zeros(self.U*2) # initialized new trajectory
        
        Action_ALL = self.all_actions[action_chosen]
        Action_ALL = np.array(Action_ALL)
        
        for u_cntt in range(self.U): # loop for the number of agents
            L_U_ind = self.L_U[u_cntt*2:u_cntt*2+2] # Location of agent i
            
            action = Action_ALL[u_cntt*2:2+u_cntt*2]
            
            self.cluster_chosen = action[0] # served cluster by agent i
            self.MOV_DIR = action[1] 
            self.v_n_rnd = self.v_u[self.MOV_DIR] # movement direction by agent i
            
            if(self.done==0):
                L_U_ind = self.Update_Trajec(L_U_ind,self.v_n_rnd) # update UAV trajectory
                
                Pwr_agent = self.Min_Power(self.cluster_chosen,L_U_ind) # pwr calculations for agent i
                self.Pwr = self.Pwr + Pwr_agent # update total power
                self.A_m = self.AoI_Calc(self.cluster_chosen,self.A_m) # age calculations

            
            L_U_new[u_cntt*2:u_cntt*2+2] = L_U_ind # update the trajectory
        
        self.Total_reward = self.Reward_Calc(self.A_m,self.Pwr/self.U)
        
        state_new = np.concatenate((np.asarray(L_U_new).reshape(-1), self.A_m), axis=None)
        
        if(self.cntt == self.length_episode): # episode length check
            self.done = 1
        self.cntt = self.cntt + 1
        
        return state_new, self.Total_reward, self.done
    