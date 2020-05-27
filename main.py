import pandas as pd
import scipy.io as sio
import os
import xlrd
from World import World
import numpy as np

def Question(transition_matrix, reward_function, gamma, theta):
    policy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    opt_value = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    flag = 1
    while flag == 1:
        delta = 0.0
        opt_value_temp = opt_value.copy()
        for s in range(0,16):
            prob_of_s = np.transpose(transition_matrix[:,s,:])
            policy_reward = np.reshape((gamma * np.matmul(np.reshape(opt_value, (1,16)), prob_of_s))  + np.reshape(reward_function[s, :], (1, 4)), (4))
            opt_value[s] = np.max(policy_reward)
            policy_action[s] = np.argmax(policy_reward) + 1
            delta = float(max(delta, float(abs(opt_value[s] - opt_value_temp[s]))))
        if delta < theta:
            flag = 0

    world = World()
    world.plot()
    world.plot_value(np.transpose(opt_value))
    world.plot_policy(np.transpose(policy_action))

def rewardFuncation (step_cost):
    reward = np.array([step_cost-1, step_cost,step_cost,step_cost,step_cost,step_cost,step_cost-1,step_cost,step_cost,step_cost,step_cost,step_cost,1+step_cost,step_cost-1,step_cost-1,step_cost])
    reward_function = np.zeros((16, 4))
    terminal = np.array([1, 7, 13, 14, 15])
    terminal = terminal - 1
    for action in range(0,4):
        for s in range(0,16):
            if s not in terminal:
                reward_function[s, action] = np.matmul(reward,transition_matrix[action, s, :])
            else:
                reward_function[s, action] = 0
    return reward_function

if __name__== "__main__":
    world = World()

    data_path = os.path.join(os.getcwd(), 'Data')  # The images path
    #mat_file = data_path + '\\Data.xlsx'
    mat_file = 'C:/Users/Daniel/PycharmProjects/Deep-Reinforcment-Learning/Data.xlsx'
    data_north = pd.read_excel(mat_file, sheet_name='North', header = None)
    data_south = pd.read_excel(mat_file, sheet_name='South', header = None)
    data_west = pd.read_excel(mat_file, sheet_name='West', header = None)
    data_east = pd.read_excel(mat_file, sheet_name='East', header = None)

    transition_matrix = np.zeros((4, 16, 16))
    transition_matrix[0, :, :] = data_north.to_numpy()
    transition_matrix[1, :, :] = data_east.to_numpy()
    transition_matrix[2, :, :] = data_south.to_numpy()
    transition_matrix[3, :, :] = data_west.to_numpy()

    # Q2.2 - value iteration gamma = 1 step_cost = -0.04 theta = 10^-4"
    Question(transition_matrix = transition_matrix, reward_function = rewardFuncation(step_cost = -0.04), gamma = 1, theta = 0.0001)
    # Q2.3 - value iteration gamma = 0.9 step_cost = -0.04 theta = 10^-4"
    Question(transition_matrix = transition_matrix, reward_function = rewardFuncation(step_cost = -0.04), gamma = 0.9, theta = 0.0001)
    # Q2.4 - value iteration gamma = 1 step_cost = -0.02 theta = 10^-4"
    Question(transition_matrix = transition_matrix, reward_function = rewardFuncation(step_cost = -0.02), gamma = 1, theta = 0.0001)
    a = 4

