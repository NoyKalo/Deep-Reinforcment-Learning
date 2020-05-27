import pandas as pd
import scipy.io as sio
import os
import xlrd
from World import World
import numpy as np
import math


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

def Policy_Iteration(transition_matrix, reward_function, gamma):
    pie_policy_iteration = 0.25 * np.ones((16, 4))
    policy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    opt_value = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0])

    flag = 1
    I = np.eye(16)
    sizes = np.shape(pie_policy_iteration)

    while flag == 1:
        prob_of_s = np.zeros((16, 16))
        avg_reward = np.zeros((16))
        for s in range(0, 16):
            for action in range(0,4):
                prob_of_s[s,:] = pie_policy_iteration[s, action] * transition_matrix[action, s, :] + prob_of_s[s,:]
                avg_reward[s] = pie_policy_iteration[s, action] * reward_function[s, action] + avg_reward[s]
        opt_value =  np.matmul(avg_reward, np.linalg.inv(I - gamma * prob_of_s))

        for s in range(0, 16):
            prob_of_s = np.transpose(transition_matrix[:,:,s])
            policy_reward = (gamma * np.matmul(opt_value,prob_of_s)) + reward_function[s,:]
            opt_value[s] = np.max(policy_reward)
            policy_action[s] = np.argmax(policy_reward) + 1

        new_policy = np.zeros((16, 4))

        for s in range(0, 16):
            for action in range(0,4):
                if (policy_action[s]-1) == action:
                    new_policy[s, action] = 1

        if (math.floor(sum(sum(new_policy == pie_policy_iteration))) / (sizes[0] * sizes[1])) == 1:
            flag = 0
        else:
            pie_policy_iteration = new_policy

    world.plot_value(np.transpose(opt_value))
    world.plot_policy(np.transpose(policy_action))

def Value_Iteration(transition_matrix, reward_function, gamma, theta):
    policy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    opt_value = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0])
    flag = 1
    while flag == 1:
        delta = 0.0
        opt_value_temp = opt_value.copy()
        for s in range(0, 16):
            if (s== 0 or s== 6 or s== 12 or s== 13 or s== 14):  # already assigned those states
                continue
            prob_of_s = transition_matrix[:, s, :]
            policy_reward = np.transpose(np.matmul(prob_of_s,(gamma * opt_value + reward_function)))
            opt_value[s] = np.max(policy_reward)
            policy_action[s] = np.argmax(policy_reward) + 1
            delta = float(max(delta, float(abs(opt_value[s] - opt_value_temp[s]))))
        if delta < theta:
            flag = 0

    world.plot_value(np.transpose(opt_value))
    world.plot_policy(np.transpose(policy_action))


if __name__== "__main__":
    world = World()
    #world.plot()
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
    Value_Iteration(transition_matrix=transition_matrix, reward_function=-0.04, gamma=1,theta=0.0001)
    # Q2.3 - value iteration gamma = 0.9 step_cost = -0.04 theta = 10^-4"
    Value_Iteration(transition_matrix=transition_matrix, reward_function=-0.04, gamma=0.9,theta=0.0001)
    # Q2.4 - value iteration gamma = 1 step_cost = -0.02 theta = 10^-4"
    Value_Iteration(transition_matrix=transition_matrix, reward_function=-0.02, gamma=1,theta=0.0001)
    # Q2.5
    Policy_Iteration(transition_matrix = transition_matrix, reward_function=-0.04, gamma = 0.9)
    a = 4
