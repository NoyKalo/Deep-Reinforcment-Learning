import pandas as pd
import scipy.io as sio
import os
import xlrd
from World import World
import numpy as np

#"2.2 - value iteration gamma = 1 step_cost = 0.04 theta = 10^-4"
def Q2_2(transition_matrix):

    policy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    opt_value = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    theta = 0.0001
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



if __name__== "__main__":
    world = World()

    data_path = os.path.join(os.getcwd(), 'Data')  # The images path
    #mat_file = data_path + '\\Data.xlsx'
    mat_file = 'C:/Users/Daniel/PycharmProjects/Deep-Reinforcment-Learning/Data.xlsx'
    data_north = pd.read_excel(mat_file, sheet_name='North', header = None)
    data_south = pd.read_excel(mat_file, sheet_name='South', header = None)
    data_west = pd.read_excel(mat_file, sheet_name='West', header = None)
    data_east = pd.read_excel(mat_file, sheet_name='East', header = None)

    reward_function = pd.read_excel(mat_file, sheet_name='Rewards')
    reward_function = reward_function.to_numpy()

    transition_matrix = np.zeros((4, 16, 16))
    transition_matrix[0, :, :] = data_north.to_numpy()
    transition_matrix[1, :, :] = data_east.to_numpy()
    transition_matrix[2, :, :] = data_south.to_numpy()
    transition_matrix[3, :, :] = data_west.to_numpy()

    p = 0.8
    b = range(1,3)
    gamma = 1

    Q2_2(transition_matrix)
    a = 4

