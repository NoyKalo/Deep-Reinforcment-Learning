import pandas as pd
import scipy.io as sio
import os
import xlrd
from World import World
import numpy as np

"2.2 - value iteration gamma = 1 step_cost = 0.04 theta = 10^-4"


def Q2_2(transition_matrix):

    Pie = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v_s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    theta = 0.0001
    flag = 1
    while flag == 1:
        delta = 0.0
        v = v_s.copy()
        for s in range(0,16):
            C = np.transpose(transition_matrix[:,s,:])
            options = np.reshape((gamma * np.matmul(np.reshape(v, (1,16)), C))  + np.reshape(reward_function[s, :], (1, 4)), (4))
            v_s[s] = np.max(options)
            delta = float(max(delta, float(abs(v[s] - v_s[s]))))
        if delta < theta:
            flag = 0

    for s in range (0,16):
        C = np.transpose(transition_matrix[:,s,:])
        options = np.reshape((gamma * np.matmul(np.reshape(v, (1,16)), C))  + np.reshape(reward_function[s, :], (1, 4)), (4))
        v_s[s] = np.max(options)
        Pie[s] = np.argmax(options)

    v_s[0] = -1
    v_s[6] = -1
    v_s[13] = -1
    v_s[14] = -1
    v_s[12] = 1
    world = World()
    world.plot()
    world.plot_value(np.transpose(v_s))
    a=4
    world.plot_policy(np.transpose(Pie))



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
    transition_matrix[1, :, :] = data_south.to_numpy()
    transition_matrix[2, :, :] = data_west.to_numpy()
    transition_matrix[3, :, :] = data_east.to_numpy()

    p = 0.8
    b = range(1,3)
    #frame_list = np.zeros((12, 2))
    #frame_list[:, 2] = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    #frame_list[:, 1] = np.array([b, b, b, b])
    gamma = 1

    Q2_2(transition_matrix)
    a = 4

