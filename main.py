import pandas as pd
import scipy.io as sio
import os
import xlrd
from World import World
import numpy as np
import math
import random


def policy_iteration_algo(transition_matrix, theta, gamma, step_cost, terminal):
    policy = np.zeros((16,4))
    states_order = range(0, 16)

    #create random policy for first step
    for i in range(0,16):
        s = states_order[i]
        if s in terminal:
            continue
        ind = random.randint(0,3)
        policy[ind] = 1

    policy_stable = 0
    step_counter = 0
    while policy_stable == 0:
        V = policy_evaluation_step(transition_matrix, policy,theta, gamma, step_cost, terminal)
        new_policy = policy_improvment_step(transition_matrix, V, gamma, step_cost)
        if np.array_equal(new_policy,policy):
            policy_stable = 1
        policy = new_policy
        step_counter = step_counter + 1

    update_new_policy = np.zeros((16))
    for i in range(0,16):
        update_new_policy[i] = np.argmax(new_policy[i,:])+1

    world.plot_value(np.transpose(V))
    world.plot_policy(np.transpose(update_new_policy))

def policy_evaluation_step(transition_matrix, policy, theta, gamma, step_cost, terminal):
    V = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0])

    delta = theta + 1
    while theta < delta:
        delta = 0
        for s in range(0,16):
            if s in terminal:
                continue
            V_temp = V[s]
            prob_of_s = transition_matrix[:, s, :]
            V[s] = np.matmul((np.matmul(np.reshape(policy[s, :], (1, 4)), prob_of_s)),np.reshape(step_cost + gamma * V, (16, 1)))
            delta = max(delta, abs(V_temp-V[s]))
    return V

def policy_improvment_step(transition_matrix, V, gamma, step_cost):
    policy = np.zeros((16, 4))
    Q_table = np.zeros((16, 4))

    for s in range(0,16):
        prob_of_s = transition_matrix[:, s, :]
        Q_table[s,:] = np.reshape(np.matmul(prob_of_s,np.reshape(step_cost + gamma * V, (16, 1))),4)
        ind = np.argmax(Q_table[s,:])
        policy[s,ind] = 1

    return policy

def Policy_Iteration(transition_matrix, reward_function, gamma):
    reward_function_matrix = rewardFuncation(reward_function)
    pie_policy_iteration = 0.25 * np.ones((16, 4))
    policy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    opt_value = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0])
    #opt_value = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    flag = 1
    I = np.eye(16)
    sizes = np.shape(pie_policy_iteration)

    #new_policy = pie_policy_iteration
    while flag == 1:
        #policy evaluation
        prob_of_s = np.zeros((16, 16))
        avg_reward = np.zeros((16))
        for s in range(0, 16):
            if (s== 0 or s== 6 or s== 12 or s== 13 or s== 14):  # already assigned those states
                continue
            for action in range(0,4):
                prob_of_s[s,:] = pie_policy_iteration[s, action] * transition_matrix[action, s, :] + prob_of_s[s,:]
                avg_reward[s] = pie_policy_iteration[s, action] * reward_function_matrix[s, action] + avg_reward[s]
        opt_value =  np.matmul(avg_reward, np.linalg.inv(I - gamma * prob_of_s))

        #policy improvement
        for s in range(0, 16):
            if (s== 0 or s== 6 or s== 12 or s== 13 or s== 14):  # already assigned those states
                continue
            prob_of_s = np.transpose(transition_matrix[:,:,s])
            policy_reward = (gamma * np.matmul(opt_value,prob_of_s)) + reward_function_matrix[s,:]
            opt_value[s] = np.max(policy_reward)
            policy_action[s] = np.argmax(policy_reward) + 1

        new_policy = np.zeros((16, 4))

        for s in range(0, 16):
            for action in range(0,4):
                if (policy_action[s]-1) == action:
                    new_policy[s, action] = 1

        #check if the policy has changed
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

    terminal = np.array([1, 7, 13, 14, 15])
    terminal = terminal - 1

    # Q2.2 - value iteration gamma = 1 step_cost = -0.04 theta = 10^-4"
    Value_Iteration(transition_matrix=transition_matrix, reward_function=-0.04, gamma=1,theta=0.0001)
    # Q2.3 - value iteration gamma = 0.9 step_cost = -0.04 theta = 10^-4"
    Value_Iteration(transition_matrix=transition_matrix, reward_function=-0.04, gamma=0.9,theta=0.0001)
    # Q2.4 - value iteration gamma = 1 step_cost = -0.02 theta = 10^-4"
    Value_Iteration(transition_matrix=transition_matrix, reward_function=-0.02, gamma=1,theta=0.0001)
    # Q2.5
    policy_iteration_algo(transition_matrix = transition_matrix, theta=0.0001, gamma = 0.9, step_cost=-0.04, terminal = terminal)
