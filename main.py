import scipy.io as sio
import os
import pandas as pd
import xlrd
from World import World
import numpy as np

if __name__== "__main__":
    world = World()
    
    data_path = os.path.join(os.getcwd(), 'Data')  # The images path
    mat_file = data_path + '\\Data.xlsx'
    data_north = pd.read_excel(mat_file, sheet_name='North')
    data_south = pd.read_excel(mat_file, sheet_name='South')
    world.plot()
    world.plot_value([np.random.random() for i in range(world.nStates)])
    world.plot_policy(np.random.randint(1, world.nActions, (world.nStates, 1)))
