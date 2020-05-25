import scipy.io as sio
import os
import pandas as pd
import xlrd

if __name__== "__main__":
    a=4
    data_path = os.path.join(os.getcwd(), 'Data')  # The images path
    mat_file = data_path + '\\Data.xlsx'
    data_north = pd.read_excel(mat_file, sheet_name='North')
    data_south = pd.read_excel(mat_file, sheet_name='South')

    a=4