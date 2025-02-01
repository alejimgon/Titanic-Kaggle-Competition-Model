# Titanic Kaggle competition model
# Author: Alejandro Jiménez-González
# Date: 2019-01-20
# Description: This script contains the model for the Titanic Kaggle competition.

# Importing the libraries
import numpy as np # Allows us to work with arrays
import pandas as pd # Allows us to import datasets and create the matrix of features and dependent variable

# Setting the path to the data folder
main_repo_folder = '/'.join(__file__.split('/')[:-1])
data_folder = f'{main_repo_folder}/data'

# Importing the dataset
dataset = pd.read_csv(f'{data_folder}/train.csv')