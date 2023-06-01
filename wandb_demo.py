"""
Import libraries needed for the homework
"""
import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Import notebook libraries
import wandb
import pickle
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Initialize the project
wandb.init(project='mlops-zoomcamp-wandb', name='experiment-1')

# Grab the X/Y data for the laod iris
X, y = load_iris(return_X_y=True)
label_names = ['Setosa', 'Versicolour', 'Virginica']

# Log model configs to wandb
params = {"C":0.1, "random_state":42}
wandb.config = params

