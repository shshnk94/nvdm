import argparse
import os 
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(description='etm cross validation wrapper')

parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
parser.add_argument('--topics', type=str, default='50', help='number of topics')
parser.add_argument('--save_path', type=str, help='save path for every run')

args = parser.parse_args()

#Hyperparameters
hyperparameters = {'epochs': ['5000'],
                   'lr': ['5e-7', '5e-6', '5e-5', '5e-4', '5e-3']}

for params in ParameterGrid(hyperparameters):

    metrics = np.zeros((int(params['epochs']), 3))
    for fold in range(3): #Hard coded values of fold

        path = os.path.join(args.save_path, 'k{}_e{}_lr{}'.format(args.topics, params['epochs'], float(params['lr'])), 'fold{}'.format(fold), 'val_scores.csv')
        df = pd.read_csv(path, header=None)
        metrics += df.values

    metrics /= 3

    min_epoch = np.argmin(metrics[:, 0])
    print('Best epoch: {} (for the learning rate {}) with average ppl of {}'.format(min_epoch + 1, params['lr'], metrics[min_epoch, 0]))
