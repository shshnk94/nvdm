import argparse
import os 
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(description='etm cross validation wrapper')

parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
parser.add_argument('--topics', type=str, default='50', help='number of topics')
parser.add_argument('--vocab_size', type=str, default='2364', help='vocabulary size')
parser.add_argument('--data_path', type=str, help='path to a fold of data')
parser.add_argument('--save_path', type=str, help='save path for every run')
parser.add_argument('--gpu', type=str, help='index of the gpu core which would contain this model')

args = parser.parse_args()

def run_script(params, fold):

    os.system('CUDA_VISIBLE_DEVICES={} python nvdm/nvdm.py --data_dir '.format(args.gpu) + args.data_path +
                  ' --n_topic ' + args.topics +
                  ' --epochs ' + params['epochs'] +
                  ' --batch_size 64' +
                  ' --vocab_size ' + args.vocab_size +
                  ' --fold ' + str(fold) +
                  ' --learning_rate ' + params['lr'] + 
                  ' --save_path ' + args.save_path + ' &')

#Hyperparameters
hyperparameters = {'epochs': ['2000'],
                   'lr': ['5e-5', '5e-3', '5e-4']}

for params in ParameterGrid(hyperparameters):
    for fold in range(3): #Hard coded values of fold
        run_script(params, fold)
