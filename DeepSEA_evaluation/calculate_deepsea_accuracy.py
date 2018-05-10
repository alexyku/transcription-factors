# Loads output from Docker DeepSEA and calculates accuracy


import argparse
import pandas as pd
import os
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np


# get command line arguments
parser = argparse.ArgumentParser(description='Data dir and out directory for prediction and accuracy files.')
parser.add_argument('deepsea_predictions_file', metavar='DeepSEA predictions', type=str, nargs='+',
                   help='output predictions file from Docker DeepSEA')
parser.add_argument('mat_target_file', metavar='mat file containing targets', type=str, nargs='+',
                   help='mat file containing targets')
parser.add_argument('output_file', metavar='Output File', type=str, nargs='+',
                   help='File to save accuracies to')

args = parser.parse_args()

# Read in predictions
# File format from predicted deepsea files is "<Line #>,<FASTA seq ID>,<Comma delimited list of 919 predictions>"
predictions = pd.read_csv(args.deepsea_predictions_file[0])

# sort by sequence ids. These may get jumpled in the file merging process
predictions['id']= predictions['name'].str.extract('(\d+)').astype(int)
predictions = predictions.sort_values('id')
predictions = predictions.drop(columns=['id'])

out_file = args.output_file[0]
targets_file = args.mat_target_file[0]

# read in targets from mat file
mat = loadmat(targets_file)
try:
    targets = mat['validdata']
except:
    targets = mat['testdata']
    
# Calculate ROC and PR for all 919 elements
roc_auc_scores = []
auprc_scores = []

col = 0
for i in predictions.columns[2:]:
    try:
        roc_auc_scores.append(roc_auc_score(targets[:,col], predictions[i]))
        auprc_scores.append(average_precision_score(targets[:,col], predictions[i]))
    except ValueError:
        roc_auc_scores.append(np.NaN)
        auprc_scores.append(np.NaN)
        
    col += 1
        
# calculate mean scores        
roc_auc_mean = np.nanmean(roc_auc_scores)
auprc_mean = np.nanmean(auprc_scores)

# append means to other scores
roc_auc_scores.insert(0, "auROC")
roc_auc_scores.append(roc_auc_mean)

auprc_scores.insert(0, "auPRC")
auprc_scores.append(auprc_mean)

# Write accuracies to file
output_df = predictions[0:0]
output_df = output_df.drop(columns=['name'])
output_df["Mean"] = []
output_df.loc[0] = roc_auc_scores
output_df.loc[1] = auprc_scores
output_df.to_csv(out_file)