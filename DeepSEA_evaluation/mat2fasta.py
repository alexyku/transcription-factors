### Converts .mat file with sequences into a fasta and corresponding label file

import pandas as pd
from scipy.io import loadmat
import numpy as np
import os
import argparse


# get command line arguments
parser = argparse.ArgumentParser(description='Data dir and out directory for mat and fata files.')
parser.add_argument('data_dir', metavar='Data Directory', type=str, nargs='+',
                   help='data directory to collect valid.mat and test.mat from.')
parser.add_argument('output_dir', metavar='Output Directory', type=str, nargs='+',
                   help='data directory to save valid.fasta and test.fasta from.')

args = parser.parse_args()

def stringify(one_hot_seq):
    """One-hot sequence to an ACTG string.

    one_hot_seq: An array with shape [length, 4] representing a one-hot-encoded
      DNA sequence. A column with all zeros is denoted by "N".

    Returns:
      A string decoding one_hot_seq.
    """
    ids = one_hot_seq.dot(np.arange(1, 5))
    # An all-zero column is denoted by "N".
    bases = np.array(list("NAGCT"))
    return "".join(bases[ids])


# file names
valid_file = os.path.join(args.data_dir[0], 'valid.mat')
test_file = os.path.join(args.data_dir[0], 'test.mat')

# Load valid.mat
mat = loadmat(valid_file)
targets = mat['validdata']
inputs = mat['validxdata']

# process validation sequences to strings
sequences = list(map(lambda i: stringify(i.transpose()), inputs))

# save validation file
out_file = os.path.join(args.output_dir[0], "valid.fasta")
print(f"Saving validation file to {out_file}...")
f= open(out_file,"w+")

for i in range(len(sequences)):
    label = f"seq{i}"
    f.write(">%s\n%s\n" %(label,sequences[i]))
        
f.close()
            
# Load test
mat = loadmat(test_file)
targets = mat['testdata']
inputs = mat['testxdata']

# process validation sequences to strings
sequences = list(map(lambda i: stringify(i.transpose()), inputs))

# save test file
out_file = os.path.join(args.output_dir[0], "test.fasta")
print(f"Saving test file to {out_file}...")
f= open(out_file,"w+")

for i in range(len(sequences)):
    label = f"seq{i}"
    f.write(">%s\n%s\n" %(label,sequences[i]))
        
f.close()