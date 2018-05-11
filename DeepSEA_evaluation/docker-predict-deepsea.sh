#!/bin/sh

# pick arguments
fasta_path=$1
output_path=$2

# Run on your sequences
docker run -v ${fasta_path}:/infile.fasta  -v ${output_path}:/output --rm haoyangz/deepsea-predict-docker python rundeepsea.py /infile.fasta /output