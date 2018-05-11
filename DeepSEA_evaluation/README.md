This folder contains all code related to DeepSEA evaluations.


# How to use this pipeline to score DeepSEA

1. Install DeepSEA with Docker:
 
```
$ ./docker-install-deepsea.sh
```

2. Process mat files to fasta format:

```
$ python mat2fasta.py <INPUT_PATH_TO_MAT_FILES> <OUTPUT_PATH_TO_FASTA_FILES>
```

3. Run predictions on fasta files:
```
$ ./docker-predict-deepsea.sh <FULL_PATH_TO_FASTA_FILE> <FULL_PATH_TO_OUTPUT_PREDICTION>
```

4. Calcuate AUC scores for predictions
```
$ python mat2fasta.py <PATH_TO_DEEPSEA_PREDITIONS/infile.fasta.out> <PATH_TO_MAT_FILE_WITH_TARGETS> <PATH_TO_OUTPUT_CSV>

```
