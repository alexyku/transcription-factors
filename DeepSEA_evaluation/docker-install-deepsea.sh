# Install DeepSEA trained models for prediction using Docker
docker pull haoyangz/deepsea-predict-docker

# Test to make sure it works
docker run -v $(pwd)/output:/output --rm haoyangz/deepsea-predict-docker python rundeepsea.py examples/deepsea/example.fasta /test-DeepSEA-output

rm -r test-DeepSEA-output