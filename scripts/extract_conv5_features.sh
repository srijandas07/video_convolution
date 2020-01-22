#!/bin/bash

mkdir -p ../results
mkdir -p ../results/frame_features
mkdir -p ../results/video_features
while IFS='' read -r line || [[ -n "$line" ]]; do
	echo "Processing File: $line"
	python extract_conv5_features.py --data_location $line --output_location $2 --model_type $3
	echo "$line conv features generated!"
done <"$1"
python max_min_pooling.py
echo "Done Successfully"
