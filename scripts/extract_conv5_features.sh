#!/bin/bash
export PATH=/home/sdas/anaconda2/bin:$PATH
module load cuda/8.0 cudnn/5.1-cuda-8.0
export PYTHONPATH=/data/stars/user/sdas/PhD_work/conv_features/models/:$PYTHONPATH
while IFS='' read -r line || [[ -n "$line" ]]; do
	echo "Processing File: $line"
	python /data/stars/user/sdas/PhD_work/conv_features/scripts/extract_conv5_features.py --data_location $line --output_location $2 --model_type $3
	echo "$line conv features generated!"
done <"$1"
python max_min_pooling.py
echo "Done Successfully"
