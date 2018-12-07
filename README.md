Action Recognition
Video analysis using frame-level convolution follwed by mx-min aggregation.

Requirement - 
Python 2.7
Keras 2.0 with tensorflow 1.5.0 in the back-end.

This script generates conv features from last fully connected layers of the network like
VGG-16 and ResNet-152 for a video and saves them in .csv.gz file.
It also computes the max_min pooling over the time frames to decibe the videos, we call video_descriptors.

First, download the pre-trained ResNet-152 weights from https://drive.google.com/uc?id=0Byy2AcGyEVxfeXExMzNNOHpEODg&export=download
at ./models/

Go to scripts and enter python extract_conv5_features.py -h to understand the input arguments to the script.
you can use extract_conv5_features.sh to execute the python script directly by resolving the dependencies.

Example- 
sh extract_conv5_features.sh filelist.txt ../results/frame_features/ VGG-16
                             OR
sh extract_conv5_features.sh filelist.txt ../results/frame_features/ ResNet-152


filelist containt the full location of the videos which is the input to the shell script.
For your dataset - 
Create the filelist.txt having the location of videos and then use
sh extract_conv5_features.sh --location_filelist.txt --type_of_model

from scripts directory

Enjoy !!!
