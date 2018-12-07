import numpy as np
from scipy.io import loadmat
import glob
import os
import sys

path = "../results/frame_features/"

def max_min_conv(video):
    stat_desc  = np.loadtxt(video, delimiter=',')
    dyn_desc = stat_desc[3:len(stat_desc),:] - stat_desc[0:len(stat_desc)-3,:]
    max_stat = np.amax(stat_desc, axis=0)
    min_stat = np.amin(stat_desc, axis=0)
    max_dyn = np.amax(dyn_desc, axis=0)
    min_dyn = np.amin(dyn_desc, axis=0)
    final_t1 = np.hstack([max_stat, min_stat, max_dyn, min_dyn])
    return final_t1

for video in os.listdir(path):

    desc = []

    features = max_min_conv(os.path.join(path, video))
    desc = np.hstack([desc, features.ravel()])
  
    np.savetxt('../results/video_descriptors/'+video, desc, delimiter=',')
