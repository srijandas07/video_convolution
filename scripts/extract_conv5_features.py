import sys
from argparse import ArgumentParser
argv = sys.argv
# Parse arguments                                                                                                                                                                        
parser = ArgumentParser(description='Helps getting things done.')
# parser.add_argument('-v', '--verbose', action='store_true', help='run verbose')                                                                                                        
parser.add_argument('--data_location', help='Location of input video files')
parser.add_argument('--output_location', help='Location of output video files')
parser.add_argument('-model_type', '--model_type', help='VGG16, ResNet-152', default='ResNet-152')
args = parser.parse_args(argv[1:])

import numpy as np, pandas as pd, os, sys
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras.models import Model

if args.model_type == 'ResNet-152':
   from resnet152 import resnet152_model
   weights_path = '../models/resnet152_weights_tf.h5'
   model = resnet152_model(weights_path)
   model = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
   f_size = 2048
else :
   from keras.applications.vgg16 import VGG16
   model = VGG16(weights='imagenet', include_top=True)
   model = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
   f_size = 4096
                                           
file_path = args.data_location                           
n_files = sorted(os.listdir(file_path))
f_p = os.path.basename(os.path.normpath(file_path))
batch_size = 64
batch_size_imgs = 250
k = 0

images = []
features_conv5 = []

def data_load_batch(k, batch_size_imgs):
    t_file = []
    for img_file in n_files[0+k:batch_size_imgs+k]:
        img_path = file_path+'/'+img_file
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        t_file.append(preprocess_input(x))
    t_file = np.asarray(t_file)
    return t_file

for i in range(0, int(len(n_files)/batch_size_imgs)):
    images = data_load_batch(k, batch_size_imgs)
    features_conv5.append(model.predict(images, batch_size=batch_size))
    k = k + batch_size_imgs
features_conv5 = np.asarray(features_conv5)
features_conv5 = np.reshape(features_conv5, [int(len(n_files)/batch_size_imgs*batch_size_imgs), f_size])
images_remain = len(n_files) - k
features_conv5_remain = []
if images_remain > 0:
   images = data_load_batch(k, images_remain)
   features_conv5_remain.append(model.predict(images, batch_size=batch_size))
features_conv5_remain = np.asarray(features_conv5_remain)
features_conv5_remain = np.squeeze(features_conv5_remain)
features_conv5 = np.vstack([features_conv5, features_conv5_remain])

np.savetxt(args.output_location+f_p+'.csv.gz', features_conv5, delimiter=',')
