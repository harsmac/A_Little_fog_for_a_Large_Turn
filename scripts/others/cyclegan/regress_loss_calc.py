from __future__ import print_function
import autopilot_loader
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import scipy.misc
import random
import numpy as np
from os import listdir
from os.path import isfile, join
import random
import matplotlib.pyplot as plt


def preprocess(full_image):
	image = np.rollaxis(scipy.misc.imresize(full_image[-150:], [128, 128]) / 255.0, -1, 0)
	image = np.reshape(image, (1, 3, 128, 128))
	img_torch = torch.from_numpy(image).float().to(device)
	return img_torch

####################################################################################################

# Load the model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = autopilot_loader.autopilot_model_loader(128).to(device)
# Can be done on batch but will do on one single pair first
loss = torch.nn.L1Loss()

diff_arr = np.zeros((3774,1))
f_train = open("cycle_angles.txt","w+")
####################################################################################################

# Assuming the images occur in pairs..
normal_images_dir = '/home/harshitha/scripts/IQA_full_train/cycle_plus_regress/real_A/' #normal weather test samples
foggy_images_dir = '/home/harshitha/scripts/IQA_full_train/cycle_plus_regress/fake_B/' #foggy weather samples


test_image_names = [f for f in listdir(normal_images_dir) if isfile(join(normal_images_dir, f))]

diff_list = []
counter = -1
diff_counter = 0


for img_name in test_image_names:
	
	counter = counter + 1

	img_name = img_name[0:-11]

	normal_image = scipy.misc.imread(normal_images_dir + str(img_name)+'_real_A.png', mode="RGB")
	foggy_image = scipy.misc.imread(foggy_images_dir + str(img_name)+'_fake_B.png', mode="RGB")


	normal_img_torch = preprocess(normal_image)
	foggy_img_torch = preprocess(foggy_image)

	angle_normal = N(normal_img_torch)
	angle_foggy = N(foggy_img_torch)
	diff = loss(angle_normal, angle_foggy)
	# print(img_name, diff, angle_normal, angle_foggy)
	diff_arr[counter] = diff
	if diff>0.5:
		#print(img_name)
		#print(diff)
		diff_counter = diff_counter + 1
		print ('image name [{}], angle_normal : {}, angle_fog : {}, diff: {:.4f}' .format(img_name, angle_normal, angle_foggy, diff),file = f_train)


print(diff_counter)
print("Mean: ", np.mean(diff_arr))
print("std: ", np.std(diff_arr))

f_train.close()
'''bin_list = []
i=0
while i<3.3:
	bin_list += [i]
	i = i+0.3
print(bin_list)
plt.hist(diff_list,bins=30,range=(0,3),cumulative=True)
plt.ylabel('Probability')
plt.show()'''
