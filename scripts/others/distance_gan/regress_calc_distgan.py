from __future__ import print_function
import autopilot_loader_dist
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
from torch.autograd import Variable

def preprocess(full_image):
	image = np.rollaxis(scipy.misc.imresize(full_image[-150:], [128, 128]) / 255.0, -1, 0)
	image = np.reshape(image, (1, 3, 128, 128))
	img_torch = torch.from_numpy(image).float().cuda(device)
	return img_torch

####################################################################################################

# Load the model

device = 0
N = autopilot_loader_dist.autopilot_model_loader(128)
N.cuda(device)
# Can be done on batch but will do on one single pair first
loss = torch.nn.L1Loss()


####################################################################################################

# Assuming the images occur in pairs..
normal_images_dir = '/home/harshitha/scripts/IQA_full_train/256_load_128_crop_A_no_self_with_cyle_with_regress_theta_0_5_alpha_0_2_epoch_25_____20_plus_5/real_A/' #normal weather test samples
foggy_images_dir = '/home/harshitha/scripts/IQA_full_train/256_load_128_crop_A_no_self_with_cyle_with_regress_theta_0_5_alpha_0_2_epoch_25_____20_plus_5/fake_B/' #foggy weather samples

f_train = open("dist_angles.txt","w+")

test_image_names = [f for f in listdir(normal_images_dir) if isfile(join(normal_images_dir, f))]

diff_arr = np.zeros((3774,1))
counter = -1

for img_name in test_image_names:
	counter = counter + 1

	img_name = img_name[0:-11]

	normal_image = scipy.misc.imread(normal_images_dir + str(img_name)+'_real_A.png', mode="RGB")
	foggy_image = scipy.misc.imread(foggy_images_dir + str(img_name)+'_fake_B.png', mode="RGB")


	normal_img_torch = preprocess(normal_image)
	foggy_img_torch = preprocess(foggy_image)

	angle_normal = N(Variable(normal_img_torch))
	angle_foggy = N(Variable(foggy_img_torch))
	diff = loss(angle_normal, angle_foggy)
	# del normal_img_torch, foggy_img_torch, normal_image, foggy_image

	# print(img_name, diff.data[0], angle_normal.data[0], angle_foggy.data[0])
	diff_arr[counter] = diff.data[0]


	if diff>0.3:
		# print("y")
		# print(img_name, diff.data[0], angle_normal.data[0], angle_foggy.data[0])
		#print(img_name)
		#print(diff.data[0])
		print ('image name [{}], angle_normal : {}, angle_fog : {}, diff: {:.4f}' .format(img_name, angle_normal, angle_foggy, diff.data[0]),file = f_train)


# print(counter)
print(np.mean(diff_arr))
print(np.std(diff_arr))

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
