from __future__ import print_function
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


# Add model architectures
class AutoPilot(nn.Module):
	def __init__(self, fc1_size, prob):
		super(AutoPilot, self).__init__()
		# Default bias for all is true
		self.conv1 = nn.Sequential(nn.Conv2d(3, 24, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv2 = nn.Sequential(nn.Conv2d(24, 36, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv3 = nn.Sequential(nn.Conv2d(36, 48, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv4 = nn.Sequential(nn.Conv2d(48, 64, kernel_size = 3, stride = 1), nn.ReLU())
		self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1), nn.ReLU())
		# To bring down the size to 7x7 one extra layer as mentioned
		self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1), nn.ReLU())

		self.fc1 = nn.Sequential(nn.Linear(fc1_size, 1164), nn.ReLU(), nn.Dropout(prob))
		self.fc2 = nn.Sequential(nn.Linear(1164, 100), nn.ReLU(), nn.Dropout(prob))
		self.fc3 = nn.Sequential(nn.Linear(100, 50),nn.ReLU(), nn.Dropout(prob))
		self.fc4 = nn.Sequential(nn.Linear(50, 10),nn.ReLU(), nn.Dropout(prob))
		self.fc5 = nn.Sequential(nn.Linear(10, 1))

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		out = self.conv6(out)
		#flatten layer
		out_size = list(out.size())
# 		print(out_size)
		# out_size[0] -->batch_size or test size as desired
		out = out.reshape(out_size[0], -1)
# 		print(out.size())
		out = self.fc1(out)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		out = self.fc5(out)
		out = torch.mul(torch.atan(out),2)
		return out

class Comma_ai(nn.Module):
	def __init__(self, fc1_size, prob):
		super(Comma_ai, self).__init__()
		# Default bias for all is true
		self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size = 8, stride = 4), nn.ReLU())
		self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 5, stride = 2), nn.ReLU())
	
		self.fc1 = nn.Sequential(nn.Linear(fc1_size, 512), nn.ReLU())
		self.fc2 = nn.Sequential(nn.Linear(512, 1))

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)

		#flatten layer
		out_size = list(out.size())
		# print(out_size)
		# out_size[0] -->batch_size or test size as desired
		out = out.reshape(out_size[0], -1)
		# print(out.size())
		out = self.fc1(out)
		out = self.fc2(out)
		out = torch.mul(torch.atan(out),2)
		return out

class PilotNet(nn.Module):
	def __init__(self, fc1_size, prob):
		super(PilotNet, self).__init__()
		# Default bias for all is true
		self.bn = nn.BatchNorm2d(3)
		self.conv1 = nn.Sequential(nn.Conv2d(3, 24, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv2 = nn.Sequential(nn.Conv2d(24, 36, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv3 = nn.Sequential(nn.Conv2d(36, 48, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv4 = nn.Sequential(nn.Conv2d(48, 64, kernel_size = 3, stride = 1), nn.ReLU())
		self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1), nn.ReLU())
		self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1), nn.ReLU())

		self.fc1 = nn.Sequential(nn.Linear(fc1_size, 100), nn.ReLU(), nn.Dropout(prob))
		self.fc2 = nn.Sequential(nn.Linear(100, 50),nn.ReLU(), nn.Dropout(prob))
		self.fc3 = nn.Sequential(nn.Linear(50, 10),nn.ReLU(), nn.Dropout(prob))
		self.fc4 = nn.Sequential(nn.Linear(10, 1))

	def forward(self, x):
		out = self.bn(x)
		out = self.conv1(out)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)		
		out = self.conv5(out)
		out = self.conv6(out)
		#flatten layer
		out_size = list(out.size())
		# print(out_size)
		# out_size[0] -->batch_size or test size as desired
		out = out.reshape(out_size[0], -1)
		# print(out.size())
		out = self.fc1(out)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		out = torch.mul(torch.atan(out),2)
		return out

# Loader functions

def autopilot_model_loader():
	fc1_size_128 = 64*7*7
	drop_prob = 0
	N_model = AutoPilot(fc1_size_128, drop_prob)
	checkpoint = torch.load('/home/harshitha/steering_models/AutoPilot/model_autopilot_no_test_epoch_390.ckpt', map_location=lambda storage, location: storage)
	N_model.load_state_dict(checkpoint)

	for param in N_model.parameters():
		param.requires_grad = False

	return N_model


def comma_ai_model_loader():
	fc1_size_128 = 64*5*5
	drop_prob = 0
	N_model = Comma_ai(fc1_size_128, drop_prob)
	checkpoint = torch.load('/home/harshitha/steering_models/Comma_ai/model_comma_ai_no_test_epoch_460.ckpt', map_location=lambda storage, location: storage)
	N_model.load_state_dict(checkpoint)

	for param in N_model.parameters():
		param.requires_grad = False

	return N_model

def pilotnet_model_loader():
	fc1_size_128 = 64*7*7
	drop_prob = 0
	N_model = PilotNet(fc1_size_128, drop_prob)
	checkpoint = torch.load('/home/harshitha/steering_models/PilotNet/model_pilotnet_no_test_epoch_280.ckpt', map_location=lambda storage, location: storage)
	N_model.load_state_dict(checkpoint)

	for param in N_model.parameters():
		param.requires_grad = False

	return N_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(full_image):
	image = np.rollaxis(scipy.misc.imresize(full_image[-150:], [128, 128]) / 127.5 -1.0, -1, 0)
	image = np.reshape(image, (1, 3, 128, 128))
	img_torch = torch.from_numpy(image).float().to(device)
	return img_torch

N = autopilot_model_loader().to(device)
# N = comma_ai_model_loader().to(device)

loss = torch.nn.L1Loss()

# diff_arr = np.zeros((1011,1))
diff_arr = np.zeros((3623,1))

f_train = open("cycle_angles_rain.txt","w+")
####################################################################################################

# Assuming the images occur in pairs..
normal_images_dir = '/home/harshitha/o_sully_zurich_m/gan_train_rain/trainA' #normal weather test samples
# Cyclegan normal comparison
# first_images_dir = '/home/harshitha/cycle_gan_new_code/pytorch-CycleGAN-and-pix2pix/results/sully_to_fog_normal_128_load_128_crop_128_lambda_3/test_latest/images/'
# foggy_images_dir = '/home/harshitha/cycle_gan_new_code/pytorch-CycleGAN-and-pix2pix/results/sully_to_fog_normal_128_load_128_crop_128_lambda_3/test_latest/images/'
# foggy_images_dir = '/home/harshitha/cycle_gan_new_code/pytorch-CycleGAN-and-pix2pix/results/om/small_test_set/images/' #foggy weather samples

# first_images_dir = '/home/harshitha/DistanceGAN/results/256_load_128_crop_A_no_self_with_cycleloss/test_25/images/'
foggy_images_dir = '/home/harshitha/cycle_gan_new_code/pytorch-CycleGAN-and-pix2pix/results/rain/test_latest/images/' #foggy weather samples


test_image_names = [f for f in listdir(normal_images_dir) if isfile(join(normal_images_dir, f))]
test_image_names = sorted(test_image_names)
print(len(test_image_names))
diff_list = []
counter = -1
diff_counter = 0

# print(test_image_names)

for img_name in test_image_names:
	
	counter = counter + 1
	# print(img_name)

	img_name = img_name[0:-4]
	# print(img_name)

	normal_image = scipy.misc.imread(foggy_images_dir + str(img_name)+'_real_A.png', mode="RGB")
	foggy_image = scipy.misc.imread(foggy_images_dir + str(img_name)+'_fake_B.png', mode="RGB")


	normal_img_torch = preprocess(normal_image)
	foggy_img_torch = preprocess(foggy_image)

	angle_normal = N(normal_img_torch)
	angle_foggy = N(foggy_img_torch)
	diff = loss(angle_normal, angle_foggy)
	print(img_name, diff, angle_normal, angle_foggy)
	diff_arr[counter] = diff
	# if diff>0.5:
	# 	#print(img_name)
	# 	#print(diff)
	# 	diff_counter = diff_counter + 1
	print ('image name [{}], angle_normal : {}, angle_fog : {}, diff: {:.4f}' .format(img_name, angle_normal, angle_foggy, diff),file = f_train)


print(diff_counter)
print("Mean: ", np.mean(diff_arr),file=f_train)
print("std: ", np.std(diff_arr),file=f_train)
print("Mean: ", np.mean(diff_arr))
print("std: ", np.std(diff_arr))

f_train.close()
