"WORKS WELL!!!!.. IT's a beauty"
from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import scipy.misc
import random
import numpy as np

class Autopilot_128_class(nn.Module):
	def __init__(self, fc1_size, prob):
		super(Autopilot_128_class, self).__init__()
		# Default bias for all is true
		self.conv1 = nn.Sequential(nn.Conv2d(3, 24, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv2 = nn.Sequential(nn.Conv2d(24, 36, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv3 = nn.Sequential(nn.Conv2d(36, 48, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv4 = nn.Sequential(nn.Conv2d(48, 64, kernel_size = 5, stride = 1), nn.ReLU())
		self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1), nn.ReLU())
		#self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1), nn.ReLU())
		self.fc1 = nn.Sequential(nn.Linear(fc1_size, 2048), nn.ReLU(), nn.Dropout(prob))
		self.fc2 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(prob))
		self.fc3 = nn.Sequential(nn.Linear(512, 128),nn.ReLU(), nn.Dropout(prob))
		self.fc4 = nn.Sequential(nn.Linear(128, 32),nn.ReLU(), nn.Dropout(prob))
		self.fc5 = nn.Sequential(nn.Linear(32, 10),nn.ReLU(), nn.Dropout(prob))
		self.fc6 = nn.Sequential(nn.Linear(10, 1))
	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		#out = self.conv6(out)
		#flatten layer
		out_size = list(out.size())
		#print(out_size)
		# out_size[0] -->batch_size or test size as desired
		out = out.reshape(out_size[0], -1)
		# print(out.size())
		out = self.fc1(out)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		out = self.fc5(out)
		out = self.fc6(out)
		out = torch.mul(torch.atan(out),2)
		return out

class Autopilot_256_class(nn.Module):
	def __init__(self, fc1_size, prob):
		super(Autopilot_256_class, self).__init__()
		# Default bias for all is true
		self.conv1 = nn.Sequential(nn.Conv2d(3, 24, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv2 = nn.Sequential(nn.Conv2d(24, 36, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv3 = nn.Sequential(nn.Conv2d(36, 48, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv4 = nn.Sequential(nn.Conv2d(48, 64, kernel_size = 5, stride = 2), nn.ReLU())
		self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1), nn.ReLU())
		self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1), nn.ReLU())
		self.fc1 = nn.Sequential(nn.Linear(fc1_size, 2048), nn.ReLU(), nn.Dropout(prob))
		self.fc2 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(prob))
		self.fc3 = nn.Sequential(nn.Linear(512, 128),nn.ReLU(), nn.Dropout(prob))
		self.fc4 = nn.Sequential(nn.Linear(128, 32),nn.ReLU(), nn.Dropout(prob))
		self.fc5 = nn.Sequential(nn.Linear(32, 10),nn.ReLU(), nn.Dropout(prob))
		self.fc6 = nn.Sequential(nn.Linear(10, 1))
	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		out = self.conv6(out)
		#flatten layer
		out_size = list(out.size())
		#print(out_size)
		# out_size[0] -->batch_size or test size as desired
		out = out.reshape(out_size[0], -1)
		# print(out.size())
		out = self.fc1(out)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		out = self.fc5(out)
		out = self.fc6(out)
		out = torch.mul(torch.atan(out),2)
		return out

def autopilot_model_loader(img_size):
	# device_str = "cuda:"+str(device_id)
	#d = torch.device('cuda:1')
	#print("dkn k v")
	if img_size == 128:
		fc1_size_128 = 64*7*7
		drop_prob = 0
		N_model_128 = Autopilot_128_class(fc1_size_128, drop_prob)
		checkpoint = torch.load('/home/harshitha/Autopilot_pytorch/python3_codes/128/model_autopilot_after_test.ckpt', map_location=lambda storage, location: storage)
		N_model_128.load_state_dict(checkpoint)

		for param in N_model_128.parameters():
			param.requires_grad = False

		return N_model_128

	if img_size == 256:
		#print("1")
		fc1_size_256 = 64*9*9
		drop_prob = 0
		N_model_256= Autopilot_256_class(fc1_size_256, drop_prob)
		#print("declared")
		checkpoint = torch.load('/home/harshitha/Autopilot_pytorch/python3_codes/256/model_autopilot_after_test.ckpt', map_location=lambda storage, location: storage)
		#print("chk loaded")
		N_model_256.load_state_dict(checkpoint)
		#print("dict loaded")

		for param in N_model_256.parameters():
			param.requires_grad = False

		return N_model_256

	else:
		print("Sorry .. Model unavailable for this size.. Train it yourself !!")
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n = autopilot_model_loader(256,1).to(device)
full_image = scipy.misc.imread("/home/harshitha/o_sully_zurich_m/driving_dataset_full/" + str(313) + ".jpg", mode="RGB")
full_image = scipy.misc.imread("/home/harshitha/cycle_gan_pytorch_3_om/pytorch_cyclgan_my_loss/results/pls_regress_only_128_200_pretrain160epoch/test_latest/images/1016_real_A.png", mode= "RGB")

image = np.rollaxis(scipy.misc.imresize(full_image[-150:], [128, 128]) / 255.0, -1, 0)
image = np.reshape(image, (1, 3, 128, 128))
img_torch = torch.from_numpy(image).float().to(device)
r = n(img_torch)
print(r)#just round off value here
print(r.item())
'''
