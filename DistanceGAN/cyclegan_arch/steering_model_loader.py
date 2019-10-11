'''
Author: Harshitha Machiraju
Description: File with functions to recall saved steering models 
Note : Change the path to the saved models in the loader functions if required
'''

from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import scipy.misc
import random
import numpy as np


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
		out = out.view(out_size[0], -1)
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
		out = out.view(out_size[0], -1)
		# print(out.size())
		out = self.fc1(out)
		out = self.fc2(out)
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
