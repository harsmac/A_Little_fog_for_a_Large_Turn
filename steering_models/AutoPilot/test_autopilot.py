from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import scipy.misc
import random
import numpy as np

device = 'cuda:0'


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


# Preprocess data

dim1 = 128
dim2 = 128
channels = 3
criterion = nn.MSELoss()
xs_train = []
ys_train = []
#read data.txt
with open("/home/harshitha/o_sully_zurich_m/full_orig_data_sully_chen/steering_model_train_test/train_data.txt") as f:
	for line in f:
		xs_train.append("/home/harshitha/o_sully_zurich_m/full_orig_data_sully_chen/steering_model_train_test/train/" + line.split()[0])
		ys_train.append(float(line.split()[1]) * scipy.pi / 180)
        
#get number of images
num_train_images = len(xs_train)
train_X = xs_train[:int(num_train_images)]
train_Y = ys_train[:int(num_train_images)]
batch_size = 1
total_steps = int(len(train_Y)/batch_size) 


def LoadTrainBatch(batch_size, train_xs, train_ys, train_batch_pointer):
	x_out = np.zeros((batch_size, channels, dim1, dim2))
	y_out = np.zeros((batch_size, 1))
	for i in range(0, batch_size):
		x_out[i,...] = np.rollaxis(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [dim1, dim2]) /127.5 -1.0, -1, 0)
		y_out[i,...] = train_ys[(train_batch_pointer + i) % num_train_images]
	train_batch_pointer += batch_size
	return x_out, y_out, train_batch_pointer


model = autopilot_model_loader().to(device)
mse = []

with torch.no_grad():
	# Load everything at once
	batch_pointer = 0 
	for step in range(total_steps):
		x, y, batch_pointer = LoadTrainBatch(batch_size, train_X, train_Y, batch_pointer)
		x_torch = torch.from_numpy(x).float().to(device)
		y_torch = torch.from_numpy(y).float().to(device)
		pred_angle = model(x_torch)
		loss = criterion(pred_angle, y_torch)
		mse+=[loss]
		if step % 100 ==0:
			print(str(step)+" steps are done")
		# print('MSE loss of the network on the test images: {} '.format(loss.item()))

# print(mse)
print(sum(mse)/float(len(mse)))


