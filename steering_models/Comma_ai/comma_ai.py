# Code implements Comma ai ..no changes (https://github.com/udacity/self-driving-car/blob/d69a77a0a1e81c486afa5c30aba09c5e59d6e4d6/steering-models/community-models/rambo/train.py)

from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import scipy.misc
import random
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


# Preprocess data

dim1 = 128
dim2 = 128
channels = 3

xs_train = []
ys_train = []
#read 	train_data.txt
with open("/home/harshitha/o_sully_zurich_m/full_orig_data_sully_chen/steering_model_train_test/train_data.txt") as f:
	for line in f:
		xs_train.append("/home/harshitha/o_sully_zurich_m/full_orig_data_sully_chen/steering_model_train_test/train/" + line.split()[0])
		ys_train.append(float(line.split()[1]) * scipy.pi / 180)
        
#get number of images
num_images_total_train = len(xs_train)

train_X = xs_train[:int(num_images_total_train)]
train_Y = ys_train[:int(num_images_total_train)]


#  Validation

xs_val = []
ys_val = []
#read valdata.txt
with open("/home/harshitha/o_sully_zurich_m/full_orig_data_sully_chen/steering_model_train_test/val_data.txt") as f:
	for line in f:
		xs_val.append("/home/harshitha/o_sully_zurich_m/full_orig_data_sully_chen/steering_model_train_test/val/" + line.split()[0])
		ys_val.append(float(line.split()[1]) * scipy.pi / 180)

#get number of images
num_images_total_val = len(xs_val)
val_X = xs_val[:int(num_images_total_val)]
val_Y = ys_val[:int(num_images_total_val)]

num_train_images = int(num_images_total_train)
num_val_images = int(num_images_total_val)


# Helper Function

def unison_shuffled_copies(a, b):
	z = list(zip(a, b))
	random.shuffle(z)
	a, b = zip(*z)
	return a,b

def LoadTrainBatch(batch_size, train_xs, train_ys, train_batch_pointer):
	x_out = np.zeros((batch_size, channels, dim1, dim2))
	y_out = np.zeros((batch_size, 1))
	for i in range(0, batch_size):
		# x_out[i,...] = np.rollaxis(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [dim1, dim2]) / 255.0, -1, 0)
		# Normalize
		x_out[i,...] = np.rollaxis(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [dim1, dim2]) /127.5 -1.0, -1, 0)

		y_out[i,...] = train_ys[(train_batch_pointer + i) % num_train_images]
	train_batch_pointer += batch_size
	return x_out, y_out, train_batch_pointer

def LoadValBatch(batch_size, val_xs, val_ys, val_batch_pointer):
	x_out = np.zeros((batch_size, channels, dim1, dim2))
	y_out = np.zeros((batch_size, 1))
	for i in range(0, batch_size):
		# x_out[i,...] = np.rollaxis(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [dim1, dim2]) / 255.0, -1, 0)
		# Normalize
		x_out[i,...] = np.rollaxis(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [dim1, dim2]) /127.5 -1.0, -1, 0)

		y_out[i,...] = val_ys[(val_batch_pointer + i) % num_val_images]
	val_batch_pointer += batch_size
	return x_out, y_out, val_batch_pointer


def weight_init(m):
	if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
		torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')


###########################################################################

# Optimizer and Decay
fc1_size = 64*5*5 #5184
# probability of dropping is 0.2
prob = 0.2
learning_rate = 1e-4
batch_size = 768
epochs = 500

model = Comma_ai(fc1_size, prob).to(device)
model.apply(weight_init)

# https://stats.stackexchange.com/questions/324616/can-weight-decay-be-higher-than-learning-rate
# weight decay is for the weight penalty
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.001)

total_steps = int(len(train_Y)/batch_size)
total_val_steps = int(num_val_images/batch_size)

train_size = len(train_Y)

f_train = open("loss_train.txt","w+")
f_val = open("loss_val.txt","w+")

##### ------------------------------------------------------------------------------------------------------ ####

for epoch in range(epochs):
	train_X, train_Y = unison_shuffled_copies(train_X, train_Y)
	train_batch_pointer = 0

	for step in range(total_steps):
		x, y, train_batch_pointer = LoadTrainBatch(batch_size, train_X, train_Y, train_batch_pointer)

		x_torch = torch.from_numpy(x).float().to(device)
		y_torch = torch.from_numpy(y).float().to(device)
		pred_angle = model(x_torch)
		loss = criterion(pred_angle, y_torch)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (step) % 10 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, epochs, step+1, total_steps, loss.item()))
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, epochs, step+1, total_steps, loss.item()),file = f_train)
	
	##### ------------------------------------------------------------------------------------------------------ ####
	if epoch % 10 == 0:
		# Load everything at once and the same set
		val_batch_pointer = 0 
		x, y, val_batch_pointer = LoadValBatch(num_val_images, val_X, val_Y, val_batch_pointer)

		x_torch = torch.from_numpy(x).float().to(device)
		y_torch = torch.from_numpy(y).float().to(device)
		pred_angle = model(x_torch)
		loss = criterion(pred_angle, y_torch)
		print('************************************************************************')
		print('Epoch [{}/{}] : MSE loss : {} '.format(epoch+1, epochs,loss.item()))
		print('************************************************************************')
		print('Epoch [{}/{}] : MSE loss : {} '.format(epoch+1, epochs,loss.item()), file = f_val)

		file_name = 'model_comma_ai_no_test_epoch_'+str(epoch)+'.ckpt'
		full_model_name = 'full_model_no_test_epoch_'+str(epoch)
		torch.save(model.state_dict(), file_name)
		torch.save(model,full_model_name)


f_train.close()
f_val.close()
# Test the model
# In test phase, we don't need gradients
with torch.no_grad():
	# Load everything at once
	val_batch_pointer = 0 
	x, y, val_batch_pointer = LoadValBatch(num_val_images, val_X, val_Y, val_batch_pointer)
	# print(y.shape)

	x_torch = torch.from_numpy(x).float().to(device)
	y_torch = torch.from_numpy(y).float().to(device)
	pred_angle = model(x_torch)
	loss = criterion(pred_angle, y_torch)

	print('MSE loss of the network on the test images: {} '.format(loss.item()))

torch.save(model.state_dict(), 'model_comma_ai_after_test.ckpt')
torch.save(model,'full_model_after_test')