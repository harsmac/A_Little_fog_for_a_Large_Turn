from autopilot_loader import *
N = autopilot_model_loader(128)

device = 0
N.cuda(0)
###########################################################################
# Preprocess data
xs = []
ys = []
dim1 = 128
dim2 = 128
channels = 3

#read data.txt
with open("/home/harshitha/o_sully_zurich_m/driving_dataset_full/data.txt") as f:
	for line in f:
		xs.append("/home/harshitha/o_sully_zurich_m/driving_dataset_full/" + line.split()[0])
# with open("/Users/harshithamachiraju/Documents/AV_Explainability/code/steering_angle/driving_dataset/data.txt") as f:
# 	for line in f:
# 		xs.append("/Users/harshithamachiraju/Documents/AV_Explainability/code/steering_angle/driving_dataset/" + line.split()[0])
 		#the paper by Nvidia uses the inverse of the turning radius,
		#but steering wheel angle is proportional to the inverse of turning radius
		#so the steering wheel angle in radians is used as the output
		ys.append(float(line.split()[1]) * scipy.pi / 180)

#get number of images
num_images_total = len(xs)

#shuffle list of images
# c = list(zip(xs, ys))
# random.shuffle(c)
# xs, ys = zip(*c)

train_X = xs[:int(num_images_total * 0.8)]
train_Y = ys[:int(num_images_total * 0.8)]

val_X = xs[-int(num_images_total * 0.2):]
val_Y = ys[-int(num_images_total * 0.2):]

num_train_images = int(num_images_total * 0.8)
num_val_images = int(num_images_total * 0.2)

def unison_shuffled_copies(a, b):
	z = list(zip(a, b))
	random.shuffle(z)
	a, b = zip(*z)
	return a,b

def LoadTrainBatch(batch_size, train_xs, train_ys, train_batch_pointer):
	x_out = np.zeros((batch_size, channels, dim1, dim2))
	y_out = np.zeros((batch_size, 1))
	for i in range(0, batch_size):
		x_out[i,...] = np.rollaxis(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [dim1, dim2]) / 255.0, -1, 0)
		y_out[i,...] = train_ys[(train_batch_pointer + i) % num_train_images]
	train_batch_pointer += batch_size
	return x_out, y_out, train_batch_pointer

def LoadValBatch(batch_size, val_xs, val_ys, val_batch_pointer):
	x_out = np.zeros((batch_size, channels, dim1, dim2))
	y_out = np.zeros((batch_size, 1))
	for i in range(0, batch_size):
		x_out[i,...] = np.rollaxis(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [dim1, dim2]) / 255.0, -1, 0)
		y_out[i,...] = val_ys[(val_batch_pointer + i) % num_val_images]
	val_batch_pointer += batch_size
	return x_out, y_out, val_batch_pointer


def weight_init(m):
	if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
		torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

###########################################################################

fc1_size = 64*7*7 #5184
# probability of dropping is 0.2
prob = 0.2
learning_rate = 1e-4
batch_size = 32 # since images r smaller
epochs = 100

# https://stats.stackexchange.com/questions/324616/can-weight-decay-be-higher-than-learning-rate
# weight decay is for the weight penalty
criterion = nn.MSELoss()

total_steps = int(len(train_Y)/batch_size)
total_val_steps = int(num_val_images/batch_size)

train_size = len(train_Y)

val_batch_pointer = 0 
x, y, val_batch_pointer = LoadValBatch(total_val_steps, val_X, val_Y, val_batch_pointer)

x_torch = torch.from_numpy(x).float().cuda(device)
y_torch = torch.from_numpy(y).float().cuda(device)
pred_angle = N(Variable(x_torch))
loss = criterion(pred_angle, Variable(y_torch))

print('MSE loss of the network on the test images: {} '.format(loss.data[0]))

