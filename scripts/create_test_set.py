import random
import shutil

xs = []
num_test = 1024

normal_img_dir = "/home/harshitha/o_sully_zurich_m/driving_dataset_full/"

with open(normal_img_dir + "data.txt") as f:
	for line in f:
		xs.append(normal_img_dir + line.split()[0])

test_X = random.sample(xs, num_test)

# Print list of test files 
with open('test_set.txt', 'w') as f:
	for item in test_X:
		f.write("%s\n" % item)

for t in test_X:
	shutil.copy(t, '/home/harshitha/o_sully_zurich_m/testA/')

