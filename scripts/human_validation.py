import numpy as np
import shutil, os

train_img_dir = '/home/harshitha/o_sully_zurich_m/trainA/'
dest_folder_1 = '/home/harshitha/o_sully_zurich_m/human_tests/batch_1/'
dest_folder_2 = '/home/harshitha/o_sully_zurich_m/human_tests/batch_2/'
dest_folder_3 = '/home/harshitha/o_sully_zurich_m/human_tests/batch_3/'
dest_folder_4 = '/home/harshitha/o_sully_zurich_m/human_tests/batch_4/'


f = open('/home/harshitha/o_sully_zurich_m/data_trainA.txt', 'r')
img_list = f.read().splitlines()
# print img_list
f.close()


random_img_array_full = np.random.choice(img_list, 50*4, replace=False)

arr_1 = random_img_array_full[0:50]
arr_2 = random_img_array_full[50:100]
arr_3 = random_img_array_full[100:150]
arr_4 = random_img_array_full[150:200]


for f in arr_1:
	f_add = train_img_dir+f
	print(f_add)
	shutil.copy(f_add, dest_folder_1)

for f in arr_2:
	f_add = train_img_dir+f
	print(f_add)
	shutil.copy(f_add, dest_folder_2)

for f in arr_3:
	f_add = train_img_dir+f
	print(f_add)
	shutil.copy(f_add, dest_folder_3)

for f in arr_4:
	f_add = train_img_dir+f
	print(f_add)
	shutil.copy(f_add, dest_folder_4)
