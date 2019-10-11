file_orig_names = []
degrees_orig = []

file_train_names = []
train_degrees = []

file_val_names = []
val_degrees = []


with open('/home/harshitha/o_sully_zurich_m/full_orig_data_sully_chen/full/data.txt') as f:
    data = f.readlines()
    for line in data:
        file_orig_names.append(line.strip().split(" ")[0][0:-4])
        degrees_orig.append(line.strip().split(" ")[1])

with open('/home/harshitha/cycle_gan_new_code/pytorch-CycleGAN-and-pix2pix/results/om/filenames_train.txt') as f:
    data_train = f.readlines()
    for line in data_train:
        file_train_names.append(line.strip().split("\n")[0][0:-11])

with open('/home/harshitha/cycle_gan_new_code/pytorch-CycleGAN-and-pix2pix/results/om/filenames_val.txt') as f:
    data_val = f.readlines()
    for line in data_val:
        # print(line.strip().split("\n")[0][0:-11])
        file_val_names.append(line.strip().split("\n")[0][0:-11])

intersection_orig_train_files = set(file_orig_names).intersection(file_train_names)
intersection_orig_val_files = set(file_orig_names).intersection(file_val_names)

# Index of common occurence
indices_orig_train = [file_orig_names.index(x) for x in intersection_orig_train_files]
indices_orig_val = [file_orig_names.index(x) for x in intersection_orig_val_files]

print("Intersection and Indicing complete")


import operator

# operator.itemgetter(*index_list)(source_input_list)

# Final train and test degrees
degree_train = operator.itemgetter(*indices_orig_train)(degrees_orig)
degree_val = operator.itemgetter(*indices_orig_val)(degrees_orig)

# Final common train and val names
files_train = intersection_orig_train_files
files_val = intersection_orig_val_files


# combine both listsand write to file
train_list = [files_train, degree_train]
val_list = [files_val, degree_val] 

print("Commencing writing to files..")

# Write to respective text files
with open("/home/harshitha/cycle_gan_new_code/pytorch-CycleGAN-and-pix2pix/results/om/train_data.txt", "w") as file:
    for x in zip(*train_list):
        file.write("{0}\t{1}\n".format(*x))

with open("/home/harshitha/cycle_gan_new_code/pytorch-CycleGAN-and-pix2pix/results/om/val_data.txt", "w") as file:
    for x in zip(*val_list):
        file.write("{0}\t{1}\n".format(*x))

print("Writing Completed !!")

