from sewar.full_ref import mse, psnr, ssim
import scipy.misc
from os import listdir
from os.path import isfile, join
import numpy as np

# Assuming the images occur in pairs..
# Assuming the images occur in pairs..
# Assuming the images occur in pairs..
normal_images_dir = '/home/harshitha/o_sully_zurich_m/gan_train/testA/' #normal weather test samples
# Cyclegan normal comparison
# first_images_dir = '/home/harshitha/cycle_gan_new_code/pytorch-CycleGAN-and-pix2pix/results/sully_to_fog_normal_128_load_128_crop_128_lambda_3/test_latest/images/'
# foggy_images_dir = '/home/harshitha/cycle_gan_new_code/pytorch-CycleGAN-and-pix2pix/results/sully_to_fog_normal_128_load_128_crop_128_lambda_3/test_latest/images/'
# foggy_images_dir = '/home/harshitha/cycle_gan_new_code/pytorch-CycleGAN-and-pix2pix/results/om/small_test_set/images/'

# foggy_images_dir = '/home/harshitha/cycle_gan_new_code/pytorch-CycleGAN-and-pix2pix/results/om/test_latest/images/' #foggy weather samples

foggy_images_dir = '/home/harshitha/DistanceGAN_comma/results/comma_fool_om/test_30/images/' #foggy weather samples
first_images_dir = '/home/harshitha/DistanceGAN/results/256_load_128_crop_A_no_self_with_cycleloss/test_25/images/'


test_image_names = [f for f in listdir(normal_images_dir) if isfile(join(normal_images_dir, f))]
diff_mse_list = []
diff_psnr_list = []
diff_ssim_list = []


counter = 0
for img_name in test_image_names:

	img_name = img_name[0:-4] # for real vs foggy
	# img_name = img_name[0:-19]
	# print(img_name)

	normal_image = scipy.misc.imread(first_images_dir + str(img_name)+'_fake_B.png', mode="RGB")
	foggy_image = scipy.misc.imread(foggy_images_dir + str(img_name)+'_fake_B.png', mode="RGB")

	diff_mse = mse(normal_image, foggy_image)
	diff_ssim = ssim(normal_image, foggy_image)
	diff_psnr = psnr(normal_image, foggy_image)

	diff_mse_list +=[diff_mse]
	diff_psnr_list += [diff_psnr]
	diff_ssim_list += [diff_ssim]

# print(diff_mse_list)
print(diff_ssim_list)
# print(diff_psnr_list)

#Get average, max and min
print("MSE stats: ")
print(np.mean(np.array(diff_mse_list)))
print(max(diff_mse_list))
print(min(diff_mse_list))
print(np.std(np.array(diff_mse_list)))
print("####"*10)

print("SSIM stats")
print(np.mean(np.array(diff_ssim_list)))
print(max(diff_ssim_list))
print(min(diff_ssim_list))
print(np.std(np.array(diff_ssim_list)))
print("####"*10)

print("PSNR stats")
print(np.mean(np.array(diff_psnr_list)))
print(max(diff_psnr_list))
print(min(diff_psnr_list))
print(np.std(np.array(diff_psnr_list)))
print("####"*10)
