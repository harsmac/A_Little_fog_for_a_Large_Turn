#!/usr/bin/python
from PIL import Image
import os, sys

path = "/home/harshitha/o_sully_zurich_m/trainA_big/"
dest_path = "/home/harshitha/o_sully_zurich_m/trainA_small/"
dirs = os.listdir( path )

def resize():
	for item in dirs:
		if os.path.isfile(path+item):
			im = Image.open(path+item)
			f = item[:-4] # remove extension
			imResize = im.resize((128,128), Image.ANTIALIAS)
			imResize.save(dest_path + f +  '_resized.jpg', 'JPEG', quality=90)

resize()