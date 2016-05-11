'''
Brent Martin 22/4/2016
Routines for manipulating images. 
Not called by the other code, but may be a useful starting point
if you wish to perform other image manipulations in python.
'''
import matplotlib.image as mpimg
from PIL import Image
import os
dir = 'C:/Brent/DATA/Teaching/COSC401/Cacophony videos/Possums/002_possum/'
filename = dir + 'possum_000.'

TEMP_FILE = 'c:/brent/temp/image.png'

def shrink(in_filename, out_filename):
	''' Creates a shrunk version of the images 1/10th of the original size '''
	img = mpimg.imread(in_filename)
	# Underlying raw PIL can only deal with png files, so convert it to png first
	mpimg.imsave(TEMP_FILE, img)
	mpimg.thumbnail(TEMP_FILE, out_filename, 0.1)

	
def shrink_images(in_folder, out_folder):
	'''
	Loops through a directory of image folders
	shrinking the image files within and saving to the destination folder
	'''
	for dir in os.listdir(in_folder):
		print dir
		in_dir = in_folder + "/" + dir
		out_dir = out_folder + "/" + dir
		if not os.path.exists(out_dir): os.makedirs(out_dir)
		for file in os.listdir(in_dir):
			in_file = in_dir + "/" + file
			out_file = out_dir + "/" + file.split('.')[0] + ".png"
			print "   converting " + in_file + " to " + out_file + "..."
			shrink(in_file, out_file)

shrink_images('C:/Brent/DATA/Teaching/COSC401/Images_large', 'C:/Brent/DATA/Teaching/COSC401/Images_small')
