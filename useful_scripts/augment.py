#script used to augment the door dataset
#in order to run type python augment.py (name of the folder to save the augmented images)

import cv2
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
import re
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import sys
import os

np.random.bit_generator = np.random._bit_generator

def boundingBoxRead(filename):
	file = open(filename)
	bbfile = []

	for line in file:
		#print(line)
		args = line.split(" ", )
		#print(args[len(args) - 1])
		args[len(args) - 1] = re.sub('[^0-9.]','', args[len(args) - 1])

		bbfile.append(args)

	return bbfile
	# print(bbfile)

def convertYolov3BBToImgaugBB(args, img_shape):
    height, width, depth = img_shape
    # for i in args:
    #     print (i)

    oclass = int(args[0])
    x_pos = float(args[1])
    y_pos = float(args[2])
    x_size = float(args[3])
    y_size = float(args[4])

    x1 = x_pos * width - (x_size * width / 2)
    y1 = y_pos * height - (y_size * height / 2)
    x2 = x_size * width + x1
    y2 = y_size * height + y1

    return (x1, y1, x2, y2, oclass)

def convertImgaugBBToYolov3BB(args, img_shape):

	height, width, depth = img_shape
    # for i in args:
    #     print (i)

	oclass = int(args[4])
	x1 = float(args[0])
	y1 = float(args[1])
	x2 = float(args[2])
	y2 = float(args[3])

	x_pos = x1 / width + ((x2 - x1) / width /  2)
	y_pos = y1 / height + ((y2 - y1) / height / 2)
	x_size = (x2 - x1) / width
	y_size = (y2 - y1) / height

	return_args = [x_pos, y_pos, x_size, y_size, oclass]

	# Skip BBs that fall outside YOLOv3 range
	for r in return_args[:4]:
	    if r > 1: 
	    	print(1)
	    	return ()
	    if r < 0: 
	    	print(2)
	    	return ()
	#print(return_args)
	return (return_args)


def convertBoundingBoxes(bbs_data, img_shape):
	# print(bbs_data)
	bb_list=[]

	args_list = []

	for data in bbs_data:

		args = convertYolov3BBToImgaugBB(data, img_shape)
		args_list.append(args)
		
		bb = ia.BoundingBox(x1=args[0], y1=args[1], x2=args[2], y2=args[3], label=args[4])
		bb_list.append(bb)


	bbs = ia.BoundingBoxesOnImage(bb_list, shape=img_shape)

	return bbs, args_list

def convertBoundingBoxesBack(bbs_data, img_shape):
	# print(bbs_data)
	bb_list=[]
	args_list = []

	for i in range (0, len(bbs_data.bounding_boxes)):

		data = (bbs_data.bounding_boxes[i].x1, bbs_data.bounding_boxes[i].y1, bbs_data.bounding_boxes[i].x2, bbs_data.bounding_boxes[i].y2, bbs_data.bounding_boxes[i].label)

		args = convertImgaugBBToYolov3BB(data, img_shape)
		
		#print(args)

		if not args:continue

		args_list.append(args)

		bb = ia.BoundingBox(x1=args[0], y1=args[1], x2=args[2], y2=args[3], label=args[4])
		bb_list.append(bb)


	bbs = ia.BoundingBoxesOnImage(bb_list, shape=img_shape)

	return bbs, args_list


def TrimBB(bbs):

	shape_img = bbs.shape

	for i in range (0, len(bbs.bounding_boxes)):
		bb = bbs.bounding_boxes[i]

		if bb.x1 >= shape_img[1]:
			bb.x1 = shape_img[1] - 1

		elif bb.x1 <= 0:
			bb.x1 = 0

		if bb.x2 >= shape_img[1]:
			bb.x2 = shape_img[1] - 1

		elif bb.x2 <= 0:
			bb.x2 = 0

		if bb.y1 >= shape_img[0]:
			bb.y1 = shape_img[0] - 1

		elif bb.y1 <= 0:
			bb.y1 = 0

		if bb.y2 >= shape_img[0]:
			bb.y2 = shape_img[0] - 1

		elif bb.y2 <= 0:
			bb.y2 = 0

		bbs.bounding_boxes[i] = bb

	# print("==BBs==")
	# print(bbs)

	return bbs



def shownImageBB(img, bbs):
	imagebb = bbs.draw_on_image(img, size=2)
	cv2.imshow("oi", cv2.resize(imagebb, (int(imagebb.shape[1] * 0.7), int(imagebb.shape[0]*0.7))))

	cv2.waitKey()

def applyAug(img, bbs):
	seq = iaa.Sequential([
	    iaa.AdditiveGaussianNoise(scale=0.05*255),
	    iaa.Fliplr(1),
	    iaa.Affine(translate_px={"x": (1, 5)})
	])

	img_aug, bbs_aug = seq(images=[img], bounding_boxes=bbs)

	img_aug = (img_aug[0][...,::-1])

	return img_aug, bbs_aug


def applyTestAug(img, bbs):
	seq = iaa.Sequential([
	    #iaa.AdditiveGaussianNoise(scale=0.05*255),
	    #iaa.Fliplr(1),

	    #iaa.Affine(translate_px={"y": (136, 136)})

	    # iaa.Affine(
     #        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
     #        translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # translate by -20 to +20 percent (per axis)
     #        #rotate=(-2, 2), # rotate by -45 to +45 degrees
     #        #shear=(-5, 5), # shear by -16 to +16 degrees
     #        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
     #        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
     #        mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
     #    )
     	
	   # iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
	   

		 		
	])

	img_aug, bbs_aug = seq(images=[img], bounding_boxes=bbs)

	#print(img_aug[0][...,::-1])

	#img_aug = (img_aug[0][...,::-1])
	img_aug = (img_aug)[0]
	# print("\nBEFORE\n")
	# print(bbs_aug)

	bbs_aug =  bbs_aug.remove_out_of_image_fraction(0.1)

	

	bbs_aug = TrimBB(bbs_aug)
	# print("\nAFTER\n")
	# print(bbs_aug)

	return img_aug, bbs_aug


def applyHeavyAug(img, bbs):

	#sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	seq = iaa.Sequential(
		[
			#iaa.Affine(translate_px={"x": (400, 400)}),
		    #iaa.AdditiveGaussianNoise(scale=0.05*255),
		    iaa.Fliplr(0.5),
		    #iaa.Flipud(0.1),

		    iaa.Sometimes(0.5, iaa.CropAndPad(
	            percent=(-0.05, 0.10),
	            pad_mode=ia.ALL,
	            pad_cval=(0, 255)
	        )),

	        iaa.Sometimes(0.5, iaa.Affine(
	            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
	            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # translate by -20 to +20 percent (per axis)
	            #rotate=(-2, 2), # rotate by -45 to +45 degrees
	            #shear=(-5, 5), # shear by -16 to +16 degrees
	            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
	            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
	            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
	        )),
		    #iaa.Affine(translate_px={"x": (1, 5)})

		    iaa.SomeOf((1,5),
		    	[
		    		iaa.ChannelShuffle(p=1.0),
		    		iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
		    		iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
		    		iaa.OneOf([
	                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
	                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.20), per_channel=0.2),
	                    iaa.Cutout(nb_iterations=(1,5), size = (0.05, 0.2), squared=False, fill_per_channel=0.5, fill_mode=("constant", "gaussian"), cval = (0,255)),
	                	iaa.SaltAndPepper((0.01,0.1)),
	                ]),

		    		iaa.OneOf([
	                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
	                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
	                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
	                ]),
	                iaa.OneOf([
	                	iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.25)), # sharpen images
	                	iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.5)), # emboss images
	                ]),

	               	iaa.BlendAlphaSimplexNoise(iaa.OneOf([
	                    iaa.EdgeDetect(alpha=(0.2, 0.8)),
	                    iaa.DirectedEdgeDetect(alpha=(0.2, 0.8), direction=(0.0, 1.0)),
	                ])),
	                # either change the brightness of the whole image (sometimes
	                # per channel) or change the brightness of subareas
	                iaa.OneOf([
	                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
	                    iaa.BlendAlphaFrequencyNoise(
	                        exponent=(-4, 0),
	                        foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
	                        background=iaa.LinearContrast((0.5, 2.0))
	                    )
	                ]),
	               
	               	iaa.OneOf([
	               		iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
	               		iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
	               		iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5), mul_saturation=(0.5,1.5))
	               	]),
	                
	                iaa.Invert(0.05, per_channel=True), # invert color channels
	                
	                iaa.Sometimes(0.5, iaa.OneOf([
	                	iaa.AveragePooling((2,8)),
	                	iaa.MaxPooling((2,8)),
	                	iaa.MinPooling((2,8)),
	                	iaa.MedianPooling((2,8)),
	                	iaa.Superpixels(p_replace=(0.1, 0.5), n_segments=(200, 300)) # convert images into their superpixel representation
	                ])),
	                
	                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
	                iaa.Grayscale(alpha=(0.0, 1.0)),
	                #iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=(0.25, 1.0), sigma=0.25)), # move pixels locally around (with random strengths)
	                #iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.03))), # sometimes move parts of the image around
	                #iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1)))

	    		],

	    		random_order=True	
	   		)
		],
		
		random_order = True
	)

	img_aug, bbs_aug = seq(images=[img], bounding_boxes=bbs)

	#img_aug = (img_aug[0][...,::-1])
	img_aug = (img_aug)[0]

	# print("\nBEFORE\n")
	# print(bbs_aug)

	bbs_aug =  bbs_aug.remove_out_of_image_fraction(0.1)

	bbs_aug = TrimBB(bbs_aug)

	# print("\nAFTER\n")
	# print(bbs_aug )

	return img_aug, bbs_aug


def saveimg(img, bb_coord, img_name, label_name):
	#print("saving img")
	cv2.imwrite(img_name, img)

	output_file = open(label_name, "w")

	#print("saving bbs")

	output_text = ""

	flag = False
	for bb in bb_coord:
		if flag == False:
			output_text = output_text + str(bb[4]) + " " + "{0:.6f}".format(bb[0]) + " " + "{0:.6f}".format(bb[1])+ " " + "{0:.6f}".format(bb[2])+ " " + "{0:.6f}".format(bb[3])
			flag = True
		else:
			output_text = output_text + "\n" + str(bb[4]) + " " + "{0:.6f}".format(bb[0]) + " " + "{0:.6f}".format(bb[1])+ " " + "{0:.6f}".format(bb[2])+ " " + "{0:.6f}".format(bb[3])
			
	output_file.write(output_text)

	output_file.close()


if __name__ == '__main__':
	

	folder = sys.argv[1] #folder recieved as arg
	print(folder)
	image_folder = folder + "/images"
	labels_folder = folder + '/labels'
	print('creating dir ' + image_folder)
	print('creating dir ' + labels_folder)
	os.makedirs(image_folder) #printing image folder created
	os.makedirs(labels_folder) #printing label folder created

	list_of_labels = []

	print("\nLOADING IMAGES")

	list_of_images = os.listdir("images/") #getting all imags

	print("IMAGES LOADED: " + str(len(list_of_images)) + "\n\nLOADING LABELS")
	for i in list_of_images:
		list_of_labels.append(i.split('.')[0] + ".txt") #getting labels that match the images

	print("LABELS LOADED: " + str(len(list_of_labels)) + "\n")

	for i in range(0, len(list_of_images)):
		current_img = "images/" + list_of_images[i]
		current_label = "labels/" + list_of_labels[i]
		#print(current_img)
		#print(current_label) 
		print("image " + str(i + 1) + " out of " + str(len(list_of_images)) + " writing to folder" + sys.argv[1])

		output_imagename = sys.argv[1] + "/images/" + sys.argv[1] + list_of_images[i]
		output_labelname = sys.argv[1] + "/labels/" + sys.argv[1] + list_of_labels[i]

		image = cv2.imread(current_img) #image load
		img_shape = image.shape
		#print(img_shape)
		bbs_data = boundingBoxRead(current_label)

		bbs_imgaug, data_imgaug = convertBoundingBoxes(bbs_data, img_shape) #returns obj BB and list BB on imgaug format
		#
		image_aug, bbs_aug = applyHeavyAug(image, bbs_imgaug) #returns the augmentation over the image and BB

		bbs_yolo, data_yolo = convertBoundingBoxesBack(bbs_aug, img_shape) #returns obj BB and list BB back on yolo format

		#print(bbs_yolo, data_yolo)

		saveimg(image_aug, data_yolo, output_imagename, output_labelname)
		#print(data_yolo)
		#cv2.imshow("oi", image_aug)
		# cv2.waitKey()
		#print(bbs_aug.bounding_boxes[0].x1)
		#shownImageBB(image_aug, bbs_aug)



	#bbs = ia.BoundingBox(x1=100.5, y1=150.5, x2=300.5, y2=500.5, label=1)
	# print(bbs.bounding_boxes)


# seq = iaa.Sequential([
#     iaa.AdditiveGaussianNoise(scale=0.05*255),
#     iaa.Fliplr(1),
#     iaa.Affine(translate_px={"x": (1, 5)})
# ])

# images_aug, bbs_aug = seq(images=[image], bounding_boxes=bbs)

# # seq_det = seq.to_deterministic()
# # image_aug = seq_det.augment_images([image])[0]
# # bbs_aug = seq_det.augment_bounding_boxes(bbs)[0]

# #print(type(image))

# #sprint(images_aug)



# after = (images_aug[0][...,::-1])

# image_before = bbs.draw_on_image(image, size=2)
# image_after = bbs_aug.draw_on_image(after, size=2)

# cv2.imshow("oi", image_before)
# cv2.waitKey()

# cv2.imshow("oi", image_after)
# cv2.waitKey()

