# DoorDetection-TinyYolo
Trying to detect doors on an image using a tiny-yolo neural net architecture.

Based on Arduengo's work: https://github.com/MiguelARD/DoorDetect-Dataset

Yolo Project: https://pjreddie.com/darknet/yolo/


Steps to run training:
- Adjust .data, train.txt and test.txt files to your own file paths.
- Make sure your dataset files do not contained words like image or raw in the name
- run ./darknet detector train cfg/0doors_test.data cfg/0doors_test.cfg darknet53.conv.74

The dataset was gathered from OpenImagesV4 and anotated by Miguel Arduengo .

The dataset and augmented images can be found here: https://drive.google.com/drive/folders/1Cy7Huu6BEkR-AINUa_J7Z6axEgAzzGzo?usp=sharing . I have only augmented and changed the name of some files to avoid troubles with yolo loading the images.	
The file "augment.py" was used to perform the augmentations on the dataset.

In order to use the dataset, all the images and labels must be in the same directory.

To use both the dataset and the augmented images, one must add all of them to the same folder and concatenate the content of both 
"train.txt" files, as well as changing the paths of both "train.txt" and "test.txt".


Useful link to start the training:
https://medium.com/@manivannan_data/how-to-train-yolov3-to-detect-custom-objects-ccbcafeb13d2
