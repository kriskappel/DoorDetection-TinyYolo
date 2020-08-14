import glob, os

list_of_images = os.listdir("./")

for i in list_of_images:
	#todo os.rename
	#oldname = i.copy()
	
	print(i.replace("raw", ""))
	os.rename(i, i.replace("raw", ""))
	i = i.replace("raw", "") 
	
	os.rename(i, i.replace("image", "egami"))
	i = i.replace("image", "egami") 

	os.rename(i, i.replace("index", "xedni"))
	i = i.replace("index", "xedni") 

	if(len(i.split('.')) > 2):
		os.rename(i, i.replace(".", "", 1))
		i = i.replace(".", "", 1)
