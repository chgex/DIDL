'''
Author: liubai
Date: 2021-03-30
LastEditTime: 2021-03-30
'''

import cv2
import matplotlib.pyplot as plt

import time
import random



# load the input image

img_path='./img.jpg'
image = cv2.imread(img_path,cv2.IMREAD_COLOR)
#  not use image = cv2.imread(img_path)

# input image
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
# initialize OpenCV's selective search implementation and set the
ss.setBaseImage(image)
# choice select search version
ss.switchToSelectiveSearchFast()

# fast but less accurate version of selective search
# ss.switchToSelectiveSearchFast()
	
# slower but more accurate version
# ss.switchToSelectiveSearchQuality()

start = time.time()
rects = ss.process()
end = time.time()
print("[info] selective search algorithm took %.2f seconds" % (end - start) )
print("[INFO] total region proposals is: {}".format(len(rects)))

# 增强边界框的色彩
rec=rects.copy()
rec[:, 2] += rec[:, 0]
rec[:, 3] += rec[:, 1]


for i in range(0, len(rec), 100):
	# clone the original image so we can draw on it
	output = image.copy()
	# loop over the current subset of region proposals
	for (x, y, w, h) in rects[i:i + 100]:
		# draw the region proposal bounding box on the image
		color = [random.randint(0, 255) for j in range(0, 3)]
		cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
	# show the output image
	tmp=cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
	plt.imshow(tmp)
	cv2.imwrite('img_bbox.jpg',output)
    


