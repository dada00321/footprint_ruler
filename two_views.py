"""
Process:
   Step 1. _get_binary_img()
   ----------------------------------------------- 
		Attributes of "binary image" (`bin_img`)
		  * Colour of foot region: black
		  * Colour of white paper region: white
		  * Colour out of white paper: black
	 ↓
   Step 2. _get_ROI()
"""

import cv2
import numpy as np
import os

def _get_binary_img(img):
	
	#!!!
	## Workable
	
	'''
	'''
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	valMask = cv2.inRange(v, 200, 255) 
	sMask = cv2.inRange(s, 0, 40)
	mask = valMask & sMask
	
	
	#!!!
	
	## Test: @dada
	
	### 1-1
	'''
	#lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	#lightness, a, b = cv2.split(lab)
	#lightnessMask = cv2.inRange(lightness, 200, 255)
	'''
	
	### 1-2
	'''
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	valMask = cv2.inRange(v, 165, 255) 
	sMask = cv2.inRange(s, 0, 30)
	mask = valMask & sMask
	'''
	
	
	
	### 2
	bin_img = cv2.bitwise_and(img, img, mask=mask)
	k = 10
	#bin_img = cv2.open(bin_img, np.ones((k,k)), iterations=1)
	#bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((k//2,k//2)))
	
	#bin_img = cv2.dilate(bin_img, None)
	#bin_img = cv2.erode(bin_img, np.ones((k,k)), iterations=1)
	#bin_img = cv2.dilate(bin_img, np.ones((k,k)), iterations=3)
	#bin_img = cv2.dilate(bin_img, np.ones((k,k)), iterations=1)
	#bin_img = cv2.dilate(bin_img, np.ones((k,k)), iterations=1)
	
	"""
	cv2.imshow("original", img)
	cv2.waitKey(0)
	cv2.imshow("binary img", bin_img)
	cv2.waitKey(0)
	
	cv2.destroyAllWindows()
	"""
	return bin_img

def _get_ROI(img, bin_img):
	# 1. Pre-processing: Extract the contour of ROI
	gray = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY)  

	# [+]: dilate (若沒做 [形態學-膨脹], bounding box 只會框到白色目標物內部的小黑孔洞)
	gray = cv2.dilate(gray, None) # 膨脹 => 目標: 補洞 (填補白色物體內部的小黑孔洞)

	ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  

	contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# 2. Add a bounding box
	x,y,w,h = cv2.boundingRect(contours[0]) # boundingRect: 找正立矩形
	#colour = tuple([255]*3)
	#colour = (5,195,221)[::-1] # cyan
	
	"""
	colour = (5,195,221)
	binary_with_bbox = bin_img.copy()
	cv2.rectangle(binary_with_bbox, (x,y), (x+w,y+h), colour, 2)
	"""
	
	"""
	cv2.imshow("result", ROI)
	cv2.waitKey(0)
	"""
	
	# 3. Crop the bounding box region 
	ROI_tmp = img[y:y+h,x:x+w]
	#cv2.imshow("ROI_tmp", ROI_tmp)
	#cv2.waitKey(0)
	
	# 4. Swap the colour of black region and white region
	ret, ROI = cv2.threshold(ROI_tmp, 120, 255, cv2.THRESH_BINARY_INV) 
	
	# 5. Use "saturation mask" to change chromatic colours to black
	#	such that "foot region" is the unique region with white colour
	hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
	hue, s, v = cv2.split(hsv)
	sMask = cv2.inRange(s, 0, 0)
	ROI = cv2.bitwise_and(ROI, ROI, mask=sMask)
	
	'''
	ROI[:15,:] = (0,255,0)
	ROI[h-15:,:] = (0,255,0)
	ROI[:,:15] = (255,0,0)
	ROI[:,w-15:] = (255,0,0)
	'''
	colour = (0,0,0)
	ROI[:15,:] = colour
	ROI[h-15:,:] = colour
	ROI[:,:15] = colour
	ROI[:,w-15:] = colour
	
	""" 
	cv2.imshow("ROI_partial", ROI)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	"""
	
	# 6. Denoise
	ROI[:50, :50, :] = 0
	ROI[:50, (w-50):w, :] = 0
	ROI[(h-50):h, :50, :] = 0
	ROI[(h-50):h, (w-50):w, :] = 0
	
	return ROI

''' Detect: corners of white paper + bounding box of foot '''
def _get_corners(ROI):
	#cv2.imshow("ROI", ROI)
	#cv2.waitKey(0)
	
	# bounding box of foot
	# ---------------
	# 1. Pre-processing: Extract the contour of ROI
	gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
	
	# Additional Process: Remove noise
	#k = 20
	#gray = cv2.dilate(gray, np.ones((k, k), np.uint8))
	h,w = gray.shape
	#cv2.imshow("gray", gray)
	#cv2.waitKey(0)
	
	# [+]: dilate (若沒做 [形態學-膨脹], bounding box 只會框到白色目標物內部的小黑孔洞)
	#gray = cv2.dilate(gray, None) # 膨脹 => 目標: 補洞 (填補白色物體內部的小洞)

	ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  

	cv2.imshow("binary", binary)
	
	contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# 2. Add a bounding box
	x, y, foot_w, foot_h = cv2.boundingRect(contours[0]) # boundingRect: 找正立矩形
	
	if max(foot_w,foot_h)/min(foot_w,foot_h) >= 4:
		x, y, foot_w, foot_h = cv2.boundingRect(contours[1]) # boundingRect: 找正立矩形
	
	print("x, y, foot_w, foot_h:\n", x, y, foot_w, foot_h)
	colour = (5,195,221)
	binary_with_bbox = ROI.copy()
	cv2.rectangle(binary_with_bbox, (x,y), (x+foot_w,y+foot_h), colour, 2)

	# 3. Show text message & processed image
	board_h, board_w, _ = binary_with_bbox.shape
	percentage_h = round(100 * foot_h / board_h, 2)
	percentage_w = round(100 * foot_w / board_w, 2)
	foot_h_mm = round(297 * foot_h / board_h, 2)
	foot_w_mm = round(210 * foot_w / board_w, 2)
	# ---
	'''
	A4: 210 x 297 | unit: mm
	'''
	print(f"board height: {board_h} px")
	print(f"board width: {board_w} px")
	print(f"foot height: {foot_h} px"+\
		  f" \n   => {percentage_h} % of board height"+\
		  f" \n   => approx. {foot_h_mm} mm")
	print(f"foot width: {foot_w} px"+\
		  f"\n   => {percentage_w} % of board width"+\
		  f" \n   => approx. {foot_w_mm} mm")
	# ---
	
	#cv2.imshow("binary_with_bbox", binary_with_bbox)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return binary_with_bbox, x, y, foot_w, foot_h, foot_h_mm, foot_w_mm
	
def _put_text(img, x, y, foot_w, foot_h, foot_h_mm, foot_w_mm):
	#font = cv2.FONT_HERSHEY_SIMPLEX
	font = cv2.FONT_HERSHEY_PLAIN

	# fontScale
	fontScale = 2

	# Blue color in BGR
	color = (5,195,221)[::-1]

	# Line thickness of 2 px
	thickness = 2

	# Using cv2.putText() method
	# 1. Put text about "foot width"
	org = (x+45, 55) # x,y
	org = (x+45, 35) # @dada
	cv2.putText(img, f"{foot_w_mm}", org, font, fontScale, color, thickness, cv2.LINE_AA)
	
	# 2. Put text about "foot height"
	org = (int(x/5), int(y+0.5*foot_h)) # x,y
	cv2.putText(img, f"{foot_h_mm}", org, font, fontScale, color, thickness, cv2.LINE_AA)

	# 3. Put text about "unit"
	thickness = 2
	org = (10, 27) # x,y
	cv2.putText(img, "unit: mm", org, font, fontScale, color, thickness, cv2.LINE_AA)

	# Show image with text 
	#---
	#cv2.imshow("img_with_bbox_and_text", img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
	"""
	realistic:
		my foot => 110 mm x 255 mm
	"""
	return img

def save_two_views(output_folder_name, left_view_imgPath, right_view_imgPath):
	dir_path = f"result/{output_folder_name}"
	os.mkdir(dir_path) if not os.path.exists(dir_path) else None
	dir_path += "/(1)_two_views"
	os.mkdir(dir_path) if not os.path.exists(dir_path) else None
	
	img = cv2.imread(left_view_imgPath)
	cv2.imshow("left_view", img)
	cv2.imwrite(f"{dir_path}/(1)_left_view.jpg", img)

	bin_img = _get_binary_img(img)
	cv2.imshow("bin_img", bin_img)
	cv2.imwrite(f"{dir_path}/(2)_bin_img.jpg", bin_img)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	ROI_left = _get_ROI(img, bin_img)
	
	### Example: @dada (Use Dada's foot in left & right sides as the input images)
	#ROI_left = cv2.erode(ROI_left, np.ones((10,10)))
	#ROI_left = cv2.erode(ROI_left, np.ones((10,10)))
	#ROI_left = cv2.dilate(ROI_left, np.ones((1,1)))
	#ROI_left = cv2.dilate(ROI_left, np.ones((10,10)))
	
	## @dada
	#k = 10
	##ROI_left = cv2.erode(ROI_left, np.ones((k,k)), iterations=3)
	cv2.imshow("ROI_left", ROI_left)
	cv2.imwrite(f"{dir_path}/(3)_ROI_left.jpg", ROI_left)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	h1,w1,_ = ROI_left.shape
	#print(h1,w1)
	
	img = cv2.imread(right_view_imgPath)
	#sleep(20)
	bin_img = _get_binary_img(img)
	ROI_right = _get_ROI(img, bin_img)
	
	## @dada
	### Because the result of `ROI_right`
	
	
	h2, w2, _ = ROI_right.shape
	#print(h2,w2)
	
	## @dada
	## The parameter and setting below: barely acceptable
	#!!!
	
	k = 6
	#ROI_right = cv2.erode(ROI_right, np.ones((k,k)), iterations=4)
	
	#!!!
	
	## The parameter and setting below: trying
	#!!!
	'''
	k = 10
	ROI_right = cv2.morphologyEx(ROI_right, cv2.MORPH_CLOSE, np.ones((k,k)), iterations=4)
	'''
	#!!!
	
	## Show `ROI_right`
	cv2.imshow("ROI_right", ROI_right)
	cv2.imwrite(f"{dir_path}/(4)_ROI_right.jpg", ROI_right)
	
	# Attempt-1
	'''
	dim = min(w1,w2), min(h1,h2)
	ROI_left = cv2.resize(ROI_left, dim, interpolation = cv2.INTER_AREA)
	ROI_right = cv2.resize(ROI_right, dim, interpolation = cv2.INTER_AREA)
	#cv2.imshow("ROI_left", ROI_left)
	#cv2.imshow("ROI_right", ROI_right)
	#---
	ROI_and = cv2.bitwise_and(ROI_left, ROI_right)
	ROI_or = cv2.bitwise_or(ROI_left, ROI_right)
	ROI = cv2.bitwise_or(ROI_and, ROI_or)
	'''
	
	# Attempt-2
	'''
	dim = min(w1,w2), min(h1,h2)
	ROI_left = cv2.resize(ROI_left, dim, interpolation = cv2.INTER_AREA)
	ROI_right = cv2.resize(ROI_right, dim, interpolation = cv2.INTER_AREA)
	#cv2.imshow("ROI_left", ROI_left)
	#cv2.imshow("ROI_right", ROI_right)
	#---
	ROI = ROI_left + ROI_right
	'''
	
	# Attempt-3: Only one part
	
	## Step 1. Obtain the right-view ROI image >> Crop as `ROI_cropped`
	binary_with_bbox_1, x1, y1, foot_w1, foot_h1, foot_h_mm_1, foot_w_mm_1 = _get_corners(ROI_right)
	img_with_bbox_and_text = _put_text(binary_with_bbox_1, x1, y1, foot_w1, foot_h1, foot_h_mm_1, foot_w_mm_1)
	#!!!
	ROI_cropped = ROI_right[y1:y1+foot_h1, x1:x1+foot_w1]
	#!!!
	#cv2.imshow("ROI_cropped", ROI_cropped)
	#cv2.waitKey()
	#cv2.destroyAllWindows()
	
	## Step 2. Obtain the left-view ROI image >> Add bounding box as `img_with_bbox_and_text`
	binary_with_bbox_2, x2, y2, foot_w2, foot_h2, foot_h_mm_2, foot_w_mm_2 = _get_corners(ROI_left)
	#!!!
	img_with_bbox_and_text = _put_text(binary_with_bbox_2, x2, y2, foot_w2, foot_h2, foot_h_mm_2, foot_w_mm_2)
	#!!!
	
	## Step X. Projective transform
	### 4 points of homography
	#[(y,x),(y+foot_h,x),(y+foot_h,x+foot_w),(y,x+foot_w)]
	
	'''
	pts_src and pts_dst are numpy arrays of points
	in source and destination images. We need at least
	4 corresponding points.
	'''
	pts_src = np.array([(y1,x1),(y1+foot_h1,x1),(y1,x1+foot_w1),(y1+foot_h1,x1+foot_w1)])
	#pts_src = [e[::-1] for e in pts_src]
	pts_dst = np.array([(y2,x2),(y2+foot_h2,x2),(y2,x2+foot_w2),(y2+foot_h2,x2+foot_w2)])
	#pts_dst = [e[::-1] for e in pts_dst]
	
	# Homography (projection)
	## Attempt-1
	"""
	h, status = cv2.find_homography(pts_dst, pts_src)
	#h, status = cv2.find_homography(pts_src, pts_dst)
	
	'''
	The calculated homography can be used to warp
	the source image to destination. Size is the
	size (width,height) of im_dst
	'''
	ROI_right_projection = cv2.warpPerspective(ROI_cropped, h, (foot_w2,foot_h2))
	cv2.imshow("A", ROI_right_projection)
	cv2.waitKey()
	cv2.destroyAllWindows()
	"""
	
	## Attempt-2
	'''
	###
	locs = {"top left": (y1,x1),
			"bottom left": (y1+foot_w1,x1),
			"top right": (y1,x1+foot_h1),
			"bottom right": (y1+foot_w1,x1+foot_h1)}
	
	replace_coordinate = [(y2,x2),(y2+foot_h2,x2),(y2,x2+foot_w2),(y2+foot_h2,x2+foot_w2)]
	cv2.waitKey()
	cv2.destroyAllWindows()
	'''

	## Attempt-3
	#print(foot_h1, foot_w1)
	#print(foot_h2, foot_w2)
	
	# Merge: img_with_bbox_and_text & ROI_cropped
	#print(foot_h2, foot_w2)
	#print(ROI_cropped.shape[:-1])
	
	#ROI_cropped_resized = cv2.resize(ROI_cropped, (foot_w2, foot_h2), interpolation = cv2.INTER_AREA)
	#ROI_right[y2:y2+foot_h2, x2:x2+foot_w2] = cv2.bitwise_or(ROI_right[y2:y2+foot_h2, x2:x2+foot_w2], ROI_cropped_resized)
	
	#cv2.imshow("img_with_bbox_and_text", img_with_bbox_and_text)
	#cv2.imshow("ROI_cropped", ROI_cropped)
	
	#cv2.waitKey()
	#cv2.destroyAllWindows()
	
	# Step 3. Save two views
	
	## Save images under the output folder
	cv2.imwrite(f"{dir_path}/(5)_ROI_cropped.jpg", ROI_cropped)
	cv2.imwrite(f"{dir_path}/(6)_img_with_bbox_and_text.jpg", img_with_bbox_and_text)
	
	## Save images for calling from another program `image_homography` afterward
	cv2.imwrite("./ROI_cropped.jpg", ROI_cropped)
	cv2.imwrite("./img_with_bbox_and_text.jpg", img_with_bbox_and_text)
	
	# Step 4. Return coordinates of `img_with_bbox_and_text`
	#col_first_coordinates = [[x2,y2],[x2,y2+foot_h2],[x2+foot_w2,y2],[x2+foot_w2,y2+foot_h2]]
	
	#return col_first_coordinates
	###
	return [x2,y2,foot_w2,foot_h2]
	
if __name__ == "__main__":
	pass