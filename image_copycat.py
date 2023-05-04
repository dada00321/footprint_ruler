import cv2 
import os

def copy_and_paste(output_folder_name, footprint_image, footprint_mask, col_first_coordinates):
	# Create Directory
	dir_path = f"result/{output_folder_name}"
	os.mkdir(dir_path) if not os.path.exists(dir_path) else None
	dir_path += "/(3)_image_copycat"
	os.mkdir(dir_path) if not os.path.exists(dir_path) else None
	
	# Show input image
	cv2.imshow("footprint image - original", footprint_image)
	cv2.imwrite(f"{dir_path}/(1)_footprint image - original.jpg", footprint_image)
	cv2.imshow("footprint mask", footprint_mask)
	cv2.imwrite(f"{dir_path}/(2)_footprint.jpg", footprint_image)
	
	## Step: Copy all pixels in footprint's bounding box
	##       and paste to footprint_image
	x2, y2, foot_w2, foot_h2 = col_first_coordinates # parse coordinates
	#footprint_image[y2:y2+foot_h2,x2:x2+foot_w2] = footprint_mask[0:foot_h2,0:foot_w2]
	footprint_image[y2+3:y2+foot_h2-2,x2+3:x2+foot_w2-2] = footprint_mask[y2+3:y2+foot_h2-2,x2+3:x2+foot_w2-2]
	cv2.imshow("footprint image", footprint_image)
	cv2.imwrite(f"{dir_path}/(3)_footprint image.jpg", footprint_image)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()