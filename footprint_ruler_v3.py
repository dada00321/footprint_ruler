from two_views import save_two_views
from image_homography import Homography
from image_copycat import copy_and_paste
from datetime import datetime

dt = datetime.today()
dt_str = f"{dt.year-2000:02d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"
output_folder_name = dt_str

''' Stage 1. Save two views (left-view & right-view footprints) 
             and then obtain corner coordinates of the left-view footprint's bounding box
'''

## Example: original
### Workable

#!!!
#left_view_imgPath = "res/original/2L.jpg"
#right_view_imgPath = "res/original/1R.jpg"
#!!!

### Renamed
left_view_imgPath = "res/original/(1)_left_view.jpg"
right_view_imgPath = "res/original/(2)_right_view.jpg"

'''
'''

## Example: dada
### Trying


'''
left_view_imgPath = "res/dada/3L_v2.jpg"
right_view_imgPath = "res/dada/3R_v2.jpg"
'''


'''
left_view_imgPath = "res/dada/4L_v2.jpg"
right_view_imgPath = "res/dada/4R_v2.jpg"
'''

'''
left_view_imgPath = "res/dada/5L_v2.jpg"
right_view_imgPath = "res/dada/5R_v2.jpg"

'''


'''
left_view_imgPath = "res/dada/6L.jpg"
right_view_imgPath = "res/dada/6R.jpg"
'''

'''
left_view_imgPath = "res/dada/7L.jpg"
right_view_imgPath = "res/dada/7R.jpg"
'''

'''
left_view_imgPath = "res/dada/8L.jpg"
right_view_imgPath = "res/dada/8R.jpg"
'''

'''
left_view_imgPath = "res/deji/2L.jpg"
right_view_imgPath = "res/deji/2R.jpg"
'''

col_first_coordinates = save_two_views(output_folder_name, left_view_imgPath, right_view_imgPath)
#print(col_first_coordinates)

''' Stage 2. Use the Homography matrix to apply Projective Transformation
			 and denoise the result 
			 (in order to remove internal noises which have black colour)
'''
homo = Homography(output_folder_name, col_first_coordinates)
footprint_image = homo.projective_transform()
footprint_mask = homo.denoise(footprint_image, col_first_coordinates)

''' Stage 3. Due to (1) Projective image 
                        => `footprint_image`
                        the projective image with colours "does have some noises" and
                    (2) Result image 
					    => `footprint_mask`
					    the result image after denoising "has no colours" 
					    in the outer regions of the footprint's bounding box,
		     in this stage, 
			 just simply copy (2) and paste into (1) to solve the problem above
'''
copy_and_paste(output_folder_name, footprint_image, footprint_mask, col_first_coordinates)
 
