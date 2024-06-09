## Import Libraries
import os
import json
import numpy as np
from PIL import Image  
## Configurations
save_dir_kidney_masks = 'AHN_Masks/Kidney'
if not os.path.exists(save_dir_kidney_masks):
      os.makedirs(save_dir_kidney_masks)
save_dir_fluid_masks = 'AHN_Masks/Fluid'
if not os.path.exists(save_dir_fluid_masks):
      os.makedirs(save_dir_fluid_masks)
## Open JSON file
f = open('data.json')
data = json.load(f)
## Read File Info
print(data.keys())
data_info = data['info']
data_info_description = data_info['description']
## Generate Masks based on Kidney and Fluid Annotations
data_categories = data['categories']
print(data_categories)
print(data['images'][0].keys())
print(data['annotations'][0].keys())
for i in range(0,len(data['images'])):
    print(f'Image No.: {i+1}')
    data_image = data['images'][i]
    # im_id = data_image['id']
    # imwidth = data_image['width']
    # imheight = data_image['height']
    # im_filename = data_image['file_name']
    # im_savedir = save_dir + '/' + data_image['file_name']
    ## Loop through Annotations
    msk_blank_kidney = np.zeros([data_image['height'],data_image['width']], dtype=float)
    msk_blank_fluid = np.zeros([data_image['height'],data_image['width']], dtype=float)
    for ii in range(0, len(data['annotations'])):
        data_annotation_temp = data['annotations'][ii]
        # annotations_id = data_annotation_temp['id']
        # annotations_cat_id = data_annotation_temp['category_id']
        # annotations_segmentation = data_annotation_temp['segmentation']
        # annotations_bbox = data_annotation_temp['bbox']
        # annotations_area = data_annotation_temp['area']
        # print(data_annotation_temp['image_id'])
        if data_annotation_temp['image_id'] == i:
            ## Create Blank Mask
            annotations = np.int_(np.transpose(np.array(data_annotation_temp['segmentation'])))
            annotations_reshaped = np.expand_dims(np.reshape(annotations, (annotations.shape[0]//2, 2)), axis=1)  # For CV2 FILL
            # For Numpy Array Fill based on Annotations Index
            '''x_coord_array = []
            y_coord_array = []
            for i in range(annotations.shape[0]):
                if i % 2 == 0:
                    x_coord_array.append(annotations[i])
                elif i % 2 == 1:
                    y_coord_array.append(annotations[i])
            x_coord_array = np.array(x_coord_array)
            y_coord_array = np.array(y_coord_array)'''
            ## Fill Blank Mask with Annotations: Kidney
            if data_annotation_temp['category_id'] == 1:
                cv2.drawContours(image=msk_blank_kidney,
                                 contours=[annotations_reshaped],
                                 contourIdx=-1,
                                 color=(255,255,255),
                                 thickness=cv2.FILLED)
                # msk_blank_kidney[x_coord_array,y_coord_array] = 255
                cv2.imwrite(save_dir_kidney_masks + '/' + data_image['file_name'], msk_blank_kidney) 
            ## Fill Blank Mask with Annotations: Fluid
            elif data_annotation_temp['category_id'] == 2:
                cv2.drawContours(image=msk_blank_fluid,
                                 contours=[annotations_reshaped],
                                 contourIdx=-1,
                                 color=(255,255,255),
                                 thickness=cv2.FILLED)
                # msk_blank_fluid[x_coord_array,y_coord_array] = 255
                cv2.imwrite(save_dir_fluid_masks + '/' + data_image['file_name'], msk_blank_fluid)