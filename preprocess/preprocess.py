#!/usr/bin/env python
# encoding: utf-8
'''
@author:
@license: (C) Copyright 2020-2023, XJTU.
@contact:
@software: MedAI
@file: preprocess.py
@time: 2022/8/3 16:50
@version:
@desc:
'''

import os
import copy
from PIL import Image
import numpy as np
import cv2
from skimage.measure import regionprops

def remove_black_border(image_path,mask_path,outputsize =256):
    # read mask
    mask = Image.open(mask_path)
    mask = np.asarray(mask)
    #read img
    img = Image.open(image_path)
    img_array = np.asarray(img)
    img_array1 = np.array(img, dtype=np.float32)
    or_shape = img_array1.shape  #the size of original image
    value_x = np.mean(img, 1) #Calculate the mean value of each column
    value_y = np.mean(img, 0) #Calculate the mean value of each row
    x_hold_range = list((len(value_x) * np.array([0.24 / 3, 2.2 / 3])).astype(np.int))
    y_hold_range = list((len(value_y) * np.array([0.8 / 3, 1.8 / 3])).astype(np.int))
    value_thresold = 5
    x_cut = np.argwhere((value_x<=value_thresold)==True)
    x_cut_min = list(x_cut[x_cut<=x_hold_range[0]])
    if x_cut_min:
        x_cut_min = max(x_cut_min)
    else:
        x_cut_min = 0

    x_cut_max = list(x_cut[x_cut>=x_hold_range[1]])
    if x_cut_max:
        x_cut_max = min(x_cut_max)
    else:
        x_cut_max = or_shape[0]
    y_cut = np.argwhere((value_y<=value_thresold)==True)
    y_cut_min = list(y_cut[y_cut<=y_hold_range[0]])
    if y_cut_min:
        y_cut_min = max(y_cut_min)
    else:
        y_cut_min = 0

    y_cut_max = list(y_cut[y_cut>=y_hold_range[1]])
    if y_cut_max:
        # print('q')
        y_cut_max = min(y_cut_max)
    else:
        y_cut_max = or_shape[1]

    # crop image
    cut_image = img_array[x_cut_min:x_cut_max,y_cut_min:y_cut_max]
    cut_mask = mask[x_cut_min:x_cut_max,y_cut_min:y_cut_max]

    #resize image
    cut_image_r = cv2.resize(cut_image, (outputsize, outputsize), cv2.INTER_CUBIC)
    mask_r = cv2.resize(cut_mask, (outputsize, outputsize))

    return cut_image,cut_mask,cut_image_r, mask_r

def get_nodule_roi(img_array,mask, c2_size=512):
    mask_c1_array_biggest=copy.deepcopy(mask)
    mask_c1_array_biggest[mask_c1_array_biggest>0]=1
    w,h=mask_c1_array_biggest.shape
    # print("w,h",w,h)
    c1_size=max(w,h)
    if np.sum(mask_c1_array_biggest) == 0:
            minr, minc, maxr, maxc = [0, 0, w,h]
    else:
        region = regionprops(mask_c1_array_biggest)[0]
        minr, minc, maxr, maxc = region.bbox

    dim1_center, dim2_center = [(maxr + minr) // 2, (maxc + minc) // 2]
    max_length = max(maxr - minr, maxc - minc)
    max_lengthl = int((c1_size/256)*80)
    preprocess1 = int((c1_size/256)*19)
    pp22 = int((c1_size/256)*31)

    if max_length > max_lengthl:
        ex_pixel = preprocess1 + max_length // 2
    else:
        ex_pixel = pp22 + max_length // 2

    dim1_cut_min = dim1_center - ex_pixel
    dim1_cut_max = dim1_center + ex_pixel
    dim2_cut_min = dim2_center - ex_pixel
    dim2_cut_max = dim2_center + ex_pixel

    if dim1_cut_min < 0:
        dim1_cut_min = 0
    if dim2_cut_min < 0:
        dim2_cut_min = 0
    if dim1_cut_max > w:
        dim1_cut_max = w
    if dim2_cut_max > h:
        dim2_cut_max = h

    img_array_roi = img_array[dim1_cut_min:dim1_cut_max, dim2_cut_min:dim2_cut_max]
    mask_roi = mask[dim1_cut_min:dim1_cut_max, dim2_cut_min:dim2_cut_max]
    print("mask_roi.shape", mask_roi.shape)

    img_array_roi = cv2.resize(img_array_roi, (c2_size, c2_size), cv2.INTER_CUBIC)
    mask_roi = cv2.resize(mask_roi, (c2_size, c2_size))
    return img_array_roi,mask_roi

if __name__=='__main__':
    osp=os.path
    dir_path=r'D:\AcademicProject\thyroid_nodule\code\preprocess\result_v3'
    dir_path_out=osp.join(dir_path,'train_dataset')
    if not osp.exists(dir_path_out):
        os.makedirs(dir_path_out)
    dir_path_mask = osp.join(dir_path,'mask')
    dir_path_image = osp.join(dir_path,'image')
    dir_path_out_img_s1 = osp.join(dir_path_out, 'image_stage1_256')
    dir_path_out_mask_s1 = osp.join(dir_path_out, 'mask_stage1_256')
    if not osp.exists(dir_path_out_img_s1):
        os.makedirs(dir_path_out_img_s1)
    if not osp.exists(dir_path_out_mask_s1):
        os.makedirs(dir_path_out_mask_s1)
    dir_path_out_img_s2 = osp.join(dir_path_out, 'image_stage2_512')
    dir_path_out_mask_s2 = osp.join(dir_path_out, 'mask_stage2_512')
    if not osp.exists(dir_path_out_img_s2):
        os.makedirs(dir_path_out_img_s2)
    if not osp.exists(dir_path_out_mask_s2):
        os.makedirs(dir_path_out_mask_s2)
    file_lsts=sorted(os.listdir(dir_path_image))
    for index, filename in enumerate(file_lsts):
        file_name = filename.split('-')[1]
        ##get data of stage 1, size is 256
        cut_image,cut_mask,cut_image_r, mask_r=remove_black_border(osp.join(dir_path_image,filename),osp.join(dir_path_mask,filename),256)
        ##get data of stage 2, size is 256
        img_array_roi,mask_roi=get_nodule_roi(cut_image,cut_mask,512)
        cv2.imwrite(osp.join(dir_path_out_img_s1, filename), cut_image_r)
        cv2.imwrite(osp.join(dir_path_out_mask_s1, filename), 255*mask_r)
        cv2.imwrite(osp.join(dir_path_out_img_s2, file_name), img_array_roi)
        cv2.imwrite(osp.join(dir_path_out_mask_s2, file_name), 255*mask_roi)
        if index>3:
            break
