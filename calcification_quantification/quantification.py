#!/usr/bin/env python
# encoding: utf-8
'''
@author:
@license: (C) Copyright 2020-2023,  XJTU.
@contact:
@software: MedAI
@file: quantification.py
@time: 2022/8/4 11:25
@version:
@desc:
'''

import pandas as pd
import pyclipper
import math
import os

import numpy as np
import cv2

def proportional_zoom_contour_ratio(contour, ratio):
    """
    Equidistant expansion of polygon outline points
    :param contour: outline format of a figure [[[x1, x2]],...],shape(-1, 1, 2)
    :param margin: The pixel distance of the outline expansion. A positive margin means expansion, and a negative margin means reduction.
    :return: Outlined contour point
    """
    poly = contour[:, 0, :]
    area_poly = cv2.contourArea(contour)  ##面积
    perimeter_poly = cv2.arcLength(contour, True)  ##周长
    poly_s = []
    pco = pyclipper.PyclipperOffset()
    pco.MiterLimit = 2
    if perimeter_poly:
        d = 2 * area_poly * (1 - ratio) / perimeter_poly  #
        pco.AddPath(poly, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        poly_s = pco.Execute(-d)
    poly_s = np.array(poly_s).reshape(-1, 1, 2).astype(int)

    return poly_s, d


def draw_axis(img, cnt, color1, color2, flag=True):
    (ell_center_x, ell_center_y), (MA, ma), angle = cv2.fitEllipse(cnt)
    # draw the contour and center of the shape on the image
    if flag:
        ell_h_point1_x = int(ell_center_x - 0.5 * MA * math.cos(angle / 180 * math.pi))
        ell_h_point1_y = int(ell_center_y - 0.5 * MA * math.sin(angle / 180 * math.pi))
        ell_h_point2_x = int(ell_center_x + 0.5 * MA * math.cos(angle / 180 * math.pi))
        ell_h_point2_y = int(ell_center_y + 0.5 * MA * math.sin(angle / 180 * math.pi))

        ell_w_point1_x = int(ell_center_x - 0.5 * ma * math.sin(angle / 180 * math.pi))
        ell_w_point1_y = int(ell_center_y + 0.5 * ma * math.cos(angle / 180 * math.pi))
        ell_w_point2_x = int(ell_center_x + 0.5 * ma * math.sin(angle / 180 * math.pi))
        ell_w_point2_y = int(ell_center_y - 0.5 * ma * math.cos(angle / 180 * math.pi))

        cv2.line(img, (ell_h_point1_x, ell_h_point1_y), (ell_h_point2_x, ell_h_point2_y), color1,
                 thickness=1)
        cv2.line(img, (ell_w_point1_x, ell_w_point1_y), (ell_w_point2_x, ell_w_point2_y), color2,
                 thickness=1)
    return ell_center_x, ell_center_y, MA, ma


def draw_axis_3(img, cnt, color1, color2):
    x, y, w, h = cv2.boundingRect(cnt)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    MM = cv2.moments(cnt)
    CX = int(MM["m10"] / MM["m00"])
    CY = int(MM["m01"] / MM["m00"])

    # draw the contour and center of the shape on the image
    ell_h_point1_x = CX
    ell_h_point1_y = int(CY - 0.5 * h)
    ell_h_point2_x = CX
    ell_h_point2_y = int(CY + 0.5 * h)

    ell_w_point1_x = int(CX - 0.5 * w)
    ell_w_point1_y = CY
    ell_w_point2_x = int(CX + 0.5 * w)
    ell_w_point2_y = CY

    cv2.line(img, (ell_h_point1_x, ell_h_point1_y), (ell_h_point2_x, ell_h_point2_y), color1,
             thickness=2)
    cv2.line(img, (ell_w_point1_x, ell_w_point1_y), (ell_w_point2_x, ell_w_point2_y), color2,
             thickness=2)
    return img


if __name__ == '__main__':
    import copy

    osp = os.path
    dir_path = r'D:\AcademicProject\thyroid_nodule\code\preprocess\result_v3\test_dataset'
    dir_path_mask = dir_path + '\mask_stage2_512'
    dir_path_image = dir_path + '\image_stage2_512'
    dir_path_out_img = osp.join(osp.dirname(dir_path_image), 'calcification_quantification_result')
    if not osp.exists(dir_path_out_img):
        os.makedirs(dir_path_out_img)
    id = []
    id_no = []
    dir_path_pred = dir_path + '\Stage2_TNSUCI_test_512_test'
    list = os.listdir(dir_path_pred)
    num = 0
    ###BGR
    colors = [(255, 0, 255), (0, 0, 255), (255, 255, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0),
              (224, 208, 64)]  # Magenta, Pure Red, White, Green, Yellow, Blue, Emerald Green
    features = []
    for index, name in enumerate(list):
        img = cv2.imread(str(osp.join(dir_path_image, name)), 1)
        mask = cv2.imread(osp.join(dir_path_mask, name), 0)
        mask_pred = cv2.imread(osp.join(dir_path_pred, name), 0)
        mask_noudle = mask_pred[:, 1024:1536]
        mask_cal = mask_pred[:, 1536:]
        img_new = copy.deepcopy(img)
        img_new1 = copy.deepcopy(img)
        mask_new = copy.deepcopy(mask)
        mask_new[mask_new > 0] = 1
        binary = np.multiply(mask_new, mask_cal)
        _, contours, hierarch = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _, contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, contours1, _ = cv2.findContours(mask_noudle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours1) < 1:
            continue
        contours11 = []
        for i in range(len(contours1)):
            try:
                contours_zoom, d1 = proportional_zoom_contour_ratio(contours1[i], 0.75)
                contours11.append(contours_zoom)
                contours_zoom, d2 = proportional_zoom_contour_ratio(contours1[i], 0.5)
                contours11.append(contours_zoom)

                contours_zoom, d1 = proportional_zoom_contour_ratio(contours1[i], 0.25)
                contours11.append(contours_zoom)
            except:
                print("failure")
        try:
            cv2.drawContours(img, contours11, -1, colors[-3], 2)
        except:
            print(len(contours11))
        try:
            cv2.drawContours(img, contours1, -1, colors[3], 2)
            # cv2.drawContours(img, contours2, -1, (0, 0, 255), 2)
            MM = cv2.moments(contours1[0])
            CX = int(MM["m10"] / MM["m00"])
            CY = int(MM["m01"] / MM["m00"])

            ###The method of aspect draw
            img = draw_axis_3(img, contours1[0], colors[0], colors[2])

            ###quantify the feature of calcification
            j = 1
            for i in range(len(contours)):
                cnt = contours[i]
                Area = cv2.contourArea(cnt)  ##Area of the calcification
                Circumference = cv2.arcLength(cnt, True)  ##Circumference the calcification
                if Area < 36:  # 36
                    cv2.drawContours(binary, [cnt], -1, (0), thickness=-1)
                    continue
                ###draw calcification region and aspect ratio
                cv2.drawContours(img, [cnt], -1, colors[-2], thickness=-1)
                try:
                    Cx, Cy, Mx, My = draw_axis(img, cnt, colors[1], colors[1], False)
                    # print(Cx,Cy)

                    Dist = []
                    for k, cnt_inter in enumerate(contours11):
                        ###Inside, Outside, or On of the contour(returns +1, -1, 0 accordingly)
                        dist = cv2.pointPolygonTest(cnt_inter, (Cx, Cy), False)
                        Dist.append(int(dist))
                    Range = 'A'
                    if Dist == [1, 1, 1]:  ##75%-100% Range=D
                        Range = "D"
                    if Dist == [1, 1, -1]:  ##50%-75% Range=C
                        Range = "C"
                    if Dist == [1, -1, -1]:  ##25%-50% Range=B
                        Range = "B"
                    if Dist == [-1, -1, -1]:  ##0%-25% Range=A
                        Range = "A"
                    ###
                    Position = 'UR'
                    if Cx <= CX and Cy <= CY:
                        Position = 'UL'
                    if Cx > CX and Cy <= CY:
                        Position = 'UR'
                    if Cx <= CX and Cy > CY:
                        Position = 'LL'
                    if Cx > CX and Cy > CY:
                        Position = 'LR'
                    cv2.putText(img, str(j), (int(Cx) - 5, int(Cy) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    feature = ["P_TT-" + name, j, round(Area / Mx, 2), round(Area / My, 2),
                               round(Area / Circumference, 2), Range, Position]
                    features.append(feature)
                    j = j + 1
                except:
                    continue
        except:
            continue

        R = np.count_nonzero(binary)
        if R > 0:
            id.append("P_TT-" + name)
        else:
            features.append(["P_TT-" + name, '', '', '', '', '', ''])
            id_no.append("P_TT-" + name)
        cv2.drawContours(img_new, contours1, -1, (0, 255, 0), 2)
        cv2.drawContours(img_new, contours2, -1, (0, 0, 255), 2)

        img_new = img_new[:, :, ::-1]
        img_new[..., 0] = np.where(binary == 255, 0, img_new[..., 0])
        img_new[..., 1] = np.where(binary == 255, 0, img_new[..., 1])
        img_new[..., 2] = np.where(binary == 255, 255, img_new[..., 2])
        img_new = img_new[..., ::-1]
        cv2.imwrite(osp.join(dir_path_out_img, name), np.hstack((img_new1, img_new, img)))
    save = pd.DataFrame(features,
                        columns=['id', 'cal_id', 'Area/mx', 'Area/my', 'Area/Circumference', 'Range', 'Position'])
    save.to_csv('cal_features.csv', index=False, header=True)
