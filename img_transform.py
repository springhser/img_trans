#!/usr/bin/env python
# encoding: utf-8
"""
@version: 0.1
@author: springhser
@license: Apache Licence 
@contact: endoffight@gmail.com
@site: http://www.springhser.com
@software: PyCharm Community Edition
@file: img_transform.py
@time: 2017/12/30 17:58
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

# 线段类
class Line:
    def __init__(self, line):
        # x1, y1, x2, y2 = l # 前两个数为起点，后两个数为终点
        self.x1 = line[0]
        self.y1 = line[1]
        self.x2 = line[2]
        self.y2 = line[3]
        # 线段中点的坐标, 为排序使用
        self.half_x = (self.x1 + self.x2) / 2
        self.half_y = (self.y1 + self.y2) / 2
    # 求出与另外一条不平行线段延长线的交点
    def get_cross_point(self, l_a):
        a1 = self.y2 - self.y1
        b1 = self.x1 - self.x2
        c1 = a1 * self.x1 + b1 * self.y1
        a2 = l_a.y2 - l_a.y1
        b2 = l_a.x1 - l_a.x2
        c2 = a2 * l_a.x1 + b2 * l_a.x2
        d = a1 * b2 - a2 * b1
        if d is 0: # 平行或共线的情况
            raise ValueError
        return (1. * (b2 * c1 - b1 * c2) / d, 1. * (a1 * c2 - a2 * c1) / d)

# 对图像进行预处理求出图像中的直线
def img_process(old_img):
    # 重新设定图像大小，方便计算
    height = old_img.shape[0]
    weight = old_img.shape[1]
    min_weight = 200
    scale = min(10., weight * 1. / min_weight)
    new_h = int(height * 1. / scale)
    new_w = int(weight * 1. / scale)
    new_img = cv2.resize(old_img, (new_w, new_h))
    gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) # 转为灰度图像
    # 利用Canny 边缘检测和霍夫变换提取直线
    highthreshold = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]  #  http://blog.csdn.net/on2way/article/details/46812121
    lowthreshold = highthreshold * 0.2
    canny_img = cv2.Canny(gray_img, lowthreshold, highthreshold)
    return cv2.HoughLinesP(canny_img, 1, np.pi / 180, new_w // 3, 50, new_w // 3, 20), new_img

# 获得四个目标点
def get_target_points(lines, img):
    # 分别提取出水平和垂直的线段
    # print(lines)
    lines_h = []  # 存放接近水平的线段
    lines_v = []  # 存放接近垂直的线段
    lines1 = lines[:, 0, :]  # 提取为二维
    for l in lines1:
        # print(l)
        line = Line(l)
        if abs(line.x1 - line.x2) > abs(line.y1 - line.y2):
            lines_h.append(line)
        else:
            lines_v.append(line)

    # 如果线段数不够两条, 直接用原图像的边缘替代
    if len(lines_h) <= 1:
        if not lines_h or lines_h[0].half_y > img.shape[0]/2:
            lines_h.append(Line((0, 0, img.shape[1] - 1, 0)))
        if not lines_h or lines_h[0].half.y <= img.shape[0] / 2:
            lines_h.append(Line((0, img.shape[0] - 1, img.shape[1] - 1, img.shape[0] - 1)))
    if len(lines_v) <= 1:
        if not lines_v or lines_v[0].half_x > img.shape[1] / 2:
            lines_v.append(Line((0, 0, 0, img.shape[0] - 1)))
        if not lines_v or lines_v[0].c_x <= img.shape[1] / 2:
            lines_v.append(Line((img.shape[1] - 1, 0, img.shape[1] - 1, img.shape[0] - 1)))

    # 获取最靠近边缘的四条线段求出他们的交点
    lines_h.sort(key=lambda line: line.half_y)
    lines_v.sort(key=lambda line: line.half_x)
    return  [lines_h[0].get_cross_point(lines_v[0]),
            lines_h[0].get_cross_point(lines_v[-1]),
            lines_h[-1].get_cross_point(lines_v[0]),
            lines_h[-1].get_cross_point(lines_v[-1])]

# 做透视变换
def per_transform(target_points, old_img):
    height = old_img.shape[0]
    weight = old_img.shape[1]
    min_weight = 200
    scale = min(10., weight * 1. / min_weight)
    # 恢复为原图像大小
    for i, p in enumerate(target_points):
        x, y = p
        target_points[i] = (x * scale, y * scale)
    # 原图像的四个点
    four_points= np.array(((0, 0),
                           (weight - 1, 0),
                           (0, height - 1),
                           (weight - 1, height - 1)),
                          np.float32)
    target_points = np.array(target_points, np.float32)
    M = cv2.getPerspectiveTransform(target_points, four_points)
    return cv2.warpPerspective(old_img, M, (weight, height))


if __name__ == '__main__':
    old_img = cv2.imread("001.jpg")
    lines, new_img = img_process(old_img)
    t_points = get_target_points(lines, new_img)
    revert_img = per_transform(t_points, old_img)
    plt.imshow(revert_img,)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


