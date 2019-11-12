#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Aravind Manoharan
aravindmanoharan05@gmail.com
https://github.com/aravindmanoharan
'''

import sys
import cv2
import numpy as np
from keras.models import model_from_json
from scipy.ndimage.measurements import label

IMAGE_NAME = sys.argv[1]
TRAINED_MODEL_WEIGHTS = 'trained_model/final_weights.h5'
TRAINED_MODEL_NAME = 'trained_model/model.json'

def add_heat(heatmap, x, y):

    heatmap[y:y+40, x:x+40] += 1

    return heatmap

def apply_threshold(heatmap, threshold):
    
    heatmap[heatmap <= threshold] = 0
    
    return heatmap

def find_area(bbox):
    
    xa, yb = bbox
    x = abs(xa[0] - yb[0])
    y = abs(xa[1] - yb[1])
    return x * y

def biggest_box(bbox_list):
    
    biggest_box_n = 0
    if len(bbox_list) == 1:
        return biggest_box_n
    
    max_area = find_area(bbox_list[0])
    biggest_box_n = 0
    
    for i in range(1, len(bbox_list)):
        
        area = find_area(bbox_list[i])
        if area > max_area:
            max_area = area
            biggest_box_n = i
    
    return biggest_box_n

def center(bbox):
    
    xa, yb = bbox
    x = int(abs(xa[0] - yb[0]) / 2)
    y = int(abs(xa[1] - yb[1]) / 2)
    
    return xa[0]+x, xa[1]+y

def draw_labeled_bboxes(labels):
    
    bbox_list = []
    for phone_number in range(1, labels[1]+1):
        
        nonzero = (labels[0] == phone_number).nonzero()
        
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_list.append(bbox)
    
    return bbox_list

def load_model(model_name, model_weights):
    
    # load json and create model
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights)
    
    return loaded_model

if __name__ == '__main__':
    
    loaded_model = load_model(TRAINED_MODEL_NAME, TRAINED_MODEL_WEIGHTS)
    
    test_image = cv2.imread(IMAGE_NAME)
    image_h = test_image.shape[0]
    image_w = test_image.shape[1]
    
    heatmap = np.zeros_like(test_image[:,:,0])
    
    stride = 5
    
    windows = []
    windows_labels = []
    
    flagged_windows = []
    for y in range(0, image_h, stride):
        for x in range(0, image_w, stride):
            if (x + 40) < image_w and (y + 40) < image_h:
                top_left = (x, y)
                bottom_right = ((x + 40), (y + 40))
                
                img = test_image[y:y+40,x:x+40]
                
                img = np.reshape(img, (-1, 40, 40, 3))
                out = int(loaded_model.predict(img))
                if out == 1:
                    heatmap = add_heat(heatmap, x, y)
    
    heatmap = apply_threshold(heatmap, np.max(heatmap) - 1)
    heat = np.clip(heatmap,0,255)
    
    labels = label(heat)
    
    bbox_list = draw_labeled_bboxes(labels)
    
    biggest_box_n = biggest_box(bbox_list)
    
    pred_x, pred_y = center(bbox_list[biggest_box_n])
    
    print(round(pred_x/image_w,4), round(pred_y/image_h,4))
    
    cv2.circle(test_image, (pred_x, pred_y), 3, (0,255,0), -1)
    cv2.imshow('output', test_image)
    cv2.waitKey(0)
    