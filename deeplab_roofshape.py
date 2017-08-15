#!/usr/bin/env python

#from __future__ import print_function
import matplotlib 
matplotlib.use('Agg') 

caffe_root = 'code/'
import sys
sys.path.insert(0, caffe_root + 'python')

import os
import numpy as np
from numpy import unravel_index
from PIL import Image as PILImage
import caffe
from utils import pascal_palette_invert, pascal_mean_values, Timer
from segmenter import Segmenter
import cv2

def main():
  img_size = 500
  img_size4seg = img_size + 5 # deeplab takes 505 * 505, which padding 5 was used.
  roof_legitimate_size = 200 # at least 200 pixels to be considered a legitimate roof
  roofshape_legitimate_size = 50 # means, at least this number of roof pixels which predicted by roof presense model should be predicted as roof shape pixels (means, non-background pixels)
  palette = pascal_palette_invert()

  gpu_id, net_path, model_path, out_path, img_path, img_seg_path = process_arguments(sys.argv)
  net = Segmenter(net_path, model_path, gpu_id)
  img, cur_h, cur_w = preprocess_image(img_path, img_size4seg)
  timer = Timer()
  timer.tic()
  rs_segm_result = net.predict([img])
  roofshape_segm_result = rs_segm_result.argmax(axis=0).astype(np.uint8)
  #belief_segm_result    = rs_segm_result[unravel_index(roofshape_segm_result, rs_segm_result.shape)] 
  #roofshape_segm_result = roofshape_segm_result[0:img_size, 0:img_size] #trim down to 500 * 500
  timer.toc()
  print '\nDeepLab semantic-segmentation took {:.4f}s'.format(timer.total_time)
  #print roofshape_segm_result.shape

  # get the roof segments np.array format
  input_seg_image = caffe.io.load_image(img_seg_path)
  seg_image = PILImage.fromarray(np.uint8(input_seg_image))
  seg_image = np.array(seg_image)

  input_seg_image = cv2.imread(img_seg_path)
  input_seg_image[input_seg_image == 1 ] = 210
  # get the roof segments and work on each segment
  labelRBG_gray = cv2.cvtColor(input_seg_image, cv2.COLOR_RGB2GRAY)
  #print input_seg_image.shape

  ret, thresh = cv2.threshold(labelRBG_gray, 127, 255, 0)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  for cnt in range(len(contours)):
    mask = np.zeros(labelRBG_gray.shape, np.uint8)
    cv2.drawContours(mask, contours, cnt, 255, -1)
    #pixelpoints = np.transpose(np.nonzero(mask))
    if cv2.contourArea(contours[cnt]) >= roof_legitimate_size:      
      roofshape_seg = roofshape_segm_result[np.nonzero(mask)]
      if np.count_nonzero(roofshape_seg) >= roofshape_legitimate_size:
        counts = np.bincount(roofshape_seg)
        ii = np.nonzero(counts)[0]
        roofshape_profile = np.vstack((ii, counts[ii])).T
        roofshape_profile = roofshape_profile[np.argsort(roofshape_profile[:, 1]), ]
        #print roofshape_profile
        shape_class = roofshape_profile[roofshape_profile.shape[0]-1, 0]
        
        scores = rs_segm_result[shape_class, :, :]
        print np.mean(scores[np.nonzero(mask)])
        seg_image[np.nonzero(mask)] = shape_class #roofshape_profile[roofshape_profile.shape[0]-1, 0] #assign the class according to the biggest(per pixel count) class
      else:
        #if too few roof shape pixel, just put everything into background. this means, overwrite the results from roof presence. at this point, I haven't seen it happened yet.
        seg_image[np.nonzero(mask)] = 0
    else:
      #if the roof segment is too small, smaller than certain pixels, just change the recognized roofs back to background, means, we consider they are not roof.
      seg_image[np.nonzero(mask)] = 0
  seg_image = seg_image * 25
  result_img = PILImage.fromarray(seg_image)
#  result_img.putpalette(palette)
  result_name = out_path + os.path.basename(img_path).split('.')[0]+'-res.png'
  result_img.save(result_name)

def preprocess_image(img_path, img_size):
  if not os.path.exists(img_path):
    print(img_path)
    return None, 0, 0

  input_image = 255 * caffe.io.load_image(img_path)
  
  image = PILImage.fromarray(np.uint8(input_image))
  image = np.array(image)
  
  mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
  reshaped_mean_vec = mean_vec.reshape(1, 1, 3);
  preprocess_img = image[:,:,::-1]
  preprocess_img = preprocess_img - reshaped_mean_vec
  
  # Pad as necessary
  cur_h, cur_w, cur_c = preprocess_img.shape
  pad_h = img_size - cur_h
  pad_w = img_size - cur_w
  preprocess_img = np.pad(preprocess_img, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)

  return preprocess_img, cur_h, cur_w

def process_arguments(argv):
  gpu_id     = None
  net_path   = None
  model_path = None 
  img_path  = None 
  out_path   = None
  img_seg_path = None

  if len(argv) >= 6:
    gpu_id     = int(argv[1])
    net_path   = argv[2]
    model_path = argv[3]
    out_path   = argv[4]
    img_path  = argv[5]
    img_seg_path = argv[6]
  else:
    help()
  return gpu_id, net_path, model_path, out_path, img_path, img_seg_path

def help():
  print('Usage: python deeplab_roofshape.py GPU_ID NET MODEL OUT_PATH IMAGE IMAGE_SEG\n'
        'GPU_ID specifies gpu number used for computation.\n'
        'NET file describing network (prototxt extension).\n'
        'MODEL file generated by caffe (caffemodel extension).\n'
        'IMAGE one image to be processed.\n'
        'IMAGE_SEGMENT image segment to be processed.')

#        , file=sys.stderr)

  exit()

def pascal_palette():
  palette = {(  0,   0,   0) : 0 ,
             (128,   0,   0) : 1 ,
             (  0, 128,   0) : 2 ,
#             (  1,   1,   1) : 1 ,
#             (  2,   2,   2) : 2 ,
             (128, 128,   0) : 3 ,
             (  0,   0, 128) : 4 ,
             (128,   0, 128) : 5 ,
             (  0, 128, 128) : 6 ,
             (128, 128, 128) : 7 ,
             ( 64,   0,   0) : 8 ,
             (192,   0,   0) : 9 ,
             ( 64, 128,   0) : 10,
             (192, 128,   0) : 11,
             ( 64,   0, 128) : 12,
             (192,   0, 128) : 13,
             ( 64, 128, 128) : 14,
             (192, 128, 128) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20 }

  return palette

def pascal_palette_invert():
  palette_list = pascal_palette().keys()
  palette = ()

  for color in palette_list:
    palette += color

  return palette

if __name__ == '__main__':
  main()
