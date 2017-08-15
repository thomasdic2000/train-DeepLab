#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/03/11

import scipy.io
import struct
import numpy as np
import time

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def pascal_classes():
  classes = {'aeroplane' : 1,  'bicycle'   : 2,  'bird'        : 3,  'boat'         : 4,
             'bottle'    : 5,  'bus'       : 6,  'car'         : 7,  'cat'          : 8,
             'chair'     : 9,  'cow'       : 10, 'diningtable' : 11, 'dog'          : 12,
             'horse'     : 13, 'motorbike' : 14, 'person'      : 15, 'potted-plant' : 16,
             'sheep'     : 17, 'sofa'      : 18, 'train'       : 19, 'tv/monitor'   : 20}

  return classes

def pascal_palette():
  palette = {(  0,   0,   0) : 0 ,
#             (128,   0,   0) : 1 ,
#             (  0, 128,   0) : 2 ,
             (  1,   1,   1) : 1 ,
             (  2,   2,   2) : 2 ,
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

def pascal_mean_values():
  return np.array([103.939, 116.779, 123.68], dtype=np.float32)

def strstr(str1, str2):
  if str1.find(str2) != -1:
    return True
  else:
    return False

# Mat to png conversion for http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html
# 'GTcls' key is for class segmentation
# 'GTinst' key is for instance segmentation
def mat2png_hariharan(mat_file, key='GTcls'):
  mat = scipy.io.loadmat(mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False)
  return mat[key].Segmentation

def convert_segmentation_mat2numpy(mat_file):
  np_segm = load_mat(mat_file)
  return np.rot90(np.fliplr(np.argmax(np_segm, axis=2)))

def load_mat(mat_file, key='data'):
  mat = scipy.io.loadmat(mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False)
  return mat[key]

# Python version of script in code/densecrf/my_script/LoadBinFile.m
def load_binary_segmentation(bin_file, dtype='int16'):
  with open(bin_file, 'rb') as bf:
    rows = struct.unpack('i', bf.read(4))[0]
    cols = struct.unpack('i', bf.read(4))[0]
    channels = struct.unpack('i', bf.read(4))[0]

    num_values = rows * cols # expect only one channel in segmentation output
    out = np.zeros(num_values, dtype=np.uint8) # expect only values between 0 and 255

    for i in range(num_values):
      out[i] = np.uint8(struct.unpack('h', bf.read(2))[0])

    return np.rot90(np.fliplr(out.reshape((cols, rows))))

def convert_from_color_segmentation(arr_3d):
  arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
  palette = pascal_palette()

  # slow!
  for i in range(0, arr_3d.shape[0]):
    for j in range(0, arr_3d.shape[1]):
      key = (arr_3d[i,j,0], arr_3d[i,j,1], arr_3d[i,j,2])
      arr_2d[i, j] = palette.get(key, 0) # default value if key was not found is 0

  return arr_2d

def create_lut(class_ids, max_id=256):
  # Index 0 is the first index used in caffe for denoting labels.
  # Therefore, index 0 is considered as default.
  lut = np.zeros(max_id, dtype=np.uint8)

  new_index = 1
  for i in class_ids:
    lut[i] = new_index
    new_index += 1

  return lut

def get_id_classes(classes):
  all_classes = pascal_classes()
  id_classes = [all_classes[c] for c in classes]
  return id_classes
