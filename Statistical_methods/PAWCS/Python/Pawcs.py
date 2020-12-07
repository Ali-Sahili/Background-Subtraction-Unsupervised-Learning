#!/usr/bin/env python

import os
import numpy as np
import ctypes as c
from numpy.ctypeslib import ndpointer
import sys
import cv2
from skimage import io


LIB_PATH    = '../build' #os.path.dirname(__file__)
LIB_NAME    = 'libPawcs'
IMG_PTR     = ndpointer(c.c_uint8, flags="C_CONTIGUOUS")
CTX_PTR     = c.c_void_p

class LBSP(object):
    def __init__(self,
                 lbsp_thresh = 0.333,
                 desc_dist_thresh_offset = 3,
                 min_color_dist_thresh = 30,
                 num_bg_samples = 50,
                 num_req_bg_samples = 2,
                 num_samples_for_moving_avg = 100):
        self._ctx = None
        self._params = (lbsp_thresh,
                        desc_dist_thresh_offset,
                        min_color_dist_thresh,
                        num_bg_samples,
                        num_req_bg_samples,
                        num_samples_for_moving_avg)
        self.lib_subsense = np.ctypeslib.load_library(LIB_NAME, LIB_PATH)
        
        # API: ss_create
        self.lib_subsense.ss_create.restype = CTX_PTR
        self.lib_subsense.ss_create.argtypes = [IMG_PTR,
                                                c.c_int,
                                                c.c_int,
                                                c.c_int,
                                                c.c_float,
                                                c.c_size_t,
                                                c.c_size_t,
                                                c.c_size_t,
                                                c.c_size_t,
                                                c.c_size_t]

        # API: ss_destroy
        self.lib_subsense.ss_destroy.restype = c.c_int
        self.lib_subsense.ss_destroy.argtypes = [CTX_PTR]

        # API: ss_apply
        self.lib_subsense.ss_apply.restype = c.c_int
        self.lib_subsense.ss_apply.argtypes = [CTX_PTR, IMG_PTR, IMG_PTR]

    def _create(self, img):
        (h, w) = img.shape[:2]        
        self.fg_mask = np.zeros((h,w), np.uint8)        
        self._ctx = self.lib_subsense.ss_create(img, self._method(), w, h, *self._params)

    def apply(self, img):                
        if self._ctx is None:
            self._create(img)
        self.lib_subsense.ss_apply(self._ctx, img, self.fg_mask)
        return self.fg_mask

    def release(self):
        self.lib_subsense.ss_destroy(self._ctx)
        self._ctx = None


class Pawcs(LBSP):
    def _method(self):
        return 2


def NumberOfDigits(Number):
    Count = 0
    while(Number > 0):
        Number = Number//10
        Count = Count + 1

    return Count




sequence_path = '/home/ali/Desktop/Research/Works_MainTopics/varna_20190125_153327_0_900/img/'
#899300
#sequence_path = '/home/ali/Desktop/MyOutput/LBP_PBAS/'

subtractor = Pawcs()

nb = '000000'
while True:
    print(nb)

    frame = io.imread(sequence_path+'varna_20190125_153327_0_900_0000'+nb+'.jpg')
    if frame is None:
        break

    fg_mask = subtractor.apply(frame)

    name = '/home/ali/Desktop/MyOutput/PAWCS/varna_20190125_153327_0_900_0000'+nb+'.jpg'

    io.imsave(name, fg_mask)

    if (int(nb) == 899300):
        break

    nb = int(nb)+100
        
    nb_digits = NumberOfDigits(int(nb))
    #print(nb_digits)

    nb = str('0'*(6-nb_digits)) + str(nb)
    #print(nb)

    #cv2.imshow('Original video', frame)
    #cv2.imshow('Foreground Mask', fg_mask)
    #cv2.waitKey(0)


subtractor.release()



        


