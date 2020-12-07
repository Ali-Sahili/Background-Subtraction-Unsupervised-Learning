#!/usr/bin/env python

import os
import numpy as np
import ctypes as c
from numpy.ctypeslib import ndpointer

LIB_PATH    = '../build' #os.path.dirname(__file__)
LIB_NAME    = 'libSubsense'
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
        self.lib_subsense.ss_apply.argtypes = [CTX_PTR, IMG_PTR, IMG_PTR,IMG_PTR]

    def _create(self, img):
        (h, w) = img.shape[:2]        
        self.fg_mask = np.zeros((h,w), np.uint8) 
        self.bg_mask = np.zeros((h,w), np.uint8)       
        self._ctx = self.lib_subsense.ss_create(img, self._method(), w, h, *self._params)

    def apply(self, img):                
        if self._ctx is None:
            self._create(img)
        self.lib_subsense.ss_apply(self._ctx, img, self.fg_mask,self.bg_mask)
        return self.fg_mask,self.bg_mask

    def release(self):
        self.lib_subsense.ss_destroy(self._ctx)
        self._ctx = None

class Subsense(LBSP):
    def _method(self):
        return 0

def main():
    import sys
    import cv2

    video_path = '/home/ali/Desktop/Research/Works_MainTopics/varna_20190125_153327_0_900/varna_20190125_153327_0_900.mp4'

    #'/home/ali/Desktop/Research/Videos_for_Testing/sequence.avi'
    #'/home/ali/Desktop/Research/Videos_for_Testing/VIRAT_S_010204_05_000856_000890.mp4'
    #'/home/ali/Desktop/Research/Videos_for_Testing/trafficdb/video/cctv052x2004080516x01638.avi'
    #'/home/ali/Desktop/Research/Videos_for_Testing/driveway-320x240.avi'


    cap = cv2.VideoCapture(video_path)
    subtractor = Subsense()


    while True:
        ret, frame = cap.read()

        if frame is None:
            break


        fg_mask,bg_mask=subtractor.apply(frame)

        cv2.imshow('Original video', frame)
        cv2.imshow('Foreground Mask', fg_mask)


        #cv2.imshow('Background Mask', bg_mask)

        keycode = cv2.waitKey(1)
        quit = ((keycode & 0xFF) == ord('q'))
        if quit:
            break

    subtractor.release()

if __name__=='__main__':
    main()



