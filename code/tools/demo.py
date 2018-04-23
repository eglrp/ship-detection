#!/usr/bin/env python

import _init_paths
import caffe
from detect.test import test_image
from detect.config import cfg
import numpy as np
import os.path as osp

if __name__ == '__main__':

    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

    image_source = cfg.ROOT_DIR + '/testimage/1.jpg'

    cfg.TEST_PROTOTXT = 'test.prototxt'
    cfg.TEST_MODEL    = 'demo.caffemodel'
    net = caffe.Net(cfg.TEST_PROTOTXT, cfg.TEST_MODEL, caffe.TEST)

    
    test_image(net, image_source, vis=True, save=False)

    '''
    for i in xrange(100):
        ind = i
        cfg.IMAGE_NUMBER = np.int_(ind)+1
        image_source = osp.join(cfg.ROOT_DIR + '/testimage', str(np.int_(ind)+1)+cfg.IMAGE_TYPE)
        net = caffe.Net(cfg.TEST_PROTOTXT, cfg.TEST_MODEL, caffe.TEST)
        if osp.exists(image_source):
            test_image(net, image_source, vis=False, save=True)
    '''