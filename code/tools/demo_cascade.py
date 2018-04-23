#!/usr/bin/env python

import _init_paths
import caffe
from detect.test_image_cascade import test_image,test_shiphead
from detect.config import cfg
import numpy as np
import os.path as osp

if __name__ == '__main__':

    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)
    '''
    image_source = cfg.ROOT_DIR + '/testimage/1.jpg'

    step1_PROTOTXT = 'test_shiphead.prototxt'
    cfg.TEST_MODEL = 'shiphead8_iter_60000.caffemodel'    

    net1 = caffe.Net(step1_PROTOTXT, cfg.TEST_MODEL, caffe.TEST)
    rois = test_shiphead(net1, image_source)

    step2_PROTOTXT = 'test_cascade_2.prototxt'
    cfg.TEST_MODEL = 'demo_cascade_2.caffemodel'

    net2 = caffe.Net(step2_PROTOTXT, cfg.TEST_MODEL, caffe.TEST)
    test_image(net2, image_source, rois, vis=True, save=False)

    '''
    for i in xrange(100):
        ind = i
        cfg.IMAGE_NUMBER = np.int_(ind)+1
        image_source = osp.join(cfg.ROOT_DIR + '/testimage', str(np.int_(ind)+1)+cfg.IMAGE_TYPE)

        if osp.exists(image_source):
            step1_PROTOTXT = 'test_shiphead.prototxt'
            cfg.TEST_MODEL = 'shiphead8_iter_60000.caffemodel'

            net1 = caffe.Net(step1_PROTOTXT, cfg.TEST_MODEL, caffe.TEST)
            rois = test_shiphead(net1, image_source)

            step2_PROTOTXT = 'test_cascade_2.prototxt'
            cfg.TEST_MODEL = 'demo_cascade_2.caffemodel'

            net2 = caffe.Net(step2_PROTOTXT, cfg.TEST_MODEL, caffe.TEST)
            test_image(net2, image_source, rois, vis=False, save=True)
