#!/usr/bin/env python

import _init_paths
import caffe
import numpy as np
import copy
import os.path as osp
from detect.train import train_net
from detect.config import cfg

def extract_data(filename):
    num  = np.array( (1,1), dtype=float )
    ind   = np.array( (1,1), dtype=float )
    roitx = np.array( (1,1), dtype=float )
    roity = np.array( (1,1), dtype=float )
    roidx = np.array( (1,1), dtype=float )
    roidy = np.array( (1,1), dtype=float )
    bboxx = np.array( (1,1), dtype=float )
    bboxy = np.array( (1,1), dtype=float )
    bboxw = np.array( (1,1), dtype=float )
    bboxh = np.array( (1,1), dtype=float )
    weight= np.array( (1,1), dtype=float )

    data = dict()
    roidb = list()
    indexdb = list()

    f = open(filename, 'r')
    for line in f.readlines():
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue 

        num, ind, roitx, roity, roidx, roidy, bboxx, bboxy, bboxw, bboxh, label, weight = [float(i) for i in line.split()]
        data['roi'] = np.array([roitx, roity, roidx, roidy])
        data['bbox_target'] = np.array([bboxx, bboxy, bboxw, bboxh])
        data['label'] = np.int_(label)
        data['weight'] = weight
        data['image'] = osp.abspath(osp.join(cfg.IMAGE_DIR, str(np.int_(ind))+cfg.IMAGE_TYPE ))
        data['imageindex'] = np.int_(ind)
        roidb.append(copy.deepcopy(data))
        indexdb.append(np.int_(ind))

    return sorted(roidb, key=lambda x: x['imageindex'], reverse=False), sorted(indexdb, key=lambda x: x, reverse=False)

if __name__ == '__main__':

    cfg.IMAGE_DIR        = cfg.ROOT_DIR + '/trainimage'
    cfg.TRAIN_SOLVER     = cfg.ROOT_DIR + '/solver_ship.prototxt'
    cfg.DATA_TXT         = cfg.ROOT_DIR + '/trainimage/shipsamplestep2roi.txt'
    cfg.SAVED_MODEL_NAME = cfg.ROOT_DIR + '/ship_s'
    cfg.PRETRAINED_MODEL = None
#    cfg.PRETRAINED_MODEL = cfg.ROOT_DIR + '/ship_s_iter_50000.caffemodel'

    data_record, index_record = extract_data(cfg.DATA_TXT)

    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

    train_net(cfg.TRAIN_SOLVER, data_record, index_record, pretrained_model=cfg.PRETRAINED_MODEL, max_iters=cfg.MAX_ITERS)