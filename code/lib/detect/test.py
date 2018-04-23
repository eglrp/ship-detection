import numpy as np
import cv2
import os.path as osp
import matplotlib.pyplot as plt
from detect.config import cfg
from detect.nms_wrapper import nms


def im_detect(net, im):

    blobs = {'data' : None}

    data_blob = np.zeros( (1, cfg.INPUT_IMAGE_HEIGHT,cfg.INPUT_IMAGE_WIDTH,3), dtype=np.float32)
    data_blob[0, 0:im.shape[0], 0:im.shape[1],:] = im
    channel_swap = (0, 3, 1, 2)
    data_blob = data_blob.transpose(channel_swap)

    blobs['data'] = data_blob

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}

    blobs_out = net.forward(**forward_kwargs)

    pred_rois = net.blobs['roi2'].data
    prob_rois = blobs_out['prob']

    return pred_rois, prob_rois 

def vis_detections(im, bbox, prob, save):
    """Visual debugging of detections."""
    if save:
        IMAGE_DIR = cfg.ROOT_DIR + '/result/image'
        TXT_DIR   = cfg.ROOT_DIR + '/result/position'

    plt.cla()
#    plt.figure(figsize=(15,15))
    if im.ndim == 3:
        im = im[:,:,0]
    plt.imshow(im)
    for i in xrange(len(bbox)):
        plt.gca().add_patch(plt.Rectangle((bbox[i][1], bbox[i][2]), (bbox[i][3]-bbox[i][1]), (bbox[i][4]-bbox[i][2]), fill=False, edgecolor='g', linewidth=3) )                  
        plt.text(bbox[i][1], bbox[i][2] - 2,'{:.3f}'.format(float(prob[i,1])),fontdict={'size': 12, 'color': 'y'})
        if save:
            with open(osp.join(TXT_DIR, str(cfg.IMAGE_NUMBER)+'.txt'),"a") as f:
               new_con = str(np.int_(bbox[i][1])) + ' ' + str(np.int_(bbox[i][2])) + ' ' + str(np.int_(bbox[i][3])) + ' ' + str(np.int_(bbox[i][4])) + '\n'
               f.write(new_con)
    if save:
        plt.savefig(osp.join(IMAGE_DIR, str(cfg.IMAGE_NUMBER)+'.jpg'),bbox_inches='tight')
    else:
        plt.show()
        print bbox

def test_image(net, image_source, vis=False, save=False):
    """Test network on a whole image."""
    im = cv2.imread(image_source)
    im = im.astype(np.float32, copy=False)

    im -= cfg.IMAGE_MEANS
    im *= cfg.IMAGE_SCALE 
#    im = cv2.resize(im, (cfg.INPUT_IMAGE_WIDTH, cfg.INPUT_IMAGE_HEIGHT) ,interpolation=cv2.INTER_LINEAR)

    pred_rois, pred_prob  = im_detect(net, im)

    inds = np.where(pred_prob[:, 1] > cfg.DETECT_THRESH)[0]
    pred_rois = pred_rois[inds, :]
    pred_prob = pred_prob[inds, :]

    dets = np.zeros( (len(pred_rois), 5), dtype=np.float32)

    dets[:,0:4] = pred_rois[:,1:5]
    dets[:,4] = pred_prob[:,1]

    keep = nms(dets, cfg.NMS)

#    vis = True
#    save= True
    if (vis or save):
        vis_detections(cv2.imread(image_source).astype(np.float32, copy=False), pred_rois[keep, :], pred_prob[keep, :], save)